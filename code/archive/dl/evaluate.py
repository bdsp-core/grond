"""
Evaluate learned model vs ridge baseline using the same patient-level CV.
Run: conda run -n foe_dl python code/dl/evaluate.py
"""

import sys
import os
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.signal import find_peaks, butter, filtfilt, coherence
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, pd_detect_pointiness_acf,
    bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
PREDICTIONS_PATH = CACHE_DIR / 'phase2_predictions.npz'
ANNOTATED_PATH = CACHE_DIR / 'annotated_pd_data.npz'

FS = 200
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


# ── Feature extraction functions (from r7_expert_spatial.py) ──────────
def preprocess_segment_r7(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    from mne.filter import notch_filter, filter_data
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def get_f_A(data, fs):
    """Method A: pd_detect_alternate(apd) event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def get_f_B(data, fs):
    """Method B: pd_detect_pointiness_acf."""
    try:
        r = pd_detect_pointiness_acf(
            data, fs, method='pointiness',
            acf_min_lag=ACF_MIN_LAG, acf_peak_threshold=ACF_THRESHOLD,
            smoothing_sigma=SMOOTHING_SIGMA, lowpass_hz=LOWPASS_HZ,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_pointiness_traces(seg, fs):
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    traces = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return np.array(traces)


def get_f_peaks(traces, fs):
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = traces[i]
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return median_finite(freqs)


def get_f_fft(traces, fs):
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        freqs[i] = fft_freqs[mask][np.argmax(fft_vals[mask])]
    return median_finite(freqs)


def get_f_tkeo_fft(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(0.02 * fs))
    for i in range(n_ch):
        x = seg[i]
        if len(x) < 3:
            continue
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
        tkeo = np.abs(tkeo)
        tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
        n = len(tkeo)
        if n < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        freqs[i] = fft_freqs[mask][np.argmax(fft_vals[mask])]
    return median_finite(freqs)


def get_f_spectral_coh(seg_bb, fs):
    nperseg = min(256, seg_bb.shape[1] // 2)
    if nperseg < 16:
        return np.nan
    coh_spectra = None
    coh_freqs = None
    for (a, b) in ADJACENT_PAIRS:
        if a >= seg_bb.shape[0] or b >= seg_bb.shape[0]:
            continue
        try:
            f_coh, Cxy = coherence(seg_bb[a], seg_bb[b], fs=fs, nperseg=nperseg)
            if coh_freqs is None:
                coh_freqs = f_coh
                coh_spectra = np.zeros_like(f_coh)
            coh_spectra = coh_spectra + Cxy
        except Exception:
            continue
    if coh_freqs is None:
        return np.nan
    coh_spectra /= len(ADJACENT_PAIRS)
    mask = (coh_freqs >= 0.3) & (coh_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    return float(coh_freqs[mask][np.argmax(coh_spectra[mask])])


def get_f_hps3(traces, fs):
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        max_idx = len(fft_vals) // 3
        if max_idx < 2:
            continue
        hps = fft_vals[:max_idx].copy()
        hps *= fft_vals[:max_idx * 2:2][:max_idx]
        hps *= fft_vals[:max_idx * 3:3][:max_idx]
        hps_freqs = fft_freqs[:max_idx]
        mask = (hps_freqs >= 0.3) & (hps_freqs <= 3.5)
        if not np.any(mask):
            continue
        freqs[i] = hps_freqs[mask][np.argmax(hps[mask])]
    return median_finite(freqs)


def impute_nan_median(X):
    X_out = X.copy()
    for fi in range(X_out.shape[1]):
        col = X_out[:, fi]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 1.0
            col[nan_mask] = med
            X_out[:, fi] = col
    return X_out


def ridge_grouped_cv(X, y, groups, n_splits=5, alpha=1.0):
    """Ridge regression with GroupKFold CV. Returns predictions array."""
    gkf = GroupKFold(n_splits=n_splits)
    predictions = np.full(len(y), np.nan)

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_tr_s = (X_train - mu) / std
        X_va_s = (X_val - mu) / std

        y_mu = np.mean(y_train)
        y_c = y_train - y_mu

        p = X_tr_s.shape[1]
        A = X_tr_s.T @ X_tr_s + alpha * np.eye(p)
        try:
            w = np.linalg.solve(A, X_tr_s.T @ y_c)
        except np.linalg.LinAlgError:
            continue
        preds = X_va_s @ w + y_mu
        predictions[val_idx] = preds

    return predictions


def extract_eventness_features(eventness_trace, fs=200):
    """Extract frequency-related features from a CNN eventness trace."""
    features = {}

    # FFT of eventness trace
    n = len(eventness_trace)
    fft_vals = np.abs(np.fft.rfft(eventness_trace - np.mean(eventness_trace)))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
    if np.any(mask):
        features['eventness_fft_freq'] = float(fft_freqs[mask][np.argmax(fft_vals[mask])])
        features['eventness_fft_power'] = float(np.max(fft_vals[mask]))
    else:
        features['eventness_fft_freq'] = np.nan
        features['eventness_fft_power'] = np.nan

    # Peak count from eventness
    peak_locs, _ = find_peaks(eventness_trace, height=0.3, distance=int(0.2 * fs))
    if len(peak_locs) >= 2:
        features['eventness_peak_freq'] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    else:
        features['eventness_peak_freq'] = np.nan

    # Mean eventness (proxy for detection confidence)
    features['eventness_mean'] = float(np.mean(eventness_trace))

    return features


def main():
    print("=" * 60)
    print("Evaluate: CNN vs Ridge Baseline Comparison")
    print("=" * 60)

    # ── Check prerequisites ───────────────────────────────────────────
    if not PREDICTIONS_PATH.exists():
        print(f"ERROR: Phase 2 predictions not found: {PREDICTIONS_PATH}")
        print("Run train_phase2.py first.")
        sys.exit(1)
    if not ANNOTATED_PATH.exists():
        print(f"ERROR: Annotated data not found: {ANNOTATED_PATH}")
        print("Run generate_weak_labels.py first.")
        sys.exit(1)

    # ── Load CNN predictions ──────────────────────────────────────────
    print("\n[1] Loading CNN predictions")
    pred_data = np.load(str(PREDICTIONS_PATH), allow_pickle=True)
    cnn_predictions = pred_data['predictions']  # (N,)
    pred_mat_names = pred_data['mat_names']     # (N,)
    patients = pred_data['patients']            # (N,)
    subtypes = pred_data['subtypes']            # (N,)

    N = len(cnn_predictions)
    n_valid_cnn = np.sum(np.isfinite(cnn_predictions))
    print(f"  Loaded {N} segments, {n_valid_cnn} valid CNN predictions")

    # ── Load annotated data for eventness features ────────────────────
    ann_data = np.load(str(ANNOTATED_PATH), allow_pickle=True)
    weak_eventness = ann_data['weak_eventness']  # (N, 2000) — for CNN eventness hybrid

    # ── Load original dataset ─────────────────────────────────────────
    print("\n[2] Loading original dataset and extracting ridge features")
    dataset = load_dataset()
    print(f"  Dataset: {len(dataset)} entries")

    # ── Extract ridge features (same as r7_expert_spatial.py) ─────────
    FEAT_NAMES = ['f_A', 'f_B', 'f_peaks', 'f_fft', 'f_tkeo_fft',
                  'f_spectral_coh', 'f_hps3', 'is_gpd']

    features = np.full((N, len(FEAT_NAMES)), np.nan)
    expert_consensus = np.full(N, np.nan)
    ridge_mat_names = []

    # We need to map pred_mat_names to dataset entries
    dataset_by_name = {e['mat_name']: e for e in dataset}

    t0 = time.time()
    for idx in range(N):
        mat_name = str(pred_mat_names[idx])
        entry = dataset_by_name.get(mat_name, None)
        if entry is None:
            ridge_mat_names.append(mat_name)
            continue

        ridge_mat_names.append(mat_name)
        expert_consensus[idx] = entry['expert_consensus_freq']

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"    Processing {idx+1}/{N} ({elapsed:.0f}s)")

        try:
            data, fs = load_eeg_data(entry)
            if data is None:
                continue

            f_A = get_f_A(data, fs)
            f_B = get_f_B(data, fs)

            seg = preprocess_segment_r7(data, fs)

            # Broadband bipolar for spectral coherence
            from mne.filter import notch_filter, filter_data
            seg_bb = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
            seg_bb = filter_data(seg_bb, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg_bb = np.array(fcn_getBanana(seg_bb))

            traces = compute_pointiness_traces(seg, fs)

            features[idx, 0] = f_A
            features[idx, 1] = f_B
            features[idx, 2] = get_f_peaks(traces, fs)
            features[idx, 3] = get_f_fft(traces, fs)
            features[idx, 4] = get_f_tkeo_fft(seg, fs)
            features[idx, 5] = get_f_spectral_coh(seg_bb, fs)
            features[idx, 6] = get_f_hps3(traces, fs)
            features[idx, 7] = 1.0 if entry['subdir'] == 'gpd' else 0.0

        except Exception as e:
            continue

    elapsed = time.time() - t0
    print(f"  Feature extraction done in {elapsed:.0f}s")

    # ── Ridge baseline with same patient-level GroupKFold ─────────────
    print("\n[3] Running ridge baseline (patient-level 5-fold CV)")

    valid_mask = np.isfinite(expert_consensus) & (expert_consensus > 0)
    valid_idx = np.where(valid_mask)[0]
    print(f"  Valid samples for ridge: {len(valid_idx)}")

    if len(valid_idx) < 10:
        print("  ERROR: Too few valid samples for ridge regression.")
        sys.exit(1)

    X_ridge = features[valid_idx].copy()
    y_ridge = np.log(expert_consensus[valid_idx])
    groups_ridge = patients[valid_idx]
    X_ridge = impute_nan_median(X_ridge)

    ridge_preds_log = ridge_grouped_cv(X_ridge, y_ridge, groups_ridge, n_splits=5, alpha=1.0)
    ridge_preds = np.exp(ridge_preds_log)

    # Map back to full array
    ridge_full = np.full(N, np.nan)
    for i, vi in enumerate(valid_idx):
        if np.isfinite(ridge_preds[i]):
            ridge_full[vi] = ridge_preds[i]

    n_valid_ridge = np.sum(np.isfinite(ridge_full))
    print(f"  Ridge predictions: {n_valid_ridge}")

    # ── Build prediction dicts ────────────────────────────────────────
    print("\n[4] Building prediction dicts and evaluating")

    cnn_dict = {}
    ridge_dict = {}
    hybrid_dict = {}

    for idx in range(N):
        mat_name = str(pred_mat_names[idx])
        if np.isfinite(cnn_predictions[idx]):
            cnn_dict[mat_name] = float(cnn_predictions[idx])
        if np.isfinite(ridge_full[idx]):
            ridge_dict[mat_name] = float(ridge_full[idx])
        # Hybrid: average CNN and ridge where both available
        if np.isfinite(cnn_predictions[idx]) and np.isfinite(ridge_full[idx]):
            hybrid_dict[mat_name] = float((cnn_predictions[idx] + ridge_full[idx]) / 2.0)
        elif np.isfinite(cnn_predictions[idx]):
            hybrid_dict[mat_name] = float(cnn_predictions[idx])
        elif np.isfinite(ridge_full[idx]):
            hybrid_dict[mat_name] = float(ridge_full[idx])

    print(f"  CNN:    {len(cnn_dict)} predictions")
    print(f"  Ridge:  {len(ridge_dict)} predictions")
    print(f"  Hybrid: {len(hybrid_dict)} predictions")

    # ── Evaluate each method ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Method 1: CNN (Phase 2)")
    cnn_metrics = evaluate_predictions(dataset, cnn_dict, 'dl_cnn_phase2')

    print("\n" + "=" * 60)
    print("Method 2: Ridge Baseline (same patient folds)")
    ridge_metrics = evaluate_predictions(dataset, ridge_dict, 'dl_ridge_baseline')

    print("\n" + "=" * 60)
    print("Method 3: Hybrid (CNN + Ridge average)")
    hybrid_metrics = evaluate_predictions(dataset, hybrid_dict, 'dl_hybrid_cnn_ridge')

    # ── CNN eventness as ridge feature ────────────────────────────────
    print("\n" + "=" * 60)
    print("Method 4: Ridge + CNN eventness features")

    # Extract features from weak eventness traces (used as CNN eventness proxy)
    eventness_feats = np.full((N, 4), np.nan)
    for idx in range(N):
        if not valid_mask[idx]:
            continue
        ev_trace = weak_eventness[idx]
        ef = extract_eventness_features(ev_trace, FS)
        eventness_feats[idx, 0] = ef.get('eventness_fft_freq', np.nan)
        eventness_feats[idx, 1] = ef.get('eventness_fft_power', np.nan)
        eventness_feats[idx, 2] = ef.get('eventness_peak_freq', np.nan)
        eventness_feats[idx, 3] = ef.get('eventness_mean', np.nan)

    # Combine standard features + eventness features
    X_augmented = np.hstack([features[valid_idx], eventness_feats[valid_idx]])
    X_augmented = impute_nan_median(X_augmented)

    aug_preds_log = ridge_grouped_cv(X_augmented, y_ridge, groups_ridge, n_splits=5, alpha=1.0)
    aug_preds = np.exp(aug_preds_log)

    aug_dict = {}
    for i, vi in enumerate(valid_idx):
        if np.isfinite(aug_preds[i]):
            aug_dict[str(pred_mat_names[vi])] = float(aug_preds[i])

    aug_metrics = evaluate_predictions(dataset, aug_dict, 'dl_ridge_plus_eventness')

    # ── Comparison Table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    methods = [
        ("CNN (Phase 2)", cnn_metrics),
        ("Ridge Baseline", ridge_metrics),
        ("Hybrid CNN+Ridge", hybrid_metrics),
        ("Ridge+Eventness", aug_metrics),
    ]

    header = f"{'Method':>25s}  {'LPD MAE':>8s}  {'GPD MAE':>8s}  {'LPD Sp.r':>8s}  {'GPD Sp.r':>8s}  {'Comb Sp.':>8s}"
    print(header)
    print("-" * len(header))

    for name, m in methods:
        lpd_mae = m.get('lpd_mae', '?')
        gpd_mae = m.get('gpd_mae', '?')
        lpd_sp = m.get('lpd_spearman_r', '?')
        gpd_sp = m.get('gpd_spearman_r', '?')
        comb = m.get('combined_spearman', '?')
        print(f"{name:>25s}  {lpd_mae:>8}  {gpd_mae:>8}  {lpd_sp:>8}  {gpd_sp:>8}  {comb:>8}")

    # ── Update dashboard ──────────────────────────────────────────────
    print("\n[5] Updating dashboard")
    try:
        sys.path.insert(0, str(CODE_DIR))
        from update_dashboard import update
        update()
    except Exception as e:
        print(f"  Warning: dashboard update failed: {e}")
        print("  You can manually run: python code/update_dashboard.py")

    print("\nDone!")


if __name__ == '__main__':
    main()
