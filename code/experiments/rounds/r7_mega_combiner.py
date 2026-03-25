"""
Round 7 Mega Combiner: combines best R6 features (TKEO, HPS, spectral coherence)
with best R7 features (multi-montage, per-expert training, spatial selection)
into one optimal model.

Target: beat r7_expert_spatial_ridge Spearman 0.471.
"""

import sys
import os
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate, fcn_getBanana, bipolar_channels, mono_channels
from pd_pointiness_acf import pd_detect_pointiness_acf, compute_pointiness_trace
from scipy.signal import find_peaks, butter, filtfilt, coherence
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = CODE_DIR.parent / 'data'

# Load templates
templates_lpd = np.load(str(DATA_DIR / 'templates_C_lpd.npy'))
templates_gpd = np.load(str(DATA_DIR / 'templates_C_gpd.npy'))

# Adjacent channel pairs for spectral coherence
ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),       # left temporal chain
    (4, 5), (5, 6), (6, 7),       # right temporal chain
    (8, 9), (9, 10), (10, 11),    # left parasagittal chain
    (12, 13), (13, 14), (14, 15), # right parasagittal chain
    (16, 17),                      # midline chain
]

FREQ_LO, FREQ_HI = 0.3, 3.5


def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
    """FFT of a 1D trace, return peak frequency in [freq_lo, freq_hi] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freqs_sub = freqs[mask]
    if np.max(fft_sub) == 0:
        return np.nan
    return freqs_sub[np.argmax(fft_sub)]


def compute_all_features(data, fs, entry):
    """Compute all features for one segment. Returns dict of feature values."""
    features = {}
    is_gpd = 1 if entry['subdir'] == 'gpd' else 0
    features['is_gpd'] = is_gpd

    # --- Method A: pd_detect_alternate ---
    try:
        res_A = pd_detect_alternate(data.copy(), fs, pk_detect='apd')
        f_A = res_A['event_frequency']
        if not np.isfinite(f_A):
            f_A = np.nan
    except:
        f_A = np.nan
    features['f_A'] = f_A

    # --- Method B: pd_detect_pointiness_acf ---
    try:
        res_B = pd_detect_pointiness_acf(
            data.copy(), fs,
            method='pointiness', lowpass_hz=15,
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        f_B = res_B['event_frequency']
        if not np.isfinite(f_B):
            f_B = np.nan
        detected_channels = res_B.get('channels', [])
        if detected_channels is None or (isinstance(detected_channels, float) and np.isnan(detected_channels)):
            detected_channels = []
        n_detected = len(detected_channels)
        # Get detected channel indices
        detected_indices = []
        for ch_name in detected_channels:
            if ch_name in bipolar_channels:
                detected_indices.append(bipolar_channels.index(ch_name))
    except:
        f_B = np.nan
        n_detected = 0
        detected_indices = []
    features['f_B'] = f_B
    features['n_detected'] = n_detected

    # --- Preprocessing for feature extraction ---
    # Notch + bandpass
    seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")

    # Bipolar montage
    seg_bip = np.array(fcn_getBanana(seg_filtered))

    # 15Hz lowpass for pointiness-based features
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(seg_bip.shape[0]):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    n_channels = seg_lp.shape[0]
    sigma_samples = max(1, int(0.02 * fs))

    # --- Compute pointiness traces for all channels ---
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # --- f_peaks: peak-count on pointiness ---
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # --- f_fft: FFT of pointiness, median across channels ---
    fft_freqs = []
    for ch in range(n_channels):
        f = compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    features['f_fft'] = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # --- f_envelope: matched-filter envelope FFT ---
    templates = templates_gpd if is_gpd else templates_lpd
    envelope_freqs = []
    for ch in range(min(n_channels, seg_lp.shape[0])):
        best_env = None
        best_corr = -1
        for t_idx in range(templates.shape[0]):
            templ = templates[t_idx]
            if len(templ) > len(seg_lp[ch]):
                continue
            # Cross-correlation envelope
            env = np.abs(np.correlate(seg_lp[ch], templ, mode='same'))
            corr_val = np.max(env)
            if corr_val > best_corr:
                best_corr = corr_val
                best_env = env
        if best_env is not None:
            f = compute_fft_peak(best_env, fs)
            if np.isfinite(f):
                envelope_freqs.append(f)
    features['f_envelope'] = float(np.median(envelope_freqs)) if envelope_freqs else np.nan

    # --- R6 features ---

    # f_tkeo: TKEO on bipolar signal → smooth → FFT
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1]**2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_hps3: Harmonic Product Spectrum on pointiness FFT
    hps_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        n_pts = len(pt)
        if n_pts < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(pt - np.mean(pt)))
        freqs_arr = np.fft.rfftfreq(n_pts, d=1.0/fs)
        # HPS: multiply P(f) * P(2f) * P(3f)
        max_idx = len(fft_vals) // 3
        if max_idx < 2:
            continue
        hps = fft_vals[:max_idx] * fft_vals[::2][:max_idx] * fft_vals[::3][:max_idx]
        hps_freqs_arr = freqs_arr[:max_idx]
        mask = (hps_freqs_arr >= FREQ_LO) & (hps_freqs_arr <= FREQ_HI)
        if not np.any(mask):
            continue
        hps_sub = hps[mask]
        freqs_sub = hps_freqs_arr[mask]
        if np.max(hps_sub) > 0:
            hps_freqs.append(freqs_sub[np.argmax(hps_sub)])
    features['f_hps3'] = float(np.median(hps_freqs)) if hps_freqs else np.nan

    # f_spectral_coh: spectral coherence between adjacent channel pairs
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs, nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except:
            continue
    features['f_spectral_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # --- R7 multi-montage features ---
    # Use first 19 channels of raw referential data
    n_ref = min(19, data.shape[0])

    # Common Average Reference montage
    seg_ref = seg_filtered[:n_ref]
    car_mean = np.mean(seg_ref, axis=0)
    seg_car = seg_ref - car_mean
    # Lowpass
    seg_car_lp = np.zeros_like(seg_car)
    for ch in range(n_ref):
        try:
            seg_car_lp[ch] = filtfilt(b_lp, a_lp, seg_car[ch])
        except:
            seg_car_lp[ch] = seg_car[ch]
    car_fft_freqs = []
    for ch in range(n_ref):
        pt = compute_pointiness_trace(seg_car_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        f = compute_fft_peak(pt, fs)
        if np.isfinite(f):
            car_fft_freqs.append(f)
    features['f_fft_car'] = float(np.median(car_fft_freqs)) if car_fft_freqs else np.nan

    # Laplacian approximation montage
    lap_fft_freqs = []
    for ch in range(1, n_ref - 1):
        lap_ch = seg_ref[ch] - 0.5 * (seg_ref[ch-1] + seg_ref[ch+1])
        try:
            lap_ch_lp = filtfilt(b_lp, a_lp, lap_ch)
        except:
            lap_ch_lp = lap_ch
        pt = compute_pointiness_trace(lap_ch_lp)
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        f = compute_fft_peak(pt, fs)
        if np.isfinite(f):
            lap_fft_freqs.append(f)
    features['f_fft_lap'] = float(np.median(lap_fft_freqs)) if lap_fft_freqs else np.nan

    # --- Spatial selection features (only on detected channels) ---
    if detected_indices:
        det_fft_freqs = []
        det_tkeo_freqs = []
        for ch in detected_indices:
            if ch >= n_channels:
                continue
            # FFT of pointiness on detected channels
            f = compute_fft_peak(pointiness_traces[ch], fs)
            if np.isfinite(f):
                det_fft_freqs.append(f)
            # TKEO on detected channels
            x = seg_lp[ch]
            if len(x) >= 3:
                tkeo = np.abs(x[1:-1]**2 - x[:-2] * x[2:])
                tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
                f_t = compute_fft_peak(tkeo_smooth, fs)
                if np.isfinite(f_t):
                    det_tkeo_freqs.append(f_t)
        features['f_fft_detected'] = float(np.median(det_fft_freqs)) if det_fft_freqs else np.nan
        features['f_tkeo_detected'] = float(np.median(det_tkeo_freqs)) if det_tkeo_freqs else np.nan
    else:
        features['f_fft_detected'] = np.nan
        features['f_tkeo_detected'] = np.nan

    return features


def ridge_loo(X, y, alpha=1.0):
    """Leave-one-out Ridge regression. Returns predictions array."""
    n, p = X.shape
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        XtX = X_train.T @ X_train + alpha * np.eye(p)
        try:
            beta = np.linalg.solve(XtX, X_train.T @ y_train)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtX, X_train.T @ y_train, rcond=None)[0]
        preds[i] = X[i] @ beta
    return preds


def get_ridge_coefficients(X, y, alpha=1.0):
    """Fit Ridge on all data, return coefficients."""
    n, p = X.shape
    XtX = X.T @ X + alpha * np.eye(p)
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    return beta


def prepare_features(feature_dicts, feature_names):
    """Build feature matrix from list of feature dicts. NaN → column median."""
    n = len(feature_dicts)
    p = len(feature_names)
    X = np.full((n, p), np.nan)
    for i, fd in enumerate(feature_dicts):
        for j, fn in enumerate(feature_names):
            X[i, j] = fd.get(fn, np.nan)
    # NaN → column median
    for j in range(p):
        col = X[:, j]
        finite_mask = np.isfinite(col)
        if np.any(finite_mask):
            med = np.median(col[finite_mask])
            col[~finite_mask] = med
        else:
            col[:] = 0.0
    return X


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    ALL_FEATURE_NAMES = [
        'f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope',
        'f_tkeo', 'f_hps3', 'f_spectral_coh',
        'f_fft_car', 'f_fft_lap',
        'f_fft_detected', 'f_tkeo_detected',
        'is_gpd', 'n_detected',
    ]

    NO_A_FEATURES = [f for f in ALL_FEATURE_NAMES if f != 'f_A']

    # Compute features for all segments
    all_features = []
    mat_names = []
    expert_freqs = []  # consensus
    expert_LB = []
    expert_PH = []
    expert_SZ = []

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing segment {idx+1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            all_features.append(None)
            mat_names.append(entry['mat_name'])
            expert_freqs.append(entry['expert_consensus_freq'])
            expert_LB.append(entry.get('expert_LB_freq', np.nan))
            expert_PH.append(entry.get('expert_PH_freq', np.nan))
            expert_SZ.append(entry.get('expert_SZ_freq', np.nan))
            continue

        feats = compute_all_features(data, fs, entry)
        all_features.append(feats)
        mat_names.append(entry['mat_name'])
        expert_freqs.append(entry['expert_consensus_freq'])
        expert_LB.append(entry.get('expert_LB_freq', np.nan))
        expert_PH.append(entry.get('expert_PH_freq', np.nan))
        expert_SZ.append(entry.get('expert_SZ_freq', np.nan))

    print(f"\nFeature extraction complete. {sum(1 for f in all_features if f is not None)} segments with features.")

    # Filter valid segments
    valid_mask = [f is not None for f in all_features]
    valid_features = [f for f in all_features if f is not None]
    valid_mat_names = [m for m, v in zip(mat_names, valid_mask) if v]
    valid_expert = np.array([e for e, v in zip(expert_freqs, valid_mask) if v])
    valid_LB = np.array([e for e, v in zip(expert_LB, valid_mask) if v])
    valid_PH = np.array([e for e, v in zip(expert_PH, valid_mask) if v])
    valid_SZ = np.array([e for e, v in zip(expert_SZ, valid_mask) if v])

    n_valid = len(valid_features)
    print(f"Valid segments: {n_valid}")

    # Print feature coverage
    for fn in ALL_FEATURE_NAMES:
        n_finite = sum(1 for f in valid_features if np.isfinite(f.get(fn, np.nan)))
        print(f"  {fn}: {n_finite}/{n_valid} finite ({100*n_finite/n_valid:.1f}%)")

    # ============================================================
    # Model evaluations
    # ============================================================

    def run_ridge_model(feature_names, alpha, experiment_name):
        """Run standard Ridge LOO-CV model."""
        print(f"\n--- {experiment_name} (alpha={alpha}) ---")
        X = prepare_features(valid_features, feature_names)
        y = np.log(valid_expert)

        preds_log = ridge_loo(X, y, alpha=alpha)
        preds = np.exp(preds_log)

        predictions = {m: float(p) for m, p in zip(valid_mat_names, preds)}
        metrics = evaluate_predictions(dataset, predictions, experiment_name)

        # Print coefficients
        beta = get_ridge_coefficients(X, y, alpha=alpha)
        print(f"\n  Feature coefficients:")
        for fn, b in zip(feature_names, beta):
            print(f"    {fn:>20s}: {b:+.4f}")

        return metrics

    def run_per_expert_model(feature_names, alpha, experiment_name):
        """Per-expert Ridge: train 3 models, average predictions."""
        print(f"\n--- {experiment_name} (alpha={alpha}, per-expert) ---")
        X_all = prepare_features(valid_features, feature_names)

        expert_arrays = {
            'LB': valid_LB,
            'PH': valid_PH,
            'SZ': valid_SZ,
        }

        # For each segment, collect predictions from each expert model
        segment_preds = np.full((n_valid, 3), np.nan)

        for e_idx, (expert_name, expert_vals) in enumerate(expert_arrays.items()):
            # Only use segments where this expert rated freq > 0
            expert_mask = np.isfinite(expert_vals) & (expert_vals > 0)
            expert_indices = np.where(expert_mask)[0]
            n_expert = len(expert_indices)

            if n_expert < 5:
                print(f"  Expert {expert_name}: only {n_expert} segments, skipping")
                continue

            X_expert = X_all[expert_indices]
            y_expert = np.log(expert_vals[expert_indices])

            # LOO-CV within this expert's subset
            preds_log = ridge_loo(X_expert, y_expert, alpha=alpha)

            # Place predictions back
            for local_i, global_i in enumerate(expert_indices):
                segment_preds[global_i, e_idx] = np.exp(preds_log[local_i])

            # For segments NOT in this expert's subset, predict using full model
            beta_full = get_ridge_coefficients(X_expert, y_expert, alpha=alpha)
            non_expert_indices = np.where(~expert_mask)[0]
            for gi in non_expert_indices:
                segment_preds[gi, e_idx] = np.exp(X_all[gi] @ beta_full)

        # Average across expert models (nanmean handles missing)
        final_preds = np.nanmean(segment_preds, axis=1)

        predictions = {m: float(p) for m, p in zip(valid_mat_names, final_preds)}
        metrics = evaluate_predictions(dataset, predictions, experiment_name)

        # Print coefficients (fit on full data with first expert for reference)
        beta = get_ridge_coefficients(X_all, np.log(valid_expert), alpha=alpha)
        print(f"\n  Feature coefficients (consensus fit):")
        for fn, b in zip(feature_names, beta):
            print(f"    {fn:>20s}: {b:+.4f}")

        return metrics

    # Run all models
    results = {}

    # a) All features, alpha=1
    results['a'] = run_ridge_model(ALL_FEATURE_NAMES, alpha=1.0, experiment_name='r7_mega_ridge')

    # b) All features, alpha=5
    results['b'] = run_ridge_model(ALL_FEATURE_NAMES, alpha=5.0, experiment_name='r7_mega_ridge_alpha5')

    # c) Per-expert, all features, alpha=1
    results['c'] = run_per_expert_model(ALL_FEATURE_NAMES, alpha=1.0, experiment_name='r7_mega_per_expert')

    # d) Per-expert, all features, alpha=5
    results['d'] = run_per_expert_model(ALL_FEATURE_NAMES, alpha=5.0, experiment_name='r7_mega_per_expert_alpha5')

    # e) No f_A, alpha=1
    results['e'] = run_ridge_model(NO_A_FEATURES, alpha=1.0, experiment_name='r7_mega_no_A')

    # f) Per-expert, no f_A
    results['f'] = run_per_expert_model(NO_A_FEATURES, alpha=1.0, experiment_name='r7_mega_per_expert_no_A')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL MODELS")
    print("=" * 70)
    print(f"{'Model':<35s} {'Comb Spearman':>14s} {'Comb MAE':>10s} {'LPD rs':>8s} {'GPD rs':>8s}")
    print("-" * 70)
    for key in ['a', 'b', 'c', 'd', 'e', 'f']:
        m = results[key]
        name = m.get('experiment', '?')
        cs = m.get('combined_spearman', np.nan)
        cm = m.get('combined_mae', np.nan)
        lr = m.get('lpd_spearman_pooled', m.get('lpd_spearman_r', np.nan))
        gr = m.get('gpd_spearman_pooled', m.get('gpd_spearman_r', np.nan))
        print(f"  {name:<33s} {cs:>14} {cm:>10} {lr:>8} {gr:>8}")
    print(f"\n  Baseline: r7_expert_spatial_ridge = 0.471 combined Spearman")

    # Find best model
    best_key = max(results.keys(), key=lambda k: results[k].get('combined_spearman', -999) if np.isfinite(results[k].get('combined_spearman', np.nan)) else -999)
    best = results[best_key]
    print(f"\n  BEST: {best['experiment']} with combined Spearman = {best.get('combined_spearman', '?')}")

    # Print coefficients for best model
    best_name = best['experiment']
    print(f"\n  Coefficients for {best_name} printed above.")
