"""
Round 7: Expert-distribution training & spatial-extent channel selection.

Approaches:
  A. Per-expert ridge models (train to each expert, average predictions)
  B. Soft-label training (target = mean of log expert freqs)
  C. Spatial-extent-guided channel selection (only detected channels)
  D. Laterality-aware channel selection (dominant hemisphere only)
  E. Combined: per-expert training + spatial features + standard features
"""

import sys
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    pd_detect_pointiness_acf, bipolar_channels,
    left_indices, right_indices,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks, coherence
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
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


# ── Feature extractors (standard) ─────────────────────────────────────
def get_f_A(data, fs):
    """Method A: pd_detect_alternate(apd) event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def get_f_B_full(data, fs):
    """Method B: pd_detect_pointiness_acf. Returns (freq, n_ch, result_dict)."""
    try:
        r = pd_detect_pointiness_acf(
            data, fs, method='pointiness',
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            smoothing_sigma=SMOOTHING_SIGMA,
            lowpass_hz=LOWPASS_HZ,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch, r
        return float(f), n_ch, r
    except Exception:
        return np.nan, 0, {}


def compute_pointiness_traces(seg, fs):
    """Compute smoothed pointiness trace per channel."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    traces = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return np.array(traces)


def get_f_peaks(traces, fs, ch_mask=None):
    """Peak-count frequency. If ch_mask given, only use those channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        if ch_mask is not None and not ch_mask[i]:
            continue
        trace = traces[i]
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return median_finite(freqs)


def get_f_fft(traces, fs, ch_mask=None):
    """FFT of pointiness. If ch_mask given, only use those channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        if ch_mask is not None and not ch_mask[i]:
            continue
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_tkeo_fft(seg, fs, ch_mask=None):
    """FFT of |TKEO| trace."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(0.02 * fs))
    for i in range(n_ch):
        if ch_mask is not None and not ch_mask[i]:
            continue
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
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_spectral_coh(seg, fs):
    """Spectral coherence peak across adjacent channel pairs."""
    nperseg = min(256, seg.shape[1] // 2)
    if nperseg < 16:
        return np.nan
    coh_spectra = []
    coh_freqs = None
    for (a, b) in ADJACENT_PAIRS:
        if a >= seg.shape[0] or b >= seg.shape[0]:
            continue
        try:
            f_coh, Cxy = coherence(seg[a], seg[b], fs=fs, nperseg=nperseg)
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
    coh_sub = coh_spectra[mask]
    freq_sub = coh_freqs[mask]
    return float(freq_sub[np.argmax(coh_sub)])


def get_f_hps3(traces, fs):
    """Harmonic Product Spectrum on pointiness FFT. Median across channels."""
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
        hps_sub = hps[mask]
        freq_sub = hps_freqs[mask]
        freqs[i] = freq_sub[np.argmax(hps_sub)]
    return median_finite(freqs)


# ── Ridge regression LOO-CV ───────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """Leave-one-out ridge regression. Returns predictions array."""
    n = X.shape[0]
    predictions = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i+1]
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_tr_s = (X_train - mu) / std
        X_te_s = (X_test - mu) / std
        y_mu = np.mean(y_train)
        y_c = y_train - y_mu
        p = X_tr_s.shape[1]
        A = X_tr_s.T @ X_tr_s + alpha * np.eye(p)
        try:
            w = np.linalg.solve(A, X_tr_s.T @ y_c)
        except np.linalg.LinAlgError:
            continue
        pred = X_te_s @ w + y_mu
        predictions[i] = pred[0]
    return predictions


def impute_nan_median(X):
    """Replace NaN with column median. If entire column NaN, use 1.0."""
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


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    N = len(dataset)
    print(f"Dataset: {N} segments")

    # Standard feature names
    STANDARD_FEAT_NAMES = [
        'f_A', 'f_B', 'f_peaks', 'f_fft', 'f_tkeo_fft',
        'f_envelope_placeholder', 'f_spectral_coh', 'f_hps3', 'is_gpd', 'n_ch',
    ]
    # Spatial feature names
    SPATIAL_FEAT_NAMES = [
        'f_fft_spatial', 'f_tkeo_spatial', 'f_peaks_spatial',
        'f_fft_top5', 'f_tkeo_top5', 'f_peaks_top5',
    ]
    # Lateral feature names
    LATERAL_FEAT_NAMES = [
        'f_fft_lateral', 'f_tkeo_lateral', 'f_peaks_lateral',
    ]

    n_std = len(STANDARD_FEAT_NAMES)
    n_spatial = len(SPATIAL_FEAT_NAMES)
    n_lateral = len(LATERAL_FEAT_NAMES)
    n_total = n_std + n_spatial + n_lateral

    features = np.full((N, n_total), np.nan)
    expert_consensus = np.full(N, np.nan)
    expert_LB = np.full(N, np.nan)
    expert_PH = np.full(N, np.nan)
    expert_SZ = np.full(N, np.nan)
    mat_names = []
    subdirs = []

    t0 = time.time()
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing segment {idx+1}/{N}  ({elapsed:.0f}s elapsed)")

        mat_names.append(entry['mat_name'])
        subdirs.append(entry['subdir'])
        expert_consensus[idx] = entry['expert_consensus_freq']
        expert_LB[idx] = entry.get('expert_LB_freq', np.nan)
        expert_PH[idx] = entry.get('expert_PH_freq', np.nan)
        expert_SZ[idx] = entry.get('expert_SZ_freq', np.nan)

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        # ─ Method A ───────────────────────────────────────────
        f_A, n_ch_A = get_f_A(data, fs)
        features[idx, 0] = f_A

        # ─ Method B (full result for spatial info) ────────────
        f_B, n_ch_B, result_B = get_f_B_full(data, fs)
        features[idx, 1] = f_B
        features[idx, 9] = n_ch_B  # n_ch

        # ─ Standard preprocessing ────────────────────────────
        seg = preprocess_segment(data, fs)

        # Broadband bipolar (no 15Hz LP) for spectral coherence
        seg_bb = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
        seg_bb = filter_data(seg_bb, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        seg_bb = np.array(fcn_getBanana(seg_bb))

        # Pointiness traces
        traces = compute_pointiness_traces(seg, fs)

        # ─ Standard features ─────────────────────────────────
        features[idx, 2] = get_f_peaks(traces, fs)          # f_peaks
        features[idx, 3] = get_f_fft(traces, fs)            # f_fft
        features[idx, 4] = get_f_tkeo_fft(seg, fs)          # f_tkeo_fft
        # f_envelope_placeholder (skip for speed, not needed for this experiment)
        features[idx, 5] = np.nan
        features[idx, 6] = get_f_spectral_coh(seg_bb, fs)   # f_spectral_coh
        features[idx, 7] = get_f_hps3(traces, fs)           # f_hps3
        features[idx, 8] = 1.0 if entry['subdir'] == 'gpd' else 0.0  # is_gpd

        # ─ Spatial channel selection (Approach C) ─────────────
        # Use Method B's channel_frequencies to identify involved channels
        ch_freqs_B = result_B.get('channel_frequencies', {})
        ch_scores_B = result_B.get('channel_pd_scores', {})

        # Mask: channels where Method B detected periodicity (finite freq)
        spatial_mask = np.zeros(18, dtype=bool)
        for ci, ch_name in enumerate(bipolar_channels):
            f_ch = ch_freqs_B.get(ch_name, np.nan)
            if np.isfinite(f_ch):
                spatial_mask[ci] = True

        # For GPD: use all channels
        is_gpd = entry['subdir'] == 'gpd'
        if is_gpd:
            effective_spatial_mask = np.ones(18, dtype=bool)
        else:
            effective_spatial_mask = spatial_mask
            # If no channels detected, fall back to all
            if not np.any(effective_spatial_mask):
                effective_spatial_mask = np.ones(18, dtype=bool)

        features[idx, n_std + 0] = get_f_fft(traces, fs, ch_mask=effective_spatial_mask)
        features[idx, n_std + 1] = get_f_tkeo_fft(seg, fs, ch_mask=effective_spatial_mask)
        features[idx, n_std + 2] = get_f_peaks(traces, fs, ch_mask=effective_spatial_mask)

        # Top 5 channels by ACF score
        scores_arr = np.array([ch_scores_B.get(ch_name, 0.0) for ch_name in bipolar_channels])
        scores_arr = np.where(np.isfinite(scores_arr), scores_arr, 0.0)
        top5_mask = np.zeros(18, dtype=bool)
        top5_idx = np.argsort(scores_arr)[::-1][:5]
        if scores_arr[top5_idx[0]] > 0:
            top5_mask[top5_idx] = True
        else:
            top5_mask[:] = True  # fallback

        features[idx, n_std + 3] = get_f_fft(traces, fs, ch_mask=top5_mask)
        features[idx, n_std + 4] = get_f_tkeo_fft(seg, fs, ch_mask=top5_mask)
        features[idx, n_std + 5] = get_f_peaks(traces, fs, ch_mask=top5_mask)

        # ─ Laterality-aware channel selection (Approach D) ────
        left_score = result_B.get('left_mean_score', 0.0)
        right_score = result_B.get('right_mean_score', 0.0)
        if not np.isfinite(left_score):
            left_score = 0.0
        if not np.isfinite(right_score):
            right_score = 0.0

        if is_gpd:
            lateral_mask = np.ones(18, dtype=bool)
        else:
            lateral_mask = np.zeros(18, dtype=bool)
            if left_score >= right_score:
                for ci in left_indices:
                    lateral_mask[ci] = True
            else:
                for ci in right_indices:
                    lateral_mask[ci] = True
            # Also include midline channels 16, 17
            lateral_mask[16] = True
            lateral_mask[17] = True

        features[idx, n_std + n_spatial + 0] = get_f_fft(traces, fs, ch_mask=lateral_mask)
        features[idx, n_std + n_spatial + 1] = get_f_tkeo_fft(seg, fs, ch_mask=lateral_mask)
        features[idx, n_std + n_spatial + 2] = get_f_peaks(traces, fs, ch_mask=lateral_mask)

    elapsed = time.time() - t0
    print(f"Feature extraction done in {elapsed:.0f}s")

    # ── Print feature coverage ────────────────────────────────────────
    ALL_FEAT_NAMES = STANDARD_FEAT_NAMES + SPATIAL_FEAT_NAMES + LATERAL_FEAT_NAMES
    print("\nFeature coverage (non-NaN counts):")
    for fi, fname in enumerate(ALL_FEAT_NAMES):
        cnt = np.sum(np.isfinite(features[:, fi]))
        print(f"  {fname:>25s}: {cnt}/{N}")

    # ── Helper: build feature matrix with imputation ──────────────────
    def build_Xy(feat_indices, target, require_target=True):
        """Select features, impute NaN, return X, y, valid mask."""
        X_raw = features[:, feat_indices].copy()
        y_raw = target.copy()
        valid = np.ones(N, dtype=bool)
        if require_target:
            valid &= np.isfinite(y_raw) & (y_raw > 0)
        X = X_raw[valid].copy()
        y = y_raw[valid].copy()
        X = impute_nan_median(X)
        return X, y, valid

    # ── Standard feature indices (skip f_envelope_placeholder) ────────
    std_idx = [0, 1, 2, 3, 4, 6, 7, 8, 9]  # f_A, f_B, f_peaks, f_fft, f_tkeo_fft, f_spectral_coh, f_hps3, is_gpd, n_ch

    # ==================================================================
    # APPROACH A: Per-expert ridge models
    # ==================================================================
    print(f"\n{'='*60}")
    print("Approach A: Per-expert ridge models")

    expert_targets = {
        'LB': expert_LB,
        'PH': expert_PH,
        'SZ': expert_SZ,
    }

    # For each expert, train a ridge model on log(freq), LOO-CV
    per_expert_preds = {}  # expert_name -> array of predictions (N,)
    for ename, etarget in expert_targets.items():
        print(f"  Training model for expert {ename}...")
        X, y, valid = build_Xy(std_idx, etarget, require_target=True)
        if len(y) < 5:
            print(f"    Skipping {ename}: only {len(y)} valid samples")
            continue
        y_log = np.log(y)
        preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
        preds = np.exp(preds_log)

        # Store back into full-size array
        full_preds = np.full(N, np.nan)
        valid_idx = np.where(valid)[0]
        for i, vi in enumerate(valid_idx):
            if np.isfinite(preds[i]):
                full_preds[vi] = preds[i]
        per_expert_preds[ename] = full_preds
        print(f"    {ename}: {np.sum(np.isfinite(full_preds))} predictions")

    # r7_per_expert_mean: average of available per-expert predictions
    pred_dict_per_expert = {}
    for idx_i in range(N):
        vals = [per_expert_preds[e][idx_i] for e in per_expert_preds
                if np.isfinite(per_expert_preds[e][idx_i])]
        if len(vals) > 0:
            pred_dict_per_expert[mat_names[idx_i]] = float(np.mean(vals))

    print(f"  Per-expert mean: {len(pred_dict_per_expert)} predictions")
    evaluate_predictions(dataset, pred_dict_per_expert, 'r7_per_expert_mean')

    # ==================================================================
    # APPROACH B: Soft-label training (mean of log expert freqs)
    # ==================================================================
    print(f"\n{'='*60}")
    print("Approach B: Soft-label training")

    # Compute mean-log-freq target
    soft_target = np.full(N, np.nan)
    for idx_i in range(N):
        log_vals = []
        for ef in [expert_LB[idx_i], expert_PH[idx_i], expert_SZ[idx_i]]:
            if np.isfinite(ef) and ef > 0:
                log_vals.append(np.log(ef))
        if log_vals:
            soft_target[idx_i] = np.exp(np.mean(log_vals))  # store as freq, will re-log below

    X_soft, y_soft, valid_soft = build_Xy(std_idx, soft_target, require_target=True)
    y_soft_log = np.log(y_soft)
    preds_soft_log = ridge_loo_cv(X_soft, y_soft_log, alpha=1.0)
    preds_soft = np.exp(preds_soft_log)

    pred_dict_soft = {}
    valid_soft_idx = np.where(valid_soft)[0]
    for i, vi in enumerate(valid_soft_idx):
        if np.isfinite(preds_soft[i]):
            pred_dict_soft[mat_names[vi]] = float(preds_soft[i])

    evaluate_predictions(dataset, pred_dict_soft, 'r7_soft_label')

    # ==================================================================
    # APPROACH C: Spatial-extent-guided channel selection
    # ==================================================================
    print(f"\n{'='*60}")
    print("Approach C: Spatial-extent-guided channel selection")

    # r7_spatial_select: ridge with spatial-selected frequency features
    spatial_idx = std_idx + [n_std + 0, n_std + 1, n_std + 2]
    X_sp, y_sp, valid_sp = build_Xy(spatial_idx, expert_consensus, require_target=True)
    y_sp_log = np.log(y_sp)
    preds_sp_log = ridge_loo_cv(X_sp, y_sp_log, alpha=1.0)
    preds_sp = np.exp(preds_sp_log)

    pred_dict_spatial = {}
    valid_sp_idx = np.where(valid_sp)[0]
    for i, vi in enumerate(valid_sp_idx):
        if np.isfinite(preds_sp[i]):
            pred_dict_spatial[mat_names[vi]] = float(preds_sp[i])

    evaluate_predictions(dataset, pred_dict_spatial, 'r7_spatial_select')

    # r7_spatial_top5: top 5 channels
    top5_idx = std_idx + [n_std + 3, n_std + 4, n_std + 5]
    X_t5, y_t5, valid_t5 = build_Xy(top5_idx, expert_consensus, require_target=True)
    y_t5_log = np.log(y_t5)
    preds_t5_log = ridge_loo_cv(X_t5, y_t5_log, alpha=1.0)
    preds_t5 = np.exp(preds_t5_log)

    pred_dict_top5 = {}
    valid_t5_idx = np.where(valid_t5)[0]
    for i, vi in enumerate(valid_t5_idx):
        if np.isfinite(preds_t5[i]):
            pred_dict_top5[mat_names[vi]] = float(preds_t5[i])

    evaluate_predictions(dataset, pred_dict_top5, 'r7_spatial_top5')

    # ==================================================================
    # APPROACH D: Laterality-aware channel selection
    # ==================================================================
    print(f"\n{'='*60}")
    print("Approach D: Laterality-aware channel selection")

    lateral_idx = std_idx + [n_std + n_spatial + 0, n_std + n_spatial + 1, n_std + n_spatial + 2]
    X_lat, y_lat, valid_lat = build_Xy(lateral_idx, expert_consensus, require_target=True)
    y_lat_log = np.log(y_lat)
    preds_lat_log = ridge_loo_cv(X_lat, y_lat_log, alpha=1.0)
    preds_lat = np.exp(preds_lat_log)

    pred_dict_lat = {}
    valid_lat_idx = np.where(valid_lat)[0]
    for i, vi in enumerate(valid_lat_idx):
        if np.isfinite(preds_lat[i]):
            pred_dict_lat[mat_names[vi]] = float(preds_lat[i])

    evaluate_predictions(dataset, pred_dict_lat, 'r7_lateral_select')

    # ==================================================================
    # APPROACH E: Combined (per-expert + spatial + standard)
    # ==================================================================
    print(f"\n{'='*60}")
    print("Approach E: Combined per-expert + spatial + standard")

    # Use all standard + spatial + lateral features
    all_idx = std_idx + \
        [n_std + i for i in range(n_spatial)] + \
        [n_std + n_spatial + i for i in range(n_lateral)]

    # Per-expert training with combined features
    per_expert_preds_combined = {}
    for ename, etarget in expert_targets.items():
        X, y, valid = build_Xy(all_idx, etarget, require_target=True)
        if len(y) < 5:
            continue
        y_log = np.log(y)
        preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
        preds = np.exp(preds_log)
        full_preds = np.full(N, np.nan)
        valid_idx = np.where(valid)[0]
        for i, vi in enumerate(valid_idx):
            if np.isfinite(preds[i]):
                full_preds[vi] = preds[i]
        per_expert_preds_combined[ename] = full_preds

    pred_dict_combined = {}
    for idx_i in range(N):
        vals = [per_expert_preds_combined[e][idx_i] for e in per_expert_preds_combined
                if np.isfinite(per_expert_preds_combined[e][idx_i])]
        if len(vals) > 0:
            pred_dict_combined[mat_names[idx_i]] = float(np.mean(vals))

    evaluate_predictions(dataset, pred_dict_combined, 'r7_expert_spatial_ridge')

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
