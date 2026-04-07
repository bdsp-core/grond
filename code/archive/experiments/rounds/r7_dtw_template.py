"""
Round 7: Dynamic Time Warping (DTW) template matching for discharge detection.

KEY INSIGHT: Fixed cross-correlation template matching failed because EEG discharge
morphology varies in duration/phase. DTW handles this temporal warping naturally.

Templates: data/templates_C_lpd.npy, data/templates_C_gpd.npy (8x50, 250ms at 200Hz)

For each channel:
  - Slide 250ms window with 50ms hop across the signal
  - DTW distance between each window and each template
  - eventness = 1 / (1 + min_DTW_across_templates)
  - FFT / peak-count on the eventness trace -> frequency

Variants:
  r7_dtw_fft:            DTW eventness -> FFT, median across channels
  r7_dtw_peaks:          DTW eventness -> peak-count
  r7_dtw_plus_peakcount: average of DTW FFT + standard peak-count
  r7_dtw_ridge:          Ridge with standard features + DTW features, LOO-CV
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
    bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ──────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02  # fraction of fs
PEAK_HEIGHT_FRAC = 0.3
PEAK_MIN_DIST = 0.2  # seconds
FS = 200
WINDOW_SAMPLES = 50   # 250ms at 200Hz
HOP_SAMPLES = 10      # 50ms hop
MAX_TEMPLATES = 8     # use all 8 templates (fast enough)


# ── DTW implementation ─────────────────────────────────────────────────
def dtw_distance(s, t):
    """DTW distance between two 1D arrays, normalized by path length."""
    n, m = len(s), len(t)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[n, m] / (n + m)


# Try to use a faster DTW if available
try:
    from dtw import dtw as _dtw_lib
    def dtw_distance_fast(s, t):
        result = _dtw_lib(s, t)
        return result.normalizedDistance
    USE_FAST_DTW = True
    print("Using dtw-python library (fast)")
except ImportError:
    try:
        from scipy.spatial.distance import cdist
        # Use vectorized numpy DTW
        def dtw_distance_fast(s, t):
            n, m = len(s), len(t)
            # Cost matrix
            cost_mat = (s.reshape(-1, 1) - t.reshape(1, -1)) ** 2
            # DP
            D = np.full((n + 1, m + 1), np.inf)
            D[0, 0] = 0.0
            for i in range(1, n + 1):
                # Vectorize inner loop: compute all j at once using rolling min
                # D[i, 1:m+1] = cost_mat[i-1, :] + min of three predecessors
                prev_diag = D[i - 1, :m]       # D[i-1, j-1]
                prev_above = D[i - 1, 1:m + 1] # D[i-1, j]
                prev_left = D[i, :m]            # D[i, j-1] - need sequential
                # Can't fully vectorize due to D[i, j-1] dependency
                for j in range(1, m + 1):
                    D[i, j] = cost_mat[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
            return D[n, m] / (n + m)
        USE_FAST_DTW = False
    except ImportError:
        dtw_distance_fast = dtw_distance
        USE_FAST_DTW = False


# Numba-accelerated DTW if available
try:
    from numba import njit
    @njit(cache=True)
    def dtw_distance_numba(s, t):
        n = len(s)
        m = len(t)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = (s[i - 1] - t[j - 1]) ** 2
                d1 = D[i - 1, j]
                d2 = D[i, j - 1]
                d3 = D[i - 1, j - 1]
                D[i, j] = cost + min(d1, min(d2, d3))
        return D[n, m] / (n + m)
    # Warm up numba
    _ = dtw_distance_numba(np.zeros(5), np.zeros(5))
    dtw_func = dtw_distance_numba
    print("Using numba-accelerated DTW")
except ImportError:
    dtw_func = dtw_distance_fast
    print("Using pure numpy DTW (no numba)")


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_standard(data, fs):
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


# ── DTW eventness computation ─────────────────────────────────────────
def compute_dtw_eventness(channel_signal, templates):
    """
    Slide a 250ms window across channel_signal with 50ms hop.
    For each window, compute DTW distance to each template.
    Return eventness trace: 1 / (1 + min_DTW_distance).
    """
    n_samples = len(channel_signal)
    if n_samples < WINDOW_SAMPLES:
        return np.array([])

    # Normalize templates once
    norm_templates = []
    for t_idx in range(min(templates.shape[0], MAX_TEMPLATES)):
        tmpl = templates[t_idx].copy()
        tmpl_std = np.std(tmpl)
        if tmpl_std > 0:
            tmpl = (tmpl - np.mean(tmpl)) / tmpl_std
        norm_templates.append(tmpl)

    n_windows = (n_samples - WINDOW_SAMPLES) // HOP_SAMPLES + 1
    eventness = np.zeros(n_windows)

    for w_idx in range(n_windows):
        start = w_idx * HOP_SAMPLES
        window = channel_signal[start:start + WINDOW_SAMPLES].copy()
        w_std = np.std(window)
        if w_std > 0:
            window = (window - np.mean(window)) / w_std
        else:
            eventness[w_idx] = 0.0
            continue

        min_dist = np.inf
        for tmpl in norm_templates:
            dist = dtw_func(window, tmpl)
            if dist < min_dist:
                min_dist = dist

        eventness[w_idx] = 1.0 / (1.0 + min_dist)

    return eventness


# ── DTW-based frequency estimators ────────────────────────────────────
def compute_dtw_fft_per_channel(seg, fs, templates):
    """FFT of smoothed DTW eventness trace per channel, peak in [0.3, 3.5] Hz."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # Effective sample rate of eventness trace
    eventness_fs = fs / HOP_SAMPLES  # 200/10 = 20 Hz

    for i in range(n_ch):
        eventness = compute_dtw_eventness(seg[i], templates)
        if len(eventness) < 10:
            continue
        eventness_smooth = gaussian_filter1d(eventness, sigma=max(1, int(SMOOTHING_SIGMA * eventness_fs)))
        if np.max(eventness_smooth) <= 0:
            continue
        n = len(eventness_smooth)
        fft_vals = np.abs(np.fft.rfft(eventness_smooth - np.mean(eventness_smooth)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / eventness_fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


def compute_dtw_peaks_per_channel(seg, fs, templates):
    """Peak-count on smoothed DTW eventness trace per channel. Require >= 3 peaks."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)

    eventness_fs = fs / HOP_SAMPLES

    for i in range(n_ch):
        eventness = compute_dtw_eventness(seg[i], templates)
        if len(eventness) < 10:
            continue
        eventness_smooth = gaussian_filter1d(eventness, sigma=max(1, int(SMOOTHING_SIGMA * eventness_fs)))
        ev_max = np.max(eventness_smooth)
        if ev_max <= 0:
            continue
        peak_height = ev_max * PEAK_HEIGHT_FRAC
        min_distance = max(1, int(PEAK_MIN_DIST * eventness_fs))
        peak_locs, _ = find_peaks(eventness_smooth, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            span = (peak_locs[-1] - peak_locs[0]) / eventness_fs
            if span > 0:
                freqs[i] = (len(peak_locs) - 1) / span
    return freqs


# ── Standard frequency estimators (for ridge features) ────────────────
def get_method_a_freq(data, fs):
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_peak_count_freq_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return freqs


def compute_fft_pointiness_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
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
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


def compute_acf_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, peaks = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=0.4,
            acf_peak_threshold=0.10,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
    return freqs


def compute_envelope_fft_per_channel(seg, fs, subdir, templates_lpd, templates_gpd):
    """Matched-filter envelope FFT (Bank C) per channel."""
    templates = templates_lpd if subdir == 'lpd' else templates_gpd
    if templates is None:
        return np.full(seg.shape[0], np.nan)

    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        ch = seg[i]
        n_samples = len(ch)
        envelope = np.zeros(n_samples)
        for t_idx in range(templates.shape[0]):
            tmpl = templates[t_idx]
            if len(tmpl) > n_samples:
                tmpl = tmpl[:n_samples]
            corr = np.abs(np.correlate(ch, tmpl, mode='same'))
            envelope = np.maximum(envelope, corr)
        if np.max(envelope) <= 0:
            continue
        n = len(envelope)
        fft_vals = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


# ── Ridge LOO-CV helper ───────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge regression. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept
        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            preds[i] = float(X_test_b @ w)
        except np.linalg.LinAlgError:
            preds[i] = np.nan
    return preds


def impute_nan_with_median(X, feature_names=None):
    """Replace NaN values with column median."""
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X[nan_mask, col_idx] = col_median
            n_imputed = int(np.sum(nan_mask))
            name = feature_names[col_idx] if feature_names else f"col_{col_idx}"
            print(f"  Imputed {n_imputed} NaN in {name} (median={col_median:.3f})")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading dataset...")
    dataset = load_dataset()
    n_total = len(dataset)
    print(f"Dataset: {n_total} segments")

    # Load templates once
    REPO_ROOT = os.path.dirname(CODE_DIR)
    try:
        templates_lpd = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
        templates_gpd = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
        print(f"Loaded templates: LPD {templates_lpd.shape}, GPD {templates_gpd.shape}")
    except Exception as e:
        print(f"FATAL: Could not load templates: {e}")
        return

    # Storage for all results
    all_samples = []

    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # ── Method A ──
        f_A = get_method_a_freq(data, fs)

        # ── Standard preprocessing ──
        seg_std = preprocess_standard(data, fs)

        # ── Standard features ──
        acf_freqs = compute_acf_per_channel(seg_std, fs)
        f_B = median_finite(acf_freqs)

        pc_freqs = compute_peak_count_freq_per_channel(seg_std, fs)
        f_peaks_pt = median_finite(pc_freqs)

        fft_pt_freqs = compute_fft_pointiness_per_channel(seg_std, fs)
        f_fft_pt = median_finite(fft_pt_freqs)

        env_fft_freqs = compute_envelope_fft_per_channel(seg_std, fs, subdir, templates_lpd, templates_gpd)
        f_envelope = median_finite(env_fft_freqs)

        # ── DTW features ──
        templates = templates_lpd if subdir == 'lpd' else templates_gpd

        dtw_fft_freqs = compute_dtw_fft_per_channel(seg_std, fs, templates)
        f_dtw_fft = median_finite(dtw_fft_freqs)

        dtw_peaks_freqs = compute_dtw_peaks_per_channel(seg_std, fs, templates)
        f_dtw_peaks = median_finite(dtw_peaks_freqs)

        # DTW FFT + standard peak-count average
        if np.isfinite(f_dtw_fft) and np.isfinite(f_peaks_pt):
            f_dtw_plus_peakcount = (f_dtw_fft + f_peaks_pt) / 2.0
        elif np.isfinite(f_dtw_fft):
            f_dtw_plus_peakcount = f_dtw_fft
        elif np.isfinite(f_peaks_pt):
            f_dtw_plus_peakcount = f_peaks_pt
        else:
            f_dtw_plus_peakcount = np.nan

        is_gpd = 1.0 if subdir == 'gpd' else 0.0

        all_samples.append({
            'mat_name': mat_name,
            'subdir': subdir,
            'expert_freq': expert_freq,
            # DTW estimates
            'f_dtw_fft': f_dtw_fft,
            'f_dtw_peaks': f_dtw_peaks,
            'f_dtw_plus_peakcount': f_dtw_plus_peakcount,
            # Ridge features
            'f_A': f_A,
            'f_B': f_B,
            'f_peaks_pt': f_peaks_pt,
            'f_fft_pt': f_fft_pt,
            'f_envelope': f_envelope,
            'is_gpd': is_gpd,
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            n_done = len(all_samples)
            rate = elapsed / max(n_done, 1)
            remaining = rate * (n_total - idx - 1)
            print(f"  Progress: {idx + 1}/{n_total} segments | {elapsed:.0f}s elapsed | ~{remaining:.0f}s remaining")

    print(f"\nProcessed {len(all_samples)} segments with data")

    # ── Evaluate simple variants ──
    variant_keys = [
        ('r7_dtw_fft', 'f_dtw_fft'),
        ('r7_dtw_peaks', 'f_dtw_peaks'),
        ('r7_dtw_plus_peakcount', 'f_dtw_plus_peakcount'),
    ]

    for exp_name, key in variant_keys:
        pred_dict = {}
        for s in all_samples:
            v = s[key]
            if np.isfinite(v):
                pred_dict[s['mat_name']] = float(v)
        print(f"\n--- {exp_name}: {len(pred_dict)} predictions ---")
        evaluate_predictions(dataset, pred_dict, exp_name)

    # ── Ridge variant: r7_dtw_ridge ──
    print("\n--- r7_dtw_ridge ---")
    ridge_feature_names = [
        'f_A', 'f_B', 'f_peaks_pt', 'f_fft_pt', 'f_envelope',
        'f_dtw_fft', 'f_dtw_peaks',
        'is_gpd',
    ]

    valid = [s for s in all_samples if np.isfinite(s['expert_freq']) and s['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"  ML samples: {n_ml}")

    X = np.array([[s[fn] for fn in ridge_feature_names] for s in valid], dtype=float)
    y_freq = np.array([s['expert_freq'] for s in valid])
    y_log = np.log(y_freq)
    mat_names = [s['mat_name'] for s in valid]

    impute_nan_with_median(X, ridge_feature_names)

    preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
    preds_ridge = np.exp(preds_log)
    preds_ridge = np.clip(preds_ridge, 0.2, 4.0)
    pred_dict_ridge = {mat_names[i]: float(preds_ridge[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_ridge, 'r7_dtw_ridge')

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
