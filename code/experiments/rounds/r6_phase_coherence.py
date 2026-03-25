"""
Round 6: Cross-channel phase coherence features for LPD frequency estimation.

Hypothesis: Experts use spatial propagation patterns — an LPD discharge "spreads"
from one electrode to the next with slight phase offsets. Channels showing the same
periodic discharge should be phase-coherent. This signal is currently unused.

Variants:
  r6_coh_weighted_fft      - Coherence-weighted FFT frequency
  r6_spectral_coherence    - Peak of average spectral coherence
  r6_coh_ridge             - Ridge on log(freq) with standard + 7 coherence features
  r6_coh_ridge_all         - Ridge with standard + coherence + spectral_coherence_freq
  r6_coh_spectral_plus_fft - Average of spectral coherence freq + FFT-of-pointiness freq
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
    bipolar_channels, left_indices, right_indices,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks, coherence as scipy_coherence
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ──────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

# Adjacent pairs within chains
LEFT_TEMPORAL_PAIRS = [(0, 1), (1, 2), (2, 3)]
RIGHT_TEMPORAL_PAIRS = [(4, 5), (5, 6), (6, 7)]
LEFT_PARASAG_PAIRS = [(8, 9), (9, 10), (10, 11)]
RIGHT_PARASAG_PAIRS = [(12, 13), (13, 14), (14, 15)]
MIDLINE_PAIRS = [(16, 17)]

LEFT_PAIRS = LEFT_TEMPORAL_PAIRS + LEFT_PARASAG_PAIRS
RIGHT_PAIRS = RIGHT_TEMPORAL_PAIRS + RIGHT_PARASAG_PAIRS
ALL_ADJACENT_PAIRS = LEFT_PAIRS + RIGHT_PAIRS + MIDLINE_PAIRS

LEFT_CH_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CH_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]


# ── Preprocessing ──────────────────────────────────────────────────────
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


# ── Per-channel helpers ───────────────────────────────────────────────
def get_method_a_freq(data, fs):
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_acf_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    scores = np.full(n_ch, 0.0)
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
        scores[i] = score
    return freqs, scores


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


# ── Phase coherence features ──────────────────────────────────────────
def compute_pointiness_traces(seg, fs):
    """Compute smoothed pointiness trace for all channels."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    n_ch = seg.shape[0]
    traces = []
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return traces


def compute_cross_corr_features(traces, fs, pairs):
    """Cross-correlate pointiness traces for given channel pairs.

    Returns list of (coherence_score, phase_lag_seconds) per pair.
    """
    results = []
    for (ch_a, ch_b) in pairs:
        if ch_a >= len(traces) or ch_b >= len(traces):
            results.append((np.nan, np.nan))
            continue

        t_a = traces[ch_a]
        t_b = traces[ch_b]

        auto_a = np.sum(t_a ** 2)
        auto_b = np.sum(t_b ** 2)

        if auto_a <= 0 or auto_b <= 0:
            results.append((np.nan, np.nan))
            continue

        # Cross-correlate
        xcorr = np.correlate(t_a, t_b, mode='full')
        # Lags: from -(N-1) to +(N-1)
        n = len(t_a)
        lags = np.arange(-(n - 1), n)

        peak_idx = np.argmax(np.abs(xcorr))
        peak_val = xcorr[peak_idx]
        lag_samples = lags[peak_idx]

        coherence_score = peak_val / np.sqrt(auto_a * auto_b)
        phase_lag = lag_samples / fs

        results.append((float(coherence_score), float(phase_lag)))

    return results


def compute_spectral_coherence_freq(seg, fs, pairs):
    """Compute average spectral coherence across adjacent pairs and find peak in [0.3, 3.5] Hz.

    Uses the raw bipolar-montaged signal (not pointiness).
    """
    n_samples = seg.shape[1]
    nperseg = min(256, n_samples // 2)
    if nperseg < 16:
        return np.nan

    sum_coh = None
    count = 0
    freqs_out = None

    for (ch_a, ch_b) in pairs:
        if ch_a >= seg.shape[0] or ch_b >= seg.shape[0]:
            continue
        try:
            f, cxy = scipy_coherence(seg[ch_a], seg[ch_b], fs=fs, nperseg=nperseg)
            if sum_coh is None:
                sum_coh = np.zeros_like(cxy)
                freqs_out = f
            sum_coh += cxy
            count += 1
        except Exception:
            continue

    if count == 0 or sum_coh is None:
        return np.nan

    avg_coh = sum_coh / count
    mask = (freqs_out >= 0.3) & (freqs_out <= 3.5)
    if not np.any(mask):
        return np.nan

    coh_sub = avg_coh[mask]
    freq_sub = freqs_out[mask]
    peak_idx = np.argmax(coh_sub)
    return float(freq_sub[peak_idx])


def aggregate_coherence_features(cross_corr_results, pairs):
    """Aggregate cross-correlation results into 7 coherence features.

    Returns: coh_mean, coh_max, coh_n_high, lag_std, coh_left, coh_right, coh_lr_ratio
    """
    all_cohs = []
    all_lags = []
    left_cohs = []
    right_cohs = []

    for idx, (coh, lag) in enumerate(cross_corr_results):
        pair = pairs[idx]
        if not np.isfinite(coh):
            continue
        all_cohs.append(coh)
        if np.isfinite(lag):
            all_lags.append(lag)

        if pair in LEFT_PAIRS:
            left_cohs.append(coh)
        elif pair in RIGHT_PAIRS:
            right_cohs.append(coh)

    if len(all_cohs) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    coh_mean = float(np.mean(all_cohs))
    coh_max = float(np.max(all_cohs))
    coh_n_high = float(np.sum(np.array(all_cohs) > 0.3))
    lag_std = float(np.std(all_lags)) if len(all_lags) >= 2 else np.nan

    coh_left = float(np.mean(left_cohs)) if len(left_cohs) > 0 else np.nan
    coh_right = float(np.mean(right_cohs)) if len(right_cohs) > 0 else np.nan

    if np.isfinite(coh_left) and np.isfinite(coh_right) and coh_right > 0:
        coh_lr_ratio = float(coh_left / coh_right)
    else:
        coh_lr_ratio = np.nan

    return coh_mean, coh_max, coh_n_high, lag_std, coh_left, coh_right, coh_lr_ratio


def compute_coherence_weighted_fft(traces, fft_freqs_per_ch, fs):
    """Weight per-channel FFT frequencies by max coherence with any adjacent channel.

    For each channel, weight = max coherence with any adjacent channel.
    Returns weighted mean of per-channel FFT frequencies.
    """
    n_ch = len(traces)

    # Build adjacency: for each channel, which channels are adjacent?
    adjacency = {i: [] for i in range(n_ch)}
    for (a, b) in ALL_ADJACENT_PAIRS:
        if a < n_ch and b < n_ch:
            adjacency[a].append(b)
            adjacency[b].append(a)

    weights = np.zeros(n_ch)
    for ch in range(n_ch):
        if len(adjacency[ch]) == 0:
            continue
        t_ch = traces[ch]
        auto_ch = np.sum(t_ch ** 2)
        if auto_ch <= 0:
            continue
        max_coh = 0.0
        for adj in adjacency[ch]:
            t_adj = traces[adj]
            auto_adj = np.sum(t_adj ** 2)
            if auto_adj <= 0:
                continue
            xcorr = np.correlate(t_ch, t_adj, mode='full')
            peak_val = np.max(np.abs(xcorr))
            coh = peak_val / np.sqrt(auto_ch * auto_adj)
            if coh > max_coh:
                max_coh = coh
        weights[ch] = max_coh

    # Weighted mean of FFT frequencies
    valid_mask = np.isfinite(fft_freqs_per_ch) & (weights > 0)
    if not np.any(valid_mask):
        return np.nan

    w = weights[valid_mask]
    f = fft_freqs_per_ch[valid_mask]
    return float(np.sum(w * f) / np.sum(w))


# ── Ridge LOO-CV helper ──────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0
        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            preds[i] = float(X_test_b @ w)
        except np.linalg.LinAlgError:
            preds[i] = np.nan
    return preds


def impute_nan_with_median(X, feature_names=None):
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

    # Load templates for envelope FFT
    REPO_ROOT = os.path.dirname(CODE_DIR)
    try:
        TEMPLATES_C_LPD = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
        TEMPLATES_C_GPD = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
        print(f"Loaded LPD templates: {TEMPLATES_C_LPD.shape}, GPD templates: {TEMPLATES_C_GPD.shape}")
    except Exception as e:
        print(f"Warning: Could not load templates: {e}")
        TEMPLATES_C_LPD = None
        TEMPLATES_C_GPD = None

    def compute_envelope_fft_per_channel(seg, fs, subdir):
        templates = TEMPLATES_C_LPD if subdir == 'lpd' else TEMPLATES_C_GPD
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

    # Collect all data
    all_samples = []

    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # --- Method A ---
        f_A = get_method_a_freq(data, fs)

        # --- Preprocess ---
        seg = preprocess_segment(data, fs)

        # --- Per-channel ACF (Method B, threshold 0.10) ---
        acf_freqs, acf_scores = compute_acf_per_channel(seg, fs)
        f_B = median_finite(acf_freqs)

        # --- Peak-count frequency ---
        pc_freqs = compute_peak_count_freq_per_channel(seg, fs)
        f_peaks = median_finite(pc_freqs)

        # --- FFT of pointiness per channel ---
        fft_pt_freqs = compute_fft_pointiness_per_channel(seg, fs)
        f_fft = median_finite(fft_pt_freqs)

        # --- Envelope FFT ---
        env_fft_freqs = compute_envelope_fft_per_channel(seg, fs, subdir)
        f_envelope = median_finite(env_fft_freqs)

        # --- is_gpd, n_ch ---
        is_gpd = 1.0 if subdir == 'gpd' else 0.0
        n_ch = float(np.sum(np.isfinite(acf_freqs)))

        # --- Phase coherence features ---
        traces = compute_pointiness_traces(seg, fs)

        # Cross-correlation features for all adjacent pairs
        cross_corr_results = compute_cross_corr_features(traces, fs, ALL_ADJACENT_PAIRS)

        # Aggregate coherence features
        coh_mean, coh_max, coh_n_high, lag_std, coh_left, coh_right, coh_lr_ratio = \
            aggregate_coherence_features(cross_corr_results, ALL_ADJACENT_PAIRS)

        # Coherence-weighted FFT frequency
        coh_weighted_fft = compute_coherence_weighted_fft(traces, fft_pt_freqs, fs)

        # Spectral coherence frequency
        spectral_coh_freq = compute_spectral_coherence_freq(seg, fs, ALL_ADJACENT_PAIRS)

        all_samples.append({
            'mat_name': mat_name,
            'subdir': subdir,
            'expert_freq': expert_freq,
            # Standard features
            'f_A': f_A,
            'f_B': f_B,
            'f_peaks': f_peaks,
            'f_fft': f_fft,
            'f_envelope': f_envelope,
            'is_gpd': is_gpd,
            'n_ch': n_ch,
            # Coherence features
            'coh_mean': coh_mean,
            'coh_max': coh_max,
            'coh_n_high': coh_n_high,
            'lag_std': lag_std,
            'coh_left': coh_left,
            'coh_right': coh_right,
            'coh_lr_ratio': coh_lr_ratio,
            # Derived frequencies
            'coh_weighted_fft': coh_weighted_fft,
            'spectral_coh_freq': spectral_coh_freq,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            print(f"  Progress: {idx + 1}/{n_total} segments ({elapsed:.0f}s)")

    # ── Prepare ML data ──
    valid = [s for s in all_samples if np.isfinite(s['expert_freq']) and s['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"\nML samples: {n_ml}")

    mat_names = [s['mat_name'] for s in valid]
    y_freq = np.array([s['expert_freq'] for s in valid])
    y_log = np.log(y_freq)

    # ── Variant a: r6_coh_weighted_fft ──
    print("\n--- r6_coh_weighted_fft ---")
    pred_dict_a = {}
    for s in valid:
        v = s['coh_weighted_fft']
        if np.isfinite(v):
            pred_dict_a[s['mat_name']] = float(np.clip(v, 0.2, 4.0))
    evaluate_predictions(dataset, pred_dict_a, 'r6_coh_weighted_fft')

    # ── Variant b: r6_spectral_coherence ──
    print("\n--- r6_spectral_coherence ---")
    pred_dict_b = {}
    for s in valid:
        v = s['spectral_coh_freq']
        if np.isfinite(v):
            pred_dict_b[s['mat_name']] = float(np.clip(v, 0.2, 4.0))
    evaluate_predictions(dataset, pred_dict_b, 'r6_spectral_coherence')

    # ── Variant c: r6_coh_ridge ──
    # Standard features + 7 coherence features, Ridge on log(freq)
    print("\n--- r6_coh_ridge ---")
    standard_feature_names = ['f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope', 'is_gpd', 'n_ch']
    coh_feature_names = ['coh_mean', 'coh_max', 'coh_n_high', 'lag_std',
                         'coh_left', 'coh_right', 'coh_lr_ratio']
    feature_names_c = standard_feature_names + coh_feature_names

    X_c = np.array([[s[f] for f in feature_names_c] for s in valid], dtype=float)
    impute_nan_with_median(X_c, feature_names_c)

    preds_log_c = ridge_loo_cv(X_c, y_log, alpha=1.0)
    preds_c = np.exp(preds_log_c)
    preds_c = np.clip(preds_c, 0.2, 4.0)
    pred_dict_c = {mat_names[i]: float(preds_c[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_c, 'r6_coh_ridge')

    # ── Variant d: r6_coh_ridge_all ──
    # Standard + coherence + spectral_coherence_freq
    print("\n--- r6_coh_ridge_all ---")
    feature_names_d = feature_names_c + ['spectral_coh_freq']
    X_d = np.array([[s[f] for f in feature_names_d] for s in valid], dtype=float)
    impute_nan_with_median(X_d, feature_names_d)

    preds_log_d = ridge_loo_cv(X_d, y_log, alpha=1.0)
    preds_d = np.exp(preds_log_d)
    preds_d = np.clip(preds_d, 0.2, 4.0)
    pred_dict_d = {mat_names[i]: float(preds_d[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_d, 'r6_coh_ridge_all')

    # ── Variant e: r6_coh_spectral_plus_fft ──
    print("\n--- r6_coh_spectral_plus_fft ---")
    pred_dict_e = {}
    for s in valid:
        sc = s['spectral_coh_freq']
        ff = s['f_fft']
        vals = [v for v in [sc, ff] if np.isfinite(v)]
        if len(vals) > 0:
            pred_dict_e[s['mat_name']] = float(np.clip(np.mean(vals), 0.2, 4.0))
    evaluate_predictions(dataset, pred_dict_e, 'r6_coh_spectral_plus_fft')

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
