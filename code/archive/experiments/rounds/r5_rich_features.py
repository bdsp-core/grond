"""
Round 5: Rich morphological, temporal, and spatial features for ML frequency prediction.

Extracts ~25 features per segment and trains Ridge regression models with LOO-CV.

Variants:
  r5_rich_ridge_logfreq    - Ridge on log(freq), all features, alpha=1.0
  r5_rich_ridge_period     - Ridge on period (1/freq), all features, alpha=1.0
  r5_rich_ridge_freq       - Ridge on raw freq, all features
  r5_rich_ridge_lpd_only   - Separate Ridge models for LPD and GPD
  r5_rich_top10            - Ridge on log(freq), top 10 features by |correlation|
  r5_rich_ridge_alpha10    - Ridge on log(freq), alpha=10
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
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ──────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

# Template paths
REPO_ROOT = os.path.dirname(CODE_DIR)
TEMPLATES_C_LPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy')
TEMPLATES_C_GPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy')

try:
    TEMPLATES_C_LPD = np.load(TEMPLATES_C_LPD_PATH)
    TEMPLATES_C_GPD = np.load(TEMPLATES_C_GPD_PATH)
    print(f"Loaded LPD templates: {TEMPLATES_C_LPD.shape}, GPD templates: {TEMPLATES_C_GPD.shape}")
except Exception as e:
    print(f"Warning: Could not load templates: {e}")
    TEMPLATES_C_LPD = None
    TEMPLATES_C_GPD = None


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


# ── Per-channel analysis helpers ──────────────────────────────────────
def get_method_a_freq(data, fs):
    """Run Method A and return event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_acf_per_channel(seg, fs):
    """ACF frequency and score per channel. Returns (freqs, scores, peak_indices_list)."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    scores = np.full(n_ch, 0.0)
    peak_indices_list = [np.array([]) for _ in range(n_ch)]
    for i in range(n_ch):
        freq, score, peaks = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
        scores[i] = score
        peak_indices_list[i] = peaks
    return freqs, scores, peak_indices_list


def compute_peak_count_freq_per_channel(seg, fs):
    """Peak-count frequency per channel. Require >= 3 peaks."""
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
    """FFT of smoothed pointiness trace per channel, peak in [0.3, 3.5] Hz."""
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
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


def compute_envelope_fft_per_channel(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C) per channel."""
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
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


# ── Morphological features ────────────────────────────────────────────
def compute_morphological_features(seg, fs):
    """Compute morphological features from pointiness trace peaks.

    Returns: (morph_width_mean, morph_width_std, morph_amp_ratio, morph_polyphasic)
    """
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    all_widths = []
    all_amplitudes = []
    all_polyphasic_fracs = []

    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, props = find_peaks(trace, height=peak_height, distance=min_distance,
                                       width=1, prominence=0)
        if len(peak_locs) < 2:
            continue

        # Widths at half prominence
        if 'widths' in props:
            all_widths.extend(props['widths'].tolist())

        # Peak amplitudes
        peak_amps = trace[peak_locs]
        all_amplitudes.extend(peak_amps.tolist())

        # Polyphasic: fraction of peaks with secondary peak within 100ms
        polyphasic_count = 0
        secondary_dist_samples = int(0.1 * fs)  # 100ms
        for loc in peak_locs:
            region_start = max(0, loc - secondary_dist_samples)
            region_end = min(len(trace), loc + secondary_dist_samples + 1)
            region = trace[region_start:region_end]
            # Find local maxima in region excluding the main peak
            local_peaks, _ = find_peaks(region)
            # Filter out the main peak itself
            local_peaks_abs = local_peaks + region_start
            secondary = [p for p in local_peaks_abs if abs(p - loc) > 3 and trace[p] > peak_height * 0.3]
            if len(secondary) > 0:
                polyphasic_count += 1
        if len(peak_locs) > 0:
            all_polyphasic_fracs.append(polyphasic_count / len(peak_locs))

    # Aggregate
    all_widths = np.array(all_widths)
    all_amplitudes = np.array(all_amplitudes)

    morph_width_mean = float(np.mean(all_widths)) if len(all_widths) > 0 else np.nan
    morph_width_std = float(np.std(all_widths)) if len(all_widths) > 1 else np.nan

    if len(all_amplitudes) > 0:
        mean_amp = np.mean(all_amplitudes)
        median_amp = np.median(all_amplitudes)
        morph_amp_ratio = float(mean_amp / median_amp) if median_amp > 0 else np.nan
    else:
        morph_amp_ratio = np.nan

    morph_polyphasic = float(np.mean(all_polyphasic_fracs)) if len(all_polyphasic_fracs) > 0 else np.nan

    return morph_width_mean, morph_width_std, morph_amp_ratio, morph_polyphasic


# ── Temporal features ─────────────────────────────────────────────────
def compute_temporal_features(seg, fs, acf_freqs, acf_scores):
    """Compute temporal consistency features.

    Returns: (temp_freq_cv, temp_interval_cv, temp_n_cycles, temp_drift)
    """
    # temp_freq_cv: CV of per-channel frequencies
    valid_freqs = acf_freqs[np.isfinite(acf_freqs)]
    if len(valid_freqs) >= 2:
        temp_freq_cv = float(np.std(valid_freqs) / np.mean(valid_freqs)) if np.mean(valid_freqs) > 0 else np.nan
    else:
        temp_freq_cv = np.nan

    # temp_interval_cv: mean CV of inter-peak intervals within each channel
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    interval_cvs = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            intervals = np.diff(peak_locs) / fs
            mean_int = np.mean(intervals)
            if mean_int > 0:
                interval_cvs.append(np.std(intervals) / mean_int)
    temp_interval_cv = float(np.mean(interval_cvs)) if len(interval_cvs) > 0 else np.nan

    # temp_n_cycles: estimated number of complete cycles in 10s
    best_freq = median_finite(acf_freqs)
    temp_n_cycles = float(best_freq * 10.0) if np.isfinite(best_freq) else np.nan

    # temp_drift: frequency from first 5s vs last 5s
    n_samples = seg.shape[1]
    half = n_samples // 2
    seg_first = seg[:, :half]
    seg_last = seg[:, half:]

    freqs_first = np.full(seg.shape[0], np.nan)
    freqs_last = np.full(seg.shape[0], np.nan)
    for i in range(seg.shape[0]):
        f1, _, _ = compute_acf_frequency(
            seg_first[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        f2, _, _ = compute_acf_frequency(
            seg_last[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs_first[i] = f1
        freqs_last[i] = f2

    med_first = median_finite(freqs_first)
    med_last = median_finite(freqs_last)
    if np.isfinite(med_first) and np.isfinite(med_last):
        temp_drift = float(med_last - med_first)
    else:
        temp_drift = np.nan

    return temp_freq_cv, temp_interval_cv, temp_n_cycles, temp_drift


# ── Spatial features ──────────────────────────────────────────────────
def compute_spatial_features(acf_freqs, acf_scores):
    """Compute spatial features from per-channel ACF results.

    Returns: (spat_n_detected, spat_lr_balance, spat_agreement,
              spat_max_score, spat_mean_score)
    """
    detected_mask = np.isfinite(acf_freqs)
    n_detected = int(np.sum(detected_mask))

    # spat_lr_balance: |n_left - n_right| / total
    n_left = int(np.sum(detected_mask[left_indices]))
    n_right = int(np.sum(detected_mask[right_indices]))
    if n_detected > 0:
        spat_lr_balance = float(abs(n_left - n_right) / n_detected)
    else:
        spat_lr_balance = np.nan

    # spat_agreement: fraction within +/-0.2 Hz of median
    valid_freqs = acf_freqs[detected_mask]
    if len(valid_freqs) >= 2:
        med_freq = np.median(valid_freqs)
        agree_count = np.sum(np.abs(valid_freqs - med_freq) <= 0.2)
        spat_agreement = float(agree_count / len(valid_freqs))
    else:
        spat_agreement = np.nan

    # spat_max_score, spat_mean_score
    spat_max_score = float(np.max(acf_scores)) if len(acf_scores) > 0 else np.nan
    if n_detected > 0:
        spat_mean_score = float(np.mean(acf_scores[detected_mask]))
    else:
        spat_mean_score = np.nan

    return float(n_detected), spat_lr_balance, spat_agreement, spat_max_score, spat_mean_score


# ── Ridge LOO-CV helper ──────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge regression. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]

        # Add intercept
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


def ridge_full_model(X, y, alpha=1.0):
    """Fit Ridge on all data, return coefficients."""
    X_b = np.column_stack([X, np.ones(len(X))])
    I_reg = np.eye(X_b.shape[1])
    I_reg[-1, -1] = 0
    try:
        w = np.linalg.solve(X_b.T @ X_b + alpha * I_reg, X_b.T @ y)
        return w
    except np.linalg.LinAlgError:
        return None


def impute_nan_with_median(X, feature_names=None):
    """Replace NaN values with column median. Modifies X in place."""
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

    # Feature names
    feature_names = [
        # Base frequency estimates (5)
        'f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope',
        # Morphological (4)
        'morph_width_mean', 'morph_width_std', 'morph_amp_ratio', 'morph_polyphasic',
        # Temporal (4)
        'temp_freq_cv', 'temp_interval_cv', 'temp_n_cycles', 'temp_drift',
        # Spatial (5)
        'spat_n_detected', 'spat_lr_balance', 'spat_agreement', 'spat_max_score', 'spat_mean_score',
        # Pattern (1)
        'is_gpd',
        # Agreement (3)
        'agree_std', 'agree_max_min_ratio', 'agree_median',
    ]
    n_features = len(feature_names)
    print(f"Extracting {n_features} features per segment...")

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

        # --- Per-channel ACF (Method B) ---
        acf_freqs, acf_scores, peak_indices_list = compute_acf_per_channel(seg, fs)
        f_B = median_finite(acf_freqs)

        # --- Peak-count frequency ---
        pc_freqs = compute_peak_count_freq_per_channel(seg, fs)
        f_peaks = median_finite(pc_freqs)

        # --- FFT of pointiness ---
        fft_pt_freqs = compute_fft_pointiness_per_channel(seg, fs)
        f_fft = median_finite(fft_pt_freqs)

        # --- Envelope FFT ---
        env_fft_freqs = compute_envelope_fft_per_channel(seg, fs, subdir)
        f_envelope = median_finite(env_fft_freqs)

        # --- Morphological features ---
        morph_width_mean, morph_width_std, morph_amp_ratio, morph_polyphasic = \
            compute_morphological_features(seg, fs)

        # --- Temporal features ---
        temp_freq_cv, temp_interval_cv, temp_n_cycles, temp_drift = \
            compute_temporal_features(seg, fs, acf_freqs, acf_scores)

        # --- Spatial features ---
        spat_n_detected, spat_lr_balance, spat_agreement, spat_max_score, spat_mean_score = \
            compute_spatial_features(acf_freqs, acf_scores)

        # --- Pattern feature ---
        is_gpd = 1.0 if subdir == 'gpd' else 0.0

        # --- Agreement features ---
        five_freqs = np.array([f_A, f_B, f_peaks, f_fft, f_envelope])
        valid_five = five_freqs[np.isfinite(five_freqs)]
        if len(valid_five) >= 2:
            agree_std = float(np.std(valid_five))
            agree_max_min_ratio = float(np.max(valid_five) / np.min(valid_five)) if np.min(valid_five) > 0 else np.nan
            agree_median = float(np.median(valid_five))
        elif len(valid_five) == 1:
            agree_std = 0.0
            agree_max_min_ratio = 1.0
            agree_median = float(valid_five[0])
        else:
            agree_std = np.nan
            agree_max_min_ratio = np.nan
            agree_median = np.nan

        # Build feature vector
        feature_vec = [
            f_A, f_B, f_peaks, f_fft, f_envelope,
            morph_width_mean, morph_width_std, morph_amp_ratio, morph_polyphasic,
            temp_freq_cv, temp_interval_cv, temp_n_cycles, temp_drift,
            spat_n_detected, spat_lr_balance, spat_agreement, spat_max_score, spat_mean_score,
            is_gpd,
            agree_std, agree_max_min_ratio, agree_median,
        ]

        all_samples.append({
            'mat_name': mat_name,
            'features': feature_vec,
            'expert_freq': expert_freq,
            'subdir': subdir,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            print(f"  Progress: {idx + 1}/{n_total} segments ({elapsed:.0f}s)")

    # ── Prepare ML data ──
    valid = [s for s in all_samples if np.isfinite(s['expert_freq']) and s['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"\nML samples: {n_ml}")

    X = np.array([s['features'] for s in valid], dtype=float)
    y_freq = np.array([s['expert_freq'] for s in valid])
    y_log = np.log(y_freq)
    y_period = 1.0 / y_freq
    subdirs = [s['subdir'] for s in valid]
    mat_names = [s['mat_name'] for s in valid]

    # Impute NaN
    impute_nan_with_median(X, feature_names)

    # ── Experiment a: r5_rich_ridge_logfreq (Ridge on log(freq), alpha=1.0) ──
    print("\n--- r5_rich_ridge_logfreq ---")
    preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
    preds_a = np.exp(preds_log)
    preds_a = np.clip(preds_a, 0.2, 4.0)
    pred_dict_a = {mat_names[i]: float(preds_a[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_a, 'r5_rich_ridge_logfreq')

    # ── Experiment b: r5_rich_ridge_period (Ridge on 1/freq, alpha=1.0) ──
    print("\n--- r5_rich_ridge_period ---")
    preds_period = ridge_loo_cv(X, y_period, alpha=1.0)
    preds_b = 1.0 / preds_period
    preds_b = np.clip(preds_b, 0.2, 4.0)
    pred_dict_b = {mat_names[i]: float(preds_b[i]) for i in range(n_ml) if np.isfinite(preds_b[i])}
    evaluate_predictions(dataset, pred_dict_b, 'r5_rich_ridge_period')

    # ── Experiment c: r5_rich_ridge_freq (Ridge on raw freq, alpha=1.0) ──
    print("\n--- r5_rich_ridge_freq ---")
    preds_c = ridge_loo_cv(X, y_freq, alpha=1.0)
    preds_c = np.clip(preds_c, 0.2, 4.0)
    pred_dict_c = {mat_names[i]: float(preds_c[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_c, 'r5_rich_ridge_freq')

    # ── Experiment d: r5_rich_ridge_lpd_only (separate models per type) ──
    print("\n--- r5_rich_ridge_lpd_only ---")
    pred_dict_d = {}
    for ptype in ['lpd', 'gpd']:
        type_mask = np.array([s == ptype for s in subdirs])
        type_indices = np.where(type_mask)[0]
        X_type = X[type_mask]
        y_type = y_log[type_mask]
        preds_type = ridge_loo_cv(X_type, y_type, alpha=1.0)
        for j, orig_idx in enumerate(type_indices):
            pred_freq = np.exp(preds_type[j])
            pred_freq = np.clip(pred_freq, 0.2, 4.0)
            pred_dict_d[mat_names[orig_idx]] = float(pred_freq)
    evaluate_predictions(dataset, pred_dict_d, 'r5_rich_ridge_lpd_only')

    # ── Experiment e: r5_rich_top10 (top 10 features by |correlation|) ──
    print("\n--- r5_rich_top10 ---")
    # Compute abs correlation of each feature with log(target)
    correlations = []
    for col_idx in range(n_features):
        col = X[:, col_idx]
        finite_mask = np.isfinite(col) & np.isfinite(y_log)
        if np.sum(finite_mask) > 3 and np.std(col[finite_mask]) > 1e-10:
            r = np.corrcoef(col[finite_mask], y_log[finite_mask])[0, 1]
            correlations.append((col_idx, abs(r), r))
        else:
            correlations.append((col_idx, 0.0, 0.0))

    correlations.sort(key=lambda x: -x[1])
    top10_indices = [c[0] for c in correlations[:10]]
    print(f"  Top 10 features: {[feature_names[i] for i in top10_indices]}")
    for c in correlations[:10]:
        print(f"    {feature_names[c[0]]:>25s}: r={c[2]:+.4f} (|r|={c[1]:.4f})")

    X_top10 = X[:, top10_indices]
    preds_log_top10 = ridge_loo_cv(X_top10, y_log, alpha=1.0)
    preds_e = np.exp(preds_log_top10)
    preds_e = np.clip(preds_e, 0.2, 4.0)
    pred_dict_e = {mat_names[i]: float(preds_e[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_e, 'r5_rich_top10')

    # ── Experiment f: r5_rich_ridge_alpha10 (stronger regularization) ──
    print("\n--- r5_rich_ridge_alpha10 ---")
    preds_log_a10 = ridge_loo_cv(X, y_log, alpha=10.0)
    preds_f = np.exp(preds_log_a10)
    preds_f = np.clip(preds_f, 0.2, 4.0)
    pred_dict_f = {mat_names[i]: float(preds_f[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_f, 'r5_rich_ridge_alpha10')

    # ── Print top 10 feature importances for best model ──
    # Find best model by LPD Spearman from saved results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCES (full model, Ridge alpha=1.0 on log(freq))")
    print("=" * 70)
    w = ridge_full_model(X, y_log, alpha=1.0)
    if w is not None:
        coefs = w[:-1]  # exclude intercept
        importance = list(zip(feature_names, coefs))
        importance.sort(key=lambda x: -abs(x[1]))
        print(f"  {'Feature':>25s}  {'Coefficient':>12s}")
        print(f"  {'-'*25}  {'-'*12}")
        for name, coef in importance[:10]:
            print(f"  {name:>25s}  {coef:+12.6f}")
        print(f"\n  {'intercept':>25s}  {w[-1]:+12.6f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
