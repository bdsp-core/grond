"""
Experiment R5: Advanced ensemble methods, stacking, and LPD-specific deep dive.

The LPD gap is our main challenge (algorithm rs=0.425 vs expert 0.53).
This experiment tries aggressive LPD-targeted approaches.

Approaches:
  a) r5_stacked_ridge: Two-level stacking with 3 ridge models
  b) r5_lpd_aggressive: Max of base estimates for LPD, ridge-on-log for GPD
  c) r5_lpd_high_freq_fix: Filter out subharmonic-locked estimates when high freq detected
  d) r5_lpd_consistency_weighted: Weight by temporal consistency
  e) r5_all_features_ridge: Ridge on all ~15 features
  f) r5_all_features_bytype: Same as (e) but separate models for LPD and GPD
  g) r5_robust_mean: Remove outliers from median, take mean of remaining
"""

import sys
import os
import numpy as np
import warnings
import traceback

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

# ── Constants ──
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

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


# ── Preprocessing ──
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


# ── Base feature extraction (5 methods) ──

def get_method_a_freq(data, fs):
    """Method A: pd_detect_alternate with apd peak detector."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_acf_freq_median(seg, fs, acf_threshold=ACF_THRESHOLD, acf_min_lag=ACF_MIN_LAG):
    """Method B: ACF frequency, median across channels."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    n_detected = 0
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=acf_min_lag,
            acf_peak_threshold=acf_threshold,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
        if np.isfinite(freq):
            n_detected += 1
    return median_finite(freqs), n_detected, freqs


def compute_peak_count_freq_median(seg, fs, height_frac=PEAK_HEIGHT_FRAC):
    """Peak-count frequency, median across channels."""
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
        peak_height = trace_max * height_frac
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return median_finite(freqs), freqs


def compute_fft_pointiness_median(seg, fs):
    """FFT of pointiness trace, median across channels."""
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
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs), freqs


def compute_envelope_fft_median(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C), median across channels."""
    templates = TEMPLATES_C_LPD if subdir == 'lpd' else TEMPLATES_C_GPD
    if templates is None:
        return np.nan, np.full(seg.shape[0], np.nan)

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
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs), freqs


# ── LPD-specific features ──

def compute_acf_freq_low_threshold(seg, fs):
    """f_B_thr005: Method B ACF with very low threshold 0.05."""
    return compute_acf_freq_median(seg, fs, acf_threshold=0.05)[0]


def compute_acf_freq_minlag025(seg, fs):
    """f_B_minlag025: Method B with acf_min_lag=0.25 (allows up to 4Hz)."""
    return compute_acf_freq_median(seg, fs, acf_min_lag=0.25)[0]


def compute_peak_count_low_threshold(seg, fs):
    """f_peaks_low_threshold: peak-count with height=max*0.15."""
    return compute_peak_count_freq_median(seg, fs, height_frac=0.15)[0]


def compute_method_a_cwt(data, fs):
    """f_A_cwt: pd_detect_alternate with cwt peak detector."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='cwt')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def compute_half_segment_consistency(seg, fs):
    """
    half_segment_consistency: |freq_first_5s - freq_last_5s|
    using peak-count method on each half.
    """
    n_samples = seg.shape[1]
    mid = n_samples // 2
    seg_first = seg[:, :mid]
    seg_second = seg[:, mid:]

    freq_first, _ = compute_peak_count_freq_median(seg_first, fs)
    freq_second, _ = compute_peak_count_freq_median(seg_second, fs)

    if np.isfinite(freq_first) and np.isfinite(freq_second):
        return abs(freq_first - freq_second), freq_first, freq_second
    else:
        return np.nan, freq_first, freq_second


# ── Ridge regression helper (numpy, no sklearn) ──

def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge on log(y). Returns predictions in original scale."""
    n = len(y)
    log_y = np.log(np.clip(y, 1e-6, None))
    predictions = np.full(n, np.nan)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(log_y, i)
        X_test = X[i:i + 1]

        # Add intercept
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = float(X_test_b @ w)
            predictions[i] = np.exp(pred_log)
        except np.linalg.LinAlgError:
            predictions[i] = np.nan

    return predictions


def ridge_loo_cv_raw(X, y, alpha=1.0):
    """LOO-CV Ridge on raw y (not log). Returns predictions."""
    n = len(y)
    predictions = np.full(n, np.nan)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i + 1]

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            predictions[i] = float(X_test_b @ w)
        except np.linalg.LinAlgError:
            predictions[i] = np.nan

    return predictions


def impute_nan_columns(X):
    """Replace NaN with column median. If entire column NaN, use 0."""
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        col = X_out[:, j]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X_out[nan_mask, j] = col_median
    return X_out


# ── Main ──
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    n_total = len(dataset)

    # Storage for all features per segment
    records = []

    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            records.append(None)
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']

        try:
            # --- Preprocess ---
            seg = preprocess_segment(data, fs)

            # --- Base features (5 methods) ---
            f_A = get_method_a_freq(data, fs)

            f_B, n_detected, _ = compute_acf_freq_median(seg, fs, acf_threshold=ACF_THRESHOLD)
            f_peaks, _ = compute_peak_count_freq_median(seg, fs)
            f_fft, _ = compute_fft_pointiness_median(seg, fs)
            f_envelope, _ = compute_envelope_fft_median(seg, fs, subdir)

            is_gpd = 1.0 if subdir == 'gpd' else 0.0

            # std and max/min ratio of base estimates
            base_estimates = np.array([f_A, f_B, f_peaks, f_fft, f_envelope])
            valid_base = base_estimates[np.isfinite(base_estimates)]
            if len(valid_base) >= 2:
                base_std = float(np.std(valid_base))
                base_maxmin_ratio = float(np.max(valid_base) / np.min(valid_base)) if np.min(valid_base) > 0 else np.nan
            else:
                base_std = np.nan
                base_maxmin_ratio = np.nan

            # --- LPD-specific features ---
            f_B_thr005 = compute_acf_freq_low_threshold(seg, fs)
            f_B_minlag025 = compute_acf_freq_minlag025(seg, fs)
            f_peaks_low = compute_peak_count_low_threshold(seg, fs)
            f_A_cwt = compute_method_a_cwt(data, fs)
            half_cons, freq_first_half, freq_second_half = compute_half_segment_consistency(seg, fs)

            rec = {
                'mat_name': mat_name,
                'subdir': subdir,
                'expert_freq': entry['expert_consensus_freq'],
                # Base features
                'f_A': f_A,
                'f_B': f_B,
                'f_peaks': f_peaks,
                'f_fft': f_fft,
                'f_envelope': f_envelope,
                'is_gpd': is_gpd,
                'n_detected': float(n_detected),
                'base_std': base_std,
                'base_maxmin_ratio': base_maxmin_ratio,
                # LPD-specific features
                'f_B_thr005': f_B_thr005,
                'f_B_minlag025': f_B_minlag025,
                'f_peaks_low': f_peaks_low,
                'f_A_cwt': f_A_cwt,
                'half_consistency': half_cons,
                'freq_first_half': freq_first_half,
                'freq_second_half': freq_second_half,
            }
            records.append(rec)

        except Exception as e:
            print(f"  ERROR on segment {idx} ({mat_name}): {e}")
            traceback.print_exc()
            records.append(None)

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # Filter to valid records
    valid_records = [r for r in records if r is not None and np.isfinite(r['expert_freq'])]
    print(f"\nValid records: {len(valid_records)}")

    # Build feature matrices
    mat_names = [r['mat_name'] for r in valid_records]
    subdirs = np.array([r['subdir'] for r in valid_records])
    expert_freqs = np.array([r['expert_freq'] for r in valid_records])
    is_gpd_arr = np.array([r['is_gpd'] for r in valid_records])

    # Base 5 frequency estimates
    f_A_arr = np.array([r['f_A'] for r in valid_records])
    f_B_arr = np.array([r['f_B'] for r in valid_records])
    f_peaks_arr = np.array([r['f_peaks'] for r in valid_records])
    f_fft_arr = np.array([r['f_fft'] for r in valid_records])
    f_env_arr = np.array([r['f_envelope'] for r in valid_records])

    n_detected_arr = np.array([r['n_detected'] for r in valid_records])
    base_std_arr = np.array([r['base_std'] for r in valid_records])
    base_maxmin_arr = np.array([r['base_maxmin_ratio'] for r in valid_records])

    # LPD-specific
    f_B_thr005_arr = np.array([r['f_B_thr005'] for r in valid_records])
    f_B_minlag025_arr = np.array([r['f_B_minlag025'] for r in valid_records])
    f_peaks_low_arr = np.array([r['f_peaks_low'] for r in valid_records])
    f_A_cwt_arr = np.array([r['f_A_cwt'] for r in valid_records])
    half_cons_arr = np.array([r['half_consistency'] for r in valid_records])

    n = len(valid_records)

    # ──────────────────────────────────────────────────────────────────────
    # (a) r5_stacked_ridge: Two-level stacking
    # Level 1: 3 ridge models (freq features, period features, log-freq features)
    # Level 2: ridge on 3 level-1 predictions + is_gpd + n_detected
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(a) r5_stacked_ridge: Two-level stacking")
    print("=" * 60)

    # Freq features: [f_A, f_B, f_peaks, f_fft, f_envelope]
    X_freq = np.column_stack([f_A_arr, f_B_arr, f_peaks_arr, f_fft_arr, f_env_arr])
    X_freq = impute_nan_columns(X_freq)

    # Period features: 1/freq
    X_period = np.column_stack([
        1.0 / np.clip(f_A_arr, 0.01, None),
        1.0 / np.clip(f_B_arr, 0.01, None),
        1.0 / np.clip(f_peaks_arr, 0.01, None),
        1.0 / np.clip(f_fft_arr, 0.01, None),
        1.0 / np.clip(f_env_arr, 0.01, None),
    ])
    # Fix NaN from original NaN freq values
    for j in range(5):
        src = [f_A_arr, f_B_arr, f_peaks_arr, f_fft_arr, f_env_arr][j]
        nan_mask = ~np.isfinite(src)
        X_period[nan_mask, j] = np.nan
    X_period = impute_nan_columns(X_period)

    # Log-freq features: log(freq)
    X_logfreq = np.column_stack([
        np.log(np.clip(f_A_arr, 0.01, None)),
        np.log(np.clip(f_B_arr, 0.01, None)),
        np.log(np.clip(f_peaks_arr, 0.01, None)),
        np.log(np.clip(f_fft_arr, 0.01, None)),
        np.log(np.clip(f_env_arr, 0.01, None)),
    ])
    for j in range(5):
        src = [f_A_arr, f_B_arr, f_peaks_arr, f_fft_arr, f_env_arr][j]
        nan_mask = ~np.isfinite(src)
        X_logfreq[nan_mask, j] = np.nan
    X_logfreq = impute_nan_columns(X_logfreq)

    # Level 1: LOO-CV for each of the 3 ridge models
    print("  Level 1: Training 3 ridge models with LOO-CV...")
    level1_freq_preds = ridge_loo_cv(X_freq, expert_freqs, alpha=1.0)
    level1_period_preds = ridge_loo_cv(X_period, expert_freqs, alpha=1.0)
    level1_logfreq_preds = ridge_loo_cv(X_logfreq, expert_freqs, alpha=1.0)

    # Level 2: Stack level-1 predictions + is_gpd + n_detected
    X_level2 = np.column_stack([
        level1_freq_preds, level1_period_preds, level1_logfreq_preds,
        is_gpd_arr, n_detected_arr,
    ])
    X_level2 = impute_nan_columns(X_level2)

    print("  Level 2: Training meta ridge model with LOO-CV...")
    stacked_preds = ridge_loo_cv(X_level2, expert_freqs, alpha=1.0)

    preds_a = {mat_names[i]: float(stacked_preds[i]) for i in range(n)}
    evaluate_predictions(dataset, preds_a, 'r5_stacked_ridge')

    # ──────────────────────────────────────────────────────────────────────
    # (b) r5_lpd_aggressive: For LPD use MAX of base estimates, GPD use ridge-on-log
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(b) r5_lpd_aggressive: MAX for LPD, ridge-on-log for GPD")
    print("=" * 60)

    # GPD ridge on log-freq
    gpd_mask = is_gpd_arr == 1.0
    lpd_mask = is_gpd_arr == 0.0

    # For GPD: ridge on log-freq features (all 5 base + n_detected)
    X_gpd_ridge = np.column_stack([X_logfreq, n_detected_arr])
    X_gpd_ridge = impute_nan_columns(X_gpd_ridge)
    gpd_ridge_preds = ridge_loo_cv(X_gpd_ridge, expert_freqs, alpha=1.0)

    preds_b = {}
    for i in range(n):
        if lpd_mask[i]:
            # LPD: use maximum of base estimates (errors are mostly underestimates)
            base = np.array([f_A_arr[i], f_B_arr[i], f_peaks_arr[i], f_fft_arr[i]])
            valid = base[np.isfinite(base)]
            if len(valid) > 0:
                preds_b[mat_names[i]] = float(np.max(valid))
            else:
                preds_b[mat_names[i]] = np.nan
        else:
            # GPD: use ridge prediction
            preds_b[mat_names[i]] = float(gpd_ridge_preds[i])

    evaluate_predictions(dataset, preds_b, 'r5_lpd_aggressive')

    # ──────────────────────────────────────────────────────────────────────
    # (c) r5_lpd_high_freq_fix: If any estimate > 1.5 Hz, use median of
    #     estimates > 1.0 Hz only. Otherwise standard median.
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(c) r5_lpd_high_freq_fix: Filter subharmonic-locked estimates")
    print("=" * 60)

    preds_c = {}
    for i in range(n):
        base = np.array([f_A_arr[i], f_B_arr[i], f_peaks_arr[i], f_fft_arr[i], f_env_arr[i]])
        valid = base[np.isfinite(base)]
        if len(valid) == 0:
            preds_c[mat_names[i]] = np.nan
            continue

        if np.any(valid > 1.5):
            high_freq = valid[valid > 1.0]
            if len(high_freq) > 0:
                preds_c[mat_names[i]] = float(np.median(high_freq))
            else:
                preds_c[mat_names[i]] = float(np.median(valid))
        else:
            preds_c[mat_names[i]] = float(np.median(valid))

    evaluate_predictions(dataset, preds_c, 'r5_lpd_high_freq_fix')

    # ──────────────────────────────────────────────────────────────────────
    # (d) r5_lpd_consistency_weighted: Weight each frequency estimate by its
    #     temporal consistency (first half vs second half)
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(d) r5_lpd_consistency_weighted: Consistency-weighted ridge")
    print("=" * 60)

    # Per-segment consistency weights for each estimate
    # We already have half_consistency for peak-count method
    # For simplicity, compute a per-segment consistency score and use it as a feature
    # along with the base frequencies

    # Compute per-method consistency scores (comparing first/second half)
    # We'll recompute on halves for each base method - but to save time,
    # just use the half_consistency we already have for peak-count and
    # derive a weight from it.
    # consistency_weight = 1 / (1 + half_consistency)  -- high when consistent
    consistency_weight = np.where(
        np.isfinite(half_cons_arr),
        1.0 / (1.0 + half_cons_arr),
        0.5  # neutral weight for NaN
    )

    # Features: base freqs * consistency_weight, plus raw features
    X_cons = np.column_stack([
        f_A_arr * consistency_weight,
        f_B_arr * consistency_weight,
        f_peaks_arr * consistency_weight,
        f_fft_arr * consistency_weight,
        f_env_arr * consistency_weight,
        consistency_weight,
        is_gpd_arr,
        n_detected_arr,
    ])
    X_cons = impute_nan_columns(X_cons)

    cons_preds = ridge_loo_cv(X_cons, expert_freqs, alpha=1.0)
    preds_d = {mat_names[i]: float(cons_preds[i]) for i in range(n)}
    evaluate_predictions(dataset, preds_d, 'r5_lpd_consistency_weighted')

    # ──────────────────────────────────────────────────────────────────────
    # (e) r5_all_features_ridge: Ridge on ALL ~15 features
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(e) r5_all_features_ridge: Ridge on all features")
    print("=" * 60)

    # Use log of freq features where possible
    def safe_log(arr):
        out = np.log(np.clip(arr, 0.01, None))
        out[~np.isfinite(arr)] = np.nan
        return out

    X_all = np.column_stack([
        safe_log(f_A_arr),
        safe_log(f_B_arr),
        safe_log(f_peaks_arr),
        safe_log(f_fft_arr),
        safe_log(f_env_arr),
        is_gpd_arr,
        n_detected_arr,
        base_std_arr,
        base_maxmin_arr,
        safe_log(f_B_thr005_arr),
        safe_log(f_B_minlag025_arr),
        safe_log(f_peaks_low_arr),
        safe_log(f_A_cwt_arr),
        half_cons_arr,
    ])
    X_all = impute_nan_columns(X_all)

    all_preds = ridge_loo_cv(X_all, expert_freqs, alpha=1.0)
    preds_e = {mat_names[i]: float(all_preds[i]) for i in range(n)}
    evaluate_predictions(dataset, preds_e, 'r5_all_features_ridge')

    # Print feature coefficients for the full model
    log_y_all = np.log(np.clip(expert_freqs, 1e-6, None))
    X_all_b = np.column_stack([X_all, np.ones(n)])
    I_reg = np.eye(X_all_b.shape[1])
    I_reg[-1, -1] = 0
    try:
        w_full = np.linalg.solve(X_all_b.T @ X_all_b + 1.0 * I_reg, X_all_b.T @ log_y_all)
        feat_names = ['log_f_A', 'log_f_B', 'log_f_peaks', 'log_f_fft', 'log_f_env',
                       'is_gpd', 'n_detected', 'base_std', 'base_maxmin',
                       'log_f_B_thr005', 'log_f_B_minlag025', 'log_f_peaks_low',
                       'log_f_A_cwt', 'half_cons']
        print("\n  Feature coefficients (full model, Ridge alpha=1.0):")
        for fname, coef in zip(feat_names, w_full[:-1]):
            print(f"    {fname:>20s}: {coef:+.4f}")
        print(f"    {'intercept':>20s}: {w_full[-1]:+.4f}")
    except Exception as e:
        print(f"  Could not compute full-model coefficients: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # (f) r5_all_features_bytype: Same as (e) but separate models for LPD/GPD
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(f) r5_all_features_bytype: Separate LPD/GPD ridge models")
    print("=" * 60)

    bytype_preds = np.full(n, np.nan)
    for i in range(n):
        # Train on same-type samples only (excluding i)
        same_type = (is_gpd_arr == is_gpd_arr[i])
        same_type[i] = False
        if np.sum(same_type) < 5:
            # Fallback: use all samples
            train_mask = np.ones(n, dtype=bool)
            train_mask[i] = False
        else:
            train_mask = same_type

        X_train = X_all[train_mask]
        y_train = log_y_all[train_mask]
        X_test = X_all[i:i + 1]

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])
        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + 1.0 * I_reg,
                                X_train_b.T @ y_train)
            bytype_preds[i] = np.exp(float(X_test_b @ w))
        except np.linalg.LinAlgError:
            bytype_preds[i] = np.nan

    preds_f = {mat_names[i]: float(bytype_preds[i]) for i in range(n)}
    evaluate_predictions(dataset, preds_f, 'r5_all_features_bytype')

    # ──────────────────────────────────────────────────────────────────────
    # (g) r5_robust_mean: Remove outliers from median, mean of remaining
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("(g) r5_robust_mean: Remove outliers > 0.5 Hz from median")
    print("=" * 60)

    preds_g = {}
    for i in range(n):
        base = np.array([f_A_arr[i], f_B_arr[i], f_peaks_arr[i], f_fft_arr[i], f_env_arr[i]])
        valid = base[np.isfinite(base)]
        if len(valid) == 0:
            preds_g[mat_names[i]] = np.nan
            continue

        med = np.median(valid)
        close = valid[np.abs(valid - med) <= 0.5]
        if len(close) > 0:
            preds_g[mat_names[i]] = float(np.mean(close))
        else:
            # All removed: use median
            preds_g[mat_names[i]] = float(med)

    evaluate_predictions(dataset, preds_g, 'r5_robust_mean')

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("R5 ENSEMBLE EXPERIMENT SUMMARY")
    print("=" * 70)
    print("All results saved to results/optimization_runs/")
    print("Run 'python code/update_dashboard.py' to update the dashboard.")


if __name__ == '__main__':
    main()
