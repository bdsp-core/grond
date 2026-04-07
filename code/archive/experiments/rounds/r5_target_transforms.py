"""
Round 5: Target-transform experiments for frequency prediction.

The dominant errors are multiplicative underestimates (algorithm says 0.5 when
expert says 1.0, or 0.75 when expert says 1.5). Predicting period or log-frequency
should linearize these errors.

Variants:
  a) r5_predict_logfreq:       target = log(expert_freq), features = frequencies
  b) r5_predict_period:        target = 1/expert_freq, features = periods
  c) r5_predict_logperiod:     target = log(1/expert_freq), features = log(periods)
  d) r5_predict_freq_raw:      target = expert_freq directly, features = frequencies
  e) r5_predict_sqrtfreq:      target = sqrt(expert_freq), features = sqrt(frequencies)
  f) r5_predict_period_rich:   target = 1/expert_freq, features = periods + extras
  g) r5_predict_logfreq_bytype: separate ridge per LPD/GPD, log-freq target
  h) r5_predict_period_bytype:  separate ridge per LPD/GPD, period target
"""

import sys
import os
import numpy as np
import warnings

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


# ── Feature extraction ─────────────────────────────────────────────────
def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


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


def compute_acf_freq_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
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


def freq_to_period(f):
    """Convert frequency to period. If f is NaN or <= 0, return NaN."""
    if not np.isfinite(f) or f <= 0:
        return np.nan
    return 1.0 / f


def impute_nan_with_median(X):
    """Replace NaN values in each column with column median."""
    X = X.copy()
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X[nan_mask, col_idx] = col_median
    return X


def ridge_loo_predict(X, y, alpha=1.0):
    """LOO-CV Ridge regression using numpy. Returns array of predictions."""
    n = len(y)
    X = impute_nan_with_median(X)
    predictions = np.full(n, np.nan)

    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]

        # Add intercept column
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept
        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            predictions[i] = float(X_test_b @ w)
        except np.linalg.LinAlgError:
            predictions[i] = np.nan

    return predictions


def clamp(val, lo=0.2, hi=4.0):
    """Clamp a value to [lo, hi]."""
    if not np.isfinite(val):
        return np.nan
    return max(lo, min(hi, val))


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Collect feature vectors for all segments
    ml_records = []

    n_total = len(dataset)
    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # --- Method A ---
        a_freq = get_method_a_freq(data, fs)

        # --- Preprocess ---
        seg = preprocess_segment(data, fs)

        # --- Per-channel features ---
        acf_freqs = compute_acf_freq_per_channel(seg, fs)
        pc_freqs = compute_peak_count_freq_per_channel(seg, fs)
        fft_pt_freqs = compute_fft_pointiness_per_channel(seg, fs)
        env_fft_freqs = compute_envelope_fft_per_channel(seg, fs, subdir)

        # --- Segment-level aggregation (median across channels) ---
        b_freq = median_finite(acf_freqs)
        pc_freq = median_finite(pc_freqs)
        fft_pt_freq = median_finite(fft_pt_freqs)
        env_fft_freq = median_finite(env_fft_freqs)
        n_detected = int(np.sum(np.isfinite(acf_freqs)))

        # --- Period versions ---
        p_A = freq_to_period(a_freq)
        p_B = freq_to_period(b_freq)
        p_peaks = freq_to_period(pc_freq)
        p_fft = freq_to_period(fft_pt_freq)
        p_envelope = freq_to_period(env_fft_freq)

        type_is_gpd = 1.0 if subdir == 'gpd' else 0.0

        ml_records.append({
            'mat_name': mat_name,
            'subdir': subdir,
            'expert_freq': expert_freq,
            # Frequency features
            'f_A': a_freq,
            'f_B': b_freq,
            'f_peaks': pc_freq,
            'f_fft': fft_pt_freq,
            'f_envelope': env_fft_freq,
            # Period features
            'p_A': p_A,
            'p_B': p_B,
            'p_peaks': p_peaks,
            'p_fft': p_fft,
            'p_envelope': p_envelope,
            # Metadata
            'is_gpd': type_is_gpd,
            'n_ch': float(n_detected),
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # Filter to valid expert frequencies
    valid = [m for m in ml_records if np.isfinite(m['expert_freq']) and m['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"\nValid ML samples: {n_ml}")

    mat_names = [m['mat_name'] for m in valid]
    subdirs = np.array([m['subdir'] for m in valid])
    expert_freqs = np.array([m['expert_freq'] for m in valid])

    # Build feature matrices
    freq_features = np.array([[m['f_A'], m['f_B'], m['f_peaks'], m['f_fft'], m['f_envelope'],
                               m['is_gpd'], m['n_ch']] for m in valid])
    period_features = np.array([[m['p_A'], m['p_B'], m['p_peaks'], m['p_fft'], m['p_envelope'],
                                 m['is_gpd'], m['n_ch']] for m in valid])

    feature_names_freq = ['f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope', 'is_gpd', 'n_ch']
    feature_names_period = ['p_A', 'p_B', 'p_peaks', 'p_fft', 'p_envelope', 'is_gpd', 'n_ch']

    alpha = 1.0

    all_experiment_predictions = {}

    # ════════════════════════════════════════════════════════════════════
    # (a) r5_predict_logfreq: target = log(expert_freq), features = frequencies
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (a) r5_predict_logfreq ---")
    y_logfreq = np.log(expert_freqs)
    preds_log = ridge_loo_predict(freq_features, y_logfreq, alpha=alpha)
    preds_a = np.exp(preds_log)
    all_experiment_predictions['r5_predict_logfreq'] = {
        mat_names[i]: float(preds_a[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_a))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (b) r5_predict_period: target = 1/expert_freq, features = periods
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (b) r5_predict_period ---")
    y_period = 1.0 / expert_freqs
    preds_period_raw = ridge_loo_predict(period_features, y_period, alpha=alpha)
    preds_b = np.array([clamp(1.0 / p) if np.isfinite(p) and p > 0 else np.nan
                        for p in preds_period_raw])
    all_experiment_predictions['r5_predict_period'] = {
        mat_names[i]: float(preds_b[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_b))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (c) r5_predict_logperiod: target = log(1/expert_freq), features = log(periods)
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (c) r5_predict_logperiod ---")
    y_logperiod = np.log(1.0 / expert_freqs)
    # Log of period features (only the 5 period columns, keep is_gpd and n_ch as-is)
    log_period_features = period_features.copy()
    for col in range(5):  # first 5 columns are periods
        for row in range(n_ml):
            v = log_period_features[row, col]
            if np.isfinite(v) and v > 0:
                log_period_features[row, col] = np.log(v)
            else:
                log_period_features[row, col] = np.nan
    preds_logperiod_raw = ridge_loo_predict(log_period_features, y_logperiod, alpha=alpha)
    preds_c = np.array([clamp(1.0 / np.exp(p)) if np.isfinite(p) else np.nan
                        for p in preds_logperiod_raw])
    all_experiment_predictions['r5_predict_logperiod'] = {
        mat_names[i]: float(preds_c[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_c))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (d) r5_predict_freq_raw: target = expert_freq, features = frequencies
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (d) r5_predict_freq_raw ---")
    y_raw = expert_freqs.copy()
    preds_raw = ridge_loo_predict(freq_features, y_raw, alpha=alpha)
    preds_d = np.array([clamp(p) for p in preds_raw])
    all_experiment_predictions['r5_predict_freq_raw'] = {
        mat_names[i]: float(preds_d[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_d))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (e) r5_predict_sqrtfreq: target = sqrt(expert_freq), features = sqrt(freq)
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (e) r5_predict_sqrtfreq ---")
    y_sqrt = np.sqrt(expert_freqs)
    sqrt_features = freq_features.copy()
    for col in range(5):  # sqrt of first 5 frequency columns
        for row in range(n_ml):
            v = sqrt_features[row, col]
            if np.isfinite(v) and v > 0:
                sqrt_features[row, col] = np.sqrt(v)
            else:
                sqrt_features[row, col] = np.nan
    preds_sqrt_raw = ridge_loo_predict(sqrt_features, y_sqrt, alpha=alpha)
    preds_e = np.array([clamp(p ** 2) if np.isfinite(p) else np.nan for p in preds_sqrt_raw])
    all_experiment_predictions['r5_predict_sqrtfreq'] = {
        mat_names[i]: float(preds_e[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_e))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (f) r5_predict_period_rich: target = 1/expert_freq, features = periods + extras
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (f) r5_predict_period_rich ---")
    # Extra features: std_of_periods, max/min_ratio
    rich_features_list = []
    for m in valid:
        periods = [m['p_A'], m['p_B'], m['p_peaks'], m['p_fft'], m['p_envelope']]
        finite_periods = [p for p in periods if np.isfinite(p) and p > 0]

        if len(finite_periods) >= 2:
            std_periods = float(np.std(finite_periods))
            max_min_ratio = max(finite_periods) / min(finite_periods)
        elif len(finite_periods) == 1:
            std_periods = 0.0
            max_min_ratio = 1.0
        else:
            std_periods = np.nan
            max_min_ratio = np.nan

        rich_features_list.append([
            m['p_A'], m['p_B'], m['p_peaks'], m['p_fft'], m['p_envelope'],
            m['is_gpd'], m['n_ch'],
            std_periods, max_min_ratio,
        ])
    rich_features = np.array(rich_features_list)
    y_period_rich = 1.0 / expert_freqs
    preds_rich_raw = ridge_loo_predict(rich_features, y_period_rich, alpha=alpha)
    preds_f = np.array([clamp(1.0 / p) if np.isfinite(p) and p > 0 else np.nan
                        for p in preds_rich_raw])
    all_experiment_predictions['r5_predict_period_rich'] = {
        mat_names[i]: float(preds_f[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_f))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (g) r5_predict_logfreq_bytype: separate ridge per LPD/GPD, log-freq target
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (g) r5_predict_logfreq_bytype ---")
    preds_g = np.full(n_ml, np.nan)
    # Remove is_gpd column (index 5) for per-type models
    freq_features_notype = np.delete(freq_features, 5, axis=1)

    for ptype in ['lpd', 'gpd']:
        mask = subdirs == ptype
        idx_type = np.where(mask)[0]
        if len(idx_type) < 3:
            continue
        X_type = freq_features_notype[mask]
        y_type = np.log(expert_freqs[mask])
        preds_type = ridge_loo_predict(X_type, y_type, alpha=alpha)
        for j, global_idx in enumerate(idx_type):
            preds_g[global_idx] = np.exp(preds_type[j])

    all_experiment_predictions['r5_predict_logfreq_bytype'] = {
        mat_names[i]: float(preds_g[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_g))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # (h) r5_predict_period_bytype: separate ridge per LPD/GPD, period target
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (h) r5_predict_period_bytype ---")
    preds_h = np.full(n_ml, np.nan)
    period_features_notype = np.delete(period_features, 5, axis=1)

    for ptype in ['lpd', 'gpd']:
        mask = subdirs == ptype
        idx_type = np.where(mask)[0]
        if len(idx_type) < 3:
            continue
        X_type = period_features_notype[mask]
        y_type = 1.0 / expert_freqs[mask]
        preds_type = ridge_loo_predict(X_type, y_type, alpha=alpha)
        for j, global_idx in enumerate(idx_type):
            p = preds_type[j]
            if np.isfinite(p) and p > 0:
                preds_h[global_idx] = clamp(1.0 / p)
            else:
                preds_h[global_idx] = np.nan

    all_experiment_predictions['r5_predict_period_bytype'] = {
        mat_names[i]: float(preds_h[i]) for i in range(n_ml)
    }
    print(f"  Done. Finite predictions: {np.sum(np.isfinite(preds_h))}/{n_ml}")

    # ════════════════════════════════════════════════════════════════════
    # Evaluate all experiments
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EVALUATING ALL ROUND 5 TARGET-TRANSFORM EXPERIMENTS")
    print("=" * 70)

    experiment_names = [
        'r5_predict_logfreq',
        'r5_predict_period',
        'r5_predict_logperiod',
        'r5_predict_freq_raw',
        'r5_predict_sqrtfreq',
        'r5_predict_period_rich',
        'r5_predict_logfreq_bytype',
        'r5_predict_period_bytype',
    ]

    summary_rows = []
    for name in experiment_names:
        metrics = evaluate_predictions(dataset, all_experiment_predictions[name], name)
        summary_rows.append(metrics)

    # ── Print results table ──
    print("\n" + "=" * 80)
    print("ROUND 5 TARGET-TRANSFORM RESULTS (sorted by combined Spearman)")
    print("=" * 80)
    header = (f"{'Experiment':<35s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baselines
    print(f"{'Method A (baseline)':<35s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
    print(f"{'r4_ml_log_ridge (R4 best)':<35s} {'---':>8s} {'---':>8s} "
          f"{'---':>7s} {'---':>7s} {'---':>8s} {'---':>9s}")
    print("-" * len(header))

    sorted_rows = sorted(summary_rows,
                         key=lambda r: -(r.get('combined_spearman', -999)
                                         if np.isfinite(r.get('combined_spearman', np.nan))
                                         else -999))
    for row in sorted_rows:
        def fmt(key, default='?'):
            v = row.get(key, default)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return f"{v:.4f}"
            return str(v)
        print(f"{row['experiment']:<35s} {fmt('lpd_mae'):>8s} {fmt('gpd_mae'):>8s} "
              f"{fmt('lpd_spearman_r'):>7s} {fmt('gpd_spearman_r'):>7s} "
              f"{fmt('combined_spearman'):>8s} {fmt('combined_mae'):>9s}")

    print("\nDone! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
