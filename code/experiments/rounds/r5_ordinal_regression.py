"""
Round 5: Ordinal regression -- predict frequency as ordered categories
matching expert annotation granularity.

Experts annotate frequency using discrete values:
  0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 Hz
Ordinal regression respects this structure.

Variants:
  a) r5_ordinal_ridge:            ridge on bin index, round to nearest, map back to Hz
  b) r5_ordinal_ridge_continuous: ridge on bin index, don't round, map linearly back to Hz
  c) r5_ordinal_ridge_logfreq:    ordinal bins on log-frequency scale
  d) r5_ordinal_bytype:           separate ordinal models for LPD and GPD
  e) r5_ordinal_plus_continuous:  average of ordinal prediction with continuous ridge-on-log-freq
  f) r5_snap_to_grid:             best continuous prediction (ridge on log-freq), snap to nearest bin
  g) r5_quantile_weighted:        softmax weighting over bin centers
  h) r5_bin_classifier:           K one-vs-rest classifiers, expected value from probabilities
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

# Expert annotation bins
BIN_CENTERS = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5])
K = len(BIN_CENTERS)

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


# ── Bin assignment helpers ──────────────────────────────────────────────
def freq_to_bin_index(freq):
    """Assign frequency to nearest bin center, return index."""
    if not np.isfinite(freq) or freq <= 0:
        return 0  # default to lowest bin
    return int(np.argmin(np.abs(BIN_CENTERS - freq)))


def bin_index_to_freq(idx):
    """Map bin index back to Hz (clamp to valid range)."""
    idx_clamped = max(0, min(K - 1, int(round(idx))))
    return BIN_CENTERS[idx_clamped]


def bin_index_to_freq_continuous(idx):
    """Map continuous bin index to Hz via linear interpolation."""
    idx_clamped = max(0.0, min(float(K - 1), idx))
    lower = int(np.floor(idx_clamped))
    upper = min(lower + 1, K - 1)
    frac = idx_clamped - lower
    return BIN_CENTERS[lower] * (1.0 - frac) + BIN_CENTERS[upper] * frac


def snap_to_nearest_bin(freq):
    """Snap a continuous frequency to nearest expert bin center."""
    if not np.isfinite(freq) or freq <= 0:
        return np.nan
    return BIN_CENTERS[np.argmin(np.abs(BIN_CENTERS - freq))]


# ── Ridge regression LOO-CV helper ─────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge regression using numpy. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    # Add intercept
    X_b = np.column_stack([X, np.ones(n)])
    p = X_b.shape[1]
    I_reg = np.eye(p)
    I_reg[-1, -1] = 0  # Don't regularize intercept

    for i in range(n):
        X_train = np.delete(X_b, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X_b[i:i+1]
        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I_reg,
                                X_train.T @ y_train)
            preds[i] = float(X_test @ w)
        except np.linalg.LinAlgError:
            preds[i] = np.nan
    return preds


def ridge_loo_cv_bytype(X, y, type_mask, alpha=1.0):
    """LOO-CV Ridge with separate models for LPD (type_mask=0) and GPD (type_mask=1).

    Returns predictions array.
    """
    n = len(y)
    preds = np.full(n, np.nan)
    # Add intercept
    X_b = np.column_stack([X, np.ones(n)])
    p = X_b.shape[1]
    I_reg = np.eye(p)
    I_reg[-1, -1] = 0

    for i in range(n):
        # Train only on same-type samples (excluding i)
        my_type = type_mask[i]
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        train_mask &= (type_mask == my_type)

        X_train = X_b[train_mask]
        y_train = y[train_mask]
        X_test = X_b[i:i+1]

        if len(y_train) < 3:
            # Fallback to all-type training
            X_train = np.delete(X_b, i, axis=0)
            y_train = np.delete(y, i, axis=0)

        try:
            w = np.linalg.solve(X_train.T @ X_train + alpha * I_reg,
                                X_train.T @ y_train)
            preds[i] = float(X_test @ w)
        except np.linalg.LinAlgError:
            preds[i] = np.nan
    return preds


def softmax(x, temperature=1.0):
    """Stable softmax."""
    x = np.asarray(x, dtype=float)
    x = x / temperature
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


# ── One-vs-rest bin classifier ──────────────────────────────────────────
def bin_classifier_loo_cv(X, y_bin_indices, alpha=1.0):
    """Train K one-vs-rest ridge classifiers, predict P(bin=k), expected value.

    Returns predictions as Hz values.
    """
    n = len(y_bin_indices)
    preds = np.full(n, np.nan)
    X_b = np.column_stack([X, np.ones(n)])
    p = X_b.shape[1]
    I_reg = np.eye(p)
    I_reg[-1, -1] = 0

    for i in range(n):
        X_train = np.delete(X_b, i, axis=0)
        y_train_indices = np.delete(y_bin_indices, i)
        X_test = X_b[i:i+1]

        # One-vs-rest probabilities
        probs = np.zeros(K)
        for k in range(K):
            y_binary = (y_train_indices == k).astype(float)
            # Skip if no positive examples
            if np.sum(y_binary) == 0:
                probs[k] = 0.0
                continue
            try:
                w = np.linalg.solve(X_train.T @ X_train + alpha * I_reg,
                                    X_train.T @ y_binary)
                raw_score = float(X_test @ w)
                probs[k] = max(0.0, raw_score)  # clip negative
            except np.linalg.LinAlgError:
                probs[k] = 0.0

        # Normalize and compute expected value
        total = np.sum(probs)
        if total > 0:
            probs = probs / total
            preds[i] = np.sum(probs * BIN_CENTERS)
        else:
            preds[i] = np.nan

    return preds


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ROUND 5: ORDINAL REGRESSION EXPERIMENTS")
    print("=" * 70)

    print("\nLoading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Collect features for all segments
    all_features = []
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

        # --- Segment-level aggregation ---
        b_freq = median_finite(acf_freqs)
        pc_freq = median_finite(pc_freqs)
        fft_pt_freq = median_finite(fft_pt_freqs)
        env_fft_freq = median_finite(env_fft_freqs)
        n_detected = int(np.sum(np.isfinite(acf_freqs)))

        type_is_gpd = 1.0 if subdir == 'gpd' else 0.0
        feature_vec = [a_freq, b_freq, pc_freq, fft_pt_freq, env_fft_freq,
                       type_is_gpd, float(n_detected)]

        all_features.append({
            'mat_name': mat_name,
            'features': feature_vec,
            'expert_freq': expert_freq,
            'subdir': subdir,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Feature extraction: {idx + 1}/{n_total} segments")

    # ── Prepare ML data ──
    valid = [m for m in all_features if np.isfinite(m['expert_freq']) and m['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"\nML samples: {n_ml}")

    X = np.array([m['features'] for m in valid])
    expert_freqs = np.array([m['expert_freq'] for m in valid])
    mat_names = [m['mat_name'] for m in valid]
    subdirs = np.array([m['subdir'] for m in valid])
    type_mask = np.array([1.0 if s == 'gpd' else 0.0 for s in subdirs])

    feature_names = ['MethodA', 'ACF_thr010', 'PeakCount', 'FFT_Pointiness',
                     'EnvelopeFFT_BankC', 'IsGPD', 'N_detected']

    # Impute NaN features with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X[nan_mask, col_idx] = col_median
            n_imputed = int(np.sum(nan_mask))
            print(f"  Imputed {n_imputed} NaN in {feature_names[col_idx]} with median={col_median:.3f}")

    # Bin assignments
    y_bin_idx = np.array([freq_to_bin_index(f) for f in expert_freqs])
    y_log_freq = np.log(expert_freqs)

    print(f"\nBin distribution:")
    for k_idx in range(K):
        count = np.sum(y_bin_idx == k_idx)
        print(f"  Bin {k_idx} ({BIN_CENTERS[k_idx]:.2f} Hz): {count} samples")

    # ════════════════════════════════════════════════════════════════════
    # Variant a) r5_ordinal_ridge: ridge on bin index, round, map back
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (a) r5_ordinal_ridge: ridge on bin index, round to nearest ---")
    y_target_a = y_bin_idx.astype(float)
    preds_a = ridge_loo_cv(X, y_target_a, alpha=1.0)
    preds_a_hz = np.array([bin_index_to_freq(p) for p in preds_a])
    pred_dict_a = {mat_names[i]: float(preds_a_hz[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_a, 'r5_ordinal_ridge')

    # ════════════════════════════════════════════════════════════════════
    # Variant b) r5_ordinal_ridge_continuous: don't round, interpolate
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (b) r5_ordinal_ridge_continuous: don't round, interpolate ---")
    preds_b_hz = np.array([bin_index_to_freq_continuous(p) for p in preds_a])
    pred_dict_b = {mat_names[i]: float(preds_b_hz[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_b, 'r5_ordinal_ridge_continuous')

    # ════════════════════════════════════════════════════════════════════
    # Variant c) r5_ordinal_ridge_logfreq: ordinal bins on log scale
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (c) r5_ordinal_ridge_logfreq: bins on log-frequency scale ---")
    log_bin_centers = np.log(BIN_CENTERS)
    y_target_c = y_bin_idx.astype(float)
    preds_c_idx = ridge_loo_cv(X, y_target_c, alpha=1.0)
    # Map back: use continuous index -> interpolate in log space, then exp
    preds_c_hz = np.full(n_ml, np.nan)
    for i in range(n_ml):
        idx_cont = preds_c_idx[i]
        if not np.isfinite(idx_cont):
            continue
        idx_clamped = max(0.0, min(float(K - 1), idx_cont))
        lower = int(np.floor(idx_clamped))
        upper = min(lower + 1, K - 1)
        frac = idx_clamped - lower
        log_f = log_bin_centers[lower] * (1.0 - frac) + log_bin_centers[upper] * frac
        preds_c_hz[i] = np.exp(log_f)
    pred_dict_c = {mat_names[i]: float(preds_c_hz[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_c, 'r5_ordinal_ridge_logfreq')

    # ════════════════════════════════════════════════════════════════════
    # Variant d) r5_ordinal_bytype: separate models for LPD and GPD
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (d) r5_ordinal_bytype: separate models for LPD and GPD ---")
    preds_d_idx = ridge_loo_cv_bytype(X, y_target_a, type_mask, alpha=1.0)
    preds_d_hz = np.array([bin_index_to_freq_continuous(p) for p in preds_d_idx])
    pred_dict_d = {mat_names[i]: float(preds_d_hz[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_d, 'r5_ordinal_bytype')

    # ════════════════════════════════════════════════════════════════════
    # Variant e) r5_ordinal_plus_continuous: average ordinal + continuous
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (e) r5_ordinal_plus_continuous: avg ordinal + continuous ---")
    # Continuous ridge on log-freq (same as R4)
    preds_continuous_log = ridge_loo_cv(X, y_log_freq, alpha=1.0)
    preds_continuous_hz = np.exp(preds_continuous_log)
    # Average with ordinal continuous prediction (variant b)
    preds_e_hz = np.full(n_ml, np.nan)
    for i in range(n_ml):
        vals = []
        if np.isfinite(preds_b_hz[i]):
            vals.append(preds_b_hz[i])
        if np.isfinite(preds_continuous_hz[i]):
            vals.append(preds_continuous_hz[i])
        if vals:
            preds_e_hz[i] = np.mean(vals)
    pred_dict_e = {mat_names[i]: float(preds_e_hz[i]) for i in range(n_ml)}
    evaluate_predictions(dataset, pred_dict_e, 'r5_ordinal_plus_continuous')

    # ════════════════════════════════════════════════════════════════════
    # Variant f) r5_snap_to_grid: best continuous -> snap to nearest bin
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (f) r5_snap_to_grid: continuous ridge log-freq, snap to grid ---")
    preds_f_hz = np.array([snap_to_nearest_bin(f) for f in preds_continuous_hz])
    pred_dict_f = {mat_names[i]: float(preds_f_hz[i]) for i in range(n_ml)
                   if np.isfinite(preds_f_hz[i])}
    evaluate_predictions(dataset, pred_dict_f, 'r5_snap_to_grid')

    # ════════════════════════════════════════════════════════════════════
    # Variant g) r5_quantile_weighted: softmax weighting over bin centers
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (g) r5_quantile_weighted: softmax distance weighting ---")
    # Use the continuous prediction as anchor, softmax-weight nearby bins
    preds_g_hz = np.full(n_ml, np.nan)
    for i in range(n_ml):
        if not np.isfinite(preds_continuous_hz[i]):
            continue
        # Distances from prediction to each bin center
        distances = np.abs(BIN_CENTERS - preds_continuous_hz[i])
        # Softmax of negative distances (temperature=0.3 for sharper weighting)
        weights = softmax(-distances, temperature=0.3)
        preds_g_hz[i] = np.sum(weights * BIN_CENTERS)
    pred_dict_g = {mat_names[i]: float(preds_g_hz[i]) for i in range(n_ml)
                   if np.isfinite(preds_g_hz[i])}
    evaluate_predictions(dataset, pred_dict_g, 'r5_quantile_weighted')

    # ════════════════════════════════════════════════════════════════════
    # Variant h) r5_bin_classifier: K one-vs-rest, expected value
    # ════════════════════════════════════════════════════════════════════
    print("\n--- (h) r5_bin_classifier: K one-vs-rest classifiers ---")
    preds_h_hz = bin_classifier_loo_cv(X, y_bin_idx, alpha=1.0)
    pred_dict_h = {mat_names[i]: float(preds_h_hz[i]) for i in range(n_ml)
                   if np.isfinite(preds_h_hz[i])}
    evaluate_predictions(dataset, pred_dict_h, 'r5_bin_classifier')

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("ROUND 5 ORDINAL REGRESSION: SUMMARY")
    print("=" * 70)

    variant_names = [
        'r5_ordinal_ridge',
        'r5_ordinal_ridge_continuous',
        'r5_ordinal_ridge_logfreq',
        'r5_ordinal_bytype',
        'r5_ordinal_plus_continuous',
        'r5_snap_to_grid',
        'r5_quantile_weighted',
        'r5_bin_classifier',
    ]

    import json
    from pathlib import Path
    runs_dir = Path(REPO_ROOT) / 'results' / 'optimization_runs'

    header = (f"{'Experiment':<35s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baseline
    print(f"{'Method A (baseline)':<35s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
    print("-" * len(header))

    rows = []
    for vname in variant_names:
        jpath = runs_dir / f'{vname}.json'
        if jpath.exists():
            with open(str(jpath)) as f:
                m = json.load(f)
            rows.append(m)

    rows.sort(key=lambda r: -(r.get('combined_spearman', -999)
                              if isinstance(r.get('combined_spearman'), (int, float))
                              and np.isfinite(r.get('combined_spearman', np.nan))
                              else -999))
    for row in rows:
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
