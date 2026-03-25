"""
Round 4 Combiner v2: 4 focused type-specific + ML combinations.

Variants:
  1. r4_type_specific_best:  LPD=ACF+peakcount avg + Bayesian, GPD=FFT_pt+peakcount avg
  2. r4_type_specific_v2:    LPD=median(MethodA, ACF, peakcount) + Bayesian, GPD=FFT_pt+peakcount avg
  3. r4_five_way_median:     median of 5 freq estimates (all types)
  4. r4_ml_log_ridge:        Ridge on log(freq), LOO-CV, 7 features
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
LPD_PRIOR = 1.25
BAYESIAN_ALGO_WEIGHT = 0.7
BAYESIAN_PRIOR_WEIGHT = 0.3
FS = 200

# Template paths
REPO_ROOT = os.path.dirname(CODE_DIR)
TEMPLATES_C_LPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy')
TEMPLATES_C_GPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy')

# Load templates at module level
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


def bayesian_nudge(freq, prior=LPD_PRIOR):
    if not np.isfinite(freq):
        return freq
    return BAYESIAN_ALGO_WEIGHT * freq + BAYESIAN_PRIOR_WEIGHT * prior


def compute_acf_freq_per_channel(seg, fs):
    """ACF frequency (Method B with thr=0.10) per channel."""
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


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    combo_names = [
        'r4_type_specific_best',
        'r4_type_specific_v2',
        'r4_five_way_median',
        'r4_ml_log_ridge',
    ]
    all_predictions = {name: {} for name in combo_names}

    # For ML: collect feature vectors
    ml_features = []

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
        b_freq = median_finite(acf_freqs)          # ACF thr=0.10
        pc_freq = median_finite(pc_freqs)           # Peak-count
        fft_pt_freq = median_finite(fft_pt_freqs)   # FFT of pointiness
        env_fft_freq = median_finite(env_fft_freqs)  # Envelope FFT (Bank C)
        n_detected = int(np.sum(np.isfinite(acf_freqs)))

        # ── Combo 1: r4_type_specific_best ──
        # LPD: ACF + peakcount average, then Bayesian nudge (prior=1.25)
        # GPD: FFT of pointiness + peak-count average
        if subdir == 'lpd':
            vals = [v for v in [b_freq, pc_freq] if np.isfinite(v)]
            freq1 = float(np.mean(vals)) if vals else np.nan
            freq1 = bayesian_nudge(freq1, prior=LPD_PRIOR)
        else:
            vals = [v for v in [fft_pt_freq, pc_freq] if np.isfinite(v)]
            freq1 = float(np.mean(vals)) if vals else np.nan
        all_predictions['r4_type_specific_best'][mat_name] = freq1

        # ── Combo 2: r4_type_specific_v2 ──
        # LPD: median(MethodA, ACF-thr010, peakcount) + Bayesian nudge (prior=1.25)
        # GPD: FFT of pointiness + peak-count average
        if subdir == 'lpd':
            candidates = [v for v in [a_freq, b_freq, pc_freq] if np.isfinite(v)]
            freq2 = float(np.median(candidates)) if candidates else np.nan
            freq2 = bayesian_nudge(freq2, prior=LPD_PRIOR)
        else:
            vals = [v for v in [fft_pt_freq, pc_freq] if np.isfinite(v)]
            freq2 = float(np.mean(vals)) if vals else np.nan
        all_predictions['r4_type_specific_v2'][mat_name] = freq2

        # ── Combo 3: r4_five_way_median ──
        # Median of 5 estimates: MethodA, ACF, peakcount, FFT-pointiness, envelope-FFT
        five = [v for v in [a_freq, b_freq, pc_freq, fft_pt_freq, env_fft_freq]
                if np.isfinite(v)]
        freq3 = float(np.median(five)) if five else np.nan
        all_predictions['r4_five_way_median'][mat_name] = freq3

        # ── Collect features for ML (combo 4) ──
        type_is_gpd = 1.0 if subdir == 'gpd' else 0.0
        feature_vec = [a_freq, b_freq, pc_freq, fft_pt_freq, env_fft_freq,
                       type_is_gpd, float(n_detected)]
        ml_features.append({
            'mat_name': mat_name,
            'features': feature_vec,
            'expert_freq': expert_freq,
            'subdir': subdir,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # ── ML model (combo 4): Ridge regression with LOO-CV ──
    print("\nRunning ML model (Ridge regression, LOO-CV on log(freq))...")

    valid_ml = [m for m in ml_features if np.isfinite(m['expert_freq']) and m['expert_freq'] > 0]
    n_ml = len(valid_ml)
    print(f"  ML samples: {n_ml}")

    X = np.array([m['features'] for m in valid_ml])
    y = np.log(np.array([m['expert_freq'] for m in valid_ml]))

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
            print(f"  Imputed {n_imputed} NaN values in {feature_names[col_idx]} with median={col_median:.3f}")

    # LOO-CV with numpy Ridge: beta = (X'X + alpha*I)^-1 X'y
    alpha = 1.0
    loo_predictions = np.full(n_ml, np.nan)

    for i in range(n_ml):
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
            pred_log = float(X_test_b @ w)
            loo_predictions[i] = np.exp(pred_log)
        except np.linalg.LinAlgError:
            loo_predictions[i] = np.nan

    # Store ML predictions
    for i, m in enumerate(valid_ml):
        all_predictions['r4_ml_log_ridge'][m['mat_name']] = float(loo_predictions[i])

    # Print feature coefficients (full model)
    X_b = np.column_stack([X, np.ones(len(X))])
    I_reg = np.eye(X_b.shape[1])
    I_reg[-1, -1] = 0
    try:
        w_full = np.linalg.solve(X_b.T @ X_b + alpha * I_reg, X_b.T @ y)
        print("\n  Feature coefficients (full model, Ridge alpha=1.0):")
        for fname, coef in zip(feature_names, w_full[:-1]):
            print(f"    {fname:>20s}: {coef:+.4f}")
        print(f"    {'intercept':>20s}: {w_full[-1]:+.4f}")
    except Exception as e:
        print(f"  Could not compute full-model coefficients: {e}")

    # ── Evaluate all combos ──
    print("\n" + "=" * 70)
    print("EVALUATING ALL ROUND 4 v2 COMBINATIONS")
    print("=" * 70)

    summary_rows = []
    for combo_name in combo_names:
        metrics = evaluate_predictions(dataset, all_predictions[combo_name], combo_name)
        summary_rows.append(metrics)

    # ── Print results table ──
    print("\n" + "=" * 70)
    print("ROUND 4 v2 COMBINER RESULTS (sorted by combined Spearman)")
    print("=" * 70)
    header = (f"{'Experiment':<35s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baselines
    print(f"{'Method A (baseline)':<35s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
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
