"""
Round 4 Combiner: Type-specific approaches + matched-filter envelope FFT + ML.

Combinations tested:
  1. r4_type_specific_best: LPD=ACF+peakcount+Bayesian, GPD=FFT_pointiness+peakcount
  2. r4_type_specific_v2: LPD=matched-filter envelope FFT, GPD=FFT+peakcount
  3. r4_type_specific_v3: LPD=median(envelope_fft, acf+peakcount, methodA), GPD=FFT+peakcount
  4. r4_envelope_plus_peakcount: avg(envelope_fft, peakcount) for all
  5. r4_envelope_plus_fft_pointiness: avg(envelope_fft, fft_pointiness) for all
  6. r4_ml_with_envelope: Ridge regression LOO-CV on log(freq)
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
GPD_PRIOR = 1.0
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


def bayesian_nudge(freq, subdir):
    if not np.isfinite(freq):
        return freq
    prior = GPD_PRIOR if subdir == 'gpd' else LPD_PRIOR
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
    """Peak-count frequency per channel."""
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
        # FFT
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
        # Restrict to [0.3, 3.5] Hz
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


def compute_envelope_fft_per_channel(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C) per channel.

    Cross-correlate each channel with each template, take max across templates
    to get envelope, then FFT of envelope, peak in [0.3, 3.5] Hz.
    """
    templates = TEMPLATES_C_LPD if subdir == 'lpd' else TEMPLATES_C_GPD
    if templates is None:
        return np.full(seg.shape[0], np.nan)

    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)

    for i in range(n_ch):
        ch = seg[i]
        n_samples = len(ch)
        # Cross-correlate with each template, take max envelope
        envelope = np.zeros(n_samples)
        for t_idx in range(templates.shape[0]):
            tmpl = templates[t_idx]
            # Trim template if longer than channel
            if len(tmpl) > n_samples:
                tmpl = tmpl[:n_samples]
            corr = np.correlate(ch, tmpl, mode='same')
            corr = np.abs(corr)
            envelope = np.maximum(envelope, corr)

        if np.max(envelope) <= 0:
            continue

        # FFT of envelope
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

    # Pre-allocate predictions
    combo_names = [
        'r4_type_specific_best',
        'r4_type_specific_v2',
        'r4_type_specific_v3',
        'r4_envelope_plus_peakcount',
        'r4_envelope_plus_fft_pointiness',
        'r4_ml_with_envelope',
    ]
    all_predictions = {name: {} for name in combo_names}

    # For ML: collect feature vectors
    ml_features = []  # list of (mat_name, feature_vec, expert_freq, subdir)

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
        b_freq = median_finite(acf_freqs)          # Method B (ACF thr=0.10)
        pc_freq = median_finite(pc_freqs)           # Peak-count
        fft_pt_freq = median_finite(fft_pt_freqs)   # FFT of pointiness
        env_fft_freq = median_finite(env_fft_freqs)  # Envelope FFT (Bank C)
        n_detected = int(np.sum(np.isfinite(acf_freqs)))

        # ── Combo 1: r4_type_specific_best ──
        # LPD: ACF thr=0.10 + peakcount avg + Bayesian (prior=1.25)
        # GPD: FFT of pointiness + peak-count average
        if subdir == 'lpd':
            vals = [v for v in [b_freq, pc_freq] if np.isfinite(v)]
            freq1 = float(np.mean(vals)) if vals else np.nan
            freq1 = bayesian_nudge(freq1, subdir)
        else:
            vals = [v for v in [fft_pt_freq, pc_freq] if np.isfinite(v)]
            freq1 = float(np.mean(vals)) if vals else np.nan
        all_predictions['r4_type_specific_best'][mat_name] = freq1

        # ── Combo 2: r4_type_specific_v2 ──
        # LPD: matched-filter envelope FFT (Bank C)
        # GPD: FFT of pointiness + peak-count average
        if subdir == 'lpd':
            freq2 = env_fft_freq
        else:
            vals = [v for v in [fft_pt_freq, pc_freq] if np.isfinite(v)]
            freq2 = float(np.mean(vals)) if vals else np.nan
        all_predictions['r4_type_specific_v2'][mat_name] = freq2

        # ── Combo 3: r4_type_specific_v3 ──
        # LPD: median(envelope_fft, acf+peakcount_avg, methodA)
        # GPD: FFT + peakcount
        if subdir == 'lpd':
            # acf+peakcount avg
            vals_bp = [v for v in [b_freq, pc_freq] if np.isfinite(v)]
            bp_avg = float(np.mean(vals_bp)) if vals_bp else np.nan
            candidates = [v for v in [env_fft_freq, bp_avg, a_freq] if np.isfinite(v)]
            freq3 = float(np.median(candidates)) if candidates else np.nan
        else:
            vals = [v for v in [fft_pt_freq, pc_freq] if np.isfinite(v)]
            freq3 = float(np.mean(vals)) if vals else np.nan
        all_predictions['r4_type_specific_v3'][mat_name] = freq3

        # ── Combo 4: r4_envelope_plus_peakcount ──
        # Average envelope FFT with peak-count, for all
        vals4 = [v for v in [env_fft_freq, pc_freq] if np.isfinite(v)]
        freq4 = float(np.mean(vals4)) if vals4 else np.nan
        all_predictions['r4_envelope_plus_peakcount'][mat_name] = freq4

        # ── Combo 5: r4_envelope_plus_fft_pointiness ──
        # Average envelope FFT with FFT-of-pointiness, for all
        vals5 = [v for v in [env_fft_freq, fft_pt_freq] if np.isfinite(v)]
        freq5 = float(np.mean(vals5)) if vals5 else np.nan
        all_predictions['r4_envelope_plus_fft_pointiness'][mat_name] = freq5

        # ── Collect features for ML (#6) ──
        type_is_gpd = 1.0 if subdir == 'gpd' else 0.0
        feature_vec = [
            a_freq if np.isfinite(a_freq) else 0.0,
            b_freq if np.isfinite(b_freq) else 0.0,
            pc_freq if np.isfinite(pc_freq) else 0.0,
            fft_pt_freq if np.isfinite(fft_pt_freq) else 0.0,
            env_fft_freq if np.isfinite(env_fft_freq) else 0.0,
            type_is_gpd,
            float(n_detected),
        ]
        # Track which features were missing (for imputation flag)
        has_all = all(np.isfinite(v) for v in [a_freq, b_freq, pc_freq, fft_pt_freq, env_fft_freq])
        ml_features.append({
            'mat_name': mat_name,
            'features': feature_vec,
            'expert_freq': expert_freq,
            'subdir': subdir,
            'has_all': has_all,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # ── ML model (#6): Ridge regression with LOO-CV ──
    print("\nRunning ML model (Ridge regression, LOO-CV)...")

    # Prepare data
    valid_ml = [m for m in ml_features if np.isfinite(m['expert_freq']) and m['expert_freq'] > 0]
    n_ml = len(valid_ml)
    print(f"  ML samples: {n_ml}")

    X = np.array([m['features'] for m in valid_ml])
    y = np.log(np.array([m['expert_freq'] for m in valid_ml]))
    mat_names_ml = [m['mat_name'] for m in valid_ml]

    feature_names = ['MethodA', 'MethodB_ACF', 'PeakCount', 'FFT_Pointiness',
                     'EnvelopeFFT', 'IsGPD', 'N_detected']

    # Try sklearn Ridge, else numpy
    try:
        from sklearn.linear_model import Ridge
        use_sklearn = True
        print("  Using sklearn Ridge")
    except ImportError:
        use_sklearn = False
        print("  Using numpy least squares")

    # LOO-CV
    loo_predictions = np.full(n_ml, np.nan)
    alpha = 1.0  # Ridge regularization

    for i in range(n_ml):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i+1]

        if use_sklearn:
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_train, y_train)
            loo_predictions[i] = np.exp(model.predict(X_test)[0])
        else:
            # Add intercept
            X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
            X_test_b = np.column_stack([X_test, np.ones(1)])
            # Ridge: (X'X + alpha*I)^-1 X'y
            I = np.eye(X_train_b.shape[1])
            I[-1, -1] = 0  # Don't regularize intercept
            try:
                w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I, X_train_b.T @ y_train)
                loo_predictions[i] = np.exp(X_test_b @ w)
            except np.linalg.LinAlgError:
                loo_predictions[i] = np.nan

    # Store ML predictions
    for i, m in enumerate(valid_ml):
        all_predictions['r4_ml_with_envelope'][m['mat_name']] = float(loo_predictions[i])

    # Print feature coefficients (fit on full data)
    if use_sklearn:
        model_full = Ridge(alpha=alpha, fit_intercept=True)
        model_full.fit(X, y)
        print("\n  Feature coefficients (full model):")
        for fname, coef in zip(feature_names, model_full.coef_):
            print(f"    {fname:>20s}: {coef:+.4f}")
        print(f"    {'intercept':>20s}: {model_full.intercept_:+.4f}")
    else:
        X_b = np.column_stack([X, np.ones(len(X))])
        I = np.eye(X_b.shape[1])
        I[-1, -1] = 0
        try:
            w = np.linalg.solve(X_b.T @ X_b + alpha * I, X_b.T @ y)
            print("\n  Feature coefficients (full model):")
            for fname, coef in zip(feature_names, w[:-1]):
                print(f"    {fname:>20s}: {coef:+.4f}")
            print(f"    {'intercept':>20s}: {w[-1]:+.4f}")
        except:
            print("  Could not compute coefficients")

    # ── Evaluate all combos ──
    print("\n" + "=" * 70)
    print("EVALUATING ALL ROUND 4 COMBINATIONS")
    print("=" * 70)

    summary_rows = []
    for combo_name in combo_names:
        metrics = evaluate_predictions(dataset, all_predictions[combo_name], combo_name)
        summary_rows.append(metrics)

    # ── Print results table ──
    print("\n" + "=" * 70)
    print("ROUND 4 COMBINER RESULTS (sorted by combined Spearman)")
    print("=" * 70)
    header = (f"{'Experiment':<40s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baselines
    print(f"{'Method A (baseline)':<40s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
    print(f"{'r2_round1_best (best LPD)':<40s} {'---':>8s} {'---':>8s} "
          f"{'0.425':>7s} {'---':>7s} {'---':>8s} {'---':>9s}")
    print(f"{'r3_fft_plus_peakcount (best GPD)':<40s} {'---':>8s} {'---':>8s} "
          f"{'---':>7s} {'0.475':>7s} {'0.375':>8s} {'---':>9s}")
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
        print(f"{row['experiment']:<40s} {fmt('lpd_mae'):>8s} {fmt('gpd_mae'):>8s} "
              f"{fmt('lpd_spearman_r'):>7s} {fmt('gpd_spearman_r'):>7s} "
              f"{fmt('combined_spearman'):>8s} {fmt('combined_mae'):>9s}")

    print("\nDone! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
