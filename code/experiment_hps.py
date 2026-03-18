"""
Round 6: Harmonic Product Spectrum (HPS) — anti-subharmonic frequency estimation.

HPS: P_final(f) = P(f) * P(2f) * P(3f)

The product naturally promotes the true fundamental frequency by requiring
energy at harmonic multiples. Subharmonic peaks (e.g. at f/2) get suppressed
because their 3rd harmonic (3*f/2) typically has no energy.

Variants:
  r6_hps3_pointiness  - 3-harmonic HPS on pointiness FFT, median across channels
  r6_hps2_pointiness  - 2-harmonic HPS on pointiness FFT
  r6_hps3_tkeo        - 3-harmonic HPS on TKEO trace FFT
  r6_hps3_raw         - 3-harmonic HPS on raw bipolar signal FFT
  r6_hps3_plus_peaks  - average of HPS3 pointiness freq + peak-count freq
  r6_hps_ridge        - Ridge on log(freq) with [f_A, f_B, f_peaks, f_fft, f_envelope,
                         f_hps3_pt, f_hps3_tkeo, is_gpd, n_ch]. LOO-CV.
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

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

# Template paths for envelope FFT
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


# ── HPS computation ──────────────────────────────────────────────────
def compute_hps(power_spectrum, n_harmonics=3, freq_lo=0.3, freq_hi=3.5, fs=200, N=None):
    """
    Harmonic Product Spectrum.

    Args:
        power_spectrum: |FFT|^2 or |FFT| array from rfft (length N//2+1)
        n_harmonics: number of harmonics (2 or 3)
        freq_lo, freq_hi: frequency range for peak search (Hz)
        fs: sampling rate
        N: original signal length (for freq resolution)

    Returns:
        peak_freq: frequency of HPS peak in [freq_lo, freq_hi]
    """
    if N is None:
        N = (len(power_spectrum) - 1) * 2

    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    n_bins = len(power_spectrum)

    # Maximum index we can use: for n_harmonics, need index * n_harmonics < n_bins
    max_idx = n_bins // n_harmonics

    if max_idx < 2:
        return np.nan

    # Build HPS: product of downsampled spectra
    hps = power_spectrum[:max_idx].copy()
    for h in range(2, n_harmonics + 1):
        # P_h[k] = P[h*k]
        downsampled = power_spectrum[np.arange(max_idx) * h]
        hps = hps * downsampled

    # Find peak in frequency range
    hps_freqs = freqs[:max_idx]
    mask = (hps_freqs >= freq_lo) & (hps_freqs <= freq_hi)
    if not np.any(mask):
        return np.nan

    hps_masked = hps[mask]
    freq_masked = hps_freqs[mask]

    peak_idx = np.argmax(hps_masked)
    return float(freq_masked[peak_idx])


# ── Per-channel HPS on pointiness ────────────────────────────────────
def compute_hps_pointiness_per_channel(seg, fs, n_harmonics=3):
    """HPS on pointiness trace FFT per channel, peak in [0.3, 3.5] Hz."""
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
        freqs[i] = compute_hps(fft_vals, n_harmonics=n_harmonics, fs=fs, N=n)
    return freqs


# ── Per-channel HPS on TKEO ──────────────────────────────────────────
def compute_tkeo_trace(signal_1d):
    """Teager-Kaiser Energy Operator: |x(n)^2 - x(n-1)*x(n+1)|"""
    x = signal_1d
    n = len(x)
    if n < 3:
        return np.zeros(n)
    tkeo = np.zeros(n)
    tkeo[1:-1] = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
    tkeo[0] = tkeo[1]
    tkeo[-1] = tkeo[-2]
    return tkeo


def compute_hps_tkeo_per_channel(seg, fs, n_harmonics=3):
    """HPS on TKEO trace FFT per channel."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    for i in range(n_ch):
        tkeo = compute_tkeo_trace(seg[i])
        tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
        if np.max(tkeo) <= 0:
            continue
        n = len(tkeo)
        fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
        freqs[i] = compute_hps(fft_vals, n_harmonics=n_harmonics, fs=fs, N=n)
    return freqs


# ── Per-channel HPS on raw bipolar ───────────────────────────────────
def compute_hps_raw_per_channel(seg, fs, n_harmonics=3):
    """HPS on raw bipolar signal FFT per channel."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        ch = seg[i]
        n = len(ch)
        fft_vals = np.abs(np.fft.rfft(ch - np.mean(ch)))
        freqs[i] = compute_hps(fft_vals, n_harmonics=n_harmonics, fs=fs, N=n)
    return freqs


# ── Existing feature extractors (for ridge model) ────────────────────
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
    """ACF frequency per channel (Method B). Returns freqs array."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, peaks = compute_acf_frequency(
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
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
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
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


# ── Ridge LOO-CV helper ──────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge regression. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]

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


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading dataset...")
    dataset = load_dataset()
    n_total = len(dataset)
    print(f"Dataset: {n_total} segments")

    # Storage for all variants
    preds_hps3_pt = {}
    preds_hps2_pt = {}
    preds_hps3_tkeo = {}
    preds_hps3_raw = {}
    preds_hps3_plus_peaks = {}

    # For ridge model
    ridge_samples = []

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {idx + 1}/{n_total} segments ({elapsed:.0f}s)")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # --- Method A freq ---
        f_A = get_method_a_freq(data, fs)

        # --- Preprocess ---
        seg = preprocess_segment(data, fs)
        n_ch = seg.shape[0]

        # --- Method B (ACF) freq ---
        acf_freqs = compute_acf_per_channel(seg, fs)
        f_B = median_finite(acf_freqs)

        # --- Peak-count freq ---
        pc_freqs = compute_peak_count_freq_per_channel(seg, fs)
        f_peaks = median_finite(pc_freqs)

        # --- FFT pointiness freq ---
        fft_pt_freqs = compute_fft_pointiness_per_channel(seg, fs)
        f_fft = median_finite(fft_pt_freqs)

        # --- Envelope FFT freq ---
        env_fft_freqs = compute_envelope_fft_per_channel(seg, fs, subdir)
        f_envelope = median_finite(env_fft_freqs)

        # --- HPS3 on pointiness ---
        hps3_pt_freqs = compute_hps_pointiness_per_channel(seg, fs, n_harmonics=3)
        f_hps3_pt = median_finite(hps3_pt_freqs)

        # --- HPS2 on pointiness ---
        hps2_pt_freqs = compute_hps_pointiness_per_channel(seg, fs, n_harmonics=2)
        f_hps2_pt = median_finite(hps2_pt_freqs)

        # --- HPS3 on TKEO ---
        hps3_tkeo_freqs = compute_hps_tkeo_per_channel(seg, fs, n_harmonics=3)
        f_hps3_tkeo = median_finite(hps3_tkeo_freqs)

        # --- HPS3 on raw ---
        hps3_raw_freqs = compute_hps_raw_per_channel(seg, fs, n_harmonics=3)
        f_hps3_raw = median_finite(hps3_raw_freqs)

        # --- Variant a: r6_hps3_pointiness ---
        if np.isfinite(f_hps3_pt):
            preds_hps3_pt[mat_name] = f_hps3_pt

        # --- Variant b: r6_hps2_pointiness ---
        if np.isfinite(f_hps2_pt):
            preds_hps2_pt[mat_name] = f_hps2_pt

        # --- Variant c: r6_hps3_tkeo ---
        if np.isfinite(f_hps3_tkeo):
            preds_hps3_tkeo[mat_name] = f_hps3_tkeo

        # --- Variant d: r6_hps3_raw ---
        if np.isfinite(f_hps3_raw):
            preds_hps3_raw[mat_name] = f_hps3_raw

        # --- Variant e: r6_hps3_plus_peaks ---
        if np.isfinite(f_hps3_pt) and np.isfinite(f_peaks):
            preds_hps3_plus_peaks[mat_name] = (f_hps3_pt + f_peaks) / 2.0
        elif np.isfinite(f_hps3_pt):
            preds_hps3_plus_peaks[mat_name] = f_hps3_pt
        elif np.isfinite(f_peaks):
            preds_hps3_plus_peaks[mat_name] = f_peaks

        # --- Collect for ridge ---
        is_gpd = 1.0 if subdir == 'gpd' else 0.0
        n_detected = int(np.sum(np.isfinite(acf_freqs)))

        ridge_samples.append({
            'mat_name': mat_name,
            'subdir': subdir,
            'expert_freq': expert_freq,
            'features': [
                f_A, f_B, f_peaks, f_fft, f_envelope,
                f_hps3_pt, f_hps3_tkeo,
                is_gpd, float(n_detected),
            ],
        })

    elapsed = time.time() - t0
    print(f"\nFeature extraction done in {elapsed:.0f}s")

    # ── Evaluate variants a-e ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Evaluating HPS variants...")
    print(f"{'=' * 60}")

    print(f"\n--- r6_hps3_pointiness ({len(preds_hps3_pt)} predictions) ---")
    evaluate_predictions(dataset, preds_hps3_pt, 'r6_hps3_pointiness')

    print(f"\n--- r6_hps2_pointiness ({len(preds_hps2_pt)} predictions) ---")
    evaluate_predictions(dataset, preds_hps2_pt, 'r6_hps2_pointiness')

    print(f"\n--- r6_hps3_tkeo ({len(preds_hps3_tkeo)} predictions) ---")
    evaluate_predictions(dataset, preds_hps3_tkeo, 'r6_hps3_tkeo')

    print(f"\n--- r6_hps3_raw ({len(preds_hps3_raw)} predictions) ---")
    evaluate_predictions(dataset, preds_hps3_raw, 'r6_hps3_raw')

    print(f"\n--- r6_hps3_plus_peaks ({len(preds_hps3_plus_peaks)} predictions) ---")
    evaluate_predictions(dataset, preds_hps3_plus_peaks, 'r6_hps3_plus_peaks')

    # ── Variant f: r6_hps_ridge ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Running Ridge LOO-CV (r6_hps_ridge)...")
    print(f"{'=' * 60}")

    feature_names = [
        'f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope',
        'f_hps3_pt', 'f_hps3_tkeo',
        'is_gpd', 'n_ch',
    ]

    valid = [s for s in ridge_samples
             if np.isfinite(s['expert_freq']) and s['expert_freq'] > 0]
    n_ml = len(valid)
    print(f"  ML samples: {n_ml}")

    X = np.array([s['features'] for s in valid], dtype=float)
    y_freq = np.array([s['expert_freq'] for s in valid])
    y_log = np.log(y_freq)
    mat_names = [s['mat_name'] for s in valid]

    # Impute NaN
    impute_nan_with_median(X, feature_names)

    # Ridge LOO-CV on log(freq)
    preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
    preds_ridge = np.exp(preds_log)
    preds_ridge = np.clip(preds_ridge, 0.2, 4.0)
    pred_dict_ridge = {mat_names[i]: float(preds_ridge[i])
                       for i in range(n_ml) if np.isfinite(preds_ridge[i])}

    evaluate_predictions(dataset, pred_dict_ridge, 'r6_hps_ridge')

    # Print feature importances
    print("\n--- Feature importances (full model) ---")
    X_b = np.column_stack([X, np.ones(len(X))])
    I_reg = np.eye(X_b.shape[1])
    I_reg[-1, -1] = 0
    try:
        w = np.linalg.solve(X_b.T @ X_b + 1.0 * I_reg, X_b.T @ y_log)
        coefs = w[:-1]
        importance = sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))
        print(f"  {'Feature':>20s}  {'Coefficient':>12s}")
        print(f"  {'-' * 20}  {'-' * 12}")
        for name, coef in importance:
            print(f"  {name:>20s}  {coef:+12.6f}")
        print(f"  {'intercept':>20s}  {w[-1]:+12.6f}")
    except np.linalg.LinAlgError:
        print("  Could not compute feature importances.")

    total_elapsed = time.time() - t0
    print(f"\nTotal time: {total_elapsed:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
