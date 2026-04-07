"""
Round 6: Teager-Kaiser Energy Operator (TKEO) experiment.

TKEO = x[n]^2 - x[n-1]*x[n+1]
A one-sample computation that suppresses 1/f background and produces sharp pulses
at each discharge. Should be better than pointiness for complex LPD morphology.

Variants:
  r6_tkeo_fft_s002      - TKEO FFT, sigma=0.02*fs
  r6_tkeo_fft_s004      - TKEO FFT, sigma=0.04*fs
  r6_tkeo_acf           - TKEO ACF, sigma=0.02*fs
  r6_tkeo_peaks         - TKEO peak-count, sigma=0.02*fs
  r6_tkeo_plus_peaks    - average of TKEO FFT + TKEO peak-count
  r6_tkeo_ridge         - Ridge on log(freq) with mixed features, LOO-CV
  r6_tkeo_fft_raw       - TKEO on raw bipolar (no 15Hz lowpass)
  r6_tkeo_fft_bandpass  - TKEO on signal bandpassed 1-15Hz
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
ACF_MIN_LAG_POINTINESS = 0.4
ACF_THRESHOLD_POINTINESS = 0.10
PEAK_HEIGHT_FRAC_POINTINESS = 0.3
FS = 200

# TKEO-specific
TKEO_ACF_MIN_LAG = 0.25  # seconds
TKEO_ACF_THRESHOLD = 0.05
TKEO_PEAK_HEIGHT_FRAC = 0.3
TKEO_PEAK_MIN_DIST = 0.2  # seconds


# ── TKEO computation ──────────────────────────────────────────────────
def compute_tkeo(x):
    """Teager-Kaiser Energy Operator: tkeo[n] = x[n]^2 - x[n-1]*x[n+1]
    Returns |tkeo| of length N-2."""
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    return np.abs(tkeo)


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


def preprocess_no_lowpass(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage (NO 15Hz lowpass)."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    return seg


def preprocess_bandpass_1_15(data, fs):
    """Notch 60Hz, bandpass 1-15Hz, bipolar montage."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 1.0, 15.0, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    return seg


def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


# ── TKEO-based frequency estimators (per channel) ─────────────────────
def compute_tkeo_fft_per_channel(seg, fs, sigma_frac):
    """FFT of smoothed TKEO trace per channel, peak in [0.3, 3.5] Hz."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(sigma_frac * fs))
    for i in range(n_ch):
        tkeo = compute_tkeo(seg[i])
        if len(tkeo) < 10:
            continue
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        if np.max(tkeo_smooth) <= 0:
            continue
        n = len(tkeo_smooth)
        fft_vals = np.abs(np.fft.rfft(tkeo_smooth - np.mean(tkeo_smooth)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


def compute_tkeo_acf_per_channel(seg, fs, sigma_frac):
    """ACF of smoothed TKEO trace per channel."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(sigma_frac * fs))
    min_lag = int(TKEO_ACF_MIN_LAG * fs)

    for i in range(n_ch):
        tkeo = compute_tkeo(seg[i])
        if len(tkeo) < 10:
            continue
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        if np.max(tkeo_smooth) <= 0:
            continue

        # Normalize
        tkeo_norm = tkeo_smooth - np.mean(tkeo_smooth)
        n = len(tkeo_norm)

        # Compute ACF via FFT
        fft_x = np.fft.rfft(tkeo_norm, n=2 * n)
        acf_full = np.fft.irfft(np.abs(fft_x) ** 2)[:n]
        if acf_full[0] > 0:
            acf_full /= acf_full[0]
        else:
            continue

        # Find first peak after min_lag
        max_lag = int(1.0 / 0.3 * fs)  # max period ~3.33s
        max_lag = min(max_lag, n - 1)
        if min_lag >= max_lag:
            continue

        acf_segment = acf_full[min_lag:max_lag + 1]
        peaks, props = find_peaks(acf_segment, height=TKEO_ACF_THRESHOLD)
        if len(peaks) == 0:
            continue
        # Pick first (shortest period) peak
        best_peak = peaks[0]
        lag = best_peak + min_lag
        freqs[i] = fs / lag
    return freqs


def compute_tkeo_peaks_per_channel(seg, fs, sigma_frac):
    """Peak-count frequency on TKEO trace per channel. Require >= 3 peaks."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(sigma_frac * fs))
    min_distance = int(TKEO_PEAK_MIN_DIST * fs)

    for i in range(n_ch):
        tkeo = compute_tkeo(seg[i])
        if len(tkeo) < 10:
            continue
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        tkeo_max = np.max(tkeo_smooth)
        if tkeo_max <= 0:
            continue
        peak_height = tkeo_max * TKEO_PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(tkeo_smooth, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            span = (peak_locs[-1] - peak_locs[0]) / fs
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


def compute_acf_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, peaks = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG_POINTINESS,
            acf_peak_threshold=ACF_THRESHOLD_POINTINESS,
            peak_height_frac=PEAK_HEIGHT_FRAC_POINTINESS,
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
        peak_height = trace_max * PEAK_HEIGHT_FRAC_POINTINESS
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


def compute_envelope_fft_per_channel(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C) per channel."""
    try:
        REPO_ROOT = os.path.dirname(CODE_DIR)
        if subdir == 'lpd':
            templates = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
        else:
            templates = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
    except Exception:
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

    # Storage for all results
    all_samples = []

    # Load templates once
    REPO_ROOT = os.path.dirname(CODE_DIR)
    try:
        templates_lpd = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
        templates_gpd = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
        print(f"Loaded templates: LPD {templates_lpd.shape}, GPD {templates_gpd.shape}")
    except Exception as e:
        print(f"Warning: Could not load templates: {e}")
        templates_lpd = None
        templates_gpd = None

    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # ── Method A ──
        f_A = get_method_a_freq(data, fs)

        # ── Standard preprocessing (with 15Hz lowpass) ──
        seg_std = preprocess_standard(data, fs)

        # ── Standard features ──
        acf_freqs_std = compute_acf_per_channel(seg_std, fs)
        f_B = median_finite(acf_freqs_std)

        pc_freqs_std = compute_peak_count_freq_per_channel(seg_std, fs)
        f_peaks_pt = median_finite(pc_freqs_std)

        fft_pt_freqs = compute_fft_pointiness_per_channel(seg_std, fs)
        f_fft_pt = median_finite(fft_pt_freqs)

        env_fft_freqs = compute_envelope_fft_per_channel(seg_std, fs, subdir)
        f_envelope = median_finite(env_fft_freqs)

        # ── TKEO on standard preprocessed signal ──
        # sigma=0.02*fs
        tkeo_fft_s002 = compute_tkeo_fft_per_channel(seg_std, fs, 0.02)
        f_tkeo_fft_s002 = median_finite(tkeo_fft_s002)

        # sigma=0.04*fs
        tkeo_fft_s004 = compute_tkeo_fft_per_channel(seg_std, fs, 0.04)
        f_tkeo_fft_s004 = median_finite(tkeo_fft_s004)

        # ACF on TKEO, sigma=0.02
        tkeo_acf_freqs = compute_tkeo_acf_per_channel(seg_std, fs, 0.02)
        f_tkeo_acf = median_finite(tkeo_acf_freqs)

        # Peak-count on TKEO, sigma=0.02
        tkeo_peaks_freqs = compute_tkeo_peaks_per_channel(seg_std, fs, 0.02)
        f_tkeo_peaks = median_finite(tkeo_peaks_freqs)

        # TKEO FFT + TKEO peaks average
        if np.isfinite(f_tkeo_fft_s002) and np.isfinite(f_tkeo_peaks):
            f_tkeo_plus_peaks = (f_tkeo_fft_s002 + f_tkeo_peaks) / 2.0
        elif np.isfinite(f_tkeo_fft_s002):
            f_tkeo_plus_peaks = f_tkeo_fft_s002
        elif np.isfinite(f_tkeo_peaks):
            f_tkeo_plus_peaks = f_tkeo_peaks
        else:
            f_tkeo_plus_peaks = np.nan

        # ── TKEO on raw bipolar (no 15Hz lowpass) ──
        seg_raw = preprocess_no_lowpass(data, fs)
        tkeo_fft_raw_freqs = compute_tkeo_fft_per_channel(seg_raw, fs, 0.02)
        f_tkeo_fft_raw = median_finite(tkeo_fft_raw_freqs)

        # ── TKEO on bandpassed 1-15Hz ──
        seg_bp = preprocess_bandpass_1_15(data, fs)
        tkeo_fft_bp_freqs = compute_tkeo_fft_per_channel(seg_bp, fs, 0.02)
        f_tkeo_fft_bandpass = median_finite(tkeo_fft_bp_freqs)

        # ── Number of valid channels for TKEO ──
        n_ch_tkeo = int(np.sum(np.isfinite(tkeo_fft_s002)))

        # ── Ridge feature: is_gpd ──
        is_gpd = 1.0 if subdir == 'gpd' else 0.0

        all_samples.append({
            'mat_name': mat_name,
            'subdir': subdir,
            'expert_freq': expert_freq,
            # Simple estimates
            'f_tkeo_fft_s002': f_tkeo_fft_s002,
            'f_tkeo_fft_s004': f_tkeo_fft_s004,
            'f_tkeo_acf': f_tkeo_acf,
            'f_tkeo_peaks': f_tkeo_peaks,
            'f_tkeo_plus_peaks': f_tkeo_plus_peaks,
            'f_tkeo_fft_raw': f_tkeo_fft_raw,
            'f_tkeo_fft_bandpass': f_tkeo_fft_bandpass,
            # Ridge features
            'f_A': f_A,
            'f_B': f_B,
            'f_peaks_pt': f_peaks_pt,
            'f_fft_pt': f_fft_pt,
            'f_envelope': f_envelope,
            'is_gpd': is_gpd,
            'n_ch_tkeo': float(n_ch_tkeo),
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            elapsed = time.time() - t0
            print(f"  Progress: {idx + 1}/{n_total} segments ({elapsed:.0f}s)")

    print(f"\nProcessed {len(all_samples)} segments with data")

    # ── Evaluate simple variants ──
    variant_keys = [
        ('r6_tkeo_fft_s002', 'f_tkeo_fft_s002'),
        ('r6_tkeo_fft_s004', 'f_tkeo_fft_s004'),
        ('r6_tkeo_acf', 'f_tkeo_acf'),
        ('r6_tkeo_peaks', 'f_tkeo_peaks'),
        ('r6_tkeo_plus_peaks', 'f_tkeo_plus_peaks'),
        ('r6_tkeo_fft_raw', 'f_tkeo_fft_raw'),
        ('r6_tkeo_fft_bandpass', 'f_tkeo_fft_bandpass'),
    ]

    for exp_name, key in variant_keys:
        pred_dict = {}
        for s in all_samples:
            v = s[key]
            if np.isfinite(v):
                pred_dict[s['mat_name']] = float(v)
        print(f"\n--- {exp_name}: {len(pred_dict)} predictions ---")
        evaluate_predictions(dataset, pred_dict, exp_name)

    # ── Ridge variant: r6_tkeo_ridge ──
    print("\n--- r6_tkeo_ridge ---")
    ridge_feature_names = [
        'f_A', 'f_B', 'f_peaks_pt', 'f_fft_pt', 'f_envelope',
        'f_tkeo_fft_s002', 'f_tkeo_peaks',
        'is_gpd', 'n_ch_tkeo',
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
    evaluate_predictions(dataset, pred_dict_ridge, 'r6_tkeo_ridge')

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
