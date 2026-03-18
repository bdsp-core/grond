"""
Round 6: Generalized Eigenvalue Decomposition (GED) spatial filtering for frequency estimation.

LPDs are lateralized — ~9 of 18 channels are pure noise from the quiet hemisphere.
Per-channel frequency → median is diluted by these noisy channels.
GED creates an optimal spatial filter that maximizes periodic signal relative to background.

Variants:
  r6_ged_fft        - FFT of GED-filtered pointiness
  r6_ged_acf        - ACF of GED-filtered pointiness
  r6_ged_peaks      - Peak-count on GED-filtered signal
  r6_ged_plus_peakcount - Average of GED FFT + per-channel peak-count
  r6_ged_ridge      - Ridge on log(freq) with GED + per-channel features
  r6_ged_top3       - Top 3 GED components, FFT on each, take median
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
    pd_detect_pointiness_acf, bipolar_channels, left_indices, right_indices,
)
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from mne.filter import notch_filter, filter_data

# ── Constants ──────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.25
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200
N_BIPOLAR = 18

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


# ── Helpers ────────────────────────────────────────────────────────────
def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    return seg


def lowpass_signal(seg, fs, cutoff=15.0):
    """Apply lowpass filter to each channel."""
    b_lp, a_lp = butter(4, cutoff / (fs / 2), btype='low')
    out = seg.copy()
    for i in range(out.shape[0]):
        try:
            out[i] = filtfilt(b_lp, a_lp, out[i])
        except ValueError:
            pass
    return out


def get_rough_freq(data, fs):
    """Get rough frequency estimate using pd_detect_pointiness_acf."""
    try:
        result = pd_detect_pointiness_acf(data, fs, acf_peak_threshold=0.10)
        f = result.get('event_frequency', np.nan)
        if f is not None and np.isfinite(f) and f > 0:
            return float(f)
    except Exception:
        pass
    return 1.0  # default


def ged_spatial_filter(seg_broadband, fs, rough_freq, n_components=1):
    """
    Generalized Eigenvalue Decomposition spatial filter.

    Args:
        seg_broadband: (18, n_samples) broadband bipolar signal (0.5-40 Hz)
        fs: sampling rate
        rough_freq: rough frequency estimate in Hz
        n_components: number of top components to return

    Returns:
        filtered_signals: (n_components, n_samples) GED-filtered signals
        eigenvalues: top n eigenvalues
    """
    n_ch, n_samples = seg_broadband.shape

    # Narrowband filter around rough_freq
    low = rough_freq - 0.3
    high = rough_freq + 0.3

    # Widen if too narrow
    if (high - low) < 0.6:
        low = rough_freq - 0.5
        high = rough_freq + 0.5

    # Clamp to valid range
    low = max(0.2, low)
    high = min(4.0, high)

    # Ensure low < high and valid for filter
    if low >= high:
        low = max(0.2, high - 0.6)
    if low >= high:
        # Can't filter, return broadband mean
        mean_sig = np.mean(seg_broadband, axis=0, keepdims=True)
        return mean_sig[:n_components], np.array([1.0] * n_components)

    # Narrowband filter (order=2, butter, filtfilt)
    try:
        b_nb, a_nb = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
        X_narrow = np.zeros_like(seg_broadband)
        for i in range(n_ch):
            X_narrow[i] = filtfilt(b_nb, a_nb, seg_broadband[i])
    except Exception:
        mean_sig = np.mean(seg_broadband, axis=0, keepdims=True)
        return mean_sig[:n_components], np.array([1.0] * n_components)

    # Covariance matrices
    T = n_samples
    S_wide = (1.0 / T) * (seg_broadband @ seg_broadband.T)
    S_freq = (1.0 / T) * (X_narrow @ X_narrow.T)

    # Solve generalized eigenvalue problem
    try:
        eigenvalues, eigenvectors = eigh(S_freq, S_wide)
    except np.linalg.LinAlgError:
        # Add regularization
        S_wide += 0.01 * np.eye(n_ch)
        try:
            eigenvalues, eigenvectors = eigh(S_freq, S_wide)
        except np.linalg.LinAlgError:
            mean_sig = np.mean(seg_broadband, axis=0, keepdims=True)
            return mean_sig[:n_components], np.array([1.0] * n_components)

    # eigh returns eigenvalues in ascending order, take top n_components (largest)
    top_indices = np.argsort(eigenvalues)[::-1][:n_components]
    top_eigenvalues = eigenvalues[top_indices]

    # Apply spatial filters
    filtered_signals = np.zeros((n_components, n_samples))
    for k, idx in enumerate(top_indices):
        w = eigenvectors[:, idx]
        filtered_signals[k] = w.T @ seg_broadband

    return filtered_signals, top_eigenvalues


def ged_fft_frequency(signal_1d, fs):
    """FFT of pointiness trace on a 1D signal. Returns peak freq in [0.3, 3.5] Hz."""
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    if np.max(trace) <= 0:
        return np.nan
    n = len(trace)
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freq_sub = fft_freqs[mask]
    return float(freq_sub[np.argmax(fft_sub)])


def ged_acf_frequency(signal_1d, fs):
    """ACF of pointiness trace on a 1D signal."""
    freq, score, _ = compute_acf_frequency(
        signal_1d, fs, method='pointiness',
        smoothing_sigma=SMOOTHING_SIGMA,
        acf_min_lag=ACF_MIN_LAG,
        acf_peak_threshold=ACF_THRESHOLD,
        peak_height_frac=PEAK_HEIGHT_FRAC,
    )
    return freq


def ged_peak_count_frequency(signal_1d, fs):
    """Peak-count frequency on a 1D signal's pointiness trace."""
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    trace_max = np.max(trace)
    if trace_max <= 0:
        return np.nan
    peak_height = trace_max * PEAK_HEIGHT_FRAC
    min_distance = int(0.2 * fs)
    peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
    if len(peak_locs) >= 3:
        return (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return np.nan


# ── Per-channel helpers (for standard features) ───────────────────────
def compute_per_channel_acf(seg, fs):
    """ACF freq per channel."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    scores = np.full(n_ch, 0.0)
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=0.4,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
        scores[i] = score
    return freqs, scores


def compute_per_channel_peaks(seg, fs):
    """Peak-count freq per channel."""
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


def compute_per_channel_fft(seg, fs):
    """FFT of pointiness per channel."""
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
    return freqs


def compute_envelope_freq(seg, fs, subdir):
    """Matched-filter envelope FFT per channel."""
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
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return freqs


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Storage for all predictions
    preds_ged_fft = {}
    preds_ged_acf = {}
    preds_ged_peaks = {}
    preds_ged_plus_peakcount = {}
    preds_ged_top3 = {}

    # Storage for ridge features
    ridge_features = []  # list of feature vectors
    ridge_targets = []   # log(expert_consensus_freq)
    ridge_mat_names = []
    ridge_subdirs = []

    t0 = time.time()

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing {idx+1}/{len(dataset)} ({elapsed:.1f}s elapsed)")

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            # ── Preprocessing ──────────────────────────────────────
            seg_broadband = preprocess_segment(data, fs)  # (18, n_samples)

            # Apply lowpass for pointiness-based per-channel analysis
            seg_lp = lowpass_signal(seg_broadband, fs, LOWPASS_HZ)

            # ── Rough frequency estimate ───────────────────────────
            rough_freq = get_rough_freq(data, fs)

            # ── GED (top 1 component) ─────────────────────────────
            filtered_1, evals_1 = ged_spatial_filter(seg_broadband, fs, rough_freq, n_components=1)

            # Apply 15Hz lowpass to GED-filtered signal
            b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
            try:
                ged_signal = filtfilt(b_lp, a_lp, filtered_1[0])
            except ValueError:
                ged_signal = filtered_1[0]

            # ── GED frequency estimates ────────────────────────────
            f_ged_fft = ged_fft_frequency(ged_signal, fs)
            f_ged_acf = ged_acf_frequency(ged_signal, fs)
            f_ged_peaks = ged_peak_count_frequency(ged_signal, fs)

            preds_ged_fft[mat_name] = f_ged_fft
            preds_ged_acf[mat_name] = f_ged_acf
            preds_ged_peaks[mat_name] = f_ged_peaks

            # ── GED top 3 ─────────────────────────────────────────
            filtered_3, evals_3 = ged_spatial_filter(seg_broadband, fs, rough_freq, n_components=3)
            top3_fft_freqs = []
            for k in range(min(3, filtered_3.shape[0])):
                try:
                    sig_k = filtfilt(b_lp, a_lp, filtered_3[k])
                except ValueError:
                    sig_k = filtered_3[k]
                f_k = ged_fft_frequency(sig_k, fs)
                if np.isfinite(f_k):
                    top3_fft_freqs.append(f_k)
            preds_ged_top3[mat_name] = float(np.median(top3_fft_freqs)) if top3_fft_freqs else np.nan

            # ── Per-channel standard features ──────────────────────
            acf_freqs, acf_scores = compute_per_channel_acf(seg_lp, fs)
            peak_freqs = compute_per_channel_peaks(seg_lp, fs)
            fft_freqs = compute_per_channel_fft(seg_lp, fs)
            envelope_freqs = compute_envelope_freq(seg_lp, fs, subdir)

            f_A = median_finite(acf_freqs)
            f_B = rough_freq  # from pd_detect_pointiness_acf
            f_peaks = median_finite(peak_freqs)
            f_fft = median_finite(fft_freqs)
            f_envelope = median_finite(envelope_freqs)

            # ── GED + peakcount combo ──────────────────────────────
            if np.isfinite(f_ged_fft) and np.isfinite(f_peaks):
                preds_ged_plus_peakcount[mat_name] = (f_ged_fft + f_peaks) / 2.0
            elif np.isfinite(f_ged_fft):
                preds_ged_plus_peakcount[mat_name] = f_ged_fft
            elif np.isfinite(f_peaks):
                preds_ged_plus_peakcount[mat_name] = f_peaks
            else:
                preds_ged_plus_peakcount[mat_name] = np.nan

            # ── Ridge features ─────────────────────────────────────
            is_gpd = 1.0 if subdir == 'gpd' else 0.0
            n_ch_detected = int(np.sum(np.isfinite(acf_freqs)))

            features = [
                f_A if np.isfinite(f_A) else 1.0,
                f_B if np.isfinite(f_B) else 1.0,
                f_peaks if np.isfinite(f_peaks) else 1.0,
                f_fft if np.isfinite(f_fft) else 1.0,
                f_envelope if np.isfinite(f_envelope) else 1.0,
                f_ged_fft if np.isfinite(f_ged_fft) else 1.0,
                f_ged_acf if np.isfinite(f_ged_acf) else 1.0,
                is_gpd,
                float(n_ch_detected),
            ]
            ridge_features.append(features)
            ridge_targets.append(np.log(entry['expert_consensus_freq']) if entry['expert_consensus_freq'] > 0 else 0.0)
            ridge_mat_names.append(mat_name)
            ridge_subdirs.append(subdir)

        except Exception as e:
            if (idx + 1) % 50 == 0:
                print(f"    Error on {mat_name}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\nProcessing complete: {elapsed:.1f}s")
    print(f"GED FFT predictions: {len(preds_ged_fft)}")

    # ── Evaluate simple variants ───────────────────────────────────────
    print("\n--- Evaluating r6_ged_fft ---")
    evaluate_predictions(dataset, preds_ged_fft, 'r6_ged_fft')

    print("\n--- Evaluating r6_ged_acf ---")
    evaluate_predictions(dataset, preds_ged_acf, 'r6_ged_acf')

    print("\n--- Evaluating r6_ged_peaks ---")
    evaluate_predictions(dataset, preds_ged_peaks, 'r6_ged_peaks')

    print("\n--- Evaluating r6_ged_plus_peakcount ---")
    evaluate_predictions(dataset, preds_ged_plus_peakcount, 'r6_ged_plus_peakcount')

    print("\n--- Evaluating r6_ged_top3 ---")
    evaluate_predictions(dataset, preds_ged_top3, 'r6_ged_top3')

    # ── Ridge regression with LOO-CV ───────────────────────────────────
    print("\n--- Running Ridge LOO-CV for r6_ged_ridge ---")
    X = np.array(ridge_features)
    y = np.array(ridge_targets)
    n = len(y)
    print(f"  Ridge: {n} samples, {X.shape[1]} features")

    if n > 5:
        # Standardize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std[X_std < 1e-10] = 1.0
        X_norm = (X - X_mean) / X_std

        # Add bias column
        X_norm = np.column_stack([X_norm, np.ones(n)])

        alpha = 1.0
        preds_ridge = {}
        for i in range(n):
            # LOO: train on all except i
            X_train = np.delete(X_norm, i, axis=0)
            y_train = np.delete(y, i)

            # Ridge closed form: w = (X^T X + alpha I)^-1 X^T y
            n_feat = X_train.shape[1]
            reg = alpha * np.eye(n_feat)
            reg[-1, -1] = 0  # don't regularize bias
            try:
                w = np.linalg.solve(X_train.T @ X_train + reg, X_train.T @ y_train)
                pred_log = X_norm[i] @ w
                pred_freq = np.exp(pred_log)
                # Clamp to reasonable range
                pred_freq = max(0.3, min(4.0, pred_freq))
                preds_ridge[ridge_mat_names[i]] = pred_freq
            except np.linalg.LinAlgError:
                preds_ridge[ridge_mat_names[i]] = np.nan

        evaluate_predictions(dataset, preds_ridge, 'r6_ged_ridge')
    else:
        print("  Not enough samples for Ridge regression")

    print("\nAll evaluations complete.")


if __name__ == '__main__':
    main()
