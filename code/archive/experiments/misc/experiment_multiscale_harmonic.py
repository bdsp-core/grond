"""
Experiment: Multi-scale pointiness + harmonic-aware ACF peak selection.

Tests:
  r2_multiscale_best     - pointiness at half_win=4,8,12,16; pick scale with highest ACF peak
  r2_harmonic_aware      - half_win=8, ACF all peaks, score = height/lag (favors fundamental)
  r2_harmonic_penalty    - ACF all peaks, check for sub-harmonic at L/2 and L/3
  r2_multiscale_harmonic - multi-scale + harmonic-aware scoring
  r2_best_combo          - best of above + bayesian nudge + average with peak-count frequency
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, bipolar_channels,
)
from mne.filter import notch_filter, filter_data

# ---------------------------------------------------------------------------
# Modified pointiness trace with configurable half_win
# ---------------------------------------------------------------------------

def compute_pointiness_trace(signal_1d, half_win=8):
    """Compute pointiness = prominence^2/width at each local max. Zero elsewhere."""
    n = len(signal_1d)
    trace = np.zeros(n)
    peaks, _ = find_peaks(signal_1d)
    for loc in peaks:
        if loc < half_win or loc >= n - half_win:
            continue
        peak_val = signal_1d[loc]
        left_valley = np.min(signal_1d[loc - half_win:loc])
        right_valley = np.min(signal_1d[loc + 1:loc + half_win + 1])
        prom = peak_val - max(left_valley, right_valley)
        if prom <= 0:
            continue
        half_prom_level = peak_val - 0.5 * prom
        width = 0
        for j in range(1, half_win + 1):
            if signal_1d[loc - j] > half_prom_level:
                width += 1
            else:
                break
        for j in range(1, half_win + 1):
            if loc + j < n and signal_1d[loc + j] > half_prom_level:
                width += 1
            else:
                break
        if width > 0:
            trace[loc] = prom ** 2 / width
    return trace


# ---------------------------------------------------------------------------
# ACF helpers
# ---------------------------------------------------------------------------

def compute_acf(trace, fs, max_lag_s=4.0):
    """Compute normalized ACF of a trace. Returns acf array (lag 0 = index 0)."""
    t = trace - np.mean(trace)
    max_lag = min(int(max_lag_s * fs), len(t) - 1)
    if max_lag < 10:
        return None
    acf = np.correlate(t, t, mode='full')
    acf = acf[len(t) - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return None
    return acf


def find_all_acf_peaks(acf, min_lag_samples, threshold=0.05):
    """Find all local maxima in ACF after min_lag with height > threshold.
    Returns list of (lag, height) tuples."""
    peaks = []
    for k in range(min_lag_samples + 1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > threshold:
            peaks.append((k, float(acf[k])))
    return peaks


def find_first_acf_peak(acf, min_lag_samples, threshold=0.05):
    """Find first significant ACF peak after min_lag. Returns (lag, height) or None."""
    for k in range(min_lag_samples + 1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > threshold:
            return (k, float(acf[k]))
    return None


def preprocess_segment(segment, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    segment = notch_filter(segment, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(segment))
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


def get_pointiness_and_acf(channel_signal, fs, half_win=8, smoothing_sigma=0.02):
    """Compute pointiness trace, smooth, and return ACF."""
    trace = compute_pointiness_trace(channel_signal, half_win=half_win)
    sigma_samples = max(1, int(smoothing_sigma * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    acf = compute_acf(trace, fs)
    return trace, acf


# ---------------------------------------------------------------------------
# Approach (a): multiscale_best
# ---------------------------------------------------------------------------

def freq_multiscale_best(channel_signal, fs, acf_min_lag=0.25, acf_peak_threshold=0.05):
    """Run pointiness at half_win=4,8,12,16. Pick scale with highest ACF peak."""
    min_lag_samples = int(acf_min_lag * fs)
    best_freq = np.nan
    best_height = 0.0

    for hw in [4, 8, 12, 16]:
        _, acf = get_pointiness_and_acf(channel_signal, fs, half_win=hw)
        if acf is None:
            continue
        peak = find_first_acf_peak(acf, min_lag_samples, threshold=acf_peak_threshold)
        if peak is not None:
            lag, height = peak
            if height > best_height:
                best_height = height
                best_freq = fs / lag

    return best_freq, best_height


# ---------------------------------------------------------------------------
# Approach (b): harmonic_aware
# ---------------------------------------------------------------------------

def freq_harmonic_aware(channel_signal, fs, acf_min_lag=0.25, acf_peak_threshold=0.05, half_win=8):
    """Find ALL ACF peaks, score by height/lag, pick highest-scored."""
    min_lag_samples = int(acf_min_lag * fs)
    _, acf = get_pointiness_and_acf(channel_signal, fs, half_win=half_win)
    if acf is None:
        return np.nan, 0.0

    peaks = find_all_acf_peaks(acf, min_lag_samples, threshold=acf_peak_threshold)
    if not peaks:
        return np.nan, 0.0

    # Score = acf_height / lag (favors shorter lags = higher frequencies)
    best_score = -1
    best_lag = None
    best_height = 0.0
    for lag, height in peaks:
        score = height / lag
        if score > best_score:
            best_score = score
            best_lag = lag
            best_height = height

    return fs / best_lag, best_height


# ---------------------------------------------------------------------------
# Approach (c): harmonic_penalty
# ---------------------------------------------------------------------------

def freq_harmonic_penalty(channel_signal, fs, acf_min_lag=0.25, acf_peak_threshold=0.05, half_win=8):
    """Find ALL ACF peaks. For each peak at lag L, check if peak near L/2 or L/3 exists.
    If sub-harmonic peak has height > 0.3*height_at_L, prefer the shorter lag (fundamental)."""
    min_lag_samples = int(acf_min_lag * fs)
    _, acf = get_pointiness_and_acf(channel_signal, fs, half_win=half_win)
    if acf is None:
        return np.nan, 0.0

    peaks = find_all_acf_peaks(acf, min_lag_samples, threshold=acf_peak_threshold)
    if not peaks:
        return np.nan, 0.0

    # Build a lookup: lag -> height
    peak_dict = {lag: height for lag, height in peaks}
    peak_lags = sorted(peak_dict.keys())

    # Start with the first (shortest lag) peak as candidate
    # Then check if any longer-lag peak is actually a subharmonic
    # For each peak, check if L/2 or L/3 has a peak nearby
    # If so, prefer L/2 or L/3

    def find_peak_near(target_lag, tolerance=3):
        """Find if there's a peak within tolerance of target_lag."""
        for lag in peak_lags:
            if abs(lag - target_lag) <= tolerance:
                return lag, peak_dict[lag]
        return None, 0.0

    # Start from the longest lag and try to replace with fundamental
    # Actually: iterate all peaks, for each check if L/2 or L/3 has a peak
    # Collect "fundamental" candidates
    fundamental_candidates = set(peak_lags)  # start with all

    for lag in peak_lags:
        height = peak_dict[lag]
        # Check L/2
        half_lag = lag / 2
        if half_lag >= min_lag_samples:
            found_lag, found_height = find_peak_near(int(round(half_lag)))
            if found_lag is not None and found_height > 0.3 * height:
                # L/2 peak exists and is significant -> L is likely a subharmonic
                fundamental_candidates.discard(lag)
        # Check L/3
        third_lag = lag / 3
        if third_lag >= min_lag_samples:
            found_lag, found_height = find_peak_near(int(round(third_lag)))
            if found_lag is not None and found_height > 0.3 * height:
                fundamental_candidates.discard(lag)

    if not fundamental_candidates:
        # Fallback: use first peak
        best_lag = peak_lags[0]
    else:
        # Pick the shortest remaining fundamental candidate
        best_lag = min(fundamental_candidates)

    return fs / best_lag, peak_dict[best_lag]


# ---------------------------------------------------------------------------
# Approach (d): multiscale + harmonic-aware
# ---------------------------------------------------------------------------

def freq_multiscale_harmonic(channel_signal, fs, acf_min_lag=0.25, acf_peak_threshold=0.05):
    """Combine multiscale pointiness with harmonic-aware scoring."""
    min_lag_samples = int(acf_min_lag * fs)
    best_score = -1
    best_freq = np.nan
    best_height = 0.0

    for hw in [4, 8, 12, 16]:
        _, acf = get_pointiness_and_acf(channel_signal, fs, half_win=hw)
        if acf is None:
            continue
        peaks = find_all_acf_peaks(acf, min_lag_samples, threshold=acf_peak_threshold)
        for lag, height in peaks:
            score = height / lag
            if score > best_score:
                best_score = score
                best_freq = fs / lag
                best_height = height

    return best_freq, best_height


# ---------------------------------------------------------------------------
# Approach (e): best combo
# ---------------------------------------------------------------------------

def freq_best_combo(channel_signal, fs, acf_min_lag=0.25, acf_peak_threshold=0.05):
    """Best of above + bayesian nudge + average with peak-count frequency."""
    # Get frequencies from all approaches
    f_ms, h_ms = freq_multiscale_best(channel_signal, fs, acf_min_lag, acf_peak_threshold)
    f_ha, h_ha = freq_harmonic_aware(channel_signal, fs, acf_min_lag, acf_peak_threshold)
    f_hp, h_hp = freq_harmonic_penalty(channel_signal, fs, acf_min_lag, acf_peak_threshold)
    f_msh, h_msh = freq_multiscale_harmonic(channel_signal, fs, acf_min_lag, acf_peak_threshold)

    # Pick best by ACF height among all approaches
    candidates = [(f_ms, h_ms), (f_ha, h_ha), (f_hp, h_hp), (f_msh, h_msh)]
    candidates = [(f, h) for f, h in candidates if np.isfinite(f)]
    if not candidates:
        return np.nan, 0.0

    best_f, best_h = max(candidates, key=lambda x: x[1])

    # Peak-count frequency: count peaks in pointiness trace
    trace = compute_pointiness_trace(channel_signal, half_win=8)
    sigma_samples = max(1, int(0.02 * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    trace_max = np.max(trace)
    if trace_max > 0:
        peak_height = trace_max * 0.3
        peak_indices, _ = find_peaks(trace, height=peak_height, distance=int(0.2 * fs))
        n_peaks = len(peak_indices)
        duration_s = len(channel_signal) / fs
        if n_peaks >= 2:
            peak_count_freq = n_peaks / duration_s
        else:
            peak_count_freq = np.nan
    else:
        peak_count_freq = np.nan

    # Average with peak-count frequency if available
    if np.isfinite(peak_count_freq):
        best_f = (best_f + peak_count_freq) / 2.0

    # Bayesian nudge toward 1.5 Hz prior (most PDs are 1-2 Hz)
    prior_mean = 1.5
    prior_weight = 0.1
    best_f = best_f * (1 - prior_weight) + prior_mean * prior_weight

    return best_f, best_h


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

APPROACHES = {
    'r2_multiscale_best': freq_multiscale_best,
    'r2_harmonic_aware': freq_harmonic_aware,
    'r2_harmonic_penalty': freq_harmonic_penalty,
    'r2_multiscale_harmonic': freq_multiscale_harmonic,
    'r2_best_combo': freq_best_combo,
}


def run_approach(dataset, approach_name, approach_fn):
    """Run a single approach across all segments."""
    print(f"\n{'#'*60}")
    print(f"Running: {approach_name}")
    print(f"{'#'*60}")

    predictions = {}
    n_total = len(dataset)

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  [{approach_name}] Progress: {idx+1}/{n_total}")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            seg = preprocess_segment(data, fs)
            channel_freqs = []

            for ch_i in range(seg.shape[0]):
                freq, height = approach_fn(
                    seg[ch_i], fs,
                    acf_min_lag=0.25,
                    acf_peak_threshold=0.05,
                )
                if np.isfinite(freq):
                    channel_freqs.append(freq)

            if channel_freqs:
                pred_freq = float(np.median(channel_freqs))
            else:
                pred_freq = np.nan

            predictions[entry['mat_name']] = pred_freq

        except Exception as e:
            predictions[entry['mat_name']] = np.nan

    evaluate_predictions(dataset, predictions, approach_name)
    return predictions


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} annotated segments")

    for approach_name, approach_fn in APPROACHES.items():
        run_approach(dataset, approach_name, approach_fn)

    print("\n\nAll approaches complete!")


if __name__ == '__main__':
    main()
