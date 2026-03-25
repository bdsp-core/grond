"""
Experiment: Subharmonic correction in ACF frequency estimation.

Problem: 73% of frequency estimation failures are due to subharmonic locking —
the ACF picks a peak at 2x, 3x, or 4x the true period, returning freq/2, freq/3, freq/4.

Fix: After finding the first ACF peak at lag L, check whether peaks exist at L/2, L/3, L/4
with ACF value > threshold * ACF[L]. If so, use the shortest valid lag (highest frequency).
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, bipolar_channels,
    notch_filter, filter_data, butter, filtfilt,
    find_peaks, gaussian_filter1d,
)


def compute_acf_frequency_subharm(signal_1d, fs, method='pointiness',
                                   smoothing_sigma=0.02, acf_min_lag=0.4,
                                   acf_peak_threshold=0.20,
                                   peak_height_frac=0.3,
                                   subharm_threshold=0.5):
    """
    ACF frequency estimation with subharmonic correction.

    After finding the first ACF peak at lag L, check L/2, L/3, L/4 for peaks
    with ACF value > subharm_threshold * ACF[L]. Use shortest valid lag.
    """
    if len(signal_1d) < 50:
        return np.nan, 0.0

    # Compute feature trace
    if method == 'pointiness':
        trace = compute_pointiness_trace(signal_1d)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Smooth
    sigma_samples = max(1, int(smoothing_sigma * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)

    # ACF
    t = trace - np.mean(trace)
    max_lag = min(int(4 * fs), len(t) - 1)
    if max_lag < 10:
        return np.nan, 0.0

    acf = np.correlate(t, t, mode='full')
    acf = acf[len(t) - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return np.nan, 0.0

    # Find first significant local max after min_lag
    min_lag_samples = int(acf_min_lag * fs)
    first_peak_lag = None
    first_peak_val = None

    for k in range(min_lag_samples + 1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > acf_peak_threshold:
            first_peak_lag = k
            first_peak_val = float(acf[k])
            break

    if first_peak_lag is None:
        return np.nan, 0.0

    # Subharmonic correction: check L/2, L/3, L/4 for valid peaks
    best_lag = first_peak_lag
    best_val = first_peak_val
    threshold_val = subharm_threshold * first_peak_val

    for divisor in [4, 3, 2]:  # check from highest freq (shortest lag) first
        sub_lag = first_peak_lag / divisor
        sub_lag_int = int(round(sub_lag))

        # Must be above minimum lag
        if sub_lag_int <= min_lag_samples or sub_lag_int < 2:
            continue
        if sub_lag_int >= len(acf) - 1:
            continue

        # Look for a peak near sub_lag_int (within +/- 2 samples)
        search_start = max(1, sub_lag_int - 2)
        search_end = min(len(acf) - 1, sub_lag_int + 3)

        for j in range(search_start, search_end):
            if j < 1 or j >= len(acf) - 1:
                continue
            if acf[j] > acf[j - 1] and acf[j] > acf[j + 1] and acf[j] > threshold_val:
                # Found a valid subharmonic peak — use it (prefer shortest lag)
                best_lag = j
                best_val = float(acf[j])
                # Since we iterate from divisor=4 down, first match is shortest lag
                # but we want the shortest valid one, so break on first match per divisor
                break
        # If we found one at this divisor, that's the shortest lag — use it
        if best_lag != first_peak_lag:
            break

    freq = fs / best_lag
    return freq, best_val


def run_on_dataset(dataset, subharm_threshold, experiment_name):
    """Run subharmonic-corrected ACF on all segments."""
    predictions = {}
    n = len(dataset)

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  [{experiment_name}] Processing {idx+1}/{n}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            # Full preprocessing pipeline from pd_detect_pointiness_acf
            segment = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
            segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg = np.array(fcn_getBanana(segment))

            # Lowpass filter
            lowpass_hz = 15.0
            b_lp, a_lp = butter(4, lowpass_hz / (fs / 2), btype='low')
            for i in range(seg.shape[0]):
                try:
                    seg[i] = filtfilt(b_lp, a_lp, seg[i])
                except ValueError:
                    pass

            # Per-channel frequency estimation
            channel_freqs = []
            for i in range(seg.shape[0]):
                freq, score = compute_acf_frequency_subharm(
                    seg[i, :], fs,
                    method='pointiness',
                    smoothing_sigma=0.02,
                    acf_min_lag=0.4,
                    acf_peak_threshold=0.20,
                    peak_height_frac=0.3,
                    subharm_threshold=subharm_threshold,
                )
                if np.isfinite(freq):
                    channel_freqs.append(freq)

            if channel_freqs:
                predictions[entry['mat_name']] = float(np.median(channel_freqs))

        except Exception as e:
            pass  # skip on error

    return predictions


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments.\n")

    thresholds = [0.3, 0.5, 0.7]

    for thresh in thresholds:
        name = f"subharm_corr_t{int(thresh*10):02d}"
        print(f"\n{'='*60}")
        print(f"Running: {name} (subharm_threshold={thresh})")
        print(f"{'='*60}")

        predictions = run_on_dataset(dataset, thresh, name)
        print(f"  Got {len(predictions)} predictions out of {len(dataset)} segments.")

        evaluate_predictions(dataset, predictions, name)


if __name__ == '__main__':
    main()
