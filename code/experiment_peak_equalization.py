"""
Experiment: Peak equalization — replace pointiness amplitudes with binary
indicators before ACF.

Hypothesis: ACF locks onto subharmonics because large peaks dominate.
By equalizing all peaks to height=1, the ACF should find the true fundamental.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, bipolar_channels
)
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data


def detect_peak_equalized(segment, fs, eq_sigma=4, acf_peak_threshold=0.05,
                          acf_min_lag=0.25, bayesian_nudge=False):
    """
    Detect PD frequency using peak-equalized ACF.

    Steps per channel:
      1. Compute pointiness trace
      2. Smooth with sigma=0.02*fs
      3. Find peaks (height=max*0.3, distance=0.2*fs)
      4. Create equalized trace: zeros, place 1.0 at each peak, smooth with eq_sigma
      5. ACF on equalized trace, find first peak after min_lag
      6. Median frequency across channels where ACF found a peak
    """
    # Preprocess
    segment = notch_filter(segment, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(segment))

    # 15Hz lowpass
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    max_lag = int(4 * fs)
    min_lag_samples = int(acf_min_lag * fs)
    channel_freqs = []

    for i in range(seg.shape[0]):
        ch = seg[i]
        if len(ch) < 50:
            continue

        # Pointiness trace
        trace = compute_pointiness_trace(ch)

        # Smooth
        sigma_samples = max(1, int(0.02 * fs))
        trace = gaussian_filter1d(trace, sigma=sigma_samples)

        # Find peaks
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * 0.3
        peak_locs, _ = find_peaks(trace, height=peak_height,
                                  distance=int(0.2 * fs))

        if len(peak_locs) < 2:
            continue

        # Equalized trace: binary peaks smoothed
        eq_trace = np.zeros(len(trace))
        eq_trace[peak_locs] = 1.0
        eq_trace = gaussian_filter1d(eq_trace, sigma=eq_sigma)

        # ACF on equalized trace
        t = eq_trace - np.mean(eq_trace)
        ml = min(max_lag, len(t) - 1)
        if ml < 10:
            continue

        acf = np.correlate(t, t, mode='full')
        acf = acf[len(t) - 1:][:ml + 1]
        if acf[0] <= 0:
            continue
        acf = acf / acf[0]

        # Find first significant peak after min_lag
        for k in range(min_lag_samples + 1, len(acf) - 1):
            if (acf[k] > acf[k - 1] and acf[k] > acf[k + 1]
                    and acf[k] > acf_peak_threshold):
                freq = fs / k
                channel_freqs.append(freq)
                break

    if not channel_freqs:
        return np.nan

    freq = float(np.median(channel_freqs))

    if bayesian_nudge:
        prior = 1.5  # typical PD frequency
        freq = 0.7 * freq + 0.3 * prior

    return freq


def run_variant(dataset, name, eq_sigma, acf_threshold, bayesian_nudge=False):
    """Run one variant across the full dataset."""
    predictions = {}
    n = len(dataset)
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [{name}] Processing {idx+1}/{n} ...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        freq = detect_peak_equalized(
            data, fs,
            eq_sigma=eq_sigma,
            acf_peak_threshold=acf_threshold,
            acf_min_lag=0.25,
            bayesian_nudge=bayesian_nudge,
        )
        predictions[entry['mat_name']] = freq

    evaluate_predictions(dataset, predictions, name)
    return predictions


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments.\n")

    variants = [
        ("r2_peakeq_s2_t005", 2, 0.05, False),
        ("r2_peakeq_s4_t005", 4, 0.05, False),
        ("r2_peakeq_s4_t010", 4, 0.10, False),
        ("r2_peakeq_s8_t010", 8, 0.10, False),
    ]

    best_preds = None
    best_name = None

    for name, sigma, thresh, bayes in variants:
        print(f"\n--- Running variant: {name} (sigma={sigma}, thresh={thresh}) ---")
        preds = run_variant(dataset, name, sigma, thresh, bayes)
        if name == "r2_peakeq_s4_t005":
            best_preds = preds
            best_name = name

    # Bayesian nudge on best variant (s4_t005)
    print("\n--- Running variant: r2_peakeq_s4_t005_bayesian ---")
    predictions_bayes = {}
    n = len(dataset)
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"  [r2_peakeq_s4_t005_bayesian] Processing {idx+1}/{n} ...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        freq = detect_peak_equalized(
            data, fs, eq_sigma=4, acf_peak_threshold=0.05,
            acf_min_lag=0.25, bayesian_nudge=True,
        )
        predictions_bayes[entry['mat_name']] = freq

    evaluate_predictions(dataset, predictions_bayes, "r2_peakeq_s4_t005_bayesian")

    print("\n\nAll variants complete.")


if __name__ == '__main__':
    main()
