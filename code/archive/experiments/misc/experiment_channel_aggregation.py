"""
Experiment: Compare different channel aggregation strategies for frequency estimation.

Instead of always taking the median frequency across all channels where ACF
detected periodicity, try strategies that weight by ACF score or filter noisy channels.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from collections import Counter

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_acf_frequency, bipolar_channels
)
from mne.filter import notch_filter, filter_data

# Best params from prior optimization
PARAMS = dict(
    lowpass_hz=15,
    smoothing_sigma=0.02,
    acf_min_lag=0.4,
    acf_peak_threshold=0.20,
    peak_height_frac=0.3,
)


def aggregate_median(freqs, scores):
    """Baseline: median of all detected channel frequencies."""
    valid = freqs[np.isfinite(freqs)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


def aggregate_best_channel(freqs, scores):
    """Frequency from the single channel with highest ACF score."""
    mask = np.isfinite(freqs)
    if not np.any(mask):
        return np.nan
    idx = np.argmax(scores * mask)  # zero out scores for non-detected channels
    return float(freqs[idx])


def aggregate_top3_mean(freqs, scores):
    """Mean frequency of top 3 channels by ACF score."""
    mask = np.isfinite(freqs)
    if not np.any(mask):
        return np.nan
    # Set non-detected scores to -inf so they rank last
    s = scores.copy()
    s[~mask] = -np.inf
    top_idx = np.argsort(s)[-3:]
    top_idx = top_idx[mask[top_idx]]
    if len(top_idx) == 0:
        return np.nan
    return float(np.mean(freqs[top_idx]))


def aggregate_weighted_mean(freqs, scores):
    """Weighted mean of frequencies, weighted by ACF score."""
    mask = np.isfinite(freqs)
    if not np.any(mask):
        return np.nan
    w = scores[mask]
    f = freqs[mask]
    if np.sum(w) == 0:
        return float(np.mean(f))
    return float(np.average(f, weights=w))


def aggregate_mode_bin(freqs, scores):
    """Bin frequencies into 0.25Hz bins, pick the mode bin's center."""
    valid = freqs[np.isfinite(freqs)]
    if len(valid) == 0:
        return np.nan
    bins = np.round(valid / 0.25) * 0.25
    counter = Counter(bins)
    mode_bin = counter.most_common(1)[0][0]
    return float(mode_bin)


def aggregate_trimmed_mean(freqs, scores):
    """Remove highest and lowest channel frequencies, mean of rest."""
    valid = freqs[np.isfinite(freqs)]
    if len(valid) == 0:
        return np.nan
    if len(valid) <= 2:
        return float(np.mean(valid))
    sorted_f = np.sort(valid)
    trimmed = sorted_f[1:-1]
    return float(np.mean(trimmed))


STRATEGIES = {
    'agg_median': aggregate_median,
    'agg_best_channel': aggregate_best_channel,
    'agg_top3_mean': aggregate_top3_mean,
    'agg_weighted_mean': aggregate_weighted_mean,
    'agg_mode_bin': aggregate_mode_bin,
    'agg_trimmed_mean': aggregate_trimmed_mean,
}


def process_segment(data, fs):
    """Run preprocessing and per-channel ACF. Return (freqs, scores) arrays."""
    # Notch + bandpass
    data = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose="ERROR")

    # Bipolar montage
    seg = np.array(fcn_getBanana(data))

    # Lowpass
    b_lp, a_lp = butter(4, PARAMS['lowpass_hz'] / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    n_ch = seg.shape[0]
    channel_freqs = np.full(n_ch, np.nan)
    channel_scores = np.full(n_ch, 0.0)

    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i, :], fs, method='pointiness',
            smoothing_sigma=PARAMS['smoothing_sigma'],
            acf_min_lag=PARAMS['acf_min_lag'],
            acf_peak_threshold=PARAMS['acf_peak_threshold'],
            peak_height_frac=PARAMS['peak_height_frac'],
        )
        channel_freqs[i] = freq
        channel_scores[i] = score

    return channel_freqs, channel_scores


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    # Pre-compute per-channel freqs and scores for all segments
    all_freqs = {}
    all_scores = {}

    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        channel_freqs, channel_scores = process_segment(data, fs)
        key = entry['mat_name']
        all_freqs[key] = channel_freqs
        all_scores[key] = channel_scores

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} segments")

    print(f"Processed {len(all_freqs)} segments total\n")

    # Evaluate each strategy
    for strat_name, strat_fn in STRATEGIES.items():
        predictions = {}
        for mat_name, freqs in all_freqs.items():
            scores = all_scores[mat_name]
            predictions[mat_name] = strat_fn(freqs, scores)

        evaluate_predictions(dataset, predictions, strat_name)
        print()


if __name__ == '__main__':
    main()
