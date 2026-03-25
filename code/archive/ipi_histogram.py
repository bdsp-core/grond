"""
Inter-peak interval (IPI) histogram for frequency estimation.
Bypasses ACF entirely by directly measuring intervals between peaks.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    bipolar_channels
)
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data


def ipi_frequency_for_segment(data, fs):
    """
    Compute IPI-based frequency estimates for a single segment.

    Returns dict with keys:
        'per_channel_median_freqs': list of median-interval freqs per channel
        'per_channel_mode_freqs': list of mode-interval freqs per channel
        'per_channel_npeaks': list of peak counts per channel
        'acf_freq': ACF-based frequency (median of valid channels)
    """
    # Preprocess
    data = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(data))

    # 15 Hz lowpass
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    median_freqs = []
    mode_freqs = []
    npeaks_list = []
    acf_freqs = []

    sigma_samples = max(1, int(0.02 * fs))
    min_distance = int(0.2 * fs)

    for i in range(seg.shape[0]):
        ch = seg[i]

        # Pointiness trace
        trace = compute_pointiness_trace(ch)
        trace = gaussian_filter1d(trace, sigma=sigma_samples)

        # Find peaks
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * 0.3
        peaks, _ = find_peaks(trace, height=peak_height, distance=min_distance)

        if len(peaks) >= 3:
            # Inter-peak intervals in seconds
            intervals = np.diff(peaks) / fs

            # Median-interval frequency
            med_interval = np.median(intervals)
            if med_interval > 0:
                median_freqs.append(1.0 / med_interval)
                npeaks_list.append(len(peaks))

            # Mode-interval frequency (bin into 0.1s bins)
            bin_edges = np.arange(0, np.max(intervals) + 0.15, 0.1)
            if len(bin_edges) >= 2:
                counts, edges = np.histogram(intervals, bins=bin_edges)
                if np.max(counts) > 0:
                    best_bin = np.argmax(counts)
                    mode_interval = (edges[best_bin] + edges[best_bin + 1]) / 2.0
                    if mode_interval > 0:
                        mode_freqs.append(1.0 / mode_interval)

        # ACF frequency for this channel (for combined method)
        acf_freq, acf_score, _ = compute_acf_frequency(
            ch, fs, method='pointiness',
            smoothing_sigma=0.02, acf_min_lag=0.25,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        if np.isfinite(acf_freq):
            acf_freqs.append(acf_freq)

    return {
        'per_channel_median_freqs': median_freqs,
        'per_channel_mode_freqs': mode_freqs,
        'per_channel_npeaks': npeaks_list,
        'acf_freq': float(np.median(acf_freqs)) if acf_freqs else np.nan,
    }


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    # Storage for all strategies
    preds_median = {}
    preds_mode = {}
    preds_weighted = {}
    preds_ipi_plus_acf = {}
    preds_ipi_plus_acf_bayesian = {}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"Processing segment {idx+1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']

        try:
            result = ipi_frequency_for_segment(data, fs)
        except Exception as e:
            print(f"  Error on {mat_name}: {e}")
            continue

        median_freqs = result['per_channel_median_freqs']
        mode_freqs = result['per_channel_mode_freqs']
        npeaks = result['per_channel_npeaks']
        acf_freq = result['acf_freq']

        # (a) r2_ipi_median: median of per-channel median-interval frequencies
        if median_freqs:
            preds_median[mat_name] = float(np.median(median_freqs))
        else:
            preds_median[mat_name] = np.nan

        # (b) r2_ipi_mode: median of per-channel mode-interval frequencies
        if mode_freqs:
            preds_mode[mat_name] = float(np.median(mode_freqs))
        else:
            preds_mode[mat_name] = np.nan

        # (c) r2_ipi_weighted: weighted mean by number of peaks
        if median_freqs and npeaks:
            weights = np.array(npeaks, dtype=float)
            freqs_arr = np.array(median_freqs)
            preds_weighted[mat_name] = float(np.average(freqs_arr, weights=weights))
        else:
            preds_weighted[mat_name] = np.nan

        # (d) r2_ipi_plus_acf: average of IPI freq and ACF freq
        ipi_freq = preds_median.get(mat_name, np.nan)
        if np.isfinite(ipi_freq) and np.isfinite(acf_freq):
            preds_ipi_plus_acf[mat_name] = (ipi_freq + acf_freq) / 2.0
        elif np.isfinite(ipi_freq):
            preds_ipi_plus_acf[mat_name] = ipi_freq
        elif np.isfinite(acf_freq):
            preds_ipi_plus_acf[mat_name] = acf_freq
        else:
            preds_ipi_plus_acf[mat_name] = np.nan

        # (e) r2_ipi_plus_acf_bayesian: (d) + bayesian nudge
        combined = preds_ipi_plus_acf[mat_name]
        if np.isfinite(combined):
            # Bayesian nudge toward population prior (1.5 Hz mean, moderate strength)
            prior_mean = 1.5
            prior_strength = 0.3  # weight of prior vs data
            preds_ipi_plus_acf_bayesian[mat_name] = (
                (1 - prior_strength) * combined + prior_strength * prior_mean
            )
        else:
            preds_ipi_plus_acf_bayesian[mat_name] = np.nan

    print("\n\nEvaluating all strategies...\n")

    evaluate_predictions(dataset, preds_median, "r2_ipi_median")
    evaluate_predictions(dataset, preds_mode, "r2_ipi_mode")
    evaluate_predictions(dataset, preds_weighted, "r2_ipi_weighted")
    evaluate_predictions(dataset, preds_ipi_plus_acf, "r2_ipi_plus_acf")
    evaluate_predictions(dataset, preds_ipi_plus_acf_bayesian, "r2_ipi_plus_acf_bayesian")

    print("\nDone!")


if __name__ == '__main__':
    main()
