"""
Run three baselines through the optimization harness (with Spearman correlation):
  A) r2_baseline_A: pd_detect_alternate(pk_detect='apd') -> event_frequency
  B) r2_baseline_B: pd_detect_pointiness_acf with best params
  C) r2_round1_best: pointiness_acf(thr=0.10) + peak-count freq average + Bayesian nudge
"""

import sys
import os
import numpy as np
from pathlib import Path

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import (pd_detect_pointiness_acf, fcn_getBanana,
                                 bipolar_channels)
from scipy.signal import find_peaks, butter, filtfilt
from mne.filter import notch_filter, filter_data

import warnings
warnings.filterwarnings('ignore')


def run_baseline_A(dataset):
    """Method A: pd_detect_alternate with pk_detect='apd'."""
    predictions = {}
    n = len(dataset)
    for idx, entry in enumerate(dataset):
        if idx % 20 == 0:
            print(f"  [A] {idx}/{n}")
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        try:
            result = pd_detect_alternate(data, fs, pk_detect='apd')
            freq = result['event_frequency']
            if hasattr(freq, 'item'):
                freq = freq.item()
            predictions[entry['mat_name']] = float(freq) if (freq is not None and np.isfinite(freq)) else np.nan
        except Exception as e:
            print(f"  [A] Error on {entry['mat_name']}: {e}")
            predictions[entry['mat_name']] = np.nan
    return predictions


def run_baseline_B(dataset):
    """Method B: pd_detect_pointiness_acf with best params."""
    predictions = {}
    n = len(dataset)
    for idx, entry in enumerate(dataset):
        if idx % 20 == 0:
            print(f"  [B] {idx}/{n}")
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        try:
            result = pd_detect_pointiness_acf(
                data, fs,
                lowpass_hz=15.0,
                smoothing_sigma=0.02,
                acf_min_lag=0.4,
                acf_peak_threshold=0.20,
                peak_height_frac=0.3,
            )
            freq = result['event_frequency']
            if hasattr(freq, 'item'):
                freq = freq.item()
            predictions[entry['mat_name']] = float(freq) if (freq is not None and np.isfinite(freq)) else np.nan
        except Exception as e:
            print(f"  [B] Error on {entry['mat_name']}: {e}")
            predictions[entry['mat_name']] = np.nan
    return predictions


def run_round1_best(dataset):
    """Round 1 best: pointiness_acf(thr=0.10) + peak-count freq + Bayesian nudge."""
    predictions = {}
    n = len(dataset)
    for idx, entry in enumerate(dataset):
        if idx % 20 == 0:
            print(f"  [C] {idx}/{n}")
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        try:
            # Step 1: ACF frequency with thr=0.10
            result = pd_detect_pointiness_acf(
                data, fs,
                lowpass_hz=15.0,
                smoothing_sigma=0.02,
                acf_min_lag=0.4,
                acf_peak_threshold=0.10,
                peak_height_frac=0.3,
            )
            acf_freq = result['event_frequency']
            if hasattr(acf_freq, 'item'):
                acf_freq = acf_freq.item()

            # Step 2: Peak-count frequency
            # Preprocess the same way as the detector
            seg = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
            seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg = np.array(fcn_getBanana(seg))
            b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
            for ch_i in range(seg.shape[0]):
                try:
                    seg[ch_i] = filtfilt(b_lp, a_lp, seg[ch_i])
                except ValueError:
                    pass

            peak_count_freqs = []
            n_samples = seg.shape[1]
            time_span = n_samples / fs

            for ch_i in range(seg.shape[0]):
                ch_data = seg[ch_i]
                ch_max = np.max(np.abs(ch_data))
                if ch_max == 0:
                    continue
                height_thr = ch_max * 0.3
                pks, _ = find_peaks(ch_data, height=height_thr, distance=int(0.2 * fs))
                n_peaks = len(pks)
                if n_peaks >= 2:
                    ch_freq = (n_peaks - 1) / time_span
                    peak_count_freqs.append(ch_freq)

            if peak_count_freqs:
                peak_count_freq = float(np.median(peak_count_freqs))
            else:
                peak_count_freq = np.nan

            # Step 3: Average the two frequency estimates
            if np.isfinite(acf_freq) and np.isfinite(peak_count_freq):
                freq = (acf_freq + peak_count_freq) / 2.0
            elif np.isfinite(acf_freq):
                freq = float(acf_freq)
            elif np.isfinite(peak_count_freq):
                freq = peak_count_freq
            else:
                freq = np.nan

            # Step 4: Bayesian nudge: 0.7*freq + 0.3*prior
            if np.isfinite(freq):
                subdir = entry['subdir']
                if subdir == 'lpd':
                    prior = 1.25
                else:  # gpd
                    prior = 1.0
                freq = 0.7 * freq + 0.3 * prior

            predictions[entry['mat_name']] = float(freq) if np.isfinite(freq) else np.nan

        except Exception as e:
            print(f"  [C] Error on {entry['mat_name']}: {e}")
            predictions[entry['mat_name']] = np.nan
    return predictions


def main():
    print("Loading dataset ...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments.\n")

    print("=== Running r2_baseline_A (Method A: pd_detect_alternate) ===")
    preds_a = run_baseline_A(dataset)
    evaluate_predictions(dataset, preds_a, "r2_baseline_A")

    print("\n=== Running r2_baseline_B (pointiness_acf best params) ===")
    preds_b = run_baseline_B(dataset)
    evaluate_predictions(dataset, preds_b, "r2_baseline_B")

    print("\n=== Running r2_round1_best (ACF+peak-count+nudge) ===")
    preds_c = run_round1_best(dataset)
    evaluate_predictions(dataset, preds_c, "r2_round1_best")

    print("\nAll baselines complete.")


if __name__ == '__main__':
    main()
