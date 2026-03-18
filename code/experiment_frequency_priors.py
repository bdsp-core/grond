"""
Experiment: Using frequency priors and constraints to improve PD frequency estimation.

Tests 6 post-processing strategies applied to pd_detect_pointiness_acf raw output.
"""

import sys
import os
import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    pd_detect_pointiness_acf, compute_pointiness_trace, compute_acf_frequency,
    fcn_getBanana, bipolar_channels,
)
from scipy.signal import butter, filtfilt
from mne.filter import notch_filter, filter_data
from scipy.ndimage import gaussian_filter1d

# Best params (defaults from evaluate_methods.py baseline)
BEST_PARAMS = dict(
    method='pointiness',
    lowpass_hz=20.0,
    smoothing_sigma=0.02,
    acf_min_lag=0.25,
    acf_peak_threshold=0.1,
    peak_height_frac=0.3,
    sync_tolerance_ms=200,
    sync_threshold=0.8,
    sync_min_peaks=5,
)

GPD_PRIOR = 1.0
LPD_PRIOR = 1.25


def get_pattern_prior(entry):
    """Return frequency prior based on pattern type."""
    return GPD_PRIOR if entry['subdir'] == 'gpd' else LPD_PRIOR


def run_raw_detection(dataset):
    """Run pd_detect_pointiness_acf on all segments, return raw results dict."""
    raw_predictions = {}
    raw_results_full = {}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  Raw detection: {idx+1}/{len(dataset)}")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            result = pd_detect_pointiness_acf(data, fs, **BEST_PARAMS)
            freq = result['event_frequency']
            raw_predictions[entry['mat_name']] = freq
            raw_results_full[entry['mat_name']] = result
        except Exception as e:
            pass

    return raw_predictions, raw_results_full


def compute_acf_for_best_channel(entry, raw_result):
    """
    Re-compute ACF for the best channel (highest score) to check for half-lag peak.
    Returns the full ACF array and fs, or (None, None).
    """
    data, fs = load_eeg_data(entry)
    if data is None:
        return None, None

    # Preprocess same as pd_detect_pointiness_acf
    segment = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(segment))

    b_lp, a_lp = butter(4, BEST_PARAMS['lowpass_hz'] / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    # Find best channel (highest ACF score)
    scores = raw_result.get('channel_pd_scores', {})
    best_ch = None
    best_score = -1
    for i, ch_name in enumerate(bipolar_channels):
        s = scores.get(ch_name, 0.0)
        if np.isfinite(s) and s > best_score:
            best_score = s
            best_ch = i

    if best_ch is None or best_score <= 0:
        return None, None

    # Compute pointiness trace and ACF for best channel
    signal_1d = seg[best_ch]
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(BEST_PARAMS['smoothing_sigma'] * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)

    t = trace - np.mean(trace)
    max_lag = min(int(4 * fs), len(t) - 1)
    if max_lag < 10:
        return None, None

    acf = np.correlate(t, t, mode='full')
    acf = acf[len(t) - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return None, None

    return acf, fs


def has_half_lag_peak(acf, fs, freq, threshold=0.08):
    """
    Check if ACF has a peak near lag corresponding to freq*2 (i.e., at half the lag).
    If freq is the subharmonic, the true frequency is freq*2, so the ACF
    should have a peak at lag = fs/(freq*2) = original_lag/2.
    """
    if acf is None or not np.isfinite(freq) or freq <= 0:
        return False

    target_lag = fs / (freq * 2)  # lag for double frequency
    target_lag_int = int(round(target_lag))

    if target_lag_int < 2 or target_lag_int >= len(acf) - 1:
        return False

    # Check in a window around target lag
    window = max(3, int(0.05 * fs))  # 50ms window
    lo = max(1, target_lag_int - window)
    hi = min(len(acf) - 1, target_lag_int + window)

    for k in range(lo, hi):
        if k >= len(acf) - 1:
            break
        if acf[k] > acf[k-1] and acf[k] > acf[k+1] and acf[k] > threshold:
            return True

    return False


# ---- Strategy functions ----

def strategy_prior_clamp(raw_predictions, dataset):
    """Clamp frequency to [0.3, 3.5] Hz range."""
    preds = {}
    for entry in dataset:
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]
        if not np.isfinite(freq):
            preds[mn] = freq
        else:
            preds[mn] = np.clip(freq, 0.3, 3.5)
    return preds


def strategy_prior_gpd_bias(raw_predictions, dataset):
    """For GPD segments, if estimated freq < 0.5 Hz, multiply by 2."""
    preds = {}
    for entry in dataset:
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]
        if not np.isfinite(freq):
            preds[mn] = freq
        elif entry['subdir'] == 'gpd' and freq < 0.5:
            preds[mn] = freq * 2
        else:
            preds[mn] = freq
    return preds


def strategy_prior_double_if_low(raw_predictions, dataset):
    """If estimated freq < 0.6 Hz, double it."""
    preds = {}
    for entry in dataset:
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]
        if not np.isfinite(freq):
            preds[mn] = freq
        elif freq < 0.6:
            preds[mn] = freq * 2
        else:
            preds[mn] = freq
    return preds


def strategy_prior_pattern_median(raw_predictions, dataset):
    """If freq is NaN or < 0.3 Hz, replace with pattern-type median."""
    preds = {}
    for entry in dataset:
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]
        prior = get_pattern_prior(entry)
        if not np.isfinite(freq) or freq < 0.3:
            preds[mn] = prior
        else:
            preds[mn] = freq
    return preds


def strategy_prior_smart_double(raw_predictions, dataset, raw_results_full):
    """If freq < 0.7 Hz AND ACF has a peak near half-lag, use freq*2."""
    preds = {}
    n_doubled = 0
    for idx, entry in enumerate(dataset):
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]

        if (idx + 1) % 100 == 0:
            print(f"  Smart double ACF check: {idx+1}/{len(dataset)}")

        if not np.isfinite(freq) or freq >= 0.7:
            preds[mn] = freq
            continue

        # Re-examine ACF for half-lag peak
        raw_result = raw_results_full.get(mn, {})
        acf, fs = compute_acf_for_best_channel(entry, raw_result)

        if has_half_lag_peak(acf, fs, freq):
            preds[mn] = freq * 2
            n_doubled += 1
        else:
            preds[mn] = freq

    print(f"  Smart double: doubled {n_doubled} predictions")
    return preds


def strategy_prior_bayesian_nudge(raw_predictions, dataset):
    """Weighted combination: 0.7 * algorithm_freq + 0.3 * pattern_prior."""
    preds = {}
    for entry in dataset:
        mn = entry['mat_name']
        if mn not in raw_predictions:
            continue
        freq = raw_predictions[mn]
        if not np.isfinite(freq):
            preds[mn] = freq
        else:
            prior = get_pattern_prior(entry)
            preds[mn] = 0.7 * freq + 0.3 * prior
    return preds


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    print("\nRunning raw pd_detect_pointiness_acf detection...")
    raw_predictions, raw_results_full = run_raw_detection(dataset)
    print(f"Got {len(raw_predictions)} raw predictions")

    # Evaluate raw baseline first
    print("\n--- Evaluating raw baseline ---")
    evaluate_predictions(dataset, raw_predictions, "prior_raw_baseline")

    # Strategy (a): prior_clamp
    print("\n--- Strategy (a): prior_clamp [0.3, 3.5] Hz ---")
    preds_a = strategy_prior_clamp(raw_predictions, dataset)
    evaluate_predictions(dataset, preds_a, "prior_clamp")

    # Strategy (b): prior_gpd_bias
    print("\n--- Strategy (b): prior_gpd_bias ---")
    preds_b = strategy_prior_gpd_bias(raw_predictions, dataset)
    evaluate_predictions(dataset, preds_b, "prior_gpd_bias")

    # Strategy (c): prior_double_if_low
    print("\n--- Strategy (c): prior_double_if_low ---")
    preds_c = strategy_prior_double_if_low(raw_predictions, dataset)
    evaluate_predictions(dataset, preds_c, "prior_double_if_low")

    # Strategy (d): prior_pattern_median
    print("\n--- Strategy (d): prior_pattern_median ---")
    preds_d = strategy_prior_pattern_median(raw_predictions, dataset)
    evaluate_predictions(dataset, preds_d, "prior_pattern_median")

    # Strategy (e): prior_smart_double
    print("\n--- Strategy (e): prior_smart_double ---")
    preds_e = strategy_prior_smart_double(raw_predictions, dataset, raw_results_full)
    evaluate_predictions(dataset, preds_e, "prior_smart_double")

    # Strategy (f): prior_bayesian_nudge
    print("\n--- Strategy (f): prior_bayesian_nudge ---")
    preds_f = strategy_prior_bayesian_nudge(raw_predictions, dataset)
    evaluate_predictions(dataset, preds_f, "prior_bayesian_nudge")

    print("\n\nAll strategies complete!")


if __name__ == '__main__':
    main()
