"""
Peak-count frequency estimation as cross-validation for ACF.

Direct counting of peaks in the pointiness trace gives an independent
frequency estimate that can catch ACF subharmonic locking.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    compute_pointiness_trace, compute_acf_frequency,
    fcn_getBanana, bipolar_channels
)
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data


def process_segment(data, fs):
    """
    Process one segment. Returns per-channel peak-count freqs, ACF freqs, and ACF scores.
    """
    # Preprocessing: notch 60Hz, bandpass 0.5-40Hz
    segment = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")

    # Bipolar montage
    seg = np.array(fcn_getBanana(segment))

    # 15Hz lowpass
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    n_channels = seg.shape[0]
    peak_freqs = np.full(n_channels, np.nan)
    acf_freqs = np.full(n_channels, np.nan)
    acf_scores = np.full(n_channels, 0.0)

    sigma_samples = max(1, int(0.02 * fs))
    min_dist = int(0.2 * fs)

    for i in range(n_channels):
        ch = seg[i, :]

        # Pointiness trace + smooth
        trace = compute_pointiness_trace(ch)
        trace_smooth = gaussian_filter1d(trace, sigma=sigma_samples)

        # --- Peak-count frequency ---
        trace_max = np.max(trace_smooth)
        if trace_max > 0:
            pks, _ = find_peaks(trace_smooth, height=trace_max * 0.3, distance=min_dist)
            if len(pks) >= 3:
                time_span = (pks[-1] - pks[0]) / fs
                if time_span > 0:
                    peak_freqs[i] = (len(pks) - 1) / time_span

        # --- ACF frequency ---
        freq, score, _ = compute_acf_frequency(
            ch, fs, method='pointiness',
            smoothing_sigma=0.02, acf_min_lag=0.25,
            acf_peak_threshold=0.1, peak_height_frac=0.3
        )
        acf_freqs[i] = freq
        acf_scores[i] = score

    return peak_freqs, acf_freqs, acf_scores


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    # Storage: mat_name -> results
    all_peak_freqs = {}
    all_acf_freqs = {}
    all_acf_scores = {}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  Processing segment {idx + 1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        peak_freqs, acf_freqs, acf_scores = process_segment(data, fs)
        mat_name = entry['mat_name']
        all_peak_freqs[mat_name] = peak_freqs
        all_acf_freqs[mat_name] = acf_freqs
        all_acf_scores[mat_name] = acf_scores

    print(f"\nProcessed {len(all_peak_freqs)} segments successfully")

    # --- Strategy a) peakcount_only ---
    preds_peakcount = {}
    for mat_name, pf in all_peak_freqs.items():
        valid = pf[np.isfinite(pf)]
        if len(valid) >= 1:
            preds_peakcount[mat_name] = float(np.median(valid))
    evaluate_predictions(dataset, preds_peakcount, "peakcount_only")

    # --- Strategy b) acf_validated ---
    preds_acf_validated = {}
    for mat_name in all_peak_freqs:
        pf = all_peak_freqs[mat_name]
        af = all_acf_freqs[mat_name]

        valid_pf = pf[np.isfinite(pf)]
        valid_af = af[np.isfinite(af)]

        if len(valid_af) == 0 and len(valid_pf) == 0:
            continue

        acf_med = float(np.median(valid_af)) if len(valid_af) > 0 else np.nan
        pk_med = float(np.median(valid_pf)) if len(valid_pf) >= 3 else np.nan

        if np.isfinite(acf_med) and np.isfinite(pk_med):
            # If peak-count is >1.5x the ACF freq, ACF likely locked subharmonic
            if pk_med > 1.5 * acf_med:
                preds_acf_validated[mat_name] = pk_med
            else:
                preds_acf_validated[mat_name] = acf_med
        elif np.isfinite(acf_med):
            preds_acf_validated[mat_name] = acf_med
        elif np.isfinite(pk_med):
            preds_acf_validated[mat_name] = pk_med
    evaluate_predictions(dataset, preds_acf_validated, "acf_validated_by_peakcount")

    # --- Strategy c) mean_acf_peaks ---
    preds_mean = {}
    for mat_name in all_peak_freqs:
        pf = all_peak_freqs[mat_name]
        af = all_acf_freqs[mat_name]

        valid_pf = pf[np.isfinite(pf)]
        valid_af = af[np.isfinite(af)]

        vals = []
        if len(valid_af) > 0:
            vals.append(float(np.median(valid_af)))
        if len(valid_pf) >= 1:
            vals.append(float(np.median(valid_pf)))

        if vals:
            preds_mean[mat_name] = float(np.mean(vals))
    evaluate_predictions(dataset, preds_mean, "mean_acf_peakcount")

    # --- Strategy d) best_channel_peaks ---
    preds_best_ch = {}
    for mat_name in all_peak_freqs:
        pf = all_peak_freqs[mat_name]
        scores = all_acf_scores[mat_name]

        # Find channel with highest ACF score
        best_ch = np.argmax(scores)
        if np.isfinite(pf[best_ch]):
            preds_best_ch[mat_name] = float(pf[best_ch])
        else:
            # Fallback: use ACF freq from best channel
            af = all_acf_freqs[mat_name]
            if np.isfinite(af[best_ch]):
                preds_best_ch[mat_name] = float(af[best_ch])
    evaluate_predictions(dataset, preds_best_ch, "best_channel_peakcount")

    print("\nAll strategies evaluated.")


if __name__ == '__main__':
    main()
