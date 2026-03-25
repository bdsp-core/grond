"""
Ensemble/hybrid experiment: combine Method A (pd_detect_alternate) and
Method B (pd_detect_pointiness_acf) frequency estimates.

Method A excels at LPD frequency (MAE 0.537).
Method B excels at GPD correlation (r=0.42).
This script tries several strategies to combine their complementary strengths.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    # Run both methods on every segment, cache results
    freqs_A = {}   # mat_name -> frequency from Method A
    freqs_B = {}   # mat_name -> frequency from Method B
    subdirs = {}   # mat_name -> 'lpd' or 'gpd'

    for idx, entry in enumerate(dataset):
        mat_name = entry['mat_name']
        subdirs[mat_name] = entry['subdir']

        data, fs = load_eeg_data(entry)
        if data is None:
            freqs_A[mat_name] = np.nan
            freqs_B[mat_name] = np.nan
            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{len(dataset)}] (skipped - no data)")
            continue

        # Method A
        try:
            res_A = pd_detect_alternate(data, fs, pk_detect='apd')
            freqs_A[mat_name] = float(res_A['event_frequency'])
        except Exception as e:
            freqs_A[mat_name] = np.nan

        # Method B
        try:
            res_B = pd_detect_pointiness_acf(
                data, fs,
                method='pointiness',
                lowpass_hz=15,
                smoothing_sigma=0.02,
                acf_min_lag=0.4,
                acf_peak_threshold=0.20,
                peak_height_frac=0.3,
                sync_threshold=0.8,
            )
            freqs_B[mat_name] = float(res_B['event_frequency'])
        except Exception as e:
            freqs_B[mat_name] = np.nan

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(dataset)}] A={freqs_A[mat_name]:.3f}, B={freqs_B[mat_name]:.3f}"
                  if np.isfinite(freqs_A[mat_name]) and np.isfinite(freqs_B[mat_name])
                  else f"  [{idx+1}/{len(dataset)}] A={freqs_A[mat_name]}, B={freqs_B[mat_name]}")

    print(f"\nAll {len(dataset)} segments processed. Building ensembles...\n")

    # ---- Strategy (a): ensemble_mean ----
    preds_mean = {}
    for mn in freqs_A:
        a, b = freqs_A[mn], freqs_B[mn]
        a_ok, b_ok = np.isfinite(a), np.isfinite(b)
        if a_ok and b_ok:
            preds_mean[mn] = (a + b) / 2.0
        elif a_ok:
            preds_mean[mn] = a
        elif b_ok:
            preds_mean[mn] = b
        else:
            preds_mean[mn] = np.nan
    evaluate_predictions(dataset, preds_mean, "ensemble_mean")

    # ---- Strategy (b/c): ensemble_A_for_LPD (= use A for LPD, B for GPD) ----
    preds_type = {}
    for mn in freqs_A:
        if subdirs[mn] == 'lpd':
            preds_type[mn] = freqs_A[mn] if np.isfinite(freqs_A[mn]) else freqs_B[mn]
        else:
            preds_type[mn] = freqs_B[mn] if np.isfinite(freqs_B[mn]) else freqs_A[mn]
    evaluate_predictions(dataset, preds_type, "ensemble_A_for_LPD_B_for_GPD")

    # ---- Strategy (d): ensemble_closer_to_1hz ----
    preds_1hz = {}
    for mn in freqs_A:
        a, b = freqs_A[mn], freqs_B[mn]
        a_ok, b_ok = np.isfinite(a), np.isfinite(b)
        if a_ok and b_ok:
            preds_1hz[mn] = a if abs(a - 1.0) <= abs(b - 1.0) else b
        elif a_ok:
            preds_1hz[mn] = a
        elif b_ok:
            preds_1hz[mn] = b
        else:
            preds_1hz[mn] = np.nan
    evaluate_predictions(dataset, preds_1hz, "ensemble_closer_to_1hz")

    # ---- Strategy (e): ensemble_higher ----
    preds_higher = {}
    for mn in freqs_A:
        a, b = freqs_A[mn], freqs_B[mn]
        a_ok, b_ok = np.isfinite(a), np.isfinite(b)
        if a_ok and b_ok:
            preds_higher[mn] = max(a, b)
        elif a_ok:
            preds_higher[mn] = a
        elif b_ok:
            preds_higher[mn] = b
        else:
            preds_higher[mn] = np.nan
    evaluate_predictions(dataset, preds_higher, "ensemble_higher")

    # ---- Strategy (f): ensemble_weighted ----
    preds_weighted = {}
    for mn in freqs_A:
        a, b = freqs_A[mn], freqs_B[mn]
        a_ok, b_ok = np.isfinite(a), np.isfinite(b)
        if a_ok and b_ok:
            if subdirs[mn] == 'lpd':
                preds_weighted[mn] = 0.7 * a + 0.3 * b
            else:
                preds_weighted[mn] = 0.3 * a + 0.7 * b
        elif a_ok:
            preds_weighted[mn] = a
        elif b_ok:
            preds_weighted[mn] = b
        else:
            preds_weighted[mn] = np.nan
    evaluate_predictions(dataset, preds_weighted, "ensemble_weighted")

    print("\n\nAll ensemble strategies evaluated.")


if __name__ == '__main__':
    main()
