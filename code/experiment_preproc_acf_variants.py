"""
Experiment: Preprocessing and ACF parameter variants for frequency estimation.

Tests different lowpass cutoffs, smoothing sigmas, acf_min_lag, and acf_peak_threshold
values to find optimal settings for different frequency ranges.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import pd_detect_pointiness_acf

# Default parameters (best known)
DEFAULTS = dict(
    lowpass_hz=15,
    smoothing_sigma=0.02,
    acf_min_lag=0.4,
    acf_peak_threshold=0.20,
    peak_height_frac=0.3,
    sync_threshold=0.8,
    method='pointiness',
)

# Variants to test
VARIANTS = {
    "preproc_minlag025": dict(acf_min_lag=0.25),
    "preproc_minlag030": dict(acf_min_lag=0.30),
    "preproc_lp10": dict(lowpass_hz=10),
    "preproc_lp20": dict(lowpass_hz=20),
    "preproc_sigma001": dict(smoothing_sigma=0.01),
    "preproc_sigma003": dict(smoothing_sigma=0.03),
    "preproc_acfthr010": dict(acf_peak_threshold=0.10),
    "preproc_acfthr030": dict(acf_peak_threshold=0.30),
    "preproc_combo_fast": dict(
        lowpass_hz=15,
        acf_min_lag=0.25,
        smoothing_sigma=0.01,
        acf_peak_threshold=0.15,
    ),
}


def run_variant(dataset, name, overrides):
    """Run a single variant and evaluate."""
    params = {**DEFAULTS, **overrides}
    print(f"\n{'#'*60}")
    print(f"Running variant: {name}")
    print(f"  Params: {overrides}")
    print(f"{'#'*60}")

    predictions = {}
    n_total = len(dataset)

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  [{name}] Progress: {idx+1}/{n_total}")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            result = pd_detect_pointiness_acf(
                data, fs,
                method=params['method'],
                acf_min_lag=params['acf_min_lag'],
                acf_peak_threshold=params['acf_peak_threshold'],
                smoothing_sigma=params['smoothing_sigma'],
                lowpass_hz=params['lowpass_hz'],
                peak_height_frac=params['peak_height_frac'],
                sync_threshold=params['sync_threshold'],
            )
            freq = result.get('event_frequency', np.nan)
            if freq is not None and np.isfinite(freq):
                predictions[entry['mat_name']] = freq
            else:
                predictions[entry['mat_name']] = np.nan
        except Exception as e:
            print(f"  Error on {entry['mat_name']}: {e}")
            predictions[entry['mat_name']] = np.nan

    metrics = evaluate_predictions(dataset, predictions, name)
    return metrics


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} annotated segments.")

    all_results = {}
    for name, overrides in VARIANTS.items():
        metrics = run_variant(dataset, name, overrides)
        all_results[name] = metrics

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY OF ALL VARIANTS")
    print(f"{'='*80}")
    print(f"{'Variant':<25s} {'LPD MAE':>8s} {'GPD MAE':>8s} {'Comb MAE':>9s} {'LPD r':>7s} {'GPD r':>7s}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*9} {'-'*7} {'-'*7}")
    for name, m in all_results.items():
        lpd_mae = m.get('lpd_mae', '?')
        gpd_mae = m.get('gpd_mae', '?')
        comb = m.get('combined_mae', '?')
        lpd_r = m.get('lpd_pearson_r', '?')
        gpd_r = m.get('gpd_pearson_r', '?')
        print(f"{name:<25s} {lpd_mae:>8} {gpd_mae:>8} {comb:>9} {lpd_r:>7} {gpd_r:>7}")

    print("\nDone.")


if __name__ == '__main__':
    main()
