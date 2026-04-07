#!/usr/bin/env python3
"""Run ensemble methods using cached per-patient scores from single methods.

Must be run AFTER all single methods have completed.
"""
import sys
import json
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from lateralization_contest.harness_v2 import (
    load_contest_data, evaluate, save_result, update_html_leaderboard, CACHE_DIR
)


def load_single_scores():
    """Load all cached per-patient scores."""
    scores = {}
    for path in sorted(CACHE_DIR.glob('*_scores.json')):
        method_name = path.stem.replace('_scores', '')
        with open(path) as f:
            scores[method_name] = json.load(f)
    return scores


def ensemble_mean_asymmetry(single_scores, method_names, patient_ids):
    """Compute mean asymmetry across methods for each patient."""
    results = {}
    for pid in patient_ids:
        asyms = []
        lat_idxs = []
        for mn in method_names:
            if mn in single_scores and pid in single_scores[mn]:
                s = single_scores[mn][pid]
                asyms.append(s['asymmetry'])
                lat_idxs.append(s['laterality_index'])
        if asyms:
            mean_asym = float(np.mean(asyms))
            mean_lat = float(np.mean(lat_idxs))
        else:
            mean_asym = 0.0
            mean_lat = 0.0
        ls = max(0, 0.5 - mean_lat / 2)
        rs = max(0, 0.5 + mean_lat / 2)
        results[pid] = {
            'left_score': ls,
            'right_score': rs,
            'laterality_index': mean_lat,
            'asymmetry': mean_asym,
            'extras': {},
        }
    return results


def ensemble_max_asymmetry(single_scores, method_names, patient_ids):
    """Max asymmetry across methods, keeping the laterality direction from the max method."""
    results = {}
    for pid in patient_ids:
        best_asym = 0.0
        best_lat = 0.0
        for mn in method_names:
            if mn in single_scores and pid in single_scores[mn]:
                s = single_scores[mn][pid]
                if s['asymmetry'] > best_asym:
                    best_asym = s['asymmetry']
                    best_lat = s['laterality_index']
        ls = max(0, 0.5 - best_lat / 2)
        rs = max(0, 0.5 + best_lat / 2)
        results[pid] = {
            'left_score': ls,
            'right_score': rs,
            'laterality_index': best_lat,
            'asymmetry': best_asym,
            'extras': {},
        }
    return results


TIER1 = ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L03_PeakToMeanRatio',
         'L04_SpectralConcentration', 'L05_TemplateMatch']
VE_METHODS = ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L09_VarExplained', 'L19_MatchedFilter']

ENSEMBLES = [
    ('L21_EnsembleTop3', 'mean', ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L03_PeakToMeanRatio']),
    ('L22_EnsembleTop5', 'mean', TIER1),
    ('L23_EnsembleVE', 'mean', VE_METHODS),
    ('L24_EnsembleAll', 'mean', None),  # None = all available
    ('L25_MaxAsymmetry', 'max', None),
]


def main():
    print("Loading data...")
    data = load_contest_data(verbose=True)
    patient_ids = list(data['segs'].keys())

    print("Loading cached single-method scores...")
    single_scores = load_single_scores()
    available = list(single_scores.keys())
    print(f"  {len(available)} methods available: {', '.join(sorted(available))}")

    for name, agg, methods in ENSEMBLES:
        if methods is None:
            methods = available
        present = [m for m in methods if m in single_scores]
        print(f"\n{'=' * 60}")
        print(f"Running: {name} ({agg} of {len(present)} methods)")
        print(f"{'=' * 60}")

        if not present:
            print(f"  SKIP: no component methods available")
            continue

        if agg == 'mean':
            results = ensemble_mean_asymmetry(single_scores, present, patient_ids)
        else:
            results = ensemble_max_asymmetry(single_scores, present, patient_ids)

        metrics = evaluate(results, data)
        save_result(name, metrics)
        print(f"  ** AUC: {metrics['primary_auc']} **  Cohen's d: {metrics['cohens_d']}")
        print(f"  Side accuracy: {metrics['side_accuracy']} ({metrics['n_side_validation']} cases)")
        update_html_leaderboard()

    print("\nDone!")


if __name__ == '__main__':
    main()
