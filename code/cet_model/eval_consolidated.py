"""
Evaluate the consolidated best method from discharge_detector.py.

Usage:
    conda run -n foe_dl python code/cet_model/eval_consolidated.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from discharge_detector import DischargeDetector, FS
from optimization_harness_v2 import load_dataset

TOLERANCE_S = 0.1


def main():
    t0 = time.time()
    print("=" * 74)
    print("  Consolidated Best Method Evaluation")
    print("  discharge_detector.py (all improvements)")
    print("=" * 74)

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    print("\nLoading detector...")
    detector = DischargeDetector()
    print(f"  Device: {detector.device}")

    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs = [], []
    n_cases = 0
    n_failed = 0

    print(f"\nEvaluating {len(gt_cases)} cases...")
    t_eval = time.time()

    for idx, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]
        subtype = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        try:
            result = detector.detect(seg, subtype=subtype, laterality=lat)
            algo_times = sorted(result['global_times'])
        except Exception as e:
            n_failed += 1
            continue

        n_cases += 1

        # Match discharges
        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    dist = abs(gt - at)
                    if dist < best_dist:
                        best_dist, best_ai = dist, ai
            if best_dist <= TOLERANCE_S and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        # Frequency
        gt_ipis = [gt_times[i+1] - gt_times[i] for i in range(len(gt_times)-1)]
        gt_freq = 1.0 / np.median(gt_ipis)
        if len(algo_times) >= 2:
            algo_ipis = [algo_times[i+1] - algo_times[i]
                         for i in range(len(algo_times)-1)]
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_eval:.0f}s)")

    elapsed = time.time() - t_eval

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    if len(gt_freqs) >= 3:
        freq_rho, _ = spearmanr(algo_freqs, gt_freqs)
    else:
        freq_rho = float('nan')

    print(f"\n{'='*74}")
    print(f"  RESULTS: Consolidated Best Method")
    print(f"{'='*74}")
    print(f"  Cases:       {n_cases} ({n_failed} failed)")
    print(f"  Sensitivity: {sens:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  F1:          {f1:.4f}")
    print(f"  Freq Rho:    {freq_rho:.4f}")
    print(f"  TP={total_tp}  FN={total_fn}  FP={total_fp}")
    print(f"  Time:        {elapsed:.0f}s")

    print(f"\n  Improvements applied:")
    print(f"    1. CET 80% threshold")
    print(f"    2. Product-boost (boost=3, floor=0.3)")
    print(f"    3. CNN+ACF ensemble freq (0.8/0.2)")
    print(f"    4. Post-hoc min-evidence filter (ratio=0.3)")

    print(f"\n  Comparison:")
    print(f"    Original max(HPP,CET)+CNN_freq+opt:  F1=0.7207")
    print(f"    This consolidated method:            F1={f1:.4f}  ({f1-0.7207:+.4f})")
    print(f"    Gold-freq reference:                 F1=0.7570")

    # Save
    save_path = PROJECT_DIR / 'data' / 'cet_cache' / 'consolidated_results.json'
    save_data = {
        'method': 'consolidated_best',
        'n_cases': n_cases,
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'freq_spearman': round(freq_rho, 4),
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
    }
    with open(str(save_path), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {save_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == '__main__':
    main()
