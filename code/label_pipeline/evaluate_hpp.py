"""Evaluate HPP discharge detection against MW ground truth."""
import json, sys, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimization_harness_v2 import load_dataset
from hpp_discharge_marking import detect_discharge_times_hpp

TOLERANCE_S = 0.1  # ±100ms

def main():
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    with open('data/labels/discharge_times.json') as f:
        hpp = json.load(f)

    gt_cases = {pid: v for pid, v in hpp.items() if v.get('review_status') == 'ground_truth'}
    print(f"Evaluating on {len(gt_cases)} ground truth cases\n")

    gt_freqs, algo_freqs, gold_freqs = [], [], []
    total_tp, total_fn, total_fp = 0, 0, 0
    match_errors = []
    gt_counts, algo_counts = [], []

    for pid, gt_data in gt_cases.items():
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
        gold_freq = float(row['gold_standard_freq'])
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        try:
            result = detect_discharge_times_hpp(
                seg, fs=200, subtype=subtype,
                freq_estimate=gold_freq, laterality=lat, refine=True)
        except Exception as e:
            continue

        algo_times = sorted(result['global_times'])

        # Frequency from IPI
        gt_ipis = [gt_times[i+1] - gt_times[i] for i in range(len(gt_times)-1)]
        gt_freq = 1.0 / np.median(gt_ipis)

        if len(algo_times) >= 2:
            algo_ipis = [algo_times[i+1] - algo_times[i] for i in range(len(algo_times)-1)]
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan

        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)
            gold_freqs.append(gold_freq)

        gt_counts.append(len(gt_times))
        algo_counts.append(len(algo_times))

        # Discharge matching with tolerance
        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)

        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    dist = abs(gt - at)
                    if dist < best_dist:
                        best_dist = dist
                        best_ai = ai
            if best_dist <= TOLERANCE_S and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True
                match_errors.append(best_dist)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

    # Results
    print("=" * 70)
    print(f"HPP ALGORITHM vs MW GROUND TRUTH (±{TOLERANCE_S*1000:.0f}ms tolerance)")
    print("=" * 70)
    print(f"\nCases evaluated: {len(gt_freqs)}")

    rho_ag, p1 = spearmanr(algo_freqs, gt_freqs)
    rho_a_gold, p2 = spearmanr(algo_freqs, gold_freqs)
    rho_gt_gold, p3 = spearmanr(gt_freqs, gold_freqs)
    mae_freq = np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs)))

    print(f"\n--- Frequency Estimation ---")
    print(f"  Algo IPI-freq vs MW IPI-freq:   rho = {rho_ag:.4f} (p={p1:.2e}), MAE = {mae_freq:.3f} Hz")
    print(f"  Algo IPI-freq vs Gold standard: rho = {rho_a_gold:.4f} (p={p2:.2e})")
    print(f"  MW IPI-freq vs Gold standard:   rho = {rho_gt_gold:.4f} (p={p3:.2e})")

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2*prec*sens / (prec+sens) if (prec+sens) > 0 else 0

    print(f"\n--- Discharge Detection ---")
    print(f"  True positives:  {total_tp}")
    print(f"  False negatives: {total_fn} (missed)")
    print(f"  False positives: {total_fp} (extra)")
    print(f"  Sensitivity:     {sens:.3f}")
    print(f"  Precision:       {prec:.3f}")
    print(f"  F1 score:        {f1:.3f}")
    print(f"  Miss rate:       {1-sens:.3f}")

    if match_errors:
        print(f"\n--- Timing Accuracy (matched only) ---")
        print(f"  Mean abs error:   {np.mean(match_errors)*1000:.1f} ms")
        print(f"  Median abs error: {np.median(match_errors)*1000:.1f} ms")
        print(f"  90th pctile:      {np.percentile(match_errors, 90)*1000:.1f} ms")

    print(f"\n--- Discharge Counts ---")
    print(f"  MW mean/case:   {np.mean(gt_counts):.1f} (std {np.std(gt_counts):.1f})")
    print(f"  Algo mean/case: {np.mean(algo_counts):.1f} (std {np.std(algo_counts):.1f})")

if __name__ == '__main__':
    main()
