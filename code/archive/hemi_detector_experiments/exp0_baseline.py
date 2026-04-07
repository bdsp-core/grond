"""
Experiment 0: Per-hemisphere baseline using current pipeline.

Runs the existing discharge detector on each hemisphere independently
to establish the F1 to beat.

Baseline A: Run on the affected hemisphere only (using laterality labels)
Baseline B: Run on both hemispheres, keep the one with more discharges

Usage:
    conda run -n foe_dl python code/hemi_detector/exp0_baseline.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from discharge_detector import (
    DischargeDetector, FS, LEFT_INDICES, RIGHT_INDICES,
    compute_channel_evidence, aggregate_evidence, combine_evidence,
    detect_active_interval, extract_candidates, dp_best_sequence,
    em_refine, posthoc_filter, estimate_frequency_acf,
)
from optimization_harness_v2 import load_dataset

TOLERANCE_S = 0.1


def run_hemisphere(detector, seg, hemi_indices, fs=FS):
    """Run the full pipeline on one hemisphere (8 channels)."""
    n_samples = seg.shape[1]
    n_ch = len(hemi_indices)

    # 1. Frequency estimation (CNN on hemisphere channels only)
    # Use PD-weighted freq from just these channels
    import torch
    all_pd_probs = []
    all_log_freqs = []
    for ch_idx in hemi_indices:
        ch_data = seg[ch_idx].astype(np.float32).copy()
        if not np.all(np.isfinite(ch_data)):
            all_pd_probs.append(0.0)
            all_log_freqs.append(0.0)
            continue
        mu, std = np.mean(ch_data), np.std(ch_data)
        ch_data = (ch_data - mu) / std if std > 1e-8 else ch_data - mu
        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(detector.device)
        pd_probs, log_freqs = [], []
        for model in detector.cnn_models:
            pd_prob, freq_pred, _ = model(x)
            pd_probs.append(pd_prob.item())
            log_freqs.append(freq_pred.item())
        all_pd_probs.append(np.mean(pd_probs))
        all_log_freqs.append(np.mean(log_freqs))

    pd_w = np.array(all_pd_probs)
    lf = np.array(all_log_freqs)
    ws = pd_w.sum()
    if ws > 1e-6:
        wlf = np.sum(pd_w * lf) / ws
    else:
        wlf = np.mean(lf)
    cnn_freq = float(np.clip(np.exp(wlf), 0.3, 3.5))

    # ACF freq on hemisphere channels
    from scipy.signal import butter, filtfilt
    b_lp, a_lp = butter(4, 20.0 / (fs / 2), btype='low')
    acf_freqs = []
    for ch_idx in hemi_indices:
        try:
            sig = filtfilt(b_lp, a_lp, seg[ch_idx])
        except:
            sig = seg[ch_idx]
        f = estimate_frequency_acf(sig, fs)
        if np.isfinite(f):
            acf_freqs.append(f)
    acf_freq = float(np.clip(np.median(acf_freqs), 0.3, 3.5)) if acf_freqs else cnn_freq

    freq_estimate = 0.8 * cnn_freq + 0.2 * acf_freq
    freq_estimate = float(np.clip(freq_estimate, 0.3, 3.5))

    # 2. Evidence (HPP + CET on hemisphere channels)
    hpp_all = np.zeros((n_ch, n_samples))
    cet_all = np.zeros((n_ch, n_samples), dtype=np.float32)
    for i, ch_idx in enumerate(hemi_indices):
        hpp_all[i] = compute_channel_evidence(seg[ch_idx], fs)
        if np.all(np.isfinite(seg[ch_idx])):
            cet_all[i] = detector.compute_cet_evidence_channel(seg[ch_idx])

    # Aggregate: median across hemisphere channels
    hpp_agg = np.median(hpp_all, axis=0)
    cet_agg = np.median(cet_all, axis=0)

    # Combine
    evidence = combine_evidence(hpp_agg, cet_agg)

    # 3. DP
    active_start, active_end = detect_active_interval(evidence, fs)
    candidates = extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    discharge_samples = dp_best_sequence(candidates, evidence, fs, freq_estimate)

    # 4. EM refine
    if len(discharge_samples) >= 3:
        discharge_samples = em_refine(evidence, discharge_samples, fs, freq_estimate)

    # 5. Post-hoc filter
    discharge_samples = posthoc_filter(discharge_samples, evidence)

    global_times = (discharge_samples / fs).tolist() if len(discharge_samples) > 0 else []
    return global_times, freq_estimate


def evaluate(predictions, gt_cases, tolerance=TOLERANCE_S):
    """Evaluate predictions against ground truth."""
    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs = [], []

    for pid, algo_times in predictions.items():
        if pid not in gt_cases:
            continue
        gt_times = sorted(gt_cases[pid]['global_times'])
        if len(gt_times) < 2:
            continue

        algo_times = sorted(algo_times)

        # Match
        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_dist:
                        best_dist, best_ai = d, ai
            if best_dist <= tolerance and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        # Freq
        gt_ipis = np.diff(gt_times)
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        if len(algo_times) >= 2:
            algo_ipis = np.diff(algo_times)
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = spearmanr(algo_freqs, gt_freqs)[0] if len(gt_freqs) >= 3 else float('nan')

    return {
        'f1': round(f1, 4), 'sensitivity': round(sens, 4), 'precision': round(prec, 4),
        'freq_spearman': round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
        'n_cases': len(predictions),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Experiment 0: Per-Hemisphere Baseline")
    print("=" * 70)

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    print("\nLoading detector models...")
    detector = DischargeDetector()

    # ================================================================
    # Baseline A: Affected hemisphere only
    # ================================================================
    print(f"\n{'='*70}")
    print("  Baseline A: Affected hemisphere (laterality-aware)")
    print(f"{'='*70}")

    predictions_a = {}
    t1 = time.time()

    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            # Check if it's a harvested case (not in patients.csv)
            pat_segs = segments.get(pid, [])
            if not pat_segs:
                continue
            seg = pat_segs[0]
            # Default to max(left, right) for unknown laterality
            try:
                times_l, _ = run_hemisphere(detector, seg, LEFT_INDICES)
                times_r, _ = run_hemisphere(detector, seg, RIGHT_INDICES)
                predictions_a[pid] = times_l if len(times_l) >= len(times_r) else times_r
            except:
                continue
        else:
            row = row.iloc[0]
            pat_segs = segments.get(pid, [])
            if not pat_segs:
                continue
            seg = pat_segs[0]
            subtype = row['subtype']
            lat = row.get('laterality', '')

            try:
                if subtype == 'gpd':
                    # GPD: run both, merge (since discharges are bilateral)
                    times_l, freq_l = run_hemisphere(detector, seg, LEFT_INDICES)
                    times_r, freq_r = run_hemisphere(detector, seg, RIGHT_INDICES)
                    # Use the hemisphere with more detections
                    predictions_a[pid] = times_l if len(times_l) >= len(times_r) else times_r
                elif lat == 'left':
                    times, _ = run_hemisphere(detector, seg, LEFT_INDICES)
                    predictions_a[pid] = times
                elif lat == 'right':
                    times, _ = run_hemisphere(detector, seg, RIGHT_INDICES)
                    predictions_a[pid] = times
                else:
                    # Unknown laterality: try both, pick better
                    times_l, _ = run_hemisphere(detector, seg, LEFT_INDICES)
                    times_r, _ = run_hemisphere(detector, seg, RIGHT_INDICES)
                    predictions_a[pid] = times_l if len(times_l) >= len(times_r) else times_r
            except Exception as e:
                continue

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gt_cases)} ({time.time()-t1:.0f}s)")

    result_a = evaluate(predictions_a, gt_cases)
    print(f"\n  Baseline A: F1={result_a['f1']:.4f}  Sens={result_a['sensitivity']:.4f}  "
          f"Prec={result_a['precision']:.4f}  FrqRho={result_a['freq_spearman']}  "
          f"N={result_a['n_cases']}")

    # ================================================================
    # Baseline B: Both hemispheres, pick best
    # ================================================================
    print(f"\n{'='*70}")
    print("  Baseline B: Both hemispheres, pick best")
    print(f"{'='*70}")

    predictions_b = {}
    t2 = time.time()

    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]

        try:
            times_l, _ = run_hemisphere(detector, seg, LEFT_INDICES)
            times_r, _ = run_hemisphere(detector, seg, RIGHT_INDICES)
            predictions_b[pid] = times_l if len(times_l) >= len(times_r) else times_r
        except:
            continue

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gt_cases)} ({time.time()-t2:.0f}s)")

    result_b = evaluate(predictions_b, gt_cases)
    print(f"\n  Baseline B: F1={result_b['f1']:.4f}  Sens={result_b['sensitivity']:.4f}  "
          f"Prec={result_b['precision']:.4f}  FrqRho={result_b['freq_spearman']}  "
          f"N={result_b['n_cases']}")

    # ================================================================
    # Also run current full pipeline for reference
    # ================================================================
    print(f"\n{'='*70}")
    print("  Reference: Current full pipeline (18 channels)")
    print(f"{'='*70}")

    predictions_ref = {}
    t3 = time.time()

    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            pat_segs = segments.get(pid, [])
            if not pat_segs:
                continue
            seg = pat_segs[0]
            try:
                result = detector.detect(seg, subtype='lpd')
                predictions_ref[pid] = result['global_times']
            except:
                continue
        else:
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
                predictions_ref[pid] = result['global_times']
            except:
                continue

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gt_cases)} ({time.time()-t3:.0f}s)")

    result_ref = evaluate(predictions_ref, gt_cases)
    print(f"\n  Reference: F1={result_ref['f1']:.4f}  Sens={result_ref['sensitivity']:.4f}  "
          f"Prec={result_ref['precision']:.4f}  FrqRho={result_ref['freq_spearman']}  "
          f"N={result_ref['n_cases']}")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline A (affected hemi):  F1={result_a['f1']:.4f}")
    print(f"  Baseline B (best of both):   F1={result_b['f1']:.4f}")
    print(f"  Reference (full 18ch):       F1={result_ref['f1']:.4f}")
    print(f"\n  Time: {time.time()-t0:.0f}s")

    # Save
    save_path = PROJECT_DIR / 'data' / 'hemi_cache'
    save_path.mkdir(exist_ok=True)
    with open(str(save_path / 'exp0_results.json'), 'w') as f:
        json.dump({
            'baseline_a': result_a,
            'baseline_b': result_b,
            'reference': result_ref,
        }, f, indent=2)
    print(f"  Saved to {save_path / 'exp0_results.json'}")


if __name__ == '__main__':
    main()
