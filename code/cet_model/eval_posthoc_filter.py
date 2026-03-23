"""
Improvement #5: Post-hoc confidence filtering.

After DP finds discharge times, apply additional filters to remove
low-confidence detections:
  A. Drop discharges with evidence peak below a threshold (relative to median)
  B. If IPI CV is very high (irregular), fall back to HPP-only or stricter DP
  C. Require minimum number of discharges (isolated peaks are likely FPs)
  D. Drop discharges at segment boundaries (first/last 0.2s)

Usage:
    conda run -n foe_dl python code/cet_model/eval_posthoc_filter.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.auto_pipeline import (
    load_cet_unet_models, load_cnn_attn_models,
    estimate_frequency_cnn, estimate_frequency_acf,
    compute_cet_evidence,
    _aggregate_evidence, _detect_active_interval,
    DEVICE, TOLERANCE_S, FS,
)
from cet_model.parameter_sweep import run_hpp_single
from optimization_harness_v2 import load_dataset
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

OPTIMIZED_PARAMS = {
    'dp_alpha': 1.275,
    'dp_beta': 0.3,
    'dp_lambda': 0.05,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}


def normalize_ev(ev):
    mx = np.max(ev)
    return ev / mx if mx > 0 else ev


def combine_max_thresholded(hpp_agg, cet_agg, cet_threshold_pct=80):
    hpp_norm = normalize_ev(hpp_agg)
    cet_norm = normalize_ev(cet_agg)
    if cet_threshold_pct > 0 and np.any(cet_norm > 0):
        thr = np.percentile(cet_norm[cet_norm > 0], cet_threshold_pct)
        cet_norm = np.where(cet_norm > thr, cet_norm, 0)
    return np.maximum(hpp_norm, cet_norm)


def posthoc_filter(algo_times_arr, evidence, fs=FS,
                   min_evidence_ratio=0.0,
                   max_ipi_cv=None,
                   min_discharges=0,
                   boundary_s=0.0):
    """Apply post-hoc filters to detected discharge times.

    Args:
        algo_times_arr: array of discharge sample indices
        evidence: combined evidence trace
        min_evidence_ratio: drop discharges with evidence < ratio * median
        max_ipi_cv: if IPI CV exceeds this, drop weakest discharges
        min_discharges: require at least this many discharges
        boundary_s: drop discharges within this many seconds of edges
    """
    if len(algo_times_arr) == 0:
        return algo_times_arr

    times = algo_times_arr.copy()
    n_samples = len(evidence)

    # A. Drop discharges at segment boundaries
    if boundary_s > 0:
        boundary_samples = int(boundary_s * fs)
        mask = (times >= boundary_samples) & (times < n_samples - boundary_samples)
        times = times[mask]

    if len(times) == 0:
        return times

    # B. Drop discharges with evidence below threshold
    if min_evidence_ratio > 0:
        peak_values = np.array([evidence[int(t)] for t in times])
        median_peak = np.median(peak_values)
        threshold = min_evidence_ratio * median_peak
        mask = peak_values >= threshold
        times = times[mask]

    if len(times) == 0:
        return times

    # C. Minimum discharge count
    if min_discharges > 0 and len(times) < min_discharges:
        return np.array([], dtype=int)

    # D. IPI CV filtering (drop outlier intervals by removing their weaker endpoint)
    if max_ipi_cv is not None and len(times) >= 3:
        t_sec = times / fs
        ipis = np.diff(t_sec)
        cv = np.std(ipis) / np.mean(ipis) if np.mean(ipis) > 0 else 0
        if cv > max_ipi_cv:
            # Try to remove the discharge causing the largest IPI deviation
            median_ipi = np.median(ipis)
            deviations = np.abs(ipis - median_ipi)
            worst_ipi_idx = np.argmax(deviations)
            # Remove the weaker of the two endpoints
            peak_values = np.array([evidence[int(t)] for t in times])
            left_val = peak_values[worst_ipi_idx]
            right_val = peak_values[worst_ipi_idx + 1]
            drop_idx = worst_ipi_idx if left_val < right_val else worst_ipi_idx + 1
            times = np.delete(times, drop_idx)

    return times


def evaluate_config(precomputed, posthoc_params, cet_threshold_pct=80):
    """Evaluate with specific post-hoc filtering config."""
    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs = [], []
    n_cases = 0

    for pid, pc in precomputed.items():
        gt_times = pc['gt_times']
        evidence = combine_max_thresholded(pc['hpp_agg'], pc['cet_agg'], cet_threshold_pct)

        try:
            algo_times_s = run_hpp_single(
                evidence, pc['hpp_all'], pc['cnn_freq'], FS, OPTIMIZED_PARAMS)
            # run_hpp_single returns times in seconds; convert to samples for filtering
            algo_samples = (algo_times_s * FS).astype(int) if len(algo_times_s) > 0 else np.array([], dtype=int)
        except Exception:
            continue

        # Apply post-hoc filters (works on sample indices)
        algo_samples = posthoc_filter(algo_samples, evidence, FS, **posthoc_params)
        algo_times = sorted((algo_samples / FS).tolist()) if len(algo_samples) > 0 else []

        n_cases += 1

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

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    if len(gt_freqs) >= 3:
        freq_rho, _ = spearmanr(algo_freqs, gt_freqs)
    else:
        freq_rho = float('nan')

    return {
        'n_cases': n_cases,
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'freq_spearman': round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("  Improvement #5: Post-hoc Confidence Filtering")
    print("=" * 78)

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    cet_models = load_cet_unet_models(device=DEVICE)
    cnn_models = load_cnn_attn_models(device=DEVICE)

    print(f"\nPrecomputing...")
    precomputed = {}
    t_pre = time.time()

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
        n_ch = min(seg.shape[0], 18)
        n_samp = seg.shape[1]

        try:
            cnn_freq = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)
            hpp_all = np.zeros((n_ch, n_samp))
            for ch in range(n_ch):
                hpp_all[ch] = _compute_channel_evidence(seg[ch], FS)
            cet_all = np.zeros((n_ch, n_samp), dtype=np.float32)
            for ch in range(n_ch):
                if np.all(np.isfinite(seg[ch])):
                    cet_all[ch] = compute_cet_evidence(seg[ch], cet_models, DEVICE)
            hpp_agg = _aggregate_evidence(hpp_all, subtype, lat)
            cet_agg = _aggregate_evidence(cet_all, subtype, lat)
        except Exception:
            continue

        precomputed[pid] = {
            'gt_times': gt_times, 'hpp_all': hpp_all,
            'hpp_agg': hpp_agg, 'cet_agg': cet_agg,
            'cnn_freq': cnn_freq, 'subtype': subtype, 'laterality': lat,
        }
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_pre:.0f}s)")

    print(f"  Precomputed {len(precomputed)} cases ({time.time()-t_pre:.0f}s)")

    # Baseline
    r_base = evaluate_config(precomputed, {
        'min_evidence_ratio': 0, 'max_ipi_cv': None,
        'min_discharges': 0, 'boundary_s': 0})
    print(f"\n  Baseline: F1={r_base['f1']:.4f}  Sens={r_base['sensitivity']:.4f}  "
          f"Prec={r_base['precision']:.4f}  FP={r_base['fp']}")

    # Sweep each filter independently
    print(f"\n{'='*78}")
    print("  A: Min evidence ratio (drop weak peaks)")
    print(f"{'='*78}")
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r = evaluate_config(precomputed, {
            'min_evidence_ratio': ratio, 'max_ipi_cv': None,
            'min_discharges': 0, 'boundary_s': 0})
        delta = r['f1'] - r_base['f1']
        print(f"  ratio={ratio:.1f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  FP={r['fp']}")

    print(f"\n{'='*78}")
    print("  B: Boundary exclusion")
    print(f"{'='*78}")
    for bs in [0.1, 0.2, 0.3, 0.5]:
        r = evaluate_config(precomputed, {
            'min_evidence_ratio': 0, 'max_ipi_cv': None,
            'min_discharges': 0, 'boundary_s': bs})
        delta = r['f1'] - r_base['f1']
        print(f"  boundary={bs:.1f}s:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  FP={r['fp']}")

    print(f"\n{'='*78}")
    print("  C: Min discharge count")
    print(f"{'='*78}")
    for md in [2, 3, 4]:
        r = evaluate_config(precomputed, {
            'min_evidence_ratio': 0, 'max_ipi_cv': None,
            'min_discharges': md, 'boundary_s': 0})
        delta = r['f1'] - r_base['f1']
        print(f"  min_discharges={md}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  FP={r['fp']}")

    print(f"\n{'='*78}")
    print("  D: IPI CV outlier removal")
    print(f"{'='*78}")
    for cv in [0.3, 0.4, 0.5, 0.6, 0.8]:
        r = evaluate_config(precomputed, {
            'min_evidence_ratio': 0, 'max_ipi_cv': cv,
            'min_discharges': 0, 'boundary_s': 0})
        delta = r['f1'] - r_base['f1']
        print(f"  max_cv={cv:.1f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  FP={r['fp']}")

    # Combined sweep of best options
    print(f"\n{'='*78}")
    print("  COMBINED: Best post-hoc filters")
    print(f"{'='*78}")

    best = {'f1': 0}
    for ratio in [0, 0.1, 0.2, 0.3]:
        for bs in [0, 0.1, 0.2]:
            for md in [0, 2, 3]:
                for cv in [None, 0.4, 0.5]:
                    r = evaluate_config(precomputed, {
                        'min_evidence_ratio': ratio, 'max_ipi_cv': cv,
                        'min_discharges': md, 'boundary_s': bs})
                    if r['f1'] > best['f1']:
                        best = {**r, 'ratio': ratio, 'boundary': bs,
                                'min_disc': md, 'cv': cv}

    print(f"  Best: ratio={best['ratio']}, boundary={best['boundary']}s, "
          f"min_disc={best['min_disc']}, max_cv={best['cv']}")
    print(f"  F1={best['f1']:.4f} ({best['f1'] - r_base['f1']:+.4f})  "
          f"Sens={best['sensitivity']:.4f}  Prec={best['precision']:.4f}  "
          f"FP={best['fp']}")

    save_path = CACHE_DIR / 'improvement_posthoc_results.json'
    with open(str(save_path), 'w') as f:
        json.dump({'baseline': r_base, 'best': best}, f, indent=2)
    print(f"\n  Saved to {save_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
