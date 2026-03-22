"""
Evaluate CET FP reduction improvements:
  Stage A: Threshold CET before max() combination
  Stage B: PD-weighted channel selection for evidence aggregation

Uses parameter_sweep infrastructure for parameterized DP.

Usage:
    conda run -n foe_dl python code/cet_model/eval_improvements.py
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
    estimate_frequency_cnn, compute_cet_evidence,
    DEVICE, TOLERANCE_S, FS,
)
from cet_model.parameter_sweep import (
    run_hpp_single,
    extract_candidates_param,
    dp_best_sequence_param,
    em_refine_param,
)
from cet_model.auto_pipeline import (
    _detect_active_interval,
)
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence
from pd_channel_detector.channel_cnn import ChannelPDNetAttention

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

OPTIMIZED_PARAMS = {
    'dp_alpha': 1.275,
    'dp_beta': 0.3,
    'dp_lambda': 0.05,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}


# ============================================================================
# Evidence aggregation strategies
# ============================================================================

def aggregate_evidence_original(evidence_all, subtype, laterality=None):
    """Original aggregation: median across channels, laterality-aware."""
    if subtype == 'gpd':
        return np.median(evidence_all, axis=0)
    if laterality == 'left':
        return np.median(evidence_all[LEFT_INDICES], axis=0)
    elif laterality == 'right':
        return np.median(evidence_all[RIGHT_INDICES], axis=0)
    else:
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)


def aggregate_evidence_pd_weighted(evidence_all, pd_probs, subtype, laterality=None,
                                    pd_threshold=0.3):
    """PD-weighted aggregation: weight channels by PD probability.

    For each hemisphere, compute weighted mean of evidence using PD probs as weights.
    Only include channels with PD prob > threshold.
    """
    n_channels = evidence_all.shape[0]
    weights = np.array(pd_probs[:n_channels])

    if subtype == 'gpd':
        # Use all channels, weighted by PD prob
        mask = weights > pd_threshold
        if mask.sum() == 0:
            mask = np.ones(n_channels, dtype=bool)  # fallback
        w = weights[mask]
        w = w / w.sum()
        return np.average(evidence_all[mask], axis=0, weights=w)

    if laterality == 'left':
        ch_indices = LEFT_INDICES
    elif laterality == 'right':
        ch_indices = RIGHT_INDICES
    else:
        # Run both hemispheres, pick the one with higher total PD weight
        left_w = weights[LEFT_INDICES]
        right_w = weights[RIGHT_INDICES]

        left_mask = left_w > pd_threshold
        right_mask = right_w > pd_threshold

        if left_mask.sum() == 0:
            left_mask = np.ones(len(LEFT_INDICES), dtype=bool)
        if right_mask.sum() == 0:
            right_mask = np.ones(len(RIGHT_INDICES), dtype=bool)

        lw = left_w[left_mask]
        lw = lw / lw.sum()
        left_ev = np.average(evidence_all[LEFT_INDICES][left_mask], axis=0, weights=lw)

        rw = right_w[right_mask]
        rw = rw / rw.sum()
        right_ev = np.average(evidence_all[RIGHT_INDICES][right_mask], axis=0, weights=rw)

        # Use hemisphere with stronger PD signal
        if left_w.sum() > right_w.sum():
            return left_ev
        else:
            return right_ev

    # Known laterality
    ch_w = weights[ch_indices]
    mask = ch_w > pd_threshold
    if mask.sum() == 0:
        mask = np.ones(len(ch_indices), dtype=bool)
    w = ch_w[mask]
    w = w / w.sum()
    return np.average(evidence_all[ch_indices][mask], axis=0, weights=w)


def aggregate_per_hemisphere(evidence_all, pd_probs, pd_threshold=0.3):
    """Run independently on each hemisphere, return both.

    Returns:
        left_evidence, right_evidence: aggregated evidence per hemisphere
    """
    n_channels = evidence_all.shape[0]
    weights = np.array(pd_probs[:n_channels])

    results = {}
    for side, indices in [('left', LEFT_INDICES), ('right', RIGHT_INDICES)]:
        ch_w = weights[indices]
        mask = ch_w > pd_threshold
        if mask.sum() == 0:
            # Fallback to all channels in hemisphere
            results[side] = np.median(evidence_all[indices], axis=0)
        else:
            w = ch_w[mask]
            w = w / w.sum()
            results[side] = np.average(evidence_all[indices][mask], axis=0, weights=w)

    return results['left'], results['right']


# ============================================================================
# Combine evidence with optional CET thresholding
# ============================================================================

def combine_evidence_thresholded(hpp_evidence, cet_evidence, cet_threshold_pct=0):
    """max(HPP, CET) with CET values below threshold percentile zeroed."""
    # Normalize both to [0, 1]
    hpp_max = np.max(hpp_evidence) if np.max(hpp_evidence) > 0 else 1.0
    cet_max = np.max(cet_evidence) if np.max(cet_evidence) > 0 else 1.0
    hpp_norm = hpp_evidence / hpp_max
    cet_norm = cet_evidence / cet_max

    if cet_threshold_pct > 0:
        threshold = np.percentile(cet_norm[cet_norm > 0], cet_threshold_pct) if np.any(cet_norm > 0) else 0
        cet_norm = np.where(cet_norm > threshold, cet_norm, 0)

    return np.maximum(hpp_norm, cet_norm)


# ============================================================================
# Per-channel PD probability computation
# ============================================================================

@torch.no_grad()
def get_channel_pd_probs(segment_18ch, cnn_models, device=DEVICE, fs=FS):
    """Get PD probability for each channel using CNN+Attention ensemble."""
    n_channels = min(segment_18ch.shape[0], 18)
    pd_probs = []

    for ch in range(n_channels):
        ch_data = segment_18ch[ch].astype(np.float32).copy()
        if not np.all(np.isfinite(ch_data)):
            pd_probs.append(0.0)
            continue

        mu = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 1e-8:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu

        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(device)

        probs = []
        for model in cnn_models:
            pd_prob, _, _ = model(x)
            probs.append(pd_prob.item())
        pd_probs.append(np.mean(probs))

    return pd_probs


# ============================================================================
# Evaluate a configuration
# ============================================================================

def evaluate_config(precomputed, params, agg_method='original', cet_threshold_pct=0,
                    pd_threshold=0.3, tolerance=TOLERANCE_S):
    """Evaluate one configuration across all cases.

    Args:
        agg_method: 'original', 'pd_weighted', 'per_hemisphere'
        cet_threshold_pct: percentile threshold for CET evidence (0=no threshold)
        pd_threshold: min PD prob for channel inclusion
    """
    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs = [], []
    n_cases = 0

    for pid, pc in precomputed.items():
        gt_times = pc['gt_times']

        # Aggregate evidence based on method
        if agg_method == 'original':
            hpp_agg = aggregate_evidence_original(
                pc['hpp_all'], pc['subtype'], pc['laterality'])
            cet_agg = aggregate_evidence_original(
                pc['cet_all'], pc['subtype'], pc['laterality'])
        elif agg_method == 'pd_weighted':
            hpp_agg = aggregate_evidence_pd_weighted(
                pc['hpp_all'], pc['pd_probs'], pc['subtype'], pc['laterality'],
                pd_threshold)
            cet_agg = aggregate_evidence_pd_weighted(
                pc['cet_all'], pc['pd_probs'], pc['subtype'], pc['laterality'],
                pd_threshold)
        elif agg_method == 'per_hemisphere':
            # Use PD-weighted hemisphere selection
            hpp_left, hpp_right = aggregate_per_hemisphere(
                pc['hpp_all'], pc['pd_probs'], pd_threshold)
            cet_left, cet_right = aggregate_per_hemisphere(
                pc['cet_all'], pc['pd_probs'], pd_threshold)

            if pc['subtype'] == 'gpd':
                # For GPD, combine both hemispheres
                hpp_agg = 0.5 * (hpp_left + hpp_right)
                cet_agg = 0.5 * (cet_left + cet_right)
            else:
                # For LPD, pick hemisphere with higher PD signal
                left_pd = sum(pc['pd_probs'][i] for i in LEFT_INDICES)
                right_pd = sum(pc['pd_probs'][i] for i in RIGHT_INDICES)
                if left_pd > right_pd:
                    hpp_agg, cet_agg = hpp_left, cet_left
                else:
                    hpp_agg, cet_agg = hpp_right, cet_right
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")

        # Combine with optional CET thresholding
        evidence = combine_evidence_thresholded(hpp_agg, cet_agg, cet_threshold_pct)

        # Run HPP DP
        try:
            algo_times_arr = run_hpp_single(
                evidence, pc['hpp_all'], pc['cnn_freq'], FS, params)
            algo_times = sorted(algo_times_arr.tolist()) if len(algo_times_arr) > 0 else []
        except Exception:
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
                        best_dist = dist
                        best_ai = ai
            if best_dist <= tolerance and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        # Frequency from IPI
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
        'agg_method': agg_method,
        'cet_threshold_pct': cet_threshold_pct,
        'pd_threshold': pd_threshold,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print("  CET FP Reduction: Stage A (threshold) + Stage B (PD-weighted channels)")
    print("=" * 78)
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Load GT
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    # Load models
    print("\nLoading models...")
    cet_models = load_cet_unet_models(device=DEVICE)
    cnn_models = load_cnn_attn_models(device=DEVICE)
    print(f"  {len(cet_models)} CET-UNet, {len(cnn_models)} CNN+Attention models")

    # Precompute everything
    print(f"\nPrecomputing evidence, frequencies, and PD probs...")
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
        n_channels = min(seg.shape[0], 18)
        n_samples = seg.shape[1]

        try:
            # CNN frequency
            cnn_freq = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)

            # Per-channel PD probs
            pd_probs = get_channel_pd_probs(seg, cnn_models, DEVICE, FS)

            # Per-channel HPP evidence
            hpp_all = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                hpp_all[ch] = _compute_channel_evidence(seg[ch], FS)

            # Per-channel CET evidence
            cet_all = np.zeros((n_channels, n_samples), dtype=np.float32)
            for ch in range(n_channels):
                if np.all(np.isfinite(seg[ch])):
                    cet_all[ch] = compute_cet_evidence(seg[ch], cet_models, DEVICE)

        except Exception:
            continue

        precomputed[pid] = {
            'gt_times': gt_times,
            'hpp_all': hpp_all,
            'cet_all': cet_all,
            'pd_probs': pd_probs,
            'cnn_freq': cnn_freq,
            'subtype': subtype,
            'laterality': lat,
        }

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_pre:.0f}s)")

    print(f"  Precomputed {len(precomputed)} cases ({time.time()-t_pre:.0f}s)")

    # ====================================================================
    # Baseline: current best (original aggregation, no threshold)
    # ====================================================================
    print(f"\n{'='*78}")
    print("  BASELINE: max(HPP,CET)+CNN_freq+opt (original aggregation)")
    print(f"{'='*78}")

    baseline = evaluate_config(precomputed, OPTIMIZED_PARAMS,
                                agg_method='original', cet_threshold_pct=0)
    print(f"  F1={baseline['f1']:.4f}  Sens={baseline['sensitivity']:.4f}  "
          f"Prec={baseline['precision']:.4f}  FrqRho={baseline['freq_spearman']}  "
          f"TP={baseline['tp']} FN={baseline['fn']} FP={baseline['fp']}")

    # ====================================================================
    # STAGE A: CET threshold sweep
    # ====================================================================
    print(f"\n{'='*78}")
    print("  STAGE A: CET Threshold Sweep (original aggregation)")
    print(f"{'='*78}")

    threshold_results = []
    for pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        t1 = time.time()
        r = evaluate_config(precomputed, OPTIMIZED_PARAMS,
                            agg_method='original', cet_threshold_pct=pct)
        elapsed = time.time() - t1
        delta = r['f1'] - baseline['f1']
        marker = " <-- BEST" if r['f1'] == max(rr['f1'] for rr in threshold_results + [r]) else ""
        print(f"  CET threshold={pct:>2d}%  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
              f"FP={r['fp']:>5d}  ({elapsed:.1f}s){marker}")
        threshold_results.append(r)

    best_threshold = max(threshold_results, key=lambda r: r['f1'])
    best_pct = best_threshold['cet_threshold_pct']
    print(f"\n  Best CET threshold: {best_pct}%  F1={best_threshold['f1']:.4f}  "
          f"(+{best_threshold['f1'] - baseline['f1']:.4f} vs baseline)")

    # ====================================================================
    # STAGE B: PD-weighted channel selection
    # ====================================================================
    print(f"\n{'='*78}")
    print("  STAGE B: PD-Weighted Channel Selection")
    print(f"{'='*78}")

    # Test with different PD thresholds
    for agg in ['pd_weighted', 'per_hemisphere']:
        print(f"\n  --- {agg} ---")
        for pd_thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            t1 = time.time()
            r = evaluate_config(precomputed, OPTIMIZED_PARAMS,
                                agg_method=agg, cet_threshold_pct=0,
                                pd_threshold=pd_thresh)
            elapsed = time.time() - t1
            delta = r['f1'] - baseline['f1']
            print(f"    pd_thresh={pd_thresh:.1f}  F1={r['f1']:.4f} ({delta:+.4f})  "
                  f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
                  f"FP={r['fp']:>5d}  ({elapsed:.1f}s)")

    # ====================================================================
    # STAGE A+B combined: best threshold + best aggregation
    # ====================================================================
    print(f"\n{'='*78}")
    print("  COMBINED: Best CET threshold + PD-weighted aggregation")
    print(f"{'='*78}")

    combined_results = []
    for agg in ['original', 'pd_weighted', 'per_hemisphere']:
        for pct in [0, best_pct, 50, 70]:
            for pd_thresh in [0.2, 0.3, 0.4]:
                if agg == 'original' and pd_thresh != 0.3:
                    continue  # pd_thresh irrelevant for original
                t1 = time.time()
                r = evaluate_config(precomputed, OPTIMIZED_PARAMS,
                                    agg_method=agg, cet_threshold_pct=pct,
                                    pd_threshold=pd_thresh)
                elapsed = time.time() - t1
                delta = r['f1'] - baseline['f1']
                combined_results.append(r)
                print(f"  {agg:<15s} cet_thr={pct:>2d}% pd_thr={pd_thresh:.1f}  "
                      f"F1={r['f1']:.4f} ({delta:+.4f})  "
                      f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
                      f"FP={r['fp']:>5d}")

    overall_best = max(combined_results, key=lambda r: r['f1'])
    print(f"\n  OVERALL BEST: {overall_best['agg_method']} cet_thr={overall_best['cet_threshold_pct']}% "
          f"pd_thr={overall_best['pd_threshold']}")
    print(f"  F1={overall_best['f1']:.4f} ({overall_best['f1'] - baseline['f1']:+.4f} vs baseline)  "
          f"Sens={overall_best['sensitivity']:.4f}  Prec={overall_best['precision']:.4f}  "
          f"FrqRho={overall_best['freq_spearman']}  "
          f"TP={overall_best['tp']} FN={overall_best['fn']} FP={overall_best['fp']}")

    # ====================================================================
    # LPD vs GPD breakdown for best config
    # ====================================================================
    print(f"\n{'='*78}")
    print("  SUBTYPE BREAKDOWN: Best config vs Baseline")
    print(f"{'='*78}")

    for subtype_filter in ['lpd', 'gpd']:
        filtered = {pid: pc for pid, pc in precomputed.items()
                    if pc['subtype'] == subtype_filter}
        n_sub = len(filtered)

        # Baseline
        r_base = evaluate_config(filtered, OPTIMIZED_PARAMS,
                                  agg_method='original', cet_threshold_pct=0)
        # Best
        r_best = evaluate_config(filtered, OPTIMIZED_PARAMS,
                                  agg_method=overall_best['agg_method'],
                                  cet_threshold_pct=overall_best['cet_threshold_pct'],
                                  pd_threshold=overall_best['pd_threshold'])

        print(f"\n  {subtype_filter.upper()} (n={n_sub}):")
        print(f"    Baseline:  F1={r_base['f1']:.4f}  Sens={r_base['sensitivity']:.4f}  "
              f"Prec={r_base['precision']:.4f}  FP={r_base['fp']}")
        print(f"    Best:      F1={r_best['f1']:.4f}  Sens={r_best['sensitivity']:.4f}  "
              f"Prec={r_best['precision']:.4f}  FP={r_best['fp']}  "
              f"({r_best['f1'] - r_base['f1']:+.4f})")

    # Save results
    save_path = CACHE_DIR / 'improvement_stage_ab_results.json'
    save_data = {
        'baseline': baseline,
        'best_threshold': best_threshold,
        'overall_best': overall_best,
        'all_combined': combined_results,
    }
    with open(str(save_path), 'w') as f:
        json.dump(save_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    print(f"\n  Results saved to {save_path}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == '__main__':
    main()
