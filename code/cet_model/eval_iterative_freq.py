"""
Improvement #3: Iterative frequency refinement.

After initial detection with CNN frequency, compute IPI from detected discharges,
use that as a refined frequency estimate, and re-run the DP.

Also tests combining CNN freq with ACF freq for a better initial estimate.

Usage:
    conda run -n foe_dl python code/cet_model/eval_iterative_freq.py
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
from cet_model.parameter_sweep import (
    compute_all_evidence,
    combine_evidence,
    run_hpp_single,
    extract_candidates_param,
    dp_best_sequence_param,
    em_refine_param,
)
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

OPTIMIZED_PARAMS = {
    'dp_alpha': 1.275,
    'dp_beta': 0.3,
    'dp_lambda': 0.05,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}

# Best CET threshold from Stage A
CET_THRESHOLD_PCT = 80


def combine_evidence_thresholded(hpp_evidence, cet_evidence, cet_threshold_pct=0):
    """max(HPP, CET) with CET below threshold percentile zeroed."""
    hpp_max = np.max(hpp_evidence) if np.max(hpp_evidence) > 0 else 1.0
    cet_max = np.max(cet_evidence) if np.max(cet_evidence) > 0 else 1.0
    hpp_norm = hpp_evidence / hpp_max
    cet_norm = cet_evidence / cet_max

    if cet_threshold_pct > 0:
        threshold = np.percentile(cet_norm[cet_norm > 0], cet_threshold_pct) if np.any(cet_norm > 0) else 0
        cet_norm = np.where(cet_norm > threshold, cet_norm, 0)

    return np.maximum(hpp_norm, cet_norm)


def run_hpp_with_freq(evidence, evidence_all_hpp, freq_estimate, params):
    """Run HPP pipeline with a specific frequency estimate."""
    return run_hpp_single(evidence, evidence_all_hpp, freq_estimate, FS, params)


def iterative_refinement(evidence, evidence_all_hpp, initial_freq, params,
                         n_iterations=2, blend_weight=0.5):
    """Iteratively refine frequency: detect -> IPI -> re-detect.

    Args:
        blend_weight: how much to trust IPI freq vs initial (0=all initial, 1=all IPI)
    """
    freq = initial_freq

    for iteration in range(n_iterations):
        # Detect with current freq
        algo_times_arr = run_hpp_with_freq(evidence, evidence_all_hpp, freq, params)

        if len(algo_times_arr) < 2:
            break

        # Compute IPI-based frequency
        times = algo_times_arr / FS
        ipis = np.diff(times)
        ipi_median = float(np.median(ipis))
        if ipi_median <= 0:
            break

        ipi_freq = 1.0 / ipi_median
        ipi_freq = float(np.clip(ipi_freq, 0.3, 3.5))

        # Blend: on first iteration use more initial, on later use more IPI
        iter_blend = blend_weight + (1 - blend_weight) * (iteration / max(n_iterations - 1, 1))
        freq = (1 - iter_blend) * initial_freq + iter_blend * ipi_freq

    # Final run with refined frequency
    return run_hpp_with_freq(evidence, evidence_all_hpp, freq, params), freq


def evaluate_method(precomputed, freq_method, params, cet_threshold_pct=0,
                    n_iterations=0, blend_weight=0.5, acf_weight=0.0):
    """Evaluate a frequency method across all cases.

    Args:
        freq_method: 'cnn', 'acf', 'ensemble' (CNN+ACF blend)
        n_iterations: 0 = no iterative refinement, >0 = iterate
        blend_weight: for iterative refinement
        acf_weight: for ensemble, weight of ACF freq (CNN weight = 1-acf_weight)
    """
    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs, input_freqs = [], [], []
    n_cases = 0

    for pid, pc in precomputed.items():
        gt_times = pc['gt_times']

        # Get evidence (max combine with threshold)
        evidence = combine_evidence_thresholded(
            pc['hpp_agg'], pc['cet_agg'], cet_threshold_pct)

        # Determine initial frequency
        if freq_method == 'cnn':
            initial_freq = pc['cnn_freq']
        elif freq_method == 'acf':
            initial_freq = pc['acf_freq']
        elif freq_method == 'ensemble':
            initial_freq = (1 - acf_weight) * pc['cnn_freq'] + acf_weight * pc['acf_freq']
            initial_freq = float(np.clip(initial_freq, 0.3, 3.5))
        else:
            raise ValueError(f"Unknown freq_method: {freq_method}")

        # Run with optional iterative refinement
        try:
            if n_iterations > 0:
                algo_times_arr, final_freq = iterative_refinement(
                    evidence, pc['hpp_all'], initial_freq, params,
                    n_iterations=n_iterations, blend_weight=blend_weight)
            else:
                algo_times_arr = run_hpp_with_freq(
                    evidence, pc['hpp_all'], initial_freq, params)
                final_freq = initial_freq

            algo_times = sorted(algo_times_arr.tolist()) if len(algo_times_arr) > 0 else []
        except Exception:
            continue

        n_cases += 1
        input_freqs.append(initial_freq)

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
            if best_dist <= TOLERANCE_S and best_ai >= 0:
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
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("  Improvement #3: Iterative Frequency Refinement")
    print("=" * 78)
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Load GT
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    # Load models
    print("\nLoading models...")
    cet_models = load_cet_unet_models(device=DEVICE)
    cnn_models = load_cnn_attn_models(device=DEVICE)

    # Precompute
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
        n_channels = min(seg.shape[0], 18)
        n_samples = seg.shape[1]

        try:
            cnn_freq = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)
            acf_freq = estimate_frequency_acf(seg, subtype, lat, FS)

            hpp_all = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                hpp_all[ch] = _compute_channel_evidence(seg[ch], FS)

            cet_all = np.zeros((n_channels, n_samples), dtype=np.float32)
            for ch in range(n_channels):
                if np.all(np.isfinite(seg[ch])):
                    cet_all[ch] = compute_cet_evidence(seg[ch], cet_models, DEVICE)

            hpp_agg = _aggregate_evidence(hpp_all, subtype, lat)
            cet_agg = _aggregate_evidence(cet_all, subtype, lat)

        except Exception:
            continue

        precomputed[pid] = {
            'gt_times': gt_times,
            'hpp_all': hpp_all,
            'hpp_agg': hpp_agg,
            'cet_agg': cet_agg,
            'cnn_freq': cnn_freq,
            'acf_freq': acf_freq,
            'gold_freq': float(row['gold_standard_freq']),
            'subtype': subtype,
            'laterality': lat,
        }

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_pre:.0f}s)")

    print(f"  Precomputed {len(precomputed)} cases ({time.time()-t_pre:.0f}s)")

    # ====================================================================
    # Baselines
    # ====================================================================
    print(f"\n{'='*78}")
    print("  BASELINES")
    print(f"{'='*78}")

    # Current best: CNN freq, no threshold
    r_base = evaluate_method(precomputed, 'cnn', OPTIMIZED_PARAMS, cet_threshold_pct=0)
    print(f"  CNN freq, no threshold:   F1={r_base['f1']:.4f}  Sens={r_base['sensitivity']:.4f}  "
          f"Prec={r_base['precision']:.4f}  FrqRho={r_base['freq_spearman']}")

    # Best from Stage A: CNN freq + 80% threshold
    r_thr = evaluate_method(precomputed, 'cnn', OPTIMIZED_PARAMS, cet_threshold_pct=CET_THRESHOLD_PCT)
    print(f"  CNN freq, 80% threshold:  F1={r_thr['f1']:.4f}  Sens={r_thr['sensitivity']:.4f}  "
          f"Prec={r_thr['precision']:.4f}  FrqRho={r_thr['freq_spearman']}")

    # ====================================================================
    # Test 1: ACF freq alone
    # ====================================================================
    print(f"\n{'='*78}")
    print("  ACF frequency alone")
    print(f"{'='*78}")

    r_acf = evaluate_method(precomputed, 'acf', OPTIMIZED_PARAMS, cet_threshold_pct=CET_THRESHOLD_PCT)
    print(f"  ACF freq:                 F1={r_acf['f1']:.4f}  Sens={r_acf['sensitivity']:.4f}  "
          f"Prec={r_acf['precision']:.4f}  FrqRho={r_acf['freq_spearman']}")

    # ====================================================================
    # Test 2: CNN+ACF ensemble
    # ====================================================================
    print(f"\n{'='*78}")
    print("  CNN+ACF Ensemble (sweep ACF weight)")
    print(f"{'='*78}")

    for acf_w in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r = evaluate_method(precomputed, 'ensemble', OPTIMIZED_PARAMS,
                            cet_threshold_pct=CET_THRESHOLD_PCT,
                            acf_weight=acf_w)
        delta = r['f1'] - r_thr['f1']
        print(f"  ACF weight={acf_w:.1f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
              f"FrqRho={r['freq_spearman']}")

    # ====================================================================
    # Test 3: Iterative refinement (CNN freq)
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Iterative Frequency Refinement (CNN initial)")
    print(f"{'='*78}")

    for n_iter in [1, 2, 3]:
        for blend in [0.3, 0.5, 0.7, 1.0]:
            r = evaluate_method(precomputed, 'cnn', OPTIMIZED_PARAMS,
                                cet_threshold_pct=CET_THRESHOLD_PCT,
                                n_iterations=n_iter, blend_weight=blend)
            delta = r['f1'] - r_thr['f1']
            print(f"  iter={n_iter} blend={blend:.1f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
                  f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
                  f"FrqRho={r['freq_spearman']}")

    # ====================================================================
    # Test 4: Ensemble + iterative
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Ensemble + Iterative (best combos)")
    print(f"{'='*78}")

    best_configs = []
    for acf_w in [0.1, 0.2, 0.3]:
        for n_iter in [1, 2]:
            for blend in [0.3, 0.5, 0.7]:
                r = evaluate_method(precomputed, 'ensemble', OPTIMIZED_PARAMS,
                                    cet_threshold_pct=CET_THRESHOLD_PCT,
                                    n_iterations=n_iter, blend_weight=blend,
                                    acf_weight=acf_w)
                delta = r['f1'] - r_thr['f1']
                best_configs.append((r, acf_w, n_iter, blend, delta))

    # Sort by F1 and show top 10
    best_configs.sort(key=lambda x: x[0]['f1'], reverse=True)
    print(f"\n  Top 10 configurations:")
    for r, acf_w, n_iter, blend, delta in best_configs[:10]:
        print(f"    acf_w={acf_w:.1f} iter={n_iter} blend={blend:.1f}:  "
              f"F1={r['f1']:.4f} ({delta:+.4f})  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  FrqRho={r['freq_spearman']}")

    # ====================================================================
    # Test 5: Re-optimize DP params with iterative freq
    # ====================================================================
    print(f"\n{'='*78}")
    print("  DP param sweep with best freq method")
    print(f"{'='*78}")

    # Get best freq config
    best_r, best_acf_w, best_n_iter, best_blend, _ = best_configs[0]
    print(f"  Using: acf_w={best_acf_w}, iter={best_n_iter}, blend={best_blend}")

    # Sweep alpha (most impactful DP param)
    for alpha in [0.8, 1.0, 1.275, 1.5, 2.0, 2.5]:
        params = dict(OPTIMIZED_PARAMS)
        params['dp_alpha'] = alpha
        r = evaluate_method(precomputed, 'ensemble', params,
                            cet_threshold_pct=CET_THRESHOLD_PCT,
                            n_iterations=best_n_iter, blend_weight=best_blend,
                            acf_weight=best_acf_w)
        delta = r['f1'] - r_thr['f1']
        print(f"    alpha={alpha:.3f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}  "
              f"FrqRho={r['freq_spearman']}")

    # Sweep lambda
    for lam in [0.02, 0.03, 0.05, 0.08, 0.1]:
        params = dict(OPTIMIZED_PARAMS)
        params['dp_lambda'] = lam
        r = evaluate_method(precomputed, 'ensemble', params,
                            cet_threshold_pct=CET_THRESHOLD_PCT,
                            n_iterations=best_n_iter, blend_weight=best_blend,
                            acf_weight=best_acf_w)
        delta = r['f1'] - r_thr['f1']
        print(f"    lambda={lam:.3f}:  F1={r['f1']:.4f} ({delta:+.4f})  "
              f"Sens={r['sensitivity']:.4f}  Prec={r['precision']:.4f}")

    # ====================================================================
    # Final summary
    # ====================================================================
    print(f"\n{'='*78}")
    print("  SUMMARY")
    print(f"{'='*78}")
    print(f"  Original baseline (no improvements):  F1={r_base['f1']:.4f}")
    print(f"  + CET 80% threshold (Stage A):        F1={r_thr['f1']:.4f}  "
          f"({r_thr['f1'] - r_base['f1']:+.4f})")
    print(f"  + Best freq refinement:               F1={best_r['f1']:.4f}  "
          f"({best_r['f1'] - r_base['f1']:+.4f} total)")

    # Save
    save_path = CACHE_DIR / 'improvement_iterative_freq_results.json'
    save_data = {
        'baseline_no_threshold': {k: v for k, v in r_base.items()},
        'baseline_with_threshold': {k: v for k, v in r_thr.items()},
        'best_freq_config': {
            'acf_weight': best_acf_w,
            'n_iterations': best_n_iter,
            'blend_weight': best_blend,
            **{k: v for k, v in best_r.items()},
        },
        'top10': [
            {'acf_weight': aw, 'n_iterations': ni, 'blend_weight': bl,
             **{k: v for k, v in r.items()}}
            for r, aw, ni, bl, _ in best_configs[:10]
        ],
    }
    with open(str(save_path), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {save_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == '__main__':
    main()
