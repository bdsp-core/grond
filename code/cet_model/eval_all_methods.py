"""
Evaluate all fair EEG-only methods including max(HPP,CET)+CNN_freq+optimized_DP.

Uses parameter_sweep infrastructure for correct parameterized DP.

Usage:
    conda run -n foe_dl python code/cet_model/eval_all_methods.py
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
    estimate_frequency_cnn, estimate_frequency_bootstrap,
    compute_cet_evidence,
    _aggregate_evidence, _detect_active_interval,
    DEVICE, TOLERANCE_S, FS,
)
from cet_model.parameter_sweep import (
    compute_all_evidence,
    combine_evidence,
    run_hpp_single,
)
from optimization_harness_v2 import load_dataset
from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence, detect_discharge_times_hpp,
)

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

# Optimized parameters from parameter sweep
OPTIMIZED_PARAMS = {
    'dp_alpha': 1.275,
    'dp_beta': 0.3,
    'dp_lambda': 0.05,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}

# Default parameters
DEFAULT_PARAMS = {
    'dp_alpha': 3.0,
    'dp_beta': 1.0,
    'dp_lambda': 0.02,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}


def evaluate_timing(algo_times_list, gt_times, tolerance=TOLERANCE_S):
    """Match algo times to GT times, return TP/FN/FP."""
    gt_matched = [False] * len(gt_times)
    algo_matched = [False] * len(algo_times_list)

    for gi, gt in enumerate(gt_times):
        best_dist, best_ai = np.inf, -1
        for ai, at in enumerate(algo_times_list):
            if not algo_matched[ai]:
                dist = abs(gt - at)
                if dist < best_dist:
                    best_dist = dist
                    best_ai = ai
        if best_dist <= tolerance and best_ai >= 0:
            gt_matched[gi] = True
            algo_matched[best_ai] = True

    tp = sum(gt_matched)
    fn = len(gt_times) - tp
    fp = len(algo_times_list) - sum(algo_matched)
    return tp, fn, fp


def main():
    t0 = time.time()
    print("=" * 74)
    print("  Full Method Comparison — Updated Gold Standard Labels")
    print("  All methods EEG-only (no gold freq except reference)")
    print("=" * 74)
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Load GT discharge times
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    # Load models
    print("\nLoading models...")
    cet_models = load_cet_unet_models(device=DEVICE)
    print(f"  {len(cet_models)} CET-UNet models")
    cnn_models = load_cnn_attn_models(device=DEVICE)
    print(f"  {len(cnn_models)} CNN+Attention models")

    # Precompute evidence and frequencies for all GT cases
    print(f"\nPrecomputing evidence + frequencies for {len(gt_cases)} cases...")
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
        gold_freq = float(row['gold_standard_freq'])

        try:
            cnn_freq = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)
            boot_freq = estimate_frequency_bootstrap(seg, subtype, lat, FS)
            hpp_ev, cet_ev, hpp_all = compute_all_evidence(
                seg, subtype, lat, cet_models, FS)
        except Exception as e:
            continue

        precomputed[pid] = {
            'gt_times': gt_times,
            'hpp_evidence': hpp_ev,
            'cet_evidence': cet_ev,
            'evidence_all_hpp': hpp_all,
            'cnn_freq': cnn_freq,
            'bootstrap_freq': boot_freq,
            'gold_freq': gold_freq,
            'subtype': subtype,
            'laterality': lat,
            'segment': seg,
        }

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_pre:.0f}s)")

    print(f"  Precomputed {len(precomputed)} cases ({time.time()-t_pre:.0f}s)")

    # Define methods to evaluate
    methods = [
        # (name, evidence_type, freq_key, dp_params, use_gold_freq)
        ('HPP+CNN_freq', 'hpp', 'cnn_freq', DEFAULT_PARAMS, False),
        ('HPP+bootstrap', 'hpp', 'bootstrap_freq', DEFAULT_PARAMS, False),
        ('CET+bootstrap', 'cet', 'bootstrap_freq', DEFAULT_PARAMS, False),
        ('CET+CNN_freq', 'cet', 'cnn_freq', DEFAULT_PARAMS, False),
        ('max(HPP,CET)+CNN_freq+opt', 'max_combine', 'cnn_freq', OPTIMIZED_PARAMS, False),
        ('HPP+gold [REF]', 'hpp', 'gold_freq', DEFAULT_PARAMS, True),
    ]

    results = []

    for method_name, ev_type, freq_key, params, is_ref in methods:
        print(f"\n{'─'*70}")
        print(f"  {method_name}")
        total_tp, total_fn, total_fp = 0, 0, 0
        gt_freqs, algo_freqs = [], []
        n_cases = 0
        t1 = time.time()

        for pid, pc in precomputed.items():
            gt_times = pc['gt_times']
            freq_est = pc[freq_key]

            # Get evidence
            try:
                evidence = combine_evidence(pc['hpp_evidence'], pc['cet_evidence'], ev_type)
            except ValueError:
                continue

            # Run HPP with parameterized DP
            if is_ref:
                # Reference: use original hpp_discharge_marking with gold freq
                try:
                    result = detect_discharge_times_hpp(
                        pc['segment'], fs=FS, subtype=pc['subtype'],
                        freq_estimate=freq_est, laterality=pc['laterality'],
                        refine=True)
                    algo_times = sorted(result['global_times'])
                except Exception:
                    continue
            else:
                try:
                    algo_times_arr = run_hpp_single(
                        evidence, pc['evidence_all_hpp'], freq_est, FS, params)
                    algo_times = sorted(algo_times_arr.tolist()) if len(algo_times_arr) > 0 else []
                except Exception:
                    continue

            n_cases += 1

            # Evaluate timing
            tp, fn, fp = evaluate_timing(algo_times, gt_times)
            total_tp += tp
            total_fn += fn
            total_fp += fp

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

        elapsed = time.time() - t1

        sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

        if len(gt_freqs) >= 3:
            freq_rho, _ = spearmanr(algo_freqs, gt_freqs)
        else:
            freq_rho = float('nan')

        r = {
            'method': method_name,
            'n_cases': n_cases,
            'sensitivity': sens,
            'precision': prec,
            'f1': f1,
            'freq_spearman': freq_rho,
            'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
            'time_s': elapsed,
        }
        results.append(r)

        ref_tag = " *" if is_ref else ""
        print(f"  N={n_cases}  Sens={sens:.3f}  Prec={prec:.3f}  F1={f1:.3f}  "
              f"FrqRho={freq_rho:.3f}  TP={total_tp} FN={total_fn} FP={total_fp}  "
              f"({elapsed:.1f}s){ref_tag}")

    # Print comparison table
    print(f"\n{'='*74}")
    print("  COMPARISON TABLE — All EEG-Only Methods (Updated Gold Standard)")
    print(f"{'='*74}")

    header = f"{'Method':<30s} {'Sens':>6s} {'Prec':>6s} {'F1':>6s} {'FrqRho':>7s} {'N':>4s}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        ref = " *" if "[REF]" in r['method'] else ""
        rho = f"{r['freq_spearman']:.3f}" if np.isfinite(r['freq_spearman']) else "  N/A"
        print(f"{r['method']:<30s} {r['sensitivity']:>6.3f} {r['precision']:>6.3f} "
              f"{r['f1']:>6.3f} {rho:>7s} {r['n_cases']:>4d}{ref}")

    print(f"\n  * = Reference method (gold freq, not deployment-ready)")

    # Save
    save_path = CACHE_DIR / 'method_comparison_updated_labels.json'
    save_data = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if isinstance(v, float) and np.isfinite(v):
                sr[k] = round(v, 4)
            else:
                sr[k] = v
        save_data.append(sr)
    with open(str(save_path), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {save_path}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print("=" * 74)


if __name__ == '__main__':
    main()
