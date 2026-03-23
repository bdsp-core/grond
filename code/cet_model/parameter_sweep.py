"""
HPP parameter sweep for discharge detection (FAIR / EEG-only setting).

Uses CNN-estimated frequency as the period prior. Tests different evidence
types (hpp, cet, and combinations) and DP parameters in a staged sweep.

Stage 1: Compare evidence types with current best DP params
Stage 2: Sweep DP params independently with best evidence type
Stage 3: Fine-tune top combinations from Stage 2

Usage:
    conda run -n foe_dl python code/cet_model/parameter_sweep.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from itertools import product
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.cet import CETUNet
from cet_model.auto_pipeline import (
    load_cet_unet_models, load_cnn_attn_models,
    estimate_frequency_cnn, compute_cet_evidence,
    _aggregate_evidence, _detect_active_interval,
    _per_channel_times,
    DEVICE, TOLERANCE_S,
)
from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

# Default DP parameters (current best)
DEFAULT_PARAMS = {
    'dp_alpha': 3.0,
    'dp_beta': 1.0,
    'dp_lambda': 0.02,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}

# Template refinement constants
TEMPLATE_HALF_MS = 150
CHANNEL_REFINE_MS = 50


# ============================================================================
# Core HPP functions with parameterized DP
# ============================================================================

def extract_candidates_param(evidence, fs, freq_estimate, active_start, active_end,
                             peak_height_frac):
    """Find local maxima within the active interval (parameterized)."""
    segment = evidence[active_start:active_end + 1]
    if len(segment) < 3:
        return np.array([], dtype=int)
    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(20, int(0.2 * T * fs))
    min_height = peak_height_frac * np.max(segment)
    peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
    strong_height = 0.5 * np.max(segment)
    strong_peaks, _ = find_peaks(segment, height=strong_height,
                                  distance=max(10, int(0.1 * T * fs)))
    peaks = np.unique(np.concatenate([peaks, strong_peaks]))
    return peaks + active_start


def dp_best_sequence_param(candidates, evidence, fs, freq_estimate,
                           dp_alpha, dp_beta, dp_lambda, max_skip=3):
    """Forward DP with parameterized scoring."""
    if len(candidates) == 0:
        return np.array([], dtype=int)
    if len(candidates) == 1:
        return candidates.copy()
    T = 1.0 / freq_estimate
    n = len(candidates)
    raw_scores = np.array([evidence[c] for c in candidates])
    node_scores = raw_scores ** 1.5
    best_score = np.full(n, -np.inf)
    best_prev = np.full(n, -1, dtype=int)
    for i in range(n):
        best_score[i] = node_scores[i] - dp_lambda
    for j in range(1, n):
        for i in range(j):
            dt = (candidates[j] - candidates[i]) / fs
            if dt <= 0 or dt > 4 * T:
                continue
            best_edge = -np.inf
            for m in range(1, max_skip + 1):
                deviation = (dt - m * T) / (m * T)
                interval_score = -dp_alpha * deviation ** 2
                skip_penalty = -dp_beta * (m - 1)
                edge = interval_score + skip_penalty
                if edge > best_edge:
                    best_edge = edge
            total = best_score[i] + best_edge + node_scores[j] - dp_lambda
            if total > best_score[j]:
                best_score[j] = total
                best_prev[j] = i
    best_end = int(np.argmax(best_score))
    path = []
    idx = best_end
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()
    return candidates[np.array(path)]


def em_refine_param(evidence, discharge_samples, fs, freq_estimate,
                    dp_alpha, dp_beta, dp_lambda, max_skip=3):
    """EM template refinement with parameterized DP."""
    n = len(evidence)
    half_win = int(TEMPLATE_HALF_MS / 1000.0 * fs)
    if len(discharge_samples) < 2:
        return discharge_samples
    snippets = []
    for s in discharge_samples:
        lo = s - half_win
        hi = s + half_win + 1
        if lo >= 0 and hi <= n:
            snippets.append(evidence[lo:hi])
    if len(snippets) < 2:
        return discharge_samples
    template = np.mean(snippets, axis=0)
    template = template - np.mean(template)
    corr = np.zeros(n)
    t_len = len(template)
    t_norm = np.sqrt(np.sum(template ** 2))
    if t_norm < 1e-10:
        return discharge_samples
    for i in range(half_win, n - half_win):
        seg = evidence[i - half_win:i + half_win + 1]
        if len(seg) != t_len:
            continue
        seg_centered = seg - np.mean(seg)
        s_norm = np.sqrt(np.sum(seg_centered ** 2))
        if s_norm < 1e-10:
            continue
        corr[i] = np.dot(template, seg_centered) / (t_norm * s_norm)
    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(30, int(0.3 * T * fs))
    min_height = 0.15 * np.max(corr) if np.max(corr) > 0 else 0
    refined_candidates, _ = find_peaks(corr, height=min_height, distance=min_dist)
    if len(refined_candidates) < 2:
        return discharge_samples
    return dp_best_sequence_param(refined_candidates, evidence, fs, freq_estimate,
                                  dp_alpha, dp_beta, dp_lambda, max_skip)


# ============================================================================
# Evidence computation + combination
# ============================================================================

def compute_all_evidence(segment_18ch, subtype, laterality, cet_models, fs=FS):
    """Compute both HPP and CET evidence for a segment.

    Returns:
        hpp_evidence: aggregated HPP evidence (n_samples,)
        cet_evidence: aggregated CET evidence (n_samples,) or None
        evidence_all_hpp: per-channel HPP evidence (n_channels, n_samples)
    """
    n_channels = min(segment_18ch.shape[0], 18)
    n_samples = segment_18ch.shape[1]

    # HPP evidence
    hpp_all = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        hpp_all[ch] = _compute_channel_evidence(segment_18ch[ch], fs)
    hpp_agg = _aggregate_evidence(hpp_all, subtype, laterality)

    # CET evidence
    cet_agg = None
    if cet_models is not None:
        cet_all = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            ch_data = segment_18ch[ch]
            if np.all(np.isfinite(ch_data)):
                cet_all[ch] = compute_cet_evidence(ch_data, cet_models, DEVICE)
        cet_agg = _aggregate_evidence(cet_all, subtype, laterality)

    return hpp_agg, cet_agg, hpp_all


def combine_evidence(hpp_evidence, cet_evidence, evidence_type):
    """Combine HPP and CET evidence according to evidence_type."""
    if evidence_type == 'hpp':
        return hpp_evidence
    if evidence_type == 'cet':
        if cet_evidence is None:
            raise ValueError("CET evidence not available")
        return cet_evidence
    if cet_evidence is None:
        return hpp_evidence

    # Normalize both to [0, 1] range for combination
    hpp_max = np.max(hpp_evidence) if np.max(hpp_evidence) > 0 else 1.0
    cet_max = np.max(cet_evidence) if np.max(cet_evidence) > 0 else 1.0
    hpp_norm = hpp_evidence / hpp_max
    cet_norm = cet_evidence / cet_max

    if evidence_type == 'max_combine':
        combined = np.maximum(hpp_norm, cet_norm)
    elif evidence_type == 'mean_combine':
        combined = 0.5 * (hpp_norm + cet_norm)
    elif evidence_type == 'weighted_combine':
        combined = 0.7 * hpp_norm + 0.3 * cet_norm
    else:
        raise ValueError(f"Unknown evidence_type: {evidence_type}")

    return combined


# ============================================================================
# Run HPP with specific parameters on one case
# ============================================================================

def run_hpp_single(evidence, evidence_all_hpp, freq_estimate, fs, params):
    """Run HPP pipeline on precomputed evidence with given parameters."""
    active_start, active_end = _detect_active_interval(evidence, fs)
    candidates = extract_candidates_param(
        evidence, fs, freq_estimate, active_start, active_end,
        params['peak_height_frac'])
    discharge_samples = dp_best_sequence_param(
        candidates, evidence, fs, freq_estimate,
        params['dp_alpha'], params['dp_beta'], params['dp_lambda'],
        params.get('max_skip', 3))
    # EM refinement
    if len(discharge_samples) >= 3:
        discharge_samples = em_refine_param(
            evidence, discharge_samples, fs, freq_estimate,
            params['dp_alpha'], params['dp_beta'], params['dp_lambda'],
            params.get('max_skip', 3))
    global_times = discharge_samples / fs if len(discharge_samples) > 0 else np.array([])
    return global_times


# ============================================================================
# Evaluate a parameter configuration across all GT cases
# ============================================================================

def evaluate_config(precomputed, gt_cases, df, params, evidence_type, tolerance=TOLERANCE_S):
    """Evaluate one parameter configuration on all GT cases.

    Args:
        precomputed: dict pid -> {hpp_evidence, cet_evidence, evidence_all_hpp,
                                   freq_estimate, subtype, laterality, gold_freq}
        gt_cases: dict pid -> {global_times, ...}
        df: dataframe
        params: dict with dp_alpha, dp_beta, dp_lambda, peak_height_frac
        evidence_type: str

    Returns:
        dict with sensitivity, precision, f1, freq_spearman
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    gt_freqs = []
    algo_freqs = []
    n_processed = 0

    for pid, gt_data in gt_cases.items():
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue
        if pid not in precomputed:
            continue

        pc = precomputed[pid]

        try:
            evidence = combine_evidence(pc['hpp_evidence'], pc['cet_evidence'],
                                        evidence_type)
        except ValueError:
            continue

        try:
            algo_times_arr = run_hpp_single(
                evidence, pc['evidence_all_hpp'], pc['freq_estimate'], FS, params)
        except Exception:
            continue

        algo_times = sorted(algo_times_arr.tolist()) if len(algo_times_arr) > 0 else []
        n_processed += 1

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

        # Discharge matching
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

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    if len(gt_freqs) >= 3:
        freq_spearman, _ = spearmanr(algo_freqs, gt_freqs)
    else:
        freq_spearman = float('nan')

    return {
        'n_cases': n_processed,
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'freq_spearman': round(freq_spearman, 4) if np.isfinite(freq_spearman) else None,
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'evidence_type': evidence_type,
        **{k: v for k, v in params.items()},
    }


# ============================================================================
# Print results table
# ============================================================================

def print_results_table(results, title, highlight_cols=None):
    """Print a nicely formatted results table sorted by F1."""
    results_sorted = sorted(results, key=lambda r: r.get('f1', 0), reverse=True)

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    header = (f"{'#':>3s} {'Evidence':<17s} {'Alpha':>6s} {'Lambda':>7s} "
              f"{'Beta':>5s} {'PkHt':>5s} {'Sens':>6s} {'Prec':>6s} "
              f"{'F1':>6s} {'FrqRho':>7s} {'N':>4s}")
    print(f"\n{header}")
    print("-" * len(header))

    for i, r in enumerate(results_sorted):
        f1_str = f"{r['f1']:.4f}" if r['f1'] is not None else "  N/A"
        rho_str = f"{r['freq_spearman']:.4f}" if r.get('freq_spearman') is not None else "  N/A"
        marker = " <-- BEST" if i == 0 else ""
        line = (f"{i+1:>3d} {r['evidence_type']:<17s} "
                f"{r['dp_alpha']:>6.1f} {r['dp_lambda']:>7.3f} "
                f"{r['dp_beta']:>5.1f} {r['peak_height_frac']:>5.2f} "
                f"{r['sensitivity']:>6.4f} {r['precision']:>6.4f} "
                f"{f1_str:>6s} {rho_str:>7s} {r['n_cases']:>4d}{marker}")
        print(line)

    print()
    return results_sorted


# ============================================================================
# Main sweep
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 90)
    print("  HPP Parameter Sweep — FAIR (EEG-only) Setting")
    print("  CNN-estimated frequency as period prior")
    print("=" * 90)
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
    print("\nLoading CET-UNet models...")
    try:
        cet_models = load_cet_unet_models(device=DEVICE)
        print(f"  Loaded {len(cet_models)} CET-UNet fold models")
    except FileNotFoundError as e:
        print(f"  CET-UNet models not found: {e}")
        cet_models = None

    print("Loading CNN+Attention models...")
    try:
        cnn_models = load_cnn_attn_models(device=DEVICE)
        print(f"  Loaded {len(cnn_models)} CNN+Attention fold models")
    except FileNotFoundError as e:
        print(f"  CNN+Attention models not found: {e}")
        cnn_models = None

    if cnn_models is None:
        print("\nERROR: CNN+Attention models required for frequency estimation.")
        return

    # ====================================================================
    # Precompute evidence and CNN frequency for all GT cases
    # ====================================================================
    print(f"\nPrecomputing evidence and CNN frequencies for {len(gt_cases)} GT cases...")
    precomputed = {}
    n_skip = 0
    t_pre = time.time()

    for idx, (pid, gt_data) in enumerate(gt_cases.items()):
        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            n_skip += 1
            continue
        row = row.iloc[0]

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            n_skip += 1
            continue

        seg = pat_segs[0]
        subtype = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        try:
            freq_est = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)
            hpp_ev, cet_ev, hpp_all = compute_all_evidence(
                seg, subtype, lat, cet_models, FS)
        except Exception as e:
            n_skip += 1
            continue

        precomputed[pid] = {
            'hpp_evidence': hpp_ev,
            'cet_evidence': cet_ev,
            'evidence_all_hpp': hpp_all,
            'freq_estimate': freq_est,
            'subtype': subtype,
            'laterality': lat,
            'gold_freq': float(row['gold_standard_freq']),
        }

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_pre
            print(f"  {idx+1}/{len(gt_cases)} precomputed ({elapsed:.0f}s)")

    elapsed_pre = time.time() - t_pre
    print(f"  Precomputed {len(precomputed)} cases, skipped {n_skip} ({elapsed_pre:.0f}s)")

    all_results = {}

    # ====================================================================
    # STAGE 1: Evidence type comparison (current best params)
    # ====================================================================
    print(f"\n{'#' * 90}")
    print(f"  STAGE 1: Evidence Type Comparison")
    print(f"  (alpha={DEFAULT_PARAMS['dp_alpha']}, lambda={DEFAULT_PARAMS['dp_lambda']}, "
          f"beta={DEFAULT_PARAMS['dp_beta']}, peak={DEFAULT_PARAMS['peak_height_frac']})")
    print(f"{'#' * 90}")

    evidence_types = ['hpp', 'cet', 'max_combine', 'mean_combine', 'weighted_combine']
    stage1_results = []

    for ev_type in evidence_types:
        if ev_type in ('cet', 'max_combine', 'mean_combine', 'weighted_combine') and cet_models is None:
            print(f"  Skipping {ev_type} (no CET models)")
            continue

        t1 = time.time()
        r = evaluate_config(precomputed, gt_cases, df, DEFAULT_PARAMS, ev_type)
        elapsed = time.time() - t1
        print(f"  {ev_type:<17s} F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  FrqRho={r.get('freq_spearman', 'N/A')}  ({elapsed:.1f}s)")
        stage1_results.append(r)

    stage1_sorted = print_results_table(stage1_results, "STAGE 1: Evidence Type Comparison")
    best_evidence = stage1_sorted[0]['evidence_type']
    best_f1_stage1 = stage1_sorted[0]['f1']
    print(f"  Best evidence type: {best_evidence} (F1={best_f1_stage1:.4f})")

    all_results['stage1'] = stage1_sorted

    # ====================================================================
    # STAGE 2: DP Parameter Sweep (best evidence, one param at a time)
    # ====================================================================
    print(f"\n{'#' * 90}")
    print(f"  STAGE 2: DP Parameter Sweep (evidence={best_evidence})")
    print(f"  Sweeping each parameter independently")
    print(f"{'#' * 90}")

    alpha_vals = [1.5, 2.0, 3.0, 5.0, 8.0]
    lambda_vals = [0.005, 0.01, 0.02, 0.05, 0.1]
    beta_vals = [0.3, 0.5, 1.0, 2.0, 3.0]
    peak_vals = [0.02, 0.03, 0.05, 0.08, 0.10]

    stage2_results = []
    best_per_param = {}

    # Sweep alpha
    print(f"\n  --- Sweeping DP_ALPHA ---")
    best_alpha_f1 = -1
    for val in alpha_vals:
        params = dict(DEFAULT_PARAMS)
        params['dp_alpha'] = val
        t1 = time.time()
        r = evaluate_config(precomputed, gt_cases, df, params, best_evidence)
        elapsed = time.time() - t1
        print(f"    alpha={val:>5.1f}  F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  ({elapsed:.1f}s)")
        stage2_results.append(r)
        if r['f1'] > best_alpha_f1:
            best_alpha_f1 = r['f1']
            best_per_param['dp_alpha'] = val

    # Sweep lambda
    print(f"\n  --- Sweeping DP_LAMBDA ---")
    best_lambda_f1 = -1
    for val in lambda_vals:
        params = dict(DEFAULT_PARAMS)
        params['dp_lambda'] = val
        t1 = time.time()
        r = evaluate_config(precomputed, gt_cases, df, params, best_evidence)
        elapsed = time.time() - t1
        print(f"    lambda={val:>6.3f}  F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  ({elapsed:.1f}s)")
        stage2_results.append(r)
        if r['f1'] > best_lambda_f1:
            best_lambda_f1 = r['f1']
            best_per_param['dp_lambda'] = val

    # Sweep beta
    print(f"\n  --- Sweeping DP_BETA ---")
    best_beta_f1 = -1
    for val in beta_vals:
        params = dict(DEFAULT_PARAMS)
        params['dp_beta'] = val
        t1 = time.time()
        r = evaluate_config(precomputed, gt_cases, df, params, best_evidence)
        elapsed = time.time() - t1
        print(f"    beta={val:>5.1f}  F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  ({elapsed:.1f}s)")
        stage2_results.append(r)
        if r['f1'] > best_beta_f1:
            best_beta_f1 = r['f1']
            best_per_param['dp_beta'] = val

    # Sweep peak_height_frac
    print(f"\n  --- Sweeping PEAK_HEIGHT_FRAC ---")
    best_peak_f1 = -1
    for val in peak_vals:
        params = dict(DEFAULT_PARAMS)
        params['peak_height_frac'] = val
        t1 = time.time()
        r = evaluate_config(precomputed, gt_cases, df, params, best_evidence)
        elapsed = time.time() - t1
        print(f"    peak={val:>5.2f}  F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  ({elapsed:.1f}s)")
        stage2_results.append(r)
        if r['f1'] > best_peak_f1:
            best_peak_f1 = r['f1']
            best_per_param['peak_height_frac'] = val

    stage2_sorted = print_results_table(stage2_results, "STAGE 2: DP Parameter Sweep (all)")

    print(f"  Best per parameter:")
    for k, v in best_per_param.items():
        print(f"    {k}: {v}")

    all_results['stage2'] = stage2_sorted

    # ====================================================================
    # STAGE 3: Fine-tune best combination
    # ====================================================================
    print(f"\n{'#' * 90}")
    print(f"  STAGE 3: Fine-tune around best parameters")
    print(f"{'#' * 90}")

    # Build candidate param set from top 3 of stage 2 + the "all best" combination
    top3_params = []
    seen = set()
    for r in stage2_sorted[:5]:
        key = (r['dp_alpha'], r['dp_lambda'], r['dp_beta'], r['peak_height_frac'])
        if key not in seen:
            seen.add(key)
            top3_params.append({
                'dp_alpha': r['dp_alpha'],
                'dp_lambda': r['dp_lambda'],
                'dp_beta': r['dp_beta'],
                'peak_height_frac': r['peak_height_frac'],
                'max_skip': 3,
            })
            if len(top3_params) >= 3:
                break

    # Also add the "best from each dimension" combination
    all_best_params = {
        'dp_alpha': best_per_param['dp_alpha'],
        'dp_lambda': best_per_param['dp_lambda'],
        'dp_beta': best_per_param['dp_beta'],
        'peak_height_frac': best_per_param['peak_height_frac'],
        'max_skip': 3,
    }
    key = (all_best_params['dp_alpha'], all_best_params['dp_lambda'],
           all_best_params['dp_beta'], all_best_params['peak_height_frac'])
    if key not in seen:
        top3_params.append(all_best_params)

    print(f"\n  Base configurations to fine-tune: {len(top3_params)}")
    for i, p in enumerate(top3_params):
        print(f"    Config {i+1}: alpha={p['dp_alpha']}, lambda={p['dp_lambda']}, "
              f"beta={p['dp_beta']}, peak={p['peak_height_frac']}")

    # Local grid: vary each param by small amounts around each base
    stage3_results = []
    stage3_seen = set()

    for base_params in top3_params:
        # Generate local perturbations
        alpha_local = [base_params['dp_alpha'] * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]
        lambda_local = [base_params['dp_lambda'] * f for f in [0.5, 0.75, 1.0, 1.5, 2.0]]
        beta_local = [base_params['dp_beta'] * f for f in [0.7, 0.85, 1.0, 1.15, 1.3]]
        peak_local = [base_params['peak_height_frac'] * f for f in [0.6, 0.8, 1.0, 1.25, 1.5]]

        # Test each perturbation individually (not full grid)
        for alpha in alpha_local:
            p = dict(base_params)
            p['dp_alpha'] = round(alpha, 3)
            key = (p['dp_alpha'], p['dp_lambda'], p['dp_beta'], p['peak_height_frac'])
            if key in stage3_seen:
                continue
            stage3_seen.add(key)
            r = evaluate_config(precomputed, gt_cases, df, p, best_evidence)
            stage3_results.append(r)

        for lam in lambda_local:
            p = dict(base_params)
            p['dp_lambda'] = round(lam, 4)
            key = (p['dp_alpha'], p['dp_lambda'], p['dp_beta'], p['peak_height_frac'])
            if key in stage3_seen:
                continue
            stage3_seen.add(key)
            r = evaluate_config(precomputed, gt_cases, df, p, best_evidence)
            stage3_results.append(r)

        for beta in beta_local:
            p = dict(base_params)
            p['dp_beta'] = round(beta, 3)
            key = (p['dp_alpha'], p['dp_lambda'], p['dp_beta'], p['peak_height_frac'])
            if key in stage3_seen:
                continue
            stage3_seen.add(key)
            r = evaluate_config(precomputed, gt_cases, df, p, best_evidence)
            stage3_results.append(r)

        for peak in peak_local:
            p = dict(base_params)
            p['peak_height_frac'] = round(peak, 4)
            key = (p['dp_alpha'], p['dp_lambda'], p['dp_beta'], p['peak_height_frac'])
            if key in stage3_seen:
                continue
            stage3_seen.add(key)
            r = evaluate_config(precomputed, gt_cases, df, p, best_evidence)
            stage3_results.append(r)

    # Also test best combination with all evidence types
    print(f"\n  Testing best params with all evidence types...")
    stage3_sorted_temp = sorted(stage3_results, key=lambda r: r.get('f1', 0), reverse=True)
    best_stage3_params = {
        'dp_alpha': stage3_sorted_temp[0]['dp_alpha'],
        'dp_lambda': stage3_sorted_temp[0]['dp_lambda'],
        'dp_beta': stage3_sorted_temp[0]['dp_beta'],
        'peak_height_frac': stage3_sorted_temp[0]['peak_height_frac'],
        'max_skip': 3,
    }
    for ev_type in evidence_types:
        if ev_type in ('cet', 'max_combine', 'mean_combine', 'weighted_combine') and cet_models is None:
            continue
        r = evaluate_config(precomputed, gt_cases, df, best_stage3_params, ev_type)
        # Tag it differently so we can see evidence type effect
        r_tagged = dict(r)
        stage3_results.append(r_tagged)

    print(f"\n  Stage 3: tested {len(stage3_results)} configurations")
    stage3_sorted = print_results_table(stage3_results,
                                         "STAGE 3: Fine-tuned Results (top 20)",
                                         )

    all_results['stage3'] = stage3_sorted[:20]  # save top 20

    # ====================================================================
    # Final summary
    # ====================================================================
    overall_best = stage3_sorted[0]
    print(f"\n{'=' * 90}")
    print(f"  FINAL BEST CONFIGURATION")
    print(f"{'=' * 90}")
    print(f"  Evidence type:     {overall_best['evidence_type']}")
    print(f"  DP_ALPHA:          {overall_best['dp_alpha']}")
    print(f"  DP_LAMBDA:         {overall_best['dp_lambda']}")
    print(f"  DP_BETA:           {overall_best['dp_beta']}")
    print(f"  PEAK_HEIGHT_FRAC:  {overall_best['peak_height_frac']}")
    print(f"  ---")
    print(f"  Sensitivity:       {overall_best['sensitivity']:.4f}")
    print(f"  Precision:         {overall_best['precision']:.4f}")
    print(f"  F1:                {overall_best['f1']:.4f}")
    print(f"  Freq Spearman:     {overall_best.get('freq_spearman', 'N/A')}")
    print(f"  Cases:             {overall_best['n_cases']}")
    print(f"  TP={overall_best['tp']}, FN={overall_best['fn']}, FP={overall_best['fp']}")

    # Compare to default
    default_r = stage1_sorted[0]  # best from stage 1 (might be different evidence)
    # Find default hpp result
    for r in stage1_sorted:
        if r['evidence_type'] == 'hpp':
            default_hpp = r
            break
    else:
        default_hpp = stage1_sorted[-1]

    print(f"\n  Improvement over default HPP:")
    print(f"    F1:    {default_hpp['f1']:.4f} -> {overall_best['f1']:.4f} "
          f"({overall_best['f1'] - default_hpp['f1']:+.4f})")
    print(f"    Sens:  {default_hpp['sensitivity']:.4f} -> {overall_best['sensitivity']:.4f}")
    print(f"    Prec:  {default_hpp['precision']:.4f} -> {overall_best['precision']:.4f}")

    # ====================================================================
    # Save results
    # ====================================================================
    output_path = CACHE_DIR / 'parameter_sweep_results.json'

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    save_data = {
        'best_config': {
            'evidence_type': overall_best['evidence_type'],
            'dp_alpha': overall_best['dp_alpha'],
            'dp_lambda': overall_best['dp_lambda'],
            'dp_beta': overall_best['dp_beta'],
            'peak_height_frac': overall_best['peak_height_frac'],
            'f1': overall_best['f1'],
            'sensitivity': overall_best['sensitivity'],
            'precision': overall_best['precision'],
            'freq_spearman': overall_best.get('freq_spearman'),
        },
        'default_hpp': {
            'f1': default_hpp['f1'],
            'sensitivity': default_hpp['sensitivity'],
            'precision': default_hpp['precision'],
        },
        'stage1': all_results['stage1'],
        'stage2': all_results['stage2'][:20],
        'stage3': all_results['stage3'],
    }

    with open(str(output_path), 'w') as f:
        json.dump(save_data, f, indent=2, default=json_default)
    print(f"\n  Results saved to {output_path}")

    elapsed_total = time.time() - t0
    print(f"\n  Total sweep time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'=' * 90}")


if __name__ == '__main__':
    main()
