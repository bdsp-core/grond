"""
Error pattern analysis for max(HPP,CET)+CNN_freq+optimized_DP method.

Analyzes per-case TP/FP/FN breakdown, frequency estimation quality,
subtype performance, and comparison between HPP-only and max_combine.

Usage:
    conda run -n foe_dl python code/analyze_error_patterns.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr, pearsonr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.auto_pipeline import (
    load_cet_unet_models, load_cnn_attn_models,
    estimate_frequency_cnn, compute_cet_evidence,
    _aggregate_evidence, _detect_active_interval,
    _per_channel_times,
    DEVICE, TOLERANCE_S,
)
from cet_model.parameter_sweep import (
    compute_all_evidence, combine_evidence, run_hpp_single,
)
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS

# ── Best configuration from parameter sweep ──────────────────────────────────
BEST_PARAMS = {
    'dp_alpha': 1.275,
    'dp_lambda': 0.05,
    'dp_beta': 0.3,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}
BEST_EVIDENCE_TYPE = 'max_combine'

# HPP-only baseline (default params from parameter_sweep.py)
HPP_PARAMS = {
    'dp_alpha': 3.0,
    'dp_beta': 1.0,
    'dp_lambda': 0.02,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}


def match_discharges(gt_times, algo_times, tolerance=TOLERANCE_S):
    """Match GT and algo discharges, return per-GT and per-algo match status."""
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
    return gt_matched, algo_matched


def per_case_f1(tp, fn, fp):
    """Compute F1 from per-case tp/fn/fp."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def compute_noise_floor(evidence, active_start, active_end):
    """Estimate noise floor as median of evidence outside active region."""
    n = len(evidence)
    outside = np.concatenate([evidence[:active_start], evidence[active_end + 1:]])
    if len(outside) < 10:
        outside = evidence
    return float(np.median(outside))


def compute_evidence_snr(evidence, active_start, active_end):
    """Signal (max in active region) vs noise (median outside) ratio."""
    active = evidence[active_start:active_end + 1]
    if len(active) == 0:
        return 0.0
    signal_peak = float(np.max(active))
    noise = compute_noise_floor(evidence, active_start, active_end)
    if noise < 1e-10:
        return float('inf') if signal_peak > 0 else 0.0
    return signal_peak / noise


def estimate_noise_floor_ratio(evidence, active_start, active_end):
    """Fraction of active interval where evidence is near the noise floor."""
    active = evidence[active_start:active_end + 1]
    if len(active) == 0:
        return 0.0
    noise = compute_noise_floor(evidence, active_start, active_end)
    threshold = max(noise * 1.5, np.max(active) * 0.05)
    return float(np.mean(active <= threshold))


def print_section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def print_subsection(title):
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")


def main():
    t0 = time.time()
    print("=" * 80)
    print("  Error Pattern Analysis: max(HPP,CET)+CNN_freq+optimized_DP")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}")
    print(f"\nBest method params:")
    print(f"  Evidence:        {BEST_EVIDENCE_TYPE}")
    print(f"  dp_alpha:        {BEST_PARAMS['dp_alpha']}")
    print(f"  dp_beta:         {BEST_PARAMS['dp_beta']}")
    print(f"  dp_lambda:       {BEST_PARAMS['dp_lambda']}")
    print(f"  peak_height_frac:{BEST_PARAMS['peak_height_frac']}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # ── Load GT ───────────────────────────────────────────────────────────────
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading CET-UNet models...")
    try:
        cet_models = load_cet_unet_models(device=DEVICE)
        print(f"  Loaded {len(cet_models)} CET-UNet fold models")
    except FileNotFoundError as e:
        print(f"  ERROR: CET-UNet models not found: {e}")
        return

    print("Loading CNN+Attention models (for frequency estimation)...")
    try:
        cnn_models = load_cnn_attn_models(device=DEVICE)
        print(f"  Loaded {len(cnn_models)} CNN+Attention fold models")
    except FileNotFoundError as e:
        print(f"  ERROR: CNN+Attention models not found: {e}")
        return

    # ── Precompute evidence + CNN freq for all GT cases ───────────────────────
    print(f"\nPrecomputing evidence and CNN frequencies for {len(gt_cases)} GT cases...")
    print("  (This may take several minutes...)")
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

        gold_freq = float(gt_data.get('gold_standard_freq', row.get('gold_standard_freq', np.nan)))

        precomputed[pid] = {
            'hpp_evidence': hpp_ev,
            'cet_evidence': cet_ev,
            'evidence_all_hpp': hpp_all,
            'freq_estimate': freq_est,
            'subtype': subtype,
            'laterality': lat,
            'gold_freq': gold_freq,
        }

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_pre
            print(f"  {idx+1}/{len(gt_cases)} ({elapsed:.0f}s)")

    elapsed_pre = time.time() - t_pre
    print(f"  Precomputed {len(precomputed)} cases, skipped {n_skip} ({elapsed_pre:.0f}s)")

    # ── Run both methods on all cases, collect per-case stats ─────────────────
    print("\nRunning best method (max_combine + optimized DP) and HPP-only baseline...")
    per_case = {}   # pid -> dict of per-case metrics

    for pid, gt_data in gt_cases.items():
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue
        if pid not in precomputed:
            continue

        pc = precomputed[pid]
        subtype = pc['subtype']
        freq_est = pc['freq_estimate']
        gold_freq = pc['gold_freq']
        freq_error = abs(freq_est - gold_freq) if np.isfinite(gold_freq) else np.nan

        # Gold freq from GT discharge times
        gt_ipis = np.diff(sorted(gt_times))
        gt_freq_ipi = float(1.0 / np.median(gt_ipis)) if len(gt_ipis) > 0 else np.nan

        # ── max_combine method ──
        try:
            ev_max = combine_evidence(pc['hpp_evidence'], pc['cet_evidence'], 'max_combine')
            times_max_arr = run_hpp_single(
                ev_max, pc['evidence_all_hpp'], freq_est, FS, BEST_PARAMS)
            times_max = sorted(times_max_arr.tolist()) if len(times_max_arr) > 0 else []
        except Exception:
            times_max = []

        gt_m, algo_m = match_discharges(gt_times, times_max)
        tp_max = sum(gt_m)
        fn_max = len(gt_times) - tp_max
        fp_max = len(times_max) - sum(algo_m)
        f1_max = per_case_f1(tp_max, fn_max, fp_max)

        # ── HPP-only method (HPP params, HPP evidence) ──
        try:
            ev_hpp = combine_evidence(pc['hpp_evidence'], pc['cet_evidence'], 'hpp')
            times_hpp_arr = run_hpp_single(
                ev_hpp, pc['evidence_all_hpp'], freq_est, FS, HPP_PARAMS)
            times_hpp = sorted(times_hpp_arr.tolist()) if len(times_hpp_arr) > 0 else []
        except Exception:
            times_hpp = []

        gt_m2, algo_m2 = match_discharges(gt_times, times_hpp)
        tp_hpp = sum(gt_m2)
        fn_hpp = len(gt_times) - tp_hpp
        fp_hpp = len(times_hpp) - sum(algo_m2)
        f1_hpp = per_case_f1(tp_hpp, fn_hpp, fp_hpp)

        # ── Evidence quality metrics ──
        active_s, active_e = _detect_active_interval(ev_max, FS)
        noise_floor = compute_noise_floor(ev_max, active_s, active_e)
        snr = compute_evidence_snr(ev_max, active_s, active_e)
        noise_floor_ratio = estimate_noise_floor_ratio(ev_max, active_s, active_e)

        # HPP evidence SNR
        ev_hpp_only = pc['hpp_evidence']
        active_s_h, active_e_h = _detect_active_interval(ev_hpp_only, FS)
        snr_hpp = compute_evidence_snr(ev_hpp_only, active_s_h, active_e_h)

        # IPI CV (regularity from algo)
        if len(times_max) >= 2:
            ipis = np.diff(sorted(times_max))
            algo_ipi_cv = float(np.std(ipis) / np.mean(ipis)) if np.mean(ipis) > 0 else np.nan
            algo_freq_from_ipi = float(1.0 / np.median(ipis))
        else:
            algo_ipi_cv = np.nan
            algo_freq_from_ipi = np.nan

        per_case[pid] = {
            'subtype': subtype,
            'laterality': pc['laterality'],
            'gold_freq': gold_freq,
            'gt_freq_ipi': gt_freq_ipi,
            'cnn_freq': freq_est,
            'freq_error': freq_error,
            # max_combine results
            'tp_max': tp_max,
            'fn_max': fn_max,
            'fp_max': fp_max,
            'f1_max': f1_max,
            'n_gt': len(gt_times),
            'n_algo_max': len(times_max),
            # HPP-only results
            'tp_hpp': tp_hpp,
            'fn_hpp': fn_hpp,
            'fp_hpp': fp_hpp,
            'f1_hpp': f1_hpp,
            'n_algo_hpp': len(times_hpp),
            # Evidence quality
            'snr_max': snr,
            'snr_hpp': snr_hpp,
            'noise_floor_ratio': noise_floor_ratio,
            'algo_ipi_cv': algo_ipi_cv,
            'algo_freq_from_ipi': algo_freq_from_ipi,
        }

    print(f"  Per-case data collected: {len(per_case)} cases")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    cases = list(per_case.values())
    n_cases = len(cases)

    print_section("OVERALL AGGREGATE PERFORMANCE")

    total_tp = sum(c['tp_max'] for c in cases)
    total_fn = sum(c['fn_max'] for c in cases)
    total_fp = sum(c['fp_max'] for c in cases)
    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1_overall = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    print(f"\n  max(HPP,CET)+CNN_freq+optimized_DP")
    print(f"    Cases:       {n_cases}")
    print(f"    TP:          {total_tp}")
    print(f"    FN:          {total_fn}")
    print(f"    FP:          {total_fp}")
    print(f"    Sensitivity: {sens:.4f}")
    print(f"    Precision:   {prec:.4f}")
    print(f"    F1:          {f1_overall:.4f}")

    # Freq Spearman
    valid = [(c['cnn_freq'], c['gt_freq_ipi']) for c in cases
             if np.isfinite(c['cnn_freq']) and np.isfinite(c['gt_freq_ipi'])]
    if valid:
        cnn_fs, gt_fs = zip(*valid)
        rho, _ = spearmanr(cnn_fs, gt_fs)
        print(f"    CNN freq Spearman vs GT IPI freq: {rho:.4f} (n={len(valid)})")

    # HPP-only baseline
    total_tp_h = sum(c['tp_hpp'] for c in cases)
    total_fn_h = sum(c['fn_hpp'] for c in cases)
    total_fp_h = sum(c['fp_hpp'] for c in cases)
    sens_h = total_tp_h / (total_tp_h + total_fn_h) if (total_tp_h + total_fn_h) > 0 else 0
    prec_h = total_tp_h / (total_tp_h + total_fp_h) if (total_tp_h + total_fp_h) > 0 else 0
    f1_h = 2 * prec_h * sens_h / (prec_h + sens_h) if (prec_h + sens_h) > 0 else 0

    print(f"\n  HPP-only (HPP evidence, CNN freq, HPP params)")
    print(f"    Cases:       {n_cases}")
    print(f"    TP:          {total_tp_h}")
    print(f"    FN:          {total_fn_h}")
    print(f"    FP:          {total_fp_h}")
    print(f"    Sensitivity: {sens_h:.4f}")
    print(f"    Precision:   {prec_h:.4f}")
    print(f"    F1:          {f1_h:.4f}")

    print(f"\n  Delta (max_combine - HPP-only):")
    print(f"    F1:          {f1_overall - f1_h:+.4f}")
    print(f"    Sensitivity: {sens - sens_h:+.4f}")
    print(f"    Precision:   {prec - prec_h:+.4f}")

    # ── ANALYSIS A: FPs from bad freq estimation ──────────────────────────────
    print_section("ANALYSIS A: FPs from Bad Frequency Estimation")

    freq_error_thresh = 0.3  # Hz
    fp_cases = [c for c in cases if c['fp_max'] > 0]
    bad_freq_fp = [c for c in fp_cases
                   if np.isfinite(c['freq_error']) and c['freq_error'] > freq_error_thresh]
    good_freq_fp = [c for c in fp_cases
                    if np.isfinite(c['freq_error']) and c['freq_error'] <= freq_error_thresh]

    print(f"\n  Cases with any FP:            {len(fp_cases)} / {n_cases}")
    total_fp_in_bad_freq = sum(c['fp_max'] for c in bad_freq_fp)
    total_fp_in_good_freq = sum(c['fp_max'] for c in good_freq_fp)
    print(f"  FP cases with freq error > {freq_error_thresh}Hz: {len(bad_freq_fp)} "
          f"({100*len(bad_freq_fp)/max(1,len(fp_cases)):.1f}% of FP cases)")
    print(f"  FP cases with freq error <= {freq_error_thresh}Hz: {len(good_freq_fp)} "
          f"({100*len(good_freq_fp)/max(1,len(fp_cases)):.1f}% of FP cases)")
    print(f"  Total FPs in bad-freq cases:  {total_fp_in_bad_freq} "
          f"({100*total_fp_in_bad_freq/max(1,total_fp):.1f}% of all FPs)")
    print(f"  Total FPs in good-freq cases: {total_fp_in_good_freq} "
          f"({100*total_fp_in_good_freq/max(1,total_fp):.1f}% of all FPs)")

    # Mean FP per case by freq error bucket
    buckets = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0), (1.0, 5.0)]
    print(f"\n  FPs by CNN freq error bucket:")
    print(f"    {'Freq Err Range':<20s} {'N Cases':>8s} {'Mean FP':>9s} {'Median FP':>10s} {'Mean F1':>8s}")
    for lo, hi in buckets:
        bucket_cases = [c for c in cases
                        if np.isfinite(c['freq_error']) and lo <= c['freq_error'] < hi]
        if not bucket_cases:
            continue
        mean_fp = np.mean([c['fp_max'] for c in bucket_cases])
        med_fp = np.median([c['fp_max'] for c in bucket_cases])
        mean_f1 = np.mean([c['f1_max'] for c in bucket_cases])
        print(f"    [{lo:.1f}, {hi:.1f}) Hz           {len(bucket_cases):>8d} {mean_fp:>9.2f} {med_fp:>10.2f} {mean_f1:>8.3f}")

    # ── ANALYSIS B: FPs from noise floor vs genuine extra peaks ───────────────
    print_section("ANALYSIS B: FPs — Noise Floor vs Genuine Extra Peaks")

    # Characterize FP cases by noise floor ratio
    noise_thresh = 0.5  # >50% of active interval near noise floor = "noisy"
    fp_high_noise = [c for c in fp_cases if c['noise_floor_ratio'] > noise_thresh]
    fp_low_noise = [c for c in fp_cases if c['noise_floor_ratio'] <= noise_thresh]

    print(f"\n  Cases with FPs, high noise floor (>{noise_thresh*100:.0f}% near floor): "
          f"{len(fp_high_noise)} ({100*len(fp_high_noise)/max(1,len(fp_cases)):.1f}%)")
    print(f"  Cases with FPs, low noise floor (low noise, genuine peaks):  "
          f"{len(fp_low_noise)} ({100*len(fp_low_noise)/max(1,len(fp_cases)):.1f}%)")

    if fp_high_noise:
        print(f"\n  High-noise FP cases summary:")
        print(f"    Mean FP per case:   {np.mean([c['fp_max'] for c in fp_high_noise]):.2f}")
        print(f"    Mean SNR:           {np.mean([c['snr_max'] for c in fp_high_noise if np.isfinite(c['snr_max'])]):.2f}")
        print(f"    Mean freq error:    {np.mean([c['freq_error'] for c in fp_high_noise if np.isfinite(c['freq_error'])]):.3f} Hz")
        print(f"    Mean F1:            {np.mean([c['f1_max'] for c in fp_high_noise]):.3f}")

    if fp_low_noise:
        print(f"\n  Low-noise FP cases summary:")
        print(f"    Mean FP per case:   {np.mean([c['fp_max'] for c in fp_low_noise]):.2f}")
        print(f"    Mean SNR:           {np.mean([c['snr_max'] for c in fp_low_noise if np.isfinite(c['snr_max'])]):.2f}")
        print(f"    Mean freq error:    {np.mean([c['freq_error'] for c in fp_low_noise if np.isfinite(c['freq_error'])]):.3f} Hz")
        print(f"    Mean F1:            {np.mean([c['f1_max'] for c in fp_low_noise]):.3f}")

    # SNR distribution for FP vs non-FP cases
    fp_snrs = [c['snr_max'] for c in fp_cases if np.isfinite(c['snr_max'])]
    no_fp_snrs = [c['snr_max'] for c in cases if c['fp_max'] == 0 and np.isfinite(c['snr_max'])]
    if fp_snrs and no_fp_snrs:
        print(f"\n  Evidence SNR comparison (max_combine):")
        print(f"    FP cases     — mean={np.mean(fp_snrs):.2f}, median={np.median(fp_snrs):.2f}, n={len(fp_snrs)}")
        print(f"    No-FP cases  — mean={np.mean(no_fp_snrs):.2f}, median={np.median(no_fp_snrs):.2f}, n={len(no_fp_snrs)}")

    # ── ANALYSIS C: Frequency error vs F1 per case ────────────────────────────
    print_section("ANALYSIS C: Relationship Between Frequency Error and Per-Case F1")

    valid_pairs = [(c['freq_error'], c['f1_max'])
                   for c in cases if np.isfinite(c['freq_error'])]
    if valid_pairs:
        ferrs, f1s = zip(*valid_pairs)
        rho_fe_f1, p_fe_f1 = spearmanr(ferrs, f1s)
        print(f"\n  Spearman rho(freq_error, F1) = {rho_fe_f1:.4f}  (p={p_fe_f1:.4f}, n={len(valid_pairs)})")

    # Quartile breakdown
    ferrs_arr = np.array([c['freq_error'] for c in cases if np.isfinite(c['freq_error'])])
    q1, q2, q3 = np.percentile(ferrs_arr, [25, 50, 75])
    print(f"\n  CNN freq error quartiles: Q1={q1:.3f}, Q2={q2:.3f}, Q3={q3:.3f} Hz")

    buckets_fe = [
        ('Very low  (<0.05 Hz)', lambda c: np.isfinite(c['freq_error']) and c['freq_error'] < 0.05),
        ('Low       (0.05-0.15)', lambda c: np.isfinite(c['freq_error']) and 0.05 <= c['freq_error'] < 0.15),
        ('Moderate  (0.15-0.30)', lambda c: np.isfinite(c['freq_error']) and 0.15 <= c['freq_error'] < 0.30),
        ('High      (0.30-0.60)', lambda c: np.isfinite(c['freq_error']) and 0.30 <= c['freq_error'] < 0.60),
        ('Very high (>0.60 Hz)', lambda c: np.isfinite(c['freq_error']) and c['freq_error'] >= 0.60),
    ]

    print(f"\n  {'Freq Error Bucket':<28s} {'N':>5s} {'Mean F1':>8s} {'Mean TP':>8s} {'Mean FN':>8s} {'Mean FP':>8s}")
    for label, filt in buckets_fe:
        bucket = [c for c in cases if filt(c)]
        if not bucket:
            continue
        m_f1 = np.mean([c['f1_max'] for c in bucket])
        m_tp = np.mean([c['tp_max'] for c in bucket])
        m_fn = np.mean([c['fn_max'] for c in bucket])
        m_fp = np.mean([c['fp_max'] for c in bucket])
        print(f"  {label:<28s} {len(bucket):>5d} {m_f1:>8.3f} {m_tp:>8.2f} {m_fn:>8.2f} {m_fp:>8.2f}")

    # ── ANALYSIS D: F1 distribution — good vs bad cases ──────────────────────
    print_section("ANALYSIS D: F1 Distribution — Good vs Bad Cases")

    f1s_all = [c['f1_max'] for c in cases]
    f1_arr = np.array(f1s_all)

    good_thresh = 0.8
    bad_thresh = 0.5

    good_cases = [c for c in cases if c['f1_max'] >= good_thresh]
    bad_cases = [c for c in cases if c['f1_max'] < bad_thresh]
    mid_cases = [c for c in cases if bad_thresh <= c['f1_max'] < good_thresh]

    print(f"\n  F1 summary (max_combine method):")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        print(f"    P{p:>3d}: {np.percentile(f1_arr, p):.3f}")

    print(f"\n  F1 >= {good_thresh}:  {len(good_cases):4d} ({100*len(good_cases)/n_cases:.1f}%)")
    print(f"  F1 [{bad_thresh},{good_thresh}): {len(mid_cases):4d} ({100*len(mid_cases)/n_cases:.1f}%)")
    print(f"  F1 <  {bad_thresh}:  {len(bad_cases):4d} ({100*len(bad_cases)/n_cases:.1f}%)")
    print(f"  F1 == 0:        {sum(1 for c in cases if c['f1_max'] == 0):4d}")

    print(f"\n  What distinguishes bad cases (F1 < {bad_thresh}):")
    def summary_stats(group, field):
        vals = [c[field] for c in group if np.isfinite(c[field])]
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):.3f}, med={np.median(vals):.3f}"

    fields_to_compare = [
        ('freq_error', 'CNN freq error (Hz)'),
        ('snr_max', 'Evidence SNR'),
        ('snr_hpp', 'HPP evidence SNR'),
        ('noise_floor_ratio', 'Noise floor ratio'),
        ('gold_freq', 'Gold freq (Hz)'),
        ('algo_ipi_cv', 'Algo IPI CV'),
    ]

    print(f"\n  {'Metric':<28s} {'GOOD (F1>0.8)':<28s} {'BAD (F1<0.5)':<28s}")
    print(f"  {'-'*84}")
    for field, label in fields_to_compare:
        g_str = summary_stats(good_cases, field)
        b_str = summary_stats(bad_cases, field)
        print(f"  {label:<28s} {g_str:<28s} {b_str:<28s}")

    # Subtype breakdown for good/bad
    for subtype in ['lpd', 'gpd']:
        g = sum(1 for c in good_cases if c['subtype'] == subtype)
        b = sum(1 for c in bad_cases if c['subtype'] == subtype)
        total_st = sum(1 for c in cases if c['subtype'] == subtype)
        print(f"\n  Subtype={subtype}: good={g} ({100*g/max(1,total_st):.1f}%), "
              f"bad={b} ({100*b/max(1,total_st):.1f}%), total={total_st}")

    # Error types in bad cases
    bad_all_fn = sum(1 for c in bad_cases if c['fp_max'] == 0 and c['fn_max'] > 0)
    bad_all_fp = sum(1 for c in bad_cases if c['fn_max'] == 0 and c['fp_max'] > 0)
    bad_mixed = sum(1 for c in bad_cases if c['fp_max'] > 0 and c['fn_max'] > 0)
    bad_zero = sum(1 for c in bad_cases if c['tp_max'] == 0)
    print(f"\n  Bad case error types:")
    print(f"    Pure FN (fp=0, fn>0):  {bad_all_fn}")
    print(f"    Pure FP (fn=0, fp>0):  {bad_all_fp}")
    print(f"    Mixed FP+FN:           {bad_mixed}")
    print(f"    Zero TP:               {bad_zero}")

    # ── ANALYSIS E: Per-case F1 distribution — HPP-only vs max_combine ────────
    print_section("ANALYSIS E: Per-Case F1 Distribution — HPP-Only vs max(HPP,CET)")

    f1_max_arr = np.array([c['f1_max'] for c in cases])
    f1_hpp_arr = np.array([c['f1_hpp'] for c in cases])
    delta_f1 = f1_max_arr - f1_hpp_arr

    print(f"\n  Per-case F1 statistics:")
    print(f"    {'Metric':<20s} {'HPP-only':>10s} {'max_combine':>12s} {'Delta':>10s}")
    print(f"    {'-'*56}")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        h = np.percentile(f1_hpp_arr, p)
        m = np.percentile(f1_max_arr, p)
        print(f"    P{p:<19d} {h:>10.3f} {m:>12.3f} {m-h:>+10.3f}")
    print(f"    {'Mean':<20s} {np.mean(f1_hpp_arr):>10.3f} {np.mean(f1_max_arr):>12.3f} "
          f"{np.mean(delta_f1):>+10.3f}")

    improved = sum(1 for d in delta_f1 if d > 0.01)
    worsened = sum(1 for d in delta_f1 if d < -0.01)
    unchanged = sum(1 for d in delta_f1 if abs(d) <= 0.01)

    print(f"\n  max_combine vs HPP-only (threshold: 0.01 F1):")
    print(f"    Improved (delta > +0.01):  {improved:4d} ({100*improved/n_cases:.1f}%)")
    print(f"    Unchanged (|delta| <= 0.01): {unchanged:4d} ({100*unchanged/n_cases:.1f}%)")
    print(f"    Worsened (delta < -0.01):  {worsened:4d} ({100*worsened/n_cases:.1f}%)")

    # Does max help uniformly or on specific subgroups?
    print(f"\n  Delta F1 by subtype (max_combine - HPP-only):")
    for subtype in ['lpd', 'gpd']:
        deltas = [c['f1_max'] - c['f1_hpp'] for c in cases if c['subtype'] == subtype]
        if deltas:
            print(f"    {subtype.upper()}: mean={np.mean(deltas):+.4f}, "
                  f"med={np.median(deltas):+.4f}, std={np.std(deltas):.4f}, n={len(deltas)}")

    print(f"\n  Delta F1 by gold freq range:")
    freq_bins = [(0.3, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.5)]
    for lo, hi in freq_bins:
        bin_cases = [c for c in cases
                     if np.isfinite(c['gold_freq']) and lo <= c['gold_freq'] < hi]
        if not bin_cases:
            continue
        deltas = [c['f1_max'] - c['f1_hpp'] for c in bin_cases]
        print(f"    [{lo:.1f},{hi:.1f}) Hz: mean={np.mean(deltas):+.4f}, "
              f"med={np.median(deltas):+.4f}, n={len(bin_cases)}")

    # ── ANALYSIS F: Cases where max(HPP,CET) is WORSE than HPP-only ──────────
    print_section("ANALYSIS F: Cases Where max(HPP,CET) is WORSE than HPP-Only")

    worse_cases = [c for c in cases if (c['f1_max'] - c['f1_hpp']) < -0.05]
    print(f"\n  Cases where max_combine F1 < HPP-only F1 - 0.05: {len(worse_cases)} "
          f"({100*len(worse_cases)/n_cases:.1f}%)")

    if worse_cases:
        print(f"\n  Characterization of worse cases:")
        print(f"    Mean delta F1:       {np.mean([c['f1_max']-c['f1_hpp'] for c in worse_cases]):+.4f}")
        print(f"    Mean max_combine F1: {np.mean([c['f1_max'] for c in worse_cases]):.4f}")
        print(f"    Mean HPP-only F1:    {np.mean([c['f1_hpp'] for c in worse_cases]):.4f}")

        for field, label in fields_to_compare:
            vals = [c[field] for c in worse_cases if np.isfinite(c[field])]
            all_vals = [c[field] for c in cases if np.isfinite(c[field])]
            if vals and all_vals:
                print(f"    {label:<28s} worse={np.mean(vals):.3f} vs all={np.mean(all_vals):.3f}")

        subtype_worse = defaultdict(int)
        subtype_total = defaultdict(int)
        for c in worse_cases:
            subtype_worse[c['subtype']] += 1
        for c in cases:
            subtype_total[c['subtype']] += 1
        print(f"\n  Subtype distribution of worse cases:")
        for st in sorted(subtype_worse.keys()):
            rate = subtype_worse[st] / max(1, subtype_total[st])
            print(f"    {st}: {subtype_worse[st]}/{subtype_total[st]} ({100*rate:.1f}%)")

        # What causes regression? Extra FPs from CET evidence?
        fp_increase = [c for c in worse_cases if c['fp_max'] > c['fp_hpp']]
        fn_increase = [c for c in worse_cases if c['fn_max'] > c['fn_hpp']]
        print(f"\n  Error mode causing regression:")
        print(f"    More FPs in max_combine: {len(fp_increase)} ({100*len(fp_increase)/max(1,len(worse_cases)):.1f}%)")
        print(f"    More FNs in max_combine: {len(fn_increase)} ({100*len(fn_increase)/max(1,len(worse_cases)):.1f}%)")
        both = [c for c in worse_cases if c['fp_max'] > c['fp_hpp'] and c['fn_max'] > c['fn_hpp']]
        print(f"    Both more FP and FN:     {len(both)} ({100*len(both)/max(1,len(worse_cases)):.1f}%)")

        # Are they low-SNR cases where CET adds noise?
        snr_worse = [c['snr_max'] for c in worse_cases if np.isfinite(c['snr_max'])]
        snr_all = [c['snr_max'] for c in cases if np.isfinite(c['snr_max'])]
        print(f"\n  SNR: worse cases mean={np.mean(snr_worse):.2f} vs all cases mean={np.mean(snr_all):.2f}")
        print(f"  Noise floor ratio: worse cases mean={np.mean([c['noise_floor_ratio'] for c in worse_cases]):.3f} "
              f"vs all mean={np.mean([c['noise_floor_ratio'] for c in cases]):.3f}")

    # ── ANALYSIS G: Subtype breakdown — LPD vs GPD ────────────────────────────
    print_section("ANALYSIS G: Subtype Breakdown — LPD vs GPD")

    for subtype in ['lpd', 'gpd']:
        st_cases = [c for c in cases if c['subtype'] == subtype]
        if not st_cases:
            continue

        tp_st = sum(c['tp_max'] for c in st_cases)
        fn_st = sum(c['fn_max'] for c in st_cases)
        fp_st = sum(c['fp_max'] for c in st_cases)
        sens_st = tp_st / (tp_st + fn_st) if (tp_st + fn_st) > 0 else 0
        prec_st = tp_st / (tp_st + fp_st) if (tp_st + fp_st) > 0 else 0
        f1_st = 2 * prec_st * sens_st / (prec_st + sens_st) if (prec_st + sens_st) > 0 else 0

        tp_st_h = sum(c['tp_hpp'] for c in st_cases)
        fn_st_h = sum(c['fn_hpp'] for c in st_cases)
        fp_st_h = sum(c['fp_hpp'] for c in st_cases)
        sens_st_h = tp_st_h / (tp_st_h + fn_st_h) if (tp_st_h + fn_st_h) > 0 else 0
        prec_st_h = tp_st_h / (tp_st_h + fp_st_h) if (tp_st_h + fp_st_h) > 0 else 0
        f1_st_h = 2 * prec_st_h * sens_st_h / (prec_st_h + sens_st_h) if (prec_st_h + sens_st_h) > 0 else 0

        print(f"\n  {subtype.upper()} (n={len(st_cases)} cases):")
        print(f"    {'Method':<25s} {'Sens':>6s} {'Prec':>6s} {'F1':>6s} {'TP':>6s} {'FN':>6s} {'FP':>6s}")
        print(f"    {'max_combine':<25s} {sens_st:>6.4f} {prec_st:>6.4f} {f1_st:>6.4f} {tp_st:>6d} {fn_st:>6d} {fp_st:>6d}")
        print(f"    {'HPP-only':<25s} {sens_st_h:>6.4f} {prec_st_h:>6.4f} {f1_st_h:>6.4f} {tp_st_h:>6d} {fn_st_h:>6d} {fp_st_h:>6d}")
        print(f"    {'Delta (max-hpp)':<25s}           {'':>6s} {'':>6s} {f1_st-f1_st_h:>+6.4f}")

        per_case_f1_st = [c['f1_max'] for c in st_cases]
        print(f"\n    Per-case F1 distribution:")
        for p in [25, 50, 75]:
            print(f"      P{p}: {np.percentile(per_case_f1_st, p):.3f}")
        print(f"      Mean: {np.mean(per_case_f1_st):.3f}")
        print(f"      F1>=0.8: {sum(1 for v in per_case_f1_st if v >= 0.8)}/{len(st_cases)}")
        print(f"      F1<0.5:  {sum(1 for v in per_case_f1_st if v < 0.5)}/{len(st_cases)}")

        # Laterality breakdown for LPD
        if subtype == 'lpd':
            for lat in ['left', 'right', None]:
                lat_label = lat if lat else 'unspecified'
                lat_cases = [c for c in st_cases if c['laterality'] == lat]
                if not lat_cases:
                    continue
                tp_l = sum(c['tp_max'] for c in lat_cases)
                fn_l = sum(c['fn_max'] for c in lat_cases)
                fp_l = sum(c['fp_max'] for c in lat_cases)
                s = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0
                p_ = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0
                f = 2 * p_ * s / (p_ + s) if (p_ + s) > 0 else 0
                print(f"\n    LPD laterality={lat_label}: n={len(lat_cases)}, "
                      f"Sens={s:.3f}, Prec={p_:.3f}, F1={f:.3f}")

        # CNN freq quality per subtype
        freq_errs_st = [c['freq_error'] for c in st_cases if np.isfinite(c['freq_error'])]
        if freq_errs_st:
            print(f"\n    CNN freq error: mean={np.mean(freq_errs_st):.3f}, "
                  f"med={np.median(freq_errs_st):.3f} Hz (n={len(freq_errs_st)})")

    # ── ANALYSIS H: Frequency range breakdown ─────────────────────────────────
    print_section("ANALYSIS H: Performance by Discharge Frequency Range")

    freq_bins = [
        ('Very low  [0.3, 0.5)', 0.3, 0.5),
        ('Low       [0.5, 0.8)', 0.5, 0.8),
        ('Medium    [0.8, 1.2)', 0.8, 1.2),
        ('High      [1.2, 1.8)', 1.2, 1.8),
        ('Very high [1.8, 3.5)', 1.8, 3.5),
    ]

    print(f"\n  {'Frequency Range':<24s} {'N':>4s} {'Sens':>6s} {'Prec':>6s} {'F1':>6s} "
          f"{'F1_hpp':>7s} {'Delta':>7s} {'FrqErr':>7s}")
    print(f"  {'-'*76}")

    for label, lo, hi in freq_bins:
        bin_cases = [c for c in cases
                     if np.isfinite(c['gold_freq']) and lo <= c['gold_freq'] < hi]
        if not bin_cases:
            continue
        tp_b = sum(c['tp_max'] for c in bin_cases)
        fn_b = sum(c['fn_max'] for c in bin_cases)
        fp_b = sum(c['fp_max'] for c in bin_cases)
        s_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
        p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
        f_b = 2 * p_b * s_b / (p_b + s_b) if (p_b + s_b) > 0 else 0

        tp_bh = sum(c['tp_hpp'] for c in bin_cases)
        fn_bh = sum(c['fn_hpp'] for c in bin_cases)
        fp_bh = sum(c['fp_hpp'] for c in bin_cases)
        s_bh = tp_bh / (tp_bh + fn_bh) if (tp_bh + fn_bh) > 0 else 0
        p_bh = tp_bh / (tp_bh + fp_bh) if (tp_bh + fp_bh) > 0 else 0
        f_bh = 2 * p_bh * s_bh / (p_bh + s_bh) if (p_bh + s_bh) > 0 else 0

        avg_ferr = np.mean([c['freq_error'] for c in bin_cases if np.isfinite(c['freq_error'])])

        print(f"  {label:<24s} {len(bin_cases):>4d} {s_b:>6.4f} {p_b:>6.4f} {f_b:>6.4f} "
              f"{f_bh:>7.4f} {f_b-f_bh:>+7.4f} {avg_ferr:>7.3f}")

    # Per-case F1 vs gold freq (correlation)
    valid_freq_f1 = [(c['gold_freq'], c['f1_max'])
                     for c in cases if np.isfinite(c['gold_freq'])]
    if valid_freq_f1:
        gfs, f1s = zip(*valid_freq_f1)
        rho_freq_f1, p_freq_f1 = spearmanr(gfs, f1s)
        print(f"\n  Spearman rho(gold_freq, F1) = {rho_freq_f1:.4f}  (p={p_freq_f1:.4f})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("SUMMARY")

    elapsed = time.time() - t0
    print(f"\n  Total analysis time: {elapsed:.0f}s")
    print(f"  Cases analyzed: {n_cases}")
    print(f"\n  Key findings:")
    print(f"    1. Overall F1 (max_combine): {f1_overall:.4f}  (HPP-only: {f1_h:.4f}, delta: {f1_overall-f1_h:+.4f})")
    if valid_pairs:
        print(f"    2. Freq error vs F1 correlation (Spearman rho): {rho_fe_f1:.4f}")
    print(f"    3. F1 >= 0.8: {len(good_cases)}/{n_cases} ({100*len(good_cases)/n_cases:.1f}%)")
    print(f"    4. F1 < 0.5:  {len(bad_cases)}/{n_cases} ({100*len(bad_cases)/n_cases:.1f}%)")
    print(f"    5. Cases where max_combine hurts (delta F1 < -0.05): {len(worse_cases)}/{n_cases}")
    print(f"    6. FPs from bad freq estimation (error >{freq_error_thresh}Hz): "
          f"{total_fp_in_bad_freq}/{total_fp} ({100*total_fp_in_bad_freq/max(1,total_fp):.1f}% of all FPs)")
    print(f"    7. Cases improved by max_combine vs HPP: {improved}/{n_cases} ({100*improved/n_cases:.1f}%)")
    print(f"    8. Cases worsened:                       {worsened}/{n_cases} ({100*worsened/n_cases:.1f}%)")
    print(f"\n{'=' * 80}")


if __name__ == '__main__':
    main()
