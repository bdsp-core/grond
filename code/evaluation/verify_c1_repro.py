#!/usr/bin/env python3
"""Reproduce the C1 production-config discharge-timing F1 on current models.

The headline manuscript number is F1=0.889, mean timing MAE=1.0 ms, n=582,
sourced from paper_materials/archive/method_comparison_table.json (committed
by hand in 2026-03-23 as 'HemiCET v2 + DP (C1)'). The script that produced
that number (code/archive/hemi_detector_experiments/run_e4_e7_experiments.py)
used a specific hyperparameter set (C1_PARAMS) that differs from the
defaults in code/pd_profiler.py and code/discharge_detector.py.

This script re-runs the C1 evaluator on the *current* retrained HemiCET v2
models + current CNN+ACF frequency ensemble, on the full set of PD ground-
truth cases. It reports F1 / sens / prec / freq rho / freq MAE / mean and
median timing MAE so we can compare directly with the archived 0.889.

Usage:
    conda run -n morgoth python code/evaluation/verify_c1_repro.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS  # type: ignore
from discharge_detector import (  # type: ignore
    DischargeDetector, detect_active_interval,
    em_refine, estimate_frequency_acf,
)
from hemi_detector.hemi_cet import HemiCET  # type: ignore

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {DEVICE}')

TOLERANCE_S = 0.1
N_SAMPLES = 2000
N_FOLDS = 5

# Frozen C1 production hyperparameters (from
# code/archive/hemi_detector_experiments/run_e4_e7_experiments.py:59-67)
C1_PARAMS = dict(
    dp_alpha=1.5,
    dp_beta=0.3,
    dp_lambda=0.05,
    peak_height_frac=0.05,
    max_skip=3,
    evidence_threshold_pct=50,
    min_evidence_ratio=0.4,
)

# Current pd_profiler.py / discharge_detector.py defaults (for comparison)
CURRENT_PROD_PARAMS = dict(
    dp_alpha=1.275,
    dp_beta=0.3,
    dp_lambda=0.05,
    peak_height_frac=0.05,
    max_skip=3,
    evidence_threshold_pct=0,
    min_evidence_ratio=0.3,
)

HEMI_DIR_V2 = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_v2'
HEMI_DIR_V1 = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet'
OUT_DIR = PROJECT_DIR / 'results' / 'c1_repro'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hemicet(model_dir):
    models = []
    for k in range(N_FOLDS):
        m = HemiCET(in_channels=8).to(DEVICE)
        ckpt = torch.load(str(model_dir / f'hemi_cet_fold{k}.pt'), map_location=DEVICE, weights_only=False)
        state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def run_hemicet_dp(seg, subtype, laterality, hemi_cet_models, detector,
                   dp_alpha=1.5, dp_beta=0.3, dp_lambda=0.05,
                   peak_height_frac=0.05, max_skip=3,
                   evidence_threshold_pct=50, min_evidence_ratio=0.4):
    """Lightly trimmed copy of run_e4_e7_experiments.run_hemicet_dp (no E5 path)."""

    def _run(indices):
        all_pd, all_lf = [], []
        for ci in indices[:8]:
            ch = seg[ci].astype(np.float32).copy()
            if not np.all(np.isfinite(ch)):
                all_pd.append(0.0); all_lf.append(0.0); continue
            mu, std = float(np.mean(ch)), float(np.std(ch))
            ch = (ch - mu) / std if std > 1e-8 else ch - mu
            x = torch.from_numpy(ch[None, None, :]).to(detector.device)
            pp, lf = [], []
            for m in detector.cnn_models:
                p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
            all_pd.append(float(np.mean(pp))); all_lf.append(float(np.mean(lf)))

        pdw = np.array(all_pd); lfs = np.array(all_lf); ws = float(pdw.sum())
        cnn_freq = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
        b, a = butter(4, 20.0 / (FS / 2), btype='low')
        acfs = []
        for ci in indices[:8]:
            try:
                sig = filtfilt(b, a, seg[ci])
            except Exception:
                sig = seg[ci]
            f2 = estimate_frequency_acf(sig, FS)
            if np.isfinite(f2):
                acfs.append(f2)
        acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn_freq
        freq = float(np.clip(0.8 * cnn_freq + 0.2 * acf, 0.3, 3.5))

        n_ch = len(indices)
        hs = np.zeros((n_ch, N_SAMPLES), dtype=np.float32)
        for i, ci in enumerate(indices):
            if ci < seg.shape[0]:
                ch_data = seg[ci].astype(np.float32).copy()
                mu2, std2 = float(np.mean(ch_data)), float(np.std(ch_data))
                hs[i] = (ch_data - mu2) / std2 if std2 > 1e-8 else ch_data - mu2

        x2 = torch.from_numpy(hs[None]).to(DEVICE)
        preds = [m(x2).squeeze().cpu().numpy() for m in hemi_cet_models]
        ev = np.mean(preds, axis=0)

        if evidence_threshold_pct > 0 and np.any(ev > 0):
            thr = np.percentile(ev[ev > 0], evidence_threshold_pct)
            ev = np.where(ev > thr, ev, 0)

        active_start, active_end = detect_active_interval(ev, FS)
        segment = ev[active_start:active_end + 1]
        if len(segment) < 3:
            return []

        T = 1.0 / freq if freq > 0 else 1.0
        min_dist = max(20, int(0.2 * T * FS))
        min_height = peak_height_frac * float(np.max(segment))
        peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
        strong_height = 0.5 * float(np.max(segment))
        strong_peaks, _ = find_peaks(segment, height=strong_height,
                                     distance=max(10, int(0.1 * T * FS)))
        candidates = np.unique(np.concatenate([peaks, strong_peaks])) + active_start

        if len(candidates) == 0:
            return []

        if len(candidates) == 1:
            ds = candidates.copy()
        else:
            n = len(candidates)
            raw_scores = np.array([ev[c] for c in candidates])
            node_scores = raw_scores ** 1.5
            best_score = np.full(n, -np.inf)
            best_prev = np.full(n, -1, dtype=int)
            for i in range(n):
                best_score[i] = node_scores[i] - dp_lambda
            for j in range(1, n):
                for i in range(j):
                    dt = (candidates[j] - candidates[i]) / FS
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
            path = []
            idx = int(np.argmax(best_score))
            while idx >= 0:
                path.append(idx)
                idx = best_prev[idx]
            path.reverse()
            ds = candidates[np.array(path)]

        if len(ds) >= 3:
            ds = em_refine(ev, ds, FS, freq)

        if min_evidence_ratio > 0 and len(ds) >= 2:
            peak_vals = np.array([ev[int(s)] for s in ds])
            threshold = min_evidence_ratio * float(np.median(peak_vals))
            ds = ds[peak_vals >= threshold]

        return (ds / FS).tolist() if len(ds) > 0 else []

    if subtype == 'gpd' or laterality not in ('left', 'right'):
        tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
        return tl if len(tl) >= len(tr) else tr
    return _run(LEFT_INDICES if laterality == 'left' else RIGHT_INDICES)


def evaluate(predictions, gt_cases, subtype_filter=None):
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs, match_errors = [], [], []
    n_used = 0

    for pid, algo_times in predictions.items():
        if pid not in gt_cases:
            continue
        gt_data = gt_cases[pid]
        if subtype_filter and gt_data.get('subtype') != subtype_filter:
            continue
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue
        n_used += 1
        algo_times = sorted(algo_times)

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_d, best_a = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_d:
                        best_d, best_a = d, ai
            if best_d <= TOLERANCE_S and best_a >= 0:
                gt_matched[gi] = True
                algo_matched[best_a] = True
                match_errors.append(best_d)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = np.diff(gt_times)
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        algo_freq = 1.0 / np.median(np.diff(algo_times)) if len(algo_times) >= 2 else np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(float(gt_freq)); algo_freqs.append(float(algo_freq))

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = float(spearmanr(algo_freqs, gt_freqs)[0]) if len(gt_freqs) >= 3 else float('nan')
    freq_mae = float(np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs)))) if len(gt_freqs) >= 3 else float('nan')
    mean_timing_ms = float(np.mean(match_errors) * 1000) if match_errors else float('nan')
    median_timing_ms = float(np.median(match_errors) * 1000) if match_errors else float('nan')

    return dict(
        n_cases=n_used,
        tp=total_tp, fn=total_fn, fp=total_fp,
        sensitivity=round(sens, 4),
        precision=round(prec, 4),
        f1=round(f1, 4),
        freq_rho=round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        freq_mae=round(freq_mae, 4) if np.isfinite(freq_mae) else None,
        mean_timing_ms=round(mean_timing_ms, 2) if np.isfinite(mean_timing_ms) else None,
        median_timing_ms=round(median_timing_ms, 2) if np.isfinite(median_timing_ms) else None,
    )


def main():
    t0 = time.time()
    print('Loading dataset + GT cases...')
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    with open(PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json') as f:
        hpp = json.load(f)
    gt_cases = {pid: v for pid, v in hpp.items()
                if v.get('review_status') == 'ground_truth'
                and v.get('subtype') in ('lpd', 'gpd')}
    print(f'  PD ground-truth cases: {len(gt_cases)}')

    print('Loading DischargeDetector (CNN+ACF)...')
    detector = DischargeDetector()
    print(f'  {len(detector.cnn_models)} CNN fold models loaded.\n')

    # Pre-filter GT cases to those resolvable by load_dataset() — drop the
    # patients whose EEG mats are not in the current data layout (the May-2026
    # cleanup moved external_drive mats out of segments.csv; those 510 patients
    # would otherwise be silently counted as 100% FN). See
    # results/c1_repro/REPRO_INVESTIGATION.md for the root-cause writeup.
    eval_gt_cases = {}
    skipped_no_seg = 0
    skipped_wrong_shape = 0
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            skipped_no_seg += 1
            continue
        if pat_segs[0].shape[1] != N_SAMPLES:
            skipped_wrong_shape += 1
            continue
        eval_gt_cases[pid] = gt
    print(f'GT-PD patients resolvable in current data layout: {len(eval_gt_cases)} '
          f'(of {len(gt_cases)}; {skipped_no_seg} have no segment, '
          f'{skipped_wrong_shape} have wrong shape)')

    print(f'\nRunning C1 (orig hemi_cet) + C1 (hemi_cet_v2) + current_prod (hemi_cet_v2) on {len(eval_gt_cases)} cases\n')

    all_results = {}
    model_configs = [
        ('C1_orig_hemi_cet',      C1_PARAMS,         HEMI_DIR_V1),
        ('C1_hemi_cet_v2',        C1_PARAMS,         HEMI_DIR_V2),
        ('current_prod_v2',       CURRENT_PROD_PARAMS, HEMI_DIR_V2),
    ]
    for cfg_name, params, model_dir in model_configs:
        print(f'--- {cfg_name}: models={model_dir.name}  params={params}')
        hemi_cet_models = load_hemicet(model_dir)
        preds = {}
        n_processed = 0
        for pid, gt in eval_gt_cases.items():
            pat_segs = segments.get(pid, [])
            seg = pat_segs[0]
            try:
                row = df[df['patient_id'] == pid]
                lat = row.iloc[0].get('laterality', '') if len(row) else ''
                if not isinstance(lat, str) or lat not in ('left', 'right'):
                    lat = None
                times = run_hemicet_dp(seg, gt['subtype'], lat,
                                       hemi_cet_models, detector, **params)
                preds[pid] = times
                n_processed += 1
            except Exception as e:
                preds[pid] = []

        result = dict(
            overall=evaluate(preds, eval_gt_cases),
            lpd=evaluate(preds, eval_gt_cases, subtype_filter='lpd'),
            gpd=evaluate(preds, eval_gt_cases, subtype_filter='gpd'),
        )
        all_results[cfg_name] = result
        print(f'  Overall: F1={result["overall"]["f1"]:.4f}  sens={result["overall"]["sensitivity"]:.4f}  '
              f'prec={result["overall"]["precision"]:.4f}  '
              f'freq_rho={result["overall"]["freq_rho"]}  '
              f'mean_timing_ms={result["overall"]["mean_timing_ms"]}  '
              f'median_timing_ms={result["overall"]["median_timing_ms"]}')
        print(f'  LPD: F1={result["lpd"]["f1"]:.4f}  GPD: F1={result["gpd"]["f1"]:.4f}  '
              f'({time.time()-t0:.0f}s elapsed)\n')

    out_path = OUT_DIR / 'c1_repro_results.json'
    with open(out_path, 'w') as f:
        json.dump({
            'archived_published': {
                'method': 'HemiCET v2 + DP (C1) — paper_materials/archive/method_comparison_table.json',
                'f1': 0.889,
                'sensitivity': 0.921,
                'precision': 0.8592,
                'freq_rho': 0.8908,
                'freq_mae': 0.183,
                'mean_timing_ms': 1.0,
                'median_timing_ms': 0.0,
                'n_cases': 582,
            },
            'current_runs': all_results,
            'C1_params': C1_PARAMS,
            'current_prod_params': CURRENT_PROD_PARAMS,
            'total_time_s': round(time.time() - t0, 1),
        }, f, indent=2)
    print(f'Saved {out_path.relative_to(PROJECT_DIR)}  ({time.time() - t0:.0f}s total)')


if __name__ == '__main__':
    main()
