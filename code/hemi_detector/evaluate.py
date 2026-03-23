"""
HemiNet Evaluation — Experiment 1.1

Loads the 5-fold trained models and evaluates on held-out folds.

Decoding strategies:
  Strategy 1: Simple peak picking (p_eff = p_event × p_active^1.5)
  Strategy 2: DP post-processing using predicted freq as period prior (Exp 4.1 preview)

Reports:
  - Event F1, Sensitivity, Precision (±100ms tolerance)
  - Frequency Spearman correlation
  - Per-fold and aggregate results

Usage:
    conda run -n foe_dl python code/hemi_detector/evaluate.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from scipy.ndimage import label as ndlabel

import torch
import torch.nn as nn

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.model import HemiNet
from hemi_detector.dataset import (
    HemiDataset, get_patient_ids, get_patient_subtypes,
    LEFT_INDICES, RIGHT_INDICES,
    _load_segment, _zscore_segment, _hemisphere_evidence,
)
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

# ── Configuration ─────────────────────────────────────────────────────────────
EVAL_CFG = {
    'hpp_path': str(PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'),
    'patients_csv': str(PROJECT_DIR / 'data' / 'labels' / 'patients.csv'),
    'eeg_dir': str(PROJECT_DIR / 'data' / 'eeg'),
    'model_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_1'),
    'n_folds': 5,
    'seed': 42,
    'batch_size': 32,
    'num_workers': 0,

    # Decoding
    'peak_min_height': 0.25,
    'active_threshold': 0.4,
    'active_min_bins': 50,
    'active_gamma': 1.5,
    'match_tolerance_s': 0.1,
    'fs_target': 100,
    'device': 'mps',

    # DP params (Strategy 2)
    'dp_alpha': 1.275,
    'dp_lambda': 0.05,
    'dp_beta': 0.3,
}


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_strategy1(
    event_logits: np.ndarray,
    active_logits: np.ndarray,
    freq_logit: float,
    cfg: Dict,
) -> List[float]:
    """Strategy 1: Peak picking on p_eff = p_event × p_active^gamma."""
    p_event = 1.0 / (1.0 + np.exp(-event_logits))
    p_active = 1.0 / (1.0 + np.exp(-active_logits))

    p_eff = p_event * (p_active ** cfg['active_gamma'])

    # Active mask
    active_bin = (p_active > cfg['active_threshold']).astype(np.float32)
    labeled, n = ndlabel(active_bin)
    active_mask = np.zeros(len(p_active), dtype=bool)
    for region_id in range(1, n + 1):
        region_mask = (labeled == region_id)
        if region_mask.sum() >= cfg['active_min_bins']:
            active_mask |= region_mask
    if not active_mask.any():
        active_mask[:] = True

    p_masked = p_eff.copy()
    p_masked[~active_mask] = 0.0

    pred_freq = float(np.clip(np.exp(freq_logit), 0.2, 5.0))
    min_dist_bins = max(3, int(cfg['fs_target'] / (pred_freq * 2.0)))

    peaks, _ = find_peaks(p_masked, height=cfg['peak_min_height'], distance=min_dist_bins)
    return [float(pk / cfg['fs_target']) for pk in peaks]


def decode_strategy2_dp(
    event_logits: np.ndarray,
    active_logits: np.ndarray,
    freq_logit: float,
    cfg: Dict,
) -> List[float]:
    """Strategy 2: Peak picking + DP post-processing.

    Uses the predicted frequency as a period prior for a lightweight DP.
    The DP selects a subset of candidate peaks that best fits a periodic pattern.
    """
    # Get candidates from Strategy 1
    candidates = decode_strategy1(event_logits, active_logits, freq_logit, cfg)
    if len(candidates) < 2:
        return candidates

    pred_freq = float(np.clip(np.exp(freq_logit), 0.2, 5.0))
    period = 1.0 / pred_freq

    # DP: find subset of candidates that minimizes deviations from ideal periodicity
    # State: (last selected time, current F1-proxy score)
    # This is a simplified version of the full DP from the existing pipeline

    n = len(candidates)
    # score[i] = best score ending at candidate i
    score = np.zeros(n)
    prev = np.full(n, -1, dtype=int)
    count = np.ones(n, dtype=int)  # number of events in best path ending at i

    # Evidence at each candidate bin
    p_event = 1.0 / (1.0 + np.exp(-event_logits))

    def evidence_at(t_s: float) -> float:
        bin_idx = int(round(t_s * cfg['fs_target']))
        bin_idx = max(0, min(len(p_event) - 1, bin_idx))
        return float(p_event[bin_idx])

    # Initialize
    for i in range(n):
        score[i] = evidence_at(candidates[i])

    # DP transitions
    alpha = cfg['dp_alpha']
    lam = cfg['dp_lambda']
    beta = cfg['dp_beta']

    for j in range(1, n):
        for i in range(j):
            ipi = candidates[j] - candidates[i]
            if ipi <= 0 or ipi > period * 3:
                continue
            # Period deviation penalty
            period_ratio = ipi / period
            if period_ratio < 0.5:
                continue
            deviation = abs(period_ratio - round(period_ratio)) / max(round(period_ratio), 1)
            penalty = lam * deviation ** 2

            # Transition score
            new_score = score[i] + evidence_at(candidates[j]) * (alpha ** (ipi / period)) - penalty
            if new_score > score[j]:
                score[j] = new_score
                prev[j] = i
                count[j] = count[i] + 1

    # Find best ending point (require at least 2 events)
    valid = [(score[j], j) for j in range(n) if count[j] >= 2]
    if not valid:
        return candidates  # Fallback to Strategy 1

    _, best_end = max(valid)

    # Traceback
    path = []
    cur = best_end
    while cur >= 0:
        path.append(candidates[cur])
        cur = prev[cur]
    path.reverse()

    return path


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(
    predictions: Dict[str, List[float]],
    gt_lookup: Dict[str, List[float]],
    tolerance_s: float = 0.1,
) -> Dict:
    """Compute F1, sensitivity, precision, and frequency Spearman."""
    tp, fn, fp = 0, 0, 0
    gt_freqs, pred_freqs = [], []

    for pid, gt_times in gt_lookup.items():
        if len(gt_times) < 2:
            continue
        pred_times = sorted(predictions.get(pid, []))
        gt_sorted = sorted(gt_times)

        gt_matched = [False] * len(gt_sorted)
        pred_matched = [False] * len(pred_times)

        for gi, gt in enumerate(gt_sorted):
            best_dist, best_pi = np.inf, -1
            for pi, pt in enumerate(pred_times):
                if not pred_matched[pi]:
                    d = abs(gt - pt)
                    if d < best_dist:
                        best_dist, best_pi = d, pi
            if best_dist <= tolerance_s and best_pi >= 0:
                gt_matched[gi] = True
                pred_matched[best_pi] = True

        tp += sum(gt_matched)
        fn += len(gt_sorted) - sum(gt_matched)
        fp += len(pred_times) - sum(pred_matched)

        # Frequency from IPI
        if len(gt_sorted) >= 2:
            gt_ipi = np.median(np.diff(gt_sorted))
            gt_freq = 1.0 / gt_ipi if gt_ipi > 0 else np.nan
        else:
            gt_freq = np.nan

        if len(pred_times) >= 2:
            pred_ipi = np.median(np.diff(pred_times))
            pred_freq = 1.0 / pred_ipi if pred_ipi > 0 else np.nan
        else:
            pred_freq = np.nan

        if np.isfinite(gt_freq) and np.isfinite(pred_freq):
            gt_freqs.append(gt_freq)
            pred_freqs.append(pred_freq)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

    rho = float(spearmanr(pred_freqs, gt_freqs)[0]) if len(gt_freqs) >= 3 else float('nan')

    return {
        'f1': round(f1, 4),
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'freq_spearman': round(rho, 4) if np.isfinite(rho) else None,
        'tp': tp, 'fn': fn, 'fp': fp,
        'n_cases': len(gt_lookup),
        'n_with_freq': len(gt_freqs),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    cfg = EVAL_CFG

    # Device
    if cfg['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    with open(cfg['hpp_path']) as f:
        hpp_data = json.load(f)

    gt_data = {k: v for k, v in hpp_data.items()
               if v.get('review_status') == 'ground_truth'}

    gt_lookup: Dict[str, List[float]] = {
        pid: sorted(v['global_times'])
        for pid, v in gt_data.items()
        if len(v.get('global_times', [])) >= 2
    }
    print(f"GT cases with ≥2 discharges: {len(gt_lookup)}")

    patients_df = pd.read_csv(cfg['patients_csv'])
    patients_df['patient_id'] = patients_df['patient_id'].astype(str)
    eeg_dir = Path(cfg['eeg_dir'])
    model_dir = Path(cfg['model_dir'])

    # Build full dataset (no augment)
    full_dataset = HemiDataset(
        hpp_data=gt_data,
        eeg_dir=eeg_dir,
        patients_df=patients_df,
        augment=False,
    )
    patient_ids = get_patient_ids(full_dataset)
    subtypes = get_patient_subtypes(full_dataset, gt_data)

    unique_pids = list(dict.fromkeys(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    pid_subtypes = {}
    for pid, subtype in zip(patient_ids, subtypes):
        pid_subtypes[pid] = subtype
    unique_subtypes = [pid_subtypes.get(pid, 'lpd') for pid in unique_pids]

    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    example_pid_indices = [pid_to_idx[pid] for pid in patient_ids]

    # ── Per-fold evaluation ───────────────────────────────────────────
    all_s1_preds: Dict[str, List[float]] = {}
    all_s2_preds: Dict[str, List[float]] = {}

    fold_results_s1 = []
    fold_results_s2 = []

    for fold_k, (train_pat_idx, val_pat_idx) in enumerate(
        skf.split(unique_pids, unique_subtypes)
    ):
        model_path = model_dir / f'fold{fold_k}' / 'best_model.pt'
        if not model_path.exists():
            print(f"  Fold {fold_k}: model not found at {model_path}, skipping")
            continue

        print(f"\n  Fold {fold_k+1}: loading model from {model_path}")
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        model = HemiNet(in_channels=8).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        val_pat_set = set(val_pat_idx)
        val_example_idx = [
            i for i, pidx in enumerate(example_pid_indices) if pidx in val_pat_set
        ]
        val_ds = Subset(full_dataset, val_example_idx)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg['batch_size'],
            shuffle=False,
            num_workers=cfg['num_workers'],
        )

        fold_s1_preds: Dict[str, List[float]] = {}
        fold_s2_preds: Dict[str, List[float]] = {}

        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                ev_logits, ac_logits, fr_logit = model(eeg)

                batch_size = eeg.shape[0]
                for i in range(batch_size):
                    pid = batch['pid'][i]
                    ev_np = ev_logits[i].cpu().numpy()
                    ac_np = ac_logits[i].cpu().numpy()
                    fr_val = fr_logit[i].item()

                    s1 = decode_strategy1(ev_np, ac_np, fr_val, cfg)
                    s2 = decode_strategy2_dp(ev_np, ac_np, fr_val, cfg)

                    # Merge across hemispheres for GPD (same patient → two examples)
                    for preds_dict, times in [(fold_s1_preds, s1), (fold_s2_preds, s2)]:
                        if pid in preds_dict:
                            combined = sorted(preds_dict[pid] + times)
                            merged = []
                            for t in combined:
                                if not merged or t - merged[-1] > 0.05:
                                    merged.append(t)
                            preds_dict[pid] = merged
                        else:
                            preds_dict[pid] = times

        # Evaluate this fold
        fold_gt = {pid: gt_lookup[pid] for pid in fold_s1_preds if pid in gt_lookup}
        r1 = evaluate_predictions(fold_s1_preds, fold_gt, cfg['match_tolerance_s'])
        r2 = evaluate_predictions(fold_s2_preds, fold_gt, cfg['match_tolerance_s'])

        print(f"  Fold {fold_k+1}  Strategy1: F1={r1['f1']:.4f}  Sens={r1['sensitivity']:.4f}  "
              f"Prec={r1['precision']:.4f}  FreqRho={r1['freq_spearman']}")
        print(f"  Fold {fold_k+1}  Strategy2: F1={r2['f1']:.4f}  Sens={r2['sensitivity']:.4f}  "
              f"Prec={r2['precision']:.4f}  FreqRho={r2['freq_spearman']}")

        fold_results_s1.append(r1)
        fold_results_s2.append(r2)

        # Aggregate for global eval
        all_s1_preds.update(fold_s1_preds)
        all_s2_preds.update(fold_s2_preds)

    # ── Global evaluation (all folds) ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1.1 — FINAL RESULTS")
    print(f"{'='*70}")

    global_s1 = evaluate_predictions(all_s1_preds, gt_lookup, cfg['match_tolerance_s'])
    global_s2 = evaluate_predictions(all_s2_preds, gt_lookup, cfg['match_tolerance_s'])

    print(f"\n  Strategy 1 (Peak picking):")
    print(f"    F1={global_s1['f1']:.4f}  Sensitivity={global_s1['sensitivity']:.4f}  "
          f"Precision={global_s1['precision']:.4f}  FreqRho={global_s1['freq_spearman']}")
    print(f"    TP={global_s1['tp']}  FN={global_s1['fn']}  FP={global_s1['fp']}")
    print(f"    N={global_s1['n_cases']} cases")

    print(f"\n  Strategy 2 (DP post-processing):")
    print(f"    F1={global_s2['f1']:.4f}  Sensitivity={global_s2['sensitivity']:.4f}  "
          f"Precision={global_s2['precision']:.4f}  FreqRho={global_s2['freq_spearman']}")
    print(f"    TP={global_s2['tp']}  FN={global_s2['fn']}  FP={global_s2['fp']}")
    print(f"    N={global_s2['n_cases']} cases")

    print(f"\n  Reference baseline: F1=0.740 (full 18ch pipeline)")
    print(f"  Per-hemisphere baseline: F1=0.672")

    # Per-fold summary
    if fold_results_s1:
        s1_f1s = [r['f1'] for r in fold_results_s1]
        s2_f1s = [r['f1'] for r in fold_results_s2]
        print(f"\n  Per-fold F1 (Strategy 1): {[round(f, 4) for f in s1_f1s]}")
        print(f"  Mean ± std: {np.mean(s1_f1s):.4f} ± {np.std(s1_f1s):.4f}")
        print(f"\n  Per-fold F1 (Strategy 2): {[round(f, 4) for f in s2_f1s]}")
        print(f"  Mean ± std: {np.mean(s2_f1s):.4f} ± {np.std(s2_f1s):.4f}")

    print(f"\n  Total evaluation time: {(time.time()-t0)/60:.1f} min")

    # Save results
    results = {
        'experiment': '1.1',
        'description': 'HemiNet Design A — U-Net + Transformer bottleneck, 8ch',
        'strategy1_global': global_s1,
        'strategy2_global': global_s2,
        'strategy1_per_fold': fold_results_s1,
        'strategy2_per_fold': fold_results_s2,
        'baseline_hemi_f1': 0.672,
        'baseline_full_f1': 0.740,
    }
    save_path = Path(cfg['model_dir']) / 'eval_results.json'
    with open(str(save_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_path}")


if __name__ == '__main__':
    main()
