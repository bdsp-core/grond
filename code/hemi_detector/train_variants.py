"""
HemiNet Variant Training Scripts — Experiments 1.2, 1.4, 1.5

Trains and evaluates three HemiNet variants using 5-fold patient-stratified CV:
  - Exp 1.2: HemiNetB (Design B — Dilated Convolutions)
  - Exp 1.4: HemiNetD (Design D — Neural Wrapper Around Existing Pipeline)
  - Exp 1.5: HemiNetPretrained (Design A + MAE Pretraining, Phase 2 fine-tuning)

Usage:
    # Run Design B (Experiment 1.2)
    conda run -n foe_dl python code/hemi_detector/train_variants.py --exp 1.2

    # Run Design D (Experiment 1.4)
    conda run -n foe_dl python code/hemi_detector/train_variants.py --exp 1.4

    # Run Design A + MAE fine-tuning (Experiment 1.5)
    # (Must run pretrain_mae.py first)
    conda run -n foe_dl python code/hemi_detector/train_variants.py --exp 1.5

Saves:
    data/hemi_cache/exp{EXP_ID}/
        fold{k}/best_model.pt
        fold{k}/training_log.json
        config.json
        cv_results.json
        eval_results.json
"""

import sys
import json
import time
import math
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from scipy.ndimage import label as ndlabel

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.model import (
    HemiNetB, HemiNetD, HemiNetPretrained,
    HemiNet, count_parameters,
)
from hemi_detector.dataset import HemiDataset, get_patient_ids, get_patient_subtypes

# ── Experiment configs ─────────────────────────────────────────────────────────

BASE_CONFIG = {
    # Data
    'hpp_path': str(PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'),
    'patients_csv': str(PROJECT_DIR / 'data' / 'labels' / 'patients.csv'),
    'eeg_dir': str(PROJECT_DIR / 'data' / 'eeg'),

    # Training
    'n_folds': 5,
    'n_epochs': 100,
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'grad_clip': 1.0,
    'eval_every': 5,
    'early_stop_patience': 20,

    # Loss weights
    'w_event': 1.0,
    'w_active': 0.3,
    'w_freq': 0.2,

    # Focal BCE params
    'focal_gamma': 2.0,
    'focal_alpha': 0.75,

    # Soft Dice weight within event loss
    'dice_weight': 0.5,

    # Huber delta for freq loss
    'huber_delta': 0.2,

    # Evaluation
    'peak_min_height': 0.25,
    'active_threshold': 0.4,
    'active_min_bins': 50,
    'active_gamma': 1.5,
    'match_tolerance_s': 0.1,
    'fs_target': 100,

    # DP params
    'dp_alpha': 1.275,
    'dp_lambda': 0.05,
    'dp_beta': 0.3,

    # Device
    'device': 'mps',
    'num_workers': 0,
    'seed': 42,
}

EXP_CONFIGS = {
    '1.2': {
        **BASE_CONFIG,
        'experiment': '1.2',
        'description': 'HemiNet Design B: Dilated Convolutions (no Transformer)',
        'model_class': 'HemiNetB',
        'cache_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_2'),
        'lr': 5e-4,
    },
    '1.4': {
        **BASE_CONFIG,
        'experiment': '1.4',
        'description': 'HemiNet Design D: Neural Wrapper Around Existing Pipeline',
        'model_class': 'HemiNetD',
        'cache_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_4'),
        'lr': 5e-4,
        # Fewer epochs for Design D since it trains faster (HPP is precomputed per batch)
        # but HPP computation is slow so be patient
        'n_epochs': 80,
        'early_stop_patience': 16,
    },
    '1.5': {
        **BASE_CONFIG,
        'experiment': '1.5',
        'description': 'HemiNet Design A + MAE Pretraining (Phase 2 fine-tuning)',
        'model_class': 'HemiNetPretrained',
        'cache_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_5_finetune'),
        'pretrained_path': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_5_pretrain' / 'best_mae.pt'),
        'lr': 1e-4,   # Lower lr for fine-tuning pretrained model
    },
}


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_model(cfg: Dict, device) -> nn.Module:
    model_class = cfg['model_class']
    if model_class == 'HemiNetB':
        model = HemiNetB(in_channels=8, dropout=0.1)
    elif model_class == 'HemiNetD':
        model = HemiNetD(in_channels=8, dropout=0.1, fs=200.0)
    elif model_class == 'HemiNetPretrained':
        model = HemiNetPretrained(in_channels=8, dropout=0.1)
        pretrained_path = cfg.get('pretrained_path', '')
        if pretrained_path and Path(pretrained_path).exists():
            print(f"  Loading pretrained encoder from {pretrained_path}")
            model.load_pretrained_encoder(pretrained_path, device=device)
        else:
            print(f"  WARNING: pretrained_path not found ({pretrained_path}), training from scratch")
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    return model.to(device)


# ── Loss Functions ────────────────────────────────────────────────────────────

def focal_bce_loss(logits, targets, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def soft_dice_loss(logits, targets, smooth=1e-4):
    p = torch.sigmoid(logits)
    num = 2.0 * (p * targets).sum() + smooth
    den = p.sum() + targets.sum() + smooth
    return 1.0 - num / den


def event_loss(logits, targets, gamma=2.0, alpha=0.75, dice_w=0.5):
    return focal_bce_loss(logits, targets, gamma, alpha) + dice_w * soft_dice_loss(logits, targets)


def active_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)


def freq_loss(pred, target, delta=0.2):
    mask = torch.isfinite(target.squeeze(-1)) if target.dim() > 1 else torch.isfinite(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    p = pred.squeeze(-1)[mask]
    t = target.squeeze(-1)[mask]
    return F.huber_loss(p, t, delta=delta)


def total_loss(ev_logits, ac_logits, fr_logit, ev_t, ac_t, fr_t, cfg):
    l_ev = event_loss(ev_logits, ev_t, cfg['focal_gamma'], cfg['focal_alpha'], cfg['dice_weight'])
    l_ac = active_loss(ac_logits, ac_t)
    l_fr = freq_loss(fr_logit, fr_t, cfg['huber_delta'])
    loss = cfg['w_event'] * l_ev + cfg['w_active'] * l_ac + cfg['w_freq'] * l_fr
    return loss, {'event': l_ev.item(), 'active': l_ac.item(), 'freq': l_fr.item(), 'total': loss.item()}


# ── LR Schedule ───────────────────────────────────────────────────────────────

def get_lr(epoch, n_epochs, lr, warmup_epochs):
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── Decoding ──────────────────────────────────────────────────────────────────

def decode_strategy1(ev_logits, ac_logits, freq_logit, cfg):
    p_event = 1.0 / (1.0 + np.exp(-ev_logits))
    p_active = 1.0 / (1.0 + np.exp(-ac_logits))
    p_eff = p_event * (p_active ** cfg['active_gamma'])

    active_bin = (p_active > cfg['active_threshold']).astype(np.float32)
    labeled, n = ndlabel(active_bin)
    active_mask = np.zeros(len(p_active), dtype=bool)
    for region_id in range(1, n + 1):
        if (labeled == region_id).sum() >= cfg['active_min_bins']:
            active_mask |= (labeled == region_id)
    if not active_mask.any():
        active_mask[:] = True

    p_masked = p_eff.copy()
    p_masked[~active_mask] = 0.0

    pred_freq = float(np.clip(np.exp(freq_logit), 0.2, 5.0))
    min_dist = max(3, int(cfg['fs_target'] / (pred_freq * 2.0)))
    peaks, _ = find_peaks(p_masked, height=cfg['peak_min_height'], distance=min_dist)
    return [float(pk / cfg['fs_target']) for pk in peaks]


def decode_strategy2_dp(ev_logits, ac_logits, freq_logit, cfg):
    candidates = decode_strategy1(ev_logits, ac_logits, freq_logit, cfg)
    if len(candidates) < 2:
        return candidates

    pred_freq = float(np.clip(np.exp(freq_logit), 0.2, 5.0))
    period = 1.0 / pred_freq
    n = len(candidates)
    score = np.zeros(n)
    prev = np.full(n, -1, dtype=int)
    count = np.ones(n, dtype=int)

    p_event = 1.0 / (1.0 + np.exp(-ev_logits))

    def evidence_at(t_s):
        bin_idx = max(0, min(len(p_event) - 1, int(round(t_s * cfg['fs_target']))))
        return float(p_event[bin_idx])

    for i in range(n):
        score[i] = evidence_at(candidates[i])

    alpha, lam = cfg['dp_alpha'], cfg['dp_lambda']
    for j in range(1, n):
        for i in range(j):
            ipi = candidates[j] - candidates[i]
            if ipi <= 0 or ipi > period * 3:
                continue
            period_ratio = ipi / period
            if period_ratio < 0.5:
                continue
            deviation = abs(period_ratio - round(period_ratio)) / max(round(period_ratio), 1)
            penalty = lam * deviation ** 2
            new_score = score[i] + evidence_at(candidates[j]) * (alpha ** (ipi / period)) - penalty
            if new_score > score[j]:
                score[j] = new_score
                prev[j] = i
                count[j] = count[i] + 1

    valid = [(score[j], j) for j in range(n) if count[j] >= 2]
    if not valid:
        return candidates

    _, best_end = max(valid)
    path = []
    cur = best_end
    while cur >= 0:
        path.append(candidates[cur])
        cur = prev[cur]
    path.reverse()
    return path


def merge_hemisphere_preds(existing, new_preds):
    combined = sorted(existing + new_preds)
    merged = []
    for t in combined:
        if not merged or t - merged[-1] > 0.05:
            merged.append(t)
    return merged


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(predictions, gt_lookup, tolerance_s=0.1):
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
                if not pred_matched[pi] and abs(gt - pt) < best_dist:
                    best_dist, best_pi = abs(gt - pt), pi
            if best_dist <= tolerance_s and best_pi >= 0:
                gt_matched[gi] = True
                pred_matched[best_pi] = True

        tp += sum(gt_matched)
        fn += len(gt_sorted) - sum(gt_matched)
        fp += len(pred_times) - sum(pred_matched)

        if len(gt_sorted) >= 2:
            gt_ipi = np.median(np.diff(gt_sorted))
            gt_f = 1.0 / gt_ipi if gt_ipi > 0 else np.nan
        else:
            gt_f = np.nan
        if len(pred_times) >= 2:
            pred_ipi = np.median(np.diff(pred_times))
            pred_f = 1.0 / pred_ipi if pred_ipi > 0 else np.nan
        else:
            pred_f = np.nan
        if np.isfinite(gt_f) and np.isfinite(pred_f):
            gt_freqs.append(gt_f)
            pred_freqs.append(pred_f)

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


@torch.no_grad()
def validate_fold(model, val_loader, gt_lookup, cfg, device):
    model.eval()
    val_losses = []
    s1_preds = {}
    s2_preds = {}

    for batch in val_loader:
        eeg = batch['eeg'].to(device)
        ev_t = batch['event_t'].to(device)
        ac_t = batch['active_t'].to(device)
        fr_t = batch['freq_t'].to(device)

        ev_logits, ac_logits, fr_logit = model(eeg)
        loss, _ = total_loss(ev_logits, ac_logits, fr_logit, ev_t, ac_t, fr_t, cfg)
        val_losses.append(loss.item())

        for i in range(eeg.shape[0]):
            pid = batch['pid'][i]
            ev_np = ev_logits[i].cpu().numpy()
            ac_np = ac_logits[i].cpu().numpy()
            fr_val = fr_logit[i].item()

            s1 = decode_strategy1(ev_np, ac_np, fr_val, cfg)
            s2 = decode_strategy2_dp(ev_np, ac_np, fr_val, cfg)

            for pdict, preds in [(s1_preds, s1), (s2_preds, s2)]:
                if pid in pdict:
                    pdict[pid] = merge_hemisphere_preds(pdict[pid], preds)
                else:
                    pdict[pid] = preds

    val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
    gt_filtered = {pid: t for pid, t in gt_lookup.items() if len(t) >= 2 and pid in s1_preds}
    m1 = compute_metrics(s1_preds, gt_filtered, cfg['match_tolerance_s'])
    m1['val_loss'] = val_loss
    m2 = compute_metrics(s2_preds, gt_filtered, cfg['match_tolerance_s'])
    m2['val_loss'] = val_loss
    return m1, m2


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_fold(fold_k, train_indices, val_indices, full_dataset, gt_lookup, cfg, device, save_dir):
    print(f"\n{'='*70}")
    print(f"  Fold {fold_k+1}/{cfg['n_folds']}")
    print(f"  Train: {len(train_indices)} examples  |  Val: {len(val_indices)} examples")
    print(f"{'='*70}")

    class AugSubset(Subset):
        def __getitem__(self, idx):
            old_aug = self.dataset.augment
            self.dataset.augment = True
            result = self.dataset[self.indices[idx]]
            self.dataset.augment = old_aug
            return result

    train_ds_aug = AugSubset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_ds_aug, batch_size=cfg['batch_size'],
                              shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'],
                            shuffle=False, num_workers=cfg['num_workers'],
                            pin_memory=False)

    # Build model
    model = build_model(cfg, device)
    print(f"  Model: {cfg['model_class']} ({count_parameters(model):,} params)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                                  weight_decay=cfg['weight_decay'])

    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    epoch_log = []

    for epoch in range(cfg['n_epochs']):
        lr_now = get_lr(epoch, cfg['n_epochs'], cfg['lr'], cfg['warmup_epochs'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Train
        model.train()
        train_losses = defaultdict(list)
        t_ep = time.time()

        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            ev_t = batch['event_t'].to(device)
            ac_t = batch['active_t'].to(device)
            fr_t = batch['freq_t'].to(device)

            optimizer.zero_grad()
            ev_logits, ac_logits, fr_logit = model(eeg)
            loss, components = total_loss(ev_logits, ac_logits, fr_logit, ev_t, ac_t, fr_t, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()

            for k, v in components.items():
                train_losses[k].append(v)

        # Validate
        val_m1, val_m2 = {}, {}
        if (epoch + 1) % cfg['eval_every'] == 0 or epoch == cfg['n_epochs'] - 1:
            val_m1, val_m2 = validate_fold(model, val_loader, gt_lookup, cfg, device)
            model.train()

            f1 = val_m1['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                no_improve = 0
                fold_dir = save_dir / f'fold{fold_k}'
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_metrics_s1': val_m1,
                    'val_metrics_s2': val_m2,
                    'config': cfg,
                }, str(fold_dir / 'best_model.pt'))
            else:
                no_improve += 1

        mean_train = {k: float(np.mean(v)) for k, v in train_losses.items()}
        epoch_info = {
            'epoch': epoch + 1,
            'lr': lr_now,
            'train_loss': mean_train.get('total', float('nan')),
            'time_s': round(time.time() - t_ep, 1),
        }
        if val_m1:
            epoch_info.update({
                'val_loss': val_m1.get('val_loss', float('nan')),
                'val_f1_s1': val_m1.get('f1', float('nan')),
                'val_f1_s2': val_m2.get('f1', float('nan')),
            })
        epoch_log.append(epoch_info)

        val_str = ''
        if val_m1:
            val_str = (f"  S1_F1={val_m1.get('f1', 0):.4f}"
                       f"  S2_F1={val_m2.get('f1', 0):.4f}")
        print(f"  Ep {epoch+1:3d}/{cfg['n_epochs']}  lr={lr_now:.1e}"
              f"  train={mean_train.get('total', float('nan')):.4f}"
              f"  {time.time()-t_ep:.1f}s{val_str}")

        if val_m1 and no_improve >= cfg['early_stop_patience']:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\n  Fold {fold_k+1} best F1={best_f1:.4f} at epoch {best_epoch}")

    # Save training log
    fold_dir = save_dir / f'fold{fold_k}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    with open(str(fold_dir / 'training_log.json'), 'w') as f:
        json.dump(epoch_log, f, indent=2)

    return {'fold': fold_k, 'best_f1': best_f1, 'best_epoch': best_epoch, 'epoch_log': epoch_log}


# ── Full Evaluation After Training ────────────────────────────────────────────

def evaluate_all_folds(cfg, full_dataset, gt_lookup, device, save_dir):
    """Load best models from all folds and do full evaluation."""
    import json

    with open(cfg['hpp_path']) as f:
        hpp_data = json.load(f)
    gt_data = {k: v for k, v in hpp_data.items() if v.get('review_status') == 'ground_truth'}
    patients_df = pd.read_csv(cfg['patients_csv'])
    patients_df['patient_id'] = patients_df['patient_id'].astype(str)

    patient_ids = get_patient_ids(full_dataset)
    subtypes = get_patient_subtypes(full_dataset, gt_data)
    unique_pids = list(dict.fromkeys(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    pid_subtypes = {pid: sub for pid, sub in zip(patient_ids, subtypes)}
    unique_subtypes = [pid_subtypes.get(pid, 'lpd') for pid in unique_pids]
    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    example_pid_indices = [pid_to_idx[pid] for pid in patient_ids]

    all_s1_preds = {}
    all_s2_preds = {}
    fold_results_s1 = []
    fold_results_s2 = []

    for fold_k, (_, val_pat_idx) in enumerate(skf.split(unique_pids, unique_subtypes)):
        model_path = save_dir / f'fold{fold_k}' / 'best_model.pt'
        if not model_path.exists():
            print(f"  Fold {fold_k}: model not found, skipping")
            continue

        print(f"  Fold {fold_k+1}: loading {model_path}")
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)

        model = build_model(cfg, device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        val_pat_set = set(val_pat_idx)
        val_idx = [i for i, pidx in enumerate(example_pid_indices) if pidx in val_pat_set]
        val_ds = Subset(full_dataset, val_idx)
        val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                                num_workers=cfg['num_workers'])

        fold_s1, fold_s2 = {}, {}
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                ev_logits, ac_logits, fr_logit = model(eeg)
                for i in range(eeg.shape[0]):
                    pid = batch['pid'][i]
                    ev_np = ev_logits[i].cpu().numpy()
                    ac_np = ac_logits[i].cpu().numpy()
                    fr_val = fr_logit[i].item()
                    s1 = decode_strategy1(ev_np, ac_np, fr_val, cfg)
                    s2 = decode_strategy2_dp(ev_np, ac_np, fr_val, cfg)
                    for pdict, preds in [(fold_s1, s1), (fold_s2, s2)]:
                        if pid in pdict:
                            pdict[pid] = merge_hemisphere_preds(pdict[pid], preds)
                        else:
                            pdict[pid] = preds

        fold_gt = {pid: gt_lookup[pid] for pid in fold_s1 if pid in gt_lookup}
        r1 = compute_metrics(fold_s1, fold_gt, cfg['match_tolerance_s'])
        r2 = compute_metrics(fold_s2, fold_gt, cfg['match_tolerance_s'])
        print(f"  Fold {fold_k+1}  S1: F1={r1['f1']:.4f} Sens={r1['sensitivity']:.4f} "
              f"Prec={r1['precision']:.4f} FreqRho={r1['freq_spearman']}")
        print(f"  Fold {fold_k+1}  S2: F1={r2['f1']:.4f} Sens={r2['sensitivity']:.4f} "
              f"Prec={r2['precision']:.4f} FreqRho={r2['freq_spearman']}")
        fold_results_s1.append(r1)
        fold_results_s2.append(r2)
        all_s1_preds.update(fold_s1)
        all_s2_preds.update(fold_s2)

    print(f"\n{'='*70}")
    print(f"  EXPERIMENT {cfg['experiment']} FINAL RESULTS")
    print(f"{'='*70}")

    global_s1 = compute_metrics(all_s1_preds, gt_lookup, cfg['match_tolerance_s'])
    global_s2 = compute_metrics(all_s2_preds, gt_lookup, cfg['match_tolerance_s'])

    print(f"\n  Strategy 1 (Peak picking):")
    print(f"    F1={global_s1['f1']:.4f}  Sens={global_s1['sensitivity']:.4f}  "
          f"Prec={global_s1['precision']:.4f}  FreqRho={global_s1['freq_spearman']}")
    print(f"    TP={global_s1['tp']} FN={global_s1['fn']} FP={global_s1['fp']}")
    print(f"\n  Strategy 2 (DP post-processing):")
    print(f"    F1={global_s2['f1']:.4f}  Sens={global_s2['sensitivity']:.4f}  "
          f"Prec={global_s2['precision']:.4f}  FreqRho={global_s2['freq_spearman']}")
    print(f"    TP={global_s2['tp']} FN={global_s2['fn']} FP={global_s2['fp']}")
    print(f"\n  Per-hemisphere baseline: F1=0.672")
    print(f"  Design A (Exp 1.1) baseline: F1=0.624 (overfit)")

    if fold_results_s1:
        s1_f1s = [r['f1'] for r in fold_results_s1]
        s2_f1s = [r['f1'] for r in fold_results_s2]
        print(f"\n  Per-fold F1 (S1): {[round(f, 4) for f in s1_f1s]} => {np.mean(s1_f1s):.4f}±{np.std(s1_f1s):.4f}")
        print(f"  Per-fold F1 (S2): {[round(f, 4) for f in s2_f1s]} => {np.mean(s2_f1s):.4f}±{np.std(s2_f1s):.4f}")

    results = {
        'experiment': cfg['experiment'],
        'description': cfg['description'],
        'strategy1_global': global_s1,
        'strategy2_global': global_s2,
        'strategy1_per_fold': fold_results_s1,
        'strategy2_per_fold': fold_results_s2,
        'baseline_hemi_f1': 0.672,
        'baseline_exp11_f1': 0.624,
    }
    save_path = save_dir / 'eval_results.json'
    with open(str(save_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_path}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train HemiNet variants')
    parser.add_argument('--exp', type=str, required=True, choices=['1.2', '1.4', '1.5'],
                        help='Experiment ID: 1.2 (Design B), 1.4 (Design D), 1.5 (MAE finetune)')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, only evaluate saved models')
    args = parser.parse_args()

    cfg = EXP_CONFIGS[args.exp]
    t0 = time.time()

    print(f"\n{'='*70}")
    print(f"  Experiment {cfg['experiment']}: {cfg['description']}")
    print(f"{'='*70}")

    # Device
    if cfg['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using MPS GPU")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print(f"Using CPU")
    cfg['device_str'] = str(device)

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # Load data
    print(f"\nLoading data...")
    with open(cfg['hpp_path']) as f:
        hpp_data = json.load(f)
    gt_data = {k: v for k, v in hpp_data.items() if v.get('review_status') == 'ground_truth'}
    print(f"GT cases: {len(gt_data)}")

    gt_lookup = {pid: sorted(v['global_times']) for pid, v in gt_data.items()
                 if len(v.get('global_times', [])) >= 2}

    patients_df = pd.read_csv(cfg['patients_csv'])
    patients_df['patient_id'] = patients_df['patient_id'].astype(str)
    eeg_dir = Path(cfg['eeg_dir'])

    full_dataset = HemiDataset(hpp_data=gt_data, eeg_dir=eeg_dir,
                               patients_df=patients_df, augment=False)
    print(f"Total examples: {len(full_dataset)}")

    # Print model info
    sample_model = build_model(cfg, torch.device('cpu'))
    n_params = count_parameters(sample_model)
    print(f"Model parameters: {n_params:,}")
    del sample_model

    save_dir = Path(cfg['cache_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(str(save_dir / 'config.json'), 'w') as f:
        json.dump({k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool, list))}, f, indent=2)

    # CV split
    patient_ids = get_patient_ids(full_dataset)
    subtypes = get_patient_subtypes(full_dataset, gt_data)
    unique_pids = list(dict.fromkeys(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    pid_subtypes = {pid: sub for pid, sub in zip(patient_ids, subtypes)}
    unique_subtypes = [pid_subtypes.get(pid, 'lpd') for pid in unique_pids]
    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])
    example_pid_indices = [pid_to_idx[pid] for pid in patient_ids]

    if not args.eval_only:
        fold_results = []
        for fold_k, (train_pat_idx, val_pat_idx) in enumerate(skf.split(unique_pids, unique_subtypes)):
            train_set = set(train_pat_idx)
            val_set = set(val_pat_idx)
            train_idx = [i for i, pidx in enumerate(example_pid_indices) if pidx in train_set]
            val_idx = [i for i, pidx in enumerate(example_pid_indices) if pidx in val_set]

            result = train_fold(fold_k, train_idx, val_idx, full_dataset,
                                gt_lookup, cfg, device, save_dir)
            fold_results.append(result)

        f1_scores = [r['best_f1'] for r in fold_results]
        print(f"\nTraining complete!")
        print(f"  Per-fold F1: {[round(r['best_f1'], 4) for r in fold_results]}")
        print(f"  Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"  Total time: {(time.time()-t0)/60:.1f} min")

        cv_summary = {
            'experiment': cfg['experiment'],
            'description': cfg['description'],
            'fold_results': [{'fold': r['fold'], 'best_f1': r['best_f1'], 'best_epoch': r['best_epoch']}
                             for r in fold_results],
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'total_time_min': round((time.time() - t0) / 60, 1),
        }
        with open(str(save_dir / 'cv_results.json'), 'w') as f:
            json.dump(cv_summary, f, indent=2)

    # Evaluate
    print(f"\nRunning final evaluation...")
    evaluate_all_folds(cfg, full_dataset, gt_lookup, device, save_dir)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
