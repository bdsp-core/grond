"""
HemiNet Training Script — Experiment 1.1

5-fold patient-stratified cross-validation.
Trains HemiNet (Design A: U-Net + Transformer bottleneck) on 8-channel hemisphere EEG.

Usage:
    conda run -n foe_dl python code/hemi_detector/train.py

Saves:
    data/hemi_cache/exp1_1/
        fold{k}/best_model.pt
        training_log.json
        cv_results.json
"""

import sys
import json
import time
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.model import HemiNet, count_parameters
from hemi_detector.dataset import HemiDataset, get_patient_ids, get_patient_subtypes

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    # Data
    'hpp_path': str(PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'),
    # patients.csv was moved to data/labels/archive_labels/ during the
    # May-2026 cleanup (it's the legacy per-patient aggregate, superseded
    # by segment_labels.csv but still consumed here by ChannelPD-Net training).
    'patients_csv': str(PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'),
    'eeg_dir': str(PROJECT_DIR / 'data' / 'eeg'),
    'cache_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_1'),

    # Training
    'n_folds': 5,
    'n_epochs': 100,
    'batch_size': 32,
    'lr': 5e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 5,
    'grad_clip': 1.0,
    'eval_every': 5,    # epochs
    'early_stop_patience': 20,  # in eval intervals (100 epochs max)

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
    'active_min_bins': 50,  # 0.5s
    'active_gamma': 1.5,
    'match_tolerance_s': 0.1,
    'fs_target': 100,  # Hz

    # Device
    'device': 'mps',  # will fall back to cpu if unavailable

    # Misc
    'num_workers': 0,  # MPS works best with 0 workers
    'seed': 42,
}


# ── Loss Functions ────────────────────────────────────────────────────────────

def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> torch.Tensor:
    """Focal Binary Cross-Entropy loss.

    Parameters
    ----------
    logits  : raw logits (any shape)
    targets : soft targets in [0, 1] (same shape)
    gamma   : focusing parameter
    alpha   : positive class weight
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = torch.exp(-bce)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-4,
) -> torch.Tensor:
    """Soft Dice loss for binary segmentation."""
    p = torch.sigmoid(logits)
    num = 2.0 * (p * targets).sum() + smooth
    den = p.sum() + targets.sum() + smooth
    return 1.0 - num / den


def event_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    dice_w: float = 0.5,
) -> torch.Tensor:
    """Focal BCE + weighted Soft Dice for event detection."""
    l_focal = focal_bce_loss(logits, targets, gamma=gamma, alpha=alpha)
    l_dice = soft_dice_loss(logits, targets)
    return l_focal + dice_w * l_dice


def active_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


def freq_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.2) -> torch.Tensor:
    """Huber loss on log-frequency. Only uses samples with finite targets."""
    mask = torch.isfinite(target.squeeze(-1)) if target.dim() > 1 else torch.isfinite(target)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    p = pred.squeeze(-1)[mask]
    t = target.squeeze(-1)[mask]
    return F.huber_loss(p, t, delta=delta)


def total_loss(
    event_logits, active_logits, freq_logit,
    event_t, active_t, freq_t,
    cfg: Dict,
) -> Tuple[torch.Tensor, Dict]:
    l_ev = event_loss(
        event_logits, event_t,
        gamma=cfg['focal_gamma'],
        alpha=cfg['focal_alpha'],
        dice_w=cfg['dice_weight'],
    )
    l_ac = active_loss(active_logits, active_t)
    l_fr = freq_loss(freq_logit, freq_t, delta=cfg['huber_delta'])

    loss = (
        cfg['w_event'] * l_ev +
        cfg['w_active'] * l_ac +
        cfg['w_freq'] * l_fr
    )
    components = {
        'event': l_ev.item(),
        'active': l_ac.item(),
        'freq': l_fr.item(),
        'total': loss.item(),
    }
    return loss, components


# ── Learning Rate Schedule ─────────────────────────────────────────────────────

def get_lr(epoch: int, n_epochs: int, lr: float, warmup_epochs: int) -> float:
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── Evaluation ────────────────────────────────────────────────────────────────

def decode_predictions(
    event_logits: np.ndarray,
    active_logits: np.ndarray,
    freq_logit: float,
    cfg: Dict,
) -> List[float]:
    """Decode model outputs to discharge times in seconds."""
    p_event = 1.0 / (1.0 + np.exp(-event_logits))   # (1000,)
    p_active = 1.0 / (1.0 + np.exp(-active_logits))  # (1000,)

    # Effective probability
    p_eff = p_event * (p_active ** cfg['active_gamma'])

    # Active mask
    active_mask = np.zeros(len(p_active), dtype=bool)
    active_bin = (p_active > cfg['active_threshold']).astype(np.float32)
    # Erode short active regions (min_bins)
    from scipy.ndimage import label as ndlabel
    labeled, n = ndlabel(active_bin)
    for region_id in range(1, n + 1):
        region_mask = (labeled == region_id)
        if region_mask.sum() >= cfg['active_min_bins']:
            active_mask |= region_mask

    # If no active region found, use full trace
    if not active_mask.any():
        active_mask[:] = True

    # Mask p_eff outside active regions
    p_masked = p_eff.copy()
    p_masked[~active_mask] = 0.0

    # Peak detection with min_distance from predicted frequency
    pred_freq = float(np.clip(np.exp(freq_logit), 0.2, 5.0))
    min_dist_bins = max(3, int(cfg['fs_target'] / (pred_freq * 2.0)))

    peaks, _ = find_peaks(
        p_masked,
        height=cfg['peak_min_height'],
        distance=min_dist_bins,
    )

    discharge_times = [float(pk / cfg['fs_target']) for pk in peaks]
    return discharge_times


def compute_f1(
    all_preds: Dict[str, List[float]],
    all_gt: Dict[str, List[float]],
    tolerance_s: float = 0.1,
) -> Dict:
    """Compute event detection F1 with ±tolerance_s matching."""
    tp, fn, fp = 0, 0, 0
    for pid, gt_times in all_gt.items():
        pred_times = all_preds.get(pid, [])
        gt_sorted = sorted(gt_times)
        pred_sorted = sorted(pred_times)

        gt_matched = [False] * len(gt_sorted)
        pred_matched = [False] * len(pred_sorted)

        for gi, gt in enumerate(gt_sorted):
            best_dist, best_pi = np.inf, -1
            for pi, pt in enumerate(pred_sorted):
                if not pred_matched[pi]:
                    d = abs(gt - pt)
                    if d < best_dist:
                        best_dist, best_pi = d, pi
            if best_dist <= tolerance_s and best_pi >= 0:
                gt_matched[gi] = True
                pred_matched[best_pi] = True

        tp += sum(gt_matched)
        fn += len(gt_sorted) - sum(gt_matched)
        fp += len(pred_sorted) - sum(pred_matched)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
    return {'f1': f1, 'sensitivity': sens, 'precision': prec,
            'tp': tp, 'fn': fn, 'fp': fp}


@torch.no_grad()
def validate_fold(
    model: nn.Module,
    val_loader: DataLoader,
    gt_lookup: Dict[str, List[float]],
    cfg: Dict,
    device: torch.device,
) -> Dict:
    """Run validation: compute loss and event F1."""
    model.eval()
    val_losses = []
    all_preds: Dict[str, List[float]] = {}

    for batch in val_loader:
        eeg = batch['eeg'].to(device)
        event_t = batch['event_t'].to(device)
        active_t = batch['active_t'].to(device)
        freq_t = batch['freq_t'].to(device)

        ev_logits, ac_logits, fr_logit = model(eeg)
        loss, _ = total_loss(ev_logits, ac_logits, fr_logit,
                              event_t, active_t, freq_t, cfg)
        val_losses.append(loss.item())

        # Decode predictions for each item in batch
        batch_size = eeg.shape[0]
        for i in range(batch_size):
            pid = batch['pid'][i]
            times = decode_predictions(
                ev_logits[i].cpu().numpy(),
                ac_logits[i].cpu().numpy(),
                fr_logit[i].item(),
                cfg,
            )
            # For GPD, two hemispheres → merge predictions by patient
            if pid in all_preds:
                # Merge: union of both hemispheres, then deduplicate
                combined = sorted(all_preds[pid] + times)
                merged = []
                for t in combined:
                    if not merged or t - merged[-1] > 0.05:  # 50ms min separation
                        merged.append(t)
                all_preds[pid] = merged
            else:
                all_preds[pid] = times

    val_loss = np.mean(val_losses) if val_losses else float('nan')

    # Filter to only patients in gt_lookup with ≥2 GT discharges
    gt_filtered = {
        pid: times for pid, times in gt_lookup.items()
        if len(times) >= 2 and pid in all_preds
    }
    metrics = compute_f1(all_preds, gt_filtered, cfg['match_tolerance_s'])
    metrics['val_loss'] = val_loss
    return metrics


# ── Training Loop ─────────────────────────────────────────────────────────────

def train_fold(
    fold_k: int,
    train_indices: List[int],
    val_indices: List[int],
    full_dataset: HemiDataset,
    gt_lookup: Dict[str, List[float]],
    cfg: Dict,
    device: torch.device,
    save_dir: Path,
) -> Dict:
    """Train one fold and return the best validation metrics."""
    print(f"\n{'='*70}")
    print(f"  Fold {fold_k+1}/{cfg['n_folds']}")
    print(f"  Train: {len(train_indices)} examples  |  Val: {len(val_indices)} examples")
    print(f"{'='*70}")

    # Create augmented train dataset, non-augmented val dataset
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)

    # Manually set augment=True for training items
    # We need to access the underlying HemiDataset and patch augment
    # Trick: wrap in a thin augmentation wrapper
    class AugSubset(Subset):
        def __getitem__(self, idx):
            item = self.dataset.examples[self.indices[idx]]
            # Temporarily enable augment
            old_aug = self.dataset.augment
            self.dataset.augment = True
            result = self.dataset[self.indices[idx]]
            self.dataset.augment = old_aug
            return result

    train_ds_aug = AugSubset(full_dataset, train_indices)

    train_loader = DataLoader(
        train_ds_aug,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=False,
    )

    # Model
    model = HemiNet(in_channels=8, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )

    best_f1 = 0.0
    best_epoch = 0
    no_improve = 0
    epoch_log = []

    for epoch in range(cfg['n_epochs']):
        # Set learning rate
        lr_now = get_lr(epoch, cfg['n_epochs'], cfg['lr'], cfg['warmup_epochs'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # ── Training ─────────────────────────────────────────────────
        model.train()
        train_losses = defaultdict(list)
        t_ep = time.time()

        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            event_t = batch['event_t'].to(device)
            active_t = batch['active_t'].to(device)
            freq_t = batch['freq_t'].to(device)

            optimizer.zero_grad()
            ev_logits, ac_logits, fr_logit = model(eeg)
            loss, components = total_loss(
                ev_logits, ac_logits, fr_logit,
                event_t, active_t, freq_t, cfg,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()

            for k, v in components.items():
                train_losses[k].append(v)

        # ── Validation (every eval_every epochs) ─────────────────────
        val_metrics = {}
        if (epoch + 1) % cfg['eval_every'] == 0 or epoch == cfg['n_epochs'] - 1:
            val_metrics = validate_fold(model, val_loader, gt_lookup, cfg, device)
            model.train()  # restore train mode

            f1 = val_metrics['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                no_improve = 0
                # Save best model
                fold_dir = save_dir / f'fold{fold_k}'
                fold_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'val_metrics': val_metrics,
                    'config': cfg,
                }, str(fold_dir / 'best_model.pt'))
            else:
                no_improve += 1

        # ── Logging ──────────────────────────────────────────────────
        mean_train = {k: float(np.mean(v)) for k, v in train_losses.items()}
        epoch_info = {
            'epoch': epoch + 1,
            'lr': lr_now,
            'train_loss': mean_train.get('total', float('nan')),
            'train_event': mean_train.get('event', float('nan')),
            'train_active': mean_train.get('active', float('nan')),
            'train_freq': mean_train.get('freq', float('nan')),
            'time_s': round(time.time() - t_ep, 1),
        }
        if val_metrics:
            epoch_info.update({
                'val_loss': val_metrics.get('val_loss', float('nan')),
                'val_f1': val_metrics.get('f1', float('nan')),
                'val_sens': val_metrics.get('sensitivity', float('nan')),
                'val_prec': val_metrics.get('precision', float('nan')),
            })
        epoch_log.append(epoch_info)

        # Print every epoch
        val_str = ''
        if val_metrics:
            val_str = (f"  val_loss={val_metrics.get('val_loss', float('nan')):.4f}"
                       f"  F1={val_metrics.get('f1', 0):.4f}"
                       f"  Sens={val_metrics.get('sensitivity', 0):.4f}"
                       f"  Prec={val_metrics.get('precision', 0):.4f}")
        print(
            f"  Epoch {epoch+1:3d}/{cfg['n_epochs']}  "
            f"lr={lr_now:.1e}  "
            f"train_loss={mean_train.get('total', float('nan')):.4f}"
            f"  (ev={mean_train.get('event', float('nan')):.4f}"
            f" ac={mean_train.get('active', float('nan')):.4f}"
            f" fr={mean_train.get('freq', float('nan')):.4f})"
            f"  {time.time()-t_ep:.1f}s"
            f"{val_str}"
        )

        # Early stopping (check only when we computed val)
        if val_metrics and no_improve >= cfg['early_stop_patience']:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for "
                  f"{no_improve * cfg['eval_every']} epochs)")
            break

    print(f"\n  Fold {fold_k+1} best F1={best_f1:.4f} at epoch {best_epoch}")
    return {
        'fold': fold_k,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'epoch_log': epoch_log,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    cfg = CONFIG

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

    # Seed
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    # ── Load data ─────────────────────────────────────────────────────
    print(f"\nLoading data...")
    with open(cfg['hpp_path']) as f:
        hpp_data = json.load(f)

    gt_data = {k: v for k, v in hpp_data.items()
               if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_data)}")

    # Ground truth lookup for evaluation
    gt_lookup: Dict[str, List[float]] = {
        pid: sorted(v['global_times'])
        for pid, v in gt_data.items()
        if len(v.get('global_times', [])) >= 2
    }

    patients_df = pd.read_csv(cfg['patients_csv'])
    patients_df['patient_id'] = patients_df['patient_id'].astype(str)

    eeg_dir = Path(cfg['eeg_dir'])

    # Build dataset (augment=False, will be toggled in train_fold)
    full_dataset = HemiDataset(
        hpp_data=gt_data,
        eeg_dir=eeg_dir,
        patients_df=patients_df,
        augment=False,
    )
    print(f"Total examples: {len(full_dataset)}")

    # Print model info
    model_info = HemiNet(in_channels=8)
    n_params = count_parameters(model_info)
    print(f"HemiNet parameters: {n_params:,}")
    del model_info

    # ── Cross-validation split ────────────────────────────────────────
    patient_ids = get_patient_ids(full_dataset)
    subtypes = get_patient_subtypes(full_dataset, gt_data)

    # Unique patients for splitting
    unique_pids = list(dict.fromkeys(patient_ids))  # preserve order
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}

    # Stratify label per patient: use subtype from hpp_data
    pid_subtypes = {}
    for pid, subtype in zip(patient_ids, subtypes):
        pid_subtypes[pid] = subtype

    unique_subtypes = [pid_subtypes.get(pid, 'lpd') for pid in unique_pids]

    skf = StratifiedKFold(n_splits=cfg['n_folds'], shuffle=True, random_state=cfg['seed'])

    fold_results = []
    save_dir = Path(cfg['cache_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(str(save_dir / 'config.json'), 'w') as f:
        json.dump({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool, list))}, f, indent=2)

    # Map from example index to patient index (for fold assignment)
    example_pid_indices = [pid_to_idx[pid] for pid in patient_ids]

    for fold_k, (train_pat_idx, val_pat_idx) in enumerate(
        skf.split(unique_pids, unique_subtypes)
    ):
        train_pat_set = set(train_pat_idx)
        val_pat_set = set(val_pat_idx)

        train_example_idx = [
            i for i, pidx in enumerate(example_pid_indices) if pidx in train_pat_set
        ]
        val_example_idx = [
            i for i, pidx in enumerate(example_pid_indices) if pidx in val_pat_set
        ]

        result = train_fold(
            fold_k=fold_k,
            train_indices=train_example_idx,
            val_indices=val_example_idx,
            full_dataset=full_dataset,
            gt_lookup=gt_lookup,
            cfg=cfg,
            device=device,
            save_dir=save_dir,
        )
        fold_results.append(result)

        # Save training log incrementally
        log_path = save_dir / f'fold{fold_k}' / 'training_log.json'
        with open(str(log_path), 'w') as f:
            json.dump(result['epoch_log'], f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    f1_scores = [r['best_f1'] for r in fold_results]
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT 1.1 RESULTS — HemiNet (Design A)")
    print(f"{'='*70}")
    for k, r in enumerate(fold_results):
        print(f"  Fold {k+1}: best F1={r['best_f1']:.4f} (epoch {r['best_epoch']})")
    print(f"  Mean F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Baseline to beat: F1=0.740 (full 18ch pipeline)")

    # Save CV summary
    cv_summary = {
        'experiment': '1.1',
        'description': 'HemiNet Design A: U-Net + Transformer bottleneck, 8ch input',
        'fold_results': [
            {
                'fold': r['fold'],
                'best_f1': r['best_f1'],
                'best_epoch': r['best_epoch'],
            }
            for r in fold_results
        ],
        'mean_f1': float(np.mean(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'total_time_min': round((time.time() - t0) / 60, 1),
        'baseline_f1': 0.740,
        'n_examples': len(full_dataset),
        'n_params': n_params,
    }
    with open(str(save_dir / 'cv_results.json'), 'w') as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n  Results saved to {save_dir}")

    return cv_summary


if __name__ == '__main__':
    main()
