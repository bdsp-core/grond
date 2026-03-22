"""
PDNetV2 Training - 5-fold patient-stratified cross-validation.

Usage:
    conda run -n foe_dl python code/pdnet_v2/train.py

Saves:
    data/pdnet_v2_cache/fold{k}_best.pt        - best model per fold
    data/pdnet_v2_cache/training_progress.json  - live training log
"""

import sys
import json
import time
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pdnet_v2.model import PDNetV2
from pdnet_v2.dataset import build_dataset, TARGET_FS, N_BINS
from optimization_harness_v2 import _load_mat_as_bipolar

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
CACHE_DIR = DATA_DIR / 'pdnet_v2_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_FOLDS = 5
N_EPOCHS = 50
LR = 3e-4   # reduced from 1e-3 for stability
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16  # start at 16, reduce if OOM
EVAL_EVERY = 5   # compute F1 every N epochs

LOSS_WEIGHTS = {
    'event': 1.0,
    'active': 0.5,
    'freq': 0.05,   # reduced from 0.2 — freq diverges early in training
    'subtype': 0.1,
    'lat': 0.1,
}

# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# ── Loss functions ─────────────────────────────────────────────────────────────

def focal_bce(logits, targets, gamma=2.0, alpha=0.75):
    """Focal binary cross-entropy loss."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def soft_dice(preds, targets, eps=1e-6):
    """Soft Dice loss."""
    intersection = (preds * targets).sum()
    denom = preds.sum() + targets.sum() + eps
    return 1.0 - 2.0 * intersection / denom


def masked_huber(preds, targets, mask, delta=0.2):
    """Huber loss applied only where mask=1."""
    n_valid = mask.sum()
    if n_valid < 1:
        return torch.tensor(0.0, device=preds.device)
    diff = (preds - targets).abs()
    huber = torch.where(diff < delta, 0.5 * diff ** 2 / delta, diff - 0.5 * delta)
    return (huber * mask).sum() / n_valid


def compute_loss(outputs, batch, device):
    """Compute total weighted loss."""
    event_logits, active_logits, freq_loghz, subtype_logits, lat_logits = outputs

    y_event = batch['y_event'].to(device)
    y_active = batch['y_active'].to(device)
    y_freq = batch['y_freq'].to(device)
    y_freq_mask = batch['y_freq_mask'].to(device)
    y_subtype = batch['y_subtype'].to(device)
    y_lat = batch['y_lat'].to(device)

    # Event loss: focal BCE + 0.5 * soft dice
    L_event = focal_bce(event_logits, y_event) + \
               0.5 * soft_dice(torch.sigmoid(event_logits), y_event)

    # Active loss: standard BCE
    L_active = F.binary_cross_entropy_with_logits(active_logits, y_active)

    # Freq loss: masked huber with larger delta for stability
    L_freq = masked_huber(freq_loghz, y_freq, y_freq_mask, delta=1.0)

    # Classification losses
    L_sub = F.cross_entropy(subtype_logits, y_subtype)
    L_lat = F.cross_entropy(lat_logits, y_lat)

    L_total = (LOSS_WEIGHTS['event'] * L_event +
               LOSS_WEIGHTS['active'] * L_active +
               LOSS_WEIGHTS['freq'] * L_freq +
               LOSS_WEIGHTS['subtype'] * L_sub +
               LOSS_WEIGHTS['lat'] * L_lat)

    return L_total, {
        'event': L_event.item(),
        'active': L_active.item(),
        'freq': L_freq.item(),
        'subtype': L_sub.item(),
        'lat': L_lat.item(),
        'total': L_total.item(),
    }


# ── Event F1 evaluation ────────────────────────────────────────────────────────

def decode_predictions(event_logits, active_logits):
    """
    Decode model outputs to discharge times (seconds).

    Args:
        event_logits: (1000,) tensor
        active_logits: (1000,) tensor

    Returns:
        list of discharge times in seconds
    """
    p_event = torch.sigmoid(event_logits).cpu().numpy()
    p_active = torch.sigmoid(active_logits).cpu().numpy()
    p_eff = p_event * (p_active ** 1.5)

    # Find active regions
    active_binary = p_active > 0.4
    min_len = 50  # 0.5s at 100 Hz

    # Label connected components
    discharge_times = []
    in_run = False
    run_start = 0

    def process_run(start, end):
        if end - start < min_len:
            return []
        p_run = p_eff[start:end]
        if p_run.max() < 0.15:
            return []
        # Estimate median frequency from event peaks in this run
        init_peaks, _ = find_peaks(p_run, height=0.15)
        if len(init_peaks) >= 2:
            ipis = np.diff(init_peaks)
            median_ipi = float(np.median(ipis))
            min_dist = max(10, int(median_ipi * 0.6))
        else:
            min_dist = 20  # default ~0.2s at 100 Hz

        peaks, props = find_peaks(p_run, height=0.25, distance=min_dist)
        times = [(start + pk) / TARGET_FS for pk in peaks]
        return times

    for i in range(len(active_binary)):
        if active_binary[i] and not in_run:
            in_run = True
            run_start = i
        elif not active_binary[i] and in_run:
            times = process_run(run_start, i)
            discharge_times.extend(times)
            in_run = False

    if in_run:
        times = process_run(run_start, len(active_binary))
        discharge_times.extend(times)

    return sorted(discharge_times)


def compute_event_f1(model, dataloader, device, tol=0.1):
    """Compute discharge timing F1 across all validation patients."""
    model.eval()
    all_tp = 0
    all_fp = 0
    all_fn = 0

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            outputs = model(eeg)
            event_logits, active_logits = outputs[0], outputs[1]

            for i in range(len(eeg)):
                # Ground truth: decode from y_event (peak times)
                y_event = batch['y_event'][i].cpu().numpy()
                from scipy.signal import find_peaks as fp
                gt_peaks, _ = fp(y_event, height=0.5)
                gt_times = [pk / TARGET_FS for pk in gt_peaks]

                # Predicted times
                pred_times = decode_predictions(event_logits[i], active_logits[i])

                # Match with tolerance
                tp, fp_, fn = _match_events(gt_times, pred_times, tol)
                all_tp += tp
                all_fp += fp_
                all_fn += fn

    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_tp + all_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def _match_events(gt_times, pred_times, tol):
    """Match predicted events to ground truth with tolerance."""
    if not gt_times:
        return 0, len(pred_times), 0
    if not pred_times:
        return 0, 0, len(gt_times)

    gt_arr = np.array(sorted(gt_times))
    pred_arr = np.array(sorted(pred_times))

    matched_gt = set()
    matched_pred = set()

    for j, pt in enumerate(pred_arr):
        dists = np.abs(gt_arr - pt)
        closest = np.argmin(dists)
        if dists[closest] <= tol and closest not in matched_gt:
            matched_gt.add(closest)
            matched_pred.add(j)

    tp = len(matched_gt)
    fp = len(pred_arr) - tp
    fn = len(gt_arr) - tp
    return tp, fp, fn


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_segments(verbose=True):
    """Load all EEG segments from disk."""
    df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_seg['patient_id'] = df_seg['patient_id'].astype(str)

    segments_by_patient = {}
    n_loaded = 0
    n_failed = 0

    for pid, group in df_seg.groupby('patient_id'):
        segs = []
        for _, row in group.iterrows():
            mat_path = EEG_DIR / row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(mat_path, row['montage'], int(row['n_channels']))
                # Ensure float32
                seg = np.array(seg, dtype=np.float32)
                segs.append(seg)
                n_loaded += 1
            except Exception as e:
                n_failed += 1
        if segs:
            segments_by_patient[str(pid)] = segs[:5]  # max 5 per patient

    if verbose:
        print(f"Loaded {n_loaded} segments from {len(segments_by_patient)} patients "
              f"({n_failed} failed)")
    return segments_by_patient


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    loss_components = {'event': 0, 'active': 0, 'freq': 0, 'subtype': 0, 'lat': 0, 'total': 0}
    n_batches = 0

    for batch in loader:
        eeg = batch['eeg'].to(device)

        optimizer.zero_grad()
        outputs = model(eeg)
        loss, components = compute_loss(outputs, batch, device)

        # Skip NaN batches
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        # Check for NaN gradients and skip if found
        has_nan_grad = False
        for p in model.parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                break
        if has_nan_grad:
            optimizer.zero_grad()
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
        n_batches += 1

    # Average
    for k in loss_components:
        loss_components[k] /= max(n_batches, 1)

    return loss_components


def validate_loss(model, loader, device):
    """Compute validation loss."""
    model.eval()
    loss_components = {'event': 0, 'active': 0, 'freq': 0, 'subtype': 0, 'lat': 0, 'total': 0}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            outputs = model(eeg)
            _, components = compute_loss(outputs, batch, device)
            for k, v in components.items():
                loss_components[k] += v
            n_batches += 1

    for k in loss_components:
        loss_components[k] /= max(n_batches, 1)

    return loss_components


# ── Main training ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PDNetV2 Training - 5-fold Cross-Validation")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    print("\nLoading patients and labels...")
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)

    with open(str(DATA_DIR / 'labels' / 'discharge_times_hpp.json')) as f:
        hpp_data = json.load(f)

    # Filter to valid patients (ground_truth, >=2 discharges, lpd/gpd)
    valid_pids = []
    valid_subtypes = []
    for pid, hpp in hpp_data.items():
        if hpp.get('review_status') != 'ground_truth':
            continue
        if len(hpp.get('global_times', [])) < 2:
            continue
        subtype = str(hpp.get('subtype', '')).lower()
        # Also check patients.csv subtype
        pat_rows = df_patients[df_patients['patient_id'] == pid]
        if len(pat_rows) > 0:
            subtype = str(pat_rows.iloc[0]['subtype']).lower()
        if subtype not in ('lpd', 'gpd'):
            continue
        valid_pids.append(pid)
        valid_subtypes.append(subtype)

    print(f"Valid patients (lpd/gpd with discharge times): {len(valid_pids)}")
    n_lpd = sum(1 for s in valid_subtypes if s == 'lpd')
    n_gpd = sum(1 for s in valid_subtypes if s == 'gpd')
    print(f"  LPD: {n_lpd}, GPD: {n_gpd}")

    # Load EEG segments
    print("\nLoading EEG segments...")
    t0 = time.time()
    segments_by_patient = load_all_segments(verbose=True)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Filter valid_pids to those with EEG data
    valid_pids_with_eeg = [pid for pid in valid_pids if pid in segments_by_patient]
    valid_subtypes_with_eeg = [valid_subtypes[valid_pids.index(pid)] for pid in valid_pids_with_eeg]
    print(f"Patients with EEG data: {len(valid_pids_with_eeg)}")

    # Cross-validation splits
    valid_pids_arr = np.array(valid_pids_with_eeg)
    subtype_labels = np.array([0 if s == 'lpd' else 1 for s in valid_subtypes_with_eeg])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(valid_pids_arr, subtype_labels))

    # Progress tracking
    progress = {
        'folds': [],
        'best_f1_per_fold': [],
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    progress_path = CACHE_DIR / 'training_progress.json'

    # ── Train each fold ──────────────────────────────────────────────────────
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*70}")

        train_pids = valid_pids_arr[train_idx].tolist()
        val_pids = valid_pids_arr[val_idx].tolist()
        print(f"Train: {len(train_pids)}, Val: {len(val_pids)}")

        # Build datasets
        train_ds = build_dataset(train_pids, segments_by_patient, hpp_data, df_patients, augment=True)
        val_ds = build_dataset(val_pids, segments_by_patient, hpp_data, df_patients, augment=False)
        print(f"Train items: {len(train_ds)}, Val items: {len(val_ds)}")

        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"WARNING: Empty dataset for fold {fold_idx+1}, skipping")
            continue

        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=False)

        # Model, optimizer, scheduler
        model = PDNetV2().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_val_f1 = -1.0
        best_epoch = 0
        fold_log = []

        for epoch in range(1, N_EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_losses = train_one_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            # Validation loss
            val_losses = validate_loss(model, val_loader, device)

            epoch_time = time.time() - epoch_start
            lr_now = optimizer.param_groups[0]['lr']

            # Compute F1 every EVAL_EVERY epochs
            val_f1 = None
            if epoch % EVAL_EVERY == 0 or epoch == N_EPOCHS:
                val_f1, val_prec, val_rec = compute_event_f1(model, val_loader, device)

                print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
                      f"Train: {train_losses['total']:.4f} "
                      f"(ev={train_losses['event']:.3f}, ac={train_losses['active']:.3f}) | "
                      f"Val: {val_losses['total']:.4f} | "
                      f"F1={val_f1:.4f} P={val_prec:.4f} R={val_rec:.4f} | "
                      f"lr={lr_now:.2e} | {epoch_time:.1f}s")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    save_path = CACHE_DIR / f'fold{fold_idx}_best.pt'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_f1': val_f1,
                        'val_precision': val_prec,
                        'val_recall': val_rec,
                        'fold': fold_idx,
                        'train_pids': train_pids,
                        'val_pids': val_pids,
                    }, str(save_path))
                    print(f"    *** New best F1={val_f1:.4f} at epoch {epoch} — saved ***")
            else:
                print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
                      f"Train: {train_losses['total']:.4f} "
                      f"(ev={train_losses['event']:.3f}, ac={train_losses['active']:.3f}) | "
                      f"Val: {val_losses['total']:.4f} | "
                      f"lr={lr_now:.2e} | {epoch_time:.1f}s")

            log_entry = {
                'epoch': epoch,
                'train_loss': train_losses,
                'val_loss': val_losses,
                'val_f1': val_f1,
                'lr': lr_now,
            }
            fold_log.append(log_entry)

        print(f"\nFold {fold_idx+1} complete. Best F1={best_val_f1:.4f} at epoch {best_epoch}")

        # Update progress file
        progress['folds'].append({
            'fold': fold_idx,
            'best_val_f1': best_val_f1,
            'best_epoch': best_epoch,
            'n_train': len(train_ds),
            'n_val': len(val_ds),
            'log': fold_log[-10:],  # last 10 epochs
        })
        progress['best_f1_per_fold'].append(best_val_f1)
        with open(str(progress_path), 'w') as f:
            json.dump(progress, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    if progress['best_f1_per_fold']:
        mean_f1 = np.mean(progress['best_f1_per_fold'])
        std_f1 = np.std(progress['best_f1_per_fold'])
        print(f"Mean validation F1: {mean_f1:.4f} ± {std_f1:.4f}")
        for i, f1 in enumerate(progress['best_f1_per_fold']):
            print(f"  Fold {i+1}: {f1:.4f}")
    print(f"\nProgress saved to: {progress_path}")
    print(f"Models saved to: {CACHE_DIR}/fold*_best.pt")

    progress['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(str(progress_path), 'w') as f:
        json.dump(progress, f, indent=2, default=str)


if __name__ == '__main__':
    main()
