#!/usr/bin/env python
"""
Training script for E2E Discharge Detector.

Phase 1: Freeze backbone, train decoder + heads, 50 epochs, lr=1e-3
Phase 2: Unfreeze backbone (lr=1e-5), decoder lr=1e-4, 100 epochs

Usage:
    python code/e2e_model/train_e2e.py --phase 1
    python code/e2e_model/train_e2e.py --phase 2 --resume
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from e2e_model.model import E2EDischargeDetector
from e2e_model.dataset import E2EDataset, build_sample_list, custom_collate, load_hpp_cache
from e2e_model.loss import HungarianMatchingLoss

CACHE_DIR = PROJECT_DIR / 'data' / 'e2e_cache'
CACHE_DIR.mkdir(exist_ok=True)
PRETRAINED_PATH = PROJECT_DIR / 'data' / 'pd_channel_cache' / 'cnn_attn_fold0.pt'


def get_device():
    """Get best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_timing_metrics(pred_times, pred_confs, gt_times, gt_mask,
                           tolerance_s=0.15):
    """Compute timing F1, sensitivity, precision, MAE.

    Args:
        pred_times: (n_queries,) numpy
        pred_confs: (n_queries,) numpy
        gt_times: (max_gt,) numpy
        gt_mask: (max_gt,) numpy
        tolerance_s: matching tolerance in seconds

    Returns:
        dict with f1, sensitivity, precision, mae
    """
    # Filter predictions by confidence
    conf_threshold = 0.5
    active = pred_confs > conf_threshold
    pred_t = np.sort(pred_times[active])

    n_gt = int(gt_mask.sum())
    gt_t = gt_times[:n_gt]

    if len(pred_t) == 0 and n_gt == 0:
        return {'f1': 1.0, 'sensitivity': 1.0, 'precision': 1.0, 'mae': 0.0}
    if len(pred_t) == 0:
        return {'f1': 0.0, 'sensitivity': 0.0, 'precision': 0.0, 'mae': float('nan')}
    if n_gt == 0:
        return {'f1': 0.0, 'sensitivity': 0.0, 'precision': 0.0, 'mae': float('nan')}

    # Match predictions to GT (greedy nearest neighbor)
    matched_gt = set()
    matched_pred = set()
    errors = []

    for i, pt in enumerate(pred_t):
        dists = np.abs(gt_t - pt)
        best_j = np.argmin(dists)
        if dists[best_j] <= tolerance_s and best_j not in matched_gt:
            matched_gt.add(best_j)
            matched_pred.add(i)
            errors.append(dists[best_j])

    tp = len(matched_gt)
    fp = len(pred_t) - tp
    fn = n_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    mae = np.mean(errors) if errors else float('nan')

    return {'f1': f1, 'sensitivity': sensitivity, 'precision': precision, 'mae': mae}


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataloader."""
    model.eval()
    total_loss = 0.0
    loss_components = {}
    timing_metrics = []
    freq_preds = []
    freq_gts = []
    lat_preds = []
    lat_gts = []

    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            hpp = batch['hpp'].to(device)

            outputs = model(eeg, hpp)
            loss, loss_dict = criterion(outputs, {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            })

            total_loss += loss.item() * eeg.shape[0]
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v * eeg.shape[0]

            # Timing metrics
            pred_t = outputs['pred_times'].cpu().numpy()
            pred_c = outputs['pred_confs'].cpu().numpy()
            gt_t = batch['gt_times'].numpy()
            gt_m = batch['gt_mask'].numpy()

            for b in range(eeg.shape[0]):
                tm = compute_timing_metrics(pred_t[b], pred_c[b], gt_t[b], gt_m[b])
                timing_metrics.append(tm)

                # Frequency
                if batch['freq'][b].item() > 0:
                    freq_preds.append(outputs['pred_freq'][b].cpu().item())
                    freq_gts.append(batch['freq'][b].item())

                # Laterality
                if batch['lat'][b].item() >= 0:
                    lat_preds.append(
                        torch.sigmoid(outputs['lat_logit'][b] * 5.0).cpu().item()
                    )
                    lat_gts.append(batch['lat'][b].item())

    n = len(timing_metrics)
    avg_loss = total_loss / max(n, 1)
    avg_components = {k: v / max(n, 1) for k, v in loss_components.items()}

    avg_f1 = np.mean([m['f1'] for m in timing_metrics])
    avg_sens = np.mean([m['sensitivity'] for m in timing_metrics])
    avg_prec = np.mean([m['precision'] for m in timing_metrics])
    mae_vals = [m['mae'] for m in timing_metrics if not np.isnan(m['mae'])]
    avg_mae = np.mean(mae_vals) if mae_vals else float('nan')

    # Frequency Spearman correlation
    from scipy.stats import spearmanr
    freq_rho = 0.0
    if len(freq_preds) >= 5:
        freq_rho, _ = spearmanr(freq_preds, freq_gts)
        if np.isnan(freq_rho):
            freq_rho = 0.0

    # Laterality AUC
    lat_auc = 0.0
    if len(lat_preds) >= 5:
        from sklearn.metrics import roc_auc_score
        try:
            # Binary: right=1, left=0
            lat_binary = [1 if g > 0.5 else 0 for g in lat_gts]
            if len(set(lat_binary)) > 1:
                lat_auc = roc_auc_score(lat_binary, lat_preds)
        except Exception:
            pass

    metrics = {
        'loss': avg_loss,
        'loss_components': avg_components,
        'timing_f1': avg_f1,
        'timing_sens': avg_sens,
        'timing_prec': avg_prec,
        'timing_mae': avg_mae,
        'freq_rho': freq_rho,
        'lat_auc': lat_auc,
    }

    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        eeg = batch['eeg'].to(device)
        hpp = batch['hpp'].to(device)

        outputs = model(eeg, hpp)
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        loss, loss_dict = criterion(outputs, batch_device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def run_fold(fold, train_samples, val_samples, args, device, hpp_cache=None):
    """Train and evaluate one fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold}: {len(train_samples)} train, {len(val_samples)} val")
    print(f"{'='*60}")

    # Datasets
    train_ds = E2EDataset(train_samples, augment=True, hpp_cache=hpp_cache)
    val_ds = E2EDataset(val_samples, augment=False, hpp_cache=hpp_cache)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=custom_collate, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate
    )

    # Model
    pretrained = str(PRETRAINED_PATH) if PRETRAINED_PATH.exists() else None
    model = E2EDischargeDetector(pretrained_path=pretrained).to(device)

    # Resume from Phase 1 checkpoint if Phase 2
    if args.phase == 2 and args.resume:
        ckpt_path = CACHE_DIR / f'e2e_phase1_fold{fold}.pt'
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
        else:
            print(f"WARNING: Phase 1 checkpoint not found at {ckpt_path}")

    # Setup training
    criterion = HungarianMatchingLoss()

    if args.phase == 1:
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=1e-4
        )
        n_epochs = args.epochs or 50
    else:
        model.unfreeze_backbone()
        param_groups = model.get_param_groups(
            lr_backbone=args.lr_backbone or 1e-5,
            lr_decoder=args.lr or 1e-4
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        n_epochs = args.epochs or 100

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    best_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        scheduler.step()

        # Evaluate every 5 epochs or last epoch
        if epoch % 5 == 0 or epoch == n_epochs:
            val_metrics = evaluate(model, val_loader, criterion, device)
            elapsed = time.time() - t0

            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_metrics['loss']:.4f} | "
                  f"F1: {val_metrics['timing_f1']:.3f} | "
                  f"Freq rho: {val_metrics['freq_rho']:.3f} | "
                  f"Lat AUC: {val_metrics['lat_auc']:.3f} | "
                  f"{elapsed:.1f}s")

            # Save best model
            if val_metrics['timing_f1'] > best_f1:
                best_f1 = val_metrics['timing_f1']
                best_epoch = epoch
                save_path = CACHE_DIR / f'e2e_phase{args.phase}_fold{fold}.pt'
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model (F1={best_f1:.3f}) at epoch {epoch}")
        else:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Train loss: {train_loss:.4f} | {elapsed:.1f}s")

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = CACHE_DIR / f'e2e_phase{args.phase}_fold{fold}_ckpt.pt'
            torch.save(model.state_dict(), ckpt_path)

    print(f"\nFold {fold} best: F1={best_f1:.3f} at epoch {best_epoch}")

    # Final evaluation on best model
    best_path = CACHE_DIR / f'e2e_phase{args.phase}_fold{fold}.pt'
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    final_metrics = evaluate(model, val_loader, criterion, device)

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Train E2E Discharge Detector')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase (1=freeze backbone, 2=end-to-end)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from Phase 1 checkpoint (for Phase 2)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 50 for phase 1, 100 for phase 2)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-3 for phase 1, 1e-4 for phase 2)')
    parser.add_argument('--lr_backbone', type=float, default=None,
                        help='Backbone learning rate for phase 2 (default: 1e-5)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run only this fold (0-indexed)')
    args = parser.parse_args()

    # Set defaults based on phase
    if args.lr is None:
        args.lr = 1e-3 if args.phase == 1 else 1e-4

    device = get_device()
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    print(f"LR: {args.lr}")
    if args.phase == 2:
        print(f"Backbone LR: {args.lr_backbone or 1e-5}")

    # Build sample list
    print("\nBuilding sample list...")
    samples = build_sample_list()
    print(f"Total samples with EEG + discharge times: {len(samples)}")

    # Load HPP cache
    hpp_cache = load_hpp_cache()
    if hpp_cache is None:
        print("WARNING: HPP cache not found. Run precompute_hpp.py first for fast training.")
        print("  python code/e2e_model/precompute_hpp.py")

    # Patient-stratified 5-fold CV
    patient_ids = np.array([s['patient_id'] for s in samples])
    unique_patients = np.unique(patient_ids)
    print(f"Unique patients: {len(unique_patients)}")

    gkf = GroupKFold(n_splits=args.n_folds)
    X_dummy = np.zeros(len(samples))
    groups = patient_ids

    all_metrics = []
    folds = list(gkf.split(X_dummy, groups=groups))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if args.fold is not None and fold_idx != args.fold:
            continue

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]

        metrics = run_fold(fold_idx, train_samples, val_samples, args, device,
                           hpp_cache=hpp_cache)
        metrics['fold'] = fold_idx
        all_metrics.append(metrics)

    # Summary
    if all_metrics:
        print(f"\n{'='*60}")
        print(f"Phase {args.phase} Summary ({len(all_metrics)} folds)")
        print(f"{'='*60}")

        for key in ['timing_f1', 'timing_sens', 'timing_prec', 'timing_mae',
                     'freq_rho', 'lat_auc']:
            vals = [m[key] for m in all_metrics if not np.isnan(m.get(key, float('nan')))]
            if vals:
                print(f"  {key}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

        # Save results
        results_path = CACHE_DIR / f'e2e_phase{args.phase}_results.json'
        serializable = []
        for m in all_metrics:
            sm = {}
            for k, v in m.items():
                if isinstance(v, (float, int, str)):
                    sm[k] = v
                elif isinstance(v, dict):
                    sm[k] = {kk: float(vv) if isinstance(vv, (float, int, np.floating)) else vv
                             for kk, vv in v.items()}
            serializable.append(sm)
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
