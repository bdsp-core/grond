"""
Train the unified multi-task CNN for joint PD/RDA analysis.

Tasks:
  1. 4-class subtype classification (LPD=0, GPD=1, LRDA=2, GRDA=3)
  2. Frequency estimation (log Hz, masked for cases without labels)
  3. Per-channel PD detection
  4. Per-channel RDA detection

Multi-task loss:
  L = lambda_sub * CE(subtype) + lambda_freq * MSE(freq) * freq_mask
    + lambda_pd * BCE(pd) * pd_mask * confidence
    + lambda_rda * BCE(rda) * rda_mask * confidence

5-fold patient-stratified CV, 30 epochs, early stopping.
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from unified_model.model import UnifiedPDModel
from unified_model.dataset import load_unified_dataset, UnifiedPDDataset

CACHE_DIR = PROJECT_DIR / 'data' / 'unified_model_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cpu')

# Loss weights
LAMBDA_SUB = 1.0
LAMBDA_FREQ = 1.0
LAMBDA_PD = 0.5
LAMBDA_RDA = 0.5

# Training hyperparameters
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-3
PATIENCE = 7

SUBTYPE_NAMES = ['LPD', 'GPD', 'LRDA', 'GRDA']


def compute_auc(y_true, y_score):
    """Compute AUC via trapezoidal rule (no sklearn dependency)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    sorted_idx = np.argsort(-y_score)
    y_sorted = y_true[sorted_idx]
    tpr_list, fpr_list = [0.0], [0.0]
    tp_cum, fp_cum = 0, 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_list.append(tp_cum / n_pos)
        fpr_list.append(fp_cum / n_neg)
    return float(np.trapz(tpr_list, fpr_list))


def compute_multiclass_auc(y_true, y_scores, n_classes=4):
    """Compute macro-average one-vs-rest AUC."""
    aucs = []
    for c in range(n_classes):
        binary_true = (y_true == c).astype(int)
        if np.sum(binary_true) == 0 or np.sum(binary_true) == len(binary_true):
            continue
        auc = compute_auc(binary_true, y_scores[:, c])
        if np.isfinite(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float('nan')


def train_one_epoch(model, loader, optimizer, scheduler):
    """Train for one epoch. Returns average loss and component losses."""
    model.train()
    total_loss = 0.0
    loss_components = {'sub': 0.0, 'freq': 0.0, 'pd': 0.0, 'rda': 0.0}
    n_batches = 0

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='none')
    bce_loss = nn.BCELoss(reduction='none')

    for batch in loader:
        (eeg, subtype, freq, freq_mask, pd_labels, rda_labels,
         pd_mask, rda_mask, conf_weights) = batch

        eeg = eeg.to(DEVICE)
        subtype = subtype.to(DEVICE)
        freq = freq.to(DEVICE)
        freq_mask = freq_mask.to(DEVICE)
        pd_labels = pd_labels.to(DEVICE)
        rda_labels = rda_labels.to(DEVICE)
        pd_mask = pd_mask.to(DEVICE)
        rda_mask = rda_mask.to(DEVICE)
        conf_weights = conf_weights.to(DEVICE)

        optimizer.zero_grad()

        subtype_logits, freq_pred, pd_probs, rda_probs = model(eeg)

        # 1. Subtype classification loss
        l_sub = ce_loss(subtype_logits, subtype)

        # 2. Frequency loss (masked)
        freq_pred_sq = freq_pred.squeeze(-1)  # (batch,)
        l_freq_raw = mse_loss(freq_pred_sq, freq)  # (batch,)
        n_freq_valid = freq_mask.sum()
        if n_freq_valid > 0:
            l_freq = (l_freq_raw * freq_mask).sum() / n_freq_valid
        else:
            l_freq = torch.tensor(0.0, device=DEVICE)

        # 3. PD channel loss (masked + confidence weighted)
        pd_probs = pd_probs.clamp(1e-7, 1 - 1e-7)
        l_pd_raw = bce_loss(pd_probs, pd_labels)  # (batch, 18)
        pd_weighted = l_pd_raw * pd_mask * conf_weights  # (batch, 18)
        n_pd_valid = (pd_mask * conf_weights).sum()
        if n_pd_valid > 0:
            l_pd = pd_weighted.sum() / n_pd_valid
        else:
            l_pd = torch.tensor(0.0, device=DEVICE)

        # 4. RDA channel loss (masked + confidence weighted)
        rda_probs = rda_probs.clamp(1e-7, 1 - 1e-7)
        l_rda_raw = bce_loss(rda_probs, rda_labels)  # (batch, 18)
        rda_weighted = l_rda_raw * rda_mask * conf_weights  # (batch, 18)
        n_rda_valid = (rda_mask * conf_weights).sum()
        if n_rda_valid > 0:
            l_rda = rda_weighted.sum() / n_rda_valid
        else:
            l_rda = torch.tensor(0.0, device=DEVICE)

        # Combined loss
        loss = (LAMBDA_SUB * l_sub + LAMBDA_FREQ * l_freq
                + LAMBDA_PD * l_pd + LAMBDA_RDA * l_rda)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        loss_components['sub'] += l_sub.item()
        loss_components['freq'] += l_freq.item()
        loss_components['pd'] += l_pd.item()
        loss_components['rda'] += l_rda.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    for k in loss_components:
        loss_components[k] /= max(n_batches, 1)

    return total_loss / max(n_batches, 1), loss_components


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model. Returns dict with predictions and metrics."""
    model.eval()

    all_subtype_logits = []
    all_subtype_labels = []
    all_freq_preds = []
    all_freq_targets = []
    all_freq_masks = []
    all_pd_probs = []
    all_pd_labels = []
    all_pd_masks = []
    all_rda_probs = []
    all_rda_labels = []
    all_rda_masks = []
    all_conf_weights = []

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='none')
    bce_loss = nn.BCELoss(reduction='none')
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        (eeg, subtype, freq, freq_mask, pd_labels, rda_labels,
         pd_mask, rda_mask, conf_weights) = batch

        eeg = eeg.to(DEVICE)
        subtype_dev = subtype.to(DEVICE)
        freq_dev = freq.to(DEVICE)
        freq_mask_dev = freq_mask.to(DEVICE)
        pd_labels_dev = pd_labels.to(DEVICE)
        rda_labels_dev = rda_labels.to(DEVICE)
        pd_mask_dev = pd_mask.to(DEVICE)
        rda_mask_dev = rda_mask.to(DEVICE)
        conf_weights_dev = conf_weights.to(DEVICE)

        subtype_logits, freq_pred, pd_probs, rda_probs = model(eeg)

        # Compute loss
        l_sub = ce_loss(subtype_logits, subtype_dev)
        freq_pred_sq = freq_pred.squeeze(-1)
        l_freq_raw = mse_loss(freq_pred_sq, freq_dev)
        n_freq_valid = freq_mask_dev.sum()
        l_freq = (l_freq_raw * freq_mask_dev).sum() / n_freq_valid if n_freq_valid > 0 else torch.tensor(0.0)
        pd_probs = pd_probs.clamp(1e-7, 1 - 1e-7)
        l_pd_raw = bce_loss(pd_probs, pd_labels_dev)
        pd_weighted = l_pd_raw * pd_mask_dev * conf_weights_dev
        n_pd_valid = (pd_mask_dev * conf_weights_dev).sum()
        l_pd = pd_weighted.sum() / n_pd_valid if n_pd_valid > 0 else torch.tensor(0.0)
        rda_probs = rda_probs.clamp(1e-7, 1 - 1e-7)
        l_rda_raw = bce_loss(rda_probs, rda_labels_dev)
        rda_weighted = l_rda_raw * rda_mask_dev * conf_weights_dev
        n_rda_valid = (rda_mask_dev * conf_weights_dev).sum()
        l_rda = rda_weighted.sum() / n_rda_valid if n_rda_valid > 0 else torch.tensor(0.0)
        loss = LAMBDA_SUB * l_sub + LAMBDA_FREQ * l_freq + LAMBDA_PD * l_pd + LAMBDA_RDA * l_rda
        total_loss += loss.item()
        n_batches += 1

        all_subtype_logits.append(subtype_logits.cpu().numpy())
        all_subtype_labels.append(subtype.numpy())
        all_freq_preds.append(freq_pred_sq.cpu().numpy())
        all_freq_targets.append(freq.numpy())
        all_freq_masks.append(freq_mask.numpy())
        all_pd_probs.append(pd_probs.cpu().numpy())
        all_pd_labels.append(pd_labels.numpy())
        all_pd_masks.append(pd_mask.numpy())
        all_rda_probs.append(rda_probs.cpu().numpy())
        all_rda_labels.append(rda_labels.numpy())
        all_rda_masks.append(rda_mask.numpy())
        all_conf_weights.append(conf_weights.numpy())

    return {
        'subtype_logits': np.concatenate(all_subtype_logits),
        'subtype_labels': np.concatenate(all_subtype_labels),
        'freq_preds': np.concatenate(all_freq_preds),
        'freq_targets': np.concatenate(all_freq_targets),
        'freq_masks': np.concatenate(all_freq_masks),
        'pd_probs': np.concatenate(all_pd_probs),
        'pd_labels': np.concatenate(all_pd_labels),
        'pd_masks': np.concatenate(all_pd_masks),
        'rda_probs': np.concatenate(all_rda_probs),
        'rda_labels': np.concatenate(all_rda_labels),
        'rda_masks': np.concatenate(all_rda_masks),
        'conf_weights': np.concatenate(all_conf_weights),
        'avg_loss': total_loss / max(n_batches, 1),
    }


def compute_metrics(results):
    """Compute all evaluation metrics from results dict."""
    metrics = {}

    # Subtype accuracy
    preds = np.argmax(results['subtype_logits'], axis=1)
    labels = results['subtype_labels']
    metrics['subtype_acc'] = float(np.mean(preds == labels))

    # Per-class accuracy
    for c in range(4):
        mask = labels == c
        if np.sum(mask) > 0:
            metrics[f'subtype_acc_{SUBTYPE_NAMES[c]}'] = float(np.mean(preds[mask] == c))
        else:
            metrics[f'subtype_acc_{SUBTYPE_NAMES[c]}'] = float('nan')

    # Subtype macro AUC (one-vs-rest)
    probs = torch.softmax(torch.from_numpy(results['subtype_logits']), dim=1).numpy()
    metrics['subtype_macro_auc'] = compute_multiclass_auc(labels, probs)

    # Frequency Spearman (on masked samples)
    freq_mask = results['freq_masks'] > 0.5
    if np.sum(freq_mask) >= 5:
        freq_preds = results['freq_preds'][freq_mask]
        freq_targets = results['freq_targets'][freq_mask]
        valid = np.isfinite(freq_preds) & np.isfinite(freq_targets)
        if np.sum(valid) >= 5:
            rho, _ = spearmanr(freq_targets[valid], freq_preds[valid])
            metrics['freq_spearman'] = float(rho) if np.isfinite(rho) else float('nan')
            # MAE in linear Hz
            pred_hz = np.exp(freq_preds[valid])
            true_hz = np.exp(freq_targets[valid])
            metrics['freq_mae'] = float(np.mean(np.abs(pred_hz - true_hz)))
        else:
            metrics['freq_spearman'] = float('nan')
            metrics['freq_mae'] = float('nan')
    else:
        metrics['freq_spearman'] = float('nan')
        metrics['freq_mae'] = float('nan')

    # PD channel AUC (on ground_truth-confidence channels only = conf weight >= 2.0)
    pd_probs_flat = results['pd_probs'].flatten()
    pd_labels_flat = results['pd_labels'].flatten()
    pd_masks_flat = results['pd_masks'].flatten()
    conf_flat = results['conf_weights'].flatten()

    # AUC on ground_truth channels (confidence weight == 2.0)
    gt_mask = (pd_masks_flat > 0.5) & (conf_flat >= 1.99)
    if np.sum(gt_mask) > 10:
        metrics['pd_channel_auc_gt'] = compute_auc(
            pd_labels_flat[gt_mask].astype(int), pd_probs_flat[gt_mask])
    else:
        metrics['pd_channel_auc_gt'] = float('nan')

    # AUC on all non-null channels
    all_mask = pd_masks_flat > 0.5
    if np.sum(all_mask) > 10:
        metrics['pd_channel_auc_all'] = compute_auc(
            pd_labels_flat[all_mask].astype(int), pd_probs_flat[all_mask])
    else:
        metrics['pd_channel_auc_all'] = float('nan')

    # RDA channel AUC
    rda_probs_flat = results['rda_probs'].flatten()
    rda_labels_flat = results['rda_labels'].flatten()
    rda_masks_flat = results['rda_masks'].flatten()

    gt_rda_mask = (rda_masks_flat > 0.5) & (conf_flat >= 1.99)
    if np.sum(gt_rda_mask) > 10:
        metrics['rda_channel_auc_gt'] = compute_auc(
            rda_labels_flat[gt_rda_mask].astype(int), rda_probs_flat[gt_rda_mask])
    else:
        metrics['rda_channel_auc_gt'] = float('nan')

    rda_all_mask = rda_masks_flat > 0.5
    if np.sum(rda_all_mask) > 10:
        metrics['rda_channel_auc_all'] = compute_auc(
            rda_labels_flat[rda_all_mask].astype(int), rda_probs_flat[rda_all_mask])
    else:
        metrics['rda_channel_auc_all'] = float('nan')

    return metrics


def combined_metric(metrics):
    """Combined validation metric for early stopping.

    Mean of: subtype accuracy, freq Spearman (clamped 0), PD channel AUC (GT).
    """
    parts = []
    if np.isfinite(metrics.get('subtype_acc', float('nan'))):
        parts.append(metrics['subtype_acc'])
    if np.isfinite(metrics.get('freq_spearman', float('nan'))):
        parts.append(max(0, metrics['freq_spearman']))
    if np.isfinite(metrics.get('pd_channel_auc_gt', float('nan'))):
        parts.append(metrics['pd_channel_auc_gt'])
    elif np.isfinite(metrics.get('pd_channel_auc_all', float('nan'))):
        parts.append(metrics['pd_channel_auc_all'])
    return float(np.mean(parts)) if parts else 0.0


def create_patient_folds(samples, n_folds=5, seed=42):
    """Create patient-stratified folds.

    Stratifies by subtype to ensure each fold has similar distribution.

    Returns:
        patient_folds: dict patient_id -> fold_idx
    """
    rng = np.random.RandomState(seed)

    # Group patients by subtype
    pid_to_subtype = {}
    for s in samples:
        pid_to_subtype[s['patient_id']] = s['subtype']

    subtype_groups = {}
    for pid, st in pid_to_subtype.items():
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(pid)

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % n_folds

    return patient_folds


def main():
    t0 = time.time()
    print("=" * 70)
    print("Unified Multi-Task CNN: Subtype + Frequency + Channel PD/RDA")
    print("=" * 70)

    # -- Load dataset --
    print("\nLoading unified dataset...")
    samples = load_unified_dataset(verbose=True)

    if len(samples) == 0:
        print("ERROR: No samples loaded. Check data paths.")
        return

    # -- Create folds --
    print(f"\nCreating {N_FOLDS}-fold patient-stratified splits...")
    patient_folds = create_patient_folds(samples, N_FOLDS)

    # Print fold statistics
    for fold in range(N_FOLDS):
        fold_pids = [p for p, f in patient_folds.items() if f == fold]
        fold_samples = [s for s in samples if patient_folds[s['patient_id']] == fold]
        st_counts = Counter(s['subtype'] for s in fold_samples)
        n_freq = sum(1 for s in fold_samples if np.isfinite(s['freq']))
        print(f"  Fold {fold}: {len(fold_pids)} patients, {len(fold_samples)} segs, "
              f"freq={n_freq}, subtypes={dict(st_counts)}")

    # -- Training --
    print(f"\n{'=' * 70}")
    print(f"Training {N_FOLDS}-fold CV (epochs={N_EPOCHS}, batch={BATCH_SIZE}, lr={LR})")
    print(f"Loss weights: sub={LAMBDA_SUB}, freq={LAMBDA_FREQ}, pd={LAMBDA_PD}, rda={LAMBDA_RDA}")
    print(f"{'=' * 70}")

    all_fold_metrics = []
    training_curves = {}

    # Collect out-of-fold predictions
    oof_subtype_logits = {}
    oof_subtype_labels = {}
    oof_freq_preds = {}
    oof_freq_targets = {}
    oof_freq_masks = {}
    oof_pd_probs = {}
    oof_pd_labels = {}
    oof_pd_masks = {}
    oof_rda_probs = {}
    oof_rda_labels = {}
    oof_rda_masks = {}
    oof_conf_weights = {}
    oof_patient_ids = {}

    for fold in range(N_FOLDS):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Split samples
        train_samples = [s for s in samples if patient_folds[s['patient_id']] != fold]
        val_samples = [s for s in samples if patient_folds[s['patient_id']] == fold]

        print(f"  Train: {len(train_samples)} segs, Val: {len(val_samples)} segs")

        train_ds = UnifiedPDDataset(train_samples, augment=True)
        val_ds = UnifiedPDDataset(val_samples, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = UnifiedPDModel().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_combined = -1.0
        best_state = None
        epochs_no_improve = 0

        fold_curves = {
            'train_loss': [], 'val_loss': [],
            'val_subtype_acc': [], 'val_freq_spearman': [],
            'val_pd_auc': [], 'val_combined': [],
        }

        for epoch in range(N_EPOCHS):
            train_loss, loss_parts = train_one_epoch(model, train_loader, optimizer, scheduler)

            val_results = evaluate(model, val_loader)
            val_metrics = compute_metrics(val_results)
            val_comb = combined_metric(val_metrics)

            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_results['avg_loss'], 6))
            fold_curves['val_subtype_acc'].append(
                round(val_metrics['subtype_acc'], 4) if np.isfinite(val_metrics['subtype_acc']) else None)
            fold_curves['val_freq_spearman'].append(
                round(val_metrics['freq_spearman'], 4) if np.isfinite(val_metrics['freq_spearman']) else None)
            fold_curves['val_pd_auc'].append(
                round(val_metrics.get('pd_channel_auc_gt', val_metrics.get('pd_channel_auc_all', float('nan'))), 4)
                if np.isfinite(val_metrics.get('pd_channel_auc_gt', val_metrics.get('pd_channel_auc_all', float('nan')))) else None)
            fold_curves['val_combined'].append(round(val_comb, 4))

            if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
                freq_sp = val_metrics['freq_spearman']
                pd_auc = val_metrics.get('pd_channel_auc_gt', val_metrics.get('pd_channel_auc_all', float('nan')))
                print(f"  Ep {epoch+1:2d}: loss={train_loss:.4f}/{val_results['avg_loss']:.4f} "
                      f"sub_acc={val_metrics['subtype_acc']:.3f} "
                      f"freq_sp={freq_sp:.3f} " if np.isfinite(freq_sp) else f"  Ep {epoch+1:2d}: loss={train_loss:.4f}/{val_results['avg_loss']:.4f} "
                      f"sub_acc={val_metrics['subtype_acc']:.3f} freq_sp=N/A ",
                      end='')
                print(f"pd_auc={pd_auc:.3f} " if np.isfinite(pd_auc) else "pd_auc=N/A ",
                      end='')
                print(f"comb={val_comb:.3f}")

            if val_comb > best_combined:
                best_combined = val_comb
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={PATIENCE})")
                    break

        training_curves[f'fold_{fold}'] = fold_curves

        # Load best model and evaluate
        if best_state is not None:
            model.load_state_dict(best_state)

        val_results = evaluate(model, val_loader)
        val_metrics = compute_metrics(val_results)
        all_fold_metrics.append(val_metrics)

        # Save out-of-fold predictions
        for i, s in enumerate(val_samples):
            pid = s['patient_id']
            if pid not in oof_subtype_logits:
                oof_subtype_logits[pid] = []
                oof_subtype_labels[pid] = s['subtype']
                oof_freq_preds[pid] = []
                oof_freq_targets[pid] = s['freq']
                oof_freq_masks[pid] = 1.0 if np.isfinite(s['freq']) else 0.0
                oof_pd_probs[pid] = []
                oof_pd_labels[pid] = s['pd_labels']
                oof_pd_masks[pid] = (s['pd_labels'] >= 0).astype(float)
                oof_rda_probs[pid] = []
                oof_rda_labels[pid] = s['rda_labels']
                oof_rda_masks[pid] = (s['rda_labels'] >= 0).astype(float)
                oof_conf_weights[pid] = s['confidence_weights']
                oof_patient_ids[pid] = pid

        # Map val results back to samples
        for i in range(len(val_samples)):
            pid = val_samples[i]['patient_id']
            oof_subtype_logits[pid].append(val_results['subtype_logits'][i])
            oof_freq_preds[pid].append(val_results['freq_preds'][i])
            oof_pd_probs[pid].append(val_results['pd_probs'][i])
            oof_rda_probs[pid].append(val_results['rda_probs'][i])

        # Save model
        save_path = CACHE_DIR / f'unified_fold{fold}.pt'
        torch.save(best_state if best_state is not None else model.state_dict(), str(save_path))

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold+1} done: combined={best_combined:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Saved to {save_path}")

    # -- Save training curves --
    curves_path = CACHE_DIR / 'training_curves.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Aggregate out-of-fold results --
    print(f"\n{'=' * 70}")
    print("OUT-OF-FOLD RESULTS (Patient-Level)")
    print(f"{'=' * 70}")

    # Patient-level subtype: average logits across segments, then argmax
    pat_subtype_preds = []
    pat_subtype_labels = []
    pat_subtype_probs = []

    for pid in sorted(oof_subtype_logits.keys()):
        logits_list = oof_subtype_logits[pid]
        if len(logits_list) == 0:
            continue
        avg_logits = np.mean(logits_list, axis=0)
        pred = np.argmax(avg_logits)
        pat_subtype_preds.append(pred)
        pat_subtype_labels.append(oof_subtype_labels[pid])
        probs = np.exp(avg_logits) / np.sum(np.exp(avg_logits))
        pat_subtype_probs.append(probs)

    pat_subtype_preds = np.array(pat_subtype_preds)
    pat_subtype_labels = np.array(pat_subtype_labels)
    pat_subtype_probs = np.array(pat_subtype_probs)

    # Overall accuracy
    sub_acc = float(np.mean(pat_subtype_preds == pat_subtype_labels))
    sub_macro_auc = compute_multiclass_auc(pat_subtype_labels, pat_subtype_probs)

    print(f"\n4-Class Subtype Classification (N={len(pat_subtype_labels)} patients):")
    print(f"  Overall accuracy:  {sub_acc:.4f}")
    print(f"  Macro AUC (OVR):   {sub_macro_auc:.4f}")

    # Per-class accuracy
    for c in range(4):
        mask = pat_subtype_labels == c
        n_c = np.sum(mask)
        if n_c > 0:
            acc_c = float(np.mean(pat_subtype_preds[mask] == c))
            print(f"  {SUBTYPE_NAMES[c]:>5s}: {acc_c:.4f} (N={n_c})")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>6s}", end='')
    for c in range(4):
        print(f" {SUBTYPE_NAMES[c]:>5s}", end='')
    print()
    for r in range(4):
        print(f"  {SUBTYPE_NAMES[r]:>6s}", end='')
        for c in range(4):
            count = np.sum((pat_subtype_labels == r) & (pat_subtype_preds == c))
            print(f" {count:5d}", end='')
        print()

    # Patient-level frequency
    print(f"\nFrequency Estimation:")
    freq_golds = []
    freq_preds_list = []
    freq_subtypes = []
    pid_to_subtype_str = {s['patient_id']: s['subtype_str'] for s in samples}

    for pid in sorted(oof_freq_preds.keys()):
        if oof_freq_masks[pid] < 0.5:
            continue
        pred_list = oof_freq_preds[pid]
        if len(pred_list) == 0:
            continue
        avg_log_pred = float(np.mean(pred_list))
        pred_hz = np.exp(avg_log_pred)
        pred_hz = np.clip(pred_hz, 0.1, 10.0)
        gold_hz = np.exp(oof_freq_targets[pid]) if np.isfinite(oof_freq_targets[pid]) else float('nan')
        if np.isfinite(gold_hz) and np.isfinite(pred_hz):
            freq_golds.append(gold_hz)
            freq_preds_list.append(pred_hz)
            freq_subtypes.append(pid_to_subtype_str.get(pid, ''))

    freq_golds = np.array(freq_golds)
    freq_preds_arr = np.array(freq_preds_list)
    freq_subtypes = np.array(freq_subtypes)

    if len(freq_golds) >= 5:
        rho_all, _ = spearmanr(freq_golds, freq_preds_arr)
        mae_all = float(np.mean(np.abs(freq_golds - freq_preds_arr)))
        print(f"  Combined (N={len(freq_golds)}): Spearman={rho_all:.4f}, MAE={mae_all:.4f} Hz")

        for st in ['lpd', 'gpd']:
            mask = freq_subtypes == st
            if np.sum(mask) >= 5:
                rho_st, _ = spearmanr(freq_golds[mask], freq_preds_arr[mask])
                mae_st = float(np.mean(np.abs(freq_golds[mask] - freq_preds_arr[mask])))
                print(f"  {st.upper():>3s}      (N={np.sum(mask)}): Spearman={rho_st:.4f}, MAE={mae_st:.4f} Hz")
    else:
        rho_all = float('nan')
        mae_all = float('nan')
        print(f"  Not enough patients with frequency labels")

    # Channel-level PD AUC (aggregate across folds)
    print(f"\nChannel-Level PD Detection:")
    all_pd_p = []
    all_pd_l = []
    all_pd_gt_p = []
    all_pd_gt_l = []

    for pid in sorted(oof_pd_probs.keys()):
        prob_list = oof_pd_probs[pid]
        if len(prob_list) == 0:
            continue
        avg_probs = np.mean(prob_list, axis=0)  # (18,)
        labels = oof_pd_labels[pid]
        masks = oof_pd_masks[pid]
        confs = oof_conf_weights[pid]

        for ch in range(18):
            if masks[ch] > 0.5:
                all_pd_p.append(avg_probs[ch])
                all_pd_l.append(int(labels[ch]))
                if confs[ch] >= 1.99:
                    all_pd_gt_p.append(avg_probs[ch])
                    all_pd_gt_l.append(int(labels[ch]))

    all_pd_p = np.array(all_pd_p)
    all_pd_l = np.array(all_pd_l)
    all_pd_gt_p = np.array(all_pd_gt_p)
    all_pd_gt_l = np.array(all_pd_gt_l)

    if len(all_pd_gt_l) > 10:
        pd_auc_gt = compute_auc(all_pd_gt_l, all_pd_gt_p)
        print(f"  AUC (ground_truth only, N={len(all_pd_gt_l)}): {pd_auc_gt:.4f}")
    else:
        pd_auc_gt = float('nan')
        print(f"  Not enough ground_truth channels for AUC")

    if len(all_pd_l) > 10:
        pd_auc_all = compute_auc(all_pd_l, all_pd_p)
        print(f"  AUC (all labeled, N={len(all_pd_l)}): {pd_auc_all:.4f}")

    # Channel-level RDA AUC
    print(f"\nChannel-Level RDA Detection:")
    all_rda_p = []
    all_rda_l = []
    all_rda_gt_p = []
    all_rda_gt_l = []

    for pid in sorted(oof_rda_probs.keys()):
        prob_list = oof_rda_probs[pid]
        if len(prob_list) == 0:
            continue
        avg_probs = np.mean(prob_list, axis=0)
        labels = oof_rda_labels[pid]
        masks = oof_rda_masks[pid]
        confs = oof_conf_weights[pid]

        for ch in range(18):
            if masks[ch] > 0.5:
                all_rda_p.append(avg_probs[ch])
                all_rda_l.append(int(labels[ch]))
                if confs[ch] >= 1.99:
                    all_rda_gt_p.append(avg_probs[ch])
                    all_rda_gt_l.append(int(labels[ch]))

    all_rda_p = np.array(all_rda_p)
    all_rda_l = np.array(all_rda_l)
    all_rda_gt_p = np.array(all_rda_gt_p)
    all_rda_gt_l = np.array(all_rda_gt_l)

    if len(all_rda_gt_l) > 10:
        rda_auc_gt = compute_auc(all_rda_gt_l, all_rda_gt_p)
        print(f"  AUC (ground_truth only, N={len(all_rda_gt_l)}): {rda_auc_gt:.4f}")
    else:
        rda_auc_gt = float('nan')
        print(f"  Not enough ground_truth channels for RDA AUC")

    if len(all_rda_l) > 10:
        rda_auc_all = compute_auc(all_rda_l, all_rda_p)
        print(f"  AUC (all labeled, N={len(all_rda_l)}): {rda_auc_all:.4f}")

    # -- Comparison with previous best --
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH PREVIOUS BEST")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30s} {'Previous':>12s} {'Unified':>12s}")
    print(f"  {'-' * 54}")

    def fmt(v):
        return f"{v:.4f}" if np.isfinite(v) else "N/A"

    print(f"  {'Freq Spearman (combined)':<30s} {'0.640':>12s} {fmt(rho_all if len(freq_golds)>=5 else float('nan')):>12s}")
    print(f"  {'Subtype AUC (macro OVR)':<30s} {'0.931':>12s} {fmt(sub_macro_auc):>12s}")
    print(f"  {'Subtype Accuracy (4-class)':<30s} {'N/A':>12s} {fmt(sub_acc):>12s}")
    if np.isfinite(pd_auc_gt):
        print(f"  {'PD Channel AUC (GT)':<30s} {'N/A':>12s} {fmt(pd_auc_gt):>12s}")
    if np.isfinite(rda_auc_gt):
        print(f"  {'RDA Channel AUC (GT)':<30s} {'N/A':>12s} {fmt(rda_auc_gt):>12s}")

    # -- Save results JSON --
    results_json = {
        'experiment': 'unified_multitask_cnn',
        'timestamp': time.time(),
        'n_folds': N_FOLDS,
        'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'lambda_sub': LAMBDA_SUB,
        'lambda_freq': LAMBDA_FREQ,
        'lambda_pd': LAMBDA_PD,
        'lambda_rda': LAMBDA_RDA,
        'subtype_accuracy': round(sub_acc, 4),
        'subtype_macro_auc': round(sub_macro_auc, 4) if np.isfinite(sub_macro_auc) else None,
        'freq_spearman_combined': round(float(rho_all), 4) if len(freq_golds) >= 5 and np.isfinite(rho_all) else None,
        'freq_mae_combined': round(mae_all, 4) if len(freq_golds) >= 5 and np.isfinite(mae_all) else None,
        'pd_channel_auc_gt': round(float(pd_auc_gt), 4) if np.isfinite(pd_auc_gt) else None,
        'rda_channel_auc_gt': round(float(rda_auc_gt), 4) if np.isfinite(rda_auc_gt) else None,
    }

    # Per-fold metrics
    for i, m in enumerate(all_fold_metrics):
        for k, v in m.items():
            key = f'fold{i}_{k}'
            if isinstance(v, float) and np.isnan(v):
                results_json[key] = None
            else:
                results_json[key] = round(v, 4) if isinstance(v, float) else v

    results_path = RESULTS_DIR / 'unified_multitask_results.json'
    with open(str(results_path), 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
