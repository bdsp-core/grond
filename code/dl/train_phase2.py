"""
Phase 2: Fine-tune pretrained backbone for frequency estimation with dual heads.
Run: conda run -n foe_dl python code/dl/train_phase2.py
"""

import sys
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from model import EEGFrequencyModel
from data_loader import IIICFrequencyDataset
from optimization_harness import load_dataset, evaluate_predictions

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
CLASSIFIER_PATH = CACHE_DIR / 'classifier_best.pt'
ANNOTATED_PATH = CACHE_DIR / 'annotated_pd_data.npz'
PREDICTIONS_PATH = CACHE_DIR / 'phase2_predictions.npz'
MODEL_SAVE_PATH = CACHE_DIR / 'frequency_model_best.pt'

# Hyperparameters
BATCH_SIZE = 32
NUM_WORKERS = 0
BACKBONE_LR = 1e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
PATIENCE = 15
N_FOLDS = 5
FS = 200


def compute_loss(eventness_pred, freq_preds, log_freq_targets, weak_eventness,
                 device, fs=200):
    """Compute combined loss: L_freq + 0.1 * L_count + 0.05 * L_peak.

    Args:
        eventness_pred: (B, 1, 2000) predicted eventness
        freq_preds: list of 3 tensors, each (B, 1) log-freq predictions
        log_freq_targets: (B, 3) log-freq targets (NaN where unavailable)
        weak_eventness: (B, 2000) pseudo-eventness labels
        device: torch device
        fs: sampling rate

    Returns:
        total_loss, (l_freq, l_count, l_peak) for logging
    """
    batch_size = log_freq_targets.shape[0]

    # ── L_freq: MSE of log-freq predictions vs log(expert_freq) per expert ──
    l_freq = torch.tensor(0.0, device=device)
    n_valid_freq = 0
    for expert_idx in range(3):
        pred = freq_preds[expert_idx].squeeze(1)  # (B,)
        target = log_freq_targets[:, expert_idx]   # (B,)
        valid = torch.isfinite(target)
        if valid.sum() > 0:
            l_freq = l_freq + nn.functional.mse_loss(pred[valid], target[valid])
            n_valid_freq += 1
    if n_valid_freq > 0:
        l_freq = l_freq / n_valid_freq

    # ── L_count: consistency between eventness sum and predicted frequency ──
    # sum(eventness)/fs should approximate number of events in 10s
    # 10 * freq gives expected number of events in 10s
    # Use mean of the 3 expert heads' predictions
    eventness_sum = eventness_pred.squeeze(1).sum(dim=1) / fs  # (B,) approx event count
    mean_log_freq = torch.stack([fp.squeeze(1) for fp in freq_preds], dim=1).mean(dim=1)  # (B,)
    expected_events = 10.0 * torch.exp(mean_log_freq)  # (B,)
    l_count = nn.functional.mse_loss(eventness_sum, expected_events.detach())

    # ── L_peak: BCE(eventness, weak_labels) weighted by agreement confidence ──
    eventness_flat = eventness_pred.squeeze(1)  # (B, 2000)
    weak_flat = weak_eventness  # (B, 2000)

    # Weight by confidence: higher weak label values get more weight
    confidence_weight = 0.5 + 0.5 * weak_flat  # range [0.5, 1.0]

    # BCE loss (eventness_pred already has sigmoid from model)
    eps = 1e-7
    eventness_clamped = torch.clamp(eventness_flat, eps, 1 - eps)
    bce = -(weak_flat * torch.log(eventness_clamped) +
            (1 - weak_flat) * torch.log(1 - eventness_clamped))
    l_peak = (bce * confidence_weight).mean()

    # Combined loss
    total = l_freq + 0.1 * l_count + 0.05 * l_peak

    return total, (l_freq.item(), l_count.item(), l_peak.item())


def train_one_epoch(model, loader, optimizer, device):
    """Train for one epoch. Returns average loss and components."""
    model.train()
    total_loss_sum = 0.0
    freq_loss_sum = 0.0
    count_loss_sum = 0.0
    peak_loss_sum = 0.0
    n_batches = 0

    for batch_x, batch_log_freqs, batch_eventness in loader:
        batch_x = batch_x.to(device)
        batch_log_freqs = batch_log_freqs.to(device)
        batch_eventness = batch_eventness.to(device)

        optimizer.zero_grad()
        eventness_pred, freq_preds = model(batch_x)
        loss, (lf, lc, lp) = compute_loss(
            eventness_pred, freq_preds, batch_log_freqs, batch_eventness, device)

        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_sum += loss.item()
            freq_loss_sum += lf
            count_loss_sum += lc
            peak_loss_sum += lp
            n_batches += 1

    if n_batches == 0:
        return 0.0, (0.0, 0.0, 0.0)
    return (total_loss_sum / n_batches,
            (freq_loss_sum / n_batches, count_loss_sum / n_batches, peak_loss_sum / n_batches))


def validate(model, loader, device):
    """Validate. Returns average loss, components, and predictions."""
    model.eval()
    total_loss_sum = 0.0
    freq_loss_sum = 0.0
    count_loss_sum = 0.0
    peak_loss_sum = 0.0
    n_batches = 0

    all_freq_preds = []  # list of (B, 3) arrays (one per batch)

    with torch.no_grad():
        for batch_x, batch_log_freqs, batch_eventness in loader:
            batch_x = batch_x.to(device)
            batch_log_freqs = batch_log_freqs.to(device)
            batch_eventness = batch_eventness.to(device)

            eventness_pred, freq_preds = model(batch_x)
            loss, (lf, lc, lp) = compute_loss(
                eventness_pred, freq_preds, batch_log_freqs, batch_eventness, device)

            if torch.isfinite(loss):
                total_loss_sum += loss.item()
                freq_loss_sum += lf
                count_loss_sum += lc
                peak_loss_sum += lp
                n_batches += 1

            # Collect frequency predictions: average of 3 expert heads, exp() to get Hz
            batch_log_preds = torch.stack([fp.squeeze(1) for fp in freq_preds], dim=1)  # (B, 3)
            mean_log_pred = batch_log_preds.mean(dim=1)  # (B,)
            hz_pred = torch.exp(mean_log_pred).cpu().numpy()  # (B,)
            all_freq_preds.append(hz_pred)

    if n_batches == 0:
        return 0.0, (0.0, 0.0, 0.0), np.array([])

    all_freq_preds = np.concatenate(all_freq_preds)
    return (total_loss_sum / n_batches,
            (freq_loss_sum / n_batches, count_loss_sum / n_batches, peak_loss_sum / n_batches),
            all_freq_preds)


def main():
    print("=" * 60)
    print("Phase 2: Frequency Estimation Fine-tuning")
    print("=" * 60)

    # ── Check prerequisites ───────────────────────────────────────────
    if not CLASSIFIER_PATH.exists():
        print(f"ERROR: Pretrained classifier not found: {CLASSIFIER_PATH}")
        print("Run train_phase1.py first.")
        sys.exit(1)
    if not ANNOTATED_PATH.exists():
        print(f"ERROR: Annotated data not found: {ANNOTATED_PATH}")
        print("Run generate_weak_labels.py first.")
        sys.exit(1)

    # ── Load pretrained classifier weights ────────────────────────────
    print("\n[1] Loading pretrained classifier weights")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")
    classifier_state = torch.load(str(CLASSIFIER_PATH), map_location=device, weights_only=True)
    print(f"  Loaded {len(classifier_state)} parameter tensors from Phase 1")

    # ── Load annotated data ───────────────────────────────────────────
    print("\n[2] Loading annotated data")
    ann_data = np.load(str(ANNOTATED_PATH), allow_pickle=True)
    segments = ann_data['segments']          # (N, 18, 2000)
    expert_freqs = ann_data['expert_freqs']  # (N, 3)
    weak_eventness = ann_data['weak_eventness']  # (N, 2000)
    patients = ann_data['patients']          # (N,)
    subtypes = ann_data['subtypes']          # (N,)

    N = len(segments)
    n_patients = len(np.unique(patients))
    n_lpd = np.sum(subtypes == 'lpd')
    n_gpd = np.sum(subtypes == 'gpd')
    print(f"  Segments: {N} (LPD={n_lpd}, GPD={n_gpd})")
    print(f"  Unique patients: {n_patients}")

    # ── Also load the original dataset for evaluate_predictions() ─────
    print("\n[3] Loading original dataset for evaluation")
    orig_dataset = load_dataset()
    # Build mapping from index to mat_name
    # The annotated data was built from load_dataset() in the same order
    # We need to reconstruct mat_names
    mat_names = []
    for entry in orig_dataset:
        mat_names.append(entry['mat_name'])
    # Verify lengths match (they might not exactly due to loading failures in generate_weak_labels)
    # We'll build the mapping based on patient+subtype matching
    print(f"  Original dataset: {len(orig_dataset)} entries")

    # Build a lookup from (patient, subtype, entry_index) to mat_name
    # Since generate_weak_labels processes load_dataset() in order, skipping failures,
    # we rebuild by iterating both lists
    segment_mat_names = []
    ann_idx = 0
    for entry in orig_dataset:
        if ann_idx >= N:
            break
        patient_id = Path(entry['mat_name']).stem.split('_')[0]
        subtype = entry['subdir']
        # Check if this entry matches the next annotated segment
        if patient_id == patients[ann_idx] and subtype == subtypes[ann_idx]:
            segment_mat_names.append(entry['mat_name'])
            ann_idx += 1
        # If no match, this entry was skipped (failed in generate_weak_labels)

    # If we couldn't match all, try a simpler approach: use load_dataset order
    if len(segment_mat_names) != N:
        print(f"  Warning: matched {len(segment_mat_names)}/{N} segments by patient+subtype")
        print("  Falling back to sequential matching...")
        segment_mat_names = []
        ann_idx = 0
        for entry in orig_dataset:
            if ann_idx >= N:
                break
            # Try to load the data to see if it would succeed
            segment_mat_names.append(entry['mat_name'])
            ann_idx += 1
        if len(segment_mat_names) != N:
            # Last resort: just use indices
            print(f"  Warning: could only match {len(segment_mat_names)} of {N}. Using first {N}.")
            segment_mat_names = [orig_dataset[i]['mat_name'] for i in range(min(N, len(orig_dataset)))]

    print(f"  Mapped {len(segment_mat_names)} segments to mat_names")

    # ── Patient-level 5-fold CV ───────────────────────────────────────
    print(f"\n[4] Setting up {N_FOLDS}-fold patient-level CV")
    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(np.arange(N), groups=patients))

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_patients = np.unique(patients[train_idx])
        val_patients = np.unique(patients[val_idx])
        overlap = set(train_patients) & set(val_patients)
        print(f"  Fold {fold_idx+1}: train={len(train_idx)} ({len(train_patients)} patients), "
              f"val={len(val_idx)} ({len(val_patients)} patients), overlap={len(overlap)}")

    # ── Train each fold ───────────────────────────────────────────────
    all_val_predictions = np.full(N, np.nan)
    fold_val_losses = []
    t0_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{N_FOLDS}")
        print(f"{'='*60}")

        t0_fold = time.time()

        # ── Build model with pretrained backbone ──────────────────
        model = EEGFrequencyModel(in_channels=18, dropout=0.1, n_experts=3).to(device)
        model.load_pretrained_backbone(classifier_state)
        model.freeze_early_blocks()

        n_params_total = sum(p.numel() for p in model.parameters())
        n_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params_total:,} total, {n_params_train:,} trainable")

        # ── Create datasets ───────────────────────────────────────
        train_patients_fold = np.unique(patients[train_idx])
        val_patients_fold = np.unique(patients[val_idx])

        train_ds = IIICFrequencyDataset(
            segments=segments, expert_freqs=expert_freqs,
            weak_eventness=weak_eventness, patients=patients,
            patient_ids_filter=train_patients_fold, augment=True, fs=FS)
        val_ds = IIICFrequencyDataset(
            segments=segments, expert_freqs=expert_freqs,
            weak_eventness=weak_eventness, patients=patients,
            patient_ids_filter=val_patients_fold, augment=False, fs=FS)

        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)

        # ── Optimizer with param groups ───────────────────────────
        # lr=1e-4 for backbone blocks 3-4, lr=1e-3 for heads
        backbone_late_params = list(model.backbone.block3.parameters()) + \
                               list(model.backbone.block4.parameters())
        head_params = list(model.eventness_head.parameters()) + \
                      list(model.freq_heads.parameters())

        optimizer = torch.optim.AdamW([
            {'params': backbone_late_params, 'lr': BACKBONE_LR},
            {'params': head_params, 'lr': HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # ── Training loop ─────────────────────────────────────────
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_val_preds = None
        best_state = None

        print(f"  {'Ep':>4s}  {'TrLoss':>8s}  {'VaLoss':>8s}  {'Lfreq':>7s}  {'Lcnt':>7s}  {'Lpk':>7s}")
        print("  " + "-" * 50)

        for epoch in range(1, EPOCHS + 1):
            train_loss, (tl_f, tl_c, tl_p) = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, (vl_f, vl_c, vl_p), val_preds = validate(model, val_loader, device)
            scheduler.step()

            if epoch <= 5 or epoch % 10 == 0 or epoch == EPOCHS:
                print(f"  {epoch:4d}  {train_loss:8.4f}  {val_loss:8.4f}  "
                      f"{vl_f:7.4f}  {vl_c:7.4f}  {vl_p:7.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_val_preds = val_preds.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
                    break

        fold_time = time.time() - t0_fold
        fold_val_losses.append(best_val_loss)
        print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch} ({fold_time:.0f}s)")

        # ── Store val predictions ─────────────────────────────────
        if best_val_preds is not None and len(best_val_preds) == len(val_idx):
            all_val_predictions[val_idx] = best_val_preds
            n_valid = np.sum(np.isfinite(best_val_preds))
            print(f"  Stored {n_valid} val predictions")
        else:
            print(f"  Warning: val predictions length mismatch "
                  f"({len(best_val_preds) if best_val_preds is not None else 0} vs {len(val_idx)})")

    # ── Evaluate all val predictions ──────────────────────────────────
    print(f"\n{'='*60}")
    print("Cross-Validation Results")
    print(f"{'='*60}")

    n_predicted = np.sum(np.isfinite(all_val_predictions))
    print(f"Total predictions: {n_predicted}/{N}")

    # Build prediction dict for evaluate_predictions
    pred_dict = {}
    for i in range(min(N, len(segment_mat_names))):
        if np.isfinite(all_val_predictions[i]):
            pred_dict[segment_mat_names[i]] = float(all_val_predictions[i])

    print(f"Prediction dict: {len(pred_dict)} entries")

    if len(pred_dict) > 0:
        metrics = evaluate_predictions(orig_dataset, pred_dict, 'dl_phase2_cnn')
    else:
        print("ERROR: No valid predictions to evaluate.")
        metrics = {}

    # ── Save predictions ──────────────────────────────────────────────
    print(f"\nSaving predictions to {PREDICTIONS_PATH}")
    os.makedirs(str(CACHE_DIR), exist_ok=True)
    np.savez_compressed(
        str(PREDICTIONS_PATH),
        predictions=all_val_predictions,
        mat_names=np.array(segment_mat_names[:N]),
        patients=patients,
        subtypes=subtypes,
        fold_val_losses=np.array(fold_val_losses),
    )

    # ── Save best model (fold with lowest val loss) ───────────────────
    best_fold_idx = int(np.argmin(fold_val_losses))
    print(f"\nBest fold: {best_fold_idx+1} (val loss={fold_val_losses[best_fold_idx]:.4f})")

    # Re-train best fold to save its model
    print(f"Re-training fold {best_fold_idx+1} to save model...")
    train_idx, val_idx = folds[best_fold_idx]

    model = EEGFrequencyModel(in_channels=18, dropout=0.1, n_experts=3).to(device)
    model.load_pretrained_backbone(classifier_state)
    model.freeze_early_blocks()

    train_patients_fold = np.unique(patients[train_idx])
    val_patients_fold = np.unique(patients[val_idx])

    train_ds = IIICFrequencyDataset(
        segments=segments, expert_freqs=expert_freqs,
        weak_eventness=weak_eventness, patients=patients,
        patient_ids_filter=train_patients_fold, augment=True, fs=FS)
    val_ds = IIICFrequencyDataset(
        segments=segments, expert_freqs=expert_freqs,
        weak_eventness=weak_eventness, patients=patients,
        patient_ids_filter=val_patients_fold, augment=False, fs=FS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    backbone_late_params = list(model.backbone.block3.parameters()) + \
                           list(model.backbone.block4.parameters())
    head_params = list(model.eventness_head.parameters()) + \
                  list(model.freq_heads.parameters())

    optimizer = torch.optim.AdamW([
        {'params': backbone_late_params, 'lr': BACKBONE_LR},
        {'params': head_params, 'lr': HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(model, train_loader, optimizer, device)
        val_loss, _, _ = validate(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    print(f"  Saved best model to: {MODEL_SAVE_PATH}")

    total_time = time.time() - t0_total
    print(f"\nTotal Phase 2 time: {total_time:.0f}s")
    print("Done!")


if __name__ == '__main__':
    main()
