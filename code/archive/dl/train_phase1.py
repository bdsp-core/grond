"""
Phase 1: Pretrain EEG backbone on LPD vs GPD classification.
Uses cached segments from data/dl_cache/external_pd_segments.npz
Run: conda run -n foe_dl python code/dl/train_phase1.py
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))

from model import EEGClassifier
from data_loader import IIICClassificationDataset

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
NPZ_PATH = CACHE_DIR / 'external_pd_segments.npz'
MODEL_SAVE_PATH = CACHE_DIR / 'classifier_best.pt'

# Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 0  # Mac compatibility
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 10


def main():
    print("=" * 60)
    print("Phase 1: LPD vs GPD Classification Pretraining")
    print("=" * 60)

    # ── Check cache exists ────────────────────────────────────────────
    if not NPZ_PATH.exists():
        print(f"ERROR: Cache file not found: {NPZ_PATH}")
        print("Run extract_external.py first to create the cached segments.")
        sys.exit(1)

    # ── Load data to get patient IDs for splitting ────────────────────
    print(f"\n[1] Loading cached data from {NPZ_PATH}")
    data = np.load(str(NPZ_PATH), allow_pickle=True)
    all_patients = data['patients']
    all_labels = data['labels']
    n_total = len(all_labels)
    n_lpd = np.sum(all_labels == 0)
    n_gpd = np.sum(all_labels == 1)
    n_patients = len(np.unique(all_patients))
    print(f"  Total segments: {n_total} (LPD={n_lpd}, GPD={n_gpd})")
    print(f"  Unique patients: {n_patients}")

    # ── Patient-level 80/20 train/val split ───────────────────────────
    print("\n[2] Patient-level 80/20 train/val split")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(np.arange(n_total), groups=all_patients))

    train_patients = np.unique(all_patients[train_idx])
    val_patients = np.unique(all_patients[val_idx])
    print(f"  Train: {len(train_idx)} segments from {len(train_patients)} patients")
    print(f"  Val:   {len(val_idx)} segments from {len(val_patients)} patients")

    # Verify no patient overlap
    overlap = set(train_patients) & set(val_patients)
    assert len(overlap) == 0, f"Patient overlap detected: {overlap}"
    print("  No patient overlap (verified).")

    # ── Create datasets and dataloaders ───────────────────────────────
    print("\n[3] Creating datasets and dataloaders")
    train_ds = IIICClassificationDataset(str(NPZ_PATH), patient_ids=train_patients, augment=True)
    val_ds = IIICClassificationDataset(str(NPZ_PATH), patient_ids=val_patients, augment=False)
    print(f"  Train dataset: {len(train_ds)} samples")
    print(f"  Val dataset:   {len(val_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # ── Build model ───────────────────────────────────────────────────
    print("\n[4] Building EEGClassifier model")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = EEGClassifier(in_channels=18, dropout=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Loss with class imbalance weighting ───────────────────────────
    # pos_weight = n_negative / n_positive (GPD is label=1)
    train_labels = all_labels[train_idx]
    n_pos = np.sum(train_labels == 1)
    n_neg = np.sum(train_labels == 0)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"  Class balance: LPD={n_neg}, GPD={n_pos}, pos_weight={pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimizer and scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n[5] Training for up to {EPOCHS} epochs (patience={PATIENCE})")
    print(f"  {'Epoch':>5s}  {'Train Loss':>10s}  {'Train Acc':>9s}  {'Val Loss':>10s}  {'Val Acc':>9s}  {'LR':>10s}")
    print("  " + "-" * 62)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)

                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss_sum += loss.item() * batch_x.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_acc:8.1%}  {val_loss:10.4f}  {val_acc:8.1%}  {current_lr:10.6f}")

        # ── Early stopping ────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            os.makedirs(str(CACHE_DIR), exist_ok=True)
            torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    elapsed = time.time() - t0
    print(f"\n[6] Training complete in {elapsed:.0f}s")
    print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"  Model saved to: {MODEL_SAVE_PATH}")

    # ── Final evaluation with best model ──────────────────────────────
    print("\n[7] Final evaluation with best model")
    model.load_state_dict(torch.load(str(MODEL_SAVE_PATH), map_location=device, weights_only=True))
    model.eval()

    val_correct = 0
    val_total = 0
    val_preds_all = []
    val_labels_all = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_x.size(0)
            val_preds_all.extend(preds.cpu().numpy().tolist())
            val_labels_all.extend(batch_y.cpu().numpy().tolist())

    val_acc_final = val_correct / max(val_total, 1)
    val_preds_arr = np.array(val_preds_all)
    val_labels_arr = np.array(val_labels_all)

    # Per-class accuracy
    lpd_mask = val_labels_arr == 0
    gpd_mask = val_labels_arr == 1
    lpd_acc = np.mean(val_preds_arr[lpd_mask] == val_labels_arr[lpd_mask]) if np.any(lpd_mask) else 0.0
    gpd_acc = np.mean(val_preds_arr[gpd_mask] == val_labels_arr[gpd_mask]) if np.any(gpd_mask) else 0.0

    print(f"  Overall val accuracy: {val_acc_final:.1%}")
    print(f"  LPD accuracy:         {lpd_acc:.1%}")
    print(f"  GPD accuracy:         {gpd_acc:.1%}")
    print("\nDone!")


if __name__ == '__main__':
    main()
