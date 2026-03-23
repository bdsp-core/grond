"""
HemiNet MAE Pretraining — Experiment 1.5 Phase 1

Trains the Design A encoder as a masked autoencoder on ALL ~2500 EEG segments
(both labeled and unlabeled, both hemispheres). No labels needed.

Pretraining task:
  1. Take 8-channel hemisphere segment (B, 8, 2000)
  2. Mask random 20% of time patches (each patch = 100 samples = 0.5s)
  3. Encode with Design A encoder → bottleneck features
  4. Decode with lightweight decoder → reconstruct masked regions
  5. Loss = MSE on masked regions only

Usage:
    conda run -n foe_dl python code/hemi_detector/pretrain_mae.py

Saves:
    data/hemi_cache/exp1_5_pretrain/
        best_mae.pt          — best checkpoint (lowest val loss)
        final_mae.pt         — final epoch checkpoint
        pretrain_log.json    — per-epoch loss log
"""

import sys
import json
import time
import math
import warnings
import numpy as np
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.model import HemiNetMAE, count_parameters
from hemi_detector.dataset import (
    _load_segment, _zscore_segment,
    LEFT_INDICES, RIGHT_INDICES,
    N_SAMPLES,
)

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    'eeg_dir': str(PROJECT_DIR / 'data' / 'eeg'),
    'save_dir': str(PROJECT_DIR / 'data' / 'hemi_cache' / 'exp1_5_pretrain'),

    # Training
    'n_epochs': 50,
    'batch_size': 64,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'warmup_epochs': 3,
    'grad_clip': 1.0,
    'val_frac': 0.1,   # 10% of segments for validation

    # Device
    'device': 'mps',
    'num_workers': 0,
    'seed': 42,
}


# ── Dataset: ALL EEG segments (no labels) ─────────────────────────────────────

class AllEEGDataset(Dataset):
    """Dataset of all EEG .mat files for unsupervised pretraining.

    Each segment yields TWO items: left hemisphere + right hemisphere.
    So total = 2 × n_files (both hemispheres per file).

    Each item:
        eeg : (8, 2000) float32 — z-scored hemisphere channels
    """

    def __init__(
        self,
        eeg_dir: Path,
        file_list: List[Path],
        augment: bool = False,
        amp_scale_range=(0.7, 1.3),
        noise_sigma: float = 0.05,
        ch_dropout_p: float = 0.15,
    ):
        self.eeg_dir = Path(eeg_dir)
        self.augment = augment
        self.amp_scale_range = amp_scale_range
        self.noise_sigma = noise_sigma
        self.ch_dropout_p = ch_dropout_p

        # Build (file_path, hemisphere) pairs
        self.examples = []
        for fp in file_list:
            self.examples.append((fp, 'left'))
            self.examples.append((fp, 'right'))

        print(f"AllEEGDataset: {len(file_list)} files → {len(self.examples)} examples "
              f"({'augment' if augment else 'no augment'})")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        fp, hemi = self.examples[idx]

        try:
            seg18 = _load_segment(fp)  # (18, 2000)
        except Exception:
            seg18 = np.zeros((18, N_SAMPLES), dtype=np.float32)

        hemi_idx = LEFT_INDICES if hemi == 'left' else RIGHT_INDICES
        seg8 = seg18[hemi_idx, :]  # (8, 2000)
        seg8 = _zscore_segment(seg8)

        if self.augment:
            scale = np.random.uniform(self.amp_scale_range[0], self.amp_scale_range[1])
            seg8 = seg8 * scale
            for ch in range(8):
                ch_std = np.std(seg8[ch])
                noise = np.random.randn(*seg8[ch].shape).astype(np.float32)
                seg8[ch] = seg8[ch] + noise * self.noise_sigma * ch_std
            if np.random.rand() < self.ch_dropout_p:
                drop_ch = np.random.randint(0, 8)
                seg8[drop_ch] = 0.0

        return {'eeg': torch.from_numpy(seg8)}


# ── Learning Rate Schedule ─────────────────────────────────────────────────────

def get_lr(epoch: int, n_epochs: int, lr: float, warmup_epochs: int) -> float:
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, cfg):
    model.train()
    losses = []

    for batch in loader:
        eeg = batch['eeg'].to(device)  # (B, 8, 2000)

        optimizer.zero_grad()
        recon, mask = model(eeg)  # recon: (B, 8, 2000), mask: (B, 1, 2000)

        # MSE loss on masked regions only
        mask_expanded = mask.expand_as(recon)  # (B, 8, 2000)
        if mask_expanded.any():
            loss = F.mse_loss(recon[mask_expanded], eeg[mask_expanded])
        else:
            loss = F.mse_loss(recon, eeg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    losses = []

    for batch in loader:
        eeg = batch['eeg'].to(device)
        recon, mask = model(eeg)

        mask_expanded = mask.expand_as(recon)
        if mask_expanded.any():
            loss = F.mse_loss(recon[mask_expanded], eeg[mask_expanded])
        else:
            loss = F.mse_loss(recon, eeg)
        losses.append(loss.item())

    return float(np.mean(losses)) if losses else float('nan')


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

    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])

    save_dir = Path(cfg['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover all EEG files ─────────────────────────────────────────
    eeg_dir = Path(cfg['eeg_dir'])
    all_mat_files = sorted(eeg_dir.glob('*.mat'))
    print(f"\nFound {len(all_mat_files)} .mat files in {eeg_dir}")

    # Filter out files that are too small or broken
    # (Quick existence check only — errors handled in dataset)
    valid_files = [f for f in all_mat_files if f.stat().st_size > 1000]
    print(f"Valid files (>1KB): {len(valid_files)}")

    # Train/val split
    np.random.shuffle(valid_files := list(valid_files))
    n_val = max(1, int(len(valid_files) * cfg['val_frac']))
    val_files = valid_files[:n_val]
    train_files = valid_files[n_val:]
    print(f"Train files: {len(train_files)} ({len(train_files)*2} examples)")
    print(f"Val files:   {len(val_files)} ({len(val_files)*2} examples)")

    # Datasets
    train_ds = AllEEGDataset(eeg_dir, train_files, augment=True)
    val_ds = AllEEGDataset(eeg_dir, val_files, augment=False)

    train_loader = DataLoader(
        train_ds,
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

    # ── Model ─────────────────────────────────────────────────────────
    model = HemiNetMAE(in_channels=8, dropout=0.1).to(device)
    n_params = count_parameters(model)
    print(f"\nHemiNetMAE parameters: {n_params:,}")
    print(f"  Encoder: {count_parameters(model.encoder):,}")
    print(f"  Decoder: {count_parameters(model.decoder):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
    )

    # ── Training loop ──────────────────────────────────────────────────
    best_val_loss = float('inf')
    epoch_log = []

    print(f"\nStarting MAE pretraining for {cfg['n_epochs']} epochs...")
    print(f"{'Epoch':>6}  {'LR':>8}  {'Train':>8}  {'Val':>8}  {'Time':>6}")
    print("-" * 45)

    for epoch in range(cfg['n_epochs']):
        lr_now = get_lr(epoch, cfg['n_epochs'], cfg['lr'], cfg['warmup_epochs'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        t_ep = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg)
        val_loss = val_epoch(model, val_loader, device)
        elapsed = time.time() - t_ep

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': cfg,
            }, str(save_dir / 'best_mae.pt'))

        epoch_info = {
            'epoch': epoch + 1,
            'lr': lr_now,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time_s': round(elapsed, 1),
        }
        epoch_log.append(epoch_info)

        print(f"{epoch+1:6d}  {lr_now:.2e}  {train_loss:.4f}  {val_loss:.4f}  {elapsed:.1f}s"
              f"{'  *' if val_loss == best_val_loss else ''}")

        # Save log incrementally
        if (epoch + 1) % 10 == 0:
            with open(str(save_dir / 'pretrain_log.json'), 'w') as f:
                json.dump(epoch_log, f, indent=2)

    # Save final checkpoint
    torch.save({
        'epoch': cfg['n_epochs'],
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'train_loss': train_loss,
        'config': cfg,
    }, str(save_dir / 'final_mae.pt'))

    # Save full log
    with open(str(save_dir / 'pretrain_log.json'), 'w') as f:
        json.dump(epoch_log, f, indent=2)

    total_min = (time.time() - t0) / 60
    print(f"\nPretraining complete!")
    print(f"  Best val MSE: {best_val_loss:.4f}")
    print(f"  Total time: {total_min:.1f} min")
    print(f"  Saved to: {save_dir}")
    print(f"\nNext step: run train_exp15.py to fine-tune with labeled data")

    return best_val_loss


if __name__ == '__main__':
    main()
