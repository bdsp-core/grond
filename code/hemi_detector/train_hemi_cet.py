"""
Train HemiCET (Experiment 5.3) — 8-channel hemisphere CET-UNet.

Trains a single-hemisphere evidence model using 5-fold patient-stratified CV.

Dataset:
  - LPD: affected hemisphere (from laterality labels)
  - GPD: both hemispheres (2 training examples per patient)
  ~816 training examples from ~665 GT cases

Targets:
  Sharp Gaussian bumps at discharge times (sigma=2 samples = 10ms at 200Hz)

Loss:
  BCE(pos_weight=20) + 0.1×mean(evidence) + 0.05×HPP_floor_loss

Usage:
    conda run -n foe_dl python code/hemi_detector/train_hemi_cet.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.hemi_cet import HemiCET, count_parameters
from optimization_harness_v2 import load_dataset, FS
from discharge_detector import LEFT_INDICES, RIGHT_INDICES
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

SAVE_DIR = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
N_FOLDS = 5
N_EPOCHS = 80
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT = 20.0
SHARPNESS_PENALTY = 0.1
HPP_FLOOR_LAMBDA = 0.05
GAUSSIAN_SIGMA = 2  # samples (10ms at 200Hz)
N_SAMPLES = 2000

# Augmentation probabilities
AMP_SCALE_LO, AMP_SCALE_HI = 0.7, 1.3
NOISE_SIGMA_FRAC = 0.05
CH_DROPOUT_P = 0.15


# ── Target construction ────────────────────────────────────────────────────

def make_target_signal(discharge_times_s, n_samples=N_SAMPLES, fs=FS,
                       sigma=GAUSSIAN_SIGMA, jitter_sigma=1.0):
    """Create target signal: sharp Gaussian bumps at each discharge time.

    Args:
        discharge_times_s: list of discharge times in seconds
        n_samples: number of samples
        fs: sampling rate
        sigma: Gaussian std in samples
        jitter_sigma: optional random jitter in samples (applied during build)

    Returns:
        (n_samples,) float32 array with values in [0, 1]
    """
    target = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32)
    for dt in discharge_times_s:
        center = dt * fs
        if jitter_sigma > 0:
            center += np.random.randn() * jitter_sigma
        bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        target = np.maximum(target, bump)
    return target


# ── Dataset ────────────────────────────────────────────────────────────────

class HemiCETDataset(Dataset):
    """Each sample is an 8-channel hemisphere segment with a target trace.

    Fields stored per sample:
      hemi_data:    (8, 2000) float32 — raw EEG (not yet z-scored)
      target:       (2000,)   float32 — sharp Gaussian discharge targets
      hpp_evidence: (2000,)   float32 — median HPP evidence (for floor loss)
    """

    def __init__(self, hemi_data, targets, hpp_evidence, augment=False):
        self.hemi_data = hemi_data.astype(np.float32)    # (N, 8, 2000)
        self.targets = targets.astype(np.float32)          # (N, 2000)
        self.hpp_evidence = hpp_evidence.astype(np.float32)  # (N, 2000)
        self.augment = augment

    def __len__(self):
        return len(self.hemi_data)

    def _zscore_channels(self, x):
        """Z-score each channel independently. x: (8, 2000)"""
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        return (x - mu) / std

    def __getitem__(self, idx):
        x = self.hemi_data[idx].copy()   # (8, 2000)
        y = self.targets[idx].copy()      # (2000,)
        h = self.hpp_evidence[idx].copy() # (2000,)

        # Per-channel z-score
        x = self._zscore_channels(x)

        if self.augment:
            x, y = self._augment(x, y)

        x_t = torch.from_numpy(x)              # (8, 2000)
        y_t = torch.from_numpy(y[np.newaxis, :])  # (1, 2000)
        h_t = torch.from_numpy(h[np.newaxis, :])  # (1, 2000)
        return x_t, y_t, h_t

    def _augment(self, x, y):
        """Apply amplitude scale, Gaussian noise, channel dropout."""
        # Amplitude scale
        scale = np.random.uniform(AMP_SCALE_LO, AMP_SCALE_HI)
        x = x * scale

        # Per-channel Gaussian noise
        for ch in range(x.shape[0]):
            ch_std = np.std(x[ch])
            noise = np.random.randn(x.shape[1]) * NOISE_SIGMA_FRAC * (ch_std + 1e-8)
            x[ch] = x[ch] + noise.astype(np.float32)

        # Channel dropout: zero one random channel with probability CH_DROPOUT_P
        if np.random.rand() < CH_DROPOUT_P:
            drop_ch = np.random.randint(0, x.shape[0])
            x[drop_ch] = 0.0

        return x, y


# ── Data preparation ───────────────────────────────────────────────────────

def prepare_hemi_cet_data(verbose=True):
    """Build hemisphere-level training examples from GT cases.

    For each GT case:
      - LPD (left)  → left hemisphere example only
      - LPD (right) → right hemisphere example only
      - LPD (unknown laterality) → both hemispheres (target on both)
      - GPD → both hemispheres (two training examples)

    Returns:
        hemi_data:    (N, 8, 2000)  EEG segments (raw, not z-scored)
        targets:      (N, 2000)     Gaussian-bump discharge targets
        hpp_evidence: (N, 2000)     median HPP evidence across 8 channels
        patient_ids:  (N,)          patient IDs (for fold splitting)
        subtypes:     (N,)          subtype strings
    """
    if verbose:
        print("Loading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    if verbose:
        print(f"Ground truth cases: {len(gt_cases)}")

    all_hemi = []
    all_targets = []
    all_hpp = []
    all_pids = []
    all_subtypes = []

    n_lpd_left = n_lpd_right = n_lpd_both = n_gpd = 0

    for pid, gt_data in gt_cases.items():
        discharge_times = gt_data['global_times']
        if len(discharge_times) < 2:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]  # (18, 2000) or similar

        if seg.shape[1] != N_SAMPLES:
            continue

        subtype = gt_data.get('subtype', 'unknown')

        # Determine laterality from patients DataFrame
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            lat = row.iloc[0].get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = 'unknown'
        else:
            lat = 'unknown'

        # Build discharge target (without jitter; jitter applied in Dataset)
        discharge_target = make_target_signal(discharge_times, jitter_sigma=0.0)

        def add_hemi(hemi_indices, pid_, subtype_):
            """Extract 8-channel block, compute HPP evidence, add to lists."""
            hemi_raw = np.zeros((8, N_SAMPLES), dtype=np.float32)
            hpp_ch = np.zeros((8, N_SAMPLES), dtype=np.float32)
            for i, ch_idx in enumerate(hemi_indices):
                if ch_idx >= seg.shape[0]:
                    continue
                ch = seg[ch_idx].astype(np.float32)
                ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
                hemi_raw[i] = ch
                hpp_ch[i] = _compute_channel_evidence(ch, FS).astype(np.float32)

            # Normalize each channel's HPP evidence to [0, 1]
            for i in range(8):
                mx = hpp_ch[i].max()
                if mx > 1e-8:
                    hpp_ch[i] /= mx

            hpp_med = np.median(hpp_ch, axis=0)

            all_hemi.append(hemi_raw)
            all_targets.append(discharge_target.copy())
            all_hpp.append(hpp_med.astype(np.float32))
            all_pids.append(pid_)
            all_subtypes.append(subtype_)

        if subtype == 'gpd':
            add_hemi(LEFT_INDICES, pid, subtype)
            add_hemi(RIGHT_INDICES, pid, subtype)
            n_gpd += 1
        elif lat == 'left':
            add_hemi(LEFT_INDICES, pid, subtype)
            n_lpd_left += 1
        elif lat == 'right':
            add_hemi(RIGHT_INDICES, pid, subtype)
            n_lpd_right += 1
        else:
            # Unknown laterality: use both, target on both
            add_hemi(LEFT_INDICES, pid, subtype)
            add_hemi(RIGHT_INDICES, pid, subtype)
            n_lpd_both += 1

    hemi_data = np.array(all_hemi, dtype=np.float32)     # (N, 8, 2000)
    targets = np.array(all_targets, dtype=np.float32)      # (N, 2000)
    hpp_evidence = np.array(all_hpp, dtype=np.float32)     # (N, 2000)
    patient_ids = np.array(all_pids)
    subtypes = np.array(all_subtypes)

    if verbose:
        print(f"Total hemisphere examples: {len(hemi_data)}")
        print(f"  GPD (both hemi):            {n_gpd} patients → {n_gpd*2} examples")
        print(f"  LPD left laterality:         {n_lpd_left} examples")
        print(f"  LPD right laterality:        {n_lpd_right} examples")
        print(f"  LPD unknown (both hemi):     {n_lpd_both} patients → {n_lpd_both*2} examples")

    return hemi_data, targets, hpp_evidence, patient_ids, subtypes


# ── Training loop ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pos_weight_tensor = torch.tensor(POS_WEIGHT, device=device)

    for x, y, hpp_ev in loader:
        x = x.to(device)       # (B, 8, 2000)
        y = y.to(device)       # (B, 1, 2000)
        hpp_ev = hpp_ev.to(device)  # (B, 1, 2000)

        optimizer.zero_grad()
        pred = model(x)  # (B, 1, 2000)

        # Weighted BCE
        weights = torch.where(y > 0.1, pos_weight_tensor, torch.ones_like(y))
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)

        # Sharpness penalty: penalize high baseline evidence
        sharpness_loss = SHARPNESS_PENALTY * pred.mean()

        # HPP floor auxiliary loss: CNN should be >= HPP at discharge peaks
        discharge_mask = (y > 0.1).float()
        hpp_floor_violation = torch.clamp(hpp_ev - pred, min=0) * discharge_mask
        floor_loss = HPP_FLOOR_LAMBDA * (hpp_floor_violation ** 2).mean()

        loss = bce_loss + sharpness_loss + floor_loss
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_val_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    pos_weight_tensor = torch.tensor(POS_WEIGHT, device=device)

    for x, y, hpp_ev in loader:
        x = x.to(device)
        y = y.to(device)
        hpp_ev = hpp_ev.to(device)

        pred = model(x)

        weights = torch.where(y > 0.1, pos_weight_tensor, torch.ones_like(y))
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)
        sharpness_loss = SHARPNESS_PENALTY * pred.mean()
        discharge_mask = (y > 0.1).float()
        hpp_floor_violation = torch.clamp(hpp_ev - pred, min=0) * discharge_mask
        floor_loss = HPP_FLOOR_LAMBDA * (hpp_floor_violation ** 2).mean()

        loss = bce_loss + sharpness_loss + floor_loss
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  HemiCET Training — Experiment 5.3")
    print("=" * 70)

    # Load data
    hemi_data, targets, hpp_evidence, patient_ids, subtypes = prepare_hemi_cet_data()

    # Check model size
    model_check = HemiCET().to(DEVICE)
    print(f"\nHemiCET parameters: {count_parameters(model_check):,}")
    del model_check

    # Create patient-stratified 5-fold splits
    print("\nCreating 5-fold patient-stratified splits...")
    unique_pids = np.unique(patient_ids)
    rng = np.random.RandomState(42)

    # Group unique patients by subtype for stratified splitting
    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids):
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = subtypes[i]

    subtype_groups = {}
    for pid in unique_pids:
        st = pid_to_subtype.get(pid, 'unknown')
        subtype_groups.setdefault(st, []).append(pid)

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % N_FOLDS

    for fold in range(N_FOLDS):
        mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
        n_fold_pats = len([p for p, f in patient_folds.items() if f == fold])
        print(f"  Fold {fold}: {n_fold_pats} patients, {mask.sum()} examples")

    # Train across folds
    print(f"\n{'='*70}")
    print("  Training HemiCET across 5 folds...")
    print(f"  Epochs={N_EPOCHS}, LR={LR}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}")
    print(f"  POS_WEIGHT={POS_WEIGHT}, SHARPNESS={SHARPNESS_PENALTY}, HPP_FLOOR={HPP_FLOOR_LAMBDA}")
    print(f"  Gaussian sigma={GAUSSIAN_SIGMA} samples ({GAUSSIAN_SIGMA*1000//FS}ms)")
    print(f"{'='*70}")

    fold_results = []
    all_val_losses = []

    for fold in range(N_FOLDS):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        val_mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_ds = HemiCETDataset(
            hemi_data[train_mask], targets[train_mask], hpp_evidence[train_mask],
            augment=True)
        val_ds = HemiCETDataset(
            hemi_data[val_mask], targets[val_mask], hpp_evidence[val_mask],
            augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)

        print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

        model = HemiCET().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=N_EPOCHS)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer,
                                         scheduler, DEVICE)
            val_loss = evaluate_val_loss(model, val_loader, DEVICE)

            # Print every epoch (as specified)
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
                  f"train={train_loss:.6f}  val={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        save_path = SAVE_DIR / f'hemi_cet_fold{fold}.pt'
        torch.save(best_state if best_state is not None else model.state_dict(),
                   str(save_path))
        all_val_losses.append(best_val_loss)
        fold_results.append({'fold': fold, 'best_val_loss': best_val_loss})

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold+1} done: best_val_loss={best_val_loss:.6f} "
              f"({fold_elapsed:.1f}s) — saved to {save_path}")

    print(f"\n{'='*70}")
    print("  Training complete")
    print(f"  Mean best val loss: {np.mean(all_val_losses):.6f}")
    for r in fold_results:
        print(f"    Fold {r['fold']}: {r['best_val_loss']:.6f}")

    # Save training summary
    summary = {
        'model': 'HemiCET',
        'n_folds': N_FOLDS,
        'n_epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'pos_weight': POS_WEIGHT,
        'sharpness_penalty': SHARPNESS_PENALTY,
        'hpp_floor_lambda': HPP_FLOOR_LAMBDA,
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'fold_results': fold_results,
        'mean_best_val_loss': float(np.mean(all_val_losses)),
        'total_training_time_s': round(time.time() - t0, 1),
    }
    summary_path = SAVE_DIR / 'training_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print('=' * 70)


if __name__ == '__main__':
    main()
