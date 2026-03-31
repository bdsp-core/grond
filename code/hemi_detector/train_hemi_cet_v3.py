"""
Train HemiCET v3 — 8-channel hemisphere CET-UNet with expanded expert labels.

Uses current discharge_times.json (1275 entries, 675 GT) which includes
MW's expanded discharge timing labels.

Changes from v1/v2:
  - Saves weights to v3_hemi_cet_fold{0-4}.pt in hemi_cache/ (does NOT overwrite v1/v2)
  - Uses MPS device
  - Same architecture and hyperparameters as v2
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

SAVE_DIR = PROJECT_DIR / 'data' / 'hemi_cache'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v3] Using device: {DEVICE}")

SAVE_PREFIX = 'v3_hemi_cet'

# Hyperparameters (same as v2)
N_FOLDS = 5
N_EPOCHS = 80
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT = 20.0
SHARPNESS_PENALTY = 0.1
HPP_FLOOR_LAMBDA = 0.05
GAUSSIAN_SIGMA = 2
N_SAMPLES = 2000

# Augmentation
AMP_SCALE_LO, AMP_SCALE_HI = 0.7, 1.3
NOISE_SIGMA_FRAC = 0.05
CH_DROPOUT_P = 0.15


# -- Target construction ----------------------------------------------------

def make_target_signal(discharge_times_s, n_samples=N_SAMPLES, fs=FS,
                       sigma=GAUSSIAN_SIGMA, jitter_sigma=1.0):
    target = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32)
    for dt in discharge_times_s:
        center = dt * fs
        if jitter_sigma > 0:
            center += np.random.randn() * jitter_sigma
        bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        target = np.maximum(target, bump)
    return target


# -- Dataset -----------------------------------------------------------------

class HemiCETDataset(Dataset):
    def __init__(self, hemi_data, targets, hpp_evidence, augment=False):
        self.hemi_data = hemi_data.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.hpp_evidence = hpp_evidence.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.hemi_data)

    def _zscore_channels(self, x):
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        return (x - mu) / std

    def __getitem__(self, idx):
        x = self.hemi_data[idx].copy()
        y = self.targets[idx].copy()
        h = self.hpp_evidence[idx].copy()

        x = self._zscore_channels(x)

        if self.augment:
            x, y = self._augment(x, y)

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y[np.newaxis, :])
        h_t = torch.from_numpy(h[np.newaxis, :])
        return x_t, y_t, h_t

    def _augment(self, x, y):
        scale = np.random.uniform(AMP_SCALE_LO, AMP_SCALE_HI)
        x = x * scale

        for ch in range(x.shape[0]):
            ch_std = np.std(x[ch])
            noise = np.random.randn(x.shape[1]) * NOISE_SIGMA_FRAC * (ch_std + 1e-8)
            x[ch] = x[ch] + noise.astype(np.float32)

        if np.random.rand() < CH_DROPOUT_P:
            drop_ch = np.random.randint(0, x.shape[0])
            x[drop_ch] = 0.0

        return x, y


# -- Data preparation -------------------------------------------------------

def prepare_hemi_cet_data(verbose=True):
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
        seg = pat_segs[0]

        if seg.shape[1] != N_SAMPLES:
            continue

        subtype = gt_data.get('subtype', 'unknown')

        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            lat = row.iloc[0].get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = 'unknown'
        else:
            lat = 'unknown'

        discharge_target = make_target_signal(discharge_times, jitter_sigma=0.0)

        def add_hemi(hemi_indices, pid_, subtype_):
            hemi_raw = np.zeros((8, N_SAMPLES), dtype=np.float32)
            hpp_ch = np.zeros((8, N_SAMPLES), dtype=np.float32)
            for i, ch_idx in enumerate(hemi_indices):
                if ch_idx >= seg.shape[0]:
                    continue
                ch = seg[ch_idx].astype(np.float32)
                ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
                hemi_raw[i] = ch
                hpp_ch[i] = _compute_channel_evidence(ch, FS).astype(np.float32)

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
            add_hemi(LEFT_INDICES, pid, subtype)
            add_hemi(RIGHT_INDICES, pid, subtype)
            n_lpd_both += 1

    hemi_data = np.array(all_hemi, dtype=np.float32)
    targets = np.array(all_targets, dtype=np.float32)
    hpp_evidence = np.array(all_hpp, dtype=np.float32)
    patient_ids = np.array(all_pids)
    subtypes = np.array(all_subtypes)

    if verbose:
        print(f"Total hemisphere examples: {len(hemi_data)}")
        print(f"  GPD (both hemi):            {n_gpd} patients -> {n_gpd*2} examples")
        print(f"  LPD left laterality:         {n_lpd_left} examples")
        print(f"  LPD right laterality:        {n_lpd_right} examples")
        print(f"  LPD unknown (both hemi):     {n_lpd_both} patients -> {n_lpd_both*2} examples")

    return hemi_data, targets, hpp_evidence, patient_ids, subtypes


# -- Training loop -----------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    pos_weight_tensor = torch.tensor(POS_WEIGHT, device=device)

    for x, y, hpp_ev in loader:
        x = x.to(device)
        y = y.to(device)
        hpp_ev = hpp_ev.to(device)

        optimizer.zero_grad()
        pred = model(x)

        weights = torch.where(y > 0.1, pos_weight_tensor, torch.ones_like(y))
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)

        sharpness_loss = SHARPNESS_PENALTY * pred.mean()

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


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("  HemiCET V3 Training — Expanded Expert Labels")
    print("=" * 70)

    hemi_data, targets, hpp_evidence, patient_ids, subtypes = prepare_hemi_cet_data()

    model_check = HemiCET().to(DEVICE)
    print(f"\nHemiCET parameters: {count_parameters(model_check):,}")
    del model_check

    print("\nCreating 5-fold patient-stratified splits...")
    unique_pids = np.unique(patient_ids)
    rng = np.random.RandomState(42)

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

    print(f"\n{'='*70}")
    print("  Training HemiCET V3 across 5 folds...")
    print(f"  Epochs={N_EPOCHS}, LR={LR}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}")
    print(f"  POS_WEIGHT={POS_WEIGHT}, SHARPNESS={SHARPNESS_PENALTY}, HPP_FLOOR={HPP_FLOOR_LAMBDA}")
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

            if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
                print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
                      f"train={train_loss:.6f}  val={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save with v3 prefix
        save_path = SAVE_DIR / f'{SAVE_PREFIX}_fold{fold}.pt'
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

    summary = {
        'model': 'HemiCET_v3',
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
    summary_path = SAVE_DIR / f'{SAVE_PREFIX}_training_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print('=' * 70)


if __name__ == '__main__':
    main()
