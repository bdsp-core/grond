"""
Train CET-UNet v3 — uses current discharge_times.json (1275 entries, 675 GT)
which includes MW's expanded expert labels.

Changes from v1:
  - Saves weights to v3_cet_unet_fold{0-4}.pt (does NOT overwrite v1)
  - Uses MPS device
  - Same architecture and hyperparameters as v1
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.cet import CETUNet
from optimization_harness_v2 import load_dataset, FS
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v3] Using device: {DEVICE}")

SAVE_PREFIX = 'v3_cet_unet'

# Training hyperparameters (same as v1)
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 7
POS_WEIGHT = 20.0
SHARPNESS_PENALTY = 0.1
GAUSSIAN_SIGMA = 2
N_SAMPLES = 2000
TOLERANCE_S = 0.1


# -- Dataset -----------------------------------------------------------------

class CETDataset(Dataset):
    def __init__(self, channels, targets, hpp_evidence=None, augment=False):
        self.channels = channels.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.hpp_evidence = hpp_evidence.astype(np.float32) if hpp_evidence is not None else None
        self.augment = augment

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        x = self.channels[idx].copy()
        y = self.targets[idx].copy()

        mu = np.mean(x)
        std = np.std(x)
        if std > 1e-8:
            x = (x - mu) / std
        else:
            x = x - mu

        if self.augment:
            x, y = self._augment(x, y)

        x_tensor = torch.from_numpy(x[np.newaxis, :])
        y_tensor = torch.from_numpy(y[np.newaxis, :])

        if self.hpp_evidence is not None:
            h = self.hpp_evidence[idx].copy()
            h_tensor = torch.from_numpy(h[np.newaxis, :])
            return x_tensor, y_tensor, h_tensor
        return x_tensor, y_tensor, torch.zeros_like(y_tensor)

    def _augment(self, x, y):
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale

        snr_db = np.random.uniform(20, 40)
        signal_power = np.mean(x ** 2)
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(x)) * np.sqrt(noise_power)
            x = x + noise.astype(np.float32)

        return x, y


def make_target_signal(discharge_times_s, n_samples=N_SAMPLES, fs=FS,
                       sigma=GAUSSIAN_SIGMA):
    target = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32)

    for dt in discharge_times_s:
        center = dt * fs
        bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        target = np.maximum(target, bump)

    return target


# -- Data preparation -------------------------------------------------------

def prepare_cet_data():
    print("Loading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    pseudo_path = PROJECT_DIR / 'data' / 'labels' / 'channel_pseudolabels.json'
    with open(str(pseudo_path)) as f:
        pseudolabels = json.load(f)

    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    all_channels = []
    all_targets = []
    all_hpp_evidence = []
    all_pids = []
    all_subtypes = []

    n_involved = 0
    n_uninvolved = 0

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

        ch_labels = pseudolabels.get(pid, {})
        ch_dict = ch_labels.get('channels', {}) if isinstance(ch_labels, dict) else {}

        discharge_target = make_target_signal(discharge_times)

        n_ch = min(seg.shape[0], 18)
        for ch_idx in range(n_ch):
            ch_data = seg[ch_idx]

            if not np.all(np.isfinite(ch_data)):
                continue

            ch_info = ch_dict.get(str(ch_idx), {})
            is_involved = ch_info.get('pd_label', 0) == 1

            if is_involved:
                target = discharge_target.copy()
                n_involved += 1
            else:
                target = np.zeros(N_SAMPLES, dtype=np.float32)
                n_uninvolved += 1

            hpp_ev = _compute_channel_evidence(ch_data, FS)
            hpp_max = hpp_ev.max()
            if hpp_max > 1e-8:
                hpp_ev = hpp_ev / hpp_max

            all_channels.append(ch_data.astype(np.float32))
            all_targets.append(target)
            all_hpp_evidence.append(hpp_ev.astype(np.float32))
            all_pids.append(pid)
            all_subtypes.append(subtype)

    channels = np.array(all_channels)
    targets = np.array(all_targets)
    hpp_evidence = np.array(all_hpp_evidence)
    patient_ids = np.array(all_pids)
    subtypes = np.array(all_subtypes)

    print(f"Total training channels: {len(channels)}")
    print(f"  Involved (with discharge targets): {n_involved}")
    print(f"  Uninvolved (zero targets): {n_uninvolved}")

    return channels, targets, hpp_evidence, patient_ids, subtypes


# -- Training loop -----------------------------------------------------------

AUX_LAMBDA = 0.5

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y, hpp_ev in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        hpp_ev = hpp_ev.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)

        weights = torch.where(y > 0.1, POS_WEIGHT, 1.0)
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)

        discharge_mask = (y > 0.1).float()
        hpp_floor_violation = torch.clamp(hpp_ev - pred, min=0) * discharge_mask
        aux_loss = AUX_LAMBDA * (hpp_floor_violation ** 2).mean()

        sharpness_loss = SHARPNESS_PENALTY * pred.mean()

        loss = bce_loss + aux_loss + sharpness_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y, hpp_ev in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        hpp_ev = hpp_ev.to(DEVICE)

        pred = model(x)
        weights = torch.where(y > 0.1, POS_WEIGHT, 1.0)
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)
        discharge_mask = (y > 0.1).float()
        hpp_floor_violation = torch.clamp(hpp_ev - pred, min=0) * discharge_mask
        aux_loss = AUX_LAMBDA * (hpp_floor_violation ** 2).mean()
        sharpness_loss = SHARPNESS_PENALTY * pred.mean()
        loss = bce_loss + aux_loss + sharpness_loss

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("CET-UNet V3 Training (expanded expert discharge labels)")
    print("=" * 70)

    channels, targets, hpp_evidence, patient_ids, subtypes = prepare_cet_data()

    # Create 5-fold patient-stratified splits
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
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(pid)

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % N_FOLDS

    for fold in range(N_FOLDS):
        fold_pids = [p for p, f in patient_folds.items() if f == fold]
        fold_mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
        n_fold = int(np.sum(fold_mask))
        n_fold_involved = int(np.sum(np.max(targets[fold_mask], axis=1) > 0.1))
        print(f"  Fold {fold}: {len(fold_pids)} patients, {n_fold} channels "
              f"({n_fold_involved} involved)")

    # Train across folds
    print("\n" + "=" * 70)
    print("Training CET-UNet V3 across 5 folds...")
    print(f"  POS_WEIGHT={POS_WEIGHT}, SHARPNESS_PENALTY={SHARPNESS_PENALTY}, AUX_LAMBDA={AUX_LAMBDA}")
    print("=" * 70)

    all_peak_results = []

    for fold in range(N_FOLDS):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        val_mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_channels = channels[train_mask]
        train_targets = targets[train_mask]
        val_channels = channels[val_mask]
        val_targets = targets[val_mask]

        print(f"  Train: {len(train_channels)} channels, Val: {len(val_channels)} channels")

        train_hpp = hpp_evidence[train_mask]
        val_hpp = hpp_evidence[val_mask]
        train_ds = CETDataset(train_channels, train_targets, train_hpp, augment=True)
        val_ds = CETDataset(val_channels, val_targets, val_hpp, augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0)

        model = CETUNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, loader=train_loader,
                                         optimizer=optimizer, scheduler=scheduler)
            val_loss = evaluate_loss(model, val_loader)

            if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={PATIENCE})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save with v3 prefix
        save_path = CACHE_DIR / f'{SAVE_PREFIX}_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_loss={best_val_loss:.6f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # Save results
    results = {
        'model': 'CETUNet_v3',
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'pos_weight': POS_WEIGHT,
        'sharpness_penalty': SHARPNESS_PENALTY,
    }
    results_path = CACHE_DIR / f'{SAVE_PREFIX}_cv_results.json'
    with open(str(results_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
