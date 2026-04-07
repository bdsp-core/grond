"""
HemiCET Optimization — Experiments E5, E6, E7 (E4 skipped — too slow)

Runs sequentially:
  E5: HemiCET with 10 channels (8 hemisphere + Fz-Cz idx16 + Cz-Pz idx17)
  E6: Multi-segment training (up to 5 segments per patient, standard 8ch)
  E7: Gold-standard freq oracle test (how much room from better freq estimation?)
      + optional CNN freq retrain evaluation

Current best (C1): F1=0.8908
  dp_alpha=1.5, dp_beta=0.3, dp_lambda=0.05,
  evidence_threshold_pct=50, min_evidence_ratio=0.4

Usage:
    conda run -n foe_dl python code/hemi_detector/run_e5_e6_e7.py

Results saved to:
    data/hemi_cache/optimization/{e5_midline,e6_multiseg,e7_freq}.json
Dashboard updated after each experiment:
    results/hemicet_optimization_dashboard.html
"""

import sys, json, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt, find_peaks

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, detect_active_interval,
    dp_best_sequence, em_refine, posthoc_filter, compute_channel_evidence,
    estimate_frequency_acf,
)
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence
from hemi_detector.hemi_cet import HemiCET, count_parameters

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Device: {DEVICE}")

TOLERANCE_S = 0.1
RESULTS_DIR = PROJECT_DIR / 'data' / 'hemi_cache' / 'optimization'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_PATH = PROJECT_DIR / 'results' / 'hemicet_optimization_dashboard.html'

# C1 best params
C1_PARAMS = dict(
    dp_alpha=1.5,
    dp_beta=0.3,
    dp_lambda=0.05,
    peak_height_frac=0.05,
    max_skip=3,
    evidence_threshold_pct=50,
    min_evidence_ratio=0.4,
)

# Training hypers (matching train_hemi_cet.py)
N_FOLDS = 5
N_EPOCHS = 80
BATCH_SIZE = 32
LR_SCRATCH = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT = 20.0
SHARPNESS_PENALTY = 0.1
HPP_FLOOR_LAMBDA = 0.05
GAUSSIAN_SIGMA = 2
N_SAMPLES = 2000

# E5 channel lists: 8 hemisphere + Fz-Cz (idx 16) + Cz-Pz (idx 17)
LEFT_INDICES_10  = list(LEFT_INDICES)  + [16, 17]
RIGHT_INDICES_10 = list(RIGHT_INDICES) + [16, 17]


# ============================================================================
# Utilities
# ============================================================================

def zscore_channels(x):
    """x: (C, T) -> z-scored per channel."""
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    return (x - mu) / std


def make_target_signal(discharge_times_s, n_samples=N_SAMPLES, fs=FS,
                       sigma=GAUSSIAN_SIGMA, jitter_sigma=0.0):
    target = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32)
    for dt in discharge_times_s:
        center = dt * fs
        if jitter_sigma > 0:
            center += np.random.randn() * jitter_sigma
        bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        target = np.maximum(target, bump)
    return target


# ============================================================================
# Models
# ============================================================================

class HemiCET10(nn.Module):
    """HemiCET with 10 input channels (8 hemisphere + 2 midline)."""

    def __init__(self, in_channels: int = 10):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.skip4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.skip3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.skip2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _match_size(d, e):
        min_len = min(d.shape[2], e.shape[2])
        return d[:, :, :min_len], e[:, :, :min_len]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d4 = self.up4(e4)
        d4, e3m = self._match_size(d4, e3)
        d4 = self.skip4(torch.cat([d4, e3m], dim=1))
        d3 = self.up3(d4)
        d3, e2m = self._match_size(d3, e2)
        d3 = self.skip3(torch.cat([d3, e2m], dim=1))
        d2 = self.up2(d3)
        d2, e1m = self._match_size(d2, e1)
        d2 = self.skip2(torch.cat([d2, e1m], dim=1))
        d1 = self.up1(d2)
        return self.head(d1)


# ============================================================================
# Dataset
# ============================================================================

class HemiCETDataset(Dataset):
    """Generic HemiCET dataset supporting any channel count."""

    def __init__(self, hemi_data, targets, hpp_evidence, augment=False):
        self.hemi_data = [x.astype(np.float32) for x in hemi_data]
        self.targets = targets.astype(np.float32)
        self.hpp_evidence = hpp_evidence.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.hemi_data)

    def _zscore(self, x):
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        std = np.where(std > 1e-8, std, 1.0)
        return (x - mu) / std

    def __getitem__(self, idx):
        x = self.hemi_data[idx].copy()
        y = self.targets[idx].copy()
        h = self.hpp_evidence[idx].copy()

        x = self._zscore(x)

        if self.augment:
            scale = np.random.uniform(0.7, 1.3)
            x = x * scale
            for ch in range(x.shape[0]):
                ch_std = np.std(x[ch])
                noise = np.random.randn(x.shape[1]) * 0.05 * (ch_std + 1e-8)
                x[ch] = x[ch] + noise.astype(np.float32)
            if np.random.rand() < 0.15:
                drop_ch = np.random.randint(0, x.shape[0])
                x[drop_ch] = 0.0

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y[np.newaxis, :])
        h_t = torch.from_numpy(h[np.newaxis, :])
        return x_t, y_t, h_t


# ============================================================================
# Training
# ============================================================================

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


def make_patient_folds(patient_ids, subtypes, n_folds=N_FOLDS, seed=42):
    rng = np.random.RandomState(seed)
    unique_pids = np.unique(patient_ids)

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
            patient_folds[pid] = i % n_folds

    return patient_folds


def train_5fold(model_class, hemi_data, targets, hpp_evidence, patient_ids, subtypes,
                save_dir, device, n_epochs=N_EPOCHS, lr=LR_SCRATCH,
                model_kwargs=None, verbose=True):
    """Train 5-fold CV and save best models. Returns fold val losses."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if model_kwargs is None:
        model_kwargs = {}

    patient_folds = make_patient_folds(patient_ids, subtypes)

    if verbose:
        for fold in range(N_FOLDS):
            mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
            n_fold_pats = len([p for p, f in patient_folds.items() if f == fold])
            print(f"  Fold {fold}: {n_fold_pats} patients, {mask.sum()} examples")

    fold_results = []
    all_val_losses = []

    for fold in range(N_FOLDS):
        fold_t0 = time.time()
        print(f"\n  --- Fold {fold+1}/{N_FOLDS} ---")

        val_mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_ds = HemiCETDataset(
            list(hemi_data[train_mask]), targets[train_mask], hpp_evidence[train_mask],
            augment=True)
        val_ds = HemiCETDataset(
            list(hemi_data[val_mask]), targets[val_mask], hpp_evidence[val_mask],
            augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss = evaluate_val_loss(model, val_loader, device)

            if epoch % 20 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                      f"train={train_loss:.6f} val={val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        save_path = save_dir / f'hemi_cet_fold{fold}.pt'
        torch.save(best_state if best_state is not None else model.state_dict(), str(save_path))
        all_val_losses.append(best_val_loss)
        fold_results.append({'fold': fold, 'best_val_loss': best_val_loss})
        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold+1} done: best_val_loss={best_val_loss:.6f} ({fold_elapsed:.1f}s)")

    print(f"\n  Mean best val loss: {np.mean(all_val_losses):.6f}")
    return fold_results


def load_hemicet_models_from_dir(model_dir, n_folds=N_FOLDS, model_class=None, model_kwargs=None):
    if model_class is None:
        model_class = HemiCET
    if model_kwargs is None:
        model_kwargs = {}
    model_dir = Path(model_dir)
    models = []
    for fold in range(n_folds):
        p = model_dir / f'hemi_cet_fold{fold}.pt'
        if p.exists():
            m = model_class(**model_kwargs)
            state = torch.load(str(p), map_location=DEVICE, weights_only=True)
            m.load_state_dict(state)
            m.to(DEVICE)
            m.eval()
            models.append(m)
        else:
            print(f"  WARNING: fold {fold} model not found at {p}")
    print(f"  Loaded {len(models)} fold models from {model_dir}")
    return models


# ============================================================================
# Data preparation
# ============================================================================

def load_gt_cases(df):
    """Load ground-truth cases from discharge_times.json."""
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    gt_cases = {}
    for pid, v in hpp_data.items():
        if v.get('review_status') != 'ground_truth':
            continue
        if len(v.get('global_times', [])) < 2:
            continue
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            v['subtype'] = row.iloc[0]['subtype']
            lat = row.iloc[0].get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = None
            v['laterality'] = lat
        gt_cases[pid] = v
    return gt_cases


def prepare_data_nch(segments, channel_lists, verbose=True):
    """Prepare training data with custom channel lists (for E5 10-ch)."""
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    gt_cases_raw = {pid: v for pid, v in hpp_data.items()
                    if v.get('review_status') == 'ground_truth'}

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    left_ch, right_ch = channel_lists
    n_ch = len(left_ch)

    if verbose:
        print(f"  GT cases: {len(gt_cases_raw)}, channels per hemi: {n_ch}")

    all_hemi, all_targets, all_hpp, all_pids, all_subtypes = [], [], [], [], []

    for pid, gt_data in gt_cases_raw.items():
        discharge_times = gt_data['global_times']
        if len(discharge_times) < 2:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        if seg.shape[1] != N_SAMPLES:
            continue
        if seg.shape[0] < 18:
            continue  # need at least 18 channels for midline indices 16, 17

        subtype = gt_data.get('subtype', 'unknown')
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            lat = row.iloc[0].get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = 'unknown'
        else:
            lat = 'unknown'

        discharge_target = make_target_signal(discharge_times, jitter_sigma=0.0)

        def add_hemi(indices):
            hemi_raw = np.zeros((n_ch, N_SAMPLES), dtype=np.float32)
            hpp_ch = np.zeros((n_ch, N_SAMPLES), dtype=np.float32)
            for i, ci in enumerate(indices):
                if ci >= seg.shape[0]:
                    continue
                ch = seg[ci].astype(np.float32)
                ch = np.nan_to_num(ch)
                hemi_raw[i] = ch
                try:
                    hpp_ch[i] = _compute_channel_evidence(ch, FS).astype(np.float32)
                except Exception:
                    pass
            for i in range(n_ch):
                mx = hpp_ch[i].max()
                if mx > 1e-8:
                    hpp_ch[i] /= mx
            hpp_med = np.median(hpp_ch, axis=0)
            all_hemi.append(hemi_raw)
            all_targets.append(discharge_target.copy())
            all_hpp.append(hpp_med.astype(np.float32))
            all_pids.append(pid)
            all_subtypes.append(subtype)

        if subtype == 'gpd':
            add_hemi(left_ch)
            add_hemi(right_ch)
        elif lat == 'left':
            add_hemi(left_ch)
        elif lat == 'right':
            add_hemi(right_ch)
        else:
            add_hemi(left_ch)
            add_hemi(right_ch)

    if verbose:
        print(f"  Total examples: {len(all_hemi)}")

    return (np.array(all_hemi, dtype=object),
            np.array(all_targets, dtype=np.float32),
            np.array(all_hpp, dtype=np.float32),
            np.array(all_pids),
            np.array(all_subtypes))


def prepare_multiseg_data(verbose=True):
    """Build training data using up to 5 segments per patient (E6)."""
    import scipy.io as sio

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    gt_cases_raw = {pid: v for pid, v in hpp_data.items()
                    if v.get('review_status') == 'ground_truth'}

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    eeg_dir = PROJECT_DIR / 'data' / 'eeg'

    if verbose:
        print(f"  GT cases: {len(gt_cases_raw)}")

    all_hemi, all_targets, all_hpp, all_pids, all_subtypes = [], [], [], [], []
    n_multiseg = n_single = 0

    for pid, gt_data in gt_cases_raw.items():
        discharge_times = gt_data['global_times']
        if len(discharge_times) < 2:
            continue

        seg_files = sorted(eeg_dir.glob(f'{pid}_seg*.mat'))[:5]
        if not seg_files:
            continue

        if len(seg_files) > 1:
            n_multiseg += 1
        else:
            n_single += 1

        subtype = gt_data.get('subtype', 'unknown')
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            lat = row.iloc[0].get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = 'unknown'
        else:
            lat = 'unknown'

        discharge_target = make_target_signal(discharge_times, jitter_sigma=0.0)

        for seg_file in seg_files:
            try:
                mat = sio.loadmat(str(seg_file))
                seg = mat['data'].astype(np.float32)
            except Exception:
                continue

            n_ch_raw = seg.shape[0]
            if seg.shape[1] != N_SAMPLES:
                if seg.shape[1] > N_SAMPLES:
                    seg = seg[:, :N_SAMPLES]
                else:
                    pad = np.zeros((n_ch_raw, N_SAMPLES - seg.shape[1]), dtype=np.float32)
                    seg = np.hstack([seg, pad])

            def add_hemi_ms(hemi_indices):
                hemi_raw = np.zeros((8, N_SAMPLES), dtype=np.float32)
                hpp_ch = np.zeros((8, N_SAMPLES), dtype=np.float32)
                for i, ci in enumerate(hemi_indices):
                    if ci >= seg.shape[0]:
                        continue
                    ch = seg[ci].astype(np.float32)
                    ch = np.nan_to_num(ch)
                    hemi_raw[i] = ch
                    try:
                        hpp_ch[i] = _compute_channel_evidence(ch, FS).astype(np.float32)
                    except Exception:
                        pass
                for i in range(8):
                    mx = hpp_ch[i].max()
                    if mx > 1e-8:
                        hpp_ch[i] /= mx
                hpp_med = np.median(hpp_ch, axis=0)
                all_hemi.append(hemi_raw)
                all_targets.append(discharge_target.copy())
                all_hpp.append(hpp_med.astype(np.float32))
                all_pids.append(pid)
                all_subtypes.append(subtype)

            if subtype == 'gpd':
                add_hemi_ms(LEFT_INDICES)
                add_hemi_ms(RIGHT_INDICES)
            elif lat == 'left':
                add_hemi_ms(LEFT_INDICES)
            elif lat == 'right':
                add_hemi_ms(RIGHT_INDICES)
            else:
                add_hemi_ms(LEFT_INDICES)
                add_hemi_ms(RIGHT_INDICES)

    if verbose:
        print(f"  Multi-segment patients: {n_multiseg}, single-segment: {n_single}")
        print(f"  Total examples: {len(all_hemi)}")

    return (np.array(all_hemi, dtype=object),
            np.array(all_targets, dtype=np.float32),
            np.array(all_hpp, dtype=np.float32),
            np.array(all_pids),
            np.array(all_subtypes))


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_predictions(predictions, gt_cases, subtype_filter=None):
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs, match_errors = [], [], []

    for pid, algo_times in predictions.items():
        if pid not in gt_cases:
            continue
        gt_data = gt_cases[pid]
        if subtype_filter and gt_data.get('subtype') != subtype_filter:
            continue
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue
        algo_times = sorted(algo_times)

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_d, best_a = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_d:
                        best_d, best_a = d, ai
            if best_d <= TOLERANCE_S and best_a >= 0:
                gt_matched[gi] = True
                algo_matched[best_a] = True
                match_errors.append(best_d)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = np.diff(gt_times)
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        if len(algo_times) >= 2:
            algo_freq = 1.0 / np.median(np.diff(algo_times))
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = spearmanr(algo_freqs, gt_freqs)[0] if len(gt_freqs) >= 3 else float('nan')
    freq_mae = float(np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs)))) if len(gt_freqs) >= 3 else float('nan')
    timing_med = float(np.median(match_errors) * 1000) if match_errors else float('nan')

    return dict(
        f1=round(f1, 4), sens=round(sens, 4), prec=round(prec, 4),
        freq_rho=round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        freq_mae=round(freq_mae, 3) if np.isfinite(freq_mae) else None,
        timing_med_ms=round(timing_med, 1) if np.isfinite(timing_med) else None,
        n_cases=len(predictions),
        gt_freqs=[round(f, 3) for f in gt_freqs],
        algo_freqs=[round(f, 3) for f in algo_freqs],
    )


def full_evaluate(predictions, gt_cases):
    overall = evaluate_predictions(predictions, gt_cases)
    lpd = evaluate_predictions(predictions, gt_cases, subtype_filter='lpd')
    gpd = evaluate_predictions(predictions, gt_cases, subtype_filter='gpd')
    return dict(overall=overall, lpd=lpd, gpd=gpd)


# ============================================================================
# HemiCET inference + DP (C1 params)
# ============================================================================

@torch.no_grad()
def run_hemicet_dp(seg, subtype, laterality, hemi_cet_models, detector,
                   dp_alpha=1.5, dp_beta=0.3, dp_lambda=0.05,
                   peak_height_frac=0.05, max_skip=3,
                   evidence_threshold_pct=50, min_evidence_ratio=0.4,
                   hemi_channels=None, gold_freq=None):
    """Run HemiCET+DP inference.

    hemi_channels: override LEFT/RIGHT_INDICES for E5 (10ch).
    gold_freq: if not None, use this as the discharge frequency instead of CNN estimate (E7 oracle).
    """

    def _run(indices):
        # --- Frequency estimation ---
        if gold_freq is not None:
            freq = float(np.clip(gold_freq, 0.3, 3.5))
        else:
            # CNN-based freq estimation (first 8 channels for compatibility)
            all_pd, all_lf = [], []
            for ci in indices[:8]:
                ch = seg[ci].astype(np.float32).copy()
                if not np.all(np.isfinite(ch)):
                    all_pd.append(0.0); all_lf.append(0.0); continue
                mu, std = np.mean(ch), np.std(ch)
                ch = (ch - mu) / std if std > 1e-8 else ch - mu
                x = torch.from_numpy(ch[None, None, :]).to(detector.device)
                pp, lf = [], []
                for m in detector.cnn_models:
                    p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
                all_pd.append(np.mean(pp)); all_lf.append(np.mean(lf))

            pdw = np.array(all_pd); lfs = np.array(all_lf); ws = pdw.sum()
            cnn_freq = float(np.clip(
                np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))

            b, a = butter(4, 20.0 / (FS / 2), btype='low')
            acfs = []
            for ci in indices[:8]:
                try:
                    sig = filtfilt(b, a, seg[ci])
                except Exception:
                    sig = seg[ci]
                f2 = estimate_frequency_acf(sig, FS)
                if np.isfinite(f2):
                    acfs.append(f2)
            acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn_freq
            freq = float(np.clip(0.8 * cnn_freq + 0.2 * acf, 0.3, 3.5))

        # --- HemiCET evidence ---
        n_ch = len(indices)
        hs = np.zeros((n_ch, N_SAMPLES), dtype=np.float32)
        for i, ci in enumerate(indices):
            if ci < seg.shape[0]:
                ch_data = seg[ci].astype(np.float32).copy()
                mu2, std2 = np.mean(ch_data), np.std(ch_data)
                hs[i] = (ch_data - mu2) / std2 if std2 > 1e-8 else ch_data - mu2

        x2 = torch.from_numpy(hs[None]).to(DEVICE)
        preds = [m(x2).squeeze().cpu().numpy() for m in hemi_cet_models]
        ev = np.mean(preds, axis=0)

        if evidence_threshold_pct > 0 and np.any(ev > 0):
            thr = np.percentile(ev[ev > 0], evidence_threshold_pct)
            ev = np.where(ev > thr, ev, 0)

        active_start, active_end = detect_active_interval(ev, FS)
        segment = ev[active_start:active_end + 1]
        if len(segment) < 3:
            return []

        T = 1.0 / freq if freq > 0 else 1.0
        min_dist = max(20, int(0.2 * T * FS))
        min_height = peak_height_frac * np.max(segment)
        peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
        strong_height = 0.5 * np.max(segment)
        strong_peaks, _ = find_peaks(segment, height=strong_height,
                                     distance=max(10, int(0.1 * T * FS)))
        candidates = np.unique(np.concatenate([peaks, strong_peaks])) + active_start

        if len(candidates) == 0:
            return []
        if len(candidates) == 1:
            ds = candidates.copy()
        else:
            n = len(candidates)
            raw_scores = np.array([ev[c] for c in candidates])
            node_scores = raw_scores ** 1.5
            best_score = np.full(n, -np.inf)
            best_prev = np.full(n, -1, dtype=int)
            for i in range(n):
                best_score[i] = node_scores[i] - dp_lambda
            for j in range(1, n):
                for i in range(j):
                    dt = (candidates[j] - candidates[i]) / FS
                    if dt <= 0 or dt > 4 * T:
                        continue
                    best_edge = -np.inf
                    for m in range(1, max_skip + 1):
                        deviation = (dt - m * T) / (m * T)
                        interval_score = -dp_alpha * deviation ** 2
                        skip_penalty = -dp_beta * (m - 1)
                        edge = interval_score + skip_penalty
                        if edge > best_edge:
                            best_edge = edge
                    total = best_score[i] + best_edge + node_scores[j] - dp_lambda
                    if total > best_score[j]:
                        best_score[j] = total
                        best_prev[j] = i
            path = []
            idx = int(np.argmax(best_score))
            while idx >= 0:
                path.append(idx)
                idx = best_prev[idx]
            path.reverse()
            ds = candidates[np.array(path)]

        if len(ds) >= 3:
            ds = em_refine(ev, ds, FS, freq)

        if min_evidence_ratio > 0 and len(ds) >= 2:
            peak_vals = np.array([ev[int(s)] for s in ds])
            threshold = min_evidence_ratio * np.median(peak_vals)
            ds = ds[peak_vals >= threshold]

        return (ds / FS).tolist() if len(ds) > 0 else []

    left_ch = hemi_channels[0] if hemi_channels else LEFT_INDICES
    right_ch = hemi_channels[1] if hemi_channels else RIGHT_INDICES

    if subtype == 'gpd' or laterality not in ('left', 'right'):
        tl, tr = _run(left_ch), _run(right_ch)
        return tl if len(tl) >= len(tr) else tr
    return _run(left_ch if laterality == 'left' else right_ch)


def run_eval_with_models(hemi_cet_models, gt_cases, segments, detector,
                         params=None, hemi_channels=None, gold_freq_map=None):
    """Evaluate HemiCET models with given params over all GT cases."""
    if params is None:
        params = C1_PARAMS
    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        gold_freq = gold_freq_map.get(pid) if gold_freq_map else None
        try:
            times = run_hemicet_dp(
                seg, gt['subtype'], gt.get('laterality'),
                hemi_cet_models, detector,
                hemi_channels=hemi_channels,
                gold_freq=gold_freq,
                **params)
            preds[pid] = times
        except Exception as e:
            preds[pid] = []
    return full_evaluate(preds, gt_cases)


# ============================================================================
# Dashboard
# ============================================================================

def load_existing_dashboard_experiments():
    """Load completed prior experiment results from JSON files."""
    experiments = []
    name_map = {
        'baseline':    ('baseline',    'Baseline HemiCET+DP'),
        'e2_dp_sweep': ('e2_dp_sweep', 'E2: DP param re-optimization'),
        'e3_posthoc':  ('e3_posthoc',  'E3: Post-hoc filtering sweep'),
        'c1_combined': ('c1_combined', 'C1: E2+E3 combined (best=0.8908)'),
    }
    for fname, (exp_id, exp_name) in name_map.items():
        fpath = RESULTS_DIR / f'{fname}.json'
        if fpath.exists():
            with open(str(fpath)) as f:
                result = json.load(f)
            experiments.append({
                'id': exp_id,
                'name': exp_name,
                'status': 'done',
                'result': result,
            })
    # Also load any previously completed E5/E6/E7
    for fname, (exp_id, exp_name) in [
        ('e5_midline',  ('e5_midline',  'E5: Midline 10ch')),
        ('e6_multiseg', ('e6_multiseg', 'E6: Multi-segment')),
        ('e7_freq',     ('e7_freq',     'E7: Freq oracle / retrain')),
    ]:
        fpath = RESULTS_DIR / f'{fname}.json'
        if fpath.exists():
            with open(str(fpath)) as f:
                result = json.load(f)
            experiments.append({
                'id': exp_id,
                'name': exp_name,
                'status': 'done',
                'result': result,
            })
    return experiments


def build_dashboard(experiments, baseline):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    rows_html = ""
    for exp in experiments:
        r = exp.get('result', {}).get('overall', {})
        lpd = exp.get('result', {}).get('lpd', {})
        gpd = exp.get('result', {}).get('gpd', {})
        status = exp.get('status', 'pending')

        status_color = {
            'pending': '#888', 'running': '#ff9800',
            'done': '#44cc88', 'error': '#ff4444'
        }.get(status, '#888')

        f1 = r.get('f1', '')
        f1_delta = ''
        baseline_f1 = baseline.get('overall', {}).get('f1') if baseline else None
        if f1 and baseline_f1:
            delta = f1 - baseline_f1
            f1_delta = f' ({delta:+.4f})'

        rows_html += f"""<tr>
  <td>{exp['name']}</td>
  <td style="color:{status_color};font-weight:bold">{status.upper()}</td>
  <td><b>{f1 or '&mdash;'}</b>{f1_delta}</td>
  <td>{r.get('sens', '&mdash;')}</td>
  <td>{r.get('prec', '&mdash;')}</td>
  <td>{r.get('freq_rho', '&mdash;')}</td>
  <td>{r.get('freq_mae', '&mdash;')}</td>
  <td>{r.get('timing_med_ms', '&mdash;')}</td>
  <td>{lpd.get('f1', '&mdash;')}</td>
  <td>{gpd.get('f1', '&mdash;')}</td>
</tr>"""

    scatter_js = "const SCATTER_DATA = {};\n"
    for exp in experiments:
        if exp.get('status') == 'done' and exp.get('result'):
            for sub in ['lpd', 'gpd']:
                r_sub = exp['result'].get(sub, {})
                gt = r_sub.get('gt_freqs', [])
                algo = r_sub.get('algo_freqs', [])
                if gt and algo:
                    key = f"{exp['id']}_{sub}"
                    scatter_js += (f"SCATTER_DATA['{key}'] = {{gt: {json.dumps(gt)}, "
                                   f"algo: {json.dumps(algo)}, "
                                   f"name: '{exp['name']} ({sub.upper()})'}};\n")

    if baseline:
        for sub in ['lpd', 'gpd']:
            r_sub = baseline.get(sub, {})
            gt = r_sub.get('gt_freqs', [])
            algo = r_sub.get('algo_freqs', [])
            if gt and algo:
                scatter_js += (f"SCATTER_DATA['baseline_{sub}'] = {{gt: {json.dumps(gt)}, "
                               f"algo: {json.dumps(algo)}, name: 'Baseline ({sub.upper()})'}};\n")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="15">
<title>HemiCET Optimization Dashboard</title>
<style>
  * {{box-sizing:border-box;margin:0;padding:0}}
  body {{background:#1a1a1a;color:#eee;font-family:'Consolas','Monaco',monospace;padding:20px}}
  h1 {{color:#ff9800;margin-bottom:10px}}
  .updated {{color:#666;font-size:12px;margin-bottom:20px}}
  table {{border-collapse:collapse;width:100%;margin-bottom:20px}}
  th {{background:#333;padding:8px 12px;text-align:left;font-size:12px;color:#aaa;border-bottom:2px solid #444}}
  td {{padding:6px 12px;border-bottom:1px solid #333;font-size:13px}}
  tr:hover {{background:#252525}}
  .best {{color:#44ff66;font-weight:bold}}
  .scatter-container {{display:flex;flex-wrap:wrap;gap:20px;margin-top:20px}}
  canvas {{background:#222;border-radius:8px}}
</style>
</head><body>
<h1>HemiCET+DP Optimization — E5/E6/E7</h1>
<div class="updated">Updated: {now} | C1 best: F1=0.8908 | Running: E5 (midline 10ch), E6 (multi-seg), E7 (freq oracle)</div>

<table>
<tr>
  <th>Experiment</th><th>Status</th><th>F1 (vs baseline)</th><th>Sens</th><th>Prec</th>
  <th>Freq rho</th><th>Freq MAE</th><th>Timing Med (ms)</th><th>LPD F1</th><th>GPD F1</th>
</tr>
{rows_html}
</table>

<div class="scatter-container" id="scatter-container"></div>

<script>
{scatter_js}

function drawScatter(canvasId, data, title) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = 350, H = 350, M = 50;
  canvas.width = W; canvas.height = H;
  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, W, H);
  const maxVal = Math.max(Math.max(...data.gt, ...data.algo) * 1.1, 4);
  function toX(v) {{ return M + (v / maxVal) * (W - M - 20); }}
  function toY(v) {{ return H - M - (v / maxVal) * (H - M - 30); }}
  ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
  for (let v = 1; v <= maxVal; v++) {{
    ctx.beginPath(); ctx.moveTo(toX(v), M-10); ctx.lineTo(toX(v), H-M); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(M, toY(v)); ctx.lineTo(W-20, toY(v)); ctx.stroke();
  }}
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(maxVal), toY(maxVal)); ctx.stroke();
  ctx.setLineDash([]);
  for (let i = 0; i < data.gt.length; i++) {{
    const diff = Math.abs(data.gt[i] - data.algo[i]);
    ctx.fillStyle = diff > 0.5 ? 'rgba(255,80,80,0.6)' : diff > 0.25 ? 'rgba(255,180,50,0.5)' : 'rgba(70,130,255,0.5)';
    ctx.beginPath(); ctx.arc(toX(data.gt[i]), toY(data.algo[i]), 3, 0, Math.PI*2); ctx.fill();
  }}
  ctx.fillStyle = '#aaa'; ctx.font = '11px Consolas'; ctx.textAlign = 'center';
  ctx.fillText('GT Frequency (Hz)', W/2, H-8);
  ctx.save(); ctx.translate(12, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Algo Frequency (Hz)', 0, 0); ctx.restore();
  ctx.fillStyle = '#eee'; ctx.font = 'bold 12px Consolas'; ctx.textAlign = 'center';
  ctx.fillText(title, W/2, 16);
}}

const container = document.getElementById('scatter-container');
const keys = Object.keys(SCATTER_DATA);
keys.forEach((key, i) => {{
  const canvas = document.createElement('canvas');
  canvas.id = 'scatter_' + i;
  canvas.width = 350; canvas.height = 350;
  container.appendChild(canvas);
  drawScatter('scatter_' + i, SCATTER_DATA[key], SCATTER_DATA[key].name);
}});
</script>
</body></html>"""

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(DASHBOARD_PATH), 'w') as f:
        f.write(html)
    print(f"  Dashboard updated: {DASHBOARD_PATH}")


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  HemiCET Optimization — E5 (midline 10ch), E6 (multi-seg), E7 (freq)")
    print("  E4 skipped (MAE pretraining too slow)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # Load shared data
    print("\nLoading dataset and GT cases...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']
    gt_cases = load_gt_cases(df)
    print(f"GT cases: {len(gt_cases)}")

    detector = DischargeDetector()

    # Load v2 models (C1 baseline)
    v2_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_v2'
    v2_models = load_hemicet_models_from_dir(v2_dir)

    # Build gold_freq_map for E7 oracle test
    gold_freq_map = {}
    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        freq = row.get('gold_standard_freq', float('nan'))
        if np.isfinite(freq) and freq > 0:
            gold_freq_map[pid] = float(freq)
    print(f"Gold freq available for {len(gold_freq_map)} patients")

    # Load existing experiments for dashboard
    experiments = load_existing_dashboard_experiments()

    # Load baseline for delta display
    baseline_path = RESULTS_DIR / 'baseline.json'
    if baseline_path.exists():
        with open(str(baseline_path)) as f:
            baseline = json.load(f)
    else:
        baseline = {}

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E5: Midline channels (10ch)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E5: Midline Channels (10ch — 8 hemi + Fz-Cz + Cz-Pz)")
    print(f"{'='*70}")

    e5_exp = {'id': 'e5_midline', 'name': 'E5: Midline 10ch', 'status': 'running'}
    # Remove any pre-loaded version to replace with running status
    experiments = [e for e in experiments if e['id'] != 'e5_midline']
    experiments.append(e5_exp)
    build_dashboard(experiments, baseline)

    e5_result = None
    try:
        e5_t0 = time.time()
        e5_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e5'
        e5_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  LEFT_INDICES_10:  {LEFT_INDICES_10}")
        print(f"  RIGHT_INDICES_10: {RIGHT_INDICES_10}")

        print("\n  Preparing 10-ch training data...")
        hemi_data_10, targets_10, hpp_10, pids_10, subs_10 = prepare_data_nch(
            segments,
            channel_lists=(LEFT_INDICES_10, RIGHT_INDICES_10),
            verbose=True)

        print(f"\n  Training 5-fold CV (HemiCET10, 10ch)...")
        train_5fold(
            model_class=HemiCET10,
            hemi_data=hemi_data_10,
            targets=targets_10,
            hpp_evidence=hpp_10,
            patient_ids=pids_10,
            subtypes=subs_10,
            save_dir=e5_save_dir,
            device=DEVICE,
            n_epochs=N_EPOCHS,
            lr=LR_SCRATCH,
            model_kwargs={'in_channels': 10},
            verbose=True,
        )

        print("\n  Evaluating E5 models with C1 params...")
        e5_models = load_hemicet_models_from_dir(
            e5_save_dir, model_class=HemiCET10, model_kwargs={'in_channels': 10})

        e5_result = run_eval_with_models(
            e5_models, gt_cases, segments, detector, C1_PARAMS,
            hemi_channels=(LEFT_INDICES_10, RIGHT_INDICES_10))

        e5_elapsed = time.time() - e5_t0
        print(f"\n  E5 RESULT: F1={e5_result['overall']['f1']} "
              f"sens={e5_result['overall']['sens']} "
              f"prec={e5_result['overall']['prec']}")
        print(f"    LPD F1={e5_result['lpd']['f1']}, GPD F1={e5_result['gpd']['f1']}")
        print(f"    freq_rho={e5_result['overall'].get('freq_rho')}  "
              f"({e5_elapsed:.0f}s elapsed)")

        e5_exp['result'] = e5_result
        e5_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e5_midline.json'), 'w') as f:
            json.dump(e5_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"\n  E5 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e5_exp['status'] = 'error'

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E6: Multi-segment training
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E6: Multi-Segment Training (up to 5 segs/patient, standard 8ch)")
    print(f"{'='*70}")

    e6_exp = {'id': 'e6_multiseg', 'name': 'E6: Multi-segment', 'status': 'running'}
    experiments = [e for e in experiments if e['id'] != 'e6_multiseg']
    experiments.append(e6_exp)
    build_dashboard(experiments, baseline)

    e6_result = None
    try:
        e6_t0 = time.time()
        e6_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e6'
        e6_save_dir.mkdir(parents=True, exist_ok=True)

        print("\n  Preparing multi-segment training data...")
        hemi_ms, targets_ms, hpp_ms, pids_ms, subs_ms = prepare_multiseg_data(verbose=True)

        print(f"\n  Training 5-fold CV (HemiCET 8ch, multi-seg data)...")
        train_5fold(
            model_class=HemiCET,
            hemi_data=hemi_ms,
            targets=targets_ms,
            hpp_evidence=hpp_ms,
            patient_ids=pids_ms,
            subtypes=subs_ms,
            save_dir=e6_save_dir,
            device=DEVICE,
            n_epochs=N_EPOCHS,
            lr=LR_SCRATCH,
            verbose=True,
        )

        print("\n  Evaluating E6 models with C1 params...")
        e6_models = load_hemicet_models_from_dir(e6_save_dir)
        e6_result = run_eval_with_models(e6_models, gt_cases, segments, detector, C1_PARAMS)

        e6_elapsed = time.time() - e6_t0
        print(f"\n  E6 RESULT: F1={e6_result['overall']['f1']} "
              f"sens={e6_result['overall']['sens']} "
              f"prec={e6_result['overall']['prec']}")
        print(f"    LPD F1={e6_result['lpd']['f1']}, GPD F1={e6_result['gpd']['f1']}")
        print(f"    freq_rho={e6_result['overall'].get('freq_rho')}  "
              f"({e6_elapsed:.0f}s elapsed)")

        e6_exp['result'] = e6_result
        e6_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e6_multiseg.json'), 'w') as f:
            json.dump(e6_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"\n  E6 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e6_exp['status'] = 'error'

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E7: Frequency oracle test
    # How much room is there from better frequency estimation?
    # Substitute gold_standard_freq as the DP prior instead of CNN-estimated freq.
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E7: Frequency Oracle Test")
    print("  (Use gold_standard_freq as DP prior — cheating reference upper bound)")
    print(f"{'='*70}")

    e7_exp = {'id': 'e7_freq', 'name': 'E7: Gold freq oracle (HemiCET v2)', 'status': 'running'}
    experiments = [e for e in experiments if e['id'] != 'e7_freq']
    experiments.append(e7_exp)
    build_dashboard(experiments, baseline)

    e7_result = None
    try:
        e7_t0 = time.time()

        print(f"  Gold freq map: {len(gold_freq_map)} patients")
        print("  Evaluating HemiCET v2 + C1 params + gold_standard_freq as DP prior...")

        e7_result = run_eval_with_models(
            v2_models, gt_cases, segments, detector, C1_PARAMS,
            gold_freq_map=gold_freq_map)

        e7_elapsed = time.time() - e7_t0
        print(f"\n  E7 RESULT (freq oracle): F1={e7_result['overall']['f1']} "
              f"sens={e7_result['overall']['sens']} "
              f"prec={e7_result['overall']['prec']}")
        print(f"    LPD F1={e7_result['lpd']['f1']}, GPD F1={e7_result['gpd']['f1']}")
        print(f"    freq_rho={e7_result['overall'].get('freq_rho')}  "
              f"({e7_elapsed:.0f}s elapsed)")

        c1_f1 = 0.8908
        oracle_f1 = e7_result['overall']['f1']
        print(f"\n  INTERPRETATION: C1={c1_f1} -> oracle={oracle_f1} "
              f"(gap={oracle_f1-c1_f1:+.4f})")
        print(f"  Freq improvement headroom: {oracle_f1 - c1_f1:.4f} F1 points")

        e7_result['experiment_note'] = (
            'E7: gold_standard_freq used as DP prior (oracle/cheating upper bound). '
            f'C1 baseline F1={c1_f1}. Oracle F1={oracle_f1}. '
            f'Headroom={oracle_f1-c1_f1:+.4f}.'
        )

        e7_exp['result'] = e7_result
        e7_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e7_freq.json'), 'w') as f:
            json.dump(e7_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"\n  E7 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e7_exp['status'] = 'error'

    build_dashboard(experiments, baseline)

    # =========================================================================
    # Summary
    # =========================================================================
    total_elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  C1 baseline:  F1=0.8908")

    if e5_result:
        f1 = e5_result['overall']['f1']
        print(f"  E5 (10ch):    F1={f1}  ({f1-0.8908:+.4f})")
    else:
        print("  E5 (10ch):    FAILED")

    if e6_result:
        f1 = e6_result['overall']['f1']
        print(f"  E6 (multiseg):F1={f1}  ({f1-0.8908:+.4f})")
    else:
        print("  E6 (multiseg):FAILED")

    if e7_result:
        f1 = e7_result['overall']['f1']
        print(f"  E7 (oracle):  F1={f1}  ({f1-0.8908:+.4f})")
        print(f"    -> Freq estimation headroom: {f1-0.8908:+.4f} F1 pts")
    else:
        print("  E7 (oracle):  FAILED")

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  Dashboard: {DASHBOARD_PATH}")
    print('=' * 70)


if __name__ == '__main__':
    main()
