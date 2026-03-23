"""
HemiCET Optimization — Experiments E4, E5, E6, E7 and Combinations

Runs sequentially:
  E4: Self-supervised MAE pretraining → fine-tune HemiCET
  E5: Add midline channels (10ch instead of 8ch)
  E6: Multi-segment training (up to 5 segments per patient)
  E7: Retrain CNN+Attention frequency model on cleaned labels

Combinations:
  C2: Best single experiment (E3+E4 or E3+E5 etc.) + C1 params
  C3: E4 + E5 (pretrain + midline)

Dashboard updated after each experiment completes.

Usage:
    conda run -n foe_dl python code/hemi_detector/run_e4_e7_experiments.py

Results saved to:
    data/hemi_cache/optimization/{e4,e5,e6,e7,c2,c3}_*.json
"""

import sys, json, time, math, warnings
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
LR_FINETUNE = 1e-4       # for MAE fine-tune (E4)
LR_SCRATCH = 3e-4         # for scratch (E5, E6)
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
POS_WEIGHT = 20.0
SHARPNESS_PENALTY = 0.1
HPP_FLOOR_LAMBDA = 0.05
GAUSSIAN_SIGMA = 2
N_SAMPLES = 2000

# MAE pretraining hypers
MAE_EPOCHS = 50
MAE_BATCH = 64
MAE_LR = 1e-3
MAE_MASK_PCT = 0.20
PATCH_SIZE = 100  # samples per patch


# ============================================================================
# Shared utilities
# ============================================================================

def zscore_channels(x):
    """x: (C, T) → (C, T) z-scored per channel."""
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
# Evaluation (from optimization_swarm.py)
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
# HemiCET inference with C1 params
# ============================================================================

@torch.no_grad()
def run_hemicet_dp(seg, subtype, laterality, hemi_cet_models, detector,
                   dp_alpha=1.5, dp_beta=0.3, dp_lambda=0.05,
                   peak_height_frac=0.05, max_skip=3,
                   evidence_threshold_pct=50, min_evidence_ratio=0.4,
                   hemi_channels=None):
    """Run HemiCET+DP. hemi_channels overrides LEFT/RIGHT_INDICES for E5 (10ch)."""

    def _run(indices):
        all_pd, all_lf = [], []
        for ci in indices[:8]:  # freq model uses first 8 for compatibility
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
        cnn_freq = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
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

        # HemiCET evidence — use all indices (may be 8 or 10)
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

    if hemi_channels is not None:
        # For E5 (10ch) — use the custom channel lists
        left_ch, right_ch = hemi_channels
        if subtype == 'gpd' or laterality not in ('left', 'right'):
            tl, tr = _run(left_ch), _run(right_ch)
            return tl if len(tl) >= len(tr) else tr
        return _run(left_ch if laterality == 'left' else right_ch)
    else:
        if subtype == 'gpd' or laterality not in ('left', 'right'):
            tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
            return tl if len(tl) >= len(tr) else tr
        return _run(LEFT_INDICES if laterality == 'left' else RIGHT_INDICES)


def run_eval_with_models(hemi_cet_models, gt_cases, segments, detector,
                          params=None, hemi_channels=None):
    """Evaluate a set of HemiCET models with given params."""
    if params is None:
        params = C1_PARAMS
    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        try:
            times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                   hemi_cet_models, detector,
                                   hemi_channels=hemi_channels,
                                   **params)
            preds[pid] = times
        except Exception as e:
            preds[pid] = []
    return full_evaluate(preds, gt_cases)


# ============================================================================
# Data preparation (shared between E4, E5, E6)
# ============================================================================

def prepare_base_data(segments, verbose=True):
    """Build hemisphere training examples from GT cases (mirrors train_hemi_cet.py)."""
    import json as _json
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = _json.load(f)

    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    dataset = load_dataset(verbose=False)
    df = dataset['df']

    if verbose:
        print(f"  GT cases: {len(gt_cases)}")

    all_hemi = []
    all_targets = []
    all_hpp = []
    all_pids = []
    all_subtypes = []

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

        def add_hemi(hemi_indices):
            hemi_raw = np.zeros((len(hemi_indices), N_SAMPLES), dtype=np.float32)
            hpp_ch = np.zeros((len(hemi_indices), N_SAMPLES), dtype=np.float32)
            for i, ch_idx in enumerate(hemi_indices):
                if ch_idx >= seg.shape[0]:
                    continue
                ch = seg[ch_idx].astype(np.float32)
                ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
                hemi_raw[i] = ch
                try:
                    hpp_ch[i] = _compute_channel_evidence(ch, FS).astype(np.float32)
                except Exception:
                    pass

            for i in range(len(hemi_indices)):
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
            add_hemi(LEFT_INDICES)
            add_hemi(RIGHT_INDICES)
        elif lat == 'left':
            add_hemi(LEFT_INDICES)
        elif lat == 'right':
            add_hemi(RIGHT_INDICES)
        else:
            add_hemi(LEFT_INDICES)
            add_hemi(RIGHT_INDICES)

    return (np.array(all_hemi, dtype=object),  # ragged if different ch counts
            np.array(all_targets, dtype=np.float32),
            np.array(all_hpp, dtype=np.float32),
            np.array(all_pids),
            np.array(all_subtypes))


# ============================================================================
# Datasets
# ============================================================================

class HemiCETDataset(Dataset):
    """Standard 8-channel HemiCET dataset."""

    def __init__(self, hemi_data, targets, hpp_evidence, augment=False):
        self.hemi_data = [x.astype(np.float32) for x in hemi_data]
        self.targets = targets.astype(np.float32)
        self.hpp_evidence = hpp_evidence.astype(np.float32)
        self.augment = augment
        self.n_channels = hemi_data[0].shape[0] if len(hemi_data) > 0 else 8

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
# Training loop
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
    """Create patient-stratified fold assignments."""
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
                pretrain_encoder_state=None, model_kwargs=None, verbose=True):
    """Train 5-fold CV. Returns fold val losses."""
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
            list(hemi_data[train_mask]),
            targets[train_mask],
            hpp_evidence[train_mask],
            augment=True)
        val_ds = HemiCETDataset(
            list(hemi_data[val_mask]),
            targets[val_mask],
            hpp_evidence[val_mask],
            augment=False)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

        model = model_class(**model_kwargs).to(device)

        # Load pretrained encoder weights if provided (E4)
        if pretrain_encoder_state is not None:
            print(f"  Loading pretrained encoder weights...")
            own_state = model.state_dict()
            loaded = 0
            for name, param in pretrain_encoder_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
            model.load_state_dict(own_state)
            print(f"  Loaded {loaded} encoder layers from pretrained weights")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            val_loss = evaluate_val_loss(model, val_loader, device)

            if epoch % 20 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}: train={train_loss:.6f} val={val_loss:.6f}")

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
    """Load HemiCET fold models from a directory."""
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
# Dashboard
# ============================================================================

def load_existing_dashboard_experiments():
    """Load existing experiment results from JSON files."""
    experiments = []

    # Try to load baseline and prior experiments from the RESULTS_DIR
    name_map = {
        'baseline': ('baseline', 'Baseline HemiCET+DP'),
        'e2_dp_sweep': ('e2_dp_sweep', 'E2: DP param re-optimization'),
        'e3_posthoc': ('e3_posthoc', 'E3: Post-hoc filtering sweep'),
        'c1_combined': ('c1_combined', 'C1: E2+E3 combined'),
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

    return experiments


def build_dashboard(experiments, baseline):
    """Build/update optimization dashboard HTML."""
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
  <td><b>{f1 or '—'}</b>{f1_delta}</td>
  <td>{r.get('sens', '—')}</td>
  <td>{r.get('prec', '—')}</td>
  <td>{r.get('freq_rho', '—')}</td>
  <td>{r.get('freq_mae', '—')}</td>
  <td>{r.get('timing_med_ms', '—')}</td>
  <td>{lpd.get('f1', '—')}</td>
  <td>{gpd.get('f1', '—')}</td>
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
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
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
<h1>HemiCET+DP Optimization — E4/E5/E6/E7</h1>
<div class="updated">Updated: {now} | Leaderboard: baseline=0.8725 E2=0.8727 E3=0.8906 C1=0.8908</div>

<table>
<tr>
  <th>Experiment</th><th>Status</th><th>F1</th><th>Sens</th><th>Prec</th>
  <th>Freq ρ</th><th>Freq MAE</th><th>Timing Med</th><th>LPD F1</th><th>GPD F1</th>
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

    with open(str(DASHBOARD_PATH), 'w') as f:
        f.write(html)


# ============================================================================
# E4: Self-supervised MAE pretraining for HemiCET
# ============================================================================

class HemiCETMAE(nn.Module):
    """Masked autoencoder wrapping HemiCET encoder for pretraining.

    Uses the HemiCET encoder (enc1..enc4) and a lightweight convolutional
    decoder to reconstruct masked input patches.
    """

    def __init__(self, in_channels=8, mask_pct=MAE_MASK_PCT, patch_size=PATCH_SIZE):
        super().__init__()
        self.in_channels = in_channels
        self.mask_pct = mask_pct
        self.patch_size = patch_size

        # Encoder — identical to HemiCET
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

        # Lightweight decoder: bottleneck → full resolution
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.GELU(),
        )
        self.head = nn.Conv1d(16, in_channels, kernel_size=1)

    def forward(self, x):
        """x: (B, C, 2000) → recon: (B, C, 2000), mask: (B, 1, 2000) bool"""
        B, C, T = x.shape
        n_patches = T // self.patch_size
        n_mask = max(1, int(n_patches * self.mask_pct))

        # Create mask
        mask = torch.zeros(B, 1, T, device=x.device, dtype=torch.bool)
        for b in range(B):
            patch_idx = torch.randperm(n_patches, device=x.device)[:n_mask]
            for pi in patch_idx:
                start = pi * self.patch_size
                end = min(start + self.patch_size, T)
                mask[b, 0, start:end] = True

        # Zero out masked patches
        x_masked = x.clone()
        x_masked[mask.expand_as(x_masked)] = 0.0

        # Encode
        e1 = self.enc1(x_masked)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decode
        d = self.dec4(e4)
        d = self._trim(d, e3.shape[2])
        d = self.dec3(d)
        d = self._trim(d, e2.shape[2])
        d = self.dec2(d)
        d = self._trim(d, e1.shape[2])
        d = self.dec1(d)
        d = self._trim(d, T)
        recon = self.head(d)

        return recon, mask

    @staticmethod
    def _trim(x, target_len):
        if x.shape[2] != target_len:
            x = x[:, :, :target_len]
        return x

    def get_encoder_state(self):
        """Return state dict for enc1..enc4 only (for transfer to HemiCET)."""
        return {k: v for k, v in self.state_dict().items()
                if k.startswith('enc')}


class AllEEGDataset(Dataset):
    """All EEG .mat files for unsupervised pretraining."""

    def __init__(self, file_list, in_channels=8, augment=False):
        self.augment = augment
        self.in_channels = in_channels
        self.examples = []
        for fp in file_list:
            self.examples.append((fp, 'left'))
            self.examples.append((fp, 'right'))
        print(f"  AllEEGDataset: {len(file_list)} files → {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fp, hemi = self.examples[idx]
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(fp))
            data = mat['data'].astype(np.float32)
            n_ch = data.shape[0]
            if data.shape[1] != N_SAMPLES:
                if data.shape[1] > N_SAMPLES:
                    data = data[:, :N_SAMPLES]
                else:
                    pad = np.zeros((n_ch, N_SAMPLES - data.shape[1]), dtype=np.float32)
                    data = np.hstack([data, pad])
        except Exception:
            data = np.zeros((18, N_SAMPLES), dtype=np.float32)

        indices = LEFT_INDICES if hemi == 'left' else RIGHT_INDICES
        n_use = min(self.in_channels, len(indices))
        seg = np.zeros((self.in_channels, N_SAMPLES), dtype=np.float32)
        for i, ci in enumerate(indices[:n_use]):
            if ci < data.shape[0]:
                ch = data[ci].astype(np.float32)
                ch = np.nan_to_num(ch)
                seg[i] = ch

        # Z-score
        for ch in range(self.in_channels):
            mu, std = np.mean(seg[ch]), np.std(seg[ch])
            if std > 1e-8:
                seg[ch] = (seg[ch] - mu) / std

        if self.augment:
            scale = np.random.uniform(0.7, 1.3)
            seg = seg * scale
            for ch in range(self.in_channels):
                ch_std = np.std(seg[ch])
                seg[ch] += (np.random.randn(N_SAMPLES) * 0.05 * (ch_std + 1e-8)).astype(np.float32)
            if np.random.rand() < 0.15:
                seg[np.random.randint(0, self.in_channels)] = 0.0

        return {'eeg': torch.from_numpy(seg)}


def get_lr_cosine(epoch, n_epochs, lr, warmup_epochs=3):
    if epoch < warmup_epochs:
        return lr * (epoch + 1) / warmup_epochs
    t = (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)
    return lr * 0.5 * (1.0 + math.cos(math.pi * t))


def run_e4_pretrain(eeg_dir, device, in_channels=8):
    """Phase 1: MAE pretraining on all EEG files."""
    save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e4_pretrain'
    save_dir.mkdir(parents=True, exist_ok=True)

    all_mat = sorted(Path(eeg_dir).glob('*.mat'))
    valid_files = [f for f in all_mat if f.stat().st_size > 1000]
    print(f"  Found {len(valid_files)} valid EEG files for MAE pretraining")

    rng = np.random.RandomState(42)
    valid_files = list(valid_files)
    rng.shuffle(valid_files)
    n_val = max(1, int(len(valid_files) * 0.1))
    val_files = valid_files[:n_val]
    train_files = valid_files[n_val:]

    train_ds = AllEEGDataset(train_files, in_channels=in_channels, augment=True)
    val_ds = AllEEGDataset(val_files, in_channels=in_channels, augment=False)

    train_loader = DataLoader(train_ds, batch_size=MAE_BATCH, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=MAE_BATCH, shuffle=False, num_workers=0)

    model = HemiCETMAE(in_channels=in_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAE_LR, weight_decay=1e-4)

    best_val_loss = float('inf')
    best_encoder_state = None
    log = []

    print(f"  Starting MAE pretraining for {MAE_EPOCHS} epochs...")
    for epoch in range(MAE_EPOCHS):
        lr_now = get_lr_cosine(epoch, MAE_EPOCHS, MAE_LR)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            optimizer.zero_grad()
            recon, mask = model(eeg)
            mask_exp = mask.expand_as(recon)
            if mask_exp.any():
                loss = F.mse_loss(recon[mask_exp], eeg[mask_exp])
            else:
                loss = F.mse_loss(recon, eeg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Val
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                recon, mask = model(eeg)
                mask_exp = mask.expand_as(recon)
                if mask_exp.any():
                    loss = F.mse_loss(recon[mask_exp], eeg[mask_exp])
                else:
                    loss = F.mse_loss(recon, eeg)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_encoder_state = model.get_encoder_state()
            torch.save({'encoder_state': best_encoder_state, 'val_loss': val_loss,
                        'epoch': epoch+1},
                       str(save_dir / 'best_encoder.pt'))

        log.append({'epoch': epoch+1, 'lr': lr_now,
                    'train_loss': train_loss, 'val_loss': val_loss})

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{MAE_EPOCHS}: "
                  f"train={train_loss:.4f} val={val_loss:.4f} "
                  f"{'*' if val_loss == best_val_loss else ''}")

    with open(str(save_dir / 'pretrain_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print(f"  MAE pretraining done. Best val MSE: {best_val_loss:.4f}")
    return best_encoder_state


# ============================================================================
# E5: HemiCET with 10 channels (add Fz-Cz, Cz-Pz)
# ============================================================================

LEFT_INDICES_10 = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17]
RIGHT_INDICES_10 = [4, 5, 6, 7, 12, 13, 14, 15, 16, 17]


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


def prepare_data_nch(segments, channel_lists, verbose=True):
    """Prepare training data with custom channel lists (for E5 10-ch)."""
    import json as _json
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = _json.load(f)

    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    left_ch, right_ch = channel_lists
    n_ch = len(left_ch)

    if verbose:
        print(f"  GT cases: {len(gt_cases)}, channels per hemi: {n_ch}")

    all_hemi, all_targets, all_hpp, all_pids, all_subtypes = [], [], [], [], []

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
        if seg.shape[0] < 18:
            continue  # need at least 18 channels for midline

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

    return (np.array(all_hemi, dtype=object),
            np.array(all_targets, dtype=np.float32),
            np.array(all_hpp, dtype=np.float32),
            np.array(all_pids),
            np.array(all_subtypes))


# ============================================================================
# E6: Multi-segment training
# ============================================================================

def find_all_segments(pid, eeg_dir):
    """Find all .mat segment files for a patient."""
    eeg_dir = Path(eeg_dir)
    segs = sorted(eeg_dir.glob(f'{pid}_seg*.mat'))
    return segs


def prepare_multiseg_data(verbose=True):
    """Build training data using up to 5 segments per patient."""
    import json as _json
    import scipy.io as sio

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = _json.load(f)

    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    dataset = load_dataset(verbose=False)
    df = dataset['df']
    eeg_dir = PROJECT_DIR / 'data' / 'eeg'

    if verbose:
        print(f"  GT cases: {len(gt_cases)}")

    all_hemi, all_targets, all_hpp, all_pids, all_subtypes = [], [], [], [], []
    n_multiseg = 0
    n_single = 0

    for pid, gt_data in gt_cases.items():
        discharge_times = gt_data['global_times']
        if len(discharge_times) < 2:
            continue

        # Find all segments for this patient (up to 5)
        seg_files = find_all_segments(pid, eeg_dir)
        if not seg_files:
            continue

        seg_files = seg_files[:5]  # cap at 5
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
# E7: Retrain CNN+Attention frequency model
# ============================================================================

def run_e7_freq_retrain():
    """Retrain CNN+Attention frequency model on 644 patients with gold_standard_freq."""
    print("\n  Importing CNN+Attention training infrastructure...")

    import sys as _sys
    _sys.path.insert(0, str(CODE_DIR))
    from pd_channel_detector.channel_cnn import ChannelPDNetAttention

    # Import the full training machinery from the hi-freq script
    from pd_channel_detector.train_cnn_attention_hifreq import (
        ChannelPDDatasetWeighted,
        train_one_epoch as cnn_train_epoch,
        evaluate as cnn_evaluate,
        compute_freq_spearman, compute_auc,
        load_hifreq_channels,
    )

    import scipy.io as sio

    CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
    EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
    HARVEST_MANIFEST = PROJECT_DIR / 'data' / 'labels' / 'harvest_manifest.json'

    # Device for E7 (MPS)
    e7_device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    # Load original channel dataset
    data_path = CACHE_DIR / 'channel_dataset.npz'
    print(f"  Loading channel dataset from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    channels_orig = data['channels']
    labels_orig = data['labels']
    patient_ids_orig = data['patient_ids']
    subtypes_orig = data['subtypes']
    n_orig = len(labels_orig)
    print(f"  Original: {n_orig} channels, PD+={int(np.sum(labels_orig==1))}")

    # Load frequency labels from main dataset
    main_dataset = load_dataset(verbose=False)
    df_main = main_dataset['df']
    pid_to_freq = {}
    for _, row in df_main.iterrows():
        pid = str(row['patient_id'])
        freq = row['gold_standard_freq']
        if np.isfinite(freq) and freq > 0:
            pid_to_freq[pid] = freq

    freq_targets_orig = np.full(n_orig, np.nan, dtype=np.float32)
    freq_weights_orig = np.ones(n_orig, dtype=np.float32)
    for i in range(n_orig):
        if labels_orig[i] == 1:
            pid = str(patient_ids_orig[i])
            if pid in pid_to_freq:
                freq_targets_orig[i] = np.log(pid_to_freq[pid])

    n_with_freq = int(np.sum(np.isfinite(freq_targets_orig) & (labels_orig == 1)))
    print(f"  PD+ with gold freq: {n_with_freq}")

    # Load hi-freq harvested data
    hi_channels, hi_pids, hi_freq_targets, hi_freq_weights, hi_labels = None, None, None, None, None
    if HARVEST_MANIFEST.exists():
        print("  Loading hi-freq harvested segments...")
        hi_channels, hi_pids, hi_freq_targets, hi_freq_weights, hi_labels = load_hifreq_channels(
            HARVEST_MANIFEST, EEG_DIR)

    if hi_channels is not None:
        channels_all = np.concatenate([channels_orig, hi_channels], axis=0)
        labels_all = np.concatenate([labels_orig, hi_labels], axis=0)
        freq_targets_all = np.concatenate([freq_targets_orig, hi_freq_targets], axis=0)
        freq_weights_all = np.concatenate([freq_weights_orig, hi_freq_weights], axis=0)
        patient_ids_all = np.concatenate([
            np.array([str(p) for p in patient_ids_orig]),
            np.array([str(p) for p in hi_pids])
        ])
        subtypes_all = np.concatenate([
            np.array([str(s) for s in subtypes_orig]),
            np.array(['lpd'] * len(hi_labels))
        ])
        print(f"  Combined: {len(labels_all)} channels total")
    else:
        channels_all = channels_orig
        labels_all = labels_orig
        freq_targets_all = freq_targets_orig
        freq_weights_all = freq_weights_orig
        patient_ids_all = np.array([str(p) for p in patient_ids_orig])
        subtypes_all = np.array([str(s) for s in subtypes_orig])

    # Create patient folds
    unique_pids = np.unique(patient_ids_all)
    rng = np.random.RandomState(42)
    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids_all):
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = subtypes_all[i]

    subtype_groups = {}
    for pid in unique_pids:
        st = pid_to_subtype.get(pid, 'unknown')
        subtype_groups.setdefault(st, []).append(pid)

    patient_folds_map = {}
    for st, pids in subtype_groups.items():
        pids_sh = list(pids)
        rng.shuffle(pids_sh)
        for i, pid in enumerate(pids_sh):
            patient_folds_map[pid] = i % 5

    n_epochs = 30
    batch_size = 128
    lr = 1e-3
    patience = 5
    alpha = 0.5

    models_by_fold = {}

    save_dir_e7 = PROJECT_DIR / 'data' / 'pd_channel_cache'

    for fold in range(5):
        fold_t0 = time.time()
        print(f"\n  --- E7 Fold {fold+1}/5 ---")

        val_mask = np.array([patient_folds_map.get(p, -1) == fold for p in patient_ids_all])
        train_mask = ~val_mask
        val_mask_orig = np.array([patient_folds_map.get(str(p), -1) == fold for p in patient_ids_orig])

        train_ds = ChannelPDDatasetWeighted(
            channels_all[train_mask], labels_all[train_mask],
            freq_targets_all[train_mask], freq_weights_all[train_mask], augment=True)
        val_ds = ChannelPDDatasetWeighted(
            channels_orig[val_mask_orig], labels_orig[val_mask_orig],
            freq_targets_orig[val_mask_orig], freq_weights_orig[val_mask_orig], augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

        model = ChannelPDNetAttention().to(e7_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_auc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(n_epochs):
            train_loss = cnn_train_epoch(model, train_loader, optimizer, scheduler, alpha=alpha)
            val_res = cnn_evaluate(model, val_loader, alpha=alpha)
            val_auc = compute_auc(val_res['labels'], val_res['pd_probs'])
            val_freq_sp = compute_freq_spearman(val_res)

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch+1:2d}: loss={train_loss:.4f} "
                      f"val_AUC={val_auc:.4f} freq_sp={val_freq_sp:.4f}")

            if np.isfinite(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  Early stop at epoch {epoch+1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        save_path = save_dir_e7 / f'cnn_attn_v2_fold{fold}.pt'
        torch.save(best_state or model.state_dict(), str(save_path))
        models_by_fold[fold] = best_state or model.state_dict()
        print(f"  Fold {fold+1} done: best_AUC={best_val_auc:.4f} ({time.time()-fold_t0:.1f}s)")

    print(f"\n  E7 training complete. Models saved to {save_dir_e7}/cnn_attn_v2_fold*.pt")
    return models_by_fold


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  HemiCET Optimization — E4, E5, E6, E7 + Combinations")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # --- Load shared data ---
    print("\nLoading dataset and GT cases...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        dt = json.load(f)

    gt_cases = {}
    for pid, v in dt.items():
        if v.get('review_status') != 'ground_truth':
            continue
        if len(v.get('global_times', [])) < 2:
            continue
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            v['subtype'] = row.iloc[0]['subtype']
            v['laterality'] = row.iloc[0].get('laterality', '')
            if not isinstance(v['laterality'], str) or v['laterality'] not in ('left', 'right'):
                v['laterality'] = None
        gt_cases[pid] = v

    detector = DischargeDetector()
    print(f"GT cases: {len(gt_cases)}")

    # Load v2 models (C1 baseline)
    v2_models = load_hemicet_models_from_dir(
        PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_v2')

    # Load existing dashboard experiments
    experiments = load_existing_dashboard_experiments()

    # Determine baseline result
    baseline_path = RESULTS_DIR / 'baseline.json'
    if baseline_path.exists():
        with open(str(baseline_path)) as f:
            baseline = json.load(f)
    else:
        baseline = {}

    # Build initial dashboard
    build_dashboard(experiments, baseline)

    eeg_dir = PROJECT_DIR / 'data' / 'eeg'

    # =========================================================================
    # E4: Self-supervised MAE pretraining
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E4: Self-supervised MAE Pretraining")
    print(f"{'='*70}")

    e4_exp = {'id': 'e4_pretrain', 'name': 'E4: MAE Pretraining', 'status': 'running'}
    experiments.append(e4_exp)
    build_dashboard(experiments, baseline)

    try:
        e4_t0 = time.time()

        # Phase 1: MAE pretraining
        print("\n  Phase 1: MAE pretraining on all EEG files...")
        encoder_state = run_e4_pretrain(eeg_dir, DEVICE, in_channels=8)

        # Phase 2: Fine-tune on labeled data
        print("\n  Phase 2: Fine-tuning on labeled data...")
        e4_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e4'
        e4_save_dir.mkdir(parents=True, exist_ok=True)

        print("  Preparing training data...")
        hemi_data, targets, hpp_evidence, patient_ids, subtypes = prepare_base_data(
            segments, verbose=True)

        train_5fold(
            model_class=HemiCET,
            hemi_data=hemi_data,
            targets=targets,
            hpp_evidence=hpp_evidence,
            patient_ids=patient_ids,
            subtypes=subtypes,
            save_dir=e4_save_dir,
            device=DEVICE,
            n_epochs=N_EPOCHS,
            lr=LR_FINETUNE,
            pretrain_encoder_state=encoder_state,
            verbose=True,
        )

        # Evaluate with C1 params
        print("\n  Evaluating E4 models...")
        e4_models = load_hemicet_models_from_dir(e4_save_dir)
        e4_result = run_eval_with_models(e4_models, gt_cases, segments, detector, C1_PARAMS)

        e4_elapsed = time.time() - e4_t0
        print(f"\n  E4 result: F1={e4_result['overall']['f1']} "
              f"LPD={e4_result['lpd']['f1']} GPD={e4_result['gpd']['f1']} "
              f"({e4_elapsed:.0f}s)")

        e4_exp['result'] = e4_result
        e4_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e4_pretrain.json'), 'w') as f:
            json.dump(e4_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"  E4 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e4_exp['status'] = 'error'
        e4_result = None

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E5: Add midline channels (10ch)
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E5: Midline Channels (10ch)")
    print(f"{'='*70}")

    e5_exp = {'id': 'e5_midline', 'name': 'E5: Midline 10ch', 'status': 'running'}
    experiments.append(e5_exp)
    build_dashboard(experiments, baseline)

    try:
        e5_t0 = time.time()
        e5_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e5'
        e5_save_dir.mkdir(parents=True, exist_ok=True)

        print("  Preparing 10-ch training data...")
        hemi_data_10, targets_10, hpp_10, pids_10, subs_10 = prepare_data_nch(
            segments,
            channel_lists=(LEFT_INDICES_10, RIGHT_INDICES_10),
            verbose=True)

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

        # Evaluate with C1 params using 10-ch inference
        print("\n  Evaluating E5 models...")
        e5_models = load_hemicet_models_from_dir(
            e5_save_dir, model_class=HemiCET10, model_kwargs={'in_channels': 10})

        e5_result = run_eval_with_models(
            e5_models, gt_cases, segments, detector, C1_PARAMS,
            hemi_channels=(LEFT_INDICES_10, RIGHT_INDICES_10))

        e5_elapsed = time.time() - e5_t0
        print(f"\n  E5 result: F1={e5_result['overall']['f1']} "
              f"LPD={e5_result['lpd']['f1']} GPD={e5_result['gpd']['f1']} "
              f"({e5_elapsed:.0f}s)")

        e5_exp['result'] = e5_result
        e5_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e5_midline.json'), 'w') as f:
            json.dump(e5_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"  E5 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e5_exp['status'] = 'error'
        e5_result = None

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E6: Multi-segment training
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E6: Multi-Segment Training")
    print(f"{'='*70}")

    e6_exp = {'id': 'e6_multiseg', 'name': 'E6: Multi-segment', 'status': 'running'}
    experiments.append(e6_exp)
    build_dashboard(experiments, baseline)

    try:
        e6_t0 = time.time()
        e6_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e6'
        e6_save_dir.mkdir(parents=True, exist_ok=True)

        print("  Preparing multi-segment training data...")
        hemi_ms, targets_ms, hpp_ms, pids_ms, subs_ms = prepare_multiseg_data(verbose=True)

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

        print("\n  Evaluating E6 models...")
        e6_models = load_hemicet_models_from_dir(e6_save_dir)
        e6_result = run_eval_with_models(e6_models, gt_cases, segments, detector, C1_PARAMS)

        e6_elapsed = time.time() - e6_t0
        print(f"\n  E6 result: F1={e6_result['overall']['f1']} "
              f"LPD={e6_result['lpd']['f1']} GPD={e6_result['gpd']['f1']} "
              f"({e6_elapsed:.0f}s)")

        e6_exp['result'] = e6_result
        e6_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e6_multiseg.json'), 'w') as f:
            json.dump(e6_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"  E6 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e6_exp['status'] = 'error'
        e6_result = None

    build_dashboard(experiments, baseline)

    # =========================================================================
    # E7: Retrain CNN+Attention frequency model
    # =========================================================================
    print(f"\n{'='*70}")
    print("  E7: Retrain CNN+Attention Frequency Model")
    print(f"{'='*70}")

    e7_exp = {'id': 'e7_freq', 'name': 'E7: New Freq Model', 'status': 'running'}
    experiments.append(e7_exp)
    build_dashboard(experiments, baseline)

    try:
        e7_t0 = time.time()

        # Train new frequency model
        run_e7_freq_retrain()

        # Evaluate: use HemiCET v2 + C1 params but load new cnn_attn_v2 freq models
        # We need to temporarily replace the detector's CNN models
        print("\n  Evaluating E7: HemiCET v2 + new freq model + C1 params...")

        # Load the new CNN+Attn models into a modified detector
        from pd_channel_detector.channel_cnn import ChannelPDNetAttention as _CPA

        cache_dir = PROJECT_DIR / 'data' / 'pd_channel_cache'
        new_cnn_models = []
        for fold in range(5):
            p = cache_dir / f'cnn_attn_v2_fold{fold}.pt'
            if p.exists():
                m = _CPA().to(detector.device)
                m.load_state_dict(torch.load(str(p), map_location=detector.device, weights_only=True))
                m.eval()
                new_cnn_models.append(m)

        print(f"  Loaded {len(new_cnn_models)} new CNN freq models")

        # Swap detector's CNN models temporarily
        old_cnn_models = detector.cnn_models
        detector.cnn_models = new_cnn_models

        e7_result = run_eval_with_models(v2_models, gt_cases, segments, detector, C1_PARAMS)

        # Restore original
        detector.cnn_models = old_cnn_models

        e7_elapsed = time.time() - e7_t0
        print(f"\n  E7 result: F1={e7_result['overall']['f1']} "
              f"LPD={e7_result['lpd']['f1']} GPD={e7_result['gpd']['f1']} "
              f"({e7_elapsed:.0f}s)")

        e7_exp['result'] = e7_result
        e7_exp['status'] = 'done'
        with open(str(RESULTS_DIR / 'e7_freq.json'), 'w') as f:
            json.dump(e7_result, f, indent=2, default=str)

    except Exception as ex:
        print(f"  E7 FAILED: {ex}")
        import traceback; traceback.print_exc()
        e7_exp['status'] = 'error'
        e7_result = None

    build_dashboard(experiments, baseline)

    # =========================================================================
    # Combinations
    # =========================================================================
    print(f"\n{'='*70}")
    print("  Combinations: C2 and C3")
    print(f"{'='*70}")

    # C2: Best single experiment (highest F1 among E4, E5, E6, E7)
    exp_f1s = []
    exp_label_model = []
    if e4_result:
        exp_f1s.append(e4_result['overall']['f1'])
        exp_label_model.append(('E4', e4_models if 'e4_models' in dir() else None, None, 8))
    if e5_result:
        exp_f1s.append(e5_result['overall']['f1'])
        exp_label_model.append(('E5', e5_models if 'e5_models' in dir() else None,
                                 (LEFT_INDICES_10, RIGHT_INDICES_10), 10))
    if e6_result:
        exp_f1s.append(e6_result['overall']['f1'])
        exp_label_model.append(('E6', e6_models if 'e6_models' in dir() else None, None, 8))
    if e7_result:
        exp_f1s.append(e7_result['overall']['f1'])
        exp_label_model.append(('E7', v2_models, None, 8))

    c2_exp = {'id': 'c2_best', 'name': 'C2: Best Single Exp', 'status': 'running'}
    experiments.append(c2_exp)
    build_dashboard(experiments, baseline)

    if exp_f1s:
        best_idx = int(np.argmax(exp_f1s))
        best_label, best_models, best_hemi_ch, _ = exp_label_model[best_idx]
        print(f"\n  C2: Best single experiment is {best_label} (F1={exp_f1s[best_idx]:.4f})")

        if best_models is not None:
            # For E7, swap freq model
            if best_label == 'E7' and e7_result:
                from pd_channel_detector.channel_cnn import ChannelPDNetAttention as _CPA2
                cache_dir = PROJECT_DIR / 'data' / 'pd_channel_cache'
                new_cnn2 = []
                for fold in range(5):
                    p = cache_dir / f'cnn_attn_v2_fold{fold}.pt'
                    if p.exists():
                        m = _CPA2().to(detector.device)
                        m.load_state_dict(torch.load(str(p), map_location=detector.device,
                                                      weights_only=True))
                        m.eval()
                        new_cnn2.append(m)
                old_cnn = detector.cnn_models
                detector.cnn_models = new_cnn2
                c2_result = run_eval_with_models(best_models, gt_cases, segments, detector,
                                                  C1_PARAMS, best_hemi_ch)
                detector.cnn_models = old_cnn
            else:
                c2_result = run_eval_with_models(best_models, gt_cases, segments, detector,
                                                  C1_PARAMS, best_hemi_ch)

            c2_result['best_experiment'] = best_label
            c2_exp['result'] = c2_result
            c2_exp['status'] = 'done'
            print(f"  C2 result: F1={c2_result['overall']['f1']} "
                  f"LPD={c2_result['lpd']['f1']} GPD={c2_result['gpd']['f1']}")
            with open(str(RESULTS_DIR / 'c2_best.json'), 'w') as f:
                json.dump(c2_result, f, indent=2, default=str)
        else:
            c2_exp['status'] = 'error'
    else:
        c2_exp['status'] = 'error'
        print("  C2: No successful experiments to combine")

    build_dashboard(experiments, baseline)

    # C3: E4 + E5 (pretrain + midline) — only if both succeeded
    c3_exp = {'id': 'c3_e4e5', 'name': 'C3: E4+E5 (pretrain+midline)', 'status': 'pending'}
    experiments.append(c3_exp)

    if e4_result and e5_result:
        c3_exp['status'] = 'running'
        build_dashboard(experiments, baseline)

        print(f"\n  C3: E4 (pretrain) + E5 (midline)")
        print("  Re-training 10-ch model with pretrained encoder...")

        try:
            c3_save_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_c3'
            c3_save_dir.mkdir(parents=True, exist_ok=True)

            # Load the pretrained encoder from E4
            e4_pretrain_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_e4_pretrain'
            pretrain_ckpt = e4_pretrain_dir / 'best_encoder.pt'
            if pretrain_ckpt.exists():
                ckpt = torch.load(str(pretrain_ckpt), map_location='cpu', weights_only=True)
                encoder_state_c3 = ckpt['encoder_state']

                # The pretrained encoder has enc1 with in_channels=8.
                # For 10-ch we need to adapt enc1 weight. Use partial weight copy.
                # We'll keep enc2/enc3/enc4 fully; for enc1 we copy the 8-ch weights
                # and leave the 2 extra channels randomly initialized.
                adapted_state = {}
                for k, v in encoder_state_c3.items():
                    if 'enc1' in k and 'weight' in k and v.shape[1] == 8:
                        # Expand from (32, 8, K) to (32, 10, K)
                        extra = torch.zeros(*v.shape[:1], 2, *v.shape[2:])
                        nn.init.kaiming_normal_(extra, mode='fan_out', nonlinearity='relu')
                        adapted_state[k] = torch.cat([v, extra], dim=1)
                    else:
                        adapted_state[k] = v
                print(f"  Adapted encoder state for 10-ch input")
            else:
                adapted_state = None
                print("  No pretrained encoder found for C3, training from scratch")

            train_5fold(
                model_class=HemiCET10,
                hemi_data=hemi_data_10,
                targets=targets_10,
                hpp_evidence=hpp_10,
                patient_ids=pids_10,
                subtypes=subs_10,
                save_dir=c3_save_dir,
                device=DEVICE,
                n_epochs=N_EPOCHS,
                lr=LR_FINETUNE if adapted_state else LR_SCRATCH,
                pretrain_encoder_state=adapted_state,
                model_kwargs={'in_channels': 10},
                verbose=True,
            )

            c3_models = load_hemicet_models_from_dir(
                c3_save_dir, model_class=HemiCET10, model_kwargs={'in_channels': 10})
            c3_result = run_eval_with_models(
                c3_models, gt_cases, segments, detector, C1_PARAMS,
                hemi_channels=(LEFT_INDICES_10, RIGHT_INDICES_10))

            c3_exp['result'] = c3_result
            c3_exp['status'] = 'done'
            print(f"\n  C3 result: F1={c3_result['overall']['f1']} "
                  f"LPD={c3_result['lpd']['f1']} GPD={c3_result['gpd']['f1']}")
            with open(str(RESULTS_DIR / 'c3_e4e5.json'), 'w') as f:
                json.dump(c3_result, f, indent=2, default=str)

        except Exception as ex:
            print(f"  C3 FAILED: {ex}")
            import traceback; traceback.print_exc()
            c3_exp['status'] = 'error'
    else:
        c3_exp['status'] = 'error' if not (e4_result and e5_result) else 'pending'
        print(f"  C3 skipped: E4={'done' if e4_result else 'failed'}, "
              f"E5={'done' if e5_result else 'failed'}")

    build_dashboard(experiments, baseline)

    # =========================================================================
    # Final summary
    # =========================================================================
    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Experiment':<40s}  {'F1':>7}  {'LPD F1':>7}  {'GPD F1':>7}")
    print(f"  {'-'*67}")

    # Reference C1
    print(f"  {'C1 (reference)':<40s}  {'0.8908':>7}  {'——':>7}  {'——':>7}")

    for exp in experiments:
        if exp['id'] in ('baseline', 'e2_dp_sweep', 'e3_posthoc', 'c1_combined'):
            continue  # skip prior experiments in summary
        r = exp.get('result', {})
        ov = r.get('overall', {}) if r else {}
        lp = r.get('lpd', {}) if r else {}
        gp = r.get('gpd', {}) if r else {}
        f1 = ov.get('f1', '—')
        delta = ''
        if isinstance(f1, float):
            delta = f' ({f1-0.8908:+.4f})'
        print(f"  {exp['name']:<40s}  {str(f1)+delta:>14}  "
              f"{str(lp.get('f1','—')):>7}  {str(gp.get('f1','—')):>7}")

    print(f"\n  Total time: {elapsed_total/60:.1f} min")
    print(f"  Dashboard: {DASHBOARD_PATH}")
    print(f"  Results: {RESULTS_DIR}")
    print('=' * 70)


if __name__ == '__main__':
    main()
