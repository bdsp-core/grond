"""
Phase 2: Retrain ChannelPD-Net with balanced LPD/GPD data and a
heteroscedastic uncertainty head for frequency estimation.

Key changes from V1/V3:
  - freq_head split into freq_head_mu + freq_head_logvar (uncertainty)
  - Heteroscedastic NLL loss instead of MSE for frequency
  - GPD frequency examples upweighted 3x (V4) or 1x (V5, balanced data)
  - GPD PD+ channels oversampled 2x during training (V4) or 1x (V5)
  - Balanced frequency labels from annotations.csv + segment_labels.csv
  - Saves weights to {prefix}_cnn_attn_fold{0-4}.pt (does NOT overwrite V1)

Usage:
  # Train V4 (original)
  python train_cnn_attention_phase2.py

  # Train V5 (with pre-built balanced dataset)
  python train_cnn_attention_phase2.py --dataset data/pd_channel_cache/channel_dataset_v5.npz --output-prefix v5

  # Evaluate only (compare V1 vs V4 vs V5)
  python train_cnn_attention_phase2.py --evaluate
"""

import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
V1_BACKUP_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache_backup_v1'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[Phase2] Using device: {DEVICE}")

SAVE_PREFIX = 'v4_cnn_attn'

CONFIG = {
    'n_folds': 5,
    'n_epochs': 30,
    'batch_size': 128,
    'lr': 1e-3,
    'alpha': 0.5,           # frequency loss weight
    'gpd_freq_weight': 3.0, # upweight GPD frequency examples (V4); overridden to 1.0 for V5
    'early_stop_patience': 5,
}

# Will be updated by CLI args
DATASET_PATH = None  # None = use V1 backup dataset + build_freq_labels


# -- V4 Model with heteroscedastic frequency head ----------------------------

class ChannelPDNetAttentionV4(nn.Module):
    """ChannelPDNetAttention with heteroscedastic frequency head.

    Same conv backbone + attention as V1, but freq_head is replaced with:
      - freq_head_mu:     Linear(64, 1) -> mean log-frequency
      - freq_head_logvar: Linear(64, 1) -> log-variance (learned uncertainty)
    """

    def __init__(self):
        super().__init__()

        # Backbone (identical to ChannelPDNetAttention)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Attention branch (identical to V1)
        self.attn_conv = nn.Conv1d(64, 1, kernel_size=1)

        # PD detection head (unchanged)
        self.pd_head = nn.Linear(64, 1)

        # Heteroscedastic frequency head
        self.freq_head_mu = nn.Linear(64, 1)       # mean log-frequency
        self.freq_head_logvar = nn.Linear(64, 1)    # log-variance (uncertainty)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            pd_prob: (batch, 1) PD probability (after sigmoid)
            freq_mu: (batch, 1) mean log-frequency prediction
            freq_logvar: (batch, 1) log-variance of frequency prediction
            attention_weights: (batch, 1, T) attention weights over time
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (batch, 64, T)

        # Attention pooling
        attn_logits = self.attn_conv(x)  # (batch, 1, T)
        attention_weights = torch.softmax(attn_logits, dim=-1)  # (batch, 1, T)
        pooled = (x * attention_weights).sum(dim=-1)  # (batch, 64)

        pd_prob = torch.sigmoid(self.pd_head(pooled))        # (batch, 1)
        freq_mu = self.freq_head_mu(pooled)                   # (batch, 1)
        freq_logvar = self.freq_head_logvar(pooled)            # (batch, 1)

        return pd_prob, freq_mu, freq_logvar, attention_weights

    @classmethod
    def from_v1_weights(cls, state_dict):
        """Initialize V4 model from V1 (ChannelPDNetAttention) weights.

        Maps V1 freq_head -> freq_head_mu, initializes freq_head_logvar to
        zeros (initial log-variance=0 -> variance=1.0).
        """
        model = cls()
        new_state = model.state_dict()

        for key, value in state_dict.items():
            if key == 'freq_head.weight':
                new_state['freq_head_mu.weight'] = value.clone()
            elif key == 'freq_head.bias':
                new_state['freq_head_mu.bias'] = value.clone()
            elif key in new_state:
                new_state[key] = value.clone()

        # Initialize freq_head_logvar to zeros (log-variance=0 -> variance=1.0)
        nn.init.zeros_(new_state['freq_head_logvar.weight'])
        nn.init.zeros_(new_state['freq_head_logvar.bias'])

        model.load_state_dict(new_state)
        return model


# -- Dataset -----------------------------------------------------------------

class ChannelPDDataset(Dataset):
    """PyTorch dataset for channel-level PD detection + frequency estimation."""

    def __init__(self, channels, labels, freq_targets, subtypes,
                 freq_weights=None, augment=False):
        self.channels = channels.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.freq_targets = freq_targets.astype(np.float32)
        self.subtypes = subtypes
        self.freq_weights = freq_weights if freq_weights is not None else np.ones(len(labels), dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.channels[idx].copy()

        # Per-channel z-score normalization
        mu = np.mean(x)
        std = np.std(x)
        if std > 1e-8:
            x = (x - mu) / std
        else:
            x = x - mu

        if self.augment:
            x = self._augment(x)

        x_tensor = torch.from_numpy(x[np.newaxis, :])
        label = torch.tensor(self.labels[idx])
        freq = torch.tensor(self.freq_targets[idx])
        mask = torch.tensor(
            1.0 if self.labels[idx] > 0.5 and np.isfinite(self.freq_targets[idx]) else 0.0
        )
        freq_w = torch.tensor(self.freq_weights[idx])

        return x_tensor, label, freq, mask, freq_w

    def _augment(self, x):
        # Random amplitude scaling (0.8-1.2x)
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale

        # Random Gaussian noise (SNR 20-40 dB)
        snr_db = np.random.uniform(20, 40)
        signal_power = np.mean(x ** 2)
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(x)) * np.sqrt(noise_power)
            x = x + noise.astype(np.float32)

        # Random circular time shift (up to 50 samples)
        shift = np.random.randint(-50, 51)
        if shift != 0:
            x = np.roll(x, shift)

        return x


# -- Training loop -----------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, alpha=0.5):
    """Train for one epoch with heteroscedastic NLL loss. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    bce = nn.BCELoss(reduction='none')

    for x, label, freq, mask, freq_w in loader:
        x = x.to(DEVICE)
        label = label.to(DEVICE)
        freq = freq.to(DEVICE)
        mask = mask.to(DEVICE)
        freq_w = freq_w.to(DEVICE)

        optimizer.zero_grad()
        pd_prob, freq_mu, freq_logvar, _attn = model(x)

        pd_prob = pd_prob.squeeze(1)
        freq_mu = freq_mu.squeeze(1)
        freq_logvar = freq_logvar.squeeze(1)

        # PD detection loss (unchanged)
        loss_bce = bce(pd_prob, label).mean()

        # Heteroscedastic NLL for frequency
        freq_clean = torch.where(torch.isnan(freq), torch.zeros_like(freq), freq)
        # NLL = 0.5 * (logvar + (mu - target)^2 / exp(logvar))
        nll = 0.5 * (freq_logvar + (freq_mu - freq_clean) ** 2 / freq_logvar.exp())
        # Regularization to prevent infinite uncertainty
        reg = 0.01 * freq_logvar.exp()
        # Apply mask (only PD+ channels with freq labels) and per-sample weight
        loss_freq_per_sample = (nll + reg) * mask * freq_w
        n_masked = (mask * freq_w).sum()
        if n_masked > 0:
            loss_freq = loss_freq_per_sample.sum() / n_masked
        else:
            loss_freq = torch.tensor(0.0, device=DEVICE)

        loss = loss_bce + alpha * loss_freq
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, alpha=0.5):
    """Evaluate model. Returns predictions, labels, and average loss."""
    model.eval()
    all_pd_probs = []
    all_labels = []
    all_freq_mu = []
    all_freq_logvar = []
    all_freq_targets = []
    all_masks = []

    bce = nn.BCELoss(reduction='none')
    total_loss = 0.0
    n_batches = 0

    for x, label, freq, mask, freq_w in loader:
        x = x.to(DEVICE)
        label_dev = label.to(DEVICE)
        freq_dev = freq.to(DEVICE)
        mask_dev = mask.to(DEVICE)
        freq_w_dev = freq_w.to(DEVICE)

        pd_prob, freq_mu, freq_logvar, _attn = model(x)
        pd_prob_sq = pd_prob.squeeze(1)
        freq_mu_sq = freq_mu.squeeze(1)
        freq_logvar_sq = freq_logvar.squeeze(1)

        loss_bce = bce(pd_prob_sq, label_dev).mean()
        freq_clean = torch.where(torch.isnan(freq_dev), torch.zeros_like(freq_dev), freq_dev)
        nll = 0.5 * (freq_logvar_sq + (freq_mu_sq - freq_clean) ** 2 / freq_logvar_sq.exp())
        reg = 0.01 * freq_logvar_sq.exp()
        loss_freq_per_sample = (nll + reg) * mask_dev * freq_w_dev
        n_masked = (mask_dev * freq_w_dev).sum()
        if n_masked > 0:
            loss_freq = loss_freq_per_sample.sum() / n_masked
        else:
            loss_freq = torch.tensor(0.0, device=DEVICE)
        loss = loss_bce + alpha * loss_freq
        total_loss += loss.item()
        n_batches += 1

        all_pd_probs.append(pd_prob_sq.cpu().numpy())
        all_labels.append(label.numpy())
        all_freq_mu.append(freq_mu_sq.cpu().numpy())
        all_freq_logvar.append(freq_logvar_sq.cpu().numpy())
        all_freq_targets.append(freq.numpy())
        all_masks.append(mask.numpy())

    result = {
        'pd_probs': np.concatenate(all_pd_probs),
        'labels': np.concatenate(all_labels),
        'freq_preds': np.concatenate(all_freq_mu),      # mu = point estimate
        'freq_logvar': np.concatenate(all_freq_logvar),
        'freq_targets': np.concatenate(all_freq_targets),
        'masks': np.concatenate(all_masks),
        'avg_loss': total_loss / max(n_batches, 1),
    }
    return result


def compute_freq_spearman(val_results):
    """Compute Spearman correlation of frequency predictions on PD+ channels."""
    masks = val_results['masks']
    freq_preds = val_results['freq_preds']
    freq_targets = val_results['freq_targets']
    valid = (masks > 0.5) & np.isfinite(freq_targets) & np.isfinite(freq_preds)
    if np.sum(valid) < 5:
        return float('nan')
    rho, _ = spearmanr(freq_targets[valid], freq_preds[valid])
    return float(rho) if np.isfinite(rho) else float('nan')


def compute_auc(y_true, y_score):
    """Compute AUC via trapezoidal rule."""
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

    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(_trapz(tpr_list, fpr_list))


# -- Build balanced frequency labels -----------------------------------------

def build_freq_labels(patient_ids, labels, subtypes):
    """Build frequency labels from annotations.csv + segment_labels.csv.

    Returns:
        freq_targets: np.array of log-frequency per channel (NaN if no label)
        freq_weights: np.array of per-channel weight (GPD upweighted)
    """
    n_total = len(labels)

    # Build segment_id -> frequency lookup from annotations.csv
    ann_path = PROJECT_DIR / 'data' / 'labels' / 'annotations.csv'
    ann = pd.read_csv(str(ann_path))
    has_freq = ann[ann.frequency_hz.notna()].copy()
    has_freq['frequency_hz'] = pd.to_numeric(has_freq['frequency_hz'], errors='coerce')
    freq_per_seg = has_freq.groupby('segment_id').agg(
        mean_freq=('frequency_hz', 'mean')
    ).reset_index()
    freq_lookup = dict(zip(freq_per_seg.segment_id, freq_per_seg.mean_freq))

    # Add MW-only from segment_labels.csv
    sl_path = PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv'
    sl = pd.read_csv(str(sl_path))
    for _, row in sl[sl.expert_freq_rater == 'MW'].iterrows():
        sid = str(row['mat_file']).replace('.mat', '')
        if sid not in freq_lookup:
            val = pd.to_numeric(row['expert_freq_hz'], errors='coerce')
            if pd.notna(val) and val > 0:
                freq_lookup[sid] = float(val)

    print(f"  Frequency lookup: {len(freq_lookup)} segments with freq labels")

    # Build patient -> frequency map (for channel dataset which uses patient_ids)
    # The channel dataset maps patient_id -> channels, so we need patient-level freq
    # First, build from segment_labels: patient -> list of freqs
    pid_freqs = {}
    for _, row in sl.iterrows():
        pid = str(row['patient_id'])
        st = str(row['subtype']).lower()
        if st not in ('lpd', 'gpd'):
            continue
        # Try expert freq first
        freq_val = pd.to_numeric(row.get('expert_freq_hz'), errors='coerce')
        if pd.isna(freq_val) or freq_val <= 0:
            # Try algo freq
            freq_val = pd.to_numeric(row.get('algo_freq_hz'), errors='coerce')
        if pd.notna(freq_val) and freq_val > 0:
            if pid not in pid_freqs:
                pid_freqs[pid] = []
            pid_freqs[pid].append(float(freq_val))

    # Also add from annotations.csv
    for _, row in has_freq.iterrows():
        pid = str(row['patient_id'])
        fval = row['frequency_hz']
        if pd.notna(fval) and fval > 0:
            if pid not in pid_freqs:
                pid_freqs[pid] = []
            pid_freqs[pid].append(float(fval))

    # Average per patient
    pid_to_freq = {pid: np.mean(fs) for pid, fs in pid_freqs.items() if len(fs) > 0}
    print(f"  Patient freq lookup: {len(pid_to_freq)} patients with freq")

    # Also load from patients.csv for fallback (fast - just CSV, no EEG loading)
    try:
        patients_path = PROJECT_DIR / 'data' / 'labels' / 'patients.csv'
        df_patients = pd.read_csv(str(patients_path))
        for _, row in df_patients.iterrows():
            pid = str(row['patient_id'])
            freq = pd.to_numeric(row.get('gold_standard_freq'), errors='coerce')
            if pd.notna(freq) and freq > 0 and pid not in pid_to_freq:
                pid_to_freq[pid] = float(freq)
        print(f"  After patients.csv fallback: {len(pid_to_freq)} patients with freq")
    except Exception as e:
        print(f"  Warning: could not load patients.csv: {e}")

    # Assign frequency labels to channels
    freq_targets = np.full(n_total, np.nan, dtype=np.float32)
    n_lpd_freq = 0
    n_gpd_freq = 0
    for i in range(n_total):
        if labels[i] == 1:  # PD+ channel only
            pid = str(patient_ids[i])
            if pid in pid_to_freq:
                freq_targets[i] = np.log(pid_to_freq[pid])
                if str(subtypes[i]).lower() == 'lpd':
                    n_lpd_freq += 1
                elif str(subtypes[i]).lower() == 'gpd':
                    n_gpd_freq += 1

    print(f"  PD+ channels with freq: LPD={n_lpd_freq}, GPD={n_gpd_freq}")

    # Build per-channel frequency weights
    # Upweight GPD to balance with LPD
    freq_weights = np.ones(n_total, dtype=np.float32)
    gpd_weight = CONFIG['gpd_freq_weight']
    for i in range(n_total):
        if labels[i] == 1 and np.isfinite(freq_targets[i]):
            if str(subtypes[i]).lower() == 'gpd':
                freq_weights[i] = gpd_weight

    return freq_targets, freq_weights, n_lpd_freq, n_gpd_freq


# -- Sampler with GPD oversampling -------------------------------------------

def build_oversampler(labels, subtypes, freq_targets):
    """Build a WeightedRandomSampler that oversamples GPD PD+ channels 2x."""
    n = len(labels)
    weights = np.ones(n, dtype=np.float64)

    for i in range(n):
        if labels[i] == 1 and str(subtypes[i]).lower() == 'gpd':
            weights[i] = 2.0  # oversample GPD PD+ by 2x

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=n,
        replacement=True,
    )
    return sampler


# -- Patient-level frequency estimation (V4) ---------------------------------

@torch.no_grad()
def patient_level_freq_estimation_v4(models_by_fold, patient_folds, dataset):
    """Evaluate V4 CNN as patient-level frequency estimator.

    Like V1 but returns uncertainty too.
    """
    df = dataset['df']
    segments = dataset['segments']

    patient_preds = {}
    patient_golds = {}
    patient_subtypes = {}
    patient_uncertainty = {}

    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        gold = row['gold_standard_freq']
        subtype = row['subtype']

        if not np.isfinite(gold) or gold <= 0:
            continue

        pat_segs = segments.get(pid, [])
        if len(pat_segs) == 0:
            continue

        fold_idx = patient_folds.get(pid, None)
        if fold_idx is not None:
            fold_indices = [fold_idx]
        else:
            fold_indices = list(models_by_fold.keys())

        all_pd_probs = []
        all_freq_mu = []
        all_freq_logvar = []

        for fi in fold_indices:
            model = ChannelPDNetAttentionV4().to(DEVICE)
            model.load_state_dict(models_by_fold[fi])
            model.eval()

            for seg in pat_segs:
                n_ch = seg.shape[0]
                for ch_idx in range(n_ch):
                    ch_data = seg[ch_idx].astype(np.float32).copy()
                    if not np.all(np.isfinite(ch_data)):
                        continue

                    mu = np.mean(ch_data)
                    std = np.std(ch_data)
                    if std > 1e-8:
                        ch_data = (ch_data - mu) / std
                    else:
                        ch_data = ch_data - mu

                    x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(DEVICE)
                    pd_prob, freq_mu, freq_logvar, _attn = model(x)

                    all_pd_probs.append(pd_prob.item())
                    all_freq_mu.append(freq_mu.item())
                    all_freq_logvar.append(freq_logvar.item())

        if len(all_pd_probs) == 0:
            continue

        pd_probs = np.array(all_pd_probs)
        freq_mu_arr = np.array(all_freq_mu)
        freq_logvar_arr = np.array(all_freq_logvar)

        # PD-weighted frequency aggregation
        weights = pd_probs.copy()
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weighted_freq_log = np.sum(weights * freq_mu_arr) / weight_sum
            # Weighted average uncertainty
            weighted_logvar = np.sum(weights * freq_logvar_arr) / weight_sum
        else:
            weighted_freq_log = np.mean(freq_mu_arr)
            weighted_logvar = np.mean(freq_logvar_arr)

        pred_freq = np.exp(weighted_freq_log)
        pred_freq = np.clip(pred_freq, 0.1, 10.0)

        patient_preds[pid] = pred_freq
        patient_golds[pid] = gold
        patient_subtypes[pid] = subtype
        patient_uncertainty[pid] = float(np.exp(weighted_logvar))  # variance in log-space

    return patient_preds, patient_golds, patient_subtypes, patient_uncertainty


# -- Patient-level freq with V1 model (for comparison) ----------------------

@torch.no_grad()
def patient_level_freq_estimation_v1(models_by_fold, patient_folds, dataset):
    """Evaluate V1 ChannelPDNetAttention as patient-level frequency estimator."""
    df = dataset['df']
    segments = dataset['segments']

    patient_preds = {}
    patient_golds = {}
    patient_subtypes = {}

    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        gold = row['gold_standard_freq']
        subtype = row['subtype']

        if not np.isfinite(gold) or gold <= 0:
            continue

        pat_segs = segments.get(pid, [])
        if len(pat_segs) == 0:
            continue

        fold_idx = patient_folds.get(pid, None)
        if fold_idx is not None:
            fold_indices = [fold_idx]
        else:
            fold_indices = list(models_by_fold.keys())

        all_pd_probs = []
        all_freq_preds = []

        for fi in fold_indices:
            model = ChannelPDNetAttention().to(DEVICE)
            model.load_state_dict(models_by_fold[fi])
            model.eval()

            for seg in pat_segs:
                n_ch = seg.shape[0]
                for ch_idx in range(n_ch):
                    ch_data = seg[ch_idx].astype(np.float32).copy()
                    if not np.all(np.isfinite(ch_data)):
                        continue

                    mu = np.mean(ch_data)
                    std = np.std(ch_data)
                    if std > 1e-8:
                        ch_data = (ch_data - mu) / std
                    else:
                        ch_data = ch_data - mu

                    x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(DEVICE)
                    pd_prob, freq_pred, _attn = model(x)

                    all_pd_probs.append(pd_prob.item())
                    all_freq_preds.append(freq_pred.item())

        if len(all_pd_probs) == 0:
            continue

        pd_probs = np.array(all_pd_probs)
        freq_preds = np.array(all_freq_preds)

        weights = pd_probs.copy()
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weighted_freq_log = np.sum(weights * freq_preds) / weight_sum
        else:
            weighted_freq_log = np.mean(freq_preds)

        pred_freq = np.exp(weighted_freq_log)
        pred_freq = np.clip(pred_freq, 0.1, 10.0)

        patient_preds[pid] = pred_freq
        patient_golds[pid] = gold
        patient_subtypes[pid] = subtype

    return patient_preds, patient_golds, patient_subtypes


# -- Metrics computation ----------------------------------------------------

def compute_group_metrics(patient_preds, patient_golds, patient_subtypes):
    """Compute frequency metrics by subtype group."""
    results_by_group = {}
    for group in ['combined', 'lpd', 'gpd']:
        gold_vals = []
        pred_vals = []
        for pid in patient_preds:
            if group != 'combined' and patient_subtypes.get(pid) != group:
                continue
            g = patient_golds[pid]
            p = patient_preds[pid]
            if np.isfinite(g) and np.isfinite(p):
                gold_vals.append(g)
                pred_vals.append(p)
        gold_arr = np.array(gold_vals)
        pred_arr = np.array(pred_vals)
        if len(gold_arr) >= 3:
            rho, _ = spearmanr(gold_arr, pred_arr)
            mae = float(np.mean(np.abs(gold_arr - pred_arr)))
        else:
            rho = float('nan')
            mae = float('nan')
        results_by_group[group] = {
            'n': len(gold_arr),
            'spearman': round(float(rho), 4) if np.isfinite(rho) else float('nan'),
            'mae': round(mae, 4) if np.isfinite(mae) else float('nan'),
        }
    return results_by_group


# -- Main: Training ----------------------------------------------------------

def main_train():
    global SAVE_PREFIX
    t0 = time.time()
    print("=" * 70)
    print(f"Phase 2: CNN+Attention {SAVE_PREFIX.upper()} with Heteroscedastic Frequency Head")
    print("=" * 70)

    use_prebuilt = DATASET_PATH is not None

    if use_prebuilt:
        # -- Load pre-built V5 dataset with freq_targets included -----------
        data_path = Path(DATASET_PATH)
        if not data_path.is_absolute():
            data_path = PROJECT_DIR / data_path
        print(f"\nLoading pre-built dataset from {data_path}...")
        data = np.load(str(data_path), allow_pickle=True)
        channels = data['channels']
        labels = data['labels'].astype(np.float32)
        patient_ids = data['patient_ids']
        subtypes = data['subtypes']
        freq_targets = data['freq_targets'].astype(np.float32)

        # For V5: equal weight (data is naturally balanced)
        freq_weights = np.ones(len(labels), dtype=np.float32)
        gpd_mask = (subtypes == 'gpd') & (labels > 0.5) & np.isfinite(freq_targets)
        freq_weights[gpd_mask] = CONFIG['gpd_freq_weight']

        n_lpd_freq = int(np.sum((subtypes == 'lpd') & (labels > 0.5) & np.isfinite(freq_targets)))
        n_gpd_freq = int(np.sum(gpd_mask))
        print(f"  PD+ channels with freq: LPD={n_lpd_freq}, GPD={n_gpd_freq}")
    else:
        # -- Load channel dataset from V1 backup ---------------------------------
        data_path = V1_BACKUP_DIR / 'channel_dataset.npz'
        print(f"\nLoading channel dataset from {data_path}...")
        data = np.load(str(data_path), allow_pickle=True)
        channels = data['channels']
        labels = data['labels']
        patient_ids = data['patient_ids']
        subtypes = data['subtypes']

        # -- Build frequency labels -----------------------------------------------
        print("\nBuilding balanced frequency labels...")
        freq_targets, freq_weights, n_lpd_freq, n_gpd_freq = build_freq_labels(
            patient_ids, labels, subtypes
        )

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    print(f"  Channels: {n_total} (pos={n_pos}, neg={n_neg})")
    print(f"  Patients: {n_patients}")
    print(f"  Subtypes: LPD={np.sum(subtypes=='lpd')}, GPD={np.sum(subtypes=='gpd')}, "
          f"LRDA={np.sum(subtypes=='lrda')}, GRDA={np.sum(subtypes=='grda')}")

    # -- Create 5-fold patient-stratified splits ------------------------------
    print("\nCreating 5-fold patient-stratified splits...")
    n_folds = CONFIG['n_folds']
    rng = np.random.RandomState(42)

    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids):
        pid = str(pid)
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = str(subtypes[i])

    subtype_groups = {}
    for pid in unique_patients:
        st = pid_to_subtype.get(str(pid), 'unknown')
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(str(pid))

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % n_folds

    for fold in range(n_folds):
        fold_pids = [p for p, f in patient_folds.items() if f == fold]
        fold_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        n_fold = int(np.sum(fold_mask))
        n_fold_pos = int(np.sum(labels[fold_mask] == 1))
        n_fold_gpd = int(np.sum((subtypes[fold_mask] == 'gpd') & (labels[fold_mask] == 1)))
        print(f"  Fold {fold}: {len(fold_pids)} patients, {n_fold} channels "
              f"(pos={n_fold_pos}, GPD_pos={n_fold_gpd})")

    # -- Try to load V1 weights for initialization ----------------------------
    print("\nChecking for V1 weights to initialize from...")
    v1_weights = {}
    for fold in range(n_folds):
        v1_path = V1_BACKUP_DIR / f'cnn_attn_fold{fold}.pt'
        if v1_path.exists():
            v1_weights[fold] = torch.load(str(v1_path), map_location='cpu', weights_only=True)
    if len(v1_weights) == n_folds:
        print(f"  Found all {n_folds} V1 weight files - will initialize from V1")
        init_from_v1 = True
    else:
        print(f"  Found {len(v1_weights)}/{n_folds} V1 weight files - training from scratch")
        init_from_v1 = False

    # -- Training loop across folds -------------------------------------------
    print("\n" + "=" * 70)
    print("Training CNN+Attention V4 across 5 folds...")
    print("=" * 70)

    alpha = CONFIG['alpha']
    n_epochs = CONFIG['n_epochs']
    batch_size = CONFIG['batch_size']
    lr = CONFIG['lr']
    patience = CONFIG['early_stop_patience']

    oof_pd_probs = np.full(n_total, np.nan)
    oof_freq_preds = np.full(n_total, np.nan)
    oof_freq_logvar = np.full(n_total, np.nan)

    training_curves = {}
    models_by_fold = {}

    for fold in range(n_folds):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        fold_curves = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_freq_spearman': [],
        }

        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_channels = channels[train_mask]
        train_labels = labels[train_mask]
        train_freqs = freq_targets[train_mask]
        train_subtypes = subtypes[train_mask]
        train_freq_weights = freq_weights[train_mask]

        val_channels = channels[val_mask]
        val_labels = labels[val_mask]
        val_freqs = freq_targets[val_mask]
        val_subtypes = subtypes[val_mask]
        val_freq_weights = freq_weights[val_mask]

        n_train_gpd_pos = int(np.sum((train_subtypes == 'gpd') & (train_labels == 1)))
        n_train_lpd_pos = int(np.sum((train_subtypes == 'lpd') & (train_labels == 1)))
        print(f"  Train: {len(train_labels)} channels (LPD+={n_train_lpd_pos}, GPD+={n_train_gpd_pos})")
        print(f"  Val:   {len(val_labels)} channels")

        train_ds = ChannelPDDataset(
            train_channels, train_labels, train_freqs, train_subtypes,
            freq_weights=train_freq_weights, augment=True
        )
        val_ds = ChannelPDDataset(
            val_channels, val_labels, val_freqs, val_subtypes,
            freq_weights=val_freq_weights, augment=False
        )

        # Build oversampler for training
        # V5 (pre-built dataset): no oversampling needed (data naturally balanced)
        # V4 (V1 dataset): GPD PD+ oversampled 2x
        if use_prebuilt:
            train_sampler = None
            shuffle_train = True
        else:
            train_sampler = build_oversampler(train_labels, train_subtypes, train_freqs)
            shuffle_train = False

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler,
            shuffle=shuffle_train if train_sampler is None else False,
            num_workers=0, drop_last=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Initialize model
        if init_from_v1 and fold in v1_weights:
            model = ChannelPDNetAttentionV4.from_v1_weights(v1_weights[fold])
            model = model.to(DEVICE)
            print(f"  Initialized from V1 fold {fold} weights")
        else:
            model = ChannelPDNetAttentionV4().to(DEVICE)
            print(f"  Initialized from scratch")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, alpha=alpha)

            val_results = evaluate(model, val_loader, alpha=alpha)
            val_auc = compute_auc(val_results['labels'], val_results['pd_probs'])
            val_loss = val_results['avg_loss']
            val_freq_sp = compute_freq_spearman(val_results)

            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_loss, 6))
            fold_curves['val_auc'].append(round(val_auc, 6) if np.isfinite(val_auc) else None)
            fold_curves['val_freq_spearman'].append(
                round(val_freq_sp, 6) if np.isfinite(val_freq_sp) else None
            )

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_AUC={val_auc:.4f}, val_freq_sp={val_freq_sp:.4f}")

            # Early stopping on val_loss (better for heteroscedastic model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

        training_curves[f'fold_{fold}'] = fold_curves

        if best_state is not None:
            model.load_state_dict(best_state)

        # Collect OOF predictions
        val_results = evaluate(model, val_loader, alpha=alpha)
        oof_pd_probs[val_mask] = val_results['pd_probs']
        oof_freq_preds[val_mask] = val_results['freq_preds']
        oof_freq_logvar[val_mask] = val_results['freq_logvar']

        # Save fold model
        save_path = CACHE_DIR / f'{SAVE_PREFIX}_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_loss={best_val_loss:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # -- Save training curves --------------------------------------------------
    curves_path = CACHE_DIR / f'{SAVE_PREFIX}_training_curves.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Channel-level OOF metrics --------------------------------------------
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RESULTS (V4 CNN + Attention + Heteroscedastic)")
    print("=" * 70)

    valid = np.isfinite(oof_pd_probs)
    y_true = labels[valid].astype(int)
    y_prob = oof_pd_probs[valid]
    ch_auc = compute_auc(y_true, y_prob)
    print(f"\n  Channel AUC: {ch_auc:.4f}")

    # Channel-level frequency by subtype
    for st in ['lpd', 'gpd']:
        freq_mask = valid & (labels == 1) & np.isfinite(freq_targets) & (subtypes == st)
        n_st = int(np.sum(freq_mask))
        if n_st >= 5:
            rho, _ = spearmanr(freq_targets[freq_mask], oof_freq_preds[freq_mask])
            mae = float(np.mean(np.abs(np.exp(freq_targets[freq_mask]) - np.exp(oof_freq_preds[freq_mask]))))
            mean_logvar = float(np.mean(oof_freq_logvar[freq_mask]))
            print(f"  {st.upper()} channel freq: N={n_st}, Spearman={rho:.4f}, MAE={mae:.4f}, "
                  f"mean_uncertainty={np.exp(mean_logvar):.4f}")
        else:
            print(f"  {st.upper()} channel freq: N={n_st} (too few)")

    # -- Patient-level frequency estimation ------------------------------------
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL FREQUENCY ESTIMATION (V4)")
    print(f"{'=' * 70}")

    main_dataset = load_dataset(verbose=False)
    patient_preds, patient_golds, patient_subtypes_map, patient_unc = \
        patient_level_freq_estimation_v4(models_by_fold, patient_folds, main_dataset)

    v4_metrics = compute_group_metrics(patient_preds, patient_golds, patient_subtypes_map)

    for group in ['combined', 'lpd', 'gpd']:
        m = v4_metrics[group]
        print(f"  {group.upper():>8s} (N={m['n']:3d}): Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}")

    # Uncertainty calibration check
    lpd_uncs = [patient_unc[p] for p in patient_unc if patient_subtypes_map.get(p) == 'lpd']
    gpd_uncs = [patient_unc[p] for p in patient_unc if patient_subtypes_map.get(p) == 'gpd']
    if lpd_uncs and gpd_uncs:
        print(f"\n  Uncertainty calibration:")
        print(f"    LPD mean uncertainty (variance): {np.mean(lpd_uncs):.4f}")
        print(f"    GPD mean uncertainty (variance): {np.mean(gpd_uncs):.4f}")
        if np.mean(gpd_uncs) > np.mean(lpd_uncs):
            print(f"    GPD > LPD uncertainty (expected - GPD is harder)")
        else:
            print(f"    LPD > GPD uncertainty (unexpected)")

    # -- Save results ----------------------------------------------------------
    version_tag = SAVE_PREFIX.replace('_cnn_attn', '')  # e.g. 'v4' or 'v5'
    results = {
        'experiment': f'phase2_{version_tag}_heteroscedastic',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'channel_auc': round(ch_auc, 4),
        'v4_combined_spearman': v4_metrics['combined']['spearman'],
        'v4_combined_mae': v4_metrics['combined']['mae'],
        'v4_combined_n': v4_metrics['combined']['n'],
        'v4_lpd_spearman': v4_metrics['lpd']['spearman'],
        'v4_lpd_mae': v4_metrics['lpd']['mae'],
        'v4_lpd_n': v4_metrics['lpd']['n'],
        'v4_gpd_spearman': v4_metrics['gpd']['spearman'],
        'v4_gpd_mae': v4_metrics['gpd']['mae'],
        'v4_gpd_n': v4_metrics['gpd']['n'],
        'lpd_mean_uncertainty': round(float(np.mean(lpd_uncs)), 4) if lpd_uncs else None,
        'gpd_mean_uncertainty': round(float(np.mean(gpd_uncs)), 4) if gpd_uncs else None,
    }

    # Replace NaN with None for JSON
    results_json = {}
    for k, v in results.items():
        if isinstance(v, float) and np.isnan(v):
            results_json[k] = None
        else:
            results_json[k] = v

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / f'phase2_{version_tag}_results.json'
    with open(str(results_path), 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.1f}s")
    print("=" * 70)

    return models_by_fold, patient_folds


# -- Main: Evaluation (V1 vs V4 comparison) ---------------------------------

def main_evaluate():
    t0 = time.time()
    print("=" * 70)
    print("Phase 2 Evaluation: V1 vs V4 vs V5 Comparison")
    print("=" * 70)

    # Load channel dataset for fold assignment
    data_path = V1_BACKUP_DIR / 'channel_dataset.npz'
    data = np.load(str(data_path), allow_pickle=True)
    patient_ids = data['patient_ids']
    subtypes = data['subtypes']
    unique_patients = np.unique(patient_ids)

    # Recreate fold assignments (same RNG seed as training)
    rng = np.random.RandomState(42)
    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids):
        pid = str(pid)
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = str(subtypes[i])

    subtype_groups = {}
    for pid in unique_patients:
        st = pid_to_subtype.get(str(pid), 'unknown')
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(str(pid))

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % 5

    # Load main dataset
    print("\nLoading main frequency dataset...")
    main_dataset = load_dataset(verbose=False)

    # -- Load V1 weights and evaluate -----------------------------------------
    print("\n--- V1 (Original) ---")
    v1_models = {}
    for fold in range(5):
        v1_path = V1_BACKUP_DIR / f'cnn_attn_fold{fold}.pt'
        if v1_path.exists():
            v1_models[fold] = torch.load(str(v1_path), map_location='cpu', weights_only=True)
        else:
            print(f"  WARNING: V1 fold {fold} not found at {v1_path}")

    if len(v1_models) == 5:
        v1_preds, v1_golds, v1_subtypes = patient_level_freq_estimation_v1(
            v1_models, patient_folds, main_dataset
        )
        v1_metrics = compute_group_metrics(v1_preds, v1_golds, v1_subtypes)
        for group in ['combined', 'lpd', 'gpd']:
            m = v1_metrics[group]
            print(f"  {group.upper():>8s} (N={m['n']:3d}): Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}")
    else:
        print("  Skipping V1 evaluation (missing weights)")
        v1_metrics = None

    # -- Load V4 weights and evaluate -----------------------------------------
    print("\n--- V4 (Heteroscedastic, GPD upweighted) ---")
    v4_models = {}
    for fold in range(5):
        v4_path = CACHE_DIR / f'v4_cnn_attn_fold{fold}.pt'
        if v4_path.exists():
            v4_models[fold] = torch.load(str(v4_path), map_location='cpu', weights_only=True)
        else:
            print(f"  WARNING: V4 fold {fold} not found at {v4_path}")

    if len(v4_models) == 5:
        v4_preds, v4_golds, v4_subtypes, v4_unc = patient_level_freq_estimation_v4(
            v4_models, patient_folds, main_dataset
        )
        v4_metrics = compute_group_metrics(v4_preds, v4_golds, v4_subtypes)
        for group in ['combined', 'lpd', 'gpd']:
            m = v4_metrics[group]
            print(f"  {group.upper():>8s} (N={m['n']:3d}): Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}")
    else:
        print("  Skipping V4 evaluation (missing weights)")
        v4_metrics = None

    # -- Load V5 weights and evaluate -----------------------------------------
    print("\n--- V5 (Full balanced dataset, equal weight) ---")
    v5_models = {}
    for fold in range(5):
        v5_path = CACHE_DIR / f'v5_cnn_attn_fold{fold}.pt'
        if v5_path.exists():
            v5_models[fold] = torch.load(str(v5_path), map_location='cpu', weights_only=True)
        else:
            print(f"  WARNING: V5 fold {fold} not found at {v5_path}")

    if len(v5_models) == 5:
        v5_preds, v5_golds, v5_subtypes, v5_unc = patient_level_freq_estimation_v4(
            v5_models, patient_folds, main_dataset
        )
        v5_metrics = compute_group_metrics(v5_preds, v5_golds, v5_subtypes)
        for group in ['combined', 'lpd', 'gpd']:
            m = v5_metrics[group]
            print(f"  {group.upper():>8s} (N={m['n']:3d}): Spearman={m['spearman']:.4f}, MAE={m['mae']:.4f}")

        # Uncertainty calibration
        lpd_uncs = [v5_unc[p] for p in v5_unc if v5_subtypes.get(p) == 'lpd']
        gpd_uncs = [v5_unc[p] for p in v5_unc if v5_subtypes.get(p) == 'gpd']
        if lpd_uncs and gpd_uncs:
            print(f"\n  Uncertainty calibration:")
            print(f"    LPD mean uncertainty: {np.mean(lpd_uncs):.4f}")
            print(f"    GPD mean uncertainty: {np.mean(gpd_uncs):.4f}")
    else:
        print("  Skipping V5 evaluation (missing weights)")
        v5_metrics = None

    # -- Comparison table ------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("V1 vs V4 vs V5 COMPARISON")
    print(f"{'=' * 70}")

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.4f}" if np.isfinite(v) else "N/A"

    def get_val(metrics, group, key):
        if metrics is None:
            return None
        m = metrics.get(group)
        if m is None:
            return None
        return m.get(key)

    print(f"  {'':>10s}  {'LPD rho':>8s}  {'LPD MAE':>8s}  {'GPD rho':>8s}  {'GPD MAE':>8s}")
    print(f"  {'-' * 50}")
    for name, metrics in [('V1', v1_metrics), ('V4', v4_metrics), ('V5', v5_metrics)]:
        lpd_rho = get_val(metrics, 'lpd', 'spearman')
        lpd_mae = get_val(metrics, 'lpd', 'mae')
        gpd_rho = get_val(metrics, 'gpd', 'spearman')
        gpd_mae = get_val(metrics, 'gpd', 'mae')
        print(f"  {name:>10s}  {fmt(lpd_rho):>8s}  {fmt(lpd_mae):>8s}  {fmt(gpd_rho):>8s}  {fmt(gpd_mae):>8s}")

    # -- Save comparison results -----------------------------------------------
    comparison = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'v1': v1_metrics,
        'v4': v4_metrics,
        'v5': v5_metrics,
    }
    # Handle NaN in nested dicts
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    comp_path = RESULTS_DIR / 'phase2_v5_comparison.json'
    with open(str(comp_path), 'w') as f:
        json.dump(sanitize(comparison), f, indent=2)
    print(f"\n  Comparison saved to {comp_path}")

    elapsed = time.time() - t0
    print(f"\nTotal evaluation time: {elapsed:.1f}s")
    print("=" * 70)


# -- Entry point --------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: V4/V5 CNN with heteroscedastic freq head')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate V1 vs V4 vs V5 (skip training)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to pre-built channel dataset (e.g. channel_dataset_v5.npz). '
                             'If provided, uses freq_targets from the file directly.')
    parser.add_argument('--output-prefix', type=str, default='v4',
                        help='Prefix for saved weights (default: v4). E.g. v5 -> v5_cnn_attn_fold0.pt')
    parser.add_argument('--gpd-freq-weight', type=float, default=None,
                        help='GPD frequency loss weight. Default: 3.0 for V4, 1.0 for V5.')
    args = parser.parse_args()

    # Apply CLI args to globals
    SAVE_PREFIX = f'{args.output_prefix}_cnn_attn'
    DATASET_PATH = args.dataset

    # Set GPD freq weight based on version if not explicitly provided
    if args.gpd_freq_weight is not None:
        CONFIG['gpd_freq_weight'] = args.gpd_freq_weight
    elif args.dataset is not None:
        # V5: balanced data, no upweighting needed
        CONFIG['gpd_freq_weight'] = 1.0

    if args.evaluate:
        main_evaluate()
    else:
        main_train()
