"""
Train ChannelPDNetAttention with augmented data including 376 harvested
high-frequency LPD segments (>2.5 Hz) to improve frequency estimation
at the high end of the PD frequency spectrum.

Key differences from train_cnn_attention.py:
  1. Loads 376 hi-freq harvested segments (bins 2.5-3.0, 3.0-3.5, 3.5+)
  2. Uses est_freq as frequency target for harvested data (noisier)
  3. Applies lower freq-loss weight (freq_weight=0.5) for harvested data
  4. Marks harvested segments as PD-positive (pd_label=1)
  5. Uses MPS GPU (Apple Silicon)

The multi-task loss with per-sample frequency weights:
  loss = BCE(pd_pred, pd_label) + alpha * mean(freq_weight_i * MSE(freq_pred_i, log_freq_i) * mask_i)

Saves models to data/pd_channel_cache/ (overwrites old ones).
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import scipy.io as sio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
HARVEST_MANIFEST = PROJECT_DIR / 'data' / 'labels' / 'harvest_manifest.json'

# Use MPS if available, else CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Using CUDA GPU")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU")


# -- Dataset -----------------------------------------------------------------

class ChannelPDDatasetWeighted(Dataset):
    """PyTorch dataset for channel-level PD detection + frequency estimation.

    Supports per-sample frequency loss weights (for noisy harvested data).
    """

    def __init__(self, channels, labels, freq_targets, freq_weights=None, augment=False):
        self.channels = channels.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.freq_targets = freq_targets.astype(np.float32)
        if freq_weights is None:
            self.freq_weights = np.ones(len(labels), dtype=np.float32)
        else:
            self.freq_weights = freq_weights.astype(np.float32)
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
        freq_weight = torch.tensor(self.freq_weights[idx])
        mask = torch.tensor(
            1.0 if self.labels[idx] > 0.5 and np.isfinite(self.freq_targets[idx]) else 0.0
        )

        return x_tensor, label, freq, mask, freq_weight

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
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    for x, label, freq, mask, freq_weight in loader:
        x = x.to(DEVICE)
        label = label.to(DEVICE)
        freq = freq.to(DEVICE)
        mask = mask.to(DEVICE)
        freq_weight = freq_weight.to(DEVICE)

        optimizer.zero_grad()
        pd_prob, freq_pred, _attn = model(x)

        pd_prob = pd_prob.squeeze(1)
        freq_pred = freq_pred.squeeze(1)

        loss_bce = bce(pd_prob, label).mean()

        freq_clean = torch.where(torch.isnan(freq), torch.zeros_like(freq), freq)
        # Per-sample weighted frequency loss
        loss_freq = (mse(freq_pred, freq_clean) * mask * freq_weight).sum()
        n_masked = (mask * freq_weight).sum()
        if n_masked > 1e-6:
            loss_freq = loss_freq / n_masked
        else:
            loss_freq = torch.tensor(0.0, device=DEVICE)

        loss = loss_bce + alpha * loss_freq
        loss.backward()
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
    all_freq_preds = []
    all_freq_targets = []
    all_masks = []
    all_freq_weights = []

    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')
    total_loss = 0.0
    n_batches = 0

    for x, label, freq, mask, freq_weight in loader:
        x = x.to(DEVICE)
        label_dev = label.to(DEVICE)
        freq_dev = freq.to(DEVICE)
        mask_dev = mask.to(DEVICE)
        freq_weight_dev = freq_weight.to(DEVICE)

        pd_prob, freq_pred, _attn = model(x)
        pd_prob_sq = pd_prob.squeeze(1)
        freq_pred_sq = freq_pred.squeeze(1)

        loss_bce = bce(pd_prob_sq, label_dev).mean()
        freq_clean = torch.where(torch.isnan(freq_dev), torch.zeros_like(freq_dev), freq_dev)
        loss_freq = (mse(freq_pred_sq, freq_clean) * mask_dev * freq_weight_dev).sum()
        n_masked = (mask_dev * freq_weight_dev).sum()
        if n_masked > 1e-6:
            loss_freq = loss_freq / n_masked
        else:
            loss_freq = torch.tensor(0.0, device=DEVICE)
        loss = loss_bce + alpha * loss_freq
        total_loss += loss.item()
        n_batches += 1

        all_pd_probs.append(pd_prob_sq.cpu().numpy())
        all_labels.append(label.numpy())
        all_freq_preds.append(freq_pred_sq.cpu().numpy())
        all_freq_targets.append(freq.numpy())
        all_masks.append(mask.numpy())
        all_freq_weights.append(freq_weight.numpy())

    result = {
        'pd_probs': np.concatenate(all_pd_probs),
        'labels': np.concatenate(all_labels),
        'freq_preds': np.concatenate(all_freq_preds),
        'freq_targets': np.concatenate(all_freq_targets),
        'masks': np.concatenate(all_masks),
        'freq_weights': np.concatenate(all_freq_weights),
        'avg_loss': total_loss / max(n_batches, 1),
    }
    return result


def compute_freq_spearman(val_results, gold_only=False):
    """Compute Spearman correlation of frequency predictions on PD+ channels.

    If gold_only=True, only uses gold-standard channels (freq_weight=1.0).
    """
    masks = val_results['masks']
    freq_preds = val_results['freq_preds']
    freq_targets = val_results['freq_targets']
    freq_weights = val_results['freq_weights']

    if gold_only:
        valid = (masks > 0.5) & np.isfinite(freq_targets) & np.isfinite(freq_preds) & (freq_weights > 0.9)
    else:
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

    return float(np.trapz(tpr_list, fpr_list))


# -- Load harvested hi-freq segments ----------------------------------------

def load_hifreq_channels(harvest_manifest_path, eeg_dir, hi_freq_bins=('2.5', '3.0', '3.5')):
    """Load single channels from harvested hi-freq LPD segments.

    Returns:
        channels: (N_channels, 2000) float32 array
        patient_ids: (N_channels,) str array
        freq_targets: (N_channels,) float32 array (log Hz)
        freq_weights: (N_channels,) float32 array (0.5 = harvested/noisy)
        labels: (N_channels,) float32 array (all 1.0 = PD+)
    """
    with open(str(harvest_manifest_path)) as f:
        manifest = json.load(f)

    hi_entries = {
        pid: v for pid, v in manifest.items()
        if any(b in v.get('bin', '') for b in hi_freq_bins)
    }
    print(f"  Hi-freq manifest entries: {len(hi_entries)}")

    all_channels = []
    all_pids = []
    all_freq_targets = []
    all_freq_weights = []
    all_labels = []

    skipped_missing = 0
    skipped_invalid = 0
    loaded_segs = 0

    for pid, info in hi_entries.items():
        seg_path = Path(eeg_dir) / f'{pid}_seg000.mat'
        if not seg_path.exists():
            skipped_missing += 1
            continue

        try:
            mat = sio.loadmat(str(seg_path))
            seg = mat['data'].astype(np.float32)  # (18-20, 2000)
        except Exception:
            skipped_invalid += 1
            continue

        est_freq = float(info['est_freq'])
        if not np.isfinite(est_freq) or est_freq <= 0:
            skipped_invalid += 1
            continue

        log_freq = np.log(est_freq)
        n_channels = seg.shape[0]
        # Only use first 18 channels (bipolar montage) if more available
        n_ch_use = min(n_channels, 18)

        for ch_idx in range(n_ch_use):
            ch_data = seg[ch_idx].copy()
            if not np.all(np.isfinite(ch_data)):
                continue
            all_channels.append(ch_data)
            all_pids.append(pid)
            all_freq_targets.append(log_freq)
            all_freq_weights.append(0.5)  # Lower weight: est_freq is noisy
            all_labels.append(1.0)         # All are PD-positive

        loaded_segs += 1

    print(f"  Loaded {loaded_segs} hi-freq segments -> {len(all_channels)} channels")
    print(f"  Skipped: {skipped_missing} missing, {skipped_invalid} invalid")

    if len(all_channels) == 0:
        return None, None, None, None, None

    channels = np.stack(all_channels, axis=0)  # (N, 2000)
    patient_ids = np.array(all_pids)
    freq_targets = np.array(all_freq_targets, dtype=np.float32)
    freq_weights = np.array(all_freq_weights, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.float32)

    return channels, patient_ids, freq_targets, freq_weights, labels


# -- Patient-level frequency estimation -------------------------------------

@torch.no_grad()
def patient_level_freq_estimation(models_by_fold, patient_folds, dataset):
    """Evaluate CNN attention model as patient-level frequency estimator.

    For each patient in the frequency estimation dataset:
      1. Load all their segments (up to 5)
      2. Run all 18 channels through the out-of-fold CNN attention model
      3. Get per-channel: PD probability and frequency prediction
      4. Aggregate: weighted mean of freq predictions, weighted by PD probability
      5. Compare to gold standard
    """
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

        # PD-weighted frequency aggregation
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


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("CNN+Attention Retraining with Hi-Freq LPD Augmentation")
    print("=" * 70)

    # -- Load original channel dataset --------------------------------------
    data_path = CACHE_DIR / 'channel_dataset.npz'
    print(f"\nLoading original channel dataset from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    channels_orig = data['channels']      # (9310, 2000)
    labels_orig = data['labels']          # (9310,)
    patient_ids_orig = data['patient_ids']
    subtypes_orig = data['subtypes']

    n_orig = len(labels_orig)
    print(f"  Original: {n_orig} channels, {int(np.sum(labels_orig==1))} PD+, {int(np.sum(labels_orig==0))} PD-")

    # -- Load frequency labels from main dataset ----------------------------
    print("\nLoading frequency labels from main dataset...")
    main_dataset = load_dataset(verbose=False)
    df_main = main_dataset['df']

    pid_to_freq = {}
    for _, row in df_main.iterrows():
        pid = str(row['patient_id'])
        freq = row['gold_standard_freq']
        if np.isfinite(freq) and freq > 0:
            pid_to_freq[pid] = freq

    freq_targets_orig = np.full(n_orig, np.nan, dtype=np.float32)
    freq_weights_orig = np.ones(n_orig, dtype=np.float32)  # Gold standard weight = 1.0
    n_with_freq = 0
    for i in range(n_orig):
        if labels_orig[i] == 1:
            pid = str(patient_ids_orig[i])
            if pid in pid_to_freq:
                freq_targets_orig[i] = np.log(pid_to_freq[pid])
                n_with_freq += 1

    print(f"  PD+ channels with gold freq labels: {n_with_freq}/{int(np.sum(labels_orig==1))}")

    # Check hi-freq coverage in original data
    gold_freqs = np.exp(freq_targets_orig[np.isfinite(freq_targets_orig)])
    n_hifreq_orig = np.sum(gold_freqs > 2.5)
    print(f"  Original PD+ channels with freq >2.5 Hz: {n_hifreq_orig} / {len(gold_freqs)}")

    # -- Load hi-freq harvested segments ------------------------------------
    print("\nLoading hi-freq harvested LPD segments...")
    channels_hf, pids_hf, freq_targets_hf, freq_weights_hf, labels_hf = load_hifreq_channels(
        HARVEST_MANIFEST, EEG_DIR
    )

    n_hf = len(labels_hf)
    print(f"  Hi-freq channels: {n_hf}")
    print(f"  Hi-freq freq range: {np.exp(freq_targets_hf.min()):.2f} - {np.exp(freq_targets_hf.max()):.2f} Hz")

    # -- Combine original + hi-freq data ------------------------------------
    print("\nCombining datasets...")
    channels_all = np.concatenate([channels_orig, channels_hf], axis=0)
    labels_all = np.concatenate([labels_orig, labels_hf], axis=0)
    freq_targets_all = np.concatenate([freq_targets_orig, freq_targets_hf], axis=0)
    freq_weights_all = np.concatenate([freq_weights_orig, freq_weights_hf], axis=0)

    # Patient IDs: orig as strings, hf as strings
    patient_ids_all = np.concatenate([
        np.array([str(p) for p in patient_ids_orig]),
        np.array([str(p) for p in pids_hf])
    ])

    # Subtypes: orig as-is, hf as 'lpd' (LPD candidates)
    subtypes_all = np.concatenate([
        np.array([str(s) for s in subtypes_orig]),
        np.array(['lpd'] * n_hf)
    ])

    n_total = len(labels_all)
    print(f"  Total channels: {n_total} (orig={n_orig}, hi-freq={n_hf})")
    print(f"  PD+: {int(np.sum(labels_all==1))}, PD-: {int(np.sum(labels_all==0))}")

    # -- Patient-level data accounting --------------------------------------
    # Hi-freq patients are new patients not in the original channel dataset
    # They'll be assigned to folds based on subtype stratification
    unique_patients_all = np.unique(patient_ids_all)
    n_patients_all = len(unique_patients_all)
    print(f"  Unique patients: {n_patients_all} (orig unique: {len(np.unique(patient_ids_orig))})")

    # -- Create 5-fold patient-stratified splits ----------------------------
    print("\nCreating 5-fold patient-stratified splits...")
    n_folds = 5
    rng = np.random.RandomState(42)

    # Build pid -> subtype mapping
    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids_all):
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = subtypes_all[i]

    subtype_groups = {}
    for pid in unique_patients_all:
        st = pid_to_subtype.get(pid, 'unknown')
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(pid)

    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % n_folds

    for fold in range(n_folds):
        fold_mask = np.array([patient_folds.get(p, -1) == fold for p in patient_ids_all])
        n_fold = int(np.sum(fold_mask))
        n_fold_pos = int(np.sum(labels_all[fold_mask] == 1))
        fold_pids = [p for p, f in patient_folds.items() if f == fold]
        print(f"  Fold {fold}: {len(fold_pids)} patients, {n_fold} channels (pos={n_fold_pos})")

    # -- Training loop across folds -----------------------------------------
    print("\n" + "=" * 70)
    print("Training CNN+Attention with hi-freq augmentation across 5 folds...")
    print("=" * 70)

    # Patient folds for original patients only (for out-of-fold eval)
    patient_folds_orig = {
        str(p): patient_folds.get(str(p), -1)
        for p in np.unique(patient_ids_orig)
    }

    oof_pd_probs = np.full(n_orig, np.nan)
    oof_freq_preds = np.full(n_orig, np.nan)

    alpha = 0.5
    n_epochs = 30
    batch_size = 128
    lr = 1e-3
    patience = 5

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
            'val_freq_sp_gold': [],
        }

        val_mask_all = np.array([patient_folds.get(p, -1) == fold for p in patient_ids_all])
        train_mask_all = ~val_mask_all

        # Val: only use original data for clean evaluation metrics
        val_mask_orig = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids_orig])

        train_channels = channels_all[train_mask_all]
        train_labels = labels_all[train_mask_all]
        train_freqs = freq_targets_all[train_mask_all]
        train_fw = freq_weights_all[train_mask_all]

        val_channels = channels_orig[val_mask_orig]
        val_labels = labels_orig[val_mask_orig]
        val_freqs = freq_targets_orig[val_mask_orig]
        val_fw = freq_weights_orig[val_mask_orig]  # All 1.0 for gold data

        print(f"  Train: {len(train_labels)} channels (incl. {int(np.sum(train_fw < 0.9))} hi-freq)")
        print(f"  Val (gold only): {len(val_labels)} channels")

        train_ds = ChannelPDDatasetWeighted(
            train_channels, train_labels, train_freqs, train_fw, augment=True
        )
        val_ds = ChannelPDDatasetWeighted(
            val_channels, val_labels, val_freqs, val_fw, augment=False
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = ChannelPDNetAttention().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_auc = -1.0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, alpha=alpha)

            val_results = evaluate(model, val_loader, alpha=alpha)
            val_auc = compute_auc(val_results['labels'], val_results['pd_probs'])
            val_loss = val_results['avg_loss']
            val_freq_sp = compute_freq_spearman(val_results)
            val_freq_sp_gold = compute_freq_spearman(val_results, gold_only=True)

            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_loss, 6))
            fold_curves['val_auc'].append(round(val_auc, 6) if np.isfinite(val_auc) else None)
            fold_curves['val_freq_spearman'].append(
                round(val_freq_sp, 6) if np.isfinite(val_freq_sp) else None
            )
            fold_curves['val_freq_sp_gold'].append(
                round(val_freq_sp_gold, 6) if np.isfinite(val_freq_sp_gold) else None
            )

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, val_AUC={val_auc:.4f}, "
                      f"val_freq_sp={val_freq_sp:.4f} (gold={val_freq_sp_gold:.4f})")

            if np.isfinite(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
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

        # Record OOF predictions on original data only
        val_results = evaluate(model, val_loader, alpha=alpha)
        oof_pd_probs[val_mask_orig] = val_results['pd_probs']
        oof_freq_preds[val_mask_orig] = val_results['freq_preds']

        # Save fold model (overwrites old ones)
        save_path = CACHE_DIR / f'cnn_attn_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_AUC={best_val_auc:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # -- Save training curves -----------------------------------------------
    curves_path = CACHE_DIR / 'training_curves_attn_hifreq.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Channel-level OOF metrics (on original 594-patient data) ----------
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RESULTS (original data only, CNN+Attention+HiFreq)")
    print("=" * 70)

    valid = np.isfinite(oof_pd_probs)
    y_true = labels_orig[valid].astype(int)
    y_prob = oof_pd_probs[valid]
    pids_valid = patient_ids_orig[valid]

    ch_auc = compute_auc(y_true, y_prob)
    print(f"\nChannel-level AUC: {ch_auc:.4f}")

    # Patient-level PD detection
    pat_true = []
    pat_prob = []
    for pat in np.unique(pids_valid):
        pat_mask = pids_valid == pat
        mean_prob = float(np.mean(y_prob[pat_mask]))
        pat_labels = y_true[pat_mask]
        pat_label = int(np.max(pat_labels))
        pat_true.append(pat_label)
        pat_prob.append(mean_prob)

    pat_true = np.array(pat_true)
    pat_prob = np.array(pat_prob)
    pat_auc = compute_auc(pat_true, pat_prob)
    print(f"Patient-level AUC: {pat_auc:.4f}")

    # Channel-level freq Spearman (gold standard only)
    freq_mask = valid & (labels_orig == 1) & np.isfinite(freq_targets_orig)
    n_freq = int(np.sum(freq_mask))
    ch_freq_rho = float('nan')
    if n_freq > 5:
        freq_pred_valid = oof_freq_preds[freq_mask]
        freq_true_valid = freq_targets_orig[freq_mask]
        ch_freq_rho, pval = spearmanr(freq_true_valid, freq_pred_valid)
        print(f"Channel freq Spearman (gold, N={n_freq}): {ch_freq_rho:.4f} (p={pval:.2e})")

    # -- Patient-level frequency estimation (full 594-patient eval) ---------
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL FREQUENCY ESTIMATION (594-patient eval)")
    print(f"{'=' * 70}")

    patient_preds, patient_golds, patient_subtypes = patient_level_freq_estimation(
        models_by_fold, patient_folds_orig, main_dataset
    )

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

    comb = results_by_group['combined']
    lpd = results_by_group['lpd']
    gpd = results_by_group['gpd']

    def fmt(v):
        return f"{v:.4f}" if np.isfinite(v) else "N/A"

    print(f"\n  {'Method':<35s} {'Combined':>10s} {'LPD':>8s} {'GPD':>8s} {'MAE':>8s}")
    print(f"  {'-' * 69}")
    print(f"  {'Original CNN+Attn (baseline)':<35s} {'0.744':>10s} {'TBD':>8s} {'TBD':>8s} {'TBD':>8s}")
    print(f"  {'CNN+Attn+HiFreq':<35s} {fmt(comb['spearman']):>10s} {fmt(lpd['spearman']):>8s} {fmt(gpd['spearman']):>8s} {fmt(comb['mae']):>8s}")
    print(f"  {'  N patients':<35s} {comb['n']:>10d} {lpd['n']:>8d} {gpd['n']:>8d}")

    # -- Save results -------------------------------------------------------
    freq_results = {
        'experiment': 'cnn_attention_hifreq_augmented',
        'timestamp': time.time(),
        'n_orig_channels': n_orig,
        'n_hifreq_channels': n_hf,
        'n_total_channels': n_total,
        'combined_spearman': comb['spearman'],
        'combined_mae': comb['mae'],
        'combined_n': comb['n'],
        'lpd_spearman': lpd['spearman'],
        'lpd_mae': lpd['mae'],
        'lpd_n': lpd['n'],
        'gpd_spearman': gpd['spearman'],
        'gpd_mae': gpd['mae'],
        'gpd_n': gpd['n'],
        'channel_auc': round(ch_auc, 4),
        'patient_auc': round(pat_auc, 4),
        'channel_freq_spearman_gold': round(float(ch_freq_rho), 4) if np.isfinite(ch_freq_rho) else None,
        'baseline_combined_spearman': 0.744,
    }

    # Replace NaN with None for JSON
    freq_results_json = {
        k: (None if isinstance(v, float) and np.isnan(v) else v)
        for k, v in freq_results.items()
    }

    results_json_path = RESULTS_DIR / 'cnn_attn_hifreq_patient_freq.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(results_json_path), 'w') as f:
        json.dump(freq_results_json, f, indent=2)
    print(f"\nResults saved to {results_json_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
