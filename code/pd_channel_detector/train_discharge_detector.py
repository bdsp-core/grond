"""
Train a frame-level discharge detector using bootstrap targets from
pointiness peaks. 5-fold patient-stratified CV.

Bootstrap target generation:
  - PD-positive channels: compute pointiness trace, find peaks (PEAK_HEIGHT_FRAC=0.3),
    place Gaussian bumps (sigma=5 samples = 25ms) at each peak
  - PD-negative channels: target is all zeros
  - Loss: BCE per sample, with positive-sample weighting

After training, evaluate by:
  - Peak-picking detector output (prob > 0.3)
  - Computing inter-peak intervals (IPIs)
  - Deriving frequency = 1 / median(IPI)
  - Comparing with gold standard frequency (Spearman)

Run: conda run -n foe_dl python code/pd_channel_detector/train_discharge_detector.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.discharge_detector import DischargeDetector
from pd_pointiness_acf import compute_pointiness_trace
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('cpu')

FS = 200
PEAK_HEIGHT_FRAC = 0.3
TARGET_SIGMA = 5  # samples = 25ms at 200Hz
POS_WEIGHT = 10.0  # weight for positive samples in BCE (peaks are sparse)


# -- Bootstrap target generation ---------------------------------------------

def generate_discharge_targets(channels, labels):
    """Generate frame-level discharge targets from pointiness peaks.

    Args:
        channels: (N, 2000) array of channel waveforms
        labels: (N,) array of PD labels (0 or 1)

    Returns:
        targets: (N, 2000) array of target discharge probabilities
        peak_counts: (N,) array of number of peaks per channel
    """
    n = len(labels)
    targets = np.zeros((n, 2000), dtype=np.float32)
    peak_counts = np.zeros(n, dtype=int)

    for i in range(n):
        if labels[i] < 0.5:
            continue  # PD-negative: target stays all zeros

        signal = channels[i].astype(np.float64)

        # Skip bad channels
        if not np.all(np.isfinite(signal)):
            continue

        # Compute pointiness trace (same as optimization_harness_v2)
        trace = compute_pointiness_trace(signal)
        sigma_samples = max(1, int(0.02 * FS))
        trace = gaussian_filter1d(trace, sigma=sigma_samples)

        # Find peaks
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        pks, _ = find_peaks(trace, height=peak_height, distance=int(0.2 * FS))

        if len(pks) < 2:
            continue

        # Create Gaussian bumps at each peak
        t = np.arange(2000)
        for pk in pks:
            bump = np.exp(-0.5 * ((t - pk) / TARGET_SIGMA) ** 2)
            targets[i] += bump

        # Clip to [0, 1]
        targets[i] = np.clip(targets[i], 0, 1)
        peak_counts[i] = len(pks)

    return targets, peak_counts


# -- Dataset ------------------------------------------------------------------

class DischargeDataset(Dataset):
    """PyTorch dataset for frame-level discharge detection."""

    def __init__(self, channels, targets, augment=False):
        self.channels = channels.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, idx):
        x = self.channels[idx].copy()
        y = self.targets[idx].copy()

        # Per-channel z-score normalization
        mu = np.mean(x)
        std = np.std(x)
        if std > 1e-8:
            x = (x - mu) / std
        else:
            x = x - mu

        if self.augment:
            x, y = self._augment(x, y)

        x_tensor = torch.from_numpy(x[np.newaxis, :])  # (1, 2000)
        y_tensor = torch.from_numpy(y[np.newaxis, :])   # (1, 2000)

        return x_tensor, y_tensor

    def _augment(self, x, y):
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

        # Random circular time shift (up to 50 samples) - shift both signal and target
        shift = np.random.randint(-50, 51)
        if shift != 0:
            x = np.roll(x, shift)
            y = np.roll(y, shift)

        return x, y


# -- Training loop ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, pos_weight):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    weight = torch.tensor([pos_weight], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight, reduction='mean')

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        # Get raw logits (before sigmoid) for BCEWithLogitsLoss
        # We need to modify forward to get logits, or use the sigmoid output with BCE
        prob = model(x)  # (B, 1, 2000) - already has sigmoid

        # Use BCE on the sigmoid output with manual weighting
        # Positive weight: scale loss on positive samples
        pos_mask = (y > 0.1).float()
        neg_mask = 1.0 - pos_mask
        bce = -(y * torch.log(prob + 1e-7) + (1 - y) * torch.log(1 - prob + 1e-7))
        weighted_bce = bce * (pos_mask * pos_weight + neg_mask * 1.0)
        loss = weighted_bce.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, pos_weight):
    """Evaluate model. Returns predictions, targets, and average loss."""
    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        prob = model(x)

        pos_mask = (y > 0.1).float()
        neg_mask = 1.0 - pos_mask
        bce = -(y * torch.log(prob + 1e-7) + (1 - y) * torch.log(1 - prob + 1e-7))
        weighted_bce = bce * (pos_mask * pos_weight + neg_mask * 1.0)
        loss = weighted_bce.mean()

        total_loss += loss.item()
        n_batches += 1

        all_probs.append(prob.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    return {
        'probs': np.concatenate(all_probs, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'avg_loss': total_loss / max(n_batches, 1),
    }


def compute_ipi_frequency(prob_signal, fs=200, prob_threshold=0.3, min_distance=40):
    """Compute frequency from a discharge probability signal via peak-picking.

    Args:
        prob_signal: (2000,) discharge probability
        fs: sampling rate
        prob_threshold: minimum probability for peak detection
        min_distance: minimum distance between peaks in samples

    Returns:
        frequency: Hz (or NaN if too few peaks)
        n_peaks: number of detected peaks
        ipi_cv: coefficient of variation of IPIs
    """
    pks, _ = find_peaks(prob_signal, height=prob_threshold, distance=min_distance)

    if len(pks) < 3:
        return np.nan, len(pks), np.nan

    ipis = np.diff(pks) / fs  # in seconds
    median_ipi = np.median(ipis)

    if median_ipi <= 0:
        return np.nan, len(pks), np.nan

    frequency = 1.0 / median_ipi
    ipi_cv = np.std(ipis) / np.mean(ipis) if np.mean(ipis) > 0 else np.nan

    return frequency, len(pks), ipi_cv


# -- Main ---------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 4a: Frame-Level Discharge Detector Training")
    print("=" * 70)

    # -- Load channel dataset --------------------------------------------------
    data_path = CACHE_DIR / 'channel_dataset.npz'
    print(f"\nLoading channel dataset from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    channels = data['channels']
    labels = data['labels']
    patient_ids = data['patient_ids']
    subtypes = data['subtypes']

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    print(f"  Channels: {n_total} (pos={n_pos}, neg={n_neg})")
    print(f"  Patients: {n_patients}")

    # -- Generate bootstrap targets --------------------------------------------
    print("\nGenerating bootstrap targets from pointiness peaks...")
    targets, peak_counts = generate_discharge_targets(channels, labels)

    n_with_targets = int(np.sum(peak_counts > 0))
    avg_peaks = np.mean(peak_counts[peak_counts > 0]) if n_with_targets > 0 else 0
    print(f"  Channels with targets: {n_with_targets}/{n_pos} PD+ channels")
    print(f"  Average peaks per channel: {avg_peaks:.1f}")
    print(f"  Target coverage: {np.mean(targets > 0.1) * 100:.2f}% of samples are positive")

    # -- Load frequency labels from main dataset ------------------------------
    print("\nLoading frequency labels from main dataset...")
    main_dataset = load_dataset(verbose=False)
    df_main = main_dataset['df']

    pid_to_freq = {}
    for _, row in df_main.iterrows():
        pid = str(row['patient_id'])
        freq = row['gold_standard_freq']
        if np.isfinite(freq) and freq > 0:
            pid_to_freq[pid] = freq

    # -- Create 5-fold patient-stratified splits --------------------------------
    print("\nCreating 5-fold patient-stratified splits...")
    n_folds = 5
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
        print(f"  Fold {fold}: {len(fold_pids)} patients, {n_fold} channels (pos={n_fold_pos})")

    # -- Training loop across folds --------------------------------------------
    print("\n" + "=" * 70)
    print("Training DischargeDetector across 5 folds...")
    print("=" * 70)

    n_epochs = 30
    batch_size = 64
    lr = 1e-3
    patience = 7

    training_curves = {}
    models_by_fold = {}

    for fold in range(n_folds):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        fold_curves = {
            'train_loss': [],
            'val_loss': [],
        }

        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_channels = channels[train_mask]
        train_targets = targets[train_mask]
        val_channels = channels[val_mask]
        val_targets = targets[val_mask]

        print(f"  Train: {len(train_channels)} channels, Val: {len(val_channels)} channels")

        train_ds = DischargeDataset(train_channels, train_targets, augment=True)
        val_ds = DischargeDataset(val_channels, val_targets, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = DischargeDetector().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         pos_weight=POS_WEIGHT)

            val_results = evaluate(model, val_loader, pos_weight=POS_WEIGHT)
            val_loss = val_results['avg_loss']

            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_loss, 6))

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

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

        # Save fold model
        save_path = CACHE_DIR / f'discharge_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_loss={best_val_loss:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # -- Save training curves --------------------------------------------------
    curves_path = CACHE_DIR / 'training_curves_discharge.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Evaluation: IPI-based frequency estimation ----------------------------
    print("\n" + "=" * 70)
    print("EVALUATION: IPI-Based Frequency Estimation")
    print("=" * 70)

    # For each PD+ channel in the validation set, run detector and compute frequency
    all_gold_freqs = []
    all_pred_freqs = []
    all_n_peaks = []
    all_ipi_cvs = []

    for fold in range(n_folds):
        model = DischargeDetector().to(DEVICE)
        model.load_state_dict(models_by_fold[fold])
        model.eval()

        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])

        for i in range(n_total):
            if not val_mask[i]:
                continue
            if labels[i] < 0.5:
                continue

            pid = str(patient_ids[i])
            gold_freq = pid_to_freq.get(pid, np.nan)
            if not np.isfinite(gold_freq):
                continue

            # Run detector
            ch_data = channels[i].astype(np.float32).copy()
            if not np.all(np.isfinite(ch_data)):
                continue

            mu = np.mean(ch_data)
            std = np.std(ch_data)
            if std > 1e-8:
                ch_data = (ch_data - mu) / std
            else:
                ch_data = ch_data - mu

            x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :])
            with torch.no_grad():
                prob = model(x)  # (1, 1, 2000)

            prob_signal = prob[0, 0].numpy()
            pred_freq, n_peaks, ipi_cv = compute_ipi_frequency(prob_signal)

            if np.isfinite(pred_freq):
                all_gold_freqs.append(gold_freq)
                all_pred_freqs.append(pred_freq)
                all_n_peaks.append(n_peaks)
                all_ipi_cvs.append(ipi_cv)

    gold_arr = np.array(all_gold_freqs)
    pred_arr = np.array(all_pred_freqs)

    print(f"\n  Channels with valid IPI frequency: {len(gold_arr)}")

    if len(gold_arr) >= 5:
        rho, pval = spearmanr(gold_arr, pred_arr)
        mae = float(np.mean(np.abs(gold_arr - pred_arr)))
        print(f"  Channel-level Spearman: {rho:.4f} (p={pval:.2e})")
        print(f"  Channel-level MAE:      {mae:.4f} Hz")
        print(f"  Mean peaks detected:    {np.mean(all_n_peaks):.1f}")
        print(f"  Mean IPI CV:            {np.nanmean(all_ipi_cvs):.3f}")
    else:
        rho = float('nan')
        mae = float('nan')
        print(f"  Not enough valid channels for Spearman ({len(gold_arr)})")

    # -- Patient-level frequency estimation ------------------------------------
    print("\n--- Patient-level frequency estimation ---")

    # Aggregate per patient: use median of channel-level predictions
    patient_gold = {}
    patient_preds_list = {}

    for fold in range(n_folds):
        model = DischargeDetector().to(DEVICE)
        model.load_state_dict(models_by_fold[fold])
        model.eval()

        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])

        for i in range(n_total):
            if not val_mask[i]:
                continue
            if labels[i] < 0.5:
                continue

            pid = str(patient_ids[i])
            gold_freq = pid_to_freq.get(pid, np.nan)
            if not np.isfinite(gold_freq):
                continue

            ch_data = channels[i].astype(np.float32).copy()
            if not np.all(np.isfinite(ch_data)):
                continue

            mu = np.mean(ch_data)
            std = np.std(ch_data)
            if std > 1e-8:
                ch_data = (ch_data - mu) / std
            else:
                ch_data = ch_data - mu

            x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :])
            with torch.no_grad():
                prob = model(x)

            prob_signal = prob[0, 0].numpy()
            pred_freq, n_peaks, ipi_cv = compute_ipi_frequency(prob_signal)

            if np.isfinite(pred_freq):
                patient_gold[pid] = gold_freq
                if pid not in patient_preds_list:
                    patient_preds_list[pid] = []
                patient_preds_list[pid].append(pred_freq)

    # Aggregate
    pat_golds = []
    pat_preds = []
    for pid in patient_gold:
        if pid in patient_preds_list and len(patient_preds_list[pid]) > 0:
            pat_golds.append(patient_gold[pid])
            pat_preds.append(float(np.median(patient_preds_list[pid])))

    pat_golds = np.array(pat_golds)
    pat_preds = np.array(pat_preds)

    if len(pat_golds) >= 5:
        pat_rho, pat_pval = spearmanr(pat_golds, pat_preds)
        pat_mae = float(np.mean(np.abs(pat_golds - pat_preds)))
        print(f"  Patients: {len(pat_golds)}")
        print(f"  Patient-level Spearman: {pat_rho:.4f} (p={pat_pval:.2e})")
        print(f"  Patient-level MAE:      {pat_mae:.4f} Hz")
    else:
        pat_rho = float('nan')
        pat_mae = float('nan')
        print(f"  Not enough patients ({len(pat_golds)})")

    # -- Save results ----------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'experiment': 'discharge_detector_ipi_freq',
        'timestamp': time.time(),
        'channel_spearman': round(float(rho), 4) if np.isfinite(rho) else None,
        'channel_mae': round(float(mae), 4) if np.isfinite(mae) else None,
        'channel_n': len(gold_arr),
        'patient_spearman': round(float(pat_rho), 4) if np.isfinite(pat_rho) else None,
        'patient_mae': round(float(pat_mae), 4) if np.isfinite(pat_mae) else None,
        'patient_n': len(pat_golds),
        'mean_peaks': round(float(np.mean(all_n_peaks)), 1) if all_n_peaks else None,
        'mean_ipi_cv': round(float(np.nanmean(all_ipi_cvs)), 3) if all_ipi_cvs else None,
    }

    results_path = RESULTS_DIR / 'discharge_detector_results.json'
    with open(str(results_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
