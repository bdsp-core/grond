"""
Train CET-UNet (CNN Evidence Trace with U-Net architecture) for frame-level
discharge detection.

Training targets: Sharp Gaussian bumps (sigma=2 samples = 10ms) at
MW-reviewed discharge times.
- Involved channels get discharge-peak targets
- Uninvolved channels get all-zeros target

Loss: weighted BCE + sharpness penalty (mean evidence).
5-fold patient-stratified CV, early stopping.

Usage:
    conda run -n foe_dl python code/cet_model/train_cet.py
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
print(f"Using device: {DEVICE}")

# Training hyperparameters
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 7
POS_WEIGHT = 20.0  # weight for positive (discharge) samples — higher for sharp targets
SHARPNESS_PENALTY = 0.1  # coefficient for mean-evidence penalty
GAUSSIAN_SIGMA = 2  # samples = 10ms at 200 Hz — very sharp peaks
N_SAMPLES = 2000
TOLERANCE_S = 0.1  # +/- 100ms for evaluation


# -- Dataset -----------------------------------------------------------------

class CETDataset(Dataset):
    """Dataset for CET training.

    Each sample is a single channel with a frame-level target signal
    and the corresponding HPP evidence trace (for auxiliary loss).
    """

    def __init__(self, channels, targets, hpp_evidence=None, augment=False):
        """
        Args:
            channels: (N, 2000) float32 array of channel data
            targets: (N, 2000) float32 array of target signals
            hpp_evidence: (N, 2000) float32 array of HPP evidence (for auxiliary loss)
            augment: whether to apply data augmentation
        """
        self.channels = channels.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.hpp_evidence = hpp_evidence.astype(np.float32) if hpp_evidence is not None else None
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

        if self.hpp_evidence is not None:
            h = self.hpp_evidence[idx].copy()
            h_tensor = torch.from_numpy(h[np.newaxis, :])  # (1, 2000)
            return x_tensor, y_tensor, h_tensor
        return x_tensor, y_tensor, torch.zeros_like(y_tensor)

    def _augment(self, x, y):
        # Random amplitude scaling (0.8-1.2x) — only affects x, not target
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale

        # Random Gaussian noise (SNR 20-40 dB)
        snr_db = np.random.uniform(20, 40)
        signal_power = np.mean(x ** 2)
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(x)) * np.sqrt(noise_power)
            x = x + noise.astype(np.float32)

        return x, y


def make_target_signal(discharge_times_s, n_samples=N_SAMPLES, fs=FS,
                       sigma=GAUSSIAN_SIGMA):
    """Create target signal with sharp Gaussian bumps at discharge times.

    Args:
        discharge_times_s: list of discharge times in seconds
        n_samples: number of samples in the signal
        fs: sampling rate
        sigma: std dev of Gaussian bumps in samples (2 = 10ms at 200 Hz)

    Returns:
        (n_samples,) float32 array with values in [0, 1]
    """
    target = np.zeros(n_samples, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float32)

    for dt in discharge_times_s:
        center = dt * fs
        bump = np.exp(-0.5 * ((t - center) / sigma) ** 2)
        target = np.maximum(target, bump)

    return target


# -- Data preparation -------------------------------------------------------

def prepare_cet_data():
    """Load data and create channel-level training samples.

    Returns:
        channels: (N, 2000) array
        targets: (N, 2000) array
        patient_ids: (N,) array of patient IDs
        subtypes: (N,) array of subtypes
    """
    print("Loading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Load ground truth discharge times
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)

    # Load channel pseudolabels for involvement info
    pseudo_path = PROJECT_DIR / 'data' / 'labels' / 'channel_pseudolabels.json'
    with open(str(pseudo_path)) as f:
        pseudolabels = json.load(f)

    # Only use ground_truth cases
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

        # Get segment
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]  # first segment, (18, 2000)

        if seg.shape[1] != N_SAMPLES:
            continue

        # Get subtype
        subtype = gt_data.get('subtype', 'unknown')

        # Get channel involvement from pseudolabels
        ch_labels = pseudolabels.get(pid, {})
        ch_dict = ch_labels.get('channels', {}) if isinstance(ch_labels, dict) else {}

        # Create target signal (sharp Gaussian bumps at discharge times)
        discharge_target = make_target_signal(discharge_times)

        n_ch = min(seg.shape[0], 18)
        for ch_idx in range(n_ch):
            ch_data = seg[ch_idx]

            # Skip channels with NaN/Inf
            if not np.all(np.isfinite(ch_data)):
                continue

            # Determine if this channel is involved
            ch_info = ch_dict.get(str(ch_idx), {})
            is_involved = ch_info.get('pd_label', 0) == 1

            if is_involved:
                target = discharge_target.copy()
                n_involved += 1
            else:
                target = np.zeros(N_SAMPLES, dtype=np.float32)
                n_uninvolved += 1

            # Compute HPP evidence for this channel (for auxiliary loss)
            hpp_ev = _compute_channel_evidence(ch_data, FS)
            # Normalize to [0, 1]
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
    print(f"  Gaussian sigma: {GAUSSIAN_SIGMA} samples ({GAUSSIAN_SIGMA * 1000 / FS:.0f}ms)")

    return channels, targets, hpp_evidence, patient_ids, subtypes


# -- Training loop -----------------------------------------------------------

AUX_LAMBDA = 0.5  # weight for "don't be worse than HPP" auxiliary loss

def train_one_epoch(model, loader, optimizer, scheduler):
    """Train one epoch with weighted BCE + HPP floor auxiliary loss + sharpness penalty."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y, hpp_ev in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        hpp_ev = hpp_ev.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)  # (B, 1, 2000)

        # Weighted BCE: positive samples weighted POS_WEIGHT x higher
        weights = torch.where(y > 0.1, POS_WEIGHT, 1.0)
        bce_loss = nn.functional.binary_cross_entropy(pred, y, weight=weights)

        # Auxiliary loss: CNN evidence should be at least as high as HPP at discharge locations
        # Only penalize where HPP > CNN (CNN should match or exceed HPP)
        # Mask to discharge locations (where y > 0.1) to focus on peaks
        discharge_mask = (y > 0.1).float()
        hpp_floor_violation = torch.clamp(hpp_ev - pred, min=0) * discharge_mask
        aux_loss = AUX_LAMBDA * (hpp_floor_violation ** 2).mean()

        # Sharpness penalty: penalize high mean evidence (encourages near-zero baseline)
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
    """Evaluate model loss on a loader."""
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


# -- Peak-based evaluation --------------------------------------------------

@torch.no_grad()
def evaluate_peaks(model, channels, targets, patient_ids, val_mask,
                   hpp_data, fs=FS):
    """Evaluate CET peak detection against MW ground truth.

    For each test channel that has discharge targets:
    1. Run CET to get evidence trace
    2. Peak-pick at threshold 0.3
    3. Compare with ground truth times at +/-100ms tolerance

    Returns dict with sensitivity, precision, F1.
    """
    model.eval()

    total_tp = 0
    total_fn = 0
    total_fp = 0
    match_errors = []

    val_indices = np.where(val_mask)[0]

    # Group by patient to get GT times
    processed_patients = set()
    for idx in val_indices:
        pid = patient_ids[idx]
        if pid in processed_patients:
            continue

        gt_data = hpp_data.get(pid, {})
        gt_times = gt_data.get('global_times', [])
        if len(gt_times) < 2:
            continue

        # Check if this channel has discharge targets (is involved)
        target = targets[idx]
        if np.max(target) < 0.1:
            continue  # uninvolved channel, skip for peak eval

        # Get CET prediction
        ch_data = channels[idx].copy()
        mu = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 1e-8:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu

        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :].astype(np.float32)).to(DEVICE)
        evidence = model(x).squeeze().cpu().numpy()  # (2000,)

        # Estimate frequency from EEG (ACF) — no gold standard used
        from pd_pointiness_acf import compute_acf_frequency
        from scipy.signal import butter, filtfilt
        b_lp, a_lp = butter(4, 20.0 / (fs / 2), btype='low')
        acf_freqs = []
        # Use patient's segment to estimate frequency
        pat_segs = None
        try:
            from optimization_harness_v2 import load_dataset
            if not hasattr(evaluate_peaks, '_segments'):
                _ds = load_dataset(verbose=False)
                evaluate_peaks._segments = _ds['segments']
            pat_segs_list = evaluate_peaks._segments.get(pid, [])
            if pat_segs_list:
                seg = pat_segs_list[0]
                n_ch = min(seg.shape[0], 18)
                for ch_i in range(n_ch):
                    try:
                        sig = filtfilt(b_lp, a_lp, seg[ch_i])
                    except ValueError:
                        sig = seg[ch_i]
                    freq_est, _, _ = compute_acf_frequency(
                        sig, fs, method='pointiness',
                        smoothing_sigma=0.02, acf_min_lag=0.4,
                        acf_peak_threshold=0.10, peak_height_frac=0.3)
                    if np.isfinite(freq_est):
                        acf_freqs.append(freq_est)
        except Exception:
            pass
        if acf_freqs:
            estimated_freq = float(np.clip(np.median(acf_freqs), 0.3, 3.5))
        else:
            estimated_freq = 1.0
        expected_period = 1.0 / estimated_freq
        min_dist = max(20, int(0.3 * expected_period * fs))

        # Peak-pick
        peaks, _ = find_peaks(evidence, height=0.3, distance=min_dist)
        pred_times = peaks / fs

        # Match predicted to GT
        gt_matched = [False] * len(gt_times)
        pred_matched = [False] * len(pred_times)

        for gi, gt in enumerate(gt_times):
            best_dist, best_pi = np.inf, -1
            for pi, pt in enumerate(pred_times):
                if not pred_matched[pi]:
                    dist = abs(gt - pt)
                    if dist < best_dist:
                        best_dist = dist
                        best_pi = pi
            if best_dist <= TOLERANCE_S and best_pi >= 0:
                gt_matched[gi] = True
                pred_matched[best_pi] = True
                match_errors.append(best_dist)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(pred_times) - sum(pred_matched)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    mean_error = np.mean(match_errors) * 1000 if match_errors else float('nan')

    return {
        'sensitivity': sens,
        'precision': prec,
        'f1': f1,
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'timing_error_ms': mean_error,
    }


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("CET-UNet Training (sigma=2, sharp targets, skip connections)")
    print("=" * 70)

    # Load data
    channels, targets, hpp_evidence, patient_ids, subtypes = prepare_cet_data()

    # Load HPP data for evaluation
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    # Create 5-fold patient-stratified splits
    print("\nCreating 5-fold patient-stratified splits...")
    unique_pids = np.unique(patient_ids)
    rng = np.random.RandomState(42)

    # Group by subtype for stratification
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
    print("Training CET-UNet across 5 folds...")
    print(f"  POS_WEIGHT={POS_WEIGHT}, SHARPNESS_PENALTY={SHARPNESS_PENALTY}, AUX_LAMBDA={AUX_LAMBDA}")
    print(f"  GAUSSIAN_SIGMA={GAUSSIAN_SIGMA} samples ({GAUSSIAN_SIGMA * 1000 / FS:.0f}ms)")
    print(f"  HPP floor auxiliary loss: ON")
    print("=" * 70)

    models_by_fold = {}
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

        # Training progress file for live dashboard
        progress_path = CACHE_DIR / 'training_progress.json'

        for epoch in range(N_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            val_loss = evaluate_loss(model, val_loader)

            if epoch % 5 == 0 or epoch == N_EPOCHS - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.6f}, "
                      f"val_loss={val_loss:.6f}")

            # Write progress for live dashboard (with curves)
            import time as _time
            if not hasattr(main, '_curves'):
                main._curves = {}
            fold_key = f'fold_{fold}'
            if fold_key not in main._curves:
                main._curves[fold_key] = {'train_loss': [], 'val_loss': []}
            main._curves[fold_key]['train_loss'].append(round(train_loss, 6))
            main._curves[fold_key]['val_loss'].append(round(val_loss, 6))

            progress = {
                'fold': fold + 1, 'total_folds': N_FOLDS,
                'epoch': epoch + 1, 'total_epochs': N_EPOCHS,
                'train_loss': round(train_loss, 6),
                'val_loss': round(val_loss, 6),
                'best_val_loss': round(best_val_loss if best_val_loss < float('inf') else val_loss, 6),
                'folds_complete': fold,
                'device': str(DEVICE),
                'timestamp': _time.time(),
                'curves': main._curves,
            }
            with open(str(progress_path), 'w') as _pf:
                json.dump(progress, _pf)
            js_path = CACHE_DIR / 'training_progress.js'
            with open(str(js_path), 'w') as _jf:
                _jf.write('var TRAINING_PROGRESS = ')
                json.dump(progress, _jf)
                _jf.write(';\n')

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

        # Save fold model as cet_unet_fold{i}.pt (keep old cet_fold{i}.pt)
        save_path = CACHE_DIR / f'cet_unet_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        # Evaluate peak detection on validation set
        peak_results = evaluate_peaks(model, channels, targets, patient_ids,
                                      val_mask, gt_cases)
        all_peak_results.append(peak_results)

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_loss={best_val_loss:.6f} ({fold_elapsed:.1f}s)")
        print(f"  Peak detection — Sens={peak_results['sensitivity']:.3f}, "
              f"Prec={peak_results['precision']:.3f}, F1={peak_results['f1']:.3f}")
        print(f"  Model saved to {save_path}")

    # Aggregate results across folds
    print("\n" + "=" * 70)
    print("CET-UNet CROSS-VALIDATION RESULTS (Raw CET peak detection)")
    print("=" * 70)

    total_tp = sum(r['tp'] for r in all_peak_results)
    total_fn = sum(r['fn'] for r in all_peak_results)
    total_fp = sum(r['fp'] for r in all_peak_results)

    overall_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_f1 = 2 * overall_prec * overall_sens / (overall_prec + overall_sens) \
        if (overall_prec + overall_sens) > 0 else 0

    timing_errors = [r['timing_error_ms'] for r in all_peak_results
                     if np.isfinite(r['timing_error_ms'])]
    mean_timing = np.mean(timing_errors) if timing_errors else float('nan')

    print(f"\n  Channel-level discharge detection (threshold=0.3, +/-100ms):")
    print(f"    True positives:  {total_tp}")
    print(f"    False negatives: {total_fn}")
    print(f"    False positives: {total_fp}")
    print(f"    Sensitivity:     {overall_sens:.3f}")
    print(f"    Precision:       {overall_prec:.3f}")
    print(f"    F1 score:        {overall_f1:.3f}")
    print(f"    Mean timing err: {mean_timing:.1f} ms")

    # Save results
    results = {
        'model': 'CETUNet',
        'gaussian_sigma': GAUSSIAN_SIGMA,
        'pos_weight': POS_WEIGHT,
        'sharpness_penalty': SHARPNESS_PENALTY,
        'sensitivity': round(overall_sens, 4),
        'precision': round(overall_prec, 4),
        'f1': round(overall_f1, 4),
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'mean_timing_error_ms': round(mean_timing, 1) if np.isfinite(mean_timing) else None,
        'per_fold': [
            {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
             for k, v in r.items()}
            for r in all_peak_results
        ],
    }
    results_path = CACHE_DIR / 'cet_unet_cv_results.json'
    with open(str(results_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
