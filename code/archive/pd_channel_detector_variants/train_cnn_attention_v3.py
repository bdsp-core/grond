"""
Train CNN+Attention v3: Use v1 CURATED dataset (815 patients, 9310 channels)
with UPDATED expert frequency labels from segment_labels.csv.

Changes from v1:
  - Loads channel_dataset.npz from pd_channel_cache_backup_v1/ (curated 815 patients)
  - Updates frequency labels from current segment_labels.csv expert_freq_hz
  - Falls back to patients.csv gold_standard_freq if no expert_freq_hz
  - Saves weights to v3_cnn_attn_fold{0-4}.pt (does NOT overwrite v1)
  - Uses MPS device for GPU acceleration
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
V1_BACKUP_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache_backup_v1'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"[v3] Using device: {DEVICE}")

SAVE_PREFIX = 'v3_cnn_attn'


# -- Dataset -----------------------------------------------------------------

class ChannelPDDataset(Dataset):
    """PyTorch dataset for channel-level PD detection + frequency estimation."""

    def __init__(self, channels, labels, freq_targets, augment=False):
        self.channels = channels.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.freq_targets = freq_targets.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.channels[idx].copy()

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

        return x_tensor, label, freq, mask

    def _augment(self, x):
        scale = np.random.uniform(0.8, 1.2)
        x = x * scale

        snr_db = np.random.uniform(20, 40)
        signal_power = np.mean(x ** 2)
        if signal_power > 1e-10:
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(len(x)) * np.sqrt(noise_power)
            x = x + noise.astype(np.float32)

        shift = np.random.randint(-50, 51)
        if shift != 0:
            x = np.roll(x, shift)

        return x


# -- Training loop -----------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, alpha=0.5):
    model.train()
    total_loss = 0.0
    n_batches = 0

    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')

    for x, label, freq, mask in loader:
        x = x.to(DEVICE)
        label = label.to(DEVICE)
        freq = freq.to(DEVICE)
        mask = mask.to(DEVICE)

        optimizer.zero_grad()
        pd_prob, freq_pred, _attn = model(x)

        pd_prob = pd_prob.squeeze(1)
        freq_pred = freq_pred.squeeze(1)

        loss_bce = bce(pd_prob, label).mean()

        freq_clean = torch.where(torch.isnan(freq), torch.zeros_like(freq), freq)
        loss_freq = (mse(freq_pred, freq_clean) * mask).sum()
        n_masked = mask.sum()
        if n_masked > 0:
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
    model.eval()
    all_pd_probs = []
    all_labels = []
    all_freq_preds = []
    all_freq_targets = []
    all_masks = []

    bce = nn.BCELoss(reduction='none')
    mse = nn.MSELoss(reduction='none')
    total_loss = 0.0
    n_batches = 0

    for x, label, freq, mask in loader:
        x = x.to(DEVICE)
        label_dev = label.to(DEVICE)
        freq_dev = freq.to(DEVICE)
        mask_dev = mask.to(DEVICE)

        pd_prob, freq_pred, _attn = model(x)
        pd_prob_sq = pd_prob.squeeze(1)
        freq_pred_sq = freq_pred.squeeze(1)

        loss_bce = bce(pd_prob_sq, label_dev).mean()
        freq_clean = torch.where(torch.isnan(freq_dev), torch.zeros_like(freq_dev), freq_dev)
        loss_freq = (mse(freq_pred_sq, freq_clean) * mask_dev).sum()
        n_masked = mask_dev.sum()
        if n_masked > 0:
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

    result = {
        'pd_probs': np.concatenate(all_pd_probs),
        'labels': np.concatenate(all_labels),
        'freq_preds': np.concatenate(all_freq_preds),
        'freq_targets': np.concatenate(all_freq_targets),
        'masks': np.concatenate(all_masks),
        'avg_loss': total_loss / max(n_batches, 1),
    }
    return result


def compute_freq_spearman(val_results):
    masks = val_results['masks']
    freq_preds = val_results['freq_preds']
    freq_targets = val_results['freq_targets']
    valid = (masks > 0.5) & np.isfinite(freq_targets) & np.isfinite(freq_preds)
    if np.sum(valid) < 5:
        return float('nan')
    rho, _ = spearmanr(freq_targets[valid], freq_preds[valid])
    return float(rho) if np.isfinite(rho) else float('nan')


def compute_auc(y_true, y_score):
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


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("V3: CNN + Attention — Curated 815 patients + Updated Expert Freq Labels")
    print("=" * 70)

    # -- Load V1 CURATED channel dataset (815 patients) ----------------------
    data_path = V1_BACKUP_DIR / 'channel_dataset.npz'
    print(f"\nLoading V1 CURATED channel dataset from {data_path}...")
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

    # -- Load UPDATED frequency labels from segment_labels.csv ---------------
    import pandas as pd
    print("\nLoading UPDATED frequency labels...")

    # Priority 1: expert_freq_hz from segment_labels.csv
    seg_labels = pd.read_csv(str(PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv'))
    pid_to_expert_freq = {}
    for _, row in seg_labels.iterrows():
        pid = str(row['patient_id'])
        expert_freq = row.get('expert_freq_hz')
        if pd.notna(expert_freq) and float(expert_freq) > 0:
            pid_to_expert_freq[pid] = float(expert_freq)

    # Priority 2: gold_standard_freq from patients.csv (fallback)
    pid_to_gold_freq = {}
    try:
        pat_df = pd.read_csv(str(PROJECT_DIR / 'data' / 'labels' / 'patients.csv'))
        for _, row in pat_df.iterrows():
            pid = str(row['patient_id'])
            freq = row.get('gold_standard_freq')
            if pd.notna(freq) and float(freq) > 0:
                pid_to_gold_freq[pid] = float(freq)
    except Exception:
        pass

    # Build combined freq map: expert takes priority
    pid_to_freq = {}
    for pid in unique_patients:
        pid_str = str(pid)
        if pid_str in pid_to_expert_freq:
            pid_to_freq[pid_str] = pid_to_expert_freq[pid_str]
        elif pid_str in pid_to_gold_freq:
            pid_to_freq[pid_str] = pid_to_gold_freq[pid_str]

    # Create freq targets (log Hz) for PD+ channels
    freq_targets = np.full(n_total, np.nan, dtype=np.float32)
    n_with_expert = 0
    n_with_gold = 0
    n_with_freq = 0
    for i in range(n_total):
        if labels[i] == 1:
            pid = str(patient_ids[i])
            if pid in pid_to_freq:
                freq_targets[i] = np.log(pid_to_freq[pid])
                n_with_freq += 1
                if pid in pid_to_expert_freq:
                    n_with_expert += 1
                else:
                    n_with_gold += 1

    print(f"  PD+ channels with frequency labels: {n_with_freq}/{n_pos}")
    print(f"    From expert_freq_hz: {n_with_expert}")
    print(f"    From gold_standard_freq (fallback): {n_with_gold}")
    print(f"  Unique patients with expert freq: {len(pid_to_expert_freq)}")

    # -- Create 5-fold patient-stratified splits ----------------------------
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

    # -- Training loop across folds -----------------------------------------
    print("\n" + "=" * 70)
    print("Training CNN+Attention V3 across 5 folds...")
    print("=" * 70)

    oof_pd_probs = np.full(n_total, np.nan)
    oof_freq_preds = np.full(n_total, np.nan)

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
        }

        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_channels = channels[train_mask]
        train_labels = labels[train_mask]
        train_freqs = freq_targets[train_mask]
        val_channels = channels[val_mask]
        val_labels = labels[val_mask]
        val_freqs = freq_targets[val_mask]

        print(f"  Train: {len(train_labels)} channels, Val: {len(val_labels)} channels")

        train_ds = ChannelPDDataset(train_channels, train_labels, train_freqs, augment=True)
        val_ds = ChannelPDDataset(val_channels, val_labels, val_freqs, augment=False)

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

            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_loss, 6))
            fold_curves['val_auc'].append(round(val_auc, 6) if np.isfinite(val_auc) else None)
            fold_curves['val_freq_spearman'].append(
                round(val_freq_sp, 6) if np.isfinite(val_freq_sp) else None
            )

            if epoch % 5 == 0 or epoch == n_epochs - 1:
                print(f"  Epoch {epoch + 1:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_AUC={val_auc:.4f}, val_freq_sp={val_freq_sp:.4f}")

            if np.isfinite(val_auc) and val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
                    break

        training_curves[f'fold_{fold}'] = fold_curves

        if best_state is not None:
            model.load_state_dict(best_state)

        val_results = evaluate(model, val_loader, alpha=alpha)
        oof_pd_probs[val_mask] = val_results['pd_probs']
        oof_freq_preds[val_mask] = val_results['freq_preds']

        # Save fold model with v3 prefix
        save_path = CACHE_DIR / f'{SAVE_PREFIX}_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_AUC={best_val_auc:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # -- Save training curves ------------------------------------------------
    curves_path = CACHE_DIR / f'{SAVE_PREFIX}_training_curves.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Compute overall channel-level metrics -------------------------------
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RESULTS (CNN + Attention V3)")
    print("=" * 70)

    valid = np.isfinite(oof_pd_probs)
    y_true = labels[valid].astype(int)
    y_prob = oof_pd_probs[valid]
    y_pred = (y_prob >= 0.5).astype(int)
    pids_valid = patient_ids[valid]

    ch_auc = compute_auc(y_true, y_prob)
    ch_acc = float(np.mean(y_true == y_pred))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ch_bal_acc = (sens + spec) / 2

    print(f"\nChannel-level metrics (N={len(y_true)}):")
    print(f"  AUC:              {ch_auc:.4f}")
    print(f"  Accuracy:         {ch_acc:.4f}")
    print(f"  Balanced Accuracy:{ch_bal_acc:.4f}")
    print(f"  Sensitivity:      {sens:.4f}")
    print(f"  Specificity:      {spec:.4f}")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    # Channel-level frequency Spearman
    print(f"\nChannel-level frequency estimation (PD+ channels only):")
    freq_mask = valid & (labels == 1) & np.isfinite(freq_targets)
    n_freq = int(np.sum(freq_mask))

    ch_freq_rho = float('nan')
    if n_freq > 5:
        freq_pred_valid = oof_freq_preds[freq_mask]
        freq_true_valid = freq_targets[freq_mask]
        ch_freq_rho, pval = spearmanr(freq_true_valid, freq_pred_valid)
        print(f"  N channels with freq labels: {n_freq}")
        print(f"  Spearman (log space):   {ch_freq_rho:.4f} (p={pval:.2e})")

    # -- Patient-level frequency estimation ---------------------------------
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL FREQUENCY ESTIMATION (CNN+Attention V3)")
    print(f"{'=' * 70}")

    main_dataset = load_dataset(verbose=False)
    df = main_dataset['df']
    segments = main_dataset['segments']

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
        all_freq_preds_list = []

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
                    all_freq_preds_list.append(freq_pred.item())

        if len(all_pd_probs) == 0:
            continue

        pd_probs_arr = np.array(all_pd_probs)
        freq_preds_arr = np.array(all_freq_preds_list)

        weights = pd_probs_arr.copy()
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weighted_freq_log = np.sum(weights * freq_preds_arr) / weight_sum
        else:
            weighted_freq_log = np.mean(freq_preds_arr)

        pred_freq = np.exp(weighted_freq_log)
        pred_freq = np.clip(pred_freq, 0.1, 10.0)

        patient_preds[pid] = pred_freq
        patient_golds[pid] = gold
        patient_subtypes[pid] = subtype

    # Compute metrics by subtype
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

    print(f"\n  CNN+Attention V3 patient-level frequency results:")
    print(f"    Combined (N={comb['n']}): Spearman={comb['spearman']:.4f}, MAE={comb['mae']:.4f}")
    print(f"    LPD      (N={lpd['n']}):  Spearman={lpd['spearman']:.4f}, MAE={lpd['mae']:.4f}")
    print(f"    GPD      (N={gpd['n']}):  Spearman={gpd['spearman']:.4f}, MAE={gpd['mae']:.4f}")

    # -- Save results as JSON ------------------------------------------------
    freq_results = {
        'experiment': 'cnn_attention_v3_curated_expert_freq',
        'version': 'v3',
        'dataset': 'v1_curated_815_patients',
        'freq_labels': 'expert_freq_hz_priority',
        'timestamp': time.time(),
        'n_patients': n_patients,
        'n_channels': n_total,
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
        'channel_freq_spearman': round(float(ch_freq_rho), 4) if np.isfinite(ch_freq_rho) else None,
    }

    # Replace NaN with None for JSON
    freq_results_json = {}
    for k, v in freq_results.items():
        if isinstance(v, float) and np.isnan(v):
            freq_results_json[k] = None
        else:
            freq_results_json[k] = v

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_json_path = RESULTS_DIR / 'v3_cnn_attn_results.json'
    with open(str(results_json_path), 'w') as f:
        json.dump(freq_results_json, f, indent=2)
    print(f"\n  V3 results saved to {results_json_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
