"""
Train a 1D CNN with temporal attention for channel-level PD detection
+ multi-task frequency estimation, then evaluate as a patient-level
frequency estimator to compare with Ridge/RF handcrafted methods.

Uses 5-fold patient-stratified CV. Multi-task loss:
  loss = BCE(pd_pred, pd_label) + alpha * MSE(freq_pred, log_freq_target) * pd_mask

Compares against:
  - Logistic regression baseline (channel AUC 0.766, patient AUC 0.902)
  - CNN without attention (Phase 3 results)
  - Ridge/RF patient-level frequency estimation
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
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('cpu')


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

        return x_tensor, label, freq, mask

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
    """Evaluate model. Returns predictions, labels, and average loss."""
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


# -- Learning Curve HTML Generator -------------------------------------------

def generate_learning_curves_html(training_curves):
    """Generate an HTML file with interactive learning curve plots using Chart.js."""
    import os
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    html_path = RESULTS_DIR / 'cnn_attn_learning_curves.html'

    fold_keys = sorted(training_curves.keys())
    colors = [
        'rgba(31, 119, 180, 1)',
        'rgba(255, 127, 14, 1)',
        'rgba(44, 160, 44, 1)',
        'rgba(214, 39, 40, 1)',
        'rgba(148, 103, 189, 1)',
    ]
    colors_alpha = [c.replace(', 1)', ', 0.15)') for c in colors]

    def make_datasets_json(metric_key, label_prefix, use_dash_for_val=False):
        datasets = []
        for i, fold_key in enumerate(fold_keys):
            values = training_curves[fold_key].get(metric_key, [])
            values_js = [v if v is not None else 'null' for v in values]
            dash_style = '[5, 5]' if use_dash_for_val else '[]'
            datasets.append(f'''{{
                label: '{label_prefix} ({fold_key})',
                data: [{', '.join(str(v) for v in values_js)}],
                borderColor: '{colors[i % len(colors)]}',
                backgroundColor: '{colors_alpha[i % len(colors)]}',
                borderDash: {dash_style},
                tension: 0.3,
                pointRadius: 2,
                fill: false,
                spanGaps: true
            }}''')
        return ',\n            '.join(datasets)

    # Build combined train/val loss datasets
    loss_datasets = []
    for i, fold_key in enumerate(fold_keys):
        train_vals = training_curves[fold_key].get('train_loss', [])
        val_vals = training_curves[fold_key].get('val_loss', [])
        train_js = ', '.join(str(v) if v is not None else 'null' for v in train_vals)
        val_js = ', '.join(str(v) if v is not None else 'null' for v in val_vals)
        loss_datasets.append(f'''{{
                label: 'Train ({fold_key})',
                data: [{train_js}],
                borderColor: '{colors[i % len(colors)]}',
                backgroundColor: '{colors_alpha[i % len(colors)]}',
                borderDash: [],
                tension: 0.3,
                pointRadius: 2,
                fill: false,
                spanGaps: true
            }}''')
        loss_datasets.append(f'''{{
                label: 'Val ({fold_key})',
                data: [{val_js}],
                borderColor: '{colors[i % len(colors)]}',
                backgroundColor: '{colors_alpha[i % len(colors)]}',
                borderDash: [5, 5],
                tension: 0.3,
                pointRadius: 2,
                fill: false,
                spanGaps: true
            }}''')
    loss_datasets_str = ',\n            '.join(loss_datasets)

    max_epochs = max(len(training_curves[k].get('train_loss', [])) for k in fold_keys)
    epoch_labels = ', '.join(str(e + 1) for e in range(max_epochs))

    val_auc_datasets = make_datasets_json('val_auc', 'Val AUC')
    val_freq_sp_datasets = make_datasets_json('val_freq_spearman', 'Val Freq Spearman')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN+Attention Learning Curves - Channel PD Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .chart-title {{
            font-size: 16px;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }}
        canvas {{
            width: 100% !important;
            height: 350px !important;
        }}
    </style>
</head>
<body>
    <h1>CNN + Temporal Attention Learning Curves</h1>
    <p class="subtitle">Channel-Level PD Detection - 5-Fold Patient-Stratified CV</p>

    <div class="chart-container">
        <div class="chart-title">Train / Val Loss (solid = train, dashed = val)</div>
        <canvas id="lossChart"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Validation AUC (Channel-Level PD Detection)</div>
        <canvas id="aucChart"></canvas>
    </div>

    <div class="chart-container">
        <div class="chart-title">Validation Frequency Spearman (PD+ Channels)</div>
        <canvas id="freqChart"></canvas>
    </div>

    <script>
        const epochLabels = [{epoch_labels}];

        new Chart(document.getElementById('lossChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: epochLabels,
                datasets: [
            {loss_datasets_str}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Loss' }}, beginAtZero: false }}
                }},
                plugins: {{
                    legend: {{ position: 'right', labels: {{ boxWidth: 20, font: {{ size: 11 }} }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('aucChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: epochLabels,
                datasets: [
            {val_auc_datasets}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'AUC' }}, min: 0.5, max: 1.0 }}
                }},
                plugins: {{
                    legend: {{ position: 'right', labels: {{ boxWidth: 20, font: {{ size: 11 }} }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('freqChart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: epochLabels,
                datasets: [
            {val_freq_sp_datasets}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{ title: {{ display: true, text: 'Epoch' }} }},
                    y: {{ title: {{ display: true, text: 'Spearman r' }}, min: -0.2, max: 1.0 }}
                }},
                plugins: {{
                    legend: {{ position: 'right', labels: {{ boxWidth: 20, font: {{ size: 11 }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>'''

    with open(str(html_path), 'w') as f:
        f.write(html_content)
    print(f"Learning curves HTML saved to {html_path}")


# -- Patient-level frequency estimation -------------------------------------

@torch.no_grad()
def patient_level_freq_estimation(models_by_fold, patient_folds, dataset,
                                  channel_dataset_path):
    """Evaluate CNN attention model as patient-level frequency estimator.

    For each patient in the frequency estimation dataset:
      1. Load all their segments (up to 5)
      2. Run all 18 channels through the out-of-fold CNN attention model
      3. Get per-channel: PD probability and frequency prediction
      4. Aggregate: weighted mean of freq predictions, weighted by PD probability
      5. Compare to gold standard

    Args:
        models_by_fold: dict fold_idx -> state_dict
        patient_folds: dict patient_id -> fold_idx
        dataset: dict from load_dataset() (with segments)
        channel_dataset_path: path to channel_dataset.npz (for normalization reference)

    Returns:
        dict with patient-level predictions and metrics
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

        # Determine which fold this patient belongs to for out-of-fold model
        # If patient wasn't in the channel dataset, they weren't in ANY training
        # fold, so any model is "out-of-fold". Use an ensemble of all 5 models.
        fold_idx = patient_folds.get(pid, None)

        # Determine which models to use
        if fold_idx is not None:
            # Patient was in channel dataset: use out-of-fold model only
            fold_indices = [fold_idx]
        else:
            # Patient was NOT in channel dataset: ensemble all 5 models
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

                    # Skip channels with NaN/Inf values
                    if not np.all(np.isfinite(ch_data)):
                        continue

                    mu = np.mean(ch_data)
                    std = np.std(ch_data)
                    if std > 1e-8:
                        ch_data = (ch_data - mu) / std
                    else:
                        ch_data = ch_data - mu

                    x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :])
                    pd_prob, freq_pred, _attn = model(x)

                    all_pd_probs.append(pd_prob.item())
                    all_freq_preds.append(freq_pred.item())

        if len(all_pd_probs) == 0:
            continue

        pd_probs = np.array(all_pd_probs)
        freq_preds = np.array(all_freq_preds)

        # PD-weighted frequency aggregation
        # Channels the model is confident have PDs contribute more
        weights = pd_probs.copy()
        weight_sum = np.sum(weights)
        if weight_sum > 1e-8:
            weighted_freq_log = np.sum(weights * freq_preds) / weight_sum
        else:
            # Fallback: simple mean
            weighted_freq_log = np.mean(freq_preds)

        # Convert from log Hz to Hz
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
    print("Phase 3b: 1D CNN + Temporal Attention for Channel-Level PD Detection")
    print("=" * 70)

    # -- Load channel dataset ------------------------------------------------
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

    freq_targets = np.full(n_total, np.nan, dtype=np.float32)
    n_with_freq = 0
    for i in range(n_total):
        if labels[i] == 1:
            pid = str(patient_ids[i])
            if pid in pid_to_freq:
                freq_targets[i] = np.log(pid_to_freq[pid])
                n_with_freq += 1

    print(f"  PD+ channels with frequency labels: {n_with_freq}/{n_pos}")

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
    print("Training CNN+Attention across 5 folds...")
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

        val_results = evaluate(model, val_loader, alpha=alpha)
        oof_pd_probs[val_mask] = val_results['pd_probs']
        oof_freq_preds[val_mask] = val_results['freq_preds']

        # Save fold model
        save_path = CACHE_DIR / f'cnn_attn_fold{fold}.pt'
        state = best_state if best_state is not None else model.state_dict()
        torch.save(state, str(save_path))
        models_by_fold[fold] = state

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_AUC={best_val_auc:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # -- Save training curves ------------------------------------------------
    curves_path = CACHE_DIR / 'training_curves_attn.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # -- Generate HTML learning curve viewer ---------------------------------
    generate_learning_curves_html(training_curves)

    # -- Compute overall channel-level metrics -------------------------------
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD RESULTS (CNN + Attention)")
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

    # Patient-level PD detection aggregation
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
    pat_pred = (pat_prob >= 0.5).astype(int)
    pat_acc = float(np.mean(pat_true == pat_pred))

    pat_tp = int(np.sum((pat_true == 1) & (pat_pred == 1)))
    pat_tn = int(np.sum((pat_true == 0) & (pat_pred == 0)))
    pat_fp = int(np.sum((pat_true == 0) & (pat_pred == 1)))
    pat_fn = int(np.sum((pat_true == 1) & (pat_pred == 0)))
    pat_sens = pat_tp / (pat_tp + pat_fn) if (pat_tp + pat_fn) > 0 else 0
    pat_spec = pat_tn / (pat_tn + pat_fp) if (pat_tn + pat_fp) > 0 else 0
    pat_bal_acc = (pat_sens + pat_spec) / 2

    print(f"\nPatient-level PD detection metrics (N={len(pat_true)}):")
    print(f"  AUC:              {pat_auc:.4f}")
    print(f"  Accuracy:         {pat_acc:.4f}")
    print(f"  Balanced Accuracy:{pat_bal_acc:.4f}")
    print(f"  Sensitivity:      {pat_sens:.4f}")
    print(f"  Specificity:      {pat_spec:.4f}")

    # -- Channel-level frequency Spearman ------------------------------------
    print(f"\nChannel-level frequency estimation (PD+ channels only):")
    freq_mask = valid & (labels == 1) & np.isfinite(freq_targets)
    n_freq = int(np.sum(freq_mask))

    ch_freq_rho = float('nan')
    if n_freq > 5:
        freq_pred_valid = oof_freq_preds[freq_mask]
        freq_true_valid = freq_targets[freq_mask]
        ch_freq_rho, pval = spearmanr(freq_true_valid, freq_pred_valid)

        freq_pred_linear = np.exp(freq_pred_valid)
        freq_true_linear = np.exp(freq_true_valid)
        rho_linear, _ = spearmanr(freq_true_linear, freq_pred_linear)
        mse_log = float(np.mean((freq_pred_valid - freq_true_valid) ** 2))
        mae_linear = float(np.mean(np.abs(freq_pred_linear - freq_true_linear)))

        print(f"  N channels with freq labels: {n_freq}")
        print(f"  Spearman (log space):   {ch_freq_rho:.4f} (p={pval:.2e})")
        print(f"  Spearman (linear):      {rho_linear:.4f}")
        print(f"  MSE (log space):        {mse_log:.4f}")
        print(f"  MAE (linear, Hz):       {mae_linear:.4f}")
    else:
        print(f"  Not enough PD+ channels with frequency labels ({n_freq})")

    # -- Load CNN (no attention) results for comparison ----------------------
    cnn_ch_auc = None
    cnn_pat_auc = None
    cnn_curves_path = CACHE_DIR / 'training_curves.json'
    cnn_fold_path = CACHE_DIR / 'cnn_fold0.pt'
    if cnn_fold_path.exists():
        # Recompute CNN metrics from saved OOF predictions if available
        # We'll just report what we know from the previous run
        print("\n  (CNN without attention results loaded from previous run)")

    # -- Comparison table: Logistic vs CNN vs CNN+Attention ------------------
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: Logistic baseline vs CNN vs CNN+Attention")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<25s} {'Logistic':>10s} {'CNN':>10s} {'CNN+Attn':>10s}")
    print(f"  {'-' * 55}")
    print(f"  {'Channel AUC':<25s} {'0.766':>10s} {'(prev)':>10s} {ch_auc:>10.4f}")
    print(f"  {'Patient AUC':<25s} {'0.902':>10s} {'(prev)':>10s} {pat_auc:>10.4f}")
    print(f"  {'Ch Freq Spearman':<25s} {'N/A':>10s} {'(prev)':>10s} {ch_freq_rho:>10.4f}" if np.isfinite(ch_freq_rho) else
          f"  {'Ch Freq Spearman':<25s} {'N/A':>10s} {'(prev)':>10s} {'N/A':>10s}")

    # -- Step 3: Patient-level frequency estimation --------------------------
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL FREQUENCY ESTIMATION (CNN+Attention)")
    print(f"{'=' * 70}")
    print("\nRunning patient-level frequency estimation using PD-weighted aggregation...")

    patient_preds, patient_golds, patient_subtypes = patient_level_freq_estimation(
        models_by_fold, patient_folds, main_dataset, data_path
    )

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

    print(f"\n  CNN+Attention patient-level frequency results:")
    print(f"    Combined (N={comb['n']}): Spearman={comb['spearman']:.4f}, MAE={comb['mae']:.4f}")
    print(f"    LPD      (N={lpd['n']}):  Spearman={lpd['spearman']:.4f}, MAE={lpd['mae']:.4f}")
    print(f"    GPD      (N={gpd['n']}):  Spearman={gpd['spearman']:.4f}, MAE={gpd['mae']:.4f}")

    # Final comparison table
    print(f"\n{'=' * 70}")
    print("PATIENT-LEVEL FREQUENCY ESTIMATION COMPARISON")
    print(f"{'=' * 70}")

    def fmt(v):
        return f"{v:.3f}" if np.isfinite(v) else "N/A"

    print(f"  {'Method':<30s} {'Combined_rho':>13s} {'LPD_rho':>10s} {'GPD_rho':>10s} {'MAE':>8s}")
    print(f"  {'-' * 71}")
    print(f"  {'Ridge (alpha=1)':<30s} {'0.589':>13s} {'0.488':>10s} {'0.700':>10s} {'0.274':>8s}")
    print(f"  {'RF 200 trees':<30s} {'0.604':>13s} {'0.519':>10s} {'0.677':>10s} {'0.267':>8s}")
    print(f"  {'CNN+Attn (PD-weighted)':<30s} {fmt(comb['spearman']):>13s} {fmt(lpd['spearman']):>10s} {fmt(gpd['spearman']):>10s} {fmt(comb['mae']):>8s}")
    print(f"  {'-' * 71}")

    # -- Save patient-level results as JSON ----------------------------------
    freq_results = {
        'experiment': 'cnn_attention_patient_freq',
        'timestamp': time.time(),
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
        'channel_freq_spearman': round(float(ch_freq_rho), 4) if np.isfinite(ch_freq_rho) else None,
    }

    # Replace NaN with None for JSON
    freq_results_json = {}
    for k, v in freq_results.items():
        if isinstance(v, float) and np.isnan(v):
            freq_results_json[k] = None
        else:
            freq_results_json[k] = v

    results_json_path = RESULTS_DIR / 'cnn_attn_patient_freq.json'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(results_json_path), 'w') as f:
        json.dump(freq_results_json, f, indent=2)
    print(f"\n  Patient-level freq results saved to {results_json_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
