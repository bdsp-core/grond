"""
Train a 1D CNN for channel-level PD detection with multi-task frequency estimation.

Uses 5-fold patient-stratified CV. Multi-task loss:
  loss = BCE(pd_pred, pd_label) + alpha * MSE(freq_pred, log_freq_target) * pd_mask

Compares against logistic regression baseline (channel AUC 0.766, patient AUC 0.902).
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

from pd_channel_detector.channel_cnn import ChannelPDNet
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('cpu')


# ── Dataset ─────────────────────────────────────────────────────────────

class ChannelPDDataset(Dataset):
    """PyTorch dataset for channel-level PD detection + frequency estimation."""

    def __init__(self, channels, labels, freq_targets, augment=False):
        """
        Args:
            channels: (N, 2000) float32 array
            labels: (N,) binary PD labels
            freq_targets: (N,) log(gold_standard_freq), NaN for PD-negative
            augment: whether to apply data augmentation
        """
        self.channels = channels.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.freq_targets = freq_targets.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.channels[idx].copy()  # (2000,)

        # Per-channel z-score normalization
        mu = np.mean(x)
        std = np.std(x)
        if std > 1e-8:
            x = (x - mu) / std
        else:
            x = x - mu

        if self.augment:
            x = self._augment(x)

        # Shape: (1, 2000) for Conv1d
        x_tensor = torch.from_numpy(x[np.newaxis, :])
        label = torch.tensor(self.labels[idx])
        freq = torch.tensor(self.freq_targets[idx])
        mask = torch.tensor(1.0 if self.labels[idx] > 0.5 and np.isfinite(self.freq_targets[idx]) else 0.0)

        return x_tensor, label, freq, mask

    def _augment(self, x):
        """Apply random data augmentation."""
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


# ── Training loop ────────────────────────────────────────────────────────

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
        pd_prob, freq_pred = model(x)

        pd_prob = pd_prob.squeeze(1)      # (B,)
        freq_pred = freq_pred.squeeze(1)  # (B,)

        # BCE loss for PD detection
        loss_bce = bce(pd_prob, label).mean()

        # MSE loss for frequency (masked to PD+ channels only)
        # Replace NaN freq targets with 0 (they'll be masked anyway)
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

        pd_prob, freq_pred = model(x)
        pd_prob_sq = pd_prob.squeeze(1)
        freq_pred_sq = freq_pred.squeeze(1)

        # Compute loss (same as training)
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

    return float(np.trapz(tpr_list, fpr_list))


# ── Learning Curve HTML Generator ─────────────────────────────────────────

def generate_learning_curves_html(training_curves):
    """Generate an HTML file with interactive learning curve plots using Chart.js."""
    import os
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    html_path = RESULTS_DIR / 'cnn_learning_curves.html'

    fold_keys = sorted(training_curves.keys())
    # Color palette for folds
    colors = [
        'rgba(31, 119, 180, 1)',   # blue
        'rgba(255, 127, 14, 1)',   # orange
        'rgba(44, 160, 44, 1)',    # green
        'rgba(214, 39, 40, 1)',    # red
        'rgba(148, 103, 189, 1)',  # purple
    ]
    colors_alpha = [c.replace(', 1)', ', 0.15)') for c in colors]

    # Build datasets JSON for each chart
    def make_datasets_json(metric_key, label_prefix, use_dash_for_val=False):
        datasets = []
        for i, fold_key in enumerate(fold_keys):
            values = training_curves[fold_key].get(metric_key, [])
            # Replace None with null for JSON
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

    # Max epochs across folds for x-axis labels
    max_epochs = max(len(training_curves[k].get('train_loss', [])) for k in fold_keys)
    epoch_labels = ', '.join(str(e + 1) for e in range(max_epochs))

    val_auc_datasets = make_datasets_json('val_auc', 'Val AUC')
    val_freq_sp_datasets = make_datasets_json('val_freq_spearman', 'Val Freq Spearman')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN Learning Curves - Channel PD Detection</title>
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
    <h1>CNN Learning Curves</h1>
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

        // Loss chart (train + val)
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

        // Val AUC chart
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

        // Val Frequency Spearman chart
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


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("Phase 3: 1D CNN for Channel-Level PD Detection")
    print("=" * 60)

    # ── Load channel dataset ──────────────────────────────────────────
    data_path = CACHE_DIR / 'channel_dataset.npz'
    print(f"\nLoading channel dataset from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    channels = data['channels']        # (N, 2000)
    labels = data['labels']            # (N,)
    patient_ids = data['patient_ids']  # (N,)
    subtypes = data['subtypes']        # (N,)

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    print(f"  Channels: {n_total} (pos={n_pos}, neg={n_neg})")
    print(f"  Patients: {n_patients}")

    # ── Load frequency labels from main dataset ───────────────────────
    print("\nLoading frequency labels from main dataset...")
    main_dataset = load_dataset(verbose=False)
    df_main = main_dataset['df']

    # Build patient_id -> gold_standard_freq mapping
    pid_to_freq = {}
    for _, row in df_main.iterrows():
        pid = str(row['patient_id'])
        freq = row['gold_standard_freq']
        if np.isfinite(freq) and freq > 0:
            pid_to_freq[pid] = freq

    # Build frequency targets: log(freq) for PD+ channels, NaN for PD- channels
    freq_targets = np.full(n_total, np.nan, dtype=np.float32)
    n_with_freq = 0
    for i in range(n_total):
        if labels[i] == 1:
            pid = str(patient_ids[i])
            if pid in pid_to_freq:
                freq_targets[i] = np.log(pid_to_freq[pid])
                n_with_freq += 1

    print(f"  PD+ channels with frequency labels: {n_with_freq}/{n_pos}")

    # ── Create 5-fold patient-stratified splits ───────────────────────
    print("\nCreating 5-fold patient-stratified splits...")
    n_folds = 5
    rng = np.random.RandomState(42)

    # Get subtype per patient (for stratification)
    pid_to_subtype = {}
    for i, pid in enumerate(patient_ids):
        pid = str(pid)
        if pid not in pid_to_subtype:
            pid_to_subtype[pid] = str(subtypes[i])

    # Group patients by subtype
    subtype_groups = {}
    for pid in unique_patients:
        st = pid_to_subtype.get(str(pid), 'unknown')
        if st not in subtype_groups:
            subtype_groups[st] = []
        subtype_groups[st].append(str(pid))

    # Stratified assignment to folds
    patient_folds = {}
    for st, pids in subtype_groups.items():
        pids_shuffled = list(pids)
        rng.shuffle(pids_shuffled)
        for i, pid in enumerate(pids_shuffled):
            patient_folds[pid] = i % n_folds

    # Print fold sizes
    for fold in range(n_folds):
        fold_pids = [p for p, f in patient_folds.items() if f == fold]
        fold_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        n_fold = int(np.sum(fold_mask))
        n_fold_pos = int(np.sum(labels[fold_mask] == 1))
        print(f"  Fold {fold}: {len(fold_pids)} patients, {n_fold} channels (pos={n_fold_pos})")

    # ── Training loop across folds ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training CNN across 5 folds...")
    print("=" * 60)

    # Collect out-of-fold predictions
    oof_pd_probs = np.full(n_total, np.nan)
    oof_freq_preds = np.full(n_total, np.nan)

    alpha = 0.5   # frequency loss weight
    n_epochs = 30
    batch_size = 128
    lr = 1e-3
    patience = 5

    # Per-epoch training curves for all folds
    training_curves = {}

    for fold in range(n_folds):
        fold_t0 = time.time()
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")

        # Initialize per-epoch curve tracking for this fold
        fold_curves = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_freq_spearman': [],
        }

        # Split into train/val
        val_mask = np.array([patient_folds.get(str(p), -1) == fold for p in patient_ids])
        train_mask = ~val_mask

        train_channels = channels[train_mask]
        train_labels = labels[train_mask]
        train_freqs = freq_targets[train_mask]
        val_channels = channels[val_mask]
        val_labels = labels[val_mask]
        val_freqs = freq_targets[val_mask]

        print(f"  Train: {len(train_labels)} channels, Val: {len(val_labels)} channels")

        # Create datasets and loaders
        train_ds = ChannelPDDataset(train_channels, train_labels, train_freqs, augment=True)
        val_ds = ChannelPDDataset(val_channels, val_labels, val_freqs, augment=False)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create model
        model = ChannelPDNet().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        # Training with early stopping
        best_val_auc = -1.0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, alpha=alpha)

            # Evaluate on validation set
            val_results = evaluate(model, val_loader, alpha=alpha)
            val_auc = compute_auc(val_results['labels'], val_results['pd_probs'])
            val_loss = val_results['avg_loss']
            val_freq_sp = compute_freq_spearman(val_results)

            # Record per-epoch metrics
            fold_curves['train_loss'].append(round(train_loss, 6))
            fold_curves['val_loss'].append(round(val_loss, 6))
            fold_curves['val_auc'].append(round(val_auc, 6) if np.isfinite(val_auc) else None)
            fold_curves['val_freq_spearman'].append(round(val_freq_sp, 6) if np.isfinite(val_freq_sp) else None)

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

        # Store this fold's curves
        training_curves[f'fold_{fold}'] = fold_curves

        # Load best model and get final val predictions
        if best_state is not None:
            model.load_state_dict(best_state)

        val_results = evaluate(model, val_loader, alpha=alpha)
        oof_pd_probs[val_mask] = val_results['pd_probs']
        oof_freq_preds[val_mask] = val_results['freq_preds']

        # Save fold model
        save_path = CACHE_DIR / f'cnn_fold{fold}.pt'
        torch.save(best_state if best_state is not None else model.state_dict(), str(save_path))

        fold_elapsed = time.time() - fold_t0
        print(f"  Fold {fold + 1} done: best_val_AUC={best_val_auc:.4f} ({fold_elapsed:.1f}s)")
        print(f"  Model saved to {save_path}")

    # ── Save training curves ─────────────────────────────────────────
    curves_path = CACHE_DIR / 'training_curves.json'
    with open(str(curves_path), 'w') as f:
        json.dump(training_curves, f, indent=2)
    print(f"\nTraining curves saved to {curves_path}")

    # ── Generate HTML learning curve viewer ───────────────────────────
    generate_learning_curves_html(training_curves)

    # ── Compute overall metrics ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUT-OF-FOLD RESULTS")
    print("=" * 60)

    # Filter to channels that got predictions
    valid = np.isfinite(oof_pd_probs)
    y_true = labels[valid].astype(int)
    y_prob = oof_pd_probs[valid]
    y_pred = (y_prob >= 0.5).astype(int)
    pids_valid = patient_ids[valid]

    # Channel-level metrics
    ch_auc = compute_auc(y_true, y_prob)
    ch_acc = float(np.mean(y_true == y_pred))
    # Balanced accuracy
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

    # Patient-level aggregation
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

    print(f"\nPatient-level metrics (N={len(pat_true)}):")
    print(f"  AUC:              {pat_auc:.4f}")
    print(f"  Accuracy:         {pat_acc:.4f}")
    print(f"  Balanced Accuracy:{pat_bal_acc:.4f}")
    print(f"  Sensitivity:      {pat_sens:.4f}")
    print(f"  Specificity:      {pat_spec:.4f}")

    # ── Comparison with baseline ──────────────────────────────────────
    print(f"\n{'─' * 40}")
    print(f"Comparison with logistic regression baseline:")
    print(f"  {'Metric':<25s} {'Baseline':>10s} {'CNN':>10s}")
    print(f"  {'Channel AUC':<25s} {'0.766':>10s} {ch_auc:>10.4f}")
    print(f"  {'Patient AUC':<25s} {'0.902':>10s} {pat_auc:>10.4f}")
    print(f"{'─' * 40}")

    # ── Frequency prediction (on PD+ channels) ───────────────────────
    print(f"\nFrequency estimation (PD+ channels only):")
    freq_mask = valid & (labels == 1) & np.isfinite(freq_targets)
    n_freq = int(np.sum(freq_mask))

    if n_freq > 5:
        freq_pred_valid = oof_freq_preds[freq_mask]
        freq_true_valid = freq_targets[freq_mask]

        # Spearman on log-space predictions vs log-space targets
        rho, pval = spearmanr(freq_true_valid, freq_pred_valid)

        # Also compute in linear space
        freq_pred_linear = np.exp(freq_pred_valid)
        freq_true_linear = np.exp(freq_true_valid)
        rho_linear, pval_linear = spearmanr(freq_true_linear, freq_pred_linear)

        # MSE in log space
        mse_log = float(np.mean((freq_pred_valid - freq_true_valid) ** 2))
        # MAE in linear space
        mae_linear = float(np.mean(np.abs(freq_pred_linear - freq_true_linear)))

        print(f"  N channels with freq labels: {n_freq}")
        print(f"  Spearman (log space):   {rho:.4f} (p={pval:.2e})")
        print(f"  Spearman (linear):      {rho_linear:.4f} (p={pval_linear:.2e})")
        print(f"  MSE (log space):        {mse_log:.4f}")
        print(f"  MAE (linear, Hz):       {mae_linear:.4f}")
    else:
        print(f"  Not enough PD+ channels with frequency labels ({n_freq})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
