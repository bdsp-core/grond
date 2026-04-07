"""
Domain Adversarial Neural Network (DANN) for purifying CNN embeddings.

Problem: CNN backbone captures patient identity rather than generalizable discharge features.
Solution: Gradient reversal layer forces backbone to learn features useful for classification
          but NOT useful for identifying the patient.

Architecture:
    Input (batch, 18, 2000) → EEGBackbone → (batch, 128, 125) → AdaptiveAvgPool1d(1) → (batch, 128)
        ├── ClassificationHead → LPD/GPD (BCE loss)
        └── GradientReversal → PatientIDHead → patient_id (CE loss)

Run: conda run -n foe_dl python code/dl/train_dann.py
"""

import sys
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings('ignore')

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from model import EEGBackbone, ClassificationHead
from data_loader import normalize_segment
from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       'mps' if torch.backends.mps.is_available() else 'cpu')

# ── Gradient Reversal Layer ───────────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ── DANN Model ────────────────────────────────────────────────────────────

class DANNModel(nn.Module):
    """Domain Adversarial Neural Network with classification + patient adversarial heads."""

    def __init__(self, n_patients, in_channels=18, dropout=0.1):
        super().__init__()
        self.backbone = EEGBackbone(in_channels, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head (LPD vs GPD)
        self.class_head = nn.Linear(128, 1)

        # Patient adversarial head (through gradient reversal)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.patient_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_patients),
        )

    def set_grl_alpha(self, alpha):
        self.grl.alpha = alpha

    def forward(self, x):
        features = self.backbone(x)         # (B, 128, 125)
        pooled = self.pool(features).squeeze(-1)  # (B, 128)

        class_logits = self.class_head(pooled)    # (B, 1)

        reversed_features = self.grl(pooled)
        patient_logits = self.patient_head(reversed_features)  # (B, n_patients)

        return class_logits, patient_logits, pooled


# ── Dataset ───────────────────────────────────────────────────────────────

class DANNDataset(Dataset):
    """Dataset returning (segment, label, patient_idx) for DANN training."""

    def __init__(self, segments, labels, patient_indices, augment=False):
        self.segments = segments
        self.labels = labels
        self.patient_indices = patient_indices
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx].copy()
        label = self.labels[idx]
        patient_idx = self.patient_indices[idx]

        if self.augment:
            # Gaussian noise
            if np.random.random() < 0.5:
                noise_scale = 0.1 * np.std(seg, axis=1, keepdims=True)
                noise_scale[noise_scale < 1e-6] = 0.01
                seg = seg + np.random.randn(*seg.shape).astype(np.float32) * noise_scale
            # Amplitude scaling
            if np.random.random() < 0.5:
                scales = np.random.uniform(0.8, 1.2, (18, 1)).astype(np.float32)
                seg = seg * scales
            # Channel dropout
            if np.random.random() < 0.2:
                n_drop = np.random.randint(1, 3)
                drop_idx = np.random.choice(18, n_drop, replace=False)
                seg[drop_idx] = 0.0

        seg = normalize_segment(seg)

        return (torch.from_numpy(seg),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(patient_idx, dtype=torch.long))


# ── SP Feature Extraction (for evaluation) ───────────────────────────────

def extract_sp_features(dataset_entries):
    """Extract 8 signal processing features for each annotated segment."""
    from pd_pointiness_acf import (
        fcn_getBanana, compute_pointiness_trace, pd_detect_pointiness_acf,
    )
    from pd_detect_alternate import pd_detect_alternate
    from scipy.signal import butter, filtfilt, find_peaks, coherence
    from scipy.ndimage import gaussian_filter1d
    from mne.filter import notch_filter, filter_data

    LOWPASS_HZ = 15.0
    SMOOTHING_SIGMA = 0.02
    ACF_MIN_LAG = 0.4
    ACF_THRESHOLD = 0.10
    PEAK_HEIGHT_FRAC = 0.3
    FS = 200
    ADJACENT_PAIRS = [
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (12, 13), (13, 14), (14, 15),
        (16, 17),
    ]

    def median_finite(arr):
        valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
            np.array([x for x in arr if np.isfinite(x)])
        return float(np.median(valid)) if len(valid) > 0 else np.nan

    N = len(dataset_entries)
    FEAT_NAMES = ['f_A', 'f_B', 'f_peaks', 'f_fft', 'f_tkeo_fft',
                  'f_spectral_coh', 'f_hps3', 'is_gpd']
    features = np.full((N, len(FEAT_NAMES)), np.nan)

    t0 = time.time()
    for idx, entry in enumerate(dataset_entries):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"    SP features: {idx+1}/{N} ({elapsed:.0f}s)")

        try:
            data, fs = load_eeg_data(entry)
            if data is None:
                continue

            # Method A
            try:
                r = pd_detect_alternate(data, fs, pk_detect='apd')
                f = r.get('event_frequency', np.nan)
                features[idx, 0] = float(f) if f is not None and np.isfinite(float(f)) else np.nan
            except Exception:
                pass

            # Method B
            try:
                r = pd_detect_pointiness_acf(
                    data, fs, method='pointiness',
                    acf_min_lag=ACF_MIN_LAG, acf_peak_threshold=ACF_THRESHOLD,
                    smoothing_sigma=SMOOTHING_SIGMA, lowpass_hz=LOWPASS_HZ,
                    peak_height_frac=PEAK_HEIGHT_FRAC)
                f = r.get('event_frequency', np.nan)
                features[idx, 1] = float(f) if f is not None and np.isfinite(float(f)) else np.nan
            except Exception:
                pass

            # Preprocess for remaining methods
            seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
            seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg_bb = np.array(fcn_getBanana(seg))  # broadband bipolar
            b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
            seg_lp = seg_bb.copy()
            for i in range(seg_lp.shape[0]):
                try:
                    seg_lp[i] = filtfilt(b_lp, a_lp, seg_lp[i])
                except ValueError:
                    pass

            # Pointiness traces
            sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
            traces = []
            for i in range(seg_lp.shape[0]):
                trace = compute_pointiness_trace(seg_lp[i])
                trace = gaussian_filter1d(trace, sigma=sigma_samples)
                traces.append(trace)
            traces = np.array(traces)

            # f_peaks
            n_ch = traces.shape[0]
            freqs_pk = np.full(n_ch, np.nan)
            min_distance = int(0.2 * fs)
            for i in range(n_ch):
                trace = traces[i]
                if np.max(trace) <= 0:
                    continue
                peak_locs, _ = find_peaks(trace, height=np.max(trace) * PEAK_HEIGHT_FRAC,
                                           distance=min_distance)
                if len(peak_locs) >= 3:
                    freqs_pk[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
            features[idx, 2] = median_finite(freqs_pk)

            # f_fft
            freqs_fft = np.full(n_ch, np.nan)
            for i in range(n_ch):
                trace = traces[i]
                if np.max(trace) <= 0:
                    continue
                n = len(trace)
                fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
                fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
                if np.any(mask):
                    freqs_fft[i] = fft_freqs[mask][np.argmax(fft_vals[mask])]
            features[idx, 3] = median_finite(freqs_fft)

            # f_tkeo_fft
            freqs_tkeo = np.full(n_ch, np.nan)
            for i in range(n_ch):
                x = seg_lp[i]
                if len(x) < 3:
                    continue
                tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
                tkeo = np.abs(tkeo)
                tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
                n = len(tkeo)
                if n < 10:
                    continue
                fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
                fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
                if np.any(mask):
                    freqs_tkeo[i] = fft_freqs[mask][np.argmax(fft_vals[mask])]
            features[idx, 4] = median_finite(freqs_tkeo)

            # f_spectral_coh
            nperseg = min(256, seg_bb.shape[1] // 2)
            if nperseg >= 16:
                coh_spectra = None
                coh_freqs = None
                for (a, b) in ADJACENT_PAIRS:
                    if a >= seg_bb.shape[0] or b >= seg_bb.shape[0]:
                        continue
                    try:
                        f_coh, Cxy = coherence(seg_bb[a], seg_bb[b], fs=fs, nperseg=nperseg)
                        if coh_freqs is None:
                            coh_freqs = f_coh
                            coh_spectra = np.zeros_like(f_coh)
                        coh_spectra += Cxy
                    except Exception:
                        continue
                if coh_freqs is not None:
                    coh_spectra /= len(ADJACENT_PAIRS)
                    mask = (coh_freqs >= 0.3) & (coh_freqs <= 3.5)
                    if np.any(mask):
                        features[idx, 5] = float(coh_freqs[mask][np.argmax(coh_spectra[mask])])

            # f_hps3
            freqs_hps = np.full(n_ch, np.nan)
            for i in range(n_ch):
                trace = traces[i]
                if np.max(trace) <= 0:
                    continue
                n = len(trace)
                fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
                fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
                max_idx = len(fft_vals) // 3
                if max_idx < 2:
                    continue
                hps = fft_vals[:max_idx].copy()
                hps *= fft_vals[:max_idx * 2:2][:max_idx]
                hps *= fft_vals[:max_idx * 3:3][:max_idx]
                hps_freqs = fft_freqs[:max_idx]
                mask = (hps_freqs >= 0.3) & (hps_freqs <= 3.5)
                if np.any(mask):
                    freqs_hps[i] = hps_freqs[mask][np.argmax(hps[mask])]
            features[idx, 6] = median_finite(freqs_hps)

            # is_gpd
            features[idx, 7] = 1.0 if entry['subdir'] == 'gpd' else 0.0

        except Exception:
            continue

    elapsed = time.time() - t0
    print(f"  SP feature extraction done in {elapsed:.0f}s")
    return features, FEAT_NAMES


def impute_nan_median(X):
    X_out = X.copy()
    for fi in range(X_out.shape[1]):
        col = X_out[:, fi]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            med = np.nanmedian(col)
            if not np.isfinite(med):
                med = 1.0
            col[nan_mask] = med
            X_out[:, fi] = col
    return X_out


# ── LOPO per-expert ridge ────────────────────────────────────────────────

def lopo_per_expert_ridge(X, dataset_entries, experiment_name, alpha=1.0):
    """Leave-one-patient-out per-expert ridge regression.
    Trains 3 separate models (one per expert), averages predictions.
    """
    N = len(dataset_entries)
    expert_keys = ['expert_LB_freq', 'expert_PH_freq', 'expert_SZ_freq']

    # Get patient IDs
    patients = []
    for entry in dataset_entries:
        # Extract patient from mat_name
        mat_name = entry['mat_name']
        patients.append(mat_name.split('_')[0] if '_' in mat_name else mat_name)

    patients = np.array(patients)
    unique_patients = np.unique(patients)
    print(f"  LOPO: {len(unique_patients)} patients, {N} segments")

    # For each expert, run LOPO
    expert_predictions = np.full((3, N), np.nan)

    for ei, ekey in enumerate(expert_keys):
        # Get valid targets for this expert
        targets = np.array([entry.get(ekey, np.nan) for entry in dataset_entries])
        valid = np.isfinite(targets) & (targets > 0)

        if np.sum(valid) < 10:
            print(f"    Expert {ekey}: too few valid ({np.sum(valid)}), skipping")
            continue

        log_targets = np.log(targets)

        for pi, pat in enumerate(unique_patients):
            test_mask = patients == pat
            train_mask = ~test_mask & valid

            if np.sum(train_mask) < 5 or not np.any(test_mask & valid):
                continue

            X_train = X[train_mask]
            y_train = log_targets[train_mask]
            X_test = X[test_mask]

            # Standardize
            mu = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            std[std == 0] = 1.0
            X_tr_s = (X_train - mu) / std
            X_te_s = (X_test - mu) / std

            y_mu = np.mean(y_train)
            y_c = y_train - y_mu

            p = X_tr_s.shape[1]
            A = X_tr_s.T @ X_tr_s + alpha * np.eye(p)
            try:
                w = np.linalg.solve(A, X_tr_s.T @ y_c)
            except np.linalg.LinAlgError:
                continue

            preds = X_te_s @ w + y_mu
            expert_predictions[ei, test_mask] = preds

    # Average across experts (where available)
    final_predictions = np.full(N, np.nan)
    for i in range(N):
        expert_preds_i = expert_predictions[:, i]
        valid_preds = expert_preds_i[np.isfinite(expert_preds_i)]
        if len(valid_preds) > 0:
            final_predictions[i] = np.exp(np.mean(valid_preds))

    # Build predictions dict
    pred_dict = {}
    for i, entry in enumerate(dataset_entries):
        if np.isfinite(final_predictions[i]):
            pred_dict[entry['mat_name']] = float(final_predictions[i])

    print(f"  LOPO predictions: {len(pred_dict)}/{N}")

    # Evaluate
    dataset = load_dataset()
    metrics = evaluate_predictions(dataset, pred_dict, experiment_name)
    return metrics


# ── Main Training ─────────────────────────────────────────────────────────

def train_dann():
    print("=" * 70)
    print("DANN: Domain Adversarial Neural Network Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\n[1] Loading external classification segments")
    ext = np.load(str(CACHE_DIR / 'external_pd_segments.npz'), allow_pickle=True)
    segments = ext['segments']       # (3816, 18, 2000)
    labels = ext['labels']           # (3816,) 0=LPD, 1=GPD
    patients = ext['patients']       # (3816,) patient IDs

    N = len(segments)
    unique_patients = np.unique(patients)
    n_patients = len(unique_patients)
    print(f"  {N} segments, {n_patients} unique patients")

    # Map patient IDs to integer indices
    patient_to_idx = {p: i for i, p in enumerate(unique_patients)}
    patient_indices = np.array([patient_to_idx[p] for p in patients])

    # ── Patient-level 80/20 split ─────────────────────────────────────
    np.random.seed(42)
    perm = np.random.permutation(n_patients)
    n_train_patients = int(0.8 * n_patients)
    train_patient_ids = set(unique_patients[perm[:n_train_patients]])
    val_patient_ids = set(unique_patients[perm[n_train_patients:]])

    train_mask = np.array([p in train_patient_ids for p in patients])
    val_mask = np.array([p in val_patient_ids for p in patients])

    print(f"  Train: {np.sum(train_mask)} segments ({len(train_patient_ids)} patients)")
    print(f"  Val:   {np.sum(val_mask)} segments ({len(val_patient_ids)} patients)")

    # Create datasets
    train_ds = DANNDataset(segments[train_mask], labels[train_mask],
                           patient_indices[train_mask], augment=True)
    val_ds = DANNDataset(segments[val_mask], labels[val_mask],
                         patient_indices[val_mask], augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # ── Build model ───────────────────────────────────────────────────
    print("\n[2] Building DANN model")
    model = DANNModel(n_patients=n_patients, in_channels=18, dropout=0.1).to(DEVICE)

    # Load pretrained backbone
    ckpt = torch.load(str(CACHE_DIR / 'classifier_best.pt'), map_location=DEVICE)
    backbone_dict = {}
    for k, v in ckpt.items():
        if k.startswith('backbone.'):
            backbone_dict[k.replace('backbone.', '')] = v
    model.backbone.load_state_dict(backbone_dict, strict=True)
    print(f"  Loaded {len(backbone_dict)} backbone parameters from pretrained classifier")

    # Also initialize classification head from pretrained
    if 'head.fc.weight' in ckpt:
        model.class_head.weight.data.copy_(ckpt['head.fc.weight'])
        model.class_head.bias.data.copy_(ckpt['head.fc.bias'])
        print("  Loaded classification head from pretrained classifier")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    class_criterion = nn.BCEWithLogitsLoss()
    patient_criterion = nn.CrossEntropyLoss()

    MAX_EPOCHS = 30
    best_val_loss = float('inf')
    best_state = None

    # ── Training loop ─────────────────────────────────────────────────
    print("\n[3] Training DANN")
    print(f"{'Ep':>3s} {'Lambda':>6s} {'CLoss':>7s} {'PLoss':>7s} {'Total':>7s} "
          f"{'VCLoss':>7s} {'VPLoss':>7s} {'ClAcc':>6s} {'PtAcc':>6s}")
    print("-" * 75)

    for epoch in range(MAX_EPOCHS):
        # Lambda schedule: ramps from 0 to 1 over training
        p = epoch / MAX_EPOCHS
        lam = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        model.set_grl_alpha(lam)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        epoch_class_loss = 0.0
        epoch_patient_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for seg_batch, label_batch, patient_batch in train_loader:
            seg_batch = seg_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            patient_batch = patient_batch.to(DEVICE)

            class_logits, patient_logits, _ = model(seg_batch)

            c_loss = class_criterion(class_logits.squeeze(-1), label_batch)
            p_loss = patient_criterion(patient_logits, patient_batch)
            total_loss = c_loss + lam * p_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_class_loss += c_loss.item()
            epoch_patient_loss += p_loss.item()
            epoch_total_loss += total_loss.item()
            n_batches += 1

        scheduler.step()

        avg_c = epoch_class_loss / max(n_batches, 1)
        avg_p = epoch_patient_loss / max(n_batches, 1)
        avg_t = epoch_total_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        val_class_loss = 0.0
        val_patient_loss = 0.0
        val_class_correct = 0
        val_patient_correct = 0
        val_total = 0
        val_batches = 0

        with torch.no_grad():
            for seg_batch, label_batch, patient_batch in val_loader:
                seg_batch = seg_batch.to(DEVICE)
                label_batch = label_batch.to(DEVICE)
                patient_batch = patient_batch.to(DEVICE)

                class_logits, patient_logits, _ = model(seg_batch)

                c_loss = class_criterion(class_logits.squeeze(-1), label_batch)
                p_loss = patient_criterion(patient_logits, patient_batch)

                val_class_loss += c_loss.item()
                val_patient_loss += p_loss.item()
                val_batches += 1

                # Accuracy
                class_pred = (torch.sigmoid(class_logits.squeeze(-1)) > 0.5).float()
                val_class_correct += (class_pred == label_batch).sum().item()
                patient_pred = patient_logits.argmax(dim=1)
                val_patient_correct += (patient_pred == patient_batch).sum().item()
                val_total += len(label_batch)

        vc = val_class_loss / max(val_batches, 1)
        vp = val_patient_loss / max(val_batches, 1)
        cl_acc = val_class_correct / max(val_total, 1) * 100
        pt_acc = val_patient_correct / max(val_total, 1) * 100

        print(f"{epoch+1:>3d} {lam:>6.3f} {avg_c:>7.4f} {avg_p:>7.4f} {avg_t:>7.4f} "
              f"{vc:>7.4f} {vp:>7.4f} {cl_acc:>5.1f}% {pt_acc:>5.1f}%")

        # Save best based on classification loss (we want good classification
        # with low patient discriminability)
        val_combined = vc
        if val_combined < best_val_loss:
            best_val_loss = val_combined
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save DANN backbone ────────────────────────────────────────────
    print(f"\n[4] Saving DANN backbone")
    dann_path = CACHE_DIR / 'dann_backbone.pt'

    # Save just the backbone state dict
    backbone_state = {}
    for k, v in best_state.items():
        if k.startswith('backbone.'):
            backbone_state[k] = v

    torch.save(backbone_state, str(dann_path))
    print(f"  Saved to {dann_path} ({len(backbone_state)} parameters)")

    return model, best_state


def evaluate_dann():
    """Extract DANN embeddings and evaluate with LOPO ridge and k-NN."""
    print("\n" + "=" * 70)
    print("DANN Evaluation: Embeddings + LOPO Ridge + k-NN")
    print("=" * 70)

    # ── Load DANN backbone ────────────────────────────────────────────
    print("\n[5] Loading DANN backbone")
    dann_path = CACHE_DIR / 'dann_backbone.pt'
    if not dann_path.exists():
        print(f"  ERROR: {dann_path} not found")
        return

    backbone = EEGBackbone(in_channels=18, dropout=0.0).to(DEVICE)
    state = torch.load(str(dann_path), map_location='cpu')
    # Keys are like 'backbone.block1...' — strip prefix
    backbone_dict = {}
    for k, v in state.items():
        key = k.replace('backbone.', '') if k.startswith('backbone.') else k
        backbone_dict[key] = v
    backbone.load_state_dict(backbone_dict, strict=True)
    backbone.eval()
    print("  Loaded DANN backbone")

    # Also load vanilla (non-DANN) backbone for comparison
    print("  Loading vanilla backbone for comparison")
    vanilla_backbone = EEGBackbone(in_channels=18, dropout=0.0).to(DEVICE)
    ckpt = torch.load(str(CACHE_DIR / 'classifier_best.pt'), map_location='cpu')
    vanilla_dict = {}
    for k, v in ckpt.items():
        if k.startswith('backbone.'):
            vanilla_dict[k.replace('backbone.', '')] = v
    vanilla_backbone.load_state_dict(vanilla_dict, strict=True)
    vanilla_backbone.eval()

    # ── Load annotated segments ───────────────────────────────────────
    print("\n[6] Extracting embeddings for 556 annotated segments")
    ann = np.load(str(CACHE_DIR / 'annotated_pd_data.npz'), allow_pickle=True)
    ann_segments = ann['segments']     # (556, 18, 2000)
    ann_patients = ann['patients']     # (556,)
    ann_subtypes = ann['subtypes']     # (556,)

    N = len(ann_segments)
    pool = nn.AdaptiveAvgPool1d(1)

    # Extract DANN embeddings
    dann_embeddings = np.zeros((N, 128), dtype=np.float32)
    vanilla_embeddings = np.zeros((N, 128), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, 64):
            batch = ann_segments[i:i+64].copy()
            # Normalize
            for j in range(len(batch)):
                batch[j] = normalize_segment(batch[j])
            batch_t = torch.from_numpy(batch).to(DEVICE)

            # DANN embeddings
            feats = backbone(batch_t)  # (B, 128, 125)
            pooled = pool(feats).squeeze(-1)  # (B, 128)
            dann_embeddings[i:i+len(batch)] = pooled.cpu().numpy()

            # Vanilla embeddings
            feats_v = vanilla_backbone(batch_t)
            pooled_v = pool(feats_v).squeeze(-1)
            vanilla_embeddings[i:i+len(batch)] = pooled_v.cpu().numpy()

    print(f"  DANN embeddings: {dann_embeddings.shape}")
    print(f"  Vanilla embeddings: {vanilla_embeddings.shape}")

    # ── PCA to 20 dims ────────────────────────────────────────────────
    print("\n[7] PCA reduction to 20 dims")
    pca_dann = PCA(n_components=20, random_state=42)
    dann_pca = pca_dann.fit_transform(dann_embeddings)
    print(f"  DANN PCA variance explained: {pca_dann.explained_variance_ratio_.sum():.3f}")

    pca_vanilla = PCA(n_components=20, random_state=42)
    vanilla_pca = pca_vanilla.fit_transform(vanilla_embeddings)
    print(f"  Vanilla PCA variance explained: {pca_vanilla.explained_variance_ratio_.sum():.3f}")

    # ── Load dataset for evaluation ───────────────────────────────────
    print("\n[8] Loading dataset and extracting SP features")
    dataset = load_dataset()

    # Map annotated segments to dataset entries
    # Build a mapping from mat_name to dataset entry
    dataset_by_name = {e['mat_name']: e for e in dataset}

    # We need to figure out which annotated segments correspond to which dataset entries
    # The annotated_pd_data.npz doesn't have mat_names, so we need to reconstruct
    # Load the phase2 predictions which has the mapping
    pred_data = np.load(str(CACHE_DIR / 'phase2_predictions.npz'), allow_pickle=True)
    pred_mat_names = pred_data['mat_names']  # (556,)

    # Build ordered dataset entries matching annotated segments
    ordered_entries = []
    valid_indices = []
    for i in range(N):
        mat_name = str(pred_mat_names[i])
        entry = dataset_by_name.get(mat_name)
        if entry is not None:
            ordered_entries.append(entry)
            valid_indices.append(i)
        else:
            ordered_entries.append({
                'mat_name': mat_name,
                'subdir': str(ann_subtypes[i]),
                'expert_consensus_freq': np.nan,
                'expert_LB_freq': np.nan,
                'expert_PH_freq': np.nan,
                'expert_SZ_freq': np.nan,
            })
            valid_indices.append(i)

    # Extract SP features
    sp_features, sp_feat_names = extract_sp_features(ordered_entries)
    sp_features = impute_nan_median(sp_features)

    # ── Combine features ──────────────────────────────────────────────
    print("\n[9] Running LOPO per-expert ridge evaluations")

    # DANN embeddings (20 PCA) + 8 SP features = 28 features
    X_dann = np.hstack([dann_pca, sp_features])
    X_dann = impute_nan_median(X_dann)
    print(f"  DANN + SP features: {X_dann.shape}")

    # Vanilla embeddings (20 PCA) + 8 SP features = 28 features
    X_vanilla = np.hstack([vanilla_pca, sp_features])
    X_vanilla = impute_nan_median(X_vanilla)
    print(f"  Vanilla + SP features: {X_vanilla.shape}")

    # SP only (baseline)
    X_sp_only = sp_features.copy()
    print(f"  SP-only features: {X_sp_only.shape}")

    # ── Run evaluations ───────────────────────────────────────────────
    print("\n--- DANN embeddings + SP (LOPO) ---")
    dann_metrics = lopo_per_expert_ridge(X_dann, ordered_entries, 'r9_dann_embeddings_lopo')

    print("\n--- Vanilla embeddings + SP (LOPO) ---")
    vanilla_metrics = lopo_per_expert_ridge(X_vanilla, ordered_entries, 'r9_vanilla_embeddings_lopo')

    print("\n--- SP only (LOPO baseline) ---")
    sp_metrics = lopo_per_expert_ridge(X_sp_only, ordered_entries, 'r9_sp_only_lopo')

    # ── k-NN with DANN embeddings ─────────────────────────────────────
    print("\n[10] Running k-NN with DANN embeddings")

    # Get valid entries with consensus freq
    expert_consensus = np.array([e.get('expert_consensus_freq', np.nan) for e in ordered_entries])
    valid_mask = np.isfinite(expert_consensus) & (expert_consensus > 0)
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) >= 10:
        # LOPO k-NN
        patients_arr = np.array([e.get('mat_name', '').split('_')[0]
                                  if '_' in e.get('mat_name', '') else e.get('mat_name', '')
                                  for e in ordered_entries])
        unique_patients = np.unique(patients_arr[valid_idx])

        knn_preds = np.full(N, np.nan)
        for pat in unique_patients:
            test_mask = (patients_arr == pat) & valid_mask
            train_mask = (patients_arr != pat) & valid_mask

            if np.sum(train_mask) < 3 or np.sum(test_mask) < 1:
                continue

            X_train = dann_pca[train_mask]
            y_train = expert_consensus[train_mask]
            X_test = dann_pca[test_mask]

            # Standardize
            mu = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            std[std == 0] = 1.0
            X_tr_s = (X_train - mu) / std
            X_te_s = (X_test - mu) / std

            knn = KNeighborsRegressor(n_neighbors=min(7, len(X_train)),
                                       weights='distance', metric='euclidean')
            knn.fit(X_tr_s, y_train)
            preds = knn.predict(X_te_s)
            knn_preds[test_mask] = preds

        knn_dict = {}
        for i in range(N):
            if np.isfinite(knn_preds[i]):
                knn_dict[ordered_entries[i]['mat_name']] = float(knn_preds[i])

        print(f"  k-NN predictions: {len(knn_dict)}")
        knn_metrics = evaluate_predictions(dataset, knn_dict, 'r9_dann_knn')
    else:
        print("  Too few valid samples for k-NN")
        knn_metrics = {}

    # ── Comparison ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Method':>35s} {'LPD Sp':>8s} {'GPD Sp':>8s} {'Comb Sp':>8s} {'LPD MAE':>8s} {'GPD MAE':>8s}"
    print(header)
    print("-" * len(header))

    for name, m in [("SP only (LOPO)", sp_metrics),
                    ("Vanilla emb + SP (LOPO)", vanilla_metrics),
                    ("DANN emb + SP (LOPO)", dann_metrics),
                    ("DANN k-NN", knn_metrics)]:
        lpd_sp = m.get('lpd_spearman_r', '?')
        gpd_sp = m.get('gpd_spearman_r', '?')
        comb = m.get('combined_spearman', '?')
        lpd_mae = m.get('lpd_mae', '?')
        gpd_mae = m.get('gpd_mae', '?')
        print(f"{name:>35s} {lpd_sp:>8} {gpd_sp:>8} {comb:>8} {lpd_mae:>8} {gpd_mae:>8}")

    return dann_metrics, vanilla_metrics, sp_metrics, knn_metrics


def main():
    t0 = time.time()

    # Phase 1: Train DANN
    model, best_state = train_dann()

    # Phase 2: Evaluate
    dann_metrics, vanilla_metrics, sp_metrics, knn_metrics = evaluate_dann()

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Update dashboard
    print("\n[11] Updating dashboard")
    try:
        from update_dashboard import update
        update()
    except Exception as e:
        print(f"  Dashboard update failed: {e}")
        print("  Run manually: python code/update_dashboard.py")


if __name__ == '__main__':
    main()
