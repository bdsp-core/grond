"""
Round 9: Leave-One-Patient-Out (LOPO) evaluation experiments.

Experiments:
1. r9_lopo_sp_only      - LOPO with 8 SP features (baseline)
2. r9_lopo_strict_gpd   - LOPO excluding abn2147 entirely
3. r9_lopo_quick_wins   - LOPO with 10 features (8 + ratio_fft_to_peaks + eigenvalue_spread)
4. r9_knn_cross_patient - Cross-patient k-NN with CNN embeddings
5. r9_lopo_knn_feature  - LOPO ridge with k-NN prediction as 9th feature

Run: conda run -n foe_dl python code/r9_lopo_experiments.py
"""

import sys
import os
import re
import time
import warnings
import numpy as np
import torch
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
DL_DIR = CODE_DIR / 'dl'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate, fcn_getBanana, bipolar_channels, mono_channels
from pd_pointiness_acf import pd_detect_pointiness_acf, compute_pointiness_trace
from mne.filter import notch_filter, filter_data

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
FREQ_LO, FREQ_HI = 0.3, 3.5

# Adjacent channel pairs for spectral coherence
ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


def extract_patient_id(mat_name):
    """Extract patient ID from mat_name like 'pat0103_20180322_...'."""
    # Patient ID is everything before the date pattern _YYYYMMDD_
    m = re.match(r'^([a-zA-Z]+\d+)_', mat_name)
    if m:
        return m.group(1)
    return mat_name.split('_')[0]


def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
    """FFT of a 1D trace, return peak frequency in [freq_lo, freq_hi] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freqs_sub = freqs[mask]
    if np.max(fft_sub) == 0:
        return np.nan
    return freqs_sub[np.argmax(fft_sub)]


def compute_sp_features(data, fs, entry):
    """Compute the 8 SP features + 2 new quick-win features for one segment."""
    features = {}
    is_gpd = 1 if entry['subdir'] == 'gpd' else 0
    features['is_gpd'] = is_gpd

    # --- Method B: pd_detect_pointiness_acf (ACF thr=0.10) ---
    try:
        res_B = pd_detect_pointiness_acf(
            data.copy(), fs,
            method='pointiness', lowpass_hz=15,
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        f_B = res_B['event_frequency']
        if not np.isfinite(f_B):
            f_B = np.nan
        detected_channels = res_B.get('channels', [])
        if detected_channels is None or (isinstance(detected_channels, float) and np.isnan(detected_channels)):
            detected_channels = []
        n_detected = len(detected_channels)
    except:
        f_B = np.nan
        n_detected = 0
    features['f_B'] = f_B
    features['n_ch'] = n_detected

    # --- Preprocessing ---
    seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg_bip = np.array(fcn_getBanana(seg_filtered))

    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(seg_bip.shape[0]):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    n_channels = seg_lp.shape[0]
    sigma_samples = max(1, int(0.02 * fs))

    # Pointiness traces
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # --- f_peaks: peak-count on pointiness ---
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # --- f_fft: FFT of pointiness ---
    fft_freqs = []
    for ch in range(n_channels):
        f = compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    features['f_fft'] = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # --- f_tkeo ---
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # --- f_coh: spectral coherence ---
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs,
                                   nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # placeholder = 0
    features['placeholder'] = 0.0

    # --- Quick-win features ---

    # ratio_fft_to_peaks
    f_fft_val = features.get('f_fft', np.nan)
    f_peaks_val = features.get('f_peaks', np.nan)
    if np.isfinite(f_fft_val) and np.isfinite(f_peaks_val) and f_peaks_val > 0 and f_fft_val > 0:
        features['ratio_fft_to_peaks'] = f_fft_val / f_peaks_val
    else:
        features['ratio_fft_to_peaks'] = np.nan

    # eigenvalue_spread: ratio of largest to second-largest eigenvalue of 18-ch covariance
    try:
        # Use seg_bip (18 x 2000 preprocessed bipolar)
        cov_mat = np.cov(seg_bip)  # 18x18
        eigvals = np.linalg.eigvalsh(cov_mat)  # sorted ascending
        if len(eigvals) >= 2 and eigvals[-2] > 0:
            features['eigenvalue_spread'] = float(eigvals[-1] / eigvals[-2])
        else:
            features['eigenvalue_spread'] = np.nan
    except:
        features['eigenvalue_spread'] = np.nan

    return features


def prepare_features_lopo(feature_dicts, feature_names, train_idx, test_idx):
    """Build feature matrices with NaN->median from TRAINING set only."""
    n_train = len(train_idx)
    n_test = len(test_idx)
    p = len(feature_names)

    X_train = np.full((n_train, p), np.nan)
    X_test = np.full((n_test, p), np.nan)

    for i, gi in enumerate(train_idx):
        for j, fn in enumerate(feature_names):
            X_train[i, j] = feature_dicts[gi].get(fn, np.nan)

    for i, gi in enumerate(test_idx):
        for j, fn in enumerate(feature_names):
            X_test[i, j] = feature_dicts[gi].get(fn, np.nan)

    # Compute medians from training set, impute both
    for j in range(p):
        col = X_train[:, j]
        finite_mask = np.isfinite(col)
        if np.any(finite_mask):
            med = np.median(col[finite_mask])
        else:
            med = 0.0
        X_train[~finite_mask, j] = med
        # Impute test with training median
        test_col = X_test[:, j]
        test_nan = ~np.isfinite(test_col)
        X_test[test_nan, j] = med

    return X_train, X_test


def ridge_fit(X, y, alpha=1.0):
    """Fit ridge regression, return coefficients."""
    p = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(p)
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    return beta


def run_lopo_per_expert(feature_dicts, feature_names, patient_ids, mat_names,
                        expert_LB, expert_PH, expert_SZ, expert_consensus,
                        dataset, experiment_name, alpha=1.0,
                        exclude_patients=None, subset_mask=None):
    """
    LOPO per-expert ridge model.

    For each patient held out:
    - Train 3 ridge models (one per expert) on remaining patients
    - For each expert model, predict held-out segments
    - Average the 3 predictions

    exclude_patients: set of patient IDs to exclude from both train AND test
    subset_mask: boolean array, True = include in analysis
    """
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name} (LOPO, per-expert ridge, alpha={alpha})")
    print(f"{'='*60}")

    n = len(feature_dicts)
    if subset_mask is None:
        subset_mask = np.ones(n, dtype=bool)

    if exclude_patients is None:
        exclude_patients = set()

    # Apply exclusions
    valid_mask = np.array([
        subset_mask[i] and patient_ids[i] not in exclude_patients
        for i in range(n)
    ])

    valid_indices = np.where(valid_mask)[0]
    valid_patients = [patient_ids[i] for i in valid_indices]
    unique_patients = sorted(set(valid_patients))

    print(f"  Segments: {len(valid_indices)}, Patients: {len(unique_patients)}")

    # Build predictions
    all_preds = np.full(n, np.nan)

    for fold_i, held_out_pat in enumerate(unique_patients):
        if (fold_i + 1) % 10 == 0 or fold_i == 0:
            print(f"  Fold {fold_i+1}/{len(unique_patients)}: holding out {held_out_pat}")

        test_idx = [i for i in valid_indices if patient_ids[i] == held_out_pat]
        train_idx = [i for i in valid_indices if patient_ids[i] != held_out_pat]

        if len(test_idx) == 0 or len(train_idx) < 5:
            continue

        X_train, X_test = prepare_features_lopo(feature_dicts, feature_names, train_idx, test_idx)

        # Per-expert training
        segment_preds = np.full((len(test_idx), 3), np.nan)

        for e_idx, (expert_name, expert_arr) in enumerate([
            ('LB', expert_LB), ('PH', expert_PH), ('SZ', expert_SZ)
        ]):
            # Build train targets for this expert
            train_expert_vals = np.array([expert_arr[i] for i in train_idx])
            train_mask = np.isfinite(train_expert_vals) & (train_expert_vals > 0)

            if np.sum(train_mask) < 3:
                continue

            X_e = X_train[train_mask]
            y_e = np.log(train_expert_vals[train_mask])
            beta = ridge_fit(X_e, y_e, alpha=alpha)

            # Predict test segments (clamp log predictions to reasonable range)
            preds_log = X_test @ beta
            preds_log = np.clip(preds_log, np.log(0.1), np.log(10.0))
            segment_preds[:, e_idx] = np.exp(preds_log)

        # Average across experts
        fold_preds = np.nanmean(segment_preds, axis=1)

        for local_i, global_i in enumerate(test_idx):
            all_preds[global_i] = fold_preds[local_i]

    # Build predictions dict
    predictions = {}
    for i in valid_indices:
        if np.isfinite(all_preds[i]):
            predictions[mat_names[i]] = float(all_preds[i])

    print(f"  Predictions: {len(predictions)} segments")

    # If we excluded patients, we need to filter the dataset for evaluate_predictions
    if exclude_patients:
        filtered_dataset = [
            e for e in dataset
            if extract_patient_id(e['mat_name']) not in exclude_patients
        ]
    else:
        filtered_dataset = dataset

    metrics = evaluate_predictions(filtered_dataset, predictions, experiment_name)
    return metrics, predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("ROUND 9: LOPO EXPERIMENTS")
    print("=" * 70)

    # ---- Load dataset ----
    print("\n[1] Loading dataset...")
    dataset = load_dataset()
    print(f"  Dataset: {len(dataset)} segments")

    # ---- Extract features for all segments ----
    print("\n[2] Computing SP features for all segments...")
    all_features = []
    mat_names = []
    patient_ids = []
    expert_consensus = []
    expert_LB_arr = []
    expert_PH_arr = []
    expert_SZ_arr = []
    subtypes = []

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing segment {idx+1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        mat_name = entry['mat_name']
        pat_id = extract_patient_id(mat_name)

        if data is None:
            all_features.append(None)
        else:
            feats = compute_sp_features(data, fs, entry)
            all_features.append(feats)

        mat_names.append(mat_name)
        patient_ids.append(pat_id)
        expert_consensus.append(entry['expert_consensus_freq'])
        expert_LB_arr.append(entry.get('expert_LB_freq', np.nan))
        expert_PH_arr.append(entry.get('expert_PH_freq', np.nan))
        expert_SZ_arr.append(entry.get('expert_SZ_freq', np.nan))
        subtypes.append(entry['subdir'])

    expert_consensus = np.array(expert_consensus)
    expert_LB_arr = np.array(expert_LB_arr)
    expert_PH_arr = np.array(expert_PH_arr)
    expert_SZ_arr = np.array(expert_SZ_arr)
    subtypes = np.array(subtypes)

    # Filter valid segments (those with features)
    valid_mask = np.array([f is not None for f in all_features])
    n_valid = int(np.sum(valid_mask))
    print(f"\n  Feature extraction complete. {n_valid}/{len(dataset)} segments with features.")

    # For LOPO we work with indices into the full array, but only where valid
    valid_indices = np.where(valid_mask)[0]

    # ============================================================
    # Experiment 1: LOPO with 8 SP features
    # ============================================================
    SP_FEATURES_8 = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch', 'placeholder']

    metrics_1, preds_1 = run_lopo_per_expert(
        all_features, SP_FEATURES_8, patient_ids, mat_names,
        expert_LB_arr, expert_PH_arr, expert_SZ_arr, expert_consensus,
        dataset, experiment_name='r9_lopo_sp_only',
        alpha=1.0, subset_mask=valid_mask
    )

    # ============================================================
    # Experiment 2: LOPO excluding abn2147
    # ============================================================
    metrics_2, preds_2 = run_lopo_per_expert(
        all_features, SP_FEATURES_8, patient_ids, mat_names,
        expert_LB_arr, expert_PH_arr, expert_SZ_arr, expert_consensus,
        dataset, experiment_name='r9_lopo_strict_gpd',
        alpha=1.0, subset_mask=valid_mask,
        exclude_patients={'abn2147'}
    )

    # Report LPD and GPD separately for strict_gpd
    print("\n  Strict GPD breakdown (excluding abn2147):")
    print(f"    LPD: n={metrics_2.get('lpd_n','?')}, MAE={metrics_2.get('lpd_mae','?')}, "
          f"Spearman={metrics_2.get('lpd_spearman_r','?')}, pooled={metrics_2.get('lpd_spearman_pooled','?')}")
    print(f"    GPD: n={metrics_2.get('gpd_n','?')}, MAE={metrics_2.get('gpd_mae','?')}, "
          f"Spearman={metrics_2.get('gpd_spearman_r','?')}, pooled={metrics_2.get('gpd_spearman_pooled','?')}")

    # ============================================================
    # Experiment 3: LOPO with 10 features (8 + quick wins)
    # ============================================================
    SP_FEATURES_10 = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch', 'placeholder',
                      'ratio_fft_to_peaks', 'eigenvalue_spread']

    metrics_3, preds_3 = run_lopo_per_expert(
        all_features, SP_FEATURES_10, patient_ids, mat_names,
        expert_LB_arr, expert_PH_arr, expert_SZ_arr, expert_consensus,
        dataset, experiment_name='r9_lopo_quick_wins',
        alpha=1.0, subset_mask=valid_mask
    )

    # ============================================================
    # Experiment 4: k-NN with CNN embeddings
    # ============================================================
    print(f"\n{'='*60}")
    print("Running: r9_knn_cross_patient (k-NN with CNN embeddings)")
    print(f"{'='*60}")

    # Load pretrained CNN backbone
    print("  Loading pretrained CNN backbone...")
    from model import EEGClassifier
    from data_loader import normalize_segment

    model = EEGClassifier(in_channels=18, dropout=0.1)
    ckpt = torch.load(str(CACHE_DIR / 'classifier_best.pt'), map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    backbone = model.backbone

    # Load preprocessed segments from annotated data
    print("  Loading preprocessed segments...")
    ann_data = np.load(str(CACHE_DIR / 'annotated_pd_data.npz'), allow_pickle=True)
    ann_segments = ann_data['segments']       # (556, 18, 2000)
    ann_expert_freqs = ann_data['expert_freqs']  # (556, 3)
    ann_patients = ann_data['patients']       # (556,)
    ann_subtypes = ann_data['subtypes']       # (556,)

    # Compute consensus frequencies for each segment in npz
    ann_consensus = np.full(len(ann_patients), np.nan)
    for i in range(len(ann_patients)):
        valid_freqs = ann_expert_freqs[i]
        vf = valid_freqs[np.isfinite(valid_freqs) & (valid_freqs > 0)]
        if len(vf) > 0:
            ann_consensus[i] = np.median(vf)

    # Extract CNN embeddings for all 556 segments
    print("  Extracting CNN embeddings...")
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for start in range(0, len(ann_segments), batch_size):
            end = min(start + batch_size, len(ann_segments))
            batch = []
            for i in range(start, end):
                seg = normalize_segment(ann_segments[i].copy())
                batch.append(seg)
            batch_tensor = torch.from_numpy(np.array(batch, dtype=np.float32))
            # backbone(x) -> (B, 128, 125), then mean(dim=-1) -> (B, 128)
            feat = backbone(batch_tensor).mean(dim=-1)
            embeddings.append(feat.numpy())
            if (start // batch_size) % 5 == 0:
                print(f"    Batch {start//batch_size + 1}/{(len(ann_segments) + batch_size - 1) // batch_size}")

    embeddings = np.concatenate(embeddings, axis=0)  # (556, 128)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    embeddings_normed = embeddings / norms

    # k-NN cross-patient
    K = 5
    knn_preds = np.full(len(ann_patients), np.nan)

    print(f"  Computing k={K} NN predictions (cross-patient)...")
    for i in range(len(ann_patients)):
        pat_i = ann_patients[i]
        # Cosine similarity = dot product of normalized vectors
        sims = embeddings_normed @ embeddings_normed[i]
        # Exclude same patient
        other_mask = ann_patients != pat_i
        sims[~other_mask] = -np.inf
        # Also need valid consensus freq
        valid_freq_mask = np.isfinite(ann_consensus)
        sims[~valid_freq_mask] = -np.inf

        # Top-k indices
        top_k = np.argsort(sims)[-K:]
        neighbor_freqs = ann_consensus[top_k]
        valid_neighbor = neighbor_freqs[np.isfinite(neighbor_freqs)]
        if len(valid_neighbor) > 0:
            knn_preds[i] = float(np.median(valid_neighbor))

    # Map npz indices to mat_names in the dataset
    # We need to match ann_data segments to dataset entries
    # The dataset and npz should have the same segments but possibly different order.
    # Let's build a mapping using patient + subtype + index within patient

    # Actually, the dataset entries have mat_names and the npz has patients/subtypes.
    # We need to connect them. Let me check if they're in the same order.
    print("  Mapping npz segments to dataset mat_names...")

    # Build a patient->entries mapping from dataset
    pat_subtype_to_dataset = {}
    for idx, entry in enumerate(dataset):
        pat = extract_patient_id(entry['mat_name'])
        st = entry['subdir']
        key = (pat, st)
        if key not in pat_subtype_to_dataset:
            pat_subtype_to_dataset[key] = []
        pat_subtype_to_dataset[key].append(idx)

    # Build same mapping from npz
    pat_subtype_to_npz = {}
    for i in range(len(ann_patients)):
        key = (ann_patients[i], ann_subtypes[i])
        if key not in pat_subtype_to_npz:
            pat_subtype_to_npz[key] = []
        pat_subtype_to_npz[key].append(i)

    # Map npz_idx -> dataset_idx by matching within (patient, subtype) groups
    npz_to_dataset = {}
    dataset_to_npz = {}
    for key in pat_subtype_to_npz:
        npz_indices = pat_subtype_to_npz[key]
        ds_indices = pat_subtype_to_dataset.get(key, [])
        # They should have same count and be in same order
        if len(npz_indices) == len(ds_indices):
            for ni, di in zip(npz_indices, ds_indices):
                npz_to_dataset[ni] = di
                dataset_to_npz[di] = ni
        else:
            print(f"    WARNING: mismatch for {key}: npz={len(npz_indices)}, ds={len(ds_indices)}")
            # Best effort: match as many as possible
            for ni, di in zip(npz_indices, ds_indices):
                npz_to_dataset[ni] = di
                dataset_to_npz[di] = ni

    # Build knn predictions dict
    knn_predictions = {}
    for npz_i, ds_i in npz_to_dataset.items():
        if np.isfinite(knn_preds[npz_i]):
            knn_predictions[mat_names[ds_i]] = float(knn_preds[npz_i])

    print(f"  k-NN predictions: {len(knn_predictions)} segments")
    metrics_4 = evaluate_predictions(dataset, knn_predictions, 'r9_knn_cross_patient')

    # ============================================================
    # Experiment 5: LOPO ridge with k-NN prediction as additional feature
    # ============================================================
    print(f"\n{'='*60}")
    print("Running: r9_lopo_knn_feature (LOPO ridge with k-NN as feature)")
    print(f"{'='*60}")

    # Add knn_pred as a feature to the feature dicts
    # We need cross-patient k-NN for each segment
    for ds_i in range(len(all_features)):
        if all_features[ds_i] is None:
            continue
        npz_i = dataset_to_npz.get(ds_i, None)
        if npz_i is not None and np.isfinite(knn_preds[npz_i]):
            all_features[ds_i]['knn_pred'] = float(knn_preds[npz_i])
        else:
            all_features[ds_i]['knn_pred'] = np.nan

    SP_FEATURES_KNN = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch', 'placeholder', 'knn_pred']

    metrics_5, preds_5 = run_lopo_per_expert(
        all_features, SP_FEATURES_KNN, patient_ids, mat_names,
        expert_LB_arr, expert_PH_arr, expert_SZ_arr, expert_consensus,
        dataset, experiment_name='r9_lopo_knn_feature',
        alpha=1.0, subset_mask=valid_mask
    )

    # ============================================================
    # Final comparison table
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Experiment':<30s} {'Comb Spear':>10s} {'Comb MAE':>10s} {'LPD rs':>8s} {'GPD rs':>8s} {'LPD MAE':>8s} {'GPD MAE':>8s}")
    print("-" * 80)

    all_metrics = [
        ('r9_lopo_sp_only', metrics_1),
        ('r9_lopo_strict_gpd', metrics_2),
        ('r9_lopo_quick_wins', metrics_3),
        ('r9_knn_cross_patient', metrics_4),
        ('r9_lopo_knn_feature', metrics_5),
    ]

    for name, m in all_metrics:
        cs = m.get('combined_spearman', np.nan)
        cm = m.get('combined_mae', np.nan)
        lr = m.get('lpd_spearman_pooled', m.get('lpd_spearman_r', np.nan))
        gr = m.get('gpd_spearman_pooled', m.get('gpd_spearman_r', np.nan))
        lm = m.get('lpd_mae', np.nan)
        gm = m.get('gpd_mae', np.nan)
        print(f"  {name:<28s} {cs:>10} {cm:>10} {lr:>8} {gr:>8} {lm:>8} {gm:>8}")

    print(f"\n  Baseline (5-fold GroupKFold): combined Spearman ~ 0.476")
    print(f"  Expert-expert baseline: combined Spearman ~ 0.50")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
