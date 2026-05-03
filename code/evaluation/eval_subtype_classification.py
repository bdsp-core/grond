#!/usr/bin/env python3
"""
Evaluate LPD vs GPD subtype classification using ChannelPD-Net features
plus handcrafted frequency/laterality features.

Features (26 total):
  - 18 per-channel PD probabilities (from predictions.json)
  - 5 frequency features: f_B (ACF), f_peaks, f_fft, f_tkeo, f_coh
  - 3 laterality features: lat_idx, lat_energy_ratio, lat_acf_ratio

Uses 5-fold patient-stratified CV with RF 300 trees.

Usage:
    conda run -n morgoth python code/evaluation/eval_subtype_classification.py
    conda run -n morgoth python code/evaluation/eval_subtype_classification.py --compute-features
"""

import argparse
import csv
import json
import sys
import time
import numpy as np
import scipy.io as sio
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
PREDICTIONS_PATH = LABELS_DIR / 'predictions.json'
FEATURES_CACHE = PROJECT_DIR / 'data' / 'evaluation_results' / 'handcrafted_features.json'
RESULTS_DIR = PROJECT_DIR / 'data' / 'evaluation_results'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / 'subtype_classification.json'

sys.path.insert(0, str(CODE_DIR))

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
FS = 200
LOWPASS_HZ = 20.0
FREQ_LO, FREQ_HI = 0.3, 3.5
SMOOTHING_SIGMA = 0.015  # seconds
ACF_MIN_LAG = 0.25
ACF_THRESHOLD = 0.15
PEAK_HEIGHT_FRAC = 0.05

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (12, 13), (13, 14), (14, 15),
]


def load_mono(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] != 19:
        return None
    return seg[:, :2000]


def mono_to_bipolar(mono):
    ch_idx = {ch: i for i, ch in enumerate(MONO_CHANNELS)}
    bip = np.zeros((18, mono.shape[1]))
    for i, (a, b) in enumerate(BIPOLAR_PAIRS):
        bip[i] = mono[ch_idx[a]] - mono[ch_idx[b]]
    return bip


def compute_sp_features(seg_bip):
    """Compute 8 handcrafted features from 18-channel bipolar EEG.

    Same features as optimization_harness_v2.py compute_sp_features().
    """
    from scipy.signal import butter, filtfilt, find_peaks, coherence as scipy_coherence
    from scipy.ndimage import gaussian_filter1d
    from pd_pointiness_acf import compute_acf_frequency, compute_pointiness_trace

    fs = FS
    n_channels = seg_bip.shape[0]
    features = {}

    # Lowpass filter
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # f_B: ACF on lowpassed bipolar channels
    acf_freqs = np.full(n_channels, np.nan)
    for ch in range(n_channels):
        freq, score, _ = compute_acf_frequency(
            seg_lp[ch], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        acf_freqs[ch] = freq
    valid_acf = acf_freqs[np.isfinite(acf_freqs)]
    features['f_B'] = float(np.median(valid_acf)) if len(valid_acf) > 0 else np.nan

    # Pointiness traces
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # f_peaks
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * PEAK_HEIGHT_FRAC, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # f_fft
    def _fft_peak(trace):
        n = len(trace)
        if n < 10:
            return np.nan
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
        if not np.any(mask):
            return np.nan
        fft_sub = fft_vals[mask]
        freqs_sub = freqs[mask]
        return freqs_sub[np.argmax(fft_sub)] if np.max(fft_sub) > 0 else np.nan

    fft_freqs = [_fft_peak(pointiness_traces[ch]) for ch in range(n_channels)]
    fft_valid = [f for f in fft_freqs if np.isfinite(f)]
    features['f_fft'] = float(np.median(fft_valid)) if fft_valid else np.nan

    # f_tkeo
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = _fft_peak(tkeo_smooth)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = scipy_coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs,
                                          nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except Exception:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # Laterality features
    peak_strengths = np.zeros(n_channels)
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx > 0:
            pks, props = find_peaks(pt, height=mx * PEAK_HEIGHT_FRAC, distance=int(0.2 * fs))
            if len(pks) >= 2:
                peak_strengths[ch] = np.mean(props['peak_heights'])

    left_s = np.mean(peak_strengths[LEFT_INDICES])
    right_s = np.mean(peak_strengths[RIGHT_INDICES])
    d = left_s + right_s
    features['lat_idx'] = float((right_s - left_s) / d) if d > 0 else 0.0

    left_energy = np.mean([np.sqrt(np.mean(seg_lp[ch] ** 2)) for ch in LEFT_INDICES])
    right_energy = np.mean([np.sqrt(np.mean(seg_lp[ch] ** 2)) for ch in RIGHT_INDICES])
    if left_energy > 0 and right_energy > 0:
        features['lat_energy_ratio'] = float(np.log(right_energy / left_energy))
    else:
        features['lat_energy_ratio'] = 0.0

    left_acf = acf_freqs[LEFT_INDICES]
    right_acf = acf_freqs[RIGHT_INDICES]
    lv = left_acf[np.isfinite(left_acf)]
    rv = right_acf[np.isfinite(right_acf)]
    if len(lv) > 0 and len(rv) > 0:
        lm, rm = np.median(lv), np.median(rv)
        d2 = lm + rm
        features['lat_acf_ratio'] = float((rm - lm) / d2) if d2 > 0 else 0.0
    else:
        features['lat_acf_ratio'] = 0.0

    return features


HANDCRAFTED_KEYS = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh',
                    'lat_idx', 'lat_energy_ratio', 'lat_acf_ratio']


def compute_and_cache_features(mat_files_and_subtypes):
    """Compute handcrafted features for all segments and cache to disk."""
    print(f"  Computing handcrafted features for {len(mat_files_and_subtypes)} segments...")
    cache = {}
    t0 = time.time()
    for i, (mat, sub) in enumerate(mat_files_and_subtypes):
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(mat_files_and_subtypes)}] ({time.time()-t0:.0f}s)...")
        mono = load_mono(mat)
        if mono is None:
            continue
        try:
            bip = mono_to_bipolar(mono)
            feats = compute_sp_features(bip)
            cache[mat] = feats
        except Exception:
            pass

    with open(FEATURES_CACHE, 'w') as f:
        json.dump(cache, f)
    print(f"  Cached {len(cache)} segments ({time.time()-t0:.0f}s) → {FEATURES_CACHE.name}")
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute-features', action='store_true',
                        help='Recompute handcrafted features from EEG (slow, ~15 min)')
    args = parser.parse_args()

    print("Evaluating LPD vs GPD subtype classification...")

    # Load channel probabilities from predictions
    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    # Load segment metadata
    segments = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get('subtype', '').lower()
            if sub not in ('lpd', 'gpd'):
                continue
            if row.get('excluded', '').lower() in ('true', '1', 'yes'):
                continue
            mat = row['mat_file']
            if mat in predictions and predictions[mat].get('channel_probs'):
                probs = predictions[mat]['channel_probs']
                if len(probs) == 18 and not any(p is None or np.isnan(p) for p in probs):
                    segments[mat] = {
                        'subtype': sub,
                        'patient_id': row.get('patient_id', mat),
                        'channel_probs': probs,
                    }

    print(f"  {len(segments)} segments with valid channel predictions")
    n_lpd = sum(1 for s in segments.values() if s['subtype'] == 'lpd')
    n_gpd = sum(1 for s in segments.values() if s['subtype'] == 'gpd')
    print(f"  LPD: {n_lpd}, GPD: {n_gpd}")

    # Load or compute handcrafted features
    if args.compute_features or not FEATURES_CACHE.exists():
        mat_sub_pairs = [(mat, info['subtype']) for mat, info in segments.items()]
        hc_features = compute_and_cache_features(mat_sub_pairs)
    else:
        print(f"  Loading cached handcrafted features from {FEATURES_CACHE.name}...")
        with open(FEATURES_CACHE) as f:
            hc_features = json.load(f)
        print(f"  Loaded {len(hc_features)} cached features")

    # Build feature vectors: 18 channel probs + 8 handcrafted = 26 features
    patients = defaultdict(list)
    for mat, info in segments.items():
        hc = hc_features.get(mat, {})
        hc_vec = [hc.get(k, np.nan) for k in HANDCRAFTED_KEYS]
        full_features = info['channel_probs'] + hc_vec  # 26 features
        patients[info['patient_id']].append((mat, info['subtype'], full_features))

    print(f"  {len(patients)} unique patients")
    n_with_hc = sum(1 for mat in segments if mat in hc_features)
    print(f"  {n_with_hc}/{len(segments)} have handcrafted features")

    # Build arrays at patient level
    patient_ids = sorted(patients.keys())
    X_all, y_all = [], []
    for pid in patient_ids:
        segs = patients[pid]
        mean_feats = np.nanmean([s[2] for s in segs], axis=0)
        label = 1 if segs[0][1] == 'gpd' else 0
        X_all.append(mean_feats)
        y_all.append(label)

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    # Impute NaN with training median (done per fold)
    print("  Running 5-fold patient-stratified CV...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = np.array(patient_ids)

    all_probs = []
    fold_models = []
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups)):
        X_tr, y_tr = X_all[train_idx].copy(), y_all[train_idx]
        X_te = X_all[test_idx].copy()

        # Impute NaN with training median
        for col in range(X_tr.shape[1]):
            med = np.nanmedian(X_tr[:, col])
            X_tr[np.isnan(X_tr[:, col]), col] = med
            X_te[np.isnan(X_te[:, col]), col] = med

        rf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        probs = rf.predict_proba(X_te)[:, 1]

        for idx, prob in zip(test_idx, probs):
            all_probs.append((patient_ids[idx], int(y_all[idx]), float(prob)))

        print(f"    Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
        fold_models.append(rf)

    # Train one final model on ALL data so the classifier can be deployed at
    # inference time without depending on a particular fold split.
    print("  Training final model on full cohort...")
    X_full = X_all.copy()
    for col in range(X_full.shape[1]):
        med = np.nanmedian(X_full[:, col])
        X_full[np.isnan(X_full[:, col]), col] = med
    final_rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    final_rf.fit(X_full, y_all)

    # Compute AUC
    labels = np.array([p[1] for p in all_probs])
    scores = np.array([p[2] for p in all_probs])
    auc = roc_auc_score(labels, scores)

    preds = (scores >= 0.5).astype(int)
    accuracy = float(np.mean(preds == labels))
    sensitivity = float(np.mean(preds[labels == 1] == 1)) if np.sum(labels == 1) > 0 else 0
    specificity = float(np.mean(preds[labels == 0] == 0)) if np.sum(labels == 0) > 0 else 0

    results = {
        'task': 'LPD vs GPD subtype classification',
        'method': 'RF 300 trees on 18 channel probs + 8 handcrafted features',
        'cv': '5-fold patient-stratified',
        'n_patients': len(all_probs),
        'n_lpd': int(np.sum(labels == 0)),
        'n_gpd': int(np.sum(labels == 1)),
        'n_segments': len(segments),
        'n_features': 26,
        'auc': round(auc, 4),
        'accuracy': round(accuracy, 4),
        'sensitivity_gpd': round(sensitivity, 4),
        'specificity_lpd': round(specificity, 4),
        'feature_names': [f'ch_prob_{i}' for i in range(18)] + HANDCRAFTED_KEYS,
    }

    print(f"\n  Results:")
    print(f"    AUC: {auc:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Sensitivity (GPD): {sensitivity:.4f}")
    print(f"    Specificity (LPD): {specificity:.4f}")
    print(f"    N patients: {len(all_probs)} ({results['n_lpd']} LPD, {results['n_gpd']} GPD)")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")

    # Persist the trained classifier so it can be reused without re-running
    # the full feature-extraction + CV loop.
    import joblib
    model_out = PROJECT_DIR / 'data' / 'models' / 'lpd_vs_gpd_rf.pkl'
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        'final': final_rf,
        'folds': fold_models,
        'feature_names': results['feature_names'],
        'cv': 'StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)',
        'hyperparams': dict(n_estimators=300, max_depth=8, min_samples_leaf=3,
                             random_state=42),
    }, model_out)
    print(f"  Saved trained models: {model_out}")


if __name__ == '__main__':
    main()
