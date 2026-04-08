#!/usr/bin/env python3
"""
Evaluate 3-way LPD vs GPD vs BIPD subtype classification.

Uses per-hemisphere discharge timing features from bipd_cache/
combined with ChannelPD-Net channel probabilities.

Features:
  - 18 per-channel PD probabilities
  - 8 timing-pair features (from L/R hemisphere discharge sequences)

Usage:
    conda run -n morgoth python code/evaluation/eval_3way_classification.py
"""

import csv
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
BIPD_CACHE = PROJECT_DIR / 'data' / 'bipd_cache'
PREDICTIONS_PATH = LABELS_DIR / 'predictions.json'
RESULTS_DIR = PROJECT_DIR / 'data' / 'evaluation_results'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / 'three_way_classification.json'


def extract_timing_features(left_times, right_times):
    """Extract features from a pair of per-hemisphere discharge time sequences.

    Features capture the degree of independence between L and R hemispheres:
    - BIPD: independent timing → low correlation, different frequencies
    - GPD: synchronized → high correlation, same frequency
    - LPD: one hemisphere dominant → asymmetric counts
    """
    lt = np.array(left_times, dtype=float)
    rt = np.array(right_times, dtype=float)

    features = {}

    # Basic counts
    features['n_left'] = len(lt)
    features['n_right'] = len(rt)
    features['count_ratio'] = min(len(lt), len(rt)) / max(len(lt), len(rt)) if max(len(lt), len(rt)) > 0 else 0
    features['count_asymmetry'] = abs(len(lt) - len(rt)) / max(len(lt), len(rt)) if max(len(lt), len(rt)) > 0 else 0

    # Frequency from IPI
    if len(lt) >= 2:
        ipi_l = np.diff(lt)
        features['freq_left'] = 1.0 / np.median(ipi_l) if np.median(ipi_l) > 0 else 0
        features['regularity_left'] = np.std(ipi_l) / np.mean(ipi_l) if np.mean(ipi_l) > 0 else 1
    else:
        features['freq_left'] = 0
        features['regularity_left'] = 1

    if len(rt) >= 2:
        ipi_r = np.diff(rt)
        features['freq_right'] = 1.0 / np.median(ipi_r) if np.median(ipi_r) > 0 else 0
        features['regularity_right'] = np.std(ipi_r) / np.mean(ipi_r) if np.mean(ipi_r) > 0 else 1
    else:
        features['freq_right'] = 0
        features['regularity_right'] = 1

    # Frequency ratio
    if features['freq_left'] > 0 and features['freq_right'] > 0:
        features['freq_ratio'] = min(features['freq_left'], features['freq_right']) / max(features['freq_left'], features['freq_right'])
    else:
        features['freq_ratio'] = 0

    # Matched fraction: how many L discharges have a close R match
    MATCH_TOL = 0.1  # 100ms tolerance
    if len(lt) > 0 and len(rt) > 0:
        matched = 0
        for t in lt:
            if np.min(np.abs(rt - t)) < MATCH_TOL:
                matched += 1
        features['matched_fraction'] = matched / len(lt)
    else:
        features['matched_fraction'] = 0

    # Phase consistency: correlation of inter-discharge intervals
    if len(lt) >= 3 and len(rt) >= 3:
        # Resample both to common grid and correlate
        min_t = max(lt[0], rt[0])
        max_t = min(lt[-1], rt[-1])
        if max_t > min_t:
            # Binary event vectors at 10ms resolution
            resolution = 0.01
            n_bins = int((max_t - min_t) / resolution) + 1
            l_events = np.zeros(n_bins)
            r_events = np.zeros(n_bins)
            for t in lt:
                idx = int((t - min_t) / resolution)
                if 0 <= idx < n_bins:
                    l_events[idx] = 1
            for t in rt:
                idx = int((t - min_t) / resolution)
                if 0 <= idx < n_bins:
                    r_events[idx] = 1
            # Smooth and correlate
            from scipy.ndimage import gaussian_filter1d
            l_smooth = gaussian_filter1d(l_events, sigma=5)
            r_smooth = gaussian_filter1d(r_events, sigma=5)
            if np.std(l_smooth) > 0 and np.std(r_smooth) > 0:
                features['phase_correlation'] = float(np.corrcoef(l_smooth, r_smooth)[0, 1])
            else:
                features['phase_correlation'] = 0
        else:
            features['phase_correlation'] = 0
    else:
        features['phase_correlation'] = 0

    return features


TIMING_FEATURE_KEYS = [
    'n_left', 'n_right', 'count_ratio', 'count_asymmetry',
    'freq_left', 'freq_right', 'freq_ratio', 'regularity_left', 'regularity_right',
    'matched_fraction', 'phase_correlation',
]


def load_hemi_detections():
    """Load per-hemisphere discharge detections from bipd_cache."""
    data = {}

    for label_file, label in [
        ('bipd_hemi_detections.json', 'bipd'),
        ('gpd_hemi_detections.json', 'gpd'),
        ('lpd_hemi_detections.json', 'lpd'),
    ]:
        path = BIPD_CACHE / label_file
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as f:
            entries = json.load(f)
        for patient_id, entry in entries.items():
            left = entry.get('left', {})
            right = entry.get('right', {})
            lt = left.get('times', [])
            rt = right.get('times', [])
            if len(lt) >= 3 and len(rt) >= 3:
                data[patient_id] = {
                    'label': label,
                    'left_times': lt,
                    'right_times': rt,
                }

    return data


def main():
    print("Evaluating 3-way LPD vs GPD vs BIPD classification...")

    # Load per-hemisphere timing data
    hemi_data = load_hemi_detections()
    label_counts = defaultdict(int)
    for v in hemi_data.values():
        label_counts[v['label']] += 1
    print(f"  Loaded per-hemisphere timing: {dict(label_counts)}")

    # Load channel probabilities
    predictions = {}
    if PREDICTIONS_PATH.exists():
        with open(PREDICTIONS_PATH) as f:
            predictions = json.load(f)

    # Map patient IDs to mat files for channel prob lookup
    patient_to_mat = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        for row in csv.DictReader(f):
            pid = row.get('patient_id', '')
            if pid in hemi_data:
                patient_to_mat.setdefault(pid, []).append(row['mat_file'])

    # Build feature vectors
    patient_ids = []
    X_list = []
    y_list = []
    label_map = {'lpd': 0, 'gpd': 1, 'bipd': 2}

    for pid, info in hemi_data.items():
        # Timing features
        timing_feats = extract_timing_features(info['left_times'], info['right_times'])
        timing_vec = [timing_feats.get(k, 0) for k in TIMING_FEATURE_KEYS]

        # Channel probability features (average across segments for this patient)
        ch_probs = None
        mats = patient_to_mat.get(pid, [])
        if mats:
            prob_list = []
            for mat in mats:
                pred = predictions.get(mat, {})
                probs = pred.get('channel_probs')
                if probs and len(probs) == 18 and not any(p is None or np.isnan(p) for p in probs):
                    prob_list.append(probs)
            if prob_list:
                ch_probs = np.mean(prob_list, axis=0).tolist()

        if ch_probs is None:
            ch_probs = [0.0] * 18  # fallback

        # Combined features: 18 channel probs + 11 timing = 29
        full_vec = ch_probs + timing_vec

        patient_ids.append(pid)
        X_list.append(full_vec)
        y_list.append(label_map[info['label']])

    X = np.array(X_list)
    y = np.array(y_list)
    patient_ids = np.array(patient_ids)

    print(f"  Total patients: {len(y)} (LPD={np.sum(y==0)}, GPD={np.sum(y==1)}, BIPD={np.sum(y==2)})")
    print(f"  Features: {X.shape[1]} (18 channel probs + {len(TIMING_FEATURE_KEYS)} timing)")

    # Impute NaN
    for col in range(X.shape[1]):
        mask = np.isnan(X[:, col]) | np.isinf(X[:, col])
        if mask.any():
            X[mask, col] = np.nanmedian(X[:, col])

    # 5-fold patient-stratified CV
    print("  Running 5-fold CV...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = np.zeros((len(y), 3))  # probabilities for each class
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, patient_ids)):
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=42, n_jobs=-1, class_weight='balanced',
        )
        rf.fit(X[train_idx], y[train_idx])
        probs = rf.predict_proba(X[test_idx])

        # Handle case where not all classes appear in fold
        for i, cls in enumerate(rf.classes_):
            all_preds[test_idx, cls] = probs[:, i]

        fold_preds = rf.predict(X[test_idx])
        fold_acc = np.mean(fold_preds == y[test_idx])
        print(f"    Fold {fold}: acc={fold_acc:.3f}")

    # Compute metrics
    y_pred = np.argmax(all_preds, axis=1)
    accuracy = float(np.mean(y_pred == y))

    # Per-class AUC (one-vs-rest)
    y_bin = label_binarize(y, classes=[0, 1, 2])
    class_names = ['LPD', 'GPD', 'BIPD']
    aucs = {}
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y):
            aucs[name] = float(roc_auc_score(y_bin[:, i], all_preds[:, i]))
        else:
            aucs[name] = None

    # Macro AUC
    valid_aucs = [v for v in aucs.values() if v is not None]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else None

    # BIPD sensitivity/specificity
    bipd_pred = (y_pred == 2)
    bipd_true = (y == 2)
    if bipd_true.sum() > 0:
        bipd_sensitivity = float(np.mean(bipd_pred[bipd_true]))
        bipd_specificity = float(np.mean(~bipd_pred[~bipd_true]))
    else:
        bipd_sensitivity = None
        bipd_specificity = None

    # Binary BIPD vs GPD AUC
    gpd_bipd_mask = (y == 1) | (y == 2)
    if gpd_bipd_mask.sum() > 10:
        bipd_vs_gpd_auc = float(roc_auc_score(
            (y[gpd_bipd_mask] == 2).astype(int),
            all_preds[gpd_bipd_mask, 2]
        ))
    else:
        bipd_vs_gpd_auc = None

    results = {
        'task': '3-way LPD vs GPD vs BIPD classification',
        'method': 'RF 300 trees on 18 channel probs + 11 timing features',
        'cv': '5-fold patient-stratified (class-balanced)',
        'n_patients': len(y),
        'n_lpd': int(np.sum(y == 0)),
        'n_gpd': int(np.sum(y == 1)),
        'n_bipd': int(np.sum(y == 2)),
        'n_features': int(X.shape[1]),
        'accuracy': round(accuracy, 4),
        'macro_auc': round(macro_auc, 4) if macro_auc else None,
        'per_class_auc': {k: round(v, 4) if v else None for k, v in aucs.items()},
        'bipd_vs_gpd_auc': round(bipd_vs_gpd_auc, 4) if bipd_vs_gpd_auc else None,
        'bipd_sensitivity': round(bipd_sensitivity, 4) if bipd_sensitivity else None,
        'bipd_specificity': round(bipd_specificity, 4) if bipd_specificity else None,
    }

    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Macro AUC: {macro_auc:.4f}" if macro_auc else "    Macro AUC: N/A")
    for name, auc in aucs.items():
        print(f"    {name} AUC (OVR): {auc:.4f}" if auc else f"    {name} AUC: N/A")
    print(f"    BIPD vs GPD AUC: {bipd_vs_gpd_auc:.4f}" if bipd_vs_gpd_auc else "    BIPD vs GPD AUC: N/A")
    print(f"    BIPD sensitivity: {bipd_sensitivity:.4f}" if bipd_sensitivity else "    BIPD sensitivity: N/A")
    print(f"    N: {results['n_lpd']} LPD, {results['n_gpd']} GPD, {results['n_bipd']} BIPD")

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
