#!/usr/bin/env python3
"""
Evaluate LPD vs GPD subtype classification using ChannelPD-Net features.

Uses per-channel PD probabilities from predictions.json as features
for a Random Forest classifier with Leave-One-Patient-Out CV.

Saves results to data/evaluation_results/subtype_classification.json.

Usage:
    conda run -n morgoth python code/evaluation/eval_subtype_classification.py
"""

import csv
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
PREDICTIONS_PATH = LABELS_DIR / 'predictions.json'
RESULTS_DIR = PROJECT_DIR / 'data' / 'evaluation_results'
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = RESULTS_DIR / 'subtype_classification.json'


def main():
    print("Evaluating LPD vs GPD subtype classification...")

    # Load predictions
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

    # Group by patient
    patients = defaultdict(list)
    for mat, info in segments.items():
        patients[info['patient_id']].append((mat, info))

    print(f"  {len(patients)} unique patients")

    # 5-fold patient-stratified CV (much faster than LOPO for 7000+ patients)
    from sklearn.model_selection import StratifiedGroupKFold

    print("  Running 5-fold patient-stratified CV...")

    # Build arrays at patient level (one entry per patient, using mean probs)
    patient_ids = sorted(patients.keys())
    X_all, y_all, groups = [], [], []
    for pid in patient_ids:
        segs = patients[pid]
        # Average channel probs across segments for this patient
        mean_probs = np.mean([info['channel_probs'] for _, info in segs], axis=0)
        label = 1 if segs[0][1]['subtype'] == 'gpd' else 0
        X_all.append(mean_probs)
        y_all.append(label)
        groups.append(pid)

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    groups = np.array(groups)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    all_probs = []
    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X_all, y_all, groups)):
        print(f"    Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_all[train_idx], y_all[train_idx])
        probs = rf.predict_proba(X_all[test_idx])[:, 1]

        for idx, prob in zip(test_idx, probs):
            all_probs.append((patient_ids[idx], int(y_all[idx]), float(prob)))

    # Compute AUC
    labels = np.array([p[1] for p in all_probs])
    scores = np.array([p[2] for p in all_probs])
    auc = roc_auc_score(labels, scores)

    # Accuracy at 0.5 threshold
    preds = (scores >= 0.5).astype(int)
    accuracy = float(np.mean(preds == labels))
    sensitivity = float(np.mean(preds[labels == 1] == 1)) if np.sum(labels == 1) > 0 else 0
    specificity = float(np.mean(preds[labels == 0] == 0)) if np.sum(labels == 0) > 0 else 0

    results = {
        'task': 'LPD vs GPD subtype classification',
        'method': 'RF 300 trees on ChannelPD-Net channel probabilities',
        'cv': '5-fold patient-stratified',
        'n_patients': len(all_probs),
        'n_lpd': int(np.sum(labels == 0)),
        'n_gpd': int(np.sum(labels == 1)),
        'n_segments': len(segments),
        'auc': round(auc, 4),
        'accuracy': round(accuracy, 4),
        'sensitivity_gpd': round(sensitivity, 4),
        'specificity_lpd': round(specificity, 4),
        'features': '18 per-channel PD probabilities from ChannelPD-Net',
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


if __name__ == '__main__':
    main()
