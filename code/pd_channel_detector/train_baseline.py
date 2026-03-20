"""
Train a logistic regression baseline for channel-level PD detection.

Uses LOPO (Leave-One-Patient-Out) cross-validation, splitting by patient_id.
Reports channel-level and patient-level metrics.
"""

import sys
import time
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_features import extract_features_batch, FEATURE_NAMES

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'


def main():
    t0 = time.time()

    # Load dataset
    data_path = CACHE_DIR / 'channel_dataset.npz'
    print(f"Loading dataset from {data_path}...")
    data = np.load(str(data_path), allow_pickle=True)
    channels = data['channels']
    labels = data['labels']
    patient_ids = data['patient_ids']
    channel_indices = data['channel_indices']
    subtypes = data['subtypes']

    n_total = len(labels)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    print(f"  Channels: {n_total} (pos={n_pos}, neg={n_neg})")
    print(f"  Patients: {n_patients}")

    # Extract features
    print("Extracting features...")
    t_feat = time.time()
    features = extract_features_batch(channels)
    print(f"  Features extracted in {time.time() - t_feat:.1f}s")
    print(f"  Feature matrix shape: {features.shape}")

    # Replace any NaN/inf with 0
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # LOPO cross-validation
    print(f"\nRunning LOPO cross-validation ({n_patients} folds)...")
    all_preds = np.full(n_total, np.nan)
    all_probs = np.full(n_total, np.nan)

    n_processed = 0
    for pat in unique_patients:
        test_mask = patient_ids == pat
        train_mask = ~test_mask

        n_test = np.sum(test_mask)
        n_train = np.sum(train_mask)
        if n_test == 0 or n_train < 10:
            continue

        # Check we have both classes in training
        train_labels = labels[train_mask]
        if len(np.unique(train_labels)) < 2:
            continue

        X_train = features[train_mask]
        y_train = train_labels
        X_test = features[test_mask]

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')
        clf.fit(X_train_s, y_train)

        # Predict
        pred_labels = clf.predict(X_test_s)
        pred_probs = clf.predict_proba(X_test_s)[:, 1]

        all_preds[test_mask] = pred_labels
        all_probs[test_mask] = pred_probs

        n_processed += 1
        if n_processed % 100 == 0:
            print(f"  Processed {n_processed}/{n_patients} patients...")

    # Filter to channels that got predictions
    valid = np.isfinite(all_probs)
    y_true = labels[valid].astype(int)
    y_pred = all_preds[valid].astype(int)
    y_prob = all_probs[valid]
    pids_valid = patient_ids[valid]

    # Channel-level metrics
    ch_auc = roc_auc_score(y_true, y_prob)
    ch_acc = accuracy_score(y_true, y_pred)
    ch_bal_acc = balanced_accuracy_score(y_true, y_pred)

    # Confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Patient-level aggregation
    pat_true = []
    pat_prob = []
    for pat in np.unique(pids_valid):
        pat_mask = pids_valid == pat
        mean_prob = float(np.mean(y_prob[pat_mask]))
        pat_labels = y_true[pat_mask]
        # Patient is "PD positive" if any channel is positive
        pat_label = int(np.max(pat_labels))
        pat_true.append(pat_label)
        pat_prob.append(mean_prob)

    pat_true = np.array(pat_true)
    pat_prob = np.array(pat_prob)
    pat_pred = (pat_prob >= 0.5).astype(int)

    if len(np.unique(pat_true)) >= 2:
        pat_auc = roc_auc_score(pat_true, pat_prob)
        pat_acc = accuracy_score(pat_true, pat_pred)
        pat_bal_acc = balanced_accuracy_score(pat_true, pat_pred)
    else:
        pat_auc = pat_acc = pat_bal_acc = float('nan')

    elapsed = time.time() - t0

    # Print results
    print(f"\n{'='*60}")
    print(f"PD Channel Detector - Logistic Regression Baseline")
    print(f"{'='*60}")
    print(f"\nChannel-level metrics (N={len(y_true)}):")
    print(f"  AUC:              {ch_auc:.4f}")
    print(f"  Accuracy:         {ch_acc:.4f}")
    print(f"  Balanced Accuracy:{ch_bal_acc:.4f}")
    print(f"  Sensitivity:      {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Sensitivity: N/A")
    print(f"  Specificity:      {tn/(tn+fp):.4f}" if (tn+fp) > 0 else "  Specificity: N/A")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")

    print(f"\nPatient-level metrics (N={len(pat_true)}):")
    print(f"  AUC:              {pat_auc:.4f}")
    print(f"  Accuracy:         {pat_acc:.4f}")
    print(f"  Balanced Accuracy:{pat_bal_acc:.4f}")

    # Feature importance (from last fold's model coefficients)
    if hasattr(clf, 'coef_'):
        print(f"\nFeature importance (last fold's coefficients):")
        coefs = clf.coef_[0]
        sorted_idx = np.argsort(np.abs(coefs))[::-1]
        for idx in sorted_idx:
            print(f"  {FEATURE_NAMES[idx]:>20s}: {coefs[idx]:+.4f}")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
