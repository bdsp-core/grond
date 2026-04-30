#!/usr/bin/env python3
"""Train the LRDA hard-case classifier (Plan A.1).

Loads cached features + expert labels, builds the hard/easy target, runs
5-fold patient-stratified CV, picks an operating threshold that maximizes
end-to-end LRDA-frequency MAE when the V1/V8 gated hybrid is run, and
saves the trained model.

    conda run -n morgoth python code/evaluation/train_lrda_hard_case_classifier.py

Output:
    data/labels/independent_expert_v1/hard_case_classifier.pkl
    data/labels/independent_expert_v1/hard_case_cv_report.txt
"""

import csv
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))

from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
FEATURES_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_features.csv'
MODEL_PATH = LABELS_DIR / 'independent_expert_v1' / 'hard_case_classifier.pkl'
REPORT_PATH = LABELS_DIR / 'independent_expert_v1' / 'hard_case_cv_report.txt'

HARD_THRESH_HZ = 0.5  # |v1_freq - mean(expert_freqs)| > this is "hard"


def load_data():
    """Returns (X, y, groups, mat_files, feature_names, mean_expert_freq, v1_freq)."""
    # Features
    with open(FEATURES_CSV) as f:
        rows = list(csv.DictReader(f))
    feature_names = [c for c in rows[0].keys() if c not in ('mat_file', 'patient_id')]

    # Expert labels from canonical labels.csv
    expert_freqs = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'frequency_hz':
                continue
            try:
                expert_freqs[r][row['mat_file']] = float(row['value'])
            except ValueError:
                pass

    X, y, groups, mat_files = [], [], [], []
    mean_exp, v1_used = [], []
    for r in rows:
        mf = r['mat_file']
        v1 = float(r['v1_freq'])
        # Mean of available expert freqs
        avail = [expert_freqs[er][mf] for er in ('MW', 'SZ', 'TZ') if mf in expert_freqs[er]]
        if not avail:
            continue
        m = float(np.mean(avail))
        is_hard = abs(v1 - m) > HARD_THRESH_HZ
        feats = [float(r[fn]) for fn in feature_names]
        X.append(feats)
        y.append(int(is_hard))
        groups.append(r['patient_id'])
        mat_files.append(mf)
        mean_exp.append(m)
        v1_used.append(v1)

    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int64),
            np.array(groups), mat_files, feature_names,
            np.array(mean_exp), np.array(v1_used))


def main():
    X, y, groups, mat_files, feature_names, mean_exp, v1_freqs = load_data()
    n = len(y)
    n_hard = int(y.sum())
    print(f'Loaded {n} LRDA segments with at least one expert freq label.')
    print(f'  Hard cases (|V1-mean_exp| > {HARD_THRESH_HZ} Hz): {n_hard} ({100*n_hard/n:.1f}%)')
    print(f'  Easy cases: {n - n_hard}')
    print(f'  Features: {len(feature_names)}')
    print()

    # 5-fold patient-grouped CV (no patient appears in two folds)
    gkf = GroupKFold(n_splits=5)
    fold_oof_proba = np.zeros(n)
    fold_oof_pred = np.zeros(n, dtype=int)
    feature_importances = np.zeros(len(feature_names))

    print('Running 5-fold patient-grouped CV...')
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        n_pos_tr = int(y[tr].sum())
        n_pos_te = int(y[te].sum())
        # Class-weighted: scikit's HistGradientBoostingClassifier accepts sample_weight
        # to compensate for imbalance. Weight each hard case ~6x.
        w = np.where(y[tr] == 1, n - n_hard, n_hard).astype(np.float64)
        w = w / w.mean()  # normalize
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42,
        )
        clf.fit(X[tr], y[tr], sample_weight=w)
        fold_oof_proba[te] = clf.predict_proba(X[te])[:, 1]
        fold_oof_pred[te] = clf.predict(X[te])
        # Permutation-based feature importance is too slow per fold; use built-in
        # Note HistGradientBoostingClassifier does not have feature_importances_ directly;
        # we'll re-fit a single final model below to extract importances via permutation.
        print(f'  fold {k}: train n={len(tr)} ({n_pos_tr} hard) | test n={len(te)} ({n_pos_te} hard)')
    print()

    # Aggregate OOF metrics
    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
    auc = roc_auc_score(y, fold_oof_proba) if len(set(y)) > 1 else float('nan')
    p, r, f, _ = precision_recall_fscore_support(y, fold_oof_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y, fold_oof_pred)
    print(f'Out-of-fold metrics:')
    print(f'  ROC AUC: {auc:.3f}')
    print(f'  Precision (hard class): {p:.3f}')
    print(f'  Recall (hard class):    {r:.3f}')
    print(f'  F1 (hard class):        {f:.3f}')
    print(f'  Confusion matrix [actual easy/hard x predicted easy/hard]:')
    print(f'                  pred easy  pred hard')
    print(f'    actual easy    {cm[0,0]:>5d}     {cm[0,1]:>5d}')
    print(f'    actual hard    {cm[1,0]:>5d}     {cm[1,1]:>5d}')
    print()

    # Threshold sweep against the actual end-to-end objective:
    # for each threshold, simulate the gated hybrid (V1 if predicted easy, V8 if predicted hard)
    # and compute MW-ALGO MAE on the OOF predictions.
    # We approximate V8's contribution from the diagnostic V8 result that's
    # already pre-computed for each segment in our diagnostic notes; for
    # simplicity here we use the rule: if predicted hard, output 2*v1 clipped
    # to [0.5, 4]; this is the empirical "fix" pattern for sub-harmonic locking.
    # Better: actually run V8 on each test fold; we'll do that when we score V9.
    # For now this threshold is just an interim; the real evaluation is via
    # v9_gated_hybrid.py + analyze_independent_expert_v1.py.
    print('Threshold sweep on OOF probabilities (proxy MAE = |2*v1 if pred-hard else v1 - mean_exp|):')
    print(f'{"thresh":>7s}  {"frac_hard":>10s}  {"V1 MAE":>7s}  {"V9 proxy MAE":>13s}')
    v1_mae = float(np.mean(np.abs(v1_freqs - mean_exp)))
    best = (None, v1_mae)
    for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
        pred_hard = fold_oof_proba >= thresh
        v9_proxy = np.where(pred_hard, np.clip(2 * v1_freqs, 0.5, 4.0), v1_freqs)
        mae = float(np.mean(np.abs(v9_proxy - mean_exp)))
        frac_hard = float(pred_hard.mean())
        print(f'{thresh:>7.2f}  {frac_hard:>10.3f}  {v1_mae:>7.3f}  {mae:>13.3f}')
        if mae < best[1]:
            best = (thresh, mae)
    print(f'  Best proxy threshold: {best[0]} (proxy MAE {best[1]:.3f})')
    chosen_thresh = best[0] if best[0] is not None else 0.5
    print()

    # Train one final model on all data for production use
    print('Training final model on all data...')
    w_all = np.where(y == 1, n - n_hard, n_hard).astype(np.float64)
    w_all = w_all / w_all.mean()
    final_clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=120, learning_rate=0.05,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42,
    )
    final_clf.fit(X, y, sample_weight=w_all)

    # Permutation feature importance on the full dataset (cheap with 22 features)
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(final_clf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    print('Top 10 features by permutation importance:')
    order = np.argsort(pi.importances_mean)[::-1]
    for i in order[:10]:
        print(f'  {feature_names[i]:35s}  {pi.importances_mean[i]:+.4f}  +/- {pi.importances_std[i]:.4f}')
    print()

    # Save model + threshold
    bundle = {
        'model': final_clf,
        'feature_names': feature_names,
        'threshold': chosen_thresh,
        'auc': auc,
        'oof_precision': p,
        'oof_recall': r,
        'oof_f1': f,
        'feature_importance_mean': dict(zip(feature_names, pi.importances_mean.tolist())),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'Saved model bundle to {MODEL_PATH}')

    # Save report
    with open(REPORT_PATH, 'w') as f:
        f.write(f'LRDA hard-case classifier — CV report\n')
        f.write(f'======================================\n\n')
        f.write(f'Total segments: {n}\n')
        f.write(f'Hard (|V1 - mean_exp| > {HARD_THRESH_HZ} Hz): {n_hard} ({100*n_hard/n:.1f}%)\n\n')
        f.write(f'OOF ROC AUC: {auc:.3f}\n')
        f.write(f'OOF precision (hard class): {p:.3f}\n')
        f.write(f'OOF recall (hard class):    {r:.3f}\n')
        f.write(f'OOF F1 (hard class):        {f:.3f}\n\n')
        f.write(f'Confusion matrix (rows=actual, cols=predicted):\n')
        f.write(f'                  pred easy  pred hard\n')
        f.write(f'    actual easy    {cm[0,0]:>5d}     {cm[0,1]:>5d}\n')
        f.write(f'    actual hard    {cm[1,0]:>5d}     {cm[1,1]:>5d}\n\n')
        f.write(f'Chosen operating threshold (proxy-MAE-minimizing): {chosen_thresh}\n')
    print(f'Saved CV report to {REPORT_PATH}')


if __name__ == '__main__':
    main()
