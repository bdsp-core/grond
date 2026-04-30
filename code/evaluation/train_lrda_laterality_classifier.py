#!/usr/bin/env python3
"""Train the LRDA laterality classifier (Plan A.2).

Loads cached features + consensus laterality labels (>=2-of-3 raters
agree), runs 5-fold patient-grouped CV, evaluates accuracy and Cohen's
kappa, and saves the trained model.

The target is binary: 0 = left, 1 = right. Consensus is computed from
MW + SZ + TZ on the 200-segment LRDA manifest (excluding rejected
segments per rater).

    conda run -n morgoth python code/evaluation/train_lrda_laterality_classifier.py

Output:
    data/labels/independent_expert_v1/lrda_laterality_classifier.pkl
    data/labels/independent_expert_v1/lrda_laterality_cv_report.txt
"""
import csv
import json
import pickle
from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
FEATURES_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'
MODEL_PATH = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_classifier.pkl'
REPORT_PATH = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_cv_report.txt'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'


def _load_rater_status_and_labels():
    """Returns (status[rater][mat_file], lat[rater][mat_file]).

    Status from raw JSONs (so we can exclude segments the rater rejected
    even if a label was somehow recorded). Lat labels from labels.csv.
    """
    status = {r: {} for r in ('MW', 'SZ', 'TZ')}
    files = [
        ('TZ/lrda_freq_labeling_results_TZ.json', 'TZ'),
        ('SZ/rda_freq_labeling_results-2.json', 'SZ'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', 'MW'),
    ]
    for rel, rater in files:
        with open(RAW_DIR / rel) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if not mf or sub != 'lrda':
                continue
            status[rater][mf] = v.get('action') or 'unknown'

    lat = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'laterality':
                continue
            v = row['value'].strip().lower()
            if v not in ('left', 'right'):
                continue
            lat[r][row['mat_file']] = v
    return status, lat


def _consensus_lat(mf, status, lat):
    """Return the consensus laterality (>=2 raters who accepted the segment
    vote the same way) or None if not enough information."""
    votes = []
    for r in ('MW', 'SZ', 'TZ'):
        if status[r].get(mf) == 'accept' and mf in lat[r]:
            votes.append(lat[r][mf])
    if len(votes) < 2:
        return None
    c = Counter(votes)
    top, count = c.most_common(1)[0]
    return top if count >= 2 else None


def load_data():
    with open(FEATURES_CSV) as f:
        rows = list(csv.DictReader(f))
    feature_names = [c for c in rows[0].keys() if c not in ('mat_file', 'patient_id')]
    status, lat = _load_rater_status_and_labels()

    X, y, groups, mat_files = [], [], [], []
    for r in rows:
        mf = r['mat_file']
        cons = _consensus_lat(mf, status, lat)
        if cons is None:
            continue
        feats = [float(r[fn]) for fn in feature_names]
        X.append(feats)
        y.append(1 if cons == 'right' else 0)
        groups.append(r['patient_id'])
        mat_files.append(mf)
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.int64),
            np.array(groups), mat_files, feature_names)


def cohen_kappa(y_true, y_pred):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(y_true, y_pred))


def main():
    X, y, groups, mat_files, feature_names = load_data()
    n = len(y)
    n_left = int((y == 0).sum())
    n_right = int((y == 1).sum())
    print(f'Loaded {n} LRDA segments with consensus laterality.')
    print(f'  Class balance: left={n_left}, right={n_right}')
    print(f'  Features: {len(feature_names)}')
    print()

    # Reference: rule-based W05 laterality from features.
    # The W05 pass-2 envelope ratio is in feature `pass2_env_log_ratio` >0 means left dominant.
    rule_pred = (np.array([X[i, feature_names.index('pass2_env_log_ratio')] for i in range(n)]) < 0).astype(int)
    rule_acc = float(np.mean(rule_pred == y))
    rule_kappa = cohen_kappa(y, rule_pred)
    print(f'Rule-based W05 laterality (current production):')
    print(f'  Accuracy:     {rule_acc:.3f}')
    print(f'  Cohen kappa:  {rule_kappa:.3f}')
    print()

    # 5-fold patient-grouped CV
    gkf = GroupKFold(n_splits=5)
    fold_oof_proba = np.zeros(n)
    fold_oof_pred = np.zeros(n, dtype=int)
    print('Running 5-fold patient-grouped CV...')
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42,
        )
        clf.fit(X[tr], y[tr])
        fold_oof_proba[te] = clf.predict_proba(X[te])[:, 1]
        fold_oof_pred[te] = clf.predict(X[te])
        fold_acc = float(np.mean(fold_oof_pred[te] == y[te]))
        print(f'  fold {k}: train={len(tr)}  test={len(te)}  test_acc={fold_acc:.3f}')

    oof_acc = float(np.mean(fold_oof_pred == y))
    oof_kappa = cohen_kappa(y, fold_oof_pred)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, fold_oof_pred)
    print()
    print(f'Out-of-fold metrics:')
    print(f'  Accuracy:     {oof_acc:.3f}  (rule was {rule_acc:.3f})')
    print(f'  Cohen kappa:  {oof_kappa:.3f}  (rule was {rule_kappa:.3f})')
    print(f'  Confusion matrix [actual L/R x predicted L/R]:')
    print(f'                  pred L  pred R')
    print(f'    actual L      {cm[0,0]:>5d}   {cm[0,1]:>5d}')
    print(f'    actual R      {cm[1,0]:>5d}   {cm[1,1]:>5d}')

    # Train final model on all data
    print()
    print('Training final model on all data...')
    final_clf = HistGradientBoostingClassifier(
        max_depth=3, max_iter=120, learning_rate=0.05,
        min_samples_leaf=5, l2_regularization=1.0, random_state=42,
    )
    final_clf.fit(X, y)

    # Permutation feature importance
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(final_clf, X, y, n_repeats=20, random_state=42, n_jobs=-1)
    print()
    print('Top 10 features by permutation importance:')
    order = np.argsort(pi.importances_mean)[::-1]
    for i in order[:10]:
        print(f'  {feature_names[i]:35s}  {pi.importances_mean[i]:+.4f}  +/- {pi.importances_std[i]:.4f}')

    bundle = {
        'model': final_clf,
        'feature_names': feature_names,
        'oof_accuracy': oof_acc,
        'oof_kappa': oof_kappa,
        'rule_accuracy': rule_acc,
        'rule_kappa': rule_kappa,
        'feature_importance_mean': dict(zip(feature_names, pi.importances_mean.tolist())),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f'\nSaved model bundle to {MODEL_PATH}')

    with open(REPORT_PATH, 'w') as f:
        f.write(f'LRDA laterality classifier — CV report\n')
        f.write(f'======================================\n\n')
        f.write(f'Total segments with consensus laterality: {n}\n')
        f.write(f'  left={n_left}, right={n_right}\n\n')
        f.write(f'Rule-based W05 (current production):\n')
        f.write(f'  Accuracy:    {rule_acc:.3f}\n')
        f.write(f'  Cohen kappa: {rule_kappa:.3f}\n\n')
        f.write(f'Learned classifier (5-fold patient-grouped OOF):\n')
        f.write(f'  Accuracy:    {oof_acc:.3f}\n')
        f.write(f'  Cohen kappa: {oof_kappa:.3f}\n\n')
        f.write(f'Confusion matrix (rows=actual, cols=predicted):\n')
        f.write(f'                  pred L  pred R\n')
        f.write(f'    actual L      {cm[0,0]:>5d}   {cm[0,1]:>5d}\n')
        f.write(f'    actual R      {cm[1,0]:>5d}   {cm[1,1]:>5d}\n\n')
        f.write(f'Top features by permutation importance:\n')
        for i in order[:15]:
            f.write(f'  {feature_names[i]:35s}  {pi.importances_mean[i]:+.4f}\n')
    print(f'Saved report to {REPORT_PATH}')


if __name__ == '__main__':
    main()
