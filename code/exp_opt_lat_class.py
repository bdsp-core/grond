"""
Laterality classification experiments: left vs right for LPD patients.

Tests different classifiers, feature subsets, and class balancing via LOPO CV.

Run: conda run -n foe python code/exp_opt_lat_class.py
"""

import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_laterality_classification,
    ALL_FEATURE_COLS, FEATURE_COLS, LATERALITY_FEATURE_COLS,
    _build_segment_level_data, RUNS_DIR,
)

CACHE_PATH = PROJECT_DIR / 'data' / 'dl_cache' / 'dataset_cache.pkl'


def load_dataset_cached(verbose=True):
    """Load dataset with disk caching to avoid recomputing features."""
    import pickle
    if CACHE_PATH.exists():
        if verbose:
            print(f"Loading cached dataset from {CACHE_PATH}...")
        with open(str(CACHE_PATH), 'rb') as f:
            dataset = pickle.load(f)
        if verbose:
            print(f"  Loaded {len(dataset['df'])} patients from cache.")
        return dataset
    else:
        if verbose:
            print("No cache found, computing features from scratch...")
        dataset = load_dataset(verbose=verbose)
        with open(str(CACHE_PATH), 'wb') as f:
            pickle.dump(dataset, f)
        if verbose:
            print(f"  Saved cache to {CACHE_PATH}")
        return dataset


# ── Generic LOPO laterality classifier ────────────────────────────────

def run_laterality_experiment(dataset, experiment_name, feat_col_names, classifier_fn):
    """
    LOPO laterality classification with configurable features and classifier.

    classifier_fn(X_train, y_train, X_test) -> test_probs (array of probs for test segments)
    """
    t0 = time.time()
    print(f"\nRunning laterality classification: {experiment_name}")

    df = dataset['df']

    lat_map = {'left': 0, 'right': 1}
    eligible = df[df['laterality'].isin(['left', 'right'])].copy()

    if len(eligible) < 10:
        print(f"  Only {len(eligible)} eligible patients -- skipping.")
        return {}

    eligible_pids = set(eligible['patient_id'].values)
    pid_to_lat = dict(zip(eligible['patient_id'], eligible['laterality'].map(lat_map)))

    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    feat_indices = [ALL_FEATURE_COLS.index(c) for c in feat_col_names]

    eligible_mask = np.array([p in eligible_pids for p in seg_pids])
    seg_pids_e = seg_pids[eligible_mask]
    seg_features_e = seg_features[eligible_mask]
    seg_lat = np.array([pid_to_lat.get(p, -1) for p in seg_pids_e])

    unique_patients = eligible['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids_e == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_features_e[train_mask][:, feat_indices].copy()
        y_train = seg_lat[train_mask]
        X_test = seg_features_e[test_mask][:, feat_indices].copy()

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        test_probs = classifier_fn(X_train, y_train, X_test)
        patient_preds[pat] = float(np.mean(test_probs))

    # Aggregate
    y_true, y_prob = [], []
    for _, row in eligible.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(pid_to_lat[pid])
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(np.mean(y_true == y_pred))
    n = len(y_true)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (sens + spec) / 2

    # AUC (trapezoidal)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos > 0 and n_neg > 0:
        sorted_idx = np.argsort(-y_prob)
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
        auc = float(np.trapz(tpr_list, fpr_list))
    else:
        auc = float('nan')

    metrics = {
        'experiment': experiment_name,
        'task': 'laterality_classification',
        'timestamp': time.time(),
        'n_patients': n,
        'n_left': int(np.sum(y_true == 0)),
        'n_right': int(np.sum(y_true == 1)),
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'sensitivity_right': round(float(sens), 4),
        'specificity_left': round(float(spec), 4),
        'auc': round(auc, 4) if np.isfinite(auc) else None,
        'confusion_matrix': {'tp_right': tp, 'tn_left': tn, 'fp': fp, 'fn': fn},
        'pred_probs': [round(v, 4) for v in y_prob.tolist()],
        'true_labels': y_true.tolist(),
    }

    out_path = RUNS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"LATERALITY CLASSIFICATION: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"  N={n} (left={metrics['n_left']}, right={metrics['n_right']})")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Sens (right):      {sens:.3f}")
    print(f"  Spec (left):       {spec:.3f}")
    if np.isfinite(auc):
        print(f"  AUC:               {auc:.3f}")
    else:
        print(f"  AUC:               N/A")
    print(f"  Confusion: TP(R)={tp} TN(L)={tn} FP={fp} FN={fn}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


# ── Classifier factories ──────────────────────────────────────────────

def make_ridge_logreg(alpha=1.0, n_iter=5):
    """IRLS ridge logistic regression."""
    def _classify(X_train, y_train, X_test):
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
        w = np.zeros(X_train_b.shape[1])
        for _ in range(n_iter):
            logits = X_train_b @ w
            logits = np.clip(logits, -10, 10)
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            W_diag = p * (1 - p)
            z = logits + (y_train - p) / W_diag
            W_X = X_train_b * W_diag[:, None]
            try:
                w = np.linalg.solve(
                    W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                    W_X.T @ z
                )
            except np.linalg.LinAlgError:
                break
        test_logits = X_test_b @ w
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -10, 10)))
        return test_probs
    return _classify


def make_gbm_classifier(n_estimators=100, max_depth=3, learning_rate=0.1):
    """GradientBoostingClassifier wrapper."""
    def _classify(X_train, y_train, X_test):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    return _classify


def make_gbm_balanced_classifier(n_estimators=100, max_depth=3, learning_rate=0.1):
    """GBM with sample weights inversely proportional to class frequency."""
    def _classify(X_train, y_train, X_test):
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, random_state=42,
        )
        # Compute inverse class frequency weights
        classes, counts = np.unique(y_train, return_counts=True)
        total = len(y_train)
        weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sw = np.array([weight_map[yi] for yi in y_train])
        clf.fit(X_train, y_train, sample_weight=sw)
        return clf.predict_proba(X_test)[:, 1]
    return _classify


def make_rf_classifier(n_estimators=200, max_depth=5, min_samples_leaf=5,
                       class_weight=None):
    """RandomForestClassifier wrapper."""
    def _classify(X_train, y_train, X_test):
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, class_weight=class_weight,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)[:, 1]
    return _classify


# ── Feature sets ──────────────────────────────────────────────────────

LAT_FREQ_FEATS = LATERALITY_FEATURE_COLS + ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']
LAT_ONLY_FEATS = list(LATERALITY_FEATURE_COLS)  # lat_idx, lat_energy_ratio, lat_acf_ratio
LAT_IDX_ONLY = ['lat_idx']


def update_dashboard():
    print("  Updating dashboard...")
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_dataset_cached(verbose=True)

    experiments = [
        # 1: Logistic Ridge alpha=1, lat+freq features
        ('lat_logreg_a1', LAT_FREQ_FEATS, make_ridge_logreg(alpha=1.0)),
        # 2: Logistic Ridge alpha=0.1
        ('lat_logreg_a01', LAT_FREQ_FEATS, make_ridge_logreg(alpha=0.1)),
        # 3: Logistic Ridge alpha=5
        ('lat_logreg_a5', LAT_FREQ_FEATS, make_ridge_logreg(alpha=5.0)),
        # 4: GBM(100, depth=3, lr=0.1)
        ('lat_gbm', LAT_FREQ_FEATS, make_gbm_classifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
        # 5: RF(200, depth=5, leaf=5)
        ('lat_rf', LAT_FREQ_FEATS, make_rf_classifier(n_estimators=200, max_depth=5, min_samples_leaf=5)),
        # 6: Only lat_idx feature, Ridge alpha=1
        ('lat_idx_only', LAT_IDX_ONLY, make_ridge_logreg(alpha=1.0)),
        # 7: Only laterality features (lat_idx, lat_energy_ratio, lat_acf_ratio)
        ('lat_only_feats', LAT_ONLY_FEATS, make_ridge_logreg(alpha=1.0)),
        # 8: GBM with balanced class weights
        ('lat_gbm_bal', LAT_FREQ_FEATS, make_gbm_balanced_classifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
        # 9: RF with class_weight='balanced'
        ('lat_rf_bal', LAT_FREQ_FEATS, make_rf_classifier(n_estimators=200, max_depth=5, min_samples_leaf=5, class_weight='balanced')),
    ]

    all_results = []
    for i, (name, feats, clf_fn) in enumerate(experiments, 1):
        # Skip experiments that already have a JSON result
        existing_path = RUNS_DIR / f'{name}.json'
        if existing_path.exists():
            print(f"\n  SKIP {i}/9: {name} -- JSON already exists at {existing_path}")
            with open(str(existing_path)) as f:
                all_results.append(json.load(f))
            continue

        print(f"\n{'='*72}")
        print(f"EXPERIMENT {i}/9: {name}")
        print(f"{'='*72}")
        m = run_laterality_experiment(dataset, name, feats, clf_fn)
        all_results.append(m)
        update_dashboard()

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Laterality Classification Experiments")
    print(f"{'='*80}")
    print(f"{'Experiment':<22s} {'Acc':>6s} {'BalAcc':>7s} {'AUC':>6s} {'Sens(R)':>8s} {'Spec(L)':>8s}")
    print(f"{'-'*22} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*8}")
    for m in all_results:
        if not m:
            continue
        auc_str = f"{m['auc']:.3f}" if m.get('auc') is not None else "N/A"
        print(f"{m['experiment']:<22s} {m['accuracy']:>6.3f} {m['balanced_accuracy']:>7.3f} "
              f"{auc_str:>6s} {m['sensitivity_right']:>8.3f} {m['specificity_left']:>8.3f}")
    print(f"{'='*80}")

    print("\n\nAll 9 laterality classification experiments complete!")
