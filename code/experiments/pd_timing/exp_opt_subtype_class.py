"""
Subtype classification experiments: LPD vs GPD.

Tests different classifiers, regularization strengths, feature subsets,
and class balancing strategies for LOPO subtype classification.

Run: conda run -n foe python code/exp_opt_subtype_class.py
"""

import sys
import json
import time
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimization_harness_v2 import (
    load_dataset, evaluate_subtype_classification,
    ALL_FEATURE_COLS, FEATURE_COLS, LATERALITY_FEATURE_COLS,
    _build_segment_level_data, RUNS_DIR,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
CACHE_PATH = PROJECT_DIR / 'data' / 'dl_cache' / 'dataset_cache.npz'


def load_dataset_cached(verbose=True):
    """Load dataset with disk caching to avoid recomputing features."""
    import pickle
    cache_pkl = CACHE_PATH.with_suffix('.pkl')
    if cache_pkl.exists():
        if verbose:
            print(f"Loading cached dataset from {cache_pkl}...")
        with open(str(cache_pkl), 'rb') as f:
            dataset = pickle.load(f)
        if verbose:
            print(f"  Loaded {len(dataset['df'])} patients from cache.")
        return dataset
    else:
        if verbose:
            print("No cache found, computing features from scratch...")
        dataset = load_dataset(verbose=verbose)
        with open(str(cache_pkl), 'wb') as f:
            pickle.dump(dataset, f)
        if verbose:
            print(f"  Saved cache to {cache_pkl}")
        return dataset


# ── Helpers ───────────────────────────────────────────────────────────

def _get_feat_indices(feature_names):
    """Get column indices into ALL_FEATURE_COLS for given feature names, excluding is_gpd."""
    return [ALL_FEATURE_COLS.index(c) for c in feature_names if c != 'is_gpd']


def _impute_and_bias(X_train, X_test):
    """NaN imputation with training median, then add intercept column."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_train[~finite, j] = med
        X_test[~np.isfinite(X_test[:, j]), j] = med
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
    return X_train_b, X_test_b


def _impute(X_train, X_test):
    """NaN imputation with training median (no intercept)."""
    X_train = X_train.copy()
    X_test = X_test.copy()
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_train[~finite, j] = med
        X_test[~np.isfinite(X_test[:, j]), j] = med
    return X_train, X_test


def _ridge_logistic(X_train_b, y_train, alpha, n_iter=5):
    """IRLS ridge logistic regression. Returns weight vector."""
    w = np.zeros(X_train_b.shape[1])
    for _ in range(n_iter):
        logits = np.clip(X_train_b @ w, -10, 10)
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        W_diag = p * (1 - p)
        z = logits + (y_train - p) / W_diag
        W_X = X_train_b * W_diag[:, None]
        try:
            w = np.linalg.solve(
                W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                W_X.T @ z,
            )
        except np.linalg.LinAlgError:
            break
    return w


def _compute_auc(y_true, y_prob):
    """Trapezoidal AUC."""
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
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


def _compute_and_save_metrics(experiment_name, y_true, y_prob, elapsed):
    """Compute classification metrics, print, save JSON, return dict."""
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (sens + spec) / 2
    auc = _compute_auc(y_true, y_prob)

    metrics = {
        'experiment': experiment_name,
        'task': 'subtype_classification',
        'timestamp': time.time(),
        'n_patients': n,
        'n_lpd': int(np.sum(y_true == 0)),
        'n_gpd': int(np.sum(y_true == 1)),
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'auc': round(float(auc), 4) if np.isfinite(auc) else None,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
    }

    out_path = RUNS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*72}")
    print(f"SUBTYPE CLASSIFICATION: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"  N={n} (LPD={metrics['n_lpd']}, GPD={metrics['n_gpd']})")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Sensitivity (GPD): {sens:.3f}")
    print(f"  Specificity (LPD): {spec:.3f}")
    if np.isfinite(auc):
        print(f"  AUC:               {auc:.3f}")
    else:
        print(f"  AUC:               N/A")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


def update_dashboard():
    subprocess.run(['python', 'code/update_dashboard_v2.py'],
                   cwd=str(PROJECT_DIR))


# ── Generic LOPO logistic ridge experiment ────────────────────────────

def run_logreg_experiment(dataset, experiment_name, alpha, feat_indices):
    """LOPO logistic ridge with configurable alpha and features."""
    t0 = time.time()
    print(f"\nRunning subtype classification: {experiment_name}")

    df = dataset['df']
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))
    seg_subtypes = np.array([1 if pid_to_subtype.get(p) == 'gpd' else 0 for p in seg_pids])

    unique_patients = df['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_features[train_mask][:, feat_indices]
        y_train = seg_subtypes[train_mask]
        X_test = seg_features[test_mask][:, feat_indices]

        X_train_b, X_test_b = _impute_and_bias(X_train, X_test)
        w = _ridge_logistic(X_train_b, y_train, alpha)

        test_logits = X_test_b @ w
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -10, 10)))
        patient_preds[pat] = float(np.mean(test_probs))

    # Aggregate at patient level
    y_true, y_prob = [], []
    for _, row in df.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(1 if row['subtype'] == 'gpd' else 0)
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    elapsed = time.time() - t0

    return _compute_and_save_metrics(experiment_name, y_true, y_prob, elapsed)


# ── Sklearn-based experiments ─────────────────────────────────────────

def run_sklearn_experiment(dataset, experiment_name, clf, feat_indices=None,
                           sample_weight_fn=None):
    """LOPO classification using a scikit-learn classifier.

    Args:
        feat_indices: column indices to use. If None, uses all except is_gpd.
        sample_weight_fn: if provided, called with y_train to produce sample_weight array.
    """
    t0 = time.time()
    print(f"\nRunning subtype classification: {experiment_name}")

    df = dataset['df']
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))
    seg_subtypes = np.array([1 if pid_to_subtype.get(p) == 'gpd' else 0 for p in seg_pids])

    # Default: all features except is_gpd
    if feat_indices is None:
        is_gpd_idx = ALL_FEATURE_COLS.index('is_gpd')
        feat_indices = [i for i in range(len(ALL_FEATURE_COLS)) if i != is_gpd_idx]

    unique_patients = df['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_features[train_mask][:, feat_indices].copy()
        y_train = seg_subtypes[train_mask]
        X_test = seg_features[test_mask][:, feat_indices].copy()

        X_train, X_test = _impute(X_train, X_test)

        try:
            from sklearn.base import clone
            model = clone(clf)
            if sample_weight_fn is not None:
                sw = sample_weight_fn(y_train)
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)
            test_probs = model.predict_proba(X_test)[:, 1]
            patient_preds[pat] = float(np.mean(test_probs))
        except Exception as e:
            print(f"  Warning: failed for patient {pat}: {e}")
            continue

    # Aggregate
    y_true, y_prob = [], []
    for _, row in df.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(1 if row['subtype'] == 'gpd' else 0)
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    elapsed = time.time() - t0

    return _compute_and_save_metrics(experiment_name, y_true, y_prob, elapsed)


def _inverse_class_weight(y):
    """Compute sample weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y])


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

    print("Loading dataset...")
    dataset = load_dataset_cached(verbose=True)

    # Feature index sets (excluding is_gpd)
    all_feat_idx = _get_feat_indices(ALL_FEATURE_COLS)
    freq_feat_names = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']
    freq_feat_idx = _get_feat_indices(freq_feat_names)

    # ── 1: sub_logreg_a1 ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 1/9: sub_logreg_a1 (Logistic Ridge, alpha=1, all feats except is_gpd)")
    print("="*72)
    run_logreg_experiment(dataset, 'sub_logreg_a1', alpha=1.0, feat_indices=all_feat_idx)
    update_dashboard()

    # ── 2: sub_rf ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 2/9: sub_rf (RandomForest 200, depth=5, min_leaf=5)")
    print("="*72)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42,
    )
    run_sklearn_experiment(dataset, 'sub_rf', rf)
    update_dashboard()

    # ── 3: sub_gbm ───────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 3/9: sub_gbm (GBM 100, depth=3, lr=0.1)")
    print("="*72)
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    run_sklearn_experiment(dataset, 'sub_gbm', gbm)
    update_dashboard()

    # ── 4: sub_rf_bal ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 4/9: sub_rf_bal (RF with class_weight='balanced')")
    print("="*72)
    rf_bal = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        class_weight='balanced', random_state=42,
    )
    run_sklearn_experiment(dataset, 'sub_rf_bal', rf_bal)
    update_dashboard()

    # ── 5: sub_gbm_bal ───────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 5/9: sub_gbm_bal (GBM with inverse class freq sample_weight)")
    print("="*72)
    gbm_bal = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    run_sklearn_experiment(dataset, 'sub_gbm_bal', gbm_bal,
                           sample_weight_fn=_inverse_class_weight)
    update_dashboard()

    # ── 6: sub_freq_only ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 6/9: sub_freq_only (Logistic Ridge alpha=1, freq features only)")
    print("="*72)
    run_logreg_experiment(dataset, 'sub_freq_only', alpha=1.0, feat_indices=freq_feat_idx)
    update_dashboard()

    # ── 7: sub_rf_300 ────────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 7/9: sub_rf_300 (RF 300, depth=8, min_leaf=3)")
    print("="*72)
    rf_300 = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=42,
    )
    run_sklearn_experiment(dataset, 'sub_rf_300', rf_300)
    update_dashboard()

    # ── 8: sub_logreg_a01 ────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 8/9: sub_logreg_a01 (Logistic Ridge, alpha=0.1)")
    print("="*72)
    run_logreg_experiment(dataset, 'sub_logreg_a01', alpha=0.1, feat_indices=all_feat_idx)
    update_dashboard()

    # ── 9: sub_logreg_a5 ─────────────────────────────────────────────
    print("\n" + "="*72)
    print("EXPERIMENT 9/9: sub_logreg_a5 (Logistic Ridge, alpha=5)")
    print("="*72)
    run_logreg_experiment(dataset, 'sub_logreg_a5', alpha=5.0, feat_indices=all_feat_idx)
    update_dashboard()

    print("\n\nAll 9 subtype classification experiments complete!")
