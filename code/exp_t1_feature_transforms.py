"""
Experiment T1: Feature transformations before Ridge regression.

Tests three transformation strategies:
  a) Log transform: log(f + 0.01) on numeric features
  b) Interaction features: base + pairwise products (f_fft*f_tkeo, f_fft*f_B, f_tkeo*f_B)
  c) Standardized: z-score per fold (train mean/std applied to test)

Each with alpha=1.0 and alpha=5.0.

Usage:
    conda run -n foe python code/exp_t1_feature_transforms.py
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, evaluate_experiment, FEATURE_COLS


# ── Helper: median imputation ────────────────────────────────────────

def _impute_median(X_train, X_test):
    """Impute NaN with training-set column medians. Returns copies."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for j in range(X_tr.shape[1]):
        col = X_tr[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_tr[~finite, j] = med
        te_col = X_te[:, j]
        X_te[~np.isfinite(te_col), j] = med
    return X_tr, X_te


def _ridge_fit_predict(X_train, y_train_log, X_test, alpha):
    """Fit Ridge (with intercept) on log-space and return exp(predictions)."""
    X_tr_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_te_b = np.column_stack([X_test, np.ones(len(X_test))])

    I_reg = np.eye(X_tr_b.shape[1])
    I_reg[-1, -1] = 0  # Don't regularize intercept

    try:
        w = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha * I_reg,
                            X_tr_b.T @ y_train_log)
        pred_log = X_te_b @ w
        pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
        return np.exp(pred_log)
    except np.linalg.LinAlgError:
        return np.full(X_test.shape[0], np.nan)


# ── Feature column indices ───────────────────────────────────────────
# FEATURE_COLS = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd']
IDX_FB = FEATURE_COLS.index('f_B')
IDX_FFT = FEATURE_COLS.index('f_fft')
IDX_TKEO = FEATURE_COLS.index('f_tkeo')
IDX_GPD = FEATURE_COLS.index('is_gpd')
NUMERIC_COLS = [i for i in range(len(FEATURE_COLS)) if FEATURE_COLS[i] != 'is_gpd']


# ── Predict functions ────────────────────────────────────────────────

def make_log_transform_fn(alpha):
    """Log-transform numeric features: log(f + 0.01), keep is_gpd as-is."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_tr, X_te = _impute_median(train_features, test_features)
        y_log = np.log(np.clip(train_labels, 0.05, 100.0))

        # Log-transform numeric columns
        for j in NUMERIC_COLS:
            X_tr[:, j] = np.log(X_tr[:, j] + 0.01)
            X_te[:, j] = np.log(X_te[:, j] + 0.01)

        return _ridge_fit_predict(X_tr, y_log, X_te, alpha)
    return _predict


def make_interaction_fn(alpha):
    """Base features + 3 pairwise interaction terms."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_tr, X_te = _impute_median(train_features, test_features)
        y_log = np.log(np.clip(train_labels, 0.05, 100.0))

        # Compute interaction terms: f_fft*f_tkeo, f_fft*f_B, f_tkeo*f_B
        def add_interactions(X):
            i1 = (X[:, IDX_FFT] * X[:, IDX_TKEO]).reshape(-1, 1)
            i2 = (X[:, IDX_FFT] * X[:, IDX_FB]).reshape(-1, 1)
            i3 = (X[:, IDX_TKEO] * X[:, IDX_FB]).reshape(-1, 1)
            return np.hstack([X, i1, i2, i3])

        X_tr = add_interactions(X_tr)
        X_te = add_interactions(X_te)

        return _ridge_fit_predict(X_tr, y_log, X_te, alpha)
    return _predict


def make_standardized_fn(alpha):
    """Z-score numeric features per fold (train stats applied to test)."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_tr, X_te = _impute_median(train_features, test_features)
        y_log = np.log(np.clip(train_labels, 0.05, 100.0))

        # Z-score numeric columns using training stats
        for j in NUMERIC_COLS:
            mu = np.mean(X_tr[:, j])
            sd = np.std(X_tr[:, j])
            if sd < 1e-12:
                sd = 1.0
            X_tr[:, j] = (X_tr[:, j] - mu) / sd
            X_te[:, j] = (X_te[:, j] - mu) / sd

        return _ridge_fit_predict(X_tr, y_log, X_te, alpha)
    return _predict


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    experiments = [
        ('t1_ridge_log_features_a1', make_log_transform_fn(alpha=1.0)),
        ('t1_ridge_log_features_a5', make_log_transform_fn(alpha=5.0)),
        ('t1_ridge_interactions_a1', make_interaction_fn(alpha=1.0)),
        ('t1_ridge_interactions_a5', make_interaction_fn(alpha=5.0)),
        ('t1_ridge_standardized_a1', make_standardized_fn(alpha=1.0)),
        ('t1_ridge_standardized_a5', make_standardized_fn(alpha=5.0)),
    ]

    for name, predict_fn in experiments:
        evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )

    print("\nAll T1 experiments complete.")
