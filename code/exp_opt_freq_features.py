"""
Experiment: Frequency feature sets with standard and frequency-balanced training.

Tests different feature subsets and interaction terms, each with both
unweighted and frequency-balanced (inverse-frequency-bin weighted) Ridge.

Frequency-balanced training uses inverse-frequency-bin weights:
  Bins: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
  Weight = 1/(count in bin), normalized so sum(weights) = N
  Weighted Ridge: (X'WX + alpha*I)w = X'Wy

Experiments (10):
  1. feat_base6_a1          - 6 base features, Ridge alpha=1 (unweighted)
  2. feat_base6_a1_bal      - 6 base features, Ridge alpha=1, freq-balanced
  3. feat_base5_no_gpd_a1   - 5 features without is_gpd, alpha=1
  4. feat_base5_no_gpd_a1_bal - same, freq-balanced
  5. feat_all9_a1           - All 9 features, alpha=1
  6. feat_all9_a1_bal       - same, freq-balanced
  7. feat_interact_a1       - 6 base + 3 interactions, alpha=1
  8. feat_interact_a1_bal   - same, freq-balanced
  9. feat_base6_a5_bal      - 6 base, alpha=5, freq-balanced
  10. feat_interact_a5_bal   - interactions, alpha=5, freq-balanced

Usage:
    conda run -n foe python code/exp_opt_freq_features.py
"""

import sys
import subprocess
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment,
    ALL_FEATURE_COLS, FEATURE_COLS, LATERALITY_FEATURE_COLS,
)


def update_dashboard():
    """Update the dashboard after each experiment."""
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


# ── Frequency-bin weighting ──────────────────────────────────────────

FREQ_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]


def compute_freq_weights(labels):
    """Compute inverse-frequency-bin weights for training samples.

    Bins: [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    Weight = 1/(count in bin), normalized so sum(weights) = N.
    """
    n = len(labels)
    bin_indices = np.digitize(labels, FREQ_BINS) - 1
    bin_indices = np.clip(bin_indices, 0, len(FREQ_BINS) - 2)

    # Count samples per bin
    bin_counts = np.bincount(bin_indices, minlength=len(FREQ_BINS) - 1)

    # Assign inverse-count weight per sample
    weights = np.zeros(n)
    for i in range(n):
        b = bin_indices[i]
        weights[i] = 1.0 / max(bin_counts[b], 1)

    # Normalize so sum = N
    weights = weights * (n / np.sum(weights))
    return weights


# ── Predict functions ────────────────────────────────────────────────

def _impute_and_intercept(X_train, X_test):
    """Impute NaN with training median and add intercept column."""
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_train[~finite, j] = med
        X_test[~np.isfinite(X_test[:, j]), j] = med

    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
    return X_train_b, X_test_b


def _ridge_solve(X_train_b, y_train, X_test_b, alpha, sample_weights=None):
    """Solve Ridge (optionally weighted) and return predictions in original freq space."""
    I_reg = np.eye(X_train_b.shape[1])
    I_reg[-1, -1] = 0  # Don't regularize intercept

    try:
        if sample_weights is not None:
            W = np.diag(sample_weights)
            w = np.linalg.solve(X_train_b.T @ W @ X_train_b + alpha * I_reg,
                                X_train_b.T @ W @ y_train)
        else:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
        pred_log = X_test_b @ w
        pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
        return np.exp(pred_log)
    except np.linalg.LinAlgError:
        return np.full(X_test_b.shape[0], np.nan)


def make_column_select_predict_fn(col_names, alpha=1.0, balanced=False):
    """Return a predict_fn that selects specific columns and runs Ridge."""
    col_indices = [ALL_FEATURE_COLS.index(c) for c in col_names]

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_train = train_features[:, col_indices].copy()
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        X_test = test_features[:, col_indices].copy()

        X_train_b, X_test_b = _impute_and_intercept(X_train, X_test)

        weights = compute_freq_weights(train_labels) if balanced else None
        return _ridge_solve(X_train_b, y_train, X_test_b, alpha, weights)

    return _predict


def make_interaction_predict_fn(alpha=1.0, balanced=False):
    """Ridge with 6 base features + 3 interaction terms (f_fft*f_tkeo, f_fft*f_B, f_tkeo*f_B)."""
    base_indices = [ALL_FEATURE_COLS.index(c) for c in FEATURE_COLS]

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        def _build_X(feats):
            X_base = feats[:, base_indices].copy()
            # Impute before computing interactions
            for j in range(X_base.shape[1]):
                col = X_base[:, j]
                finite = np.isfinite(col)
                med = np.median(col[finite]) if np.any(finite) else 0.0
                X_base[~finite, j] = med

            base_names = list(FEATURE_COLS)
            idx_fft = base_names.index('f_fft')
            idx_tkeo = base_names.index('f_tkeo')
            idx_B = base_names.index('f_B')

            inter1 = (X_base[:, idx_fft] * X_base[:, idx_tkeo]).reshape(-1, 1)
            inter2 = (X_base[:, idx_fft] * X_base[:, idx_B]).reshape(-1, 1)
            inter3 = (X_base[:, idx_tkeo] * X_base[:, idx_B]).reshape(-1, 1)
            return np.hstack([X_base, inter1, inter2, inter3])

        X_train = _build_X(train_features)
        X_test = _build_X(test_features)

        # Final impute pass (interactions should be clean, but be safe)
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        weights = compute_freq_weights(train_labels) if balanced else None
        return _ridge_solve(X_train_b, y_train, X_test_b, alpha, weights)

    return _predict


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading dataset once...")
    dataset = load_dataset(verbose=True)

    n_patients = len(dataset['df'])
    print(f"\n  Dataset: {n_patients} patients")
    assert n_patients == 594, (
        f"Expected 594 patients but got {n_patients}. "
        "Check patients.csv for excluded/missing rows."
    )

    base5_cols = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']

    experiments = [
        # 1. 6 base features, alpha=1, unweighted baseline
        ('feat_base6_a1',
         make_column_select_predict_fn(FEATURE_COLS, alpha=1.0, balanced=False)),

        # 2. 6 base features, alpha=1, frequency-balanced
        ('feat_base6_a1_bal',
         make_column_select_predict_fn(FEATURE_COLS, alpha=1.0, balanced=True)),

        # 3. 5 features without is_gpd, alpha=1
        ('feat_base5_no_gpd_a1',
         make_column_select_predict_fn(base5_cols, alpha=1.0, balanced=False)),

        # 4. 5 features without is_gpd, alpha=1, balanced
        ('feat_base5_no_gpd_a1_bal',
         make_column_select_predict_fn(base5_cols, alpha=1.0, balanced=True)),

        # 5. All 9 features, alpha=1
        ('feat_all9_a1',
         make_column_select_predict_fn(ALL_FEATURE_COLS, alpha=1.0, balanced=False)),

        # 6. All 9 features, alpha=1, balanced
        ('feat_all9_a1_bal',
         make_column_select_predict_fn(ALL_FEATURE_COLS, alpha=1.0, balanced=True)),

        # 7. 6 base + 3 interactions, alpha=1
        ('feat_interact_a1',
         make_interaction_predict_fn(alpha=1.0, balanced=False)),

        # 8. 6 base + 3 interactions, alpha=1, balanced
        ('feat_interact_a1_bal',
         make_interaction_predict_fn(alpha=1.0, balanced=True)),

        # 9. 6 base, alpha=5, balanced
        ('feat_base6_a5_bal',
         make_column_select_predict_fn(FEATURE_COLS, alpha=5.0, balanced=True)),

        # 10. interactions, alpha=5, balanced
        ('feat_interact_a5_bal',
         make_interaction_predict_fn(alpha=5.0, balanced=True)),
    ]

    for name, predict_fn in experiments:
        print(f"\n{'#'*72}")
        print(f"# Experiment: {name}")
        print(f"{'#'*72}")

        evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )
        update_dashboard()

    print("\n\nAll 10 experiments complete!")
