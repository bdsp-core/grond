"""
Experiment: Compare regression models with inverse-frequency-bin weighting.

Tests 14 configurations:
  - Standard (unweighted): Ridge alpha=1,5; RF 100,200; GBM 200
  - Frequency-balanced (weighted): same models with inverse-freq-bin sample weights
  - Fine-bin balanced: Ridge alpha=1, RF 100 with finer frequency bins
  - Ridge alpha=0.1 balanced
  - RF 100 with oversampling instead of weighting

All train in log-space, impute NaN with training median, clip predictions.

Usage:
    conda run -n foe python code/exp_opt_freq_models.py
"""

import sys
import subprocess
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, evaluate_experiment, ridge_predict_fn


# ── Helpers ──────────────────────────────────────────────────────────

def _impute(X_train, X_test):
    """Impute NaN with training column medians. Returns copies."""
    Xtr = X_train.copy()
    Xte = X_test.copy()
    for j in range(Xtr.shape[1]):
        col = Xtr[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        Xtr[~finite, j] = med
        Xte[~np.isfinite(Xte[:, j]), j] = med
    return Xtr, Xte


def _add_intercept(X):
    return np.column_stack([X, np.ones(X.shape[0])])


def _clip_log_pred(pred_log):
    pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
    return np.exp(pred_log)


STANDARD_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
FINE_BINS = [0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]


def _compute_freq_weights(train_labels, bins):
    """Compute inverse-frequency-bin sample weights.

    Bins each sample's gold_standard_freq, then weight = 1/(count in bin).
    Normalizes so weights sum to N.
    """
    freqs = np.asarray(train_labels, dtype=float)
    n = len(freqs)
    bin_indices = np.digitize(freqs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    # Count samples in each bin
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    # Avoid division by zero
    bin_counts = np.maximum(bin_counts, 1)

    # Weight = 1 / count_in_bin for each sample
    weights = np.array([1.0 / bin_counts[bi] for bi in bin_indices])

    # Normalize so weights sum to N
    weights = weights * (n / np.sum(weights))
    return weights


def update_dashboard():
    """Push latest results to the dashboard."""
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


# ── Predict functions ────────────────────────────────────────────────

def ridge_weighted_predict_fn(alpha=1.0, bins=None):
    """Ridge regression with weighted least squares using freq-bin weights.

    Solves: (X'WX + alpha*I) w = X'W y
    where W = diag(weights).
    """
    if bins is None:
        bins = STANDARD_BINS

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        # Compute frequency-bin weights on original (not log) labels
        weights = _compute_freq_weights(train_labels, bins)

        Xtr_b = _add_intercept(Xtr)
        Xte_b = _add_intercept(Xte)

        I_reg = np.eye(Xtr_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        # Weighted least squares: (X'WX + alpha*I) w = X'Wy
        W = np.diag(weights)
        try:
            w = np.linalg.solve(
                Xtr_b.T @ W @ Xtr_b + alpha * I_reg,
                Xtr_b.T @ W @ y_train
            )
            return _clip_log_pred(Xte_b @ w)
        except np.linalg.LinAlgError:
            return np.full(Xte.shape[0], np.nan)

    return _predict


def tree_predict_fn(model_cls, **model_kwargs):
    """Unweighted tree model predict_fn. Log-space training."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        model = model_cls(random_state=42, **model_kwargs)
        model.fit(Xtr, y_train)

        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


def tree_weighted_predict_fn(model_cls, bins=None, **model_kwargs):
    """Tree model with freq-bin sample_weight. Log-space training."""
    if bins is None:
        bins = STANDARD_BINS

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        weights = _compute_freq_weights(train_labels, bins)

        model = model_cls(random_state=42, **model_kwargs)
        model.fit(Xtr, y_train, sample_weight=weights)

        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


def rf_oversample_predict_fn(**model_kwargs):
    """RF with oversampling of rare frequency bins instead of weighting.

    Duplicates samples from under-represented bins so all bins have
    roughly equal representation.
    """
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train_raw = train_labels.copy()
        y_train = np.log(np.clip(y_train_raw, 0.05, 100.0))

        bins = STANDARD_BINS
        freqs = np.asarray(y_train_raw, dtype=float)
        bin_indices = np.digitize(freqs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
        max_count = int(np.max(bin_counts))

        # Oversample: duplicate samples from each bin up to max_count
        rng = np.random.RandomState(42)
        new_X = []
        new_y = []
        for b in range(len(bins) - 1):
            mask = bin_indices == b
            if np.sum(mask) == 0:
                continue
            Xb = Xtr[mask]
            yb = y_train[mask]
            n_in_bin = len(yb)
            n_needed = max_count - n_in_bin
            new_X.append(Xb)
            new_y.append(yb)
            if n_needed > 0:
                idx = rng.randint(0, n_in_bin, size=n_needed)
                new_X.append(Xb[idx])
                new_y.append(yb[idx])

        Xtr_os = np.vstack(new_X)
        y_train_os = np.concatenate(new_y)

        model = RandomForestRegressor(random_state=42, **model_kwargs)
        model.fit(Xtr_os, y_train_os)

        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    experiments = [
        # ── Standard (unweighted) ──
        ("ridge_a1", ridge_predict_fn(alpha=1.0)),
        ("ridge_a5", ridge_predict_fn(alpha=5.0)),
        ("rf_100", tree_predict_fn(
            RandomForestRegressor,
            n_estimators=100, max_depth=5, min_samples_leaf=5)),
        ("rf_200", tree_predict_fn(
            RandomForestRegressor,
            n_estimators=200, max_depth=8, min_samples_leaf=3)),
        ("gbm_200_d3", tree_predict_fn(
            GradientBoostingRegressor,
            n_estimators=200, max_depth=3, learning_rate=0.05)),

        # ── Frequency-balanced (weighted) ──
        ("ridge_a1_bal", ridge_weighted_predict_fn(alpha=1.0)),
        ("ridge_a5_bal", ridge_weighted_predict_fn(alpha=5.0)),
        ("rf_100_bal", tree_weighted_predict_fn(
            RandomForestRegressor,
            n_estimators=100, max_depth=5, min_samples_leaf=5)),
        ("rf_200_bal", tree_weighted_predict_fn(
            RandomForestRegressor,
            n_estimators=200, max_depth=8, min_samples_leaf=3)),
        ("gbm_200_d3_bal", tree_weighted_predict_fn(
            GradientBoostingRegressor,
            n_estimators=200, max_depth=3, learning_rate=0.05)),

        # ── Fine-bin balanced ──
        ("ridge_a1_bal_fine", ridge_weighted_predict_fn(alpha=1.0, bins=FINE_BINS)),
        ("rf_100_bal_fine", tree_weighted_predict_fn(
            RandomForestRegressor,
            bins=FINE_BINS,
            n_estimators=100, max_depth=5, min_samples_leaf=5)),

        # ── Additional ──
        ("ridge_a01_bal", ridge_weighted_predict_fn(alpha=0.1)),
        ("rf_100_oversample", rf_oversample_predict_fn(
            n_estimators=100, max_depth=5, min_samples_leaf=5)),
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

    print("\n" + "=" * 72)
    print("All 14 model comparison experiments complete.")
    print("=" * 72)
