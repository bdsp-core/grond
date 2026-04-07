"""
Run remaining frequency model experiments that don't already have JSON results.

Experiments:
  1. ridge_a01_bal: sklearn Ridge alpha=0.1, freq-balanced weights
  2. ridge_a1_bal_fine: sklearn Ridge alpha=1, finer bins
  3. rf_100_bal: RF(100,depth=5,leaf=5), freq-balanced
  4. rf_200_bal: RF(200,depth=8,leaf=3), freq-balanced
  5. rf_100_bal_fine: RF(100,depth=5,leaf=5), finer bins
  6. rf_100_oversample: RF(100,depth=5,leaf=5), oversample rare bins
  7. gbm_200_d3: GBM(200,depth=3,lr=0.05), unweighted
  8. gbm_200_d3_bal: GBM(200,depth=3,lr=0.05), freq-balanced

Usage:
    conda run -n foe python code/exp_opt_freq_models_remaining.py
"""

import sys
import subprocess
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, RUNS_DIR,
    ALL_FEATURE_COLS, FEATURE_COLS,
)


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


def _clip_log_pred(pred_log):
    pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
    return np.exp(pred_log)


STANDARD_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
FINE_BINS = [0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]


def _compute_freq_weights(train_labels, bins):
    """Inverse-frequency-bin sample weights, normalized so sum = N."""
    freqs = np.asarray(train_labels, dtype=float)
    n = len(freqs)
    bin_indices = np.digitize(freqs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    bin_counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    bin_counts = np.maximum(bin_counts, 1)
    weights = np.array([1.0 / bin_counts[bi] for bi in bin_indices])
    weights = weights * (n / np.sum(weights))
    return weights


def update_dashboard():
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


def already_done(name):
    return (RUNS_DIR / f'{name}.json').exists()


# ── Predict functions ────────────────────────────────────────────────

def sklearn_ridge_balanced_fn(alpha=1.0, bins=None):
    """sklearn Ridge with freq-balanced sample_weight."""
    if bins is None:
        bins = STANDARD_BINS

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        weights = _compute_freq_weights(train_labels, bins)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(Xtr, y_train, sample_weight=weights)
        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

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
    """RF with oversampling of rare frequency bins instead of weighting."""
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

        rng = np.random.RandomState(42)
        new_X, new_y = [], []
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


def make_interaction_balanced_fn(alpha=5.0, bins=None):
    """Ridge with 6 base + 3 interactions, freq-balanced via sklearn Ridge sample_weight."""
    if bins is None:
        bins = STANDARD_BINS
    base_indices = [ALL_FEATURE_COLS.index(c) for c in FEATURE_COLS]

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        def _build_X(feats):
            X_base = feats[:, base_indices].copy()
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

        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        weights = _compute_freq_weights(train_labels, bins)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train, sample_weight=weights)
        pred_log = model.predict(X_test)
        return _clip_log_pred(pred_log)

    return _predict


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    experiments = [
        # 1. Ridge alpha=0.1, freq-balanced (sklearn)
        ('ridge_a01_bal', sklearn_ridge_balanced_fn(alpha=0.1, bins=STANDARD_BINS)),

        # 2. Ridge alpha=1, finer bins (sklearn)
        ('ridge_a1_bal_fine', sklearn_ridge_balanced_fn(alpha=1.0, bins=FINE_BINS)),

        # 3. RF(100, depth=5, leaf=5), freq-balanced
        ('rf_100_bal', tree_weighted_predict_fn(
            RandomForestRegressor,
            bins=STANDARD_BINS,
            n_estimators=100, max_depth=5, min_samples_leaf=5)),

        # 4. RF(200, depth=8, leaf=3), freq-balanced
        ('rf_200_bal', tree_weighted_predict_fn(
            RandomForestRegressor,
            bins=STANDARD_BINS,
            n_estimators=200, max_depth=8, min_samples_leaf=3)),

        # 5. RF(100, depth=5, leaf=5), finer bins
        ('rf_100_bal_fine', tree_weighted_predict_fn(
            RandomForestRegressor,
            bins=FINE_BINS,
            n_estimators=100, max_depth=5, min_samples_leaf=5)),

        # 6. RF(100, depth=5, leaf=5), oversample rare bins
        ('rf_100_oversample', rf_oversample_predict_fn(
            n_estimators=100, max_depth=5, min_samples_leaf=5)),

        # 7. GBM(200, depth=3, lr=0.05), unweighted
        ('gbm_200_d3', tree_predict_fn(
            GradientBoostingRegressor,
            n_estimators=200, max_depth=3, learning_rate=0.05)),

        # 8. GBM(200, depth=3, lr=0.05), freq-balanced
        ('gbm_200_d3_bal', tree_weighted_predict_fn(
            GradientBoostingRegressor,
            bins=STANDARD_BINS,
            n_estimators=200, max_depth=3, learning_rate=0.05)),

        # 9. Ridge alpha=5, 6 base + 3 interactions, freq-balanced (sklearn)
        ('feat_interact_a5_bal', make_interaction_balanced_fn(alpha=5.0, bins=STANDARD_BINS)),
    ]

    # Filter to only experiments without existing JSON results
    to_run = [(name, fn) for name, fn in experiments if not already_done(name)]

    if not to_run:
        print("All experiments already have JSON results. Nothing to do.")
        sys.exit(0)

    skipped = len(experiments) - len(to_run)
    print(f"Found {len(to_run)} experiments to run (skipping {skipped} already done):")
    for name, _ in to_run:
        print(f"  - {name}")

    # Load dataset once
    dataset = load_dataset(verbose=True)

    n_done = 0
    for name, predict_fn in to_run:
        # Double-check before running (in case a prior iteration just created it)
        if already_done(name):
            print(f"\n  SKIP (already exists): {name}")
            continue

        print(f"\n{'#'*72}")
        print(f"# Experiment: {name} ({n_done+1}/{len(to_run)})")
        print(f"{'#'*72}")

        evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )
        update_dashboard()
        n_done += 1

    print(f"\n{'='*72}")
    print(f"Done. Ran {n_done} experiments, skipped {skipped}.")
    print(f"{'='*72}")
