"""
Bias-correction experiments for Ridge frequency estimation.

Ridge regression systematically underestimates high-frequency LPDs due to
shrinkage toward the mean. This script tests 8 strategies to correct this bias.

Run: conda run -n foe python code/exp_opt_freq_bias.py
"""

import sys
import subprocess
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, ridge_predict_fn,
    ALL_FEATURE_COLS, _build_segment_level_data,
)


# ── Helper: impute NaN with training median, return medians for test ──

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


def _ridge_solve(X_b, y, alpha):
    """Ridge regression with intercept column already appended."""
    I_reg = np.eye(X_b.shape[1])
    I_reg[-1, -1] = 0  # Don't regularize intercept
    try:
        w = np.linalg.solve(X_b.T @ X_b + alpha * I_reg, X_b.T @ y)
        return w
    except np.linalg.LinAlgError:
        return None


def _clip_log_pred(pred_log):
    pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
    return np.exp(pred_log)


def _update_dashboard():
    subprocess.run(['python', 'code/update_dashboard_v2.py'], cwd=str(PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════
# Experiment 1: Ridge alpha=0.01
# ═══════════════════════════════════════════════════════════════════════

def make_ridge_predict_fn(alpha):
    """Ridge in log-space with specified alpha."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        Xtr_b = _add_intercept(Xtr)
        Xte_b = _add_intercept(Xte)
        w = _ridge_solve(Xtr_b, y_train, alpha)
        if w is None:
            return np.full(Xte.shape[0], np.nan)
        return _clip_log_pred(Xte_b @ w)
    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Experiment 4: Residual correction (two-stage Ridge)
# ═══════════════════════════════════════════════════════════════════════

def residual_correction_predict_fn():
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        Xtr_b = _add_intercept(Xtr)
        Xte_b = _add_intercept(Xte)

        # Stage 1: Ridge alpha=1
        w1 = _ridge_solve(Xtr_b, y_train, alpha=1.0)
        if w1 is None:
            return np.full(Xte.shape[0], np.nan)

        train_pred1 = Xtr_b @ w1
        residuals = y_train - train_pred1

        # Stage 2: Ridge alpha=0.5 on residuals
        w2 = _ridge_solve(Xtr_b, residuals, alpha=0.5)
        if w2 is None:
            return _clip_log_pred(Xte_b @ w1)

        test_pred1 = Xte_b @ w1
        test_pred2 = Xte_b @ w2
        return _clip_log_pred(test_pred1 + test_pred2)

    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Experiment 5: Quantile (median) regression via GBR
# ═══════════════════════════════════════════════════════════════════════

def quantile_median_predict_fn():
    from sklearn.ensemble import GradientBoostingRegressor

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        model = GradientBoostingRegressor(
            loss='quantile', alpha=0.5,
            n_estimators=200, max_depth=3, learning_rate=0.05,
            random_state=42,
        )
        model.fit(Xtr, y_train)
        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Experiment 6: Weighted Ridge (freq^2 sample weights)
# ═══════════════════════════════════════════════════════════════════════

def weighted_high_freq_predict_fn():
    from sklearn.linear_model import Ridge

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        # Weights proportional to freq^2
        weights = train_labels ** 2

        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(Xtr, y_train, sample_weight=weights)
        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Experiment 7: Piecewise Ridge (low/high freq blend)
# ═══════════════════════════════════════════════════════════════════════

def piecewise_ridge_predict_fn():
    from sklearn.linear_model import LogisticRegression

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        # Split training into low/high frequency
        is_high = (train_labels >= 1.0).astype(int)

        # Train separate Ridge models
        Xtr_b = _add_intercept(Xtr)
        Xte_b = _add_intercept(Xte)

        low_mask = is_high == 0
        high_mask = is_high == 1

        # Need at least a few samples in each group
        if np.sum(low_mask) < 3 or np.sum(high_mask) < 3:
            w = _ridge_solve(Xtr_b, y_train, alpha=1.0)
            if w is None:
                return np.full(Xte.shape[0], np.nan)
            return _clip_log_pred(Xte_b @ w)

        w_low = _ridge_solve(Xtr_b[low_mask], y_train[low_mask], alpha=1.0)
        w_high = _ridge_solve(Xtr_b[high_mask], y_train[high_mask], alpha=1.0)

        if w_low is None or w_high is None:
            w = _ridge_solve(Xtr_b, y_train, alpha=1.0)
            if w is None:
                return np.full(Xte.shape[0], np.nan)
            return _clip_log_pred(Xte_b @ w)

        # Train logistic classifier to predict high/low
        # Use is_gpd and f_fft as features
        is_gpd_idx = ALL_FEATURE_COLS.index('is_gpd')
        f_fft_idx = ALL_FEATURE_COLS.index('f_fft')
        X_cls_train = Xtr[:, [is_gpd_idx, f_fft_idx]]
        X_cls_test = Xte[:, [is_gpd_idx, f_fft_idx]]

        clf = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
        clf.fit(X_cls_train, is_high)
        p_high = clf.predict_proba(X_cls_test)[:, 1]

        # Blend predictions
        pred_low = Xte_b @ w_low
        pred_high = Xte_b @ w_high
        pred_log = (1 - p_high) * pred_low + p_high * pred_high
        return _clip_log_pred(pred_log)

    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Experiment 8: GBM quantile median (different hyperparams)
# ═══════════════════════════════════════════════════════════════════════

def gbm_quantile_median_predict_fn():
    from sklearn.ensemble import GradientBoostingRegressor

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        Xtr, Xte = _impute(train_features, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        model = GradientBoostingRegressor(
            loss='quantile', alpha=0.5,
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42,
        )
        model.fit(Xtr, y_train)
        pred_log = model.predict(Xte)
        return _clip_log_pred(pred_log)

    return _predict


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("BIAS-CORRECTION EXPERIMENTS FOR RIDGE FREQUENCY ESTIMATION")
    print("=" * 72)

    dataset = load_dataset(verbose=True)

    experiments = [
        ('ridge_a001',           make_ridge_predict_fn(alpha=0.01)),
        ('ridge_a01',            make_ridge_predict_fn(alpha=0.1)),
        ('ridge_a005',           make_ridge_predict_fn(alpha=0.05)),
        ('residual_correction',  residual_correction_predict_fn()),
        ('quantile_median',      quantile_median_predict_fn()),
        ('weighted_high_freq',   weighted_high_freq_predict_fn()),
        ('piecewise_ridge',      piecewise_ridge_predict_fn()),
        ('gbm_quantile_median',  gbm_quantile_median_predict_fn()),
    ]

    all_results = {}

    for name, predict_fn in experiments:
        print(f"\n{'#' * 72}")
        print(f"# Experiment: {name}")
        print(f"{'#' * 72}")

        metrics = evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )
        all_results[name] = metrics

        # Update dashboard after each experiment
        _update_dashboard()

    # ── Final summary table ──────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("FINAL SUMMARY: BIAS-CORRECTION EXPERIMENTS")
    print(f"{'=' * 90}")
    print(f"  {'Experiment':<25s} {'Combined ρ':>12s} {'LPD ρ':>10s} {'GPD ρ':>10s} "
          f"{'Combined MAE':>14s} {'LPD MAE':>10s} {'GPD MAE':>10s}")
    print(f"  {'-' * 85}")

    for name, m in all_results.items():
        cs = m.get('combined_spearman', float('nan'))
        ls = m.get('lpd_spearman', float('nan'))
        gs = m.get('gpd_spearman', float('nan'))
        cm = m.get('combined_mae', float('nan'))
        lm = m.get('lpd_mae', float('nan'))
        gm = m.get('gpd_mae', float('nan'))

        def _fmt(v):
            return f"{v:.4f}" if np.isfinite(v) else "N/A"

        print(f"  {name:<25s} {_fmt(cs):>12s} {_fmt(ls):>10s} {_fmt(gs):>10s} "
              f"{_fmt(cm):>14s} {_fmt(lm):>10s} {_fmt(gm):>10s}")

    print(f"{'=' * 90}")
    print("Done.")
