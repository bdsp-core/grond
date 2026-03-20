"""
Experiment: Subtype-specific frequency models.

Trains SEPARATE models for LPD and GPD patients instead of one combined model.
Tests Ridge at multiple alpha values, a variant without is_gpd feature,
and GBM (gradient boosted trees) per subtype.
"""

import sys
import subprocess
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, ALL_FEATURE_COLS, _build_segment_level_data
)

# Column indices in the ALL_FEATURE_COLS feature matrix
GPD_INDEX = ALL_FEATURE_COLS.index('is_gpd')
ALL_INDICES = list(range(len(ALL_FEATURE_COLS)))
NO_GPD_INDICES = [i for i in ALL_INDICES if i != GPD_INDEX]


def _ridge_fit_predict(X_train, y_train_log, X_test, alpha):
    """Fit Ridge on log-space labels, return predictions in original space."""
    X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
    X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

    I_reg = np.eye(X_train_b.shape[1])
    I_reg[-1, -1] = 0  # Don't regularize intercept

    try:
        w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                            X_train_b.T @ y_train_log)
        pred_log = X_test_b @ w
        pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
        return np.exp(pred_log)
    except np.linalg.LinAlgError:
        return np.full(X_test.shape[0], np.nan)


def _impute_nan(X_train, X_test):
    """Impute NaN with training column medians. Modifies arrays in place."""
    for j in range(X_train.shape[1]):
        col = X_train[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_train[~finite, j] = med
        test_col = X_test[:, j]
        X_test[~np.isfinite(test_col), j] = med


def subtype_ridge_predict_fn(alpha=1.0, feat_indices=None):
    """Separate Ridge(alpha) for LPD vs GPD segments."""
    if feat_indices is None:
        feat_indices = ALL_INDICES

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # Determine test subtype from is_gpd feature
        test_is_gpd = test_features[:, GPD_INDEX]
        test_gpd = np.nanmean(test_is_gpd) > 0.5

        # Split training data by subtype
        train_is_gpd = train_features[:, GPD_INDEX]
        gpd_mask = train_is_gpd > 0.5
        lpd_mask = ~gpd_mask

        # Select features
        X_train_all = train_features[:, feat_indices].copy()
        X_test = test_features[:, feat_indices].copy()
        y_train_all = np.log(np.clip(train_labels, 0.05, 100.0))

        if test_gpd:
            X_train = X_train_all[gpd_mask]
            y_train = y_train_all[gpd_mask]
        else:
            X_train = X_train_all[lpd_mask]
            y_train = y_train_all[lpd_mask]

        if len(X_train) < 3:
            return np.full(X_test.shape[0], np.nan)

        _impute_nan(X_train, X_test)
        return _ridge_fit_predict(X_train, y_train, X_test, alpha)

    return _predict


def subtype_gbm_predict_fn(n_estimators=100, max_depth=3):
    """Separate GBM for LPD vs GPD segments."""
    from sklearn.ensemble import GradientBoostingRegressor

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # Determine test subtype
        test_is_gpd = test_features[:, GPD_INDEX]
        test_gpd = np.nanmean(test_is_gpd) > 0.5

        # Split training by subtype
        train_is_gpd = train_features[:, GPD_INDEX]
        gpd_mask = train_is_gpd > 0.5
        lpd_mask = ~gpd_mask

        X_train_all = train_features[:, ALL_INDICES].copy()
        X_test = test_features[:, ALL_INDICES].copy()
        y_train_all = np.log(np.clip(train_labels, 0.05, 100.0))

        if test_gpd:
            X_train = X_train_all[gpd_mask]
            y_train = y_train_all[gpd_mask]
        else:
            X_train = X_train_all[lpd_mask]
            y_train = y_train_all[lpd_mask]

        if len(X_train) < 3:
            return np.full(X_test.shape[0], np.nan)

        _impute_nan(X_train, X_test)

        try:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred_log = model.predict(X_test)
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except Exception:
            return np.full(X_test.shape[0], np.nan)

    return _predict


def update_dashboard():
    """Run dashboard update."""
    subprocess.run(['python', 'code/update_dashboard_v2.py'],
                   cwd=str(CODE_DIR.parent))


if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    # 1. Separate Ridge(alpha=1) for LPD vs GPD
    evaluate_experiment(
        dataset,
        experiment_name='subtype_specific_a1',
        predict_fn=subtype_ridge_predict_fn(alpha=1.0),
        eval_type='patient_lopo',
    )
    update_dashboard()

    # 2. Separate Ridge(alpha=5) for LPD vs GPD
    evaluate_experiment(
        dataset,
        experiment_name='subtype_specific_a5',
        predict_fn=subtype_ridge_predict_fn(alpha=5.0),
        eval_type='patient_lopo',
    )
    update_dashboard()

    # 3. Separate Ridge(alpha=10) for LPD vs GPD
    evaluate_experiment(
        dataset,
        experiment_name='subtype_specific_a10',
        predict_fn=subtype_ridge_predict_fn(alpha=10.0),
        eval_type='patient_lopo',
    )
    update_dashboard()

    # 4. Separate models WITHOUT is_gpd as feature (redundant when split)
    evaluate_experiment(
        dataset,
        experiment_name='subtype_specific_notype',
        predict_fn=subtype_ridge_predict_fn(alpha=1.0, feat_indices=NO_GPD_INDICES),
        eval_type='patient_lopo',
    )
    update_dashboard()

    # 5. Separate GBM(n_estimators=100, max_depth=3) per subtype
    evaluate_experiment(
        dataset,
        experiment_name='subtype_specific_gbm',
        predict_fn=subtype_gbm_predict_fn(n_estimators=100, max_depth=3),
        eval_type='patient_lopo',
    )
    update_dashboard()

    print("\nAll subtype-specific experiments complete.")
