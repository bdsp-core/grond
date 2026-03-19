"""
Experiment T1: Subtype-specific Ridge models.

Trains SEPARATE Ridge models for LPD and GPD instead of one unified model.
Uses 5 features (f_B, f_peaks, f_fft, f_tkeo, f_coh) without is_gpd flag
since the subtype split already encodes that information.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, evaluate_experiment

FEATURE_COLS = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']
# Map feature name -> index in the 6-col feature matrix from harness
# Harness order: f_B(0), f_peaks(1), f_fft(2), f_tkeo(3), f_coh(4), is_gpd(5)
FEAT_INDICES = [0, 1, 2, 3, 4]  # all except is_gpd
GPD_INDEX = 5  # is_gpd column index


def subtype_specific_predict_fn(alpha=1.0):
    """Return a predict_fn that trains separate Ridge models per subtype."""

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # Determine test subtype from is_gpd feature
        test_is_gpd = test_features[:, GPD_INDEX]
        # All test segments belong to the same patient, so same subtype
        test_gpd = np.nanmean(test_is_gpd) > 0.5

        # Extract only the 5 feature columns (no is_gpd)
        X_train_all = train_features[:, FEAT_INDICES].copy()
        X_test = test_features[:, FEAT_INDICES].copy()
        y_train_all = np.log(np.clip(train_labels, 0.05, 100.0))

        # Split training data by subtype
        train_is_gpd = train_features[:, GPD_INDEX]
        gpd_mask = train_is_gpd > 0.5
        lpd_mask = ~gpd_mask

        if test_gpd:
            X_train = X_train_all[gpd_mask]
            y_train = y_train_all[gpd_mask]
        else:
            X_train = X_train_all[lpd_mask]
            y_train = y_train_all[lpd_mask]

        if len(X_train) < 3:
            return np.full(X_test.shape[0], np.nan)

        # Impute NaN with subtype-specific training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            test_col = X_test[:, j]
            X_test[~np.isfinite(test_col), j] = med

        # Ridge regression with intercept
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.nan)

    return _predict


if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    for alpha, suffix in [(1.0, 'a1'), (5.0, 'a5'), (10.0, 'a10')]:
        evaluate_experiment(
            dataset,
            experiment_name=f't1_subtype_specific_{suffix}',
            predict_fn=subtype_specific_predict_fn(alpha=alpha),
            eval_type='patient_lopo',
        )
