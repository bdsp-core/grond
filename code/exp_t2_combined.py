"""
Experiment T2: Combined strategies from Tier 1.

Combines expanded features + interactions + subtype-specific models.

Experiments:
  - t2_expanded_interactions_a5:  14 features (11 expanded + 3 interactions), alpha=5.0
  - t2_expanded_interactions_a10: 14 features (11 expanded + 3 interactions), alpha=10.0
  - t2_subtype_expanded_a5:  subtype-specific Ridge on 10 expanded features, alpha=5.0
  - t2_subtype_expanded_a10: subtype-specific Ridge on 10 expanded features, alpha=10.0
  - t2_ensemble_3way: average of 3 Ridge models
"""

import sys
import time
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

import optimization_harness_v2 as harness
from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, ridge_predict_fn,
    FEATURE_COLS,
)
from exp_t1_expanded_features import (
    add_expanded_features, EXPANDED_FEATURE_COLS,
)


# ── Feature column definitions ────────────────────────────────────────

# 11 expanded features + 3 interaction terms = 14
INTERACTION_FEATURE_COLS = EXPANDED_FEATURE_COLS + [
    'f_fft_x_f_tkeo', 'f_fft_x_f_B', 'f_tkeo_x_f_B'
]

# 10 expanded features (no is_gpd, for subtype-specific models)
EXPANDED_NO_GPD_COLS = [c for c in EXPANDED_FEATURE_COLS if c != 'is_gpd']


# ── Precompute interaction features ──────────────────────────────────

def add_interaction_features(dataset):
    """Add interaction terms to each feature dict in the dataset."""
    features = dataset['features']
    n_computed = 0
    for pid in features:
        for feat_dict in features[pid]:
            f_fft = feat_dict.get('f_fft', np.nan)
            f_tkeo = feat_dict.get('f_tkeo', np.nan)
            f_B = feat_dict.get('f_B', np.nan)
            feat_dict['f_fft_x_f_tkeo'] = f_fft * f_tkeo
            feat_dict['f_fft_x_f_B'] = f_fft * f_B
            feat_dict['f_tkeo_x_f_B'] = f_tkeo * f_B
            n_computed += 1
    print(f"  Interaction features computed for {n_computed} segments")


# ── Custom predict_fns ────────────────────────────────────────────────

def subtype_expanded_predict_fn(alpha=5.0):
    """Train separate Ridge models for LPD and GPD using 10 expanded features (no is_gpd)."""
    feature_cols = EXPANDED_NO_GPD_COLS
    n_feats = len(feature_cols)

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # train_features/test_features are built from whatever FEATURE_COLS is set to.
        # We need the is_gpd column to split, and the rest for modeling.
        # Since we'll set FEATURE_COLS to EXPANDED_FEATURE_COLS before calling,
        # the feature matrix includes is_gpd. We need to find it.

        # EXPANDED_FEATURE_COLS = [..., 'is_gpd', ...] - it's index 5
        is_gpd_idx = EXPANDED_FEATURE_COLS.index('is_gpd')

        # Extract is_gpd from the feature matrices
        train_is_gpd = train_features[:, is_gpd_idx]
        test_is_gpd = test_features[:, is_gpd_idx]

        # Build feature matrices WITHOUT is_gpd
        non_gpd_indices = [i for i, c in enumerate(EXPANDED_FEATURE_COLS) if c != 'is_gpd']
        X_train_all = train_features[:, non_gpd_indices]
        X_test_all = test_features[:, non_gpd_indices]

        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        predictions = np.full(len(test_features), np.nan)

        for gpd_val in [0.0, 1.0]:
            train_mask = train_is_gpd == gpd_val
            test_mask = test_is_gpd == gpd_val

            if np.sum(train_mask) < 5 or np.sum(test_mask) == 0:
                # Fall back to combined model if not enough data
                if np.sum(test_mask) > 0:
                    # Use all training data for this subset
                    train_mask = np.ones(len(train_features), dtype=bool)

            X_tr = X_train_all[train_mask].copy()
            y_tr = y_train[train_mask]
            X_te = X_test_all[test_mask].copy()

            if len(X_te) == 0:
                continue

            # Impute NaN with training median
            for j in range(X_tr.shape[1]):
                col = X_tr[:, j]
                finite = np.isfinite(col)
                med = np.median(col[finite]) if np.any(finite) else 0.0
                X_tr[~finite, j] = med
                te_col = X_te[:, j]
                X_te[~np.isfinite(te_col), j] = med

            # Add intercept
            X_tr_b = np.column_stack([X_tr, np.ones(len(X_tr))])
            X_te_b = np.column_stack([X_te, np.ones(len(X_te))])

            I_reg = np.eye(X_tr_b.shape[1])
            I_reg[-1, -1] = 0  # Don't regularize intercept

            try:
                w = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha * I_reg,
                                    X_tr_b.T @ y_tr)
                pred_log = X_te_b @ w
                pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
                predictions[test_mask] = np.exp(pred_log)
            except np.linalg.LinAlgError:
                predictions[test_mask] = np.nan

        return predictions

    return _predict


def ensemble_3way_predict_fn():
    """Ensemble: average of 3 Ridge models.

    1. Ridge on 6 base features (alpha=1.0)
    2. Ridge on 11 expanded features (alpha=10.0)
    3. Subtype-specific Ridge on 5 base features (alpha=5.0)

    The harness passes feature matrices built from the current FEATURE_COLS.
    We'll set FEATURE_COLS to EXPANDED_FEATURE_COLS (11 features) so we have
    access to all features. Then we extract subsets within the predict_fn.
    """
    base_indices = [EXPANDED_FEATURE_COLS.index(c) for c in FEATURE_COLS]
    is_gpd_idx = EXPANDED_FEATURE_COLS.index('is_gpd')
    # Base features without is_gpd for subtype-specific model
    base_no_gpd_indices = [EXPANDED_FEATURE_COLS.index(c)
                           for c in FEATURE_COLS if c != 'is_gpd']

    def _ridge_fit_predict(X_train, y_train_log, X_test, alpha):
        """Fit Ridge and predict."""
        # Impute NaN
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            te_col = X_test[:, j]
            X_test[~np.isfinite(te_col), j] = med

        X_tr_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_te_b = np.column_stack([X_test, np.ones(len(X_test))])

        I_reg = np.eye(X_tr_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_tr_b.T @ X_tr_b + alpha * I_reg,
                                X_tr_b.T @ y_train_log)
            pred_log = X_te_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(len(X_test), np.nan)

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        y_train_log = np.log(np.clip(train_labels, 0.05, 100.0))
        n_test = len(test_features)

        # ── Model 1: Ridge on 6 base features, alpha=1.0 ─────────────
        X_tr1 = train_features[:, base_indices].copy()
        X_te1 = test_features[:, base_indices].copy()
        pred1 = _ridge_fit_predict(X_tr1, y_train_log, X_te1, alpha=1.0)

        # ── Model 2: Ridge on 11 expanded features, alpha=10.0 ───────
        X_tr2 = train_features.copy()
        X_te2 = test_features.copy()
        pred2 = _ridge_fit_predict(X_tr2, y_train_log, X_te2, alpha=10.0)

        # ── Model 3: Subtype-specific Ridge on 5 base features, alpha=5.0
        train_is_gpd = train_features[:, is_gpd_idx]
        test_is_gpd = test_features[:, is_gpd_idx]
        pred3 = np.full(n_test, np.nan)

        for gpd_val in [0.0, 1.0]:
            tr_mask = train_is_gpd == gpd_val
            te_mask = test_is_gpd == gpd_val

            if np.sum(tr_mask) < 5 or np.sum(te_mask) == 0:
                if np.sum(te_mask) > 0:
                    tr_mask = np.ones(len(train_features), dtype=bool)

            X_tr3 = train_features[tr_mask][:, base_no_gpd_indices].copy()
            y_tr3 = y_train_log[tr_mask]
            X_te3 = test_features[te_mask][:, base_no_gpd_indices].copy()

            if len(X_te3) == 0:
                continue

            pred3[te_mask] = _ridge_fit_predict(X_tr3, y_tr3, X_te3, alpha=5.0)

        # ── Ensemble: simple average ─────────────────────────────────
        preds = np.stack([pred1, pred2, pred3], axis=0)
        # Average ignoring NaN
        with np.errstate(all='ignore'):
            ensemble = np.nanmean(preds, axis=0)

        return ensemble

    return _predict


# ── Helper: run with custom feature cols ─────────────────────────────

def run_lopo_custom(dataset, experiment_name, predict_fn, feature_cols):
    """Run LOPO with custom feature columns by temporarily swapping FEATURE_COLS."""
    original_cols = harness.FEATURE_COLS
    harness.FEATURE_COLS = list(feature_cols)

    try:
        metrics = evaluate_experiment(
            dataset,
            experiment_name=experiment_name,
            predict_fn=predict_fn,
            eval_type='patient_lopo',
        )
    finally:
        harness.FEATURE_COLS = original_cols

    return metrics


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    # Load dataset
    dataset = load_dataset(verbose=True)

    # Compute expanded features
    print("\nComputing expanded features...")
    t_feat = time.time()
    add_expanded_features(dataset)
    print(f"  Done in {time.time() - t_feat:.1f}s")

    # Compute interaction features
    print("\nComputing interaction features...")
    add_interaction_features(dataset)

    all_results = {}

    # ── Experiment 1: Expanded + interactions, alpha=5.0 ──────────────
    metrics = run_lopo_custom(
        dataset, 't2_expanded_interactions_a5',
        predict_fn=ridge_predict_fn(alpha=5.0),
        feature_cols=INTERACTION_FEATURE_COLS,
    )
    all_results['t2_expanded_interactions_a5'] = metrics

    # ── Experiment 2: Expanded + interactions, alpha=10.0 ─────────────
    metrics = run_lopo_custom(
        dataset, 't2_expanded_interactions_a10',
        predict_fn=ridge_predict_fn(alpha=10.0),
        feature_cols=INTERACTION_FEATURE_COLS,
    )
    all_results['t2_expanded_interactions_a10'] = metrics

    # ── Experiment 3: Subtype-specific with expanded, alpha=5.0 ───────
    metrics = run_lopo_custom(
        dataset, 't2_subtype_expanded_a5',
        predict_fn=subtype_expanded_predict_fn(alpha=5.0),
        feature_cols=EXPANDED_FEATURE_COLS,
    )
    all_results['t2_subtype_expanded_a5'] = metrics

    # ── Experiment 4: Subtype-specific with expanded, alpha=10.0 ──────
    metrics = run_lopo_custom(
        dataset, 't2_subtype_expanded_a10',
        predict_fn=subtype_expanded_predict_fn(alpha=10.0),
        feature_cols=EXPANDED_FEATURE_COLS,
    )
    all_results['t2_subtype_expanded_a10'] = metrics

    # ── Experiment 5: 3-way ensemble ──────────────────────────────────
    metrics = run_lopo_custom(
        dataset, 't2_ensemble_3way',
        predict_fn=ensemble_3way_predict_fn(),
        feature_cols=EXPANDED_FEATURE_COLS,
    )
    all_results['t2_ensemble_3way'] = metrics

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY: All T2 Experiments")
    print(f"{'='*72}")
    print(f"  {'Experiment':<35s} {'Spearman':>10s} {'MAE':>8s}")
    print(f"  {'-'*55}")

    for name, m in all_results.items():
        sp = m.get('combined_spearman', float('nan'))
        mae = m.get('combined_mae', float('nan'))
        sp_str = f"{sp:.4f}" if np.isfinite(sp) else "N/A"
        mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        print(f"  {name:<35s} {sp_str:>10s} {mae_str:>8s}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'='*72}")
