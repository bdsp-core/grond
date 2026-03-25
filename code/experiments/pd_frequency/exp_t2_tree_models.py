"""
Experiment T2: Tree-based models (Random Forest, Gradient Boosting).

With 318 patients, tree models may now have enough data to work
(they failed with only 43 patients before).

Runs 5 experiments:
  - t2_random_forest_100: RF, 100 trees, max_depth=5, min_samples_leaf=5
  - t2_random_forest_200: RF, 200 trees, max_depth=5, min_samples_leaf=5
  - t2_gbm_100_d3: GBM, 100 estimators, max_depth=3, lr=0.1
  - t2_gbm_200_d3: GBM, 200 estimators, max_depth=3, lr=0.05
  - t2_gbm_100_d5: GBM, 100 estimators, max_depth=5, lr=0.1

Usage:
    conda run -n foe python code/exp_t2_tree_models.py
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimization_harness_v2 import load_dataset, evaluate_experiment


def tree_predict_fn(model_cls, **model_kwargs):
    """Return a predict_fn that trains a scikit-learn tree model on log(freq).

    Features are extracted from the feature matrix (f_B, f_peaks, f_fft,
    f_tkeo, f_coh, is_gpd). NaNs are imputed with training-set medians.
    Training is done in log-space; predictions are exp'd back to Hz.
    """
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_train = train_features.copy()
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        X_test = test_features.copy()

        # Median imputation using training set
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            test_col = X_test[:, j]
            X_test[~np.isfinite(test_col), j] = med

        model = model_cls(random_state=42, **model_kwargs)
        model.fit(X_train, y_train)

        pred_log = model.predict(X_test)
        pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
        return np.exp(pred_log)

    return _predict


if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    experiments = [
        ("t2_random_forest_100", RandomForestRegressor,
         dict(n_estimators=100, max_depth=5, min_samples_leaf=5)),
        ("t2_random_forest_200", RandomForestRegressor,
         dict(n_estimators=200, max_depth=5, min_samples_leaf=5)),
        ("t2_gbm_100_d3", GradientBoostingRegressor,
         dict(n_estimators=100, max_depth=3, learning_rate=0.1)),
        ("t2_gbm_200_d3", GradientBoostingRegressor,
         dict(n_estimators=200, max_depth=3, learning_rate=0.05)),
        ("t2_gbm_100_d5", GradientBoostingRegressor,
         dict(n_estimators=100, max_depth=5, learning_rate=0.1)),
    ]

    for name, cls, kwargs in experiments:
        evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=tree_predict_fn(cls, **kwargs),
            eval_type='patient_lopo',
        )

    print("\nAll T2 tree-model experiments complete.")
