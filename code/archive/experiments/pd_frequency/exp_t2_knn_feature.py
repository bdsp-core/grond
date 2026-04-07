"""
Experiment T2: k-NN frequency estimate as an additional Ridge feature.

For each held-out patient, find the k most similar training patients
(by Euclidean distance in standardized feature space) and use their
average gold_standard_freq as a new feature (f_knn), then train Ridge
on log(freq) with 7 features.

Experiments:
  t2_knn3_ridge_a1   k=3,  alpha=1.0
  t2_knn5_ridge_a1   k=5,  alpha=1.0
  t2_knn10_ridge_a1  k=10, alpha=1.0
  t2_knn5_ridge_a5   k=5,  alpha=5.0

Usage:
    conda run -n foe python code/exp_t2_knn_feature.py
"""

import numpy as np
from optimization_harness_v2 import load_dataset, evaluate_experiment


def knn_ridge_predict_fn(k=5, alpha=1.0):
    """Ridge with k-NN frequency estimate appended as 7th feature."""

    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_train = train_features.copy()  # (N_train, 6)
        y_train_raw = train_labels.copy()
        X_test = test_features.copy()    # (N_test, 6)

        # 1. Impute NaN with training median
        medians = np.zeros(X_train.shape[1])
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            medians[j] = med
            X_train[~finite, j] = med
            tc = X_test[:, j]
            X_test[~np.isfinite(tc), j] = med

        # 2. Standardize (z-score from training set)
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0
        X_train_z = (X_train - means) / stds
        X_test_z = (X_test - means) / stds

        # 3. Compute k-NN feature for training set (leave-one-out within train)
        f_knn_train = np.zeros(len(X_train_z))
        for i in range(len(X_train_z)):
            dists = np.sqrt(np.sum((X_train_z - X_train_z[i]) ** 2, axis=1))
            dists[i] = np.inf  # exclude self
            nn_idx = np.argpartition(dists, k)[:k]
            f_knn_train[i] = np.mean(y_train_raw[nn_idx])

        # 4. Compute k-NN feature for test set
        f_knn_test = np.zeros(len(X_test_z))
        for i in range(len(X_test_z)):
            dists = np.sqrt(np.sum((X_train_z - X_test_z[i]) ** 2, axis=1))
            nn_idx = np.argpartition(dists, k)[:k]
            f_knn_test[i] = np.mean(y_train_raw[nn_idx])

        # 5. Augment feature matrices with f_knn
        X_train_aug = np.column_stack([X_train, f_knn_train])  # (N_train, 7)
        X_test_aug = np.column_stack([X_test, f_knn_test])     # (N_test, 7)

        # 6. Train Ridge on log(freq)
        y_train = np.log(np.clip(y_train_raw, 0.05, 100.0))

        # Add intercept
        X_train_b = np.column_stack([X_train_aug, np.ones(len(X_train_aug))])
        X_test_b = np.column_stack([X_test_aug, np.ones(len(X_test_aug))])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        try:
            w = np.linalg.solve(
                X_train_b.T @ X_train_b + alpha * I_reg,
                X_train_b.T @ y_train
            )
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.nan)

    return _predict


if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    experiments = [
        ('t2_knn3_ridge_a1',  3,  1.0),
        ('t2_knn5_ridge_a1',  5,  1.0),
        ('t2_knn10_ridge_a1', 10, 1.0),
        ('t2_knn5_ridge_a5',  5,  5.0),
    ]

    for name, k, alpha in experiments:
        evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=knn_ridge_predict_fn(k=k, alpha=alpha),
            eval_type='patient_lopo',
        )
