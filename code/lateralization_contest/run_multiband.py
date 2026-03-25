#!/usr/bin/env python3
"""Multi-band feature extraction + classification for LRDA vs GRDA.

Tests:
1. Single best band (0.5-5 Hz)
2. Multiple bands concatenated
3. Various classifiers on each

Results written to v3 leaderboard.
"""
import sys
import json
import time
import pickle
import warnings
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

V2_CACHE = PROJECT_DIR / 'results' / 'lateralization_contest_v2' / '_cache'
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v3'

from lateralization_contest.optimize_bandpass import extract_spatial_all


def load_balanced():
    with open(V2_CACHE / 'lateral_v2_data.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    segs = data['segs']
    lrda_pids = df[df['subtype'] == 'lrda']['patient_id'].values
    grda_pids = df[df['subtype'] == 'grda']['patient_id'].values
    np.random.seed(42)
    grda_sub = np.random.choice(grda_pids, size=len(lrda_pids), replace=False)
    use_pids = set(list(lrda_pids) + list(grda_sub))
    df_bal = df[df['patient_id'].isin(use_pids)].copy().reset_index(drop=True)
    labels = (df_bal['subtype'] == 'lrda').astype(int).values
    return df_bal, segs, labels


def extract_band(pids, segs, lo, hi):
    """Extract spatial_all features for a single band."""
    feat_dicts = [extract_spatial_all(segs[pid], lo, hi) for pid in pids]
    all_keys = sorted(set(k for d in feat_dicts for k in d))
    X = np.array([[d.get(k, 0.0) for k in all_keys] for d in feat_dicts], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Prefix keys with band info
    prefixed_keys = [f'b{lo:.1f}_{hi:.0f}_{k}' for k in all_keys]
    return X, prefixed_keys


CLASSIFIERS = {
    'GBM_deep': lambda: GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42),
    'GBM': lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42),
    'RF_deep': lambda: RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42),
    'RF': lambda: RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42),
    'ExtraTrees': lambda: ExtraTreesClassifier(
        n_estimators=300, max_depth=8, random_state=42),
    'SVM': lambda: Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', probability=True, C=1.0))]),
    'LogReg_L1': lambda: Pipeline([('s', StandardScaler()),
                                    ('c', LogisticRegression(max_iter=2000, C=0.3, penalty='l1', solver='saga'))]),
}

# Feature set configs: name -> list of (lo, hi) bands
BAND_CONFIGS = {
    'band_0.5_5': [(0.5, 5.0)],
    'band_0.5_6': [(0.5, 6.0)],
    'band_0.5_4+0.5_8': [(0.5, 4.0), (0.5, 8.0)],
    'band_0.5_3+0.5_6': [(0.5, 3.0), (0.5, 6.0)],
    'band_0.5_4+0.5_6': [(0.5, 4.0), (0.5, 6.0)],
    'band_0.5_5+1_8': [(0.5, 5.0), (1.0, 8.0)],
    'tri_0.5_3+0.5_5+0.5_8': [(0.5, 3.0), (0.5, 5.0), (0.5, 8.0)],
    'tri_0.5_4+0.5_6+0.5_10': [(0.5, 4.0), (0.5, 6.0), (0.5, 10.0)],
    'quad_0.5_3+0.5_5+0.5_8+1_6': [(0.5, 3.0), (0.5, 5.0), (0.5, 8.0), (1.0, 6.0)],
}


def update_leaderboard():
    """Regenerate v3 leaderboard from all results."""
    from lateralization_contest.ml_classify_v3 import update_leaderboard as _update
    _update()


def main():
    print("=" * 70)
    print("  Multi-Band LRDA vs GRDA Classification")
    print("=" * 70)

    df_bal, segs, labels = load_balanced()
    pids = df_bal['patient_id'].values
    print(f"Balanced: {(labels==1).sum()} LRDA + {(labels==0).sum()} GRDA = {len(labels)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Pre-extract all needed bands
    print("\nExtracting features per band...")
    band_cache = {}
    needed_bands = set()
    for bands in BAND_CONFIGS.values():
        for b in bands:
            needed_bands.add(b)

    for lo, hi in sorted(needed_bands):
        t0 = time.time()
        X, keys = extract_band(pids, segs, lo, hi)
        band_cache[(lo, hi)] = (X, keys)
        print(f"  [{lo:.1f}-{hi:.0f}] Hz: {X.shape[1]} features ({time.time()-t0:.0f}s)")

    # Build concatenated feature matrices
    print("\nBuilding feature matrices...")
    feat_matrices = {}
    for config_name, bands in BAND_CONFIGS.items():
        Xs = [band_cache[b][0] for b in bands]
        keys = []
        for b in bands:
            keys.extend(band_cache[b][1])
        X = np.hstack(Xs)
        feat_matrices[config_name] = (X, keys)
        print(f"  {config_name}: {X.shape[1]} features")

    # Run all combos
    combos = []
    # Single best band with all classifiers
    for clf_name in CLASSIFIERS:
        combos.append(('band_0.5_5', clf_name))
        combos.append(('band_0.5_6', clf_name))
    # Multi-band configs with top classifiers
    for config_name in BAND_CONFIGS:
        if config_name.startswith('band_0.5_5') or config_name.startswith('band_0.5_6'):
            continue  # already covered
        for clf_name in ['GBM_deep', 'GBM', 'RF_deep', 'RF']:
            combos.append((config_name, clf_name))

    print(f"\nRunning {len(combos)} combinations...")
    print(f"{'Features':<35} {'Clf':<12} {'AUC':>8} {'Std':>7} {'nFeat':>6}")
    print("-" * 75)

    for config_name, clf_name in combos:
        method_id = f"{config_name}__{clf_name}"
        result_file = RESULTS_DIR / f"{method_id}.json"
        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            print(f"{config_name:<35} {clf_name:<12} {r['auc_mean']:>8.4f} {r['auc_std']:>7.4f} {r['n_features']:>6}  CACHED")
            continue

        X, keys = feat_matrices[config_name]
        clf = CLASSIFIERS[clf_name]()

        t0 = time.time()
        try:
            y_prob = cross_val_predict(clf, X, labels, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            auc = float(roc_auc_score(labels, y_prob))
            fold_aucs = []
            for tr, te in cv.split(X, labels):
                fold_aucs.append(float(roc_auc_score(labels[te], y_prob[te])))
            auc_std = float(np.std(fold_aucs))
        except Exception as e:
            auc, auc_std = 0.5, 0.0
            print(f"  ERROR: {e}")

        elapsed = time.time() - t0
        marker = ' ***' if auc > 0.81 else ' **' if auc > 0.80 else ' *' if auc > 0.79 else ''

        result = {
            'features': config_name,
            'classifier': clf_name,
            'auc_mean': round(auc, 4),
            'auc_std': round(auc_std, 4),
            'n_features': X.shape[1],
            'time': round(elapsed, 1),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"{config_name:<35} {clf_name:<12} {auc:>8.4f} {auc_std:>7.4f} {X.shape[1]:>6}{marker}")
        update_leaderboard()

    # Final leaderboard
    update_leaderboard()
    print("\nDone! Leaderboard updated.")


if __name__ == '__main__':
    main()
