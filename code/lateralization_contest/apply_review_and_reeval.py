#!/usr/bin/env python3
"""Apply MW's review decisions and re-evaluate top 10 models.

1. Load review results (relabeled cases)
2. Update the cached dataset with corrected labels
3. Re-run top 10 models with corrected labels
4. Update leaderboard
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

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

V2_CACHE = PROJECT_DIR / 'results' / 'lateralization_contest_v2' / '_cache'
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v3'

from lateralization_contest.optimize_bandpass import extract_spatial_all


def main():
    print("=" * 70)
    print("  Apply Review + Re-evaluate Top Models")
    print("=" * 70)

    # Load review results
    review = json.loads(Path('/dev/stdin').read_text()) if not Path(PROJECT_DIR / 'results' / 'lateral_review_results.json').exists() else None
    review_path = PROJECT_DIR / 'results' / 'lateral_review_results.json'
    if review_path.exists():
        with open(review_path) as f:
            review = json.load(f)
    else:
        print("ERROR: No review results found")
        return

    decisions = review['decisions']
    relabeled = [pid for pid, d in decisions.items() if d['decision'] == 'accept']
    kept = [pid for pid, d in decisions.items() if d['decision'] == 'reject']
    print(f"Review: {len(relabeled)} relabeled, {len(kept)} kept, {len(decisions)} total")

    # Count label changes
    lrda_to_grda = sum(1 for pid in relabeled if decisions[pid]['iiic_label'] == 'lrda')
    grda_to_lrda = sum(1 for pid in relabeled if decisions[pid]['iiic_label'] == 'grda')
    print(f"  LRDA → GRDA: {lrda_to_grda}")
    print(f"  GRDA → LRDA: {grda_to_lrda}")

    # Load original data
    with open(V2_CACHE / 'lateral_v2_data.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['df'].copy()
    segs = data['segs']

    # Apply label corrections
    n_updated = 0
    for pid, d in decisions.items():
        if d['decision'] != 'accept':
            continue
        new_label = d['new_label']
        mask = df['patient_id'] == pid
        if mask.any():
            old = df.loc[mask, 'subtype'].values[0]
            if old != new_label:
                df.loc[mask, 'subtype'] = new_label
                n_updated += 1

    print(f"\nUpdated {n_updated} labels in dataset")
    print(f"  LRDA: {(df['subtype']=='lrda').sum()}, GRDA: {(df['subtype']=='grda').sum()}")

    # Save corrected dataset
    corrected_data = {'df': df, 'segs': segs}
    corrected_path = V2_CACHE / 'lateral_v2_data_corrected.pkl'
    with open(corrected_path, 'wb') as f:
        pickle.dump(corrected_data, f)
    print(f"Saved corrected dataset to {corrected_path}")

    # Balance
    lrda_pids = df[df['subtype'] == 'lrda']['patient_id'].values
    grda_pids = df[df['subtype'] == 'grda']['patient_id'].values
    np.random.seed(42)
    n_min = min(len(lrda_pids), len(grda_pids))
    if len(lrda_pids) > n_min:
        lrda_sub = np.random.choice(lrda_pids, size=n_min, replace=False)
        grda_sub = grda_pids
    else:
        lrda_sub = lrda_pids
        grda_sub = np.random.choice(grda_pids, size=n_min, replace=False)
    use_pids = set(list(lrda_sub) + list(grda_sub))
    df_bal = df[df['patient_id'].isin(use_pids)].copy().reset_index(drop=True)
    labels = (df_bal['subtype'] == 'lrda').astype(int).values
    pids = df_bal['patient_id'].values
    print(f"\nBalanced: {(labels==1).sum()} LRDA + {(labels==0).sum()} GRDA = {len(labels)}")

    # Define top 10 models to re-evaluate
    TOP_MODELS = [
        ('band_0.5_5', 'GBM', dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),
        ('band_0.5_5', 'GBM_deep', dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
        ('band_0.5_6', 'GBM', dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),
        ('band_0.5_6', 'GBM_deep', dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
        ('spatial_all_0.5_4', 'GBM_deep', dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
        ('spatial_all_0.5_4', 'GBM', dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),
        ('band_0.5_5', 'RF_deep', None),
        ('band_0.5_5', 'RF', None),
        ('band_0.5_4+0.5_8', 'GBM', dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),
        ('band_0.5_5+1_8', 'GBM_deep', dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42)),
    ]

    # Extract features for needed bands
    print("\nExtracting features...")
    band_cache = {}
    needed_bands = set()
    for feat_name, _, _ in TOP_MODELS:
        if feat_name == 'spatial_all_0.5_4':
            needed_bands.add((0.5, 4.0))
        elif feat_name == 'band_0.5_5':
            needed_bands.add((0.5, 5.0))
        elif feat_name == 'band_0.5_6':
            needed_bands.add((0.5, 6.0))
        elif feat_name == 'band_0.5_4+0.5_8':
            needed_bands.add((0.5, 4.0))
            needed_bands.add((0.5, 8.0))
        elif feat_name == 'band_0.5_5+1_8':
            needed_bands.add((0.5, 5.0))
            needed_bands.add((1.0, 8.0))

    for lo, hi in sorted(needed_bands):
        t0 = time.time()
        feat_dicts = [extract_spatial_all(segs[pid], lo, hi) for pid in pids]
        all_keys = sorted(set(k for d in feat_dicts for k in d))
        X = np.array([[d.get(k, 0.0) for k in all_keys] for d in feat_dicts], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        band_cache[(lo, hi)] = (X, all_keys)
        print(f"  [{lo:.1f}-{hi:.0f}] Hz: {X.shape[1]} features ({time.time()-t0:.0f}s)")

    # Build feature matrices
    def get_X(feat_name):
        if feat_name == 'spatial_all_0.5_4':
            return band_cache[(0.5, 4.0)]
        elif feat_name == 'band_0.5_5':
            return band_cache[(0.5, 5.0)]
        elif feat_name == 'band_0.5_6':
            return band_cache[(0.5, 6.0)]
        elif feat_name == 'band_0.5_4+0.5_8':
            X1, k1 = band_cache[(0.5, 4.0)]
            X2, k2 = band_cache[(0.5, 8.0)]
            return np.hstack([X1, X2]), k1 + k2
        elif feat_name == 'band_0.5_5+1_8':
            X1, k1 = band_cache[(0.5, 5.0)]
            X2, k2 = band_cache[(1.0, 8.0)]
            return np.hstack([X1, X2]), k1 + k2

    # Run evaluation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    print(f"\n{'='*80}")
    print(f"  Re-evaluation with corrected labels")
    print(f"{'='*80}")
    print(f"{'Features':<25} {'Clf':<12} {'AUC':>8} {'±Std':>7}")
    print(f"{'-'*80}")

    for feat_name, clf_name, clf_params in TOP_MODELS:
        X, keys = get_X(feat_name)

        if clf_params:
            clf = GradientBoostingClassifier(**clf_params)
        elif clf_name == 'RF_deep':
            clf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42)

        y_prob = cross_val_predict(clf, X, labels, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
        auc = float(roc_auc_score(labels, y_prob))
        fold_aucs = []
        for tr, te in cv.split(X, labels):
            fold_aucs.append(float(roc_auc_score(labels[te], y_prob[te])))
        auc_std = float(np.std(fold_aucs))

        marker = ' ***' if auc > 0.84 else ' **' if auc > 0.82 else ' *' if auc > 0.80 else ''
        print(f"{feat_name:<25} {clf_name:<12} {auc:>8.4f} {auc_std:>7.4f}{marker}")

        result = {
            'features': f'{feat_name}_corrected',
            'classifier': clf_name,
            'auc_mean': round(auc, 4),
            'auc_std': round(auc_std, 4),
            'n_features': X.shape[1],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        results.append(result)

        # Save to v3 results
        method_id = f"{feat_name}_corrected__{clf_name}"
        with open(RESULTS_DIR / f"{method_id}.json", 'w') as f:
            json.dump(result, f, indent=2)

    # Update leaderboard
    from lateralization_contest.ml_classify_v3 import update_leaderboard
    update_leaderboard()

    print(f"\n{'='*80}")
    print("Done! Leaderboard updated with corrected-label results.")


if __name__ == '__main__':
    main()
