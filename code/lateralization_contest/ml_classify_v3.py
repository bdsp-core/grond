#!/usr/bin/env python3
"""V3 ML Classification Contest — comprehensive features for LRDA vs GRDA.

Runs a specific batch of (feature_set, classifier) combinations.
Usage:
    python ml_classify_v3.py batch1   # runs combos 0-9
    python ml_classify_v3.py batch2   # runs combos 10-19
    python ml_classify_v3.py batch3   # runs combos 20-29
    python ml_classify_v3.py all      # runs everything
"""
import sys
import json
import time
import pickle
import warnings
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR_V2 = PROJECT_DIR / 'results' / 'lateralization_contest_v2' / '_cache'
FEAT_CACHE = RESULTS_DIR / '_feat_cache.pkl'

from lateralization_contest.features_v3 import (
    extract_all, extract_cached, FEATURE_FAMILIES
)

# ═══════════════════════════════════════════════════════════════
# Feature set definitions
# ═══════════════════════════════════════════════════════════════

FEATURE_SETS = {
    'focality': ['focality'],
    'homologous': ['homologous'],
    'connectivity': ['connectivity'],
    'rhythmicity': ['rhythmicity'],
    'waveform': ['waveform'],
    'timefreq': ['timefreq'],
    'propagation': ['propagation'],
    'perchannel': ['perchannel'],
    'bgsub': ['bgsub'],
    'spatial_all': ['focality', 'homologous', 'perchannel', 'bgsub'],
    'sync_all': ['connectivity', 'homologous'],
    'rhythm_morph': ['rhythmicity', 'waveform'],
    'clinical_core': ['focality', 'homologous', 'connectivity', 'rhythmicity'],
    'all_v3': list(FEATURE_FAMILIES.keys()),
    'all_v3+cached': None,  # special: all families + round 2 cached scores
}

CLASSIFIERS = {
    'RF': lambda: RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42),
    'RF_deep': lambda: RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=3, random_state=42),
    'GBM': lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                               subsample=0.8, random_state=42),
    'GBM_deep': lambda: GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                                    subsample=0.8, random_state=42),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=42),
    'LogReg': lambda: Pipeline([('s', StandardScaler()), ('c', LogisticRegression(max_iter=2000, C=1.0))]),
    'LogReg_L1': lambda: Pipeline([('s', StandardScaler()),
                                    ('c', LogisticRegression(max_iter=2000, C=0.3, penalty='l1', solver='saga'))]),
    'SVM': lambda: Pipeline([('s', StandardScaler()), ('c', SVC(kernel='rbf', probability=True, C=1.0))]),
    'LDA': lambda: Pipeline([('s', StandardScaler()), ('c', LinearDiscriminantAnalysis())]),
    'MLP': lambda: Pipeline([('s', StandardScaler()),
                              ('c', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42))]),
}

# Define which combos to run (feature_set, classifier)
ALL_COMBOS = []
# Individual families with RF (to see which family matters)
for fs in ['focality', 'homologous', 'connectivity', 'rhythmicity', 'waveform',
           'timefreq', 'propagation', 'perchannel', 'bgsub']:
    ALL_COMBOS.append((fs, 'RF'))

# Combined feature sets with multiple classifiers
for fs in ['spatial_all', 'sync_all', 'rhythm_morph', 'clinical_core', 'all_v3', 'all_v3+cached']:
    for clf in ['RF', 'RF_deep', 'GBM', 'GBM_deep', 'ExtraTrees',
                'LogReg', 'LogReg_L1', 'SVM', 'LDA', 'MLP']:
        ALL_COMBOS.append((fs, clf))

BATCHES = {
    'batch1': ALL_COMBOS[:20],
    'batch2': ALL_COMBOS[20:40],
    'batch3': ALL_COMBOS[40:60],
    'batch4': ALL_COMBOS[60:],
    'all': ALL_COMBOS,
}


def load_data():
    cache_file = CACHE_DIR_V2 / 'lateral_v2_data.pkl'
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


def extract_feature_matrix(df_bal, segs, families, include_cached=False):
    """Extract features for all patients. Uses caching."""
    cache_key = '+'.join(sorted(families)) + ('+cached' if include_cached else '')

    # Try loading from feature cache
    if FEAT_CACHE.exists():
        with open(FEAT_CACHE, 'rb') as f:
            feat_cache = pickle.load(f)
    else:
        feat_cache = {}

    if cache_key in feat_cache:
        return feat_cache[cache_key]

    feat_dicts = []
    for _, row in df_bal.iterrows():
        pid = row['patient_id']
        seg = segs[pid]
        feats = extract_all(seg, families)
        if include_cached:
            feats.update(extract_cached(pid))
        feat_dicts.append(feats)

    all_keys = sorted(set(k for d in feat_dicts for k in d))
    X = np.array([[d.get(k, 0.0) for k in all_keys] for d in feat_dicts], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    result = (X, all_keys)
    feat_cache[cache_key] = result
    with open(FEAT_CACHE, 'wb') as f:
        pickle.dump(feat_cache, f)

    return result


def update_leaderboard():
    """Generate HTML leaderboard from all saved results."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        if path.name.startswith('_'):
            continue
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and 'auc_mean' in data:
            results.append(data)
    if not results:
        return

    results.sort(key=lambda r: -r.get('auc_mean', 0))

    rows = ""
    for i, r in enumerate(results):
        auc = r['auc_mean']
        color = ('#44cc88' if auc > 0.80 else '#88cc44' if auc > 0.75 else
                 '#cccc44' if auc > 0.70 else '#cc8844' if auc > 0.65 else '#cc4444')
        bar_w = max(0, (auc - 0.5) * 200)
        rows += (
            f"<tr><td>{i+1}</td>"
            f"<td style='font-weight:bold'>{r['features']}</td>"
            f"<td>{r['classifier']}</td>"
            f"<td style='color:{color};font-weight:bold;font-size:1.1em'>{auc:.4f}</td>"
            f"<td>±{r['auc_std']:.4f}</td>"
            f"<td><div style='background:#333;border-radius:4px;height:14px;width:120px'>"
            f"<div style='background:{color};height:100%;width:{bar_w}px;border-radius:4px'></div></div></td>"
            f"<td>{r['n_features']}</td>"
            f"</tr>\n"
        )

    n = len(results)
    best = results[0] if results else {}
    html = f"""<!DOCTYPE html><html><head><title>V3 LRDA vs GRDA Contest</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#44cc88;margin-bottom:5px}}
h2{{color:#888;font-weight:normal;font-size:14px;margin-top:0}}
table{{border-collapse:collapse;width:100%;margin-top:15px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555;font-size:12px}}
td{{padding:6px 10px;border-bottom:1px solid #333}}
tr:hover{{background:#2a2a2a}}
tr:first-child td{{background:#1a2a1a}}
.best{{color:#44cc88;font-size:24px;margin:10px 0}}
</style></head><body>
<h1>V3 Contest: LRDA vs GRDA Classification</h1>
<h2>Comprehensive features — 9 families — 10-fold stratified CV — balanced (311+311)</h2>
<div class="best">Best AUC: {best.get('auc_mean', 0):.4f} ({best.get('features', '?')} + {best.get('classifier', '?')})</div>
<p style="color:#777">{n}/{len(ALL_COMBOS)} combinations · Updated {time.strftime('%H:%M:%S')}</p>
<table>
<tr><th>#</th><th>Features</th><th>Classifier</th><th>AUC ↓</th><th>Std</th><th>AUC bar</th><th>nFeat</th></tr>
{rows}</table>
<p style="color:#555;font-size:11px;margin-top:20px">
Feature families: focality, homologous pairs, connectivity, rhythmicity (Hilbert freq),
waveform, time-frequency stability, propagation, per-channel, background subtraction
</p></body></html>"""

    out = RESULTS_DIR.parent / 'v3_lrda_grda_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)


def main():
    batch_name = sys.argv[1] if len(sys.argv) > 1 else 'all'
    combos = BATCHES.get(batch_name, ALL_COMBOS)
    print(f"V3 Contest — batch '{batch_name}': {len(combos)} combinations")

    data = load_data()
    df = data['df']
    segs = data['segs']

    # Balance
    lrda_pids = df[df['subtype'] == 'lrda']['patient_id'].values
    grda_pids = df[df['subtype'] == 'grda']['patient_id'].values
    np.random.seed(42)
    grda_sub = np.random.choice(grda_pids, size=len(lrda_pids), replace=False)
    use_pids = set(list(lrda_pids) + list(grda_sub))
    df_bal = df[df['patient_id'].isin(use_pids)].copy().reset_index(drop=True)
    labels = (df_bal['subtype'] == 'lrda').astype(int).values
    print(f"Balanced: {(labels==1).sum()} LRDA + {(labels==0).sum()} GRDA = {len(labels)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Pre-extract all needed feature sets
    print("Extracting features...")
    needed_families = set()
    for fs_name, _ in combos:
        if fs_name == 'all_v3+cached':
            needed_families.update(FEATURE_FAMILIES.keys())
        elif fs_name in FEATURE_SETS:
            families = FEATURE_SETS[fs_name]
            if families:
                needed_families.update(families)

    feat_matrices = {}
    for fs_name in set(fs for fs, _ in combos):
        if fs_name == 'all_v3+cached':
            families = list(FEATURE_FAMILIES.keys())
            X, keys = extract_feature_matrix(df_bal, segs, families, include_cached=True)
        else:
            families = FEATURE_SETS[fs_name]
            X, keys = extract_feature_matrix(df_bal, segs, families, include_cached=False)
        feat_matrices[fs_name] = (X, keys)
        print(f"  {fs_name}: {X.shape[1]} features")

    print(f"\nRunning {len(combos)} combinations...")
    for i, (fs_name, clf_name) in enumerate(combos):
        method_id = f"{fs_name}__{clf_name}"
        result_file = RESULTS_DIR / f"{method_id}.json"

        # Skip if already done
        if result_file.exists():
            print(f"  [{i+1}/{len(combos)}] {method_id} — CACHED")
            continue

        X, keys = feat_matrices[fs_name]
        clf = CLASSIFIERS[clf_name]()

        t0 = time.time()
        try:
            # Get predicted probabilities via cross-validation
            y_prob = cross_val_predict(clf, X, labels, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            auc = float(roc_auc_score(labels, y_prob))
            # Also get per-fold AUCs for std
            fold_aucs = []
            for train_idx, test_idx in cv.split(X, labels):
                fold_aucs.append(float(roc_auc_score(labels[test_idx], y_prob[test_idx])))
            auc_std = float(np.std(fold_aucs))
        except Exception as e:
            auc = 0.5
            auc_std = 0.0
            print(f"    ERROR: {e}")

        elapsed = time.time() - t0

        result = {
            'features': fs_name,
            'classifier': clf_name,
            'auc_mean': round(auc, 4),
            'auc_std': round(auc_std, 4),
            'n_features': X.shape[1],
            'time': round(elapsed, 1),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        marker = '***' if auc > 0.75 else '**' if auc > 0.70 else '*' if auc > 0.65 else ''
        print(f"  [{i+1}/{len(combos)}] {method_id}: AUC={auc:.4f}±{auc_std:.4f} "
              f"({X.shape[1]}feat, {elapsed:.1f}s) {marker}")

        update_leaderboard()

    print("\nDone!")
    update_leaderboard()


if __name__ == '__main__':
    main()
