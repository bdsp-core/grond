#!/usr/bin/env python3
"""ML-based LRDA vs GRDA classification — unconstrained approaches.

Balanced data (subsample GRDA to match LRDA), 10-fold stratified CV.
Tries multiple feature sets × classifiers, reports AUC for each.
"""
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch, hilbert, butter, sosfiltfilt, coherence
from scipy.stats import kurtosis, skew
from numpy.fft import fft, ifft
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v2'
CACHE_DIR = RESULTS_DIR / '_cache'

FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])
MIDLINE_CHS = np.array([16, 17])


# ═══════════════════════════════════════════════════════════════════
#  Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def extract_features_basic(seg):
    """Basic per-hemisphere features (power, variance, peaks)."""
    sos = butter(4, [0.5 / (FS / 2), 4.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg, axis=1)

    feats = {}
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        data = seg_f[chs]
        feats[f'{side}_power_mean'] = np.mean(data ** 2)
        feats[f'{side}_power_std'] = np.std(np.mean(data ** 2, axis=1))
        feats[f'{side}_var'] = np.var(data)
        feats[f'{side}_rms'] = np.sqrt(np.mean(data ** 2))

        # Per-channel power distribution
        ch_powers = np.array([np.var(data[i]) for i in range(len(chs))])
        feats[f'{side}_power_max'] = np.max(ch_powers)
        feats[f'{side}_power_gini'] = _gini(ch_powers)
        feats[f'{side}_power_entropy'] = _entropy(ch_powers)

    # Asymmetry features
    feats['power_asym'] = (feats['R_power_mean'] - feats['L_power_mean']) / (
        feats['R_power_mean'] + feats['L_power_mean'] + 1e-12)
    feats['power_asym_abs'] = abs(feats['power_asym'])
    feats['var_asym'] = abs(feats['R_var'] - feats['L_var']) / (feats['R_var'] + feats['L_var'] + 1e-12)

    return feats


def extract_features_spectral(seg):
    """Spectral features per hemisphere."""
    sos = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg, axis=1)

    feats = {}
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        sig = np.mean(seg_f[chs], axis=0)
        f, pxx = welch(sig, fs=FS, nperseg=400)
        delta = (f >= 0.5) & (f <= 3.5)

        if delta.any() and pxx[delta].sum() > 0:
            pxx_d = pxx[delta]
            peak_idx = np.argmax(pxx_d)
            peak_f = f[delta][peak_idx]
            feats[f'{side}_peak_freq'] = peak_f
            feats[f'{side}_peak_power'] = float(pxx_d[peak_idx])
            feats[f'{side}_mean_power'] = float(np.mean(pxx_d))
            feats[f'{side}_peak_ratio'] = float(pxx_d[peak_idx] / np.mean(pxx_d))

            # Spectral concentration
            narrow = (f >= peak_f - 0.3) & (f <= peak_f + 0.3) & delta
            feats[f'{side}_spec_conc'] = float(pxx[narrow].sum() / pxx[delta].sum())

            # Spectral flatness
            pxx_pos = pxx_d[pxx_d > 0]
            if len(pxx_pos) > 1:
                geo = np.exp(np.mean(np.log(pxx_pos)))
                feats[f'{side}_spec_flat'] = float(geo / np.mean(pxx_pos))
            else:
                feats[f'{side}_spec_flat'] = 1.0

            # Spectral entropy
            p = pxx_d / pxx_d.sum()
            p = p[p > 0]
            feats[f'{side}_spec_entropy'] = float(-np.sum(p * np.log(p)))
        else:
            for k in ['peak_freq', 'peak_power', 'mean_power', 'peak_ratio',
                       'spec_conc', 'spec_flat', 'spec_entropy']:
                feats[f'{side}_{k}'] = 0.0

    # Spectral asymmetry
    for k in ['peak_power', 'mean_power', 'peak_ratio', 'spec_conc', 'spec_flat', 'spec_entropy']:
        lv, rv = feats.get(f'L_{k}', 0), feats.get(f'R_{k}', 0)
        feats[f'{k}_asym'] = abs(rv - lv) / (abs(rv) + abs(lv) + 1e-12)
    feats['freq_diff'] = abs(feats.get('L_peak_freq', 0) - feats.get('R_peak_freq', 0))

    return feats


def extract_features_rhythm(seg):
    """Rhythmicity features per hemisphere."""
    sos = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg, axis=1)

    feats = {}
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        sig = np.mean(seg_f[chs], axis=0)

        # ACF peak
        x = sig - np.mean(sig)
        n = len(x)
        acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
        acf = acf / max(acf[0], 1e-12)
        min_lag, max_lag = int(FS / 3.5), min(int(FS / 0.5), n - 1)
        acf_seg = acf[min_lag:max_lag]
        feats[f'{side}_acf_peak'] = float(np.max(acf_seg)) if len(acf_seg) > 0 else 0.0

        # Hilbert regularity
        analytic = hilbert(sig)
        inst_freq = np.diff(np.unwrap(np.angle(analytic))) / (2 * np.pi / FS)
        inst_freq = inst_freq[(inst_freq > 0.3) & (inst_freq < 4.0)]
        if len(inst_freq) > 10:
            feats[f'{side}_hilbert_cv'] = float(np.std(inst_freq) / max(np.median(inst_freq), 0.01))
            feats[f'{side}_hilbert_med'] = float(np.median(inst_freq))
        else:
            feats[f'{side}_hilbert_cv'] = 1.0
            feats[f'{side}_hilbert_med'] = 0.0

        # Kurtosis & skewness
        feats[f'{side}_kurtosis'] = float(kurtosis(sig))
        feats[f'{side}_skewness'] = float(skew(sig))

        # Gradient energy ratio (smoothness)
        sig_e = np.mean(sig ** 2)
        grad_e = np.mean(np.diff(sig) ** 2)
        feats[f'{side}_smoothness'] = float(sig_e / (sig_e + grad_e + 1e-12))

    # Asymmetries
    for k in ['acf_peak', 'hilbert_cv', 'kurtosis', 'smoothness']:
        lv, rv = feats.get(f'L_{k}', 0), feats.get(f'R_{k}', 0)
        feats[f'{k}_asym'] = abs(rv - lv) / (abs(rv) + abs(lv) + 1e-12)

    return feats


def extract_features_connectivity(seg):
    """Cross-hemisphere connectivity features."""
    sos = butter(4, [0.5 / (FS / 2), 4.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg, axis=1)

    feats = {}

    # Inter-hemisphere envelope correlation
    left_env = np.abs(hilbert(np.mean(seg_f[LEFT_CHS], axis=0)))
    right_env = np.abs(hilbert(np.mean(seg_f[RIGHT_CHS], axis=0)))
    if np.std(left_env) > 1e-12 and np.std(right_env) > 1e-12:
        feats['inter_hemi_corr'] = float(np.corrcoef(left_env, right_env)[0, 1])
    else:
        feats['inter_hemi_corr'] = 0.0
    feats['inter_hemi_corr_abs'] = abs(feats['inter_hemi_corr'])

    # Intra-hemisphere coherence
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        plv_vals = []
        phases = np.array([np.angle(hilbert(seg_f[ch])) for ch in chs])
        for i in range(len(chs)):
            for j in range(i + 1, min(i + 3, len(chs))):
                plv = float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j])))))
                plv_vals.append(plv)
        feats[f'{side}_plv'] = float(np.mean(plv_vals)) if plv_vals else 0.0

    feats['plv_asym'] = abs(feats['R_plv'] - feats['L_plv']) / (
        feats['R_plv'] + feats['L_plv'] + 1e-12)

    # SVD dominance per hemisphere
    for side, chs in [('L', LEFT_CHS), ('R', RIGHT_CHS)]:
        try:
            _, s, _ = np.linalg.svd(seg_f[chs], full_matrices=False)
            feats[f'{side}_svd_dom'] = float(s[0] / s.sum()) if s.sum() > 1e-12 else 0.0
            feats[f'{side}_svd_ratio12'] = float(s[0] / s[1]) if s[1] > 1e-12 else 0.0
        except:
            feats[f'{side}_svd_dom'] = 0.0
            feats[f'{side}_svd_ratio12'] = 0.0

    feats['svd_dom_asym'] = abs(feats['R_svd_dom'] - feats['L_svd_dom']) / (
        feats['R_svd_dom'] + feats['L_svd_dom'] + 1e-12)

    return feats


def extract_features_channel(seg):
    """Per-channel features (18 channels × key metrics)."""
    sos = butter(4, [0.5 / (FS / 2), 4.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg, axis=1)

    feats = {}
    for ch in range(18):
        sig = seg_f[ch]
        feats[f'ch{ch:02d}_power'] = float(np.var(sig))
        feats[f'ch{ch:02d}_rms'] = float(np.sqrt(np.mean(sig ** 2)))

        # Spectral peak ratio
        f, pxx = welch(sig, fs=FS, nperseg=400)
        delta = (f >= 0.5) & (f <= 3.5)
        if delta.any() and pxx[delta].mean() > 0:
            feats[f'ch{ch:02d}_peak_ratio'] = float(np.max(pxx[delta]) / np.mean(pxx[delta]))
        else:
            feats[f'ch{ch:02d}_peak_ratio'] = 0.0

    # Spatial pattern: which channels are most active?
    powers = np.array([feats[f'ch{ch:02d}_power'] for ch in range(18)])
    if powers.sum() > 0:
        powers_norm = powers / powers.sum()
        feats['spatial_entropy'] = float(-np.sum(powers_norm[powers_norm > 0] *
                                                  np.log(powers_norm[powers_norm > 0])))
        feats['spatial_gini'] = _gini(powers)
        feats['n_active_channels'] = float(np.sum(powers > 0.1 * np.max(powers)))
    else:
        feats['spatial_entropy'] = 0.0
        feats['spatial_gini'] = 0.0
        feats['n_active_channels'] = 0.0

    return feats


def extract_features_cached(pid, cache_dir=CACHE_DIR):
    """Load cached per-patient scores from Round 2 single methods."""
    feats = {}
    for path in sorted(cache_dir.glob('*_scores.json')):
        method_name = path.stem.replace('_scores', '')
        with open(path) as f:
            scores = json.load(f)
        if pid in scores:
            s = scores[pid]
            feats[f'{method_name}_asym'] = s['asymmetry']
            feats[f'{method_name}_lat'] = s['laterality_index']
            feats[f'{method_name}_left'] = s['left_score']
            feats[f'{method_name}_right'] = s['right_score']
    return feats


def extract_all_features(seg, pid=None):
    """Extract all feature sets, return combined dict."""
    feats = {}
    feats.update(extract_features_basic(seg))
    feats.update(extract_features_spectral(seg))
    feats.update(extract_features_rhythm(seg))
    feats.update(extract_features_connectivity(seg))
    feats.update(extract_features_channel(seg))
    if pid:
        feats.update(extract_features_cached(pid))
    return feats


def _gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


def _entropy(x):
    x = np.abs(x)
    s = x.sum()
    if s == 0:
        return 0.0
    p = x / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

FEATURE_SETS = {
    'cached_only': lambda seg, pid: extract_features_cached(pid),
    'basic': lambda seg, pid: extract_features_basic(seg),
    'spectral': lambda seg, pid: extract_features_spectral(seg),
    'rhythm': lambda seg, pid: extract_features_rhythm(seg),
    'connectivity': lambda seg, pid: extract_features_connectivity(seg),
    'basic+spectral': lambda seg, pid: {**extract_features_basic(seg),
                                         **extract_features_spectral(seg)},
    'all_signal': lambda seg, pid: {**extract_features_basic(seg),
                                     **extract_features_spectral(seg),
                                     **extract_features_rhythm(seg),
                                     **extract_features_connectivity(seg)},
    'all_signal+channel': lambda seg, pid: {**extract_features_basic(seg),
                                             **extract_features_spectral(seg),
                                             **extract_features_rhythm(seg),
                                             **extract_features_connectivity(seg),
                                             **extract_features_channel(seg)},
    'everything': lambda seg, pid: extract_all_features(seg, pid),
}

CLASSIFIERS = {
    'LogReg': lambda: Pipeline([('scaler', StandardScaler()),
                                 ('clf', LogisticRegression(max_iter=1000, C=1.0))]),
    'LogReg_L1': lambda: Pipeline([('scaler', StandardScaler()),
                                    ('clf', LogisticRegression(max_iter=1000, C=0.5,
                                                                penalty='l1', solver='saga'))]),
    'RF': lambda: RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    'RF_deep': lambda: RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42),
    'ExtraTrees': lambda: ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42),
    'GBM': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                               learning_rate=0.1, random_state=42),
    'GBM_deep': lambda: GradientBoostingClassifier(n_estimators=200, max_depth=6,
                                                    learning_rate=0.05, random_state=42),
    'AdaBoost': lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': lambda: Pipeline([('scaler', StandardScaler()),
                              ('clf', SVC(kernel='rbf', probability=True, C=1.0))]),
    'SVM_linear': lambda: Pipeline([('scaler', StandardScaler()),
                                     ('clf', SVC(kernel='linear', probability=True, C=1.0))]),
    'KNN': lambda: Pipeline([('scaler', StandardScaler()),
                              ('clf', KNeighborsClassifier(n_neighbors=11))]),
    'LDA': lambda: Pipeline([('scaler', StandardScaler()),
                              ('clf', LinearDiscriminantAnalysis())]),
    'MLP': lambda: Pipeline([('scaler', StandardScaler()),
                              ('clf', MLPClassifier(hidden_layer_sizes=(64, 32),
                                                     max_iter=500, random_state=42))]),
    'MLP_large': lambda: Pipeline([('scaler', StandardScaler()),
                                    ('clf', MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                                                           max_iter=500, random_state=42))]),
}


def main():
    import pickle

    print("Loading data...")
    cache_file = CACHE_DIR / 'lateral_v2_data.pkl'
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    df = data['df']
    segs = data['segs']

    # Balance: subsample GRDA to match LRDA
    lrda_pids = df[df['subtype'] == 'lrda']['patient_id'].values
    grda_pids = df[df['subtype'] == 'grda']['patient_id'].values
    n_lrda = len(lrda_pids)
    np.random.seed(42)
    grda_sub = np.random.choice(grda_pids, size=n_lrda, replace=False)
    use_pids = set(list(lrda_pids) + list(grda_sub))

    df_bal = df[df['patient_id'].isin(use_pids)].copy().reset_index(drop=True)
    print(f"Balanced: {(df_bal['subtype']=='lrda').sum()} LRDA + "
          f"{(df_bal['subtype']=='grda').sum()} GRDA = {len(df_bal)}")

    # Extract features for each feature set
    print("\nExtracting features...")
    feature_matrices = {}
    labels = (df_bal['subtype'] == 'lrda').astype(int).values

    for fs_name, fs_func in FEATURE_SETS.items():
        t0 = time.time()
        feat_dicts = []
        for _, row in df_bal.iterrows():
            pid = row['patient_id']
            seg = segs[pid]
            feat_dicts.append(fs_func(seg, pid))

        # Convert to matrix
        all_keys = sorted(set(k for d in feat_dicts for k in d))
        X = np.array([[d.get(k, 0.0) for k in all_keys] for d in feat_dicts], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        feature_matrices[fs_name] = (X, all_keys)
        print(f"  {fs_name}: {X.shape[1]} features ({time.time()-t0:.1f}s)")

    # Run cross-validation for each combination
    print(f"\n{'=' * 90}")
    print(f"  LRDA vs GRDA Classification — 10-fold Stratified CV")
    print(f"{'=' * 90}")
    print(f"{'Features':<22} {'Classifier':<10} {'AUC':>8} {'Std':>6} {'nFeat':>6}")
    print(f"{'-' * 90}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    for fs_name, (X, keys) in feature_matrices.items():
        for clf_name, clf_factory in CLASSIFIERS.items():
            t0 = time.time()
            clf = clf_factory()
            try:
                scores = cross_val_score(clf, X, labels, cv=cv, scoring='roc_auc', n_jobs=-1)
                mean_auc = float(np.mean(scores))
                std_auc = float(np.std(scores))
            except Exception as e:
                mean_auc = 0.0
                std_auc = 0.0

            elapsed = time.time() - t0
            results.append({
                'features': fs_name,
                'classifier': clf_name,
                'auc_mean': round(mean_auc, 4),
                'auc_std': round(std_auc, 4),
                'n_features': X.shape[1],
                'time': round(elapsed, 1),
            })
            print(f"{fs_name:<22} {clf_name:<10} {mean_auc:>8.4f} {std_auc:>6.4f} {X.shape[1]:>6}")

    # Sort and show top results
    results.sort(key=lambda r: -r['auc_mean'])
    print(f"\n{'=' * 90}")
    print(f"  TOP 10 COMBINATIONS")
    print(f"{'=' * 90}")
    for i, r in enumerate(results[:10]):
        print(f"  {i+1}. {r['features']} + {r['classifier']}: "
              f"AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f} ({r['n_features']} features)")

    # Save results
    out = RESULTS_DIR / 'ml_classification_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

    # Generate HTML leaderboard
    _update_ml_leaderboard(results)

    # Feature importance for best model
    print(f"\n{'=' * 90}")
    print(f"  FEATURE IMPORTANCE (best RF/GBM model)")
    print(f"{'=' * 90}")
    best_tree = [r for r in results if r['classifier'] in ('RF', 'GBM')]
    if best_tree:
        best = best_tree[0]
        X, keys = feature_matrices[best['features']]
        if best['classifier'] == 'RF':
            clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        else:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(X, labels)
        importances = clf.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        for i, idx in enumerate(top_idx):
            print(f"  {i+1:>3}. {keys[idx]:<40} {importances[idx]:.4f}")


def _update_ml_leaderboard(results):
    """Generate auto-updating HTML leaderboard for ML results."""
    results_sorted = sorted(results, key=lambda r: -r['auc_mean'])
    n = len(results_sorted)

    rows = ""
    for i, r in enumerate(results_sorted):
        auc = r['auc_mean']
        color = ('#44cc88' if auc > 0.70 else '#88cc44' if auc > 0.65 else
                 '#cccc44' if auc > 0.60 else '#cc8844' if auc > 0.55 else '#cc4444')
        bar_width = max(0, (auc - 0.5) * 200)  # scale 0.5-1.0 to 0-100%
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td style='font-weight:bold'>{r['features']}</td>"
            f"<td>{r['classifier']}</td>"
            f"<td style='color:{color};font-weight:bold;font-size:1.1em'>{auc:.4f}</td>"
            f"<td>±{r['auc_std']:.4f}</td>"
            f"<td><div style='background:#333;border-radius:4px;height:14px;width:100px'>"
            f"<div style='background:{color};height:100%;width:{bar_width}px;border-radius:4px'>"
            f"</div></div></td>"
            f"<td>{r['n_features']}</td>"
            f"<td>{r['time']:.1f}s</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html><html><head><title>ML LRDA vs GRDA Classification</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#44cc88;margin-bottom:5px}}
h2{{color:#888;font-weight:normal;font-size:14px;margin-top:0}}
table{{border-collapse:collapse;width:100%;margin-top:15px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555;font-size:12px}}
td{{padding:6px 10px;border-bottom:1px solid #333}}
tr:hover{{background:#2a2a2a}}
.meta{{color:#666;font-size:11px;margin-top:15px}}
</style></head><body>
<h1>ML Classification: LRDA vs GRDA</h1>
<h2>Unconstrained approaches — 10-fold stratified CV, balanced data (311+311)</h2>
<p style="color:#aaa">{n} feature×classifier combinations tested</p>
<p style="color:#777;font-size:12px">Updated {time.strftime('%H:%M:%S')}</p>
<table>
<tr><th>#</th><th>Features</th><th>Classifier</th>
<th>AUC ↓</th><th>Std</th><th>AUC bar</th><th>nFeat</th><th>Time</th></tr>
{rows}</table>
<div class="meta">
<p>Feature sets: cached_only (20 method scores), basic, spectral, rhythm, connectivity,
basic+spectral, all_signal, all_signal+channel, everything</p>
<p>Classifiers: LogReg, LogReg_L1, RF, RF_deep, ExtraTrees, GBM, GBM_deep, AdaBoost,
SVM, SVM_linear, KNN, LDA, MLP, MLP_large</p>
</div>
</body></html>"""

    out = RESULTS_DIR.parent / 'ml_lrda_grda_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)
    print(f"\nLeaderboard: {out}")


if __name__ == '__main__':
    main()
