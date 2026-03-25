#!/usr/bin/env python3
"""Optimize the bandpass filter for LRDA vs GRDA classification.

Tests multiple (lo, hi) combinations using the winning feature set + classifier.
Extracts fresh features for each bandpass setting.
"""
import sys
import json
import time
import pickle
import warnings
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, welch, hilbert, find_peaks
from scipy.stats import kurtosis, skew, pearsonr
from numpy.fft import fft, ifft
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

V2_CACHE = PROJECT_DIR / 'results' / 'lateralization_contest_v2' / '_cache'
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v3'

FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])
HOMOLOGOUS_PAIRS = [
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
]


def bp_filter(seg, lo, hi):
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, seg, axis=1)


def extract_spatial_all(seg, lo, hi):
    """Extract spatial_all features (focality + homologous + perchannel + bgsub)
    with a given bandpass."""
    seg_f = bp_filter(seg, lo, hi)
    feats = {}

    # ── Focality ──
    ch_power = np.array([np.var(seg_f[ch]) for ch in range(18)])
    total = ch_power.sum()
    if total > 1e-12:
        p = ch_power / total
        feats['foc_entropy'] = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        feats['foc_gini'] = _gini(ch_power)
        feats['foc_max_frac'] = float(np.max(p))
        sp = np.sort(p)[::-1]
        feats['foc_top2_frac'] = float(sp[:2].sum())
        feats['foc_top3_frac'] = float(sp[:3].sum())
        feats['foc_n_active'] = float(np.sum(ch_power > 0.1 * np.max(ch_power)))
        max_ch = np.argmax(ch_power)
        feats['foc_spread'] = float(np.sqrt(np.average((np.arange(18) - max_ch) ** 2, weights=p)))
        lt = ch_power[LEFT_CHS].sum()
        rt = ch_power[RIGHT_CHS].sum()
        feats['foc_left_frac'] = float(lt / total)
        feats['foc_right_frac'] = float(rt / total)
        feats['foc_hemi_imbalance'] = abs(lt - rt) / (lt + rt + 1e-12)
        feats['foc_within_left_var'] = float(np.var(ch_power[LEFT_CHS]))
        feats['foc_within_right_var'] = float(np.var(ch_power[RIGHT_CHS]))
        feats['foc_within_var_ratio'] = abs(feats['foc_within_left_var'] - feats['foc_within_right_var']) / (
            feats['foc_within_left_var'] + feats['foc_within_right_var'] + 1e-12)
    else:
        for k in ['entropy', 'gini', 'max_frac', 'top2_frac', 'top3_frac', 'n_active',
                   'spread', 'left_frac', 'right_frac', 'hemi_imbalance',
                   'within_left_var', 'within_right_var', 'within_var_ratio']:
            feats[f'foc_{k}'] = 0.0

    # ── Homologous pairs ──
    corrs, plvs, power_ratios, freq_diffs = [], [], [], []
    for i, (l_ch, r_ch) in enumerate(HOMOLOGOUS_PAIRS):
        l_sig, r_sig = seg_f[l_ch], seg_f[r_ch]
        if np.std(l_sig) > 1e-12 and np.std(r_sig) > 1e-12:
            c = float(pearsonr(l_sig, r_sig)[0])
        else:
            c = 0.0
        corrs.append(c)
        feats[f'hom_corr_{i}'] = c

        p1 = np.angle(hilbert(l_sig))
        p2 = np.angle(hilbert(r_sig))
        plv = float(np.abs(np.mean(np.exp(1j * (p1 - p2)))))
        plvs.append(plv)
        feats[f'hom_plv_{i}'] = plv

        lp, rp = np.var(l_sig), np.var(r_sig)
        pr = abs(lp - rp) / (lp + rp + 1e-12)
        power_ratios.append(pr)
        feats[f'hom_power_ratio_{i}'] = pr

        for sig, side in [(l_sig, 'l'), (r_sig, 'r')]:
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= lo) & (f <= hi)
            if delta.any() and pxx[delta].sum() > 0:
                feats[f'hom_peakf_{side}_{i}'] = float(f[delta][np.argmax(pxx[delta])])
            else:
                feats[f'hom_peakf_{side}_{i}'] = 0.0
        freq_diffs.append(abs(feats[f'hom_peakf_l_{i}'] - feats[f'hom_peakf_r_{i}']))

    feats['hom_corr_mean'] = float(np.mean(corrs))
    feats['hom_corr_min'] = float(np.min(corrs))
    feats['hom_corr_std'] = float(np.std(corrs))
    feats['hom_plv_mean'] = float(np.mean(plvs))
    feats['hom_plv_min'] = float(np.min(plvs))
    feats['hom_plv_std'] = float(np.std(plvs))
    feats['hom_power_ratio_mean'] = float(np.mean(power_ratios))
    feats['hom_power_ratio_max'] = float(np.max(power_ratios))
    feats['hom_power_ratio_std'] = float(np.std(power_ratios))
    feats['hom_freq_diff_mean'] = float(np.mean(freq_diffs))
    feats['hom_freq_diff_max'] = float(np.max(freq_diffs))
    feats['hom_n_asym_pairs'] = float(np.sum(np.array(power_ratios) > 0.3))

    # ── Per-channel ──
    for ch in range(18):
        sig = seg_f[ch]
        feats[f'ch{ch:02d}_power'] = float(np.var(sig))
        f, pxx = welch(sig, fs=FS, nperseg=400)
        delta = (f >= lo) & (f <= hi)
        if delta.any() and pxx[delta].mean() > 0:
            feats[f'ch{ch:02d}_peak_ratio'] = float(np.max(pxx[delta]) / np.mean(pxx[delta]))
        else:
            feats[f'ch{ch:02d}_peak_ratio'] = 0.0
        # Hilbert Q
        if np.std(sig) > 1e-10:
            analytic = hilbert(sig)
            inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
            mask = (inst_freq > 0.3) & (inst_freq < max(hi, 4.0))
            valid = inst_freq[mask]
            if len(valid) > 20:
                cv = np.std(valid) / max(np.median(valid), 0.01)
                feats[f'ch{ch:02d}_hilbert_q'] = max(0, 1 - 2 * cv)
            else:
                feats[f'ch{ch:02d}_hilbert_q'] = 0.0
        else:
            feats[f'ch{ch:02d}_hilbert_q'] = 0.0

    # ── Background subtraction ──
    residual_powers, residual_rhythmicities = [], []
    for l_ch, r_ch in HOMOLOGOUS_PAIRS:
        l_sig = seg_f[l_ch]
        r_sig = seg_f[r_ch]
        l_n = l_sig / max(np.std(l_sig), 1e-12)
        r_n = r_sig / max(np.std(r_sig), 1e-12)
        residual = l_n - r_n
        residual_powers.append(float(np.var(residual)))
        if np.std(residual) > 1e-10:
            analytic = hilbert(residual)
            inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
            mask = (inst_freq > 0.3) & (inst_freq < max(hi, 4.0))
            valid = inst_freq[mask]
            if len(valid) > 20:
                cv = np.std(valid) / max(np.median(valid), 0.01)
                residual_rhythmicities.append(max(0, 1 - 2 * cv))
            else:
                residual_rhythmicities.append(0.0)
        else:
            residual_rhythmicities.append(0.0)

    feats['bgsub_residual_power_mean'] = float(np.mean(residual_powers))
    feats['bgsub_residual_power_max'] = float(np.max(residual_powers))
    feats['bgsub_residual_power_std'] = float(np.std(residual_powers))
    feats['bgsub_residual_rhythm_mean'] = float(np.mean(residual_rhythmicities))
    feats['bgsub_residual_rhythm_max'] = float(np.max(residual_rhythmicities))

    return feats


def _gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * x) - (n + 1) * np.sum(x)) / (n * np.sum(x)))


# Bandpass settings to test
BANDPASS_GRID = [
    (0.25, 3.0),
    (0.25, 4.0),
    (0.25, 5.0),
    (0.25, 6.0),
    (0.25, 8.0),
    (0.3, 3.0),
    (0.3, 4.0),
    (0.3, 5.0),
    (0.3, 6.0),
    (0.3, 8.0),
    (0.5, 3.0),
    (0.5, 4.0),   # current default
    (0.5, 5.0),
    (0.5, 6.0),
    (0.5, 8.0),
    (0.5, 10.0),
    (0.1, 3.0),
    (0.1, 4.0),
    (0.1, 6.0),
    (1.0, 4.0),
    (1.0, 6.0),
    (1.0, 8.0),
]


def main():
    print("=" * 70)
    print("  Bandpass Optimization for LRDA vs GRDA")
    print("=" * 70)

    # Load data
    with open(V2_CACHE / 'lateral_v2_data.pkl', 'rb') as f:
        data = pickle.load(f)
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
    pids = df_bal['patient_id'].values
    print(f"Balanced: {(labels==1).sum()} LRDA + {(labels==0).sum()} GRDA = {len(labels)}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = []

    print(f"\nTesting {len(BANDPASS_GRID)} bandpass settings...")
    print(f"{'Lo':>6} {'Hi':>6} {'AUC':>8} {'Std':>7} {'nFeat':>6} {'Time':>6}")
    print("-" * 50)

    for lo, hi in BANDPASS_GRID:
        t0 = time.time()

        # Extract features
        feat_dicts = []
        for pid in pids:
            seg = segs[pid]
            feat_dicts.append(extract_spatial_all(seg, lo, hi))

        all_keys = sorted(set(k for d in feat_dicts for k in d))
        X = np.array([[d.get(k, 0.0) for k in all_keys] for d in feat_dicts], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Evaluate
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42)
        try:
            y_prob = cross_val_predict(clf, X, labels, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
            auc = float(roc_auc_score(labels, y_prob))
            fold_aucs = []
            for tr, te in cv.split(X, labels):
                fold_aucs.append(float(roc_auc_score(labels[te], y_prob[te])))
            auc_std = float(np.std(fold_aucs))
        except:
            auc, auc_std = 0.5, 0.0

        elapsed = time.time() - t0
        marker = ' ***' if auc > 0.80 else ' **' if auc > 0.79 else ' *' if auc > 0.78 else ''
        print(f"{lo:>6.2f} {hi:>6.1f} {auc:>8.4f} {auc_std:>7.4f} {X.shape[1]:>6} {elapsed:>5.0f}s{marker}")

        results.append({
            'lo': lo, 'hi': hi,
            'auc_mean': round(auc, 4),
            'auc_std': round(auc_std, 4),
            'n_features': X.shape[1],
        })

    # Sort and show best
    results.sort(key=lambda r: -r['auc_mean'])
    print(f"\n{'=' * 50}")
    print("TOP 5 BANDPASS SETTINGS:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. [{r['lo']:.2f} - {r['hi']:.1f}] Hz: AUC = {r['auc_mean']:.4f} ± {r['auc_std']:.4f}")
    print(f"\nCurrent default [0.50 - 4.0] Hz: "
          f"AUC = {[r for r in results if r['lo']==0.5 and r['hi']==4.0][0]['auc_mean']:.4f}")

    # Save
    out = RESULTS_DIR / 'bandpass_optimization.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == '__main__':
    main()
