#!/usr/bin/env python3
"""LRDA laterality v2 -- rhythmicity features + peak-locked topographic centroid.

Evaluates two fixes for the LRDA laterality gap, head-to-head against
the V12/V1 baseline (pass-2 narrowband envelope rule).

Fix 1 (rhythmicity, addresses the 3-of-6 amplitude-vs-rhythmicity errors):
  - spec_conc_log_ratio:  log(L_concentration / R_concentration), where
                          concentration = power in [est_freq +- 0.4] /
                          power in [0.5, 4] Hz, computed per hemisphere
                          on the average of bipolar channels in that
                          hemisphere.
  - acf_peak_log_ratio:   log(L_acf_peak / R_acf_peak), where acf_peak
                          is the autocorrelation height at lag = round
                          (FS / est_freq).
  - if_cv_inv_log_ratio:  log(R_if_cv / L_if_cv) computed per hemisphere
                          on the narrowband-filtered signal.  Positive
                          means left has lower IF CV (more rhythmic).

Fix 2 (peak-locked topographic centroid, addresses the unanimous-amplitude
errors):
  - peak_topo_centroid_log_ratio: detect narrowband envelope peaks of the
                          per-segment dominant signal; for each peak
                          take the per-channel absolute voltage; average
                          across peaks; report log(mean |L topo| /
                          mean |R topo|).

For each segment in the 155-segment majority-accept consensus laterality
set, we evaluate seven rules:

    V12 baseline:          pass2_env_log_ratio > 0
    Fix1a:                 spec_conc_log_ratio > 0
    Fix1b:                 acf_peak_log_ratio > 0
    Fix1c:                 if_cv_inv_log_ratio > 0
    Fix1_sum:              sum of Fix-1 features > 0
    Fix2:                  peak_topo_centroid_log_ratio > 0
    Hybrid_v2 (vote):      majority(pass2_env, spec_conc, peak_topo_centroid)
    Trained classifier:    HGB on V1's 16 features + 4 new features,
                           5-fold patient-grouped OOF.

Reports overall accuracy, per-rater Cohen's kappa, and the per-rule call
on the 6 known V12 errors.

Output:
    data/labels/independent_expert_v1/lrda_laterality_v2_features.csv
    data/labels/independent_expert_v1/lrda_laterality_v2_eval.txt

    conda run -n morgoth python code/evaluation/lrda_laterality_v2_eval.py
"""
from __future__ import annotations
import csv
import json
import sys
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
V1_FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'
OUT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v2_features.csv'
OUT_REPORT = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v2_eval.txt'

V12_FREQ_BAND = (0.5, 4.5)


# -------------------- Feature computation --------------------

def _safe_log_ratio(a: float, b: float) -> float:
    eps = 1e-9
    return float(np.log(max(a, eps) / max(b, eps)))


def _hilbert_if_cv_signal(sig: np.ndarray) -> float:
    """IF coefficient of variation for a signal. Lower = more rhythmic."""
    if np.std(sig) < 1e-10:
        return float('nan')
    a = hilbert(sig)
    inst = np.diff(np.unwrap(np.angle(a))) * FS / (2 * np.pi)
    mask = (inst > 0.3) & (inst < 4.5)
    valid = inst[mask]
    if len(valid) < 20:
        return float('nan')
    med = float(np.median(valid))
    if med <= 1e-6:
        return float('nan')
    return float(np.std(valid) / med)


def _hemisphere_signal(seg_pre: np.ndarray, chs: np.ndarray) -> np.ndarray:
    """Mean of the bipolar channels for one hemisphere."""
    return np.mean(seg_pre[chs], axis=0)


def _spec_concentration(sig: np.ndarray, est_freq: float, bw: float = 0.4,
                         lo_band: float = 0.5, hi_band: float = 4.0) -> float:
    """Power in [est_freq +- bw] / power in [lo_band, hi_band], on a Welch PSD."""
    n = min(len(sig), 1024)
    f, pxx = welch(sig, fs=FS, nperseg=n)
    in_peak = (f >= max(lo_band, est_freq - bw)) & (f <= min(hi_band, est_freq + bw))
    in_total = (f >= lo_band) & (f <= hi_band)
    p_peak = float(np.trapezoid(pxx[in_peak], f[in_peak])) if in_peak.any() else 0.0
    p_total = float(np.trapezoid(pxx[in_total], f[in_total])) if in_total.any() else 0.0
    if p_total <= 0:
        return 0.0
    return p_peak / p_total


def _acf_peak_height(sig: np.ndarray, est_freq: float) -> float:
    """Autocorrelation height at lag = round(FS / est_freq), normalized by lag-0."""
    sig = sig - np.mean(sig)
    n = len(sig)
    full = np.correlate(sig, sig, mode='full')
    mid = full[n - 1:]
    lag = int(round(FS / max(est_freq, 0.25)))
    if lag <= 0 or lag >= n:
        return 0.0
    if mid[0] <= 0:
        return 0.0
    return float(mid[lag] / mid[0])


def _peak_locked_topography(seg_pre: np.ndarray, est_freq: float, bw: float = 0.4) -> np.ndarray | None:
    """Detect narrowband envelope peaks on the *neutral* mean of both hemispheric
    envelopes (so peak detection is not biased toward whichever side has more
    raw amplitude), then average channel-wise |voltage| at peak times.

    Returns shape (n_channels,) or None if too few peaks were found.
    """
    sos2 = butter(3, [max(est_freq - bw, 0.1) / (FS / 2), min(est_freq + bw, FS / 2 - 0.1) / (FS / 2)],
                  btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos2, seg_pre, axis=1)
    # Per-hemisphere envelope, then average → neutral peak-detection signal.
    L_env = np.abs(hilbert(np.mean(seg_nb[LEFT_CHS], axis=0)))
    R_env = np.abs(hilbert(np.mean(seg_nb[RIGHT_CHS], axis=0)))
    # Normalize each by its own max so amplitudes are comparable, then average.
    L_norm = L_env / max(L_env.max(), 1e-9)
    R_norm = R_env / max(R_env.max(), 1e-9)
    env = (L_norm + R_norm) / 2.0
    # Min distance between peaks: 0.6 / est_freq seconds.
    min_dist = max(int(0.6 * FS / max(est_freq, 0.25)), 1)
    # Lower prominence threshold: the normalized signal has range ~[0,1].
    peaks, _ = find_peaks(env, distance=min_dist, prominence=0.10)
    if len(peaks) < 3:
        # Fallback: relax prominence requirement
        peaks, _ = find_peaks(env, distance=min_dist)
    if len(peaks) < 3:
        return None
    n_ch = seg_pre.shape[0]
    topo = np.zeros(n_ch, dtype=np.float64)
    for p in peaks:
        topo += np.abs(seg_nb[:, p])
    topo /= len(peaks)
    return topo


def featurize_v2(seg_bi: np.ndarray, v1_pass1_log_ratio: float, est_freq: float) -> dict:
    """Compute the new v2 features given a segment and the V1 estimated freq."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)

    # Hemisphere mean signals (bandpassed broad)
    left_sig = _hemisphere_signal(seg_pre, LEFT_CHS)
    right_sig = _hemisphere_signal(seg_pre, RIGHT_CHS)

    # Spectral concentration (rhythmicity) per hemisphere
    L_conc = _spec_concentration(left_sig, est_freq)
    R_conc = _spec_concentration(right_sig, est_freq)
    spec_conc_log_ratio = _safe_log_ratio(L_conc + 1e-3, R_conc + 1e-3)

    # ACF peak per hemisphere
    L_acf = _acf_peak_height(left_sig, est_freq)
    R_acf = _acf_peak_height(right_sig, est_freq)
    acf_peak_log_ratio = _safe_log_ratio(max(L_acf, 1e-3), max(R_acf, 1e-3))

    # IF CV per hemisphere on the narrowband signal (lower = more rhythmic)
    sos2 = butter(3, [max(est_freq - 0.4, 0.1) / (FS / 2), min(est_freq + 0.4, FS / 2 - 0.1) / (FS / 2)],
                  btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos2, seg_pre, axis=1)
    L_nb = np.mean(seg_nb[LEFT_CHS], axis=0)
    R_nb = np.mean(seg_nb[RIGHT_CHS], axis=0)
    L_cv = _hilbert_if_cv_signal(L_nb)
    R_cv = _hilbert_if_cv_signal(R_nb)
    if not np.isfinite(L_cv) or not np.isfinite(R_cv):
        if_cv_inv_log_ratio = 0.0
    else:
        if_cv_inv_log_ratio = _safe_log_ratio(max(R_cv, 1e-3), max(L_cv, 1e-3))  # + means left more rhythmic

    # Peak-locked topographic centroid
    topo = _peak_locked_topography(seg_pre, est_freq)
    if topo is None:
        peak_topo_log_ratio = 0.0
        n_peaks_used = 0
    else:
        L_mass = float(np.sum(topo[LEFT_CHS]))
        R_mass = float(np.sum(topo[RIGHT_CHS]))
        peak_topo_log_ratio = _safe_log_ratio(L_mass, R_mass)
        n_peaks_used = -1  # marker; we don't return the actual count for now

    return {
        'spec_conc_log_ratio': spec_conc_log_ratio,
        'acf_peak_log_ratio': acf_peak_log_ratio,
        'if_cv_inv_log_ratio': if_cv_inv_log_ratio,
        'peak_topo_log_ratio': peak_topo_log_ratio,
        'L_concentration': L_conc,
        'R_concentration': R_conc,
        'L_acf_peak': L_acf,
        'R_acf_peak': R_acf,
    }


# -------------------- Consensus laterality --------------------

def load_status_and_lat():
    status = {r: {} for r in ('MW', 'SZ', 'TZ')}
    files = [
        ('TZ/lrda_freq_labeling_results_TZ.json', 'TZ'),
        ('SZ/rda_freq_labeling_results-2.json', 'SZ'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', 'MW'),
    ]
    for rel, rater in files:
        with open(RAW_DIR / rel) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf and sub == 'lrda':
                status[rater][mf] = v.get('action') or 'unknown'

    lat = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'laterality':
                continue
            v = row['value'].strip().lower()
            if v in ('left', 'right'):
                lat[r][row['mat_file']] = v
    return status, lat


def consensus_lat(mf, status, lat):
    votes = []
    for r in ('MW', 'SZ', 'TZ'):
        if status[r].get(mf) == 'accept' and mf in lat[r]:
            votes.append(lat[r][mf])
    if len(votes) < 2:
        return None
    c = Counter(votes)
    top, count = c.most_common(1)[0]
    return top if count >= 2 else None


# -------------------- Main --------------------

def main():
    # Load V1 cached features (so we don't re-run the V1 pipeline)
    with open(V1_FEAT_CSV) as f:
        v1_rows = {r['mat_file']: r for r in csv.DictReader(f)}
    print(f'V1 cached features: {len(v1_rows)} segments')

    status, lat = load_status_and_lat()

    # Compute v2 features for every manifest segment that has v1 features
    v2_rows = {}
    print('Computing v2 features (rhythmicity + peak-locked centroid)...')
    for i, (mf, v1) in enumerate(v1_rows.items()):
        seg = load_segment(mf)
        if seg is None:
            print(f'  WARNING: {mf} unloadable; skipping')
            continue
        est_freq = float(v1['est_freq'])
        feats = featurize_v2(seg, v1_pass1_log_ratio=float(v1['pass1_var_log_ratio']),
                              est_freq=est_freq)
        feats['mat_file'] = mf
        feats['patient_id'] = v1.get('patient_id', '')
        feats['est_freq'] = est_freq
        v2_rows[mf] = feats
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(v1_rows)}')

    # Save the v2 features
    fields = ['mat_file', 'patient_id', 'est_freq',
              'spec_conc_log_ratio', 'acf_peak_log_ratio',
              'if_cv_inv_log_ratio', 'peak_topo_log_ratio',
              'L_concentration', 'R_concentration', 'L_acf_peak', 'R_acf_peak']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for mf, feats in v2_rows.items():
            w.writerow({k: feats.get(k, '') for k in fields})
    print(f'Wrote {OUT_CSV}')

    # Build the evaluation set: 155 consensus-laterality segments
    eval_rows = []
    for mf, v1 in v1_rows.items():
        cons = consensus_lat(mf, status, lat)
        if cons is None or mf not in v2_rows:
            continue
        v2 = v2_rows[mf]
        merged = {**v1, **v2}
        merged['mat_file'] = mf
        merged['consensus'] = cons
        merged['cons_int'] = 0 if cons == 'left' else 1   # 0 left, 1 right
        merged['mw_lat'] = lat['MW'].get(mf, '-')
        merged['sz_lat'] = lat['SZ'].get(mf, '-')
        merged['tz_lat'] = lat['TZ'].get(mf, '-')
        eval_rows.append(merged)
    n = len(eval_rows)
    print(f'\nConsensus-laterality eval set: {n} segments')

    # ---------------- Rules ----------------
    def call_from_score(score):
        return 'left' if float(score) > 0 else 'right'

    rules = {}

    rules['V12 baseline (pass2_env)'] = lambda r: call_from_score(r['pass2_env_log_ratio'])
    rules['Fix1a: spec_conc_log_ratio'] = lambda r: call_from_score(r['spec_conc_log_ratio'])
    rules['Fix1b: acf_peak_log_ratio'] = lambda r: call_from_score(r['acf_peak_log_ratio'])
    rules['Fix1c: if_cv_inv_log_ratio'] = lambda r: call_from_score(r['if_cv_inv_log_ratio'])
    rules['Fix1_sum'] = lambda r: call_from_score(
        float(r['spec_conc_log_ratio']) + float(r['acf_peak_log_ratio']) + float(r['if_cv_inv_log_ratio']))
    rules['Fix2: peak_topo_log_ratio'] = lambda r: call_from_score(r['peak_topo_log_ratio'])
    rules['Hybrid_v2 vote(pass2,spec_conc,peak_topo)'] = lambda r: call_from_score(
        (1 if float(r['pass2_env_log_ratio']) > 0 else -1) +
        (1 if float(r['spec_conc_log_ratio']) > 0 else -1) +
        (1 if float(r['peak_topo_log_ratio']) > 0 else -1))
    rules['Hybrid_5-vote(pass2,spec_conc,peak_topo,acf,max_ch)'] = lambda r: call_from_score(
        (1 if float(r['pass2_env_log_ratio']) > 0 else -1) +
        (1 if float(r['spec_conc_log_ratio']) > 0 else -1) +
        (1 if float(r['peak_topo_log_ratio']) > 0 else -1) +
        (1 if float(r['acf_peak_log_ratio']) > 0 else -1) +
        (1 if float(r['lr_max_ch_log_ratio']) > 0 else -1))

    # ---------------- Trained classifier on combined feature set ----------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    feat_cols_v1 = ['pass1_var_log_ratio', 'pass2_env_log_ratio',
                    'narrowband_var_log_ratio', 'top3_var_log_ratio',
                    'spectral_peak_prom_log_ratio', 'pass1_pass2_agreement',
                    'top3_uniform_agreement', 'left_artifact_score',
                    'right_artifact_score', 'left_ch_dispersion',
                    'right_ch_dispersion', 'est_freq', 'est_freq_if_cv',
                    'left_max_ch_var', 'right_max_ch_var', 'lr_max_ch_log_ratio']
    feat_cols_v2 = ['spec_conc_log_ratio', 'acf_peak_log_ratio',
                    'if_cv_inv_log_ratio', 'peak_topo_log_ratio']
    all_cols = feat_cols_v1 + feat_cols_v2

    X = np.array([[float(r[c]) for c in all_cols] for r in eval_rows])
    y = np.array([r['cons_int'] for r in eval_rows])
    groups = np.array([r['patient_id'] for r in eval_rows])

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(n, dtype=int)
    for k, (tr, te) in enumerate(gkf.split(X, y, groups)):
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42)
        clf.fit(X[tr], y[tr])
        oof_pred[te] = clf.predict(X[te])
    trained_calls = ['left' if p == 0 else 'right' for p in oof_pred]
    rules['Trained_v2 (HGB OOF, 16+4 feats)'] = lambda r, _i=[0]: trained_calls[_i.__setitem__(0, _i[0]+1) or (_i[0]-1)]
    # The closure trick is too clever; replace with an explicit dict to avoid reentrancy issues
    rules.pop('Trained_v2 (HGB OOF, 16+4 feats)', None)

    # ---------------- Evaluation ----------------
    def cohen_kappa_pair(preds_str, rater_dict, mfs):
        """Cohen kappa between predictions (list[str]) and rater (dict[mf]->str), only on segments rater labeled."""
        a, b = [], []
        for i, mf in enumerate(mfs):
            if mf in rater_dict:
                a.append(preds_str[i])
                b.append(rater_dict[mf])
        if len(a) < 5:
            return float('nan'), 0
        cats = sorted(set(a) | set(b))
        if len(cats) < 2:
            return 1.0, len(a)
        po = sum(1 for x, y in zip(a, b) if x == y) / len(a)
        from collections import Counter
        ca = Counter(a); cb = Counter(b)
        pe = sum((ca[c] / len(a)) * (cb[c] / len(b)) for c in cats)
        if pe >= 1:
            return float('nan'), len(a)
        return (po - pe) / (1 - pe), len(a)

    mfs = [r['mat_file'] for r in eval_rows]
    cons_str = [r['consensus'] for r in eval_rows]
    rater_dicts = {r: lat[r] for r in ('MW', 'SZ', 'TZ')}

    # Identify the V12 baseline disagreement cases
    v12_calls = [rules['V12 baseline (pass2_env)'](r) for r in eval_rows]
    v12_errors = [(i, mfs[i], cons_str[i], v12_calls[i]) for i in range(n) if v12_calls[i] != cons_str[i]]
    print(f'\nV12 baseline errors: {len(v12_errors)}')
    for i, mf, cons, call in v12_errors:
        print(f'  #{i+1}  {mf[:30]:30s}  consensus={cons}  algo={call}')

    # Build report
    report_lines = []
    report_lines.append('LRDA laterality v2 evaluation')
    report_lines.append('=' * 80)
    report_lines.append(f'Eval set: {n} segments with majority-accept consensus laterality.')
    report_lines.append('')
    report_lines.append(f'{"rule":<55s}  {"acc":>6s}  {"kappa_MW":>9s}  {"kappa_SZ":>9s}  {"kappa_TZ":>9s}  {"kappa_mean":>10s}')
    report_lines.append('-' * 110)

    rule_results = {}
    rule_calls = {}
    for name, fn in rules.items():
        calls = [fn(r) for r in eval_rows]
        rule_calls[name] = calls
        acc = sum(1 for c, gt in zip(calls, cons_str) if c == gt) / n
        # Per-rater kappa
        kappas = {}
        for rater in ('MW', 'SZ', 'TZ'):
            k, nn = cohen_kappa_pair(calls, rater_dicts[rater], mfs)
            kappas[rater] = (k, nn)
        kmean = float(np.mean([kappas[r][0] for r in ('MW', 'SZ', 'TZ') if not np.isnan(kappas[r][0])]))
        rule_results[name] = {'acc': acc, 'kappas': kappas, 'kappa_mean': kmean}
        report_lines.append(
            f'{name:<55s}  {acc:>6.3f}  {kappas["MW"][0]:>9.3f}  {kappas["SZ"][0]:>9.3f}  '
            f'{kappas["TZ"][0]:>9.3f}  {kmean:>10.3f}')

    # Trained classifier
    rule_calls['Trained_v2 (HGB OOF, 16+4 feats)'] = trained_calls
    acc = sum(1 for c, gt in zip(trained_calls, cons_str) if c == gt) / n
    kappas = {}
    for rater in ('MW', 'SZ', 'TZ'):
        k, nn = cohen_kappa_pair(trained_calls, rater_dicts[rater], mfs)
        kappas[rater] = (k, nn)
    kmean = float(np.mean([kappas[r][0] for r in ('MW', 'SZ', 'TZ') if not np.isnan(kappas[r][0])]))
    rule_results['Trained_v2 (HGB OOF, 16+4 feats)'] = {'acc': acc, 'kappas': kappas, 'kappa_mean': kmean}
    report_lines.append(
        f'{"Trained_v2 (HGB OOF, 16+4 feats)":<55s}  {acc:>6.3f}  '
        f'{kappas["MW"][0]:>9.3f}  {kappas["SZ"][0]:>9.3f}  {kappas["TZ"][0]:>9.3f}  {kmean:>10.3f}')

    report_lines.append('')
    report_lines.append('Per-rule call on the 6 V12 baseline disagreement cases:')
    report_lines.append(f'{"#":>3s}  {"freq":>4s}  {"cons":>5s}  ' +
                        '  '.join(f'{n[:18]:>18s}' for n in rule_calls.keys()))
    for i, mf, cons, _ in v12_errors:
        f = float(eval_rows[i]['est_freq'])
        row = f'{i+1:>3d}  {f:>4.2f}  {cons:>5s}  '
        for name in rule_calls:
            call = rule_calls[name][i]
            mark = '*' if call == cons else ' '
            row += f' {call+mark:>18s}'
        report_lines.append(row)

    report_lines.append('')
    report_lines.append('Star (*) = rule got this case correct (matches consensus).')

    print('\n'.join(report_lines))
    OUT_REPORT.write_text('\n'.join(report_lines) + '\n')
    print(f'\nWrote {OUT_REPORT}')


if __name__ == '__main__':
    main()
