#!/usr/bin/env python3
"""V13 -- rhythmicity-by-construction LRDA laterality rule.

Motivation: V12 lateralizes via amplitude (which hemisphere has more
narrowband envelope amplitude at the estimated rhythm frequency).
Amplitude-based laterality is fallible by construction:

  - Single-channel artifact (electrode pop, sweat drift, motion) inflates
    one hemisphere's variance.
  - Asymmetric volume conduction can project a midline or contralateral
    generator preferentially to one hemisphere's bipolar montage.
  - Slow drift / breathing artifact leaks into the bottom of the LRDA
    band and biases pass-1 variance.
  - Background asymmetry unrelated to the rhythm (focal slowing, posterior
    rhythm asymmetries, residual fast activity) can dominate variance.

V13 lateralizes via four amplitude-normalized rhythmicity measures:

  1. Q-factor per hemisphere: f_peak / FWHM at est_freq, computed on the
     mean PSD of the hemisphere's bipolar channels.
     (Both numerator and denominator are frequencies -- amplitude-free.)

  2. Within-hemisphere phase-locking value (PLV) at est_freq: for each
     within-hemisphere channel pair, |mean(exp(i*(phi_c1 - phi_c2)))|;
     mean across pairs.
     (Phase-only -- amplitude-free by construction.)

  3. Peak-amplitude consistency per hemisphere: per channel, detect peaks
     in the narrowband envelope; report CV of peak heights (std/mean,
     scale-free); take per-hemisphere mean. Lower CV = more rhythmic.
     (Mean-normalized -- scale-free.)

  4. Spectral peak prominence per hemisphere: max prominence in [0.5, 4] Hz
     normalized by the hemisphere's max PSD value.
     (Within-channel-normalized -- amplitude-free as a ratio.)

Each metric yields an L vs R log-ratio with sign convention "positive =
left more rhythmic." V13's call is then a vote / weighted sum across
the four signed log-ratios.

Output:
    data/labels/independent_expert_v1/lrda_laterality_v13_features.csv
    data/labels/independent_expert_v1/lrda_laterality_v13_eval.txt

    conda run -n morgoth python code/evaluation/lrda_laterality_v13_eval.py
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from itertools import combinations
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
OUT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v13_features.csv'
OUT_REPORT = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v13_eval.txt'


# -------------------- Per-channel / per-hemisphere primitives --------------------

def _safe_log_ratio(a, b, eps=1e-9):
    return float(np.log(max(a, eps) / max(b, eps)))


def _per_channel_qfactor(seg_pre: np.ndarray, chs: np.ndarray, est_freq: float) -> float:
    """Mean per-channel Q-factor (f_peak / FWHM near est_freq) across the
    given channels. Welch PSD with nperseg = full segment for resolution."""
    n = seg_pre.shape[1]
    qs = []
    for ch in chs:
        f, pxx = welch(seg_pre[ch], fs=FS, nperseg=min(n, 2048))
        # Restrict to [0.5, 4] Hz
        mask = (f >= 0.5) & (f <= 4.0)
        f_m = f[mask]; p_m = pxx[mask]
        if not len(f_m):
            continue
        # Find local peak nearest est_freq
        peaks, _ = find_peaks(p_m)
        if not len(peaks):
            # Fallback: argmax
            ip = int(np.argmax(p_m))
        else:
            ip = peaks[int(np.argmin(np.abs(f_m[peaks] - est_freq)))]
        f_pk = float(f_m[ip])
        if f_pk <= 0:
            continue
        half = p_m[ip] / 2.0
        # Walk left and right to find FWHM
        l = ip
        while l > 0 and p_m[l] >= half:
            l -= 1
        r = ip
        while r < len(p_m) - 1 and p_m[r] >= half:
            r += 1
        fwhm = float(f_m[r] - f_m[l])
        if fwhm <= 0:
            continue
        qs.append(f_pk / fwhm)
    if not qs:
        return float('nan')
    return float(np.mean(qs))


def _within_hemisphere_plv(seg_nb: np.ndarray, chs: np.ndarray) -> float:
    """Mean within-hemisphere PLV across all channel pairs."""
    if len(chs) < 2:
        return float('nan')
    phases = np.array([np.angle(hilbert(seg_nb[ch])) for ch in chs])  # (n_ch, n_samp)
    plvs = []
    for i, j in combinations(range(len(chs)), 2):
        d = phases[i] - phases[j]
        plv = float(np.abs(np.mean(np.exp(1j * d))))
        plvs.append(plv)
    return float(np.mean(plvs))


def _per_channel_peak_cv(seg_nb: np.ndarray, chs: np.ndarray, est_freq: float) -> float:
    """Mean per-channel CV of narrowband envelope peak heights. Lower = more
    rhythmic. Returns nan if too few channels had >=3 peaks."""
    min_dist = max(int(0.5 * FS / max(est_freq, 0.25)), 1)
    cvs = []
    for ch in chs:
        env = np.abs(hilbert(seg_nb[ch]))
        peaks, _ = find_peaks(env, distance=min_dist, prominence=np.std(env) * 0.3)
        if len(peaks) < 3:
            continue
        heights = env[peaks]
        if heights.mean() <= 0:
            continue
        cv = float(np.std(heights) / heights.mean())
        cvs.append(cv)
    if not cvs:
        return float('nan')
    return float(np.mean(cvs))


def _per_hemisphere_peak_prom(seg_pre: np.ndarray, chs: np.ndarray) -> float:
    """Max spectral peak prominence in [0.5, 4] Hz, normalized by max PSD value,
    averaged across the hemisphere's channels."""
    n = seg_pre.shape[1]
    proms = []
    for ch in chs:
        f, pxx = welch(seg_pre[ch], fs=FS, nperseg=min(n, 1024))
        mask = (f >= 0.5) & (f <= 4.0)
        p_m = pxx[mask]
        if not len(p_m) or p_m.max() <= 0:
            continue
        peaks, props = find_peaks(p_m, prominence=p_m.max() * 0.05)
        if not len(peaks):
            proms.append(0.0)
            continue
        proms.append(float(np.max(props['prominences']) / p_m.max()))
    if not proms:
        return 0.0
    return float(np.mean(proms))


# -------------------- V13 featurizer --------------------

def featurize_v13(seg_bi: np.ndarray, est_freq: float) -> dict:
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)

    # Narrowband filter at est_freq +- 0.5 Hz for PLV and peak-CV
    lo = max(est_freq - 0.5, 0.1)
    hi = min(est_freq + 0.5, FS / 2 - 0.1)
    sos_nb = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos_pre, seg_bi, axis=1)
    seg_nb = sosfiltfilt(sos_nb, seg_nb, axis=1)

    L_q = _per_channel_qfactor(seg_pre, LEFT_CHS, est_freq)
    R_q = _per_channel_qfactor(seg_pre, RIGHT_CHS, est_freq)
    if not np.isfinite(L_q) or not np.isfinite(R_q):
        q_log_ratio = 0.0
    else:
        q_log_ratio = _safe_log_ratio(L_q, R_q)

    L_plv = _within_hemisphere_plv(seg_nb, LEFT_CHS)
    R_plv = _within_hemisphere_plv(seg_nb, RIGHT_CHS)
    if not np.isfinite(L_plv) or not np.isfinite(R_plv):
        plv_log_ratio = 0.0
    else:
        plv_log_ratio = _safe_log_ratio(L_plv, R_plv)

    L_peak_cv = _per_channel_peak_cv(seg_nb, LEFT_CHS, est_freq)
    R_peak_cv = _per_channel_peak_cv(seg_nb, RIGHT_CHS, est_freq)
    if not np.isfinite(L_peak_cv) or not np.isfinite(R_peak_cv):
        peak_cv_inv_log_ratio = 0.0
    else:
        # invert sign: positive = left has lower CV = more rhythmic
        peak_cv_inv_log_ratio = _safe_log_ratio(R_peak_cv, L_peak_cv)

    L_prom = _per_hemisphere_peak_prom(seg_pre, LEFT_CHS)
    R_prom = _per_hemisphere_peak_prom(seg_pre, RIGHT_CHS)
    prom_log_ratio = _safe_log_ratio(L_prom + 0.01, R_prom + 0.01)

    return {
        'q_log_ratio': q_log_ratio,
        'plv_log_ratio': plv_log_ratio,
        'peak_cv_inv_log_ratio': peak_cv_inv_log_ratio,
        'prom_log_ratio': prom_log_ratio,
        'L_q': L_q if np.isfinite(L_q) else 0.0,
        'R_q': R_q if np.isfinite(R_q) else 0.0,
        'L_plv': L_plv if np.isfinite(L_plv) else 0.0,
        'R_plv': R_plv if np.isfinite(R_plv) else 0.0,
        'L_peak_cv': L_peak_cv if np.isfinite(L_peak_cv) else 0.0,
        'R_peak_cv': R_peak_cv if np.isfinite(R_peak_cv) else 0.0,
        'L_prom': L_prom,
        'R_prom': R_prom,
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
    c = Counter(votes); top, n = c.most_common(1)[0]
    return top if n >= 2 else None


def cohen_kappa_pair(preds, rater_dict, mfs):
    a, b = [], []
    for i, mf in enumerate(mfs):
        if mf in rater_dict:
            a.append(preds[i]); b.append(rater_dict[mf])
    if len(a) < 5:
        return float('nan'), 0
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return 1.0, len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[c] / len(a)) * (cb[c] / len(b)) for c in cats)
    if pe >= 1:
        return float('nan'), len(a)
    return (po - pe) / (1 - pe), len(a)


# -------------------- Main --------------------

def main():
    with open(V1_FEAT_CSV) as f:
        v1_rows = {r['mat_file']: r for r in csv.DictReader(f)}
    print(f'V1 cached features: {len(v1_rows)} segments')

    status, lat = load_status_and_lat()

    # Compute V13 features
    v13_rows = {}
    if OUT_CSV.exists():
        cached = list(csv.DictReader(open(OUT_CSV)))
        for r in cached:
            v13_rows[r['mat_file']] = r
        print(f'Loaded {len(v13_rows)} cached V13 features.')

    todo = [mf for mf in v1_rows if mf not in v13_rows]
    if todo:
        print(f'Computing V13 features for {len(todo)} segments...')
        for i, mf in enumerate(todo):
            seg = load_segment(mf)
            if seg is None:
                continue
            est_freq = float(v1_rows[mf]['est_freq'])
            feats = featurize_v13(seg, est_freq)
            feats['mat_file'] = mf
            feats['patient_id'] = v1_rows[mf].get('patient_id', '')
            feats['est_freq'] = est_freq
            v13_rows[mf] = feats
            if (i + 1) % 50 == 0:
                print(f'  {i+1}/{len(todo)}')
                _save(v13_rows)
        _save(v13_rows)

    # Build evaluation set: 155 consensus
    eval_rows = []
    for mf, v1 in v1_rows.items():
        cons = consensus_lat(mf, status, lat)
        if cons is None or mf not in v13_rows:
            continue
        v13 = v13_rows[mf]
        merged = {**v1, **v13}
        merged['mat_file'] = mf
        merged['consensus'] = cons
        eval_rows.append(merged)
    n = len(eval_rows)
    print(f'\nConsensus-laterality eval set: {n} segments')

    cons_str = [r['consensus'] for r in eval_rows]
    mfs = [r['mat_file'] for r in eval_rows]

    # ---------------- Rules ----------------
    def call_score(score):
        return 'left' if float(score) > 0 else 'right'

    def vote_call(*scores):
        s = sum(1 if float(x) > 0 else -1 for x in scores)
        # Tie -> use the score with the largest magnitude
        if s == 0:
            best = max(scores, key=lambda x: abs(float(x)))
            return call_score(best)
        return 'left' if s > 0 else 'right'

    def weighted_sum_call(*scores):
        # Each score is a log-ratio; sum of all
        return call_score(sum(float(x) for x in scores))

    rules = {}
    rules['V12 baseline (pass2_env, amplitude)'] = lambda r: call_score(r['pass2_env_log_ratio'])
    rules['V13a: Q-factor only'] = lambda r: call_score(r['q_log_ratio'])
    rules['V13b: PLV only'] = lambda r: call_score(r['plv_log_ratio'])
    rules['V13c: peak-CV-inv only'] = lambda r: call_score(r['peak_cv_inv_log_ratio'])
    rules['V13d: spectral peak prom only'] = lambda r: call_score(r['prom_log_ratio'])
    rules['V13_vote (4-way majority)'] = lambda r: vote_call(
        r['q_log_ratio'], r['plv_log_ratio'], r['peak_cv_inv_log_ratio'], r['prom_log_ratio'])
    rules['V13_sum (sum of normalized scores)'] = lambda r: weighted_sum_call(
        r['q_log_ratio'], r['plv_log_ratio'], r['peak_cv_inv_log_ratio'], r['prom_log_ratio'])
    rules['V13_max (most decisive feature)'] = lambda r: call_score(
        max([r['q_log_ratio'], r['plv_log_ratio'], r['peak_cv_inv_log_ratio'], r['prom_log_ratio']],
            key=lambda x: abs(float(x))))

    # ---------------- Evaluation ----------------
    def report_rule(name, calls):
        acc = sum(1 for c, gt in zip(calls, cons_str) if c == gt) / n
        kappas = {}
        for rater in ('MW', 'SZ', 'TZ'):
            k, nn = cohen_kappa_pair(calls, lat[rater], mfs)
            kappas[rater] = (k, nn)
        kmean = float(np.mean([kappas[r][0] for r in ('MW', 'SZ', 'TZ') if not np.isnan(kappas[r][0])]))
        return acc, kappas, kmean

    rule_calls = {name: [fn(r) for r in eval_rows] for name, fn in rules.items()}

    print('\n' + '=' * 80)
    print('Held-out evaluation on 155-segment 3-rater consensus set')
    print('=' * 80)
    print(f'{"rule":<48s}  {"acc":>6s}  {"k_MW":>6s}  {"k_SZ":>6s}  {"k_TZ":>6s}  {"k_mean":>7s}')
    print('-' * 90)
    rows_out = []
    for name in rules:
        acc, kappas, kmean = report_rule(name, rule_calls[name])
        rows_out.append((name, acc, kappas, kmean))
        print(f'{name:<48s}  {acc:>6.3f}  {kappas["MW"][0]:>6.3f}  '
              f'{kappas["SZ"][0]:>6.3f}  {kappas["TZ"][0]:>6.3f}  {kmean:>7.3f}')

    # ---------------- V12 errors: per-rule behavior ----------------
    v12_calls = rule_calls['V12 baseline (pass2_env, amplitude)']
    v12_errors = [(i, mfs[i], cons_str[i], v12_calls[i]) for i in range(n) if v12_calls[i] != cons_str[i]]
    print(f'\nV12 baseline errors ({len(v12_errors)} cases). Per-rule behavior:')
    headers = ['#', 'freq', 'cons', 'V12'] + [f'V13{x}' for x in ['a', 'b', 'c', 'd', 'vote', 'sum', 'max']]
    print('  ' + '  '.join(f'{h:>10s}' for h in headers))
    short_names = {
        'V12 baseline (pass2_env, amplitude)': 'V12',
        'V13a: Q-factor only': 'V13a',
        'V13b: PLV only': 'V13b',
        'V13c: peak-CV-inv only': 'V13c',
        'V13d: spectral peak prom only': 'V13d',
        'V13_vote (4-way majority)': 'V13vote',
        'V13_sum (sum of normalized scores)': 'V13sum',
        'V13_max (most decisive feature)': 'V13max',
    }
    keys_in_order = ['V12 baseline (pass2_env, amplitude)',
                     'V13a: Q-factor only', 'V13b: PLV only',
                     'V13c: peak-CV-inv only', 'V13d: spectral peak prom only',
                     'V13_vote (4-way majority)', 'V13_sum (sum of normalized scores)',
                     'V13_max (most decisive feature)']
    for i, mf, cons, _ in v12_errors:
        f = float(eval_rows[i]['est_freq'])
        cells = [str(i+1), f'{f:.2f}', cons]
        for k in keys_in_order:
            call = rule_calls[k][i]
            mark = '*' if call == cons else ' '
            cells.append(call + mark)
        print('  ' + '  '.join(f'{c:>10s}' for c in cells))

    # ---------------- New errors introduced by best V13 vs V12 ----------------
    best_name, best_acc, best_kappas, best_kmean = max(rows_out[1:], key=lambda r: r[3])
    print(f'\nBest V13 variant by mean kappa: {best_name}  (acc {best_acc:.3f}, mean k {best_kmean:.3f})')
    best_calls = rule_calls[best_name]
    new_errors = [(i, mfs[i], cons_str[i], v12_calls[i], best_calls[i])
                  for i in range(n) if v12_calls[i] == cons_str[i] and best_calls[i] != cons_str[i]]
    fixed = [(i, mfs[i], cons_str[i], v12_calls[i], best_calls[i])
             for i in range(n) if v12_calls[i] != cons_str[i] and best_calls[i] == cons_str[i]]
    print(f'  V12-error cases fixed by {short_names[best_name]}: {len(fixed)}')
    for i, mf, cons, v12c, bc in fixed:
        print(f'    #{i+1:>3d}  {float(eval_rows[i]["est_freq"]):.2f} Hz  cons={cons}  V12={v12c}  best={bc}')
    print(f'  Cases V12-correct but {short_names[best_name]} now wrong: {len(new_errors)}')
    for i, mf, cons, v12c, bc in new_errors[:15]:
        print(f'    #{i+1:>3d}  {float(eval_rows[i]["est_freq"]):.2f} Hz  cons={cons}  V12={v12c}  best={bc}')
    if len(new_errors) > 15:
        print(f'    ... ({len(new_errors) - 15} more)')

    # ---------------- Bootstrap V13 vs V12 mean kappa ----------------
    rng = np.random.default_rng(42)
    n_boot = 2000
    deltas = np.empty(n_boot)
    rater_lats = {r: lat[r] for r in ('MW', 'SZ', 'TZ')}
    mfs_arr = np.array(mfs)
    v12_arr = np.array(v12_calls)
    best_arr = np.array(best_calls)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub_mfs = mfs_arr[idx]
        sub_v12 = v12_arr[idx]
        sub_best = best_arr[idx]
        kvs, kbs = [], []
        for rater in ('MW', 'SZ', 'TZ'):
            rd = rater_lats[rater]
            mask = np.array([m in rd for m in sub_mfs])
            if mask.sum() < 5:
                continue
            kv,_ = cohen_kappa_pair(sub_v12[mask].tolist(), rd, sub_mfs[mask].tolist())
            kb,_ = cohen_kappa_pair(sub_best[mask].tolist(), rd, sub_mfs[mask].tolist())
            if not (np.isnan(kv) or np.isnan(kb)):
                kvs.append(kv); kbs.append(kb)
        if kvs and kbs:
            deltas[b] = float(np.mean(kbs)) - float(np.mean(kvs))
        else:
            deltas[b] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    p_two = float(min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))))
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    delta_pt = best_kmean - rows_out[0][3]
    print(f'\nBootstrap: {short_names[best_name]} mean kappa - V12 mean kappa')
    print(f'  point estimate: {delta_pt:+.3f}')
    print(f'  95% CI:         [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    print(f'  two-sided p:    {p_two:.3f}')

    # ---------------- Save report ----------------
    report = []
    report.append('LRDA laterality V13 (rhythmicity-by-construction) -- evaluation')
    report.append('=' * 80)
    report.append(f'Eval set: 155-segment 3-rater majority-accept consensus.')
    report.append('')
    report.append(f'{"rule":<48s}  {"acc":>6s}  {"k_MW":>6s}  {"k_SZ":>6s}  {"k_TZ":>6s}  {"k_mean":>7s}')
    for name, acc, kappas, kmean in rows_out:
        report.append(f'{name:<48s}  {acc:>6.3f}  {kappas["MW"][0]:>6.3f}  '
                      f'{kappas["SZ"][0]:>6.3f}  {kappas["TZ"][0]:>6.3f}  {kmean:>7.3f}')
    report.append('')
    report.append(f'Best V13 variant: {best_name}')
    report.append(f'Bootstrap delta (best V13 - V12) mean kappa: {delta_pt:+.3f}, '
                  f'95% CI [{ci[0]:+.3f},{ci[1]:+.3f}], p={p_two:.3f}')
    OUT_REPORT.write_text('\n'.join(report) + '\n')
    print(f'\nWrote {OUT_REPORT}')


def _save(v13_rows):
    fields = ['mat_file', 'patient_id', 'est_freq',
              'q_log_ratio', 'plv_log_ratio', 'peak_cv_inv_log_ratio', 'prom_log_ratio',
              'L_q', 'R_q', 'L_plv', 'R_plv', 'L_peak_cv', 'R_peak_cv', 'L_prom', 'R_prom']
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for mf, r in v13_rows.items():
            w.writerow({k: r.get(k, '') for k in fields})


if __name__ == '__main__':
    main()
