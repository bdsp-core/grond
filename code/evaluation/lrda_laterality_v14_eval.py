#!/usr/bin/env python3
"""V14 -- V12 (amplitude) default with V13 (rhythmicity) unanimous override.

Principle: amplitude-based laterality (V12) is correct on ~96% of segments
but fails predictably on a few. The four amplitude-normalized rhythmicity
features (V13: Q-factor, PLV, peak-CV, peak-prom) catch most of those
failures but are individually noisy at the per-segment level. V14
combines them: trust V12 by default, override only when ALL four V13
features unanimously disagree -- the strongest possible "amplitude is
lying here" evidence.

Variants tested:
    V14_unanimous:  flip V12 only when 4 of 4 V13 features disagree
    V14_strict-3:   flip V12 only when 3 of 4 V13 features disagree
    V14_strict-3-conf: 3 of 4 disagree AND each disagreer has |log_ratio| > 0.3
    V14_3-with-mag: 3 of 4 disagree AND average |log_ratio| of disagreers > 0.5

Output:
    data/labels/independent_expert_v1/lrda_laterality_v14_eval.txt

    conda run -n morgoth python code/evaluation/lrda_laterality_v14_eval.py
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
V1_FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'
V13_FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v13_features.csv'
OUT_REPORT = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v14_eval.txt'

V13_KEYS = ['q_log_ratio', 'plv_log_ratio', 'peak_cv_inv_log_ratio', 'prom_log_ratio']


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


def main():
    v1 = {r['mat_file']: r for r in csv.DictReader(open(V1_FEAT_CSV))}
    v13 = {r['mat_file']: r for r in csv.DictReader(open(V13_FEAT_CSV))}
    status, lat = load_status_and_lat()

    eval_rows = []
    for mf, v1r in v1.items():
        cons = consensus_lat(mf, status, lat)
        if cons is None or mf not in v13:
            continue
        merged = {**v1r, **v13[mf]}
        merged['mat_file'] = mf
        merged['consensus'] = cons
        eval_rows.append(merged)
    n = len(eval_rows)
    print(f'Eval set: {n} segments (3-rater majority-accept consensus)')

    cons_str = [r['consensus'] for r in eval_rows]
    mfs = [r['mat_file'] for r in eval_rows]
    v12_str = ['left' if float(r['pass2_env_log_ratio']) > 0 else 'right' for r in eval_rows]

    # Pre-compute V13 feature signs and magnitudes per row
    def v13_signs_and_mags(r):
        signs = []  # +1 = points left
        mags = []
        for k in V13_KEYS:
            v = float(r[k])
            signs.append(1 if v > 0 else -1)
            mags.append(abs(v))
        return signs, mags

    # ---------------- Hybrid rules ----------------
    def v14_unanimous(r, v12c):
        """Flip V12 only when 4 of 4 V13 features disagree with V12."""
        signs, _ = v13_signs_and_mags(r)
        v12_sign = 1 if v12c == 'left' else -1
        n_disagree = sum(1 for s in signs if s != v12_sign)
        if n_disagree == 4:
            return 'right' if v12_sign == 1 else 'left'
        return v12c

    def v14_strict3(r, v12c):
        """Flip V12 when at least 3 of 4 V13 features disagree."""
        signs, _ = v13_signs_and_mags(r)
        v12_sign = 1 if v12c == 'left' else -1
        n_disagree = sum(1 for s in signs if s != v12_sign)
        if n_disagree >= 3:
            return 'right' if v12_sign == 1 else 'left'
        return v12c

    def v14_strict3_conf(r, v12c, conf=0.3):
        """3-of-4 disagree AND each disagreer has |log_ratio| > conf."""
        signs, mags = v13_signs_and_mags(r)
        v12_sign = 1 if v12c == 'left' else -1
        disagreers = [(s, m) for s, m in zip(signs, mags) if s != v12_sign]
        if len(disagreers) >= 3 and all(m > conf for _, m in disagreers):
            return 'right' if v12_sign == 1 else 'left'
        return v12c

    def v14_3_with_avg_mag(r, v12c, mag_thresh=0.5):
        """3-of-4 disagree AND average |log_ratio| of disagreers > mag_thresh."""
        signs, mags = v13_signs_and_mags(r)
        v12_sign = 1 if v12c == 'left' else -1
        disagreers = [(s, m) for s, m in zip(signs, mags) if s != v12_sign]
        if len(disagreers) >= 3 and float(np.mean([m for _, m in disagreers])) > mag_thresh:
            return 'right' if v12_sign == 1 else 'left'
        return v12c

    def v14_4_thresh(r, v12c, mag_thresh=0.0):
        """4-of-4 disagree AND average |log_ratio| > mag_thresh (loose magnitude gate)."""
        signs, mags = v13_signs_and_mags(r)
        v12_sign = 1 if v12c == 'left' else -1
        if all(s != v12_sign for s in signs) and float(np.mean(mags)) > mag_thresh:
            return 'right' if v12_sign == 1 else 'left'
        return v12c

    rules = {
        'V12 baseline (amplitude)': lambda i: v12_str[i],
        'V14_unanimous (flip when 4-of-4 V13 disagree)': lambda i: v14_unanimous(eval_rows[i], v12_str[i]),
        'V14_strict-3 (flip when 3-of-4 V13 disagree)': lambda i: v14_strict3(eval_rows[i], v12_str[i]),
        'V14_strict-3 + each |log_ratio|>0.3': lambda i: v14_strict3_conf(eval_rows[i], v12_str[i], 0.3),
        'V14_strict-3 + avg |log_ratio|>0.5': lambda i: v14_3_with_avg_mag(eval_rows[i], v12_str[i], 0.5),
        'V14_strict-3 + avg |log_ratio|>0.3': lambda i: v14_3_with_avg_mag(eval_rows[i], v12_str[i], 0.3),
        'V14_4 + avg |log_ratio|>0.5':       lambda i: v14_4_thresh(eval_rows[i], v12_str[i], 0.5),
    }

    rule_calls = {name: [fn(i) for i in range(n)] for name, fn in rules.items()}

    def report_rule(calls):
        acc = sum(1 for c, gt in zip(calls, cons_str) if c == gt) / n
        kappas = {}
        for rater in ('MW', 'SZ', 'TZ'):
            k, nn = cohen_kappa_pair(calls, lat[rater], mfs)
            kappas[rater] = (k, nn)
        kmean = float(np.mean([kappas[r][0] for r in ('MW', 'SZ', 'TZ') if not np.isnan(kappas[r][0])]))
        return acc, kappas, kmean

    print('\n' + '=' * 90)
    print('V14: V12 (amplitude) + V13 (rhythmicity) hybrid override variants')
    print('=' * 90)
    print(f'{"rule":<55s}  {"acc":>6s}  {"k_MW":>6s}  {"k_SZ":>6s}  {"k_TZ":>6s}  {"k_mean":>7s}')
    print('-' * 100)
    rows_out = []
    for name in rules:
        acc, kappas, kmean = report_rule(rule_calls[name])
        rows_out.append((name, acc, kappas, kmean))
        print(f'{name:<55s}  {acc:>6.3f}  {kappas["MW"][0]:>6.3f}  '
              f'{kappas["SZ"][0]:>6.3f}  {kappas["TZ"][0]:>6.3f}  {kmean:>7.3f}')

    # ---------------- Per-case: V12 errors + new errors per V14 variant ----------------
    v12_errors = [(i, mfs[i], cons_str[i], v12_str[i]) for i in range(n) if v12_str[i] != cons_str[i]]
    print(f'\nV12 baseline errors ({len(v12_errors)} cases). Per-rule behavior:')
    headers = ['#', 'freq', 'cons', 'V12'] + ['un', 'st3', 's3c', 's3a5', 's3a3', '4a5']
    print('  ' + '  '.join(f'{h:>10s}' for h in headers))
    short = list(rules.keys())
    for i, mf, cons, _ in v12_errors:
        f = float(eval_rows[i]['est_freq'])
        cells = [str(i+1), f'{f:.2f}', cons]
        for name in short:
            call = rule_calls[name][i]
            mark = '*' if call == cons else ' '
            cells.append(call + mark)
        print('  ' + '  '.join(f'{c:>10s}' for c in cells))

    print('\nNew errors per V14 variant (cases V12 was right but V14 now wrong):')
    for name in short[1:]:
        new_errors = [(i, float(eval_rows[i]['est_freq']), cons_str[i], v12_str[i], rule_calls[name][i])
                      for i in range(n) if v12_str[i] == cons_str[i] and rule_calls[name][i] != cons_str[i]]
        fixed = [(i, float(eval_rows[i]['est_freq']), cons_str[i], v12_str[i], rule_calls[name][i])
                 for i in range(n) if v12_str[i] != cons_str[i] and rule_calls[name][i] == cons_str[i]]
        n_flipped = sum(1 for i in range(n) if v12_str[i] != rule_calls[name][i])
        print(f'\n  {name}')
        print(f'    Flips: {n_flipped} total. Fixed V12 errors: {len(fixed)}. New errors: {len(new_errors)}.')
        for i, f, c, vc, ec in fixed:
            print(f'      FIXED  #{i+1:>3d}  {f:.2f} Hz  cons={c}  V12={vc}  V14={ec}')
        for i, f, c, vc, ec in new_errors[:8]:
            print(f'      NEW    #{i+1:>3d}  {f:.2f} Hz  cons={c}  V12={vc}  V14={ec}')
        if len(new_errors) > 8:
            print(f'      ... ({len(new_errors) - 8} more)')

    # ---------------- Bootstrap significance: each V14 vs V12 ----------------
    print('\n' + '=' * 90)
    print('Bootstrap significance: V14 mean kappa vs V12 mean kappa (paired segment-bootstrap, n=2000)')
    print('=' * 90)
    rng_seed_base = 42
    rater_lats = {r: lat[r] for r in ('MW', 'SZ', 'TZ')}
    mfs_arr = np.array(mfs)
    v12_arr = np.array(v12_str)

    for name in short[1:]:
        calls = rule_calls[name]
        calls_arr = np.array(calls)
        rng = np.random.default_rng(rng_seed_base)
        n_boot = 2000
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            sub_mfs = mfs_arr[idx]
            sub_v12 = v12_arr[idx]
            sub_v14 = calls_arr[idx]
            kvs, kbs = [], []
            for rater in ('MW', 'SZ', 'TZ'):
                rd = rater_lats[rater]
                mask = np.array([m in rd for m in sub_mfs])
                if mask.sum() < 5:
                    continue
                kv,_ = cohen_kappa_pair(sub_v12[mask].tolist(), rd, sub_mfs[mask].tolist())
                kb,_ = cohen_kappa_pair(sub_v14[mask].tolist(), rd, sub_mfs[mask].tolist())
                if not (np.isnan(kv) or np.isnan(kb)):
                    kvs.append(kv); kbs.append(kb)
            if kvs and kbs:
                deltas[b] = float(np.mean(kbs)) - float(np.mean(kvs))
            else:
                deltas[b] = np.nan
        deltas = deltas[~np.isnan(deltas)]
        p_two = float(min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))))
        ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
        # Find this rule's point estimate
        delta_pt = next(rec[3] for rec in rows_out if rec[0] == name) - rows_out[0][3]
        print(f'  {name:<55s}')
        print(f'    delta mean kappa: {delta_pt:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  p={p_two:.3f}')

    # Save report
    lines = []
    lines.append('LRDA laterality V14 (V12 + V13-override hybrid) -- evaluation')
    lines.append('=' * 80)
    lines.append(f'Eval set: {n} segments (3-rater majority-accept consensus).')
    lines.append('')
    lines.append(f'{"rule":<55s}  {"acc":>6s}  {"k_MW":>6s}  {"k_SZ":>6s}  {"k_TZ":>6s}  {"k_mean":>7s}')
    for name, acc, kappas, kmean in rows_out:
        lines.append(f'{name:<55s}  {acc:>6.3f}  {kappas["MW"][0]:>6.3f}  '
                     f'{kappas["SZ"][0]:>6.3f}  {kappas["TZ"][0]:>6.3f}  {kmean:>7.3f}')
    OUT_REPORT.write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {OUT_REPORT}')


if __name__ == '__main__':
    main()
