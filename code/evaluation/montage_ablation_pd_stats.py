#!/usr/bin/env python3
"""Significance tests for the PD-Profiler montage ablation.

Point estimates alone cannot settle whether the GPD-frequency advantage of the
monopolar / fusion montages is real, so this runs a paired, patient-stratified
bootstrap of each alternative montage against the production bipolar montage.

Paired: every resample draws the same patients for both montages and recomputes
both metrics on that resample, so the comparison is on identical segments.
Patient-stratified: patients are the resampling unit (a patient may contribute
several segments), matching the significance tests used elsewhere in the paper.

    conda run -n morgoth python code/evaluation/montage_ablation_pd_stats.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS = PROJECT_DIR / 'results' / 'montage_ablation'
IN_PATH = RESULTS / 'montage_ablation_pd.json'
OUT_PATH = RESULTS / 'montage_ablation_pd_stats.json'

N_BOOT = 2000
SEED = 42
BASELINE = 'bipolar'


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float); rx -= rx.mean()
    ry = np.argsort(np.argsort(y)).astype(float); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else np.nan


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = np.argsort(np.argsort(s)) + 1.0
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def mae(x, y):
    return float(np.mean(np.abs(np.asarray(x, float) - np.asarray(y, float))))


# ---- metric definitions over a list of per-segment records --------------------

def m_gpd_rho(recs):
    r = [x for x in recs if x['subtype'] == 'gpd']
    return spearman([x['expert_freq'] for x in r], [x['cnn_freq'] for x in r])


def m_lpd_rho(recs):
    r = [x for x in recs if x['subtype'] == 'lpd']
    return spearman([x['expert_freq'] for x in r], [x['cnn_freq'] for x in r])


def m_gpd_mae(recs):
    r = [x for x in recs if x['subtype'] == 'gpd']
    return mae([x['expert_freq'] for x in r], [x['cnn_freq'] for x in r])


def m_lat_auc(recs):
    r = [x for x in recs if x['subtype'] == 'lpd' and x['lat_true'] in ('left', 'right')]
    return auc(np.array([1 if x['lat_true'] == 'right' else 0 for x in r]),
               np.array([x['lat_score'] for x in r]))


METRICS = {
    'GPD frequency rho': m_gpd_rho,
    'LPD frequency rho': m_lpd_rho,
    'GPD frequency MAE': m_gpd_mae,
    'LPD laterality AUC': m_lat_auc,
}
# Metrics where lower is better, so that a positive delta always favours the
# alternative montage.
LOWER_IS_BETTER = {'GPD frequency MAE'}


def paired_bootstrap(by_patient, mats_base, mats_alt, fn, lower_better, rng):
    """Resample patients; recompute the metric for both montages on each draw."""
    pids = list(by_patient)
    sign = -1.0 if lower_better else 1.0
    obs = sign * (fn([mats_alt[m] for p in pids for m in by_patient[p]])
                  - fn([mats_base[m] for p in pids for m in by_patient[p]]))
    deltas = []
    n = len(pids)
    for _ in range(N_BOOT):
        draw = rng.choice(n, size=n, replace=True)
        mats = [m for i in draw for m in by_patient[pids[i]]]
        a = fn([mats_alt[m] for m in mats])
        b = fn([mats_base[m] for m in mats])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(sign * (a - b))
    if len(deltas) < 100:
        return None
    d = np.array(deltas)
    p = 2 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return dict(delta=float(obs),
                ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p=min(1.0, p))


def main():
    data = json.loads(IN_PATH.read_text())
    montages = list(data)
    if BASELINE not in montages:
        raise SystemExit(f'{BASELINE} missing from {IN_PATH}')
    for m in montages:
        if 'per_segment' not in data[m]:
            raise SystemExit(f'{m} has no per_segment records; re-run '
                             'montage_ablation_pd.py with the updated script')

    per = {m: data[m]['per_segment'] for m in montages}
    common = set.intersection(*(set(v) for v in per.values()))
    by_patient = defaultdict(list)
    for mat in sorted(common):
        by_patient[per[BASELINE][mat]['patient_id']].append(mat)
    print(f'{len(common):,} segments common to all {len(montages)} montages, '
          f'{len(by_patient):,} patients\n')

    out = {'n_segments': len(common), 'n_patients': len(by_patient),
           'n_boot': N_BOOT, 'baseline': BASELINE, 'metrics': {}}

    for label, fn in METRICS.items():
        lower = label in LOWER_IS_BETTER
        base_val = fn([per[BASELINE][m] for m in sorted(common)])
        print(f'== {label} ==   bipolar (production): {base_val:.3f}'
              f'{"  (lower is better)" if lower else ""}')
        out['metrics'][label] = {'baseline_value': base_val, 'alternatives': {}}
        for m in montages:
            if m == BASELINE:
                continue
            rng = np.random.default_rng(SEED)
            val = fn([per[m][x] for x in sorted(common)])
            r = paired_bootstrap(by_patient, per[BASELINE], per[m], fn, lower, rng)
            out['metrics'][label]['alternatives'][m] = dict(value=val, **(r or {}))
            if r:
                star = '*' if r['p'] < 0.05 else ' '
                print(f'   {m:16} {val:6.3f}   d={r["delta"]:+.3f} '
                      f'[{r["ci"][0]:+.3f}, {r["ci"][1]:+.3f}]  p={r["p"]:.3f} {star}')
            else:
                print(f'   {m:16} {val:6.3f}   (bootstrap failed)')
        print()

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print('wrote', OUT_PATH)


if __name__ == '__main__':
    main()
