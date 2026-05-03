#!/usr/bin/env python3
"""Phase 3 comparison harness: retrained vs published numbers.

Loads the result JSONs that the Phase 1 retrain regenerates and diffs each
headline metric against the value cited in the manuscript (paper_materials/
manuscript.tex). Each row is tagged within-tolerance / outside-tolerance.

Usage:
    conda run -n morgoth python code/evaluation/compare_retrain_to_published.py \
        --out results/retrain_v1/comparison_report.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

# (label, manuscript_value, retrain_path, retrain_key, abs_tolerance)
PUBLISHED = [
    # Lateralization (Table 3) -- AUCs
    ('LPD-vs-GPD AUC',         0.911,
     'data/evaluation_results/subtype_classification.json',  'auc',           0.005),
    ('3-way macro AUC',        0.862,
     'data/evaluation_results/three_way_classification.json', 'macro_auc',    0.030),  # cohort grew, wider tol
    ('LPD one-vs-rest AUC',    0.832,
     'data/evaluation_results/three_way_classification.json', ['per_class_auc', 'LPD'],  0.030),
    ('GPD one-vs-rest AUC',    0.835,
     'data/evaluation_results/three_way_classification.json', ['per_class_auc', 'GPD'],  0.030),
    ('BIPD one-vs-rest AUC',   0.920,
     'data/evaluation_results/three_way_classification.json', ['per_class_auc', 'BIPD'], 0.030),
    ('BIPD-vs-GPD AUC',        0.937,
     'data/evaluation_results/three_way_classification.json', 'bipd_vs_gpd_auc',         0.030),
]


def deep_get(d, keypath):
    if isinstance(keypath, str):
        return d.get(keypath)
    cur = d
    for k in keypath:
        if cur is None:
            return None
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/retrain_v1/comparison_report.md')
    args = ap.parse_args()

    rows = []
    for label, pub, path, key, tol in PUBLISHED:
        rj_path = PROJECT_DIR / path
        if not rj_path.exists():
            rows.append((label, pub, None, None, tol, 'MISSING'))
            continue
        with open(rj_path) as f:
            data = json.load(f)
        retrain = deep_get(data, key)
        if retrain is None:
            rows.append((label, pub, None, None, tol, 'MISSING_KEY'))
            continue
        delta = retrain - pub
        within = 'OK' if abs(delta) <= tol else 'OUT_OF_TOL'
        rows.append((label, pub, retrain, delta, tol, within))

    # ---- Console + markdown output ----
    print(f'\n{"Label":<30s} {"Pub":>8s} {"Retrain":>9s} {"Delta":>8s} {"Tol":>6s}  Status')
    print('-' * 80)
    md = ['# Phase 1 retrain vs published comparison',
          '',
          f'Repo: {PROJECT_DIR}',
          '',
          '| Metric | Published | Retrain | Δ | Tolerance | Status |',
          '|---|---:|---:|---:|---:|:---:|']
    for label, pub, retrain, delta, tol, status in rows:
        if retrain is None:
            line = f'{label:<30s} {pub:>8.4f} {"--":>9s} {"--":>8s} {tol:>6.3f}  {status}'
            md_line = f'| {label} | {pub:.4f} | — | — | {tol:.3f} | {status} |'
        else:
            line = f'{label:<30s} {pub:>8.4f} {retrain:>9.4f} {delta:+8.4f} {tol:>6.3f}  {status}'
            md_line = (f'| {label} | {pub:.4f} | {retrain:.4f} | '
                        f'{delta:+.4f} | {tol:.3f} | **{status}** |')
        print(line)
        md.append(md_line)

    n_ok = sum(1 for r in rows if r[-1] == 'OK')
    n_out = sum(1 for r in rows if r[-1] == 'OUT_OF_TOL')
    n_miss = sum(1 for r in rows if r[-1].startswith('MISSING'))
    summary = (f'\nSummary: {n_ok}/{len(rows)} within tolerance, '
                f'{n_out} outside tolerance, {n_miss} missing.')
    print(summary)
    md.append('')
    md.append(summary.strip())

    out_path = PROJECT_DIR / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'\nReport saved to {out_path.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
