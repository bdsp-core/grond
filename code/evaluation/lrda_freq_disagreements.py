#!/usr/bin/env python3
"""List LRDA segments where the V14 algorithm frequency disagrees with any
human rater's frequency by more than a threshold (default 0.25 Hz).

Outputs:
    paper_materials/independent_expert_tasks/lrda/lrda_freq_disagreements_v14.csv
        Full per-segment table with each rater's freq, V14 algo freq, max diff,
        and which raters exceeded the threshold.
    paper_materials/independent_expert_tasks/lrda/lrda_freq_disagreements_v14.md
        Human-readable sorted summary (worst-first).
    paper_materials/independent_expert_tasks/lrda/lrda_freq_disagreements_v14_manifest.csv
        Minimal mat_file/patient_id manifest sorted by max-diff descending; can
        be passed to any tool that takes a `--manifest` flag (e.g. the
        re-scoring HTML reviewer).

    conda run -n morgoth python code/evaluation/lrda_freq_disagreements.py
    conda run -n morgoth python code/evaluation/lrda_freq_disagreements.py --threshold 0.5
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'

RATERS = ['MW', 'SZ', 'TZ', 'LB', 'PH']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threshold', type=float, default=0.25,
                    help='Max-diff threshold in Hz (default 0.25)')
    ap.add_argument('--out-prefix', type=str, default='lrda_freq_disagreements_v14',
                    help='Output filename prefix (default lrda_freq_disagreements_v14)')
    args = ap.parse_args()

    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    labels = pd.read_csv(LABELS_DIR / 'labels.csv')

    # LRDA segments + V14 algo freq.  V14 reuses V12 frequency (only laterality
    # differs), so algo_freq_hz in segment_labels.csv is the V14/V12 value.
    lrda = sl[sl.subtype == 'lrda'].copy()
    lrda['algo_freq_hz'] = pd.to_numeric(lrda['algo_freq_hz'], errors='coerce')
    lrda = lrda[lrda.algo_freq_hz.notna()][['mat_file', 'patient_id', 'algo_freq_hz']]
    print(f'LRDA segments with V14/V12 algo_freq_hz: {len(lrda)}')

    # Per-rater frequency labels
    fr = labels[labels.label_type == 'frequency_hz'].copy()
    fr['value'] = pd.to_numeric(fr['value'], errors='coerce')
    fr = fr[fr.value.notna() & (fr.value > 0) & fr.rater.isin(RATERS)]
    fr = fr[fr.mat_file.isin(set(lrda.mat_file))]

    # Pivot to one row per mat_file with each rater's freq as column
    rater_pivot = fr.pivot_table(index='mat_file', columns='rater', values='value', aggfunc='mean')
    rater_pivot = rater_pivot.reindex(columns=RATERS)  # keep canonical order

    merged = lrda.set_index('mat_file').join(rater_pivot, how='inner')
    print(f'LRDA segments with both algo and at least one human freq: {len(merged)}')

    rows = []
    for mf, row in merged.iterrows():
        algo = float(row['algo_freq_hz'])
        diffs = {}
        for r in RATERS:
            v = row.get(r, np.nan)
            if pd.notna(v):
                diffs[r] = float(v) - algo
        max_abs = max((abs(d) for d in diffs.values()), default=0.0)
        over = sorted([r for r, d in diffs.items() if abs(d) > args.threshold])
        if max_abs <= args.threshold:
            continue
        rec = {
            'mat_file': mf,
            'patient_id': row.get('patient_id', ''),
            'algo_freq_hz': round(algo, 3),
            'max_abs_diff_hz': round(max_abs, 3),
            'raters_over_threshold': ';'.join(over),
            'n_raters': sum(1 for r in RATERS if pd.notna(row.get(r))),
        }
        for r in RATERS:
            v = row.get(r, np.nan)
            rec[f'{r}_freq'] = round(float(v), 2) if pd.notna(v) else ''
            rec[f'{r}_diff'] = round(float(v) - algo, 2) if pd.notna(v) else ''
        rows.append(rec)

    rows.sort(key=lambda r: -r['max_abs_diff_hz'])
    print(f'\nDisagreement cases (|diff| > {args.threshold} Hz): {len(rows)}')

    # Distribution
    diffs = [r['max_abs_diff_hz'] for r in rows]
    if diffs:
        print(f'  median max-diff: {np.median(diffs):.2f} Hz')
        print(f'  p75 max-diff:    {np.percentile(diffs, 75):.2f} Hz')
        print(f'  p90 max-diff:    {np.percentile(diffs, 90):.2f} Hz')
        print(f'  worst:           {max(diffs):.2f} Hz')

    # ---------- CSV ----------
    cols = (['mat_file', 'patient_id', 'algo_freq_hz', 'max_abs_diff_hz',
             'raters_over_threshold', 'n_raters'] +
            [f'{r}_freq' for r in RATERS] +
            [f'{r}_diff' for r in RATERS])
    csv_path = TASKS_DIR / f'{args.out_prefix}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nWrote {csv_path.relative_to(PROJECT_DIR)}')

    # ---------- Manifest ----------
    manifest_path = TASKS_DIR / f'{args.out_prefix}_manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['mat_file', 'patient_id'])
        for r in rows:
            w.writerow([r['mat_file'], r['patient_id']])
    print(f'Wrote {manifest_path.relative_to(PROJECT_DIR)}  ({len(rows)} cases)')

    # ---------- Markdown ----------
    md_path = TASKS_DIR / f'{args.out_prefix}.md'
    md = []
    md.append(f'# LRDA frequency disagreements: V14 algorithm vs human raters')
    md.append('')
    md.append(f'Found **{len(rows)}** LRDA segments where the V14 algorithm '
              f'frequency differs from at least one human rater by more than '
              f'**{args.threshold} Hz**.')
    md.append('')
    md.append(f'- Algorithm: V14 (V12 NB-Hilbert frequency: pass-1 0.5–4.5 Hz, '
              f'pass-2 narrowband half-width 0.5 Hz, top-3 channels, freq cap 4.5 Hz).')
    md.append(f'- Raters considered: {", ".join(RATERS)}.')
    md.append(f'- Per-rater coverage among the {len(rows)} flagged segments:')
    for r in RATERS:
        n = sum(1 for row in rows if row.get(f'{r}_freq', '') != '')
        md.append(f'  - {r}: {n}')
    md.append('')
    md.append('Threshold-exceedance breakdown:')
    md.append(f'  - max-diff > 0.25 Hz: {sum(1 for d in diffs if d > 0.25)}')
    md.append(f'  - max-diff > 0.50 Hz: {sum(1 for d in diffs if d > 0.50)}')
    md.append(f'  - max-diff > 0.75 Hz: {sum(1 for d in diffs if d > 0.75)}')
    md.append(f'  - max-diff > 1.00 Hz: {sum(1 for d in diffs if d > 1.00)}')
    md.append('')
    md.append('## Top 50 cases (sorted by max |diff|, worst first)')
    md.append('')
    md.append('| # | mat_file (short) | algo | MW | SZ | TZ | LB | PH | max|Δ| | raters > thresh |')
    md.append('|---|---|---|---|---|---|---|---|---|---|')
    for i, r in enumerate(rows[:50]):
        short = r['mat_file'].replace('.mat', '').replace('sub-', '')[:28]
        cells = [str(i+1), f'`{short}`', f'{r["algo_freq_hz"]:.2f}']
        for rater in RATERS:
            v = r.get(f'{rater}_freq', '')
            d = r.get(f'{rater}_diff', '')
            if v == '':
                cells.append('—')
            else:
                cells.append(f'{v:.2f} ({d:+.2f})')
        cells.append(f'**{r["max_abs_diff_hz"]:.2f}**')
        cells.append(r['raters_over_threshold'])
        md.append('| ' + ' | '.join(cells) + ' |')
    if len(rows) > 50:
        md.append('')
        md.append(f'_... {len(rows) - 50} more cases in the CSV._')
    md.append('')
    md.append(f'## How to re-score')
    md.append(f'')
    md.append(f'1. The CSV at `{csv_path.relative_to(PROJECT_DIR)}` has every '
              f'flagged case sorted by max |diff|.')
    md.append(f'2. The minimal manifest at `{manifest_path.relative_to(PROJECT_DIR)}` '
              f'(mat_file, patient_id) can feed an HTML reviewer.')
    md.append(f'3. After re-scoring, run `code/data_management/apply_lrda_corrections.py` '
              f'with the corrections JSON to merge the new labels into labels.csv.')
    md_path.write_text('\n'.join(md) + '\n')
    print(f'Wrote {md_path.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
