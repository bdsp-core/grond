#!/usr/bin/env python3
"""Rank LRDA segments by expert-vs-algorithm frequency disagreement.

For each segment in the LRDA manifest with at least one expert frequency
label, compute

    score = |mean(available expert freqs) - algorithm frequency|

and report the top N (default 20) segments. Also writes a sub-manifest
CSV that the existing LRDA viewer can consume and a side-by-side
markdown summary with all four raters' labels per case.

This is the broader cousin of lrda_disagreement_triage.py:
- triage:    cases where MW specifically disagrees with both algo AND
             at least one other expert (7 cases; investigates whether
             MW is the outlier)
- this:      cases where the experts as a group disagree with algo
             (regardless of which experts; investigates the algorithm
             itself)

Usage:
    conda run -n morgoth python code/evaluation/lrda_top_disagreements.py
    conda run -n morgoth python code/evaluation/lrda_top_disagreements.py --top 30
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'


def load_labels():
    """Return (lrda_rows_in_manifest_order, freq, lat) where freq[rater][mat_file] = float."""
    with open(TASKS_DIR / 'manifest.csv') as f:
        lrda_rows = list(csv.DictReader(f))
    lrda_mfs = {r['mat_file'] for r in lrda_rows}

    freq = {r: {} for r in ('MW', 'SZ', 'TZ', 'ALGO')}
    lat = {r: {} for r in ('MW', 'SZ', 'TZ', 'ALGO')}

    # MW, SZ, TZ -- canonical labels.csv (any round)
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            rater = row['rater']
            if rater not in ('MW', 'SZ', 'TZ'):
                continue
            mf = row['mat_file']
            if mf not in lrda_mfs:
                continue
            lt = row['label_type']
            v = row['value'].strip()
            if lt == 'frequency_hz':
                try:
                    freq[rater][mf] = float(v)
                except ValueError:
                    pass
            elif lt == 'laterality':
                vl = v.lower()
                if vl in ('left', 'right'):
                    lat[rater][mf] = vl

    # ALGO -- from rater export JSONs (algorithm pre-fills are
    # deterministic per segment; any rater's file works)
    for fp in [
        'data/labels/raw_inputs/independent_expert_v1/TZ/lrda_freq_labeling_results_TZ.json',
        'data/labels/raw_inputs/independent_expert_v1/SZ/rda_freq_labeling_results-2.json',
        'data/labels/raw_inputs/independent_expert_v1/MW/rda_freq_labeling_results-mbw2.json',
    ]:
        with open(PROJECT_DIR / fp) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf not in lrda_mfs or sub != 'lrda':
                continue
            if mf not in freq['ALGO'] and v.get('w05_freq') is not None:
                freq['ALGO'][mf] = float(v['w05_freq'])
            if mf not in lat['ALGO'] and v.get('w05_laterality') in ('left', 'right'):
                lat['ALGO'][mf] = v['w05_laterality']

    return lrda_rows, freq, lat


def score_one(mf, freq, lat):
    """Return a per-segment dict if it can be scored; else None."""
    exp_freqs = {r: freq[r][mf] for r in ('MW', 'SZ', 'TZ') if mf in freq[r]}
    algo_freq = freq['ALGO'].get(mf)
    if not exp_freqs or algo_freq is None:
        return None
    mean_exp = sum(exp_freqs.values()) / len(exp_freqs)
    max_exp_diff = max(abs(v - algo_freq) for v in exp_freqs.values())
    return {
        'mat_file': mf,
        'mw_freq': freq['MW'].get(mf),
        'sz_freq': freq['SZ'].get(mf),
        'tz_freq': freq['TZ'].get(mf),
        'algo_freq': algo_freq,
        'mean_expert_freq': mean_exp,
        'max_expert_algo_diff': max_exp_diff,
        'score': abs(mean_exp - algo_freq),
        'n_experts': len(exp_freqs),
        'mw_lat': lat['MW'].get(mf),
        'sz_lat': lat['SZ'].get(mf),
        'tz_lat': lat['TZ'].get(mf),
        'algo_lat': lat['ALGO'].get(mf),
    }


def write_manifest(top_rows, manifest_lookup, out_path):
    """Write a sub-manifest CSV in the same schema as manifest.csv."""
    fieldnames = list(next(iter(manifest_lookup.values())).keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        w.writeheader()
        for r in top_rows:
            w.writerow(manifest_lookup[r['mat_file']])


def write_summary(top_rows, out_path, n):
    """Write a markdown side-by-side summary."""
    lines = []
    lines.append(f'# Top {n} LRDA expert-vs-algorithm disagreements\n')
    lines.append('Sorted by `score = |mean(expert freqs) - algo freq|`. Cases at the top of the table contribute most to the LRDA-frequency expert-vs-algorithm gap on figS5.\n')
    lines.append('\n## Summary statistics\n')

    n_algo_below = sum(1 for r in top_rows if r['mean_expert_freq'] > r['algo_freq'])
    n_algo_above = sum(1 for r in top_rows if r['mean_expert_freq'] < r['algo_freq'])
    n_double = sum(1 for r in top_rows
                   if r['algo_freq'] > 0
                   and 1.6 <= r['mean_expert_freq'] / r['algo_freq'] <= 2.4)
    lines.append(f'- Algorithm BELOW expert mean: **{n_algo_below}** of {len(top_rows)}\n')
    lines.append(f'- Algorithm ABOVE expert mean: **{n_algo_above}** of {len(top_rows)}\n')
    lines.append(f'- Algorithm at ~half expert mean (ratio 1.6-2.4): **{n_double}** of {len(top_rows)}  -- classic sub-harmonic locking pattern\n')
    expert_freqs_in_top = [r['mean_expert_freq'] for r in top_rows]
    if expert_freqs_in_top:
        lines.append(f'- Expert-mean frequency distribution: '
                     f'min={min(expert_freqs_in_top):.2f}, '
                     f'median={sorted(expert_freqs_in_top)[len(expert_freqs_in_top)//2]:.2f}, '
                     f'max={max(expert_freqs_in_top):.2f} Hz\n')

    lines.append('\n## Per-case table (sorted by disagreement)\n')
    lines.append('| # | segment_id | MW | SZ | TZ | mean exp | ALGO | score | ratio | lat (MW/SZ/TZ/ALGO) |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for i, r in enumerate(top_rows, 1):
        seg = r['mat_file'].replace('.mat', '')
        def fmt(v): return f'{v:.2f}' if v is not None else '—'
        ratio = (r['mean_expert_freq'] / r['algo_freq']) if r['algo_freq'] > 0 else float('nan')
        ratio_str = f'{ratio:.2f}×' if r['algo_freq'] > 0 else '—'
        lat_str = (
            (r['mw_lat'] or '—')[:1].upper() + '/' +
            (r['sz_lat'] or '—')[:1].upper() + '/' +
            (r['tz_lat'] or '—')[:1].upper() + '/' +
            (r['algo_lat'] or '—')[:1].upper()
        )
        lines.append(
            f'| {i} | `{seg}` | {fmt(r["mw_freq"])} | {fmt(r["sz_freq"])} | {fmt(r["tz_freq"])} | '
            f'**{r["mean_expert_freq"]:.2f}** | **{r["algo_freq"]:.2f}** | '
            f'**{r["score"]:.2f}** | {ratio_str} | {lat_str} |'
        )

    lines.append('\n## How to read this\n')
    lines.append('- **score**: the headline metric -- magnitude of disagreement in Hz.\n')
    lines.append('- **ratio**: mean-expert / algorithm. Values near 2.0 indicate the algorithm is reading the sub-harmonic.\n')
    lines.append('- **lat**: laterality letters in order MW / SZ / TZ / ALGO. "L"/"R"/"--".\n')
    lines.append('\n## To review interactively\n')
    lines.append(f'Open the focused viewer at `top{n}_disagreement_viewer.html` (regenerated from this manifest with the standard LRDA viewer) -- arrow keys to navigate, up/down to step through frequency buttons and watch which value the green narrowband overlay locks onto.\n')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top-disagreement cases to report (default 20)')
    args = parser.parse_args()

    lrda_rows, freq, lat = load_labels()
    manifest_lookup = {r['mat_file']: r for r in lrda_rows}

    print(f'Scoring {len(lrda_rows)} LRDA manifest segments...')
    scored = []
    for row in lrda_rows:
        s = score_one(row['mat_file'], freq, lat)
        if s is not None:
            scored.append(s)
    print(f'  Segments with at least one expert + algorithm label: {len(scored)}')
    print()

    scored.sort(key=lambda x: -x['score'])
    top = scored[:args.top]

    print(f'Top {args.top} cases by |mean_expert - algo|:')
    print(f'{"#":>3s}  {"segment_id":42s}  {"MW":>5s}  {"SZ":>5s}  {"TZ":>5s}  {"mean":>5s}  {"ALGO":>5s}  {"score":>5s}')
    for i, r in enumerate(top, 1):
        seg = r['mat_file'].replace('.mat', '')[:42]
        def f(v): return f'{v:.2f}' if v is not None else '  -- '
        print(f'{i:>3d}  {seg:42s}  {f(r["mw_freq"]):>5s}  {f(r["sz_freq"]):>5s}  {f(r["tz_freq"]):>5s}  {r["mean_expert_freq"]:>5.2f}  {r["algo_freq"]:>5.2f}  {r["score"]:>5.2f}')

    manifest_path = TASKS_DIR / f'top{args.top}_disagreement_manifest.csv'
    summary_path = TASKS_DIR / f'top{args.top}_disagreement_summary.md'
    write_manifest(top, manifest_lookup, manifest_path)
    write_summary(top, summary_path, args.top)
    print()
    print(f'Wrote {manifest_path}')
    print(f'Wrote {summary_path}')

    n_below = sum(1 for r in top if r['mean_expert_freq'] > r['algo_freq'])
    n_above = sum(1 for r in top if r['mean_expert_freq'] < r['algo_freq'])
    n_double = sum(1 for r in top
                   if r['algo_freq'] > 0
                   and 1.6 <= r['mean_expert_freq'] / r['algo_freq'] <= 2.4)
    print()
    print(f'Direction summary:')
    print(f'  Algorithm BELOW expert mean: {n_below}/{args.top}')
    print(f'  Algorithm ABOVE expert mean: {n_above}/{args.top}')
    print(f'  ~Sub-harmonic (ratio 1.6-2.4x): {n_double}/{args.top}')


if __name__ == '__main__':
    main()
