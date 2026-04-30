#!/usr/bin/env python3
"""Refresh `algo_freq_hz` (NB-Hilbert / V12) in segment_labels.csv on every
RDA segment with a frequency label in labels.csv.

Background: `algo_freq_hz` was originally populated for a frozen subset
of segments years ago. The labels.csv canonical set has grown since
(MW added ~1000 new LRDA + ~1500 new GRDA frequency labels). When fig4
joins on labels.csv expert frequency, the overlap with the legacy
algo_freq_hz subset is small (~182 LRDA / ~182 GRDA), making the figure
n's silly relative to the actual labeled set.

This script runs V12 NB-Hilbert (the shipping RDA-Profiler frequency
estimator: pass-1 0.5-4.5 Hz, pass-2 narrowband half-width 0.5 Hz,
top-3 channels, frequency cap 4.5 Hz) on every RDA segment that has a
frequency label in labels.csv but no current `algo_freq_hz` value, and
writes the result back to segment_labels.csv.

By default keeps existing values. Pass --force to overwrite all.

    conda run -n morgoth python code/evaluation/refresh_algo_freq_rda.py
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))

from generate_v12_predictions import w05_v12  # type: ignore
from generate_rda_freq_labeler import load_segment, FS  # type: ignore

LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'labels.csv'
SEGMENT_LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv'


def labeled_rda_mat_files():
    """Set of mat_files of subtype lrda/grda that have a labels.csv freq label."""
    out = set()
    with open(LABELS_CSV) as f:
        for r in csv.DictReader(f):
            if r.get('label_type') == 'frequency_hz':
                v = (r.get('value') or '').strip()
                try:
                    if float(v) > 0:
                        out.add(r['mat_file'])
                except ValueError:
                    pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true',
                    help='Recompute even for segments that already have an algo_freq_hz value.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most N segments (debugging).')
    args = ap.parse_args()

    target = labeled_rda_mat_files()
    print(f'labels.csv-labeled mat_files: {len(target)}')

    with open(SEGMENT_LABELS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f'segment_labels.csv rows: {len(rows)}')

    todo = []
    for r in rows:
        if r.get('subtype', '').lower() not in ('lrda', 'grda'):
            continue
        if r['mat_file'] not in target:
            continue
        cur = (r.get('algo_freq_hz') or '').strip()
        if cur and not args.force:
            try:
                if np.isfinite(float(cur)):
                    continue
            except ValueError:
                pass
        todo.append(r)
    print(f'RDA segments to (re)compute V12 algo_freq_hz on: {len(todo)}')
    if args.limit:
        todo = todo[:args.limit]
        print(f'  (truncated to {len(todo)} for debugging)')

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for i, r in enumerate(todo):
        seg = load_segment(r['mat_file'])
        if seg is None:
            n_fail += 1
            continue
        try:
            f, _lat = w05_v12(seg)
            if f is None or not np.isfinite(float(f)):
                r['algo_freq_hz'] = ''
                n_fail += 1
            else:
                r['algo_freq_hz'] = f'{float(f):.2f}'
                n_ok += 1
        except Exception:
            n_fail += 1

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - (i + 1)) / max(rate, 0.01)
            print(f'  [{i+1}/{len(todo)}] ok={n_ok} fail={n_fail}  rate={rate:.1f}/s  eta={eta:.0f}s')

    print(f'\nDone: ok={n_ok} fail={n_fail}  total time={time.time()-t0:.0f}s')

    # Write updated segment_labels.csv
    with open(SEGMENT_LABELS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'Wrote {SEGMENT_LABELS_CSV}')


if __name__ == '__main__':
    main()
