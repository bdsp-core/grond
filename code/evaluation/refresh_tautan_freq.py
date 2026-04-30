#!/usr/bin/env python3
"""Refresh `tautan_freq_hz` in segments.csv against the current canonical
labeled set (every mat_file with a frequency_hz row in labels.csv).

Background: fig4 (frequency scatter) was showing inconsistent coverage
between the PDProfiler row and the Tautan row because Tautan was computed
on an older subset of segments that didn't fully overlap with the current
labeled set. This script runs `pd_detect_alternate(pk_detect='apd')` --
the Tautan baseline -- on every labeled segment that doesn't yet have a
fresh `tautan_freq_hz` value, then updates segments.csv in place.

Idempotent: skips segments that already have a finite `tautan_freq_hz`
unless `--force` is passed. After running, regenerate fig4 with
`paper_materials/generate_fig6.py` and rerun any tables that depend on
Tautan numbers.

    conda run -n morgoth python code/evaluation/refresh_tautan_freq.py
    conda run -n morgoth python code/evaluation/refresh_tautan_freq.py --force
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
sys.path.insert(0, str(PROJECT_DIR / 'code'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'pd_detector_alternate'))

from pd_detect_alternate import pd_detect_alternate  # type: ignore

LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'labels.csv'
SEGMENTS_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
FS = 200


def load_eeg(mat_file):
    p = EEG_DIR / mat_file
    if not p.exists():
        return None
    m = sio.loadmat(str(p))
    key = [k for k in m.keys() if not k.startswith('_')][0]
    seg = m[key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] != 19:
        return None
    return seg[:, :2000]


def labeled_mat_files():
    """Set of mat_files that have at least one frequency_hz row in labels.csv."""
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
                    help='Recompute tautan_freq_hz even for segments that already have a value.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most N segments (debugging).')
    args = ap.parse_args()

    target_mfs = labeled_mat_files()
    print(f'Labeled mat_files (canonical set): {len(target_mfs)}')

    # Read segments.csv
    with open(SEGMENTS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f'segments.csv rows: {len(rows)}')

    if 'tautan_freq_hz' not in fieldnames:
        fieldnames.append('tautan_freq_hz')

    todo = []
    for r in rows:
        if r['mat_file'] not in target_mfs:
            continue
        cur = (r.get('tautan_freq_hz') or '').strip()
        if cur and not args.force:
            try:
                if np.isfinite(float(cur)):
                    continue
            except ValueError:
                pass
        todo.append(r)
    print(f'Segments to (re)compute Tautan on: {len(todo)}')
    if args.limit:
        todo = todo[:args.limit]
        print(f'  (truncated to {len(todo)} for debugging)')

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for i, r in enumerate(todo):
        mono = load_eeg(r['mat_file'])
        if mono is None:
            n_fail += 1
            continue
        try:
            res = pd_detect_alternate(mono.copy(), FS, pk_detect='apd')
            f = res.get('event_frequency') if isinstance(res, dict) else None
            if hasattr(f, 'item'):
                f = f.item()
            if f is None or not np.isfinite(float(f)):
                r['tautan_freq_hz'] = ''
                n_fail += 1
            else:
                r['tautan_freq_hz'] = f'{float(f):.4f}'
                n_ok += 1
        except Exception:
            n_fail += 1

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - (i + 1)) / max(rate, 0.01)
            print(f'  [{i+1}/{len(todo)}] ok={n_ok} fail={n_fail}  '
                  f'rate={rate:.1f}/s  eta={eta:.0f}s')

    print(f'\nDone: ok={n_ok} fail={n_fail}  total time={time.time()-t0:.0f}s')

    # Write updated segments.csv
    with open(SEGMENTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'Wrote {SEGMENTS_CSV}')


if __name__ == '__main__':
    main()
