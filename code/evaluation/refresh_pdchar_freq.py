#!/usr/bin/env python3
"""Refresh `pdchar_freq_hz` in segments.csv against the current production
PD-Profiler output, on every PD segment with at least one rater frequency
label in labels.csv.

Background: `pdchar_freq_hz` was originally populated by an older PD pipeline
that pre-dates the current PD-Profiler (CNN+ACF prior + HemiCET-UNet evidence
+ DP + EM + IPI-derived frequency). The legacy column is therefore stale
relative to the production algorithm described in the manuscript. This script
re-runs the production PD-Profiler on the canonical labels.csv-labeled set
and writes the fresh `pdchar_freq_hz` value back to segments.csv.

Same idempotency contract as refresh_tautan_freq.py / refresh_algo_freq_rda.py:
skips segments that already have a value unless --force is given.

    conda run -n morgoth python code/evaluation/refresh_pdchar_freq.py
    conda run -n morgoth python code/evaluation/refresh_pdchar_freq.py --force
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

from pd_profiler import PDProfiler  # type: ignore

LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'labels.csv'
SEGMENTS_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
FS = 200

MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2']
BIPOLAR_PAIRS = [
    ('Fp1','F7'), ('F7','T3'), ('T3','T5'), ('T5','O1'),
    ('Fp2','F8'), ('F8','T4'), ('T4','T6'), ('T6','O2'),
    ('Fp1','F3'), ('F3','C3'), ('C3','P3'), ('P3','O1'),
    ('Fp2','F4'), ('F4','C4'), ('C4','P4'), ('P4','O2'),
    ('Fz','Cz'), ('Cz','Pz'),
]


def load_mono(mat_file):
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


def mono_to_bipolar(mono):
    ch_idx = {ch: i for i, ch in enumerate(MONO_CHANNELS)}
    bip = np.zeros((18, mono.shape[1]))
    for i, (a, b) in enumerate(BIPOLAR_PAIRS):
        bip[i] = mono[ch_idx[a]] - mono[ch_idx[b]]
    return bip


def labeled_pd_mat_files():
    """Set of mat_files of subtype lpd/gpd that have a labels.csv freq label."""
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
                    help='Recompute even for segments that already have a value.')
    ap.add_argument('--limit', type=int, default=None,
                    help='Process at most N segments (debugging).')
    args = ap.parse_args()

    target = labeled_pd_mat_files()
    print(f'labels.csv-labeled mat_files: {len(target)}')

    with open(SEGMENTS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    print(f'segments.csv rows: {len(rows)}')
    if 'pdchar_freq_hz' not in fieldnames:
        fieldnames.append('pdchar_freq_hz')

    todo = []
    for r in rows:
        if r.get('subtype', '').lower() not in ('lpd', 'gpd'):
            continue
        if r['mat_file'] not in target:
            continue
        cur = (r.get('pdchar_freq_hz') or '').strip()
        if cur and not args.force:
            try:
                if np.isfinite(float(cur)):
                    continue
            except ValueError:
                pass
        todo.append(r)
    print(f'PD segments to (re)compute pdchar_freq_hz on: {len(todo)}')
    if args.limit:
        todo = todo[:args.limit]
        print(f'  (truncated to {len(todo)} for debugging)')

    print('Loading PDProfiler...')
    pc = PDProfiler()
    print('Done.')

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for i, r in enumerate(todo):
        mono = load_mono(r['mat_file'])
        if mono is None:
            n_fail += 1
            continue
        try:
            bip = mono_to_bipolar(mono)
            result = pc.characterize(bip, subtype=r['subtype'].lower())
            f = result.get('frequency')
            if f is None or not np.isfinite(float(f)):
                r['pdchar_freq_hz'] = ''
                n_fail += 1
            else:
                r['pdchar_freq_hz'] = f'{float(f):.4f}'
                n_ok += 1
        except Exception:
            n_fail += 1

        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            eta = (len(todo) - (i + 1)) / max(rate, 0.01)
            print(f'  [{i+1}/{len(todo)}] ok={n_ok} fail={n_fail}  rate={rate:.1f}/s  eta={eta:.0f}s')

    print(f'\nDone: ok={n_ok} fail={n_fail}  total time={time.time()-t0:.0f}s')

    with open(SEGMENTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'Wrote {SEGMENTS_CSV}')


if __name__ == '__main__':
    main()
