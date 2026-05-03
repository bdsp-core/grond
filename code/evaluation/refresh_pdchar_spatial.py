#!/usr/bin/env python3
"""Refresh per-channel CNN predictions and pdchar_spatial_extent.

Re-runs the production PDProfiler on every PD segment with a label or with
spatial-extent annotations, refreshes:
  - data/labels/predictions.json (channel_probs per segment)
  - data/labels/segments.csv (pdchar_spatial_extent column)
  - paper_materials/spatial_inference_cache.json (entries for spatial-extent-annotated segments)

Uses threshold = 0.62 to binarize per-channel probs into the spatial-extent
fraction (the default 0.5 saturates because the per-channel CNN was trained
on segment-level labels; see code/evaluation/run_all_inference.py docstring).

Usage:
    conda run -n morgoth python code/evaluation/refresh_pdchar_spatial.py
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))
from pd_profiler import PDProfiler  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEGMENTS_CSV = LABELS_DIR / 'segments.csv'
PREDICTIONS_JSON = LABELS_DIR / 'predictions.json'
SPATIAL_CACHE = PROJECT_DIR / 'paper_materials' / 'spatial_inference_cache.json'

PD_THRESHOLD = 0.62

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])


def load_eeg(mat_file):
    import scipy.io as sio
    p = EEG_DIR / mat_file
    if not p.exists():
        return None, None
    mat = sio.loadmat(str(p))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] != 19:
        return None, None
    bipolar = seg[BIPOLAR_INDICES[:, 0]] - seg[BIPOLAR_INDICES[:, 1]]
    return seg, bipolar


def main():
    print('Loading segments + existing predictions/cache ...')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    pd_mfs = set(sl[sl.subtype.isin(['lpd', 'gpd']) & (sl.excluded != True)].mat_file)
    print(f'  PD segments to (re)compute: {len(pd_mfs)}')

    if PREDICTIONS_JSON.exists():
        with open(PREDICTIONS_JSON) as f:
            predictions = json.load(f)
    else:
        predictions = {}

    if SPATIAL_CACHE.exists():
        with open(SPATIAL_CACHE) as f:
            spatial_cache = json.load(f)
    else:
        spatial_cache = {}

    pc = PDProfiler()
    print('PDProfiler loaded.\n')

    n_ok, n_fail = 0, 0
    t0 = time.time()
    for i, mf in enumerate(sorted(pd_mfs)):
        sub = sl[sl.mat_file == mf].subtype.iloc[0]
        mono, bip = load_eeg(mf)
        if mono is None:
            n_fail += 1
            continue
        try:
            result = pc.characterize(bip, subtype=sub)
            probs = np.asarray(result.get('channel_probs', [0]*18), dtype=float)
            if probs.shape[0] != 18:
                n_fail += 1
                continue
            entry = predictions.get(mf, {})
            entry['channel_probs'] = probs.tolist()
            predictions[mf] = entry
            # Spatial cache (only for segments that already had cache entries
            # OR that have spatial annotations)
            if mf in spatial_cache:
                ce = spatial_cache[mf]
                ce['pdchar_channel_probs'] = probs.tolist()
                ce['pdchar_spatial_extent'] = float(np.mean(probs > PD_THRESHOLD))
                spatial_cache[mf] = ce
            n_ok += 1
        except Exception:
            n_fail += 1

        if (i + 1) % 200 == 0 or i + 1 == len(pd_mfs):
            rate = (i + 1) / max(time.time() - t0, 0.001)
            eta = (len(pd_mfs) - i - 1) / max(rate, 0.001)
            print(f'  {i+1}/{len(pd_mfs)}  ok={n_ok} fail={n_fail}  '
                  f'rate={rate:.1f}/s  ETA={eta/60:.1f} min')

    print(f'\nDone: {n_ok} ok, {n_fail} fail. Writing outputs...')

    # Write predictions.json
    with open(PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)
    print(f'  wrote {PREDICTIONS_JSON.relative_to(PROJECT_DIR)}')

    # Write spatial cache
    with open(SPATIAL_CACHE, 'w') as f:
        json.dump(spatial_cache, f)
    print(f'  wrote {SPATIAL_CACHE.relative_to(PROJECT_DIR)}')

    # Update segments.csv pdchar_spatial_extent column from refreshed probs
    print('Updating segments.csv pdchar_spatial_extent ...')
    with open(SEGMENTS_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if 'pdchar_spatial_extent' not in fieldnames:
        fieldnames.append('pdchar_spatial_extent')

    n_set = 0
    for r in rows:
        mf = r['mat_file']
        if r.get('subtype', '').lower() not in ('lpd', 'gpd'):
            continue
        cp = predictions.get(mf, {}).get('channel_probs')
        if not isinstance(cp, list) or len(cp) != 18:
            continue
        se = float(np.mean(np.array(cp) > PD_THRESHOLD))
        r['pdchar_spatial_extent'] = f'{se:.4f}'
        n_set += 1

    with open(SEGMENTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'  set pdchar_spatial_extent on {n_set} rows')

    # Quick distribution check
    arr = np.array([
        float(r['pdchar_spatial_extent']) for r in rows
        if r.get('pdchar_spatial_extent', '').strip()
        and r.get('subtype', '').lower() in ('lpd', 'gpd')
    ])
    print(f'  pdchar_spatial_extent distribution: n={len(arr)}  mean={arr.mean():.3f}  '
          f'unique values: {len(set(np.round(arr, 3)))}')


if __name__ == '__main__':
    main()
