#!/usr/bin/env python3
"""
Run all model inference and save predictions for performance evaluation.

Stores results in two files:
  - data/labels/segments.csv: scalar predictions (laterality, freq, spatial extent)
    New columns: pdchar_laterality, pdchar_spatial_extent, tautan_spatial_extent,
                 w05_laterality, w05_freq_hz, rda_plv_spatial_extent
  - data/labels/predictions.json: per-channel/per-discharge predictions
    Keys: {mat_file: {channel_probs: [...], discharge_times: [...], ...}}

This only needs to be re-run when:
  - Model weights change (retrained)
  - New EEG segments are added
  - Algorithm code is modified

Usage:
    conda run -n morgoth python code/evaluation/run_all_inference.py
    conda run -n morgoth python code/evaluation/run_all_inference.py --pd-only
    conda run -n morgoth python code/evaluation/run_all_inference.py --rda-only
"""

import argparse
import csv
import json
import sys
import time
import numpy as np
import scipy.io as sio
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

SEGMENTS_CSV = LABELS_DIR / 'segments.csv'
PREDICTIONS_JSON = LABELS_DIR / 'predictions.json'

FS = 200
MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
LEFT_CH = [0, 1, 2, 3, 8, 9, 10, 11]  # bipolar left channels
RIGHT_CH = [4, 5, 6, 7, 12, 13, 14, 15]


def load_mono(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[key].astype(np.float64)
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


def load_segments():
    """Load segments.csv as list of dicts."""
    with open(SEGMENTS_CSV) as f:
        return list(csv.DictReader(f))


def save_segments(rows, fieldnames):
    """Save segments.csv."""
    with open(SEGMENTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pd_inference(segments, predictions):
    """Run PDProfiler on LPD/GPD segments."""
    from pd_profiler import PDProfiler
    pc = PDProfiler()

    pd_segs = [s for s in segments
                if s.get('subtype', '').lower() in ('lpd', 'gpd')
                and s.get('excluded', '').lower() not in ('true', '1', 'yes')]

    print(f"\nRunning PDProfiler on {len(pd_segs)} PD segments...")
    t0 = time.time()
    n_ok, n_fail = 0, 0

    for i, seg in enumerate(pd_segs):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pd_segs)}] ({time.time()-t0:.0f}s)...")

        mat_file = seg['mat_file']
        mono = load_mono(mat_file)
        if mono is None:
            n_fail += 1
            continue

        try:
            bip = mono_to_bipolar(mono)
            result = pc.characterize(bip, subtype=seg['subtype'].lower())

            # Scalar predictions → segments.csv
            probs = np.array(result.get('channel_probs', [0]*18))
            left_mean = float(np.mean(probs[LEFT_CH]))
            right_mean = float(np.mean(probs[RIGHT_CH]))
            seg['pdchar_laterality'] = 'left' if left_mean > right_mean else 'right'
            # Threshold 0.62 (not 0.5): the per-channel CNN saturates above 0.5
            # even on uninvolved channels, so a 0.5 threshold gives a degenerate
            # binary {0, 1} spatial-extent distribution. 0.62 yields a non-degenerate
            # fraction-involved distribution and matches the downstream figure scripts
            # (generate_fig_spatial_scatter.py, generate_fig_irr.py).
            seg['pdchar_spatial_extent'] = float(np.mean(probs > 0.62))

            # Per-channel predictions → predictions.json
            predictions[mat_file] = predictions.get(mat_file, {})
            predictions[mat_file]['channel_probs'] = probs.tolist()
            predictions[mat_file]['pdchar_laterality'] = seg['pdchar_laterality']

            n_ok += 1
        except Exception as e:
            n_fail += 1

    print(f"  PD done: {n_ok} OK, {n_fail} failed ({time.time()-t0:.0f}s)")


def run_tautan_inference(segments, predictions):
    """Run Tautan et al. on all segments for spatial extent."""
    from pd_detect_alternate import pd_detect_alternate

    non_excluded = [s for s in segments
                     if s.get('excluded', '').lower() not in ('true', '1', 'yes')
                     and s.get('subtype', '').lower() in ('lpd', 'gpd', 'lrda', 'grda')]

    print(f"\nRunning Tautan et al. on {len(non_excluded)} segments...")
    t0 = time.time()
    n_ok, n_fail = 0, 0

    for i, seg in enumerate(non_excluded):
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(non_excluded)}] ({time.time()-t0:.0f}s)...")

        mat_file = seg['mat_file']
        mono = load_mono(mat_file)
        if mono is None:
            n_fail += 1
            continue

        try:
            result = pd_detect_alternate(mono.copy(), FS, pk_detect='apd')
            if isinstance(result, dict) and result.get('spatial_extent') is not None:
                seg['tautan_spatial_extent'] = float(result['spatial_extent'])
            n_ok += 1
        except Exception:
            n_fail += 1

    print(f"  Tautan done: {n_ok} OK, {n_fail} failed ({time.time()-t0:.0f}s)")


def run_rda_inference(segments, predictions):
    """Run W05 + RDA-PLV on LRDA/GRDA segments."""
    from rda_spatial_extent import rda_spatial_extent

    rda_segs = [s for s in segments
                 if s.get('subtype', '').lower() in ('lrda', 'grda')
                 and s.get('excluded', '').lower() not in ('true', '1', 'yes')]

    print(f"\nRunning RDA-PLV on {len(rda_segs)} RDA segments...")
    t0 = time.time()
    n_ok, n_fail = 0, 0

    for i, seg in enumerate(rda_segs):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rda_segs)}] ({time.time()-t0:.0f}s)...")

        mat_file = seg['mat_file']
        mono = load_mono(mat_file)
        if mono is None:
            n_fail += 1
            continue

        try:
            bip = mono_to_bipolar(mono)
            freq = float(seg.get('pdchar_freq_hz') or seg.get('algo_freq_hz') or 1.0)
            if freq <= 0 or not np.isfinite(freq):
                freq = 1.0

            result = rda_spatial_extent(bip, freq)

            seg['rda_plv_spatial_extent'] = float(result.get('spatial_extent', 0))

            # Per-channel scores → predictions.json
            predictions[mat_file] = predictions.get(mat_file, {})
            predictions[mat_file]['rda_channel_scores'] = result['channel_scores'].tolist() \
                if hasattr(result['channel_scores'], 'tolist') else list(result['channel_scores'])

            n_ok += 1
        except Exception:
            n_fail += 1

    print(f"  RDA done: {n_ok} OK, {n_fail} failed ({time.time()-t0:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description='Run all model inference')
    parser.add_argument('--pd-only', action='store_true', help='Only run PD inference')
    parser.add_argument('--rda-only', action='store_true', help='Only run RDA inference')
    parser.add_argument('--tautan-only', action='store_true', help='Only run Tautan inference')
    args = parser.parse_args()

    run_all = not (args.pd_only or args.rda_only or args.tautan_only)

    print("=" * 60)
    print("Running Model Inference")
    print("=" * 60)

    segments = load_segments()
    fieldnames = list(segments[0].keys())

    # Add new columns if not present
    for col in ['pdchar_laterality', 'pdchar_spatial_extent',
                'tautan_spatial_extent', 'rda_plv_spatial_extent']:
        if col not in fieldnames:
            fieldnames.append(col)
        for seg in segments:
            if col not in seg:
                seg[col] = ''

    # Load existing predictions
    if PREDICTIONS_JSON.exists():
        with open(PREDICTIONS_JSON) as f:
            predictions = json.load(f)
        print(f"Loaded {len(predictions)} existing predictions")
    else:
        predictions = {}

    if run_all or args.pd_only:
        run_pd_inference(segments, predictions)

    if run_all or args.tautan_only:
        run_tautan_inference(segments, predictions)

    if run_all or args.rda_only:
        run_rda_inference(segments, predictions)

    # Save
    print(f"\nSaving segments.csv ({len(segments)} rows, {len(fieldnames)} columns)...")
    save_segments(segments, fieldnames)

    print(f"Saving predictions.json ({len(predictions)} entries)...")
    with open(PREDICTIONS_JSON, 'w') as f:
        json.dump(predictions, f)

    print("\nDone!")


if __name__ == '__main__':
    main()
