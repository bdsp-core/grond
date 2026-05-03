#!/usr/bin/env python3
"""
Pre-compute spatial inference results for all figure generators.

Runs PDProfiler, Tautan et al., and RDA-PLV on all segments with
spatial ground truth, caching results to JSON for instant figure generation.

This only needs to be re-run when:
  - Model weights change (retrained models)
  - New spatial ground truth labels are added
  - The spatial algorithms are modified

Usage:
    conda run -n morgoth python paper_materials/precompute_spatial_cache.py
"""

import sys
import json
import csv
import numpy as np
import scipy.io as sio
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
CACHE_PATH = SCRIPT_DIR / 'spatial_inference_cache.json'

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

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


def get_segments_with_spatial_gt():
    """Get all segments that have spatial ground truth labels."""
    segments = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mat_file = row['mat_file']
            subtype = row.get('subtype', '')
            if not subtype:
                continue
            # Check if excluded
            if row.get('excluded', '').lower() in ('true', '1', 'yes'):
                continue
            # Check for spatial annotations
            has_spatial = row.get('has_channel_involvement', '').lower() in ('true', '1', 'yes')
            if has_spatial:
                segments[mat_file] = {
                    'subtype': subtype,
                    'freq': float(row.get('pdchar_freq_hz', 0) or 0),
                }
    return segments


def main():
    print("=" * 60)
    print("Pre-computing spatial inference cache")
    print("=" * 60)

    segments = get_segments_with_spatial_gt()
    print(f"Found {len(segments)} segments with spatial GT")

    # Initialize models
    from pd_profiler import PDProfiler
    from pd_detect_alternate import pd_detect_alternate
    from rda_spatial_extent import rda_spatial_extent

    pc = PDProfiler()

    cache = {}
    n_ok, n_fail = 0, 0

    for i, (mat_file, info) in enumerate(segments.items()):
        if i % 50 == 0:
            print(f"  [{i}/{len(segments)}] {mat_file[:40]}...")

        mono = load_mono(mat_file)
        if mono is None:
            n_fail += 1
            continue

        entry = {'subtype': info['subtype']}

        try:
            # PDProfiler (PD subtypes)
            if info['subtype'] in ('lpd', 'gpd'):
                bip = mono_to_bipolar(mono)
                result = pc.characterize(bip, subtype=info['subtype'])
                entry['pdchar_channel_probs'] = result.get('channel_probs', [0]*18)
                if isinstance(entry['pdchar_channel_probs'], np.ndarray):
                    entry['pdchar_channel_probs'] = entry['pdchar_channel_probs'].tolist()
                # Threshold 0.62 (not 0.5): the per-channel CNN was trained on segment-level
                # positive labels and saturates above 0.5 even on uninvolved channels;
                # 0.62 is the threshold that yields a non-degenerate fraction-involved
                # distribution and matches what generate_fig_spatial_scatter.py and
                # generate_fig_irr.py use downstream.
                entry['pdchar_spatial_extent'] = float(np.mean(np.array(entry['pdchar_channel_probs']) > 0.62))

            # RDA-PLV (RDA subtypes)
            if info['subtype'] in ('lrda', 'grda'):
                bip = mono_to_bipolar(mono)
                freq = info['freq'] if info['freq'] > 0 else 1.0
                rda_result = rda_spatial_extent(bip, freq)
                entry['rda_channel_scores'] = rda_result['channel_scores'].tolist() if hasattr(rda_result['channel_scores'], 'tolist') else list(rda_result['channel_scores'])
                entry['rda_spatial_extent'] = float(rda_result.get('spatial_extent', 0))
                entry['rda_spatial_continuous'] = float(rda_result.get('spatial_extent_continuous', 0))

            # Tautan et al. (all subtypes)
            result_t = pd_detect_alternate(mono.copy(), FS, pk_detect='apd')
            if result_t is not None and 'spatial_extent' in result_t:
                entry['tautan_spatial_extent'] = float(result_t['spatial_extent'])
            else:
                entry['tautan_spatial_extent'] = None

            n_ok += 1
        except Exception as e:
            entry['error'] = str(e)
            n_fail += 1

        cache[mat_file] = entry

    # Save cache
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f"\nDone: {n_ok} OK, {n_fail} failed")
    print(f"Saved: {CACHE_PATH} ({len(cache)} entries)")


if __name__ == '__main__':
    main()
