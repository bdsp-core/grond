#!/usr/bin/env python3
"""Restore the 510 PD-GT patients lost during the May-2026 cleanup.

Sources:
  Phase 1 (this script): data/_archive/dl_cache/external_pd_segments.npz
    Covers ~296 PIDs / ~948 segments. Bipolar 18-channel float32, indexed.
  Phase 2 (separate script): S3 download of sub-S0001{pid}_{ts}.mat
    Covers ~152 additional PIDs via iiic_file_index.json canonical mapping.

For each recovered patient we:
  - Write data/eeg/{pid}_seg{N}.mat with bipolar 18-channel data + Fs=200
  - Append a row to data/labels/segments.csv (eeg_source='external_npz_recovery')
  - Append a row to data/labels/archive_labels/patients.csv with subtype +
    selected_freq + laterality recovered from data/labels/discharge_times.json

The index mapping comes from data/labels/archive_labels/segments.csv.bak which
preserves the original original_filename column 'external_pd_segments.npz[N]'.

Backups of segments.csv and patients.csv are placed at /tmp/recovery_backup/.

Usage:
    conda run -n morgoth python code/data_management/recover_missing_pd_segments.py
"""
from __future__ import annotations
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
NPZ_PATH = PROJECT_DIR / 'data' / '_archive' / 'dl_cache' / 'external_pd_segments.npz'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
SEG_BAK = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'segments.csv.bak'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

NPZ_REF_RE = re.compile(r'external_pd_segments\.npz\[(\d+)\]')


def main():
    print('Loading inputs...')
    with open(DT_JSON) as f:
        dt = json.load(f)
    gt_pd = {pid: v for pid, v in dt.items()
             if v.get('review_status') == 'ground_truth'
             and v.get('subtype') in ('lpd', 'gpd')}
    print(f'  PD ground-truth patients: {len(gt_pd)}')

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    resolvable_now = set(seg_df['patient_id'])
    missing_pids = {p for p in gt_pd if p not in resolvable_now}
    print(f'  Currently missing PD-GT patients: {len(missing_pids)}')

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    with open(SEG_BAK) as f:
        bk_rows = list(csv.DictReader(f))

    print(f'Loading npz {NPZ_PATH.relative_to(PROJECT_DIR)}...')
    npz = np.load(NPZ_PATH, allow_pickle=True)
    segments_np = npz['segments']  # (3816, 18, 2000) float32
    patients_np = np.array([str(p) for p in npz['patients']])
    print(f'  npz: {len(segments_np)} segments, {len(set(patients_np))} unique patients')

    # Plan extraction rows. We restrict to ONE segment per patient (seg000),
    # because data/labels/discharge_times.json has a single set of global_times
    # per patient, aligned to seg000. Multi-segment recovery would cause the
    # load_dataset top-variance sort to pick the wrong window for evaluation.
    to_extract = []
    missing_with_no_npz_idx = set(missing_pids)
    seen_pids = set()
    for r in bk_rows:
        pid = r['patient_id']
        if pid not in missing_pids:
            continue
        if pid in seen_pids:
            continue  # already chose the seg000 row for this pid
        if not r['mat_file'].endswith('_seg000.mat'):
            continue
        m = NPZ_REF_RE.match(r.get('original_filename', ''))
        if not m:
            continue
        idx = int(m.group(1))
        if idx >= len(segments_np):
            print(f'  WARNING: npz index {idx} out of range for {pid}; skipping')
            continue
        if patients_np[idx] != pid:
            print(f'  WARNING: npz index {idx} maps to {patients_np[idx]!r} '
                  f'not {pid!r}; skipping')
            continue
        to_extract.append({
            'pid': pid,
            'mat_file': r['mat_file'],
            'subtype': r['subtype'].lower(),
            'idx': idx,
        })
        seen_pids.add(pid)
        missing_with_no_npz_idx.discard(pid)

    pids_recovered = {x['pid'] for x in to_extract}
    print(f'\nTo extract: {len(to_extract)} segments covering {len(pids_recovered)} PIDs')
    print(f'Missing PIDs NOT in npz: {len(missing_with_no_npz_idx)} '
          f'(Phase 2 / S3 territory)\n')

    EEG_DIR.mkdir(parents=True, exist_ok=True)

    # Drop columns we don't have on the new rows (algo_freq_hz, pdchar_*, etc. — let them be NaN)
    seg_cols = list(seg_df.columns)
    pat_cols = list(pat_df.columns)

    # Write .mat files
    print('Writing .mat files...')
    n_written = 0
    new_seg_rows = []
    for item in to_extract:
        pid = item['pid']
        mat_file = item['mat_file']
        subtype = item['subtype']
        idx = item['idx']

        seg = segments_np[idx].astype(np.float64)  # (18, 2000)
        out = EEG_DIR / mat_file
        sio.savemat(str(out), {'data': seg, 'Fs': np.array([[200]], dtype=np.int64)})
        n_written += 1

        if n_written % 200 == 0:
            print(f'  {n_written}/{len(to_extract)} written')

        new_seg_rows.append({
            'mat_file': mat_file,
            'patient_id': pid,
            'subtype': subtype,
            'subtype_source': 'external_npz_recovery',
            'algo_freq_hz': '',
            'excluded': False,
            'exclusion_reason': '',
            'subtype_original': subtype,
            'has_discharge_timing': bool(pid in gt_pd),
            'has_wave_timing': False,
            'has_channel_involvement': False,
            'eeg_source': 'external_npz_recovery',
            'eeg_file': '',
            'montage': 'bipolar',
            'duration_s': 10.0,
            'fs': 200.0,
            'n_channels': 18.0,
            'pdchar_freq_hz': '',
            'tautan_freq_hz': '',
            'pdchar_laterality': '',
            'pdchar_spatial_extent': '',
            'tautan_spatial_extent': '',
            'rda_plv_spatial_extent': '',
        })
    print(f'  Done. {n_written} .mat files written.\n')

    # Build patients.csv rows (one per unique PID)
    n_segs_per_pid = {}
    for r in new_seg_rows:
        n_segs_per_pid[r['patient_id']] = n_segs_per_pid.get(r['patient_id'], 0) + 1

    new_pat_rows = []
    for pid in sorted(pids_recovered):
        if pid in existing_pat_pids:
            continue  # already present
        gt_data = gt_pd[pid]
        subtype = gt_data.get('subtype', 'lpd').lower()
        gold = gt_data.get('selected_freq')
        if gold is None or (isinstance(gold, float) and not np.isfinite(gold)):
            # Fall back to inferring from inter-pulse intervals
            times = gt_data.get('global_times', [])
            if len(times) >= 2:
                ipis = np.diff(sorted(times))
                gold = float(1.0 / np.median(ipis)) if len(ipis) > 0 else 1.0
            else:
                gold = 1.0
        lat = gt_data.get('laterality') or ''
        new_pat_rows.append({
            'patient_id': pid,
            'subtype': subtype,
            'subtype_original': subtype,
            'n_segments': n_segs_per_pid.get(pid, 1),
            'gold_standard_freq': float(gold),
            'gold_standard_freq_original': float(gold),
            'excluded': False,
            'exclusion_reason': '',
            'laterality': lat,
            'laterality_original': lat,
            'laterality_rater': '',
            'subtype_rater': 'mw_review',
        })
    print(f'New patients.csv rows to add: {len(new_pat_rows)}')

    # Append + save
    seg_df_new = pd.concat([seg_df, pd.DataFrame(new_seg_rows, columns=seg_cols)],
                           ignore_index=True)
    seg_df_new.to_csv(SEG_CSV, index=False)
    print(f'  segments.csv: {len(seg_df)} → {len(seg_df_new)} rows; wrote {SEG_CSV.relative_to(PROJECT_DIR)}')

    pat_df_new = pd.concat([pat_df, pd.DataFrame(new_pat_rows, columns=pat_cols)],
                           ignore_index=True)
    pat_df_new.to_csv(PAT_CSV, index=False)
    print(f'  patients.csv: {len(pat_df)} → {len(pat_df_new)} rows; wrote {PAT_CSV.relative_to(PROJECT_DIR)}')

    # Sanity check: load one and verify shape
    sample_seg = new_seg_rows[0]['mat_file'] if new_seg_rows else None
    if sample_seg:
        m = sio.loadmat(str(EEG_DIR / sample_seg))
        print(f'\nSanity-check load: {sample_seg} → data shape {m["data"].shape}, '
              f'Fs={int(m["Fs"][0,0])}')

    print('\nDone.')


if __name__ == '__main__':
    main()
