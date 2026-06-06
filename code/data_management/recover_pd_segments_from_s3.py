#!/usr/bin/env python3
"""Phase 2 recovery: download S3 segments using the original harvest_manifest pointers.

After Phase 1 (`recover_missing_pd_segments.py`), there are still ~205 PD-GT
patients with no resolvable EEG. We use the labeled record of which exact
sub-S0001 file we originally annotated, preserved in the four harvest manifests
under `data/labels/archive_labels/`:
  - harvest_manifest.json           (general PD)
  - bipd_harvest_manifest.json
  - other_harvest_manifest.json
  - rda_harvest_manifest.json

Each entry's `s3_file` field names the canonical sub-S0001 .mat file that the
annotator viewed when creating the discharge_times.json entry; that file lives
at s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/.

We download, save as data/eeg/{pid}_seg000.mat in the same format as the
existing external_drive entries ((19, 2000) monopolar; key 'data', 'Fs'=200),
and append rows to segments.csv + archive_labels/patients.csv.

Pre-conditions:
  - AWS credentials in env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    conda run -n morgoth python code/data_management/recover_pd_segments_from_s3.py
"""
from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

ARCHIVE = PROJECT_DIR / 'data' / 'labels' / 'archive_labels'
MANIFESTS = [
    ('harvest_manifest.json', 'pid_keyed'),
    ('bipd_harvest_manifest.json', 'segid_keyed'),
    ('other_harvest_manifest.json', 'segid_keyed'),
    ('rda_harvest_manifest.json', 'pid_keyed'),
]

S3_BUCKET = 'bdsp-opendata-credentialed'
S3_PREFIX = 'iiic-freq3/data/eeg/'


def build_pid_to_s3_map():
    """Aggregate pid -> set of s3 filenames from all harvest manifests."""
    pid_to_files = {}
    for name, mode in MANIFESTS:
        with open(ARCHIVE / name) as f:
            d = json.load(f)
        for k, v in d.items():
            if not isinstance(v, dict):
                continue
            if mode == 'pid_keyed':
                pid = str(k)
            else:  # segid_keyed
                pid = str(v.get('patient_id') or k.split('_')[0])
            s3f = v.get('s3_file', '')
            if s3f:
                pid_to_files.setdefault(pid, set()).add(s3f)
    return pid_to_files


def main():
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        print('ERROR: AWS_ACCESS_KEY_ID not set.')
        sys.exit(1)

    print('Loading inputs...')
    with open(DT_JSON) as f:
        dt = json.load(f)
    gt_pd = {pid: v for pid, v in dt.items()
             if v.get('review_status') == 'ground_truth'
             and v.get('subtype') in ('lpd', 'gpd')}

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    resolvable = set(seg_df['patient_id'])
    missing = sorted({p for p in gt_pd if p not in resolvable})
    print(f'  Still-missing PD-GT patients (post-Phase 1): {len(missing)}')

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    print('Aggregating harvest manifests...')
    pid_to_files = build_pid_to_s3_map()
    print(f'  Total harvest-tracked PIDs: {len(pid_to_files)}')

    # Plan downloads — unambiguous single-file pids only
    plan = []
    ambiguous = []
    no_file = []
    for pid in missing:
        files = pid_to_files.get(pid)
        if not files:
            no_file.append(pid)
            continue
        if len(files) > 1:
            ambiguous.append((pid, files))
            continue
        plan.append({'pid': pid, 's3_file': next(iter(files))})

    print(f'  Plan: {len(plan)} downloads (unambiguous single-file pids)')
    print(f'  Skipped (no harvest entry): {len(no_file)}')
    print(f'  Skipped (multiple candidates): {len(ambiguous)}')

    if not plan:
        print('Nothing to do.')
        return

    EEG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path('/tmp/s3_recovery_phase2')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    new_seg_rows = []
    new_pat_rows = []
    seg_cols = list(seg_df.columns)
    pat_cols = list(pat_df.columns)

    n_ok = 0
    n_fail = 0
    for i, item in enumerate(plan):
        pid = item['pid']
        s3_file = item['s3_file']
        # Some s3_file values are bare filenames, some are full s3 paths
        if s3_file.startswith('s3://'):
            s3_uri = s3_file
            src_filename = Path(s3_file).name
        else:
            s3_uri = f's3://{S3_BUCKET}/{S3_PREFIX}{s3_file}'
            src_filename = s3_file
        tmp_path = tmp_dir / src_filename
        out_mat = f'{pid}_seg000.mat'
        out_path = EEG_DIR / out_mat

        # Download
        try:
            r = subprocess.run(['aws', 's3', 'cp', s3_uri, str(tmp_path),
                                '--no-progress'],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                if (i + 1) % 25 == 0 or i < 3:
                    print(f'  [{i+1}/{len(plan)}] {pid}: aws cp FAILED '
                          f'({r.stderr.strip()[:80]})')
                n_fail += 1
                continue
        except subprocess.TimeoutExpired:
            print(f'  [{i+1}/{len(plan)}] {pid}: TIMEOUT')
            n_fail += 1
            continue

        # Copy directly with no transformation — preserve the same format as
        # existing external_drive entries (typically (19, 2000) monopolar but
        # the pipeline treats it as bipolar; F1 reproducibility on the original
        # 147 cohort confirms this works end-to-end).
        try:
            m = sio.loadmat(str(tmp_path))
            keys = [k for k in m if not k.startswith('_')]
            data_key = 'data' if 'data' in keys else keys[0]
            data = m[data_key]
            if data.shape[0] > data.shape[1]:
                data = data.T
            if data.shape[1] != 2000:
                print(f'  [{i+1}/{len(plan)}] {pid}: unexpected shape {data.shape}; skip')
                n_fail += 1
                continue
            sio.savemat(str(out_path), {'data': data.astype(np.float64),
                                        'Fs': np.array([[200]], dtype=np.int64)})
            n_ok += 1
        except Exception as e:
            print(f'  [{i+1}/{len(plan)}] {pid}: load/save FAILED: {e}')
            n_fail += 1
            continue

        if (i + 1) % 25 == 0:
            print(f'  [{i+1}/{len(plan)}] ok={n_ok} fail={n_fail}')

        subtype = gt_pd[pid]['subtype']
        new_seg_rows.append({
            'mat_file': out_mat,
            'patient_id': pid,
            'subtype': subtype,
            'subtype_source': 's3_recovery_harvest_manifest',
            'algo_freq_hz': '',
            'excluded': False,
            'exclusion_reason': '',
            'subtype_original': subtype,
            'has_discharge_timing': True,
            'has_wave_timing': False,
            'has_channel_involvement': False,
            'eeg_source': 's3_iiic_freq3_recovery',
            'eeg_file': src_filename,
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

        if pid not in existing_pat_pids:
            gt_data = gt_pd[pid]
            gold = gt_data.get('selected_freq')
            if gold is None or (isinstance(gold, float) and not np.isfinite(gold)):
                times = gt_data.get('global_times', [])
                if len(times) >= 2:
                    ipis = np.diff(sorted(times))
                    gold = float(1.0 / np.median(ipis))
                else:
                    gold = 1.0
            lat = gt_data.get('laterality') or ''
            new_pat_rows.append({
                'patient_id': pid,
                'subtype': subtype,
                'subtype_original': subtype,
                'n_segments': 1,
                'gold_standard_freq': float(gold),
                'gold_standard_freq_original': float(gold),
                'excluded': False,
                'exclusion_reason': '',
                'laterality': lat,
                'laterality_original': lat,
                'laterality_rater': '',
                'subtype_rater': 'mw_review',
            })
            existing_pat_pids.add(pid)

    print(f'\nDownload result: ok={n_ok}, fail={n_fail}')
    print(f'New segments.csv rows: {len(new_seg_rows)}')
    print(f'New patients.csv rows: {len(new_pat_rows)}')

    seg_df_new = pd.concat([seg_df, pd.DataFrame(new_seg_rows, columns=seg_cols)],
                           ignore_index=True)
    seg_df_new.to_csv(SEG_CSV, index=False)
    pat_df_new = pd.concat([pat_df, pd.DataFrame(new_pat_rows, columns=pat_cols)],
                           ignore_index=True)
    pat_df_new.to_csv(PAT_CSV, index=False)
    print(f'segments.csv: {len(seg_df)} → {len(seg_df_new)} rows')
    print(f'patients.csv: {len(pat_df)} → {len(pat_df_new)} rows')


if __name__ == '__main__':
    main()
