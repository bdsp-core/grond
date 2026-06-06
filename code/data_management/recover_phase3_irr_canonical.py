#!/usr/bin/env python3
"""Phase 3 recovery: use IRR manifest mat_file as the canonical filename.

The paper_materials/independent_expert_tasks/{lpd,gpd,lrda,grda}/manifest.csv
files record the EXACT 10-second segment mat files used in the canonical IRR
evaluation. These are the authoritative source for "what file did the user
label." We prefer the IRR manifest over harvest_manifest because the latter
records the harvest pipeline's original candidate, which may differ from the
file the rater ultimately annotated.

This script:
  - Walks the four IRR manifests, mapping pid -> mat_file.
  - For each still-missing PD-GT patient that appears in any IRR manifest,
    downloads the canonical mat_file from
    s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/.
  - Saves to data/eeg/{pid}_seg000.mat in the same format as existing
    external_drive entries.
  - Appends rows to segments.csv + patients.csv.

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    conda run -n morgoth python code/data_management/recover_phase3_irr_canonical.py
"""
from __future__ import annotations
import csv
import json
import os
import re
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
IRR_ROOT = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'

S3_BUCKET = 'bdsp-opendata-credentialed'
S3_PREFIX = 'iiic-freq3/data/eeg/'

PID_RE = re.compile(r'sub-S0001(\d+)_')


def build_irr_pid_to_mat():
    """Read the four IRR manifests; return pid -> set of mat_file."""
    out = {}
    for sub in ('lpd', 'gpd', 'lrda', 'grda'):
        manifest = IRR_ROOT / sub / 'manifest.csv'
        if not manifest.exists():
            continue
        df = pd.read_csv(manifest)
        for _, row in df.iterrows():
            mat_file = row['mat_file']
            m = PID_RE.match(str(mat_file))
            if m:
                pid = m.group(1)
                out.setdefault(pid, set()).add(mat_file)
    return out


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
    print(f'  Still-missing PD-GT (post earlier phases): {len(missing)}')

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    irr_map = build_irr_pid_to_mat()
    print(f'  IRR PID->mat_file map: {len(irr_map)} PIDs')

    plan = []
    for pid in missing:
        files = irr_map.get(pid)
        if not files:
            continue
        # Prefer the file from the matching subtype manifest if possible
        # (multiple files in the same manifest are rare but pick first)
        plan.append({'pid': pid, 's3_file': sorted(files)[0]})
    print(f'  Plan: {len(plan)} IRR-canonical downloads')

    if not plan:
        print('Nothing to do.')
        return

    EEG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path('/tmp/s3_recovery_phase3')
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
        s3_uri = f's3://{S3_BUCKET}/{S3_PREFIX}{s3_file}'
        tmp_path = tmp_dir / s3_file
        out_mat = f'{pid}_seg000.mat'
        out_path = EEG_DIR / out_mat

        try:
            r = subprocess.run(['aws', 's3', 'cp', s3_uri, str(tmp_path),
                                '--no-progress'],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                if i < 3:
                    print(f'  [{i+1}/{len(plan)}] {pid}: 404 - {s3_file}')
                n_fail += 1
                continue
        except subprocess.TimeoutExpired:
            print(f'  [{i+1}/{len(plan)}] {pid}: TIMEOUT')
            n_fail += 1
            continue

        try:
            m = sio.loadmat(str(tmp_path))
            keys = [k for k in m if not k.startswith('_')]
            data_key = 'data' if 'data' in keys else keys[0]
            data = m[data_key]
            if data.shape[0] > data.shape[1]:
                data = data.T
            if data.shape[1] != 2000:
                print(f'  [{i+1}/{len(plan)}] {pid}: shape {data.shape}; skip')
                n_fail += 1
                continue
            sio.savemat(str(out_path), {'data': data.astype(np.float64),
                                        'Fs': np.array([[200]], dtype=np.int64)})
            n_ok += 1
        except Exception as e:
            print(f'  [{i+1}/{len(plan)}] {pid}: save failed: {e}')
            n_fail += 1
            continue

        if (i + 1) % 25 == 0:
            print(f'  [{i+1}/{len(plan)}] ok={n_ok} fail={n_fail}')

        subtype = gt_pd[pid]['subtype']
        new_seg_rows.append({
            'mat_file': out_mat,
            'patient_id': pid,
            'subtype': subtype,
            'subtype_source': 's3_recovery_irr_canonical',
            'algo_freq_hz': '',
            'excluded': False,
            'exclusion_reason': '',
            'subtype_original': subtype,
            'has_discharge_timing': True,
            'has_wave_timing': False,
            'has_channel_involvement': False,
            'eeg_source': 's3_iiic_freq3_recovery',
            'eeg_file': s3_file,
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

    print(f'\nResult: ok={n_ok}, fail={n_fail}')
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
