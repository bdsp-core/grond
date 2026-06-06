#!/usr/bin/env python3
"""Recover missing PD-GT EEG segments from IIIC-AlexandraData/data/eeg/.

Alex's directory has 2,241 _seg000.mat files in (18, 2000) float32 bipolar
format. Spotlight identified 297 of our 501 missing PD-GT patients as having
files there. Copies them into data/eeg/ + registers them in
segments.csv and patients.csv.

Notes on format: the files are 18-channel bipolar float32 with HF content
attenuated above ~20 Hz (PSD analysis). However, controlled experiment
shows the HemiCET-UNet pipeline produces essentially identical inference
on filtered vs unfiltered data (CNN PD prob, freq, evidence peaks all
match within rounding), so the format mismatch should not break F1.

Usage:
    conda run -n morgoth python code/data_management/recover_from_alex_data.py
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

ALEX_DIR = Path('/Users/mwestover/GithubRepos/IIIC-AlexandraData/data/eeg')


def main():
    if not ALEX_DIR.exists():
        print(f'ERROR: {ALEX_DIR} does not exist.')
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
    print(f'  Still-missing PD-GT patients: {len(missing)}')

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    # Find Alex's seg000 files matching missing PIDs
    plan = []
    for pid in missing:
        src = ALEX_DIR / f'{pid}_seg000.mat'
        if src.exists():
            plan.append(pid)
    print(f'  Found in Alex dir: {len(plan)}')

    if not plan:
        print('Nothing to copy.')
        return

    EEG_DIR.mkdir(parents=True, exist_ok=True)

    seg_cols = list(seg_df.columns)
    pat_cols = list(pat_df.columns)
    new_seg_rows = []
    new_pat_rows = []

    for i, pid in enumerate(plan):
        src = ALEX_DIR / f'{pid}_seg000.mat'
        dst = EEG_DIR / f'{pid}_seg000.mat'
        shutil.copy2(str(src), str(dst))
        if (i + 1) % 50 == 0:
            print(f'  copied {i + 1}/{len(plan)}')

        subtype = gt_pd[pid]['subtype']
        new_seg_rows.append({
            'mat_file': f'{pid}_seg000.mat',
            'patient_id': pid,
            'subtype': subtype,
            'subtype_source': 'alex_recovery',
            'algo_freq_hz': '',
            'excluded': False,
            'exclusion_reason': '',
            'subtype_original': subtype,
            'has_discharge_timing': True,
            'has_wave_timing': False,
            'has_channel_involvement': False,
            'eeg_source': 'alex_recovery',
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

    print(f'\nDone copying. {len(new_seg_rows)} files added.')
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
