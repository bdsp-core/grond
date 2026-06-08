#!/usr/bin/env python3
"""Extract unfiltered 10s EEG slices from the bank for high-NCC matches.

For each query in bank_match_results.csv with NCC >= threshold:
  - Open the matched bank entry's data30s (20-ch monopolar @ 200 Hz × 30s)
  - Slice the 10s window starting at bank_window_start_samples
  - Drop EKG → 19-ch monopolar → 18-ch bipolar via standard banana
  - Save as data/eeg/{pid}_seg000.mat (overwrites the filtered version)
  - Update segments.csv: eeg_source = 'eeg_bank_unfiltered_recovery'
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
MATCH_CSV = PROJECT_DIR / 'results' / 'c1_repro' / 'bank_match_results.csv'
H5_PATH = '/Volumes/Extreme SSD/.eeg_cache/eeg-test/eeg_bank_spec.h5'

NCC_THRESHOLD = 0.9  # only accept high-confidence matches
FS = 200
WIN_SAMPLES = 2000

MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
        'Fp2','F4','C4','P4','F8','T4','T6','O2']
PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
         ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
         ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
         ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
         ('Fz','Cz'),('Cz','Pz')]
BIP_IDX = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])


def main():
    matches = pd.read_csv(MATCH_CSV)
    matches['pid'] = matches['pid'].astype(str)
    matches['bank_seg_id'] = matches['bank_seg_id'].astype(str)
    print(f'Loaded {len(matches)} match results', flush=True)

    high_conf = matches[matches['best_ncc'] >= NCC_THRESHOLD].copy()
    print(f'  {len(high_conf)} above NCC threshold {NCC_THRESHOLD}', flush=True)
    print(f'  By source:', flush=True)
    for src, grp in high_conf.groupby('source'):
        print(f'    {src}: {len(grp)}', flush=True)

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    print(f'\nOpening {H5_PATH}...', flush=True)
    n_ok = 0
    n_fail = 0
    t0 = time.time()
    updated_rows = []  # (mat_file, new eeg_source values)

    with h5py.File(H5_PATH, 'r') as f:
        seg = f['segments']
        for i, row in enumerate(high_conf.itertuples(index=False)):
            try:
                bank_obj = seg[row.bank_seg_id]
                if 'data30s' not in bank_obj:
                    n_fail += 1
                    continue
                data30s = np.array(bank_obj['data30s'])  # (20, 6000)
                if data30s.shape != (20, 6000):
                    n_fail += 1
                    continue
                if not np.isfinite(data30s).all():
                    n_fail += 1
                    continue
                mono = data30s[:19].astype(np.float64)  # drop EKG
                bip = mono[BIP_IDX[:, 0]] - mono[BIP_IDX[:, 1]]
                start = int(row.bank_window_start_samples)
                end = start + WIN_SAMPLES
                if end > bip.shape[1]:
                    n_fail += 1
                    continue
                window = bip[:, start:end]
                # Save (overwrites filtered version)
                out_path = EEG_DIR / row.mat_file
                sio.savemat(str(out_path),
                            {'data': window.astype(np.float64),
                             'Fs': np.array([[FS]], dtype=np.int64)})
                n_ok += 1
                updated_rows.append(row.mat_file)
            except Exception as e:
                n_fail += 1
                continue

            if (i + 1) % 100 == 0:
                print(f'  [{i+1}/{len(high_conf)}] {time.time()-t0:.0f}s, ok={n_ok}, fail={n_fail}',
                      flush=True)

    print(f'\nExtracted {n_ok} / {len(high_conf)} (fail={n_fail})', flush=True)

    # Update segments.csv: change eeg_source to 'eeg_bank_unfiltered_recovery' for extracted patients
    print(f'\nUpdating segments.csv...', flush=True)
    mat_set = set(updated_rows)
    seg_df.loc[seg_df['mat_file'].isin(mat_set), 'eeg_source'] = 'eeg_bank_unfiltered_recovery'
    seg_df.to_csv(SEG_CSV, index=False)
    print(f'  Updated {len(mat_set)} rows in segments.csv', flush=True)

    # Save audit
    audit = high_conf[['pid', 'mat_file', 'source', 'bank_seg_id',
                       'bank_window_start_s', 'best_ncc']].copy()
    audit['extracted'] = audit['mat_file'].isin(mat_set)
    audit_out = PROJECT_DIR / 'results' / 'c1_repro' / 'bank_extracted_audit.csv'
    audit.to_csv(audit_out, index=False)
    print(f'  Audit: {audit_out.relative_to(PROJECT_DIR)}', flush=True)


if __name__ == '__main__':
    main()
