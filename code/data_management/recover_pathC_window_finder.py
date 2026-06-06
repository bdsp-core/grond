#!/usr/bin/env python3
"""Path (C): re-extract missing 10-second EEG segments from S3 morgoth1.

After Steps 1 + 2 (Alex + IRR), some PD-GT patients still have
discharge_times.json annotations but no resolvable EEG. Their multi-recording
source data lives in
   s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/{LPD,GPD,IIIC}/segments_raw/sub-S0001{pid}_{ts}.mat
Each S3 object is a v7.3 .mat file containing a 10-minute 20-channel monopolar
EEG recording (data shape (120000, 20)) plus event_time markers.

For each remaining missing PID:
  1. Look up candidate recordings (preferring the subtype-matching subdir).
  2. Download one at a time.
  3. Convert to bipolar; slide a 10-second window (stride 1 s) across the
     recording; for each candidate window, compute alignment score against
     the GT discharge times (energy at each window-relative GT time).
  4. Accept the first window whose alignment score exceeds 2× the median
     window score; otherwise try the next recording. If we exhaust all
     candidates, save the best window we found regardless and flag low
     confidence in the provenance log.

Saves successful extractions to data/eeg/{pid}_seg000.mat and registers them
in segments.csv + patients.csv. Provenance log written to
results/c1_repro/pathC_provenance.csv with source-file + window-start + score.

Pre-conditions:
  - AWS credentials available in env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.
  - /tmp/morgoth1_keys/pid_to_recordings.json (output of an earlier indexing
    pass listing candidate S3 keys per pid in preferred order).

Usage:
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    conda run -n morgoth python code/data_management/recover_pathC_window_finder.py
"""
from __future__ import annotations
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import h5py

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
KEY_INDEX = Path('/tmp/morgoth1_keys/pid_to_recordings.json')
S3_BUCKET = 'bdsp-opendata-credentialed'

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1','F7'), ('F7','T3'), ('T3','T5'), ('T5','O1'),
    ('Fp2','F8'), ('F8','T4'), ('T4','T6'), ('T6','O2'),
    ('Fp1','F3'), ('F3','C3'), ('C3','P3'), ('P3','O1'),
    ('Fp2','F4'), ('F4','C4'), ('C4','P4'), ('P4','O2'),
    ('Fz','Cz'), ('Cz','Pz'),
]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])

FS = 200
WIN_SAMPLES = 2000
ACCEPT_THRESHOLD = 2.0  # accept window if its score is >= 2x median


def load_morgoth1_mat(path):
    """Load a morgoth1/segments_raw .mat file. Returns (data, channels, fs).

    Tries v7.3 (h5py) first, falls back to v5 (scipy.io).
    Data is reshaped to (19, N) bipolar-source monopolar (drops EKG).
    """
    try:
        with h5py.File(path, 'r') as f:
            data = np.array(f['data']).astype(np.float64)
            # h5py reads data with axes transposed - check shapes
            # Source is (120000, 20) in matlab, but h5py reads as (20, 120000)
            if data.shape[0] == 20 and data.shape[1] > 1000:
                pass  # already (20, N)
            elif data.shape[1] == 20 and data.shape[0] > 1000:
                data = data.T
            else:
                return None
            fs = float(f['Fs'][0, 0]) if 'Fs' in f else FS
    except (OSError, KeyError):
        # v5 fallback
        try:
            m = sio.loadmat(str(path))
            data = m.get('data')
            if data is None: return None
            data = data.astype(np.float64)
            if data.shape[0] == 20:
                pass
            elif data.shape[1] == 20:
                data = data.T
            else:
                return None
            fs = float(m['Fs'][0, 0]) if 'Fs' in m else FS
        except Exception:
            return None

    if int(fs) != FS:
        # Skip non-200-Hz recordings
        return None
    # Drop EKG (channel 20)
    mono = data[:19]
    return mono


def to_bipolar(mono):
    return mono[BIPOLAR_INDICES[:, 0]] - mono[BIPOLAR_INDICES[:, 1]]


def score_windows(bipolar, gt_times):
    """Slide 10s window with 1s stride; compute alignment score per window.

    Score = sum over GT times of channel-summed-|signal| at (window_start + gt).
    Returns: (best_start, best_score, normalized_score, distribution_stats).
    """
    n_ch, n = bipolar.shape
    if n < WIN_SAMPLES:
        return None
    energy = np.sum(np.abs(bipolar), axis=0)
    gt_samples = np.array([int(round(t * FS)) for t in gt_times])
    stride = FS
    starts = np.arange(0, n - WIN_SAMPLES + 1, stride)
    if len(starts) == 0:
        return None
    scores = np.zeros(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        score = 0.0
        for gs in gt_samples:
            idx = s + gs
            if idx < 0 or idx >= n:
                continue
            lo = max(0, idx - 10)
            hi = min(n, idx + 11)
            score += float(energy[lo:hi].mean())
        scores[i] = score
    best_i = int(np.argmax(scores))
    return {
        'start': int(starts[best_i]),
        'score': float(scores[best_i]),
        'median': float(np.median(scores)),
        'norm': float(scores[best_i] / max(np.median(scores), 1e-9)),
        'max_score': float(scores.max()),
    }


def aws_download(s3_key, local_path):
    s3_uri = f's3://{S3_BUCKET}/{s3_key}'
    try:
        r = subprocess.run(['aws', 's3', 'cp', s3_uri, str(local_path),
                            '--no-progress', '--quiet'],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main():
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        print('ERROR: AWS_ACCESS_KEY_ID not set.')
        sys.exit(1)
    if not KEY_INDEX.exists():
        print(f'ERROR: {KEY_INDEX} not found. Run S3 indexing first.')
        sys.exit(1)

    print('Loading inputs...')
    with open(DT_JSON) as f:
        dt = json.load(f)
    gt_pd = {pid: v for pid, v in dt.items()
             if v.get('review_status') == 'ground_truth'
             and v.get('subtype') in ('lpd', 'gpd')
             and isinstance(v.get('global_times'), list)
             and len(v.get('global_times')) >= 2}

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    resolvable = set(seg_df['patient_id'])
    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    with open(KEY_INDEX) as f:
        pid_to_keys = json.load(f)

    missing = sorted({p for p in gt_pd if p not in resolvable})
    candidates = [p for p in missing if p in pid_to_keys]
    no_source = [p for p in missing if p not in pid_to_keys]
    print(f'  Still-missing PD-GT after Steps 1+2: {len(missing)}')
    print(f'  With S3 morgoth1 candidate recordings: {len(candidates)}')
    print(f'  Without any S3 source:                 {len(no_source)}')

    EEG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path('/tmp/morgoth1_dl')
    tmp_dir.mkdir(parents=True, exist_ok=True)

    seg_cols = list(seg_df.columns)
    pat_cols = list(pat_df.columns)
    new_seg_rows = []
    new_pat_rows = []
    provenance = []

    n_high = 0
    n_low = 0
    n_fail = 0
    t0 = time.time()
    for i, pid in enumerate(candidates):
        gt_times = gt_pd[pid]['global_times']
        keys = pid_to_keys[pid]

        best_overall = None
        for s3_key in keys:
            tmp_path = tmp_dir / Path(s3_key).name
            if not aws_download(s3_key, tmp_path):
                continue
            try:
                mono = load_morgoth1_mat(tmp_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                continue
            if mono is None or mono.shape[1] < WIN_SAMPLES:
                tmp_path.unlink(missing_ok=True)
                continue
            bip = to_bipolar(mono)
            scored = score_windows(bip, gt_times)
            if scored is None:
                tmp_path.unlink(missing_ok=True)
                continue
            scored['key'] = s3_key
            scored['bipolar_slice'] = bip[:, scored['start']:scored['start'] + WIN_SAMPLES]
            if best_overall is None or scored['score'] > best_overall['score']:
                best_overall = scored
            # Accept early if confident
            if scored['norm'] >= ACCEPT_THRESHOLD:
                tmp_path.unlink(missing_ok=True)
                break
            tmp_path.unlink(missing_ok=True)

        if best_overall is None:
            n_fail += 1
            provenance.append({
                'pid': pid, 'status': 'fail_no_window',
                'source_file': '', 'window_start_s': '',
                'alignment_score': '', 'norm_score_vs_median': '',
            })
            continue

        # Save the extracted slice
        out_mat = f'{pid}_seg000.mat'
        out_path = EEG_DIR / out_mat
        sio.savemat(str(out_path), {'data': best_overall['bipolar_slice'].astype(np.float64),
                                    'Fs': np.array([[FS]], dtype=np.int64)})

        is_high = best_overall['norm'] >= ACCEPT_THRESHOLD
        if is_high:
            n_high += 1
        else:
            n_low += 1
        provenance.append({
            'pid': pid,
            'status': 'ok_high' if is_high else 'ok_low_confidence',
            'source_file': best_overall['key'],
            'window_start_s': round(best_overall['start'] / FS, 3),
            'alignment_score': round(best_overall['score'], 1),
            'norm_score_vs_median': round(best_overall['norm'], 2),
        })

        subtype = gt_pd[pid]['subtype']
        new_seg_rows.append({
            'mat_file': out_mat,
            'patient_id': pid,
            'subtype': subtype,
            'subtype_source': 'pathC_window_finder',
            'algo_freq_hz': '',
            'excluded': False,
            'exclusion_reason': '',
            'subtype_original': subtype,
            'has_discharge_timing': True,
            'has_wave_timing': False,
            'has_channel_involvement': False,
            'eeg_source': 'pathC_morgoth1_s3',
            'eeg_file': best_overall['key'],
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
                ipis = np.diff(sorted(gt_times))
                gold = float(1.0 / np.median(ipis)) if len(ipis) > 0 else 1.0
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

        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(candidates) - i - 1) / max(rate, 1e-6)
        if (i + 1) % 5 == 0 or (i + 1) == len(candidates):
            print(f'  [{i+1}/{len(candidates)}]  high={n_high}  low={n_low}  '
                  f'fail={n_fail}  rate={rate:.2f}/s  ETA={eta/60:.1f} min')

    print(f'\nResult: ok={n_high + n_low} (high-conf {n_high}, low-conf {n_low}), '
          f'fail={n_fail}, no_source={len(no_source)}')

    # Provenance log
    prov_path = PROJECT_DIR / 'results' / 'c1_repro' / 'pathC_provenance.csv'
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pid', 'status', 'source_file',
                                          'window_start_s', 'alignment_score',
                                          'norm_score_vs_median'],
                           lineterminator='\n')
        w.writeheader()
        w.writerows(provenance)
    print(f'Provenance log saved: {prov_path}')

    # Append + save CSVs
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
