#!/usr/bin/env python3
"""Restore data/eeg/ to 19-channel monopolar canonical format.

Several recovery scripts wrote 18-channel bipolar versions of segments to
data/eeg/ instead of preserving the 19-channel monopolar referential montage
that the rest of the pipeline (and discharge-locked Laplacian topography in
particular) expects. This script restores those entries to 19-channel
monopolar by re-sourcing each from its canonical location:

  - alex_recovery / iiic_unfiltered_user_reviewed / orphan w/ bipolar shape:
    re-read from /Users/mwestover/GithubRepos/IIIC-AlexandraData/data/eeg/
    (20-ch monopolar+EKG; drop EKG → 19-ch)

  - pathC_morgoth1_s3:  re-read from
    /Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/morgoth1/
    using the existing window-finder. Save as 19-channel monopolar instead
    of 18-channel bipolar.

  - sub-S0001-backed entries: re-copy from
    /Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/iiic-freq3/
    /data/eeg/  (already 19-ch monopolar; just re-copy without conversion)

Files that are already (19, 2000) are left untouched.
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

ALEX_DIR = Path('/Users/mwestover/GithubRepos/IIIC-AlexandraData/data/eeg')
SUB_S0001_CACHE = Path('/Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/iiic-freq3/data/eeg')
MORGOTH1_ROOT = Path('/Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/morgoth1/data/internal_dataset')

FS = 200
WIN_SAMPLES = 2000


def restore_from_alex(pid, mat_file):
    """Drop EKG channel (last of 20) → 19-ch monopolar."""
    src = ALEX_DIR / mat_file
    if not src.exists(): return None, 'no_alex_source'
    m = sio.loadmat(str(src))
    data = m.get('data')
    if data is None: return None, 'no_data_key'
    if data.shape != (20, 2000):
        return None, f'wrong_shape_{data.shape}'
    return data[:19], 'ok'


def restore_from_sub_S0001(pid, mat_file, eeg_file_hint):
    """eeg_file_hint is the sub-S0001 filename if known; otherwise pick first match."""
    candidate = None
    if eeg_file_hint and not pd.isna(eeg_file_hint):
        ef = SUB_S0001_CACHE / eeg_file_hint
        if not eeg_file_hint.endswith('.mat'):
            ef = SUB_S0001_CACHE / (eeg_file_hint + '.mat')
        if ef.exists(): candidate = ef
    if candidate is None:
        # Find any sub-S0001<pid>_*.mat
        for f in SUB_S0001_CACHE.glob(f'sub-S0001{pid}_*.mat'):
            candidate = f; break
    if candidate is None:
        return None, 'no_subS0001_match'
    m = sio.loadmat(str(candidate))
    data = m.get('data')
    if data is None: return None, 'no_data_key'
    if data.shape != (19, 2000):
        return None, f'wrong_shape_{data.shape}'
    return data, 'ok'


# Path-C window finder (re-extract 10-s window from morgoth1 source for given GT times)
def extract_pathC_window(pid, gt_times, subtype):
    """Re-extract a 10-second 19-channel monopolar window from morgoth1 source."""
    # morgoth1 dir layout: /BIRD,GPD,GRDA,IIIC,LPD,LRDA,SEIZURE,SPIKES_HM/
    # Each contains segments_raw/{pid}_*.mat
    subdirs_to_search = [subtype.upper(), 'IIIC', 'LPD', 'GPD', 'LRDA', 'GRDA']
    found = None
    for sd in subdirs_to_search:
        d = MORGOTH1_ROOT / sd / 'segments_raw'
        if not d.exists(): continue
        for f in d.glob(f'{pid}_*.mat'):
            found = f; break
        if found: break
    if found is None:
        return None, 'no_morgoth1_match'

    try:
        import h5py
        with h5py.File(str(found), 'r') as h:
            # morgoth1 .mat is v7.3 (HDF5)
            keys = list(h.keys())
            data_key = 'data' if 'data' in keys else keys[0]
            data = np.array(h[data_key]).T if h[data_key].ndim == 2 else np.array(h[data_key])
            # Expect 19 or 20 channels
            if data.shape[0] in (19, 20):
                if data.shape[0] == 20:
                    data = data[:19]
                full = data
            elif data.shape[1] in (19, 20):
                full = data.T[:19]
            else:
                return None, f'wrong_shape_{data.shape}'
    except Exception as e:
        return None, f'load_err_{type(e).__name__}'

    # Slide a 10-s window across; pick the one where energy sum at gt times is max
    n = full.shape[1]
    if n < WIN_SAMPLES:
        return None, f'too_short_{n}'
    # Bipolar for energy scoring (we don't save this version)
    BIP_IDX_simple = [(0,4),(4,5),(5,6),(6,7),(11,15),(15,16),(16,17),(17,18)]  # any subset
    energy = np.sum(np.abs(full[:8]), axis=0)  # crude but fast
    stride = FS
    starts = np.arange(0, n - WIN_SAMPLES + 1, stride)
    if len(starts) == 0:
        return None, 'no_starts'
    gt_samples = np.array([int(round(t * FS)) for t in gt_times])
    scores = np.zeros(len(starts))
    for i, s in enumerate(starts):
        score = 0.0
        for gs in gt_samples:
            idx = s + gs
            lo, hi = max(0, idx - 10), min(n, idx + 11)
            score += float(energy[lo:hi].max())
        scores[i] = score
    best = int(starts[int(np.argmax(scores))])
    return full[:, best:best + WIN_SAMPLES], 'ok'


def main():
    print('Loading inputs...', flush=True)
    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    seg_df = seg_df.drop_duplicates(subset=['mat_file'])
    with open(DT_JSON) as f:
        dt = json.load(f)

    # Backup
    BACKUP_DIR = PROJECT_DIR / 'data' / 'eeg.bipolar_backup'
    BACKUP_DIR.mkdir(exist_ok=True)

    # Find files in data/eeg/ that are currently (18, 2000)
    print('Scanning data/eeg/ for bipolar files...', flush=True)
    bipolar_files = []
    t0 = time.time()
    for i, row in enumerate(seg_df.itertuples(index=False)):
        p = EEG_DIR / row.mat_file
        if not p.exists(): continue
        try:
            m = sio.loadmat(str(p))
            d = m.get('data')
            if d is not None and d.shape == (18, 2000):
                bipolar_files.append({
                    'mat_file': row.mat_file,
                    'patient_id': row.patient_id,
                    'eeg_source': row.eeg_source,
                    'eeg_file': row.eeg_file,
                })
        except Exception:
            pass
        if (i + 1) % 2000 == 0:
            print(f'  scanned {i+1}/{len(seg_df)} ({time.time()-t0:.0f}s)', flush=True)
    print(f'  found {len(bipolar_files)} bipolar entries to restore', flush=True)

    counts = {}
    for b in bipolar_files:
        counts[b['eeg_source']] = counts.get(b['eeg_source'], 0) + 1
    print(f'  by source: {counts}', flush=True)

    # Restore
    n_ok = 0; n_fail = 0
    fail_reasons = {}
    t0 = time.time()
    for i, b in enumerate(bipolar_files):
        pid = b['patient_id']
        mat_file = b['mat_file']

        # Try strategies in order of speed
        new_data = None
        reason = None
        # Strategy A: sub-S0001 cache (works for IRR + reviewed cohorts)
        new_data, reason = restore_from_sub_S0001(pid, mat_file, b['eeg_file'])
        # Strategy B: Alex backup (works for alex_recovery)
        if new_data is None and b['eeg_source'] == 'alex_recovery':
            new_data, reason = restore_from_alex(pid, mat_file)
        # Strategy C: morgoth1 source (works for pathC)
        if new_data is None and b['eeg_source'] in ('pathC_morgoth1_s3',):
            gt = dt.get(pid, {})
            gt_times = gt.get('global_times', [])
            subtype = gt.get('subtype', 'IIIC')
            if gt_times:
                new_data, reason = extract_pathC_window(pid, gt_times, subtype)

        if new_data is None:
            n_fail += 1
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            continue

        # Backup current bipolar, then save monopolar
        cur_path = EEG_DIR / mat_file
        bk_path = BACKUP_DIR / mat_file
        if not bk_path.exists():
            shutil.copy2(cur_path, bk_path)
        sio.savemat(str(cur_path),
                    {'data': new_data.astype(np.float64),
                     'Fs': np.array([[FS]], dtype=np.int64)})
        n_ok += 1

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(bipolar_files)}] {time.time()-t0:.0f}s, ok={n_ok} fail={n_fail}', flush=True)

    print(f'\nDone: ok={n_ok}, fail={n_fail}', flush=True)
    if fail_reasons:
        print(f'Fail reasons: {fail_reasons}', flush=True)
    print(f'Bipolar backups saved to {BACKUP_DIR}', flush=True)


if __name__ == '__main__':
    main()
