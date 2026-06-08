#!/usr/bin/env python3
"""Re-extract bank-matched 10s windows using GT-discharge-aligned window finder.

For each patient that has both:
  - A high-NCC match in bank_match_results.csv (found the right recording)
  - GT discharge_times in discharge_times.json

Slide a 10s window through the matched 30s bank entry; pick the window
where the SUM OF ENERGY at GT_discharge_times (re-referenced to window start)
is maximum. This is the path-C window-finder logic but applied to the
NCC-identified bank recording.

This should give us 10s slices where discharges actually occur at the
GT-annotated times.
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
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
MATCH_CSV = PROJECT_DIR / 'results' / 'c1_repro' / 'bank_match_results.csv'
H5_PATH = '/Volumes/Extreme SSD/.eeg_cache/eeg-test/eeg_bank_spec.h5'

NCC_THRESHOLD = 0.9
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


def find_gt_aligned_window(bip, gt_times):
    """Slide 10s window with 1-sample stride; pick window where sum of
    energy at GT_times is max.

    Returns (best_start_sample, best_score, normalized_score) or (None, None, None).
    """
    n_ch, n = bip.shape
    if n < WIN_SAMPLES:
        return None, None, None
    energy = np.sum(np.abs(bip), axis=0)
    gt_samples = np.array([int(round(t * FS)) for t in gt_times])
    if len(gt_samples) == 0:
        return None, None, None

    # Stride 5 samples (25 ms) for reasonable speed vs accuracy
    stride = 5
    starts = np.arange(0, n - WIN_SAMPLES + 1, stride)
    scores = np.zeros(len(starts), dtype=np.float64)
    for i, s in enumerate(starts):
        score = 0.0
        for gs in gt_samples:
            idx = s + gs
            if 0 <= idx < n:
                # Local max in ±50 ms window
                lo = max(0, idx - 10)
                hi = min(n, idx + 11)
                score += float(energy[lo:hi].max())
        scores[i] = score

    best_i = int(np.argmax(scores))
    best = int(starts[best_i])
    norm = float(scores[best_i] / max(np.median(scores), 1e-9))
    return best, float(scores[best_i]), norm


def main():
    matches = pd.read_csv(MATCH_CSV)
    matches['pid'] = matches['pid'].astype(str)
    matches['bank_seg_id'] = matches['bank_seg_id'].astype(str)
    print(f'Loaded {len(matches)} match results', flush=True)

    hi = matches[matches['best_ncc'] >= NCC_THRESHOLD].copy()
    print(f'  {len(hi)} above NCC threshold {NCC_THRESHOLD}', flush=True)

    with open(DT_JSON) as f:
        dt = json.load(f)
    # Restrict to PD-GT patients only (those with timing labels)
    hi['has_gt'] = hi['pid'].apply(lambda p: p in dt
                                   and dt[p].get('review_status') == 'ground_truth'
                                   and dt[p].get('subtype') in ('lpd', 'gpd')
                                   and isinstance(dt[p].get('global_times'), list)
                                   and len(dt[p]['global_times']) >= 2)
    hi_gt = hi[hi['has_gt']].copy()
    print(f'  {len(hi_gt)} of them have PD-GT labels', flush=True)

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    n_ok = 0
    n_fail = 0
    aligned_audit = []
    t0 = time.time()

    with h5py.File(H5_PATH, 'r') as f:
        seg = f['segments']
        for i, row in enumerate(hi_gt.itertuples(index=False)):
            try:
                gt_times = dt[row.pid]['global_times']
                bank_obj = seg[row.bank_seg_id]
                if 'data30s' not in bank_obj:
                    n_fail += 1
                    continue
                data30s = np.array(bank_obj['data30s'])
                if data30s.shape != (20, 6000) or not np.isfinite(data30s).all():
                    n_fail += 1
                    continue
                mono = data30s[:19].astype(np.float64)
                bip = mono[BIP_IDX[:, 0]] - mono[BIP_IDX[:, 1]]
                start, score, norm = find_gt_aligned_window(bip, gt_times)
                if start is None:
                    n_fail += 1
                    continue
                window = bip[:, start:start + WIN_SAMPLES]
                # Save (overwrite)
                out_path = EEG_DIR / row.mat_file
                sio.savemat(str(out_path),
                            {'data': window.astype(np.float64),
                             'Fs': np.array([[FS]], dtype=np.int64)})
                aligned_audit.append({
                    'pid': row.pid,
                    'mat_file': row.mat_file,
                    'bank_seg_id': row.bank_seg_id,
                    'ncc_window_start_s': row.bank_window_start_s,
                    'gt_aligned_window_start_s': start / FS,
                    'gt_aligned_norm_score': norm,
                })
                n_ok += 1
            except Exception:
                n_fail += 1
                continue

            if (i + 1) % 50 == 0:
                print(f'  [{i+1}/{len(hi_gt)}] {time.time()-t0:.0f}s ok={n_ok} fail={n_fail}',
                      flush=True)

    print(f'\nDone: ok={n_ok}, fail={n_fail}', flush=True)
    audit_df = pd.DataFrame(aligned_audit)
    audit_df.to_csv(PROJECT_DIR / 'results' / 'c1_repro' / 'bank_gt_aligned_audit.csv',
                    index=False)
    if len(audit_df):
        print(f'\n## GT-aligned window-start distribution:')
        print(audit_df['gt_aligned_window_start_s'].describe())
        print(f'\n## NCC-vs-GT-aligned offset agreement:')
        diff = audit_df['gt_aligned_window_start_s'] - audit_df['ncc_window_start_s']
        print(diff.describe())


if __name__ == '__main__':
    main()
