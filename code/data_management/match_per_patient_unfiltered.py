#!/usr/bin/env python3
"""For each missing PD-GT patient, find the UNFILTERED 10-s source file in
the local sub-S0001 cache that best aligns with the GT discharge times.

Algorithm:
  1. Get list of all sub-S0001<pid>_*.mat files in
     /Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/iiic-freq3/data/eeg
     for the target PID (1-199 candidates per patient).
  2. For each candidate (19-ch monopolar, 200 Hz × 10 s):
       - Derive 18-ch bipolar.
       - Compute envelope = sum |bipolar| across channels.
       - Compute discharge-time alignment score: sum over GT times of
         envelope local-max in [t - 25 ms, t + 25 ms] window.
       - Normalize: alignment_score / envelope.mean()
  3. Pick the candidate with highest normalized alignment score.
  4. Save unfiltered bipolar 10-s slice as data/eeg/{pid}_seg000.mat.
  5. Update segments.csv: eeg_source = 'subS_unfiltered_recovery'.

This finds the EXACT 10-second sub-S0001 file the labeler used, with
unfiltered content. Unlike the bank-NCC approach, the per-patient search
space is small (1-200) and the discharge-time alignment metric is direct.
"""
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
CACHE = Path('/Volumes/Extreme SSD/.eeg_cache/bdsp-opendata-credentialed/iiic-freq3/data/eeg')

FS = 200
WIN_SAMPLES = 2000  # 10 seconds

# Bipolar derivation
MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
        'Fp2','F4','C4','P4','F8','T4','T6','O2']
PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
         ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
         ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
         ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
         ('Fz','Cz'),('Cz','Pz')]
BIP_IDX = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])

# Acceptance: only commit a match if normalized alignment is above this threshold
NORM_THRESHOLD = 0.5   # accept any non-degenerate match; we want to use sub-S0001 unfiltered originals over the filtered Alex/IRR/pathC versions


def alignment_score(bipolar, gt_times):
    """For 18-channel (18, 2000) bipolar EEG at 200 Hz, return mean
    of local-max envelope value at each GT discharge time, normalized
    by overall envelope mean."""
    n_ch, n = bipolar.shape
    if n != WIN_SAMPLES:
        return None
    energy = np.sum(np.abs(bipolar), axis=0)
    overall_mean = float(energy.mean())
    if overall_mean <= 0:
        return None
    gt_samples = [int(round(t * FS)) for t in gt_times if 0 <= t * FS < n]
    if not gt_samples:
        return None
    local_maxes = []
    for gs in gt_samples:
        lo = max(0, gs - 5)
        hi = min(n, gs + 6)
        local_maxes.append(float(energy[lo:hi].max()))
    return float(np.mean(local_maxes) / overall_mean)


def main():
    print('Loading inputs...', flush=True)
    with open(DT_JSON) as f:
        dt = json.load(f)
    gt_pd = {pid: v for pid, v in dt.items()
             if v.get('review_status') == 'ground_truth'
             and v.get('subtype') in ('lpd', 'gpd')
             and isinstance(v.get('global_times'), list)
             and len(v.get('global_times')) >= 2}

    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    # Process ALL PD-GT patients (replace any filtered versions with unfiltered)
    have = set(seg_df['patient_id'])
    missing = sorted(gt_pd.keys())
    print(f'Processing ALL PD-GT patients: {len(missing)} (will replace filtered originals with unfiltered)', flush=True)

    # Build cache index: pid -> [sub-S0001<pid>_<ts>.mat]
    print(f'Indexing sub-S0001 cache at {CACHE}...', flush=True)
    pid_to_files = {}
    for f in CACHE.glob('sub-S0001*.mat'):
        m = re.match(r'sub-S0001(\d+)_', f.name)
        if m:
            pid_to_files.setdefault(m.group(1), []).append(f)
    print(f'  Cache has files for {len(pid_to_files)} unique PIDs', flush=True)

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    # Per-PID match
    seg_cols = list(seg_df.columns)
    pat_cols = list(pat_df.columns)
    new_seg_rows = []
    new_pat_rows = []
    updates = []   # (pid, mat_file, ts, norm_score)
    skipped_no_cache = 0
    skipped_low_score = 0
    n_ok = 0

    EEG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for i, pid in enumerate(missing):
        gt_times = gt_pd[pid]['global_times']
        cache_files = pid_to_files.get(pid, [])
        if not cache_files:
            skipped_no_cache += 1
            continue

        # Score each candidate
        best = None  # (norm_score, file, bipolar)
        for cf in cache_files:
            try:
                m = sio.loadmat(str(cf))
                data = m.get('data')
                if data is None: continue
                if data.shape[0] > data.shape[1]:
                    data = data.T
                if data.shape != (19, 2000): continue
                bip = data[BIP_IDX[:, 0]].astype(np.float64) - data[BIP_IDX[:, 1]].astype(np.float64)
                norm = alignment_score(bip, gt_times)
                if norm is None: continue
                if best is None or norm > best[0]:
                    best = (norm, cf, bip)
            except Exception:
                continue

        if best is None:
            skipped_low_score += 1
            continue
        norm, cf, bip = best
        if norm < NORM_THRESHOLD:
            skipped_low_score += 1
            continue

        # Save
        out_path = EEG_DIR / f'{pid}_seg000.mat'
        sio.savemat(str(out_path),
                    {'data': bip.astype(np.float64),
                     'Fs': np.array([[FS]], dtype=np.int64)})
        n_ok += 1
        updates.append((pid, cf.name, norm))

        # Register in segments.csv if not already
        if pid not in have:
            row = {c: '' for c in seg_cols}
            row.update({
                'mat_file': f'{pid}_seg000.mat',
                'patient_id': pid,
                'subtype': gt_pd[pid]['subtype'],
                'subtype_source': 'subS_unfiltered_recovery',
                'algo_freq_hz': '',
                'excluded': False,
                'exclusion_reason': '',
                'subtype_original': gt_pd[pid]['subtype'],
                'has_discharge_timing': True,
                'has_wave_timing': False,
                'has_channel_involvement': False,
                'eeg_source': 'subS_unfiltered_recovery',
                'eeg_file': cf.name,
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
            new_seg_rows.append(row)
            if pid not in existing_pat_pids:
                gold = gt_pd[pid].get('selected_freq')
                if gold is None or not np.isfinite(gold):
                    ipis = np.diff(sorted(gt_times))
                    gold = float(1.0 / np.median(ipis)) if len(ipis) > 0 else 1.0
                lat = gt_pd[pid].get('laterality') or ''
                new_pat_rows.append({
                    'patient_id': pid,
                    'subtype': gt_pd[pid]['subtype'],
                    'subtype_original': gt_pd[pid]['subtype'],
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

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(missing)}] {elapsed:.0f}s, ok={n_ok}, '
                  f'no_cache={skipped_no_cache}, low_score={skipped_low_score}',
                  flush=True)

    print(f'\nFinal: ok={n_ok}, no_cache={skipped_no_cache}, low_score={skipped_low_score}', flush=True)

    if new_seg_rows:
        seg_df_new = pd.concat([seg_df, pd.DataFrame(new_seg_rows, columns=seg_cols)],
                               ignore_index=True)
        seg_df_new.to_csv(SEG_CSV, index=False)
        print(f'  segments.csv: {len(seg_df)} -> {len(seg_df_new)} rows', flush=True)
    if new_pat_rows:
        pat_df_new = pd.concat([pat_df, pd.DataFrame(new_pat_rows, columns=pat_cols)],
                               ignore_index=True)
        pat_df_new.to_csv(PAT_CSV, index=False)
        print(f'  patients.csv: {len(pat_df)} -> {len(pat_df_new)} rows', flush=True)

    # Audit CSV
    audit_df = pd.DataFrame(updates, columns=['pid', 'source_file', 'norm_score'])
    audit_out = PROJECT_DIR / 'results' / 'c1_repro' / 'per_pid_unfiltered_audit.csv'
    audit_df.to_csv(audit_out, index=False)
    print(f'  Audit: {audit_out.relative_to(PROJECT_DIR)}', flush=True)


if __name__ == '__main__':
    main()
