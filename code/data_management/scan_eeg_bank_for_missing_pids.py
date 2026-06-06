#!/usr/bin/env python3
"""Scan the local eeg_bank_spec.h5 for our 501 missing PD-GT patients.

Reads the per-entry `patient_id` attr from every one of the 132,291
segments in `/Volumes/Extreme SSD/.eeg_cache/eeg-test/eeg_bank_spec.h5`,
builds a pid→seg_id map, intersects with our missing PD-GT cohort, and
reports.

For each hit, extracts the 30-second EEG (sample rate 200 Hz, 20-channel
monopolar including EKG, dropping EKG → 19-channel monopolar → 18-channel
bipolar via standard banana montage), runs the path-C-style energy-aligned
window-finder against the patient's GT discharge times to pick the labeled
10-second window, and saves to data/eeg/{pid}_seg000.mat.

Usage:
    conda run -n morgoth python code/data_management/scan_eeg_bank_for_missing_pids.py
"""
from __future__ import annotations
import csv
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
PAT_CSV = PROJECT_DIR / 'data' / 'labels' / 'archive_labels' / 'patients.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

H5_PATH = Path('/Volumes/Extreme SSD/.eeg_cache/eeg-test/eeg_bank_spec.h5')

# Bipolar derivation
MONO_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
BIPOLAR_PAIRS = [
    ('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
    ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
    ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
    ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
    ('Fz','Cz'),('Cz','Pz'),
]
BIPOLAR_INDICES = np.array([[MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)]
                            for a, b in BIPOLAR_PAIRS])
FS = 200
WIN_SAMPLES = 2000  # 10 seconds


def to_bipolar(mono):
    return mono[BIPOLAR_INDICES[:, 0]] - mono[BIPOLAR_INDICES[:, 1]]


def find_best_window(bipolar, gt_times):
    """Slide a 10s window with 1s stride; return (best_start_sample, norm_score)."""
    n_ch, n = bipolar.shape
    if n < WIN_SAMPLES:
        return None, None
    energy = np.sum(np.abs(bipolar), axis=0)
    gt_samples = np.array([int(round(t * FS)) for t in gt_times])
    stride = FS
    starts = np.arange(0, n - WIN_SAMPLES + 1, stride)
    if len(starts) == 0:
        return None, None
    scores = np.zeros(len(starts))
    for i, s in enumerate(starts):
        score = 0.0
        for gs in gt_samples:
            idx = s + gs
            lo, hi = max(0, idx - 10), min(n, idx + 11)
            if hi > lo:
                score += float(energy[lo:hi].mean())
        scores[i] = score
    best_i = int(np.argmax(scores))
    best = int(starts[best_i])
    norm = float(scores[best_i] / max(np.median(scores), 1e-9))
    return best, norm


def main():
    if not H5_PATH.exists():
        print(f'ERROR: {H5_PATH} not found. Download must complete first.')
        sys.exit(1)
    size_gb = H5_PATH.stat().st_size / 1e9
    print(f'Found {H5_PATH} ({size_gb:.1f} GB)')

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
    missing = sorted({p for p in gt_pd if p not in set(seg_df['patient_id'])})
    print(f'  Still-missing PD-GT patients (post-prior-recovery): {len(missing)}')

    # Originally-missing (pre-recovery) for the larger search
    try:
        seg_pre = pd.read_csv('/tmp/recovery_backup/segments.csv.pre_recovery')
        seg_pre['patient_id'] = seg_pre['patient_id'].astype(str)
        orig_missing = sorted(set(gt_pd) - set(seg_pre['patient_id']))
    except FileNotFoundError:
        orig_missing = missing
    print(f'  Originally-missing (pre-recovery): {len(orig_missing)}')

    pat_df = pd.read_csv(PAT_CSV)
    pat_df['patient_id'] = pat_df['patient_id'].astype(str)
    existing_pat_pids = set(pat_df['patient_id'])

    print(f'\nOpening {H5_PATH.name} and walking all entries...')
    t0 = time.time()
    with h5py.File(H5_PATH, 'r') as f:
        seg = f['segments']
        n_total = len(seg)
        print(f'  Total segments: {n_total}')
        # Iterate all entries, collect (pid, seg_id, attrs needed for extraction)
        pid_to_segs = {}
        n_scanned = 0
        n_with_pid = 0
        all_attr_keys = set()
        for k in seg:
            obj = seg[k]
            attrs = dict(obj.attrs)
            all_attr_keys.update(attrs.keys())
            pid_raw = attrs.get('patient_id', '')
            n_scanned += 1
            if not pid_raw:
                continue
            n_with_pid += 1
            pid = str(pid_raw).rstrip('.0')
            pid_to_segs.setdefault(pid, []).append({
                'seg_id': k,
                'eeg_duration_s': float(attrs.get('eeg_duration_s', 0)),
                'fs_hz30s': float(attrs.get('fs_hz30s', 200)),
                'source_window_center_s': float(attrs.get('source_window_center_s', 0)),
                'spec_source': str(attrs.get('spec_source', '')),
                'pattern_class': str(attrs.get('pattern_class', '')),
            })
            if n_scanned % 1000 == 0:
                elapsed = time.time() - t0
                rate = n_scanned / elapsed
                eta = (n_total - n_scanned) / rate
                print(f'    [{n_scanned}/{n_total}] {elapsed:.0f}s elapsed, '
                      f'rate={rate:.0f}/s, ETA={eta/60:.1f} min, '
                      f'with_pid={n_with_pid}, unique_pids={len(pid_to_segs)}')
        elapsed = time.time() - t0
        print(f'  Walk complete: {elapsed:.0f}s ({n_scanned} entries; '
              f'{n_with_pid} with patient_id; {len(pid_to_segs)} unique pids)')
        print(f'  All attr keys: {sorted(all_attr_keys)}')

        # Cross-check against missing PIDs
        post_recovery_hits = sorted(set(missing) & set(pid_to_segs))
        orig_missing_hits = sorted(set(orig_missing) & set(pid_to_segs))
        print(f'\n  HITS:')
        print(f'    Still-missing (post-recovery) PIDs found: {len(post_recovery_hits)} / {len(missing)}')
        if post_recovery_hits:
            print(f'      Sample: {post_recovery_hits[:5]}')
        print(f'    Originally-missing PIDs found: {len(orig_missing_hits)} / {len(orig_missing)}')

        # Save the pid→seg_id mapping
        OUT_DIR = PROJECT_DIR / 'results' / 'c1_repro'
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_DIR / 'eeg_bank_pid_to_seg.json', 'w') as out:
            json.dump(pid_to_segs, out, indent=2, default=str)
        print(f'  Saved pid→seg_id map: {(OUT_DIR / "eeg_bank_pid_to_seg.json").relative_to(PROJECT_DIR)}')

        if not post_recovery_hits and not orig_missing_hits:
            print('\nNo missing PIDs found in eeg_bank_spec.h5. Done.')
            return

        # For each STILL-missing PD-GT hit, extract the labeled 10s window
        EEG_DIR.mkdir(parents=True, exist_ok=True)
        new_seg_rows, new_pat_rows = [], []
        seg_cols = list(seg_df.columns)
        pat_cols = list(pat_df.columns)

        print(f'\nExtracting EEG for {len(post_recovery_hits)} still-missing PD-GT hits...')
        n_ok, n_lowconf, n_fail = 0, 0, 0
        for i, pid in enumerate(post_recovery_hits):
            gt_data = gt_pd[pid]
            gt_times = gt_data['global_times']
            candidates = pid_to_segs[pid]
            # Prefer subtype-matching candidates; fall back to others
            expected_sub = gt_data['subtype'].upper()
            candidates_sorted = sorted(candidates,
                                       key=lambda c: (0 if c['pattern_class'].upper() == expected_sub else 1))
            best_overall = None
            for c in candidates_sorted:
                k = c['seg_id']
                obj = seg[k]
                # data30s should be present
                if 'data30s' not in obj:
                    continue
                data30s = np.array(obj['data30s'])  # (20, 6000)
                if data30s.shape[1] < WIN_SAMPLES:
                    continue
                if data30s.shape[0] == 20:
                    mono = data30s[:19]  # drop EKG
                elif data30s.shape[0] == 19:
                    mono = data30s
                else:
                    continue
                bip = to_bipolar(mono.astype(np.float64))
                start, norm = find_best_window(bip, gt_times)
                if start is None:
                    continue
                scored = {'seg_id': k, 'start': start, 'norm': norm,
                          'bip_slice': bip[:, start:start + WIN_SAMPLES],
                          'pattern_class': c['pattern_class']}
                if best_overall is None or scored['norm'] > best_overall['norm']:
                    best_overall = scored
                if scored['norm'] >= 2.0:
                    break

            if best_overall is None:
                n_fail += 1
                print(f'  [{i+1}/{len(post_recovery_hits)}] {pid}: FAIL (no usable window)')
                continue

            out_mat = f'{pid}_seg000.mat'
            sio.savemat(str(EEG_DIR / out_mat),
                        {'data': best_overall['bip_slice'].astype(np.float64),
                         'Fs': np.array([[FS]], dtype=np.int64)})
            if best_overall['norm'] >= 2.0:
                n_ok += 1
            else:
                n_lowconf += 1

            new_seg_rows.append({
                'mat_file': out_mat,
                'patient_id': pid,
                'subtype': gt_data['subtype'],
                'subtype_source': 'eeg_bank_spec_h5',
                'algo_freq_hz': '',
                'excluded': False,
                'exclusion_reason': '',
                'subtype_original': gt_data['subtype'],
                'has_discharge_timing': True,
                'has_wave_timing': False,
                'has_channel_involvement': False,
                'eeg_source': 'eeg_bank_spec_h5',
                'eeg_file': best_overall['seg_id'],
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
                gold = gt_data.get('selected_freq')
                if gold is None or (isinstance(gold, float) and not np.isfinite(gold)):
                    ipis = np.diff(sorted(gt_times))
                    gold = float(1.0 / np.median(ipis)) if len(ipis) > 0 else 1.0
                lat = gt_data.get('laterality') or ''
                new_pat_rows.append({
                    'patient_id': pid,
                    'subtype': gt_data['subtype'],
                    'subtype_original': gt_data['subtype'],
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

            print(f'  [{i+1}/{len(post_recovery_hits)}] {pid}: seg_id={best_overall["seg_id"]} '
                  f'start={best_overall["start"]/FS:.1f}s norm={best_overall["norm"]:.2f}')

    print(f'\nExtracted: ok={n_ok} (high-conf), low_conf={n_lowconf}, fail={n_fail}')
    if new_seg_rows:
        seg_df_new = pd.concat([seg_df, pd.DataFrame(new_seg_rows, columns=seg_cols)],
                               ignore_index=True)
        seg_df_new.to_csv(SEG_CSV, index=False)
        pat_df_new = pd.concat([pat_df, pd.DataFrame(new_pat_rows, columns=pat_cols)],
                               ignore_index=True)
        pat_df_new.to_csv(PAT_CSV, index=False)
        print(f'  segments.csv: {len(seg_df)} → {len(seg_df_new)} rows')
        print(f'  patients.csv: {len(pat_df)} → {len(pat_df_new)} rows')


if __name__ == '__main__':
    main()
