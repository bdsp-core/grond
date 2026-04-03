"""
Build V5 channel-level PD dataset from ALL segments with expert frequency labels.

Key difference from V1 build_channel_dataset.py:
  - Uses ALL segments (not just first per patient) -- covers multiple time windows
  - Includes ALL PD segments with expert_freq_hz labels (not just patients.csv)
  - Reads from segment_labels.csv (canonical source) instead of patients.csv
  - No GPD subsampling (all 18 channels per GPD segment)
  - Stores freq_targets directly (log-frequency for PD+ channels, NaN for PD-)
  - Appends GRDA/LRDA channels from V1 dataset as PD-negative hard examples

Expected output: ~40K PD channels + ~1.3K RDA channels = ~41K total channels
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
V1_BACKUP_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache_backup_v1'

LEFT = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT = [4, 5, 6, 7, 12, 13, 14, 15]


def load_bipolar(mat_file):
    """Load a .mat file and return (18, N) bipolar array."""
    mat = sio.loadmat(str(EEG_DIR / mat_file))
    data = mat['data'].astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T
    # fcn_getBanana handles 19ch and 20ch monopolar -> 18ch bipolar
    # Also handles 18ch bipolar (passthrough when already correct)
    if data.shape[0] == 18:
        return data
    bi = np.array(fcn_getBanana(data)).astype(np.float32)
    return bi


def main():
    t0 = time.time()
    print("=" * 70)
    print("Building V5 channel-level PD dataset")
    print("  Source: ALL segments with expert frequency labels")
    print("=" * 70)

    # Load segment labels
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    eeg_files = set(os.listdir(str(EEG_DIR)))

    # Get all PD segments with expert frequency labels
    pd_segs = sl[
        (sl.subtype.isin(['lpd', 'gpd'])) &
        (sl.excluded != True) &
        (sl.expert_freq_hz.notna()) &
        (sl.mat_file.isin(eeg_files))
    ].copy()

    print(f"\nPD segments with expert freq: {len(pd_segs)}")
    print(f"  LPD: {len(pd_segs[pd_segs.subtype == 'lpd'])}")
    print(f"  GPD: {len(pd_segs[pd_segs.subtype == 'gpd'])}")
    print(f"  Unique patients: {pd_segs.patient_id.nunique()}")

    channels_list = []
    labels_list = []       # 1 for PD+, 0 for PD-
    patient_ids_list = []
    channel_indices_list = []
    subtypes_list = []
    freq_targets_list = []  # log-frequency for PD+ channels, NaN for PD-

    stats = {
        'gpd_positive': 0,
        'lpd_ipsi_positive': 0,
        'lpd_contra_negative': 0,
        'lpd_bilateral_positive': 0,
        'lpd_unknown_positive': 0,
        'skipped_load_error': 0,
        'skipped_short': 0,
        'skipped_nan': 0,
    }

    n_total = len(pd_segs)
    for i, (_, row) in enumerate(pd_segs.iterrows()):
        try:
            seg_bi = load_bipolar(row['mat_file'])
        except Exception as e:
            stats['skipped_load_error'] += 1
            continue

        if seg_bi.shape[1] < 2000:
            stats['skipped_short'] += 1
            continue

        seg_bi = seg_bi[:, :2000].astype(np.float32)

        # Check for NaN/Inf in entire segment
        if not np.all(np.isfinite(seg_bi)):
            # Check per-channel
            finite_mask = np.all(np.isfinite(seg_bi), axis=1)
            if not np.any(finite_mask):
                stats['skipped_nan'] += 1
                continue
        else:
            finite_mask = np.ones(seg_bi.shape[0], dtype=bool)

        freq = float(row['expert_freq_hz'])
        subtype = row['subtype']
        pid = str(row['patient_id'])
        lat = str(row.get('laterality', '')).lower().strip()
        if lat == 'nan' or lat == '':
            lat = ''

        log_freq = np.log(freq) if freq > 0 else np.nan

        for ch in range(min(18, seg_bi.shape[0])):
            if not finite_mask[ch]:
                continue

            # Determine if this channel is PD+
            if subtype == 'gpd':
                if lat in ('left', 'right'):
                    # Rare GPD with laterality: treat like LPD
                    if lat == 'left':
                        is_pd = ch in LEFT
                    else:
                        is_pd = ch in RIGHT
                else:
                    # GPD or bilateral GPD: all channels PD+
                    is_pd = True
                if is_pd:
                    stats['gpd_positive'] += 1

            elif subtype == 'lpd':
                if lat == 'left':
                    is_pd = ch in LEFT
                    if is_pd:
                        stats['lpd_ipsi_positive'] += 1
                    else:
                        stats['lpd_contra_negative'] += 1
                elif lat == 'right':
                    is_pd = ch in RIGHT
                    if is_pd:
                        stats['lpd_ipsi_positive'] += 1
                    else:
                        stats['lpd_contra_negative'] += 1
                elif lat == 'bilateral':
                    is_pd = True  # all non-midline channels
                    stats['lpd_bilateral_positive'] += 1
                else:
                    # Unknown laterality -- assume PD+ (conservative)
                    is_pd = True
                    stats['lpd_unknown_positive'] += 1
            else:
                is_pd = True

            channels_list.append(seg_bi[ch])
            labels_list.append(1 if is_pd else 0)
            patient_ids_list.append(pid)
            channel_indices_list.append(ch)
            subtypes_list.append(subtype)
            freq_targets_list.append(log_freq if is_pd and np.isfinite(log_freq) else np.nan)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{n_total} segments processed ({elapsed:.1f}s)")

    print(f"\n  All {n_total} PD segments processed")
    print(f"  PD channels so far: {len(channels_list)}")

    # -- Add GRDA/LRDA channels from V1 dataset as PD-negative hard examples --
    print("\nAppending GRDA/LRDA hard negatives from V1 dataset...")
    v1_path = V1_BACKUP_DIR / 'channel_dataset.npz'
    if v1_path.exists():
        ds_v1 = np.load(str(v1_path), allow_pickle=True)
        v1_subtypes = ds_v1['subtypes']
        rda_mask = np.isin(v1_subtypes, ['grda', 'lrda'])
        n_rda = int(np.sum(rda_mask))
        print(f"  Found {n_rda} RDA channels in V1 dataset")

        v1_channels = ds_v1['channels'][rda_mask]
        v1_labels = ds_v1['labels'][rda_mask]  # should be all 0
        v1_pids = ds_v1['patient_ids'][rda_mask]
        v1_ch_idx = ds_v1['channel_indices'][rda_mask]
        v1_st = v1_subtypes[rda_mask]

        for j in range(n_rda):
            channels_list.append(v1_channels[j])
            labels_list.append(int(v1_labels[j]))  # 0 (PD-)
            patient_ids_list.append(str(v1_pids[j]))
            channel_indices_list.append(int(v1_ch_idx[j]))
            subtypes_list.append(str(v1_st[j]))
            freq_targets_list.append(np.nan)  # no freq for RDA

        stats['rda_negative'] = n_rda
    else:
        print(f"  WARNING: V1 dataset not found at {v1_path}, skipping RDA negatives")
        stats['rda_negative'] = 0

    # -- Convert to arrays and save -------------------------------------------
    print("\nConverting to numpy arrays...")
    channels = np.array(channels_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    patient_ids = np.array(patient_ids_list, dtype=str)
    channel_indices = np.array(channel_indices_list, dtype=np.int8)
    subtypes = np.array(subtypes_list, dtype=str)
    freq_targets = np.array(freq_targets_list, dtype=np.float32)

    # Final NaN/Inf check
    finite_mask = np.all(np.isfinite(channels), axis=1)
    n_removed = int(np.sum(~finite_mask))
    if n_removed > 0:
        print(f"  Removed {n_removed} channels with NaN/Inf values")
        channels = channels[finite_mask]
        labels = labels[finite_mask]
        patient_ids = patient_ids[finite_mask]
        channel_indices = channel_indices[finite_mask]
        subtypes = subtypes[finite_mask]
        freq_targets = freq_targets[finite_mask]

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / 'channel_dataset_v5.npz'
    np.savez(
        str(out_path),
        channels=channels,
        labels=labels,
        patient_ids=patient_ids,
        channel_indices=channel_indices,
        subtypes=subtypes,
        freq_targets=freq_targets,
    )

    # -- Print statistics -----------------------------------------------------
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    n_total_ch = len(labels)
    n_patients = len(set(patient_ids))

    n_lpd = int(np.sum(subtypes == 'lpd'))
    n_gpd = int(np.sum(subtypes == 'gpd'))
    n_grda = int(np.sum(subtypes == 'grda'))
    n_lrda = int(np.sum(subtypes == 'lrda'))

    n_freq_valid = int(np.sum(np.isfinite(freq_targets)))
    n_freq_lpd = int(np.sum(np.isfinite(freq_targets) & (subtypes == 'lpd')))
    n_freq_gpd = int(np.sum(np.isfinite(freq_targets) & (subtypes == 'gpd')))

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"V5 Channel Dataset saved to: {out_path}")
    print(f"{'=' * 70}")
    print(f"  Total channels:     {n_total_ch}")
    print(f"  Positive (PD+):     {n_pos} ({100 * n_pos / n_total_ch:.1f}%)")
    print(f"  Negative (PD-):     {n_neg} ({100 * n_neg / n_total_ch:.1f}%)")
    print(f"  Unique patients:    {n_patients}")
    print(f"\n  By subtype:")
    print(f"    LPD channels:     {n_lpd}")
    print(f"    GPD channels:     {n_gpd}")
    print(f"    GRDA channels:    {n_grda}")
    print(f"    LRDA channels:    {n_lrda}")
    print(f"\n  Freq targets (PD+ with log-freq):")
    print(f"    Total:            {n_freq_valid}")
    print(f"    LPD:              {n_freq_lpd}")
    print(f"    GPD:              {n_freq_gpd}")
    print(f"\n  Breakdown:")
    print(f"    GPD positive:              {stats['gpd_positive']}")
    print(f"    LPD ipsilateral positive:  {stats['lpd_ipsi_positive']}")
    print(f"    LPD contralateral negative:{stats['lpd_contra_negative']}")
    print(f"    LPD bilateral positive:    {stats['lpd_bilateral_positive']}")
    print(f"    LPD unknown lat positive:  {stats['lpd_unknown_positive']}")
    print(f"    RDA negative (from V1):    {stats['rda_negative']}")
    print(f"    Skipped (load error):      {stats['skipped_load_error']}")
    print(f"    Skipped (short):           {stats['skipped_short']}")
    print(f"    Skipped (NaN):             {stats['skipped_nan']}")
    print(f"\n  Dataset size: {channels.nbytes / 1e9:.2f} GB")
    print(f"  Build time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
