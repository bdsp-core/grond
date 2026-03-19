"""
Round 4 annotation selection: GPD (all available) + LPD (up to 30) for frequency annotation.
Priority is GPD — we need ~27+ more GPD patients to reach 100 total.

Steps:
1. Load cached external segments (3816 segments, 18ch bipolar, 2000 samples @ 200Hz)
2. Exclude previously annotated patients (canonical + candidates + round2 + round3)
3. For GPD: select ALL available patients, stratified across 5 frequency bins
4. For LPD: select up to 30 patients, stratified across 5 frequency bins
5. Save .mat files and manifest.csv
"""

import sys, os, json
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
CACHE = DATA / 'dl_cache'
OUT = DATA / '_archive' / 'pd_round4'
FS = 200  # sampling rate

FREQ_BINS = [
    (0.0, 0.6),
    (0.6, 0.8),
    (0.8, 1.2),
    (1.2, 2.0),
    (2.0, 10.0),
]


def load_previously_annotated():
    """Load all patient IDs from previous annotation rounds."""
    pids = set()

    # Round 1: canonical dataset
    labels_path = DATA / '_archive' / 'canonical_dataset' / 'labels.csv'
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  canonical_dataset: {len(df)} rows, {len(df['patient_id'].unique())} unique patients")

    # Round 1 candidates
    cand_path = DATA / '_archive' / 'pd_round1_candidates' / 'manifest.csv'
    if cand_path.exists():
        df = pd.read_csv(cand_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  pd_round1_candidates: {len(df)} rows")

    # Round 2
    r2_path = DATA / '_archive' / 'pd_round2' / 'manifest.csv'
    if r2_path.exists():
        df = pd.read_csv(r2_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  pd_round2: {len(df)} rows")

    # Round 3
    r3_path = DATA / '_archive' / 'pd_round3' / 'manifest.csv'
    if r3_path.exists():
        df = pd.read_csv(r3_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  pd_round3: {len(df)} rows")

    print(f"  Total previously annotated patient IDs: {len(pids)}")
    return pids


def compute_pointiness_trace(signal_1d, half_win=8):
    """Compute |d^2x/dt^2| pointiness trace."""
    n = len(signal_1d)
    pt = np.zeros(n)
    for i in range(half_win, n - half_win):
        left = signal_1d[i - half_win:i]
        right = signal_1d[i:i + half_win]
        mid = signal_1d[i]
        avg_neighbors = (np.mean(left) + np.mean(right)) / 2
        pt[i] = abs(mid - avg_neighbors)
    return pt


def compute_frequency_estimates(segment, fs=200):
    """
    Compute quick frequency estimates from a single bipolar segment (18, 2000).
    Returns dict with f_fft, f_tkeo, f_peaks.
    """
    from scipy.signal import butter, filtfilt, find_peaks
    from scipy.ndimage import gaussian_filter1d

    n_channels = segment.shape[0]
    n_samples = segment.shape[1]

    # Lowpass at 15 Hz for pointiness
    try:
        b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    except:
        b_lp, a_lp = None, None

    fft_freqs_all = []
    tkeo_freqs_all = []
    peak_counts_all = []

    for ch in range(n_channels):
        sig = segment[ch, :]

        # Skip flat/dead channels
        if np.std(sig) < 1e-6:
            continue

        # Apply lowpass
        if b_lp is not None:
            try:
                sig_lp = filtfilt(b_lp, a_lp, sig)
            except:
                sig_lp = sig
        else:
            sig_lp = sig

        # Pointiness trace
        pt = compute_pointiness_trace(sig_lp)
        pt = gaussian_filter1d(pt, sigma=fs * 0.02)

        # FFT of pointiness
        pt_centered = pt - np.mean(pt)
        if np.std(pt_centered) > 1e-10:
            fft_vals = np.abs(np.fft.rfft(pt_centered))
            freqs = np.fft.rfftfreq(len(pt_centered), d=1.0/fs)
            # Restrict to [0.3, 3.5] Hz
            mask = (freqs >= 0.3) & (freqs <= 3.5)
            if mask.sum() > 0:
                peak_idx = np.argmax(fft_vals[mask])
                fft_freqs_all.append(freqs[mask][peak_idx])

        # TKEO
        tkeo = np.zeros(len(sig_lp))
        tkeo[1:-1] = sig_lp[1:-1]**2 - sig_lp[:-2] * sig_lp[2:]
        tkeo = np.abs(tkeo)
        tkeo = gaussian_filter1d(tkeo, sigma=fs * 0.02)
        tkeo_centered = tkeo - np.mean(tkeo)
        if np.std(tkeo_centered) > 1e-10:
            fft_vals_t = np.abs(np.fft.rfft(tkeo_centered))
            freqs_t = np.fft.rfftfreq(len(tkeo_centered), d=1.0/fs)
            mask_t = (freqs_t >= 0.3) & (freqs_t <= 3.5)
            if mask_t.sum() > 0:
                peak_idx_t = np.argmax(fft_vals_t[mask_t])
                tkeo_freqs_all.append(freqs_t[mask_t][peak_idx_t])

        # Peak count (pointiness peaks)
        mx = np.max(pt)
        if mx > 0:
            pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
            n_peaks = len(pks)
            duration = n_samples / fs
            if n_peaks > 1:
                peak_counts_all.append((n_peaks - 1) / duration)

    f_fft = np.median(fft_freqs_all) if fft_freqs_all else np.nan
    f_tkeo = np.median(tkeo_freqs_all) if tkeo_freqs_all else np.nan
    f_peaks = np.median(peak_counts_all) if peak_counts_all else np.nan

    return {
        'f_fft': round(f_fft, 3) if np.isfinite(f_fft) else np.nan,
        'f_tkeo': round(f_tkeo, 3) if np.isfinite(f_tkeo) else np.nan,
        'f_peaks': round(f_peaks, 3) if np.isfinite(f_peaks) else np.nan,
    }


def assign_frequency_bin(freq_est):
    """Assign a frequency bin label based on the frequency estimate."""
    if not np.isfinite(freq_est):
        return 'unknown'
    for lo, hi in FREQ_BINS:
        if lo <= freq_est < hi:
            return f'[{lo:.1f}, {hi:.1f})'
    return 'unknown'


def select_stratified(df, n_target, subtype_label):
    """
    Select up to n_target patients with frequency stratification across 5 bins.
    If n_target >= len(df), take all.
    """
    target_per_bin = max(1, n_target // len(FREQ_BINS))

    # Use f_fft as primary frequency estimate, fill with f_tkeo, then f_peaks
    df = df.copy()
    df['freq_est'] = df['f_fft']
    mask = df['freq_est'].isna()
    df.loc[mask, 'freq_est'] = df.loc[mask, 'f_tkeo']
    mask = df['freq_est'].isna()
    df.loc[mask, 'freq_est'] = df.loc[mask, 'f_peaks']

    # Assign bins
    df['frequency_bin'] = df['freq_est'].apply(assign_frequency_bin)

    # If we want all of them, just return all
    if n_target >= len(df):
        print(f"\n  Taking ALL {len(df)} available {subtype_label} patients:")
        for lo, hi in FREQ_BINS:
            bin_label = f'[{lo:.1f}, {hi:.1f})'
            count = (df['frequency_bin'] == bin_label).sum()
            print(f"    {bin_label} Hz: {count}")
        unk = (df['frequency_bin'] == 'unknown').sum()
        if unk > 0:
            print(f"    unknown: {unk}")
        return df

    # Otherwise, stratified selection
    selected = []
    remaining = df.copy()

    print(f"\n  Frequency bin selection for {subtype_label} (target {n_target}):")
    for lo, hi in FREQ_BINS:
        bin_label = f'[{lo:.1f}, {hi:.1f})'
        in_bin = remaining[remaining['frequency_bin'] == bin_label]
        n_available = len(in_bin)
        n_take = min(target_per_bin, n_available)
        if n_take > 0:
            picks = in_bin.sample(n=n_take, random_state=42)
            selected.append(picks)
            remaining = remaining[~remaining.index.isin(picks.index)]
        print(f"    {bin_label} Hz: {n_available} available, took {n_take}")

    # Include unknowns
    unknowns = remaining[remaining['frequency_bin'] == 'unknown']
    if len(unknowns) > 0:
        selected.append(unknowns)
        remaining = remaining[~remaining.index.isin(unknowns.index)]
        print(f"    unknown: took {len(unknowns)}")

    so_far = pd.concat(selected) if selected else pd.DataFrame()
    shortfall = n_target - len(so_far)
    if shortfall > 0 and len(remaining) > 0:
        extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=123)
        so_far = pd.concat([so_far, extra])
        print(f"    Filled {len(extra)} extra from remaining pool")

    print(f"  Total selected: {len(so_far)}")
    return so_far


def process_subtype(segments, labels, patients, prev_pids, label_val, subtype_name, n_target):
    """Process one subtype (LPD or GPD): filter, pick best segment, compute freqs, select."""

    # Filter to this subtype
    mask = labels == label_val
    sub_segments = segments[mask]
    sub_patients = patients[mask]
    unique_all = np.unique(sub_patients)
    print(f"\n  {subtype_name}: {len(sub_segments)} segments, {len(unique_all)} unique patients")

    # Exclude previously annotated
    exclude_mask = np.array([str(p) in prev_pids for p in sub_patients])
    n_excluded = exclude_mask.sum()
    sub_segments = sub_segments[~exclude_mask]
    sub_patients = sub_patients[~exclude_mask]
    unique_remaining = np.unique(sub_patients)
    print(f"  Excluded {n_excluded} segments from previously annotated patients")
    print(f"  Remaining: {len(sub_segments)} segments, {len(unique_remaining)} unique patients")

    if len(unique_remaining) == 0:
        print(f"  No unannotated {subtype_name} patients available!")
        return pd.DataFrame(), sub_segments, sub_patients

    # Pick 1 segment per patient (highest variance)
    best_idx = {}
    for i, pid in enumerate(sub_patients):
        var = np.var(sub_segments[i])
        pid_str = str(pid)
        if pid_str not in best_idx or var > best_idx[pid_str][1]:
            best_idx[pid_str] = (i, var)

    selected_indices = [v[0] for v in best_idx.values()]
    selected_pats = [str(sub_patients[i]) for i in selected_indices]
    selected_segs = [sub_segments[i] for i in selected_indices]
    print(f"  Selected {len(selected_pats)} segments (1 per patient)")

    # Compute frequency estimates
    print(f"  Computing frequency estimates...")
    records = []
    for i, (pid, seg) in enumerate(zip(selected_pats, selected_segs)):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"    Processing {i+1}/{len(selected_pats)}...")
        freqs = compute_frequency_estimates(seg, fs=FS)
        records.append({
            'patient_id': pid,
            'seg_idx': selected_indices[i],
            'subtype': subtype_name,
            **freqs,
        })

    df = pd.DataFrame(records)

    valid_fft = df['f_fft'].dropna()
    valid_tkeo = df['f_tkeo'].dropna()
    valid_peaks = df['f_peaks'].dropna()
    print(f"\n  Frequency stats ({subtype_name}):")
    if len(valid_fft) > 0:
        print(f"    f_fft:   median={valid_fft.median():.3f}, range=[{valid_fft.min():.3f}, {valid_fft.max():.3f}]")
    if len(valid_tkeo) > 0:
        print(f"    f_tkeo:  median={valid_tkeo.median():.3f}, range=[{valid_tkeo.min():.3f}, {valid_tkeo.max():.3f}]")
    if len(valid_peaks) > 0:
        print(f"    f_peaks: median={valid_peaks.median():.3f}, range=[{valid_peaks.min():.3f}, {valid_peaks.max():.3f}]")

    # Select with stratification
    selected_df = select_stratified(df, n_target, subtype_name)

    return selected_df, sub_segments, sub_patients


def main():
    print("=" * 60)
    print("Round 4: GPD (all) + LPD (up to 30) annotation selection")
    print("=" * 60)

    # 1. Load cached segments
    print("\n[1] Loading cached external segments...")
    npz = np.load(str(CACHE / 'external_pd_segments.npz'), allow_pickle=True)
    segments = npz['segments']  # (3816, 18, 2000)
    labels = npz['labels']      # 0=LPD, 1=GPD
    patients = npz['patients']  # patient IDs
    print(f"  Loaded {len(segments)} segments ({(labels==0).sum()} LPD, {(labels==1).sum()} GPD)")
    print(f"  {len(np.unique(patients))} unique patients total")

    # 2. Load previously annotated
    print("\n[2] Loading previously annotated patient IDs...")
    prev_pids = load_previously_annotated()

    # 3. Process GPD (take ALL available)
    print("\n[3] Processing GPD candidates (take all available)...")
    gpd_selected, gpd_segments, gpd_patients = process_subtype(
        segments, labels, patients, prev_pids,
        label_val=1, subtype_name='gpd', n_target=200  # large number = take all
    )

    # 4. Process LPD (up to 30)
    print("\n[4] Processing LPD candidates (up to 30)...")
    lpd_selected, lpd_segments, lpd_patients = process_subtype(
        segments, labels, patients, prev_pids,
        label_val=0, subtype_name='lpd', n_target=30
    )

    # 5. Save .mat files
    print("\n[5] Saving .mat files...")
    manifest_rows = []

    # GPD
    gpd_dir = OUT / 'gpd'
    gpd_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(gpd_selected.iterrows(), 1):
        pid = row['patient_id']
        seg_idx = int(row['seg_idx'])

        # Find the segment in the filtered arrays
        # seg_idx is relative to the filtered (non-excluded) subtype array
        seg_data = gpd_segments[seg_idx]

        file_name = f"r4_gpd_{rank:03d}_pat{pid}"
        mat_path = gpd_dir / f"{file_name}.mat"

        scipy.io.savemat(str(mat_path), {
            'data': seg_data.astype(np.float64),
            'Fs': np.array([[FS]]),
        })

        consensus = np.nanmedian([row['f_fft'], row['f_tkeo'], row['f_peaks']])
        disagreement = np.nanstd([row['f_fft'], row['f_tkeo'], row['f_peaks']])

        manifest_rows.append({
            'patient_id': pid,
            'file_name': file_name,
            'subtype': 'gpd',
            'f_fft': f"{row['f_fft']:.3f}" if np.isfinite(row['f_fft']) else '',
            'f_tkeo': f"{row['f_tkeo']:.3f}" if np.isfinite(row['f_tkeo']) else '',
            'f_peaks': f"{row['f_peaks']:.3f}" if np.isfinite(row['f_peaks']) else '',
            'frequency_bin': row.get('frequency_bin', ''),
            'disagreement': f"{disagreement:.3f}" if np.isfinite(disagreement) else '',
            'consensus_estimate': f"{consensus:.3f}" if np.isfinite(consensus) else '',
        })

    print(f"  Saved {len(gpd_selected)} GPD .mat files to {gpd_dir}")

    # LPD
    lpd_dir = OUT / 'lpd'
    lpd_dir.mkdir(parents=True, exist_ok=True)
    for rank, (_, row) in enumerate(lpd_selected.iterrows(), 1):
        pid = row['patient_id']
        seg_idx = int(row['seg_idx'])

        seg_data = lpd_segments[seg_idx]

        file_name = f"r4_lpd_{rank:03d}_pat{pid}"
        mat_path = lpd_dir / f"{file_name}.mat"

        scipy.io.savemat(str(mat_path), {
            'data': seg_data.astype(np.float64),
            'Fs': np.array([[FS]]),
        })

        consensus = np.nanmedian([row['f_fft'], row['f_tkeo'], row['f_peaks']])
        disagreement = np.nanstd([row['f_fft'], row['f_tkeo'], row['f_peaks']])

        manifest_rows.append({
            'patient_id': pid,
            'file_name': file_name,
            'subtype': 'lpd',
            'f_fft': f"{row['f_fft']:.3f}" if np.isfinite(row['f_fft']) else '',
            'f_tkeo': f"{row['f_tkeo']:.3f}" if np.isfinite(row['f_tkeo']) else '',
            'f_peaks': f"{row['f_peaks']:.3f}" if np.isfinite(row['f_peaks']) else '',
            'frequency_bin': row.get('frequency_bin', ''),
            'disagreement': f"{disagreement:.3f}" if np.isfinite(disagreement) else '',
            'consensus_estimate': f"{consensus:.3f}" if np.isfinite(consensus) else '',
        })

    print(f"  Saved {len(lpd_selected)} LPD .mat files to {lpd_dir}")

    # 6. Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  Saved manifest ({len(manifest_rows)} rows) to {manifest_path}")

    # Also save as JSON for HTML viewer
    manifest_json_path = OUT / 'manifest.json'
    with open(manifest_json_path, 'w') as f:
        json.dump(manifest_rows, f)

    # 7. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_gpd = len(gpd_selected)
    n_lpd = len(lpd_selected)
    print(f"  GPD candidates: {n_gpd}")
    print(f"  LPD candidates: {n_lpd}")
    print(f"  Total candidates: {n_gpd + n_lpd}")

    print(f"\n  GPD frequency distribution:")
    if len(gpd_selected) > 0:
        for lo, hi in FREQ_BINS:
            bin_label = f'[{lo:.1f}, {hi:.1f})'
            count = (gpd_selected['frequency_bin'] == bin_label).sum()
            print(f"    {bin_label} Hz: {count}")
        unk = (gpd_selected['frequency_bin'] == 'unknown').sum()
        if unk > 0:
            print(f"    unknown: {unk}")

    print(f"\n  LPD frequency distribution:")
    if len(lpd_selected) > 0:
        for lo, hi in FREQ_BINS:
            bin_label = f'[{lo:.1f}, {hi:.1f})'
            count = (lpd_selected['frequency_bin'] == bin_label).sum()
            print(f"    {bin_label} Hz: {count}")
        unk = (lpd_selected['frequency_bin'] == 'unknown').sum()
        if unk > 0:
            print(f"    unknown: {unk}")

    print(f"\n  Output directory: {OUT}")
    print(f"  Next steps:")
    print(f"    1. Run: conda run -n foe python code/generate_round4_images.py")
    print(f"    2. Run: python code/build_round4_viewer.py")


if __name__ == '__main__':
    main()
