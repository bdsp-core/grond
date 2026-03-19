"""
Round 3 annotation selection: 50 LPD patients for frequency annotation.
Focuses exclusively on LPD to balance the dataset.

Steps:
1. Load cached external segments (3816 segments, 18ch bipolar, 2000 samples @ 200Hz)
2. Exclude previously annotated patients (canonical + round2)
3. Filter to LPD only
4. Pick 1 segment per patient
5. Compute quick frequency estimates (pointiness FFT, TKEO FFT, peak count)
6. Select 50 patients with frequency diversity across 5 bins
7. Save .mat files and manifest.csv
"""

import sys, os, json
import numpy as np
import pandas as pd
import scipy.io
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
CACHE = DATA / 'dl_cache'
OUT = DATA / '_archive' / 'annotation_round3'
FS = 200  # sampling rate

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
    cand_path = DATA / '_archive' / 'annotation_candidates' / 'manifest.csv'
    if cand_path.exists():
        df = pd.read_csv(cand_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  annotation_candidates: {len(df)} rows")

    # Round 2
    r2_path = DATA / '_archive' / 'annotation_round2' / 'manifest.csv'
    if r2_path.exists():
        df = pd.read_csv(r2_path)
        pids.update(df['patient_id'].astype(str).tolist())
        print(f"  annotation_round2: {len(df)} rows")

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


def select_diverse(df, n=50):
    """Select n patients targeting frequency diversity across 5 bins."""
    bins = [
        (0.0, 0.6),
        (0.6, 0.8),
        (0.8, 1.2),
        (1.2, 2.0),
        (2.0, 10.0),
    ]
    target_per_bin = n // len(bins)  # 10

    # Use f_fft as primary frequency estimate
    df = df.copy()
    df['freq_est'] = df['f_fft']
    # Fill NaN with f_tkeo, then f_peaks
    mask = df['freq_est'].isna()
    df.loc[mask, 'freq_est'] = df.loc[mask, 'f_tkeo']
    mask = df['freq_est'].isna()
    df.loc[mask, 'freq_est'] = df.loc[mask, 'f_peaks']

    selected = []
    remaining = df.copy()

    print(f"\n  Frequency bin selection:")
    for lo, hi in bins:
        in_bin = remaining[(remaining['freq_est'] >= lo) & (remaining['freq_est'] < hi)]
        n_available = len(in_bin)
        n_take = min(target_per_bin, n_available)
        if n_take > 0:
            picks = in_bin.sample(n=n_take, random_state=42)
            selected.append(picks)
            remaining = remaining[~remaining.index.isin(picks.index)]
        print(f"    [{lo:.1f}, {hi:.1f}) Hz: {n_available} available, took {n_take}")

    # Fill from remaining if we haven't reached n
    so_far = pd.concat(selected) if selected else pd.DataFrame()
    shortfall = n - len(so_far)
    if shortfall > 0 and len(remaining) > 0:
        # Fill proportionally from adjacent bins with surplus
        extra = remaining.sample(n=min(shortfall, len(remaining)), random_state=123)
        so_far = pd.concat([so_far, extra])
        print(f"    Filled {len(extra)} extra from remaining pool")

    print(f"  Total selected: {len(so_far)}")
    return so_far


def main():
    print("=" * 60)
    print("Round 3: LPD-only annotation selection (50 patients)")
    print("=" * 60)

    # 1. Load cached segments
    print("\n[1] Loading cached external segments...")
    npz = np.load(str(CACHE / 'external_pd_segments.npz'), allow_pickle=True)
    segments = npz['segments']  # (3816, 18, 2000)
    labels = npz['labels']      # 0=LPD, 1=GPD
    patients = npz['patients']  # patient IDs
    print(f"  Loaded {len(segments)} segments ({(labels==0).sum()} LPD, {(labels==1).sum()} GPD)")

    # 2. Load previously annotated
    print("\n[2] Loading previously annotated patient IDs...")
    prev_pids = load_previously_annotated()

    # 3. Filter to LPD only
    print("\n[3] Filtering to LPD segments only...")
    lpd_mask = labels == 0
    lpd_segments = segments[lpd_mask]
    lpd_patients = patients[lpd_mask]
    print(f"  {len(lpd_segments)} LPD segments, {len(np.unique(lpd_patients))} unique patients")

    # 4. Exclude previously annotated
    print("\n[4] Excluding previously annotated patients...")
    exclude_mask = np.array([p in prev_pids for p in lpd_patients])
    n_excluded = exclude_mask.sum()
    lpd_segments = lpd_segments[~exclude_mask]
    lpd_patients = lpd_patients[~exclude_mask]
    unique_remaining = np.unique(lpd_patients)
    print(f"  Excluded {n_excluded} segments from previously annotated patients")
    print(f"  Remaining: {len(lpd_segments)} segments, {len(unique_remaining)} unique patients")

    # 5. Pick 1 segment per patient (the one with highest signal variance)
    print("\n[5] Picking 1 segment per patient...")
    best_idx = {}
    for i, pid in enumerate(lpd_patients):
        var = np.var(lpd_segments[i])
        if pid not in best_idx or var > best_idx[pid][1]:
            best_idx[pid] = (i, var)

    selected_indices = [v[0] for v in best_idx.values()]
    selected_pats = [lpd_patients[i] for i in selected_indices]
    selected_segs = [lpd_segments[i] for i in selected_indices]
    print(f"  Selected {len(selected_pats)} segments (1 per patient)")

    # 6. Compute frequency estimates
    print("\n[6] Computing frequency estimates...")
    records = []
    for i, (pid, seg) in enumerate(zip(selected_pats, selected_segs)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(selected_pats)}...")
        freqs = compute_frequency_estimates(seg, fs=FS)
        records.append({
            'patient_id': pid,
            'seg_idx': selected_indices[i],
            **freqs,
        })

    df = pd.DataFrame(records)
    print(f"\n  Frequency stats:")
    print(f"    f_fft:   median={df['f_fft'].median():.3f}, range=[{df['f_fft'].min():.3f}, {df['f_fft'].max():.3f}]")
    print(f"    f_tkeo:  median={df['f_tkeo'].median():.3f}, range=[{df['f_tkeo'].min():.3f}, {df['f_tkeo'].max():.3f}]")
    print(f"    f_peaks: median={df['f_peaks'].median():.3f}, range=[{df['f_peaks'].min():.3f}, {df['f_peaks'].max():.3f}]")

    # 7. Select 50 with frequency diversity
    print("\n[7] Selecting 50 patients with frequency diversity...")
    selected = select_diverse(df, n=50)

    # 8. Save .mat files
    print("\n[8] Saving .mat files...")
    lpd_dir = OUT / 'lpd'
    lpd_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for rank, (_, row) in enumerate(selected.iterrows(), 1):
        pid = row['patient_id']
        seg_idx = int(row['seg_idx'])
        seg_data = lpd_segments[seg_idx]

        file_name = f"r3_lpd_{rank:03d}_pat{pid}"
        mat_path = lpd_dir / f"{file_name}.mat"

        # Save as (20, 2000) monopolar-like format for compatibility
        # Actually the cached data is already bipolar (18, 2000)
        # Save with Fs and data keys
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
            'disagreement': f"{disagreement:.3f}" if np.isfinite(disagreement) else '',
            'consensus_estimate': f"{consensus:.3f}" if np.isfinite(consensus) else '',
        })

    # 9. Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUT / 'manifest.csv'
    manifest_df.to_csv(manifest_path, index=False)
    print(f"  Saved {len(manifest_rows)} .mat files to {lpd_dir}")
    print(f"  Saved manifest to {manifest_path}")

    # Also save the manifest as JSON for the HTML viewer
    manifest_json_path = OUT / 'manifest.json'
    with open(manifest_json_path, 'w') as f:
        json.dump(manifest_rows, f)

    print("\n  Done! Next step: run generate_round3_images.py with conda run -n foe")


if __name__ == '__main__':
    main()
