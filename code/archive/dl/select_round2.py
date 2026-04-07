"""
Round 2 annotation selection: 50 NEW patients targeting frequency diversity.

Computes consensus frequency per segment from cached external segments,
then selects 50 patients stratified by frequency bin to fill gaps in the
existing annotation set (especially extreme frequencies).

Run: conda run -n foe_dl python code/dl/select_round2.py
"""

import sys
import os
import time
import warnings
import csv
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat

warnings.filterwarnings('ignore')

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_pointiness_acf import (
    compute_pointiness_trace, compute_acf_frequency,
    bipolar_channels,
)

# ── Constants ──────────────────────────────────────────────────────────
FS = 200
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
LOWPASS_HZ = 15.0
TKEO_SIGMA = 0.02

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
OUTPUT_DIR = PROJECT_DIR / 'data' / '_archive' / 'pd_round2'

# Target distribution
BIN_TARGETS = [
    (0.0,  0.5,  10, 'slow_<0.5'),
    (0.5,  0.75, 10, '0.5-0.75'),
    (0.75, 1.0,   5, '0.75-1.0'),
    (1.0,  1.5,   5, '1.0-1.5'),
    (1.5,  2.5,  10, '1.5-2.5'),
    (2.5,  10.0, 10, 'fast_>2.5'),
]
TOTAL_TARGET = 50


# ── Helpers ────────────────────────────────────────────────────────────
def median_finite(arr):
    valid = arr[np.isfinite(arr)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def compute_tkeo(x):
    return np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])


# ── Frequency estimators (same as round 1) ────────────────────────────
def estimate_f_B(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
    return median_finite(freqs)


def estimate_f_peaks(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            span = (peak_locs[-1] - peak_locs[0]) / fs
            if span > 0:
                freqs[i] = (len(peak_locs) - 1) / span
    return median_finite(freqs)


def estimate_f_fft(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def estimate_f_tkeo(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(TKEO_SIGMA * fs))
    for i in range(n_ch):
        tkeo = compute_tkeo(seg[i])
        if len(tkeo) < 10:
            continue
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        if np.max(tkeo_smooth) <= 0:
            continue
        n = len(tkeo_smooth)
        fft_vals = np.abs(np.fft.rfft(tkeo_smooth - np.mean(tkeo_smooth)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def compute_consensus(f_B, f_peaks, f_fft, f_tkeo):
    vals = np.array([f_B, f_peaks, f_fft, f_tkeo])
    valid = vals[np.isfinite(vals)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


def compute_disagreement(f_B, f_peaks, f_fft, f_tkeo):
    vals = np.array([f_B, f_peaks, f_fft, f_tkeo])
    valid = vals[np.isfinite(vals)]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("Round 2: Select 50 Patients for Frequency-Diverse Annotation")
    print("=" * 60)

    # ── 1. Load external segments ─────────────────────────────────────
    print("\n[1] Loading external segments...")
    ext_path = CACHE_DIR / 'external_pd_segments.npz'
    ext = np.load(str(ext_path), allow_pickle=True)
    segments = ext['segments']    # (N, 18, 2000)
    labels = ext['labels']        # 0=LPD, 1=GPD
    patients = ext['patients']    # patient IDs
    N = len(segments)
    print(f"  {N} segments, {len(np.unique(patients))} unique patients")
    print(f"  LPD: {np.sum(labels == 0)}, GPD: {np.sum(labels == 1)}")

    # ── 2. Build exclusion set ────────────────────────────────────────
    print("\n[2] Building exclusion set...")
    exclude_patients = set()

    # Original 43 annotated patients (different ID format, but check anyway)
    ann_path = CACHE_DIR / 'annotated_pd_data.npz'
    ann = np.load(str(ann_path), allow_pickle=True)
    for pid in ann['patients']:
        exclude_patients.add(str(pid))
    print(f"  Original annotated: {len(ann['patients'])} segments, {len(set(ann['patients']))} unique patients")

    # Round 1: 77 new patients from frequency_annotations.csv
    r1_csv = PROJECT_DIR / 'data' / '_archive' / 'pd_round1_candidates' / 'frequency_annotations.csv'
    r1_patients = set()
    if r1_csv.exists():
        with open(str(r1_csv)) as f:
            for row in csv.DictReader(f):
                r1_patients.add(row['patient_id'])
                exclude_patients.add(row['patient_id'])
        print(f"  Round 1 patients: {len(r1_patients)}")
    else:
        # Fall back to manifest.csv
        r1_manifest = PROJECT_DIR / 'data' / '_archive' / 'pd_round1_candidates' / 'manifest.csv'
        if r1_manifest.exists():
            with open(str(r1_manifest)) as f:
                for row in csv.DictReader(f):
                    r1_patients.add(row['patient_id'])
                    exclude_patients.add(row['patient_id'])
            print(f"  Round 1 patients (from manifest): {len(r1_patients)}")

    print(f"  Total excluded patient IDs: {len(exclude_patients)}")

    # ── 3. Compute frequency estimates per segment ────────────────────
    print(f"\n[3] Computing frequency estimates for {N} segments...")
    results = []
    for idx in range(N):
        seg = segments[idx]

        f_B = estimate_f_B(seg, FS)
        f_peaks = estimate_f_peaks(seg, FS)
        f_fft = estimate_f_fft(seg, FS)
        f_tkeo = estimate_f_tkeo(seg, FS)

        consensus = compute_consensus(f_B, f_peaks, f_fft, f_tkeo)
        disagreement = compute_disagreement(f_B, f_peaks, f_fft, f_tkeo)

        results.append({
            'idx': idx,
            'patient_id': str(patients[idx]),
            'subtype': 'lpd' if labels[idx] == 0 else 'gpd',
            'f_B': f_B,
            'f_peaks': f_peaks,
            'f_fft': f_fft,
            'f_tkeo': f_tkeo,
            'disagreement': disagreement,
            'consensus': consensus,
        })

        if (idx + 1) % 200 == 0 or (idx + 1) == N:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{N} segments ({elapsed:.0f}s)")

    # ── 4. Select best segment per patient (exclude annotated) ────────
    print("\n[4] Selecting one segment per patient...")
    patient_best = {}
    for r in results:
        pid = r['patient_id']
        if pid in exclude_patients:
            continue
        if np.isnan(r['consensus']):
            continue
        # Pick segment closest to extreme frequency if multiple, else highest disagreement
        if pid not in patient_best or r['disagreement'] > patient_best[pid]['disagreement']:
            patient_best[pid] = r

    print(f"  {len(patient_best)} eligible patients (after exclusion)")

    # ── 5. Stratified selection by frequency bins ─────────────────────
    print("\n[5] Stratified selection targeting frequency diversity...")

    # Separate into bins
    bin_pools = {}
    for lo, hi, target, name in BIN_TARGETS:
        pool = [r for r in patient_best.values() if lo <= r['consensus'] < hi]
        # Sort by how extreme the frequency is (distance from center of bin)
        center = (lo + hi) / 2
        pool.sort(key=lambda x: abs(x['consensus'] - center))
        bin_pools[name] = pool
        print(f"  Bin [{lo:.1f}, {hi:.1f}): {len(pool)} available, target {target}")

    selected_pids = set()
    selected = []

    # First pass: fill each bin up to target
    for lo, hi, target, name in BIN_TARGETS:
        pool = bin_pools[name]
        count = 0
        for r in pool:
            if r['patient_id'] in selected_pids:
                continue
            selected.append(r)
            selected_pids.add(r['patient_id'])
            count += 1
            if count >= target:
                break
        print(f"  Bin {name}: selected {count}/{target}")

    # Second pass: fill remaining from adjacent bins
    remaining = TOTAL_TARGET - len(selected)
    if remaining > 0:
        print(f"\n  Filling {remaining} remaining slots from overflow bins...")
        # Prioritize extreme frequency patients
        all_remaining = [r for r in patient_best.values()
                         if r['patient_id'] not in selected_pids]
        # Sort by distance from median (1.0 Hz) - want extremes first
        all_remaining.sort(key=lambda x: -abs(x['consensus'] - 1.0))
        for r in all_remaining:
            if len(selected) >= TOTAL_TARGET:
                break
            selected.append(r)
            selected_pids.add(r['patient_id'])

    print(f"\n  Total selected: {len(selected)}")

    # Print distribution of selected
    print("\n  Selected frequency distribution:")
    for lo, hi, target, name in BIN_TARGETS:
        count = sum(1 for r in selected if lo <= r['consensus'] < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {count} selected (target: {target})")

    lpd_count = sum(1 for r in selected if r['subtype'] == 'lpd')
    gpd_count = sum(1 for r in selected if r['subtype'] == 'gpd')
    print(f"\n  LPD: {lpd_count}, GPD: {gpd_count}")

    # ── 6. Save .mat files and manifest ───────────────────────────────
    print(f"\n[6] Saving to {OUTPUT_DIR}...")
    for subdir in ['lpd', 'gpd', 'images']:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Sort selected by consensus frequency for nice ordering
    selected.sort(key=lambda x: x['consensus'])

    manifest_rows = []
    for i, r in enumerate(selected):
        seg_idx = r['idx']
        subtype = r['subtype']
        patient_id = r['patient_id']

        file_name = f"r2_{subtype}_{i + 1:03d}_pat{patient_id}"

        # Save raw bipolar EEG as .mat
        mat_path = OUTPUT_DIR / subtype / f"{file_name}.mat"
        seg_data = segments[seg_idx]
        savemat(str(mat_path), {
            'data': seg_data,
            'fs': FS,
            'channels': bipolar_channels,
            'patient_id': patient_id,
        })

        manifest_rows.append({
            'patient_id': patient_id,
            'file_name': file_name,
            'subtype': subtype,
            'f_B': f'{r["f_B"]:.3f}' if np.isfinite(r['f_B']) else '',
            'f_peaks': f'{r["f_peaks"]:.3f}' if np.isfinite(r['f_peaks']) else '',
            'f_fft': f'{r["f_fft"]:.3f}' if np.isfinite(r['f_fft']) else '',
            'f_tkeo': f'{r["f_tkeo"]:.3f}' if np.isfinite(r['f_tkeo']) else '',
            'disagreement': f'{r["disagreement"]:.3f}',
            'consensus_estimate': f'{r["consensus"]:.3f}' if np.isfinite(r['consensus']) else '',
        })

    # Write manifest CSV
    manifest_path = OUTPUT_DIR / 'manifest.csv'
    fieldnames = ['patient_id', 'file_name', 'subtype', 'f_B', 'f_peaks',
                  'f_fft', 'f_tkeo', 'disagreement', 'consensus_estimate']
    with open(str(manifest_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Manifest: {manifest_path} ({len(manifest_rows)} rows)")

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s")
    freqs = [r['consensus'] for r in selected]
    print(f"  Frequency range: {min(freqs):.3f} - {max(freqs):.3f} Hz")
    print(f"  Median: {np.median(freqs):.3f} Hz")
    print(f"\nNext steps:")
    print(f"  1. Generate PNGs: conda run -n foe python code/dl/generate_round2_pngs.py")
    print(f"  2. Open viewer:   open data/pd_round2/annotation_viewer.html")


if __name__ == '__main__':
    main()
