"""
Active learning: select most informative external segments for expert annotation.

Computes 5 frequency estimates per segment, selects the most uncertain (highest
inter-method disagreement) per patient, then picks top 50 LPD + 30 GPD candidates.

Run: conda run -n foe_dl python code/dl/select_for_annotation.py
"""

import sys
import os
import time
import warnings
import numpy as np
import csv
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence as scipy_coherence
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
TKEO_PEAK_MIN_DIST = 0.2

# Adjacent pairs for spectral coherence
LEFT_TEMPORAL_PAIRS = [(0, 1), (1, 2), (2, 3)]
RIGHT_TEMPORAL_PAIRS = [(4, 5), (5, 6), (6, 7)]
LEFT_PARASAG_PAIRS = [(8, 9), (9, 10), (10, 11)]
RIGHT_PARASAG_PAIRS = [(12, 13), (13, 14), (14, 15)]
MIDLINE_PAIRS = [(16, 17)]
ALL_ADJACENT_PAIRS = (LEFT_TEMPORAL_PAIRS + RIGHT_TEMPORAL_PAIRS +
                      LEFT_PARASAG_PAIRS + RIGHT_PARASAG_PAIRS + MIDLINE_PAIRS)

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
OUTPUT_DIR = PROJECT_DIR / 'data' / '_archive' / 'annotation_candidates'

N_LPD = 50
N_GPD = 30


# ── Helpers ────────────────────────────────────────────────────────────
def median_finite(arr):
    valid = arr[np.isfinite(arr)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def compute_tkeo(x):
    """Teager-Kaiser Energy Operator: |x[n]^2 - x[n-1]*x[n+1]|"""
    return np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])


# ── Per-segment frequency estimators ──────────────────────────────────
def estimate_f_B(seg, fs):
    """f_B: ACF of pointiness trace, threshold=0.10, median across channels."""
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
    """f_peaks: peak-count frequency on pointiness trace, median across channels."""
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
    """f_fft: FFT of pointiness trace, peak in [0.3, 3.5] Hz, median across channels."""
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
    """f_tkeo: FFT of TKEO trace, peak in [0.3, 3.5] Hz, median across channels."""
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


def estimate_f_coh(seg, fs):
    """f_coh: peak of average spectral coherence across adjacent pairs in [0.3, 3.5] Hz."""
    n_samples = seg.shape[1]
    nperseg = min(256, n_samples // 2)
    if nperseg < 16:
        return np.nan

    sum_coh = None
    count = 0
    freqs_out = None

    for (ch_a, ch_b) in ALL_ADJACENT_PAIRS:
        if ch_a >= seg.shape[0] or ch_b >= seg.shape[0]:
            continue
        try:
            f, cxy = scipy_coherence(seg[ch_a], seg[ch_b], fs=fs, nperseg=nperseg)
            if sum_coh is None:
                sum_coh = np.zeros_like(cxy)
                freqs_out = f
            sum_coh += cxy
            count += 1
        except Exception:
            continue

    if count == 0 or sum_coh is None:
        return np.nan

    avg_coh = sum_coh / count
    mask = (freqs_out >= 0.3) & (freqs_out <= 3.5)
    if not np.any(mask):
        return np.nan

    coh_sub = avg_coh[mask]
    freq_sub = freqs_out[mask]
    return float(freq_sub[np.argmax(coh_sub)])


def compute_disagreement(f_B, f_peaks, f_fft, f_tkeo, f_coh):
    """NaN-aware standard deviation of the 5 frequency estimates."""
    vals = np.array([f_B, f_peaks, f_fft, f_tkeo, f_coh])
    valid = vals[np.isfinite(vals)]
    if len(valid) < 2:
        return 0.0  # Not enough estimates to disagree
    return float(np.std(valid))


def compute_consensus(f_B, f_peaks, f_fft, f_tkeo, f_coh):
    """NaN-aware median of the 5 frequency estimates."""
    vals = np.array([f_B, f_peaks, f_fft, f_tkeo, f_coh])
    valid = vals[np.isfinite(vals)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("Active Learning: Select Segments for Expert Annotation")
    print("=" * 60)

    # ── 1. Load external segments ─────────────────────────────────────
    print("\n[1] Loading external segments...")
    ext_path = CACHE_DIR / 'external_pd_segments.npz'
    ext = np.load(str(ext_path), allow_pickle=True)
    segments = ext['segments']    # (N, 18, 2000) already preprocessed bipolar
    labels = ext['labels']        # 0=LPD, 1=GPD
    patients = ext['patients']    # patient IDs
    N = len(segments)
    n_patients = len(np.unique(patients))
    print(f"  {N} segments, {n_patients} patients")
    print(f"  LPD: {np.sum(labels == 0)}, GPD: {np.sum(labels == 1)}")

    # ── 2. Load annotated patients (to exclude) ──────────────────────
    print("\n[2] Loading annotated patient list (to exclude)...")
    ann_path = CACHE_DIR / 'annotated_pd_data.npz'
    ann = np.load(str(ann_path), allow_pickle=True)
    annotated_patients = set(ann['patients'])
    print(f"  {len(annotated_patients)} annotated patients")
    # Note: patient ID formats differ (external='112968514', annotated='abn1047')
    # so there should be no overlap, but we check anyway
    overlap = set(patients) & annotated_patients
    print(f"  Overlap with external: {len(overlap)} patients")

    # ── 3. Compute 5 frequency estimates per segment ─────────────────
    print(f"\n[3] Computing frequency estimates for {N} segments...")
    results = []
    for idx in range(N):
        seg = segments[idx]  # (18, 2000)

        f_B = estimate_f_B(seg, FS)
        f_peaks = estimate_f_peaks(seg, FS)
        f_fft = estimate_f_fft(seg, FS)
        f_tkeo = estimate_f_tkeo(seg, FS)
        f_coh = estimate_f_coh(seg, FS)

        disagreement = compute_disagreement(f_B, f_peaks, f_fft, f_tkeo, f_coh)
        consensus = compute_consensus(f_B, f_peaks, f_fft, f_tkeo, f_coh)

        results.append({
            'idx': idx,
            'patient_id': patients[idx],
            'subtype': 'lpd' if labels[idx] == 0 else 'gpd',
            'f_B': f_B,
            'f_peaks': f_peaks,
            'f_fft': f_fft,
            'f_tkeo': f_tkeo,
            'f_coh': f_coh,
            'disagreement': disagreement,
            'consensus_estimate': consensus,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == N:
            elapsed = time.time() - t0
            print(f"  {idx + 1}/{N} segments ({elapsed:.0f}s)")

    # ── 4. Select 1 segment per patient (highest disagreement) ───────
    print("\n[4] Selecting best segment per patient (highest disagreement)...")
    patient_best = {}
    for r in results:
        pid = r['patient_id']
        if pid in annotated_patients:
            continue
        if pid not in patient_best or r['disagreement'] > patient_best[pid]['disagreement']:
            patient_best[pid] = r

    print(f"  {len(patient_best)} patients after excluding annotated")

    # ── 5. Separate by subtype, rank by disagreement ─────────────────
    print("\n[5] Ranking by disagreement per subtype...")
    lpd_candidates = sorted(
        [r for r in patient_best.values() if r['subtype'] == 'lpd'],
        key=lambda x: x['disagreement'], reverse=True
    )
    gpd_candidates = sorted(
        [r for r in patient_best.values() if r['subtype'] == 'gpd'],
        key=lambda x: x['disagreement'], reverse=True
    )
    print(f"  LPD candidates: {len(lpd_candidates)}")
    print(f"  GPD candidates: {len(gpd_candidates)}")

    selected_lpd = lpd_candidates[:N_LPD]
    selected_gpd = gpd_candidates[:N_GPD]
    selected = selected_lpd + selected_gpd
    print(f"  Selected: {len(selected_lpd)} LPD + {len(selected_gpd)} GPD = {len(selected)}")

    # ── 6. Save .mat files and manifest ──────────────────────────────
    print(f"\n[6] Saving candidates to {OUTPUT_DIR}...")
    for subdir in ['lpd', 'gpd', 'images']:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for i, r in enumerate(selected):
        seg_idx = r['idx']
        subtype = r['subtype']
        patient_id = r['patient_id']

        # File name
        rank = i + 1 if subtype == 'lpd' else i - len(selected_lpd) + 1
        if subtype == 'gpd':
            rank = selected.index(r) - len(selected_lpd) + 1
        file_name = f"{subtype}_{rank:03d}_pat{patient_id}"

        # Save raw bipolar EEG as .mat
        mat_path = OUTPUT_DIR / subtype / f"{file_name}.mat"
        seg_data = segments[seg_idx]  # (18, 2000)
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
            'f_coh': f'{r["f_coh"]:.3f}' if np.isfinite(r['f_coh']) else '',
            'disagreement': f'{r["disagreement"]:.3f}',
            'consensus_estimate': f'{r["consensus_estimate"]:.3f}' if np.isfinite(r['consensus_estimate']) else '',
        })

    # Write manifest CSV
    manifest_path = OUTPUT_DIR / 'manifest.csv'
    fieldnames = ['patient_id', 'file_name', 'subtype', 'f_B', 'f_peaks',
                  'f_fft', 'f_tkeo', 'f_coh', 'disagreement', 'consensus_estimate']
    with open(str(manifest_path), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"  Manifest: {manifest_path} ({len(manifest_rows)} rows)")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f}s")
    print(f"  LPD selected: {len(selected_lpd)}")
    print(f"  GPD selected: {len(selected_gpd)}")
    if selected_lpd:
        disag_lpd = [r['disagreement'] for r in selected_lpd]
        print(f"  LPD disagreement range: {min(disag_lpd):.3f} - {max(disag_lpd):.3f}")
    if selected_gpd:
        disag_gpd = [r['disagreement'] for r in selected_gpd]
        print(f"  GPD disagreement range: {min(disag_gpd):.3f} - {max(disag_gpd):.3f}")
    print(f"\nNext steps:")
    print(f"  1. Generate PNGs: conda run -n foe_dl python code/dl/generate_annotation_pngs.py")
    print(f"  2. Open viewer:   open code/dl/annotation_viewer.html")


if __name__ == '__main__':
    main()
