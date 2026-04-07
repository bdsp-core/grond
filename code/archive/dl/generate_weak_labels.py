"""
Generate weak eventness labels from existing detectors for Phase 2 supervision.
Run: conda run -n foe_dl python code/dl/generate_weak_labels.py
"""

import sys
import os
import time
import warnings
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, butter, filtfilt

warnings.filterwarnings('ignore')

# Setup paths
DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data
from pd_pointiness_acf import fcn_getBanana, compute_pointiness_trace
from data_loader import preprocess_segment

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
OUTPUT_PATH = CACHE_DIR / 'annotated_pd_data.npz'
FS = 200
TARGET_LEN = 2000  # 10s at 200 Hz


def extract_patient_id(mat_name):
    """Extract patient ID from mat filename (first part before underscore)."""
    stem = Path(mat_name).stem
    parts = stem.split('_')
    return parts[0] if parts else stem


def compute_agreement_trace(seg_bipolar, fs=200):
    """Compute cross-channel agreement trace for eventness.

    For each of 18 bipolar channels:
      - Compute pointiness trace
      - Smooth with gaussian_filter1d(sigma=4)
      - Find peaks (height=max*0.3, distance=40 samples)
    Then for each time step, count how many channels have a peak within +/-25 samples.
    Normalize, smooth, clip to [0, 1].

    Args:
        seg_bipolar: (18, N) preprocessed bipolar EEG

    Returns:
        eventness: (N,) agreement trace in [0, 1]
    """
    n_channels, n_samples = seg_bipolar.shape
    all_peak_locs = []

    for ch in range(n_channels):
        signal = seg_bipolar[ch]
        # Compute pointiness trace
        trace = compute_pointiness_trace(signal)
        # Smooth
        trace = gaussian_filter1d(trace, sigma=4)
        # Find peaks
        trace_max = np.max(trace)
        if trace_max <= 0:
            all_peak_locs.append(np.array([], dtype=int))
            continue
        peak_height = trace_max * 0.3
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=40)
        all_peak_locs.append(peak_locs)

    # Build agreement trace: for each time step, count channels with a peak within +/-25 samples
    agreement = np.zeros(n_samples, dtype=np.float32)
    window_radius = 25  # +/-25 samples = 50ms at 200Hz

    for ch_peaks in all_peak_locs:
        for pk in ch_peaks:
            start = max(0, pk - window_radius)
            end = min(n_samples, pk + window_radius + 1)
            agreement[start:end] += 1.0

    # Normalize by max
    max_agreement = np.max(agreement)
    if max_agreement > 0:
        agreement /= max_agreement

    # Smooth
    agreement = gaussian_filter1d(agreement, sigma=20)

    # Clip to [0, 1]
    agreement = np.clip(agreement, 0, 1)

    return agreement


def main():
    print("=" * 60)
    print("Generate Weak Eventness Labels for Phase 2")
    print("=" * 60)

    # ── Load annotated dataset ────────────────────────────────────────
    print("\n[1] Loading annotated dataset...")
    dataset = load_dataset()
    N = len(dataset)
    print(f"  Found {N} annotated segments")

    # ── Process each segment ──────────────────────────────────────────
    print(f"\n[2] Processing {N} segments...")
    all_segments = []
    all_expert_freqs = []
    all_weak_eventness = []
    all_patients = []
    all_subtypes = []
    processed = 0
    failed = 0

    t0 = time.time()
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing {idx+1}/{N}  ({elapsed:.0f}s elapsed, {failed} failed)")

        try:
            # Load raw EEG
            data, fs = load_eeg_data(entry)
            if data is None:
                failed += 1
                continue

            # Preprocess: bipolar montage + bandpass + lowpass
            seg_bipolar = preprocess_segment(data, fs)  # (18, N_samples)

            # Ensure exactly TARGET_LEN samples
            n_samples = seg_bipolar.shape[1]
            if n_samples < TARGET_LEN:
                pad = TARGET_LEN - n_samples
                seg_bipolar = np.pad(seg_bipolar, ((0, 0), (0, pad)), mode='constant')
            elif n_samples > TARGET_LEN:
                # Take center crop
                start = (n_samples - TARGET_LEN) // 2
                seg_bipolar = seg_bipolar[:, start:start + TARGET_LEN]

            # Compute agreement-based weak eventness labels
            weak_eventness = compute_agreement_trace(seg_bipolar, fs)

            # Ensure exact length after processing
            if len(weak_eventness) != TARGET_LEN:
                if len(weak_eventness) < TARGET_LEN:
                    weak_eventness = np.pad(weak_eventness, (0, TARGET_LEN - len(weak_eventness)),
                                            mode='constant')
                else:
                    weak_eventness = weak_eventness[:TARGET_LEN]

            # Expert frequencies: [LB, PH, SZ]
            expert_freqs = np.array([
                entry.get('expert_LB_freq', np.nan),
                entry.get('expert_PH_freq', np.nan),
                entry.get('expert_SZ_freq', np.nan),
            ], dtype=np.float32)

            # Patient ID
            patient_id = extract_patient_id(entry['mat_name'])

            # Subtype
            subtype = entry['subdir']  # 'lpd' or 'gpd'

            all_segments.append(seg_bipolar.astype(np.float32))
            all_expert_freqs.append(expert_freqs)
            all_weak_eventness.append(weak_eventness.astype(np.float32))
            all_patients.append(patient_id)
            all_subtypes.append(subtype)
            processed += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"    Warning: failed on {entry['mat_name']}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\n[3] Processing complete in {elapsed:.0f}s")
    print(f"  Successfully processed: {processed}")
    print(f"  Failed: {failed}")

    if processed == 0:
        print("ERROR: No segments were processed successfully.")
        sys.exit(1)

    # ── Convert to arrays ─────────────────────────────────────────────
    segments_arr = np.array(all_segments, dtype=np.float32)      # (N, 18, 2000)
    expert_freqs_arr = np.array(all_expert_freqs, dtype=np.float32)  # (N, 3)
    weak_eventness_arr = np.array(all_weak_eventness, dtype=np.float32)  # (N, 2000)
    patients_arr = np.array(all_patients)                        # (N,)
    subtypes_arr = np.array(all_subtypes)                        # (N,)

    print(f"\n[4] Array shapes:")
    print(f"  segments:       {segments_arr.shape}")
    print(f"  expert_freqs:   {expert_freqs_arr.shape}")
    print(f"  weak_eventness: {weak_eventness_arr.shape}")
    print(f"  patients:       {patients_arr.shape}")
    print(f"  subtypes:       {subtypes_arr.shape}")

    # ── Summary statistics ────────────────────────────────────────────
    n_lpd = np.sum(subtypes_arr == 'lpd')
    n_gpd = np.sum(subtypes_arr == 'gpd')
    n_patients_unique = len(np.unique(patients_arr))
    print(f"\n  LPD: {n_lpd}, GPD: {n_gpd}")
    print(f"  Unique patients: {n_patients_unique}")

    # Expert frequency coverage
    for i, name in enumerate(['LB', 'PH', 'SZ']):
        n_valid = np.sum(np.isfinite(expert_freqs_arr[:, i]) & (expert_freqs_arr[:, i] > 0))
        print(f"  Expert {name}: {n_valid}/{processed} valid frequencies")

    # Eventness statistics
    mean_eventness = np.mean(weak_eventness_arr)
    max_eventness = np.max(weak_eventness_arr)
    n_nonzero = np.sum(np.any(weak_eventness_arr > 0.1, axis=1))
    print(f"  Eventness: mean={mean_eventness:.3f}, max={max_eventness:.3f}, "
          f"segments with activity={n_nonzero}/{processed}")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(str(CACHE_DIR), exist_ok=True)
    np.savez_compressed(
        str(OUTPUT_PATH),
        segments=segments_arr,
        expert_freqs=expert_freqs_arr,
        weak_eventness=weak_eventness_arr,
        patients=patients_arr,
        subtypes=subtypes_arr,
    )
    file_size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\n[5] Saved to: {OUTPUT_PATH} ({file_size_mb:.1f} MB)")
    print("Done!")


if __name__ == '__main__':
    main()
