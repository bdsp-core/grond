"""
Recompute signal-processing features for the 77 new annotation candidates
from their raw .mat files on the external drive, using the EXACT same pipeline
as the original 43 patients in r10_expanded_evaluation.py.

This fixes the feature distribution mismatch: the original CSV features were
computed on pre-processed bipolar segments (18ch, 2000 samples), but the
original 43 patients' features are computed from raw 20ch EEG through the
full pipeline (notch -> bandpass -> bipolar -> lowpass -> feature extraction).

Run: conda run -n foe python code/recompute_new77_features.py
"""

import sys
import os
import glob
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.signal import find_peaks, butter, filtfilt, coherence as scipy_coherence
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Setup paths
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_pointiness_acf import (
    pd_detect_pointiness_acf, compute_pointiness_trace, fcn_getBanana,
    bipolar_channels,
)
from mne.filter import notch_filter, filter_data
import hdf5storage

# ── Constants (same as r10_expanded_evaluation.py) ────────────────────
FS = 200
FREQ_LO, FREQ_HI = 0.3, 3.5
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]

RAW_DIR = Path('/Volumes/sanD_photos/IIIC/segments_raw')
ANNOT_CSV = PROJECT_DIR / 'data' / '_archive' / 'annotation_candidates' / 'frequency_annotations.csv'
OUTPUT_CSV = PROJECT_DIR / 'data' / '_archive' / 'annotation_candidates' / 'frequency_annotations_with_features.csv'


# ── Feature computation (identical to r10_expanded_evaluation.py) ─────

def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
    """FFT peak frequency in [freq_lo, freq_hi] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freqs_sub = freqs[mask]
    if np.max(fft_sub) == 0:
        return np.nan
    return freqs_sub[np.argmax(fft_sub)]


def compute_sp_features_from_eeg(data, fs, is_gpd):
    """Compute SP features from raw 20-channel EEG data.

    Identical to r10_expanded_evaluation.py::compute_sp_features_from_eeg.
    """
    features = {}
    features['is_gpd'] = float(is_gpd)

    # f_B: ACF thr=0.10 (pd_detect_pointiness_acf does its own preprocessing)
    try:
        res_B = pd_detect_pointiness_acf(
            data.copy(), fs,
            method='pointiness', lowpass_hz=15,
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        f_B = res_B['event_frequency']
        if not np.isfinite(f_B):
            f_B = np.nan
        detected_channels = res_B.get('channels', [])
        if detected_channels is None or (isinstance(detected_channels, float) and np.isnan(detected_channels)):
            detected_channels = []
        n_detected = len(detected_channels)
    except:
        f_B = np.nan
        n_detected = 0
    features['f_B'] = f_B
    features['n_ch'] = n_detected

    # Preprocessing for per-channel features
    seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg_bip = np.array(fcn_getBanana(seg_filtered))

    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(seg_bip.shape[0]):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    n_channels = seg_lp.shape[0]
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # Pointiness traces
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # f_peaks
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # f_fft
    fft_freqs = []
    for ch in range(n_channels):
        f = compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    features['f_fft'] = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # f_tkeo
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh (on seg_bip, NOT seg_lp — same as r10)
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = scipy_coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs,
                                          nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    return features


def find_mat_file(patient_id):
    """Find a raw .mat file for this patient on the external drive."""
    pattern = str(RAW_DIR / f'sub-S0001{patient_id}_*.mat')
    files = sorted(glob.glob(pattern))
    if files:
        return files[0]
    return None


def extract_10s_window(mat_path):
    """Load raw .mat and extract central 10s window around event_time.

    Returns (data, fs) where data is (20, 2000) at 200Hz, or (None, None).
    """
    try:
        # These are HDF5 v7.3 .mat files — use hdf5storage directly
        mat = hdf5storage.loadmat(mat_path)

        data = mat['data']
        fs_val = mat['Fs']
        if hasattr(fs_val, 'flat'):
            fs_val = float(fs_val.flat[0])
        else:
            fs_val = float(fs_val)
        fs = int(fs_val)

        # Ensure (channels, samples)
        if data.shape[0] > data.shape[1]:
            data = data.T

        n_channels, n_samples = data.shape

        # Parse event_time and filename timestamp to find offset
        event_time_raw = mat.get('event_time', None)
        fname = os.path.basename(mat_path)

        center_sample = n_samples // 2  # default: center of recording

        if event_time_raw is not None:
            try:
                # Extract event time string from nested numpy array
                et = event_time_raw
                # Unwrap nested arrays until we get a string
                while isinstance(et, np.ndarray) and et.size >= 1:
                    et = et.flat[0]
                event_str = str(et).strip()

                # Parse filename timestamp (sub-S0001XXXXX_YYYYMMDDHHMMSS.mat)
                ts_part = fname.split('_')[-1].replace('.mat', '')
                file_dt = datetime.strptime(ts_part, '%Y%m%d%H%M%S')
                event_dt = datetime.strptime(event_str, '%d-%b-%Y %H:%M:%S')

                offset_s = (event_dt - file_dt).total_seconds()
                if 0 <= offset_s <= n_samples / fs:
                    center_sample = int(offset_s * fs)
            except Exception:
                pass  # Fall back to center of recording

        # Extract 10s window centered on event
        half_win = int(5 * fs)  # 5s = 1000 samples at 200Hz
        start = max(0, center_sample - half_win)
        end = start + 2 * half_win
        if end > n_samples:
            end = n_samples
            start = max(0, end - 2 * half_win)

        segment = data[:, start:end].astype(np.float64)

        # Verify shape
        if segment.shape[0] != 20 or segment.shape[1] != 2 * half_win:
            return None, None

        return segment, fs

    except Exception as e:
        print(f"    ERROR loading {mat_path}: {e}")
        return None, None


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("RECOMPUTE FEATURES FOR 77 NEW PATIENTS FROM RAW .MAT FILES")
    print("=" * 70)

    # 1. Load annotation CSV
    print(f"\n[1] Loading annotations from {ANNOT_CSV}...")
    df = pd.read_csv(str(ANNOT_CSV))
    print(f"  Total rows: {len(df)}")

    # Skip rows with expert_annotation == 'skip'
    df_active = df[df['expert_annotation'] != 'skip'].copy()
    n_skip = len(df) - len(df_active)
    print(f"  Skipped: {n_skip}, Active: {len(df_active)}")

    # 2. Process each patient
    print(f"\n[2] Processing {len(df_active)} patients...")
    print(f"  Raw .mat directory: {RAW_DIR}")

    results = []
    n_success = 0
    n_failed = 0
    n_no_mat = 0

    for idx, (_, row) in enumerate(df_active.iterrows()):
        patient_id = str(row['patient_id'])
        file_name = row['file_name']
        subtype = row['subtype']
        expert_annotation = row['expert_annotation']
        is_gpd = 1 if subtype == 'gpd' else 0

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{len(df_active)}] {file_name} ({elapsed:.0f}s elapsed, "
              f"{n_success} ok, {n_failed} failed, {n_no_mat} no .mat)")

        # Find raw .mat file
        mat_path = find_mat_file(patient_id)
        if mat_path is None:
            print(f"    WARNING: No .mat file found for patient {patient_id}")
            n_no_mat += 1
            results.append({
                'patient_id': patient_id,
                'file_name': file_name,
                'subtype': subtype,
                'expert_annotation': expert_annotation,
                'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                'f_tkeo': np.nan, 'f_coh': np.nan,
                'is_gpd': is_gpd, 'n_ch': 0,
            })
            continue

        # Load and extract 10s window
        data, fs = extract_10s_window(mat_path)
        if data is None:
            print(f"    WARNING: Could not extract data for patient {patient_id}")
            n_failed += 1
            results.append({
                'patient_id': patient_id,
                'file_name': file_name,
                'subtype': subtype,
                'expert_annotation': expert_annotation,
                'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                'f_tkeo': np.nan, 'f_coh': np.nan,
                'is_gpd': is_gpd, 'n_ch': 0,
            })
            continue

        # Compute features
        try:
            feats = compute_sp_features_from_eeg(data, fs, is_gpd)
            results.append({
                'patient_id': patient_id,
                'file_name': file_name,
                'subtype': subtype,
                'expert_annotation': expert_annotation,
                'f_B': feats['f_B'],
                'f_peaks': feats['f_peaks'],
                'f_fft': feats['f_fft'],
                'f_tkeo': feats['f_tkeo'],
                'f_coh': feats['f_coh'],
                'is_gpd': feats['is_gpd'],
                'n_ch': feats['n_ch'],
            })
            n_success += 1
        except Exception as e:
            print(f"    WARNING: Feature extraction failed for {patient_id}: {e}")
            n_failed += 1
            results.append({
                'patient_id': patient_id,
                'file_name': file_name,
                'subtype': subtype,
                'expert_annotation': expert_annotation,
                'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                'f_tkeo': np.nan, 'f_coh': np.nan,
                'is_gpd': is_gpd, 'n_ch': 0,
            })

    elapsed = time.time() - t0
    print(f"\n[3] Done processing. {n_success} success, {n_failed} failed, {n_no_mat} no .mat ({elapsed:.0f}s)")

    # 3. Save results
    df_out = pd.DataFrame(results)
    col_order = ['patient_id', 'file_name', 'subtype', 'expert_annotation',
                 'f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch']
    df_out = df_out[col_order]

    # Round numeric columns
    for col in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']:
        df_out[col] = df_out[col].round(3)

    df_out.to_csv(str(OUTPUT_CSV), index=False)
    print(f"\n[4] Saved to {OUTPUT_CSV}")
    print(f"  Shape: {df_out.shape}")

    # 4. Summary statistics comparison
    print("\n" + "=" * 70)
    print("FEATURE DISTRIBUTION SUMMARY (new features from raw .mat)")
    print("=" * 70)
    for col in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'n_ch']:
        vals = df_out[col].dropna()
        if len(vals) > 0:
            print(f"  {col:8s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"median={vals.median():.3f}  [{vals.min():.3f}, {vals.max():.3f}]  "
                  f"n_valid={len(vals)}/{len(df_out)}")
        else:
            print(f"  {col:8s}: no valid values")

    # Compare with old CSV features
    print("\n  --- Old CSV features (for comparison) ---")
    df_old = pd.read_csv(str(ANNOT_CSV))
    df_old = df_old[df_old['expert_annotation'] != 'skip']
    for col in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']:
        vals = pd.to_numeric(df_old[col], errors='coerce').dropna()
        if len(vals) > 0:
            print(f"  {col:8s}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
                  f"median={vals.median():.3f}  [{vals.min():.3f}, {vals.max():.3f}]  "
                  f"n_valid={len(vals)}/{len(df_old)}")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
