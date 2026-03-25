"""
Auto-label RDA wave peaks using bandpass filtering + peak detection.

For each of the 549 RDA cases with expert-verified frequency annotations:
  1. Load 18-channel bipolar EEG segment
  2. Bandpass filter around the known frequency (f ± bandwidth)
  3. Average filtered signal across involved channels per hemisphere
  4. Find peaks (local maxima) of the averaged filtered signal
  5. Find zero crossings for wave onset/offset times
  6. Compute quality metrics (IPI CV, peak consistency)
  7. Assign review tier: auto-accept, quick-review, or manual-edit

Output: rda_wave_labels.json with per-case wave peak times, quality metrics,
        and review tier assignments.

Usage:
    conda run -n foe python code/rda_detector/auto_label_rda_waves.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import (
    detrend, butter, filtfilt, find_peaks, hilbert
)
from scipy.ndimage import gaussian_filter1d
import scipy.io as sio

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

# ── Constants ────────────────────────────────────────────────────────
FS = 200
N_SAMPLES = 2000  # 10 seconds at 200 Hz

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
RDA_CACHE_DIR = DATA_DIR / 'rda_cache'

OUT_DIR = DATA_DIR / 'labels'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEFT_CHANNELS = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CHANNELS = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_CHANNELS = [16, 17]

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Review tier thresholds
TIER1_IPI_CV_MAX = 0.20     # auto-accept
TIER2_IPI_CV_MAX = 0.35     # quick review
# above TIER2 → manual edit (Tier 3)

# Minimum expected peaks for a valid detection (at least 3 waves)
MIN_PEAKS = 3


def load_segment(mat_file):
    """Load a .mat file and return (18, N) bipolar array.

    Tries cached preprocessed version first, then loads from raw .mat.
    """
    # Try RDA cache first (preprocessed with notch + bandpass + detrend)
    segment_id = Path(mat_file).stem
    cache_path = RDA_CACHE_DIR / f'{segment_id}.npy'
    if cache_path.exists():
        seg = np.load(str(cache_path))
        if seg.shape[0] == 18:
            return seg

    # Load raw .mat
    mat_path = EEG_DIR / mat_file
    if not mat_path.exists():
        return None

    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)

    # Convert to bipolar if monopolar
    if data.shape[0] == 20 or (data.shape[1] == 20 and data.shape[0] != 20):
        if data.shape[1] == 20:
            data = data.T
        seg_bi = np.array(fcn_getBanana(data), dtype=np.float64)
    elif data.shape[0] == 18:
        seg_bi = data.astype(np.float64)
    elif data.shape[1] == 18:
        seg_bi = data.T.astype(np.float64)
    else:
        return None

    # Basic preprocessing: detrend per channel
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')

    return seg_bi


def bandpass_filter(signal, f_center, bandwidth=0.4, fs=FS, order=4):
    """Zero-phase bandpass filter around f_center ± bandwidth.

    Args:
        signal: 1D array
        f_center: center frequency in Hz
        bandwidth: half-bandwidth in Hz (filter is f_center ± bandwidth)
        fs: sampling rate
        order: Butterworth filter order

    Returns:
        filtered signal (same length as input)
    """
    nyq = fs / 2.0
    lo = max(f_center - bandwidth, 0.1) / nyq
    hi = min(f_center + bandwidth, nyq - 0.1) / nyq

    if lo >= hi or lo <= 0 or hi >= 1:
        return signal

    b, a = butter(order, [lo, hi], btype='band')
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def find_involved_channels(seg_bi, f_center, bandwidth=0.4):
    """Find the most involved channels per hemisphere using variance explained.

    Returns:
        left_channels: list of channel indices (top channels in left hemisphere)
        right_channels: list of channel indices (top channels in right hemisphere)
        laterality_index: float in [-1, 1], negative = left dominant
    """
    n_ch = seg_bi.shape[0]
    ve_scores = np.zeros(n_ch)

    for ch in range(n_ch):
        filtered = bandpass_filter(seg_bi[ch], f_center, bandwidth)
        total_var = np.var(seg_bi[ch])
        if total_var < 1e-12:
            continue
        residual_var = np.var(seg_bi[ch] - filtered)
        ve_scores[ch] = max(0, 1.0 - residual_var / total_var)

    # Top channels per hemisphere (at least top 3, with VE > 0.05)
    left_ve = [(ch, ve_scores[ch]) for ch in LEFT_CHANNELS if ve_scores[ch] > 0.05]
    right_ve = [(ch, ve_scores[ch]) for ch in RIGHT_CHANNELS if ve_scores[ch] > 0.05]

    left_ve.sort(key=lambda x: -x[1])
    right_ve.sort(key=lambda x: -x[1])

    left_top = [ch for ch, _ in left_ve[:4]] if left_ve else LEFT_CHANNELS[:4]
    right_top = [ch for ch, _ in right_ve[:4]] if right_ve else RIGHT_CHANNELS[:4]

    # Laterality index
    left_mean = np.mean([ve_scores[ch] for ch in LEFT_CHANNELS])
    right_mean = np.mean([ve_scores[ch] for ch in RIGHT_CHANNELS])
    total = left_mean + right_mean
    if total > 1e-8:
        lat_index = (right_mean - left_mean) / total  # negative = left dominant
    else:
        lat_index = 0.0

    return left_top, right_top, lat_index, ve_scores


def detect_wave_peaks(signal, f_center, fs=FS):
    """Detect wave peaks in a filtered signal.

    Args:
        signal: 1D filtered signal (averaged across channels)
        f_center: expected frequency in Hz
        fs: sampling rate

    Returns:
        peak_times: array of peak times in seconds
        peak_indices: array of peak sample indices
        trough_indices: array of trough sample indices (for onset/offset)
    """
    expected_period = 1.0 / f_center
    min_distance_samples = int(0.5 * expected_period * fs)  # at least half a period apart
    min_distance_samples = max(min_distance_samples, 10)

    # Find peaks (maxima)
    peak_indices, peak_props = find_peaks(
        signal,
        distance=min_distance_samples,
        prominence=0.05 * np.std(signal),  # at least 5% of signal std
    )

    # Find troughs (minima) — these are wave onsets/offsets
    trough_indices, _ = find_peaks(
        -signal,
        distance=min_distance_samples,
    )

    peak_times = peak_indices / fs

    return peak_times, peak_indices, trough_indices


def compute_wave_triplets(peak_indices, trough_indices, fs=FS):
    """Compute onset/peak/offset triplets from peaks and troughs.

    For each peak, onset = nearest preceding trough, offset = nearest following trough.

    Returns:
        list of dicts with 'onset', 'peak', 'offset' times in seconds
    """
    triplets = []
    troughs = np.array(trough_indices)

    for peak_idx in peak_indices:
        # Find nearest preceding trough (onset)
        preceding = troughs[troughs < peak_idx]
        if len(preceding) > 0:
            onset_idx = preceding[-1]
        else:
            onset_idx = 0  # start of segment

        # Find nearest following trough (offset)
        following = troughs[troughs > peak_idx]
        if len(following) > 0:
            offset_idx = following[0]
        else:
            offset_idx = peak_idx  # no offset found

        triplets.append({
            'onset': round(float(onset_idx / fs), 4),
            'peak': round(float(peak_idx / fs), 4),
            'offset': round(float(offset_idx / fs), 4),
        })

    return triplets


def compute_quality_metrics(peak_times):
    """Compute quality metrics for auto-detected peaks.

    Returns:
        dict with ipi_cv, ipi_mean, ipi_std, n_peaks, frequency
    """
    if len(peak_times) < 2:
        return {
            'ipi_cv': float('inf'),
            'ipi_mean': float('nan'),
            'ipi_std': float('nan'),
            'n_peaks': len(peak_times),
            'frequency': float('nan'),
        }

    ipis = np.diff(peak_times)
    ipi_mean = float(np.mean(ipis))
    ipi_std = float(np.std(ipis))
    ipi_cv = ipi_std / ipi_mean if ipi_mean > 0 else float('inf')

    return {
        'ipi_cv': round(ipi_cv, 4),
        'ipi_mean': round(ipi_mean, 4),
        'ipi_std': round(ipi_std, 4),
        'n_peaks': len(peak_times),
        'frequency': round(1.0 / np.median(ipis), 4) if len(ipis) > 0 else float('nan'),
    }


def assign_review_tier(metrics, annotated_freq):
    """Assign review tier based on quality metrics.

    Tier 1: auto-accept (clean, consistent peaks)
    Tier 2: quick review (moderate variability, accept/reject en bloc)
    Tier 3: manual edit (poor auto-detection, needs add/delete)

    Returns: 1, 2, or 3
    """
    ipi_cv = metrics['ipi_cv']
    n_peaks = metrics['n_peaks']
    detected_freq = metrics['frequency']

    # Too few peaks → manual
    if n_peaks < MIN_PEAKS:
        return 3

    # Check frequency agreement with annotated frequency
    if np.isfinite(detected_freq) and annotated_freq > 0:
        freq_ratio = detected_freq / annotated_freq
        # If detected frequency is way off (e.g., harmonic), needs review
        if freq_ratio < 0.6 or freq_ratio > 1.6:
            return 3

    if ipi_cv <= TIER1_IPI_CV_MAX:
        return 1
    elif ipi_cv <= TIER2_IPI_CV_MAX:
        return 2
    else:
        return 3


def process_one_case(seg_bi, annotated_freq, subtype):
    """Run the full auto-labeling pipeline on one case.

    Args:
        seg_bi: (18, N_SAMPLES) bipolar EEG
        annotated_freq: expert-annotated frequency in Hz
        subtype: 'lrda' or 'grda'

    Returns:
        dict with wave_peaks, triplets, quality metrics, tier, laterality info
    """
    # Adaptive bandwidth: narrower for higher frequencies, wider for lower
    # Ensures we capture the fundamental without harmonics
    bandwidth = min(0.4, annotated_freq * 0.3)
    bandwidth = max(bandwidth, 0.15)

    # Find involved channels
    left_top, right_top, lat_index, ve_scores = find_involved_channels(
        seg_bi, annotated_freq, bandwidth)

    # Determine which hemisphere(s) to use
    if subtype == 'lrda':
        # Use the more involved hemisphere
        if lat_index < -0.15:
            involved_channels = left_top
            laterality = 'left'
        elif lat_index > 0.15:
            involved_channels = right_top
            laterality = 'right'
        else:
            # Ambiguous — use left by default, flag for review
            involved_channels = left_top
            laterality = 'uncertain'
    else:
        # GRDA — use both hemispheres
        involved_channels = left_top + right_top
        laterality = 'bilateral'

    # Bandpass filter each involved channel and average
    filtered_channels = []
    for ch in involved_channels:
        filtered = bandpass_filter(seg_bi[ch], annotated_freq, bandwidth)
        # Z-score normalize before averaging to equalize channel contributions
        std = np.std(filtered)
        if std > 1e-8:
            filtered = filtered / std
        filtered_channels.append(filtered)

    if not filtered_channels:
        return None

    avg_filtered = np.mean(filtered_channels, axis=0)

    # Smooth slightly to remove any residual high-freq noise
    avg_filtered = gaussian_filter1d(avg_filtered, sigma=2)

    # Detect peaks and troughs
    peak_times, peak_indices, trough_indices = detect_wave_peaks(
        avg_filtered, annotated_freq)

    # Compute triplets (onset, peak, offset)
    triplets = compute_wave_triplets(peak_indices, trough_indices)

    # Quality metrics
    metrics = compute_quality_metrics(peak_times)

    # Assign review tier
    tier = assign_review_tier(metrics, annotated_freq)

    return {
        'wave_peaks': [round(float(t), 4) for t in peak_times],
        'triplets': triplets,
        'n_waves': len(peak_times),
        'annotated_freq': annotated_freq,
        'detected_freq': metrics['frequency'],
        'ipi_cv': metrics['ipi_cv'],
        'ipi_mean': metrics['ipi_mean'],
        'n_peaks': metrics['n_peaks'],
        'review_tier': tier,
        'laterality': laterality,
        'lat_index': round(lat_index, 4),
        'involved_channels': involved_channels,
        've_scores': {BIPOLAR_CHANNELS[i]: round(float(ve_scores[i]), 4)
                      for i in range(len(BIPOLAR_CHANNELS))},
    }


def main():
    print("=" * 70)
    print("RDA Wave Auto-Labeling Pipeline")
    print("=" * 70)

    # ── Load annotated cases ──────────────────────────────────────────
    annot_df = pd.read_csv(str(LABELS_DIR / 'archive_labels' / 'rda_freq_annotations_mw.csv'))
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    rda_segs = seg_df[seg_df['subtype'].isin(['grda', 'lrda'])].copy()

    # Merge to get mat_file and subtype
    merged = annot_df.merge(
        rda_segs[['patient_id', 'subtype', 'segment_id', 'mat_file']].drop_duplicates('patient_id'),
        on='patient_id', how='left'
    )

    print(f"Annotated RDA cases: {len(merged)}")
    print(f"  GRDA: {(merged['subtype'] == 'grda').sum()}")
    print(f"  LRDA: {(merged['subtype'] == 'lrda').sum()}")
    print()

    # ── Process each case ─────────────────────────────────────────────
    results = {}
    tier_counts = {1: 0, 2: 0, 3: 0}
    n_failed = 0

    for idx, row in merged.iterrows():
        patient_id = str(row['patient_id'])
        mat_file = row['mat_file']
        annotated_freq = float(row['annotated_freq'])
        subtype = row['subtype']

        if pd.isna(mat_file):
            n_failed += 1
            continue

        # Load segment
        seg_bi = load_segment(mat_file)
        if seg_bi is None:
            n_failed += 1
            continue

        # Ensure correct shape
        if seg_bi.shape[0] != 18:
            n_failed += 1
            continue

        # Process
        result = process_one_case(seg_bi, annotated_freq, subtype)
        if result is None:
            n_failed += 1
            continue

        result['patient_id'] = patient_id
        result['segment_id'] = str(row.get('segment_id', ''))
        result['subtype'] = subtype
        result['review_status'] = 'pending'

        results[patient_id] = result
        tier_counts[result['review_tier']] += 1

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(merged)}...")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\nProcessed: {len(results)} / {len(merged)} cases")
    print(f"Failed to load/process: {n_failed}")
    print()
    print("Review tier distribution:")
    print(f"  Tier 1 (auto-accept):  {tier_counts[1]:4d}  ({100*tier_counts[1]/max(1,len(results)):.1f}%)")
    print(f"  Tier 2 (quick review): {tier_counts[2]:4d}  ({100*tier_counts[2]/max(1,len(results)):.1f}%)")
    print(f"  Tier 3 (manual edit):  {tier_counts[3]:4d}  ({100*tier_counts[3]/max(1,len(results)):.1f}%)")

    # Frequency agreement stats
    freq_ratios = []
    for pid, r in results.items():
        if np.isfinite(r['detected_freq']) and r['annotated_freq'] > 0:
            freq_ratios.append(r['detected_freq'] / r['annotated_freq'])
    if freq_ratios:
        fr = np.array(freq_ratios)
        print(f"\nFrequency agreement (detected/annotated):")
        print(f"  Median ratio: {np.median(fr):.3f}")
        print(f"  Within ±10%:  {100*np.mean(np.abs(fr - 1.0) < 0.1):.1f}%")
        print(f"  Within ±20%:  {100*np.mean(np.abs(fr - 1.0) < 0.2):.1f}%")

    # IPI CV stats
    cvs = [r['ipi_cv'] for r in results.values() if np.isfinite(r['ipi_cv'])]
    if cvs:
        print(f"\nIPI CV distribution:")
        print(f"  Median: {np.median(cvs):.3f}")
        print(f"  Mean:   {np.mean(cvs):.3f}")
        print(f"  < 0.20: {100*np.mean(np.array(cvs) < 0.20):.1f}%")
        print(f"  < 0.35: {100*np.mean(np.array(cvs) < 0.35):.1f}%")

    # ── Save results ──────────────────────────────────────────────────
    out_path = OUT_DIR / 'rda_wave_labels.json'
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Also save a compact summary CSV for easy review
    summary_records = []
    for pid, r in results.items():
        summary_records.append({
            'patient_id': pid,
            'subtype': r['subtype'],
            'annotated_freq': r['annotated_freq'],
            'detected_freq': r['detected_freq'],
            'n_waves': r['n_waves'],
            'ipi_cv': r['ipi_cv'],
            'review_tier': r['review_tier'],
            'laterality': r['laterality'],
        })
    summary_df = pd.DataFrame(summary_records)
    summary_path = OUT_DIR / 'rda_wave_labels_summary.csv'
    summary_df.to_csv(str(summary_path), index=False)
    print(f"Saved summary to {summary_path}")


if __name__ == '__main__':
    main()
