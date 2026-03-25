"""
Round 4 experiment: Alternans-aware frequency correction.

Some periodic discharges show strong/weak/strong/weak alternation.
Standard methods detect half the true frequency because weak peaks
are missed. This experiment detects alternans and corrects for it.

Variants tested:
  a) r4_alternans_fft          - if alternans_score > 0.3, double FFT freq
  b) r4_alternans_acf          - if alternans_score > 0.3, double ACF freq
  c) r4_alternans_peakcount_all - recount peaks with lower threshold (0.15)
  d) r4_alternans_adaptive     - if alternans, use low-threshold peak count; else normal
  e) r4_alternans_triple       - median of (corrected FFT, corrected ACF, low-thr peak count)
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import fcn_getBanana, compute_pointiness_trace, bipolar_channels


# ---------------------------------------------------------------------------
# Preprocessing (same as r3)
# ---------------------------------------------------------------------------

def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    data = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(data))
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


def compute_smoothed_pointiness(seg, fs):
    """Compute smoothed pointiness trace for each channel. Returns (n_ch, n_samples)."""
    sigma_samples = max(1, int(0.02 * fs))
    traces = []
    for i in range(seg.shape[0]):
        tr = compute_pointiness_trace(seg[i])
        tr = gaussian_filter1d(tr, sigma=sigma_samples)
        traces.append(tr)
    return np.array(traces)


# ---------------------------------------------------------------------------
# Frequency estimation helpers (from r3)
# ---------------------------------------------------------------------------

def fft_frequency(trace, fs, fmin=0.3, fmax=3.5):
    """Find peak frequency in FFT of trace within [fmin, fmax] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan, False
    fft_vals = np.fft.rfft(trace - np.mean(trace))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, False
    power_in_range = power[mask]
    freqs_in_range = freqs[mask]
    peak_idx = np.argmax(power_in_range)
    peak_power = power_in_range[peak_idx]
    mean_power = np.mean(power_in_range)
    if peak_power > 2.0 * mean_power:
        return float(freqs_in_range[peak_idx]), True
    return np.nan, False


def acf_frequency(trace, fs, acf_peak_threshold=0.10, acf_min_lag=0.25):
    """ACF-based frequency estimation with threshold 0.10."""
    n = len(trace)
    if n < 50:
        return np.nan
    t = trace - np.mean(trace)
    max_lag = min(int(4 * fs), n - 1)
    if max_lag < 10:
        return np.nan
    acf = np.correlate(t, t, mode='full')
    acf = acf[n - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return np.nan
    min_lag_samples = int(acf_min_lag * fs)
    for k in range(min_lag_samples + 1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > acf_peak_threshold:
            freq = fs / k
            return freq
    return np.nan


def peakcount_frequency(trace, fs, height_frac=0.3):
    """Estimate frequency by counting prominent peaks in pointiness trace."""
    if len(trace) < 10:
        return np.nan
    mx = np.max(trace)
    if mx <= 0:
        return np.nan
    height_thr = mx * height_frac
    min_dist = int(0.2 * fs)
    peaks, props = find_peaks(trace, height=height_thr, distance=min_dist)
    if len(peaks) < 2:
        return np.nan
    intervals = np.diff(peaks) / fs
    freq = 1.0 / np.median(intervals)
    if 0.3 <= freq <= 3.5:
        return freq
    return np.nan


def median_valid(freqs):
    """Median of finite values, or NaN."""
    valid = [f for f in freqs if np.isfinite(f)]
    if not valid:
        return np.nan
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Alternans detection
# ---------------------------------------------------------------------------

def detect_alternans_channel(trace, fs):
    """Detect alternans (strong/weak alternation) in a single-channel pointiness trace.

    Returns True if alternans pattern is detected.
    """
    mx = np.max(trace)
    if mx <= 0:
        return False

    height_thr = mx * 0.3
    min_dist = int(0.2 * fs)
    peaks, props = find_peaks(trace, height=height_thr, distance=min_dist)

    if len(peaks) < 4:
        return False

    peak_heights = props['peak_heights']
    even_heights = peak_heights[0::2]
    odd_heights = peak_heights[1::2]

    mean_even = np.mean(even_heights)
    mean_odd = np.mean(odd_heights)

    if mean_odd == 0 or mean_even == 0:
        return False

    alternans_ratio = mean_even / mean_odd

    # Check if ratio indicates alternans: between 0.3 and 0.7, or inverse is
    if 0.3 <= alternans_ratio <= 0.7:
        return True
    inv_ratio = 1.0 / alternans_ratio
    if 0.3 <= inv_ratio <= 0.7:
        return True

    return False


def compute_alternans_score(traces, fs):
    """Compute alternans_score = fraction of detected channels showing alternans.

    Only counts channels that have >= 4 peaks (i.e. channels where alternans
    detection is even attempted).
    """
    n_ch = traces.shape[0]
    n_detected = 0
    n_alternans = 0

    for ci in range(n_ch):
        tr = traces[ci]
        mx = np.max(tr)
        if mx <= 0:
            continue
        height_thr = mx * 0.3
        min_dist = int(0.2 * fs)
        peaks, _ = find_peaks(tr, height=height_thr, distance=min_dist)
        if len(peaks) < 4:
            continue
        n_detected += 1
        if detect_alternans_channel(tr, fs):
            n_alternans += 1

    if n_detected == 0:
        return 0.0
    return n_alternans / n_detected


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} annotated segments")

    # Predictions dicts for each variant
    preds_alt_fft = {}
    preds_alt_acf = {}
    preds_alt_peakcount_all = {}
    preds_alt_adaptive = {}
    preds_alt_triple = {}

    n_total = len(dataset)
    n_skipped = 0
    n_alternans_detected = 0

    for idx, entry in enumerate(dataset):
        mat_name = entry['mat_name']

        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{n_total} segments processed "
                  f"({n_skipped} skipped, {n_alternans_detected} alternans detected)")

        data, fs = load_eeg_data(entry)
        if data is None:
            n_skipped += 1
            continue

        try:
            seg = preprocess_segment(data, fs)
            traces = compute_smoothed_pointiness(seg, fs)
        except Exception:
            n_skipped += 1
            continue

        n_ch = traces.shape[0]

        # --- Alternans detection ---
        alternans_score = compute_alternans_score(traces, fs)
        has_alternans = alternans_score > 0.3
        if has_alternans:
            n_alternans_detected += 1

        # --- Per-channel frequency estimates ---
        ch_fft = []
        ch_acf = []
        ch_peakcount_normal = []
        ch_peakcount_low = []   # low threshold for catching weak peaks

        for ci in range(n_ch):
            tr = traces[ci]

            # FFT frequency
            f_fft, v_fft = fft_frequency(tr, fs)
            if v_fft:
                ch_fft.append(f_fft)

            # ACF frequency (thr=0.10)
            f_acf = acf_frequency(tr, fs, acf_peak_threshold=0.10)
            if np.isfinite(f_acf):
                ch_acf.append(f_acf)

            # Normal peak-count (height_frac=0.3)
            f_pk = peakcount_frequency(tr, fs, height_frac=0.3)
            if np.isfinite(f_pk):
                ch_peakcount_normal.append(f_pk)

            # Low-threshold peak-count (height_frac=0.15) to catch weak peaks
            f_pk_low = peakcount_frequency(tr, fs, height_frac=0.15)
            if np.isfinite(f_pk_low):
                ch_peakcount_low.append(f_pk_low)

        # Aggregate across channels
        freq_fft = median_valid(ch_fft)
        freq_acf = median_valid(ch_acf)
        freq_pk_normal = median_valid(ch_peakcount_normal)
        freq_pk_low = median_valid(ch_peakcount_low)

        # --- Apply alternans corrections ---

        # (a) r4_alternans_fft: if alternans, double FFT frequency
        if has_alternans and np.isfinite(freq_fft):
            corrected_fft = freq_fft * 2.0
            # Clamp to reasonable range
            preds_alt_fft[mat_name] = corrected_fft if corrected_fft <= 3.5 else freq_fft
        else:
            preds_alt_fft[mat_name] = freq_fft

        # (b) r4_alternans_acf: if alternans, double ACF frequency
        if has_alternans and np.isfinite(freq_acf):
            corrected_acf = freq_acf * 2.0
            preds_alt_acf[mat_name] = corrected_acf if corrected_acf <= 3.5 else freq_acf
        else:
            preds_alt_acf[mat_name] = freq_acf

        # (c) r4_alternans_peakcount_all: always use low-threshold peak count
        preds_alt_peakcount_all[mat_name] = freq_pk_low

        # (d) r4_alternans_adaptive: if alternans use low-threshold; else normal
        if has_alternans:
            preds_alt_adaptive[mat_name] = freq_pk_low
        else:
            preds_alt_adaptive[mat_name] = freq_pk_normal

        # (e) r4_alternans_triple: median of (corrected FFT, corrected ACF, low-thr peak count)
        corr_fft = freq_fft
        corr_acf = freq_acf
        if has_alternans:
            if np.isfinite(freq_fft):
                doubled = freq_fft * 2.0
                corr_fft = doubled if doubled <= 3.5 else freq_fft
            if np.isfinite(freq_acf):
                doubled = freq_acf * 2.0
                corr_acf = doubled if doubled <= 3.5 else freq_acf

        triple_vals = [v for v in [corr_fft, corr_acf, freq_pk_low] if np.isfinite(v)]
        preds_alt_triple[mat_name] = float(np.median(triple_vals)) if triple_vals else np.nan

    print(f"\nDone processing. {n_skipped} segments skipped out of {n_total}.")
    print(f"Alternans detected in {n_alternans_detected} segments (alternans_score > 0.3).")

    # Evaluate each variant
    print("\n" + "=" * 60)
    print("EVALUATING ALL VARIANTS")
    print("=" * 60)

    variants = [
        ("r4_alternans_fft", preds_alt_fft),
        ("r4_alternans_acf", preds_alt_acf),
        ("r4_alternans_peakcount_all", preds_alt_peakcount_all),
        ("r4_alternans_adaptive", preds_alt_adaptive),
        ("r4_alternans_triple", preds_alt_triple),
    ]

    for name, preds in variants:
        n_valid = sum(1 for v in preds.values() if np.isfinite(v))
        print(f"\n--- {name} ({n_valid} valid predictions) ---")
        evaluate_predictions(dataset, preds, name)

    print("\nAll experiments complete.")


if __name__ == '__main__':
    main()
