"""
Round 3 experiment: FFT and Cepstral analysis of pointiness traces.

Problem: ACF locks onto subharmonics. FFT and cepstral methods can better
separate fundamentals from harmonics.

Variants tested:
  a) r3_fft_pointiness   - FFT of pointiness trace, peak in [0.3, 3.5] Hz
  b) r3_cepstral         - Cepstral analysis of pointiness trace
  c) r3_fft_of_acf       - FFT of ACF of pointiness trace
  d) r3_fft_plus_peakcount - average of FFT freq and peak-count freq
  e) r3_triple_fft_A_peaks - median of (FFT freq, Method A freq, peak-count freq)
  f) r3_fft_bayesian     - FFT freq with Bayesian prior nudge
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
from pd_detect_alternate import pd_detect_alternate

# ---------------------------------------------------------------------------
# Preprocessing: shared across all methods
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
# Method A: FFT of pointiness trace
# ---------------------------------------------------------------------------

def fft_frequency(trace, fs, fmin=0.3, fmax=3.5):
    """Find peak frequency in FFT of trace within [fmin, fmax] Hz.
    Returns (freq, is_valid). is_valid requires peak > 2x mean power in range.
    """
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


# ---------------------------------------------------------------------------
# Method B: Cepstral analysis
# ---------------------------------------------------------------------------

def cepstral_frequency(trace, fs, fmin=0.3, fmax=3.5):
    """Find frequency via cepstral peak in quefrency range [1/fmax, 1/fmin].
    Returns (freq, is_valid).
    """
    n = len(trace)
    if n < 10:
        return np.nan, False

    fft_vals = np.fft.rfft(trace - np.mean(trace))
    power = np.abs(fft_vals) ** 2
    log_power = np.log(power + 1e-12)

    # Cepstrum = inverse FFT of log power spectrum
    cepstrum = np.fft.irfft(log_power)

    # Quefrency bounds: 1/fmax to 1/fmin (in samples)
    q_min = int(fs / fmax)   # 0.28s * fs ~ high freq bound
    q_max = int(fs / fmin)   # 3.3s * fs ~ low freq bound
    q_max = min(q_max, len(cepstrum) - 1)

    if q_min >= q_max or q_min < 1:
        return np.nan, False

    ceps_range = cepstrum[q_min:q_max + 1]
    mean_ceps = np.mean(np.abs(ceps_range))

    peak_idx = np.argmax(ceps_range)
    peak_val = ceps_range[peak_idx]

    if peak_val > 2.0 * mean_ceps:
        quefrency = (q_min + peak_idx) / fs
        return float(1.0 / quefrency), True
    return np.nan, False


# ---------------------------------------------------------------------------
# Method C: FFT of ACF
# ---------------------------------------------------------------------------

def fft_of_acf_frequency(trace, fs, fmin=0.3, fmax=3.5):
    """Compute ACF of trace, then FFT of ACF, peak in [fmin, fmax].
    Returns (freq, is_valid).
    """
    n = len(trace)
    if n < 50:
        return np.nan, False

    t = trace - np.mean(trace)
    max_lag = min(int(4 * fs), n - 1)
    if max_lag < 10:
        return np.nan, False

    acf = np.correlate(t, t, mode='full')
    acf = acf[n - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return np.nan, False

    # FFT of the ACF
    fft_vals = np.fft.rfft(acf)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(acf), d=1.0 / fs)

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


# ---------------------------------------------------------------------------
# Peak-count frequency (simple: count peaks in pointiness trace)
# ---------------------------------------------------------------------------

def peakcount_frequency(trace, fs, duration_s=None):
    """Estimate frequency by counting prominent peaks in pointiness trace."""
    if duration_s is None:
        duration_s = len(trace) / fs
    if duration_s <= 0 or len(trace) < 10:
        return np.nan

    mx = np.max(trace)
    if mx <= 0:
        return np.nan
    height_thr = mx * 0.3
    min_dist = int(0.2 * fs)
    peaks, _ = find_peaks(trace, height=height_thr, distance=min_dist)
    if len(peaks) < 2:
        return np.nan
    intervals = np.diff(peaks) / fs
    freq = 1.0 / np.median(intervals)
    if 0.3 <= freq <= 3.5:
        return freq
    return np.nan


# ---------------------------------------------------------------------------
# Per-channel aggregation helper
# ---------------------------------------------------------------------------

def median_valid(freqs):
    """Median of finite values, or NaN."""
    valid = [f for f in freqs if np.isfinite(f)]
    if not valid:
        return np.nan
    return float(np.median(valid))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} annotated segments")

    # Predictions dicts for each variant
    preds_fft = {}
    preds_cepstral = {}
    preds_fft_acf = {}
    preds_fft_plus_pk = {}
    preds_triple = {}
    preds_bayesian = {}

    # Prior for Bayesian nudge: population median ~1.5 Hz for PDs
    PRIOR_FREQ = 1.5

    n_total = len(dataset)
    n_skipped = 0

    for idx, entry in enumerate(dataset):
        mat_name = entry['mat_name']

        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{n_total} segments processed "
                  f"({n_skipped} skipped)")

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
        duration_s = traces.shape[1] / fs

        # Per-channel frequency estimates
        ch_fft = []
        ch_ceps = []
        ch_fft_acf = []
        ch_peakcount = []

        for ci in range(n_ch):
            tr = traces[ci]

            f_fft, v_fft = fft_frequency(tr, fs)
            if v_fft:
                ch_fft.append(f_fft)

            f_ceps, v_ceps = cepstral_frequency(tr, fs)
            if v_ceps:
                ch_ceps.append(f_ceps)

            f_facf, v_facf = fft_of_acf_frequency(tr, fs)
            if v_facf:
                ch_fft_acf.append(f_facf)

            f_pk = peakcount_frequency(tr, fs, duration_s)
            if np.isfinite(f_pk):
                ch_peakcount.append(f_pk)

        # (a) FFT pointiness
        freq_fft = median_valid(ch_fft)
        preds_fft[mat_name] = freq_fft

        # (b) Cepstral
        freq_ceps = median_valid(ch_ceps)
        preds_cepstral[mat_name] = freq_ceps

        # (c) FFT of ACF
        freq_facf = median_valid(ch_fft_acf)
        preds_fft_acf[mat_name] = freq_facf

        # (d) FFT + peak-count average
        freq_pk = median_valid(ch_peakcount)
        if np.isfinite(freq_fft) and np.isfinite(freq_pk):
            preds_fft_plus_pk[mat_name] = (freq_fft + freq_pk) / 2.0
        elif np.isfinite(freq_fft):
            preds_fft_plus_pk[mat_name] = freq_fft
        elif np.isfinite(freq_pk):
            preds_fft_plus_pk[mat_name] = freq_pk
        else:
            preds_fft_plus_pk[mat_name] = np.nan

        # (e) Triple: median of (FFT freq, Method A freq, peak-count freq)
        try:
            rA = pd_detect_alternate(data, fs, pk_detect='apd')
            freq_A = rA.get('event_frequency', np.nan)
            if not np.isfinite(freq_A):
                freq_A = np.nan
        except Exception:
            freq_A = np.nan

        triple_vals = [v for v in [freq_fft, freq_A, freq_pk] if np.isfinite(v)]
        preds_triple[mat_name] = float(np.median(triple_vals)) if triple_vals else np.nan

        # (f) Bayesian nudge: 0.7*FFT + 0.3*prior
        if np.isfinite(freq_fft):
            preds_bayesian[mat_name] = 0.7 * freq_fft + 0.3 * PRIOR_FREQ
        else:
            preds_bayesian[mat_name] = np.nan

    print(f"\nDone processing. {n_skipped} segments skipped out of {n_total}.")

    # Evaluate each variant
    print("\n" + "=" * 60)
    print("EVALUATING ALL VARIANTS")
    print("=" * 60)

    variants = [
        ("r3_fft_pointiness", preds_fft),
        ("r3_cepstral", preds_cepstral),
        ("r3_fft_of_acf", preds_fft_acf),
        ("r3_fft_plus_peakcount", preds_fft_plus_pk),
        ("r3_triple_fft_A_peaks", preds_triple),
        ("r3_fft_bayesian", preds_bayesian),
    ]

    for name, preds in variants:
        n_valid = sum(1 for v in preds.values() if np.isfinite(v))
        print(f"\n--- {name} ({n_valid} valid predictions) ---")
        evaluate_predictions(dataset, preds, name)

    print("\nAll experiments complete.")


if __name__ == '__main__':
    main()
