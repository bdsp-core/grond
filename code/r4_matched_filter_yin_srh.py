"""
Round 4 experiment: Matched-filter discharge likelihood envelope + YIN/SRH frequency estimation.

Uses template banks (C for LPD/GPD-specific, D for universal) to create a
discharge likelihood envelope via cross-correlation, then estimates periodicity
using YIN (CMNDF), SRH, and FFT methods.

Variants:
  r4_yin_bankC        - YIN threshold=0.2, Bank C
  r4_yin_bankD        - YIN threshold=0.2, Bank D
  r4_yin_t01_bankC    - YIN threshold=0.1 (stricter)
  r4_yin_t03_bankC    - YIN threshold=0.3 (more permissive)
  r4_srh_bankC        - SRH on Bank C envelope
  r4_srh_bankD        - SRH on Bank D envelope
  r4_fft_envelope_C   - FFT of Bank C envelope
  r4_fft_envelope_D   - FFT of Bank D envelope
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from mne.filter import notch_filter, filter_data
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import fcn_getBanana

# ---------------------------------------------------------------------------
# Load template banks
# ---------------------------------------------------------------------------
DATA_DIR = CODE_DIR.parent / 'data'
templates_C_lpd = np.load(str(DATA_DIR / 'templates_C_lpd.npy'))  # (8, 50)
templates_C_gpd = np.load(str(DATA_DIR / 'templates_C_gpd.npy'))  # (8, 50)
templates_D = np.load(str(DATA_DIR / 'templates_D.npy'))          # (8, 50)

print(f"Templates loaded: C_lpd={templates_C_lpd.shape}, C_gpd={templates_C_gpd.shape}, D={templates_D.shape}")

# ---------------------------------------------------------------------------
# Preprocessing
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


# ---------------------------------------------------------------------------
# Matched filter: cross-correlate each channel with templates
# ---------------------------------------------------------------------------

def compute_discharge_envelope(seg, templates):
    """
    For each channel, cross-correlate with each template, take max across
    templates at each time point -> discharge_likelihood_envelope[channel, t].
    """
    n_ch, n_t = seg.shape
    n_templates, template_len = templates.shape
    envelope = np.zeros((n_ch, n_t))

    for ch in range(n_ch):
        channel = seg[ch]
        # z-score normalize the channel
        std = np.std(channel)
        if std < 1e-10:
            continue
        channel_normed = (channel - np.mean(channel)) / std

        correlations = np.zeros((n_templates, n_t))
        for ti in range(n_templates):
            # Normalize template (should already be, but ensure)
            tmpl = templates[ti]
            tmpl_std = np.std(tmpl)
            if tmpl_std < 1e-10:
                continue
            tmpl_normed = (tmpl - np.mean(tmpl)) / tmpl_std / template_len
            corr = np.correlate(channel_normed, tmpl_normed, mode='same')
            correlations[ti] = corr

        # Max correlation across templates at each time point
        envelope[ch] = np.max(correlations, axis=0)

    return envelope


# ---------------------------------------------------------------------------
# YIN / CMNDF frequency estimation
# ---------------------------------------------------------------------------

def yin_frequency(envelope_channel, fs, threshold=0.2, fmin=0.3, fmax=4.0):
    """
    Estimate frequency using YIN's cumulative mean normalized difference function.

    Returns (freq, is_valid).
    """
    n = len(envelope_channel)
    tau_min = int(fs / fmax)   # min lag (highest freq)
    tau_max = int(fs / fmin)   # max lag (lowest freq)
    tau_max = min(tau_max, n // 2)

    if tau_max <= tau_min or tau_min < 1:
        return np.nan, False

    # Compute difference function d(tau)
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = envelope_channel[:n - tau] - envelope_channel[tau:n]
        d[tau] = np.sum(diff ** 2)

    # Cumulative mean normalized difference function d'(tau)
    d_prime = np.ones(tau_max + 1)
    cumsum = 0.0
    for tau in range(1, tau_max + 1):
        cumsum += d[tau]
        if cumsum < 1e-10:
            d_prime[tau] = 1.0
        else:
            d_prime[tau] = d[tau] / (cumsum / tau)

    # Find first tau in valid range where d'(tau) < threshold
    for tau in range(tau_min, tau_max + 1):
        if d_prime[tau] < threshold:
            # Parabolic interpolation for sub-sample accuracy
            if tau > 0 and tau < tau_max:
                s0 = d_prime[tau - 1]
                s1 = d_prime[tau]
                s2 = d_prime[tau + 1]
                denom = 2.0 * (2.0 * s1 - s2 - s0)
                if abs(denom) > 1e-10:
                    tau_refined = tau + (s0 - s2) / denom
                else:
                    tau_refined = float(tau)
            else:
                tau_refined = float(tau)
            freq = fs / tau_refined
            return freq, True

    return np.nan, False


# ---------------------------------------------------------------------------
# SRH (Summation of Residual Harmonics) frequency estimation
# ---------------------------------------------------------------------------

def srh_frequency(envelope_channel, fs, fmin=0.3, fmax=3.5, n_harmonics=4):
    """
    Estimate frequency using Summation of Residual Harmonics.

    S(f) = E(f) + sum_{k=2}^{n_harmonics} [E(k*f) - 0.5 * E((k-0.5)*f)]

    Returns (freq, is_valid).
    """
    n = len(envelope_channel)
    if n < 10:
        return np.nan, False

    # Power spectrum
    fft_vals = np.fft.rfft(envelope_channel - np.mean(envelope_channel))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    freq_res = freqs[1] if len(freqs) > 1 else 1.0

    # Interpolation helper: get power at arbitrary frequency
    def get_power(f):
        idx = f / freq_res
        idx_lo = int(np.floor(idx))
        idx_hi = idx_lo + 1
        if idx_lo < 0 or idx_hi >= len(power):
            return 0.0
        frac = idx - idx_lo
        return power[idx_lo] * (1 - frac) + power[idx_hi] * frac

    candidates = np.arange(fmin, fmax + 0.01, 0.05)
    srh_scores = np.zeros(len(candidates))

    for ci, f0 in enumerate(candidates):
        s = get_power(f0)
        for k in range(2, n_harmonics + 1):
            s += get_power(k * f0) - 0.5 * get_power((k - 0.5) * f0)
        srh_scores[ci] = s

    best_idx = np.argmax(srh_scores)
    best_freq = candidates[best_idx]
    best_score = srh_scores[best_idx]

    # Validity: peak should be prominent
    mean_score = np.mean(srh_scores)
    is_valid = best_score > 2.0 * mean_score if mean_score > 0 else False

    return best_freq, is_valid


# ---------------------------------------------------------------------------
# FFT of envelope frequency estimation
# ---------------------------------------------------------------------------

def fft_envelope_frequency(envelope_channel, fs, fmin=0.3, fmax=3.5):
    """Find peak frequency in FFT of envelope within [fmin, fmax] Hz."""
    n = len(envelope_channel)
    if n < 10:
        return np.nan, False

    fft_vals = np.fft.rfft(envelope_channel - np.mean(envelope_channel))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan, False

    power_range = power[mask]
    freqs_range = freqs[mask]
    peak_idx = np.argmax(power_range)
    peak_freq = freqs_range[peak_idx]
    peak_power = power_range[peak_idx]
    mean_power = np.mean(power_range)

    is_valid = peak_power > 2.0 * mean_power if mean_power > 0 else False
    return float(peak_freq), is_valid


# ---------------------------------------------------------------------------
# Aggregate channel estimates
# ---------------------------------------------------------------------------

def aggregate_channel_estimates(freqs, valid_flags):
    """Take median of valid channel estimates."""
    valid = [f for f, v in zip(freqs, valid_flags) if v and np.isfinite(f)]
    if valid:
        return float(np.median(valid))
    return np.nan


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_segment(seg, fs, templates, method='yin', yin_threshold=0.2):
    """
    Process a preprocessed segment with matched filter + frequency estimation.

    method: 'yin', 'srh', or 'fft'
    Returns estimated frequency (float or nan).
    """
    envelope = compute_discharge_envelope(seg, templates)
    n_ch = envelope.shape[0]

    freqs = []
    valids = []

    for ch in range(n_ch):
        env_ch = envelope[ch]
        if np.std(env_ch) < 1e-10:
            freqs.append(np.nan)
            valids.append(False)
            continue

        if method == 'yin':
            f, v = yin_frequency(env_ch, fs, threshold=yin_threshold)
        elif method == 'srh':
            f, v = srh_frequency(env_ch, fs)
        elif method == 'fft':
            f, v = fft_envelope_frequency(env_ch, fs)
        else:
            f, v = np.nan, False

        freqs.append(f)
        valids.append(v)

    return aggregate_channel_estimates(freqs, valids)


# ---------------------------------------------------------------------------
# Run all variants
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Define all variants
    variants = {
        'r4_yin_bankC':        {'method': 'yin', 'bank': 'C', 'yin_threshold': 0.2},
        'r4_yin_bankD':        {'method': 'yin', 'bank': 'D', 'yin_threshold': 0.2},
        'r4_yin_t01_bankC':    {'method': 'yin', 'bank': 'C', 'yin_threshold': 0.1},
        'r4_yin_t03_bankC':    {'method': 'yin', 'bank': 'C', 'yin_threshold': 0.3},
        'r4_srh_bankC':        {'method': 'srh', 'bank': 'C'},
        'r4_srh_bankD':        {'method': 'srh', 'bank': 'D'},
        'r4_fft_envelope_C':   {'method': 'fft', 'bank': 'C'},
        'r4_fft_envelope_D':   {'method': 'fft', 'bank': 'D'},
    }

    # Initialize prediction dicts
    all_preds = {name: {} for name in variants}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"Processing segment {idx + 1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        seg = preprocess_segment(data, fs)
        subdir = entry['subdir']
        mat_name = entry['mat_name']

        # Pre-compute envelopes for each bank to avoid redundant work
        templates_C = templates_C_lpd if subdir == 'lpd' else templates_C_gpd
        envelope_C = compute_discharge_envelope(seg, templates_C)
        envelope_D = compute_discharge_envelope(seg, templates_D)

        for var_name, cfg in variants.items():
            bank = cfg['bank']
            envelope = envelope_C if bank == 'C' else envelope_D
            method = cfg['method']
            yin_threshold = cfg.get('yin_threshold', 0.2)

            n_ch = envelope.shape[0]
            freqs = []
            valids = []

            for ch in range(n_ch):
                env_ch = envelope[ch]
                if np.std(env_ch) < 1e-10:
                    freqs.append(np.nan)
                    valids.append(False)
                    continue

                if method == 'yin':
                    f, v = yin_frequency(env_ch, fs, threshold=yin_threshold)
                elif method == 'srh':
                    f, v = srh_frequency(env_ch, fs)
                elif method == 'fft':
                    f, v = fft_envelope_frequency(env_ch, fs)
                else:
                    f, v = np.nan, False

                freqs.append(f)
                valids.append(v)

            pred = aggregate_channel_estimates(freqs, valids)
            all_preds[var_name][mat_name] = pred

    # Evaluate all variants
    print("\n" + "=" * 60)
    print("EVALUATING ALL VARIANTS")
    print("=" * 60)

    for var_name in variants:
        n_valid = sum(1 for v in all_preds[var_name].values() if np.isfinite(v))
        print(f"\n{var_name}: {n_valid} valid predictions out of {len(all_preds[var_name])}")
        evaluate_predictions(dataset, all_preds[var_name], var_name)


if __name__ == '__main__':
    main()
