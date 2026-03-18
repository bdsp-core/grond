"""
R4: Cross-channel salience aggregation experiment.

Aggregate frequency evidence across channels BEFORE picking a frequency,
instead of per-channel-frequency then median.

Methods:
  r4_salience_uniform       - equal-weight sum of per-channel salience
  r4_salience_weighted      - quality-weighted sum
  r4_salience_top5          - sum of top-5 channels by quality
  r4_salience_srh           - uniform aggregation + SRH scoring
  r4_salience_weighted_srh  - weighted aggregation + SRH scoring
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, fftconvolve
from mne.filter import notch_filter, filter_data

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import fcn_getBanana

# Load templates
DATA_DIR = CODE_DIR.parent / 'data'
templates_C_lpd = np.load(DATA_DIR / 'templates_C_lpd.npy')  # [8, 50]
templates_C_gpd = np.load(DATA_DIR / 'templates_C_gpd.npy')  # [8, 50]

# Frequency grid
FREQ_GRID = np.arange(0.3, 3.5, 0.02)


def matched_filter_envelope(channel_signal, templates):
    """Compute matched-filter envelope for one channel.

    Cross-correlate with all templates, take max across templates at each time point.
    Returns the envelope (1D array, same length as channel_signal).
    """
    n = len(channel_signal)
    envelope = np.zeros(n)
    for t in range(templates.shape[0]):
        tmpl = templates[t]
        # Normalize template
        tmpl = tmpl - np.mean(tmpl)
        norm = np.linalg.norm(tmpl)
        if norm < 1e-10:
            continue
        tmpl = tmpl / norm
        # Cross-correlate (use fftconvolve for speed)
        xcorr = fftconvolve(channel_signal, tmpl[::-1], mode='same')
        envelope = np.maximum(envelope, np.abs(xcorr))
    return envelope


def salience_curve(envelope, fs, freq_grid):
    """Compute salience curve S(f) = |FFT(envelope)|^2 evaluated at freq_grid."""
    env = envelope - np.mean(envelope)
    n = len(env)
    fft_vals = np.fft.rfft(env)
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    power = np.abs(fft_vals) ** 2
    # Interpolate power at the requested frequencies
    s = np.interp(freq_grid, fft_freqs, power, left=0.0, right=0.0)
    return s


def apply_srh(S, freq_grid):
    """Apply Subharmonic-to-Harmonic Ratio scoring.

    SRH(f) = S(f) + sum_{k=2}^{4} [S(k*f) - 0.5*S((k-0.5)*f)]

    Out-of-range frequencies are treated as 0.
    """
    f_max = freq_grid[-1]
    srh = S.copy()
    for k in range(2, 5):
        for i, f in enumerate(freq_grid):
            harmonic_f = k * f
            subharm_f = (k - 0.5) * f
            # Lookup harmonic
            if harmonic_f <= f_max:
                h_val = np.interp(harmonic_f, freq_grid, S)
            else:
                h_val = 0.0
            # Lookup sub-harmonic
            if subharm_f <= f_max:
                sh_val = np.interp(subharm_f, freq_grid, S)
            else:
                sh_val = 0.0
            srh[i] += h_val - 0.5 * sh_val
    return srh


def process_segment(entry, data, fs):
    """Process one segment. Returns dict of method_name -> predicted_freq."""
    # Preprocess
    data = notch_filter(data.astype(np.float64), fs, 60, n_jobs=1, verbose="ERROR")
    data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(data))

    # 15Hz lowpass
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    # Select templates based on subdir
    if entry['subdir'] == 'lpd':
        templates = templates_C_lpd
    else:
        templates = templates_C_gpd

    n_channels = seg.shape[0]  # 18

    # Per-channel: matched-filter envelope, salience curve, quality weight
    saliences = np.zeros((n_channels, len(FREQ_GRID)))
    weights = np.zeros(n_channels)

    for c in range(n_channels):
        env = matched_filter_envelope(seg[c], templates)
        saliences[c] = salience_curve(env, fs, FREQ_GRID)
        weights[c] = np.max(env)

    # --- Aggregation methods ---
    results = {}

    # 1) Uniform
    S_uniform = np.sum(saliences, axis=0)
    results['r4_salience_uniform'] = FREQ_GRID[np.argmax(S_uniform)]

    # 2) Weighted
    S_weighted = np.sum(weights[:, None] * saliences, axis=0)
    results['r4_salience_weighted'] = FREQ_GRID[np.argmax(S_weighted)]

    # 3) Top-5
    top5_idx = np.argsort(weights)[-5:]
    S_top5 = np.sum(saliences[top5_idx], axis=0)
    results['r4_salience_top5'] = FREQ_GRID[np.argmax(S_top5)]

    # 4) SRH on uniform
    srh_uniform = apply_srh(S_uniform, FREQ_GRID)
    results['r4_salience_srh'] = FREQ_GRID[np.argmax(srh_uniform)]

    # 5) SRH on weighted
    srh_weighted = apply_srh(S_weighted, FREQ_GRID)
    results['r4_salience_weighted_srh'] = FREQ_GRID[np.argmax(srh_weighted)]

    return results


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    method_names = [
        'r4_salience_uniform',
        'r4_salience_weighted',
        'r4_salience_top5',
        'r4_salience_srh',
        'r4_salience_weighted_srh',
    ]
    predictions = {m: {} for m in method_names}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(dataset)}")

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            res = process_segment(entry, data, fs)
            for m in method_names:
                predictions[m][entry['mat_name']] = res[m]
        except Exception as e:
            print(f"  Error on {entry['mat_name']}: {e}")
            continue

    print(f"\nProcessed {len(predictions[method_names[0]])} segments successfully")

    # Evaluate each method
    for m in method_names:
        evaluate_predictions(dataset, predictions[m], m)


if __name__ == '__main__':
    main()
