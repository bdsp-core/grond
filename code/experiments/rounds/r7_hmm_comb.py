"""
Round 7: Within-segment candidate tracking with HMM-style decoding and comb-fit scoring.

KEY INSIGHT: Instead of picking one global frequency from a single FFT/ACF peak,
track frequency candidates across overlapping windows and decode the most consistent
frequency using HMM/Viterbi logic. This directly targets subharmonic locking.

Variants:
  r7_windowed_vote  - Approach A: windowed candidate tracking with voting
  r7_hmm_viterbi    - Approach B: HMM Viterbi decoding over windowed candidates
  r7_comb_fit       - Approach C: comb-fit scoring with pointiness trace
  r7_comb_fit_tkeo  - Approach C: comb-fit scoring with TKEO trace
  r7_decoder_ridge  - Ridge on log(freq) with standard + new features, LOO-CV
"""

import sys
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    pd_detect_pointiness_acf, bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks, coherence
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

# Frequency grid for windowed voting and HMM
FREQ_BINS = np.arange(0.25, 3.75, 0.25)  # 0.25, 0.50, ..., 3.50 -> 13 bins
N_STATES = len(FREQ_BINS)

# Window parameters
WIN_SEC = 3.0
HOP_SEC = 0.5

# HMM transition matrix
def build_transition_matrix():
    """Build HMM transition matrix. Diagonal-dominant with subharmonic penalties."""
    n = N_STATES
    T = np.full((n, n), 1e-6)  # base uniform
    for i in range(n):
        T[i, i] = 0.7  # self-transition
        # Adjacent bins
        if i > 0:
            T[i, i - 1] = 0.05
        if i < n - 1:
            T[i, i + 1] = 0.05
        # Octave jumps (penalized)
        fi = FREQ_BINS[i]
        for j in range(n):
            fj = FREQ_BINS[j]
            ratio = fj / fi if fi > 0 else 0
            if abs(ratio - 0.5) < 0.05:  # f -> f/2
                T[i, j] = 0.001
            elif abs(ratio - 1.0 / 3.0) < 0.05:  # f -> f/3
                T[i, j] = 0.0005
            elif abs(ratio - 2.0) < 0.1:  # f -> 2f
                T[i, j] = 0.001
    # Normalize rows
    for i in range(n):
        T[i] /= T[i].sum()
    return T

TRANS_MATRIX = build_transition_matrix()
LOG_TRANS = np.log(TRANS_MATRIX + 1e-30)

# Emission: gaussian kernel sigma
EMISSION_SIGMA = 0.3  # Hz


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


# ── Windowed FFT candidates ──────────────────────────────────────────
def get_windowed_candidates(trace, fs, n_top=3):
    """Split trace into overlapping windows, compute FFT, return top candidates per window.

    Returns list of lists: candidates[window_idx] = list of (freq, amplitude) tuples.
    """
    n_samples = len(trace)
    win_samples = int(WIN_SEC * fs)
    hop_samples = int(HOP_SEC * fs)

    if n_samples < win_samples:
        return []

    candidates = []
    start = 0
    while start + win_samples <= n_samples:
        window = trace[start:start + win_samples]
        window = window - np.mean(window)
        if np.max(np.abs(window)) < 1e-10:
            candidates.append([])
            start += hop_samples
            continue

        fft_vals = np.abs(np.fft.rfft(window))
        fft_freqs = np.fft.rfftfreq(win_samples, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)

        if not np.any(mask):
            candidates.append([])
            start += hop_samples
            continue

        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]

        # Find top n_top peaks
        if len(fft_sub) < 3:
            candidates.append([])
            start += hop_samples
            continue

        # Use find_peaks for proper peak detection
        peaks_idx, props = find_peaks(fft_sub, height=0)
        if len(peaks_idx) == 0:
            # Fallback: just take the argmax
            best_idx = np.argmax(fft_sub)
            candidates.append([(freq_sub[best_idx], fft_sub[best_idx])])
        else:
            # Sort by amplitude, take top n_top
            sorted_peaks = sorted(peaks_idx, key=lambda p: fft_sub[p], reverse=True)[:n_top]
            cands = [(freq_sub[p], fft_sub[p]) for p in sorted_peaks]
            candidates.append(cands)

        start += hop_samples

    return candidates


# ── Approach A: Windowed Vote ─────────────────────────────────────────
def windowed_vote_channel(trace, fs):
    """Windowed voting for a single channel. Returns frequency or NaN."""
    candidates = get_windowed_candidates(trace, fs)
    if not candidates:
        return np.nan

    votes = np.zeros(N_STATES)
    for window_cands in candidates:
        if not window_cands:
            continue
        # Top candidate votes for nearest bin
        best_freq = window_cands[0][0]
        bin_idx = np.argmin(np.abs(FREQ_BINS - best_freq))
        votes[bin_idx] += 1

    if np.sum(votes) == 0:
        return np.nan
    return float(FREQ_BINS[np.argmax(votes)])


def windowed_vote_segment(seg, fs):
    """Approach A: windowed vote, median across channels."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        if np.max(trace) <= 0:
            continue
        freqs[i] = windowed_vote_channel(trace, fs)
    return median_finite(freqs)


# ── Approach B: HMM Viterbi ──────────────────────────────────────────
def viterbi_channel(trace, fs):
    """Run Viterbi decoding on windowed candidates for a single channel.
    Returns modal state frequency or NaN."""
    candidates = get_windowed_candidates(trace, fs)
    if not candidates or len(candidates) < 2:
        return np.nan

    n_windows = len(candidates)

    # Compute emission log-probabilities: P(obs | state) for each window
    # Using gaussian kernel centered on each state frequency
    log_emission = np.full((n_windows, N_STATES), -20.0)  # very low default
    for t, window_cands in enumerate(candidates):
        if not window_cands:
            # Uniform if no candidates
            log_emission[t, :] = np.log(1.0 / N_STATES)
            continue
        for s in range(N_STATES):
            state_freq = FREQ_BINS[s]
            # Sum gaussian contributions from all candidates (weighted by amplitude)
            total = 0.0
            for freq, amp in window_cands:
                gauss = np.exp(-0.5 * ((freq - state_freq) / EMISSION_SIGMA) ** 2)
                total += gauss * amp
            if total > 0:
                log_emission[t, s] = np.log(total + 1e-30)

    # Viterbi algorithm in log space
    # Initialize
    log_pi = np.log(np.ones(N_STATES) / N_STATES)  # uniform prior
    V = np.full((n_windows, N_STATES), -np.inf)
    backptr = np.zeros((n_windows, N_STATES), dtype=int)

    V[0] = log_pi + log_emission[0]

    for t in range(1, n_windows):
        for s in range(N_STATES):
            # V[t, s] = max over prev states of V[t-1, prev] + log_trans[prev, s] + log_emission[t, s]
            scores = V[t - 1] + LOG_TRANS[:, s]
            best_prev = np.argmax(scores)
            V[t, s] = scores[best_prev] + log_emission[t, s]
            backptr[t, s] = best_prev

    # Backtrace
    path = np.zeros(n_windows, dtype=int)
    path[-1] = np.argmax(V[-1])
    for t in range(n_windows - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]

    # Modal state
    counts = np.bincount(path, minlength=N_STATES)
    modal_state = np.argmax(counts)
    return float(FREQ_BINS[modal_state])


def hmm_viterbi_segment(seg, fs):
    """Approach B: HMM Viterbi, median across channels."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        if np.max(trace) <= 0:
            continue
        freqs[i] = viterbi_channel(trace, fs)
    return median_finite(freqs)


# ── Approach C: Comb-fit scoring ──────────────────────────────────────
def comb_fit_channel(eventness, fs, tolerance_sec=0.05):
    """Comb-fit scoring for a single channel.
    eventness: smoothed eventness trace (pointiness or TKEO).
    Returns best frequency or NaN.
    """
    n_samples = len(eventness)
    duration = n_samples / fs

    if np.max(eventness) <= 0:
        return np.nan

    # Find peaks in eventness trace
    peak_height = np.max(eventness) * 0.2
    min_dist = int(0.15 * fs)
    peak_locs, peak_props = find_peaks(eventness, height=peak_height, distance=min_dist)
    if len(peak_locs) < 2:
        return np.nan
    peak_vals = eventness[peak_locs]

    tol_samples = int(tolerance_sec * fs)

    best_score = -np.inf
    best_T = np.nan

    for T in np.arange(0.28, 3.3, 0.02):
        T_samples = T * fs
        n_expected = int(duration / T)
        if n_expected < 3:
            continue

        best_phase_score = -np.inf
        n_phases = max(1, int(T / (T / 20)))
        for phi_idx in range(n_phases):
            phi = phi_idx * T / n_phases * fs  # in samples

            score = 0.0
            n_matched = 0
            n_teeth = 0

            tooth = phi
            while tooth < n_samples:
                n_teeth += 1
                tooth_int = int(round(tooth))
                # Find nearest peak within tolerance
                lo = max(0, tooth_int - tol_samples)
                hi = min(n_samples, tooth_int + tol_samples + 1)

                # Check if any peak falls within tolerance
                matched = False
                for pl_idx in range(len(peak_locs)):
                    if lo <= peak_locs[pl_idx] <= hi:
                        score += peak_vals[pl_idx]
                        matched = True
                        n_matched += 1
                        break

                if not matched:
                    # Penalty for missing tooth
                    score -= np.median(peak_vals) * 0.5

                tooth += T_samples

            if n_teeth > 0:
                # Normalize by number of expected teeth
                score = score / n_teeth
                # Bonus for high match rate
                match_rate = n_matched / n_teeth if n_teeth > 0 else 0
                score *= (1.0 + match_rate)

            if score > best_phase_score:
                best_phase_score = score

        if best_phase_score > best_score:
            best_score = best_phase_score
            best_T = T

    if np.isfinite(best_T) and best_T > 0:
        return 1.0 / best_T
    return np.nan


def comb_fit_segment(seg, fs, use_tkeo=False):
    """Approach C: comb-fit, median across channels."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        if use_tkeo:
            x = seg[i]
            if len(x) < 3:
                continue
            eventness = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
            eventness = gaussian_filter1d(eventness, sigma=sigma_samples)
        else:
            eventness = compute_pointiness_trace(seg[i])
            eventness = gaussian_filter1d(eventness, sigma=sigma_samples)
        if np.max(eventness) <= 0:
            continue
        freqs[i] = comb_fit_channel(eventness, fs)
    return median_finite(freqs)


# ── Standard feature extractors (for ridge) ──────────────────────────
def get_f_A(data, fs):
    """Method A: pd_detect_alternate(apd) event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def get_f_B(data, fs):
    """Method B: pd_detect_pointiness_acf with acf_peak_threshold=0.10."""
    try:
        r = pd_detect_pointiness_acf(
            data, fs, method='pointiness',
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            smoothing_sigma=SMOOTHING_SIGMA,
            lowpass_hz=LOWPASS_HZ,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def compute_pointiness_traces(seg, fs):
    """Compute smoothed pointiness trace per channel."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    traces = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return np.array(traces)


def get_f_peaks(traces, fs):
    """Peak-count on pointiness. Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = traces[i]
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return median_finite(freqs)


def get_f_fft(traces, fs):
    """FFT of pointiness, peak in [0.3, 3.5] Hz. Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
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


def get_f_tkeo_fft(seg, fs):
    """FFT of |TKEO| trace. Median across channels."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(0.02 * fs))
    for i in range(n_ch):
        x = seg[i]
        if len(x) < 3:
            continue
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
        tkeo = np.abs(tkeo)
        tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
        n = len(tkeo)
        if n < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_envelope(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C). Median across channels."""
    REPO_ROOT = os.path.dirname(CODE_DIR)
    try:
        if subdir == 'lpd':
            templates = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
        else:
            templates = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
    except Exception:
        return np.nan
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        ch = seg[i]
        n_samples = len(ch)
        envelope = np.zeros(n_samples)
        for t_idx in range(templates.shape[0]):
            tmpl = templates[t_idx]
            if len(tmpl) > n_samples:
                tmpl = tmpl[:n_samples]
            corr = np.abs(np.correlate(ch, tmpl, mode='same'))
            envelope = np.maximum(envelope, corr)
        if np.max(envelope) <= 0:
            continue
        n = len(envelope)
        fft_vals = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_spectral_coh(seg, fs):
    """Spectral coherence peak across adjacent channel pairs."""
    ADJACENT_PAIRS = [
        (0, 1), (1, 2), (2, 3),
        (4, 5), (5, 6), (6, 7),
        (8, 9), (9, 10), (10, 11),
        (12, 13), (13, 14), (14, 15),
        (16, 17),
    ]
    nperseg = min(256, seg.shape[1] // 2)
    if nperseg < 16:
        return np.nan
    coh_spectra = None
    coh_freqs = None
    for (a, b) in ADJACENT_PAIRS:
        if a >= seg.shape[0] or b >= seg.shape[0]:
            continue
        try:
            f_coh, Cxy = coherence(seg[a], seg[b], fs=fs, nperseg=nperseg)
            if coh_freqs is None:
                coh_freqs = f_coh
                coh_spectra = np.zeros_like(f_coh)
            coh_spectra = coh_spectra + Cxy
        except Exception:
            continue
    if coh_freqs is None:
        return np.nan
    coh_spectra /= len(ADJACENT_PAIRS)
    mask = (coh_freqs >= 0.3) & (coh_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    coh_sub = coh_spectra[mask]
    freq_sub = coh_freqs[mask]
    return float(freq_sub[np.argmax(coh_sub)])


def get_f_hps3(traces, fs):
    """Harmonic Product Spectrum on pointiness FFT. Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        max_idx = len(fft_vals) // 3
        if max_idx < 2:
            continue
        hps = fft_vals[:max_idx].copy()
        hps *= fft_vals[:max_idx * 2:2][:max_idx]
        hps *= fft_vals[:max_idx * 3:3][:max_idx]
        hps_freqs = fft_freqs[:max_idx]
        mask = (hps_freqs >= 0.3) & (hps_freqs <= 3.5)
        if not np.any(mask):
            continue
        hps_sub = hps[mask]
        freq_sub = hps_freqs[mask]
        freqs[i] = freq_sub[np.argmax(hps_sub)]
    return median_finite(freqs)


# ── Ridge LOO-CV ──────────────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """LOO-CV Ridge regression. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]
        # Standardize
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_tr_s = (X_train - mu) / std
        X_te_s = (X_test - mu) / std
        y_mu = np.mean(y_train)
        y_c = y_train - y_mu
        p = X_tr_s.shape[1]
        A = X_tr_s.T @ X_tr_s + alpha * np.eye(p)
        try:
            w = np.linalg.solve(A, X_tr_s.T @ y_c)
        except np.linalg.LinAlgError:
            continue
        pred = X_te_s @ w + y_mu
        preds[i] = pred[0]
    return preds


def impute_nan_with_median(X, feature_names=None):
    """Replace NaN values with column median."""
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X[nan_mask, col_idx] = col_median
            n_imputed = int(np.sum(nan_mask))
            name = feature_names[col_idx] if feature_names else f"col_{col_idx}"
            print(f"  Imputed {n_imputed} NaN in {name} (median={col_median:.3f})")


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Loading dataset...")
    dataset = load_dataset()
    N = len(dataset)
    print(f"Dataset: {N} segments")

    # Feature storage
    FEATURE_NAMES = [
        'f_A', 'f_B_thr010', 'f_peaks', 'f_fft', 'f_tkeo_fft',
        'f_envelope', 'f_spectral_coh', 'f_hps3',
        'is_gpd', 'n_ch',
        'f_windowed_vote', 'f_hmm_viterbi', 'f_comb_fit', 'f_comb_fit_tkeo',
    ]
    features = np.full((N, len(FEATURE_NAMES)), np.nan)
    expert_freqs = np.full(N, np.nan)
    mat_names = []
    subdirs = []

    # Predictions for simple variants
    preds_vote = {}
    preds_hmm = {}
    preds_comb = {}
    preds_comb_tkeo = {}

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing segment {idx+1}/{N}  ({elapsed:.0f}s elapsed)")

        mat_names.append(entry['mat_name'])
        subdirs.append(entry['subdir'])
        expert_freqs[idx] = entry['expert_consensus_freq']

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        # ─ Method A (does its own preprocessing) ──────────────
        f_A, n_ch_A = get_f_A(data, fs)
        features[idx, 0] = f_A

        # ─ Method B (does its own preprocessing) ─────────────
        f_B, n_ch_B = get_f_B(data, fs)
        features[idx, 1] = f_B
        features[idx, 9] = n_ch_B  # n_ch

        # ─ Standard preprocessing ────────────────────────────
        seg = preprocess_segment(data, fs)

        # Also need broadband bipolar (no 15Hz LP) for spectral coherence
        seg_bb = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
        seg_bb = filter_data(seg_bb, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        seg_bb = np.array(fcn_getBanana(seg_bb))

        # Pointiness traces (reused)
        traces = compute_pointiness_traces(seg, fs)

        # ─ f_peaks ────────────────────────────────────────────
        features[idx, 2] = get_f_peaks(traces, fs)

        # ─ f_fft ──────────────────────────────────────────────
        features[idx, 3] = get_f_fft(traces, fs)

        # ─ f_tkeo_fft ─────────────────────────────────────────
        features[idx, 4] = get_f_tkeo_fft(seg, fs)

        # ─ f_envelope ─────────────────────────────────────────
        features[idx, 5] = get_f_envelope(seg, fs, entry['subdir'])

        # ─ f_spectral_coh ─────────────────────────────────────
        features[idx, 6] = get_f_spectral_coh(seg_bb, fs)

        # ─ f_hps3 ─────────────────────────────────────────────
        features[idx, 7] = get_f_hps3(traces, fs)

        # ─ is_gpd ─────────────────────────────────────────────
        features[idx, 8] = 1.0 if entry['subdir'] == 'gpd' else 0.0

        # ─ Approach A: Windowed Vote ──────────────────────────
        f_wv = windowed_vote_segment(seg, fs)
        features[idx, 10] = f_wv
        if np.isfinite(f_wv):
            preds_vote[entry['mat_name']] = float(f_wv)

        # ─ Approach B: HMM Viterbi ───────────────────────────
        f_hmm = hmm_viterbi_segment(seg, fs)
        features[idx, 11] = f_hmm
        if np.isfinite(f_hmm):
            preds_hmm[entry['mat_name']] = float(f_hmm)

        # ─ Approach C: Comb-fit (pointiness) ─────────────────
        f_comb = comb_fit_segment(seg, fs, use_tkeo=False)
        features[idx, 12] = f_comb
        if np.isfinite(f_comb):
            preds_comb[entry['mat_name']] = float(f_comb)

        # ─ Approach C: Comb-fit (TKEO) ───────────────────────
        f_comb_tkeo = comb_fit_segment(seg, fs, use_tkeo=True)
        features[idx, 13] = f_comb_tkeo
        if np.isfinite(f_comb_tkeo):
            preds_comb_tkeo[entry['mat_name']] = float(f_comb_tkeo)

    elapsed = time.time() - t0
    print(f"Feature extraction done in {elapsed:.0f}s")

    # ── Print feature coverage ────────────────────────────────────────
    print("\nFeature coverage (non-NaN counts):")
    for fi, fname in enumerate(FEATURE_NAMES):
        cnt = np.sum(np.isfinite(features[:, fi]))
        print(f"  {fname:>20s}: {cnt}/{N}")

    # ── Evaluate simple variants ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("Evaluating r7_windowed_vote...")
    evaluate_predictions(dataset, preds_vote, 'r7_windowed_vote')

    print(f"\n{'='*60}")
    print("Evaluating r7_hmm_viterbi...")
    evaluate_predictions(dataset, preds_hmm, 'r7_hmm_viterbi')

    print(f"\n{'='*60}")
    print("Evaluating r7_comb_fit...")
    evaluate_predictions(dataset, preds_comb, 'r7_comb_fit')

    print(f"\n{'='*60}")
    print("Evaluating r7_comb_fit_tkeo...")
    evaluate_predictions(dataset, preds_comb_tkeo, 'r7_comb_fit_tkeo')

    # ── Approach E: Ridge on log(freq) with all features ─────────────
    print(f"\n{'='*60}")
    print("Running r7_decoder_ridge...")

    # Build valid mask
    valid = np.isfinite(expert_freqs) & (expert_freqs > 0)
    X = features[valid].copy()
    y = expert_freqs[valid].copy()
    valid_mat_names = [mat_names[i] for i in range(N) if valid[i]]

    print(f"  Ridge samples: {len(y)}")
    impute_nan_with_median(X, FEATURE_NAMES)

    y_log = np.log(y)
    preds_log = ridge_loo_cv(X, y_log, alpha=1.0)
    preds_ridge = np.exp(preds_log)
    preds_ridge = np.clip(preds_ridge, 0.2, 4.0)

    pred_dict_ridge = {}
    for i in range(len(valid_mat_names)):
        if np.isfinite(preds_ridge[i]):
            pred_dict_ridge[valid_mat_names[i]] = float(preds_ridge[i])

    evaluate_predictions(dataset, pred_dict_ridge, 'r7_decoder_ridge')

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")
    print("Done! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
