"""
Round 7: Multi-montage inputs + event-vs-background GED spatial filtering.

Key insight: We've been forcing everything into one longitudinal bipolar montage,
but experts use multiple views. Different montages capture different aspects of
epileptiform activity.

Creates 3 montage views from raw referential data:
  a) Bipolar (fcn_getBanana) -> 18 channels
  b) Common Average Reference (CAR) -> 19 channels
  c) Laplacian approximation -> 19 channels

Event-vs-background GED per montage:
  - Find candidate discharge times via pointiness on bipolar montage
  - Extract event/background epochs
  - Compute S_event, S_bg covariance matrices
  - GED: eigh(S_event, S_bg) -> top eigenvector -> project -> 1D signal
  - Frequency estimation on each GED component

Variants:
  r7_ged_bipolar_event   - event-vs-bg GED on bipolar montage -> FFT
  r7_ged_car_event       - event-vs-bg GED on CAR montage -> FFT
  r7_ged_multi_ridge     - Ridge on log(freq) with standard 10 features + GED features
  r7_multi_montage_fft   - Per-montage FFT of pointiness, median across montages
  r7_multi_montage_ridge - Ridge with per-montage frequency estimates as features
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
    pd_detect_pointiness_acf, bipolar_channels, mono_channels,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks, coherence
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from mne.filter import notch_filter, filter_data

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200
N_BIPOLAR = 18
N_MONO = 19  # first 19 channels (skip EKG if present)
EVENT_WIN_SAMPLES = 80  # 400ms at 200Hz

# 10-20 channel order (first 19, no EKG)
CHAN_NAMES_19 = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# Laplacian neighbors for each of the 19 channels in the 10-20 system
# Index mapping: 0=Fp1,1=F3,2=C3,3=P3,4=F7,5=T3,6=T5,7=O1,
#                8=Fz,9=Cz,10=Pz,11=Fp2,12=F4,13=C4,14=P4,15=F8,16=T4,17=T6,18=O2
LAPLACIAN_NEIGHBORS = {
    0:  [1, 4, 8],        # Fp1: F3, F7, Fz
    1:  [0, 2, 4, 8],     # F3: Fp1, C3, F7, Fz
    2:  [1, 3, 5, 9],     # C3: F3, P3, T3, Cz
    3:  [2, 6, 10],       # P3: C3, T5, Pz
    4:  [0, 1, 5],        # F7: Fp1, F3, T3
    5:  [4, 2, 6],        # T3: F7, C3, T5
    6:  [5, 3, 7],        # T5: T3, P3, O1
    7:  [6, 3, 10],       # O1: T5, P3, Pz
    8:  [0, 1, 9, 11, 12],# Fz: Fp1, F3, Cz, Fp2, F4
    9:  [2, 8, 10, 13],   # Cz: C3, Fz, Pz, C4
    10: [3, 9, 14, 7, 18],# Pz: P3, Cz, P4, O1, O2
    11: [12, 15, 8],      # Fp2: F4, F8, Fz
    12: [11, 13, 15, 8],  # F4: Fp2, C4, F8, Fz
    13: [12, 14, 16, 9],  # C4: F4, P4, T4, Cz
    14: [13, 17, 10],     # P4: C4, T6, Pz
    15: [11, 12, 16],     # F8: Fp2, F4, T4
    16: [15, 13, 17],     # T4: F8, C4, T6
    17: [16, 14, 18],     # T6: T4, P4, O2
    18: [17, 14, 10],     # O2: T6, P4, Pz
}

# Adjacent channel pairs for spectral coherence (on bipolar montage)
ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]

# Templates
REPO_ROOT = os.path.dirname(CODE_DIR)
try:
    TEMPLATES_C_LPD = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy'))
    TEMPLATES_C_GPD = np.load(os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy'))
except Exception:
    TEMPLATES_C_LPD = None
    TEMPLATES_C_GPD = None


# ── Helpers ───────────────────────────────────────────────────────────
def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def lowpass_signal(seg, fs, cutoff=15.0):
    b_lp, a_lp = butter(4, cutoff / (fs / 2), btype='low')
    out = seg.copy()
    for i in range(out.shape[0]):
        try:
            out[i] = filtfilt(b_lp, a_lp, out[i])
        except ValueError:
            pass
    return out


def preprocess_raw(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz on raw referential data."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    return seg


# ── Montage creation ─────────────────────────────────────────────────
def make_bipolar(data_preprocessed):
    """Bipolar longitudinal montage. Input: (>=19, n_samples). Output: (18, n_samples)."""
    return np.array(fcn_getBanana(data_preprocessed))


def make_car(data_preprocessed):
    """Common average reference. Input: (>=19, n_samples). Output: (19, n_samples)."""
    raw19 = data_preprocessed[:N_MONO]
    avg = np.mean(raw19, axis=0, keepdims=True)
    return raw19 - avg


def make_laplacian(data_preprocessed):
    """Laplacian approximation. Input: (>=19, n_samples). Output: (19, n_samples)."""
    raw19 = data_preprocessed[:N_MONO]
    lap = np.zeros_like(raw19)
    for i in range(N_MONO):
        neighbors = LAPLACIAN_NEIGHBORS[i]
        neighbor_mean = np.mean(raw19[neighbors], axis=0)
        lap[i] = raw19[i] - neighbor_mean
    return lap


# ── Pointiness / FFT helpers ─────────────────────────────────────────
def pointiness_fft_freq(signal_1d, fs):
    """FFT of pointiness trace, peak in [0.3, 3.5] Hz."""
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    if np.max(trace) <= 0:
        return np.nan
    n = len(trace)
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freq_sub = fft_freqs[mask]
    return float(freq_sub[np.argmax(fft_sub)])


def tkeo_fft_freq(signal_1d, fs):
    """FFT of |TKEO| trace, peak in [0.3, 3.5] Hz."""
    x = signal_1d
    if len(x) < 3:
        return np.nan
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    tkeo = np.abs(tkeo)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
    n = len(tkeo)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freq_sub = fft_freqs[mask]
    return float(freq_sub[np.argmax(fft_sub)])


def peak_count_freq(signal_1d, fs):
    """Peak-count on pointiness trace."""
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    trace_max = np.max(trace)
    if trace_max <= 0:
        return np.nan
    peak_height = trace_max * PEAK_HEIGHT_FRAC
    min_distance = int(0.2 * fs)
    peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
    if len(peak_locs) >= 3:
        return (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return np.nan


def per_channel_fft_median(seg_lp, fs):
    """FFT of pointiness per channel, return median."""
    n_ch = seg_lp.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freqs[i] = pointiness_fft_freq(seg_lp[i], fs)
    return median_finite(freqs)


# ── Event-vs-background GED ──────────────────────────────────────────
def find_candidate_peaks_bipolar(seg_bipolar_lp, fs):
    """Find candidate discharge times from pointiness on bipolar montage.

    Returns array of peak sample indices.
    """
    n_ch = seg_bipolar_lp.shape[0]
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # Compute pointiness amplitude per channel
    ch_amplitudes = np.zeros(n_ch)
    ch_traces = []
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg_bipolar_lp[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        ch_traces.append(trace)
        ch_amplitudes[i] = np.max(trace)

    # Top 3 channels by pointiness amplitude
    top3 = np.argsort(ch_amplitudes)[::-1][:3]

    # Merge peaks from top 3 channels
    all_peaks = []
    min_distance = int(0.2 * fs)
    for ch_idx in top3:
        trace = ch_traces[ch_idx]
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        all_peaks.extend(peak_locs.tolist())

    if len(all_peaks) == 0:
        return np.array([])

    # Sort and remove duplicates within 0.1s
    all_peaks = np.sort(np.unique(all_peaks))
    # Merge nearby peaks
    merged = [all_peaks[0]]
    for p in all_peaks[1:]:
        if p - merged[-1] > int(0.1 * fs):
            merged.append(p)
    return np.array(merged)


def event_vs_bg_ged(montage_data, peak_times, fs, n_components=1):
    """Event-vs-background GED on a montage.

    Args:
        montage_data: (n_ch, n_samples) montage data (lowpassed)
        peak_times: array of peak sample indices
        fs: sampling rate
        n_components: number of top components

    Returns:
        filtered_signals: (n_components, n_samples)
    """
    n_ch, n_samples = montage_data.shape
    half_win = EVENT_WIN_SAMPLES // 2  # 40 samples each side

    if len(peak_times) < 3:
        # Not enough events, return channel mean
        return np.mean(montage_data, axis=0, keepdims=True)[:n_components]

    # Extract event epochs
    event_covs = []
    for pt in peak_times:
        start = pt - half_win
        end = pt + half_win
        if start < 0 or end > n_samples:
            continue
        epoch = montage_data[:, start:end]
        cov = epoch @ epoch.T / epoch.shape[1]
        event_covs.append(cov)

    if len(event_covs) < 3:
        return np.mean(montage_data, axis=0, keepdims=True)[:n_components]

    S_event = np.mean(event_covs, axis=0)

    # Extract background epochs (between peaks)
    bg_covs = []
    for i in range(len(peak_times) - 1):
        mid = (peak_times[i] + peak_times[i + 1]) // 2
        start = mid - half_win
        end = mid + half_win
        if start < 0 or end > n_samples:
            continue
        # Make sure we're not overlapping with any event
        too_close = False
        for pt in peak_times:
            if abs(mid - pt) < EVENT_WIN_SAMPLES:
                too_close = True
                break
        if too_close:
            continue
        epoch = montage_data[:, start:end]
        cov = epoch @ epoch.T / epoch.shape[1]
        bg_covs.append(cov)

    if len(bg_covs) < 2:
        # Not enough background, use full-segment covariance as background
        S_bg = montage_data @ montage_data.T / n_samples
    else:
        S_bg = np.mean(bg_covs, axis=0)

    # Regularize
    S_bg += 0.01 * np.eye(n_ch)

    # GED
    try:
        eigenvalues, eigenvectors = eigh(S_event, S_bg)
    except np.linalg.LinAlgError:
        return np.mean(montage_data, axis=0, keepdims=True)[:n_components]

    # Take top n_components (largest eigenvalues, last in ascending order)
    filtered = np.zeros((n_components, n_samples))
    for k in range(n_components):
        idx = -(k + 1)
        w = eigenvectors[:, idx]
        filtered[k] = w @ montage_data

    return filtered


# ── Standard feature extractors (from r6_mega_combiner) ──────────────
def get_f_A(data, fs):
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        ch = r.get('channels', None)
        if ch is not None and hasattr(ch, '__len__'):
            n_ch = len(ch)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def get_f_B(data, fs):
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
        n_ch = len(r.get('channels', []))
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def compute_pointiness_traces(seg, fs):
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    traces = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return np.array(traces)


def get_f_peaks(traces, fs):
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
    templates = TEMPLATES_C_LPD if subdir == 'lpd' else TEMPLATES_C_GPD
    if templates is None:
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


# ── Ridge LOO-CV ─────────────────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    n = X.shape[0]
    predictions = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i+1]
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
        predictions[i] = pred[0]
    return predictions


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    N = len(dataset)
    print(f"Dataset: {N} segments")

    # ── Storage ───────────────────────────────────────────────────────
    # Simple variant predictions
    preds_ged_bipolar_event = {}
    preds_ged_car_event = {}
    preds_multi_montage_fft = {}

    # Feature names for ridge models
    # Standard 10 features + GED features + per-montage features
    STANDARD_FEATURES = [
        'f_A', 'f_B_thr010', 'f_peaks', 'f_fft', 'f_tkeo_fft',
        'f_envelope', 'f_spectral_coh', 'is_gpd', 'n_ch',
    ]
    GED_FEATURES = [
        'ged_bipolar_fft', 'ged_car_fft', 'ged_bipolar_tkeo',
    ]
    MONTAGE_FEATURES = [
        'fft_bipolar', 'fft_car', 'fft_laplacian',
    ]

    N_STD = len(STANDARD_FEATURES)
    N_GED = len(GED_FEATURES)
    N_MONTAGE = len(MONTAGE_FEATURES)
    N_ALL = N_STD + N_GED + N_MONTAGE

    features = np.full((N, N_ALL), np.nan)
    expert_freqs = np.full(N, np.nan)
    mat_names = []
    subdirs = []

    t0 = time.time()

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing {idx+1}/{N} ({elapsed:.1f}s elapsed)")

        mat_names.append(entry['mat_name'])
        subdirs.append(entry['subdir'])
        expert_freqs[idx] = entry['expert_consensus_freq']

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        try:
            n_channels = data.shape[0]
            if n_channels < 19:
                continue

            # ── Preprocess raw referential data ───────────────────
            data_pp = preprocess_raw(data, fs)

            # ── Create 3 montages ─────────────────────────────────
            bipolar = make_bipolar(data_pp)        # (18, n_samples)
            car = make_car(data_pp)                # (19, n_samples)
            laplacian = make_laplacian(data_pp)    # (19, n_samples)

            # ── Lowpass all montages ──────────────────────────────
            bipolar_lp = lowpass_signal(bipolar, fs, LOWPASS_HZ)
            car_lp = lowpass_signal(car, fs, LOWPASS_HZ)
            laplacian_lp = lowpass_signal(laplacian, fs, LOWPASS_HZ)

            # ── Standard features (on bipolar montage) ────────────
            # f_A
            f_A, n_ch_A = get_f_A(data, fs)
            features[idx, 0] = f_A

            # f_B (thr010)
            f_B, n_ch_B = get_f_B(data, fs)
            features[idx, 1] = f_B
            features[idx, 8] = n_ch_B  # n_ch

            # Pointiness traces on bipolar lowpassed
            traces_bp = compute_pointiness_traces(bipolar_lp, fs)

            # f_peaks
            features[idx, 2] = get_f_peaks(traces_bp, fs)

            # f_fft
            features[idx, 3] = get_f_fft(traces_bp, fs)

            # f_tkeo_fft
            features[idx, 4] = get_f_tkeo_fft(bipolar_lp, fs)

            # f_envelope
            features[idx, 5] = get_f_envelope(bipolar_lp, fs, entry['subdir'])

            # f_spectral_coh (on broadband bipolar)
            features[idx, 6] = get_f_spectral_coh(bipolar, fs)

            # is_gpd
            features[idx, 7] = 1.0 if entry['subdir'] == 'gpd' else 0.0

            # ── Find candidate peaks on bipolar montage ───────────
            peak_times = find_candidate_peaks_bipolar(bipolar_lp, fs)

            # ── Event-vs-background GED: Bipolar ─────────────────
            ged_bp = event_vs_bg_ged(bipolar_lp, peak_times, fs, n_components=1)
            ged_bp_signal = ged_bp[0]
            # Apply lowpass
            try:
                b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
                ged_bp_signal = filtfilt(b_lp, a_lp, ged_bp_signal)
            except Exception:
                pass

            f_ged_bp_fft = pointiness_fft_freq(ged_bp_signal, fs)
            f_ged_bp_tkeo = tkeo_fft_freq(ged_bp_signal, fs)
            f_ged_bp_peaks = peak_count_freq(ged_bp_signal, fs)

            preds_ged_bipolar_event[entry['mat_name']] = f_ged_bp_fft
            features[idx, N_STD + 0] = f_ged_bp_fft      # ged_bipolar_fft
            features[idx, N_STD + 2] = f_ged_bp_tkeo      # ged_bipolar_tkeo

            # ── Event-vs-background GED: CAR ─────────────────────
            ged_car = event_vs_bg_ged(car_lp, peak_times, fs, n_components=1)
            ged_car_signal = ged_car[0]
            try:
                ged_car_signal = filtfilt(b_lp, a_lp, ged_car_signal)
            except Exception:
                pass

            f_ged_car_fft = pointiness_fft_freq(ged_car_signal, fs)
            preds_ged_car_event[entry['mat_name']] = f_ged_car_fft
            features[idx, N_STD + 1] = f_ged_car_fft      # ged_car_fft

            # ── Per-montage FFT of pointiness (median across channels) ──
            fft_bipolar = per_channel_fft_median(bipolar_lp, fs)
            fft_car = per_channel_fft_median(car_lp, fs)
            fft_laplacian = per_channel_fft_median(laplacian_lp, fs)

            features[idx, N_STD + N_GED + 0] = fft_bipolar
            features[idx, N_STD + N_GED + 1] = fft_car
            features[idx, N_STD + N_GED + 2] = fft_laplacian

            # ── Multi-montage FFT median ──────────────────────────
            montage_estimates = [v for v in [fft_bipolar, fft_car, fft_laplacian]
                                 if np.isfinite(v)]
            if montage_estimates:
                preds_multi_montage_fft[entry['mat_name']] = float(np.median(montage_estimates))

        except Exception as e:
            if (idx + 1) % 50 == 0:
                print(f"    Error on {entry['mat_name']}: {e}")
            continue

    elapsed = time.time() - t0
    print(f"\nFeature extraction complete: {elapsed:.1f}s")
    print(f"GED bipolar predictions: {len(preds_ged_bipolar_event)}")
    print(f"GED CAR predictions: {len(preds_ged_car_event)}")
    print(f"Multi-montage FFT predictions: {len(preds_multi_montage_fft)}")

    # ── Feature coverage ──────────────────────────────────────────────
    all_feat_names = STANDARD_FEATURES + GED_FEATURES + MONTAGE_FEATURES
    print("\nFeature coverage:")
    for fi, fname in enumerate(all_feat_names):
        cnt = int(np.sum(np.isfinite(features[:, fi])))
        print(f"  {fname:>20s}: {cnt}/{N}")

    # ── Evaluate simple variants ──────────────────────────────────────
    print("\n--- Evaluating r7_ged_bipolar_event ---")
    evaluate_predictions(dataset, preds_ged_bipolar_event, 'r7_ged_bipolar_event')

    print("\n--- Evaluating r7_ged_car_event ---")
    evaluate_predictions(dataset, preds_ged_car_event, 'r7_ged_car_event')

    print("\n--- Evaluating r7_multi_montage_fft ---")
    evaluate_predictions(dataset, preds_multi_montage_fft, 'r7_multi_montage_fft')

    # ── Impute NaN with column median for ridge ───────────────────────
    def prepare_for_ridge(feat_indices):
        X_raw = features[:, feat_indices].copy()
        valid = np.isfinite(expert_freqs)
        X = X_raw[valid].copy()
        y = expert_freqs[valid].copy()
        names_v = [mat_names[i] for i in range(N) if valid[i]]
        # Impute NaN with column median
        for fi in range(X.shape[1]):
            col = X[:, fi]
            nan_mask = ~np.isfinite(col)
            if np.any(nan_mask):
                med = np.nanmedian(col)
                if not np.isfinite(med):
                    med = 1.0
                col[nan_mask] = med
                X[:, fi] = col
        return X, y, names_v

    # ── r7_ged_multi_ridge ────────────────────────────────────────────
    # Standard 9 features + GED bipolar FFT + GED CAR FFT + GED bipolar TKEO
    print("\n--- Running r7_ged_multi_ridge (LOO-CV) ---")
    ridge_idx = list(range(N_STD)) + [N_STD + 0, N_STD + 1, N_STD + 2]
    X_ridge, y_ridge, names_ridge = prepare_for_ridge(ridge_idx)
    print(f"  Ridge: {X_ridge.shape[0]} samples, {X_ridge.shape[1]} features")

    if X_ridge.shape[0] > 5:
        y_log = np.log(y_ridge)
        preds_log = ridge_loo_cv(X_ridge, y_log, alpha=1.0)
        preds_ridge = np.exp(preds_log)
        # Clamp
        preds_ridge = np.clip(preds_ridge, 0.3, 4.0)
        pred_dict = {}
        for i in range(len(names_ridge)):
            if np.isfinite(preds_ridge[i]):
                pred_dict[names_ridge[i]] = float(preds_ridge[i])
        evaluate_predictions(dataset, pred_dict, 'r7_ged_multi_ridge')
    else:
        print("  Not enough samples for ridge")

    # ── r7_multi_montage_ridge ────────────────────────────────────────
    # Standard 9 features + per-montage FFT estimates
    print("\n--- Running r7_multi_montage_ridge (LOO-CV) ---")
    montage_ridge_idx = list(range(N_STD)) + [N_STD + N_GED + i for i in range(N_MONTAGE)]
    X_mr, y_mr, names_mr = prepare_for_ridge(montage_ridge_idx)
    print(f"  Ridge: {X_mr.shape[0]} samples, {X_mr.shape[1]} features")

    if X_mr.shape[0] > 5:
        y_log = np.log(y_mr)
        preds_log = ridge_loo_cv(X_mr, y_log, alpha=1.0)
        preds_mr = np.exp(preds_log)
        preds_mr = np.clip(preds_mr, 0.3, 4.0)
        pred_dict = {}
        for i in range(len(names_mr)):
            if np.isfinite(preds_mr[i]):
                pred_dict[names_mr[i]] = float(preds_mr[i])
        evaluate_predictions(dataset, pred_dict, 'r7_multi_montage_ridge')
    else:
        print("  Not enough samples for ridge")

    total = time.time() - t0
    print(f"\nTotal time: {total:.0f}s")
    print("All r7 evaluations complete.")


if __name__ == '__main__':
    main()
