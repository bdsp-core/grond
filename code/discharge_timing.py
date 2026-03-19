"""
Discharge timing detection for LPDs and GPDs.

Detects individual discharge times on each channel using:
1. Per-channel event evidence (pointiness + TKEO)
2. Aggregate evidence by subtype (GPD vs LPD)
3. Dynamic programming for optimal discharge sequence

Usage:
    from discharge_timing import detect_discharge_times
    result = detect_discharge_times(segment, fs=200, freq_estimate=1.0, subtype='gpd')
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d

# Add code/ to path for imports
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, fcn_getBanana

# Channel groupings
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_INDICES = [16, 17]


def _compute_channel_evidence(signal_1d, fs):
    """Compute event evidence E_c(t) for a single channel.

    Steps:
    1. Apply 15 Hz lowpass (butter order 4, filtfilt)
    2. Compute pointiness trace
    3. Compute TKEO
    4. Smooth both with gaussian_filter1d
    5. Z-score normalize each
    6. Combine: E_c(t) = 0.5 * pointiness_zscore + 0.5 * tkeo_zscore
    """
    n = len(signal_1d)
    if n < 10:
        return np.zeros(n)

    # 1. Lowpass at 15 Hz
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    try:
        sig_lp = filtfilt(b_lp, a_lp, signal_1d)
    except ValueError:
        sig_lp = signal_1d.copy()

    # 2. Pointiness trace
    pt = compute_pointiness_trace(sig_lp)

    # 3. TKEO
    tkeo = np.zeros(n)
    if n >= 3:
        tkeo[1:-1] = np.abs(sig_lp[1:-1] ** 2 - sig_lp[:-2] * sig_lp[2:])

    # 4. Smooth both
    sigma = max(1, int(0.02 * fs))
    pt_smooth = gaussian_filter1d(pt, sigma=sigma)
    tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma)

    # 5. Z-score normalize
    def zscore(x):
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return np.zeros_like(x)
        return (x - m) / s

    pt_z = zscore(pt_smooth)
    tkeo_z = zscore(tkeo_smooth)

    # 6. Combine
    evidence = 0.5 * pt_z + 0.5 * tkeo_z
    return evidence


def _auto_detect_involved(evidence_all, n_channels):
    """Auto-detect involved channels from evidence signals.

    Channels with mean evidence > 0.5 std above overall mean are "involved".
    If fewer than 3 channels qualify, take top 6 by mean evidence.
    """
    mean_evidence = np.array([np.mean(evidence_all[ch]) for ch in range(n_channels)])
    overall_mean = np.mean(mean_evidence)
    overall_std = np.std(mean_evidence)

    threshold = overall_mean + 0.5 * overall_std
    involved = [ch for ch in range(n_channels) if mean_evidence[ch] > threshold]

    if len(involved) < 3:
        sorted_chs = sorted(range(n_channels), key=lambda c: mean_evidence[c], reverse=True)
        involved = sorted_chs[:min(6, n_channels)]

    return involved


def _aggregate_evidence(evidence_all, involved_channels, subtype, n_channels):
    """Aggregate evidence by subtype.

    GPD: mean across all involved channels
    LPD: mean across involved channels in dominant hemisphere only
    """
    if not involved_channels:
        return np.zeros(evidence_all.shape[1])

    if subtype == 'gpd':
        # Mean across all involved channels
        return np.mean(evidence_all[involved_channels], axis=0)
    else:
        # LPD: use dominant hemisphere only
        left_involved = [ch for ch in involved_channels if ch in LEFT_INDICES]
        right_involved = [ch for ch in involved_channels if ch in RIGHT_INDICES]
        midline_involved = [ch for ch in involved_channels if ch in MIDLINE_INDICES]

        left_mean = np.mean([np.mean(evidence_all[ch]) for ch in left_involved]) if left_involved else 0
        right_mean = np.mean([np.mean(evidence_all[ch]) for ch in right_involved]) if right_involved else 0

        if left_mean >= right_mean:
            dominant = left_involved + midline_involved
        else:
            dominant = right_involved + midline_involved

        if not dominant:
            dominant = involved_channels

        return np.mean(evidence_all[dominant], axis=0)


def _find_candidate_peaks(evidence, fs, freq_estimate):
    """Find local maxima of evidence signal with constraints."""
    min_sep = max(int(0.3 / freq_estimate * fs), int(0.15 * fs))
    med = np.median(evidence)
    std = np.std(evidence)
    min_height = med + 0.3 * std

    peaks, props = find_peaks(evidence, distance=min_sep, height=min_height)
    return peaks


def _dp_best_sequence(candidate_peaks, evidence, fs, freq_estimate,
                      alpha=5.0, beta=1.0, lam=0.5):
    """Dynamic programming for best global discharge sequence.

    Viterbi-style DP over candidate peaks, penalizing:
    - Deviation from expected interval T = 1/freq_estimate
    - Skipped discharges
    - Complexity (number of events)
    """
    if len(candidate_peaks) == 0:
        return np.array([])
    if len(candidate_peaks) == 1:
        return candidate_peaks.copy()

    T = 1.0 / freq_estimate
    T_samples = T * fs
    n = len(candidate_peaks)

    node_scores = np.array([evidence[p] for p in candidate_peaks])

    # DP
    best_score = np.full(n, -np.inf)
    best_prev = np.full(n, -1, dtype=int)

    # Initialize: each peak can start a sequence
    for i in range(n):
        best_score[i] = node_scores[i] - lam

    for j in range(1, n):
        for i in range(j):
            dt = (candidate_peaks[j] - candidate_peaks[i]) / fs
            if dt > 3.5 * T:
                continue
            if dt <= 0:
                continue

            # Try m = 1, 2, 3 (allow up to 2 skipped discharges)
            best_edge = -np.inf
            for m in [1, 2, 3]:
                interval_score = -alpha * ((dt - m * T) / (m * T)) ** 2
                skip_penalty = -beta * (m - 1)
                edge = interval_score + skip_penalty
                if edge > best_edge:
                    best_edge = edge

            total = best_score[i] + best_edge + node_scores[j] - lam
            if total > best_score[j]:
                best_score[j] = total
                best_prev[j] = i

    # Traceback from best-scoring endpoint
    best_end = np.argmax(best_score)
    path = []
    idx = best_end
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()

    return candidate_peaks[path]


def _fft_peak_frequency(trace, fs, freq_lo=0.3, freq_hi=3.5):
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


def _compute_sp_features(segment, fs, is_gpd):
    """Compute signal processing features for Ridge frequency estimation."""
    n_channels = min(segment.shape[0], 18)
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    sigma_samples = max(1, int(0.02 * fs))

    seg_lp = np.zeros_like(segment[:n_channels])
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, segment[ch])
        except ValueError:
            seg_lp[ch] = segment[ch]

    # Pointiness traces
    pt_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pt_traces.append(pt)

    # f_B (ACF)
    acf_freqs = []
    for ch in range(n_channels):
        try:
            freq, score, _ = compute_acf_frequency(
                seg_lp[ch], fs, method='pointiness',
                smoothing_sigma=0.02, acf_min_lag=0.4,
                acf_peak_threshold=0.10, peak_height_frac=0.3)
            if np.isfinite(freq):
                acf_freqs.append(freq)
        except Exception:
            pass
    f_B = float(np.median(acf_freqs)) if acf_freqs else np.nan

    # f_peaks
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pt_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    f_peaks = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # f_fft
    fft_freqs = []
    for ch in range(n_channels):
        f = _fft_peak_frequency(pt_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    f_fft = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # f_tkeo
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = _fft_peak_frequency(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    f_tkeo = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh (spectral coherence)
    from scipy.signal import coherence as scipy_coherence
    ADJACENT_PAIRS = [(0,1),(1,2),(2,3),(4,5),(5,6),(6,7),
                      (8,9),(9,10),(10,11),(12,13),(13,14),(14,15),(16,17)]
    coh_freqs = []
    for (a, b) in ADJACENT_PAIRS:
        if a >= n_channels or b >= n_channels:
            continue
        try:
            f_c, Cxy = scipy_coherence(segment[a], segment[b], fs=fs,
                                        nperseg=min(256, segment.shape[1]))
            mask = (f_c >= 0.3) & (f_c <= 3.5)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_sub = f_c[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_sub[np.argmax(Cxy_sub)])
        except Exception:
            continue
    f_coh = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # Interaction features
    f_fft_v = f_fft if np.isfinite(f_fft) else 0
    f_tkeo_v = f_tkeo if np.isfinite(f_tkeo) else 0
    f_B_v = f_B if np.isfinite(f_B) else 0

    return {
        'f_B': f_B, 'f_peaks': f_peaks, 'f_fft': f_fft,
        'f_tkeo': f_tkeo, 'f_coh': f_coh, 'is_gpd': float(is_gpd),
        'f_fft_x_f_tkeo': f_fft_v * f_tkeo_v,
        'f_fft_x_f_B': f_fft_v * f_B_v,
        'f_tkeo_x_f_B': f_tkeo_v * f_B_v,
    }


# Cached Ridge model (loaded from saved weights)
_ridge_cache = {}

def _load_ridge_model():
    """Load pre-trained Ridge model weights from data/dl_cache/ridge_freq_model.npz."""
    if 'w' in _ridge_cache:
        return _ridge_cache['w'], _ridge_cache['feature_cols'], _ridge_cache['medians']

    model_path = Path(__file__).resolve().parent.parent / 'data' / 'dl_cache' / 'ridge_freq_model.npz'
    npz = np.load(str(model_path), allow_pickle=True)
    w = npz['weights']
    feature_cols = list(npz['feature_cols'])
    medians = dict(zip(npz['medians_keys'], npz['medians_vals']))

    _ridge_cache['w'] = w
    _ridge_cache['feature_cols'] = feature_cols
    _ridge_cache['medians'] = medians
    return w, feature_cols, medians


def estimate_frequency(segment, fs, subtype='lpd'):
    """Estimate discharge frequency using pre-trained Ridge model.

    Loads saved weights from data/dl_cache/ridge_freq_model.npz (trained with
    t2_expanded_interactions_a5: Ridge on 9 features, alpha=5.0).
    """
    is_gpd = 1 if subtype == 'gpd' else 0
    feats = _compute_sp_features(segment, fs, is_gpd)

    w, feature_cols, medians = _load_ridge_model()

    x = np.array([feats.get(c, np.nan) for c in feature_cols])
    for j, col in enumerate(feature_cols):
        if not np.isfinite(x[j]):
            x[j] = medians.get(col, 0.0)

    x_b = np.append(x, 1.0)
    pred_log = np.clip(x_b @ w, np.log(0.1), np.log(10.0))
    return float(np.exp(pred_log))


def detect_involved_channels(segment, fs, freq_estimate, subtype='gpd',
                             ve_threshold=0.20):
    """Detect which channels are involved using variance-explained method.

    For each channel, bandpass narrowly around freq_estimate and measure
    what fraction of variance the narrowband reconstruction explains.
    Channels with VE > threshold are involved.

    Subtype-aware: for LPDs, restrict to the dominant hemisphere (+ midline).
    For GPDs, keep all channels that pass threshold.

    Args:
        segment: (18, N) bipolar EEG at fs Hz
        fs: sampling rate
        freq_estimate: estimated discharge frequency in Hz
        subtype: 'lpd' or 'gpd'
        ve_threshold: minimum variance explained to be "involved" (default 0.20)

    Returns:
        involved: list of channel indices
        ve_per_channel: array of variance explained per channel (18,)
    """
    n_channels = min(segment.shape[0], 18)
    bw = 0.3  # half-bandwidth in Hz
    lo = max(0.1, freq_estimate - bw)
    hi = min(fs / 2 - 1, freq_estimate + bw)

    try:
        b, a = butter(3, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    except ValueError:
        lo = max(0.1, freq_estimate - 0.5)
        hi = min(fs / 2 - 1, freq_estimate + 0.5)
        b, a = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band')

    ve_per_channel = np.zeros(n_channels)
    for ch in range(n_channels):
        sig = segment[ch]
        var_total = np.var(sig)
        if var_total < 1e-10:
            continue
        try:
            nb = filtfilt(b, a, sig)
            ve_per_channel[ch] = np.var(nb) / var_total
        except Exception:
            continue

    # For LPDs: restrict to dominant hemisphere + midline
    if subtype == 'lpd':
        left_ve = sum(ve_per_channel[ch] for ch in LEFT_INDICES if ch < n_channels)
        right_ve = sum(ve_per_channel[ch] for ch in RIGHT_INDICES if ch < n_channels)

        if left_ve >= right_ve:
            allowed = set(ch for ch in LEFT_INDICES if ch < n_channels)
        else:
            allowed = set(ch for ch in RIGHT_INDICES if ch < n_channels)
        # Always include midline
        allowed.update(ch for ch in MIDLINE_INDICES if ch < n_channels)

        involved = [ch for ch in range(n_channels)
                     if ve_per_channel[ch] > ve_threshold and ch in allowed]
    else:
        # GPD: all channels that pass threshold
        involved = [ch for ch in range(n_channels)
                     if ve_per_channel[ch] > ve_threshold]

    # Fallback: if too few channels, take top channels (respecting subtype constraint)
    if len(involved) < 2:
        if subtype == 'lpd':
            candidates = sorted(allowed, key=lambda c: ve_per_channel[c], reverse=True)
        else:
            candidates = sorted(range(n_channels), key=lambda c: ve_per_channel[c], reverse=True)
        involved = candidates[:max(2, len(candidates) // 4)]

    return involved, ve_per_channel


def detect_discharge_times(segment, fs, freq_estimate, subtype,
                           involved_channels=None):
    """
    Detect individual discharge times in a PD segment.

    Args:
        segment: (18, N) bipolar EEG at fs Hz
        fs: sampling rate (200)
        freq_estimate: estimated discharge frequency in Hz
        subtype: 'lpd' or 'gpd'
        involved_channels: list of channel indices (if None, auto-detect)

    Returns:
        dict with:
            'global_times': array of detected discharge times (seconds)
            'channel_times': dict of {channel_idx: array of times}
            'evidence': (18, N) evidence signal per channel
            'aggregate_evidence': (N,) aggregated evidence signal
            'active_interval': (start_sec, end_sec) or None
            'involved_channels': list of involved channel indices
    """
    n_channels, n_samples = segment.shape

    # Clamp freq_estimate to reasonable range
    freq_estimate = np.clip(freq_estimate, 0.3, 3.5)
    if not np.isfinite(freq_estimate):
        freq_estimate = 1.0

    # Step 1: Build per-channel evidence
    evidence_all = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        evidence_all[ch] = _compute_channel_evidence(segment[ch], fs)

    # Auto-detect involved channels if needed
    if involved_channels is None:
        involved_channels = _auto_detect_involved(evidence_all, n_channels)

    # Step 2: Aggregate evidence
    agg_evidence = _aggregate_evidence(evidence_all, involved_channels, subtype, n_channels)

    # Step 3: Extract candidate peaks from aggregate evidence
    candidate_peaks = _find_candidate_peaks(agg_evidence, fs, freq_estimate)

    # Step 4: DP for best global sequence
    if len(candidate_peaks) >= 2:
        global_peak_samples = _dp_best_sequence(candidate_peaks, agg_evidence, fs, freq_estimate)
    elif len(candidate_peaks) == 1:
        global_peak_samples = candidate_peaks.copy()
    else:
        global_peak_samples = np.array([], dtype=int)

    global_times = global_peak_samples / fs

    # Step 5: Per-channel discharge times
    T = 1.0 / freq_estimate
    tolerance_samples = int(T / 4 * fs)
    channel_times = {}

    for ch in involved_channels:
        # Find per-channel candidate peaks
        ch_peaks = _find_candidate_peaks(evidence_all[ch], fs, freq_estimate)
        ch_times = []
        for gt_sample in global_peak_samples:
            if len(ch_peaks) == 0:
                ch_times.append(gt_sample / fs)
                continue
            dists = np.abs(ch_peaks.astype(float) - gt_sample)
            best_idx = np.argmin(dists)
            if dists[best_idx] <= tolerance_samples:
                ch_times.append(ch_peaks[best_idx] / fs)
            else:
                ch_times.append(gt_sample / fs)
        channel_times[ch] = np.array(ch_times)

    # Step 6: Active interval
    if len(global_times) >= 2:
        active_interval = (float(global_times[0]), float(global_times[-1]))
    else:
        active_interval = None

    return {
        'global_times': global_times,
        'channel_times': channel_times,
        'evidence': evidence_all,
        'aggregate_evidence': agg_evidence,
        'active_interval': active_interval,
        'involved_channels': involved_channels,
    }
