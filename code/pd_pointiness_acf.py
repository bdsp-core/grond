"""
Periodic discharge detector using pointiness trace + autocorrelation.

Alternative to pd_detect_alternate.py. Same output interface.

Methods:
  - 'pointiness': prominence²/width at each local max → smooth → ACF
  - 'd2': |d²x/dt²| → smooth → ACF
"""

import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# Channel definitions (same as pd_detect_alternate)
bipolar_channels = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
mono_channels = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG',
]

left_indices = [0, 1, 2, 3, 8, 9, 10, 11]
right_indices = [4, 5, 6, 7, 12, 13, 14, 15]

region_channel_map = {
    'LF': [0, 8, 9, 1],
    'RF': [4, 5, 12, 13],
    'LT': [2, 3],
    'RT': [6, 7],
    'LCP': [10],
    'RCP': [14],
    'LO': [11],
    'RO': [15],
}


def fcn_getBanana(X):
    """Apply bipolar longitudinal montage."""
    bipolar_ids = np.array([
        [mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])]
        for bc in bipolar_channels
    ])
    return X[bipolar_ids[:, 0]] - X[bipolar_ids[:, 1]]


def compute_pointiness_trace(signal_1d, half_win=8):
    """Compute pointiness = prominence²/width at each local max. Zero elsewhere."""
    n = len(signal_1d)
    trace = np.zeros(n)
    peaks, _ = find_peaks(signal_1d)
    for loc in peaks:
        if loc < half_win or loc >= n - half_win:
            continue
        peak_val = signal_1d[loc]
        left_valley = np.min(signal_1d[loc - half_win:loc])
        right_valley = np.min(signal_1d[loc + 1:loc + half_win + 1])
        prom = peak_val - max(left_valley, right_valley)
        if prom <= 0:
            continue
        half_prom_level = peak_val - 0.5 * prom
        width = 0
        for j in range(1, half_win + 1):
            if signal_1d[loc - j] > half_prom_level:
                width += 1
            else:
                break
        for j in range(1, half_win + 1):
            if loc + j < n and signal_1d[loc + j] > half_prom_level:
                width += 1
            else:
                break
        if width > 0:
            trace[loc] = prom ** 2 / width
    return trace


def compute_acf_frequency(signal_1d, fs, method='pointiness',
                           smoothing_sigma=0.02, acf_min_lag=0.25,
                           acf_peak_threshold=0.1,
                           peak_height_frac=0.3,
                           log_compress=False,
                           percentile_norm=False,
                           percentile_window_s=5.0,
                           percentile_val=95):
    """Compute periodicity frequency for one channel using ACF of a feature trace.

    Returns:
        (frequency, acf_peak_height, peak_indices)
        frequency: Hz (or NaN if no periodicity detected)
        acf_peak_height: height of first ACF peak (0-1), serves as score
        peak_indices: indices of peaks in the feature trace (for synchrony)
    """
    if len(signal_1d) < 50:
        return np.nan, 0.0, np.array([])

    # Compute feature trace
    if method == 'pointiness':
        trace = compute_pointiness_trace(signal_1d)
    elif method == 'd2':
        trace = np.abs(np.diff(signal_1d, n=2))
        # Pad to same length
        trace = np.concatenate([trace, [0, 0]])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Smooth
    sigma_samples = max(1, int(smoothing_sigma * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)

    # Log compression: reduces dominance of large peaks in ACF
    if log_compress:
        trace = np.log1p(trace)

    # Local percentile normalization: divide by running percentile to equalize peak heights
    if percentile_norm:
        from scipy.ndimage import maximum_filter1d
        win_samples = max(3, int(percentile_window_s * fs))
        # Approximate running percentile using downsampled computation
        n = len(trace)
        step = max(1, win_samples // 20)
        sample_pts = np.arange(0, n, step)
        half_win = win_samples // 2
        pvals = np.array([
            np.percentile(trace[max(0, p - half_win):min(n, p + half_win)], percentile_val)
            for p in sample_pts
        ])
        # Interpolate back to full resolution
        running_pct = np.interp(np.arange(n), sample_pts, pvals)
        running_pct = np.maximum(running_pct, 1e-10)
        trace = trace / running_pct

    # Find prominent peaks in the trace (for synchrony analysis)
    trace_max = np.max(trace)
    peak_height = trace_max * peak_height_frac if trace_max > 0 else 0
    peak_indices, _ = find_peaks(trace, height=peak_height, distance=int(0.2 * fs))

    # ACF
    t = trace - np.mean(trace)
    max_lag = min(4 * fs, len(t) - 1)
    if max_lag < 10:
        return np.nan, 0.0, peak_indices

    acf = np.correlate(t, t, mode='full')
    acf = acf[len(t) - 1:][:max_lag + 1]
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return np.nan, 0.0, peak_indices

    # Find first significant local max after min_lag
    min_lag_samples = int(acf_min_lag * fs)
    for k in range(min_lag_samples + 1, len(acf) - 1):
        if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > acf_peak_threshold:
            freq = fs / k
            return freq, float(acf[k]), peak_indices

    return np.nan, 0.0, peak_indices


def pd_detect_pointiness_acf(segment, fs,
                               method='pointiness',
                               acf_min_lag=0.25,
                               acf_peak_threshold=0.1,
                               smoothing_sigma=0.02,
                               lowpass_hz=20.0,
                               peak_height_frac=0.3,
                               sync_tolerance_ms=200,
                               sync_threshold=0.8,
                               sync_min_peaks=5,
                               log_compress=False,
                               percentile_norm=False,
                               percentile_window_s=5.0,
                               percentile_val=95):
    """
    Detect periodic discharges using pointiness/d2 + ACF approach.

    Same output interface as pd_detect_alternate.
    """
    # Filter
    segment = notch_filter(segment, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")

    # Bipolar montage
    seg = np.array(fcn_getBanana(segment))

    # Lowpass filter before feature extraction — removes high-freq noise that
    # inflates d2/pointiness on frontal channels while preserving PD waveforms
    b_lp, a_lp = butter(4, lowpass_hz / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass

    # Per-channel analysis
    channel_scores = np.full(len(seg), 0.0)
    channel_freqs = np.full(len(seg), np.nan)
    all_channel_peaks = {}
    detected_channels = []

    for i in range(seg.shape[0]):
        freq, score, peaks = compute_acf_frequency(
            seg[i, :], fs, method=method,
            smoothing_sigma=smoothing_sigma,
            acf_min_lag=acf_min_lag,
            acf_peak_threshold=acf_peak_threshold,
            peak_height_frac=peak_height_frac,
            log_compress=log_compress,
            percentile_norm=percentile_norm,
            percentile_window_s=percentile_window_s,
            percentile_val=percentile_val,
        )
        channel_scores[i] = score
        channel_freqs[i] = freq
        if len(peaks) > 0:
            all_channel_peaks[i] = peaks
        if np.isfinite(freq):
            detected_channels.append(bipolar_channels[i])

    # Build output dicts
    channel_pd_scores = {bipolar_channels[i]: channel_scores[i]
                         for i in range(len(bipolar_channels))}
    channel_frequencies = {bipolar_channels[i]: channel_freqs[i]
                           for i in range(len(bipolar_channels))}

    # Region scores
    region_scores = {}
    for region, idxs in region_channel_map.items():
        region_scores[region] = float(np.mean(channel_scores[idxs]))

    # Laterality
    left_mean = np.mean([region_scores[r] for r in ['LF', 'LT', 'LCP', 'LO']])
    right_mean = np.mean([region_scores[r] for r in ['RF', 'RT', 'RCP', 'RO']])
    denom = right_mean + left_mean
    laterality_index = (right_mean - left_mean) / denom if denom > 0 else 0.0

    # Synchrony-based L/G classification
    # Only use peaks from channels where ACF detected periodicity (finite frequency)
    left_all_peaks = np.sort(np.concatenate(
        [all_channel_peaks[i] for i in left_indices
         if i in all_channel_peaks and np.isfinite(channel_freqs[i])]
    )) if any(i in all_channel_peaks and np.isfinite(channel_freqs[i])
              for i in left_indices) else np.array([])
    right_all_peaks = np.sort(np.concatenate(
        [all_channel_peaks[i] for i in right_indices
         if i in all_channel_peaks and np.isfinite(channel_freqs[i])]
    )) if any(i in all_channel_peaks and np.isfinite(channel_freqs[i])
              for i in right_indices) else np.array([])

    synchrony_ratio = np.nan
    if len(left_all_peaks) >= sync_min_peaks and len(right_all_peaks) >= sync_min_peaks:
        tol_samples = int(sync_tolerance_ms * fs / 1000)
        matched_left = 0
        for lp in left_all_peaks:
            idx = np.searchsorted(right_all_peaks, lp)
            for ci in [idx - 1, idx]:
                if 0 <= ci < len(right_all_peaks):
                    if abs(right_all_peaks[ci] - lp) <= tol_samples:
                        matched_left += 1
                        break
        matched_right = 0
        for rp in right_all_peaks:
            idx = np.searchsorted(left_all_peaks, rp)
            for ci in [idx - 1, idx]:
                if 0 <= ci < len(left_all_peaks):
                    if abs(left_all_peaks[ci] - rp) <= tol_samples:
                        matched_right += 1
                        break
        synchrony_ratio = (matched_left / len(left_all_peaks)
                           + matched_right / len(right_all_peaks)) / 2.0

    # Spatial areas
    spatial_areas = []
    for ch in detected_channels:
        for region, ch_list in {
            'LF': ['Fp1-F7', 'Fp1-F3', 'F3-C3', 'F7-T3'],
            'RF': ['Fp2-F8', 'F8-T4', 'Fp2-F4', 'F4-C4'],
            'LT': ['T3-T5', 'T5-O1'], 'RT': ['T4-T6', 'T6-O2'],
            'LCP': ['C3-P3'], 'RCP': ['C4-P4'],
            'LO': ['P3-O1'], 'RO': ['P4-O2'],
        }.items():
            if ch in ch_list:
                spatial_areas.append(region)
    spatial_areas = list(set(spatial_areas))

    n_detected = len(detected_channels)
    spatial_extent = n_detected / 18.0

    if n_detected == 0:
        type_event = np.nan
        event_frequency = np.nan
    else:
        valid_freqs = channel_freqs[np.isfinite(channel_freqs)]
        event_frequency = float(np.median(valid_freqs))

        # L/G classification
        if np.isfinite(synchrony_ratio):
            type_event = "GPD" if synchrony_ratio > sync_threshold else "LPD"
        else:
            type_event = "GPD" if spatial_extent > 0.8 else "LPD"

    return {
        "type_event": type_event,
        "event_frequency": event_frequency,
        "spatial_extent": spatial_extent,
        "spatial_areas": spatial_areas,
        "channels": detected_channels,
        "channel_pd_scores": channel_pd_scores,
        "channel_frequencies": channel_frequencies,
        "region_scores": region_scores,
        "laterality_index": laterality_index,
        "left_mean_score": left_mean,
        "right_mean_score": right_mean,
        "all_channel_peaks": all_channel_peaks,
        "synchrony_ratio": synchrony_ratio,
    }
