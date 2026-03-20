"""
PD morphological consistency scoring.

Computes a consistency score for each EEG segment by measuring how
similar successive periodic discharges are to each other.

Two approaches combined:
  A) Cross-correlation of successive discharge windows
  B) CV of discharge shape features (amplitude, duration, area)

Higher score = more consistent = more likely a real periodic discharge.

Usage:
    from pd_consistency_score import compute_pd_consistency
    result = compute_pd_consistency(segment_18ch, fs=200, expected_freq=1.5)
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend
from scipy.ndimage import gaussian_filter1d


def compute_pd_consistency(segment_18ch, fs=200, expected_freq=None):
    """Compute PD morphological consistency score for an EEG segment.

    Args:
        segment_18ch: (18, N) bipolar EEG array
        fs: sampling rate (default 200 Hz)
        expected_freq: expected discharge frequency in Hz (optional).
            If None, estimated via autocorrelation.

    Returns:
        dict with keys:
            consistency_score: combined score in [0, 1]
            median_xcorr: median cross-correlation of consecutive discharges
            shape_cv: mean CV of shape features
            n_discharges: number of detected discharges
            peak_channel: index of the channel used for analysis
    """
    seg = np.asarray(segment_18ch, dtype=np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg.shape

    # ── Lowpass filter at 20 Hz (same as existing pipeline) ──────────
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        seg_lp = np.zeros_like(seg)
        for ch in range(n_channels):
            try:
                seg_lp[ch] = filtfilt(b, a, seg[ch])
            except ValueError:
                seg_lp[ch] = seg[ch]
    else:
        seg_lp = seg.copy()

    # Detrend each channel
    for ch in range(n_channels):
        seg_lp[ch] = detrend(seg_lp[ch], type='linear')

    # ── Find channel with highest energy in pointiness trace ─────────
    # Pointiness = prominence^2 / width approximation via second derivative
    channel_energies = np.zeros(n_channels)
    for ch in range(n_channels):
        d2 = np.abs(np.diff(seg_lp[ch], n=2))
        d2_smooth = gaussian_filter1d(d2, sigma=max(1, int(0.02 * fs)))
        channel_energies[ch] = np.sum(d2_smooth ** 2)

    peak_channel = int(np.argmax(channel_energies))
    signal = seg_lp[peak_channel]

    # ── Estimate frequency if not provided ───────────────────────────
    if expected_freq is None or expected_freq <= 0:
        expected_freq = _estimate_freq_acf(signal, fs)

    # Fallback if ACF estimation fails
    if expected_freq is None or expected_freq <= 0:
        expected_freq = 1.5  # default assumption

    # ── Detect discharge peaks ───────────────────────────────────────
    # Use pointiness (absolute second derivative, smoothed) for peak detection
    d2 = np.abs(np.diff(signal, n=2))
    d2_smooth = gaussian_filter1d(d2, sigma=max(1, int(0.02 * fs)))

    min_distance = int(0.5 * fs / expected_freq)  # half the expected period
    min_distance = max(min_distance, int(0.15 * fs))  # at least 150ms apart

    height_thresh = np.max(d2_smooth) * 0.3
    peaks, _ = find_peaks(d2_smooth, height=height_thresh, distance=min_distance)

    n_discharges = len(peaks)

    # Need at least 3 discharges for meaningful statistics
    if n_discharges < 3:
        return {
            'consistency_score': 0.0,
            'median_xcorr': 0.0,
            'shape_cv': 1.0,
            'n_discharges': n_discharges,
            'peak_channel': peak_channel,
        }

    # ── Extract windows around each peak: +/- 150ms ─────────────────
    half_win = int(0.150 * fs)  # 150ms = 30 samples at 200 Hz
    windows = []
    for pk in peaks:
        start = pk - half_win
        end = pk + half_win + 1
        if start < 0 or end > len(signal):
            continue
        win = signal[start:end].copy()
        windows.append(win)

    n_discharges = len(windows)
    if n_discharges < 3:
        return {
            'consistency_score': 0.0,
            'median_xcorr': 0.0,
            'shape_cv': 1.0,
            'n_discharges': n_discharges,
            'peak_channel': peak_channel,
        }

    # ── A) Cross-correlation of consecutive discharge windows ────────
    # Normalize each window to zero-mean, unit-variance
    norm_windows = []
    for win in windows:
        w = win - np.mean(win)
        std = np.std(w)
        if std > 0:
            w = w / std
        norm_windows.append(w)

    # Pairwise cross-correlation between ALL consecutive windows
    xcorrs = []
    for i in range(len(norm_windows) - 1):
        w1 = norm_windows[i]
        w2 = norm_windows[i + 1]
        # Normalized cross-correlation (already zero-mean, unit-var)
        cc = np.dot(w1, w2) / len(w1)
        xcorrs.append(cc)

    median_xcorr = float(np.median(xcorrs)) if xcorrs else 0.0
    # Clip to [0, 1] — negative correlations mean inconsistent
    median_xcorr = float(np.clip(median_xcorr, 0.0, 1.0))

    # ── B) CV of discharge shape features ────────────────────────────
    peak_amplitudes = []
    durations = []
    areas = []

    for win in windows:
        abs_win = np.abs(win)

        # Peak amplitude: max absolute value
        peak_amp = np.max(abs_win)
        peak_amplitudes.append(peak_amp)

        # Duration: width at half-max of the absolute signal
        half_max = peak_amp / 2.0
        above_half = abs_win >= half_max
        if np.any(above_half):
            indices = np.where(above_half)[0]
            dur = (indices[-1] - indices[0]) / fs
        else:
            dur = 0.0
        durations.append(dur)

        # Area: sum of absolute values
        area = np.sum(abs_win)
        areas.append(area)

    def _cv(values):
        """Coefficient of variation (std/mean), returns 1.0 if degenerate."""
        arr = np.array(values)
        m = np.mean(arr)
        if m <= 0:
            return 1.0
        return float(np.std(arr) / m)

    cv_amp = _cv(peak_amplitudes)
    cv_dur = _cv(durations)
    cv_area = _cv(areas)
    shape_cv = float(np.mean([cv_amp, cv_dur, cv_area]))

    # Clip shape_cv to [0, 1]
    shape_cv = float(np.clip(shape_cv, 0.0, 1.0))

    # ── C) Combined score ────────────────────────────────────────────
    consistency_score = median_xcorr * (1.0 - shape_cv)
    consistency_score = float(np.clip(consistency_score, 0.0, 1.0))

    return {
        'consistency_score': round(consistency_score, 4),
        'median_xcorr': round(median_xcorr, 4),
        'shape_cv': round(shape_cv, 4),
        'n_discharges': n_discharges,
        'peak_channel': peak_channel,
    }


def _estimate_freq_acf(signal, fs, freq_lo=0.3, freq_hi=3.5):
    """Estimate dominant frequency via autocorrelation.

    Returns frequency in Hz, or None if no clear peak found.
    """
    sig = signal - np.mean(signal)
    n = len(sig)
    if n < fs:
        return None

    # Compute ACF via FFT for efficiency
    fft_len = 2 * n
    fft_sig = np.fft.rfft(sig, n=fft_len)
    acf_full = np.fft.irfft(np.abs(fft_sig) ** 2, n=fft_len)
    acf = acf_full[:n]

    # Normalize
    if acf[0] > 0:
        acf = acf / acf[0]
    else:
        return None

    # Search for peaks in the valid lag range
    min_lag = int(fs / freq_hi)  # shortest period
    max_lag = int(fs / freq_lo)  # longest period
    max_lag = min(max_lag, n - 1)

    if min_lag >= max_lag:
        return None

    acf_segment = acf[min_lag:max_lag + 1]
    if len(acf_segment) < 3:
        return None

    peaks, props = find_peaks(acf_segment, height=0.1)
    if len(peaks) == 0:
        return None

    # Pick the highest peak
    best = peaks[np.argmax(props['peak_heights'])]
    best_lag = best + min_lag

    if best_lag > 0:
        return fs / best_lag
    return None
