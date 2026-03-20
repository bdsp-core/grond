"""
Per-channel feature extraction for the PD channel detector.

Computes 12 features from each 2000-sample (200 Hz) bipolar channel signal.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, welch

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
from scipy.ndimage import gaussian_filter1d

FS = 200
FREQ_LO = 0.3
FREQ_HI = 3.5
LOWPASS_HZ = 15.0

FEATURE_NAMES = [
    'pointiness_mean',
    'pointiness_max',
    'pointiness_std',
    'acf_peak_height',
    'acf_frequency',
    'fft_peak_power',
    'fft_peak_freq',
    'amplitude_rms',
    'amplitude_range',
    'peak_count',
    'peak_regularity',
    'line_length',
    'spectral_entropy',
]


def extract_channel_features(signal_1d, fs=FS):
    """Extract features from a single channel (1D signal, 2000 samples at 200Hz).

    Returns: dict with FEATURE_NAMES keys
    """
    x = np.asarray(signal_1d, dtype=np.float64)
    n = len(x)
    features = {}

    # Lowpass filter
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    try:
        x_lp = filtfilt(b_lp, a_lp, x)
    except ValueError:
        x_lp = x.copy()

    # 1-3. Pointiness trace stats
    pt = compute_pointiness_trace(x_lp)
    pt_smooth = gaussian_filter1d(pt, sigma=max(1, int(0.02 * fs)))
    features['pointiness_mean'] = float(np.mean(pt_smooth))
    features['pointiness_max'] = float(np.max(pt_smooth))
    features['pointiness_std'] = float(np.std(pt_smooth))

    # 4-5. ACF features
    freq, acf_height, peak_indices = compute_acf_frequency(
        x_lp, fs, method='pointiness',
        smoothing_sigma=0.02, acf_min_lag=0.4,
        acf_peak_threshold=0.10, peak_height_frac=0.3,
    )
    features['acf_peak_height'] = float(acf_height)
    features['acf_frequency'] = float(freq) if np.isfinite(freq) else 0.0

    # 6-7. FFT in PD range
    fft_vals = np.abs(np.fft.rfft(x_lp - np.mean(x_lp)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
    if np.any(mask):
        fft_sub = fft_vals[mask]
        freqs_sub = freqs[mask]
        peak_idx = np.argmax(fft_sub)
        features['fft_peak_power'] = float(fft_sub[peak_idx])
        features['fft_peak_freq'] = float(freqs_sub[peak_idx])
    else:
        features['fft_peak_power'] = 0.0
        features['fft_peak_freq'] = 0.0

    # 8. Amplitude RMS (of lowpassed signal)
    features['amplitude_rms'] = float(np.sqrt(np.mean(x_lp ** 2)))

    # 9. Amplitude range
    features['amplitude_range'] = float(np.max(x_lp) - np.min(x_lp))

    # 10. Peak count (pointiness peaks)
    mx = np.max(pt_smooth)
    if mx > 0:
        pks, _ = find_peaks(pt_smooth, height=mx * 0.3, distance=int(0.2 * fs))
        features['peak_count'] = float(len(pks))
    else:
        features['peak_count'] = 0.0

    # 11. Peak regularity: 1 - CV of inter-peak intervals
    if mx > 0:
        pks, _ = find_peaks(pt_smooth, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            ipis = np.diff(pks) / fs  # inter-peak intervals in seconds
            mean_ipi = np.mean(ipis)
            if mean_ipi > 0:
                cv = np.std(ipis) / mean_ipi
                features['peak_regularity'] = float(max(0, 1.0 - cv))
            else:
                features['peak_regularity'] = 0.0
        else:
            features['peak_regularity'] = 0.0
    else:
        features['peak_regularity'] = 0.0

    # 12. Line length
    features['line_length'] = float(np.sum(np.abs(np.diff(x_lp))))

    # 13. Spectral entropy in PD range (0.3-3.5 Hz)
    try:
        f_psd, psd = welch(x_lp, fs=fs, nperseg=min(256, n))
        pd_mask = (f_psd >= FREQ_LO) & (f_psd <= FREQ_HI)
        if np.any(pd_mask):
            psd_pd = psd[pd_mask]
            psd_pd = psd_pd / np.sum(psd_pd) if np.sum(psd_pd) > 0 else psd_pd
            # Avoid log(0)
            psd_pd = np.clip(psd_pd, 1e-12, None)
            features['spectral_entropy'] = float(-np.sum(psd_pd * np.log(psd_pd)))
        else:
            features['spectral_entropy'] = 0.0
    except Exception:
        features['spectral_entropy'] = 0.0

    return features


def extract_features_batch(channels, fs=FS):
    """Extract features for an array of channels.

    Args:
        channels: (N, 2000) array
        fs: sampling rate

    Returns:
        feature_matrix: (N, n_features) array
    """
    n = channels.shape[0]
    n_feat = len(FEATURE_NAMES)
    feat_matrix = np.zeros((n, n_feat), dtype=np.float32)

    for i in range(n):
        feats = extract_channel_features(channels[i], fs)
        for j, name in enumerate(FEATURE_NAMES):
            feat_matrix[i, j] = feats.get(name, 0.0)

    return feat_matrix
