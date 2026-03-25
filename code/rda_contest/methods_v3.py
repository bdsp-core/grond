"""Round 3 contest methods — targeting expert-level perceptual features.

These methods capture aspects of RDA that experts likely use but were not
measured in Rounds 1-2:
  - Spatial coherence across channels
  - Signal-to-background ratio
  - Delta dominance (z-scored)
  - Contiguous channel count
  - Temporal stability of frequency
"""
import numpy as np
from scipy.signal import welch, hilbert
from numpy.fft import fft, ifft

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS
from .methods_v2 import _best_hemi_signal, _spectral_peak_freq, _hilbert_regularity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hemi_top_channels(seg_f, hemi_chs, n=4):
    """Return indices of top-n channels by power within a hemisphere."""
    ch_power = np.array([np.var(seg_f[ch]) for ch in hemi_chs])
    top_idx = np.argsort(ch_power)[::-1][:n]
    return hemi_chs[top_idx]


def _better_hemisphere(seg_f):
    """Return (better_hemi_chs, worse_hemi_chs) based on mean delta power."""
    left_p = np.mean([np.var(seg_f[ch]) for ch in LEFT_CHS])
    right_p = np.mean([np.var(seg_f[ch]) for ch in RIGHT_CHS])
    if left_p >= right_p:
        return LEFT_CHS, RIGHT_CHS
    return RIGHT_CHS, LEFT_CHS


def _fft_peak_freq(sig):
    """Dominant frequency from FFT magnitude in 0.5-3.5 Hz."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    mag = np.abs(np.fft.rfft(sig))
    mask = (freqs >= 0.5) & (freqs <= 3.5)
    if not mask.any() or mag[mask].sum() == 0:
        return 1.0
    return float(freqs[mask][np.argmax(mag[mask])])


def _acf_peak_for_channel(sig):
    """ACF peak height in delta-range lags for a single channel."""
    x = sig - np.mean(sig)
    n = len(x)
    acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
    acf = acf / max(acf[0], 1e-12)
    min_lag = int(FS / 3.5)
    max_lag = min(int(FS / 0.5), n - 1)
    seg = acf[min_lag:max_lag]
    if len(seg) == 0:
        return 0.0, 1.0
    peak_idx = np.argmax(seg)
    peak_val = seg[peak_idx]
    freq = FS / (min_lag + peak_idx) if (min_lag + peak_idx) > 0 else 1.0
    return float(peak_val), float(freq)


# ---------------------------------------------------------------------------
# Method 1: Spatial Coherence
# ---------------------------------------------------------------------------

class V3_SpatialCoherence(RDAMethod):
    """Cross-channel phase coherence at the dominant delta frequency.

    High coherence across channels in a hemisphere indicates a spatially
    organized rhythm — a hallmark of clear RDA.
    """
    name = "V3_SpatialCoherence"
    description = "Mean pairwise coherence among top-4 channels on best hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_hemi, _ = _better_hemisphere(seg_f)

        # Find dominant frequency from the best hemisphere mean signal
        top4 = _hemi_top_channels(seg_f, best_hemi, n=4)
        hemi_mean = np.mean(seg_f[top4], axis=0)
        dom_freq = _fft_peak_freq(hemi_mean)

        # Narrowband filter each top-4 channel at the dominant frequency
        nb = self.narrowband(seg_f[top4], dom_freq, bw=0.3)  # (4, 2000)

        # Pairwise Pearson correlations among the 4 channels
        n_ch = nb.shape[0]
        corrs = []
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                denom_i = np.std(nb[i])
                denom_j = np.std(nb[j])
                if denom_i < 1e-12 or denom_j < 1e-12:
                    corrs.append(0.0)
                else:
                    r = np.corrcoef(nb[i], nb[j])[0, 1]
                    corrs.append(max(0.0, float(r)))

        q = float(np.mean(corrs)) if corrs else 0.0
        return {'freq': dom_freq, 'q_score': q}


# ---------------------------------------------------------------------------
# Method 2: Signal-to-Background Ratio
# ---------------------------------------------------------------------------

class V3_SignalToBackground(RDAMethod):
    """How much does the RDA rhythm stand out from the background EEG?

    Computes SNR = var(narrowband signal) / var(residual background).
    High SNR means the rhythmic component dominates.
    """
    name = "V3_SignalToBackground"
    description = "Narrowband signal variance vs. residual background variance"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Find dominant freq via Welch PSD
        dom_freq, _ = _spectral_peak_freq(best_sig)

        # Narrowband at the dominant frequency
        # narrowband expects (n_ch, samples) or at least 2D
        signal_nb = self.narrowband(best_sig.reshape(1, -1), dom_freq, bw=0.3)[0]

        # Residual = prefiltered - narrowband
        residual = best_sig - signal_nb

        var_signal = np.var(signal_nb)
        var_background = np.var(residual)

        if var_background < 1e-12:
            snr = 5.0 if var_signal > 1e-12 else 0.0
        else:
            snr = var_signal / var_background

        # Normalize: SNR of 5 maps to q=1.0
        q = min(snr / 5.0, 1.0)
        return {'freq': dom_freq, 'q_score': q}


# ---------------------------------------------------------------------------
# Method 3: Delta Z-Score (delta fraction on best hemisphere)
# ---------------------------------------------------------------------------

class V3_DeltaZScore(RDAMethod):
    """Delta power fraction on the best hemisphere.

    Higher delta fraction = more delta-dominated spectrum = more likely RDA.
    Uses the raw bipolar signal (not prefiltered) for broadband power.
    """
    name = "V3_DeltaZScore"
    description = "Fraction of total power in delta band on best hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_hemi, _ = _better_hemisphere(seg_f)
        top3 = _hemi_top_channels(seg_f, best_hemi, n=3)

        delta_fractions = []
        for ch in top3:
            # Use raw bipolar for broadband power measurement
            f, pxx = welch(seg_bi[ch], fs=FS, nperseg=400)

            total_mask = (f >= 0.5) & (f <= 40.0)
            delta_mask = (f >= 0.5) & (f <= 3.5)

            total_power = np.sum(pxx[total_mask])
            delta_power = np.sum(pxx[delta_mask])

            if total_power < 1e-12:
                delta_fractions.append(0.0)
            else:
                delta_fractions.append(delta_power / total_power)

        q = float(np.mean(delta_fractions))

        # Get frequency from the prefiltered best-hemisphere signal
        best_sig = np.mean(seg_f[top3], axis=0)
        freq, _ = _spectral_peak_freq(best_sig)

        return {'freq': freq, 'q_score': q}


# ---------------------------------------------------------------------------
# Method 4: Contiguous Channels
# ---------------------------------------------------------------------------

class V3_ContiguousChannels(RDAMethod):
    """Fraction of channels showing meaningful delta rhythm.

    More channels with a clear ACF peak in the delta range indicates a
    more widespread (and therefore more obvious) RDA pattern.
    """
    name = "V3_ContiguousChannels"
    description = "Fraction of all 18 channels with ACF peak > 0.3 in delta range"

    ACF_THRESHOLD = 0.3

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        n_channels = seg_f.shape[0]

        rhythmic_count = 0
        rhythmic_freqs = []

        for ch in range(n_channels):
            acf_val, freq = _acf_peak_for_channel(seg_f[ch])
            if acf_val > self.ACF_THRESHOLD:
                rhythmic_count += 1
                rhythmic_freqs.append(freq)

        q = rhythmic_count / n_channels

        if rhythmic_freqs:
            freq = float(np.median(rhythmic_freqs))
        else:
            # Fallback: use best hemisphere signal
            best_sig, _, _ = _best_hemi_signal(seg_f)
            freq, _ = _spectral_peak_freq(best_sig)

        return {'freq': freq, 'q_score': q}


# ---------------------------------------------------------------------------
# Method 5: Temporal Stability
# ---------------------------------------------------------------------------

class V3_TemporalStability(RDAMethod):
    """Is the dominant frequency stable over time?

    Splits the segment into 2-second windows and checks whether the
    estimated frequency is consistent. Low variance = stable = clear RDA.
    """
    name = "V3_TemporalStability"
    description = "Frequency stability across 2-second windows (low std = stable)"

    WINDOW_SEC = 2.0

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        n_samples = len(best_sig)
        win_len = int(self.WINDOW_SEC * FS)
        n_windows = n_samples // win_len

        if n_windows < 2:
            # Segment too short for windowed analysis; fall back
            freq, _ = _spectral_peak_freq(best_sig)
            return {'freq': freq, 'q_score': 0.0}

        window_freqs = []
        for i in range(n_windows):
            start = i * win_len
            end = start + win_len
            win = best_sig[start:end]
            f, pxx = welch(win, fs=FS, nperseg=min(win_len, 400))
            delta_mask = (f >= 0.5) & (f <= 3.5)
            if delta_mask.any() and pxx[delta_mask].sum() > 0:
                win_freq = float(f[delta_mask][np.argmax(pxx[delta_mask])])
            else:
                win_freq = np.nan
            window_freqs.append(win_freq)

        window_freqs = np.array(window_freqs)
        valid = window_freqs[np.isfinite(window_freqs)]

        if len(valid) < 2:
            freq, _ = _spectral_peak_freq(best_sig)
            return {'freq': freq, 'q_score': 0.0}

        std_freq = float(np.std(valid))
        freq = float(np.median(valid))

        # q_score: std of 0.5 Hz or more → q=0; std of 0 → q=1
        q = max(0.0, 1.0 - std_freq / 0.5)

        return {'freq': freq, 'q_score': q}
