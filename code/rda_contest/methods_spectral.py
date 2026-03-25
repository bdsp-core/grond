"""Spectral and template-based RDA analysis methods."""
import numpy as np
from scipy.signal import welch, find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS, FREQ_GRID


def _top3_laterality(channel_scores):
    """Pick top-3 channels respecting laterality (best hemisphere)."""
    left_mean = np.nanmean(channel_scores[LEFT_CHS])
    right_mean = np.nanmean(channel_scores[RIGHT_CHS])
    if left_mean >= right_mean:
        hem_chs = LEFT_CHS
    else:
        hem_chs = RIGHT_CHS
    hem_scores = channel_scores[hem_chs]
    top3_idx = np.argsort(hem_scores)[-3:]
    return hem_chs[top3_idx]


class S1_WaveformCorrelation(RDAMethod):
    """Segment into cycles via zero crossings, measure morphological consistency."""
    name = "S1_WaveformCorrelation"
    description = "Cross-correlation of consecutive cycle waveforms"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        filtered = self.prefilter(seg_bi)

        # Pick channel with highest variance in delta band
        variances = np.var(filtered, axis=1)
        best_ch = int(np.argmax(variances))
        sig = filtered[best_ch]

        # Find positive-going zero crossings
        crossings = []
        for i in range(len(sig) - 1):
            if sig[i] <= 0 and sig[i + 1] > 0:
                crossings.append(i)

        if len(crossings) < 5:
            # Need at least 4 full cycles (5 crossings)
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        # Extract cycles
        cycles = []
        durations = []
        for i in range(len(crossings) - 1):
            start, end = crossings[i], crossings[i + 1]
            if end - start < 10:  # skip tiny segments
                continue
            cycles.append(sig[start:end])
            durations.append((end - start) / FS)

        if len(cycles) < 4:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        # Resample all cycles to common length and compute pairwise xcorr
        target_len = 100
        resampled = []
        for c in cycles:
            x_old = np.linspace(0, 1, len(c))
            x_new = np.linspace(0, 1, target_len)
            resampled.append(np.interp(x_new, x_old, c))

        correlations = []
        for i in range(len(resampled) - 1):
            a = resampled[i]
            b = resampled[i + 1]
            a_norm = a - a.mean()
            b_norm = b - b.mean()
            denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
            if denom < 1e-12:
                continue
            r = np.sum(a_norm * b_norm) / denom
            correlations.append(r)

        if len(correlations) == 0:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        q_score = float(np.mean(correlations))
        freq = 1.0 / np.median(durations)

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {'n_cycles': len(cycles), 'best_ch': best_ch},
        }


class S2_EnvelopeContinuity(RDAMethod):
    """Amplitude envelope continuity — high min/mean ratio means sustained rhythm."""
    name = "S2_EnvelopeContinuity"
    description = "Hilbert envelope min/mean ratio across candidate frequencies"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        filtered = self.prefilter(seg_bi)

        # Pick channel with highest variance
        variances = np.var(filtered, axis=1)
        best_ch = int(np.argmax(variances))
        sig = filtered[best_ch]

        candidate_freqs = np.arange(0.5, 3.55, 0.25)
        best_score = -1.0
        best_freq = np.nan

        # Gaussian smoothing sigma: 100ms at 200Hz = 20 samples
        sigma = 0.1 * FS

        for freq in candidate_freqs:
            nb = self.narrowband(sig[np.newaxis, :], freq, bw=0.3)[0]
            analytic = hilbert(nb)
            envelope = np.abs(analytic)
            envelope = gaussian_filter1d(envelope, sigma=sigma)

            mean_env = np.mean(envelope)
            if mean_env < 1e-12:
                continue
            score = np.min(envelope) / mean_env

            if score > best_score:
                best_score = score
                best_freq = freq

        if best_score < 0:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

        q_score = float(np.clip(best_score, 0.0, 1.0))
        return {
            'freq': float(best_freq),
            'q_score': q_score,
            'extras': {'best_ch': best_ch},
        }


class S3_SpectralEntropy(RDAMethod):
    """Low spectral entropy in delta band indicates narrow-band rhythmic activity."""
    name = "S3_SpectralEntropy"
    description = "Normalized spectral entropy (low = rhythmic)"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        filtered = self.prefilter(seg_bi)
        n_chs = filtered.shape[0]

        entropies = []
        peak_freqs = []

        for ch in range(n_chs):
            freqs, psd = welch(filtered[ch], fs=FS, nperseg=400)
            # Restrict to 0.3-5Hz
            mask = (freqs >= 0.3) & (freqs <= 5.0)
            freqs_band = freqs[mask]
            psd_band = psd[mask]

            if len(psd_band) < 2 or np.sum(psd_band) < 1e-20:
                entropies.append(1.0)  # max entropy (no info)
                peak_freqs.append(np.nan)
                continue

            # Normalize to probability distribution
            p = psd_band / np.sum(psd_band)
            p = p[p > 0]  # avoid log(0)
            H = -np.sum(p * np.log(p))
            H_max = np.log(len(psd_band))
            H_norm = H / H_max if H_max > 0 else 1.0
            entropies.append(H_norm)
            peak_freqs.append(freqs_band[np.argmax(psd_band)])

        entropies = np.array(entropies)
        peak_freqs = np.array(peak_freqs)

        q_score = float(1.0 - np.nanmedian(entropies))
        freq = float(np.nanmedian(peak_freqs))

        return {
            'freq': freq,
            'q_score': q_score,
            'extras': {'median_entropy': float(np.nanmedian(entropies))},
        }


class T1_TemplateMatch(RDAMethod):
    """Sinusoidal template matching across frequency grid."""
    name = "T1_TemplateMatch"
    description = "Correlation with sin/cos templates at each candidate frequency"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        filtered = self.prefilter(seg_bi)
        n_chs, n_samp = filtered.shape
        t = np.arange(n_samp) / FS

        best_score = -1.0
        best_freq = np.nan

        # Pre-compute per-channel variance for laterality selection
        channel_max_corr = np.zeros(n_chs)

        for freq in FREQ_GRID:
            sin_t = np.sin(2 * np.pi * freq * t)
            cos_t = np.cos(2 * np.pi * freq * t)

            for ch in range(n_chs):
                sig = filtered[ch]
                sig_norm = sig - sig.mean()
                sig_std = np.sqrt(np.sum(sig_norm**2))
                if sig_std < 1e-12:
                    continue

                # Correlation with sin and cos, take max magnitude
                r_sin = np.sum(sig_norm * sin_t) / (sig_std * np.sqrt(np.sum(sin_t**2)))
                r_cos = np.sum(sig_norm * cos_t) / (sig_std * np.sqrt(np.sum(cos_t**2)))
                r = np.sqrt(r_sin**2 + r_cos**2)
                channel_max_corr[ch] = r

            # Laterality-aware top-3
            top3 = _top3_laterality(channel_max_corr)
            score = float(np.max(channel_max_corr[top3]))

            if score > best_score:
                best_score = score
                best_freq = freq

        q_score = float(np.clip(best_score, 0.0, 1.0))
        return {
            'freq': float(best_freq),
            'q_score': q_score,
            'extras': {},
        }


class T2_PeakRegularity(RDAMethod):
    """Peak detection + inter-peak interval regularity."""
    name = "T2_PeakRegularity"
    description = "CV of inter-peak intervals (low CV = regular = RDA)"

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        filtered = self.prefilter(seg_bi)
        n_chs = filtered.shape[0]

        min_dist = int(FS / 4)  # minimum distance between peaks

        channel_cv = np.full(n_chs, np.inf)
        channel_median_ipi = np.full(n_chs, np.nan)

        for ch in range(n_chs):
            peaks, _ = find_peaks(filtered[ch], distance=min_dist)
            if len(peaks) < 3:
                continue
            ipis = np.diff(peaks) / FS  # in seconds
            mean_ipi = np.mean(ipis)
            if mean_ipi < 1e-12:
                continue
            cv = np.std(ipis) / mean_ipi
            channel_cv[ch] = cv
            channel_median_ipi[ch] = np.median(ipis)

        # q_score per channel: 1 - 2*CV
        channel_q = np.maximum(0.0, 1.0 - 2.0 * channel_cv)

        # Laterality-aware top-3
        top3 = _top3_laterality(channel_q)
        best_idx = top3[np.argmax(channel_q[top3])]
        q_score = float(channel_q[best_idx])

        median_ipi = channel_median_ipi[best_idx]
        freq = 1.0 / median_ipi if np.isfinite(median_ipi) and median_ipi > 0 else np.nan

        return {
            'freq': float(freq),
            'q_score': q_score,
            'extras': {'best_ch': int(best_idx)},
        }
