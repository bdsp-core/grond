"""Signal-based spatial localization methods (S1-S8)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, find_peaks, hilbert
from scipy.ndimage import gaussian_filter1d
from .base import SpatialMethod, FS, REGIONS, CHANNEL_TO_REGIONS, REGION_TO_CHANNELS, LEFT_CHS, RIGHT_CHS


class S1_PointinessMax(SpatialMethod):
    """Per-channel pointiness → region involvement."""
    name = "S1_PointinessMax"
    description = "Max pointiness trace per channel, map to regions"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            pt = compute_pointiness_trace(seg_bi[ch])
            scores[ch] = float(np.max(gaussian_filter1d(pt, sigma=4)))
        # Normalize to [0, 1]
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.3}


class S2_PointinessMean(SpatialMethod):
    """Per-channel mean pointiness → region involvement."""
    name = "S2_PointinessMean"
    description = "Mean pointiness per channel, threshold=0.25"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            pt = compute_pointiness_trace(seg_bi[ch])
            scores[ch] = float(np.mean(gaussian_filter1d(pt, sigma=4)))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.25}


class S3_FFTPeakPower(SpatialMethod):
    """FFT peak power in PD band per channel."""
    name = "S3_FFTPeakPower"
    description = "FFT peak power in 0.3-3.5 Hz per channel"

    def _analyze(self, seg_bi, subtype):
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            x = seg_bi[ch] - np.mean(seg_bi[ch])
            fft_vals = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), d=1.0/FS)
            mask = (freqs >= 0.3) & (freqs <= 3.5)
            if np.any(mask):
                scores[ch] = float(np.max(fft_vals[mask]))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class S4_BandpowerRatio(SpatialMethod):
    """Ratio of PD-band power to total power per channel."""
    name = "S4_BandpowerRatio"
    description = "PD band (0.5-3.5 Hz) power / total power ratio"

    def _analyze(self, seg_bi, subtype):
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            f, pxx = welch(seg_bi[ch], FS, nperseg=min(400, len(seg_bi[ch])))
            total = np.sum(pxx)
            if total > 0:
                mask = (f >= 0.5) & (f <= 3.5)
                scores[ch] = float(np.sum(pxx[mask]) / total)
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


class S5_ACFPeakHeight(SpatialMethod):
    """Autocorrelation peak height in PD range per channel."""
    name = "S5_ACFPeakHeight"
    description = "ACF peak height at 0.3-3.5 Hz lag per channel"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            pt = compute_pointiness_trace(seg_bi[ch])
            pt_smooth = gaussian_filter1d(pt, sigma=4)
            # ACF of pointiness
            x = pt_smooth - np.mean(pt_smooth)
            n = len(x)
            acf = np.correlate(x, x, mode='full')[n-1:]
            if acf[0] > 0:
                acf = acf / acf[0]
            # Peak in PD range (0.3 - 3.5 Hz => lag 57-667 samples)
            lo_lag = int(FS / 3.5)
            hi_lag = int(FS / 0.3)
            hi_lag = min(hi_lag, len(acf) - 1)
            if lo_lag < hi_lag:
                segment = acf[lo_lag:hi_lag+1]
                scores[ch] = float(np.max(segment)) if len(segment) > 0 else 0.0
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.3}


class S6_LineLength(SpatialMethod):
    """Line length (sum of absolute differences) after lowpass."""
    name = "S6_LineLength"
    description = "Line length after 15 Hz lowpass, proxy for sharpness"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=15.0)
        scores = np.zeros(18)
        for ch in range(min(18, seg_filt.shape[0])):
            scores[ch] = float(np.sum(np.abs(np.diff(seg_filt[ch]))))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


class S7_EnvelopePeakiness(SpatialMethod):
    """Peakiness of the bandpass envelope — sharp peaks = PDs."""
    name = "S7_EnvelopePeakiness"
    description = "Peak count in PD-band envelope per channel"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=3.5)
        scores = np.zeros(18)
        for ch in range(min(18, seg_filt.shape[0])):
            envelope = np.abs(hilbert(seg_filt[ch]))
            peaks, props = find_peaks(envelope, height=np.mean(envelope), distance=int(0.2 * FS))
            if len(peaks) > 0:
                scores[ch] = float(np.mean(props['peak_heights']))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class S8_SpectralEntropy(SpatialMethod):
    """Low spectral entropy in PD band = more periodic = more likely involved."""
    name = "S8_SpectralEntropy"
    description = "Inverse spectral entropy in PD band (lower = more periodic)"

    def _analyze(self, seg_bi, subtype):
        scores = np.zeros(18)
        for ch in range(min(18, seg_bi.shape[0])):
            f, pxx = welch(seg_bi[ch], FS, nperseg=min(400, len(seg_bi[ch])))
            mask = (f >= 0.3) & (f <= 3.5)
            if np.any(mask):
                pxx_sub = pxx[mask]
                total = np.sum(pxx_sub)
                if total > 0:
                    p = pxx_sub / total
                    p = p[p > 0]
                    entropy = -np.sum(p * np.log2(p))
                    max_entropy = np.log2(len(p))
                    # Inverse: low entropy = high score
                    scores[ch] = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}
