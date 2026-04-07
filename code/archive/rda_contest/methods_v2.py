"""Round 2 contest methods — informed by Round 1 findings.

Key improvements:
- Best-hemisphere channel selection for all features
- Amplitude consistency (best single feature from exploration)
- Multi-feature combinations
- Laterality as a feature
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, hilbert, find_peaks
from scipy.stats import pearsonr
from numpy.fft import fft, ifft

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS, FREQ_GRID


def _best_hemi_signal(seg_f):
    """Get the mean of top-3 channels on the better hemisphere."""
    ch_power = np.array([np.var(seg_f[ch]) for ch in range(18)])
    left_top3 = np.mean(np.sort(ch_power[LEFT_CHS])[::-1][:3])
    right_top3 = np.mean(np.sort(ch_power[RIGHT_CHS])[::-1][:3])
    if left_top3 > right_top3:
        best_chs = LEFT_CHS[np.argsort(ch_power[LEFT_CHS])[::-1][:3]]
    else:
        best_chs = RIGHT_CHS[np.argsort(ch_power[RIGHT_CHS])[::-1][:3]]
    return np.mean(np.abs(seg_f[best_chs]), axis=0), best_chs, ch_power


def _acf_peak_delta(sig):
    """ACF peak height in delta-range lags."""
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


def _amp_consistency(sig):
    """CV of peak amplitudes — low CV = consistent amplitude = RDA."""
    peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
    if len(peaks) < 3:
        return 0.0
    amps = sig[peaks]
    cv = np.std(amps) / max(np.mean(amps), 1e-12)
    return float(max(0, 1 - cv))


def _hilbert_regularity(sig):
    """Instantaneous frequency regularity via Hilbert. Low CV = regular."""
    analytic = hilbert(sig)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi / FS)
    inst_freq = inst_freq[(inst_freq > 0.3) & (inst_freq < 4.0)]
    if len(inst_freq) < 10:
        return 0.0, 1.0
    cv = np.std(inst_freq) / max(np.median(inst_freq), 0.01)
    reg = max(0, 1 - 2 * cv)
    freq = float(np.median(inst_freq))
    return float(reg), freq


def _laterality_index(ch_power):
    """Absolute laterality index from channel powers."""
    left_p = np.mean(ch_power[LEFT_CHS])
    right_p = np.mean(ch_power[RIGHT_CHS])
    return abs(right_p - left_p) / max(left_p + right_p, 1e-12)


def _spectral_peak_freq(sig):
    """Dominant frequency from Welch PSD on 0.5-3.5 Hz."""
    f, pxx = welch(sig, fs=FS, nperseg=400)
    delta_mask = (f >= 0.5) & (f <= 3.5)
    if not delta_mask.any() or pxx[delta_mask].sum() == 0:
        return 1.0, 0.0
    peak_idx = np.argmax(pxx[delta_mask])
    peak_p = pxx[delta_mask][peak_idx]
    mean_p = np.mean(pxx[delta_mask])
    freq = float(f[delta_mask][peak_idx])
    prominence = float(peak_p / max(mean_p, 1e-12))
    return freq, prominence


class V2_AmpConsistency(RDAMethod):
    """Amplitude consistency — best single feature from Round 1 exploration."""
    name = "V2_AmpConsistency"
    description = "Peak amplitude CV on best hemisphere (low CV = consistent = RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)
        q = _amp_consistency(best_sig)
        freq, _ = _spectral_peak_freq(best_sig)
        return {'freq': freq, 'q_score': q}


class V2_AmpConsistencyLat(RDAMethod):
    """Amplitude consistency × laterality index."""
    name = "V2_AmpConsistencyLat"
    description = "Peak amplitude CV × laterality (more lateralized + consistent = clearer RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, ch_power = _best_hemi_signal(seg_f)
        amp_q = _amp_consistency(best_sig)
        lat = _laterality_index(ch_power)
        # Combine: amp consistency is primary, laterality is a boost
        q = amp_q * (1 + 0.5 * lat)
        q = min(q, 1.0)
        freq, _ = _spectral_peak_freq(best_sig)
        return {'freq': freq, 'q_score': q}


class V2_HilbertBestHemi(RDAMethod):
    """Hilbert CV on best hemisphere only."""
    name = "V2_HilbertBestHemi"
    description = "Hilbert instantaneous frequency CV on best hemisphere channels"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)
        reg, freq = _hilbert_regularity(best_sig)
        return {'freq': freq, 'q_score': reg}


class V2_ACFBestHemi(RDAMethod):
    """ACF peak on best hemisphere."""
    name = "V2_ACFBestHemi"
    description = "Autocorrelation peak on best hemisphere (high = rhythmic)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)
        acf_val, freq = _acf_peak_delta(best_sig)
        return {'freq': freq, 'q_score': max(0, acf_val)}


class V2_MultiFeature(RDAMethod):
    """Multi-feature combination of best Round 1 features."""
    name = "V2_MultiFeature"
    description = "Weighted combination: amp_consistency + ACF + Hilbert + laterality"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, ch_power = _best_hemi_signal(seg_f)

        amp_q = _amp_consistency(best_sig)
        acf_val, acf_freq = _acf_peak_delta(best_sig)
        hilbert_reg, hilbert_freq = _hilbert_regularity(best_sig)
        lat = _laterality_index(ch_power)
        freq, prominence = _spectral_peak_freq(best_sig)

        # Weighted combination (weights from feature importance in exploration)
        q = (0.4 * amp_q +
             0.2 * max(0, acf_val) +
             0.2 * hilbert_reg +
             0.1 * lat +
             0.1 * min(prominence / 10, 1.0))
        q = min(max(q, 0), 1.0)

        return {'freq': freq, 'q_score': q}


class V2_ACFxAmp(RDAMethod):
    """ACF × amplitude consistency (best combo from Round 1)."""
    name = "V2_ACFxAmp"
    description = "Product of ACF peak and amplitude consistency on best hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)
        amp_q = _amp_consistency(best_sig)
        acf_val, freq = _acf_peak_delta(best_sig)
        q = amp_q * max(0, acf_val)
        return {'freq': freq, 'q_score': min(q, 1.0)}


class V2_TemplateBestHemi(RDAMethod):
    """Sinusoidal template correlation on best hemisphere."""
    name = "V2_TemplateBestHemi"
    description = "Template match (sin+cos) on best hemisphere top-3 channels"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        t = np.arange(len(best_sig)) / FS
        best_corr = 0.0
        best_freq = 1.0

        for f in FREQ_GRID:
            sin_t = np.sin(2 * np.pi * f * t)
            cos_t = np.cos(2 * np.pi * f * t)
            # Fit a*sin + b*cos + c
            basis = np.column_stack([sin_t, cos_t, np.ones(len(t))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(basis, best_sig, rcond=None)
                fitted = basis @ coeffs
                ss_res = np.var(best_sig - fitted)
                ss_tot = np.var(best_sig)
                r2 = max(0, 1 - ss_res / max(ss_tot, 1e-12))
                if r2 > best_corr:
                    best_corr = r2
                    best_freq = f
            except np.linalg.LinAlgError:
                pass

        return {'freq': best_freq, 'q_score': min(best_corr, 1.0)}


class V2_SpectralEntropyBestHemi(RDAMethod):
    """Spectral entropy on best hemisphere."""
    name = "V2_SpectralEntropyBestHemi"
    description = "Spectral entropy on best hemisphere (low = narrow-band = rhythmic)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        f, pxx = welch(best_sig, fs=FS, nperseg=400)
        delta_mask = (f >= 0.3) & (f <= 5.0)
        pxx_d = pxx[delta_mask]
        if pxx_d.sum() == 0:
            return {'freq': np.nan, 'q_score': 0.0}

        # Normalize to probability
        p = pxx_d / pxx_d.sum()
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(p))
        norm_entropy = entropy / max(max_entropy, 1e-12)

        q = max(0, 1 - norm_entropy)
        freq = float(f[delta_mask][np.argmax(pxx_d)])
        return {'freq': freq, 'q_score': q}
