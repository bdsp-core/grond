"""Rhythmicity-based lateralization methods (L06-L10).

All methods process each hemisphere independently.
"""
import numpy as np
from scipy.signal import welch, hilbert, find_peaks
from numpy.fft import fft, ifft

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _hemi_top_signal(seg, chs, top_k=4):
    """Mean of top-k channels by power on a hemisphere."""
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0)


def _acf_peak(sig, min_hz=0.5, max_hz=3.5):
    """ACF peak height in delta-range lags."""
    x = sig - np.mean(sig)
    n = len(x)
    acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
    acf = acf / max(acf[0], 1e-12)
    min_lag = int(FS / max_hz)
    max_lag = min(int(FS / min_hz), n - 1)
    seg_acf = acf[min_lag:max_lag]
    if len(seg_acf) == 0:
        return 0.0
    return float(np.max(seg_acf))


def _hilbert_regularity(sig):
    """Instantaneous frequency regularity via Hilbert transform."""
    analytic = hilbert(sig)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) / (2 * np.pi / FS)
    inst_freq = inst_freq[(inst_freq > 0.3) & (inst_freq < 4.0)]
    if len(inst_freq) < 10:
        return 0.0
    cv = np.std(inst_freq) / max(np.median(inst_freq), 0.01)
    return float(max(0, 1 - 2 * cv))


class L06_ACFPeak(LateralMethod):
    """Autocorrelation peak height per hemisphere (high = rhythmic)."""
    name = "L06_ACFPeak"
    description = "ACF peak in delta lags per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = _acf_peak(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = _acf_peak(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L07_HilbertCV(LateralMethod):
    """Hilbert instantaneous frequency regularity per hemisphere."""
    name = "L07_HilbertCV"
    description = "Hilbert inst. frequency CV per hemisphere (low CV = rhythmic)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = _hilbert_regularity(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = _hilbert_regularity(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L08_SpectralConcentration(LateralMethod):
    """Spectral concentration: fraction of delta power in a narrow peak per hemisphere."""
    name = "L08_SpectralConcentration"
    description = "Fraction of delta power in ±0.3 Hz peak band per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def concentration(sig):
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]
            narrow = (f >= peak_f - 0.3) & (f <= peak_f + 0.3) & delta
            return float(pxx[narrow].sum() / pxx[delta].sum())

        ls = concentration(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = concentration(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L09_ZeroCrossingRegularity(LateralMethod):
    """Zero-crossing interval regularity per hemisphere."""
    name = "L09_ZeroCrossingRegularity"
    description = "CV of zero-crossing intervals per hemisphere (low CV = rhythmic)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def zc_regularity(sig):
            crossings = np.where(np.diff(np.sign(sig)))[0]
            if len(crossings) < 4:
                return 0.0
            intervals = np.diff(crossings) / FS
            cv = np.std(intervals) / max(np.mean(intervals), 1e-6)
            return float(max(0, 1 - 2 * cv))

        ls = zc_regularity(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = zc_regularity(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L10_EnvelopeRegularity(LateralMethod):
    """Envelope amplitude regularity per hemisphere (low CV = consistent RDA)."""
    name = "L10_EnvelopeRegularity"
    description = "Envelope amplitude CV per hemisphere (low = consistent)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def env_reg(sig):
            envelope = np.abs(hilbert(sig))
            peaks, _ = find_peaks(envelope, distance=int(FS / 3.5))
            if len(peaks) < 3:
                return 0.0
            cv = np.std(envelope[peaks]) / max(np.mean(envelope[peaks]), 1e-6)
            return float(max(0, 1 - cv))

        ls = env_reg(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = env_reg(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
