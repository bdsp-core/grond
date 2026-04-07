"""Power-based lateralization methods (L01-L05).

All methods process each hemisphere independently.
"""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _hemi_power(seg, chs, lo=0.5, hi=4.0):
    """Mean delta bandpower across hemisphere channels."""
    f, pxx = welch(seg[chs], fs=FS, nperseg=400, axis=1)
    mask = (f >= lo) & (f <= hi)
    return float(np.mean(pxx[:, mask]))


def _hemi_peak_power(seg, chs, lo=0.5, hi=4.0):
    """Peak power in delta band on hemisphere mean signal."""
    sig = np.mean(seg[chs], axis=0)
    f, pxx = welch(sig, fs=FS, nperseg=400)
    mask = (f >= lo) & (f <= hi)
    if not mask.any():
        return 0.0
    return float(np.max(pxx[mask]))


def _normalize_pair(left_val, right_val):
    """Normalize left and right values to [0, 1] scores."""
    mx = max(left_val, right_val, 1e-12)
    return left_val / mx, right_val / mx


class L01_DeltaBandpower(LateralMethod):
    """Mean delta (0.5-4 Hz) bandpower per hemisphere."""
    name = "L01_DeltaBandpower"
    description = "Mean Welch PSD in 0.5-4 Hz per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        lp = _hemi_power(seg_f, LEFT_CHS)
        rp = _hemi_power(seg_f, RIGHT_CHS)
        ls, rs = _normalize_pair(lp, rp)
        return {'left_score': ls, 'right_score': rs}


class L02_NarrowbandPeak(LateralMethod):
    """Peak power at best delta frequency per hemisphere."""
    name = "L02_NarrowbandPeak"
    description = "Peak FFT power in 0.5-3.5 Hz per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        lp = _hemi_peak_power(seg_f, LEFT_CHS, 0.5, 3.5)
        rp = _hemi_peak_power(seg_f, RIGHT_CHS, 0.5, 3.5)
        ls, rs = _normalize_pair(lp, rp)
        return {'left_score': ls, 'right_score': rs}


class L03_PeakToMeanRatio(LateralMethod):
    """Peak-to-mean power ratio in delta band per hemisphere (spectral prominence)."""
    name = "L03_PeakToMeanRatio"
    description = "Peak/mean power ratio in delta band per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def prominence(chs):
            sig = np.mean(seg_f[chs], axis=0)
            f, pxx = welch(sig, fs=FS, nperseg=400)
            mask = (f >= 0.5) & (f <= 3.5)
            if not mask.any() or pxx[mask].mean() == 0:
                return 0.0
            return float(np.max(pxx[mask]) / np.mean(pxx[mask]))

        lp = prominence(LEFT_CHS)
        rp = prominence(RIGHT_CHS)
        ls, rs = _normalize_pair(lp, rp)
        return {'left_score': ls, 'right_score': rs}


class L04_BandpowerRatio(LateralMethod):
    """Delta power / total power per hemisphere (relative delta dominance)."""
    name = "L04_BandpowerRatio"
    description = "Delta/total power ratio per hemisphere"

    def _analyze(self, seg_bi):
        def ratio(chs):
            sig = np.mean(seg_bi[chs], axis=0)
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 4.0)
            total = pxx.sum()
            if total == 0:
                return 0.0
            return float(pxx[delta].sum() / total)

        lr = ratio(LEFT_CHS)
        rr = ratio(RIGHT_CHS)
        ls, rs = _normalize_pair(lr, rr)
        return {'left_score': ls, 'right_score': rs}


class L05_RMSAmplitude(LateralMethod):
    """RMS amplitude of delta-filtered signal per hemisphere."""
    name = "L05_RMSAmplitude"
    description = "RMS amplitude of 0.5-4 Hz filtered signal per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)
        lrms = float(np.sqrt(np.mean(seg_f[LEFT_CHS] ** 2)))
        rrms = float(np.sqrt(np.mean(seg_f[RIGHT_CHS] ** 2)))
        ls, rs = _normalize_pair(lrms, rrms)
        return {'left_score': ls, 'right_score': rs}
