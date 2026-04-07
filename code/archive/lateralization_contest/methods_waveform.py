"""Waveform-based lateralization methods (L16-L20).

All methods process each hemisphere independently.
"""
import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import kurtosis

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _hemi_top_signal(seg, chs, top_k=4):
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0)


class L16_AmplitudeConsistency(LateralMethod):
    """Peak amplitude consistency per hemisphere (low CV = consistent = RDA)."""
    name = "L16_AmplitudeConsistency"
    description = "Peak amplitude CV per hemisphere (1 - CV)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def amp_consistency(sig):
            peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
            if len(peaks) < 3:
                return 0.0
            amps = sig[peaks]
            cv = np.std(amps) / max(np.mean(amps), 1e-12)
            return float(max(0, 1 - cv))

        ls = amp_consistency(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = amp_consistency(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L17_PeakRegularity(LateralMethod):
    """Inter-peak interval regularity per hemisphere (low IPI CV = rhythmic)."""
    name = "L17_PeakRegularity"
    description = "IPI regularity (1 - IPI_CV) per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def ipi_regularity(sig):
            peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
            if len(peaks) < 3:
                return 0.0
            ipis = np.diff(peaks) / FS
            cv = np.std(ipis) / max(np.mean(ipis), 1e-6)
            return float(max(0, 1 - cv))

        ls = ipi_regularity(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = ipi_regularity(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L18_WaveformSymmetry(LateralMethod):
    """Waveform symmetry: rise/fall time ratio per hemisphere.

    RDA waves tend to be more symmetric than artifacts.
    """
    name = "L18_WaveformSymmetry"
    description = "Rise/fall time symmetry per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def symmetry(sig):
            peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
            troughs, _ = find_peaks(-sig, distance=int(FS / 3.5))
            if len(peaks) < 2 or len(troughs) < 2:
                return 0.0

            # Compute rise and fall times
            rises, falls = [], []
            for i in range(min(len(peaks), len(troughs)) - 1):
                if peaks[i] < troughs[i]:
                    # Peak before trough: this is a fall
                    falls.append(troughs[i] - peaks[i])
                    if i + 1 < len(peaks):
                        rises.append(peaks[i + 1] - troughs[i])
                else:
                    rises.append(peaks[i] - troughs[i])
                    if i + 1 < len(troughs):
                        falls.append(troughs[i + 1] - peaks[i])

            if not rises or not falls:
                return 0.0
            ratio = np.mean(rises) / max(np.mean(falls), 1e-6)
            # Score: 1.0 when ratio=1.0 (symmetric), decreasing as asymmetric
            return float(max(0, 1 - abs(ratio - 1)))

        ls = symmetry(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = symmetry(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L19_PeakToTrough(LateralMethod):
    """Peak-to-trough amplitude ratio per hemisphere.

    Higher peak-to-trough = stronger rhythmic delta.
    """
    name = "L19_PeakToTrough"
    description = "Mean peak-to-trough amplitude per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def p2t(sig):
            peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
            troughs, _ = find_peaks(-sig, distance=int(FS / 3.5))
            if len(peaks) < 2 or len(troughs) < 2:
                return 0.0
            peak_amp = np.mean(sig[peaks])
            trough_amp = np.mean(-sig[troughs])
            return float(peak_amp + trough_amp)

        ls = p2t(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = p2t(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L20_Kurtosis(LateralMethod):
    """Signal kurtosis per hemisphere.

    Rhythmic delta tends to have lower kurtosis (more sinusoidal, platykurtic)
    vs. spiky transients (leptokurtic). We invert so higher = more rhythmic.
    """
    name = "L20_Kurtosis"
    description = "Inverted kurtosis per hemisphere (low kurt = sinusoidal)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def inv_kurt(sig):
            k = kurtosis(sig, fisher=True)
            # Pure sine wave has kurtosis = -1.5 (fisher)
            # Score: map [-2, 6] to [1, 0]
            return float(np.clip(1 - (k + 1.5) / 7.5, 0, 1))

        ls = inv_kurt(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = inv_kurt(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
