"""Tier 1: Top 5 methods from Round 1 (re-run on IIIC data).

L01-L05: NarrowbandVE, MultiChannelVE, PeakToMeanRatio, SpectralConcentration, TemplateMatch
"""
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _hemi_top_signal(seg, chs, top_k=4):
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0)


def _var_explained_best_freq(sig, freq_grid=np.arange(0.5, 3.55, 0.05)):
    t = np.arange(len(sig)) / FS
    best_ve = 0.0
    total_var = np.var(sig)
    if total_var < 1e-12:
        return 0.0
    for f in freq_grid:
        basis = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(len(t))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
            residual_var = np.var(sig - basis @ coeffs)
            ve = max(0, 1 - residual_var / total_var)
            if ve > best_ve:
                best_ve = ve
        except np.linalg.LinAlgError:
            pass
    return float(best_ve)


class L01_NarrowbandVE(LateralMethod):
    """Narrowband variance explained: filter at best freq, compute VE per hemisphere."""
    name = "L01_NarrowbandVE"
    description = "Narrowband (±0.3 Hz) variance explained per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def nbve(sig):
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]
            lo = max(peak_f - 0.3, 0.1)
            hi = min(peak_f + 0.3, FS / 2 - 0.1)
            if lo >= hi:
                return 0.0
            sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
            nb = sosfiltfilt(sos, sig)
            total_var = np.var(sig)
            if total_var < 1e-12:
                return 0.0
            return float(max(0, 1 - np.var(sig - nb) / total_var))

        ls = nbve(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = nbve(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L02_MultiChannelVE(LateralMethod):
    """Mean variance explained across ALL channels per hemisphere."""
    name = "L02_MultiChannelVE"
    description = "Mean VE across all 8 channels per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        freq_grid = np.arange(0.5, 3.55, 0.1)

        def mean_ve(chs):
            return float(np.mean([_var_explained_best_freq(seg_f[ch], freq_grid) for ch in chs]))

        ls = mean_ve(LEFT_CHS)
        rs = mean_ve(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L03_PeakToMeanRatio(LateralMethod):
    """Peak-to-mean power ratio in delta band per hemisphere."""
    name = "L03_PeakToMeanRatio"
    description = "Peak/mean power ratio in 0.5-3.5 Hz per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def prominence(chs):
            sig = np.mean(seg_f[chs], axis=0)
            f, pxx = welch(sig, fs=FS, nperseg=400)
            mask = (f >= 0.5) & (f <= 3.5)
            if not mask.any() or pxx[mask].mean() == 0:
                return 0.0
            return float(np.max(pxx[mask]) / np.mean(pxx[mask]))

        ls = prominence(LEFT_CHS)
        rs = prominence(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L04_SpectralConcentration(LateralMethod):
    """Fraction of delta power in a narrow peak per hemisphere."""
    name = "L04_SpectralConcentration"
    description = "Power in ±0.3 Hz peak / total delta power per hemisphere"

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


class L05_TemplateMatch(LateralMethod):
    """Best sinusoidal template correlation per hemisphere."""
    name = "L05_TemplateMatch"
    description = "Peak Pearson corr with sin template per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def template_corr(sig):
            t = np.arange(len(sig)) / FS
            best_corr = 0.0
            for f in np.arange(0.5, 3.55, 0.05):
                c_sin = abs(np.corrcoef(sig, np.sin(2 * np.pi * f * t))[0, 1]) if np.std(sig) > 1e-12 else 0
                c_cos = abs(np.corrcoef(sig, np.cos(2 * np.pi * f * t))[0, 1]) if np.std(sig) > 1e-12 else 0
                best_corr = max(best_corr, c_sin, c_cos)
            return float(best_corr)

        ls = template_corr(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = template_corr(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
