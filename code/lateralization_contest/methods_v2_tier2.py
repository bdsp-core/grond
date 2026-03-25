"""Tier 2: Strong contenders from Round 1 (L06-L10).

ACFPeak, SVDDominance, IntraHemiPLV, VarExplained, DeltaBandpower
"""
import numpy as np
from scipy.signal import welch, hilbert, find_peaks
from numpy.fft import fft, ifft

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


class L06_ACFPeak(LateralMethod):
    """Autocorrelation peak height per hemisphere."""
    name = "L06_ACFPeak"
    description = "ACF peak in delta lags per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def acf_peak(sig):
            x = sig - np.mean(sig)
            n = len(x)
            acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
            acf = acf / max(acf[0], 1e-12)
            min_lag, max_lag = int(FS / 3.5), min(int(FS / 0.5), n - 1)
            seg_acf = acf[min_lag:max_lag]
            return float(np.max(seg_acf)) if len(seg_acf) > 0 else 0.0

        ls = acf_peak(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = acf_peak(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L07_SVDDominance(LateralMethod):
    """SVD first component ratio per hemisphere."""
    name = "L07_SVDDominance"
    description = "SVD σ1/Σσ per hemisphere (high = coherent)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def svd_dom(chs):
            try:
                _, s, _ = np.linalg.svd(seg_f[chs], full_matrices=False)
                return float(s[0] / s.sum()) if s.sum() > 1e-12 else 0.0
            except:
                return 0.0

        ls = svd_dom(LEFT_CHS)
        rs = svd_dom(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L08_IntraHemiPLV(LateralMethod):
    """Phase-locking value within each hemisphere."""
    name = "L08_IntraHemiPLV"
    description = "Within-hemisphere PLV in delta band"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def mean_plv(chs):
            phases = np.array([np.angle(hilbert(seg_f[ch])) for ch in chs])
            plv_vals = []
            for i in range(len(chs)):
                for j in range(i + 1, min(i + 3, len(chs))):
                    plv = float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j])))))
                    plv_vals.append(plv)
            return float(np.mean(plv_vals)) if plv_vals else 0.0

        ls = mean_plv(LEFT_CHS)
        rs = mean_plv(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L09_VarExplained(LateralMethod):
    """Best sinusoidal R² per hemisphere."""
    name = "L09_VarExplained"
    description = "Best R² of sin+cos fit per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = _var_explained_best_freq(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = _var_explained_best_freq(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L10_DeltaBandpower(LateralMethod):
    """Mean delta bandpower per hemisphere."""
    name = "L10_DeltaBandpower"
    description = "Mean Welch PSD in 0.5-4 Hz per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def bp(chs):
            f, pxx = welch(seg_f[chs], fs=FS, nperseg=400, axis=1)
            mask = (f >= 0.5) & (f <= 4.0)
            return float(np.mean(pxx[:, mask]))

        ls = bp(LEFT_CHS)
        rs = bp(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
