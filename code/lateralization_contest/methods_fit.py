"""Variance-explained / model-fitting lateralization methods (L11-L15).

All methods process each hemisphere independently.
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
    """Best variance explained by sinusoidal fit across frequency grid."""
    t = np.arange(len(sig)) / FS
    best_ve = 0.0
    total_var = np.var(sig)
    if total_var < 1e-12:
        return 0.0

    for f in freq_grid:
        basis = np.column_stack([
            np.sin(2 * np.pi * f * t),
            np.cos(2 * np.pi * f * t),
            np.ones(len(t))
        ])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
            fitted = basis @ coeffs
            residual_var = np.var(sig - fitted)
            ve = max(0, 1 - residual_var / total_var)
            if ve > best_ve:
                best_ve = ve
        except np.linalg.LinAlgError:
            pass

    return float(best_ve)


def _template_corr(sig, freq_grid=np.arange(0.5, 3.55, 0.05)):
    """Best Pearson correlation with sinusoidal template."""
    t = np.arange(len(sig)) / FS
    best_corr = 0.0

    for f in freq_grid:
        sin_t = np.sin(2 * np.pi * f * t)
        cos_t = np.cos(2 * np.pi * f * t)
        # Correlation with best-phase sinusoid
        c_sin = abs(np.corrcoef(sig, sin_t)[0, 1]) if np.std(sig) > 1e-12 else 0
        c_cos = abs(np.corrcoef(sig, cos_t)[0, 1]) if np.std(sig) > 1e-12 else 0
        c = max(c_sin, c_cos)
        if c > best_corr:
            best_corr = c

    return float(best_corr)


class L11_VarExplained(LateralMethod):
    """Variance explained by best sinusoidal fit per hemisphere."""
    name = "L11_VarExplained"
    description = "Best R² of sin+cos fit across 0.5-3.5 Hz per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = _var_explained_best_freq(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = _var_explained_best_freq(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L12_TemplateMatch(LateralMethod):
    """Best sinusoidal template correlation per hemisphere."""
    name = "L12_TemplateMatch"
    description = "Peak Pearson corr with sin template per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        ls = _template_corr(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = _template_corr(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L13_AR2Periodicity(LateralMethod):
    """AR(2) model periodicity score per hemisphere.

    AR(2): x[t] = a1*x[t-1] + a2*x[t-2] + e
    Periodic when discriminant (a1² + 4*a2) < 0 and a2 close to -1.
    """
    name = "L13_AR2Periodicity"
    description = "AR(2) oscillatory strength per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def ar2_score(sig):
            # Fit AR(2) via least squares
            n = len(sig)
            if n < 10:
                return 0.0
            X = np.column_stack([sig[1:n - 1], sig[:n - 2]])
            y = sig[2:]
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                a1, a2 = coeffs
                # Discriminant: negative means oscillatory
                disc = a1 ** 2 + 4 * a2
                if disc < 0:
                    # Oscillatory. Score based on how close a2 is to -1
                    return float(min(1.0, abs(a2)))
                else:
                    return 0.0
            except:
                return 0.0

        ls = ar2_score(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = ar2_score(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L14_NarrowbandVE(LateralMethod):
    """Narrowband variance explained: filter at best freq, compute VE per hemisphere."""
    name = "L14_NarrowbandVE"
    description = "Narrowband (±0.3 Hz) variance explained per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def nbve(sig):
            # Find peak frequency
            from scipy.signal import welch
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]

            # Narrowband filter
            lo = max(peak_f - 0.3, 0.1)
            hi = min(peak_f + 0.3, FS / 2 - 0.1)
            if lo >= hi:
                return 0.0
            sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
            nb = sosfiltfilt(sos, sig)

            total_var = np.var(sig)
            if total_var < 1e-12:
                return 0.0
            residual_var = np.var(sig - nb)
            return float(max(0, 1 - residual_var / total_var))

        ls = nbve(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = nbve(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L15_MultiChannelVE(LateralMethod):
    """Mean variance explained across ALL channels per hemisphere (not just top-k)."""
    name = "L15_MultiChannelVE"
    description = "Mean VE across all 8 channels per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        freq_grid = np.arange(0.5, 3.55, 0.1)  # coarser for speed

        def mean_ve(chs):
            ves = []
            for ch in chs:
                ve = _var_explained_best_freq(seg_f[ch], freq_grid)
                ves.append(ve)
            return float(np.mean(ves))

        ls = mean_ve(LEFT_CHS)
        rs = mean_ve(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
