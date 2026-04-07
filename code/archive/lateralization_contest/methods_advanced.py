"""Advanced lateralization methods (L21-L25): connectivity, coherence, SVD.

All methods process each hemisphere independently (or compare hemispheres
in a symmetric way).
"""
import numpy as np
from scipy.signal import welch, coherence, hilbert, butter, sosfiltfilt
from scipy.stats import pearsonr

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


class L21_IntraHemiCoherence(LateralMethod):
    """Mean pairwise coherence within each hemisphere in delta band.

    High intra-hemisphere coherence = channels are synchronized = RDA on that side.
    """
    name = "L21_IntraHemiCoherence"
    description = "Mean within-hemisphere coherence in 0.5-4 Hz"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def mean_coh(chs):
            coh_vals = []
            for i in range(len(chs)):
                for j in range(i + 1, min(i + 3, len(chs))):  # limit pairs for speed
                    f, cxy = coherence(seg_f[chs[i]], seg_f[chs[j]],
                                       fs=FS, nperseg=200)
                    delta = (f >= 0.5) & (f <= 4.0)
                    if delta.any():
                        coh_vals.append(float(np.mean(cxy[delta])))
            return float(np.mean(coh_vals)) if coh_vals else 0.0

        ls = mean_coh(LEFT_CHS)
        rs = mean_coh(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L22_IntraHemiPLV(LateralMethod):
    """Phase-locking value within each hemisphere at delta frequencies.

    PLV measures phase synchrony — high PLV = channels oscillating together.
    """
    name = "L22_IntraHemiPLV"
    description = "Within-hemisphere phase-locking value in delta band"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def mean_plv(chs):
            # Hilbert transform each channel
            phases = []
            for ch in chs:
                analytic = hilbert(seg_f[ch])
                phases.append(np.angle(analytic))
            phases = np.array(phases)

            plv_vals = []
            for i in range(len(chs)):
                for j in range(i + 1, min(i + 3, len(chs))):
                    phase_diff = phases[i] - phases[j]
                    plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
                    plv_vals.append(plv)
            return float(np.mean(plv_vals)) if plv_vals else 0.0

        ls = mean_plv(LEFT_CHS)
        rs = mean_plv(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L23_InterHemiCorr(LateralMethod):
    """Cross-hemisphere envelope correlation.

    Low inter-hemisphere correlation suggests activity is lateralized.
    We return (1 - corr) as an asymmetry boost, combined with per-hemi power.
    """
    name = "L23_InterHemiCorr"
    description = "Inter-hemisphere envelope correlation (inverted)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        # Envelope of best channels per hemisphere
        def top_envelope(chs):
            powers = np.array([np.var(seg_f[ch]) for ch in chs])
            top_idx = chs[np.argsort(powers)[::-1][:4]]
            sig = np.mean(seg_f[top_idx], axis=0)
            return np.abs(hilbert(sig))

        left_env = top_envelope(LEFT_CHS)
        right_env = top_envelope(RIGHT_CHS)

        # Inter-hemisphere correlation
        if np.std(left_env) < 1e-12 or np.std(right_env) < 1e-12:
            inter_corr = 0.0
        else:
            inter_corr = abs(pearsonr(left_env, right_env)[0])

        # Per-hemisphere power (using envelope mean as RDA strength proxy)
        left_power = float(np.mean(left_env))
        right_power = float(np.mean(right_env))

        # Score: power × (1 - inter_corr) gives lateralized cases higher asymmetry
        asymmetry_boost = 1.0 - inter_corr
        ls = left_power * (1.0 + asymmetry_boost)
        rs = right_power * (1.0 + asymmetry_boost)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L24_EnvelopeCorrelation(LateralMethod):
    """Per-channel envelope amplitude, scored per hemisphere.

    Simple approach: mean envelope amplitude across channels per hemisphere.
    """
    name = "L24_EnvelopeAmplitude"
    description = "Mean analytic envelope amplitude per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def env_amp(chs):
            amps = []
            for ch in chs:
                env = np.abs(hilbert(seg_f[ch]))
                amps.append(float(np.mean(env)))
            return float(np.mean(amps))

        ls = env_amp(LEFT_CHS)
        rs = env_amp(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L25_SVDDominance(LateralMethod):
    """SVD first component dominance per hemisphere.

    If RDA is present on a hemisphere, the first singular value should
    dominate (channels are coherent). High σ1/sum(σ) = more coherent = RDA.
    """
    name = "L25_SVDDominance"
    description = "SVD first component ratio per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def svd_dominance(chs):
            data = seg_f[chs]  # (n_chs, 2000)
            try:
                _, s, _ = np.linalg.svd(data, full_matrices=False)
                if s.sum() < 1e-12:
                    return 0.0
                return float(s[0] / s.sum())
            except:
                return 0.0

        ls = svd_dominance(LEFT_CHS)
        rs = svd_dominance(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
