"""Tier 3: New methods (L11-L20).

Novel approaches not tested in Round 1.
"""
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt, find_peaks
from scipy.stats import pearsonr

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _hemi_top_signal(seg, chs, top_k=4):
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0)


class L11_WaveletRidge(LateralMethod):
    """CWT ridge strength at delta frequency per hemisphere."""
    name = "L11_WaveletRidge"
    description = "Morlet wavelet ridge energy in 0.5-3.5 Hz per hemisphere"

    def _analyze(self, seg_bi):
        from scipy.signal import morlet2, cwt

        seg_f = self.prefilter(seg_bi)

        def ridge_strength(sig):
            freqs = np.arange(0.5, 3.6, 0.25)
            widths = freqs * 5  # ~5 cycles per wavelet
            # Use scipy CWT with Ricker wavelet as proxy (faster than complex Morlet)
            from scipy.signal import ricker
            w = np.array([FS / f for f in freqs])
            try:
                cwtm = np.abs(cwt(sig, ricker, w))
                # Ridge: max across frequencies at each time point
                ridge = np.max(cwtm, axis=0)
                return float(np.mean(ridge))
            except:
                return 0.0

        ls = ridge_strength(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = ridge_strength(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L12_EnvelopePeakedness(LateralMethod):
    """Envelope peak-to-mean ratio per hemisphere.

    Rhythmic delta has a peaked envelope (peaks at each wave crest).
    """
    name = "L12_EnvelopePeakedness"
    description = "Envelope peak/mean ratio per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def peakedness(sig):
            env = np.abs(hilbert(sig))
            peaks, _ = find_peaks(env, distance=int(FS / 3.5))
            if len(peaks) < 2:
                return 0.0
            return float(np.mean(env[peaks]) / max(np.mean(env), 1e-12))

        ls = peakedness(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = peakedness(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L13_PhaseConsistency(LateralMethod):
    """Phase consistency across estimated cycles per hemisphere.

    Cut the signal into segments of length 1/peak_freq, compute phase alignment.
    """
    name = "L13_PhaseConsistency"
    description = "Inter-cycle phase consistency per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def phase_con(sig):
            # Find dominant frequency
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]
            cycle_len = int(FS / peak_f)
            if cycle_len < 10:
                return 0.0

            # Cut into cycles and compute phase alignment
            n_cycles = len(sig) // cycle_len
            if n_cycles < 3:
                return 0.0

            phases = []
            for i in range(n_cycles):
                chunk = sig[i * cycle_len:(i + 1) * cycle_len]
                analytic = hilbert(chunk)
                phases.append(np.angle(analytic[cycle_len // 2]))

            # Phase consistency = resultant vector length
            phases = np.array(phases)
            plv = float(np.abs(np.mean(np.exp(1j * phases))))
            return plv

        ls = phase_con(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = phase_con(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L14_CrossChannelCorr(LateralMethod):
    """Mean pairwise correlation within hemisphere."""
    name = "L14_CrossChannelCorr"
    description = "Mean within-hemisphere channel correlation"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def mean_corr(chs):
            corrs = []
            for i in range(len(chs)):
                for j in range(i + 1, len(chs)):
                    if np.std(seg_f[chs[i]]) > 1e-12 and np.std(seg_f[chs[j]]) > 1e-12:
                        c = abs(pearsonr(seg_f[chs[i]], seg_f[chs[j]])[0])
                        corrs.append(c)
            return float(np.mean(corrs)) if corrs else 0.0

        ls = mean_corr(LEFT_CHS)
        rs = mean_corr(RIGHT_CHS)
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L15_SpectralFlatness(LateralMethod):
    """Spectral flatness (Wiener entropy) per hemisphere.

    Low spectral flatness = tonal/rhythmic (RDA-like).
    We invert: score = 1 - flatness.
    """
    name = "L15_SpectralFlatness"
    description = "1 - spectral flatness in delta band per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def inv_flatness(sig):
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 4.0)
            pxx_d = pxx[delta]
            pxx_d = pxx_d[pxx_d > 0]
            if len(pxx_d) < 2:
                return 0.0
            geo_mean = np.exp(np.mean(np.log(pxx_d)))
            arith_mean = np.mean(pxx_d)
            flatness = geo_mean / max(arith_mean, 1e-12)
            return float(max(0, 1 - flatness))

        ls = inv_flatness(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = inv_flatness(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L16_GradientEnergy(LateralMethod):
    """Energy of signal derivative per hemisphere.

    RDA has smooth, slow oscillations → low derivative energy relative to signal energy.
    Score = signal_energy / (derivative_energy + signal_energy).
    """
    name = "L16_GradientEnergy"
    description = "Signal smoothness (low derivative energy) per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def smoothness(sig):
            sig_e = np.mean(sig ** 2)
            grad_e = np.mean(np.diff(sig) ** 2)
            if sig_e + grad_e < 1e-12:
                return 0.0
            return float(sig_e / (sig_e + grad_e))

        ls = smoothness(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = smoothness(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L17_CepstralPeak(LateralMethod):
    """Cepstral peak prominence per hemisphere.

    The cepstrum reveals periodicity as a peak at the period quefrency.
    """
    name = "L17_CepstralPeak"
    description = "Cepstral peak prominence in delta range per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def cepstral_peak(sig):
            # Power cepstrum
            spectrum = np.fft.fft(sig)
            log_spectrum = np.log(np.abs(spectrum) + 1e-12)
            cepstrum = np.real(np.fft.ifft(log_spectrum))

            # Look for peak in delta range (quefrency = period in samples)
            min_q = int(FS / 3.5)  # ~57 samples (3.5 Hz)
            max_q = min(int(FS / 0.5), len(cepstrum) // 2)  # ~400 samples (0.5 Hz)
            if min_q >= max_q:
                return 0.0

            seg_c = cepstrum[min_q:max_q]
            peak_val = np.max(seg_c)
            mean_val = np.mean(np.abs(seg_c))
            if mean_val < 1e-12:
                return 0.0
            return float(max(0, peak_val / mean_val))

        ls = cepstral_peak(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = cepstral_peak(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L18_SubbandEntropy(LateralMethod):
    """Entropy in delta subband per hemisphere.

    Low entropy = organized/rhythmic signal. Score = 1 - normalized_entropy.
    """
    name = "L18_SubbandEntropy"
    description = "1 - normalized entropy in delta band per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def inv_entropy(sig):
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.3) & (f <= 5.0)
            pxx_d = pxx[delta]
            if pxx_d.sum() == 0:
                return 0.0
            p = pxx_d / pxx_d.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log(p))
            max_entropy = np.log(len(p))
            if max_entropy < 1e-12:
                return 0.0
            return float(max(0, 1 - entropy / max_entropy))

        ls = inv_entropy(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = inv_entropy(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L19_MatchedFilter(LateralMethod):
    """Matched filter response per hemisphere.

    For each hemisphere, estimate the dominant frequency, create a matched
    sinusoidal template, and measure the correlation.
    """
    name = "L19_MatchedFilter"
    description = "Matched sinusoidal filter response per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def matched(sig):
            # Find dominant freq
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]

            # Fit sin+cos at that frequency
            t = np.arange(len(sig)) / FS
            basis = np.column_stack([np.sin(2 * np.pi * peak_f * t),
                                     np.cos(2 * np.pi * peak_f * t),
                                     np.ones(len(t))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
                fitted = basis @ coeffs
                total_var = np.var(sig)
                if total_var < 1e-12:
                    return 0.0
                return float(max(0, 1 - np.var(sig - fitted) / total_var))
            except:
                return 0.0

        ls = matched(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = matched(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}


class L20_CoherenceWithTemplate(LateralMethod):
    """Coherence between hemispheric signal and a fitted sinusoid.

    High coherence = signal closely tracks a periodic template.
    """
    name = "L20_CoherenceWithTemplate"
    description = "Coherence with best-fit sinusoid per hemisphere"

    def _analyze(self, seg_bi):
        from scipy.signal import coherence
        seg_f = self.prefilter(seg_bi)

        def coh_with_template(sig):
            f, pxx = welch(sig, fs=FS, nperseg=400)
            delta = (f >= 0.5) & (f <= 3.5)
            if not delta.any() or pxx[delta].sum() == 0:
                return 0.0
            peak_f = f[delta][np.argmax(pxx[delta])]

            t = np.arange(len(sig)) / FS
            template = np.sin(2 * np.pi * peak_f * t)
            fc, cxy = coherence(sig, template, fs=FS, nperseg=200)
            narrow = (fc >= peak_f - 0.5) & (fc <= peak_f + 0.5)
            if not narrow.any():
                return 0.0
            return float(np.max(cxy[narrow]))

        ls = coh_with_template(_hemi_top_signal(seg_f, LEFT_CHS))
        rs = coh_with_template(_hemi_top_signal(seg_f, RIGHT_CHS))
        ls, rs = _normalize_pair(ls, rs)
        return {'left_score': ls, 'right_score': rs}
