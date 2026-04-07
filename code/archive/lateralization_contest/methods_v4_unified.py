"""V4 Unified Methods — simultaneously estimate classification, lateralization, and frequency.

Each method processes hemispheres independently and returns:
    left_score, right_score  — RDA strength per hemisphere (for lateralization + classification)
    left_freq, right_freq    — estimated frequency per hemisphere
    freq                     — best frequency estimate (from dominant hemisphere)

All methods extend LateralMethod but add frequency output.
"""
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt, find_peaks
from scipy.stats import pearsonr
from numpy.fft import fft, ifft

from .base import LateralMethod, FS, LEFT_CHS, RIGHT_CHS


def _normalize_pair(lv, rv):
    mx = max(lv, rv, 1e-12)
    return lv / mx, rv / mx


def _hemi_top_signal(seg, chs, top_k=4):
    powers = np.array([np.var(seg[ch]) for ch in chs])
    top_idx = chs[np.argsort(powers)[::-1][:top_k]]
    return np.mean(seg[top_idx], axis=0)


def _hilbert_freq_cv(sig):
    """Hilbert instantaneous frequency: returns (median_freq, cv, q_score)."""
    if np.std(sig) < 1e-10:
        return np.nan, 1.0, 0.0
    analytic = hilbert(sig)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(inst_phase) * FS / (2.0 * np.pi)
    mask = (inst_freq > 0.3) & (inst_freq < 4.0)
    valid = inst_freq[mask]
    if len(valid) < 20:
        return np.nan, 1.0, 0.0
    med_f = float(np.median(valid))
    cv = float(np.std(valid) / max(med_f, 1e-6))
    q = max(0.0, 1.0 - 2.0 * cv)
    return med_f, cv, q


def _acf_freq(sig):
    """ACF-based frequency estimation."""
    x = sig - np.mean(sig)
    n = len(x)
    acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
    acf = acf / max(acf[0], 1e-12)
    min_lag = int(FS / 3.5)
    max_lag = min(int(FS / 0.5), n - 1)
    seg_acf = acf[min_lag:max_lag]
    if len(seg_acf) == 0:
        return np.nan, 0.0
    peak_idx = np.argmax(seg_acf)
    peak_val = float(seg_acf[peak_idx])
    freq = FS / (min_lag + peak_idx) if (min_lag + peak_idx) > 0 else np.nan
    return freq, peak_val


def _spectral_peak_freq(sig):
    """Welch PSD peak frequency in delta band."""
    f, pxx = welch(sig, fs=FS, nperseg=400)
    delta = (f >= 0.5) & (f <= 3.5)
    if not delta.any() or pxx[delta].sum() == 0:
        return np.nan, 0.0
    peak_idx = np.argmax(pxx[delta])
    return float(f[delta][peak_idx]), float(pxx[delta][peak_idx] / np.mean(pxx[delta]))


def _ve_best_freq(sig, freq_grid=np.arange(0.5, 3.55, 0.05)):
    """Best sinusoidal fit frequency and VE score."""
    t = np.arange(len(sig)) / FS
    best_ve, best_freq = 0.0, np.nan
    total_var = np.var(sig)
    if total_var < 1e-12:
        return np.nan, 0.0
    for f in freq_grid:
        basis = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones(len(t))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(basis, sig, rcond=None)
            ve = max(0, 1 - np.var(sig - basis @ coeffs) / total_var)
            if ve > best_ve:
                best_ve = ve
                best_freq = f
        except:
            pass
    return float(best_freq), float(best_ve)


def _ipi_freq(sig):
    """Frequency from inter-peak intervals."""
    peaks, _ = find_peaks(sig, distance=int(FS / 3.5))
    if len(peaks) < 3:
        return np.nan, 0.0
    ipis = np.diff(peaks) / FS
    freq = 1.0 / np.median(ipis)
    cv = np.std(ipis) / max(np.mean(ipis), 1e-6)
    quality = max(0, 1 - cv)
    return float(freq), float(quality)


class _UnifiedBase(LateralMethod):
    """Base for unified methods that output freq + lateralization."""

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        left_sig = _hemi_top_signal(seg_f, LEFT_CHS)
        right_sig = _hemi_top_signal(seg_f, RIGHT_CHS)

        left_score, left_freq = self._score_hemisphere(left_sig)
        right_score, right_freq = self._score_hemisphere(right_sig)

        ls, rs = _normalize_pair(left_score, right_score)

        # Frequency from dominant hemisphere
        if ls >= rs:
            freq = left_freq if np.isfinite(left_freq) else right_freq
        else:
            freq = right_freq if np.isfinite(right_freq) else left_freq

        return {
            'left_score': ls, 'right_score': rs,
            'extras': {
                'left_freq': float(left_freq) if np.isfinite(left_freq) else None,
                'right_freq': float(right_freq) if np.isfinite(right_freq) else None,
                'freq': float(freq) if np.isfinite(freq) else None,
            }
        }

    def _score_hemisphere(self, sig):
        """Return (score, freq) for one hemisphere. Override in subclass."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════
# Unified methods
# ═══════════════════════════════════════════════════════════════

class U01_HilbertCV(_UnifiedBase):
    """M3_HilbertCV adapted: Hilbert Q-score + frequency per hemisphere."""
    name = "U01_HilbertCV"
    description = "Hilbert inst. freq CV per hemisphere → quality + frequency"

    def _score_hemisphere(self, sig):
        freq, cv, q = _hilbert_freq_cv(sig)
        return q, freq


class U02_ACFPeakFreq(_UnifiedBase):
    """ACF peak height + frequency per hemisphere."""
    name = "U02_ACFPeakFreq"
    description = "ACF peak height per hemisphere → rhythmicity + frequency"

    def _score_hemisphere(self, sig):
        freq, peak_val = _acf_freq(sig)
        return max(0, peak_val), freq


class U03_SpectralPeak(_UnifiedBase):
    """Spectral peak prominence + frequency per hemisphere."""
    name = "U03_SpectralPeak"
    description = "Welch PSD peak prominence per hemisphere → dominance + frequency"

    def _score_hemisphere(self, sig):
        freq, prominence = _spectral_peak_freq(sig)
        return min(prominence / 10, 1.0), freq


class U04_VarExplained(_UnifiedBase):
    """Sinusoidal VE + best frequency per hemisphere."""
    name = "U04_VarExplained"
    description = "Best sin+cos R² per hemisphere → VE + frequency"

    def _score_hemisphere(self, sig):
        freq, ve = _ve_best_freq(sig)
        return ve, freq


class U05_IPIRegularity(_UnifiedBase):
    """Peak-based IPI regularity + frequency per hemisphere."""
    name = "U05_IPIRegularity"
    description = "IPI regularity per hemisphere → rhythmicity + frequency"

    def _score_hemisphere(self, sig):
        freq, quality = _ipi_freq(sig)
        return quality, freq


class U06_EnvelopeAmplitude(_UnifiedBase):
    """Envelope amplitude (V4 winner) + Hilbert frequency per hemisphere."""
    name = "U06_EnvAmp_HilbertFreq"
    description = "Envelope amplitude + Hilbert frequency per hemisphere"

    def _score_hemisphere(self, sig):
        env = np.abs(hilbert(sig))
        score = float(np.mean(env))
        freq, _, _ = _hilbert_freq_cv(sig)
        return score, freq


class U07_RMS_ACFFreq(_UnifiedBase):
    """RMS amplitude (V4 #2) + ACF frequency per hemisphere."""
    name = "U07_RMS_ACFFreq"
    description = "RMS amplitude + ACF frequency per hemisphere"

    def _score_hemisphere(self, sig):
        score = float(np.sqrt(np.mean(sig ** 2)))
        freq, _ = _acf_freq(sig)
        return score, freq


class U08_Bandpower_SpectralFreq(_UnifiedBase):
    """Delta bandpower (V4 #3) + spectral peak frequency per hemisphere."""
    name = "U08_BP_SpectralFreq"
    description = "Delta bandpower + spectral peak freq per hemisphere"

    def _score_hemisphere(self, sig):
        score = float(np.var(sig))
        freq, _ = _spectral_peak_freq(sig)
        return score, freq


class U09_NarrowbandVE(_UnifiedBase):
    """Narrowband VE: find best freq via spectral peak, then VE at that freq."""
    name = "U09_NarrowbandVE"
    description = "Narrowband VE at spectral peak freq per hemisphere"

    def _score_hemisphere(self, sig):
        peak_freq, _ = _spectral_peak_freq(sig)
        if not np.isfinite(peak_freq):
            return 0.0, np.nan
        lo = max(peak_freq - 0.3, 0.1)
        hi = min(peak_freq + 0.3, FS / 2 - 0.1)
        if lo >= hi:
            return 0.0, peak_freq
        sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        nb = sosfiltfilt(sos, sig)
        total_var = np.var(sig)
        if total_var < 1e-12:
            return 0.0, peak_freq
        ve = max(0, 1 - np.var(sig - nb) / total_var)
        return ve, peak_freq


class U10_MultiChannel_HilbertFreq(LateralMethod):
    """All-channel Hilbert frequency + mean envelope per hemisphere."""
    name = "U10_MultiCh_HilbertFreq"
    description = "Per-channel Hilbert freq + envelope across all channels per hemisphere"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        def score_hemi(chs):
            freqs, envs = [], []
            for ch in chs:
                sig = seg_f[ch]
                env = np.abs(hilbert(sig))
                envs.append(np.mean(env))
                freq, _, q = _hilbert_freq_cv(sig)
                if np.isfinite(freq) and q > 0.1:
                    freqs.append(freq)
            score = float(np.mean(envs)) if envs else 0.0
            freq = float(np.median(freqs)) if freqs else np.nan
            return score, freq

        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        ls_n, rs_n = _normalize_pair(ls, rs)

        if ls_n >= rs_n:
            freq = lf if np.isfinite(lf) else rf
        else:
            freq = rf if np.isfinite(rf) else lf

        return {
            'left_score': ls_n, 'right_score': rs_n,
            'extras': {'left_freq': lf, 'right_freq': rf, 'freq': freq}
        }


class U11_HilbertCV_MultiChannel(LateralMethod):
    """M3_HilbertCV faithful adaptation: top-3 per hemisphere, q-score + freq."""
    name = "U11_HilbertCV_Top3"
    description = "M3_HilbertCV per hemisphere: top-3 channels, median freq + CV"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
        seg_narrow = sosfiltfilt(sos, seg_f, axis=1)

        def score_hemi(chs):
            powers = np.array([np.var(seg_narrow[ch]) for ch in chs])
            top3 = chs[np.argsort(powers)[::-1][:3]]
            ch_freqs, ch_cvs = [], []
            for ch in top3:
                sig = seg_narrow[ch]
                if np.std(sig) < 1e-10:
                    continue
                analytic = hilbert(sig)
                inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
                mask = (inst_freq > 0.3) & (inst_freq < 4.0)
                valid = inst_freq[mask]
                if len(valid) < 20:
                    continue
                ch_freqs.append(float(np.median(valid)))
                ch_cvs.append(float(np.std(valid) / max(np.median(valid), 1e-6)))
            if not ch_freqs:
                return 0.0, np.nan
            freq = float(np.median(ch_freqs))
            cv = float(np.median(ch_cvs))
            q = max(0.0, 1.0 - 2.0 * cv)
            return q, freq

        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        ls_n, rs_n = _normalize_pair(ls, rs)

        if ls_n >= rs_n:
            freq = lf if np.isfinite(lf) else rf
        else:
            freq = rf if np.isfinite(rf) else lf

        return {
            'left_score': ls_n, 'right_score': rs_n,
            'extras': {'left_freq': lf, 'right_freq': rf, 'freq': freq}
        }


class U12_EnvAmp_VEFreq(_UnifiedBase):
    """Envelope amplitude (best lateralizer) + VE frequency (best freq estimator)."""
    name = "U12_EnvAmp_VEFreq"
    description = "Envelope amplitude for score + sinusoidal VE for frequency"

    def _score_hemisphere(self, sig):
        env = np.abs(hilbert(sig))
        score = float(np.mean(env))
        freq, _ = _ve_best_freq(sig, np.arange(0.5, 3.55, 0.1))  # coarser for speed
        return score, freq


class U13_PLV_HilbertFreq(LateralMethod):
    """Intra-hemisphere PLV for score + Hilbert for frequency."""
    name = "U13_PLV_HilbertFreq"
    description = "Within-hemisphere PLV + Hilbert frequency"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.5, hi=4.0)

        def score_hemi(chs):
            phases = np.array([np.angle(hilbert(seg_f[ch])) for ch in chs])
            plvs = []
            for i in range(len(chs)):
                for j in range(i + 1, min(i + 3, len(chs))):
                    plv = float(np.abs(np.mean(np.exp(1j * (phases[i] - phases[j])))))
                    plvs.append(plv)
            score = float(np.mean(plvs)) if plvs else 0.0

            # Frequency from top channel
            powers = np.array([np.var(seg_f[ch]) for ch in chs])
            top = chs[np.argmax(powers)]
            freq, _, _ = _hilbert_freq_cv(seg_f[top])
            return score, freq

        ls, lf = score_hemi(LEFT_CHS)
        rs, rf = score_hemi(RIGHT_CHS)
        ls_n, rs_n = _normalize_pair(ls, rs)

        if ls_n >= rs_n:
            freq = lf if np.isfinite(lf) else rf
        else:
            freq = rf if np.isfinite(rf) else lf

        return {
            'left_score': ls_n, 'right_score': rs_n,
            'extras': {'left_freq': lf, 'right_freq': rf, 'freq': freq}
        }


class U14_Bandpower_IPIFreq(_UnifiedBase):
    """Delta bandpower for score + IPI-based frequency."""
    name = "U14_BP_IPIFreq"
    description = "Delta bandpower + IPI frequency per hemisphere"

    def _score_hemisphere(self, sig):
        score = float(np.var(sig))
        freq, _ = _ipi_freq(sig)
        return score, freq


class U15_EnvAmp_ACFFreq(_UnifiedBase):
    """Envelope amplitude + ACF frequency — combining V4 winner with robust freq."""
    name = "U15_EnvAmp_ACFFreq"
    description = "Envelope amplitude + ACF frequency per hemisphere"

    def _score_hemisphere(self, sig):
        env = np.abs(hilbert(sig))
        score = float(np.mean(env))
        freq, _ = _acf_freq(sig)
        return score, freq
