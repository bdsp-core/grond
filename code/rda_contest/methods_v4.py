"""Round 4 contest methods — 8 new approaches targeting identified gaps.

Methods:
  - FOOOF peak height above aperiodic background
  - PCA delta dominance (first principal component)
  - Half-segment frequency stability
  - Wiener entropy (spectral flatness)
  - Ensemble/channel disagreement
  - ACF at multiple lags
  - Inter-hemisphere coherence / power ratio
  - Delta power ratio
"""
import numpy as np
from scipy.signal import welch
from scipy.stats import gmean
from numpy.fft import fft, ifft

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS
from .methods_v2 import _best_hemi_signal, _spectral_peak_freq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x, center=0.5, scale=5.0):
    """Sigmoid normalization to [0, 1]."""
    return float(1.0 / (1.0 + np.exp(-scale * (x - center))))


def _acf_full(sig):
    """Full normalized autocorrelation function."""
    x = sig - np.mean(sig)
    n = len(x)
    acf = np.real(ifft(np.abs(fft(x, 2 * n)) ** 2))[:n]
    acf = acf / max(acf[0], 1e-12)
    return acf


# ---------------------------------------------------------------------------
# V4_FOOOFPeakHeight
# ---------------------------------------------------------------------------

class V4_FOOOFPeakHeight(RDAMethod):
    """FOOOF peak height above 1/f aperiodic background."""
    name = "V4_FOOOFPeakHeight"
    description = "Peak power above aperiodic fit in delta band via FOOOF"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Compute PSD on the ORIGINAL (unfiltered) signal for FOOOF —
        # but we need best_chs from filtered. Re-select on original.
        seg_f_orig = self.prefilter(seg_bi, lo=0.3, hi=40.0)
        best_sig_wide, _, _ = _best_hemi_signal(seg_f_orig)

        f_psd, pxx = welch(best_sig_wide, fs=FS, nperseg=400)

        try:
            from fooof import FOOOF
            fm = FOOOF(
                peak_width_limits=[0.3, 2.0],
                max_n_peaks=3,
                aperiodic_mode='fixed',
                verbose=False,
            )
            fm.fit(f_psd, pxx, [0.5, 15])
            peak_params = fm.get_params('peak_params')

            if peak_params is None or (peak_params.ndim == 1 and len(peak_params) == 0):
                freq, _ = _spectral_peak_freq(best_sig)
                return {'freq': freq, 'q_score': 0.0}

            # peak_params shape: (n_peaks, 3) — [center_freq, power, bandwidth]
            if peak_params.ndim == 1:
                peak_params = peak_params.reshape(1, -1)

            # Find peaks in delta range (0.5-3.5 Hz)
            delta_mask = (peak_params[:, 0] >= 0.5) & (peak_params[:, 0] <= 3.5)
            if not delta_mask.any():
                freq, _ = _spectral_peak_freq(best_sig)
                return {'freq': freq, 'q_score': 0.0}

            delta_peaks = peak_params[delta_mask]
            best_peak_idx = np.argmax(delta_peaks[:, 1])
            freq = float(delta_peaks[best_peak_idx, 0])
            height = float(delta_peaks[best_peak_idx, 1])

            # Normalize height via sigmoid (typical heights 0-2 log power)
            q = _sigmoid(height, center=0.5, scale=3.0)
            return {'freq': freq, 'q_score': q}

        except ImportError:
            # Fallback if fooof not installed: use spectral peak prominence
            freq, prominence = _spectral_peak_freq(best_sig)
            q = _sigmoid(prominence, center=5.0, scale=0.5)
            return {'freq': freq, 'q_score': q}


# ---------------------------------------------------------------------------
# V4_PCA_DeltaDominance
# ---------------------------------------------------------------------------

class V4_PCA_DeltaDominance(RDAMethod):
    """First principal component dominance of delta-filtered channels."""
    name = "V4_PCA_DeltaDominance"
    description = "Variance explained by PC1 of delta-filtered 18 channels"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        # Center each channel
        seg_centered = seg_f - seg_f.mean(axis=1, keepdims=True)

        # SVD: seg_centered is (18, 2000)
        U, S, Vt = np.linalg.svd(seg_centered, full_matrices=False)

        # Variance explained by PC1
        var_explained = S ** 2
        total_var = var_explained.sum()
        if total_var < 1e-12:
            return {'freq': np.nan, 'q_score': 0.0}

        pc1_ratio = float(var_explained[0] / total_var)

        # PC1 time course: first row of Vt scaled by S[0]
        pc1_timecourse = S[0] * Vt[0, :]

        # Spectral peak of PC1
        freq, _ = _spectral_peak_freq(pc1_timecourse)

        # q_score = variance explained ratio (typically 0.3-0.9)
        # Scale so that ~0.6+ maps to high confidence
        q = _sigmoid(pc1_ratio, center=0.4, scale=8.0)

        return {'freq': freq, 'q_score': q, 'extras': {'pc1_var': pc1_ratio}}


# ---------------------------------------------------------------------------
# V4_HalfSegmentStability
# ---------------------------------------------------------------------------

class V4_HalfSegmentStability(RDAMethod):
    """Frequency stability between first and second half of segment."""
    name = "V4_HalfSegmentStability"
    description = "Frequency agreement between segment halves (stable = RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        n = len(best_sig)
        half = n // 2
        sig1 = best_sig[:half]
        sig2 = best_sig[half:]

        # Dominant freq from each half via Welch
        f1, _ = _spectral_peak_freq(sig1)
        f2, _ = _spectral_peak_freq(sig2)

        mean_f = (f1 + f2) / 2.0
        if mean_f < 0.01:
            return {'freq': np.nan, 'q_score': 0.0}

        # q_score: 1 if identical, 0 if frequencies differ by >50% of mean
        q = max(0.0, 1.0 - 2.0 * abs(f1 - f2) / mean_f)

        return {
            'freq': float(mean_f),
            'q_score': float(q),
            'extras': {'f_half1': f1, 'f_half2': f2},
        }


# ---------------------------------------------------------------------------
# V4_WienerEntropy
# ---------------------------------------------------------------------------

class V4_WienerEntropy(RDAMethod):
    """Spectral flatness (Wiener entropy) in delta band — low = rhythmic."""
    name = "V4_WienerEntropy"
    description = "1 - spectral flatness in delta band (peaked spectrum = RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        f, pxx = welch(best_sig, fs=FS, nperseg=400)
        delta_mask = (f >= 0.5) & (f <= 3.5)
        pxx_delta = pxx[delta_mask]

        if len(pxx_delta) == 0 or pxx_delta.sum() < 1e-12:
            return {'freq': np.nan, 'q_score': 0.0}

        # Avoid log(0) — clip tiny values
        pxx_delta = np.clip(pxx_delta, 1e-20, None)

        geo_mean = float(gmean(pxx_delta))
        arith_mean = float(np.mean(pxx_delta))

        if arith_mean < 1e-20:
            return {'freq': np.nan, 'q_score': 0.0}

        wiener = geo_mean / arith_mean  # 0 = pure tone, 1 = white noise
        q = 1.0 - wiener

        freq = float(f[delta_mask][np.argmax(pxx_delta)])

        return {
            'freq': freq,
            'q_score': float(np.clip(q, 0.0, 1.0)),
            'extras': {'wiener_entropy': wiener},
        }


# ---------------------------------------------------------------------------
# V4_EnsembleDisagreement
# ---------------------------------------------------------------------------

class V4_EnsembleDisagreement(RDAMethod):
    """Channel frequency agreement — fraction of channels near median freq."""
    name = "V4_EnsembleDisagreement"
    description = "Fraction of channels agreeing on delta frequency (high = clear RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        # Compute dominant delta freq for each channel
        ch_freqs = []
        ch_powers = []
        for ch in range(min(18, seg_f.shape[0])):
            freq_ch, prom = _spectral_peak_freq(seg_f[ch])
            ch_freqs.append(freq_ch)
            ch_powers.append(np.var(seg_f[ch]))

        ch_freqs = np.array(ch_freqs)
        ch_powers = np.array(ch_powers)

        # Select top-8 channels by power
        top8 = np.argsort(ch_powers)[::-1][:8]
        top_freqs = ch_freqs[top8]

        if len(top_freqs) < 2:
            return {'freq': np.nan, 'q_score': 0.0}

        median_freq = float(np.median(top_freqs))
        if median_freq < 0.1:
            return {'freq': np.nan, 'q_score': 0.0}

        # Fraction of top-8 channels within ±0.3 Hz of median
        agreement = np.mean(np.abs(top_freqs - median_freq) <= 0.3)
        q = float(agreement)

        return {
            'freq': median_freq,
            'q_score': q,
            'extras': {'channel_freqs': top_freqs.tolist()},
        }


# ---------------------------------------------------------------------------
# V4_ACFMultipleLags
# ---------------------------------------------------------------------------

class V4_ACFMultipleLags(RDAMethod):
    """ACF values at lag T, 2T, 3T where T = 1/f."""
    name = "V4_ACFMultipleLags"
    description = "Mean ACF at lags 1/f, 2/f, 3/f (sustained rhythm = RDA)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Estimate fundamental frequency
        freq, _ = _spectral_peak_freq(best_sig)
        if freq < 0.3 or not np.isfinite(freq):
            return {'freq': np.nan, 'q_score': 0.0}

        acf = _acf_full(best_sig)
        n = len(acf)

        # Sample ACF at T, 2T, 3T
        period_samples = FS / freq  # samples per cycle
        acf_vals = []
        for k in [1, 2, 3]:
            lag = int(round(k * period_samples))
            if 0 < lag < n:
                acf_vals.append(acf[lag])

        if len(acf_vals) == 0:
            return {'freq': freq, 'q_score': 0.0}

        q = float(np.mean(acf_vals))
        q = max(0.0, q)  # ACF can be negative

        return {
            'freq': freq,
            'q_score': min(q, 1.0),
            'extras': {'acf_lags': acf_vals},
        }


# ---------------------------------------------------------------------------
# V4_InterHemiCoherence
# ---------------------------------------------------------------------------

class V4_InterHemiCoherence(RDAMethod):
    """Max hemispheric power ratio at estimated RDA frequency."""
    name = "V4_InterHemiCoherence"
    description = "Max(left, right) narrowband power ratio — at least one hemisphere has RDA"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Estimate frequency
        freq, _ = _spectral_peak_freq(best_sig)
        if freq < 0.3 or not np.isfinite(freq):
            return {'freq': np.nan, 'q_score': 0.0}

        # Narrowband filter at estimated freq
        seg_nb = self.narrowband(seg_bi, freq, bw=0.3)

        # Compute power ratio for each hemisphere
        # power_ratio = var(narrowband) / var(broadband signal) per hemisphere
        left_nb_var = np.mean([np.var(seg_nb[ch]) for ch in LEFT_CHS])
        right_nb_var = np.mean([np.var(seg_nb[ch]) for ch in RIGHT_CHS])
        left_total_var = np.mean([np.var(seg_bi[ch]) for ch in LEFT_CHS])
        right_total_var = np.mean([np.var(seg_bi[ch]) for ch in RIGHT_CHS])

        left_ratio = left_nb_var / max(left_total_var, 1e-12)
        right_ratio = right_nb_var / max(right_total_var, 1e-12)

        q = float(max(left_ratio, right_ratio))
        # Normalize: typical ratios are 0-0.5; scale up
        q = min(q * 2.0, 1.0)

        return {
            'freq': freq,
            'q_score': q,
            'extras': {
                'left_power_ratio': float(left_ratio),
                'right_power_ratio': float(right_ratio),
            },
        }


# ---------------------------------------------------------------------------
# V4_DeltaPowerRatio
# ---------------------------------------------------------------------------

class V4_DeltaPowerRatio(RDAMethod):
    """Delta band power relative to total power on best hemisphere."""
    name = "V4_DeltaPowerRatio"
    description = "Power in 0.5-3.5Hz / power in 0.5-40Hz on best hemisphere top-3 channels"

    def _analyze(self, seg_bi):
        # Use UNFILTERED signal for this method
        ch_power = np.array([np.var(seg_bi[ch]) for ch in range(18)])
        left_top3_p = np.mean(np.sort(ch_power[LEFT_CHS])[::-1][:3])
        right_top3_p = np.mean(np.sort(ch_power[RIGHT_CHS])[::-1][:3])

        if left_top3_p > right_top3_p:
            best_chs = LEFT_CHS[np.argsort(ch_power[LEFT_CHS])[::-1][:3]]
        else:
            best_chs = RIGHT_CHS[np.argsort(ch_power[RIGHT_CHS])[::-1][:3]]

        best_sig = np.mean(np.abs(seg_bi[best_chs]), axis=0)

        # Compute PSD
        f, pxx = welch(best_sig, fs=FS, nperseg=400)

        # Delta band: 0.5-3.5 Hz
        delta_mask = (f >= 0.5) & (f <= 3.5)
        # Total band: 0.5-40 Hz
        total_mask = (f >= 0.5) & (f <= 40.0)

        delta_power = pxx[delta_mask].sum()
        total_power = pxx[total_mask].sum()

        if total_power < 1e-12:
            return {'freq': np.nan, 'q_score': 0.0}

        delta_ratio = float(delta_power / total_power)
        q = delta_ratio  # directly use ratio as quality

        # Spectral peak in delta band for freq estimate
        if delta_mask.any() and pxx[delta_mask].sum() > 0:
            freq = float(f[delta_mask][np.argmax(pxx[delta_mask])])
        else:
            freq = np.nan

        return {
            'freq': freq,
            'q_score': float(np.clip(q, 0.0, 1.0)),
            'extras': {'delta_ratio': delta_ratio},
        }
