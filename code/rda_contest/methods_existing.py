"""Six existing/baseline RDA analysis methods."""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.ndimage import uniform_filter1d

from .base import RDAMethod, LEFT_CHS, RIGHT_CHS, FREQ_GRID, FS


# ---------------------------------------------------------------------------
# E1 — Variance Explained search
# ---------------------------------------------------------------------------
class E1_VESearch(RDAMethod):
    name = "E1_VESearch"
    description = "Variance-explained sweep across FREQ_GRID with laterality-aware top-3 scoring."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch = seg.shape[0]
        var_orig = np.var(seg, axis=1)  # (n_ch,)
        var_orig = np.where(var_orig > 0, var_orig, 1e-30)

        best_score = -1.0
        best_freq = np.nan

        for f in FREQ_GRID:
            filt = self.narrowband(seg, f)
            ve = np.var(filt, axis=1) / var_orig  # (n_ch,)

            # Top-3 mean per hemisphere
            left_ve = np.sort(ve[LEFT_CHS[LEFT_CHS < n_ch]])[::-1]
            right_ve = np.sort(ve[RIGHT_CHS[RIGHT_CHS < n_ch]])[::-1]
            left_score = np.mean(left_ve[:3]) if len(left_ve) >= 3 else np.mean(left_ve) if len(left_ve) > 0 else 0.0
            right_score = np.mean(right_ve[:3]) if len(right_ve) >= 3 else np.mean(right_ve) if len(right_ve) > 0 else 0.0
            score = max(left_score, right_score)

            if score > best_score:
                best_score = score
                best_freq = f

        if best_score <= 0:
            return {'freq': np.nan, 'q_score': 0.0}
        return {'freq': float(best_freq), 'q_score': float(np.clip(best_score, 0, 1))}


# ---------------------------------------------------------------------------
# E2 — FFT Peak
# ---------------------------------------------------------------------------
class E2_FFTPeak(RDAMethod):
    name = "E2_FFTPeak"
    description = "FFT peak detection: peak-to-background ratio in 0.5–3.5 Hz band."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch, n_samp = seg.shape

        # Welch PSD per channel, average across channels
        freqs, pxx = welch(seg, fs=FS, nperseg=min(1024, n_samp), axis=1)
        pxx_mean = np.mean(pxx, axis=0)  # (n_freqs,)

        # Restrict to 0.5–3.5 Hz
        mask = (freqs >= 0.5) & (freqs <= 3.5)
        if not np.any(mask):
            return {'freq': np.nan, 'q_score': 0.0}

        freqs_band = freqs[mask]
        pxx_band = pxx_mean[mask]

        peak_idx = np.argmax(pxx_band)
        peak_freq = freqs_band[peak_idx]
        peak_power = pxx_band[peak_idx]
        mean_power = np.mean(pxx_band)

        if mean_power <= 0:
            return {'freq': np.nan, 'q_score': 0.0}

        ratio = peak_power / mean_power
        # Sigmoid-style normalization: map ratio to [0, 1]
        # ratio of 1 means flat spectrum (q~0), ratio > 5 is strong peak (q~1)
        q = 1.0 / (1.0 + np.exp(-0.8 * (ratio - 3.0)))

        return {'freq': float(peak_freq), 'q_score': float(q)}


# ---------------------------------------------------------------------------
# E3 — NVO Dual Band (delta + theta nuisance)
# ---------------------------------------------------------------------------
class E3_NVODualBand(RDAMethod):
    name = "E3_NVODualBand"
    description = "Narrowband VE with theta nuisance removal, 3-s sliding window, laterality-aware."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch, n_samp = seg.shape

        # Theta nuisance band (use broader prefilter first, then narrowband)
        seg_raw = seg_bi  # unfiltered for theta
        try:
            sos_theta = butter(4, [6.0 / (FS / 2), 8.0 / (FS / 2)], btype='bandpass', output='sos')
            theta = sosfiltfilt(sos_theta, seg_raw, axis=1)
        except Exception:
            theta = np.zeros_like(seg)

        var_orig = np.var(seg, axis=1)
        var_orig = np.where(var_orig > 0, var_orig, 1e-30)

        # VE from theta alone
        ve_theta = np.var(theta, axis=1) / var_orig  # per channel

        win_samp = int(3.0 * FS)  # 3-second window
        step = win_samp // 2  # 50% overlap

        best_score = -1.0
        best_freq = np.nan

        for f in FREQ_GRID:
            delta_filt = self.narrowband(seg, f)

            # Sliding window partial VE
            window_scores = []
            for start in range(0, max(1, n_samp - win_samp + 1), step):
                end = start + win_samp
                if end > n_samp:
                    break
                seg_win = seg[:, start:end]
                delta_win = delta_filt[:, start:end]
                theta_win = theta[:, start:end]

                var_win = np.var(seg_win, axis=1)
                var_win = np.where(var_win > 0, var_win, 1e-30)
                ve_full = (np.var(delta_win, axis=1) + np.var(theta_win, axis=1)) / var_win
                ve_theta_only = np.var(theta_win, axis=1) / var_win
                partial_ve = np.clip(ve_full - ve_theta_only, 0, None)

                # Top-3 per hemisphere
                left_pve = np.sort(partial_ve[LEFT_CHS[LEFT_CHS < n_ch]])[::-1]
                right_pve = np.sort(partial_ve[RIGHT_CHS[RIGHT_CHS < n_ch]])[::-1]
                l_score = np.mean(left_pve[:3]) if len(left_pve) >= 3 else (np.mean(left_pve) if len(left_pve) > 0 else 0.0)
                r_score = np.mean(right_pve[:3]) if len(right_pve) >= 3 else (np.mean(right_pve) if len(right_pve) > 0 else 0.0)
                window_scores.append(max(l_score, r_score))

            if window_scores:
                score = np.max(window_scores)
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_freq = f

        if best_score <= 0:
            return {'freq': np.nan, 'q_score': 0.0}
        return {'freq': float(best_freq), 'q_score': float(np.clip(best_score, 0, 1))}


# ---------------------------------------------------------------------------
# E4 — Spectral Concentration
# ---------------------------------------------------------------------------
class E4_SpectralConcentration(RDAMethod):
    name = "E4_SpectralConcentration"
    description = "Spectral concentration: C_delta * quality factor Q."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch, n_samp = seg.shape

        freqs, pxx = welch(seg, fs=FS, nperseg=min(1024, n_samp), axis=1)
        pxx_mean = np.mean(pxx, axis=0)

        # Delta band mask 0.5–3.5 Hz
        delta_mask = (freqs >= 0.5) & (freqs <= 3.5)
        if not np.any(delta_mask):
            return {'freq': np.nan, 'q_score': 0.0}

        total_power = np.sum(pxx_mean[freqs > 0])
        if total_power <= 0:
            return {'freq': np.nan, 'q_score': 0.0}

        delta_power = np.sum(pxx_mean[delta_mask])
        C_delta = delta_power / total_power

        # Peak frequency
        freqs_delta = freqs[delta_mask]
        pxx_delta = pxx_mean[delta_mask]
        peak_idx = np.argmax(pxx_delta)
        f_peak = freqs_delta[peak_idx]
        peak_val = pxx_delta[peak_idx]

        # 3 dB bandwidth: find where power drops below peak/2
        half_power = peak_val / 2.0
        above = pxx_delta >= half_power
        if np.sum(above) < 2:
            bandwidth_3dB = freqs_delta[1] - freqs_delta[0]  # minimal resolution
        else:
            indices = np.where(above)[0]
            bandwidth_3dB = freqs_delta[indices[-1]] - freqs_delta[indices[0]]
            if bandwidth_3dB <= 0:
                bandwidth_3dB = freqs_delta[1] - freqs_delta[0]

        Q = f_peak / bandwidth_3dB
        q_score = C_delta * min(Q / 10.0, 1.0)

        return {'freq': float(f_peak), 'q_score': float(np.clip(q_score, 0, 1))}


# ---------------------------------------------------------------------------
# E5 — FOOOF spectral parameterization
# ---------------------------------------------------------------------------
class E5_FOOOF(RDAMethod):
    name = "E5_FOOOF"
    description = "FOOOF 1/f + peaks model; delta peak SNR as quality."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        from fooof import FOOOF

        seg = self.prefilter(seg_bi)
        n_ch, n_samp = seg.shape

        freqs, pxx = welch(seg, fs=FS, nperseg=min(1024, n_samp), axis=1)
        pxx_mean = np.mean(pxx, axis=0)

        # FOOOF requires power > 0
        pxx_mean = np.where(pxx_mean > 0, pxx_mean, 1e-30)

        fm = FOOOF(
            peak_width_limits=[0.3, 2.0],
            max_n_peaks=6,
            min_peak_height=0.05,
            peak_threshold=1.5,
            verbose=False,
        )
        fm.fit(freqs, pxx_mean, freq_range=[0.5, 4.0])

        peaks = fm.peak_params_  # (n_peaks, 3): center_freq, power, bandwidth
        if peaks is None or len(peaks) == 0:
            return {'freq': np.nan, 'q_score': 0.0}

        # Filter to delta range
        delta_peaks = peaks[(peaks[:, 0] >= 0.5) & (peaks[:, 0] <= 3.5)]
        if len(delta_peaks) == 0:
            return {'freq': np.nan, 'q_score': 0.0}

        # Strongest delta peak by power (column 1)
        best_idx = np.argmax(delta_peaks[:, 1])
        peak_freq = delta_peaks[best_idx, 0]
        peak_power = delta_peaks[best_idx, 1]  # in log space, above aperiodic

        # peak_power is already the SNR (power above 1/f fit in log space)
        # Normalize: peak_power of ~1 is decent, ~3+ is very strong
        q_score = 1.0 / (1.0 + np.exp(-1.5 * (peak_power - 1.0)))

        return {'freq': float(peak_freq), 'q_score': float(np.clip(q_score, 0, 1))}


# ---------------------------------------------------------------------------
# E6 — ACF Peak
# ---------------------------------------------------------------------------
class E6_ACFPeak(RDAMethod):
    name = "E6_ACFPeak"
    description = "Autocorrelation peak detection on mean absolute amplitude."

    def _analyze(self, seg_bi: np.ndarray) -> dict:
        seg = self.prefilter(seg_bi)
        n_ch, n_samp = seg.shape

        # Mean absolute amplitude across channels
        mean_abs = np.mean(np.abs(seg), axis=0)  # (n_samp,)

        # Zero-mean
        mean_abs = mean_abs - np.mean(mean_abs)
        norm = np.dot(mean_abs, mean_abs)
        if norm <= 0:
            return {'freq': np.nan, 'q_score': 0.0}

        # Full autocorrelation via FFT
        n_fft = 2 * n_samp
        fft_x = np.fft.rfft(mean_abs, n=n_fft)
        acf_full = np.fft.irfft(fft_x * np.conj(fft_x), n=n_fft)[:n_samp]
        acf = acf_full / norm  # normalize so acf[0]=1

        # Lag range for 0.5–3.5 Hz
        lag_min = int(FS / 3.5)   # ~57 samples
        lag_max = int(FS / 0.5)   # 400 samples
        lag_max = min(lag_max, n_samp - 1)

        if lag_min >= lag_max:
            return {'freq': np.nan, 'q_score': 0.0}

        acf_segment = acf[lag_min:lag_max + 1]
        if len(acf_segment) == 0:
            return {'freq': np.nan, 'q_score': 0.0}

        peak_idx_local = np.argmax(acf_segment)
        peak_lag = lag_min + peak_idx_local
        peak_height = acf_segment[peak_idx_local]

        if peak_lag <= 0 or peak_height <= 0:
            return {'freq': np.nan, 'q_score': 0.0}

        freq = FS / peak_lag

        return {'freq': float(freq), 'q_score': float(np.clip(peak_height, 0, 1))}
