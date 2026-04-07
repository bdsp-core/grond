"""Round 5 contest methods — advanced signal processing approaches.

Methods:
- V5_TFRidgeStability: Time-frequency ridge stability via STFT
- V5_HarmonicStructure: Harmonic power ratios (2f, 3f vs f)
- V5_ICA_DeltaComponent: FastICA dominant delta component
- V5_RecurrenceQuantification: Recurrence plot determinism
- V5_PhaseAmplitudeCoupling: Delta phase modulating higher freq amplitude
- V5_GrangerLaterality: Granger causality asymmetry between hemispheres
- V5_WaveformAsymmetry: Rise/fall time asymmetry of delta cycles
"""
import warnings
import numpy as np
from scipy.signal import stft, welch, hilbert, find_peaks, decimate, butter, sosfiltfilt
from scipy.spatial.distance import cdist

from .base import RDAMethod, FS, LEFT_CHS, RIGHT_CHS
from .methods_v2 import _best_hemi_signal, _spectral_peak_freq


class V5_TFRidgeStability(RDAMethod):
    """Time-frequency ridge stability via short-time Fourier transform."""
    name = "V5_TFRidgeStability"
    description = "STFT ridge stability: fraction of time where dominant freq stays within ±0.3Hz of median"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # STFT: 1s window (200 samples), 0.25s hop (50 samples)
        nperseg = int(1.0 * FS)  # 200
        noverlap = nperseg - int(0.25 * FS)  # 200 - 50 = 150
        f_stft, t_stft, Zxx = stft(best_sig, fs=FS, nperseg=nperseg,
                                     noverlap=noverlap)

        # Restrict to 0.5-3.5 Hz
        freq_mask = (f_stft >= 0.5) & (f_stft <= 3.5)
        if not freq_mask.any():
            return {'freq': np.nan, 'q_score': 0.0}

        f_delta = f_stft[freq_mask]
        power = np.abs(Zxx[freq_mask, :]) ** 2

        if power.shape[1] < 3:
            return {'freq': np.nan, 'q_score': 0.0}

        # Ridge: instantaneous dominant frequency at each time step
        ridge_indices = np.argmax(power, axis=0)
        ridge_freqs = f_delta[ridge_indices]

        median_freq = np.median(ridge_freqs)

        # Fraction of time points within ±0.3 Hz of median
        stable_fraction = np.mean(np.abs(ridge_freqs - median_freq) <= 0.3)

        return {
            'freq': float(median_freq),
            'q_score': float(stable_fraction),
            'extras': {'n_time_points': len(ridge_freqs)}
        }


class V5_HarmonicStructure(RDAMethod):
    """Ratio of power at harmonics 2f and 3f to fundamental f."""
    name = "V5_HarmonicStructure"
    description = "Harmonic ratio: (power_2f + power_3f) / power_f — structured repetition"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi, lo=0.3, hi=12.0)  # wider band to capture harmonics
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Find dominant freq from Welch PSD in delta band
        f_psd, pxx = welch(best_sig, fs=FS, nperseg=400)
        delta_mask = (f_psd >= 0.5) & (f_psd <= 3.5)
        if not delta_mask.any() or pxx[delta_mask].sum() == 0:
            return {'freq': np.nan, 'q_score': 0.0}

        fund_freq = float(f_psd[delta_mask][np.argmax(pxx[delta_mask])])

        # Compute power around f, 2f, 3f using narrow windows
        def _band_power(sig, center, bw=0.3):
            lo = max(center - bw, 0.1)
            hi = min(center + bw, FS / 2 - 0.1)
            if lo >= hi:
                return 0.0
            sos = butter(4, [lo / (FS / 2), hi / (FS / 2)],
                         btype='bandpass', output='sos')
            filtered = sosfiltfilt(sos, sig)
            return float(np.var(filtered))

        # Use the wider-filtered signal for harmonic extraction
        power_f = _band_power(best_sig, fund_freq)
        power_2f = _band_power(best_sig, 2 * fund_freq)
        power_3f = _band_power(best_sig, 3 * fund_freq)

        if power_f < 1e-12:
            return {'freq': fund_freq, 'q_score': 0.0}

        harmonic_ratio = (power_2f + power_3f) / power_f

        # q_score: reward some harmonics but not too many
        q = min(harmonic_ratio * 2, 1.0)

        return {
            'freq': fund_freq,
            'q_score': float(q),
            'extras': {'harmonic_ratio': float(harmonic_ratio),
                       'power_f': power_f, 'power_2f': power_2f,
                       'power_3f': power_3f}
        }


class V5_ICA_DeltaComponent(RDAMethod):
    """FastICA on delta-filtered signal — check if one IC dominates."""
    name = "V5_ICA_DeltaComponent"
    description = "ICA variance explained by top delta component (single coherent source = RDA)"

    def _analyze(self, seg_bi):
        from sklearn.decomposition import FastICA
        from sklearn.exceptions import ConvergenceWarning

        seg_f = self.prefilter(seg_bi)

        n_channels = seg_f.shape[0]
        n_components = min(8, n_channels)

        # Transpose to (samples, channels) for ICA
        X = seg_f.T  # (2000, 18)

        # Check for degenerate input
        if np.std(X) < 1e-10:
            return {'freq': np.nan, 'q_score': 0.0}

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            try:
                ica = FastICA(n_components=n_components,
                              whiten='unit-variance',
                              max_iter=200,
                              random_state=42)
                sources = ica.fit_transform(X)  # (2000, n_components)
            except Exception:
                return {'freq': np.nan, 'q_score': 0.0}

        # Variance explained by each component
        component_vars = np.var(sources, axis=0)
        total_var = component_vars.sum()

        if total_var < 1e-12:
            return {'freq': np.nan, 'q_score': 0.0}

        # Top IC by variance
        top_idx = np.argmax(component_vars)
        var_explained = component_vars[top_idx] / total_var

        # Frequency of top IC
        top_ic = sources[:, top_idx]
        freq, _ = _spectral_peak_freq(top_ic)

        return {
            'freq': freq,
            'q_score': float(var_explained),
            'extras': {'n_components': n_components,
                       'top_ic_var_ratio': float(var_explained)}
        }


class V5_RecurrenceQuantification(RDAMethod):
    """Recurrence plot determinism — periodic signals have diagonal structures."""
    name = "V5_RecurrenceQuantification"
    description = "RQA determinism on downsampled delta signal (high = periodic/structured)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Downsample to 50 Hz (factor 4)
        x = decimate(best_sig, 4)  # ~500 samples
        n = len(x)

        if n < 20:
            return {'freq': np.nan, 'q_score': 0.0}

        # Normalize
        x_std = np.std(x)
        if x_std < 1e-10:
            return {'freq': np.nan, 'q_score': 0.0}

        # Threshold = 0.3 * std
        threshold = 0.3 * x_std

        # Recurrence matrix via cdist
        x_col = x.reshape(-1, 1)
        dist_matrix = cdist(x_col, x_col, metric='cityblock')
        R = (dist_matrix < threshold).astype(np.int8)

        # Total recurrence points (excluding main diagonal)
        np.fill_diagonal(R, 0)
        total_recurrence = R.sum()

        if total_recurrence < 3:
            return {'freq': _spectral_peak_freq(best_sig)[0], 'q_score': 0.0}

        # Determinism: fraction of recurrence points on diagonal lines of length >= 3
        diag_points = 0
        # Check diagonals (both upper and lower)
        for k in range(1, n):
            diag = np.diag(R, k)
            # Find runs of 1s of length >= 3
            if len(diag) < 3:
                continue
            run_len = 0
            for val in diag:
                if val:
                    run_len += 1
                else:
                    if run_len >= 3:
                        diag_points += run_len
                    run_len = 0
            if run_len >= 3:
                diag_points += run_len

        # Count both upper and lower triangles (symmetric matrix)
        determinism = (2 * diag_points) / max(total_recurrence, 1)
        determinism = min(determinism, 1.0)

        freq, _ = _spectral_peak_freq(best_sig)

        return {
            'freq': freq,
            'q_score': float(determinism),
            'extras': {'total_recurrence': int(total_recurrence),
                       'diag_points': int(diag_points)}
        }


class V5_PhaseAmplitudeCoupling(RDAMethod):
    """Delta phase modulating higher frequency amplitude (modulation index)."""
    name = "V5_PhaseAmplitudeCoupling"
    description = "Phase-amplitude coupling: delta phase modulates theta/alpha amplitude"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig_delta, _, _ = _best_hemi_signal(seg_f)

        # Extract delta phase (0.5-3.5 Hz)
        sos_delta = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)],
                           btype='bandpass', output='sos')
        delta_sig = sosfiltfilt(sos_delta, best_sig_delta)
        analytic_delta = hilbert(delta_sig)
        delta_phase = np.angle(analytic_delta)

        # Extract theta/alpha amplitude (4-13 Hz) from unfiltered input
        best_sig_raw, _, _ = _best_hemi_signal(seg_bi)
        sos_high = butter(4, [4.0 / (FS / 2), 13.0 / (FS / 2)],
                          btype='bandpass', output='sos')
        high_sig = sosfiltfilt(sos_high, best_sig_raw)
        high_amp = np.abs(hilbert(high_sig))

        # Modulation index: bin phase into 18 bins, compute KL divergence
        n_bins = 18
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        mean_amp = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (delta_phase >= bin_edges[i]) & (delta_phase < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_amp[i] = np.mean(high_amp[mask])

        # Normalize to probability distribution
        total = mean_amp.sum()
        if total < 1e-12:
            freq, _ = _spectral_peak_freq(best_sig_delta)
            return {'freq': freq, 'q_score': 0.0}

        p = mean_amp / total
        # Uniform distribution
        q_uniform = np.ones(n_bins) / n_bins

        # KL divergence (avoid log(0))
        p_safe = np.clip(p, 1e-12, None)
        kl_div = np.sum(p_safe * np.log(p_safe / q_uniform))

        # Normalize: max possible KL for 18 bins is log(18)
        max_kl = np.log(n_bins)
        mi_normalized = kl_div / max_kl if max_kl > 0 else 0.0
        mi_normalized = float(np.clip(mi_normalized, 0.0, 1.0))

        freq, _ = _spectral_peak_freq(best_sig_delta)

        return {
            'freq': freq,
            'q_score': mi_normalized,
            'extras': {'kl_divergence': float(kl_div),
                       'mean_amp_per_bin': mean_amp.tolist()}
        }


class V5_GrangerLaterality(RDAMethod):
    """Spectral Granger causality asymmetry between hemispheres."""
    name = "V5_GrangerLaterality"
    description = "Granger causality asymmetry: lateralized source detection via VAR model"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)

        # Left and right hemisphere mean signals
        left_sig = np.mean(seg_f[LEFT_CHS], axis=0)
        right_sig = np.mean(seg_f[RIGHT_CHS], axis=0)

        n = len(left_sig)
        p = 5  # VAR order

        if n <= p + 1:
            return {'freq': np.nan, 'q_score': 0.0}

        # Build VAR design matrix
        # Y = [left_t, right_t] for t = p..n-1
        Y = np.column_stack([left_sig[p:], right_sig[p:]])  # (n-p, 2)

        # X = lagged values [left_{t-1}, right_{t-1}, ..., left_{t-p}, right_{t-p}]
        X_parts = []
        for lag in range(1, p + 1):
            X_parts.append(np.column_stack([left_sig[p - lag:n - lag],
                                            right_sig[p - lag:n - lag]]))
        X = np.hstack(X_parts)  # (n-p, 2*p)

        # Full model: both left and right lags predict each variable
        try:
            # Full model residuals
            beta_full, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            resid_full = Y - X @ beta_full
            var_full_left = np.var(resid_full[:, 0])
            var_full_right = np.var(resid_full[:, 1])

            # Restricted model for left (only left lags, no right cross-terms)
            X_left_only = X[:, ::2]  # columns 0, 2, 4, ... = left lags
            beta_r_left, _, _, _ = np.linalg.lstsq(X_left_only, Y[:, 0:1], rcond=None)
            resid_r_left = Y[:, 0:1] - X_left_only @ beta_r_left
            var_restricted_left = np.var(resid_r_left)

            # Restricted model for right (only right lags, no left cross-terms)
            X_right_only = X[:, 1::2]  # columns 1, 3, 5, ... = right lags
            beta_r_right, _, _, _ = np.linalg.lstsq(X_right_only, Y[:, 1:2], rcond=None)
            resid_r_right = Y[:, 1:2] - X_right_only @ beta_r_right
            var_restricted_right = np.var(resid_r_right)
        except (np.linalg.LinAlgError, ValueError):
            return {'freq': np.nan, 'q_score': 0.0}

        # Granger causality: log ratio of restricted vs full residual variance
        # GC_R->L: how much right helps predict left
        gc_r_to_l = np.log(max(var_restricted_left, 1e-12) /
                           max(var_full_left, 1e-12))
        gc_r_to_l = max(gc_r_to_l, 0.0)

        # GC_L->R: how much left helps predict right
        gc_l_to_r = np.log(max(var_restricted_right, 1e-12) /
                           max(var_full_right, 1e-12))
        gc_l_to_r = max(gc_l_to_r, 0.0)

        # Asymmetry measure
        max_gc = max(gc_r_to_l, gc_l_to_r)
        if max_gc < 1e-12:
            q = 0.0
        else:
            q = abs(gc_l_to_r - gc_r_to_l) / max_gc

        # Frequency from whichever hemisphere has stronger GC
        if gc_l_to_r >= gc_r_to_l:
            freq, _ = _spectral_peak_freq(left_sig)
        else:
            freq, _ = _spectral_peak_freq(right_sig)

        return {
            'freq': freq,
            'q_score': float(np.clip(q, 0.0, 1.0)),
            'extras': {'gc_l_to_r': float(gc_l_to_r),
                       'gc_r_to_l': float(gc_r_to_l)}
        }


class V5_WaveformAsymmetry(RDAMethod):
    """Asymmetry of individual delta cycles (rise vs fall time)."""
    name = "V5_WaveformAsymmetry"
    description = "Waveform asymmetry: rise/fall time ratio per cycle (biological vs artifact)"

    def _analyze(self, seg_bi):
        seg_f = self.prefilter(seg_bi)
        best_sig, _, _ = _best_hemi_signal(seg_f)

        # Find peaks and troughs
        min_dist = int(FS / 3.5)  # minimum distance for delta range
        peaks, _ = find_peaks(best_sig, distance=min_dist)
        troughs, _ = find_peaks(-best_sig, distance=min_dist)

        if len(peaks) < 2 or len(troughs) < 2:
            return {'freq': np.nan, 'q_score': 0.0}

        # Build cycles: trough -> peak -> trough
        asymmetries = []
        cycle_durations = []

        for i in range(len(troughs) - 1):
            t1 = troughs[i]
            t2 = troughs[i + 1]

            # Find peak between these two troughs
            mid_peaks = peaks[(peaks > t1) & (peaks < t2)]
            if len(mid_peaks) == 0:
                continue

            pk = mid_peaks[0]
            rise_time = pk - t1
            fall_time = t2 - pk
            total = rise_time + fall_time

            if total < 2:
                continue

            asym = abs(rise_time - fall_time) / total
            asymmetries.append(asym)
            cycle_durations.append(total)

        if len(asymmetries) < 3:
            return {'freq': np.nan, 'q_score': 0.0}

        median_asym = float(np.median(asymmetries))
        median_duration = float(np.median(cycle_durations))

        # q_score peaks around 0.3 asymmetry
        q = 1.0 - abs(median_asym - 0.3) * 3.0
        q = float(np.clip(q, 0.0, 1.0))

        # Frequency from cycle duration
        freq = FS / median_duration if median_duration > 0 else np.nan

        return {
            'freq': freq,
            'q_score': q,
            'extras': {'median_asymmetry': median_asym,
                       'n_cycles': len(asymmetries),
                       'median_duration_samples': median_duration}
        }
