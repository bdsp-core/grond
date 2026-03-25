"""Hybrid and advanced spatial localization methods (H1-H8)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks, welch, hilbert
from scipy.ndimage import gaussian_filter1d
from .base import SpatialMethod, FS, REGIONS, REGION_TO_CHANNELS, LEFT_CHS, RIGHT_CHS


class H1_EnsembleVote(SpatialMethod):
    """Ensemble of simple features: pointiness + FFT + ACF. Majority vote."""
    name = "H1_EnsembleVote"
    description = "Majority vote of pointiness, FFT power, and ACF peak"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
        n_ch = min(18, seg_bi.shape[0])

        pt_scores = np.zeros(n_ch)
        fft_scores = np.zeros(n_ch)
        acf_scores = np.zeros(n_ch)

        for ch in range(n_ch):
            x = seg_bi[ch]
            # Pointiness
            pt = compute_pointiness_trace(x)
            pt_scores[ch] = np.max(gaussian_filter1d(pt, sigma=4))
            # FFT
            fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))
            freqs = np.fft.rfftfreq(len(x), d=1.0/FS)
            mask = (freqs >= 0.3) & (freqs <= 3.5)
            if np.any(mask):
                fft_scores[ch] = np.max(fft_vals[mask])
            # ACF
            freq, acf_h, _ = compute_acf_frequency(
                x, FS, method='pointiness', smoothing_sigma=0.02,
                acf_min_lag=0.4, acf_peak_threshold=0.10, peak_height_frac=0.3)
            acf_scores[ch] = acf_h

        # Normalize each
        for arr in [pt_scores, fft_scores, acf_scores]:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        # Ensemble: average
        ensemble = (pt_scores + fft_scores + acf_scores) / 3.0
        return {'region_scores': self.channel_scores_to_regions(ensemble), 'threshold': 0.35}


class H2_WeightedEnsemble(SpatialMethod):
    """Weighted ensemble: 0.4*pointiness + 0.3*FFT + 0.3*ACF."""
    name = "H2_WeightedEnsemble"
    description = "Weighted blend: 0.4*point + 0.3*FFT + 0.3*ACF"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
        n_ch = min(18, seg_bi.shape[0])

        pt_scores = np.zeros(n_ch)
        fft_scores = np.zeros(n_ch)
        acf_scores = np.zeros(n_ch)

        for ch in range(n_ch):
            x = seg_bi[ch]
            pt = compute_pointiness_trace(x)
            pt_scores[ch] = np.max(gaussian_filter1d(pt, sigma=4))
            fft_vals = np.abs(np.fft.rfft(x - np.mean(x)))
            freqs = np.fft.rfftfreq(len(x), d=1.0/FS)
            mask = (freqs >= 0.3) & (freqs <= 3.5)
            if np.any(mask):
                fft_scores[ch] = np.max(fft_vals[mask])
            freq, acf_h, _ = compute_acf_frequency(
                x, FS, method='pointiness', smoothing_sigma=0.02,
                acf_min_lag=0.4, acf_peak_threshold=0.10, peak_height_frac=0.3)
            acf_scores[ch] = acf_h

        for arr in [pt_scores, fft_scores, acf_scores]:
            mx = arr.max()
            if mx > 0:
                arr /= mx

        ensemble = 0.4 * pt_scores + 0.3 * fft_scores + 0.3 * acf_scores
        return {'region_scores': self.channel_scores_to_regions(ensemble), 'threshold': 0.35}


class H3_AdaptiveThreshold(SpatialMethod):
    """Pointiness with subtype-adaptive threshold (lower for GPD)."""
    name = "H3_AdaptiveThreshold"
    description = "Pointiness with GPD=0.25 / LPD=0.4 thresholds"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        n_ch = min(18, seg_bi.shape[0])
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            pt = compute_pointiness_trace(seg_bi[ch])
            scores[ch] = np.max(gaussian_filter1d(pt, sigma=4))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        threshold = 0.25 if subtype == 'gpd' else 0.40
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': threshold}


class H4_PowerRankPercentile(SpatialMethod):
    """Rank channels by PD-band power, select top percentile."""
    name = "H4_PowerRankPercentile"
    description = "Top 60th percentile channels by PD-band power"

    def _analyze(self, seg_bi, subtype):
        n_ch = min(18, seg_bi.shape[0])
        scores = np.zeros(n_ch)
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=3.5)
        for ch in range(n_ch):
            scores[ch] = np.var(seg_filt[ch])
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        # Adaptive: GPD tends to be more widespread
        pctl = 0.30 if subtype == 'gpd' else 0.40
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': pctl}


class H5_SymmetryAware(SpatialMethod):
    """For GPD, enforce bilateral symmetry; for LPD, allow asymmetry."""
    name = "H5_SymmetryAware"
    description = "Pointiness + bilateral symmetry constraint for GPD"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        n_ch = min(18, seg_bi.shape[0])
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            pt = compute_pointiness_trace(seg_bi[ch])
            scores[ch] = np.max(gaussian_filter1d(pt, sigma=4))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx

        region_scores = self.channel_scores_to_regions(scores)

        if subtype == 'gpd':
            # Enforce symmetry: if one side is involved, the mirror is too
            pairs = [('LF', 'RF'), ('LT', 'RT'), ('LCP', 'RCP'), ('LO', 'RO')]
            for r1, r2 in pairs:
                avg = (region_scores[r1] + region_scores[r2]) / 2.0
                region_scores[r1] = avg
                region_scores[r2] = avg

        threshold = 0.30 if subtype == 'gpd' else 0.40
        return {'region_scores': region_scores, 'threshold': threshold}


class H6_TKEO(SpatialMethod):
    """Teager-Kaiser energy operator — highlights sharp transients."""
    name = "H6_TKEO"
    description = "TKEO (Teager-Kaiser) energy per channel"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=15.0)
        n_ch = min(18, seg_filt.shape[0])
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            x = seg_filt[ch]
            # TKEO: x[n]^2 - x[n-1]*x[n+1]
            tkeo = x[1:-1]**2 - x[:-2] * x[2:]
            tkeo = np.maximum(tkeo, 0)
            scores[ch] = float(np.mean(tkeo))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class H7_VEPerChannel(SpatialMethod):
    """Variance explained by best-fit sinusoid per channel in PD range."""
    name = "H7_VEPerChannel"
    description = "Sinusoidal VE sweep (0.5-3.5 Hz) per channel"

    def _analyze(self, seg_bi, subtype):
        n_ch = min(18, seg_bi.shape[0])
        n_samples = seg_bi.shape[1]
        t = np.arange(n_samples) / FS
        scores = np.zeros(n_ch)

        freq_grid = np.arange(0.5, 3.55, 0.1)

        for ch in range(n_ch):
            x = seg_bi[ch] - np.mean(seg_bi[ch])
            total_var = np.var(x)
            if total_var < 1e-12:
                continue
            best_ve = 0.0
            for freq in freq_grid:
                basis = np.column_stack([np.sin(2 * np.pi * freq * t),
                                         np.cos(2 * np.pi * freq * t),
                                         np.ones(n_samples)])
                coef, _, _, _ = np.linalg.lstsq(basis, x, rcond=None)
                residual = x - basis @ coef
                ve = max(0.0, 1.0 - np.var(residual) / total_var)
                best_ve = max(best_ve, ve)
            scores[ch] = best_ve

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class H8_GradientSharpness(SpatialMethod):
    """Kurtosis of signal gradient — high kurtosis = sharp transients."""
    name = "H8_GradientSharpness"
    description = "Kurtosis of first derivative (sharp transient detector)"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=15.0)
        n_ch = min(18, seg_filt.shape[0])
        scores = np.zeros(n_ch)
        from scipy.stats import kurtosis
        for ch in range(n_ch):
            grad = np.diff(seg_filt[ch])
            k = kurtosis(grad, fisher=True)
            scores[ch] = max(0.0, float(k))
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}
