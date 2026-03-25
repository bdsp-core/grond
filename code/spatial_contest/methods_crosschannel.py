"""Cross-channel spatial localization methods (X1-X6)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, coherence, welch
from scipy.ndimage import gaussian_filter1d
from .base import SpatialMethod, FS, REGIONS, REGION_TO_CHANNELS, LEFT_CHS, RIGHT_CHS


class X1_CoherenceNetwork(SpatialMethod):
    """Mean coherence of each channel with all others in PD band."""
    name = "X1_CoherenceNetwork"
    description = "Avg coherence with other channels in 0.5-3.5 Hz"

    def _analyze(self, seg_bi, subtype):
        n_ch = min(18, seg_bi.shape[0])
        coh_scores = np.zeros(n_ch)
        nperseg = min(400, seg_bi.shape[1])

        for ch in range(n_ch):
            coh_sum = 0.0
            n_pairs = 0
            for ch2 in range(n_ch):
                if ch2 == ch:
                    continue
                f, cxy = coherence(seg_bi[ch], seg_bi[ch2], FS, nperseg=nperseg)
                mask = (f >= 0.5) & (f <= 3.5)
                if np.any(mask):
                    coh_sum += float(np.mean(cxy[mask]))
                    n_pairs += 1
            if n_pairs > 0:
                coh_scores[ch] = coh_sum / n_pairs

        mx = coh_scores.max()
        if mx > 0:
            coh_scores = coh_scores / mx
        return {'region_scores': self.channel_scores_to_regions(coh_scores), 'threshold': 0.4}


class X2_CrossCorrPeak(SpatialMethod):
    """Peak cross-correlation with the most active channel."""
    name = "X2_CrossCorrPeak"
    description = "Cross-corr peak with highest-power channel in PD band"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=3.5)
        n_ch = min(18, seg_filt.shape[0])

        # Find reference: channel with most power
        powers = np.array([np.var(seg_filt[ch]) for ch in range(n_ch)])
        ref_ch = int(np.argmax(powers))
        ref = seg_filt[ref_ch]
        ref_norm = ref / (np.linalg.norm(ref) + 1e-12)

        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            sig = seg_filt[ch]
            sig_norm = sig / (np.linalg.norm(sig) + 1e-12)
            cc = np.correlate(ref_norm, sig_norm, mode='full')
            scores[ch] = float(np.max(np.abs(cc)))

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class X3_PeakSynchrony(SpatialMethod):
    """Synchrony of pointiness peaks across channels."""
    name = "X3_PeakSynchrony"
    description = "Fraction of peaks synchronous (within 25ms) with other channels"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        n_ch = min(18, seg_bi.shape[0])
        sync_tol = int(0.025 * FS)  # 25ms

        # Find peaks per channel
        all_peaks = []
        for ch in range(n_ch):
            pt = compute_pointiness_trace(seg_bi[ch])
            pt_smooth = gaussian_filter1d(pt, sigma=4)
            threshold = np.percentile(pt_smooth, 75)
            peaks, _ = find_peaks(pt_smooth, height=threshold, distance=int(0.15 * FS))
            all_peaks.append(peaks)

        from scipy.signal import find_peaks

        # For each channel, count how many of its peaks are synchronous with >=2 other channels
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            if len(all_peaks[ch]) == 0:
                continue
            sync_count = 0
            for pk in all_peaks[ch]:
                n_sync = 0
                for ch2 in range(n_ch):
                    if ch2 == ch or len(all_peaks[ch2]) == 0:
                        continue
                    if np.min(np.abs(all_peaks[ch2] - pk)) <= sync_tol:
                        n_sync += 1
                if n_sync >= 2:
                    sync_count += 1
            scores[ch] = sync_count / len(all_peaks[ch])

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.3}


class X4_TemplateCorrelation(SpatialMethod):
    """Build average waveform template from best channel, correlate with all."""
    name = "X4_TemplateCorrelation"
    description = "Template from best channel, NCC score on all channels"

    def _analyze(self, seg_bi, subtype):
        from pd_pointiness_acf import compute_pointiness_trace
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=15.0)
        n_ch = min(18, seg_filt.shape[0])

        # Find best channel by pointiness
        pt_scores = np.zeros(n_ch)
        for ch in range(n_ch):
            pt = compute_pointiness_trace(seg_filt[ch])
            pt_scores[ch] = np.mean(pt)
        best_ch = int(np.argmax(pt_scores))

        # Build template from best channel: find peaks, average snippets
        ref = seg_filt[best_ch]
        pt = compute_pointiness_trace(ref)
        pt_smooth = gaussian_filter1d(pt, sigma=4)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(pt_smooth, height=np.percentile(pt_smooth, 70),
                              distance=int(0.2 * FS))

        half_win = int(0.15 * FS)  # 150ms each side
        if len(peaks) < 3:
            # Fallback: use power
            scores = np.array([np.var(seg_filt[ch]) for ch in range(n_ch)])
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.3}

        # Average template on best channel
        snippets = []
        for pk in peaks:
            if pk - half_win >= 0 and pk + half_win < len(ref):
                snippets.append(ref[pk - half_win:pk + half_win])
        if len(snippets) < 2:
            scores = np.array([np.var(seg_filt[ch]) for ch in range(n_ch)])
            mx = scores.max()
            if mx > 0:
                scores = scores / mx
            return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.3}

        template = np.mean(snippets, axis=0)
        template = template - np.mean(template)
        t_norm = np.linalg.norm(template)
        if t_norm < 1e-12:
            return {'region_scores': {r: 0.0 for r in REGIONS}, 'threshold': 0.5}

        # NCC of template with each channel
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            sig = seg_filt[ch]
            # Sliding NCC
            best_ncc = 0.0
            for start in range(0, len(sig) - len(template), int(0.05 * FS)):
                snippet = sig[start:start + len(template)]
                snippet = snippet - np.mean(snippet)
                s_norm = np.linalg.norm(snippet)
                if s_norm > 1e-12:
                    ncc = float(np.dot(template, snippet) / (t_norm * s_norm))
                    best_ncc = max(best_ncc, ncc)
            scores[ch] = best_ncc

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}


class X5_PhaseCoherence(SpatialMethod):
    """Phase-locking value between each channel and the reference."""
    name = "X5_PhaseCoherence"
    description = "PLV with highest-power channel in PD band"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.5, hi=3.5)
        n_ch = min(18, seg_filt.shape[0])

        # Reference = highest power channel
        powers = np.array([np.var(seg_filt[ch]) for ch in range(n_ch)])
        ref_ch = int(np.argmax(powers))

        # Analytic signal
        ref_phase = np.angle(np.array([complex(0, 1)] * 1 + list(np.zeros(1)))[0] +
                             seg_filt[ref_ch])
        from scipy.signal import hilbert
        ref_analytic = hilbert(seg_filt[ref_ch])
        ref_phase = np.angle(ref_analytic)

        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            ch_analytic = hilbert(seg_filt[ch])
            ch_phase = np.angle(ch_analytic)
            phase_diff = ch_phase - ref_phase
            plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
            scores[ch] = plv

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


class X6_MutualInfoPDBand(SpatialMethod):
    """Simplified mutual information in PD band between channels and reference."""
    name = "X6_MutualInfoPDBand"
    description = "Binned MI with best channel in 0.3-3.5 Hz band"

    def _analyze(self, seg_bi, subtype):
        seg_filt = self.prefilter(seg_bi, lo=0.3, hi=3.5)
        n_ch = min(18, seg_filt.shape[0])

        powers = np.array([np.var(seg_filt[ch]) for ch in range(n_ch)])
        ref_ch = int(np.argmax(powers))
        ref = seg_filt[ref_ch]

        n_bins = 20
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            sig = seg_filt[ch]
            # 2D histogram
            c_xy, _, _ = np.histogram2d(ref, sig, bins=n_bins)
            c_xy = c_xy / c_xy.sum()
            c_x = c_xy.sum(axis=1)
            c_y = c_xy.sum(axis=0)
            # MI
            mi = 0.0
            for i in range(n_bins):
                for j in range(n_bins):
                    if c_xy[i, j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                        mi += c_xy[i, j] * np.log2(c_xy[i, j] / (c_x[i] * c_y[j]))
            scores[ch] = mi

        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.35}
