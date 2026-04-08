"""
PD-Profiler: Unified PD Characterization Pipeline

Given a 10-second EEG segment labeled as LPD or GPD, this pipeline produces:
  1. Lateralization (LPD only): left vs right (AUC 0.989)
  2. Spatial localization: which brain regions are involved (Hybrid CNN+PLV)
  3. Discharge timing: when each discharge occurs (HemiCET-UNet + DP with
     CNN-weighted evidence)
  4. Frequency: median 1/IPI (Hz)

Architecture:
  ChannelPDNetAttention (per-channel CNN)
    → channel PD probabilities (18 values)
    → laterality: compare left vs right hemisphere mean probs
    → evidence weighting: weight channels by PD probability before aggregation
    → spatial localization: CNN picks reference channels, PLV finds connected regions

  HemiCET-UNet (8-channel U-Net per hemisphere)
    → frame-level discharge evidence trace
    → combined with handcrafted evidence (pointiness + TKEO)
    → product-boosted: max(HPP, CET) + 3×HPP×CET

  DP + EM (dynamic programming + expectation-maximization)
    → approximately-periodic discharge sequence
    → template refinement
    → post-hoc filtering

Usage:
    from pd_profiler import PDProfiler
    profiler = PDProfiler()
    result = profiler.characterize(eeg_18ch, subtype='lpd')
    # result['laterality'] = 'left'
    # result['regions'] = ['LF', 'LT', 'LCP', 'LO']
    # result['region_scores'] = {'LF': 0.82, 'RF': 0.31, ...}
    # result['discharge_times'] = [0.42, 1.18, 1.95, ...]
    # result['frequency'] = 1.31
"""

import numpy as np
import torch
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert

# ── Path setup ──
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

import sys
sys.path.insert(0, str(CODE_DIR))

from discharge_detector import (
    FS, LEFT_INDICES, RIGHT_INDICES,
    compute_channel_evidence, combine_evidence,
    detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter,
    per_channel_times,
)
from pd_channel_detector.channel_cnn import ChannelPDNetAttention

# ── Constants ──
REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']
REGION_TO_CHANNELS = {
    'LF': [0, 8], 'RF': [4, 12],
    'LT': [1, 2], 'RT': [5, 6],
    'LCP': [9, 10], 'RCP': [13, 14],
    'LO': [3, 11], 'RO': [7, 15],
}


class PDProfiler:
    """Unified pipeline for PD characterization (manuscript: PD-Profiler)."""

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self._channel_cnn_models = None
        self._hemi_cet_models = None
        self._cet_models = None

    # ── Model loading ──

    def _load_channel_cnn(self):
        if self._channel_cnn_models is not None:
            return self._channel_cnn_models
        model_dir = PROJECT_DIR / 'data' / 'pd_channel_cache'
        models = []
        for fold in range(5):
            path = model_dir / f'cnn_attn_fold{fold}.pt'
            if path.exists():
                m = ChannelPDNetAttention()
                m.load_state_dict(torch.load(str(path), map_location='cpu',
                                             weights_only=True))
                m.to(self.device)
                m.eval()
                models.append(m)
        self._channel_cnn_models = models
        return models

    def _load_cet_models(self):
        """Load per-channel CET-UNet models for evidence computation."""
        if self._cet_models is not None:
            return self._cet_models
        from discharge_detector import DischargeDetector
        det = DischargeDetector()
        self._cet_models = det.cet_models
        self._cet_compute = det.compute_cet_evidence_channel
        self._freq_estimator = det
        return self._cet_models

    # ── Step 1: Per-channel PD probabilities (ChannelPDNetAttention) ──

    def get_channel_probs(self, segment_18ch):
        """Get per-channel PD probability from CNN ensemble.

        Args:
            segment_18ch: (18, 2000) bipolar EEG

        Returns:
            probs: (18,) array of PD probabilities per channel
        """
        models = self._load_channel_cnn()
        n_ch = min(18, segment_18ch.shape[0])
        probs = np.zeros(n_ch)

        for ch in range(n_ch):
            ch_sig = segment_18ch[ch:ch + 1, :].astype(np.float32)
            std = np.std(ch_sig)
            if std > 1e-8:
                ch_sig = (ch_sig - np.mean(ch_sig)) / std
            x = torch.tensor(ch_sig[np.newaxis, :, :],
                             dtype=torch.float32).to(self.device)
            fold_probs = []
            with torch.no_grad():
                for m in models:
                    out = m(x)
                    p = torch.sigmoid(out[0]).item()
                    fold_probs.append(p)
            probs[ch] = np.mean(fold_probs)

        return probs

    # ── Step 2: Laterality (LPD only) ──

    def detect_laterality(self, channel_probs):
        """Determine laterality from per-channel PD probabilities.

        Compares mean PD probability of left vs right hemisphere channels.

        Returns:
            laterality: 'left' or 'right'
            confidence: absolute difference in hemisphere means
        """
        left_mean = np.mean(channel_probs[LEFT_INDICES])
        right_mean = np.mean(channel_probs[RIGHT_INDICES])
        laterality = 'left' if left_mean > right_mean else 'right'
        confidence = abs(left_mean - right_mean)
        return laterality, confidence

    # ── Step 3: Spatial localization (Hybrid CNN+PLV) ──

    def detect_regions(self, segment_18ch, channel_probs, laterality=None):
        """Identify involved brain regions using Hybrid CNN+PLV.

        CNN picks reference channels (top 3 by PD probability),
        then PLV identifies which other channels are phase-locked.

        For LPD with known laterality, reference channels are restricted
        to the ipsilateral hemisphere so the spatial localizer seeds
        from the correct side.

        Returns:
            region_scores: dict region -> score (0-1)
            involved_regions: list of regions above threshold
        """
        n_ch = min(18, segment_18ch.shape[0])

        # For LPD: restrict reference channels to ipsilateral hemisphere
        if laterality == 'left':
            hemi_probs = channel_probs.copy()
            hemi_probs[RIGHT_INDICES] = 0  # suppress contralateral
            top_chs = np.argsort(hemi_probs)[::-1][:3]
        elif laterality == 'right':
            hemi_probs = channel_probs.copy()
            hemi_probs[LEFT_INDICES] = 0
            top_chs = np.argsort(hemi_probs)[::-1][:3]
        else:
            top_chs = np.argsort(channel_probs)[::-1][:3]

        # Bandpass 0.5-3.5 Hz for PLV
        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)],
                     btype='bandpass', output='sos')
        seg_f = sosfiltfilt(sos, segment_18ch[:n_ch], axis=1)

        # Phase of each channel
        phases = np.zeros_like(seg_f)
        for ch in range(n_ch):
            phases[ch] = np.angle(hilbert(seg_f[ch]))

        # Reference phase from top CNN channels
        ref_phase = np.angle(np.mean(np.exp(1j * phases[top_chs]), axis=0))

        # Combined score: CNN prob + PLV
        scores = np.zeros(n_ch)
        for ch in range(n_ch):
            phase_diff = phases[ch] - ref_phase
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            scores[ch] = channel_probs[ch] * 0.5 + plv * 0.5

        # Map to regions (max score across contributing channels)
        region_scores = {}
        for region, chs in REGION_TO_CHANNELS.items():
            valid_chs = [c for c in chs if c < n_ch]
            if valid_chs:
                region_scores[region] = float(max(scores[c] for c in valid_chs))
            else:
                region_scores[region] = 0.0

        # Threshold at 0.4
        involved = [r for r, s in region_scores.items() if s > 0.4]
        return region_scores, involved

    # ── Step 4: Discharge timing (CNN-weighted HemiCET+DP) ──

    def detect_discharges(self, segment_18ch, subtype, laterality, channel_probs):
        """Detect discharge timing using CNN-weighted evidence aggregation.

        Args:
            segment_18ch: (18, 2000)
            subtype: 'lpd' or 'gpd'
            laterality: 'left', 'right', or None
            channel_probs: (18,) CNN PD probabilities for weighting

        Returns:
            times: list of discharge times in seconds
            frequency: IPI-derived frequency (Hz)
            freq_estimate_input: CNN+ACF frequency estimate used as prior
        """
        self._load_cet_models()
        det = self._freq_estimator
        n_ch = min(18, segment_18ch.shape[0])
        n_samp = segment_18ch.shape[1]

        # Frequency estimate
        cnn_freq = det.estimate_frequency(segment_18ch)
        acf_freq = det.estimate_frequency_acf_multichannel(
            segment_18ch, subtype, laterality)
        if np.isfinite(acf_freq):
            freq_est = 0.8 * cnn_freq + 0.2 * acf_freq
        else:
            freq_est = cnn_freq
        freq_est = float(np.clip(freq_est, 0.3, 3.5))

        # Per-channel evidence
        hpp_all = np.zeros((n_ch, n_samp))
        cet_all = np.zeros((n_ch, n_samp), dtype=np.float32)
        for ch in range(n_ch):
            hpp_all[ch] = compute_channel_evidence(segment_18ch[ch], FS)
            if np.all(np.isfinite(segment_18ch[ch])):
                cet_all[ch] = self._cet_compute(segment_18ch[ch])

        # CNN-weighted aggregation
        weights = np.clip(channel_probs[:n_ch], 0.05, None)

        if subtype == 'gpd':
            ch_idx = np.arange(n_ch)
        elif laterality == 'left':
            ch_idx = LEFT_INDICES
        elif laterality == 'right':
            ch_idx = RIGHT_INDICES
        else:
            # LPD no laterality: weighted max of hemispheres
            lw = weights[LEFT_INDICES]
            rw = weights[RIGHT_INDICES]
            hpp_left = np.average(hpp_all[LEFT_INDICES], weights=lw, axis=0)
            hpp_right = np.average(hpp_all[RIGHT_INDICES], weights=rw, axis=0)
            cet_left = np.average(cet_all[LEFT_INDICES], weights=lw, axis=0)
            cet_right = np.average(cet_all[RIGHT_INDICES], weights=rw, axis=0)
            hpp_agg = np.maximum(hpp_left, hpp_right)
            cet_agg = np.maximum(cet_left, cet_right)
            ch_idx = None

        if ch_idx is not None:
            w = weights[ch_idx]
            hpp_agg = np.average(hpp_all[ch_idx], weights=w, axis=0)
            cet_agg = np.average(cet_all[ch_idx], weights=w, axis=0)

        evidence = combine_evidence(hpp_agg, cet_agg)

        # DP inference
        active_start, active_end = detect_active_interval(evidence, FS)
        candidates = extract_candidates(evidence, FS, freq_est,
                                        active_start, active_end)
        discharge_samples = dp_best_sequence(candidates, evidence, FS, freq_est)
        if len(discharge_samples) >= 3:
            discharge_samples = em_refine(evidence, discharge_samples, FS, freq_est)
        discharge_samples = posthoc_filter(discharge_samples, evidence)

        times = (discharge_samples / FS).tolist() if len(discharge_samples) > 0 else []

        if len(times) >= 2:
            ipi_freq = 1.0 / float(np.median(np.diff(times)))
        else:
            ipi_freq = freq_est

        return times, ipi_freq, freq_est

    # ── Main entry point ──

    def characterize(self, segment_18ch, subtype='lpd'):
        """Full PD characterization pipeline.

        Args:
            segment_18ch: (18, 2000) bipolar EEG at 200 Hz
            subtype: 'lpd' or 'gpd'

        Returns:
            dict with:
                subtype: 'lpd' or 'gpd'
                laterality: 'left'/'right' (LPD) or None (GPD)
                laterality_confidence: float
                regions: list of involved region names
                region_scores: dict region -> score
                discharge_times: list of float (seconds)
                frequency: float (Hz, from IPI)
                freq_estimate_input: float (Hz, CNN+ACF prior)
                n_discharges: int
                channel_probs: (18,) array
        """
        segment_18ch = np.asarray(segment_18ch, dtype=np.float64)
        if segment_18ch.shape[0] > segment_18ch.shape[1]:
            segment_18ch = segment_18ch.T

        # Step 1: Per-channel PD probabilities
        channel_probs = self.get_channel_probs(segment_18ch)

        # Step 2: Laterality (LPD only)
        if subtype == 'lpd':
            laterality, lat_conf = self.detect_laterality(channel_probs)
        else:
            laterality = None
            lat_conf = 0.0

        # Step 3: Spatial localization (uses laterality to seed from correct side)
        region_scores, involved_regions = self.detect_regions(
            segment_18ch, channel_probs, laterality=laterality)

        # Step 4: Discharge timing + frequency
        times, frequency, freq_input = self.detect_discharges(
            segment_18ch, subtype, laterality, channel_probs)

        return {
            'subtype': subtype,
            'laterality': laterality,
            'laterality_confidence': lat_conf,
            'regions': involved_regions,
            'region_scores': region_scores,
            'discharge_times': times,
            'frequency': frequency,
            'freq_estimate_input': freq_input,
            'n_discharges': len(times),
            'channel_probs': channel_probs.tolist(),
        }


if __name__ == '__main__':
    import scipy.io as sio

    print("PD-Profiler — Quick Test")
    print("=" * 50)

    profiler = PDProfiler()

    # Load a test segment
    mat = sio.loadmat('data/eeg/111353221_seg000.mat')
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:18, :2000]

    for subtype in ['lpd', 'gpd']:
        print(f"\n--- {subtype.upper()} ---")
        result = profiler.characterize(seg, subtype=subtype)
        print(f"  Laterality: {result['laterality']} (conf={result['laterality_confidence']:.3f})")
        print(f"  Regions: {result['regions']}")
        print(f"  Region scores: {', '.join(f'{r}={s:.2f}' for r, s in result['region_scores'].items())}")
        print(f"  Discharges: {result['n_discharges']} at freq={result['frequency']:.2f} Hz")
        print(f"  Times: {[f'{t:.2f}' for t in result['discharge_times'][:5]]}{'...' if len(result['discharge_times']) > 5 else ''}")
