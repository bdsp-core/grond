"""Baseline and reference methods (B1-B4)."""
import numpy as np
from .base import SpatialMethod, FS, REGIONS, LEFT_CHS, RIGHT_CHS


class B1_AllRegions(SpatialMethod):
    """Predict all 8 regions are involved (upper bound on recall)."""
    name = "B1_AllRegions"
    description = "Always predict all 8 regions involved"

    def _analyze(self, seg_bi, subtype):
        return {'region_scores': {r: 1.0 for r in REGIONS}, 'threshold': 0.5}


class B2_SubtypeDefault(SpatialMethod):
    """GPD: all regions. LPD: 4 regions on one side based on power."""
    name = "B2_SubtypeDefault"
    description = "GPD=all, LPD=ipsilateral 4 regions by power"

    def _analyze(self, seg_bi, subtype):
        if subtype == 'gpd':
            return {'region_scores': {r: 1.0 for r in REGIONS}, 'threshold': 0.5}

        # LPD: check which hemisphere has more power
        left_power = np.mean([np.var(seg_bi[ch]) for ch in LEFT_CHS])
        right_power = np.mean([np.var(seg_bi[ch]) for ch in RIGHT_CHS])

        scores = {}
        if left_power >= right_power:
            for r in REGIONS:
                scores[r] = 1.0 if r.startswith('L') else 0.0
        else:
            for r in REGIONS:
                scores[r] = 1.0 if r.startswith('R') else 0.0
        return {'region_scores': scores, 'threshold': 0.5}


class B3_RMSThreshold(SpatialMethod):
    """Simple RMS amplitude threshold per channel."""
    name = "B3_RMSThreshold"
    description = "Per-channel RMS > median RMS = involved"

    def _analyze(self, seg_bi, subtype):
        n_ch = min(18, seg_bi.shape[0])
        rms = np.array([np.sqrt(np.mean(seg_bi[ch]**2)) for ch in range(n_ch)])
        mx = rms.max()
        if mx > 0:
            scores = rms / mx
        else:
            scores = np.zeros(n_ch)
        return {'region_scores': self.channel_scores_to_regions(scores), 'threshold': 0.4}


class B4_RandomBaseline(SpatialMethod):
    """Random predictions (sanity check — should score near 0)."""
    name = "B4_RandomBaseline"
    description = "Random region scores (sanity check baseline)"

    def _analyze(self, seg_bi, subtype):
        rng = np.random.RandomState(hash(seg_bi.tobytes()) % (2**31))
        scores = {r: float(rng.random()) for r in REGIONS}
        return {'region_scores': scores, 'threshold': 0.5}
