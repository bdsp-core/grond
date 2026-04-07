"""Base class for spatial localization contest methods."""
import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import butter, sosfiltfilt

FS = 200

# 8 canonical brain regions and the bipolar channels that map to each
REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']

# Bipolar channel index -> region mapping
# Channels: 0=Fp1-F7, 1=F7-T3, 2=T3-T5, 3=T5-O1
#           4=Fp2-F8, 5=F8-T4, 6=T4-T6, 7=T6-O2
#           8=Fp1-F3, 9=F3-C3, 10=C3-P3, 11=P3-O1
#           12=Fp2-F4, 13=F4-C4, 14=C4-P4, 15=P4-O2
#           16=Fz-Cz, 17=Cz-Pz
CHANNEL_TO_REGIONS = {
    0: ['LF', 'LT'],   # Fp1-F7: left frontal + left temporal
    1: ['LF', 'LT'],   # F7-T3: left frontal + left temporal
    2: ['LT'],          # T3-T5: left temporal
    3: ['LT', 'LO'],   # T5-O1: left temporal + left occipital
    4: ['RF', 'RT'],    # Fp2-F8
    5: ['RF', 'RT'],    # F8-T4
    6: ['RT'],          # T4-T6
    7: ['RT', 'RO'],    # T6-O2
    8: ['LF'],          # Fp1-F3: left frontal
    9: ['LF', 'LCP'],   # F3-C3: left frontal + left central-parietal
    10: ['LCP'],         # C3-P3: left central-parietal
    11: ['LCP', 'LO'],  # P3-O1: left central-parietal + left occipital
    12: ['RF'],          # Fp2-F4
    13: ['RF', 'RCP'],   # F4-C4
    14: ['RCP'],         # C4-P4
    15: ['RCP', 'RO'],   # P4-O2
    16: [],              # Fz-Cz: midline (no canonical region)
    17: [],              # Cz-Pz: midline
}

# Inverse: region -> channel indices
REGION_TO_CHANNELS = {}
for ch, regs in CHANNEL_TO_REGIONS.items():
    for r in regs:
        REGION_TO_CHANNELS.setdefault(r, []).append(ch)

LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])

BIPOLAR_NAMES = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


class SpatialMethod(ABC):
    """Base class for spatial localization methods.

    Each method must produce:
        region_scores: dict mapping region name -> involvement score [0, 1]
        spatial_extent: float in [0, 1] (fraction of regions involved)
    """
    name: str = "unnamed"
    description: str = ""

    def analyze(self, seg_bi: np.ndarray, subtype: str = 'gpd') -> dict:
        """Process a single (18, 2000) bipolar segment.

        Returns dict with:
            'region_scores': {region: float} for each of 8 regions
            'spatial_extent': float [0,1]
            'involved_regions': list of region names predicted involved
        """
        try:
            result = self._analyze(seg_bi, subtype)
            scores = result.get('region_scores', {})
            # Ensure all 8 regions present
            for r in REGIONS:
                if r not in scores:
                    scores[r] = 0.0
                scores[r] = float(np.clip(scores[r], 0.0, 1.0))
            threshold = result.get('threshold', 0.5)
            involved = [r for r in REGIONS if scores[r] >= threshold]
            se = len(involved) / len(REGIONS)
            return {
                'region_scores': scores,
                'spatial_extent': se,
                'involved_regions': involved,
                'threshold': threshold,
            }
        except Exception as e:
            return {
                'region_scores': {r: 0.0 for r in REGIONS},
                'spatial_extent': 0.0,
                'involved_regions': [],
                'threshold': 0.5,
                'error': str(e),
            }

    @abstractmethod
    def _analyze(self, seg_bi: np.ndarray, subtype: str) -> dict:
        """Implement in subclass. Return dict with 'region_scores'."""
        pass

    @staticmethod
    def prefilter(seg: np.ndarray, lo=0.3, hi=15.0) -> np.ndarray:
        """Bandpass filter for PD-focused analysis."""
        sos = butter(4, [lo / (FS/2), hi / (FS/2)], btype='bandpass', output='sos')
        return sosfiltfilt(sos, seg, axis=1)

    @staticmethod
    def channel_scores_to_regions(channel_scores):
        """Convert per-channel scores (18,) to per-region scores (8 regions).

        For each region, take the max score across its contributing channels.
        """
        region_scores = {}
        for region in REGIONS:
            chs = REGION_TO_CHANNELS.get(region, [])
            if chs:
                region_scores[region] = float(np.max([channel_scores[ch] for ch in chs]))
            else:
                region_scores[region] = 0.0
        return region_scores
