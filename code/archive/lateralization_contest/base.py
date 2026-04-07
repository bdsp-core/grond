"""Base class for lateralization contest methods.

Every method processes each hemisphere independently and returns:
    left_score: float [0, 1] — strength of RDA on left hemisphere
    right_score: float [0, 1] — strength of RDA on right hemisphere

From these, we derive:
    laterality_index = (right_score - left_score) / (right_score + left_score + 1e-12)
        Negative → left-dominant, Positive → right-dominant, ~0 → bilateral

Design constraint: if hemispheres are swapped, the prediction must flip.
"""
import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import butter, sosfiltfilt

FS = 200

LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])

BIPOLAR_NAMES = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


class LateralMethod(ABC):
    """Base class for lateralization methods.

    Each method MUST process hemispheres independently. The analyze() wrapper
    ensures output is properly formatted and clipped.
    """
    name: str = "unnamed"
    description: str = ""

    def analyze(self, seg_bi: np.ndarray) -> dict:
        """Process a (18, 2000) bipolar segment.

        Returns dict with 'left_score', 'right_score', 'laterality_index'.
        """
        try:
            result = self._analyze(seg_bi)
            ls = float(np.clip(result.get('left_score', 0.0), 0.0, 1.0))
            rs = float(np.clip(result.get('right_score', 0.0), 0.0, 1.0))
            if not np.isfinite(ls):
                ls = 0.0
            if not np.isfinite(rs):
                rs = 0.0
            denom = ls + rs + 1e-12
            lat_idx = (rs - ls) / denom
            return {
                'left_score': ls,
                'right_score': rs,
                'laterality_index': float(lat_idx),
                'asymmetry': float(abs(ls - rs) / denom),
                'extras': result.get('extras', {}),
            }
        except Exception as e:
            return {
                'left_score': 0.0,
                'right_score': 0.0,
                'laterality_index': 0.0,
                'asymmetry': 0.0,
                'extras': {'error': str(e)},
            }

    @abstractmethod
    def _analyze(self, seg_bi: np.ndarray) -> dict:
        """Implement in subclass. Must return dict with 'left_score', 'right_score'."""
        pass

    @staticmethod
    def prefilter(seg: np.ndarray, lo=0.3, hi=5.0) -> np.ndarray:
        """Bandpass filter for delta-focused analysis."""
        sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        return sosfiltfilt(sos, seg, axis=1)

    @staticmethod
    def narrowband(seg: np.ndarray, freq: float, bw: float = 0.3) -> np.ndarray:
        """Narrowband filter at freq ± bw Hz."""
        lo = max(freq - bw, 0.1)
        hi = min(freq + bw, FS / 2 - 0.1)
        if lo >= hi:
            return np.zeros_like(seg)
        sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        return sosfiltfilt(sos, seg, axis=1)

    @staticmethod
    def hemi_mean(seg: np.ndarray, chs: np.ndarray, top_k: int = 4) -> np.ndarray:
        """Mean of top-k channels by power on a hemisphere."""
        powers = np.array([np.var(seg[ch]) for ch in chs])
        top_idx = chs[np.argsort(powers)[::-1][:top_k]]
        return np.mean(seg[top_idx], axis=0)

    @staticmethod
    def score_hemisphere(seg: np.ndarray, chs: np.ndarray, func, top_k: int = 4, **kwargs) -> float:
        """Apply a scoring function to the mean of top-k channels on a hemisphere."""
        powers = np.array([np.var(seg[ch]) for ch in chs])
        top_idx = chs[np.argsort(powers)[::-1][:top_k]]
        sig = np.mean(seg[top_idx], axis=0)
        return func(sig, **kwargs)
