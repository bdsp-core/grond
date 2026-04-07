"""Base class for RDA contest methods."""
import numpy as np
from abc import ABC, abstractmethod
from scipy.signal import butter, sosfiltfilt

FS = 200

class RDAMethod(ABC):
    """Base class for RDA analysis methods.

    Each method must produce:
        freq: float — estimated RDA frequency in Hz (0.25-3.5)
        q_score: float — quality/confidence score (0=not RDA, 1=clear RDA)
    """
    name: str = "unnamed"
    description: str = ""

    def analyze(self, seg_bi: np.ndarray) -> dict:
        """Process a single (18, 2000) bipolar segment.

        Returns dict with 'freq', 'q_score', and optional 'extras'.
        """
        try:
            result = self._analyze(seg_bi)
            freq = result.get('freq', np.nan)
            q = result.get('q_score', 0.0)
            # Clamp
            if not np.isfinite(freq):
                freq = np.nan
            if not np.isfinite(q):
                q = 0.0
            q = float(np.clip(q, 0.0, 1.0))
            return {'freq': freq, 'q_score': q, 'extras': result.get('extras', {})}
        except Exception:
            return {'freq': np.nan, 'q_score': 0.0, 'extras': {}}

    @abstractmethod
    def _analyze(self, seg_bi: np.ndarray) -> dict:
        """Implement in subclass."""
        pass

    @staticmethod
    def prefilter(seg: np.ndarray, lo=0.3, hi=5.0) -> np.ndarray:
        """Bandpass filter for delta-focused analysis."""
        sos = butter(4, [lo / (FS/2), hi / (FS/2)], btype='bandpass', output='sos')
        return sosfiltfilt(sos, seg, axis=1)

    @staticmethod
    def narrowband(seg: np.ndarray, freq: float, bw: float = 0.3) -> np.ndarray:
        """Narrowband filter at freq ± bw Hz."""
        lo = max(freq - bw, 0.1)
        hi = min(freq + bw, FS/2 - 0.1)
        if lo >= hi:
            return np.zeros_like(seg)
        sos = butter(4, [lo / (FS/2), hi / (FS/2)], btype='bandpass', output='sos')
        return sosfiltfilt(sos, seg, axis=1)

LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])
FREQ_GRID = np.arange(0.5, 3.55, 0.05)
