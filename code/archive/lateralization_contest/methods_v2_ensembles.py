"""Tier 4: Ensemble methods (L21-L25).

These combine per-patient scores from single methods.
Must be run AFTER all single methods have completed.
"""
import json
import numpy as np
from pathlib import Path

from .base import LateralMethod

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / 'results' / 'lateralization_contest_v2' / '_cache'

# Methods grouped by tier for ensemble selection
TIER1_METHODS = ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L03_PeakToMeanRatio',
                 'L04_SpectralConcentration', 'L05_TemplateMatch']
TIER2_METHODS = ['L06_ACFPeak', 'L07_SVDDominance', 'L08_IntraHemiPLV',
                 'L09_VarExplained', 'L10_DeltaBandpower']
VE_METHODS = ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L09_VarExplained', 'L19_MatchedFilter']
ALL_SINGLE = TIER1_METHODS + TIER2_METHODS + [f'L{i:02d}_{n}' for i, n in [
    (11, 'WaveletRidge'), (12, 'EnvelopePeakedness'), (13, 'PhaseConsistency'),
    (14, 'CrossChannelCorr'), (15, 'SpectralFlatness'), (16, 'GradientEnergy'),
    (17, 'CepstralPeak'), (18, 'SubbandEntropy'), (19, 'MatchedFilter'),
    (20, 'CoherenceWithTemplate')]]


def _load_scores(method_names):
    """Load per-patient scores from saved cache files."""
    all_scores = {}
    for name in method_names:
        path = CACHE_DIR / f'{name}_scores.json'
        if path.exists():
            with open(path) as f:
                all_scores[name] = json.load(f)
    return all_scores


class _EnsembleBase(LateralMethod):
    """Base for ensemble methods that combine cached single-method scores."""
    method_names = []
    _scores_cache = None

    def _load(self):
        if self._scores_cache is None:
            self.__class__._scores_cache = _load_scores(self.method_names)
        return self._scores_cache

    def _get_asymmetries(self, seg_bi):
        """Not used — ensemble methods override analyze() directly."""
        return {}

    def _analyze(self, seg_bi):
        # Not used for ensembles
        return {'left_score': 0.0, 'right_score': 0.0}


class L21_EnsembleTop3(_EnsembleBase):
    """Mean asymmetry of top 3 Round 1 methods."""
    name = "L21_EnsembleTop3"
    description = "Mean asymmetry of L01, L02, L03"
    method_names = ['L01_NarrowbandVE', 'L02_MultiChannelVE', 'L03_PeakToMeanRatio']

    def analyze(self, seg_bi, patient_id=None):
        # Ensemble methods are called differently — see run_ensembles.py
        return {'left_score': 0, 'right_score': 0, 'laterality_index': 0, 'asymmetry': 0}


class L22_EnsembleTop5(_EnsembleBase):
    """Mean asymmetry of top 5 Round 1 methods."""
    name = "L22_EnsembleTop5"
    description = "Mean asymmetry of L01-L05"
    method_names = TIER1_METHODS


class L23_EnsembleVE(_EnsembleBase):
    """Mean asymmetry of all VE-based methods."""
    name = "L23_EnsembleVE"
    description = "Mean asymmetry of VE-based methods"
    method_names = VE_METHODS


class L24_EnsembleAll(_EnsembleBase):
    """Mean asymmetry of all 20 single methods."""
    name = "L24_EnsembleAll"
    description = "Mean asymmetry of all single methods"
    method_names = ALL_SINGLE


class L25_MaxAsymmetry(_EnsembleBase):
    """Max asymmetry across all single methods."""
    name = "L25_MaxAsymmetry"
    description = "Max asymmetry across all single methods"
    method_names = ALL_SINGLE
