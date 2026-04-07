"""Utilities for loading pre-computed spatial inference cache.

The spatial inference cache (spatial_inference_cache.json) stores per-segment
results from PDCharacterizer, Tautan et al., and RDA-PLV, avoiding the need
to re-run inference (~5 minutes) every time a figure is regenerated.

To generate the cache:
    conda run -n morgoth python paper_materials/precompute_spatial_cache.py

To check if cache should be used:
    import os
    if os.environ.get('USE_SPATIAL_CACHE') and load_spatial_cache() is not None:
        # use cache
"""

import json
import os
import numpy as np
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / 'spatial_inference_cache.json'


def load_spatial_cache():
    """Load spatial inference cache. Returns dict or None if not available."""
    if not CACHE_PATH.exists():
        return None
    with open(CACHE_PATH) as f:
        return json.load(f)


def use_cache():
    """Check if we should use the cache (env var set + cache exists)."""
    return os.environ.get('USE_SPATIAL_CACHE') and CACHE_PATH.exists()


def get_cached_pdchar_spatial(cache, mat_files, default=np.nan):
    """Get PDCharacterizer spatial extent from cache for a list of mat files."""
    results = np.full(len(mat_files), default)
    for i, mf in enumerate(mat_files):
        entry = cache.get(mf, {})
        if 'pdchar_spatial_extent' in entry:
            results[i] = entry['pdchar_spatial_extent']
    return results


def get_cached_tautan_spatial(cache, mat_files, default=np.nan):
    """Get Tautan et al. spatial extent from cache for a list of mat files."""
    results = np.full(len(mat_files), default)
    for i, mf in enumerate(mat_files):
        entry = cache.get(mf, {})
        if 'tautan_spatial_extent' in entry:
            val = entry['tautan_spatial_extent']
            if val is not None:
                results[i] = val
    return results


def get_cached_rda_spatial(cache, mat_files, mode='threshold', default=np.nan):
    """Get RDA-PLV spatial extent from cache for a list of mat files."""
    key = 'rda_spatial_extent' if mode == 'threshold' else 'rda_spatial_continuous'
    results = np.full(len(mat_files), default)
    for i, mf in enumerate(mat_files):
        entry = cache.get(mf, {})
        if key in entry:
            results[i] = entry[key]
    return results


def get_cached_channel_scores(cache, mat_file, method='pdchar'):
    """Get per-channel scores from cache for a single mat file."""
    entry = cache.get(mat_file, {})
    if method == 'pdchar':
        return entry.get('pdchar_channel_probs')
    elif method == 'rda':
        return entry.get('rda_channel_scores')
    return None
