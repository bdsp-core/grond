"""
Wrapper for RDA analysis with per-channel scores and lateralization.

Calls the modified rda1b_fft detector and returns a clean result dict
including continuous per-channel RDA scores and a laterality index.
"""

import numpy as np
import rda_detector as rda


def analyze_rda(segment, fs, channel_filter=1):
    """
    Analyze an EEG segment for rhythmic delta activity.

    Parameters:
        segment: numpy array, shape (19, n_samples) - raw 19-channel EEG at fs Hz
        fs: sampling frequency (typically 200)
        channel_filter: 1 to reject noisy channels, 0 to include all

    Returns:
        dict with keys:
            # Existing outputs (backward compatible)
            type_event: 'LRDA', 'GRDA', or NaN
            event_frequency: median frequency across detected channels (Hz)
            spatial_extent: fraction of channels with RDA (0-1)
            spatial_areas: list of region labels with RDA

            # Per-channel outputs
            channel_scores: dict of {channel_name: rda_score} for all 18 bipolar channels
            channel_frequencies: dict of {channel_name: peak_freq_hz} for all 18 bipolar channels
            region_scores: dict of {region_name: mean_rda_score} for 8 regions

            # Lateralization
            laterality_index: float in [-1, +1], negative=left, positive=right
            left_mean_score: mean RDA score across left hemisphere channels
            right_mean_score: mean RDA score across right hemisphere channels
    """
    data_obj, spectra, freqs = rda.rda1b_fft(segment, fs, channel_filter)

    return {
        # Original outputs
        'type_event': data_obj['type_event'],
        'event_frequency': data_obj['event_frequency'],
        'spatial_extent': data_obj['spatial_extent'],
        'spatial_areas': data_obj['spatial_areas'],

        # Per-channel outputs
        'channel_scores': data_obj['channel_rda_scores'],
        'channel_frequencies': data_obj['channel_frequencies'],
        'region_scores': data_obj['region_scores'],

        # Lateralization
        'laterality_index': data_obj['laterality_index'],
        'left_mean_score': data_obj['left_mean_score'],
        'right_mean_score': data_obj['right_mean_score'],
    }
