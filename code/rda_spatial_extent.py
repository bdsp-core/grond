"""
RDA Spatial Extent Estimation

Estimates per-channel RDA involvement and spatial extent from 18-channel
bipolar EEG using variance explained (VE), narrowband SNR, and phase
locking value (PLV) metrics.
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 200


def _bandpass(sig, lo, hi, order=4):
    """Bandpass filter using sosfiltfilt. sig shape: (n_channels, n_samples)."""
    nyq = FS / 2.0
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 0.9999)
    if lo_n >= hi_n:
        return np.zeros_like(sig)
    sos = butter(order, [lo_n, hi_n], btype='band', output='sos')
    return sosfiltfilt(sos, sig, axis=-1)


def _compute_ve(sig_broadband, freq_hz):
    """Compute variance explained by sin+cos fit at given frequency.

    sig_broadband: 1D array (n_samples,)
    Returns VE in [0, 1], clipped.
    """
    n = len(sig_broadband)
    t = np.arange(n) / FS
    omega = 2 * np.pi * freq_hz
    # Design matrix: sin, cos, intercept
    A = np.column_stack([np.sin(omega * t), np.cos(omega * t), np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(A, sig_broadband, rcond=None)
    y_hat = A @ coeffs
    residual = sig_broadband - y_hat
    var_sig = np.var(sig_broadband)
    if var_sig < 1e-12:
        return 0.0
    ve = 1.0 - np.var(residual) / var_sig
    return float(np.clip(ve, 0.0, 1.0))


def _compute_snr(sig_broadband, sig_narrowband):
    """SNR = var(narrowband) / var(broadband), clipped to [0, 1]."""
    var_bb = np.var(sig_broadband)
    if var_bb < 1e-12:
        return 0.0
    snr = np.var(sig_narrowband) / var_bb
    return float(np.clip(snr, 0.0, 1.0))


def _compute_plv(phases_ch, phases_ref):
    """Phase locking value between channel and reference phases."""
    diff = phases_ch - phases_ref
    plv = np.abs(np.mean(np.exp(1j * diff)))
    return float(plv)


def compute_channel_metrics(seg_bi, freq_hz):
    """Compute VE, SNR, and PLV for all 18 channels.

    Parameters
    ----------
    seg_bi : ndarray (18, 2000)
        Bipolar EEG segment at 200 Hz.
    freq_hz : float
        Estimated RDA frequency.

    Returns
    -------
    dict with keys:
        ve : ndarray (18,)
        snr : ndarray (18,)
        plv : ndarray (18,)
    """
    n_ch = seg_bi.shape[0]

    # Broadband filter: 0.3-5 Hz
    bb = _bandpass(seg_bi, 0.3, 5.0, order=4)

    # Narrowband filter: freq +/- 0.4 Hz
    nb_lo = max(freq_hz - 0.4, 0.1)
    nb_hi = freq_hz + 0.4
    nb = _bandpass(seg_bi, nb_lo, nb_hi, order=3)

    # Per-channel VE and SNR
    ve = np.zeros(n_ch)
    snr = np.zeros(n_ch)
    for ch in range(n_ch):
        ve[ch] = _compute_ve(bb[ch], freq_hz)
        snr[ch] = _compute_snr(bb[ch], nb[ch])

    # PLV: reference = mean phase of top-3 VE channels on dominant hemisphere
    # Channels 0-7: left temporal + left parasagittal lower half
    # Channels 8-15: right temporal + right parasagittal lower half
    # Channels 16-17: midline
    # Hemisphere grouping for bipolar:
    #   Left: 0-3 (LT chain), 8-11 (LP chain)
    #   Right: 4-7 (RT chain), 12-15 (RP chain)
    #   Midline: 16-17
    left_idx = [0, 1, 2, 3, 8, 9, 10, 11]
    right_idx = [4, 5, 6, 7, 12, 13, 14, 15]

    left_ve_mean = np.mean(ve[left_idx])
    right_ve_mean = np.mean(ve[right_idx])

    if left_ve_mean >= right_ve_mean:
        dominant_idx = left_idx
    else:
        dominant_idx = right_idx

    # Top 3 channels by VE on dominant hemisphere
    dom_ve = [(i, ve[i]) for i in dominant_idx]
    dom_ve.sort(key=lambda x: x[1], reverse=True)
    top3_idx = [x[0] for x in dom_ve[:3]]

    # Get analytic signal phase for narrowband
    from scipy.signal import hilbert
    analytic = hilbert(nb, axis=-1)
    phases = np.angle(analytic)

    # Reference phase = circular mean of top-3 channels
    ref_phase = np.angle(np.mean(np.exp(1j * phases[top3_idx]), axis=0))

    plv = np.zeros(n_ch)
    for ch in range(n_ch):
        plv[ch] = _compute_plv(phases[ch], ref_phase)

    # Amplitude-weighted PLV: PLV × (channel narrowband envelope / max envelope)
    # This downweights contralateral channels that are phase-locked via volume
    # conduction but have low-amplitude delta compared to the dominant side.
    nb_amp = np.array([np.mean(np.abs(analytic[ch])) for ch in range(n_ch)])
    max_amp = np.max(nb_amp) if np.max(nb_amp) > 1e-10 else 1.0
    amp_ratio = nb_amp / max_amp
    plv_amp = plv * amp_ratio

    return {'ve': ve, 'snr': snr, 'plv': plv, 'plv_amp': plv_amp,
            'nb_amplitude': nb_amp, 'amp_ratio': amp_ratio}


def rda_spatial_extent(seg_bi, freq_hz, threshold=0.15, metric='plv_amp',
                       blend_weights=None, relative_threshold=None):
    """Estimate RDA spatial extent from bipolar EEG.

    Evaluated on 208 LRDA/GRDA segments against 3-rater ground truth.
    Best approach: PLV×Amplitude at threshold=0.15.
        - PLV×Amp (T=0.15): MAE=0.126, r=0.672 (LRDA r=0.568, GRDA MAE=0.121)
        - PLV only (T=0.32): MAE=0.196, r=0.373 (LRDA r=0.040 — no laterality)
    Amplitude weighting downweights contralateral volume-conducted signals.

    Parameters
    ----------
    seg_bi : ndarray (18, n_samples)
        Bipolar EEG segment at 200 Hz.
    freq_hz : float
        Estimated RDA frequency in Hz.
    threshold : float
        Absolute score threshold for binary channel involvement.
        Default 0.62 (optimal for PLV). Ignored if relative_threshold set.
    metric : str
        One of 've', 'snr', 'plv', 'blend'. Default 'plv'.
    blend_weights : dict, optional
        Weights for blend, e.g. {'ve': 0.5, 'plv': 0.5}.
    relative_threshold : float, optional
        If set, threshold = relative_threshold * max(channel_scores).
        Optimal value is 0.70 for PLV (MAE=0.209).

    Returns
    -------
    dict with keys:
        channel_scores : ndarray (18,) - per-channel involvement scores
        spatial_extent : float - fraction of channels involved (0-1)
        n_involved : int - number of involved channels
        spatial_extent_continuous : float - mean of channel scores (no threshold)
        metrics : dict - all raw metrics (ve, snr, plv arrays)
    """
    if seg_bi.shape[0] != 18:
        raise ValueError(f"Expected 18 channels, got {seg_bi.shape[0]}")

    metrics = compute_channel_metrics(seg_bi, freq_hz)

    if metric == 've':
        scores = metrics['ve']
    elif metric == 'snr':
        scores = metrics['snr']
    elif metric == 'plv':
        scores = metrics['plv']
    elif metric == 'plv_amp':
        scores = metrics['plv_amp']
    elif metric == 'blend':
        if blend_weights is None:
            blend_weights = {'ve': 0.5, 'plv': 0.5}
        scores = (blend_weights.get('ve', 0) * metrics['ve'] +
                  blend_weights.get('snr', 0) * metrics['snr'] +
                  blend_weights.get('plv', 0) * metrics['plv'] +
                  blend_weights.get('plv_amp', 0) * metrics['plv_amp'])
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Determine effective threshold
    if relative_threshold is not None:
        eff_threshold = relative_threshold * np.max(scores)
    else:
        eff_threshold = threshold

    involved = scores >= eff_threshold
    n_involved = int(np.sum(involved))
    spatial_extent = n_involved / 18.0
    spatial_extent_continuous = float(np.mean(scores))

    return {
        'channel_scores': scores,
        'spatial_extent': spatial_extent,
        'n_involved': n_involved,
        'spatial_extent_continuous': spatial_extent_continuous,
        'metrics': metrics,
    }
