#!/usr/bin/env python
"""
Phase 1: Confidence-Weighted Fusion of CNN + ACF Frequency Estimates
====================================================================

No retraining required. Uses existing model weights and adds an adaptive
fusion layer based on estimator confidence proxies.

Variants tested:
  V1_InverseVariance  — Bayesian inverse-variance fusion
  V2_SoftSwitch       — Sigmoid-based switch on confidence ratio
  V3_MaxConfidence    — Take the more confident estimate
  V4_ACFDominant      — Fixed 0.3*CNN + 0.7*ACF
  V5_PureACF          — ACF only (B1 reference)
  V6_AdaptiveSubtype  — Laterality-adaptive weighting
  V7_InvVar_WithIPI   — 3-way fusion: CNN + ACF + baseline IPI
  V8_ACF_Refined      — Confidence-weighted CNN+ACF as DP prior
  V9_ChannelConsensus — Per-channel IPI median across periodic channels
  V10_BestCombo       — Best elements from V1-V9

Usage:
    nohup /opt/homebrew/Caskroom/miniforge/base/envs/morgoth/bin/python -u \
        code/hemi_detector/confidence_weighted_freq.py \
        > /tmp/confidence_fusion.log 2>&1 &
"""

import sys
import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr

import torch

# ── Path setup ──
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))

from pd_profiler import PDProfiler
from pd_pointiness_acf import fcn_getBanana, pd_detect_pointiness_acf
from pd_pointiness_acf import compute_acf_frequency, compute_pointiness_trace
from discharge_detector import (
    compute_channel_evidence, combine_evidence,
    detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, FS,
    estimate_frequency_acf, DischargeDetector,
    LEFT_INDICES, RIGHT_INDICES,
)
import discharge_detector as dd

RESULTS_DIR = PROJECT_DIR / 'results' / 'gpd_freq_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOWPASS_HZ = 20.0


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (reused from gpd_freq_contest.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_expert_freq():
    """Load expert frequency labels from annotations.csv + segment_labels.csv."""
    expert_freq = {}

    ann_path = PROJECT_DIR / 'data' / 'labels' / 'annotations.csv'
    if ann_path.exists():
        ann = pd.read_csv(ann_path)
        has_freq = ann[ann.frequency_hz.notna()].copy()
        has_freq['frequency_hz'] = pd.to_numeric(has_freq['frequency_hz'],
                                                  errors='coerce')
        has_freq = has_freq[has_freq.frequency_hz.notna()]
        freq_agg = (has_freq.groupby('segment_id')
                    .agg(mean_freq=('frequency_hz', 'mean'))
                    .reset_index())
        expert_freq = dict(zip(freq_agg.segment_id, freq_agg.mean_freq))

    sl_path = PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv'
    if sl_path.exists():
        sl = pd.read_csv(sl_path)
        mw_rows = sl[sl.expert_freq_rater == 'MW']
        for _, row in mw_rows.iterrows():
            sid = row['mat_file'].replace('.mat', '')
            if sid not in expert_freq:
                val = pd.to_numeric(row.get('expert_freq_hz'), errors='coerce')
                if pd.notna(val):
                    expert_freq[sid] = float(val)

    return expert_freq


def load_segments():
    """Load segment list from segment_labels.csv for LPD and GPD."""
    sl = pd.read_csv(PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv')
    pd_segs = sl[sl.subtype.isin(['lpd', 'gpd']) & (~sl.excluded)].copy()
    return pd_segs


# ══════════════════════════════════════════════════════════════════════════════
# Extended precomputation — adds per-fold and per-channel freq details
# ══════════════════════════════════════════════════════════════════════════════

def compute_per_channel_cnn_details(det, segment_18ch):
    """Run CNN ensemble and return per-channel, per-fold frequency details.

    Returns:
        channel_pd_probs: (n_ch,) mean PD prob per channel
        channel_log_freqs: (n_ch,) mean log-freq per channel
        fold_log_freqs: (n_folds, n_ch) per-fold log-freq
        fold_pd_probs: (n_folds, n_ch) per-fold PD prob
    """
    n_channels = min(segment_18ch.shape[0], 18)
    n_folds = len(det.cnn_models)
    device = det.device

    fold_log_freqs = np.zeros((n_folds, n_channels))
    fold_pd_probs = np.zeros((n_folds, n_channels))

    with torch.no_grad():
        for ch in range(n_channels):
            ch_data = segment_18ch[ch].astype(np.float32).copy()
            if not np.all(np.isfinite(ch_data)):
                continue

            mu, std = np.mean(ch_data), np.std(ch_data)
            ch_data = (ch_data - mu) / std if std > 1e-8 else ch_data - mu
            x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(device)

            for fi, model in enumerate(det.cnn_models):
                pd_prob, freq_pred, _ = model(x)
                fold_pd_probs[fi, ch] = pd_prob.item()
                fold_log_freqs[fi, ch] = freq_pred.item()

    channel_pd_probs = np.mean(fold_pd_probs, axis=0)
    channel_log_freqs = np.mean(fold_log_freqs, axis=0)

    return channel_pd_probs, channel_log_freqs, fold_log_freqs, fold_pd_probs


def compute_per_channel_acf_details(segment_18ch):
    """Compute ACF frequency per channel with confidence info.

    Returns:
        acf_freqs: list of (channel_idx, frequency, acf_peak_height)
    """
    n_channels = min(segment_18ch.shape[0], 18)
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    results = []
    for ch in range(n_channels):
        try:
            sig = filtfilt(b_lp, a_lp, segment_18ch[ch])
        except ValueError:
            sig = segment_18ch[ch]

        pt = compute_pointiness_trace(sig)
        if np.max(pt) < 1e-10:
            continue

        # ACF computation with peak height
        pt_centered = pt - np.mean(pt)
        n = len(pt_centered)
        acf = np.correlate(pt_centered, pt_centered, mode='full')[n - 1:]
        if acf[0] > 0:
            acf = acf / acf[0]
        else:
            continue

        min_lag = int(0.4 * FS)
        max_lag = min(int(3.0 * FS), len(acf) - 1)
        if max_lag <= min_lag:
            continue

        acf_seg = acf[min_lag:max_lag + 1]
        peaks, props = find_peaks(acf_seg, height=0.1)
        if len(peaks) == 0:
            continue

        best_peak = peaks[0]
        peak_height = props['peak_heights'][0]
        freq = FS / (best_peak + min_lag)
        if 0.2 < freq < 5.0:
            results.append((ch, freq, float(peak_height)))

    return results


def precompute_segment(mat_file, subtype, pc, det):
    """Compute all per-segment data including extended details."""
    mat = sio.loadmat(str(PROJECT_DIR / 'data' / 'eeg' / mat_file))
    data = mat['data']
    if data.shape[0] > data.shape[1]:
        data = data.T
    n_samples = min(2000, data.shape[1])

    mono = None
    if data.shape[0] >= 19:
        mono = data[:19, :n_samples].astype(float)

    seg_bi = fcn_getBanana(data[:, :n_samples])

    # Run full PDProfiler for baseline
    result = pc.characterize(seg_bi, subtype=subtype)

    # Per-channel evidence
    n_ch = min(18, seg_bi.shape[0])
    hpp_all = np.zeros((n_ch, seg_bi.shape[1]))
    cet_all = np.zeros((n_ch, seg_bi.shape[1]), dtype=np.float32)
    for ch in range(n_ch):
        hpp_all[ch] = compute_channel_evidence(seg_bi[ch], FS)
        if np.all(np.isfinite(seg_bi[ch])):
            cet_all[ch] = pc._cet_compute(seg_bi[ch])

    # Standard frequency estimates
    cnn_freq = det.estimate_frequency(seg_bi)
    acf_freq = det.estimate_frequency_acf_multichannel(seg_bi, subtype,
                                                        result.get('laterality'))

    # Extended: per-channel/per-fold CNN details
    ch_pd_probs, ch_log_freqs, fold_log_freqs, fold_pd_probs = \
        compute_per_channel_cnn_details(det, seg_bi)

    # Extended: per-channel ACF details
    acf_details = compute_per_channel_acf_details(seg_bi)

    return {
        'mat_file': mat_file,
        'subtype': subtype,
        'seg_bi': seg_bi,
        'mono': mono,
        'channel_probs': np.array(result['channel_probs'][:18]),
        'hpp_all': hpp_all,
        'cet_all': cet_all,
        'cnn_freq': cnn_freq,
        'acf_freq': acf_freq,
        'baseline_times': result.get('discharge_times', []),
        'baseline_freq': result.get('frequency', np.nan),
        'laterality': result.get('laterality'),
        # Extended fields
        'ch_pd_probs': ch_pd_probs,
        'ch_log_freqs': ch_log_freqs,
        'fold_log_freqs': fold_log_freqs,
        'fold_pd_probs': fold_pd_probs,
        'acf_details': acf_details,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_baseline_freq_est(seg_cache):
    """Baseline blended freq: 0.8*CNN + 0.2*ACF."""
    cnn = seg_cache['cnn_freq']
    acf = seg_cache['acf_freq']
    if np.isfinite(acf):
        return float(np.clip(0.8 * cnn + 0.2 * acf, 0.3, 3.5))
    return float(np.clip(cnn, 0.3, 3.5))


def get_full_evidence(seg_cache):
    """Standard all-channel CNN-weighted evidence."""
    hpp_all = seg_cache['hpp_all']
    cet_all = seg_cache['cet_all']
    weights = np.clip(seg_cache['channel_probs'][:18], 0.05, None)
    n_ch = hpp_all.shape[0]
    ch_idx = np.arange(n_ch)
    hpp = np.average(hpp_all[ch_idx], weights=weights[:n_ch], axis=0)
    cet = np.average(cet_all[ch_idx], weights=weights[:n_ch], axis=0)
    return combine_evidence(hpp, cet)


def run_standard_dp(evidence, freq_est):
    """Run DP pipeline with a given frequency estimate."""
    freq_est = float(np.clip(freq_est, 0.3, 3.5))
    active_start, active_end = detect_active_interval(evidence, FS)
    cands = extract_candidates(evidence, FS, freq_est, active_start, active_end)
    times = dp_best_sequence(cands, evidence, FS, freq_est)
    if len(times) >= 3:
        times = em_refine(evidence, times, FS, freq_est)
    times = posthoc_filter(times, evidence)
    return times


def ipi_freq_from_times(times_samples):
    """Compute IPI-derived frequency from discharge sample indices."""
    t = np.sort(times_samples) / FS
    if len(t) >= 2:
        return 1.0 / float(np.median(np.diff(t)))
    return np.nan


def _baseline_lpd(seg_cache):
    """For LPD: always return baseline result."""
    t = seg_cache['baseline_times']
    t_samp = (np.array(t) * FS).astype(int) if len(t) > 0 else np.array([])
    return t_samp, seg_cache['baseline_freq']


# ══════════════════════════════════════════════════════════════════════════════
# Confidence proxies
# ══════════════════════════════════════════════════════════════════════════════

def compute_cnn_confidence(seg_cache):
    """CNN confidence based on channel-level frequency consistency.

    If PD+ channels agree on frequency -> high confidence.
    Also uses fold disagreement as secondary signal.
    """
    ch_pd_probs = seg_cache['ch_pd_probs']
    ch_log_freqs = seg_cache['ch_log_freqs']
    fold_log_freqs = seg_cache['fold_log_freqs']

    # Channel consistency: std of freq across PD+ channels
    pd_mask = ch_pd_probs > 0.5
    if pd_mask.sum() >= 2:
        pd_freqs = np.exp(ch_log_freqs[pd_mask])
        cnn_channel_std = np.std(pd_freqs)
        channel_confidence = 1.0 / (cnn_channel_std + 0.1)
    elif pd_mask.sum() == 1:
        channel_confidence = 0.5
    else:
        channel_confidence = 0.3

    # Fold consistency: std of ensemble-averaged freq across folds
    # Each fold -> weighted mean freq
    fold_freqs = []
    for fi in range(fold_log_freqs.shape[0]):
        w = np.clip(seg_cache['fold_pd_probs'][fi], 0.01, None)
        w_sum = w.sum()
        if w_sum > 1e-6:
            wlf = np.sum(w * fold_log_freqs[fi]) / w_sum
            fold_freqs.append(np.exp(wlf))
        else:
            fold_freqs.append(np.exp(np.mean(fold_log_freqs[fi])))
    fold_std = np.std(fold_freqs)
    fold_confidence = 1.0 / (fold_std + 0.1)

    # Combine: geometric mean
    return float(np.sqrt(channel_confidence * fold_confidence))


def compute_acf_confidence(seg_cache):
    """ACF confidence based on peak height and cross-channel consistency."""
    acf_details = seg_cache['acf_details']

    if len(acf_details) < 2:
        return 0.1

    freqs = [d[1] for d in acf_details]
    peaks = [d[2] for d in acf_details]

    # Peak height: median ACF peak across channels (0 to ~1)
    peak_confidence = float(np.median(peaks))

    # Consistency: inverse std of freq across channels
    freq_std = np.std(freqs)
    consistency = 1.0 / (freq_std + 0.1)
    consistency = min(consistency, 5.0)  # cap

    return float(peak_confidence * consistency)


def compute_acf_freq_from_details(seg_cache):
    """Compute median ACF frequency from per-channel details."""
    acf_details = seg_cache['acf_details']
    if len(acf_details) == 0:
        # Fallback to stored acf_freq
        return seg_cache['acf_freq']
    freqs = [d[1] for d in acf_details]
    return float(np.median(freqs))


# ══════════════════════════════════════════════════════════════════════════════
# 10 Fusion Variants (GPD only; LPD always uses baseline)
# ══════════════════════════════════════════════════════════════════════════════

def method_V1_InverseVariance(seg_cache):
    """V1: Bayesian inverse-variance fusion of CNN and ACF."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)

    sigma2_cnn = 1.0 / (cnn_conf + 0.01)
    sigma2_acf = 1.0 / (acf_conf + 0.01)
    w_cnn = (1.0 / sigma2_cnn) / (1.0 / sigma2_cnn + 1.0 / sigma2_acf)

    freq_est = w_cnn * cnn_freq + (1.0 - w_cnn) * acf_freq_val
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V2_SoftSwitch(seg_cache):
    """V2: Sigmoid-based switch — high ACF confidence shifts toward ACF."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)

    ratio = acf_conf / (cnn_conf + 0.01)
    w_acf = 1.0 / (1.0 + np.exp(-2.0 * (ratio - 1.0)))

    freq_est = (1.0 - w_acf) * cnn_freq + w_acf * acf_freq_val
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V3_MaxConfidence(seg_cache):
    """V3: Take whichever estimator is more confident."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)

    if acf_conf > cnn_conf:
        freq_est = acf_freq_val
    else:
        freq_est = cnn_freq
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V4_ACFDominant(seg_cache):
    """V4: Fixed 0.3*CNN + 0.7*ACF (shift weight toward ACF for GPD)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    freq_est = 0.3 * cnn_freq + 0.7 * acf_freq_val
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V5_PureACF(seg_cache):
    """V5: Pure ACF (B1 reference — contest winner)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = seg_cache['cnn_freq']
    freq_est = float(np.clip(acf_freq_val, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V6_AdaptiveSubtype(seg_cache):
    """V6: Laterality-adaptive — bilateral (GPD-like) favors ACF."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    # Compute laterality index from channel probs
    probs = seg_cache['channel_probs']
    left_mean = np.mean(probs[LEFT_INDICES])
    right_mean = np.mean(probs[RIGHT_INDICES])
    total = left_mean + right_mean
    if total > 1e-6:
        lat_idx = abs(left_mean - right_mean) / total
    else:
        lat_idx = 0.0

    # Bilateral (lat_idx near 0) -> more ACF weight
    w_cnn = 0.2 + 0.6 * lat_idx  # 0.2 for bilateral, 0.8 for strongly lateralized
    freq_est = w_cnn * cnn_freq + (1.0 - w_cnn) * acf_freq_val
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V7_InvVar_WithIPI(seg_cache):
    """V7: 3-way inverse-variance fusion: CNN + ACF + baseline IPI freq."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq
    ipi_freq = seg_cache['baseline_freq']
    if not np.isfinite(ipi_freq):
        ipi_freq = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)
    # IPI confidence: moderate fixed value (it's already post-DP)
    ipi_conf = 1.0

    estimates = [cnn_freq, acf_freq_val, ipi_freq]
    confidences = [cnn_conf, acf_conf, ipi_conf]

    weights = [c / (1.0 / (c + 0.01)) for c in confidences]
    # Simplify: weight = confidence
    w_total = sum(confidences) + 1e-10
    freq_est = sum(e * c for e, c in zip(estimates, confidences)) / w_total
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_V8_ACF_Refined(seg_cache):
    """V8: Confidence-weighted CNN+ACF as DP prior, then take IPI."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)

    # Use inverse-variance for the prior
    sigma2_cnn = 1.0 / (cnn_conf + 0.01)
    sigma2_acf = 1.0 / (acf_conf + 0.01)
    w_cnn = (1.0 / sigma2_cnn) / (1.0 / sigma2_cnn + 1.0 / sigma2_acf)
    prior_freq = w_cnn * cnn_freq + (1.0 - w_cnn) * acf_freq_val
    prior_freq = float(np.clip(prior_freq, 0.3, 3.5))

    # Run DP with this refined prior
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, prior_freq)

    # The output frequency is IPI-derived (unlike V1 which just uses IPI)
    freq = ipi_freq_from_times(times)

    # If IPI is unreliable (too few discharges), use the prior
    if not np.isfinite(freq):
        freq = prior_freq
    elif len(times) < 5:
        # Blend IPI with prior for low-discharge cases
        freq = 0.5 * freq + 0.5 * prior_freq

    return times, freq


def method_V9_ChannelConsensus(seg_cache):
    """V9: Per-channel DP, take median IPI freq across periodic channels."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    seg_bi = seg_cache['seg_bi']
    n_ch = min(18, seg_bi.shape[0])
    freq_est = get_baseline_freq_est(seg_cache)

    ch_ipi_freqs = []
    ch_n_discharges = []

    for ch in range(n_ch):
        evidence_ch = compute_channel_evidence(seg_bi[ch], FS)
        if np.max(evidence_ch) < 0.05:
            continue
        try:
            ch_times = run_standard_dp(evidence_ch, freq_est)
            if len(ch_times) >= 3:
                ch_freq = ipi_freq_from_times(ch_times)
                if np.isfinite(ch_freq) and 0.2 < ch_freq < 5.0:
                    ch_ipi_freqs.append(ch_freq)
                    ch_n_discharges.append(len(ch_times))
        except Exception:
            pass

    if len(ch_ipi_freqs) >= 2:
        # Weight by number of discharges detected
        weights = np.array(ch_n_discharges, dtype=float)
        freq_consensus = float(np.average(ch_ipi_freqs, weights=weights))
        freq_consensus = float(np.clip(freq_consensus, 0.3, 3.5))
    elif len(ch_ipi_freqs) == 1:
        freq_consensus = float(np.clip(ch_ipi_freqs[0], 0.3, 3.5))
    else:
        freq_consensus = freq_est

    # Run final DP on full evidence with consensus freq
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_consensus)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_consensus
    return times, freq


def method_V10_BestCombo(seg_cache):
    """V10: Best combo — confidence-weighted ACF-dominant + channel consensus.

    Uses V1 (inverse-variance) for the freq prior, but caps CNN weight at 0.5
    (so ACF always gets at least 50% weight for GPD). Also incorporates
    channel consensus as a tiebreaker.
    """
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)

    cnn_freq = seg_cache['cnn_freq']
    acf_freq_val = compute_acf_freq_from_details(seg_cache)
    if not np.isfinite(acf_freq_val):
        acf_freq_val = cnn_freq

    cnn_conf = compute_cnn_confidence(seg_cache)
    acf_conf = compute_acf_confidence(seg_cache)

    # Inverse-variance but cap CNN weight at 0.5 for GPD
    sigma2_cnn = 1.0 / (cnn_conf + 0.01)
    sigma2_acf = 1.0 / (acf_conf + 0.01)
    w_cnn = (1.0 / sigma2_cnn) / (1.0 / sigma2_cnn + 1.0 / sigma2_acf)
    w_cnn = min(w_cnn, 0.5)  # ACF gets at least 50%

    prior_freq = w_cnn * cnn_freq + (1.0 - w_cnn) * acf_freq_val

    # Also get channel consensus freq if available
    acf_details = seg_cache['acf_details']
    if len(acf_details) >= 3:
        acf_freqs = [d[1] for d in acf_details]
        acf_peaks = [d[2] for d in acf_details]
        # Weight by peak height
        consensus_freq = float(np.average(acf_freqs, weights=acf_peaks))
        # Blend prior with consensus
        freq_est = 0.6 * prior_freq + 0.4 * consensus_freq
    else:
        freq_est = prior_freq

    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


# ══════════════════════════════════════════════════════════════════════════════
# Method registry
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    'Baseline':            None,
    'V1_InverseVariance':  method_V1_InverseVariance,
    'V2_SoftSwitch':       method_V2_SoftSwitch,
    'V3_MaxConfidence':    method_V3_MaxConfidence,
    'V4_ACFDominant':      method_V4_ACFDominant,
    'V5_PureACF':          method_V5_PureACF,
    'V6_AdaptiveSubtype':  method_V6_AdaptiveSubtype,
    'V7_InvVar_WithIPI':   method_V7_InvVar_WithIPI,
    'V8_ACF_Refined':      method_V8_ACF_Refined,
    'V9_ChannelConsensus': method_V9_ChannelConsensus,
    'V10_BestCombo':       method_V10_BestCombo,
}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_method(method_name, method_fn, cache_list, expert_freq):
    """Run a method on all segments and compute metrics."""
    results = {'lpd': [], 'gpd': []}
    per_segment = []

    for seg in cache_list:
        sid = seg['mat_file'].replace('.mat', '')
        sub = seg['subtype']

        if sid not in expert_freq:
            continue

        gt_freq = expert_freq[sid]
        if not np.isfinite(gt_freq) or gt_freq <= 0:
            continue

        try:
            if method_fn is None:
                pred_freq = seg['baseline_freq']
            else:
                _, pred_freq = method_fn(seg)

            if not np.isfinite(pred_freq) or pred_freq <= 0:
                pred_freq = np.nan
        except Exception as e:
            print(f"  ERROR {method_name} on {sid}: {e}")
            traceback.print_exc()
            pred_freq = np.nan

        per_segment.append({
            'segment_id': sid,
            'subtype': sub,
            'gt_freq': float(gt_freq),
            'pred_freq': float(pred_freq) if np.isfinite(pred_freq) else None,
        })

        if np.isfinite(pred_freq):
            results[sub].append((gt_freq, pred_freq))

    # Compute metrics
    metrics = {}
    for sub in ['lpd', 'gpd']:
        pairs = results[sub]
        if len(pairs) >= 5:
            gt = [p[0] for p in pairs]
            pr = [p[1] for p in pairs]
            rho, pval = spearmanr(gt, pr)
            mae = float(np.mean(np.abs(np.array(gt) - np.array(pr))))
            metrics[sub] = {
                'n': len(pairs),
                'rho': float(rho) if np.isfinite(rho) else 0.0,
                'pval': float(pval) if np.isfinite(pval) else 1.0,
                'mae': mae,
            }
        else:
            metrics[sub] = {'n': len(pairs), 'rho': 0.0, 'pval': 1.0,
                            'mae': 999.0}

    return metrics, per_segment


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table and results saving
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_results, baseline_metrics):
    """Print formatted comparison table."""
    bl_lpd = baseline_metrics.get('lpd', {})
    bl_gpd = baseline_metrics.get('gpd', {})

    print("\n" + "=" * 95)
    print("CONFIDENCE-WEIGHTED FUSION RESULTS")
    print("=" * 95)
    hdr = (f"{'Method':<25s} {'LPD rho':>8s} {'LPD MAE':>8s} "
           f"{'GPD rho':>8s} {'GPD MAE':>8s} {'LPD ok?':>8s} {'GPD delta':>10s}")
    print(hdr)
    print("-" * 95)

    for name in METHODS:
        m = all_results.get(name)
        if m is None:
            continue
        lpd = m.get('lpd', {})
        gpd = m.get('gpd', {})

        lpd_rho = lpd.get('rho', 0.0)
        lpd_mae = lpd.get('mae', 999.0)
        gpd_rho = gpd.get('rho', 0.0)
        gpd_mae = gpd.get('mae', 999.0)

        # LPD degradation check
        lpd_ok = 'YES' if lpd_rho >= bl_lpd.get('rho', 0) - 0.01 else 'NO'
        if name == 'Baseline':
            lpd_ok = '--'

        # GPD delta vs baseline
        gpd_delta = gpd_rho - bl_gpd.get('rho', 0)
        delta_str = f"{gpd_delta:+.3f}" if name != 'Baseline' else '--'

        print(f"{name:<25s} {lpd_rho:>8.3f} {lpd_mae:>8.3f} "
              f"{gpd_rho:>8.3f} {gpd_mae:>8.3f} {lpd_ok:>8s} {delta_str:>10s}")

    print("=" * 95)


def save_all_results(all_results, all_per_segment):
    """Save combined results to JSON."""
    out = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Phase 1: Confidence-weighted fusion of CNN + ACF frequency',
        'methods': {},
    }
    for name in METHODS:
        if name in all_results:
            out['methods'][name] = {
                'metrics': all_results[name],
                'per_segment': all_per_segment.get(name, []),
            }

    out_path = RESULTS_DIR / 'confidence_fusion_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Phase 1: Confidence-Weighted Fusion — 10 Variants")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    t0 = time.time()

    # Load expert labels
    print("\nLoading expert frequency labels...")
    expert_freq = load_expert_freq()
    print(f"  {len(expert_freq)} segments with expert freq")

    # Load segment list
    print("Loading segment list...")
    segs_df = load_segments()
    print(f"  {len(segs_df)} PD segments "
          f"({(segs_df.subtype == 'lpd').sum()} LPD, "
          f"{(segs_df.subtype == 'gpd').sum()} GPD)")

    # Filter to segments with expert freq AND existing mat files
    eeg_dir = PROJECT_DIR / 'data' / 'eeg'
    seg_list = []
    for _, row in segs_df.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        if sid in expert_freq and (eeg_dir / row['mat_file']).exists():
            seg_list.append(row)
    print(f"  {len(seg_list)} segments with expert freq + EEG files")

    # Initialize models
    print("\nInitializing models...")
    pc = PDProfiler()
    det = DischargeDetector()
    print("  Models loaded.")

    # Precompute
    print(f"\nPrecomputing per-segment data ({len(seg_list)} segments)...")
    cache_list = []
    for i, row in enumerate(seg_list):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(seg_list)}] {row['mat_file']} "
                  f"({elapsed:.0f}s elapsed)")
        try:
            sc = precompute_segment(row['mat_file'], row['subtype'], pc, det)
            cache_list.append(sc)
        except Exception as e:
            print(f"  SKIP {row['mat_file']}: {e}")

    n_gpd = sum(1 for s in cache_list if s['subtype'] == 'gpd')
    n_lpd = sum(1 for s in cache_list if s['subtype'] == 'lpd')
    precompute_time = time.time() - t0
    print(f"  Cached {len(cache_list)} segments ({n_gpd} GPD, {n_lpd} LPD)")
    print(f"  Precompute time: {precompute_time:.1f}s")

    # Confidence distribution summary
    print("\nConfidence distribution (GPD segments):")
    gpd_cnn_confs = []
    gpd_acf_confs = []
    for seg in cache_list:
        if seg['subtype'] == 'gpd':
            gpd_cnn_confs.append(compute_cnn_confidence(seg))
            gpd_acf_confs.append(compute_acf_confidence(seg))
    if gpd_cnn_confs:
        print(f"  CNN confidence: mean={np.mean(gpd_cnn_confs):.3f}, "
              f"median={np.median(gpd_cnn_confs):.3f}, "
              f"std={np.std(gpd_cnn_confs):.3f}")
        print(f"  ACF confidence: mean={np.mean(gpd_acf_confs):.3f}, "
              f"median={np.median(gpd_acf_confs):.3f}, "
              f"std={np.std(gpd_acf_confs):.3f}")

    # Run all methods
    all_results = {}
    all_per_segment = {}
    baseline_metrics = None
    method_names = list(METHODS.keys())

    for mi, method_name in enumerate(method_names):
        method_fn = METHODS[method_name]
        print(f"\n[{mi+1}/{len(method_names)}] Running: {method_name}")
        t1 = time.time()

        metrics, per_seg = evaluate_method(
            method_name, method_fn, cache_list, expert_freq)

        elapsed = time.time() - t1
        gpd = metrics.get('gpd', {})
        lpd = metrics.get('lpd', {})
        print(f"  GPD: rho={gpd.get('rho', 0):.3f}, "
              f"MAE={gpd.get('mae', 999):.3f}, n={gpd.get('n', 0)}")
        print(f"  LPD: rho={lpd.get('rho', 0):.3f}, "
              f"MAE={lpd.get('mae', 999):.3f}, n={lpd.get('n', 0)}")
        print(f"  Time: {elapsed:.1f}s")

        all_results[method_name] = metrics
        all_per_segment[method_name] = per_seg

        if method_name == 'Baseline':
            baseline_metrics = metrics

    # Print comparison table
    if baseline_metrics:
        print_comparison_table(all_results, baseline_metrics)

    # Save results
    save_all_results(all_results, all_per_segment)

    # Final ranking
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s "
          f"(precompute: {precompute_time:.1f}s)")

    print("\nFinal GPD Frequency Ranking:")
    ranked = sorted(all_results.items(),
                    key=lambda x: -x[1].get('gpd', {}).get('rho', 0))
    for i, (name, m) in enumerate(ranked):
        gpd = m.get('gpd', {})
        bl_rho = baseline_metrics.get('gpd', {}).get('rho', 0) if baseline_metrics else 0
        delta = gpd.get('rho', 0) - bl_rho
        marker = ' ***' if delta > 0.05 else (' **' if delta > 0.02 else
                                               (' *' if delta > 0 else ''))
        print(f"  {i+1:2d}. {name:25s}  rho={gpd.get('rho', 0):.3f}  "
              f"MAE={gpd.get('mae', 999):.3f}  delta={delta:+.3f}{marker}")


if __name__ == '__main__':
    main()
