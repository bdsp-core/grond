#!/usr/bin/env python
"""
GPD Frequency Estimation Contest — 25 methods
==============================================

Evaluates 25 different approaches to GPD frequency estimation.
LPD path is never modified (always uses baseline PDCharacterizer result).

Usage:
    nohup /opt/homebrew/Caskroom/miniforge/base/envs/morgoth/bin/python -u \
        code/hemi_detector/gpd_freq_contest.py > /tmp/gpd_freq_contest.log 2>&1 &
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
from scipy.signal import butter, filtfilt, welch
from scipy.stats import spearmanr

# ── Path setup ──
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))

from pd_characterizer import PDCharacterizer
from pd_pointiness_acf import fcn_getBanana, pd_detect_pointiness_acf
from pd_pointiness_acf import compute_acf_frequency, compute_pointiness_trace
from discharge_detector import (
    compute_channel_evidence, combine_evidence,
    detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, FS,
)
from discharge_detector import LEFT_INDICES, RIGHT_INDICES
import discharge_detector as dd

RESULTS_DIR = PROJECT_DIR / 'results' / 'gpd_freq_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Expert frequency labels ──

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


# ── Precompute ──

def precompute_segment(mat_file, subtype, pc):
    """Compute all per-segment data once."""
    mat = sio.loadmat(str(PROJECT_DIR / 'data' / 'eeg' / mat_file))
    data = mat['data']
    if data.shape[0] > data.shape[1]:
        data = data.T
    n_samples = min(2000, data.shape[1])

    mono = None
    if data.shape[0] >= 19:
        mono = data[:19, :n_samples].astype(float)

    seg_bi = fcn_getBanana(data[:, :n_samples])

    # Run full PDCharacterizer for baseline
    result = pc.characterize(seg_bi, subtype=subtype)

    # Per-channel evidence
    n_ch = min(18, seg_bi.shape[0])
    hpp_all = np.zeros((n_ch, seg_bi.shape[1]))
    cet_all = np.zeros((n_ch, seg_bi.shape[1]), dtype=np.float32)
    for ch in range(n_ch):
        hpp_all[ch] = compute_channel_evidence(seg_bi[ch], FS)
        if np.all(np.isfinite(seg_bi[ch])):
            cet_all[ch] = pc._cet_compute(seg_bi[ch])

    # Frequency estimates
    det = pc._freq_estimator
    cnn_freq = det.estimate_frequency(seg_bi)
    acf_freq = det.estimate_frequency_acf_multichannel(
        seg_bi, subtype, result.get('laterality'))

    # Alexandra's frequency
    alex_freq = np.nan
    if mono is not None:
        try:
            alex = pd_detect_pointiness_acf(mono, fs=200)
            alex_freq = alex.get('event_frequency', np.nan)
        except Exception:
            pass

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
        'alex_freq': alex_freq,
        'baseline_times': result.get('discharge_times', []),
        'baseline_freq': result.get('frequency', np.nan),
        'laterality': result.get('laterality'),
    }


# ── Helper: run DP on evidence ──

def run_standard_dp(evidence, freq_est):
    """Run detect_active_interval -> extract_candidates -> dp -> em -> posthoc."""
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


def run_hemi_dp(evidence_hemi, freq_est):
    """Run DP on a single hemisphere's evidence."""
    freq_est = float(np.clip(freq_est, 0.3, 3.5))
    active = detect_active_interval(evidence_hemi, FS)
    cands = extract_candidates(evidence_hemi, FS, freq_est, active[0], active[1])
    times = dp_best_sequence(cands, evidence_hemi, FS, freq_est)
    if len(times) >= 3:
        times = em_refine(evidence_hemi, times, FS, freq_est)
    times = posthoc_filter(times, evidence_hemi)
    return times


def get_hemi_evidence(seg_cache, indices):
    """Compute CNN-weighted evidence for a set of channel indices."""
    hpp_all = seg_cache['hpp_all']
    cet_all = seg_cache['cet_all']
    weights = np.clip(seg_cache['channel_probs'][:18], 0.05, None)
    w = weights[indices]
    hpp = np.average(hpp_all[indices], weights=w, axis=0)
    cet = np.average(cet_all[indices], weights=w, axis=0)
    return combine_evidence(hpp, cet)


def get_baseline_freq_est(seg_cache):
    """Compute the baseline blended freq estimate (0.8*CNN + 0.2*ACF)."""
    cnn = seg_cache['cnn_freq']
    acf = seg_cache['acf_freq']
    if np.isfinite(acf):
        return float(np.clip(0.8 * cnn + 0.2 * acf, 0.3, 3.5))
    return float(np.clip(cnn, 0.3, 3.5))


def get_full_evidence(seg_cache):
    """Compute standard all-channel CNN-weighted evidence (GPD baseline)."""
    hpp_all = seg_cache['hpp_all']
    cet_all = seg_cache['cet_all']
    weights = np.clip(seg_cache['channel_probs'][:18], 0.05, None)
    n_ch = hpp_all.shape[0]
    ch_idx = np.arange(n_ch)
    hpp = np.average(hpp_all[ch_idx], weights=weights[:n_ch], axis=0)
    cet = np.average(cet_all[ch_idx], weights=weights[:n_ch], axis=0)
    return combine_evidence(hpp, cet)


# ══════════════════════════════════════════════════════════════════════════════
# 25 METHODS
# ══════════════════════════════════════════════════════════════════════════════
# Each returns (discharge_times_samples, frequency_hz).
# For LPD segments, all methods return (baseline_times, baseline_freq).

def _baseline_lpd(seg_cache):
    """For LPD: always return baseline result."""
    t = seg_cache['baseline_times']
    t_samp = (np.array(t) * FS).astype(int) if len(t) > 0 else np.array([])
    return t_samp, seg_cache['baseline_freq']


# ── A: Hemisphere strategies ──

def _hemi_freqs(seg_cache):
    """Run DP on left and right hemispheres independently. Returns (left_times, right_times, left_freq, right_freq)."""
    freq_est = get_baseline_freq_est(seg_cache)
    ev_left = get_hemi_evidence(seg_cache, LEFT_INDICES)
    ev_right = get_hemi_evidence(seg_cache, RIGHT_INDICES)
    t_left = run_hemi_dp(ev_left, freq_est)
    t_right = run_hemi_dp(ev_right, freq_est)
    f_left = ipi_freq_from_times(t_left)
    f_right = ipi_freq_from_times(t_right)
    return t_left, t_right, f_left, f_right


def method_A1(seg_cache):
    """A1: Average of left and right hemi IPI freq."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    tl, tr, fl, fr = _hemi_freqs(seg_cache)
    freqs = [f for f in [fl, fr] if np.isfinite(f)]
    if freqs:
        freq = np.mean(freqs)
    else:
        freq = seg_cache['baseline_freq']
    combined = np.sort(np.concatenate([tl, tr]))
    return combined, freq


def method_A2(seg_cache):
    """A2: Take freq from hemisphere with more discharges."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    tl, tr, fl, fr = _hemi_freqs(seg_cache)
    if len(tl) >= len(tr):
        return tl, fl if np.isfinite(fl) else seg_cache['baseline_freq']
    else:
        return tr, fr if np.isfinite(fr) else seg_cache['baseline_freq']


def method_A3(seg_cache):
    """A3: Pool discharge times from both hemispheres, compute IPI on combined."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    tl, tr, fl, fr = _hemi_freqs(seg_cache)
    combined = np.sort(np.concatenate([tl, tr]))
    # Deduplicate: remove times within 50ms of each other
    if len(combined) > 1:
        deduped = [combined[0]]
        for t in combined[1:]:
            if (t - deduped[-1]) > 0.05 * FS:
                deduped.append(t)
        combined = np.array(deduped)
    freq = ipi_freq_from_times(combined)
    if not np.isfinite(freq):
        freq = seg_cache['baseline_freq']
    return combined, freq


def method_A4(seg_cache):
    """A4: Weight each hemi's freq by its mean CNN channel probability."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    tl, tr, fl, fr = _hemi_freqs(seg_cache)
    probs = seg_cache['channel_probs']
    wl = np.mean(probs[LEFT_INDICES])
    wr = np.mean(probs[RIGHT_INDICES])
    freqs, weights = [], []
    if np.isfinite(fl):
        freqs.append(fl); weights.append(wl)
    if np.isfinite(fr):
        freqs.append(fr); weights.append(wr)
    if freqs:
        freq = np.average(freqs, weights=weights)
    else:
        freq = seg_cache['baseline_freq']
    combined = np.sort(np.concatenate([tl, tr]))
    return combined, freq


def method_A5(seg_cache):
    """A5: Take hemi with highest mean evidence value."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    freq_est = get_baseline_freq_est(seg_cache)
    ev_left = get_hemi_evidence(seg_cache, LEFT_INDICES)
    ev_right = get_hemi_evidence(seg_cache, RIGHT_INDICES)
    if np.mean(ev_left) >= np.mean(ev_right):
        times = run_hemi_dp(ev_left, freq_est)
    else:
        times = run_hemi_dp(ev_right, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = seg_cache['baseline_freq']
    return times, freq


# ── B: Frequency prior strategies ──

def method_B1(seg_cache):
    """B1: Pure ACF as freq prior."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    freq_est = seg_cache['acf_freq']
    if not np.isfinite(freq_est):
        freq_est = seg_cache['cnn_freq']
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_B2(seg_cache):
    """B2: Pure CNN as freq prior."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    freq_est = seg_cache['cnn_freq']
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_B3(seg_cache):
    """B3: Median ACF across all 18 channels individually."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    seg_bi = seg_cache['seg_bi']
    b, a = butter(4, 20 / (FS / 2), btype='low')
    ch_freqs = []
    for ch in range(min(18, seg_bi.shape[0])):
        try:
            sig = filtfilt(b, a, seg_bi[ch])
            pt = compute_pointiness_trace(sig, FS)
            f = compute_acf_frequency(pt, FS)
            if np.isfinite(f) and 0.2 < f < 5.0:
                ch_freqs.append(f)
        except Exception:
            pass
    if ch_freqs:
        freq_est = float(np.median(ch_freqs))
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_B4(seg_cache):
    """B4: Welch PSD peak in [0.3, 3.0] Hz on combined evidence."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    evidence = get_full_evidence(seg_cache)
    freqs_w, psd = welch(evidence, fs=FS, nperseg=min(512, len(evidence)))
    mask = (freqs_w >= 0.3) & (freqs_w <= 3.0)
    if np.any(mask):
        freq_est = float(freqs_w[mask][np.argmax(psd[mask])])
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_B5(seg_cache):
    """B5: ACF on top 4 channels by CNN probability."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    seg_bi = seg_cache['seg_bi']
    probs = seg_cache['channel_probs']
    top_k = np.argsort(probs)[-4:]
    b, a = butter(4, 20 / (FS / 2), btype='low')
    ch_freqs = []
    for ch in top_k:
        if ch >= seg_bi.shape[0]:
            continue
        try:
            sig = filtfilt(b, a, seg_bi[ch])
            pt = compute_pointiness_trace(sig, FS)
            f = compute_acf_frequency(pt, FS)
            if np.isfinite(f) and 0.2 < f < 5.0:
                ch_freqs.append(f)
        except Exception:
            pass
    if ch_freqs:
        freq_est = float(np.median(ch_freqs))
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


# ── C: Evidence aggregation strategies ──

def _custom_evidence(seg_cache, agg_fn):
    """Compute evidence with a custom channel aggregation function."""
    hpp_all = seg_cache['hpp_all']
    cet_all = seg_cache['cet_all']
    hpp_agg = agg_fn(hpp_all, seg_cache)
    cet_agg = agg_fn(cet_all, seg_cache)
    return combine_evidence(hpp_agg, cet_agg)


def method_C1(seg_cache):
    """C1: np.median instead of np.average across channels."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    def agg(arr, sc):
        return np.median(arr[:min(18, arr.shape[0])], axis=0)
    evidence = _custom_evidence(seg_cache, agg)
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_C2(seg_cache):
    """C2: Only top 6 channels by CNN prob."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    probs = seg_cache['channel_probs']
    top6 = np.argsort(probs)[-6:]
    def agg(arr, sc):
        w = np.clip(sc['channel_probs'][top6], 0.05, None)
        return np.average(arr[top6], weights=w, axis=0)
    evidence = _custom_evidence(seg_cache, agg)
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_C3(seg_cache):
    """C3: Equal weights (no CNN weighting)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    def agg(arr, sc):
        return np.mean(arr[:min(18, arr.shape[0])], axis=0)
    evidence = _custom_evidence(seg_cache, agg)
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_C4(seg_cache):
    """C4: np.max across channels."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    def agg(arr, sc):
        return np.max(arr[:min(18, arr.shape[0])], axis=0)
    evidence = _custom_evidence(seg_cache, agg)
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_C5(seg_cache):
    """C5: Softmax of CNN probs as weights."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    def agg(arr, sc):
        p = sc['channel_probs'][:arr.shape[0]]
        # Softmax with temperature
        e = np.exp(p - np.max(p))
        w = e / e.sum()
        return np.average(arr[:len(w)], weights=w, axis=0)
    evidence = _custom_evidence(seg_cache, agg)
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


# ── D: DP parameter strategies ──

def method_D1(seg_cache):
    """D1: DP_ALPHA = 0.8 (more lenient timing)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    orig_alpha = dd.DP_ALPHA
    try:
        dd.DP_ALPHA = 0.8
        evidence = get_full_evidence(seg_cache)
        freq_est = get_baseline_freq_est(seg_cache)
        times = run_standard_dp(evidence, freq_est)
        freq = ipi_freq_from_times(times)
        if not np.isfinite(freq):
            freq = float(np.clip(freq_est, 0.3, 3.5))
        return times, freq
    finally:
        dd.DP_ALPHA = orig_alpha


def method_D2(seg_cache):
    """D2: DP_ALPHA = 2.0 (stricter timing)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    orig_alpha = dd.DP_ALPHA
    try:
        dd.DP_ALPHA = 2.0
        evidence = get_full_evidence(seg_cache)
        freq_est = get_baseline_freq_est(seg_cache)
        times = run_standard_dp(evidence, freq_est)
        freq = ipi_freq_from_times(times)
        if not np.isfinite(freq):
            freq = float(np.clip(freq_est, 0.3, 3.5))
        return times, freq
    finally:
        dd.DP_ALPHA = orig_alpha


def method_D3(seg_cache):
    """D3: freq_est clipped to [0.2, 4.0] instead of [0.3, 3.5]."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    cnn = seg_cache['cnn_freq']
    acf = seg_cache['acf_freq']
    if np.isfinite(acf):
        freq_est = 0.8 * cnn + 0.2 * acf
    else:
        freq_est = cnn
    freq_est = float(np.clip(freq_est, 0.2, 4.0))
    evidence = get_full_evidence(seg_cache)
    # Run DP manually with wider clip
    active_start, active_end = detect_active_interval(evidence, FS)
    cands = extract_candidates(evidence, FS, freq_est, active_start, active_end)
    times = dp_best_sequence(cands, evidence, FS, freq_est)
    if len(times) >= 3:
        times = em_refine(evidence, times, FS, freq_est)
    times = posthoc_filter(times, evidence)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = freq_est
    return times, freq


def method_D4(seg_cache):
    """D4: MAX_SKIP = 5."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    orig_skip = dd.MAX_SKIP
    try:
        dd.MAX_SKIP = 5
        evidence = get_full_evidence(seg_cache)
        freq_est = get_baseline_freq_est(seg_cache)
        times = run_standard_dp(evidence, freq_est)
        freq = ipi_freq_from_times(times)
        if not np.isfinite(freq):
            freq = float(np.clip(freq_est, 0.3, 3.5))
        return times, freq
    finally:
        dd.MAX_SKIP = orig_skip


def method_D5(seg_cache):
    """D5: Raise active interval threshold — zero out bottom 40% of evidence."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    evidence = get_full_evidence(seg_cache).copy()
    threshold = np.percentile(evidence, 40)
    evidence[evidence < threshold] = 0.0
    freq_est = get_baseline_freq_est(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


# ── E: Hybrid strategies ──

def method_E1(seg_cache):
    """E1: Alexandra's freq as prior, standard evidence + DP."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    freq_est = seg_cache['alex_freq']
    if not np.isfinite(freq_est) or freq_est < 0.2 or freq_est > 5.0:
        freq_est = get_baseline_freq_est(seg_cache)
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_E2(seg_cache):
    """E2: A1 (both hemi avg) + D1 (relaxed alpha=0.8)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    orig_alpha = dd.DP_ALPHA
    try:
        dd.DP_ALPHA = 0.8
        freq_est = get_baseline_freq_est(seg_cache)
        ev_left = get_hemi_evidence(seg_cache, LEFT_INDICES)
        ev_right = get_hemi_evidence(seg_cache, RIGHT_INDICES)
        tl = run_hemi_dp(ev_left, freq_est)
        tr = run_hemi_dp(ev_right, freq_est)
        fl = ipi_freq_from_times(tl)
        fr = ipi_freq_from_times(tr)
        freqs = [f for f in [fl, fr] if np.isfinite(f)]
        if freqs:
            freq = np.mean(freqs)
        else:
            freq = seg_cache['baseline_freq']
        combined = np.sort(np.concatenate([tl, tr]))
        return combined, freq
    finally:
        dd.DP_ALPHA = orig_alpha


def method_E3(seg_cache):
    """E3: B5 (ACF top-K prior) + C2 (top 6 channels evidence)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    # ACF on top 4 channels for freq prior
    seg_bi = seg_cache['seg_bi']
    probs = seg_cache['channel_probs']
    top4 = np.argsort(probs)[-4:]
    b, a = butter(4, 20 / (FS / 2), btype='low')
    ch_freqs = []
    for ch in top4:
        if ch >= seg_bi.shape[0]:
            continue
        try:
            sig = filtfilt(b, a, seg_bi[ch])
            pt = compute_pointiness_trace(sig, FS)
            f = compute_acf_frequency(pt, FS)
            if np.isfinite(f) and 0.2 < f < 5.0:
                ch_freqs.append(f)
        except Exception:
            pass
    if ch_freqs:
        freq_est = float(np.median(ch_freqs))
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    # Top 6 channels evidence
    top6 = np.argsort(probs)[-6:]
    hpp_all = seg_cache['hpp_all']
    cet_all = seg_cache['cet_all']
    w = np.clip(probs[top6], 0.05, None)
    hpp_agg = np.average(hpp_all[top6], weights=w, axis=0)
    cet_agg = np.average(cet_all[top6], weights=w, axis=0)
    evidence = combine_evidence(hpp_agg, cet_agg)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_E4(seg_cache):
    """E4: Average of cnn_freq, acf_freq, and baseline IPI freq."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    vals = []
    for v in [seg_cache['cnn_freq'], seg_cache['acf_freq'],
              seg_cache['baseline_freq']]:
        if np.isfinite(v):
            vals.append(v)
    if vals:
        freq_est = float(np.mean(vals))
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    evidence = get_full_evidence(seg_cache)
    times = run_standard_dp(evidence, freq_est)
    freq = ipi_freq_from_times(times)
    if not np.isfinite(freq):
        freq = float(np.clip(freq_est, 0.3, 3.5))
    return times, freq


def method_E5(seg_cache):
    """E5: A1 (both hemi) + B4 (spectral prior from Welch PSD)."""
    if seg_cache['subtype'] != 'gpd':
        return _baseline_lpd(seg_cache)
    # Spectral freq prior from full evidence
    evidence_full = get_full_evidence(seg_cache)
    freqs_w, psd = welch(evidence_full, fs=FS, nperseg=min(512, len(evidence_full)))
    mask = (freqs_w >= 0.3) & (freqs_w <= 3.0)
    if np.any(mask):
        freq_est = float(freqs_w[mask][np.argmax(psd[mask])])
    else:
        freq_est = get_baseline_freq_est(seg_cache)
    # Hemisphere DP
    ev_left = get_hemi_evidence(seg_cache, LEFT_INDICES)
    ev_right = get_hemi_evidence(seg_cache, RIGHT_INDICES)
    tl = run_hemi_dp(ev_left, freq_est)
    tr = run_hemi_dp(ev_right, freq_est)
    fl = ipi_freq_from_times(tl)
    fr = ipi_freq_from_times(tr)
    freqs = [f for f in [fl, fr] if np.isfinite(f)]
    if freqs:
        freq = np.mean(freqs)
    else:
        freq = seg_cache['baseline_freq']
    combined = np.sort(np.concatenate([tl, tr]))
    return combined, freq


# ══════════════════════════════════════════════════════════════════════════════
# Method registry
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    'Baseline':      None,  # special: uses cached baseline
    'A1_HemiAvg':    method_A1,
    'A2_HemiMore':   method_A2,
    'A3_HemiPool':   method_A3,
    'A4_HemiCNNWt':  method_A4,
    'A5_HemiBestEv': method_A5,
    'B1_ACFPrior':   method_B1,
    'B2_CNNPrior':   method_B2,
    'B3_MedianACF':  method_B3,
    'B4_WelchPrior': method_B4,
    'B5_ACFTopK':    method_B5,
    'C1_MedianEv':   method_C1,
    'C2_Top6Ch':     method_C2,
    'C3_EqualWt':    method_C3,
    'C4_MaxEv':      method_C4,
    'C5_SoftmaxWt':  method_C5,
    'D1_Alpha08':    method_D1,
    'D2_Alpha20':    method_D2,
    'D3_WideClip':   method_D3,
    'D4_Skip5':      method_D4,
    'D5_ThreshEv':   method_D5,
    'E1_AlexPrior':  method_E1,
    'E2_HemiAlpha':  method_E2,
    'E3_ACFTop6':    method_E3,
    'E4_AvgFreqs':   method_E4,
    'E5_HemiWelch':  method_E5,
}


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_method(method_name, method_fn, cache_list, expert_freq):
    """Run a method on all segments and compute metrics."""
    results = {'lpd': [], 'gpd': []}
    per_segment = []

    for i, seg in enumerate(cache_list):
        sid = seg['mat_file'].replace('.mat', '')
        sub = seg['subtype']

        if sid not in expert_freq:
            continue

        gt_freq = expert_freq[sid]
        if not np.isfinite(gt_freq) or gt_freq <= 0:
            continue

        try:
            if method_fn is None:
                # Baseline
                pred_freq = seg['baseline_freq']
            else:
                _, pred_freq = method_fn(seg)

            if not np.isfinite(pred_freq) or pred_freq <= 0:
                pred_freq = np.nan
        except Exception as e:
            print(f"  ERROR {method_name} on {sid}: {e}")
            pred_freq = np.nan

        per_segment.append({
            'segment_id': sid,
            'subtype': sub,
            'gt_freq': gt_freq,
            'pred_freq': pred_freq,
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
            mae = np.mean(np.abs(np.array(gt) - np.array(pr)))
            metrics[sub] = {
                'n': len(pairs),
                'rho': float(rho) if np.isfinite(rho) else 0.0,
                'pval': float(pval) if np.isfinite(pval) else 1.0,
                'mae': float(mae),
            }
        else:
            metrics[sub] = {'n': len(pairs), 'rho': 0.0, 'pval': 1.0, 'mae': 999.0}

    return metrics, per_segment


# ══════════════════════════════════════════════════════════════════════════════
# Leaderboard HTML
# ══════════════════════════════════════════════════════════════════════════════

def generate_leaderboard(all_results):
    """Generate leaderboard HTML from all method results."""
    rows = []
    for name, metrics in all_results.items():
        gpd = metrics.get('gpd', {})
        lpd = metrics.get('lpd', {})
        rows.append({
            'name': name,
            'gpd_rho': gpd.get('rho', 0.0),
            'gpd_mae': gpd.get('mae', 999.0),
            'gpd_n': gpd.get('n', 0),
            'lpd_rho': lpd.get('rho', 0.0),
            'lpd_mae': lpd.get('mae', 999.0),
            'lpd_n': lpd.get('n', 0),
        })

    # Sort: baseline always first, then by GPD rho descending
    baseline = [r for r in rows if r['name'] == 'Baseline']
    others = sorted([r for r in rows if r['name'] != 'Baseline'],
                    key=lambda x: -x['gpd_rho'])

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="10">
<title>GPD Frequency Contest Leaderboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 20px; background: #f5f5f5; }}
h1 {{ color: #333; }}
.timestamp {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
table {{ border-collapse: collapse; width: 100%; background: white;
         box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
th {{ background: #2c3e50; color: white; padding: 10px 14px; text-align: left;
      font-size: 13px; }}
td {{ padding: 8px 14px; border-bottom: 1px solid #eee; font-size: 13px; }}
tr:hover {{ background: #f8f9fa; }}
tr.baseline {{ background: #e8f4fd; font-weight: bold; }}
tr.baseline:hover {{ background: #d0eafc; }}
.green {{ color: #27ae60; font-weight: bold; }}
.yellow {{ color: #f39c12; font-weight: bold; }}
.red {{ color: #e74c3c; }}
.flag {{ color: #e74c3c; font-size: 11px; }}
.rank {{ color: #999; }}
.method-group {{ color: #888; font-size: 11px; }}
</style>
</head>
<body>
<h1>GPD Frequency Contest Leaderboard</h1>
<p class="timestamp">Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refreshes every 10s</p>
<p>{len(all_results)} / {len(METHODS)} methods complete</p>
<table>
<tr>
  <th>#</th>
  <th>Method</th>
  <th>GPD rho</th>
  <th>GPD MAE</th>
  <th>GPD N</th>
  <th>LPD rho</th>
  <th>LPD MAE</th>
  <th>LPD N</th>
  <th>Notes</th>
</tr>
"""
    def row_html(r, rank, is_baseline=False):
        cls = ' class="baseline"' if is_baseline else ''
        gpd_rho = r['gpd_rho']
        if gpd_rho > 0.4:
            rho_cls = 'green'
        elif gpd_rho > 0.25:
            rho_cls = 'yellow'
        else:
            rho_cls = 'red'
        lpd_flag = ' <span class="flag">[LPD degraded]</span>' if r['lpd_rho'] < 0.65 and r['lpd_n'] >= 5 else ''
        notes = ''
        if is_baseline:
            notes = 'Baseline'
        return f"""<tr{cls}>
  <td class="rank">{rank}</td>
  <td>{r['name']}</td>
  <td class="{rho_cls}">{gpd_rho:.3f}</td>
  <td>{r['gpd_mae']:.3f}</td>
  <td>{r['gpd_n']}</td>
  <td>{r['lpd_rho']:.3f}</td>
  <td>{r['lpd_mae']:.3f}</td>
  <td>{r['lpd_n']}</td>
  <td>{notes}{lpd_flag}</td>
</tr>
"""

    rank = 0
    for r in baseline:
        rank += 1
        html += row_html(r, rank, is_baseline=True)
    for r in others:
        rank += 1
        html += row_html(r, rank)

    html += """</table>
</body>
</html>
"""
    out_path = RESULTS_DIR / 'leaderboard.html'
    with open(out_path, 'w') as f:
        f.write(html)
    print(f"  Leaderboard saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GPD Frequency Contest — 25 methods")
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

    # Precompute
    print("\nPrecomputing per-segment data...")
    pc = PDCharacterizer()
    cache_list = []
    for i, row in enumerate(seg_list):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Precomputing {i+1}/{len(seg_list)}: {row['mat_file']}")
        try:
            sc = precompute_segment(row['mat_file'], row['subtype'], pc)
            cache_list.append(sc)
        except Exception as e:
            print(f"  SKIP {row['mat_file']}: {e}")
    print(f"  Cached {len(cache_list)} segments "
          f"({sum(1 for s in cache_list if s['subtype']=='gpd')} GPD, "
          f"{sum(1 for s in cache_list if s['subtype']=='lpd')} LPD)")
    precompute_time = time.time() - t0
    print(f"  Precompute time: {precompute_time:.1f}s")

    # Run all methods
    all_results = {}
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
        print(f"  GPD: rho={gpd.get('rho', 0):.3f}, MAE={gpd.get('mae', 999):.3f}, "
              f"n={gpd.get('n', 0)}")
        print(f"  LPD: rho={lpd.get('rho', 0):.3f}, MAE={lpd.get('mae', 999):.3f}, "
              f"n={lpd.get('n', 0)}")
        print(f"  Time: {elapsed:.1f}s")

        # Save per-method results
        result_data = {
            'method': method_name,
            'metrics': metrics,
            'per_segment': per_seg,
            'timestamp': datetime.now().isoformat(),
            'elapsed_s': elapsed,
        }
        with open(RESULTS_DIR / f'{method_name}.json', 'w') as f:
            json.dump(result_data, f, indent=2, default=str)

        all_results[method_name] = metrics
        generate_leaderboard(all_results)

    # Final summary
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"DONE. Total time: {total_time:.1f}s "
          f"(precompute: {precompute_time:.1f}s)")
    print(f"Results: {RESULTS_DIR}")
    print(f"Leaderboard: {RESULTS_DIR / 'leaderboard.html'}")

    # Print final ranking
    print("\nFinal GPD Frequency Ranking:")
    ranked = sorted(all_results.items(),
                    key=lambda x: -x[1].get('gpd', {}).get('rho', 0))
    for i, (name, m) in enumerate(ranked):
        gpd = m.get('gpd', {})
        print(f"  {i+1:2d}. {name:20s}  rho={gpd.get('rho', 0):.3f}  "
              f"MAE={gpd.get('mae', 999):.3f}")


if __name__ == '__main__':
    main()
