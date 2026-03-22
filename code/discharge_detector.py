"""
Automated Periodic Discharge Detection from Raw EEG

Self-contained implementation of the best-performing method for detecting
periodic discharges (LPD/GPD) in 10-second EEG segments. No gold standard
labels are used as input — the system operates from raw EEG only.

Method: product-boosted max(HPP, CET-UNet) + CNN+ACF frequency
        + optimized DP + post-hoc filtering
Performance: F1≈0.74, evaluated on 593 patients with expert-reviewed labels

Architecture overview:
    EEG (18ch × 2000 samples @ 200 Hz)
        │
        ├─ CNN+Attention (5-fold ensemble)
        │    → per-channel PD probability + log-frequency
        │    → PD-weighted CNN frequency estimate
        │
        ├─ ACF on pointiness traces (per channel)
        │    → ACF frequency estimate
        │    → Ensemble: 0.8×CNN + 0.2×ACF
        │
        ├─ Handcrafted evidence (per channel):
        │    pointiness trace (prominence²/width at peaks)
        │    + TKEO on 20 Hz lowpassed signal
        │    → weighted combination → Gaussian smooth → E_hpp(t)
        │
        └─ CET-UNet (5-fold ensemble, per channel):
             → frame-level discharge evidence E_cet(t)
        │
        ├─ Aggregate channels (median, laterality-aware)
        └─ Combine: threshold CET (80th pct), floor (0.3),
                    product-boost: max(HPP,CET) + 3×HPP×CET
                │
                └─ Active interval detection
                   → candidate peak extraction (min height 5% of max)
                   → forward DP with approximately-periodic prior
                      (α=1.275, β=0.3, λ=0.05, max_skip=3)
                   → EM template refinement
                   → post-hoc filter (drop peaks < 0.3× median)
                   → per-channel timing (±50ms)
                │
                Output: discharge times, IPI-derived frequency

Dependencies: numpy, scipy, torch
Model weights: data/cet_cache/cet_unet_fold{0-4}.pt
               data/pd_channel_cache/cnn_attn_fold{0-4}.pt

Usage:
    from discharge_detector import DischargeDetector
    detector = DischargeDetector()
    result = detector.detect(eeg_18ch, subtype='lpd', laterality='left')
    # result['global_times'] = [0.42, 1.18, 1.95, 2.71, ...]
    # result['frequency'] = 1.31  (Hz, from IPI)
    # result['freq_estimate_input'] = 1.28  (Hz, CNN estimate)

Authors: Westover Lab, Massachusetts General Hospital
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# ── Path setup ────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

# ── Constants ─────────────────────────────────────────────────────────
FS = 200  # sampling rate (Hz)

# Channel indices in 18-channel bipolar montage (double banana)
#   0-3:   Fp1-F7, F7-T3, T3-T5, T5-O1     (left temporal chain)
#   4-7:   Fp2-F8, F8-T4, T4-T6, T6-O2     (right temporal chain)
#   8-11:  Fp1-F3, F3-C3, C3-P3, P3-O1     (left parasagittal chain)
#   12-15: Fp2-F4, F4-C4, C4-P4, P4-O2     (right parasagittal chain)
#   16-17: Fz-Cz, Cz-Pz                     (midline)
LEFT_INDICES = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_INDICES = np.array([4, 5, 6, 7, 12, 13, 14, 15])

# Handcrafted evidence parameters
LOWPASS_HZ = 20.0
POINTINESS_WEIGHT = 0.6
TKEO_WEIGHT = 0.4
SMOOTH_SIGMA_SAMPLES = 3  # ~15 ms at 200 Hz

# Active interval detection
ROLLING_WINDOW_S = 1.0
ACTIVE_THRESHOLD_FRAC = 0.5
MIN_ACTIVE_SECONDS = 3.0
ACTIVE_EXPAND_S = 0.5

# DP parameters (optimized via parameter sweep)
DP_ALPHA = 1.275     # timing deviation penalty (loosened for noisy freq)
DP_BETA = 0.3        # skip penalty
DP_LAMBDA = 0.05     # new-sequence cost
PEAK_HEIGHT_FRAC = 0.05
MAX_SKIP = 3

# EM refinement
TEMPLATE_HALF_MS = 150  # ±150 ms for template extraction
CHANNEL_REFINE_MS = 50  # ±50 ms for per-channel refinement

# Evaluation
TOLERANCE_S = 0.1  # ±100ms for discharge matching


# ═══════════════════════════════════════════════════════════════════════
# Neural network architectures
# ═══════════════════════════════════════════════════════════════════════

class ChannelPDNetAttention(nn.Module):
    """1D CNN with temporal attention for PD detection + frequency estimation.

    Input:  (batch, 1, 2000) — one z-scored EEG channel
    Output: pd_prob    (batch, 1) — PD probability [0, 1]
            freq_pred  (batch, 1) — log frequency (Hz)
            attn_wts   (batch, 1, T) — attention weights over time

    Architecture:
        4 conv blocks (1→16→32→64→64), each: Conv1d(stride=2) → BN → GELU → Dropout
        Attention: Conv1d(64→1) → softmax → weighted pooling → (64,)
        PD head: Linear(64→1) → sigmoid
        Freq head: Linear(64→1)
    """

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16), nn.GELU(), nn.Dropout(0.1))
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32), nn.GELU(), nn.Dropout(0.1))
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1))
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2))
        self.attn_conv = nn.Conv1d(64, 1, kernel_size=1)
        self.pd_head = nn.Linear(64, 1)
        self.freq_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        attn_logits = self.attn_conv(x)
        attn_wts = torch.softmax(attn_logits, dim=-1)
        pooled = (x * attn_wts).sum(dim=-1)
        pd_prob = torch.sigmoid(self.pd_head(pooled))
        freq_pred = self.freq_head(pooled)
        return pd_prob, freq_pred, attn_wts


class CETUNet(nn.Module):
    """U-Net for frame-level discharge evidence at full temporal resolution.

    Input:  (batch, 1, 2000) — one z-scored EEG channel
    Output: (batch, 1, 2000) — discharge evidence in [0, 1]

    Architecture:
        Encoder: 4 conv blocks with stride=2 downsampling
            e1: 1→16  (2000→1000)  k=51
            e2: 16→32 (1000→500)   k=25
            e3: 32→64 (500→250)    k=13
            e4: 64→64 (250→125)    k=7

        Decoder: 4 ConvTranspose blocks with skip connections from encoder
            d4: 64→64 (125→250), cat(d4, e3) → Conv(128→64)
            d3: 64→32 (250→500), cat(d3, e2) → Conv(64→32)
            d2: 32→16 (500→1000), cat(d2, e1) → Conv(32→16)
            d1: 16→8  (1000→2000), Conv(8→1) → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16), nn.GELU())
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32), nn.GELU())
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64), nn.GELU())
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.GELU())

        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.GELU())
        self.skip4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU())

        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32), nn.GELU())
        self.skip3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU())

        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16), nn.GELU())
        self.skip2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.GELU())

        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8), nn.GELU())
        self.head = nn.Sequential(nn.Conv1d(8, 1, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def _match_size(dec, enc):
        min_len = min(dec.shape[2], enc.shape[2])
        return dec[:, :, :min_len], enc[:, :, :min_len]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.up4(e4)
        d4, e3m = self._match_size(d4, e3)
        d4 = self.skip4(torch.cat([d4, e3m], dim=1))

        d3 = self.up3(d4)
        d3, e2m = self._match_size(d3, e2)
        d3 = self.skip3(torch.cat([d3, e2m], dim=1))

        d2 = self.up2(d3)
        d2, e1m = self._match_size(d2, e1)
        d2 = self.skip2(torch.cat([d2, e1m], dim=1))

        d1 = self.up1(d2)
        return self.head(d1)


# ═══════════════════════════════════════════════════════════════════════
# Signal processing: handcrafted evidence
# ═══════════════════════════════════════════════════════════════════════

def compute_pointiness_trace(signal_1d, half_win=8):
    """Compute pointiness = prominence² / width at each local maximum.

    For each local max in the signal, computes:
        prominence = peak_value - max(left_valley, right_valley)
        width = number of samples above half-prominence level
        pointiness = prominence² / width

    Returns a trace that is zero everywhere except at local maxima.
    """
    n = len(signal_1d)
    trace = np.zeros(n)
    peaks, _ = find_peaks(signal_1d)
    for loc in peaks:
        if loc < half_win or loc >= n - half_win:
            continue
        peak_val = signal_1d[loc]
        left_valley = np.min(signal_1d[loc - half_win:loc])
        right_valley = np.min(signal_1d[loc + 1:loc + half_win + 1])
        prom = peak_val - max(left_valley, right_valley)
        if prom <= 0:
            continue
        half_prom_level = peak_val - 0.5 * prom
        width = 0
        for j in range(1, half_win + 1):
            if signal_1d[loc - j] > half_prom_level:
                width += 1
            else:
                break
        for j in range(1, half_win + 1):
            if loc + j < n and signal_1d[loc + j] > half_prom_level:
                width += 1
            else:
                break
        if width > 0:
            trace[loc] = prom ** 2 / width
    return trace


def compute_channel_evidence(signal_1d, fs=FS):
    """Compute handcrafted discharge evidence E_hpp(t) for one channel.

    Steps:
        1. Pointiness trace on raw signal
        2. TKEO (Teager-Kaiser Energy Operator) on 20 Hz lowpassed signal
        3. Z-score normalize both within the window
        4. Combine: 0.6 × pointiness_z + 0.4 × tkeo_z
        5. Gaussian smooth (σ=3 samples ≈ 15ms)
        6. Clip negatives to zero
    """
    n = len(signal_1d)
    if n < 10:
        return np.zeros(n)

    pt = compute_pointiness_trace(signal_1d)

    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    try:
        sig_lp = filtfilt(b_lp, a_lp, signal_1d)
    except ValueError:
        sig_lp = signal_1d.copy()

    tkeo = np.zeros(n)
    if n >= 3:
        tkeo[1:-1] = np.abs(sig_lp[1:-1] ** 2 - sig_lp[:-2] * sig_lp[2:])

    def zscore(x):
        m, s = np.mean(x), np.std(x)
        return (x - m) / s if s > 1e-10 else np.zeros_like(x)

    evidence = POINTINESS_WEIGHT * zscore(pt) + TKEO_WEIGHT * zscore(tkeo)
    evidence = gaussian_filter1d(evidence, sigma=SMOOTH_SIGMA_SAMPLES)
    return np.clip(evidence, 0, None)


# ═══════════════════════════════════════════════════════════════════════
# Evidence aggregation
# ═══════════════════════════════════════════════════════════════════════

def aggregate_evidence(evidence_all, subtype, laterality=None):
    """Aggregate per-channel evidence into a single trace E(t).

    GPD: median across all 18 channels
    LPD with known laterality: median across ipsilateral 8 channels
    LPD without laterality: max(left_median, right_median) at each time point
    """
    if subtype == 'gpd':
        return np.median(evidence_all, axis=0)
    if laterality == 'left':
        return np.median(evidence_all[LEFT_INDICES], axis=0)
    elif laterality == 'right':
        return np.median(evidence_all[RIGHT_INDICES], axis=0)
    else:
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)


def combine_evidence(hpp_evidence, cet_evidence, cet_threshold_pct=80,
                     boost_weight=3.0, cet_floor=0.3):
    """Combine handcrafted and CNN evidence with product-boost.

    Steps:
        1. Normalize both traces to [0, 1]
        2. Threshold CET: zero out values below the 80th percentile
           (suppresses CET noise floor that causes false positives)
        3. Product-boost: max(HPP, CET) + boost_weight × HPP × CET
           (amplifies regions where both methods agree)
    """
    hpp_max = np.max(hpp_evidence) if np.max(hpp_evidence) > 0 else 1.0
    cet_max = np.max(cet_evidence) if np.max(cet_evidence) > 0 else 1.0
    hpp_norm = hpp_evidence / hpp_max
    cet_norm = cet_evidence / cet_max

    # Threshold CET to suppress noise floor
    if cet_threshold_pct > 0 and np.any(cet_norm > 0):
        thr = np.percentile(cet_norm[cet_norm > 0], cet_threshold_pct)
        cet_norm = np.where(cet_norm > thr, cet_norm, 0)

    # Floor: zero out very low CET values
    cet_clean = np.where(cet_norm > cet_floor, cet_norm, 0)

    # Product-boost: max + agreement bonus
    base = np.maximum(hpp_norm, cet_clean)
    agreement = hpp_norm * cet_clean
    return base + boost_weight * agreement


# ═══════════════════════════════════════════════════════════════════════
# Dynamic programming for discharge timing
# ═══════════════════════════════════════════════════════════════════════

def detect_active_interval(evidence, fs=FS):
    """Find the longest contiguous interval where the evidence is active.

    Active = rolling mean (1s window) exceeds 50% of the global rolling max.
    Interval is expanded by 0.5s on each side.
    """
    n = len(evidence)
    win = int(ROLLING_WINDOW_S * fs)
    cs = np.cumsum(evidence)
    rolling = np.zeros(n)
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        rolling[i] = (cs[hi - 1] - (cs[lo - 1] if lo > 0 else 0)) / (hi - lo)
    threshold = ACTIVE_THRESHOLD_FRAC * np.max(rolling)
    above = rolling > threshold
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0
    for i in range(n):
        if above[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_len = 0
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len
    if best_len < int(MIN_ACTIVE_SECONDS * fs):
        return 0, n - 1
    expand = int(ACTIVE_EXPAND_S * fs)
    return max(0, best_start - expand), min(n - 1, best_start + best_len - 1 + expand)


def extract_candidates(evidence, fs, freq_estimate, active_start, active_end):
    """Find candidate discharge peaks within the active interval.

    Two rounds of peak detection:
      1. Standard: height > 5% of max, min distance = 0.2 × T
      2. Strong: height > 50% of max, min distance = 0.1 × T (captures obvious peaks)
    Both sets are merged and deduplicated.
    """
    segment = evidence[active_start:active_end + 1]
    if len(segment) < 3:
        return np.array([], dtype=int)
    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(20, int(0.2 * T * fs))
    min_height = PEAK_HEIGHT_FRAC * np.max(segment)
    peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
    strong_height = 0.5 * np.max(segment)
    strong_peaks, _ = find_peaks(segment, height=strong_height,
                                  distance=max(10, int(0.1 * T * fs)))
    peaks = np.unique(np.concatenate([peaks, strong_peaks]))
    return peaks + active_start


def dp_best_sequence(candidates, evidence, fs, freq_estimate):
    """Find optimal discharge sequence via forward dynamic programming.

    Scoring:
      Node score = evidence[candidate]^1.5 (superlinear: favor strong peaks)
      Edge score = -α × (deviation_from_periodic)² - β × (skips - 1)
        where deviation = (dt - m×T) / (m×T) for skip count m
      New-sequence cost = -λ per candidate

    The DP finds the path through candidates that maximizes:
      Σ(node_scores) + Σ(edge_scores) - λ × path_length
    """
    if len(candidates) == 0:
        return np.array([], dtype=int)
    if len(candidates) == 1:
        return candidates.copy()
    T = 1.0 / freq_estimate
    n = len(candidates)
    raw_scores = np.array([evidence[c] for c in candidates])
    node_scores = raw_scores ** 1.5

    best_score = np.full(n, -np.inf)
    best_prev = np.full(n, -1, dtype=int)
    for i in range(n):
        best_score[i] = node_scores[i] - DP_LAMBDA

    for j in range(1, n):
        for i in range(j):
            dt = (candidates[j] - candidates[i]) / fs
            if dt <= 0 or dt > 4 * T:
                continue
            best_edge = -np.inf
            for m in range(1, MAX_SKIP + 1):
                deviation = (dt - m * T) / (m * T)
                interval_score = -DP_ALPHA * deviation ** 2
                skip_penalty = -DP_BETA * (m - 1)
                edge = interval_score + skip_penalty
                if edge > best_edge:
                    best_edge = edge
            total = best_score[i] + best_edge + node_scores[j] - DP_LAMBDA
            if total > best_score[j]:
                best_score[j] = total
                best_prev[j] = i

    path = []
    idx = int(np.argmax(best_score))
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()
    return candidates[np.array(path)]


def em_refine(evidence, discharge_samples, fs, freq_estimate):
    """Refine discharge times using template cross-correlation.

    Steps:
      1. Extract evidence snippets (±150ms) around each detected discharge
      2. Average to create a discharge template
      3. Normalized cross-correlation of template with full evidence trace
      4. Re-run peak detection + DP on the correlation peaks
    """
    n = len(evidence)
    half_win = int(TEMPLATE_HALF_MS / 1000.0 * fs)
    if len(discharge_samples) < 2:
        return discharge_samples

    snippets = []
    for s in discharge_samples:
        lo, hi = s - half_win, s + half_win + 1
        if lo >= 0 and hi <= n:
            snippets.append(evidence[lo:hi])
    if len(snippets) < 2:
        return discharge_samples

    template = np.mean(snippets, axis=0) - np.mean(np.mean(snippets, axis=0))
    t_norm = np.sqrt(np.sum(template ** 2))
    if t_norm < 1e-10:
        return discharge_samples

    corr = np.zeros(n)
    t_len = len(template)
    for i in range(half_win, n - half_win):
        seg = evidence[i - half_win:i + half_win + 1]
        if len(seg) != t_len:
            continue
        seg_c = seg - np.mean(seg)
        s_norm = np.sqrt(np.sum(seg_c ** 2))
        if s_norm < 1e-10:
            continue
        corr[i] = np.dot(template, seg_c) / (t_norm * s_norm)

    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(30, int(0.3 * T * fs))
    min_height = 0.15 * np.max(corr) if np.max(corr) > 0 else 0
    refined_candidates, _ = find_peaks(corr, height=min_height, distance=min_dist)
    if len(refined_candidates) < 2:
        return discharge_samples
    return dp_best_sequence(refined_candidates, evidence, fs, freq_estimate)


def per_channel_times(evidence_all, global_samples, fs=FS):
    """Refine global discharge times per channel within ±50ms.

    For each global discharge, finds the local evidence peak in each
    channel within a ±50ms window.
    """
    refine_win = int(CHANNEL_REFINE_MS / 1000.0 * fs)
    n_channels, n_samples = evidence_all.shape
    channel_times = {}
    for ch in range(n_channels):
        ch_times = []
        for gs in global_samples:
            lo = max(0, gs - refine_win)
            hi = min(n_samples, gs + refine_win + 1)
            window = evidence_all[ch, lo:hi]
            if len(window) == 0:
                ch_times.append(gs / fs)
            else:
                ch_times.append((lo + int(np.argmax(window))) / fs)
        channel_times[ch] = ch_times
    return channel_times


# ═══════════════════════════════════════════════════════════════════════
# Post-hoc confidence filtering
# ═══════════════════════════════════════════════════════════════════════

def posthoc_filter(discharge_samples, evidence, min_evidence_ratio=0.3):
    """Drop detected discharges with evidence below a fraction of the median.

    Removes low-confidence detections where the evidence peak at the
    detected discharge time is less than min_evidence_ratio × median
    peak value across all detections.
    """
    if len(discharge_samples) < 2 or min_evidence_ratio <= 0:
        return discharge_samples
    peak_values = np.array([evidence[int(s)] for s in discharge_samples])
    threshold = min_evidence_ratio * np.median(peak_values)
    mask = peak_values >= threshold
    return discharge_samples[mask]


# ═══════════════════════════════════════════════════════════════════════
# ACF frequency estimation (handcrafted, no neural network)
# ═══════════════════════════════════════════════════════════════════════

def estimate_frequency_acf(signal_1d, fs=FS):
    """Estimate dominant frequency from autocorrelation of pointiness trace.

    Steps:
        1. Compute pointiness trace
        2. Autocorrelation of the pointiness trace
        3. Find first peak in ACF after min lag (0.4s = max 2.5 Hz)
        4. Frequency = 1 / lag_at_peak
    """
    pt = compute_pointiness_trace(signal_1d)
    if np.max(pt) < 1e-10:
        return np.nan

    # Autocorrelation
    pt_centered = pt - np.mean(pt)
    n = len(pt_centered)
    acf = np.correlate(pt_centered, pt_centered, mode='full')[n-1:]
    if acf[0] > 0:
        acf = acf / acf[0]

    # Find first peak after minimum lag
    min_lag = int(0.4 * fs)  # 0.4s = max 2.5 Hz
    max_lag = min(int(3.0 * fs), len(acf) - 1)  # 3.0s = min 0.33 Hz
    if max_lag <= min_lag:
        return np.nan

    segment = acf[min_lag:max_lag + 1]
    peaks, props = find_peaks(segment, height=0.1)
    if len(peaks) == 0:
        return np.nan

    # Take the first (shortest lag) peak
    best_lag = peaks[0] + min_lag
    return fs / best_lag


# ═══════════════════════════════════════════════════════════════════════
# Main detector class
# ═══════════════════════════════════════════════════════════════════════

class DischargeDetector:
    """Automated periodic discharge detector from raw EEG.

    Loads CNN+Attention (frequency estimation) and CET-UNet (evidence)
    model ensembles on initialization, then detects discharges from raw
    18-channel bipolar EEG segments.

    Args:
        cet_model_dir: directory containing cet_unet_fold{0-4}.pt
        cnn_model_dir: directory containing cnn_attn_fold{0-4}.pt
        device: 'cpu', 'cuda', or 'mps'
        n_folds: number of ensemble models to load
    """

    def __init__(self, cet_model_dir=None, cnn_model_dir=None,
                 device=None, n_folds=5):
        if cet_model_dir is None:
            cet_model_dir = PROJECT_DIR / 'data' / 'cet_cache'
        if cnn_model_dir is None:
            cnn_model_dir = PROJECT_DIR / 'data' / 'pd_channel_cache'
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        self.device = device

        self.cet_models = self._load_models(
            CETUNet, cet_model_dir, 'cet_unet_fold', n_folds)
        self.cnn_models = self._load_models(
            ChannelPDNetAttention, cnn_model_dir, 'cnn_attn_fold', n_folds)

    def _load_models(self, model_class, model_dir, prefix, n_folds):
        models = []
        model_dir = Path(model_dir)
        for fold in range(n_folds):
            path = model_dir / f'{prefix}{fold}.pt'
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            model = model_class()
            model.load_state_dict(torch.load(
                str(path), map_location=self.device, weights_only=True))
            model.to(self.device)
            model.eval()
            models.append(model)
        return models

    @torch.no_grad()
    def estimate_frequency(self, segment_18ch):
        """Estimate discharge frequency from raw EEG using CNN+Attention.

        Runs each channel through the 5-fold CNN+Attention ensemble.
        PD probability is used as weight for frequency averaging
        (PD-weighted mean of log-frequency predictions).

        Returns: float, estimated frequency in Hz, clipped to [0.3, 3.5]
        """
        n_channels = min(segment_18ch.shape[0], 18)
        all_pd_probs = []
        all_log_freqs = []

        for ch in range(n_channels):
            ch_data = segment_18ch[ch].astype(np.float32).copy()
            if not np.all(np.isfinite(ch_data)):
                all_pd_probs.append(0.0)
                all_log_freqs.append(0.0)
                continue

            mu, std = np.mean(ch_data), np.std(ch_data)
            ch_data = (ch_data - mu) / std if std > 1e-8 else ch_data - mu
            x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(self.device)

            pd_probs, log_freqs = [], []
            for model in self.cnn_models:
                pd_prob, freq_pred, _ = model(x)
                pd_probs.append(pd_prob.item())
                log_freqs.append(freq_pred.item())

            all_pd_probs.append(np.mean(pd_probs))
            all_log_freqs.append(np.mean(log_freqs))

        pd_weights = np.array(all_pd_probs)
        log_freqs = np.array(all_log_freqs)

        weight_sum = pd_weights.sum()
        if weight_sum > 1e-6:
            weighted_log_freq = np.sum(pd_weights * log_freqs) / weight_sum
        else:
            weighted_log_freq = np.mean(log_freqs)

        return float(np.clip(np.exp(weighted_log_freq), 0.3, 3.5))

    @torch.no_grad()
    def compute_cet_evidence_channel(self, channel_data):
        """Run CET-UNet ensemble on a single channel.

        Returns: numpy array (n_samples,) — evidence in [0, 1]
        """
        x = channel_data.astype(np.float32).copy()
        mu, std = np.mean(x), np.std(x)
        x = (x - mu) / std if std > 1e-8 else x - mu
        x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).to(self.device)
        predictions = []
        for model in self.cet_models:
            pred = model(x_tensor).squeeze().cpu().numpy()
            predictions.append(pred)
        return np.mean(predictions, axis=0).astype(np.float32)

    def estimate_frequency_acf_multichannel(self, segment_18ch, subtype,
                                             laterality=None):
        """Estimate frequency from ACF across multiple channels.

        Computes ACF frequency per channel and takes the median across
        relevant channels (all for GPD, lateralized for LPD).

        Returns: float, estimated frequency in Hz (or NaN if no valid estimate)
        """
        n_channels = min(segment_18ch.shape[0], 18)
        b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

        acf_freqs = []
        for ch in range(n_channels):
            try:
                sig = filtfilt(b_lp, a_lp, segment_18ch[ch])
            except ValueError:
                sig = segment_18ch[ch]
            freq = estimate_frequency_acf(sig, FS)
            if np.isfinite(freq):
                acf_freqs.append(freq)

        if not acf_freqs:
            return np.nan
        return float(np.clip(np.median(acf_freqs), 0.3, 3.5))

    def detect(self, segment_18ch, subtype='lpd', laterality=None, refine=True):
        """Detect periodic discharges in a 10-second EEG segment.

        Args:
            segment_18ch: (18, 2000) bipolar EEG at 200 Hz
            subtype: 'lpd' or 'gpd'
            laterality: 'left', 'right', or None
            refine: whether to apply EM template refinement

        Returns:
            dict with:
                global_times: list of discharge times in seconds
                channel_times: dict ch -> list of times
                frequency: float, IPI-derived frequency (Hz)
                freq_estimate_input: float, CNN+ACF ensemble frequency (Hz)
                n_discharges: int
                active_interval: (start_s, end_s)
        """
        n_channels = min(segment_18ch.shape[0], 18)
        n_samples = segment_18ch.shape[1]
        fs = FS

        # 1. Estimate frequency: CNN+ACF ensemble (0.8 CNN + 0.2 ACF)
        cnn_freq = self.estimate_frequency(segment_18ch)
        acf_freq = self.estimate_frequency_acf_multichannel(
            segment_18ch, subtype, laterality)
        if np.isfinite(acf_freq):
            freq_estimate = 0.8 * cnn_freq + 0.2 * acf_freq
        else:
            freq_estimate = cnn_freq
        freq_estimate = float(np.clip(freq_estimate, 0.3, 3.5))

        # 2. Compute per-channel evidence (both handcrafted and CNN)
        hpp_all = np.zeros((n_channels, n_samples))
        cet_all = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            hpp_all[ch] = compute_channel_evidence(segment_18ch[ch], fs)
            if np.all(np.isfinite(segment_18ch[ch])):
                cet_all[ch] = self.compute_cet_evidence_channel(segment_18ch[ch])

        # 3. Aggregate across channels
        hpp_agg = aggregate_evidence(hpp_all, subtype, laterality)
        cet_agg = aggregate_evidence(cet_all, subtype, laterality)

        # 4. Combine: product-boosted max(HPP, CET) with CET thresholding
        evidence = combine_evidence(hpp_agg, cet_agg)

        # 5. DP inference
        active_start, active_end = detect_active_interval(evidence, fs)
        candidates = extract_candidates(evidence, fs, freq_estimate,
                                        active_start, active_end)
        discharge_samples = dp_best_sequence(candidates, evidence, fs,
                                              freq_estimate)

        # 6. EM refinement
        if refine and len(discharge_samples) >= 3:
            discharge_samples = em_refine(evidence, discharge_samples, fs,
                                           freq_estimate)

        # 7. Post-hoc confidence filter: drop weak detections
        discharge_samples = posthoc_filter(discharge_samples, evidence)

        # 8. Per-channel timing
        channel_times = per_channel_times(hpp_all, discharge_samples, fs)

        # 8. Output
        global_times = (discharge_samples / fs).tolist() if len(discharge_samples) > 0 else []

        if len(global_times) >= 2:
            ipis = np.diff(global_times)
            ipi_freq = 1.0 / float(np.median(ipis))
        else:
            ipi_freq = np.nan

        return {
            'global_times': global_times,
            'channel_times': {int(k): v for k, v in channel_times.items()},
            'frequency': ipi_freq,
            'freq_estimate_input': freq_estimate,
            'n_discharges': len(discharge_samples),
            'active_interval': (active_start / fs, active_end / fs),
        }


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import scipy.io as sio

    print("Discharge Detector — Self-contained implementation")
    print("=" * 60)

    detector = DischargeDetector()
    print(f"Models loaded on {detector.device}")

    # Demo: load first available EEG file
    eeg_dir = PROJECT_DIR / 'data' / 'eeg'
    mat_files = sorted(eeg_dir.glob('*.mat'))
    if mat_files:
        mat = sio.loadmat(str(mat_files[0]))
        data_key = [k for k in mat.keys() if not k.startswith('_')][0]
        seg = mat[data_key]
        if seg.shape[0] > seg.shape[1]:
            seg = seg.T
        seg = seg[:18, :2000]
        print(f"\nProcessing {mat_files[0].name}: {seg.shape}")

        result = detector.detect(seg, subtype='lpd')
        print(f"  Discharges found: {result['n_discharges']}")
        print(f"  Times: {[f'{t:.3f}' for t in result['global_times'][:10]]}")
        print(f"  CNN freq estimate: {result['freq_estimate_input']:.2f} Hz")
        print(f"  IPI frequency: {result['frequency']:.2f} Hz")
    else:
        print("\nNo EEG files found in data/eeg/. Provide a .mat file to test.")
