"""
Fully automated EEG-only discharge detection pipeline.

No gold-standard frequency is used anywhere. The system estimates frequency
from EEG using ACF bootstrap and/or CNN+Attention models.

Methods evaluated:
  1. HPP (handcrafted evidence, bootstrap freq)
  2. CET-UNet+HPP (CNN evidence, bootstrap freq)
  3. HPP (handcrafted evidence, CNN freq)
  4. CET-UNet+HPP (CNN evidence, CNN freq)
  5. HPP (handcrafted evidence, gold freq) — reference only

Usage:
    conda run -n foe_dl python code/cet_model/auto_pipeline.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.cet import CETUNet
from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from pd_pointiness_acf import compute_acf_frequency

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'
CNN_CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# HPP parameters (same as current best)
PEAK_HEIGHT_FRAC = 0.05
DP_ALPHA = 3.0
DP_BETA = 1.0
DP_LAMBDA = 0.02
MAX_SKIP = 3
LOWPASS_HZ = 20.0

# Active interval detection
ROLLING_WINDOW_S = 1.0
ACTIVE_THRESHOLD_FRAC = 0.5
MIN_ACTIVE_SECONDS = 3.0
ACTIVE_EXPAND_S = 0.5

# EM refinement
TEMPLATE_HALF_MS = 150
CHANNEL_REFINE_MS = 50

TOLERANCE_S = 0.1  # +/-100ms for evaluation


# ============================================================================
# Frequency estimation methods (EEG-only, no gold standard)
# ============================================================================

def estimate_frequency_acf(segment_18ch, subtype, laterality=None, fs=FS):
    """Estimate discharge frequency from EEG using ACF on pointiness traces.

    Computes ACF frequency per channel and takes the median across relevant
    channels (all for GPD, lateralized for LPD).

    Returns:
        float: estimated frequency in Hz, clipped to [0.3, 3.5]
    """
    n_channels = min(segment_18ch.shape[0], 18)
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')

    acf_freqs = []
    for ch in range(n_channels):
        try:
            sig = filtfilt(b_lp, a_lp, segment_18ch[ch])
        except ValueError:
            sig = segment_18ch[ch]
        freq, score, _ = compute_acf_frequency(
            sig, fs, method='pointiness',
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3)
        if np.isfinite(freq):
            acf_freqs.append(freq)

    if not acf_freqs:
        return 1.0

    return float(np.clip(np.median(acf_freqs), 0.3, 3.5))


@torch.no_grad()
def estimate_frequency_cnn(segment_18ch, cnn_models, device=DEVICE, fs=FS):
    """Estimate discharge frequency using CNN+Attention models.

    Runs each channel through the 5-fold CNN+Attention ensemble. Uses
    PD probability as weight for frequency averaging (PD-weighted mean
    of log-frequency predictions).

    Args:
        segment_18ch: (18, 2000) EEG segment
        cnn_models: list of ChannelPDNetAttention models
        device: torch device
        fs: sampling rate

    Returns:
        float: estimated frequency in Hz, clipped to [0.3, 3.5]
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

        # z-score normalize
        mu = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 1e-8:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu

        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(device)

        # Ensemble average
        pd_probs = []
        log_freqs = []
        for model in cnn_models:
            pd_prob, freq_pred, _ = model(x)
            pd_probs.append(pd_prob.item())
            log_freqs.append(freq_pred.item())

        all_pd_probs.append(np.mean(pd_probs))
        all_log_freqs.append(np.mean(log_freqs))

    pd_weights = np.array(all_pd_probs)
    log_freqs = np.array(all_log_freqs)

    # PD-weighted average of log-frequency
    weight_sum = pd_weights.sum()
    if weight_sum > 1e-6:
        weighted_log_freq = np.sum(pd_weights * log_freqs) / weight_sum
    else:
        weighted_log_freq = np.mean(log_freqs)

    freq = np.exp(weighted_log_freq)
    return float(np.clip(freq, 0.3, 3.5))


def estimate_frequency_bootstrap(segment_18ch, subtype, laterality=None, fs=FS):
    """Bootstrap frequency estimation: ACF -> HPP -> IPI -> refined freq.

    Steps:
        1. Get rough frequency from ACF (median across channels)
        2. Run HPP with handcrafted evidence + rough freq -> discharge times
        3. If >= 2 discharges found, compute IPI -> refined frequency
        4. Return refined frequency (or ACF freq if bootstrap fails)

    Returns:
        float: estimated frequency in Hz, clipped to [0.3, 3.5]
    """
    # Step 1: ACF frequency
    acf_freq = estimate_frequency_acf(segment_18ch, subtype, laterality, fs)

    # Step 2: Run HPP with ACF freq to get rough discharge times
    n_channels = min(segment_18ch.shape[0], 18)
    n_samples = segment_18ch.shape[1]

    # Compute handcrafted evidence
    from label_pipeline.hpp_discharge_marking import _compute_channel_evidence
    evidence_all = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        evidence_all[ch] = _compute_channel_evidence(segment_18ch[ch], fs)

    # Aggregate evidence
    evidence = _aggregate_evidence(evidence_all, subtype, laterality)

    # Active interval
    active_start, active_end = _detect_active_interval(evidence, fs)

    # Candidates + DP with ACF freq
    candidates = _extract_candidates(evidence, fs, acf_freq, active_start, active_end)
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, acf_freq)

    # Step 3: Compute IPI-based frequency if enough discharges
    if len(discharge_samples) >= 2:
        global_times = discharge_samples / fs
        ipis = np.diff(global_times)
        ipi_median = float(np.median(ipis))
        if ipi_median > 0:
            ipi_freq = 1.0 / ipi_median
            refined_freq = float(np.clip(ipi_freq, 0.3, 3.5))
            return refined_freq

    # Fallback: return ACF frequency
    return acf_freq


# ============================================================================
# HPP components (shared with hpp_discharge_marking.py)
# ============================================================================

def _aggregate_evidence(evidence_all, subtype, laterality=None):
    """Aggregate per-channel evidence into a single E(t)."""
    if subtype == 'gpd':
        return np.median(evidence_all, axis=0)
    if laterality == 'left':
        channels = LEFT_INDICES
    elif laterality == 'right':
        channels = RIGHT_INDICES
    else:
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)
    return np.median(evidence_all[channels], axis=0)


def _detect_active_interval(evidence, fs):
    """Find the longest contiguous interval where rolling mean > threshold."""
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
                best_start = cur_start
                best_len = cur_len
            cur_len = 0
    if cur_len > best_len:
        best_start = cur_start
        best_len = cur_len
    min_samples = int(MIN_ACTIVE_SECONDS * fs)
    if best_len < min_samples:
        return 0, n - 1
    expand = int(ACTIVE_EXPAND_S * fs)
    start = max(0, best_start - expand)
    end = min(n - 1, best_start + best_len - 1 + expand)
    return start, end


def _extract_candidates(evidence, fs, freq_estimate, active_start, active_end):
    """Find local maxima within the active interval."""
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


def _dp_best_sequence(candidates, evidence, fs, freq_estimate):
    """Find optimal discharge sequence via forward DP."""
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
    best_end = int(np.argmax(best_score))
    path = []
    idx = best_end
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()
    return candidates[np.array(path)]


def _em_refine(evidence, discharge_samples, fs, freq_estimate):
    """One iteration of template-based refinement."""
    n = len(evidence)
    half_win = int(TEMPLATE_HALF_MS / 1000.0 * fs)
    if len(discharge_samples) < 2:
        return discharge_samples
    snippets = []
    for s in discharge_samples:
        lo = s - half_win
        hi = s + half_win + 1
        if lo >= 0 and hi <= n:
            snippets.append(evidence[lo:hi])
    if len(snippets) < 2:
        return discharge_samples
    template = np.mean(snippets, axis=0)
    template = template - np.mean(template)
    corr = np.zeros(n)
    t_len = len(template)
    t_norm = np.sqrt(np.sum(template ** 2))
    if t_norm < 1e-10:
        return discharge_samples
    for i in range(half_win, n - half_win):
        seg = evidence[i - half_win:i + half_win + 1]
        if len(seg) != t_len:
            continue
        seg_centered = seg - np.mean(seg)
        s_norm = np.sqrt(np.sum(seg_centered ** 2))
        if s_norm < 1e-10:
            continue
        corr[i] = np.dot(template, seg_centered) / (t_norm * s_norm)
    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(30, int(0.3 * T * fs))
    min_height = 0.15 * np.max(corr) if np.max(corr) > 0 else 0
    refined_candidates, _ = find_peaks(corr, height=min_height, distance=min_dist)
    if len(refined_candidates) < 2:
        return discharge_samples
    return _dp_best_sequence(refined_candidates, evidence, fs, freq_estimate)


def _per_channel_times(evidence_all, global_samples, fs):
    """Refine discharge times per channel within +/- 50ms of global times."""
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
                local_peak = lo + int(np.argmax(window))
                ch_times.append(local_peak / fs)
        channel_times[ch] = ch_times
    return channel_times


# ============================================================================
# CET evidence computation
# ============================================================================

@torch.no_grad()
def compute_cet_evidence(channel_data, cet_models, device=DEVICE):
    """Run CET-UNet ensemble on a single channel."""
    x = channel_data.astype(np.float32).copy()
    mu = np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = (x - mu) / std
    else:
        x = x - mu
    x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).to(device)
    predictions = []
    for model in cet_models:
        pred = model(x_tensor).squeeze().cpu().numpy()
        predictions.append(pred)
    return np.mean(predictions, axis=0).astype(np.float32)


# ============================================================================
# Main detection function — fully automated, no gold freq
# ============================================================================

def detect_discharges_auto(segment_18ch, subtype, laterality=None,
                           evidence_type='hpp', freq_method='bootstrap',
                           cet_models=None, cnn_models=None,
                           fs=FS, refine=True):
    """Fully automated discharge detection — no gold standard frequency.

    Args:
        segment_18ch: (18, 2000) bipolar EEG
        subtype: 'lpd' or 'gpd'
        laterality: 'left', 'right', or None
        evidence_type: 'hpp' (handcrafted) or 'cet' (CET-UNet CNN)
        freq_method: 'acf', 'bootstrap', or 'cnn'
        cet_models: list of CETUNet models (required if evidence_type='cet')
        cnn_models: list of ChannelPDNetAttention models (required if freq_method='cnn')
        fs: sampling rate
        refine: whether to do EM template refinement

    Returns:
        dict with global_times, channel_times, frequency, ipi_cv, etc.
    """
    n_channels = min(segment_18ch.shape[0], 18)
    n_samples = segment_18ch.shape[1]

    # Step 1: Estimate frequency from EEG only
    if freq_method == 'acf':
        freq_estimate = estimate_frequency_acf(segment_18ch, subtype, laterality, fs)
    elif freq_method == 'bootstrap':
        freq_estimate = estimate_frequency_bootstrap(segment_18ch, subtype, laterality, fs)
    elif freq_method == 'cnn':
        if cnn_models is None:
            raise ValueError("cnn_models required for freq_method='cnn'")
        freq_estimate = estimate_frequency_cnn(segment_18ch, cnn_models, DEVICE, fs)
    else:
        raise ValueError(f"Unknown freq_method: {freq_method}")

    # Step 2: Compute evidence
    if evidence_type == 'hpp':
        from label_pipeline.hpp_discharge_marking import _compute_channel_evidence
        evidence_all = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            evidence_all[ch] = _compute_channel_evidence(segment_18ch[ch], fs)
    elif evidence_type == 'cet':
        if cet_models is None:
            raise ValueError("cet_models required for evidence_type='cet'")
        evidence_all = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            ch_data = segment_18ch[ch]
            if not np.all(np.isfinite(ch_data)):
                continue
            evidence_all[ch] = compute_cet_evidence(ch_data, cet_models, DEVICE)
    else:
        raise ValueError(f"Unknown evidence_type: {evidence_type}")

    # Step 3: Aggregate + HPP DP
    evidence = _aggregate_evidence(evidence_all, subtype, laterality)
    active_start, active_end = _detect_active_interval(evidence, fs)
    candidates = _extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)

    # Step 4: EM refinement
    if refine and len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)

    # Step 5: Per-channel timing
    channel_times = _per_channel_times(evidence_all, discharge_samples, fs)

    # Output
    global_times = discharge_samples / fs if len(discharge_samples) > 0 else np.array([])

    if len(global_times) >= 2:
        ipis = np.diff(global_times)
        ipi_median = float(np.median(ipis))
        ipi_freq = 1.0 / ipi_median if ipi_median > 0 else np.nan
        ipi_cv = float(np.std(ipis) / np.mean(ipis)) if np.mean(ipis) > 0 else np.nan
    else:
        ipi_freq = np.nan
        ipi_cv = np.nan

    return {
        'global_times': global_times.tolist() if len(global_times) > 0 else [],
        'channel_times': {int(k): v for k, v in channel_times.items()},
        'frequency': ipi_freq,
        'freq_estimate_input': freq_estimate,
        'ipi_cv': ipi_cv,
        'active_interval': (active_start / fs, active_end / fs),
        'n_discharges': len(discharge_samples),
        'evidence_signal': evidence,
        'candidates': (candidates / fs).tolist() if len(candidates) > 0 else [],
    }


# ============================================================================
# Model loading helpers
# ============================================================================

def load_cet_unet_models(n_folds=5, device=DEVICE):
    """Load ensemble of CET-UNet models from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cet_unet_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CET-UNet model not found: {model_path}")
        model = CETUNet()
        model.load_state_dict(torch.load(str(model_path), map_location=device,
                                          weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
    return models


def load_cnn_attn_models(n_folds=5, device=DEVICE):
    """Load ensemble of CNN+Attention models from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CNN_CACHE_DIR / f'cnn_attn_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CNN+Attention model not found: {model_path}")
        model = ChannelPDNetAttention()
        model.load_state_dict(torch.load(str(model_path), map_location=device,
                                          weights_only=True))
        model.to(device)
        model.eval()
        models.append(model)
    return models


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_method(method_name, df, segments, gt_cases,
                    evidence_type='hpp', freq_method='bootstrap',
                    cet_models=None, cnn_models=None,
                    use_gold_freq=False):
    """Evaluate a detection method on all GT cases.

    Args:
        use_gold_freq: if True, use gold standard freq (reference only)

    Returns dict with sensitivity, precision, F1, freq_spearman, timing_error.
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    match_errors = []

    gt_freqs = []
    algo_freqs = []
    freq_estimates = []

    n_processed = 0
    n_failed = 0

    t_start = time.time()

    for pid, gt_data in gt_cases.items():
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]
        subtype = row['subtype']
        gold_freq = float(row['gold_standard_freq'])
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        try:
            if use_gold_freq:
                # Reference method: use gold standard freq with HPP
                from label_pipeline.hpp_discharge_marking import detect_discharge_times_hpp
                result = detect_discharge_times_hpp(
                    seg, fs=FS, subtype=subtype,
                    freq_estimate=gold_freq, laterality=lat, refine=True)
            else:
                result = detect_discharges_auto(
                    seg, subtype=subtype, laterality=lat,
                    evidence_type=evidence_type, freq_method=freq_method,
                    cet_models=cet_models, cnn_models=cnn_models,
                    fs=FS, refine=True)
        except Exception as e:
            n_failed += 1
            continue

        algo_times = sorted(result['global_times'])
        n_processed += 1

        # Frequency from IPI
        gt_ipis = [gt_times[i+1] - gt_times[i] for i in range(len(gt_times)-1)]
        gt_freq = 1.0 / np.median(gt_ipis)

        if len(algo_times) >= 2:
            algo_ipis = [algo_times[i+1] - algo_times[i]
                         for i in range(len(algo_times)-1)]
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan

        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

        if not use_gold_freq and 'freq_estimate_input' in result:
            freq_estimates.append(result['freq_estimate_input'])

        # Discharge matching
        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)

        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    dist = abs(gt - at)
                    if dist < best_dist:
                        best_dist = dist
                        best_ai = ai
            if best_dist <= TOLERANCE_S and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True
                match_errors.append(best_dist)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

    t_elapsed = time.time() - t_start

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    mean_timing = np.mean(match_errors) * 1000 if match_errors else float('nan')
    median_timing = np.median(match_errors) * 1000 if match_errors else float('nan')

    if len(gt_freqs) >= 3:
        freq_spearman, _ = spearmanr(algo_freqs, gt_freqs)
        freq_mae = np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs)))
    else:
        freq_spearman = float('nan')
        freq_mae = float('nan')

    return {
        'method': method_name,
        'n_cases': n_processed,
        'n_failed': n_failed,
        'sensitivity': sens,
        'precision': prec,
        'f1': f1,
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'freq_spearman': freq_spearman,
        'freq_mae': freq_mae,
        'mean_timing_ms': mean_timing,
        'median_timing_ms': median_timing,
        'eval_time_s': t_elapsed,
    }


def evaluate_all_methods(dataset, gt_cases):
    """Run all methods and print comparison table."""
    df = dataset['df']
    segments = dataset['segments']

    results = []

    # Load models
    print("Loading CET-UNet models...")
    try:
        cet_models = load_cet_unet_models(device=DEVICE)
        print(f"  Loaded {len(cet_models)} CET-UNet fold models on {DEVICE}")
    except FileNotFoundError as e:
        print(f"  CET-UNet models not found: {e}")
        cet_models = None

    print("Loading CNN+Attention models...")
    try:
        cnn_models = load_cnn_attn_models(device=DEVICE)
        print(f"  Loaded {len(cnn_models)} CNN+Attention fold models on {DEVICE}")
    except FileNotFoundError as e:
        print(f"  CNN+Attention models not found: {e}")
        cnn_models = None

    # Method 1: HPP (handcrafted evidence, bootstrap freq)
    print("\n" + "-" * 70)
    print("Method 1: HPP (handcrafted, bootstrap freq) — EEG-only")
    r = evaluate_method('HPP+bootstrap', df, segments, gt_cases,
                        evidence_type='hpp', freq_method='bootstrap')
    results.append(r)
    print(f"  {r['n_cases']} cases | F1={r['f1']:.3f} | FreqRho={r['freq_spearman']:.3f} | "
          f"{r['eval_time_s']:.1f}s")

    # Method 2: CET-UNet+HPP (CNN evidence, bootstrap freq)
    if cet_models is not None:
        print("\n" + "-" * 70)
        print("Method 2: CET-UNet+HPP (CNN evidence, bootstrap freq) — EEG-only")
        r = evaluate_method('CET+bootstrap', df, segments, gt_cases,
                            evidence_type='cet', freq_method='bootstrap',
                            cet_models=cet_models)
        results.append(r)
        print(f"  {r['n_cases']} cases | F1={r['f1']:.3f} | FreqRho={r['freq_spearman']:.3f} | "
              f"{r['eval_time_s']:.1f}s")

    # Method 3: HPP (handcrafted evidence, CNN freq)
    if cnn_models is not None:
        print("\n" + "-" * 70)
        print("Method 3: HPP (handcrafted, CNN freq) — EEG-only")
        r = evaluate_method('HPP+CNN_freq', df, segments, gt_cases,
                            evidence_type='hpp', freq_method='cnn',
                            cnn_models=cnn_models)
        results.append(r)
        print(f"  {r['n_cases']} cases | F1={r['f1']:.3f} | FreqRho={r['freq_spearman']:.3f} | "
              f"{r['eval_time_s']:.1f}s")

    # Method 4: CET-UNet+HPP (CNN evidence, CNN freq)
    if cet_models is not None and cnn_models is not None:
        print("\n" + "-" * 70)
        print("Method 4: CET-UNet+HPP (CNN evidence, CNN freq) — EEG-only")
        r = evaluate_method('CET+CNN_freq', df, segments, gt_cases,
                            evidence_type='cet', freq_method='cnn',
                            cet_models=cet_models, cnn_models=cnn_models)
        results.append(r)
        print(f"  {r['n_cases']} cases | F1={r['f1']:.3f} | FreqRho={r['freq_spearman']:.3f} | "
              f"{r['eval_time_s']:.1f}s")

    # Method 5: HPP (handcrafted, gold freq) — REFERENCE ONLY
    print("\n" + "-" * 70)
    print("Method 5: HPP (handcrafted, GOLD freq) — REFERENCE ONLY, not deployment-ready")
    r = evaluate_method('HPP+gold [REF]', df, segments, gt_cases,
                        use_gold_freq=True)
    results.append(r)
    print(f"  {r['n_cases']} cases | F1={r['f1']:.3f} | FreqRho={r['freq_spearman']:.3f} | "
          f"{r['eval_time_s']:.1f}s")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 74)
    print("  Fully Automated EEG-Only Discharge Detection Pipeline")
    print("  NO gold-standard frequency used (except reference method)")
    print("=" * 74)
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Load GT discharge times
    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}
    print(f"Ground truth cases: {len(gt_cases)}")

    # Run all methods
    results = evaluate_all_methods(dataset, gt_cases)

    # Print comparison table
    print("\n" + "=" * 74)
    print("  COMPARISON TABLE — All EEG-Only Methods")
    print("=" * 74)

    def fmt(v, digits=3):
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{digits}f}"
        return "  N/A"

    header = (f"{'Method':<22s} {'Sens':>6s} {'Prec':>6s} {'F1':>6s} "
              f"{'FrqRho':>7s} {'FrqMAE':>7s} {'Tmg(ms)':>8s} {'N':>4s}")
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        ref_tag = " *" if "[REF]" in r['method'] else ""
        line = (f"{r['method']:<22s} "
                f"{fmt(r['sensitivity']):>6s} "
                f"{fmt(r['precision']):>6s} "
                f"{fmt(r['f1']):>6s} "
                f"{fmt(r['freq_spearman']):>7s} "
                f"{fmt(r['freq_mae']):>7s} "
                f"{fmt(r['mean_timing_ms'], 1):>8s} "
                f"{r['n_cases']:>4d}{ref_tag}")
        print(line)

    print(f"\n  * = Reference method using gold-standard frequency (not deployment-ready)")

    print(f"\n  Discharge counts:")
    for r in results:
        print(f"    {r['method']:<22s} TP={r['tp']}, FN={r['fn']}, FP={r['fp']}")

    # Save results
    comparison = {}
    for r in results:
        key = r['method'].lower().replace(' ', '_').replace('+', '_').replace('[', '').replace(']', '')
        comparison[key] = {
            k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in r.items()
        }
    comparison_path = CACHE_DIR / 'auto_pipeline_comparison.json'
    with open(str(comparison_path), 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Results saved to {comparison_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 74)


if __name__ == '__main__':
    main()
