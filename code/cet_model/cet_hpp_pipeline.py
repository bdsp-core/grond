"""
CET+HPP pipeline: CNN Evidence Trace + HPP Dynamic Programming.

Replaces the handcrafted pointiness+TKEO evidence in HPP with learned
CNN evidence, then runs the same DP algorithm for discharge detection.

Evaluates three methods:
  1. HPP-only (handcrafted evidence)
  2. CET+HPP (old CETModel, no skip connections, sigma=10)
  3. CET-UNet+HPP (new CETUNet, skip connections, sigma=2)

Usage:
    conda run -n foe_dl python code/cet_model/cet_hpp_pipeline.py
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

from cet_model.cet import CETModel, CETUNet
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from pd_pointiness_acf import compute_acf_frequency

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'
DEVICE = torch.device('cpu')

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


# -- Load CET models --------------------------------------------------------

def load_cet_models(n_folds=5):
    """Load ensemble of old CET models (no skip connections) from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cet_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CET model not found: {model_path}")
        model = CETModel()
        model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


def load_cet_unet_models(n_folds=5):
    """Load ensemble of CET-UNet models (with skip connections) from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cet_unet_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CET-UNet model not found: {model_path}")
        model = CETUNet()
        model.load_state_dict(torch.load(str(model_path), map_location=DEVICE))
        model.eval()
        models.append(model)
    return models


# -- CET evidence computation -----------------------------------------------

@torch.no_grad()
def compute_cet_evidence(channel_data, cet_models):
    """Run CET ensemble on a single channel.

    Args:
        channel_data: (2000,) float64/32 array
        cet_models: list of CETModel or CETUNet instances

    Returns:
        (2000,) float32 evidence trace (mean of ensemble)
    """
    x = channel_data.astype(np.float32).copy()

    # Per-channel z-score
    mu = np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = (x - mu) / std
    else:
        x = x - mu

    x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :])  # (1, 1, 2000)

    # Ensemble: average predictions from all folds
    predictions = []
    for model in cet_models:
        pred = model(x_tensor).squeeze().numpy()  # (2000,)
        predictions.append(pred)

    return np.mean(predictions, axis=0).astype(np.float32)


# -- HPP components (reused from hpp_discharge_marking.py) -------------------

def _aggregate_evidence(evidence_all, subtype, laterality=None):
    """Aggregate per-channel CET evidence into a single E(t)."""
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


# -- Main CET+HPP detection function ----------------------------------------

def detect_discharge_times_cet_hpp(segment_18ch, fs=200, subtype='lpd',
                                    freq_estimate=None, laterality=None,
                                    cet_models=None, refine=True):
    """CET+HPP discharge timing detection.

    1. Run each channel through CET models (ensemble of 5 folds)
    2. Aggregate by subtype
    3. Run HPP DP on the CNN evidence

    Returns same format as detect_discharge_times_hpp.
    """
    n_channels = min(segment_18ch.shape[0], 18)
    n_samples = segment_18ch.shape[1]

    # Estimate frequency if not provided
    if freq_estimate is None or not np.isfinite(freq_estimate) or freq_estimate <= 0:
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
        freq_estimate = float(np.median(acf_freqs)) if acf_freqs else 1.0

    freq_estimate = np.clip(freq_estimate, 0.3, 3.5)

    # A. Per-channel CET evidence
    evidence_all = np.zeros((n_channels, n_samples), dtype=np.float32)
    for ch in range(n_channels):
        ch_data = segment_18ch[ch]
        if not np.all(np.isfinite(ch_data)):
            continue
        evidence_all[ch] = compute_cet_evidence(ch_data, cet_models)

    # B. Aggregate by subtype
    evidence = _aggregate_evidence(evidence_all, subtype, laterality)

    # C. Active interval
    active_start, active_end = _detect_active_interval(evidence, fs)

    # D. Candidate peaks
    candidates = _extract_candidates(evidence, fs, freq_estimate,
                                     active_start, active_end)

    # E. DP
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)

    # F. EM refinement
    if refine and len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)

    # G. Per-channel timing
    channel_times = _per_channel_times(evidence_all, discharge_samples, fs)

    # Compute output metrics
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
        'ipi_cv': ipi_cv,
        'active_interval': (active_start / fs, active_end / fs),
        'n_discharges': len(discharge_samples),
        'evidence_signal': evidence,
        'candidates': (candidates / fs).tolist() if len(candidates) > 0 else [],
    }


# -- Evaluation --------------------------------------------------------------

def evaluate_method(method_name, detect_fn, df, segments, gt_cases,
                    cet_models=None):
    """Run a detection method on all GT cases and compute metrics.

    Returns dict with sensitivity, precision, F1, freq_spearman, timing_error.
    """
    total_tp = 0
    total_fn = 0
    total_fp = 0
    match_errors = []

    gt_freqs = []
    algo_freqs = []
    gold_freqs = []

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
            if cet_models is not None:
                result = detect_fn(
                    seg, fs=200, subtype=subtype,
                    freq_estimate=gold_freq, laterality=lat,
                    cet_models=cet_models, refine=True)
            else:
                result = detect_fn(
                    seg, fs=200, subtype=subtype,
                    freq_estimate=gold_freq, laterality=lat, refine=True)
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
            gold_freqs.append(gold_freq)

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


# -- Main --------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 70)
    print("Timing Detection Comparison: HPP-only vs CET+HPP vs CET-UNet+HPP")
    print("=" * 70)

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

    # Import HPP detection function
    sys.path.insert(0, str(CODE_DIR / 'label_pipeline'))
    from hpp_discharge_marking import detect_discharge_times_hpp

    # ---- Method 1: HPP-only ----
    print("\n" + "-" * 70)
    print("Running HPP-only evaluation...")
    hpp_results = evaluate_method('HPP-only', detect_discharge_times_hpp,
                                  df, segments, gt_cases)
    print(f"  Done: {hpp_results['n_cases']} cases, "
          f"F1={hpp_results['f1']:.3f}, {hpp_results['eval_time_s']:.1f}s")

    # ---- Method 2: Old CET+HPP ----
    old_cet_results = None
    try:
        print("\nLoading old CET models...")
        old_cet_models = load_cet_models()
        print(f"  Loaded {len(old_cet_models)} fold models")

        print("\nRunning CET+HPP (old) evaluation...")
        old_cet_results = evaluate_method('CET+HPP (old)',
                                           detect_discharge_times_cet_hpp,
                                           df, segments, gt_cases,
                                           cet_models=old_cet_models)
        print(f"  Done: {old_cet_results['n_cases']} cases, "
              f"F1={old_cet_results['f1']:.3f}, {old_cet_results['eval_time_s']:.1f}s")
    except FileNotFoundError as e:
        print(f"  Old CET models not found, skipping: {e}")

    # ---- Method 3: CET-UNet+HPP ----
    unet_results = None
    try:
        print("\nLoading CET-UNet models...")
        unet_models = load_cet_unet_models()
        print(f"  Loaded {len(unet_models)} fold models")

        print("\nRunning CET-UNet+HPP evaluation...")
        unet_results = evaluate_method('CET-UNet+HPP',
                                        detect_discharge_times_cet_hpp,
                                        df, segments, gt_cases,
                                        cet_models=unet_models)
        print(f"  Done: {unet_results['n_cases']} cases, "
              f"F1={unet_results['f1']:.3f}, {unet_results['eval_time_s']:.1f}s")
    except FileNotFoundError as e:
        print(f"  CET-UNet models not found, skipping: {e}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("=== Timing Detection Comparison ===")
    print("=" * 70)

    def fmt(v, digits=3):
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{digits}f}"
        return "N/A"

    all_results = [hpp_results]
    if old_cet_results is not None:
        all_results.append(old_cet_results)
    if unet_results is not None:
        all_results.append(unet_results)

    header = f"{'Method':<20s} {'Sens':>7s} {'Prec':>7s} {'F1':>7s} " \
             f"{'FreqRho':>8s} {'Timing(ms)':>10s}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in all_results:
        line = f"{r['method']:<20s} " \
               f"{fmt(r['sensitivity']):>7s} " \
               f"{fmt(r['precision']):>7s} " \
               f"{fmt(r['f1']):>7s} " \
               f"{fmt(r['freq_spearman']):>8s} " \
               f"{fmt(r['mean_timing_ms'], 1):>10s}"
        print(line)

    print(f"\n  Discharge counts:")
    for r in all_results:
        print(f"    {r['method']:<20s} TP={r['tp']}, FN={r['fn']}, FP={r['fp']}")

    # Save comparison results
    comparison = {}
    for r in all_results:
        key = r['method'].lower().replace(' ', '_').replace('+', '_').replace('(', '').replace(')', '')
        comparison[key] = {
            k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in r.items()
        }
    comparison_path = CACHE_DIR / 'cet_hpp_comparison.json'
    with open(str(comparison_path), 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved to {comparison_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
