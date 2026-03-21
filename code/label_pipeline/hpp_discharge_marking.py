"""
HPP (Hidden Point Process) discharge marking algorithm.

MAP inference using dynamic programming over candidate peaks with an
approximately-periodic prior. Pure signal processing + DP (no deep learning).

Usage:
    conda run -n foe python code/label_pipeline/hpp_discharge_marking.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

# ── Path setup ────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
from optimization_harness_v2 import (
    load_dataset, LEFT_INDICES, RIGHT_INDICES, FS,
)

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 20.0
POINTINESS_WEIGHT = 0.6
TKEO_WEIGHT = 0.4
SMOOTH_SIGMA_SAMPLES = 3  # ~15 ms at 200 Hz
ROLLING_WINDOW_S = 1.0
ACTIVE_THRESHOLD_FRAC = 0.5
MIN_ACTIVE_SECONDS = 3.0
ACTIVE_EXPAND_S = 0.5
PEAK_HEIGHT_FRAC = 0.05   # optimized for sensitivity — easier to delete FPs than add FNs

# DP parameters
DP_ALPHA = 3.0       # loosened from 5.0 — more flexible timing tolerance
DP_BETA = 1.0        # skip penalty
DP_LAMBDA = 0.02     # lowered from 0.1 — favors finding more discharges
MAX_SKIP = 3         # max skipped discharges

# EM refinement
TEMPLATE_HALF_MS = 150  # +/- 150 ms for template extraction
CHANNEL_REFINE_MS = 50  # +/- 50 ms for per-channel refinement


# ── A. Per-channel evidence signal ────────────────────────────────────

def _compute_channel_evidence(signal_1d, fs):
    """Compute discharge likelihood E_c(t) for one channel.

    1. Pointiness trace on raw signal
    2. TKEO on 20 Hz lowpassed signal
    3. Z-score both within the window
    4. Combine: 0.6 * pointiness_z + 0.4 * tkeo_z
    5. Smooth with Gaussian (sigma=3 samples)
    6. Clip negative to 0
    """
    n = len(signal_1d)
    if n < 10:
        return np.zeros(n)

    # 1. Pointiness trace
    pt = compute_pointiness_trace(signal_1d)

    # 2. TKEO on lowpassed signal
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    try:
        sig_lp = filtfilt(b_lp, a_lp, signal_1d)
    except ValueError:
        sig_lp = signal_1d.copy()

    tkeo = np.zeros(n)
    if n >= 3:
        tkeo[1:-1] = np.abs(sig_lp[1:-1] ** 2 - sig_lp[:-2] * sig_lp[2:])

    # 3. Z-score both
    def zscore(x):
        m = np.mean(x)
        s = np.std(x)
        if s < 1e-10:
            return np.zeros_like(x)
        return (x - m) / s

    pt_z = zscore(pt)
    tkeo_z = zscore(tkeo)

    # 4. Combine
    evidence = POINTINESS_WEIGHT * pt_z + TKEO_WEIGHT * tkeo_z

    # 5. Smooth
    evidence = gaussian_filter1d(evidence, sigma=SMOOTH_SIGMA_SAMPLES)

    # 6. Clip negative
    evidence = np.clip(evidence, 0, None)

    return evidence


# ── B. Aggregate by class ─────────────────────────────────────────────

def _aggregate_evidence(evidence_all, subtype, laterality=None):
    """Aggregate per-channel evidence into a single E(t).

    GPD: median across all channels
    LPD with laterality: median of ipsilateral channels
    LPD without laterality: max of left-median and right-median
    """
    if subtype == 'gpd':
        return np.median(evidence_all, axis=0)

    # LPD
    if laterality == 'left':
        channels = LEFT_INDICES
    elif laterality == 'right':
        channels = RIGHT_INDICES
    else:
        # Unknown laterality: max of left vs right median
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)

    return np.median(evidence_all[channels], axis=0)


# ── C. Detect active interval ─────────────────────────────────────────

def _detect_active_interval(evidence, fs):
    """Find the longest contiguous interval where rolling mean > threshold.

    Returns (start_sample, end_sample).
    """
    n = len(evidence)
    win = int(ROLLING_WINDOW_S * fs)

    # Rolling mean via cumsum
    cs = np.cumsum(evidence)
    rolling = np.zeros(n)
    for i in range(n):
        lo = max(0, i - win // 2)
        hi = min(n, i + win // 2 + 1)
        rolling[i] = (cs[hi - 1] - (cs[lo - 1] if lo > 0 else 0)) / (hi - lo)

    threshold = ACTIVE_THRESHOLD_FRAC * np.max(rolling)
    above = rolling > threshold

    # Find longest contiguous run
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
        # Use full window
        return 0, n - 1

    # Expand by 0.5s each side
    expand = int(ACTIVE_EXPAND_S * fs)
    start = max(0, best_start - expand)
    end = min(n - 1, best_start + best_len - 1 + expand)
    return start, end


# ── D. Extract candidate peaks ────────────────────────────────────────

def _extract_candidates(evidence, fs, freq_estimate, active_start, active_end):
    """Find local maxima within the active interval."""
    segment = evidence[active_start:active_end + 1]
    if len(segment) < 3:
        return np.array([], dtype=int)

    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(20, int(0.2 * T * fs))  # reduced from 0.3T to 0.2T
    min_height = PEAK_HEIGHT_FRAC * np.max(segment)

    peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)

    # Also include any strong peaks that might have been suppressed by distance
    strong_height = 0.5 * np.max(segment)
    strong_peaks, _ = find_peaks(segment, height=strong_height, distance=max(10, int(0.1 * T * fs)))
    peaks = np.unique(np.concatenate([peaks, strong_peaks]))

    # Shift back to global indices
    return peaks + active_start


# ── E. Dynamic programming ────────────────────────────────────────────

def _dp_best_sequence(candidates, evidence, fs, freq_estimate):
    """Find optimal discharge sequence via forward DP.

    Maximizes: sum(s(c_i)) + sum(transition_scores) - lambda * K
    """
    if len(candidates) == 0:
        return np.array([], dtype=int)
    if len(candidates) == 1:
        return candidates.copy()

    T = 1.0 / freq_estimate
    n = len(candidates)

    # Square evidence scores so strong peaks are strongly favored
    raw_scores = np.array([evidence[c] for c in candidates])
    node_scores = raw_scores ** 1.5  # superlinear weighting of evidence

    best_score = np.full(n, -np.inf)
    best_prev = np.full(n, -1, dtype=int)

    # Each candidate can start a new sequence
    for i in range(n):
        best_score[i] = node_scores[i] - DP_LAMBDA

    for j in range(1, n):
        for i in range(j):
            dt = (candidates[j] - candidates[i]) / fs
            if dt <= 0 or dt > 4 * T:
                continue

            # Best skip score over m=1,2,3
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

    # Traceback from best endpoint
    best_end = int(np.argmax(best_score))
    path = []
    idx = best_end
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()

    return candidates[np.array(path)]


# ── F. EM refinement (1 iteration) ────────────────────────────────────

def _em_refine(evidence, discharge_samples, fs, freq_estimate):
    """One iteration of template-based refinement.

    1. Average waveform snippets around detected times -> template
    2. Cross-correlate template with evidence -> refined candidates
    3. Re-run DP
    """
    n = len(evidence)
    half_win = int(TEMPLATE_HALF_MS / 1000.0 * fs)

    if len(discharge_samples) < 2:
        return discharge_samples

    # 1. Extract and average snippets
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

    # 2. Cross-correlate template with evidence
    # Normalized cross-correlation
    from numpy import correlate
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

    # 3. Find peaks of cross-correlation as refined candidates
    T = 1.0 / freq_estimate if freq_estimate > 0 else 1.0
    min_dist = max(30, int(0.3 * T * fs))
    min_height = 0.15 * np.max(corr) if np.max(corr) > 0 else 0

    refined_candidates, _ = find_peaks(corr, height=min_height, distance=min_dist)

    if len(refined_candidates) < 2:
        return discharge_samples

    # 4. Re-run DP with refined candidates
    return _dp_best_sequence(refined_candidates, evidence, fs, freq_estimate)


# ── G. Per-channel timing ─────────────────────────────────────────────

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


# ── Main detection function ───────────────────────────────────────────

def detect_discharge_times_hpp(segment_18ch, fs=200, subtype='lpd',
                                freq_estimate=None, laterality=None,
                                involved_channels=None, refine=True):
    """Detect discharge times using the HPP (Hidden Point Process) algorithm.

    Args:
        segment_18ch: (18, N) bipolar EEG
        fs: sampling rate
        subtype: 'lpd' or 'gpd'
        freq_estimate: estimated frequency in Hz (if None, estimate via ACF)
        laterality: 'left', 'right', or None
        involved_channels: list of channel indices, or None (auto-detect)
        refine: whether to do EM template refinement

    Returns:
        dict with:
            'global_times': list of discharge times in seconds
            'channel_times': dict mapping channel_idx -> list of times
            'frequency': estimated frequency from median IPI
            'ipi_cv': coefficient of variation of inter-discharge intervals
            'active_interval': (start, end) in seconds
            'n_discharges': int
            'evidence_signal': E(t) array (for visualization)
            'candidates': list of candidate times considered
    """
    n_channels = min(segment_18ch.shape[0], 18)
    n_samples = segment_18ch.shape[1]

    # Estimate frequency if not provided
    if freq_estimate is None or not np.isfinite(freq_estimate) or freq_estimate <= 0:
        # Use ACF on median pointiness trace
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

    # A. Per-channel evidence
    evidence_all = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        evidence_all[ch] = _compute_channel_evidence(segment_18ch[ch], fs)

    # B. Aggregate
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

    # IPI statistics
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


# ── Run on full dataset ──────────────────────────────────────────────

def mark_all_cases():
    """Run HPP discharge marking on all patients and save results."""
    print("=" * 72)
    print("HPP Discharge Marking — Full Dataset")
    print("=" * 72)

    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # Load existing results to preserve ground_truth cases
    output_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'
    existing = {}
    if output_path.exists():
        with open(str(output_path)) as f:
            existing = json.load(f)

    results = {}
    n_skipped_gt = 0
    gold_freqs = []
    ipi_freqs = []
    n_discharges_list = []
    ipi_cv_list = []
    n_failed = 0

    t0 = time.time()

    for idx, (_, row) in enumerate(df.iterrows()):
        pid = str(row['patient_id'])

        # Skip ground_truth cases — never overwrite human-reviewed data
        if pid in existing and existing[pid].get('review_status') == 'ground_truth':
            results[pid] = existing[pid]
            n_skipped_gt += 1
            continue
        subtype = row['subtype']
        gold = float(row['gold_standard_freq'])
        laterality = row.get('laterality', '')
        if not isinstance(laterality, str) or laterality not in ('left', 'right'):
            laterality = None

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            n_failed += 1
            continue

        # Use first segment
        seg = pat_segs[0]

        try:
            result = detect_discharge_times_hpp(
                seg, fs=FS, subtype=subtype,
                freq_estimate=gold, laterality=laterality,
                refine=True,
            )
        except Exception as e:
            print(f"  FAILED {pid}: {e}")
            n_failed += 1
            continue

        # Store serializable result (no numpy arrays)
        results[pid] = {
            'global_times': result['global_times'],
            'frequency': result['frequency'],
            'ipi_cv': result['ipi_cv'],
            'active_interval': list(result['active_interval']),
            'n_discharges': result['n_discharges'],
            'candidates': result['candidates'],
            'subtype': subtype,
            'gold_standard_freq': gold,
            'laterality': laterality,
        }

        # Collect stats
        if np.isfinite(result['frequency']) and np.isfinite(gold):
            gold_freqs.append(gold)
            ipi_freqs.append(result['frequency'])
        n_discharges_list.append(result['n_discharges'])
        if np.isfinite(result['ipi_cv']):
            ipi_cv_list.append(result['ipi_cv'])

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(df)} patients ({elapsed:.0f}s)")

    elapsed = time.time() - t0

    # Save results
    out_dir = PROJECT_DIR / 'data' / 'labels'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'discharge_times_hpp.json'

    def json_default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2, default=json_default)

    # Summary statistics
    print(f"\n{'=' * 72}")
    print(f"HPP Discharge Marking — Results")
    print(f"{'=' * 72}")
    print(f"  Patients processed: {len(results)}")
    print(f"  Ground truth (preserved): {n_skipped_gt}")
    print(f"  Re-generated (auto): {len(results) - n_skipped_gt}")
    print(f"  Failed: {n_failed}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Saved to: {out_path}")
    print()

    if n_discharges_list:
        print(f"  Mean discharges per case: {np.mean(n_discharges_list):.1f} "
              f"(std {np.std(n_discharges_list):.1f})")

    if ipi_cv_list:
        print(f"  Mean IPI CV: {np.mean(ipi_cv_list):.3f} "
              f"(std {np.std(ipi_cv_list):.3f})")

    # Frequency correlation with gold standard
    if len(gold_freqs) >= 5:
        gold_arr = np.array(gold_freqs)
        ipi_arr = np.array(ipi_freqs)

        rs, p_val = spearmanr(gold_arr, ipi_arr)
        print(f"\n  IPI-derived frequency vs gold standard:")
        print(f"    N = {len(gold_arr)}")
        print(f"    Spearman r = {rs:.4f} (p = {p_val:.2e})")

        # Also compute MAE
        mae = np.mean(np.abs(gold_arr - ipi_arr))
        print(f"    MAE = {mae:.3f} Hz")

        # Median absolute error
        med_ae = np.median(np.abs(gold_arr - ipi_arr))
        print(f"    Median AE = {med_ae:.3f} Hz")

    print(f"{'=' * 72}")


if __name__ == '__main__':
    mark_all_cases()
