"""
Round 2 Combiner: test 8 combination strategies using the best findings from Round 1/2.

Methods combined:
  A: pd_detect_alternate (apd) event_frequency
  B: pointiness ACF (thr=0.10, min_lag=0.25, sigma=0.02, lp=15Hz, ph_frac=0.3)
  Peak-count: n_peaks / duration from pointiness trace
  IPI: 1 / median(inter-peak-intervals) from pointiness trace
  Peak-equalized ACF: binary peaks smoothed sigma=8, ACF thr=0.10, min_lag=0.25

8 combinations tested (see docstrings below).
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

# ── Constants ──────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.25
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
GPD_PRIOR = 1.0
LPD_PRIOR = 1.25
BAYESIAN_ALGO_WEIGHT = 0.7
BAYESIAN_PRIOR_WEIGHT = 0.3
PEAKEQ_SIGMA = 8
PEAKEQ_ACF_THR = 0.10
PEAKEQ_MIN_LAG = 0.25
FS = 200


# ── Preprocessing ──────────────────────────────────────────────────────
def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


# ── Per-channel analysis ───────────────────────────────────────────────
def analyze_all_channels(seg, fs):
    """Compute ACF freq, peak-count freq, IPI freq, and peak-equalized ACF freq
    for every channel.

    Returns:
        acf_freqs, peak_count_freqs, ipi_freqs, peakeq_freqs
        each shape (n_channels,)
    """
    n_ch = seg.shape[0]
    acf_freqs = np.full(n_ch, np.nan)
    peak_count_freqs = np.full(n_ch, np.nan)
    ipi_freqs = np.full(n_ch, np.nan)
    peakeq_freqs = np.full(n_ch, np.nan)

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    max_lag = int(4 * fs)
    min_lag_samples = int(PEAKEQ_MIN_LAG * fs)
    duration_s = seg.shape[1] / fs

    for i in range(n_ch):
        ch = seg[i]
        if len(ch) < 50:
            continue

        # --- ACF frequency (Method B) ---
        freq_b, score_b, peak_indices = compute_acf_frequency(
            ch, fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        acf_freqs[i] = freq_b

        # --- Pointiness trace for peaks ---
        trace = compute_pointiness_trace(ch)
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)

        # --- Peak-count frequency ---
        if len(peak_locs) >= 2:
            peak_count_freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)

        # --- IPI frequency ---
        if len(peak_locs) >= 3:
            intervals = np.diff(peak_locs) / fs
            med_interval = np.median(intervals)
            if med_interval > 0:
                ipi_freqs[i] = 1.0 / med_interval

        # --- Peak-equalized ACF frequency ---
        if len(peak_locs) >= 2:
            eq_trace = np.zeros(len(trace))
            eq_trace[peak_locs] = 1.0
            eq_trace = gaussian_filter1d(eq_trace, sigma=PEAKEQ_SIGMA)
            # ACF on equalized trace
            t = eq_trace - np.mean(eq_trace)
            ml = min(max_lag, len(t) - 1)
            if ml > 10:
                acf = np.correlate(t, t, mode='full')
                acf = acf[len(t) - 1:][:ml + 1]
                if acf[0] > 0:
                    acf = acf / acf[0]
                    for k in range(min_lag_samples + 1, len(acf) - 1):
                        if acf[k] > acf[k-1] and acf[k] > acf[k+1] and acf[k] > PEAKEQ_ACF_THR:
                            peakeq_freqs[i] = fs / k
                            break

    return acf_freqs, peak_count_freqs, ipi_freqs, peakeq_freqs


# ── Helpers ────────────────────────────────────────────────────────────
def median_finite(arr):
    """Median of finite values, or NaN."""
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else [x for x in arr if np.isfinite(x)]
    if isinstance(valid, np.ndarray):
        return float(np.median(valid)) if len(valid) > 0 else np.nan
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def get_method_a_freq(data, fs):
    """Run Method A and return event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def bayesian_nudge(freq, subdir):
    """Apply Bayesian nudge: 0.7*algo + 0.3*prior."""
    if not np.isfinite(freq):
        return freq
    prior = GPD_PRIOR if subdir == 'gpd' else LPD_PRIOR
    return BAYESIAN_ALGO_WEIGHT * freq + BAYESIAN_PRIOR_WEIGHT * prior


# ── 8 Combination strategies ──────────────────────────────────────────

def r2_quad_median(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """1. Median of 4: A, B(thr=0.10), peak-count, IPI."""
    vals = [v for v in [a_freq, b_freq, pc_freq, ipi_freq] if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def r2_quad_median_bayesian(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """2. Quad median + Bayesian nudge."""
    freq = r2_quad_median(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir)
    return bayesian_nudge(freq, subdir)


def r2_five_median(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """3. Median of 5: A, B, peak-count, IPI, peak-equalized ACF."""
    vals = [v for v in [a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq] if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def r2_triple_median_ipi(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """4. Median of 3: A, IPI, peak-count (replace B with IPI)."""
    vals = [v for v in [a_freq, ipi_freq, pc_freq] if np.isfinite(v)]
    return float(np.median(vals)) if vals else np.nan


def r2_weighted_quad(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """5. Weighted mean: A(0.3), B(0.2), peak-count(0.3), IPI(0.2)."""
    sources = [(a_freq, 0.3), (b_freq, 0.2), (pc_freq, 0.3), (ipi_freq, 0.2)]
    vals = [(v, w) for v, w in sources if np.isfinite(v)]
    if not vals:
        return np.nan
    total_w = sum(w for _, w in vals)
    return float(sum(v * w for v, w in vals) / total_w)


def r2_lpd_optimized(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """6. LPD: round1_best (ACF+peakcount avg + Bayesian); GPD: triple median(A,B,pc)."""
    if subdir == 'lpd':
        # ACF thr=0.10 + peak-count average + Bayesian nudge
        vals = [v for v in [b_freq, pc_freq] if np.isfinite(v)]
        freq = float(np.mean(vals)) if vals else np.nan
        return bayesian_nudge(freq, subdir)
    else:
        # GPD: triple median of A, B, peak-count
        vals = [v for v in [a_freq, b_freq, pc_freq] if np.isfinite(v)]
        return float(np.median(vals)) if vals else np.nan


def r2_adaptive_median(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """7. Triple median(A,B,pc), but if result < 0.6 Hz, recompute excluding B."""
    vals = [v for v in [a_freq, b_freq, pc_freq] if np.isfinite(v)]
    result = float(np.median(vals)) if vals else np.nan
    if np.isfinite(result) and result < 0.6:
        # B likely subharmonic-locked; use A, peak-count, IPI instead
        vals2 = [v for v in [a_freq, pc_freq, ipi_freq] if np.isfinite(v)]
        result = float(np.median(vals2)) if vals2 else result
    return result


def r2_robust_consensus(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir):
    """8. Compute all 4, remove outliers (>0.5 Hz from group median), mean of rest."""
    estimates = np.array([v for v in [a_freq, b_freq, pc_freq, ipi_freq] if np.isfinite(v)])
    if len(estimates) == 0:
        return np.nan
    group_median = np.median(estimates)
    inliers = estimates[np.abs(estimates - group_median) <= 0.5]
    if len(inliers) == 0:
        return float(group_median)
    return float(np.mean(inliers))


COMBOS = {
    'r2_quad_median': r2_quad_median,
    'r2_quad_median_bayesian': r2_quad_median_bayesian,
    'r2_five_median': r2_five_median,
    'r2_triple_median_ipi': r2_triple_median_ipi,
    'r2_weighted_quad': r2_weighted_quad,
    'r2_lpd_optimized': r2_lpd_optimized,
    'r2_adaptive_median': r2_adaptive_median,
    'r2_robust_consensus': r2_robust_consensus,
}


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Pre-allocate predictions dicts
    all_predictions = {name: {} for name in COMBOS}

    n_total = len(dataset)
    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']

        # --- Method A ---
        a_freq = get_method_a_freq(data, fs)

        # --- Preprocess for B/peaks/IPI/peakeq ---
        seg = preprocess_segment(data, fs)

        # --- All channel-level estimates ---
        acf_freqs, peak_count_freqs, ipi_freqs, peakeq_freqs = analyze_all_channels(seg, fs)

        # --- Aggregate to segment-level ---
        b_freq = median_finite(acf_freqs)
        pc_freq = median_finite(peak_count_freqs)
        ipi_freq = median_finite(ipi_freqs)
        peakeq_freq = median_finite(peakeq_freqs)

        # --- Apply all 8 combination strategies ---
        for combo_name, combo_fn in COMBOS.items():
            try:
                freq = combo_fn(a_freq, b_freq, pc_freq, ipi_freq, peakeq_freq, subdir)
            except Exception:
                freq = np.nan
            all_predictions[combo_name][mat_name] = freq

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # Evaluate all combos
    print("\n" + "=" * 70)
    print("EVALUATING ALL ROUND 2 COMBINATIONS")
    print("=" * 70)

    summary_rows = []
    for combo_name in COMBOS:
        metrics = evaluate_predictions(dataset, all_predictions[combo_name], combo_name)
        summary_rows.append(metrics)

    # Print results table
    print("\n" + "=" * 70)
    print("ROUND 2 COMBINER RESULTS (sorted by combined Spearman)")
    print("=" * 70)
    header = (f"{'Experiment':<35s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baselines
    print(f"{'Method A (baseline)':<35s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
    print("-" * len(header))

    sorted_rows = sorted(summary_rows,
                         key=lambda r: -(r.get('combined_spearman', -999)
                                         if np.isfinite(r.get('combined_spearman', np.nan))
                                         else -999))
    for row in sorted_rows:
        def fmt(key, default='?'):
            v = row.get(key, default)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return f"{v:.4f}"
            return str(v)
        print(f"{row['experiment']:<35s} {fmt('lpd_mae'):>8s} {fmt('gpd_mae'):>8s} "
              f"{fmt('lpd_spearman_r'):>7s} {fmt('gpd_spearman_r'):>7s} "
              f"{fmt('combined_spearman'):>8s} {fmt('combined_mae'):>9s}")

    print("\nDone! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
