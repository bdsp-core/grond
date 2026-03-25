"""
Combination experiments: test combinations of the top-performing approaches.

Combos tested:
1. combo_acfthr010_peakcount: ACF thr=0.10, average ACF freq with peak-count freq
2. combo_acfthr010_subharm: ACF thr=0.10 + subharmonic correction (L/2,L/3,L/4 @ 0.3)
3. combo_acfthr010_peakcount_bayesian: combo#1 + Bayesian nudge (0.7*f + 0.3*prior)
4. combo_acfthr010_subharm_peakcount: ACF thr=0.10 + subharm + average with peak-count
5. combo_acfthr010_weighted_peakcount: ACF thr=0.10 + weighted channels + peak-count
6. combo_kitchen_sink: all of the above combined
7. combo_acfthr010_ensemble_higher: max(Method A freq, ACF-thr-0.10 Method B freq)
8. combo_best_plus_A: average of combo#1 with Method A freq
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
ACF_THRESHOLD = 0.10
ACF_MIN_LAG = 0.25
SMOOTHING_SIGMA = 0.02
LOWPASS_HZ = 15.0
PEAK_HEIGHT_FRAC = 0.3
SUBHARM_ACF_THRESHOLD = 0.3   # for subharmonic correction
GPD_PRIOR = 1.0
LPD_PRIOR = 1.25
BAYESIAN_ALGO_WEIGHT = 0.7
BAYESIAN_PRIOR_WEIGHT = 0.3
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
def analyze_channels(seg, fs, acf_threshold=ACF_THRESHOLD):
    """Run pointiness + ACF on each channel.

    Returns:
        acf_freqs: array shape (n_channels,) – ACF frequency per channel (NaN if none)
        acf_scores: array shape (n_channels,) – ACF peak height per channel
        peak_count_freqs: array shape (n_channels,) – peak-count frequency per channel
    """
    n_ch = seg.shape[0]
    acf_freqs = np.full(n_ch, np.nan)
    acf_scores = np.full(n_ch, 0.0)
    peak_count_freqs = np.full(n_ch, np.nan)

    for i in range(n_ch):
        freq, score, peak_indices = compute_acf_frequency(
            seg[i, :], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=acf_threshold,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        acf_freqs[i] = freq
        acf_scores[i] = score

        # Peak-count frequency: count peaks in pointiness trace, divide by duration
        if len(peak_indices) >= 2:
            duration_s = len(seg[i, :]) / fs
            peak_count_freqs[i] = len(peak_indices) / duration_s

    return acf_freqs, acf_scores, peak_count_freqs


# ── Subharmonic correction ─────────────────────────────────────────────
def subharmonic_correct_channel(signal_1d, fs, base_freq):
    """Check if L/2, L/3, L/4 has higher ACF peak; if so, use the lower freq."""
    if not np.isfinite(base_freq) or base_freq <= 0:
        return base_freq

    # Compute ACF once
    trace = compute_pointiness_trace(signal_1d)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    trace = gaussian_filter1d(trace, sigma=sigma_samples)
    t = trace - np.mean(trace)
    max_lag = min(int(4 * fs), len(t) - 1)
    if max_lag < 10:
        return base_freq

    acf = np.correlate(t, t, mode='full')
    acf = acf[len(t) - 1:][:max_lag + 1]
    if acf[0] <= 0:
        return base_freq
    acf = acf / acf[0]

    base_lag = int(round(fs / base_freq))

    best_freq = base_freq
    best_acf_val = acf[base_lag] if base_lag < len(acf) else 0.0

    for divisor in [2, 3, 4]:
        sub_lag = base_lag * divisor
        if sub_lag >= len(acf) - 1:
            continue
        # Find peak near sub_lag (within +/- 10% window)
        window = max(3, int(sub_lag * 0.1))
        lo = max(0, sub_lag - window)
        hi = min(len(acf) - 1, sub_lag + window)
        region = acf[lo:hi + 1]
        local_max_idx = lo + np.argmax(region)
        local_max_val = acf[local_max_idx]
        if local_max_val > SUBHARM_ACF_THRESHOLD and local_max_val >= best_acf_val * 0.8:
            best_freq = fs / local_max_idx
            best_acf_val = local_max_val

    return best_freq


def subharmonic_correct_all(seg, fs, acf_freqs):
    """Apply subharmonic correction to all channels."""
    corrected = acf_freqs.copy()
    for i in range(seg.shape[0]):
        if np.isfinite(acf_freqs[i]):
            corrected[i] = subharmonic_correct_channel(seg[i, :], fs, acf_freqs[i])
    return corrected


# ── Aggregation strategies ─────────────────────────────────────────────
def aggregate_median(freqs):
    """Median of valid (finite) frequencies."""
    valid = freqs[np.isfinite(freqs)]
    return float(np.median(valid)) if len(valid) > 0 else np.nan


def aggregate_weighted_mean(freqs, weights):
    """Weighted mean of valid frequencies, weighted by ACF scores."""
    mask = np.isfinite(freqs) & (weights > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(freqs[mask], weights=weights[mask]))


# ── Bayesian nudge ─────────────────────────────────────────────────────
def bayesian_nudge(freq, subdir):
    """Apply Bayesian nudge: 0.7*algo + 0.3*prior."""
    if not np.isfinite(freq):
        return freq
    prior = GPD_PRIOR if subdir == 'gpd' else LPD_PRIOR
    return BAYESIAN_ALGO_WEIGHT * freq + BAYESIAN_PRIOR_WEIGHT * prior


# ── Method A wrapper ───────────────────────────────────────────────────
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


# ── Combination strategies ─────────────────────────────────────────────
def combo_acfthr010_peakcount(seg, fs, subdir, data_raw=None):
    """ACF thr=0.10, average ACF freq with peak-count freq."""
    acf_freqs, acf_scores, pc_freqs = analyze_channels(seg, fs, acf_threshold=0.10)
    acf_med = aggregate_median(acf_freqs)
    pc_med = aggregate_median(pc_freqs)
    vals = [v for v in [acf_med, pc_med] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def combo_acfthr010_subharm(seg, fs, subdir, data_raw=None):
    """ACF thr=0.10 + subharmonic correction."""
    acf_freqs, acf_scores, _ = analyze_channels(seg, fs, acf_threshold=0.10)
    corrected = subharmonic_correct_all(seg, fs, acf_freqs)
    return aggregate_median(corrected)


def combo_acfthr010_peakcount_bayesian(seg, fs, subdir, data_raw=None):
    """combo#1 + Bayesian nudge."""
    freq = combo_acfthr010_peakcount(seg, fs, subdir)
    return bayesian_nudge(freq, subdir)


def combo_acfthr010_subharm_peakcount(seg, fs, subdir, data_raw=None):
    """ACF thr=0.10 + subharmonic correction + average with peak-count."""
    acf_freqs, acf_scores, pc_freqs = analyze_channels(seg, fs, acf_threshold=0.10)
    corrected = subharmonic_correct_all(seg, fs, acf_freqs)
    sub_med = aggregate_median(corrected)
    pc_med = aggregate_median(pc_freqs)
    vals = [v for v in [sub_med, pc_med] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def combo_acfthr010_weighted_peakcount(seg, fs, subdir, data_raw=None):
    """ACF thr=0.10, weighted-mean channel aggregation, average with peak-count."""
    acf_freqs, acf_scores, pc_freqs = analyze_channels(seg, fs, acf_threshold=0.10)
    acf_wt = aggregate_weighted_mean(acf_freqs, acf_scores)
    pc_med = aggregate_median(pc_freqs)
    vals = [v for v in [acf_wt, pc_med] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


def combo_kitchen_sink(seg, fs, subdir, data_raw=None):
    """ACF thr=0.10 + subharm + weighted channels + peak-count + Bayesian."""
    acf_freqs, acf_scores, pc_freqs = analyze_channels(seg, fs, acf_threshold=0.10)
    corrected = subharmonic_correct_all(seg, fs, acf_freqs)
    acf_wt = aggregate_weighted_mean(corrected, acf_scores)
    pc_med = aggregate_median(pc_freqs)
    vals = [v for v in [acf_wt, pc_med] if np.isfinite(v)]
    freq = float(np.mean(vals)) if vals else np.nan
    return bayesian_nudge(freq, subdir)


def combo_acfthr010_ensemble_higher(seg, fs, subdir, data_raw=None):
    """Run Method A and ACF-thr-0.10 Method B, pick higher frequency."""
    # Method B: ACF thr=0.10 median
    acf_freqs, _, _ = analyze_channels(seg, fs, acf_threshold=0.10)
    b_freq = aggregate_median(acf_freqs)
    # Method A
    a_freq = get_method_a_freq(data_raw, fs) if data_raw is not None else np.nan
    vals = [v for v in [a_freq, b_freq] if np.isfinite(v)]
    if not vals:
        return np.nan
    return float(max(vals))


def combo_best_plus_A(seg, fs, subdir, data_raw=None):
    """Average of (ACF thr=0.10 + peak-count mean) with Method A freq."""
    combo1_freq = combo_acfthr010_peakcount(seg, fs, subdir)
    a_freq = get_method_a_freq(data_raw, fs) if data_raw is not None else np.nan
    vals = [v for v in [combo1_freq, a_freq] if np.isfinite(v)]
    return float(np.mean(vals)) if vals else np.nan


# ── Main ───────────────────────────────────────────────────────────────
COMBOS = {
    'combo_acfthr010_peakcount': combo_acfthr010_peakcount,
    'combo_acfthr010_subharm': combo_acfthr010_subharm,
    'combo_acfthr010_peakcount_bayesian': combo_acfthr010_peakcount_bayesian,
    'combo_acfthr010_subharm_peakcount': combo_acfthr010_subharm_peakcount,
    'combo_acfthr010_weighted_peakcount': combo_acfthr010_weighted_peakcount,
    'combo_kitchen_sink': combo_kitchen_sink,
    'combo_acfthr010_ensemble_higher': combo_acfthr010_ensemble_higher,
    'combo_best_plus_A': combo_best_plus_A,
}

# Which combos need raw data for Method A
NEEDS_RAW = {'combo_acfthr010_ensemble_higher', 'combo_best_plus_A'}


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

        # Preprocess once
        seg = preprocess_segment(data, fs)
        subdir = entry['subdir']
        mat_name = entry['mat_name']

        for combo_name, combo_fn in COMBOS.items():
            raw = data if combo_name in NEEDS_RAW else None
            try:
                freq = combo_fn(seg, fs, subdir, data_raw=raw)
            except Exception as e:
                freq = np.nan
            all_predictions[combo_name][mat_name] = freq

        if (idx + 1) % 100 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # Evaluate all combos
    print("\n" + "=" * 70)
    print("EVALUATING ALL COMBINATIONS")
    print("=" * 70)

    summary_rows = []
    for combo_name in COMBOS:
        metrics = evaluate_predictions(dataset, all_predictions[combo_name], combo_name)
        summary_rows.append(metrics)

    # Print results table
    print("\n" + "=" * 70)
    print("COMBINATION RESULTS SUMMARY (sorted by combined MAE)")
    print("=" * 70)
    header = f"{'Experiment':<42s} {'LPD MAE':>8s} {'GPD MAE':>8s} {'Combined':>9s}"
    print(header)
    print("-" * len(header))

    # Add baselines
    print(f"{'Method A (baseline)':.<42s} {'0.537':>8s} {'0.274':>8s} {'0.406':>9s}")
    print(f"{'Expert-expert':.<42s} {'0.594':>8s} {'0.315':>8s} {'0.455':>9s}")
    print("-" * len(header))

    sorted_rows = sorted(summary_rows,
                         key=lambda r: r.get('combined_mae', 999))
    for row in sorted_rows:
        lpd = row.get('lpd_mae', '?')
        gpd = row.get('gpd_mae', '?')
        comb = row.get('combined_mae', '?')
        lpd_s = f"{lpd:.4f}" if isinstance(lpd, (int, float)) else str(lpd)
        gpd_s = f"{gpd:.4f}" if isinstance(gpd, (int, float)) else str(gpd)
        comb_s = f"{comb:.4f}" if isinstance(comb, (int, float)) else str(comb)
        print(f"{row['experiment']:<42s} {lpd_s:>8s} {gpd_s:>8s} {comb_s:>9s}")

    print("\nDone! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
