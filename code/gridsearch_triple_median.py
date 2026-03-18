"""
Deep parameter grid search for the triple-median approach.

Triple median = median(Method_A_freq, Method_B_acf_freq, peak_count_freq)

Method A is pre-computed once. Method B (pointiness+ACF) and peak-count
frequency are recomputed per parameter combo.

Parameter grid (432 combos):
  lowpass_hz:        [10, 15, 20, 25]
  smoothing_sigma:   [0.01, 0.02, 0.03]
  acf_min_lag:       [0.2, 0.3, 0.4]
  acf_peak_threshold:[0.05, 0.10, 0.15, 0.20]
  peak_height_frac:  [0.2, 0.3, 0.4]

Optimization: pre-filters segments once per lowpass_hz value, then for each
param combo only recomputes pointiness+ACF (skips synchrony/L-G classification).
"""

import sys
import os
import time
import itertools
import numpy as np
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    bipolar_channels,
)
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from mne.filter import notch_filter, filter_data

FS = 200

# Bayesian nudge constants
GPD_PRIOR = 1.0
LPD_PRIOR = 1.25
BAYESIAN_ALGO_WEIGHT = 0.7
BAYESIAN_PRIOR_WEIGHT = 0.3


# ── Parameter grid ────────────────────────────────────────────────────
GRID = {
    'lowpass_hz':        [10, 15, 20, 25],
    'smoothing_sigma':   [0.01, 0.02, 0.03],
    'acf_min_lag':       [0.2, 0.3, 0.4],
    'acf_peak_threshold':[0.05, 0.10, 0.15, 0.20],
    'peak_height_frac':  [0.2, 0.3, 0.4],
}

PARAM_NAMES = list(GRID.keys())
PARAM_VALUES = [GRID[k] for k in PARAM_NAMES]
ALL_COMBOS = list(itertools.product(*PARAM_VALUES))
print(f"Total parameter combinations: {len(ALL_COMBOS)}")


def bayesian_nudge(freq, subdir):
    if not np.isfinite(freq):
        return freq
    prior = GPD_PRIOR if subdir == 'gpd' else LPD_PRIOR
    return BAYESIAN_ALGO_WEIGHT * freq + BAYESIAN_PRIOR_WEIGHT * prior


def get_method_a_freq(data, fs):
    """Run Method A (pd_detect_alternate) and return event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


def preprocess_base(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage. Returns bipolar segment."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    return seg


def apply_lowpass(seg, fs, lowpass_hz):
    """Apply lowpass filter to bipolar segment."""
    b_lp, a_lp = butter(4, lowpass_hz / (fs / 2), btype='low')
    out = np.zeros_like(seg)
    for i in range(seg.shape[0]):
        try:
            out[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            out[i] = seg[i]
    return out


def compute_method_b_and_peakcount(seg_lp, fs, smoothing_sigma, acf_min_lag,
                                    acf_peak_threshold, peak_height_frac):
    """Lightweight Method B + peak-count frequency computation.

    Returns (acf_median_freq, peakcount_median_freq).
    Skips synchrony / L-G classification.
    """
    n_ch = seg_lp.shape[0]
    acf_freqs = np.full(n_ch, np.nan)
    pc_freqs = np.full(n_ch, np.nan)

    for i in range(n_ch):
        freq, score, peak_indices = compute_acf_frequency(
            seg_lp[i, :], fs, method='pointiness',
            smoothing_sigma=smoothing_sigma,
            acf_min_lag=acf_min_lag,
            acf_peak_threshold=acf_peak_threshold,
            peak_height_frac=peak_height_frac,
        )
        acf_freqs[i] = freq

        # Peak-count frequency
        if len(peak_indices) >= 2:
            duration_s = len(seg_lp[i, :]) / fs
            pc_freqs[i] = len(peak_indices) / duration_s

    acf_valid = acf_freqs[np.isfinite(acf_freqs)]
    pc_valid = pc_freqs[np.isfinite(pc_freqs)]
    acf_med = float(np.median(acf_valid)) if len(acf_valid) > 0 else np.nan
    pc_med = float(np.median(pc_valid)) if len(pc_valid) > 0 else np.nan
    return acf_med, pc_med


def quick_spearman(dataset, predictions):
    """Compute Spearman correlations for LPD and GPD separately, return combined."""
    by_type = {'lpd': {'e': [], 'p': []}, 'gpd': {'e': [], 'p': []}}
    for entry in dataset:
        mat_name = entry['mat_name']
        if mat_name not in predictions:
            continue
        pred = predictions[mat_name]
        if not np.isfinite(pred):
            continue
        expert = entry['expert_consensus_freq']
        if not np.isfinite(expert):
            continue
        by_type[entry['subdir']]['e'].append(expert)
        by_type[entry['subdir']]['p'].append(pred)

    results = {}
    for ptype in ['lpd', 'gpd']:
        e = np.array(by_type[ptype]['e'])
        p = np.array(by_type[ptype]['p'])
        if len(e) >= 3:
            rs, _ = spearmanr(p, e)
            results[ptype] = rs
        else:
            results[ptype] = np.nan

    lpd_rs = results.get('lpd', np.nan)
    gpd_rs = results.get('gpd', np.nan)
    if np.isfinite(lpd_rs) and np.isfinite(gpd_rs):
        results['combined'] = (lpd_rs + gpd_rs) / 2.0
    else:
        results['combined'] = np.nan
    return results


def main():
    t0 = time.time()

    # ── Load dataset ──────────────────────────────────────────────────
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # ── Pre-compute Method A for all segments ─────────────────────────
    print("Pre-computing Method A frequencies...")
    method_a_freqs = {}  # mat_name -> freq
    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        method_a_freqs[entry['mat_name']] = get_method_a_freq(data, fs)
        if (idx + 1) % 50 == 0:
            print(f"  Method A: {idx+1}/{len(dataset)}")
    print(f"Method A done: {len(method_a_freqs)} segments, "
          f"{sum(1 for v in method_a_freqs.values() if np.isfinite(v))} valid")

    # ── Pre-process base (notch+bandpass+bipolar) for all segments ────
    print("Pre-processing base signals (notch + bandpass + bipolar)...")
    base_segments = {}  # mat_name -> (bipolar_seg, subdir)
    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        seg = preprocess_base(data, fs)
        base_segments[entry['mat_name']] = (seg, entry['subdir'])
        if (idx + 1) % 50 == 0:
            print(f"  Base preprocess: {idx+1}/{len(dataset)}")
    print(f"Base preprocessing done: {len(base_segments)} segments")

    # ── Pre-compute lowpass-filtered versions for each lowpass_hz ─────
    print("Pre-computing lowpass-filtered versions...")
    lowpass_cache = {}  # (lowpass_hz, mat_name) -> seg_lp
    for lp_hz in GRID['lowpass_hz']:
        count = 0
        for mat_name, (seg, subdir) in base_segments.items():
            lowpass_cache[(lp_hz, mat_name)] = apply_lowpass(seg, fs, lp_hz)
            count += 1
        print(f"  Lowpass {lp_hz}Hz: {count} segments cached")
    print("Lowpass caching done.")

    # ── Grid search ───────────────────────────────────────────────────
    print(f"\nStarting grid search over {len(ALL_COMBOS)} parameter combinations...")
    top_results = []  # list of (combined_spearman, params_dict, spearman_dict)

    mat_names_list = list(base_segments.keys())
    subdirs = {mn: base_segments[mn][1] for mn in mat_names_list}

    for combo_idx, combo_vals in enumerate(ALL_COMBOS):
        params = dict(zip(PARAM_NAMES, combo_vals))
        lp_hz = params['lowpass_hz']
        sigma = params['smoothing_sigma']
        min_lag = params['acf_min_lag']
        acf_thr = params['acf_peak_threshold']
        ph_frac = params['peak_height_frac']

        predictions = {}
        for mat_name in mat_names_list:
            seg_lp = lowpass_cache[(lp_hz, mat_name)]
            acf_med, pc_med = compute_method_b_and_peakcount(
                seg_lp, FS, sigma, min_lag, acf_thr, ph_frac
            )
            a_freq = method_a_freqs.get(mat_name, np.nan)

            # Triple median
            vals = [v for v in [a_freq, acf_med, pc_med] if np.isfinite(v)]
            if vals:
                predictions[mat_name] = float(np.median(vals))
            else:
                predictions[mat_name] = np.nan

        # Evaluate
        spearman = quick_spearman(dataset, predictions)
        combined_rs = spearman.get('combined', np.nan)

        top_results.append((combined_rs, params, spearman, predictions))

        # Progress report every 50 iterations
        if (combo_idx + 1) % 50 == 0 or (combo_idx + 1) == len(ALL_COMBOS):
            elapsed = time.time() - t0
            pct = 100.0 * (combo_idx + 1) / len(ALL_COMBOS)
            print(f"\n--- Progress: {combo_idx+1}/{len(ALL_COMBOS)} ({pct:.1f}%) "
                  f"[{elapsed:.0f}s elapsed] ---")
            # Show current top 5
            valid_results = [(rs, p, sp) for rs, p, sp, _ in top_results
                             if np.isfinite(rs)]
            valid_results.sort(key=lambda x: -x[0])
            print(f"  Top combos so far (by combined Spearman):")
            for rank, (rs, p, sp) in enumerate(valid_results[:5], 1):
                print(f"    #{rank}: combined_rs={rs:.4f} "
                      f"(LPD={sp['lpd']:.4f}, GPD={sp['gpd']:.4f}) "
                      f"lp={p['lowpass_hz']}, sig={p['smoothing_sigma']}, "
                      f"lag={p['acf_min_lag']}, thr={p['acf_peak_threshold']}, "
                      f"ph={p['peak_height_frac']}")

    # ── Final top 10 ──────────────────────────────────────────────────
    valid_results_full = [(rs, p, sp, pred) for rs, p, sp, pred in top_results
                          if np.isfinite(rs)]
    valid_results_full.sort(key=lambda x: -x[0])

    print("\n" + "=" * 80)
    print("TOP 10 PARAMETER COMBINATIONS (by combined Spearman)")
    print("=" * 80)
    header = (f"{'Rank':>4s}  {'Comb_Rs':>8s}  {'LPD_Rs':>8s}  {'GPD_Rs':>8s}  "
              f"{'LP_Hz':>5s}  {'Sigma':>6s}  {'MinLag':>6s}  {'ACF_Thr':>7s}  {'PH_Frac':>7s}")
    print(header)
    print("-" * len(header))
    for rank, (rs, p, sp, _) in enumerate(valid_results_full[:10], 1):
        print(f"{rank:>4d}  {rs:>8.4f}  {sp['lpd']:>8.4f}  {sp['gpd']:>8.4f}  "
              f"{p['lowpass_hz']:>5}  {p['smoothing_sigma']:>6.3f}  "
              f"{p['acf_min_lag']:>6.2f}  {p['acf_peak_threshold']:>7.3f}  "
              f"{p['peak_height_frac']:>7.2f}")

    # ── Full evaluation of the BEST combo ─────────────────────────────
    if valid_results_full:
        best_rs, best_params, best_sp, best_preds = valid_results_full[0]
        print(f"\n\nBest params: {best_params}")
        print(f"Best combined Spearman: {best_rs:.4f}")

        # Full evaluation via harness
        exp_name = (f"gridsearch_best_lp{best_params['lowpass_hz']}"
                    f"_sig{best_params['smoothing_sigma']}"
                    f"_lag{best_params['acf_min_lag']}"
                    f"_thr{best_params['acf_peak_threshold']}"
                    f"_ph{best_params['peak_height_frac']}")
        print(f"\nRunning full evaluate_predictions() for: {exp_name}")
        evaluate_predictions(dataset, best_preds, exp_name)

        # ── Best combo + Bayesian nudge ───────────────────────────────
        print("\n\nRunning best combo + Bayesian nudge...")
        bayesian_preds = {}
        for mat_name, freq in best_preds.items():
            subdir = subdirs.get(mat_name, 'lpd')
            bayesian_preds[mat_name] = bayesian_nudge(freq, subdir)

        exp_name_bayesian = exp_name + "_bayesian"
        evaluate_predictions(dataset, bayesian_preds, exp_name_bayesian)

    total_time = time.time() - t0
    print(f"\n\nTotal grid search time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
