"""
Generate IRR comparison figure: expert-expert vs expert-algorithm reliability.

Computes ICC(3,1), Percentage Agreement, and MAE for frequency and spatial extent.
Compares 4 expert raters (LB, PH, SZ, MW) with PDProfiler (W05) and Tautan et al.

Layout: 2x4 subplot
  Row 1: RDA (LRDA + GRDA)
  Row 2: PD (LPD + GPD)
  Col A: ICC for Frequency
  Col B: PA for Frequency (by subtype)
  Col C: ICC for Spatial Extent
  Col D: PA for Spatial Extent (by subtype)

Usage:
    nohup conda run -n morgoth python -u paper_materials/generate_fig_irr.py > /tmp/fig_irr.log 2>&1 &
"""

import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'figS1_irr_comparison.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──
EXPERTS = ['LB', 'PH', 'SZ', 'MW']
SPATIAL_THRESHOLD = 0.62  # Optimized via threshold sweep (generate_threshold_sweep.py)

# Subtype colors
COLORS = {
    'lpd': '#F29030',
    'gpd': '#F0D020',
    'lrda': '#7CB342',
    'grda': '#81D4FA',
}

# Neutral bar colors
BAR_COLORS = {
    'ee': '#4A90D9',     # Blue for expert-expert
    'pdchar': '#E67E22',  # Orange for PDChar/W05
    'tautan': '#27AE60',  # Green for Tautan
    'ours': '#8E44AD',   # Purple for RDA-PLV (Ours)
}

# Frequency bins: <1, 1-1.5, 1.5-2, 2-2.5, 2.5-3, >3
FREQ_BINS = [0, 1.0, 1.5, 2.0, 2.5, 3.0, np.inf]
FREQ_BIN_LABELS = ['<1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '>3']

# Spatial extent bins (by channel count, converted to ratio)
SPATIAL_BINS = [0, 0.222, 0.556, 0.778, 1.001]  # 0-4ch, 5-10ch, 11-14ch, 15-18ch
SPATIAL_BIN_LABELS = ['1-4ch', '5-10ch', '11-14ch', '15-18ch']

MONO_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]


# ═══════════════════════════════════════════════════════════════════
# Statistical functions
# ═══════════════════════════════════════════════════════════════════

def _icc31_on_complete(ratings_matrix):
    """Core ICC(3,1) on a complete (no NaN) matrix. Returns NaN if < 10 rows or < 2 cols."""
    n, k = ratings_matrix.shape
    if n < 10 or k < 2:
        return np.nan

    grand_mean = np.mean(ratings_matrix)
    subj_means = np.mean(ratings_matrix, axis=1)
    rater_means = np.mean(ratings_matrix, axis=0)

    BSS = k * np.sum((subj_means - grand_mean) ** 2)
    BRS = n * np.sum((rater_means - grand_mean) ** 2)
    TSS = np.sum((ratings_matrix - grand_mean) ** 2)
    ESS = TSS - BSS - BRS

    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    BMS = BSS / df_subjects
    EMS = ESS / df_error

    return (BMS - EMS) / (BMS + (k - 1) * EMS)


def _find_best_rater_subset(ratings_matrix, min_rows=10):
    """Find the largest subset of raters with >= min_rows complete cases.

    Tries all raters first, then drops raters with most NaN one at a time.
    Returns (sub_matrix, col_indices) or (None, None) if no valid subset.
    """
    n, k = ratings_matrix.shape
    cols = list(range(k))

    # Try all columns first
    complete = ~np.isnan(ratings_matrix[:, cols]).any(axis=1)
    if complete.sum() >= min_rows:
        return ratings_matrix[np.ix_(complete, cols)], cols

    # Progressively drop the rater with most NaN in incomplete rows
    remaining = list(cols)
    while len(remaining) >= 2:
        sub = ratings_matrix[:, remaining]
        complete = ~np.isnan(sub).any(axis=1)
        if complete.sum() >= min_rows:
            return sub[complete], remaining

        # Find rater with most NaN and drop
        nan_counts = np.isnan(sub).sum(axis=0)
        worst = remaining[np.argmax(nan_counts)]
        remaining.remove(worst)

    return None, None


def compute_icc31(ratings_matrix):
    """
    Compute ICC(3,1) — two-way mixed, single measures, consistency.

    ratings_matrix: (n_subjects, n_raters) numpy array. May contain NaN.
    Finds the best subset of raters with >= 10 complete rows.
    """
    sub, cols = _find_best_rater_subset(ratings_matrix, min_rows=10)
    if sub is None:
        return np.nan
    return _icc31_on_complete(sub)


def compute_icc31_ci(ratings_matrix, alpha=0.05):
    """Compute ICC(3,1) with confidence interval using F-distribution.

    Finds the best subset of raters with >= 10 complete rows.
    """
    from scipy.stats import f as fdist

    sub, cols = _find_best_rater_subset(ratings_matrix, min_rows=10)
    if sub is None:
        return np.nan, np.nan, np.nan

    n, k = sub.shape
    grand_mean = np.mean(sub)
    subj_means = np.mean(sub, axis=1)
    rater_means = np.mean(sub, axis=0)

    BSS = k * np.sum((subj_means - grand_mean) ** 2)
    BRS = n * np.sum((rater_means - grand_mean) ** 2)
    TSS = np.sum((sub - grand_mean) ** 2)
    ESS = TSS - BSS - BRS

    df_subjects = n - 1
    df_error = (n - 1) * (k - 1)

    BMS = BSS / df_subjects
    EMS = ESS / df_error

    icc = (BMS - EMS) / (BMS + (k - 1) * EMS)

    F_val = BMS / EMS
    FL = F_val / fdist.ppf(1 - alpha / 2, df_subjects, df_error)
    FU = F_val / fdist.ppf(alpha / 2, df_subjects, df_error)

    ci_lower = (FL - 1) / (FL + k - 1)
    ci_upper = (FU - 1) / (FU + k - 1)

    return icc, ci_lower, ci_upper


def categorize_frequency(freq):
    """Assign frequency to bin index."""
    for i in range(len(FREQ_BINS) - 1):
        if freq <= FREQ_BINS[i + 1]:
            return i
    return len(FREQ_BINS) - 2


def categorize_spatial(extent):
    """Assign spatial extent to bin index."""
    for i in range(len(SPATIAL_BINS) - 1):
        if extent <= SPATIAL_BINS[i + 1]:
            return i
    return len(SPATIAL_BINS) - 2


def compute_pa(ratings_matrix, categorize_fn):
    """
    Compute percentage agreement with NaN handling.

    ratings_matrix: (n_subjects, n_raters) — may contain NaN
    categorize_fn: function to categorize continuous values into bins

    PA = mean over all rater pairs of (N_agree_in_pair / N_valid_in_pair) * 100
    Skip pairs with < 5 valid segments.
    """
    n, k = ratings_matrix.shape
    if n < 1 or k < 2:
        return np.nan

    # Categorize all values (NaN stays as -1 sentinel)
    cat_matrix = np.full_like(ratings_matrix, -1, dtype=int)
    for i in range(n):
        for j in range(k):
            if np.isfinite(ratings_matrix[i, j]):
                cat_matrix[i, j] = categorize_fn(ratings_matrix[i, j])

    # Compute pairwise agreement, only using segments where both raters have data
    n_pairs = 0
    total_agreement = 0
    for r1, r2 in combinations(range(k), 2):
        both_valid = (cat_matrix[:, r1] >= 0) & (cat_matrix[:, r2] >= 0)
        n_valid = both_valid.sum()
        if n_valid < 5:
            continue
        agree = np.sum(cat_matrix[both_valid, r1] == cat_matrix[both_valid, r2])
        total_agreement += agree / n_valid
        n_pairs += 1

    if n_pairs == 0:
        return np.nan

    return (total_agreement / n_pairs) * 100


def compute_mae(values_a, values_b):
    """Mean absolute error between two arrays."""
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(values_a[mask] - values_b[mask]))


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load annotations and segment labels."""
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    # Merge subtype into annotations
    ann = ann.merge(sl[['mat_file', 'subtype', 'excluded']], on='mat_file', how='left')

    # Filter to non-excluded
    ann = ann[ann['excluded'] != True].copy()
    sl = sl[sl['excluded'] != True].copy()

    return ann, sl


def get_expert_frequency_matrix(ann, subtypes):
    """
    Build (n_segments, n_experts) matrix of expert frequency ratings.
    Returns matrix (with NaN where raters didn't label), mat_files list.
    Only includes segments with at least 2 raters having data.
    """
    mask = (ann['rater'].isin(EXPERTS) &
            ann['frequency_hz'].notna() &
            ann['subtype'].isin(subtypes))
    sub = ann[mask].copy()

    # Pivot to get rater columns
    pivot = sub.pivot_table(index='mat_file', columns='rater',
                           values='frequency_hz', aggfunc='first')

    # Ensure all expert columns exist (some may be missing entirely)
    for exp in EXPERTS:
        if exp not in pivot.columns:
            pivot[exp] = np.nan

    # Keep only segments with at least 2 raters having data
    n_valid = pivot[EXPERTS].notna().sum(axis=1)
    pivot = pivot[n_valid >= 2]

    matrix = pivot[EXPERTS].values
    mat_files = pivot.index.tolist()

    return matrix, mat_files


def get_expert_spatial_matrix(ann, subtypes):
    """
    Build (n_segments, n_experts) matrix of expert spatial extent ratings.
    Returns matrix (with NaN where raters didn't label), mat_files list.
    Only includes segments with at least 2 raters having data.
    """
    mask = (ann['rater'].isin(EXPERTS) &
            ann['spatial_extent'].notna() &
            ann['subtype'].isin(subtypes))
    sub = ann[mask].copy()

    pivot = sub.pivot_table(index='mat_file', columns='rater',
                           values='spatial_extent', aggfunc='first')

    # Ensure all expert columns exist
    for exp in EXPERTS:
        if exp not in pivot.columns:
            pivot[exp] = np.nan

    # Keep only segments with at least 2 raters having data
    n_valid = pivot[EXPERTS].notna().sum(axis=1)
    pivot = pivot[n_valid >= 2]

    matrix = pivot[EXPERTS].values
    mat_files = pivot.index.tolist()

    return matrix, mat_files


def get_subtypes_for_mats(sl, mat_files):
    """Get subtype for each mat_file."""
    sl_map = sl.set_index('mat_file')['subtype'].to_dict()
    return [sl_map.get(m, 'unknown') for m in mat_files]


# ═══════════════════════════════════════════════════════════════════
# Model inference
# ═══════════════════════════════════════════════════════════════════

def load_eeg_mono(mat_file):
    """Load 19-channel monopolar EEG at 200 Hz."""
    mat_path = EEG_DIR / mat_file
    if not mat_path.exists():
        return None
    mat = sio.loadmat(str(mat_path))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] < 19 or seg.shape[1] < 2000:
        return None
    return seg[:19, :2000]


def mono_to_bipolar(mono_19ch):
    """Convert 19-channel monopolar to 18-channel bipolar banana montage."""
    ch_idx = {ch: i for i, ch in enumerate(MONO_CHANNELS)}
    bipolar = np.zeros((18, mono_19ch.shape[1]))
    for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
        bipolar[i] = mono_19ch[ch_idx[ch1]] - mono_19ch[ch_idx[ch2]]
    return bipolar


def run_pdchar_spatial(mat_files, subtypes_list):
    """Run PDProfiler on LPD/GPD segments, return spatial extent array."""
    from pd_profiler import PDProfiler
    charzer = PDProfiler()

    results = np.full(len(mat_files), np.nan)
    n_ok = 0

    for i, (mat_file, subtype) in enumerate(zip(mat_files, subtypes_list)):
        if i % 100 == 0:
            print(f"  [PDChar spatial] {i}/{len(mat_files)}...", flush=True)

        if subtype not in ('lpd', 'gpd'):
            continue

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            bipolar = mono_to_bipolar(mono)
            result = charzer.characterize(bipolar[:18], subtype=subtype)
            channel_probs = result['channel_probs']
            # Spatial extent = count(prob > 0.38) / 18
            se = sum(1 for p in channel_probs if p > SPATIAL_THRESHOLD) / 18.0
            results[i] = se
            n_ok += 1
        except Exception as e:
            continue

    print(f"  [PDChar spatial] Done: {n_ok}/{len(mat_files)}", flush=True)
    return results


def run_tautan_spatial(mat_files):
    """Run Tautan et al. on segments, return spatial extent array."""
    from pd_detect_alternate import pd_detect_alternate

    results = np.full(len(mat_files), np.nan)
    n_ok = 0

    for i, mat_file in enumerate(mat_files):
        if i % 100 == 0:
            print(f"  [Tautan spatial] {i}/{len(mat_files)}...", flush=True)

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            result = pd_detect_alternate(mono, 200, pk_detect='apd')
            se = result.get('spatial_extent', np.nan)
            if se is not None and np.isfinite(se):
                results[i] = se
                n_ok += 1
        except Exception as e:
            continue

    print(f"  [Tautan spatial] Done: {n_ok}/{len(mat_files)}", flush=True)
    return results


def run_tautan_frequency(mat_files):
    """Run Tautan et al. for frequency estimation."""
    from pd_detect_alternate import pd_detect_alternate

    results = np.full(len(mat_files), np.nan)
    n_ok = 0

    for i, mat_file in enumerate(mat_files):
        if i % 100 == 0:
            print(f"  [Tautan freq] {i}/{len(mat_files)}...", flush=True)

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            result = pd_detect_alternate(mono, 200, pk_detect='apd')
            freq = result.get('event_frequency', np.nan)
            if freq is not None and np.isfinite(freq):
                results[i] = freq
                n_ok += 1
        except Exception as e:
            continue

    print(f"  [Tautan freq] Done: {n_ok}/{len(mat_files)}", flush=True)
    return results


def run_rda_plv_spatial(mat_files, sl, mode='threshold'):
    """Run RDA spatial extent via PLV for LRDA/GRDA segments.

    Parameters
    ----------
    mat_files : list of str
    sl : DataFrame with 'mat_file' and 'pdchar_freq_hz' columns
    mode : 'threshold' or 'continuous'
        'threshold' uses spatial_extent (PLV×Amp at threshold=0.15)
        'continuous' uses spatial_extent_continuous (mean PLV)

    Returns
    -------
    results : ndarray (n,) with spatial extent values or NaN
    """
    from rda_spatial_extent import rda_spatial_extent

    freq_map = sl.set_index('mat_file')['pdchar_freq_hz'].to_dict()

    results = np.full(len(mat_files), np.nan)
    n_ok = 0

    for i, mat_file in enumerate(mat_files):
        if i % 100 == 0:
            print(f"  [RDA-PLV spatial ({mode})] {i}/{len(mat_files)}...", flush=True)

        freq_hz = freq_map.get(mat_file, np.nan)
        if not np.isfinite(freq_hz) or freq_hz <= 0:
            continue

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            bipolar = mono_to_bipolar(mono)
            result = rda_spatial_extent(bipolar[:18], freq_hz, threshold=0.15, metric='plv_amp')
            if mode == 'continuous':
                se = result['spatial_extent_continuous']
            else:
                se = result['spatial_extent']
            if np.isfinite(se):
                results[i] = se
                n_ok += 1
        except Exception:
            continue

    print(f"  [RDA-PLV spatial ({mode})] Done: {n_ok}/{len(mat_files)}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Cached frequency from segment_labels
# ═══════════════════════════════════════════════════════════════════

def get_cached_algo_freq(sl, mat_files, column):
    """Get cached algorithm frequency from segment_labels.csv."""
    sl_map = sl.set_index('mat_file')[column].to_dict()
    results = np.array([sl_map.get(m, np.nan) for m in mat_files], dtype=float)
    n_ok = np.isfinite(results).sum()
    print(f"  [Cached {column}] {n_ok}/{len(mat_files)} available", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════
# Compute metrics per condition
# ═══════════════════════════════════════════════════════════════════

def compute_all_metrics_freq(expert_matrix, algo_values, categorize_fn):
    """
    Compute ICC, PA, MAE for frequency.
    expert_matrix: (n, n_experts) — may contain NaN
    algo_values: (n,) or None
    """
    # Expert-expert (ICC drops NaN rows internally, PA handles NaN pairwise)
    ee_icc, ee_ci_lo, ee_ci_hi = compute_icc31_ci(expert_matrix)
    ee_pa = compute_pa(expert_matrix, categorize_fn)

    expert_mean = np.nanmean(expert_matrix, axis=1)

    results = {
        'ee_icc': ee_icc, 'ee_icc_ci': (ee_ci_lo, ee_ci_hi),
        'ee_pa': ee_pa,
    }

    if algo_values is not None:
        # Expert+Algorithm: add algo as extra column
        valid = np.isfinite(algo_values)
        if valid.sum() >= 10:
            combined = np.column_stack([expert_matrix[valid], algo_values[valid].reshape(-1, 1)])
            # ICC will internally drop rows with any NaN; PA handles NaN pairwise
            ea_icc, ea_ci_lo, ea_ci_hi = compute_icc31_ci(combined)
            ea_pa = compute_pa(combined, categorize_fn)
            ea_mae = compute_mae(expert_mean[valid], algo_values[valid])
            results['ea_icc'] = ea_icc
            results['ea_icc_ci'] = (ea_ci_lo, ea_ci_hi)
            results['ea_pa'] = ea_pa
            results['ea_mae'] = ea_mae
            results['ea_n'] = int(valid.sum())
        else:
            results['ea_icc'] = np.nan
            results['ea_icc_ci'] = (np.nan, np.nan)
            results['ea_pa'] = np.nan
            results['ea_mae'] = np.nan
            results['ea_n'] = int(valid.sum())

    return results


def compute_all_metrics_spatial(expert_matrix, algo_values, categorize_fn):
    """Same as freq but for spatial extent."""
    return compute_all_metrics_freq(expert_matrix, algo_values, categorize_fn)


def compute_pa_by_subtype(expert_matrix, algo_values, subtypes, categorize_fn, target_subtypes):
    """Compute PA split by subtype. Returns dict[subtype -> PA]."""
    results = {}
    subtypes = np.array(subtypes)

    for st in target_subtypes:
        mask = subtypes == st
        if mask.sum() < 5:
            results[st] = {'ee_pa': np.nan, 'ea_pa': np.nan}
            continue

        ee_pa = compute_pa(expert_matrix[mask], categorize_fn)

        if algo_values is not None:
            valid = mask & np.isfinite(algo_values)
            if valid.sum() >= 5:
                combined = np.column_stack([expert_matrix[valid], algo_values[valid].reshape(-1, 1)])
                ea_pa = compute_pa(combined, categorize_fn)
            else:
                ea_pa = np.nan
        else:
            ea_pa = np.nan

        results[st] = {'ee_pa': ee_pa, 'ea_pa': ea_pa}

    return results


def compute_icc_by_subtype(expert_matrix, algo_values, subtypes, target_subtypes):
    """Compute ICC split by subtype. Returns dict[subtype -> {ee_icc, ee_ci, ea_icc, ea_ci}]."""
    results = {}
    subtypes = np.array(subtypes)

    for st in target_subtypes:
        mask = subtypes == st
        if mask.sum() < 5:
            results[st] = {'ee_icc': np.nan, 'ee_icc_ci': (np.nan, np.nan),
                           'ea_icc': np.nan, 'ea_icc_ci': (np.nan, np.nan)}
            continue

        ee_icc, ee_lo, ee_hi = compute_icc31_ci(expert_matrix[mask])

        ea_icc, ea_lo, ea_hi = np.nan, np.nan, np.nan
        if algo_values is not None:
            valid = mask & np.isfinite(algo_values)
            if valid.sum() >= 10:
                combined = np.column_stack([expert_matrix[valid], algo_values[valid].reshape(-1, 1)])
                ea_icc, ea_lo, ea_hi = compute_icc31_ci(combined)

        results[st] = {
            'ee_icc': ee_icc, 'ee_icc_ci': (ee_lo, ee_hi),
            'ea_icc': ea_icc, 'ea_icc_ci': (ea_lo, ea_hi),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_icc_bars(ax, values, labels, ci_list=None, title='', ylabel='ICC(3,1)',
                  bar_colors=None):
    """Plot grouped ICC bars with error bars."""
    x = np.arange(len(values))
    if bar_colors is None:
        bar_colors = [BAR_COLORS['ee'], BAR_COLORS['pdchar'], BAR_COLORS['tautan']][:len(values)]
    bars = ax.bar(x, values, width=0.5, color=bar_colors,
                  edgecolor='black', linewidth=0.8, zorder=3)

    if ci_list is not None:
        for i, (val, ci) in enumerate(zip(values, ci_list)):
            if np.isfinite(val) and ci is not None and np.isfinite(ci[0]):
                ax.errorbar(i, val, yerr=[[val - ci[0]], [ci[1] - val]],
                           fmt='none', color='black', capsize=4, linewidth=1.5, zorder=4)

    # Add value labels
    for i, v in enumerate(values):
        if np.isfinite(v):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom',
                   fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha='right')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_icc_bars_by_subtype(ax, icc_data, subtypes_order, group_labels, title='', ylabel='ICC(3,1)'):
    """
    Plot ICC bars split by subtype, same layout as PA bars.

    icc_data: dict[group_label -> dict[subtype -> icc_value]]
    """
    n_groups = len(group_labels)
    n_subtypes = len(subtypes_order)

    bar_width = 0.15
    group_width = n_subtypes * bar_width + 0.05

    for g_idx, g_label in enumerate(group_labels):
        for s_idx, st in enumerate(subtypes_order):
            x = g_idx * (group_width + 0.15) + s_idx * bar_width
            val = icc_data[g_label].get(st, np.nan)
            if np.isfinite(val):
                ax.bar(x, val, width=bar_width, color=COLORS[st],
                       edgecolor='black', linewidth=0.6, zorder=3)
                ax.text(x, val + 0.02, f'{val:.2f}', ha='center', va='bottom',
                       fontsize=6, fontweight='bold')
            else:
                ax.bar(x, 0, width=bar_width, color='lightgray',
                       edgecolor='black', linewidth=0.6, zorder=3, hatch='//')

    group_centers = []
    for g_idx in range(n_groups):
        center = g_idx * (group_width + 0.15) + (n_subtypes - 1) * bar_width / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=7.5, rotation=15, ha='right')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.75, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.8)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_pa_bars_by_subtype(ax, pa_data, subtypes_order, group_labels, title='', ylabel='PA (%)'):
    """
    Plot PA bars split by subtype.

    pa_data: dict[group_label -> dict[subtype -> PA]]
    subtypes_order: list of subtypes
    group_labels: list of group names (e.g., ['Annotators', 'Ann. & Ours', 'Ann. & Tautan'])
    """
    n_groups = len(group_labels)
    n_subtypes = len(subtypes_order)

    bar_width = 0.15
    group_width = n_subtypes * bar_width + 0.05

    for g_idx, g_label in enumerate(group_labels):
        for s_idx, st in enumerate(subtypes_order):
            x = g_idx * (group_width + 0.15) + s_idx * bar_width
            val = pa_data[g_label].get(st, np.nan)
            if np.isfinite(val):
                ax.bar(x, val, width=bar_width, color=COLORS[st],
                       edgecolor='black', linewidth=0.6, zorder=3)
                ax.text(x, val + 1, f'{val:.0f}', ha='center', va='bottom',
                       fontsize=6.5, fontweight='bold')
            else:
                ax.bar(x, 0, width=bar_width, color='lightgray',
                       edgecolor='black', linewidth=0.6, zorder=3, hatch='//')

    # X-axis labels
    group_centers = []
    for g_idx in range(n_groups):
        center = g_idx * (group_width + 0.15) + (n_subtypes - 1) * bar_width / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=7.5, rotation=15, ha='right')
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("IRR Comparison Figure: Expert-Expert vs Expert-Algorithm")
    print("=" * 70)

    ann, sl = load_data()

    # ──────────────────────────────────────────────────────────────
    # FREQUENCY DATA
    # ──────────────────────────────────────────────────────────────
    print("\n── Frequency: PD (LPD + GPD) ──")
    pd_freq_matrix, pd_freq_mats = get_expert_frequency_matrix(ann, ['lpd', 'gpd'])
    pd_freq_subtypes = get_subtypes_for_mats(sl, pd_freq_mats)
    print(f"  PD freq segments with >=2 experts: {len(pd_freq_mats)}")

    # Get cached algo frequencies for PD
    pd_pdchar_freq = get_cached_algo_freq(sl, pd_freq_mats, 'pdchar_freq_hz')
    pd_tautan_freq = get_cached_algo_freq(sl, pd_freq_mats, 'tautan_freq_hz')

    print("\n── Frequency: RDA (LRDA + GRDA) ──")
    rda_freq_matrix, rda_freq_mats = get_expert_frequency_matrix(ann, ['lrda', 'grda'])
    rda_freq_subtypes = get_subtypes_for_mats(sl, rda_freq_mats)
    print(f"  RDA freq segments with >=2 experts: {len(rda_freq_mats)}")

    # Get cached algo frequencies for RDA
    rda_pdchar_freq = get_cached_algo_freq(sl, rda_freq_mats, 'pdchar_freq_hz')  # W05
    rda_tautan_freq = get_cached_algo_freq(sl, rda_freq_mats, 'tautan_freq_hz')

    # ──────────────────────────────────────────────────────────────
    # SPATIAL EXTENT DATA
    # ──────────────────────────────────────────────────────────────
    print("\n── Spatial: PD (LPD + GPD) ──")
    pd_spat_matrix, pd_spat_mats = get_expert_spatial_matrix(ann, ['lpd', 'gpd'])
    pd_spat_subtypes = get_subtypes_for_mats(sl, pd_spat_mats)
    print(f"  PD spatial segments with >=2 experts: {len(pd_spat_mats)}")

    print("\n── Spatial: RDA (LRDA + GRDA) ──")
    rda_spat_matrix, rda_spat_mats = get_expert_spatial_matrix(ann, ['lrda', 'grda'])
    rda_spat_subtypes = get_subtypes_for_mats(sl, rda_spat_mats)
    print(f"  RDA spatial segments with >=2 experts: {len(rda_spat_mats)}")

    # ──────────────────────────────────────────────────────────────
    # RUN MODEL INFERENCE FOR SPATIAL EXTENT (or load from cache)
    # ──────────────────────────────────────────────────────────────
    from spatial_cache_utils import use_cache, load_spatial_cache, \
        get_cached_pdchar_spatial, get_cached_tautan_spatial, get_cached_rda_spatial

    if use_cache():
        print("\n── Loading spatial results from cache ──")
        _cache = load_spatial_cache()
        pd_pdchar_spat = get_cached_pdchar_spatial(_cache, pd_spat_mats)
        pd_tautan_spat = get_cached_tautan_spatial(_cache, pd_spat_mats)
        rda_tautan_spat = get_cached_tautan_spatial(_cache, rda_spat_mats)
        rda_plv_spat_thr = get_cached_rda_spatial(_cache, rda_spat_mats, mode='threshold')
        rda_plv_spat_cont = get_cached_rda_spatial(_cache, rda_spat_mats, mode='continuous')
        print(f"  Loaded from cache: PD={np.sum(~np.isnan(pd_pdchar_spat))}, RDA={np.sum(~np.isnan(rda_plv_spat_thr))}")
    else:
        print("\n── Running PDProfiler for PD spatial extent ──")
        pd_pdchar_spat = run_pdchar_spatial(pd_spat_mats, pd_spat_subtypes)

        print("\n── Running Tautan for PD spatial extent ──")
        pd_tautan_spat = run_tautan_spatial(pd_spat_mats)

        print("\n── Running Tautan for RDA spatial extent ──")
        rda_tautan_spat = run_tautan_spatial(rda_spat_mats)

        print("\n── Running RDA-PLV for RDA spatial extent (threshold mode) ──")
        rda_plv_spat_thr = run_rda_plv_spatial(rda_spat_mats, sl, mode='threshold')

        print("\n── Running RDA-PLV for RDA spatial extent (continuous mode) ──")
        rda_plv_spat_cont = run_rda_plv_spatial(rda_spat_mats, sl, mode='continuous')

    # Also run Tautan frequency for segments missing cached values
    # (Use cached values from segment_labels where available)

    # ──────────────────────────────────────────────────────────────
    # COMPUTE METRICS
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70)

    # -- PD Frequency --
    print("\n── PD Frequency Metrics ──")
    pd_freq_ee = compute_all_metrics_freq(pd_freq_matrix, None, categorize_frequency)
    pd_freq_pdchar = compute_all_metrics_freq(pd_freq_matrix, pd_pdchar_freq, categorize_frequency)
    pd_freq_tautan = compute_all_metrics_freq(pd_freq_matrix, pd_tautan_freq, categorize_frequency)

    print(f"  ee-IRR: ICC={pd_freq_ee['ee_icc']:.3f} [{pd_freq_ee['ee_icc_ci'][0]:.3f}, {pd_freq_ee['ee_icc_ci'][1]:.3f}], PA={pd_freq_ee['ee_pa']:.1f}%")
    print(f"  ea-IRR (PDChar): ICC={pd_freq_pdchar.get('ea_icc', np.nan):.3f}, PA={pd_freq_pdchar.get('ea_pa', np.nan):.1f}%, MAE={pd_freq_pdchar.get('ea_mae', np.nan):.3f}")
    print(f"  ea-IRR (Tautan): ICC={pd_freq_tautan.get('ea_icc', np.nan):.3f}, PA={pd_freq_tautan.get('ea_pa', np.nan):.1f}%, MAE={pd_freq_tautan.get('ea_mae', np.nan):.3f}")

    # PD Frequency PA by subtype
    pd_freq_pa_ee_sub = compute_pa_by_subtype(pd_freq_matrix, None, pd_freq_subtypes, categorize_frequency, ['lpd', 'gpd'])
    pd_freq_pa_pdchar_sub = compute_pa_by_subtype(pd_freq_matrix, pd_pdchar_freq, pd_freq_subtypes, categorize_frequency, ['lpd', 'gpd'])
    pd_freq_pa_tautan_sub = compute_pa_by_subtype(pd_freq_matrix, pd_tautan_freq, pd_freq_subtypes, categorize_frequency, ['lpd', 'gpd'])

    # -- RDA Frequency --
    print("\n── RDA Frequency Metrics ──")
    rda_freq_ee = compute_all_metrics_freq(rda_freq_matrix, None, categorize_frequency)
    rda_freq_pdchar = compute_all_metrics_freq(rda_freq_matrix, rda_pdchar_freq, categorize_frequency)
    rda_freq_tautan = compute_all_metrics_freq(rda_freq_matrix, rda_tautan_freq, categorize_frequency)

    print(f"  ee-IRR: ICC={rda_freq_ee['ee_icc']:.3f} [{rda_freq_ee['ee_icc_ci'][0]:.3f}, {rda_freq_ee['ee_icc_ci'][1]:.3f}], PA={rda_freq_ee['ee_pa']:.1f}%")
    print(f"  ea-IRR (W05): ICC={rda_freq_pdchar.get('ea_icc', np.nan):.3f}, PA={rda_freq_pdchar.get('ea_pa', np.nan):.1f}%, MAE={rda_freq_pdchar.get('ea_mae', np.nan):.3f}")
    print(f"  ea-IRR (Tautan): ICC={rda_freq_tautan.get('ea_icc', np.nan):.3f}, PA={rda_freq_tautan.get('ea_pa', np.nan):.1f}%, MAE={rda_freq_tautan.get('ea_mae', np.nan):.3f}")

    # RDA Frequency PA by subtype
    rda_freq_pa_ee_sub = compute_pa_by_subtype(rda_freq_matrix, None, rda_freq_subtypes, categorize_frequency, ['lrda', 'grda'])
    rda_freq_pa_pdchar_sub = compute_pa_by_subtype(rda_freq_matrix, rda_pdchar_freq, rda_freq_subtypes, categorize_frequency, ['lrda', 'grda'])
    rda_freq_pa_tautan_sub = compute_pa_by_subtype(rda_freq_matrix, rda_tautan_freq, rda_freq_subtypes, categorize_frequency, ['lrda', 'grda'])

    # -- PD Spatial --
    print("\n── PD Spatial Metrics ──")
    pd_spat_ee = compute_all_metrics_spatial(pd_spat_matrix, None, categorize_spatial)
    pd_spat_pdchar = compute_all_metrics_spatial(pd_spat_matrix, pd_pdchar_spat, categorize_spatial)
    pd_spat_tautan = compute_all_metrics_spatial(pd_spat_matrix, pd_tautan_spat, categorize_spatial)

    print(f"  ee-IRR: ICC={pd_spat_ee['ee_icc']:.3f} [{pd_spat_ee['ee_icc_ci'][0]:.3f}, {pd_spat_ee['ee_icc_ci'][1]:.3f}], PA={pd_spat_ee['ee_pa']:.1f}%")
    print(f"  ea-IRR (PDChar): ICC={pd_spat_pdchar.get('ea_icc', np.nan):.3f}, PA={pd_spat_pdchar.get('ea_pa', np.nan):.1f}%, MAE={pd_spat_pdchar.get('ea_mae', np.nan):.3f}")
    print(f"  ea-IRR (Tautan): ICC={pd_spat_tautan.get('ea_icc', np.nan):.3f}, PA={pd_spat_tautan.get('ea_pa', np.nan):.1f}%, MAE={pd_spat_tautan.get('ea_mae', np.nan):.3f}")

    # PD Spatial PA by subtype
    pd_spat_pa_ee_sub = compute_pa_by_subtype(pd_spat_matrix, None, pd_spat_subtypes, categorize_spatial, ['lpd', 'gpd'])
    pd_spat_pa_pdchar_sub = compute_pa_by_subtype(pd_spat_matrix, pd_pdchar_spat, pd_spat_subtypes, categorize_spatial, ['lpd', 'gpd'])
    pd_spat_pa_tautan_sub = compute_pa_by_subtype(pd_spat_matrix, pd_tautan_spat, pd_spat_subtypes, categorize_spatial, ['lpd', 'gpd'])

    # -- RDA Spatial --
    print("\n── RDA Spatial Metrics ──")
    rda_spat_ee = compute_all_metrics_spatial(rda_spat_matrix, None, categorize_spatial)
    rda_spat_tautan = compute_all_metrics_spatial(rda_spat_matrix, rda_tautan_spat, categorize_spatial)

    # RDA-PLV: use threshold mode (PLV×Amp at T=0.15)
    rda_spat_plv = compute_all_metrics_spatial(rda_spat_matrix, rda_plv_spat_thr, categorize_spatial)
    rda_plv_spat_best = rda_plv_spat_thr
    plv_mode_chosen = 'threshold'
    print(f"  RDA-PLV mode: {plv_mode_chosen} (ICC={rda_spat_plv.get('ea_icc', np.nan):.3f})")

    print(f"  ee-IRR: ICC={rda_spat_ee['ee_icc']:.3f} [{rda_spat_ee['ee_icc_ci'][0]:.3f}, {rda_spat_ee['ee_icc_ci'][1]:.3f}], PA={rda_spat_ee['ee_pa']:.1f}%")
    print(f"  ea-IRR (Tautan): ICC={rda_spat_tautan.get('ea_icc', np.nan):.3f}, PA={rda_spat_tautan.get('ea_pa', np.nan):.1f}%, MAE={rda_spat_tautan.get('ea_mae', np.nan):.3f}")
    print(f"  ea-IRR (RDA-PLV): ICC={rda_spat_plv.get('ea_icc', np.nan):.3f}, PA={rda_spat_plv.get('ea_pa', np.nan):.1f}%, MAE={rda_spat_plv.get('ea_mae', np.nan):.3f}")

    # RDA Spatial PA by subtype
    rda_spat_pa_ee_sub = compute_pa_by_subtype(rda_spat_matrix, None, rda_spat_subtypes, categorize_spatial, ['lrda', 'grda'])
    rda_spat_pa_tautan_sub = compute_pa_by_subtype(rda_spat_matrix, rda_tautan_spat, rda_spat_subtypes, categorize_spatial, ['lrda', 'grda'])
    rda_spat_pa_plv_sub = compute_pa_by_subtype(rda_spat_matrix, rda_plv_spat_best, rda_spat_subtypes, categorize_spatial, ['lrda', 'grda'])

    # ──────────────────────────────────────────────────────────────
    # FIGURE
    # ──────────────────────────────────────────────────────────────
    print("\n── Generating figure ──")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.35, left=0.06, right=0.96, top=0.90, bottom=0.12)

    # ── Compute ICC by subtype ──
    rda_freq_icc_pdchar_sub = compute_icc_by_subtype(rda_freq_matrix, rda_pdchar_freq, rda_freq_subtypes, ['lrda', 'grda'])
    rda_freq_icc_tautan_sub = compute_icc_by_subtype(rda_freq_matrix, rda_tautan_freq, rda_freq_subtypes, ['lrda', 'grda'])
    rda_freq_icc_ee_sub = compute_icc_by_subtype(rda_freq_matrix, None, rda_freq_subtypes, ['lrda', 'grda'])

    rda_spat_icc_tautan_sub = compute_icc_by_subtype(rda_spat_matrix, rda_tautan_spat, rda_spat_subtypes, ['lrda', 'grda'])
    rda_spat_icc_plv_sub = compute_icc_by_subtype(rda_spat_matrix, rda_plv_spat_best, rda_spat_subtypes, ['lrda', 'grda'])
    rda_spat_icc_ee_sub = compute_icc_by_subtype(rda_spat_matrix, None, rda_spat_subtypes, ['lrda', 'grda'])

    # ── Row 0: RDA ──
    # (A) RDA Frequency ICC by subtype
    rda_freq_icc_data = {
        'Annotators': {st: rda_freq_icc_ee_sub[st]['ee_icc'] for st in ['lrda', 'grda']},
        'Ann. & Ours': {st: rda_freq_icc_pdchar_sub[st]['ea_icc'] for st in ['lrda', 'grda']},
        'Ann. & Tautan': {st: rda_freq_icc_tautan_sub[st]['ea_icc'] for st in ['lrda', 'grda']},
    }
    plot_icc_bars_by_subtype(axes[0, 0], rda_freq_icc_data, ['lrda', 'grda'],
                              ['Annotators', 'Ann. & Ours', 'Ann. & Tautan'],
                              title=f'RDA Frequency ICC')

    # (B) RDA Frequency PA by subtype
    rda_freq_pa_data = {
        'Annotators': {st: rda_freq_pa_ee_sub[st]['ee_pa'] for st in ['lrda', 'grda']},
        'Ann. & Ours': {st: rda_freq_pa_pdchar_sub[st]['ea_pa'] for st in ['lrda', 'grda']},
        'Ann. & Tautan': {st: rda_freq_pa_tautan_sub[st]['ea_pa'] for st in ['lrda', 'grda']},
    }
    plot_pa_bars_by_subtype(axes[0, 1], rda_freq_pa_data, ['lrda', 'grda'],
                            ['Annotators', 'Ann. & Ours', 'Ann. & Tautan'],
                            title=f'RDA Frequency PA')

    # (C) RDA Spatial ICC by subtype — includes RDA-PLV (Ours)
    rda_spat_icc_data = {
        'Annotators': {st: rda_spat_icc_ee_sub[st]['ee_icc'] for st in ['lrda', 'grda']},
        'Ann. & Ours': {st: rda_spat_icc_plv_sub[st]['ea_icc'] for st in ['lrda', 'grda']},
        'Ann. & Tautan': {st: rda_spat_icc_tautan_sub[st]['ea_icc'] for st in ['lrda', 'grda']},
    }
    plot_icc_bars_by_subtype(axes[0, 2], rda_spat_icc_data, ['lrda', 'grda'],
                              ['Annotators', 'Ann. & Ours', 'Ann. & Tautan'],
                              title=f'RDA Spatial ICC')

    # (D) RDA Spatial PA by subtype — includes RDA-PLV (Ours)
    rda_spat_pa_data = {
        'Annotators': {st: rda_spat_pa_ee_sub[st]['ee_pa'] for st in ['lrda', 'grda']},
        'Ann. & Ours': {st: rda_spat_pa_plv_sub[st]['ea_pa'] for st in ['lrda', 'grda']},
        'Ann. & Tautan': {st: rda_spat_pa_tautan_sub[st]['ea_pa'] for st in ['lrda', 'grda']},
    }
    plot_pa_bars_by_subtype(axes[0, 3], rda_spat_pa_data, ['lrda', 'grda'],
                            ['Annotators', 'Ann. & Ours', 'Ann. & Tautan'],
                            title=f'RDA Spatial PA')

    # ── Row 1: PD ──
    # (A) PD Frequency ICC
    # ── Compute PD ICC by subtype ──
    pd_freq_icc_pdchar_sub = compute_icc_by_subtype(pd_freq_matrix, pd_pdchar_freq, pd_freq_subtypes, ['lpd', 'gpd'])
    pd_freq_icc_tautan_sub = compute_icc_by_subtype(pd_freq_matrix, pd_tautan_freq, pd_freq_subtypes, ['lpd', 'gpd'])
    pd_freq_icc_ee_sub = compute_icc_by_subtype(pd_freq_matrix, None, pd_freq_subtypes, ['lpd', 'gpd'])

    pd_spat_icc_pdchar_sub = compute_icc_by_subtype(pd_spat_matrix, pd_pdchar_spat, pd_spat_subtypes, ['lpd', 'gpd'])
    pd_spat_icc_tautan_sub = compute_icc_by_subtype(pd_spat_matrix, pd_tautan_spat, pd_spat_subtypes, ['lpd', 'gpd'])
    pd_spat_icc_ee_sub = compute_icc_by_subtype(pd_spat_matrix, None, pd_spat_subtypes, ['lpd', 'gpd'])

    # (A) PD Frequency ICC by subtype
    pd_freq_icc_data = {
        'Annotators': {st: pd_freq_icc_ee_sub[st]['ee_icc'] for st in ['lpd', 'gpd']},
        'Ann. & PDChar': {st: pd_freq_icc_pdchar_sub[st]['ea_icc'] for st in ['lpd', 'gpd']},
        'Ann. & Tautan': {st: pd_freq_icc_tautan_sub[st]['ea_icc'] for st in ['lpd', 'gpd']},
    }
    plot_icc_bars_by_subtype(axes[1, 0], pd_freq_icc_data, ['lpd', 'gpd'],
                              ['Annotators', 'Ann. & PDChar', 'Ann. & Tautan'],
                              title=f'PD Frequency ICC')

    # (B) PD Frequency PA by subtype
    pd_freq_pa_data = {
        'Annotators': {st: pd_freq_pa_ee_sub[st]['ee_pa'] for st in ['lpd', 'gpd']},
        'Ann. & PDChar': {st: pd_freq_pa_pdchar_sub[st]['ea_pa'] for st in ['lpd', 'gpd']},
        'Ann. & Tautan': {st: pd_freq_pa_tautan_sub[st]['ea_pa'] for st in ['lpd', 'gpd']},
    }
    plot_pa_bars_by_subtype(axes[1, 1], pd_freq_pa_data, ['lpd', 'gpd'],
                            ['Annotators', 'Ann. & PDChar', 'Ann. & Tautan'],
                            title=f'PD Frequency PA')

    # (C) PD Spatial ICC by subtype
    pd_spat_icc_data = {
        'Annotators': {st: pd_spat_icc_ee_sub[st]['ee_icc'] for st in ['lpd', 'gpd']},
        'Ann. & PDChar': {st: pd_spat_icc_pdchar_sub[st]['ea_icc'] for st in ['lpd', 'gpd']},
        'Ann. & Tautan': {st: pd_spat_icc_tautan_sub[st]['ea_icc'] for st in ['lpd', 'gpd']},
    }
    plot_icc_bars_by_subtype(axes[1, 2], pd_spat_icc_data, ['lpd', 'gpd'],
                              ['Annotators', 'Ann. & PDChar', 'Ann. & Tautan'],
                              title=f'PD Spatial ICC')

    # (D) PD Spatial PA by subtype
    pd_spat_pa_data = {
        'Annotators': {st: pd_spat_pa_ee_sub[st]['ee_pa'] for st in ['lpd', 'gpd']},
        'Ann. & PDChar': {st: pd_spat_pa_pdchar_sub[st]['ea_pa'] for st in ['lpd', 'gpd']},
        'Ann. & Tautan': {st: pd_spat_pa_tautan_sub[st]['ea_pa'] for st in ['lpd', 'gpd']},
    }
    plot_pa_bars_by_subtype(axes[1, 3], pd_spat_pa_data, ['lpd', 'gpd'],
                            ['Annotators', 'Ann. & PDChar', 'Ann. & Tautan'],
                            title=f'PD Spatial PA')

    # ── Legend ──
    # Subtype legend
    legend_patches = [mpatches.Patch(facecolor=COLORS[st], edgecolor='black', linewidth=0.6,
                                      label=st.upper()) for st in ['lpd', 'gpd', 'lrda', 'grda']]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4,
               fontsize=9, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 0.01))

    # Row labels
    fig.text(0.01, 0.72, 'RDA', fontsize=14, fontweight='bold', va='center', rotation=90)
    fig.text(0.01, 0.32, 'PD', fontsize=14, fontweight='bold', va='center', rotation=90)

    # Panel labels
    for col, letter in enumerate(['A', 'B', 'C', 'D']):
        for row in range(2):
            ax = axes[row, col]
            ax.text(-0.12, 1.08, f'{letter}{row+1}', transform=ax.transAxes,
                   fontsize=12, fontweight='bold', va='top')

    fig.suptitle('Inter-Rater Reliability: Expert-Expert vs Expert-Algorithm',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n  Figure saved: {OUT_PATH}")

    # ──────────────────────────────────────────────────────────────
    # SUMMARY TABLE
    # ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL METRICS")
    print("=" * 70)

    def fmt(v, decimals=3):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "  N/A"
        return f"{v:.{decimals}f}"

    def fmt_ci(ci):
        if ci is None or np.isnan(ci[0]):
            return ""
        return f" [{ci[0]:.3f}, {ci[1]:.3f}]"

    print(f"\n{'Measure':<30} {'ee-IRR':>10} {'ea-IRR (W05/PDChar)':>22} {'ea-IRR (Tautan)':>18}")
    print("-" * 82)

    # PD Frequency
    print(f"{'PD Freq ICC':<30} {fmt(pd_freq_ee['ee_icc']):>10}{fmt_ci(pd_freq_ee['ee_icc_ci'])}"
          f" {fmt(pd_freq_pdchar.get('ea_icc')):>10}"
          f" {fmt(pd_freq_tautan.get('ea_icc')):>10}")
    print(f"{'PD Freq PA (%)':<30} {fmt(pd_freq_ee['ee_pa'], 1):>10}"
          f" {fmt(pd_freq_pdchar.get('ea_pa'), 1):>22}"
          f" {fmt(pd_freq_tautan.get('ea_pa'), 1):>18}")
    print(f"{'PD Freq MAE (Hz)':<30} {'':>10}"
          f" {fmt(pd_freq_pdchar.get('ea_mae')):>22}"
          f" {fmt(pd_freq_tautan.get('ea_mae')):>18}")

    # RDA Frequency
    print(f"{'RDA Freq ICC':<30} {fmt(rda_freq_ee['ee_icc']):>10}"
          f" {fmt(rda_freq_pdchar.get('ea_icc')):>22}"
          f" {fmt(rda_freq_tautan.get('ea_icc')):>18}")
    print(f"{'RDA Freq PA (%)':<30} {fmt(rda_freq_ee['ee_pa'], 1):>10}"
          f" {fmt(rda_freq_pdchar.get('ea_pa'), 1):>22}"
          f" {fmt(rda_freq_tautan.get('ea_pa'), 1):>18}")
    print(f"{'RDA Freq MAE (Hz)':<30} {'':>10}"
          f" {fmt(rda_freq_pdchar.get('ea_mae')):>22}"
          f" {fmt(rda_freq_tautan.get('ea_mae')):>18}")

    # PD Spatial
    print(f"{'PD Spatial ICC':<30} {fmt(pd_spat_ee['ee_icc']):>10}"
          f" {fmt(pd_spat_pdchar.get('ea_icc')):>22}"
          f" {fmt(pd_spat_tautan.get('ea_icc')):>18}")
    print(f"{'PD Spatial PA (%)':<30} {fmt(pd_spat_ee['ee_pa'], 1):>10}"
          f" {fmt(pd_spat_pdchar.get('ea_pa'), 1):>22}"
          f" {fmt(pd_spat_tautan.get('ea_pa'), 1):>18}")
    print(f"{'PD Spatial MAE':<30} {'':>10}"
          f" {fmt(pd_spat_pdchar.get('ea_mae')):>22}"
          f" {fmt(pd_spat_tautan.get('ea_mae')):>18}")

    # RDA Spatial
    print(f"{'RDA Spatial ICC':<30} {fmt(rda_spat_ee['ee_icc']):>10}"
          f" {'N/A':>22}"
          f" {fmt(rda_spat_tautan.get('ea_icc')):>18}")
    print(f"{'RDA Spatial PA (%)':<30} {fmt(rda_spat_ee['ee_pa'], 1):>10}"
          f" {'N/A':>22}"
          f" {fmt(rda_spat_tautan.get('ea_pa'), 1):>18}")
    print(f"{'RDA Spatial MAE':<30} {'':>10}"
          f" {'N/A':>22}"
          f" {fmt(rda_spat_tautan.get('ea_mae')):>18}")

    # RDA-PLV Spatial (Ours)
    print(f"\n  RDA-PLV Spatial (Ours, mode={plv_mode_chosen}):")
    print(f"    ICC={fmt(rda_spat_plv.get('ea_icc'))}, PA={fmt(rda_spat_plv.get('ea_pa'), 1)}%, MAE={fmt(rda_spat_plv.get('ea_mae'))}")

    # Per-subtype PA breakdown
    print(f"\n{'Per-Subtype PA (%)':}")
    print("-" * 70)
    for st in ['lpd', 'gpd']:
        print(f"  {st.upper()} Freq PA: ee={fmt(pd_freq_pa_ee_sub[st]['ee_pa'], 1)}, "
              f"W05={fmt(pd_freq_pa_pdchar_sub[st]['ea_pa'], 1)}, "
              f"Tautan={fmt(pd_freq_pa_tautan_sub[st]['ea_pa'], 1)}")
        print(f"  {st.upper()} Spat PA: ee={fmt(pd_spat_pa_ee_sub[st]['ee_pa'], 1)}, "
              f"PDChar={fmt(pd_spat_pa_pdchar_sub[st]['ea_pa'], 1)}, "
              f"Tautan={fmt(pd_spat_pa_tautan_sub[st]['ea_pa'], 1)}")
    for st in ['lrda', 'grda']:
        print(f"  {st.upper()} Freq PA: ee={fmt(rda_freq_pa_ee_sub[st]['ee_pa'], 1)}, "
              f"W05={fmt(rda_freq_pa_pdchar_sub[st]['ea_pa'], 1)}, "
              f"Tautan={fmt(rda_freq_pa_tautan_sub[st]['ea_pa'], 1)}")
        print(f"  {st.upper()} Spat PA: ee={fmt(rda_spat_pa_ee_sub[st]['ee_pa'], 1)}, "
              f"Tautan={fmt(rda_spat_pa_tautan_sub[st]['ea_pa'], 1)}, "
              f"RDA-PLV={fmt(rda_spat_pa_plv_sub[st]['ea_pa'], 1)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
