"""
Generate threshold sweep figure: spatial extent agreement vs threshold.

For PD (PDProfiler) and RDA (RDA-PLV), sweep the threshold used to
binarize per-channel scores, converting to spatial extent = count(score > T) / 18.
Plot MAE, Pearson r, and ICC(3,1) as functions of the threshold.

Layout: 2x3 subplot
  Row 1: PD (LPD + GPD)
  Row 2: RDA (LRDA + GRDA)
  Col 1: MAE vs threshold
  Col 2: Pearson r vs threshold
  Col 3: ICC(3,1) vs threshold

Usage:
    conda run -n morgoth python paper_materials/generate_threshold_sweep.py
"""

import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
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
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'threshold_sweep_spatial.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──
SPATIAL_EXPERTS = ['LB', 'PH', 'SZ']
THRESHOLDS = np.arange(0.01, 1.00, 0.01)

COLORS = {
    'lpd': '#F29030',
    'gpd': '#F0D020',
    'lrda': '#7CB342',
    'grda': '#81D4FA',
}

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
# ICC(3,1)
# ═══════════════════════════════════════════════════════════════════

def icc31(ratings_matrix):
    """ICC(3,1) on a matrix that may contain NaN. Drops rows with any NaN."""
    mask = ~np.isnan(ratings_matrix).any(axis=1)
    mat = ratings_matrix[mask]
    n, k = mat.shape
    if n < 10 or k < 2:
        return np.nan

    grand_mean = np.mean(mat)
    subj_means = np.mean(mat, axis=1)
    rater_means = np.mean(mat, axis=0)

    BSS = k * np.sum((subj_means - grand_mean) ** 2)
    BRS = n * np.sum((rater_means - grand_mean) ** 2)
    TSS = np.sum((mat - grand_mean) ** 2)
    ESS = TSS - BSS - BRS

    BMS = BSS / (n - 1)
    EMS = ESS / ((n - 1) * (k - 1))

    return (BMS - EMS) / (BMS + (k - 1) * EMS)


# ═══════════════════════════════════════════════════════════════════
# EEG loading
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


# ═══════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════

def load_data():
    """Load annotations and segment labels, return qualifying segments."""
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    # Merge subtype into annotations
    ann = ann.merge(sl[['mat_file', 'subtype', 'excluded']], on='mat_file', how='left')

    # Filter to non-excluded
    ann = ann[ann['excluded'] != True].copy()
    sl = sl[sl['excluded'] != True].copy()

    return ann, sl


def get_spatial_segments(ann, subtypes):
    """Get segments with spatial_extent from at least 2 of LB/PH/SZ.

    Returns:
        mat_files: list of mat_file names
        expert_matrix: (n, 3) array with LB/PH/SZ spatial_extent (NaN where missing)
        expert_mean: (n,) array of nanmean across raters
        subtype_list: list of subtype strings
    """
    mask = (ann['rater'].isin(SPATIAL_EXPERTS) &
            ann['spatial_extent'].notna() &
            ann['subtype'].isin(subtypes))
    sub = ann[mask].copy()

    pivot = sub.pivot_table(index='mat_file', columns='rater',
                            values='spatial_extent', aggfunc='first')

    for exp in SPATIAL_EXPERTS:
        if exp not in pivot.columns:
            pivot[exp] = np.nan

    # At least 2 raters
    n_valid = pivot[SPATIAL_EXPERTS].notna().sum(axis=1)
    pivot = pivot[n_valid >= 2]

    mat_files = pivot.index.tolist()
    expert_matrix = pivot[SPATIAL_EXPERTS].values  # (n, 3)
    expert_mean = np.nanmean(expert_matrix, axis=1)

    # Get subtypes
    subtype_map = ann.drop_duplicates('mat_file').set_index('mat_file')['subtype'].to_dict()
    subtype_list = [subtype_map.get(m, '') for m in mat_files]

    return mat_files, expert_matrix, expert_mean, subtype_list


# ═══════════════════════════════════════════════════════════════════
# Inference: get per-channel scores
# ═══════════════════════════════════════════════════════════════════

def get_pd_channel_scores(mat_files, subtype_list):
    """Run PDProfiler, return (n, 18) channel probability matrix."""
    from pd_profiler import PDProfiler
    pc = PDProfiler()

    scores = np.full((len(mat_files), 18), np.nan)
    n_ok = 0

    for i, (mat_file, subtype) in enumerate(zip(mat_files, subtype_list)):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [PD channel scores] {i+1}/{len(mat_files)}...", flush=True)

        if subtype not in ('lpd', 'gpd'):
            continue

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            bipolar = mono_to_bipolar(mono)
            result = pc.characterize(bipolar[:18], subtype=subtype)
            scores[i] = np.array(result['channel_probs'])
            n_ok += 1
        except Exception:
            continue

    print(f"  [PD channel scores] Done: {n_ok}/{len(mat_files)}", flush=True)
    return scores


def get_rda_channel_scores(mat_files, sl):
    """Run rda_spatial_extent, return (n, 18) channel score matrix."""
    from rda_spatial_extent import rda_spatial_extent

    freq_map = sl.set_index('mat_file')['pdchar_freq_hz'].to_dict()

    scores = np.full((len(mat_files), 18), np.nan)
    n_ok = 0

    for i, mat_file in enumerate(mat_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [RDA channel scores] {i+1}/{len(mat_files)}...", flush=True)

        freq_hz = freq_map.get(mat_file, np.nan)
        if not np.isfinite(freq_hz) or freq_hz <= 0:
            continue

        mono = load_eeg_mono(mat_file)
        if mono is None:
            continue

        try:
            bipolar = mono_to_bipolar(mono)
            result = rda_spatial_extent(bipolar[:18], freq_hz, threshold=0.5)
            scores[i] = np.array(result['channel_scores'])
            n_ok += 1
        except Exception:
            continue

    print(f"  [RDA channel scores] Done: {n_ok}/{len(mat_files)}", flush=True)
    return scores


# ═══════════════════════════════════════════════════════════════════
# Threshold sweep metrics
# ═══════════════════════════════════════════════════════════════════

def sweep_metrics(channel_scores, expert_matrix, expert_mean, subtype_list, thresholds):
    """Sweep thresholds and compute MAE, Pearson r, ICC for each.

    Parameters
    ----------
    channel_scores : (n, 18) array of per-channel scores (NaN rows = missing)
    expert_matrix : (n, 3) array of expert spatial extent ratings
    expert_mean : (n,) array of mean expert spatial extent
    subtype_list : list of subtype strings
    thresholds : array of thresholds to sweep

    Returns
    -------
    results : dict mapping label -> dict with 'mae', 'pearson', 'icc' arrays
    """
    # Valid mask: rows where we have both channel scores and expert data
    valid = ~np.isnan(channel_scores).any(axis=1) & np.isfinite(expert_mean)

    subtypes_unique = sorted(set(s for s in subtype_list if s))
    subtype_arr = np.array(subtype_list)

    results = {}

    # Combined + per-subtype
    groups = {'combined': valid}
    for st in subtypes_unique:
        groups[st] = valid & (subtype_arr == st)

    for label, mask in groups.items():
        n = mask.sum()
        mae_arr = np.full(len(thresholds), np.nan)
        pearson_arr = np.full(len(thresholds), np.nan)
        icc_arr = np.full(len(thresholds), np.nan)

        if n < 5:
            results[label] = {'mae': mae_arr, 'pearson': pearson_arr, 'icc': icc_arr, 'n': n}
            continue

        cs = channel_scores[mask]   # (n, 18)
        em = expert_mean[mask]      # (n,)
        ex = expert_matrix[mask]    # (n, 3)

        for ti, t in enumerate(thresholds):
            pred = np.sum(cs > t, axis=1) / 18.0

            # MAE
            mae_arr[ti] = np.mean(np.abs(pred - em))

            # Pearson r
            if np.std(pred) > 0 and np.std(em) > 0:
                r, _ = pearsonr(pred, em)
                pearson_arr[ti] = r

            # ICC(3,1): experts + algo as columns
            # Build (n, 4) matrix: [LB, PH, SZ, algo]
            ratings = np.column_stack([ex, pred[:, None]])  # (n, 4)
            icc_arr[ti] = icc31(ratings)

        results[label] = {'mae': mae_arr, 'pearson': pearson_arr, 'icc': icc_arr, 'n': n}

    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_figure(pd_results, rda_results, thresholds):
    """Create the 2x3 threshold sweep figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Spatial Extent: Agreement vs Threshold', fontsize=15, fontweight='bold', y=0.98)

    metrics = [
        ('mae', 'MAE (lower is better)', 'min'),
        ('pearson', 'Pearson r (higher is better)', 'max'),
        ('icc', 'ICC(3,1) (higher is better)', 'max'),
    ]

    row_data = [
        ('PD (PDProfiler)', pd_results, ['lpd', 'gpd']),
        ('RDA (RDA-PLV)', rda_results, ['lrda', 'grda']),
    ]

    for row_i, (row_label, results, subtypes) in enumerate(row_data):
        for col_i, (metric_key, metric_label, opt_dir) in enumerate(metrics):
            ax = axes[row_i, col_i]

            # Plot per-subtype lines
            for st in subtypes:
                if st in results:
                    vals = results[st][metric_key]
                    n = results[st]['n']
                    if np.all(np.isnan(vals)):
                        continue
                    ax.plot(thresholds, vals, color=COLORS[st], linewidth=1.5,
                            label=f'{st.upper()} (n={n})', alpha=0.8)

            # Plot combined as thick black line
            if 'combined' in results:
                vals = results['combined'][metric_key]
                n = results['combined']['n']
                if not np.all(np.isnan(vals)):
                    ax.plot(thresholds, vals, color='black', linewidth=2.5,
                            label=f'Combined (n={n})')

                    # Find optimal threshold
                    finite_mask = np.isfinite(vals)
                    if finite_mask.any():
                        if opt_dir == 'min':
                            best_idx = np.nanargmin(vals)
                        else:
                            best_idx = np.nanargmax(vals)
                        best_t = thresholds[best_idx]
                        best_v = vals[best_idx]

                        ax.axvline(best_t, color='red', linestyle='--', linewidth=1, alpha=0.7)
                        ax.plot(best_t, best_v, 'ro', markersize=8, zorder=5)

                        # Annotate
                        y_off = 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0])
                        if opt_dir == 'min':
                            y_off = -y_off
                        ax.annotate(f'T={best_t:.2f}\n{metric_key.upper()}={best_v:.3f}',
                                    xy=(best_t, best_v),
                                    xytext=(best_t + 0.08, best_v + y_off),
                                    fontsize=8, color='red', fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

            ax.set_xlabel('Threshold', fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

            if col_i == 0:
                ax.set_title(f'{row_label}  |  {metric_label}', fontsize=11, fontweight='bold')
            else:
                ax.set_title(metric_label, fontsize=11, fontweight='bold')

        # Add row label on left
        axes[row_i, 0].set_ylabel(f'{row_label}\n{metrics[0][1]}', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {OUT_PATH}")
    plt.close(fig)


def print_optimal_thresholds(results, label, thresholds):
    """Print optimal thresholds for each metric."""
    print(f"\n{'='*60}")
    print(f"Optimal thresholds: {label}")
    print(f"{'='*60}")
    if 'combined' not in results:
        print("  No combined results available.")
        return

    vals = results['combined']
    for metric_key, opt_dir in [('mae', 'min'), ('pearson', 'max'), ('icc', 'max')]:
        v = vals[metric_key]
        finite = np.isfinite(v)
        if not finite.any():
            print(f"  {metric_key.upper()}: no valid values")
            continue
        if opt_dir == 'min':
            best_idx = np.nanargmin(v)
        else:
            best_idx = np.nanargmax(v)
        print(f"  {metric_key.upper()}: threshold={thresholds[best_idx]:.2f}, value={v[best_idx]:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("Loading data...", flush=True)
    ann, sl = load_data()

    # ── PD segments ──
    print("\n--- PD (LPD + GPD) ---", flush=True)
    pd_mats, pd_expert, pd_mean, pd_subtypes = get_spatial_segments(ann, ['lpd', 'gpd'])
    print(f"  {len(pd_mats)} segments with spatial extent from >= 2 raters", flush=True)

    from spatial_cache_utils import use_cache, load_spatial_cache
    _use_cache = use_cache()
    _cache = load_spatial_cache() if _use_cache else None

    if _use_cache and _cache:
        print("Loading PDProfiler scores from cache...", flush=True)
        pd_scores = np.full((len(pd_mats), 18), np.nan)
        for i, mf in enumerate(pd_mats):
            entry = _cache.get(mf, {})
            probs = entry.get('pdchar_channel_probs')
            if probs is not None:
                pd_scores[i] = probs
    else:
        print("Running PDProfiler inference...", flush=True)
        pd_scores = get_pd_channel_scores(pd_mats, pd_subtypes)

    print("Sweeping PD thresholds...", flush=True)
    pd_results = sweep_metrics(pd_scores, pd_expert, pd_mean, pd_subtypes, THRESHOLDS)

    # ── RDA segments ──
    print("\n--- RDA (LRDA + GRDA) ---", flush=True)
    rda_mats, rda_expert, rda_mean, rda_subtypes = get_spatial_segments(ann, ['lrda', 'grda'])
    print(f"  {len(rda_mats)} segments with spatial extent from >= 2 raters", flush=True)

    if _use_cache and _cache:
        print("Loading RDA-PLV scores from cache...", flush=True)
        rda_scores = np.full((len(rda_mats), 18), np.nan)
        for i, mf in enumerate(rda_mats):
            entry = _cache.get(mf, {})
            scores = entry.get('rda_channel_scores')
            if scores is not None:
                rda_scores[i] = scores
    else:
        print("Running RDA-PLV inference...", flush=True)
        rda_scores = get_rda_channel_scores(rda_mats, sl)

    print("Sweeping RDA thresholds...", flush=True)
    rda_results = sweep_metrics(rda_scores, rda_expert, rda_mean, rda_subtypes, THRESHOLDS)

    # ── Plot ──
    print("\nGenerating figure...", flush=True)
    plot_figure(pd_results, rda_results, THRESHOLDS)

    # ── Print optimal thresholds ──
    print_optimal_thresholds(pd_results, 'PD (PDProfiler)', THRESHOLDS)
    print_optimal_thresholds(rda_results, 'RDA (RDA-PLV)', THRESHOLDS)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
