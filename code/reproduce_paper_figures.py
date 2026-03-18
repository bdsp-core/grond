"""
Reproduce Figures 5 & 6 from Tăuțan et al. 2025 (J Neural Eng 22:066027).

Compares expert-expert IRR with expert-algorithm IRR using:
  - ICC (intra-class correlation coefficient)
  - PA (mean pairwise percentage agreement)

for frequency and spatial extent of LPD, GPD segments.

Methods compared:
  - PD2a (Alexandra's original = our Method A / pd_detect_alternate)
  - Method B (pointiness+ACF, no normalization)
  - Method B+pnorm (pointiness+ACF with percentile normalization)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import f as f_dist
import hdf5storage
import scipy.io
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf

DATA_DIR = CODE_DIR.parent / 'data' / 'dataset_eeg'
ANN_DIR = CODE_DIR.parent / 'data' / 'annotations'
RESULTS_DIR = CODE_DIR.parent / 'results'


def compute_icc3(ratings_matrix):
    """Compute ICC(3,1) — two-way mixed, single measures, consistency.

    ratings_matrix: (n_subjects, n_raters) array
    Returns: ICC value, 95% CI lower, 95% CI upper
    """
    n, k = ratings_matrix.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan

    # Remove rows with any NaN
    mask = ~np.any(np.isnan(ratings_matrix), axis=1)
    ratings_matrix = ratings_matrix[mask]
    n = ratings_matrix.shape[0]
    if n < 2:
        return np.nan, np.nan, np.nan

    grand_mean = np.mean(ratings_matrix)
    row_means = np.mean(ratings_matrix, axis=1)
    col_means = np.mean(ratings_matrix, axis=0)

    # Sum of squares
    SSR = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
    SSC = n * np.sum((col_means - grand_mean) ** 2)  # between raters
    SST = np.sum((ratings_matrix - grand_mean) ** 2)  # total
    SSE = SST - SSR - SSC  # residual

    # Mean squares
    MSR = SSR / (n - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    # ICC(3,1) = (MSR - MSE) / (MSR + (k-1)*MSE)
    icc = (MSR - MSE) / (MSR + (k - 1) * MSE)

    # Confidence interval (Shrout & Fleiss, 1979)
    F_val = MSR / MSE if MSE > 0 else np.inf
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    try:
        F_L = F_val / f_dist.ppf(0.975, df1, df2)
        F_U = F_val / f_dist.ppf(0.025, df1, df2)
        ci_lower = (F_L - 1) / (F_L + k - 1)
        ci_upper = (F_U - 1) / (F_U + k - 1)
    except:
        ci_lower, ci_upper = np.nan, np.nan

    return icc, ci_lower, ci_upper


def categorize_frequency(freq):
    """Categorize frequency into bins per paper: <1, 1-1.5, 1.5-2, 2-2.5, 2.5-3, >3 Hz."""
    if not np.isfinite(freq) or freq <= 0:
        return 0
    elif freq < 1:
        return 1
    elif freq <= 1.5:
        return 2
    elif freq <= 2:
        return 3
    elif freq <= 2.5:
        return 4
    elif freq <= 3:
        return 5
    else:
        return 6


def categorize_spatial(spatial):
    """Categorize spatial extent into bins per paper: 1-4, 5-10, 11-14, 15-18 channels."""
    if not np.isfinite(spatial) or spatial <= 0:
        return 0
    n_ch = int(round(spatial * 18))
    if n_ch <= 4:
        return 1
    elif n_ch <= 10:
        return 2
    elif n_ch <= 14:
        return 3
    else:
        return 4


def compute_pairwise_agreement(ratings_matrix, categorize_fn=None):
    """Compute mean pairwise percentage agreement.

    PA = (1/a) * sum over pairs of (N_agree / N_total) * 100
    """
    n, k = ratings_matrix.shape
    if n < 1 or k < 2:
        return np.nan

    if categorize_fn is not None:
        cat_matrix = np.array([[categorize_fn(ratings_matrix[i, j])
                                 for j in range(k)] for i in range(n)])
    else:
        cat_matrix = ratings_matrix

    agreements = []
    for j1 in range(k):
        for j2 in range(j1 + 1, k):
            mask = ~(np.isnan(ratings_matrix[:, j1]) | np.isnan(ratings_matrix[:, j2]))
            if np.sum(mask) == 0:
                continue
            c1 = cat_matrix[mask, j1]
            c2 = cat_matrix[mask, j2]
            agree = np.sum(c1 == c2)
            agreements.append(agree / np.sum(mask) * 100)

    return np.mean(agreements) if agreements else np.nan


def load_per_expert_annotations():
    """Load annotations keeping individual expert ratings separate."""
    records = {}

    for pattern, subdir in [('LPDS', 'lpd'), ('GPDS', 'gpd')]:
        for expert_file in sorted(ANN_DIR.glob(f'{pattern}_*')):
            expert = expert_file.stem.split('_')[1]
            df = pd.read_csv(expert_file)
            for _, row in df.iterrows():
                mat_name = Path(row['files']).stem.replace('_score', '') + '.mat'
                key = (subdir, mat_name)
                if key not in records:
                    records[key] = {}
                freq = row['frequency']
                spatial = row['spatial']
                # Handle non-numeric spatial
                try:
                    spatial = float(spatial)
                except (ValueError, TypeError):
                    spatial = np.nan
                try:
                    freq = float(freq)
                except (ValueError, TypeError):
                    freq = np.nan
                records[key][expert] = {'frequency': freq, 'spatial': spatial}

    return records


def run_algorithms_on_dataset(records):
    """Run all three algorithms on each segment in the annotated dataset."""
    algo_results = {}
    n_total = len(records)
    n_processed = 0
    n_skipped = 0

    for (subdir, mat_name), expert_data in records.items():
        mat_path = DATA_DIR / subdir / mat_name
        if not mat_path.exists():
            n_skipped += 1
            continue

        try:
            try:
                mat = scipy.io.loadmat(str(mat_path))
            except NotImplementedError:
                mat = hdf5storage.loadmat(str(mat_path))

            data = mat.get('data_50sec', mat.get('data'))
            if data is None:
                n_skipped += 1
                continue
            if data.shape[0] > data.shape[1]:
                data = data.T
            fs = 200

            # Method A (PD2a)
            rA = pd_detect_alternate(data, fs, pk_detect='apd')

            # Method B (pointiness+ACF, best params)
            rB = pd_detect_pointiness_acf(data, fs, method='pointiness',
                                           lowpass_hz=15, smoothing_sigma=0.02,
                                           acf_min_lag=0.4, acf_peak_threshold=0.20,
                                           peak_height_frac=0.3, sync_threshold=0.8)

            # Method B+pnorm (percentile normalization)
            rBpn = pd_detect_pointiness_acf(data, fs, method='pointiness',
                                             lowpass_hz=15, smoothing_sigma=0.02,
                                             acf_min_lag=0.4, acf_peak_threshold=0.20,
                                             peak_height_frac=0.3, sync_threshold=0.8,
                                             percentile_norm=True,
                                             percentile_window_s=10.0,
                                             percentile_val=90)

            algo_results[(subdir, mat_name)] = {
                'A_freq': rA.get('event_frequency', np.nan),
                'A_spatial': rA.get('spatial_extent', np.nan),
                'B_freq': rB.get('event_frequency', np.nan),
                'B_spatial': rB.get('spatial_extent', np.nan),
                'Bpn_freq': rBpn.get('event_frequency', np.nan),
                'Bpn_spatial': rBpn.get('spatial_extent', np.nan),
            }

            n_processed += 1
            if n_processed % 50 == 0:
                print(f'  Processed {n_processed} segments...')

        except Exception as e:
            n_skipped += 1

    print(f'  Processed {n_processed}, skipped {n_skipped} of {n_total} segments')
    return algo_results


def build_rating_matrices(records, algo_results, pattern_type):
    """Build rating matrices for ICC/PA computation.

    Returns dict with keys like 'freq_ee', 'freq_ea_A', etc.
    Each value is (n_segments, n_raters) array.
    """
    experts = ['LB', 'PH', 'SZ']
    subdir = pattern_type

    # Find segments with all 3 experts AND algorithm results
    valid_keys = []
    for (sd, mat_name), expert_data in records.items():
        if sd != subdir:
            continue
        if (sd, mat_name) not in algo_results:
            continue
        # Need all 3 experts with frequency > 0
        expert_freqs = []
        for e in experts:
            if e in expert_data:
                f = expert_data[e]['frequency']
                if np.isfinite(f) and f > 0:
                    expert_freqs.append(f)
        if len(expert_freqs) == 3:
            valid_keys.append((sd, mat_name))

    n = len(valid_keys)
    print(f'  {pattern_type.upper()}: {n} segments with all 3 experts and algorithm results')

    # Expert-expert matrices
    freq_ee = np.full((n, 3), np.nan)
    spatial_ee = np.full((n, 3), np.nan)

    # Expert-algorithm matrices (3 experts + 1 algorithm = 4 raters)
    freq_ea_A = np.full((n, 4), np.nan)
    freq_ea_B = np.full((n, 4), np.nan)
    freq_ea_Bpn = np.full((n, 4), np.nan)
    spatial_ea_A = np.full((n, 4), np.nan)
    spatial_ea_B = np.full((n, 4), np.nan)
    spatial_ea_Bpn = np.full((n, 4), np.nan)

    for i, key in enumerate(valid_keys):
        sd, mat_name = key
        ed = records[key]
        ar = algo_results[key]

        for j, e in enumerate(experts):
            if e in ed:
                freq_ee[i, j] = ed[e]['frequency']
                spatial_ee[i, j] = ed[e]['spatial']
                freq_ea_A[i, j] = ed[e]['frequency']
                freq_ea_B[i, j] = ed[e]['frequency']
                freq_ea_Bpn[i, j] = ed[e]['frequency']
                spatial_ea_A[i, j] = ed[e]['spatial']
                spatial_ea_B[i, j] = ed[e]['spatial']
                spatial_ea_Bpn[i, j] = ed[e]['spatial']

        freq_ea_A[i, 3] = ar['A_freq']
        freq_ea_B[i, 3] = ar['B_freq']
        freq_ea_Bpn[i, 3] = ar['Bpn_freq']
        spatial_ea_A[i, 3] = ar['A_spatial']
        spatial_ea_B[i, 3] = ar['B_spatial']
        spatial_ea_Bpn[i, 3] = ar['Bpn_spatial']

    return {
        'freq_ee': freq_ee, 'spatial_ee': spatial_ee,
        'freq_ea_A': freq_ea_A, 'freq_ea_B': freq_ea_B, 'freq_ea_Bpn': freq_ea_Bpn,
        'spatial_ea_A': spatial_ea_A, 'spatial_ea_B': spatial_ea_B, 'spatial_ea_Bpn': spatial_ea_Bpn,
        'n': n,
    }


def plot_figure6(lpd_matrices, gpd_matrices):
    """Reproduce Figure 6: ICC and PA for LPD and GPD segments.

    4 panels: (A) ICC freq, (B) PA freq, (C) ICC spatial, (D) PA spatial
    Each panel has grouped bars for Annotators, PD2a, Method B, Method B+pnorm
    with LPD and GPD side by side.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    categories = ['Annotators', 'Annotators &\nPD2a', 'Annotators &\nMethod B', 'Annotators &\nMethod B+pnorm']
    x = np.arange(len(categories))
    width = 0.35

    # Colors matching the paper
    lpd_color = '#4CAF50'  # green
    gpd_color = '#FF9800'  # orange

    # Compute all metrics
    metrics = {}
    for ptype, matrices in [('LPD', lpd_matrices), ('GPD', gpd_matrices)]:
        # ICC frequency
        icc_ee, ci_l, ci_u = compute_icc3(matrices['freq_ee'])
        icc_A, _, _ = compute_icc3(matrices['freq_ea_A'])
        icc_B, _, _ = compute_icc3(matrices['freq_ea_B'])
        icc_Bpn, _, _ = compute_icc3(matrices['freq_ea_Bpn'])
        metrics[f'{ptype}_icc_freq'] = [icc_ee, icc_A, icc_B, icc_Bpn]

        # PA frequency
        pa_ee = compute_pairwise_agreement(matrices['freq_ee'], categorize_frequency)
        pa_A = compute_pairwise_agreement(matrices['freq_ea_A'], categorize_frequency)
        pa_B = compute_pairwise_agreement(matrices['freq_ea_B'], categorize_frequency)
        pa_Bpn = compute_pairwise_agreement(matrices['freq_ea_Bpn'], categorize_frequency)
        metrics[f'{ptype}_pa_freq'] = [pa_ee, pa_A, pa_B, pa_Bpn]

        # ICC spatial
        icc_ee_s, _, _ = compute_icc3(matrices['spatial_ee'])
        icc_A_s, _, _ = compute_icc3(matrices['spatial_ea_A'])
        icc_B_s, _, _ = compute_icc3(matrices['spatial_ea_B'])
        icc_Bpn_s, _, _ = compute_icc3(matrices['spatial_ea_Bpn'])
        metrics[f'{ptype}_icc_spatial'] = [icc_ee_s, icc_A_s, icc_B_s, icc_Bpn_s]

        # PA spatial
        pa_ee_s = compute_pairwise_agreement(matrices['spatial_ee'], categorize_spatial)
        pa_A_s = compute_pairwise_agreement(matrices['spatial_ea_A'], categorize_spatial)
        pa_B_s = compute_pairwise_agreement(matrices['spatial_ea_B'], categorize_spatial)
        pa_Bpn_s = compute_pairwise_agreement(matrices['spatial_ea_Bpn'], categorize_spatial)
        metrics[f'{ptype}_pa_spatial'] = [pa_ee_s, pa_A_s, pa_B_s, pa_Bpn_s]

    # Panel A: ICC for frequency
    ax = axes[0, 0]
    ax.set_title('A. Frequency of Event', fontsize=13, fontweight='bold')
    vals_lpd = [v * 100 for v in metrics['LPD_icc_freq']]
    vals_gpd = [v * 100 for v in metrics['GPD_icc_freq']]
    bars1 = ax.bar(x - width/2, vals_lpd, width, label='LPD', color=lpd_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_gpd, width, label='GPD', color=gpd_color, alpha=0.85)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{int(round(h))}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('ICC [%]', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel B: PA for frequency
    ax = axes[0, 1]
    ax.set_title('B. Frequency of Event', fontsize=13, fontweight='bold')
    vals_lpd = metrics['LPD_pa_freq']
    vals_gpd = metrics['GPD_pa_freq']
    bars1 = ax.bar(x - width/2, vals_lpd, width, label='LPD', color=lpd_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_gpd, width, label='GPD', color=gpd_color, alpha=0.85)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{int(round(h))}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('PA [%]', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel C: ICC for spatial extent
    ax = axes[1, 0]
    ax.set_title('C. Spatial Extent', fontsize=13, fontweight='bold')
    vals_lpd = [v * 100 for v in metrics['LPD_icc_spatial']]
    vals_gpd = [v * 100 for v in metrics['GPD_icc_spatial']]
    bars1 = ax.bar(x - width/2, vals_lpd, width, label='LPD', color=lpd_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_gpd, width, label='GPD', color=gpd_color, alpha=0.85)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{int(round(h))}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('ICC [%]', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Panel D: PA for spatial extent
    ax = axes[1, 1]
    ax.set_title('D. Spatial Extent', fontsize=13, fontweight='bold')
    vals_lpd = metrics['LPD_pa_spatial']
    vals_gpd = metrics['GPD_pa_spatial']
    bars1 = ax.bar(x - width/2, vals_lpd, width, label='LPD', color=lpd_color, alpha=0.85)
    bars2 = ax.bar(x + width/2, vals_gpd, width, label='GPD', color=gpd_color, alpha=0.85)
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{int(round(h))}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('PA [%]', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9, rotation=15, ha='right')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('ICC and PA: Expert-Expert vs Expert-Algorithm Agreement\n'
                 '(LPD and GPD segments, reproducing Figure 6 from Tăuțan et al. 2025)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig, metrics


def plot_mae_table(lpd_matrices, gpd_matrices, records, algo_results):
    """Compute and plot MAE table (reproducing Table 1 from paper)."""
    results = []

    for ptype, subdir in [('LPD', 'lpd'), ('GPD', 'gpd')]:
        expert_freqs = []
        expert_spatials = []
        algo_freqs = {'A': [], 'B': [], 'Bpn': []}
        algo_spatials = {'A': [], 'B': [], 'Bpn': []}

        for (sd, mat_name), expert_data in records.items():
            if sd != subdir:
                continue
            if (sd, mat_name) not in algo_results:
                continue

            # Consensus expert value
            freqs = [expert_data[e]['frequency'] for e in ['LB', 'PH', 'SZ']
                     if e in expert_data and np.isfinite(expert_data[e]['frequency'])
                     and expert_data[e]['frequency'] > 0]
            spatials = [expert_data[e]['spatial'] for e in ['LB', 'PH', 'SZ']
                        if e in expert_data and np.isfinite(expert_data[e].get('spatial', np.nan))
                        and expert_data[e]['spatial'] > 0]

            if not freqs:
                continue

            consensus_freq = np.median(freqs)
            consensus_spatial = np.median(spatials) if spatials else np.nan
            ar = algo_results[(sd, mat_name)]

            for method, key in [('A', 'A'), ('B', 'B'), ('Bpn', 'Bpn')]:
                f = ar[f'{key}_freq']
                s = ar[f'{key}_spatial']
                if np.isfinite(f):
                    algo_freqs[method].append((consensus_freq, f))
                if np.isfinite(s) and np.isfinite(consensus_spatial):
                    algo_spatials[method].append((consensus_spatial, s))

        for method in ['A', 'B', 'Bpn']:
            if algo_freqs[method]:
                pairs = algo_freqs[method]
                mae_f = np.mean([abs(a - b) for a, b in pairs])
                ci_f = 1.96 * np.std([abs(a - b) for a, b in pairs]) / np.sqrt(len(pairs))
            else:
                mae_f, ci_f = np.nan, np.nan
            if algo_spatials[method]:
                pairs = algo_spatials[method]
                mae_s = np.mean([abs(a - b) for a, b in pairs])
                ci_s = 1.96 * np.std([abs(a - b) for a, b in pairs]) / np.sqrt(len(pairs))
            else:
                mae_s, ci_s = np.nan, np.nan

            results.append({
                'Pattern': ptype, 'Method': method,
                'MAE_freq': mae_f, 'CI_freq': ci_f,
                'MAE_spatial': mae_s, 'CI_spatial': ci_s,
                'N_freq': len(algo_freqs[method]),
                'N_spatial': len(algo_spatials[method]),
            })

    return pd.DataFrame(results)


def main():
    print('Loading expert annotations...')
    records = load_per_expert_annotations()
    print(f'  Loaded {len(records)} segments')

    print('\nRunning algorithms on annotated dataset...')
    algo_results = run_algorithms_on_dataset(records)

    print('\nBuilding rating matrices...')
    lpd_matrices = build_rating_matrices(records, algo_results, 'lpd')
    gpd_matrices = build_rating_matrices(records, algo_results, 'gpd')

    # Generate Figure 6 reproduction
    print('\nGenerating Figure 6 (ICC and PA)...')
    fig6, metrics = plot_figure6(lpd_matrices, gpd_matrices)
    fig6.savefig(str(RESULTS_DIR / 'figure6_icc_pa_pd.png'), dpi=150, bbox_inches='tight')
    plt.close(fig6)
    print(f'  Saved: {RESULTS_DIR / "figure6_icc_pa_pd.png"}')

    # Print metrics table
    print('\n' + '=' * 80)
    print('ICC AND PA RESULTS')
    print('=' * 80)
    for ptype in ['LPD', 'GPD']:
        print(f'\n--- {ptype} ---')
        print(f'  {"Metric":<20s} {"Annotators":>12s} {"PD2a (A)":>12s} {"Method B":>12s} {"B+pnorm":>12s}')
        print('  ' + '-' * 68)
        for metric_name, key in [('ICC Freq', 'icc_freq'), ('PA Freq', 'pa_freq'),
                                  ('ICC Spatial', 'icc_spatial'), ('PA Spatial', 'pa_spatial')]:
            vals = metrics[f'{ptype}_{key}']
            if 'icc' in key:
                vals = [v * 100 for v in vals]
            print(f'  {metric_name:<20s} {vals[0]:>11.1f}% {vals[1]:>11.1f}% {vals[2]:>11.1f}% {vals[3]:>11.1f}%')

    # MAE table
    print('\nComputing MAE (reproducing Table 1)...')
    mae_df = plot_mae_table(lpd_matrices, gpd_matrices, records, algo_results)
    print('\n' + '=' * 80)
    print('MAE TABLE')
    print('=' * 80)
    print(f'  {"Pattern":<6s} {"Method":<8s} {"MAE Freq":>10s} {"95% CI":>12s} {"N":>5s}'
          f'  {"MAE Spatial":>12s} {"95% CI":>12s} {"N":>5s}')
    print('  ' + '-' * 75)
    for _, row in mae_df.iterrows():
        method_names = {'A': 'PD2a', 'B': 'Method B', 'Bpn': 'B+pnorm'}
        freq_ci = f'[{row["MAE_freq"]-row["CI_freq"]:.2f},{row["MAE_freq"]+row["CI_freq"]:.2f}]'
        spat_ci = f'[{row["MAE_spatial"]-row["CI_spatial"]:.2f},{row["MAE_spatial"]+row["CI_spatial"]:.2f}]'
        print(f'  {row["Pattern"]:<6s} {method_names[row["Method"]]:<8s} '
              f'{row["MAE_freq"]:>10.2f} {freq_ci:>12s} {row["N_freq"]:>5d}'
              f'  {row["MAE_spatial"]:>12.2f} {spat_ci:>12s} {row["N_spatial"]:>5d}')
    mae_df.to_csv(str(RESULTS_DIR / 'mae_table.csv'), index=False)

    # Generate scatter plots (frequency)
    print('\nGenerating scatter plots...')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    method_labels = ['PD2a (Method A)', 'Method B\n(pointiness+ACF)', 'Method B+pnorm\n(percentile norm)']
    freq_keys = ['A_freq', 'B_freq', 'Bpn_freq']

    for ptype_i, (ptype, subdir) in enumerate([('LPD', 'lpd'), ('GPD', 'gpd')]):
        for mi, (method_label, freq_key) in enumerate(zip(method_labels, freq_keys)):
            ax = axes[ptype_i, mi]
            expert_vals = []
            algo_vals = []
            for (sd, mat_name), expert_data in records.items():
                if sd != subdir or (sd, mat_name) not in algo_results:
                    continue
                freqs = [expert_data[e]['frequency'] for e in ['LB', 'PH', 'SZ']
                         if e in expert_data and np.isfinite(expert_data[e]['frequency'])
                         and expert_data[e]['frequency'] > 0]
                if not freqs:
                    continue
                consensus = np.median(freqs)
                algo_f = algo_results[(sd, mat_name)][freq_key]
                if np.isfinite(algo_f):
                    expert_vals.append(consensus)
                    algo_vals.append(algo_f)

            if expert_vals:
                ax.scatter(expert_vals, algo_vals, alpha=0.4, s=20,
                          color='#4CAF50' if ptype == 'LPD' else '#FF9800')
                # Identity line
                lim = max(max(expert_vals), max(algo_vals)) * 1.1
                ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5)
                mae = np.mean(np.abs(np.array(expert_vals) - np.array(algo_vals)))
                from scipy.stats import pearsonr
                r, _ = pearsonr(expert_vals, algo_vals)
                ax.text(0.05, 0.95, f'MAE={mae:.2f}\nr={r:.2f}\nn={len(expert_vals)}',
                       transform=ax.transAxes, fontsize=9, va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_xlabel('Expert consensus freq (Hz)', fontsize=9)
            ax.set_ylabel('Algorithm freq (Hz)', fontsize=9)
            if ptype_i == 0:
                ax.set_title(method_label, fontsize=11, fontweight='bold')
            ax.set_xlim(0, 4)
            ax.set_ylim(0, 4)

            # Add pattern label on right
            if mi == 2:
                ax.text(1.05, 0.5, ptype, transform=ax.transAxes, fontsize=14,
                       fontweight='bold', va='center', rotation=-90)

    fig.suptitle('Frequency: Expert Consensus vs Algorithm\n(Scatter plots by pattern type)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.97, 0.93])
    fig.savefig(str(RESULTS_DIR / 'frequency_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {RESULTS_DIR / "frequency_scatter.png"}')

    # Bland-Altman plots
    print('Generating Bland-Altman plots...')
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for ptype_i, (ptype, subdir) in enumerate([('LPD', 'lpd'), ('GPD', 'gpd')]):
        for mi, (method_label, freq_key) in enumerate(zip(method_labels, freq_keys)):
            ax = axes[ptype_i, mi]
            expert_vals = []
            algo_vals = []
            for (sd, mat_name), expert_data in records.items():
                if sd != subdir or (sd, mat_name) not in algo_results:
                    continue
                freqs = [expert_data[e]['frequency'] for e in ['LB', 'PH', 'SZ']
                         if e in expert_data and np.isfinite(expert_data[e]['frequency'])
                         and expert_data[e]['frequency'] > 0]
                if not freqs:
                    continue
                consensus = np.median(freqs)
                algo_f = algo_results[(sd, mat_name)][freq_key]
                if np.isfinite(algo_f):
                    expert_vals.append(consensus)
                    algo_vals.append(algo_f)

            if expert_vals:
                means = (np.array(expert_vals) + np.array(algo_vals)) / 2
                diffs = np.array(algo_vals) - np.array(expert_vals)
                ax.scatter(means, diffs, alpha=0.4, s=20,
                          color='#4CAF50' if ptype == 'LPD' else '#FF9800')
                bias = np.mean(diffs)
                sd = np.std(diffs)
                ax.axhline(bias, color='red', linewidth=1, label=f'Bias={bias:.2f}')
                ax.axhline(bias + 1.96*sd, color='gray', linewidth=0.8, linestyle='--')
                ax.axhline(bias - 1.96*sd, color='gray', linewidth=0.8, linestyle='--')
                ax.axhline(0, color='black', linewidth=0.5)

            ax.set_xlabel('Mean of expert & algorithm (Hz)', fontsize=9)
            ax.set_ylabel('Algorithm - Expert (Hz)', fontsize=9)
            if ptype_i == 0:
                ax.set_title(method_label, fontsize=11, fontweight='bold')
            if mi == 2:
                ax.text(1.05, 0.5, ptype, transform=ax.transAxes, fontsize=14,
                       fontweight='bold', va='center', rotation=-90)

    fig.suptitle('Bland-Altman: Frequency Agreement\n(Algorithm vs Expert Consensus)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 0.97, 0.93])
    fig.savefig(str(RESULTS_DIR / 'bland_altman.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {RESULTS_DIR / "bland_altman.png"}')

    print('\nDone!')


if __name__ == '__main__':
    main()
