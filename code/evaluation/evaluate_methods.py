"""
Compare Method A (original pd_detect_alternate) vs Method B (pointiness+ACF)
across L/G classification, frequency accuracy, and spatial extent.
"""

import sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wilcoxon, pearsonr, spearmanr
import scipy.io
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf
import hdf5storage

REPO_ROOT = CODE_DIR.parent
TEST_DIR = REPO_ROOT / 'test_case_images' / 'raw_eeg'
DATA_DIR = REPO_ROOT / 'data' / '_archive' / 'dataset_eeg'
ANN_DIR = REPO_ROOT / 'data' / '_archive' / 'pd_expert_raw'
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Parameter grids for Method B
LOWPASS_HZ_GRID = [10, 15, 20, 25, 30]
ACF_MIN_LAG_GRID = [0.25, 0.4]
ACF_PEAK_THR_GRID = [0.1, 0.15, 0.2, 0.25, 0.3]
SIGMA_GRID = [0.01, 0.02, 0.03]
SYNC_THR_GRID = [0.4, 0.6, 0.8]
PEAK_HEIGHT_FRAC_GRID = [0.2, 0.3, 0.4]


def compute_metrics(gt, pred):
    gt, pred = np.array(gt), np.array(pred)
    n = len(gt)
    tp = np.sum((gt == 'G') & (pred == 'G'))
    tn = np.sum((gt == 'L') & (pred == 'L'))
    fp = np.sum((gt == 'L') & (pred == 'G'))
    fn = np.sum((gt == 'G') & (pred == 'L'))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (sens + spec) / 2
    acc = (tp + tn) / n if n > 0 else 0
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n) if n > 0 else 0
    kappa = (acc - pe) / (1 - pe) if (1 - pe) > 0 else 0
    return {'bal_acc': bal_acc, 'kappa': kappa, 'sens': sens, 'spec': spec,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}


def mcnemar_test(gt, old, new):
    gt, old, new = np.array(gt), np.array(old), np.array(new)
    b = np.sum((gt == old) & (gt != new))
    c = np.sum((gt != old) & (gt == new))
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'p': 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {'b': b, 'c': c, 'p': p}


def run_lg_evaluation():
    print('=' * 70)
    print('SECTION 1: L/G CLASSIFICATION (50 PD test cases)')
    print('=' * 70)

    # Load PD test cases
    mat_files = sorted(TEST_DIR.glob('*.mat'))
    pd_cases = []
    for f in mat_files:
        ptype = f.stem.split('_', 2)[0]
        if ptype not in ('lpd', 'gpd'):
            continue
        mat = scipy.io.loadmat(str(f))
        data = mat['data']
        if data.shape[0] > data.shape[1]:
            data = data.T
        fs = int(mat['Fs'].ravel()[0])
        pd_cases.append({
            'file': f, 'name': f.stem, 'ptype': ptype,
            'gt_lg': 'G' if ptype == 'gpd' else 'L',
            'data': data, 'fs': fs,
        })
    print(f'Loaded {len(pd_cases)} PD test cases')

    # Method A
    print('\nRunning Method A (original)...')
    for c in pd_cases:
        r = pd_detect_alternate(c['data'], c['fs'], pk_detect='apd')
        te = str(r.get('type_event', '')).upper()
        c['A_lg'] = 'G' if te == 'GPD' else 'L'

    gt = [c['gt_lg'] for c in pd_cases]
    a_pred = [c['A_lg'] for c in pd_cases]
    a_metrics = compute_metrics(gt, a_pred)
    lpd_a = sum(1 for c in pd_cases if c['ptype'] == 'lpd' and c['A_lg'] == 'L')
    gpd_a = sum(1 for c in pd_cases if c['ptype'] == 'gpd' and c['A_lg'] == 'G')
    print(f'  Method A: LPD={lpd_a}/25, GPD={gpd_a}/25, '
          f'bal_acc={a_metrics["bal_acc"]:.3f}, kappa={a_metrics["kappa"]:.3f}')

    # Grid search Methods B (pointiness) and B' (d2)
    print('\nGrid searching Methods B and B\'...')
    best_results = {}

    det_param_combos = list(itertools.product(
        LOWPASS_HZ_GRID, SIGMA_GRID, ACF_MIN_LAG_GRID,
        ACF_PEAK_THR_GRID, PEAK_HEIGHT_FRAC_GRID))
    n_combos = len(det_param_combos)

    for method_name, method_key in [('B_pointiness', 'pointiness'), ('B_d2', 'd2')]:
        print(f'\n  {method_name}: {n_combos} detector combos × {len(pd_cases)} cases')

        # Phase 1: Run detector for each param combo
        detector_cache = {}
        for ci, (lp_hz, sigma, min_lag, peak_thr, ph_frac) in enumerate(det_param_combos):
            key = (lp_hz, sigma, min_lag, peak_thr, ph_frac)
            results = []
            for c in pd_cases:
                r = pd_detect_pointiness_acf(
                    c['data'], c['fs'], method=method_key,
                    lowpass_hz=lp_hz, smoothing_sigma=sigma,
                    acf_min_lag=min_lag, acf_peak_threshold=peak_thr,
                    peak_height_frac=ph_frac,
                    sync_tolerance_ms=200, sync_threshold=0.8, sync_min_peaks=5,
                )
                results.append(r)
            detector_cache[key] = results
            if (ci + 1) % 100 == 0:
                print(f'    detector runs: {ci+1}/{n_combos}')
        print(f'    detector runs: {n_combos}/{n_combos}')

        # Phase 2: Sweep sync_threshold over cached results
        all_grid_results = []
        for key, det_results in detector_cache.items():
            lp_hz, sigma, min_lag, peak_thr, ph_frac = key
            for sync_thr in SYNC_THR_GRID:
                preds = []
                for r in det_results:
                    sr = r.get('synchrony_ratio', np.nan)
                    se = r.get('spatial_extent', 0)
                    if np.isfinite(sr):
                        lg = 'G' if sr > sync_thr else 'L'
                    else:
                        lg = 'G' if se > 0.8 else 'L'
                    preds.append(lg)
                m = compute_metrics(gt, preds)
                all_grid_results.append({
                    'method': method_name, 'lowpass_hz': lp_hz,
                    'sigma': sigma, 'min_lag': min_lag,
                    'peak_thr': peak_thr, 'ph_frac': ph_frac,
                    'sync_thr': sync_thr, **m,
                })

        grid_df = pd.DataFrame(all_grid_results).sort_values('bal_acc', ascending=False)
        best = grid_df.iloc[0]
        best_results[method_name] = best
        print(f'\n    Best: lp={best["lowpass_hz"]}Hz, sigma={best["sigma"]}, '
              f'min_lag={best["min_lag"]}, peak_thr={best["peak_thr"]}, '
              f'ph_frac={best["ph_frac"]}, sync_thr={best["sync_thr"]}')
        print(f'    bal_acc={best["bal_acc"]:.3f}, kappa={best["kappa"]:.3f}, '
              f'G_recall={best["sens"]:.3f}, L_recall={best["spec"]:.3f}')

        # Store best predictions
        best_key = (best['lowpass_hz'], best['sigma'], best['min_lag'],
                    best['peak_thr'], best['ph_frac'])
        best_det = detector_cache[best_key]
        best_preds = []
        for r in best_det:
            sr = r.get('synchrony_ratio', np.nan)
            se = r.get('spatial_extent', 0)
            if np.isfinite(sr):
                lg = 'G' if sr > best['sync_thr'] else 'L'
            else:
                lg = 'G' if se > 0.8 else 'L'
            best_preds.append(lg)
        best_results[method_name + '_preds'] = best_preds
        best_results[method_name + '_det'] = best_det

        grid_df.to_csv(str(RESULTS_DIR / f'{method_name}_grid_search.csv'), index=False)

        # Show top 10 for insight
        print('\n    Top 10 parameter combos:')
        for _, row in grid_df.head(10).iterrows():
            print(f'      lp={row["lowpass_hz"]:.0f} sig={row["sigma"]:.2f} '
                  f'lag={row["min_lag"]:.2f} pthr={row["peak_thr"]:.2f} '
                  f'phf={row["ph_frac"]:.1f} sthr={row["sync_thr"]:.1f} '
                  f'→ bal={row["bal_acc"]:.3f} G={row["sens"]:.2f} L={row["spec"]:.2f}')

    # Summary
    print('\n' + '=' * 70)
    print('L/G CLASSIFICATION SUMMARY')
    print('=' * 70)
    header = f'{"Method":<30s} {"LPD":>5s} {"GPD":>5s} {"Bal.Acc":>8s} {"Kappa":>7s}'
    print(header)
    print('-' * len(header))
    print(f'{"Method A (original)":<30s} {lpd_a:>3d}/25 {gpd_a:>3d}/25 '
          f'{a_metrics["bal_acc"]:>8.3f} {a_metrics["kappa"]:>7.3f}')

    for mn in ['B_pointiness', 'B_d2']:
        b = best_results[mn]
        preds = best_results[mn + '_preds']
        lpd_b = sum(1 for c, p in zip(pd_cases, preds) if c['ptype'] == 'lpd' and p == 'L')
        gpd_b = sum(1 for c, p in zip(pd_cases, preds) if c['ptype'] == 'gpd' and p == 'G')
        print(f'{mn:<30s} {lpd_b:>3d}/25 {gpd_b:>3d}/25 '
              f'{b["bal_acc"]:>8.3f} {b["kappa"]:>7.3f}')
        mc = mcnemar_test(gt, a_pred, preds)
        sig = '*' if mc['p'] < 0.05 else ''
        print(f'  vs A: McNemar p={mc["p"]:.4f}{sig} (b={mc["b"]}, c={mc["c"]})')

    return pd_cases, best_results


def load_expert_annotations():
    records = {}
    for pattern in ['LPDS', 'GPDS']:
        subdir = 'lpd' if 'LPD' in pattern else 'gpd'
        for expert_file in sorted(ANN_DIR.glob(f'{pattern}_*')):
            expert = expert_file.stem.split('_')[1]
            df = pd.read_csv(expert_file)
            for _, row in df.iterrows():
                mat_name = Path(row['files']).stem.replace('_score', '') + '.mat'
                if mat_name not in records:
                    records[mat_name] = {'subdir': subdir, 'experts': {}}
                records[mat_name]['experts'][expert] = {
                    'frequency': row['frequency'],
                    'spatial': row['spatial'],
                }
    for mat_name, rec in records.items():
        freqs = [v['frequency'] for v in rec['experts'].values() if v['frequency'] > 0]
        spatials = [v['spatial'] for v in rec['experts'].values()
                    if isinstance(v['spatial'], (int, float)) and v['spatial'] > 0]
        rec['consensus_freq'] = float(np.median(freqs)) if freqs else np.nan
        rec['consensus_spatial'] = float(np.median(spatials)) if spatials else np.nan
    n_with_freq = sum(1 for r in records.values() if np.isfinite(r['consensus_freq']))
    print(f'Loaded annotations for {len(records)} segments, {n_with_freq} with freq>0')
    return records


def run_frequency_evaluation(best_results):
    print('\n' + '=' * 70)
    print('SECTION 2: FREQUENCY ACCURACY (annotated dataset)')
    print('=' * 70)

    annotations = load_expert_annotations()

    # Get best params from grid search
    best_b = best_results.get('B_pointiness', {})
    best_b2 = best_results.get('B_d2', {})

    results = []
    n_processed = 0

    for mat_name, ann in annotations.items():
        if not np.isfinite(ann['consensus_freq']):
            continue
        mat_path = DATA_DIR / ann['subdir'] / mat_name
        if not mat_path.exists():
            continue
        try:
            try:
                mat = scipy.io.loadmat(str(mat_path))
            except NotImplementedError:
                mat = hdf5storage.loadmat(str(mat_path))
            data = mat.get('data_50sec', mat.get('data'))
            if data.shape[0] > data.shape[1]:
                data = data.T
            fs = 200

            rA = pd_detect_alternate(data, fs, pk_detect='apd')

            rB = pd_detect_pointiness_acf(
                data, fs, method='pointiness',
                lowpass_hz=best_b.get('lowpass_hz', 20),
                smoothing_sigma=best_b.get('sigma', 0.02),
                acf_min_lag=best_b.get('min_lag', 0.25),
                acf_peak_threshold=best_b.get('peak_thr', 0.1),
                peak_height_frac=best_b.get('ph_frac', 0.3))

            rB2 = pd_detect_pointiness_acf(
                data, fs, method='d2',
                lowpass_hz=best_b2.get('lowpass_hz', 20),
                smoothing_sigma=best_b2.get('sigma', 0.02),
                acf_min_lag=best_b2.get('min_lag', 0.25),
                acf_peak_threshold=best_b2.get('peak_thr', 0.1),
                peak_height_frac=best_b2.get('ph_frac', 0.3))

            results.append({
                'file': mat_name, 'subdir': ann['subdir'],
                'expert_freq': ann['consensus_freq'],
                'expert_spatial': ann['consensus_spatial'],
                'A_freq': rA.get('event_frequency', np.nan),
                'A_spatial': rA.get('spatial_extent', np.nan),
                'B_freq': rB.get('event_frequency', np.nan),
                'B_spatial': rB.get('spatial_extent', np.nan),
                'B2_freq': rB2.get('event_frequency', np.nan),
                'B2_spatial': rB2.get('spatial_extent', np.nan),
                'A_n_ch': sum(1 for v in rA.get('channel_pd_scores', {}).values()
                              if isinstance(v, float) and not np.isnan(v) and v > 0),
                'B_n_ch': len(rB.get('channels', [])),
                'B2_n_ch': len(rB2.get('channels', [])),
            })
            n_processed += 1
            if n_processed % 50 == 0:
                print(f'  Processed {n_processed} segments...')
        except Exception as e:
            pass

    print(f'\nProcessed {n_processed} segments')
    df = pd.DataFrame(results)
    df.to_csv(str(RESULTS_DIR / 'method_comparison_results.csv'), index=False)

    print('\n--- FREQUENCY ACCURACY ---')
    header = f'{"Method":<25s} {"N":>4s} {"MAE":>7s} {"Pearson":>8s} {"Spearman":>8s}'
    print(header)
    print('-' * len(header))
    for label, col in [('A (original)', 'A_freq'),
                        ('B (pointiness+ACF)', 'B_freq'),
                        ("B' (d2+ACF)", 'B2_freq')]:
        valid = df[np.isfinite(df[col]) & np.isfinite(df['expert_freq'])]
        if len(valid) < 3:
            print(f'{label:<25s} {len(valid):>4d}  insufficient data')
            continue
        mae = np.mean(np.abs(valid[col] - valid['expert_freq']))
        r_p, _ = pearsonr(valid[col], valid['expert_freq'])
        r_s, _ = spearmanr(valid[col], valid['expert_freq'])
        print(f'{label:<25s} {len(valid):>4d} {mae:>7.3f} {r_p:>8.3f} {r_s:>8.3f}')

    print('\n--- SPATIAL EXTENT ---')
    for label, col in [('A', 'A_spatial'), ('B', 'B_spatial'), ("B'", 'B2_spatial')]:
        valid = df[np.isfinite(df[col]) & np.isfinite(df['expert_spatial'])]
        if len(valid) < 3:
            continue
        mae = np.mean(np.abs(valid[col] - valid['expert_spatial']))
        r_p, _ = pearsonr(valid[col], valid['expert_spatial'])
        print(f'  {label}: MAE={mae:.3f}, r={r_p:.3f} (n={len(valid)})')

    print('\n--- CHANNELS DETECTED ---')
    for label, col in [('A', 'A_n_ch'), ('B', 'B_n_ch'), ("B'", 'B2_n_ch')]:
        print(f'  {label}: {df[col].mean():.1f} ± {df[col].std():.1f}')

    return df


def generate_figures(freq_df, best_results):
    print('\n' + '=' * 70)
    print('GENERATING FIGURES')
    print('=' * 70)

    # Frequency scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, col) in zip(axes, [('A (original)', 'A_freq'),
                                        ('B (pointiness+ACF)', 'B_freq'),
                                        ("B' (d2+ACF)", 'B2_freq')]):
        valid = freq_df[np.isfinite(freq_df[col]) & np.isfinite(freq_df['expert_freq'])]
        colors = ['#cc3333' if s == 'lpd' else '#3333cc' for s in valid['subdir']]
        ax.scatter(valid['expert_freq'], valid[col], c=colors, alpha=0.4, s=15)
        lims = [0, max(4, valid['expert_freq'].max(), valid[col].max()) * 1.05]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Expert frequency (Hz)'); ax.set_ylabel('Detector frequency (Hz)')
        mae = np.mean(np.abs(valid[col] - valid['expert_freq']))
        r, _ = pearsonr(valid[col], valid['expert_freq']) if len(valid) > 2 else (0, 0)
        ax.set_title(f'{label}\nMAE={mae:.3f}, r={r:.3f}')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(RESULTS_DIR / 'frequency_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved frequency_scatter.png')

    # Bland-Altman
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (label, col) in zip(axes, [('A', 'A_freq'), ('B', 'B_freq'), ("B'", 'B2_freq')]):
        valid = freq_df[np.isfinite(freq_df[col]) & np.isfinite(freq_df['expert_freq'])]
        mean_vals = (valid[col] + valid['expert_freq']) / 2
        diff_vals = valid[col] - valid['expert_freq']
        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=10)
        md = np.mean(diff_vals); sd = np.std(diff_vals)
        ax.axhline(md, color='red', linewidth=1)
        ax.axhline(md + 1.96*sd, color='gray', linewidth=0.8, linestyle='--')
        ax.axhline(md - 1.96*sd, color='gray', linewidth=0.8, linestyle='--')
        ax.set_xlabel('Mean freq (Hz)'); ax.set_ylabel('Detector - Expert (Hz)')
        ax.set_title(f'{label}: bias={md:.3f}, LoA=[{md-1.96*sd:.2f}, {md+1.96*sd:.2f}]')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(RESULTS_DIR / 'bland_altman.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved bland_altman.png')

    # Parameter heatmaps
    for mn in ['B_pointiness', 'B_d2']:
        grid_file = RESULTS_DIR / f'{mn}_grid_search.csv'
        if not grid_file.exists():
            continue
        gdf = pd.read_csv(grid_file)
        best = best_results[mn]
        # Heatmap: lowpass_hz vs acf_peak_threshold at best other params
        sub = gdf[(gdf['sigma'] == best['sigma']) &
                  (gdf['min_lag'] == best['min_lag']) &
                  (gdf['ph_frac'] == best['ph_frac']) &
                  (gdf['sync_thr'] == best['sync_thr'])]
        if len(sub) == 0:
            continue
        pivot = sub.pivot_table(values='bal_acc', index='peak_thr',
                                 columns='lowpass_hz', aggfunc='first')
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.0f}' for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.2f}' for v in pivot.index])
        ax.set_xlabel('Lowpass Hz')
        ax.set_ylabel('ACF peak threshold')
        ax.set_title(f'{mn} Balanced Accuracy')
        plt.colorbar(im, ax=ax)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7)
        fig.savefig(str(RESULTS_DIR / f'{mn}_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved {mn}_heatmap.png')

    print('  Done with figures')


def main():
    pd_cases, best_results = run_lg_evaluation()
    freq_df = run_frequency_evaluation(best_results)
    generate_figures(freq_df, best_results)
    print('\nDone! All results saved to results/')


if __name__ == '__main__':
    main()
