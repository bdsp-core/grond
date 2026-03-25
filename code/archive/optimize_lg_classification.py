"""
Optimize L/G classification using peak synchrony (PD) and bilateral scores (RDA).

Compares the current spatial_extent > 0.80 method against:
  - PD: synchrony-based (are left/right peaks time-locked?)
  - RDA: bilateral score (does the weaker hemisphere exceed a threshold?)

Evaluates on 100 expert-labeled test cases (25 each: LPD, GPD, LRDA, GRDA).
"""

import sys, os, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mne.filter import notch_filter, filter_data
import warnings
import scipy.io

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from pd_detect_alternate import pd_detect_alternate
from rda1b_fft import rda1b_fft
from browse_results import get_bipolar, detect_pd_peaks

TEST_DIR = CODE_DIR.parent / 'test_case_images'
RESULTS_DIR = CODE_DIR.parent / 'results'

# Channel indices
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]

# Parameter grids
TOLERANCE_MS_GRID = [25, 50, 75, 100, 125, 150, 175, 200]
SYNC_THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_PEAKS_GRID = [2, 3, 4, 5]
BILATERAL_THRESHOLD_GRID = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]


def load_test_cases():
    """Load all 100 test cases, parse ground truth from filenames."""
    mat_files = sorted(TEST_DIR.glob('*.mat'))
    cases = []
    for f in mat_files:
        name = f.stem
        parts = name.split('_', 2)
        ptype = parts[0]  # lpd, gpd, lrda, grda
        gt_lg = 'G' if ptype in ('gpd', 'grda') else 'L'
        modality = 'PD' if ptype in ('lpd', 'gpd') else 'RDA'
        cases.append({
            'file': f,
            'name': name,
            'pattern_type': ptype,
            'gt_lg': gt_lg,
            'modality': modality,
        })
    print(f'Loaded {len(cases)} test cases: '
          f'{sum(1 for c in cases if c["pattern_type"]=="lpd")} LPD, '
          f'{sum(1 for c in cases if c["pattern_type"]=="gpd")} GPD, '
          f'{sum(1 for c in cases if c["pattern_type"]=="lrda")} LRDA, '
          f'{sum(1 for c in cases if c["pattern_type"]=="grda")} GRDA')
    return cases


def run_detectors(cases):
    """Run appropriate detector on each case, cache results."""
    print('\nRunning detectors...')
    for i, case in enumerate(cases):
        mat = scipy.io.loadmat(str(case['file']))
        data = mat['data']
        fs = int(mat['Fs'].ravel()[0])
        if data.shape[0] > data.shape[1]:
            data = data.T

        if case['modality'] == 'PD':
            result = pd_detect_alternate(data, fs, pk_detect='apd')
            case['detector_result'] = result
            case['current_lg'] = 'G' if str(result.get('type_event', '')).upper() == 'GPD' else 'L'
        else:
            result, _, _ = rda1b_fft(data, fs, channel_filter=0)
            case['detector_result'] = result
            case['current_lg'] = 'G' if str(result.get('type_event', '')).upper() == 'GRDA' else 'L'

        # Also get bipolar for peak detection
        filtered = notch_filter(data.astype(float), fs, 60, n_jobs=1, verbose="ERROR")
        filtered = filter_data(filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        case['seg_bi'] = get_bipolar(filtered)
        case['fs'] = fs

        print(f'  [{i+1}/{len(cases)}] {case["name"]}: gt={case["gt_lg"]}, '
              f'current={case["current_lg"]}, type={result.get("type_event","N/A")}')

    return cases


def extract_peaks(cases):
    """Extract per-hemisphere peak indices for PD cases."""
    print('\nExtracting peaks for PD cases...')
    for case in cases:
        if case['modality'] != 'PD':
            continue

        seg_bi = case['seg_bi']
        fs = case['fs']

        left_peaks = []
        right_peaks = []
        for ch_idx in range(18):
            pks = detect_pd_peaks(seg_bi[ch_idx, :], fs)
            if len(pks) > 0:
                if ch_idx in LEFT_INDICES:
                    left_peaks.append(pks)
                elif ch_idx in RIGHT_INDICES:
                    right_peaks.append(pks)

        # Flatten to sorted arrays
        case['left_peaks'] = np.sort(np.concatenate(left_peaks)) if left_peaks else np.array([])
        case['right_peaks'] = np.sort(np.concatenate(right_peaks)) if right_peaks else np.array([])
        case['n_left'] = len(case['left_peaks'])
        case['n_right'] = len(case['right_peaks'])

    return cases


def compute_synchrony(left_peaks, right_peaks, fs, tolerance_ms):
    """Compute synchrony ratio between left and right peak arrays."""
    if len(left_peaks) == 0 or len(right_peaks) == 0:
        return 0.0

    tol_samples = int(tolerance_ms * fs / 1000)

    # For each left peak, check if any right peak is within tolerance
    matched_left = 0
    for lp in left_peaks:
        idx = np.searchsorted(right_peaks, lp)
        for candidate_idx in [idx - 1, idx]:
            if 0 <= candidate_idx < len(right_peaks):
                if abs(right_peaks[candidate_idx] - lp) <= tol_samples:
                    matched_left += 1
                    break

    # For each right peak, check if any left peak is within tolerance
    matched_right = 0
    for rp in right_peaks:
        idx = np.searchsorted(left_peaks, rp)
        for candidate_idx in [idx - 1, idx]:
            if 0 <= candidate_idx < len(left_peaks):
                if abs(left_peaks[candidate_idx] - rp) <= tol_samples:
                    matched_right += 1
                    break

    frac_left = matched_left / len(left_peaks)
    frac_right = matched_right / len(right_peaks)
    return (frac_left + frac_right) / 2.0


def classify_pd_synchrony(case, tolerance_ms, sync_threshold, min_peaks):
    """Classify a PD case as L or G using synchrony."""
    if case['n_left'] < min_peaks or case['n_right'] < min_peaks:
        return case['current_lg']  # fallback
    sync = compute_synchrony(case['left_peaks'], case['right_peaks'],
                              case['fs'], tolerance_ms)
    return 'G' if sync > sync_threshold else 'L'


def classify_rda_bilateral(case, bilateral_threshold):
    """Classify an RDA case as L or G using bilateral score comparison."""
    result = case['detector_result']
    left_mean = result.get('left_mean_score', 0)
    right_mean = result.get('right_mean_score', 0)
    if np.isnan(left_mean) or np.isnan(right_mean):
        return case['current_lg']
    weaker = min(left_mean, right_mean)
    return 'G' if weaker > bilateral_threshold else 'L'


def compute_metrics(gt_labels, pred_labels):
    """Compute balanced accuracy, per-class counts, kappa."""
    gt = np.array(gt_labels)
    pred = np.array(pred_labels)
    n = len(gt)

    tp = np.sum((gt == 'G') & (pred == 'G'))
    tn = np.sum((gt == 'L') & (pred == 'L'))
    fp = np.sum((gt == 'L') & (pred == 'G'))
    fn = np.sum((gt == 'G') & (pred == 'L'))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # recall for G
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # recall for L
    balanced_acc = (sensitivity + specificity) / 2
    accuracy = (tp + tn) / n if n > 0 else 0

    # Cohen's kappa
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0

    return {
        'balanced_acc': balanced_acc,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'kappa': kappa,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def mcnemar_test(gt, pred_old, pred_new):
    """McNemar's test comparing two methods."""
    gt, pred_old, pred_new = np.array(gt), np.array(pred_old), np.array(pred_new)
    correct_old = (gt == pred_old)
    correct_new = (gt == pred_new)
    # b = old correct, new wrong; c = old wrong, new correct
    b = np.sum(correct_old & ~correct_new)
    c = np.sum(~correct_old & correct_new)
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'chi2': 0, 'p_value': 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / n  # with continuity correction
    from scipy.stats import chi2 as chi2_dist
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return {'b': b, 'c': c, 'chi2': chi2, 'p_value': p}


def run_baseline(cases):
    """Report baseline (current method) accuracy."""
    print('\n' + '=' * 70)
    print('BASELINE: Current spatial_extent > 0.80 method')
    print('=' * 70)

    for modality in ['PD', 'RDA']:
        subset = [c for c in cases if c['modality'] == modality]
        gt = [c['gt_lg'] for c in subset]
        pred = [c['current_lg'] for c in subset]
        m = compute_metrics(gt, pred)

        types = set(c['pattern_type'] for c in subset)
        print(f'\n  {modality} ({len(subset)} segments):')
        for t in sorted(types):
            type_cases = [c for c in subset if c['pattern_type'] == t]
            correct = sum(1 for c in type_cases if c['current_lg'] == c['gt_lg'])
            print(f'    {t.upper()}: {correct}/{len(type_cases)} correct')
        print(f'    Balanced acc: {m["balanced_acc"]:.3f}  |  Kappa: {m["kappa"]:.3f}')
        print(f'    Confusion: TP={m["tp"]} TN={m["tn"]} FP={m["fp"]} FN={m["fn"]}')


def grid_search(cases):
    """Grid search over all parameter combinations."""
    print('\n' + '=' * 70)
    print('GRID SEARCH')
    print('=' * 70)

    pd_cases = [c for c in cases if c['modality'] == 'PD']
    rda_cases = [c for c in cases if c['modality'] == 'RDA']

    # --- PD grid search ---
    print('\nPD: Searching over tolerance_ms × sync_threshold × min_peaks...')
    pd_results = []
    pd_gt = [c['gt_lg'] for c in pd_cases]

    for tol, sync_thr, min_pk in itertools.product(
            TOLERANCE_MS_GRID, SYNC_THRESHOLD_GRID, MIN_PEAKS_GRID):
        pred = [classify_pd_synchrony(c, tol, sync_thr, min_pk) for c in pd_cases]
        m = compute_metrics(pd_gt, pred)
        n_fallback = sum(1 for c in pd_cases
                         if c['n_left'] < min_pk or c['n_right'] < min_pk)
        pd_results.append({
            'tolerance_ms': tol,
            'sync_threshold': sync_thr,
            'min_peaks': min_pk,
            'balanced_acc': m['balanced_acc'],
            'accuracy': m['accuracy'],
            'kappa': m['kappa'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'n_fallback': n_fallback,
            **{k: m[k] for k in ['tp', 'tn', 'fp', 'fn']},
        })

    pd_df = pd.DataFrame(pd_results).sort_values('balanced_acc', ascending=False)
    best_pd = pd_df.iloc[0]
    print(f'\n  Best PD params: tolerance={best_pd["tolerance_ms"]}ms, '
          f'sync_threshold={best_pd["sync_threshold"]}, min_peaks={best_pd["min_peaks"]}')
    print(f'  Balanced acc: {best_pd["balanced_acc"]:.3f}  |  Kappa: {best_pd["kappa"]:.3f}')
    print(f'  G recall: {best_pd["sensitivity"]:.3f}  |  L recall: {best_pd["specificity"]:.3f}')
    print(f'  Fallbacks: {best_pd["n_fallback"]}/50')

    # --- RDA grid search ---
    print('\nRDA: Searching over bilateral_threshold...')
    rda_results = []
    rda_gt = [c['gt_lg'] for c in rda_cases]

    for bt in BILATERAL_THRESHOLD_GRID:
        pred = [classify_rda_bilateral(c, bt) for c in rda_cases]
        m = compute_metrics(rda_gt, pred)
        rda_results.append({
            'bilateral_threshold': bt,
            'balanced_acc': m['balanced_acc'],
            'accuracy': m['accuracy'],
            'kappa': m['kappa'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            **{k: m[k] for k in ['tp', 'tn', 'fp', 'fn']},
        })

    rda_df = pd.DataFrame(rda_results).sort_values('balanced_acc', ascending=False)
    best_rda = rda_df.iloc[0]
    print(f'\n  Best RDA params: bilateral_threshold={best_rda["bilateral_threshold"]}')
    print(f'  Balanced acc: {best_rda["balanced_acc"]:.3f}  |  Kappa: {best_rda["kappa"]:.3f}')
    print(f'  G recall: {best_rda["sensitivity"]:.3f}  |  L recall: {best_rda["specificity"]:.3f}')

    return pd_df, rda_df, pd_cases, rda_cases


def loo_cv(cases, pd_df, rda_df):
    """Leave-one-out cross-validation for both PD and RDA."""
    print('\n' + '=' * 70)
    print('LEAVE-ONE-OUT CROSS-VALIDATION')
    print('=' * 70)

    pd_cases = [c for c in cases if c['modality'] == 'PD']
    rda_cases = [c for c in cases if c['modality'] == 'RDA']

    # --- PD LOO-CV ---
    pd_loo_preds = []
    pd_gt = [c['gt_lg'] for c in pd_cases]
    # Get top-10 param combos to consider
    top_pd_params = pd_df.head(10)[['tolerance_ms', 'sync_threshold', 'min_peaks']].values

    for hold_out_idx in range(len(pd_cases)):
        train_cases = [c for j, c in enumerate(pd_cases) if j != hold_out_idx]
        train_gt = [c['gt_lg'] for c in train_cases]

        # Find best params on training set
        best_acc = -1
        best_pred = pd_cases[hold_out_idx]['current_lg']
        for tol, sync_thr, min_pk in top_pd_params:
            train_pred = [classify_pd_synchrony(c, tol, sync_thr, int(min_pk))
                          for c in train_cases]
            m = compute_metrics(train_gt, train_pred)
            if m['balanced_acc'] > best_acc:
                best_acc = m['balanced_acc']
                best_pred = classify_pd_synchrony(pd_cases[hold_out_idx],
                                                   tol, sync_thr, int(min_pk))
        pd_loo_preds.append(best_pred)

    pd_loo_metrics = compute_metrics(pd_gt, pd_loo_preds)
    print(f'\n  PD LOO-CV balanced acc: {pd_loo_metrics["balanced_acc"]:.3f}  '
          f'|  Kappa: {pd_loo_metrics["kappa"]:.3f}')

    # --- RDA LOO-CV ---
    rda_loo_preds = []
    rda_gt = [c['gt_lg'] for c in rda_cases]

    for hold_out_idx in range(len(rda_cases)):
        train_cases = [c for j, c in enumerate(rda_cases) if j != hold_out_idx]
        train_gt = [c['gt_lg'] for c in train_cases]

        best_acc = -1
        best_pred = rda_cases[hold_out_idx]['current_lg']
        for bt in BILATERAL_THRESHOLD_GRID:
            train_pred = [classify_rda_bilateral(c, bt) for c in train_cases]
            m = compute_metrics(train_gt, train_pred)
            if m['balanced_acc'] > best_acc:
                best_acc = m['balanced_acc']
                best_pred = classify_rda_bilateral(rda_cases[hold_out_idx], bt)
        rda_loo_preds.append(best_pred)

    rda_loo_metrics = compute_metrics(rda_gt, rda_loo_preds)
    print(f'  RDA LOO-CV balanced acc: {rda_loo_metrics["balanced_acc"]:.3f}  '
          f'|  Kappa: {rda_loo_metrics["kappa"]:.3f}')

    return pd_loo_preds, rda_loo_preds


def stability_analysis(pd_df, rda_df):
    """Analyze parameter stability around the optimum."""
    print('\n' + '=' * 70)
    print('STABILITY ANALYSIS')
    print('=' * 70)

    best_pd = pd_df.iloc[0]
    peak_acc = best_pd['balanced_acc']

    print(f'\nPD (peak balanced acc = {peak_acc:.3f}):')
    for param, grid in [('tolerance_ms', TOLERANCE_MS_GRID),
                         ('sync_threshold', SYNC_THRESHOLD_GRID),
                         ('min_peaks', MIN_PEAKS_GRID)]:
        # Filter to rows where other params match best
        other_params = {'tolerance_ms', 'sync_threshold', 'min_peaks'} - {param}
        mask = pd.Series([True] * len(pd_df))
        for op in other_params:
            mask = mask & (pd_df[op] == best_pd[op])
        subset = pd_df[mask].sort_values(param)

        stable_range = subset[subset['balanced_acc'] >= peak_acc - 0.02][param]
        print(f'  {param}: optimal={best_pd[param]}, '
              f'stable range=[{stable_range.min()}, {stable_range.max()}] '
              f'(within 2% of peak)')

    best_rda = rda_df.iloc[0]
    peak_acc_rda = best_rda['balanced_acc']
    stable = rda_df[rda_df['balanced_acc'] >= peak_acc_rda - 0.02]['bilateral_threshold']
    print(f'\nRDA (peak balanced acc = {peak_acc_rda:.3f}):')
    print(f'  bilateral_threshold: optimal={best_rda["bilateral_threshold"]}, '
          f'stable range=[{stable.min()}, {stable.max()}]')


def mcnemar_comparison(cases, pd_df, rda_df):
    """McNemar's test comparing current vs new method."""
    print('\n' + '=' * 70)
    print("McNEMAR'S TEST: Current vs Synchrony/Bilateral")
    print('=' * 70)

    best_pd = pd_df.iloc[0]
    best_rda = rda_df.iloc[0]

    for modality in ['PD', 'RDA']:
        subset = [c for c in cases if c['modality'] == modality]
        gt = [c['gt_lg'] for c in subset]
        old_pred = [c['current_lg'] for c in subset]

        if modality == 'PD':
            new_pred = [classify_pd_synchrony(c, best_pd['tolerance_ms'],
                                               best_pd['sync_threshold'],
                                               int(best_pd['min_peaks']))
                        for c in subset]
        else:
            new_pred = [classify_rda_bilateral(c, best_rda['bilateral_threshold'])
                        for c in subset]

        result = mcnemar_test(gt, old_pred, new_pred)
        print(f'\n  {modality}: b={result["b"]} (old✓ new✗), c={result["c"]} (old✗ new✓)')
        print(f'    chi²={result["chi2"]:.3f}, p={result["p_value"]:.4f}')
        if result['p_value'] < 0.05:
            print(f'    → Significant difference (p < 0.05)')
        else:
            print(f'    → Not significant at p < 0.05')


def generate_heatmap(pd_df):
    """Generate heatmap of PD accuracy vs tolerance_ms × sync_threshold."""
    best_min_peaks = pd_df.iloc[0]['min_peaks']
    subset = pd_df[pd_df['min_peaks'] == best_min_peaks]

    pivot = subset.pivot_table(values='balanced_acc',
                                index='sync_threshold',
                                columns='tolerance_ms',
                                aggfunc='first')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                    vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{v:.1f}' for v in pivot.index])
    ax.set_xlabel('tolerance_ms')
    ax.set_ylabel('sync_threshold')
    ax.set_title(f'PD Balanced Accuracy (min_peaks={int(best_min_peaks)})')
    plt.colorbar(im, ax=ax, label='Balanced Accuracy')

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if val > 0.6 else 'white')

    out_path = RESULTS_DIR / 'pd_synchrony_heatmap.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nHeatmap saved to {out_path}')


def generate_peak_alignment_plots(cases, pd_df):
    """Generate peak alignment plots for misclassified GPD cases."""
    best_pd = pd_df.iloc[0]
    tol = best_pd['tolerance_ms']
    sync_thr = best_pd['sync_threshold']
    min_pk = int(best_pd['min_peaks'])

    # Find GPD cases that current method misclassified
    gpd_cases = [c for c in cases if c['pattern_type'] == 'gpd']
    misclassified = [c for c in gpd_cases if c['current_lg'] == 'L']

    if not misclassified:
        print('No GPD cases misclassified by current method — no alignment plots to make.')
        return

    n_plots = min(4, len(misclassified))
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    tol_samples = int(tol * 200 / 1000)

    for i, case in enumerate(misclassified[:n_plots]):
        ax = axes[i]
        lp = case['left_peaks']
        rp = case['right_peaks']
        sync = compute_synchrony(lp, rp, case['fs'], tol)
        new_lg = classify_pd_synchrony(case, tol, sync_thr, min_pk)

        # Plot left peaks as red ticks on top, right as blue on bottom
        ax.eventplot([lp], lineoffsets=1, linelengths=0.4, colors='red', label='Left peaks')
        ax.eventplot([rp], lineoffsets=0, linelengths=0.4, colors='blue', label='Right peaks')

        # Shade tolerance windows around left peaks
        for pk in lp:
            ax.axvspan(pk - tol_samples, pk + tol_samples,
                       alpha=0.1, color='green', linewidth=0)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Right', 'Left'])
        ax.set_xlabel('Sample index')
        name_short = case['name'][:40]
        ax.set_title(f'{name_short}  |  Expert: GPD  |  Old: {case["current_lg"]}PD  '
                     f'|  New: {new_lg}PD  |  sync={sync:.2f}', fontsize=9)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    out_path = RESULTS_DIR / 'peak_alignment_plots.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Peak alignment plots saved to {out_path}')


def print_summary_table(cases, pd_df, rda_df, pd_loo_preds, rda_loo_preds):
    """Print the final comparison table."""
    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)

    best_pd = pd_df.iloc[0]
    best_rda = rda_df.iloc[0]

    pd_cases = [c for c in cases if c['modality'] == 'PD']
    rda_cases = [c for c in cases if c['modality'] == 'RDA']

    # Current method
    pd_gt = [c['gt_lg'] for c in pd_cases]
    rda_gt = [c['gt_lg'] for c in rda_cases]
    pd_old = [c['current_lg'] for c in pd_cases]
    rda_old = [c['current_lg'] for c in rda_cases]

    # New method (optimized)
    pd_new = [classify_pd_synchrony(c, best_pd['tolerance_ms'],
                                     best_pd['sync_threshold'],
                                     int(best_pd['min_peaks']))
              for c in pd_cases]
    rda_new = [classify_rda_bilateral(c, best_rda['bilateral_threshold'])
               for c in rda_cases]

    rows = []
    for label, pd_pred, rda_pred in [
            ('Current (spatial_extent)', pd_old, rda_old),
            ('Synchrony/Bilateral (optimized)', pd_new, rda_new),
            ('Synchrony/Bilateral (LOO-CV)', pd_loo_preds, rda_loo_preds)]:

        # Per-type accuracy
        lpd_correct = sum(1 for c, p in zip(pd_cases, pd_pred)
                          if c['pattern_type'] == 'lpd' and p == c['gt_lg'])
        gpd_correct = sum(1 for c, p in zip(pd_cases, pd_pred)
                          if c['pattern_type'] == 'gpd' and p == c['gt_lg'])
        lrda_correct = sum(1 for c, p in zip(rda_cases, rda_pred)
                           if c['pattern_type'] == 'lrda' and p == c['gt_lg'])
        grda_correct = sum(1 for c, p in zip(rda_cases, rda_pred)
                           if c['pattern_type'] == 'grda' and p == c['gt_lg'])

        pd_m = compute_metrics(pd_gt, pd_pred)
        rda_m = compute_metrics(rda_gt, rda_pred)
        all_gt = pd_gt + rda_gt
        all_pred = list(pd_pred) + list(rda_pred)
        all_m = compute_metrics(all_gt, all_pred)

        rows.append({
            'Method': label,
            'LPD': f'{lpd_correct}/25',
            'GPD': f'{gpd_correct}/25',
            'PD_bal': f'{pd_m["balanced_acc"]:.3f}',
            'LRDA': f'{lrda_correct}/25',
            'GRDA': f'{grda_correct}/25',
            'RDA_bal': f'{rda_m["balanced_acc"]:.3f}',
            'Overall': f'{all_m["balanced_acc"]:.3f}',
        })

    df = pd.DataFrame(rows)
    print(f'\n{df.to_string(index=False)}')

    print(f'\nOptimal PD params: tolerance_ms={best_pd["tolerance_ms"]}, '
          f'sync_threshold={best_pd["sync_threshold"]}, min_peaks={int(best_pd["min_peaks"])}')
    print(f'Optimal RDA param: bilateral_threshold={best_rda["bilateral_threshold"]}')


def main():
    cases = load_test_cases()
    cases = run_detectors(cases)
    cases = extract_peaks(cases)

    run_baseline(cases)

    pd_df, rda_df, _, _ = grid_search(cases)

    pd_loo_preds, rda_loo_preds = loo_cv(cases, pd_df, rda_df)

    stability_analysis(pd_df, rda_df)
    mcnemar_comparison(cases, pd_df, rda_df)

    generate_heatmap(pd_df)
    generate_peak_alignment_plots(cases, pd_df)

    print_summary_table(cases, pd_df, rda_df, pd_loo_preds, rda_loo_preds)


if __name__ == '__main__':
    main()
