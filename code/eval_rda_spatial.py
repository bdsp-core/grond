"""
Evaluate RDA spatial extent estimation against 3-rater ground truth.

Sweeps thresholds for VE, SNR, PLV, and blends.
Reports MAE, Pearson r, ICC(3,1) at optimal threshold.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from rda_spatial_extent import compute_channel_metrics

# ---------- Constants ----------
FS = 200
EEG_DIR = Path(__file__).parent.parent / 'data' / 'eeg'
LABELS_DIR = Path(__file__).parent.parent / 'data' / 'labels'

MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1',
                 'Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
BIPOLAR_PAIRS = [
    ('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
    ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
    ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
    ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
    ('Fz','Cz'),('Cz','Pz'),
]
BIPOLAR_INDICES = np.array([[MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)]
                            for a, b in BIPOLAR_PAIRS])


def load_bipolar(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] == 19:
        return seg[BIPOLAR_INDICES[:, 0]] - seg[BIPOLAR_INDICES[:, 1]]
    elif seg.shape[0] >= 18:
        return seg[:18]
    return None


def icc_3_1(ratings):
    """Compute ICC(3,1) for a ratings matrix (n_subjects x k_raters).

    Two-way mixed, single measures, consistency.
    """
    n, k = ratings.shape
    if n < 2 or k < 2:
        return np.nan

    # Grand mean
    grand_mean = np.mean(ratings)

    # Row means, col means
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    # Sum of squares
    SS_total = np.sum((ratings - grand_mean) ** 2)
    SS_rows = k * np.sum((row_means - grand_mean) ** 2)
    SS_cols = n * np.sum((col_means - grand_mean) ** 2)
    SS_error = SS_total - SS_rows - SS_cols

    # Mean squares
    MS_rows = SS_rows / (n - 1)
    MS_error = SS_error / ((n - 1) * (k - 1))

    # ICC(3,1)
    icc = (MS_rows - MS_error) / (MS_rows + (k - 1) * MS_error)
    return icc


def percentage_agreement(pred, gt, tol=0.1):
    """Fraction of predictions within tolerance of GT."""
    return np.mean(np.abs(pred - gt) <= tol)


def evaluate_metric(all_metrics, gt_spatial, metric_name, metric_key,
                    thresholds=None):
    """Sweep thresholds and find optimal for a given metric."""
    if thresholds is None:
        thresholds = np.arange(0.02, 0.96, 0.01)

    n_seg = len(gt_spatial)
    scores_matrix = np.array([m[metric_key] for m in all_metrics])  # (n_seg, 18)

    best_mae = 999
    best_thr = 0
    best_pred = None

    for thr in thresholds:
        involved = scores_matrix >= thr
        pred_extent = np.sum(involved, axis=1) / 18.0
        mae = np.mean(np.abs(pred_extent - gt_spatial))
        if mae < best_mae:
            best_mae = mae
            best_thr = thr
            best_pred = pred_extent

    # Pearson r
    if np.std(best_pred) > 1e-8 and np.std(gt_spatial) > 1e-8:
        r = np.corrcoef(best_pred, gt_spatial)[0, 1]
    else:
        r = 0.0

    # PA at different tolerances
    pa_01 = percentage_agreement(best_pred, gt_spatial, tol=1/18)
    pa_02 = percentage_agreement(best_pred, gt_spatial, tol=2/18)

    # ICC(3,1) with algorithm as 4th rater
    return {
        'metric': metric_name,
        'best_threshold': best_thr,
        'mae': best_mae,
        'pearson_r': r,
        'pa_1ch': pa_01,
        'pa_2ch': pa_02,
        'predictions': best_pred,
    }


def evaluate_blend(all_metrics, gt_spatial, weights, name,
                   thresholds=None):
    """Evaluate a weighted blend of metrics."""
    if thresholds is None:
        thresholds = np.arange(0.02, 0.96, 0.01)

    n_seg = len(gt_spatial)
    scores_matrix = np.zeros((n_seg, 18))
    for m_key, w in weights.items():
        arr = np.array([m[m_key] for m in all_metrics])
        scores_matrix += w * arr

    best_mae = 999
    best_thr = 0
    best_pred = None

    for thr in thresholds:
        involved = scores_matrix >= thr
        pred_extent = np.sum(involved, axis=1) / 18.0
        mae = np.mean(np.abs(pred_extent - gt_spatial))
        if mae < best_mae:
            best_mae = mae
            best_thr = thr
            best_pred = pred_extent

    if np.std(best_pred) > 1e-8 and np.std(gt_spatial) > 1e-8:
        r = np.corrcoef(best_pred, gt_spatial)[0, 1]
    else:
        r = 0.0

    pa_01 = percentage_agreement(best_pred, gt_spatial, tol=1/18)
    pa_02 = percentage_agreement(best_pred, gt_spatial, tol=2/18)

    return {
        'metric': name,
        'best_threshold': best_thr,
        'mae': best_mae,
        'pearson_r': r,
        'pa_1ch': pa_01,
        'pa_2ch': pa_02,
        'predictions': best_pred,
    }


def main():
    print("=" * 70)
    print("RDA Spatial Extent Evaluation")
    print("=" * 70)

    # Load data
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    # Filter RDA, not excluded, with pdchar_freq_hz
    rda = sl[(sl['subtype'].isin(['lrda', 'grda'])) &
             (sl['excluded'] == False) &
             (sl['pdchar_freq_hz'].notna())].copy()

    # Find segments with all 3 raters
    raters_needed = {'LB', 'PH', 'SZ'}
    ann_spatial = ann[ann['spatial_extent'].notna()].copy()
    seg_raters = ann_spatial.groupby('segment_id')['rater'].apply(set)
    three_rater_segs = seg_raters[seg_raters.apply(
        lambda x: raters_needed.issubset(x))].index

    # Map segment_id -> mat_file
    ann_mat = ann_spatial[['segment_id', 'mat_file']].drop_duplicates()
    mat_map = dict(zip(ann_mat['segment_id'], ann_mat['mat_file']))
    three_rater_mats = {mat_map[sid] for sid in three_rater_segs
                        if sid in mat_map}

    # Qualifying segments
    qualifying = rda[rda['mat_file'].isin(three_rater_mats)].copy()
    print(f"\nQualifying segments: {len(qualifying)}")
    print(f"  LRDA: {(qualifying['subtype'] == 'lrda').sum()}")
    print(f"  GRDA: {(qualifying['subtype'] == 'grda').sum()}")

    # Build GT spatial extent (mean of 3 raters) per mat_file
    # And per-rater spatial extent for ICC
    ann_3rater = ann_spatial[
        (ann_spatial['segment_id'].isin(three_rater_segs)) &
        (ann_spatial['rater'].isin(raters_needed))
    ].copy()
    ann_3rater['mat_file_from_seg'] = ann_3rater['segment_id'].map(mat_map)

    gt_by_mat = {}
    rater_by_mat = {}  # {mat_file: {rater: spatial_extent}}
    for mat_file in qualifying['mat_file']:
        rows = ann_3rater[ann_3rater['mat_file_from_seg'] == mat_file]
        if len(rows) < 3:
            # Try matching on mat_file column directly
            rows = ann_3rater[ann_3rater['mat_file'] == mat_file]
        if len(rows) >= 3:
            gt_by_mat[mat_file] = rows['spatial_extent'].mean()
            rater_by_mat[mat_file] = {
                r: rows[rows['rater'] == r]['spatial_extent'].values[0]
                for r in raters_needed if len(rows[rows['rater'] == r]) > 0
            }

    print(f"Segments with GT spatial extent: {len(gt_by_mat)}")

    # Compute metrics for each segment
    print("\nComputing per-channel metrics...")
    all_metrics = []
    gt_spatial = []
    rater_data = []  # list of {LB, PH, SZ} dicts
    mat_files_used = []
    subtypes_used = []
    freqs_used = []
    n_loaded = 0
    n_failed = 0

    for _, row in qualifying.iterrows():
        mat_file = row['mat_file']
        if mat_file not in gt_by_mat:
            continue

        freq_hz = row['pdchar_freq_hz']
        if np.isnan(freq_hz) or freq_hz < 0.3:
            continue

        seg = load_bipolar(mat_file)
        if seg is None:
            n_failed += 1
            continue

        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f"  Processed {n_loaded} segments...")

        try:
            metrics = compute_channel_metrics(seg, freq_hz)
            all_metrics.append(metrics)
            gt_spatial.append(gt_by_mat[mat_file])
            rater_data.append(rater_by_mat[mat_file])
            mat_files_used.append(mat_file)
            subtypes_used.append(row['subtype'])
            freqs_used.append(freq_hz)
        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"  Error on {mat_file}: {e}")

    gt_spatial = np.array(gt_spatial)
    subtypes_used = np.array(subtypes_used)
    print(f"\nSuccessfully processed: {len(all_metrics)} segments")
    print(f"Failed: {n_failed}")
    print(f"GT spatial extent: mean={gt_spatial.mean():.3f}, "
          f"std={gt_spatial.std():.3f}, "
          f"range=[{gt_spatial.min():.3f}, {gt_spatial.max():.3f}]")

    # Inter-rater ICC for the 3 human raters
    rater_matrix_3 = np.zeros((len(rater_data), 3))
    for i, rd in enumerate(rater_data):
        for j, r in enumerate(['LB', 'PH', 'SZ']):
            rater_matrix_3[i, j] = rd.get(r, np.nan)

    icc_human = icc_3_1(rater_matrix_3)
    print(f"\nInter-rater ICC(3,1) [3 humans]: {icc_human:.4f}")

    # Evaluate individual metrics
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 70)

    results = []
    for metric_name, metric_key in [('VE', 've'), ('SNR', 'snr'), ('PLV', 'plv')]:
        res = evaluate_metric(all_metrics, gt_spatial, metric_name, metric_key)
        results.append(res)

    # Blends
    blend_configs = [
        ('VE+SNR (0.6/0.4)', {'ve': 0.6, 'snr': 0.4}),
        ('VE+SNR (0.5/0.5)', {'ve': 0.5, 'snr': 0.5}),
        ('VE+PLV (0.6/0.4)', {'ve': 0.6, 'plv': 0.4}),
        ('VE+PLV (0.5/0.5)', {'ve': 0.5, 'plv': 0.5}),
        ('VE+SNR+PLV (0.5/0.25/0.25)', {'ve': 0.5, 'snr': 0.25, 'plv': 0.25}),
        ('VE+SNR+PLV (0.4/0.3/0.3)', {'ve': 0.4, 'snr': 0.3, 'plv': 0.3}),
        ('SNR+PLV (0.5/0.5)', {'snr': 0.5, 'plv': 0.5}),
    ]
    for name, weights in blend_configs:
        res = evaluate_blend(all_metrics, gt_spatial, weights, name)
        results.append(res)

    # Print results table
    print(f"\n{'Metric':<35} {'Thr':>5} {'MAE':>7} {'r':>7} "
          f"{'PA±1ch':>7} {'PA±2ch':>7}")
    print("-" * 75)
    for res in sorted(results, key=lambda x: x['mae']):
        print(f"{res['metric']:<35} {res['best_threshold']:>5.2f} "
              f"{res['mae']:>7.4f} {res['pearson_r']:>7.4f} "
              f"{res['pa_1ch']:>7.3f} {res['pa_2ch']:>7.3f}")

    # Best result
    best = min(results, key=lambda x: x['mae'])
    print(f"\nBest: {best['metric']} (threshold={best['best_threshold']:.2f})")
    print(f"  MAE: {best['mae']:.4f}")
    print(f"  Pearson r: {best['pearson_r']:.4f}")

    # ICC(3,1) with algorithm as 4th rater for top-3 methods
    print("\n" + "=" * 70)
    print("ICC(3,1) WITH ALGORITHM AS 4TH RATER")
    print("=" * 70)

    top3 = sorted(results, key=lambda x: x['mae'])[:5]
    for res in top3:
        rater_matrix_4 = np.zeros((len(rater_data), 4))
        for i, rd in enumerate(rater_data):
            for j, r in enumerate(['LB', 'PH', 'SZ']):
                rater_matrix_4[i, j] = rd.get(r, np.nan)
            rater_matrix_4[i, 3] = res['predictions'][i]

        icc_4 = icc_3_1(rater_matrix_4)
        print(f"  {res['metric']:<35} ICC(3,1)={icc_4:.4f}  "
              f"(vs human-only={icc_human:.4f}, "
              f"delta={icc_4 - icc_human:+.4f})")

    # Breakdown by subtype for best method
    print("\n" + "=" * 70)
    print(f"SUBTYPE BREAKDOWN (Best: {best['metric']})")
    print("=" * 70)

    for st in ['lrda', 'grda']:
        mask = subtypes_used == st
        if mask.sum() == 0:
            continue
        gt_sub = gt_spatial[mask]
        pred_sub = best['predictions'][mask]
        mae_sub = np.mean(np.abs(pred_sub - gt_sub))
        if np.std(pred_sub) > 1e-8 and np.std(gt_sub) > 1e-8:
            r_sub = np.corrcoef(pred_sub, gt_sub)[0, 1]
        else:
            r_sub = 0.0
        print(f"  {st.upper()}: n={mask.sum()}, MAE={mae_sub:.4f}, r={r_sub:.4f}, "
              f"GT mean={gt_sub.mean():.3f}, Pred mean={pred_sub.mean():.3f}")

    # Per-rater comparison: algorithm vs each rater
    print("\n" + "=" * 70)
    print("PER-RATER COMPARISON (Best method)")
    print("=" * 70)

    for r_name in ['LB', 'PH', 'SZ']:
        rater_vals = np.array([rd[r_name] for rd in rater_data])
        mae_r = np.mean(np.abs(best['predictions'] - rater_vals))
        r_r = np.corrcoef(best['predictions'], rater_vals)[0, 1]
        print(f"  Algo vs {r_name}: MAE={mae_r:.4f}, r={r_r:.4f}")

    # Also pairwise human-human MAE
    print("\n  Human-human pairwise MAE:")
    for i, r1 in enumerate(['LB', 'PH', 'SZ']):
        for j, r2 in enumerate(['LB', 'PH', 'SZ']):
            if j <= i:
                continue
            v1 = np.array([rd[r1] for rd in rater_data])
            v2 = np.array([rd[r2] for rd in rater_data])
            mae_hh = np.mean(np.abs(v1 - v2))
            r_hh = np.corrcoef(v1, v2)[0, 1]
            print(f"    {r1} vs {r2}: MAE={mae_hh:.4f}, r={r_hh:.4f}")

    # Continuous approach: mean of channel scores as spatial extent proxy
    print("\n" + "=" * 70)
    print("CONTINUOUS (NO THRESHOLD) APPROACHES")
    print("=" * 70)
    continuous_results = {}
    for metric_name, metric_key in [('VE', 've'), ('SNR', 'snr'), ('PLV', 'plv')]:
        scores_arr = np.array([m[metric_key] for m in all_metrics])  # (n, 18)
        # Mean score
        pred_mean = np.mean(scores_arr, axis=1)
        mae_mean = np.mean(np.abs(pred_mean - gt_spatial))
        r_mean = np.corrcoef(pred_mean, gt_spatial)[0, 1] if np.std(pred_mean) > 1e-8 else 0
        continuous_results[metric_name] = pred_mean
        print(f"  {metric_name} mean score:  MAE={mae_mean:.4f}, r={r_mean:.4f}")

    # ICC for continuous PLV mean
    print("\n  ICC(3,1) with continuous PLV mean as 4th rater:")
    for metric_name in ['PLV', 'SNR']:
        rater_matrix_4c = np.zeros((len(rater_data), 4))
        for i, rd in enumerate(rater_data):
            for j, r in enumerate(['LB', 'PH', 'SZ']):
                rater_matrix_4c[i, j] = rd.get(r, np.nan)
            rater_matrix_4c[i, 3] = continuous_results[metric_name][i]
        icc_4c = icc_3_1(rater_matrix_4c)
        print(f"    {metric_name} continuous: ICC(3,1)={icc_4c:.4f} "
              f"(delta={icc_4c - icc_human:+.4f})")

    # Subtype breakdown for continuous PLV
    print("\n  Continuous PLV subtype breakdown:")
    pred_plv_cont = continuous_results['PLV']
    for st in ['lrda', 'grda']:
        mask = subtypes_used == st
        if mask.sum() == 0:
            continue
        gt_sub = gt_spatial[mask]
        pred_sub = pred_plv_cont[mask]
        mae_sub = np.mean(np.abs(pred_sub - gt_sub))
        r_sub = np.corrcoef(pred_sub, gt_sub)[0, 1] if np.std(pred_sub) > 1e-8 and np.std(gt_sub) > 1e-8 else 0
        print(f"    {st.upper()}: n={mask.sum()}, MAE={mae_sub:.4f}, r={r_sub:.4f}, "
              f"GT mean={gt_sub.mean():.3f}, Pred mean={pred_sub.mean():.3f}")

    # Relative threshold: threshold = fraction of per-segment max score
    print("\n" + "=" * 70)
    print("RELATIVE THRESHOLD (frac of per-segment max)")
    print("=" * 70)
    for metric_name, metric_key in [('PLV', 'plv'), ('VE+PLV (0.5/0.5)', None)]:
        if metric_key:
            scores_arr = np.array([m[metric_key] for m in all_metrics])
        else:
            scores_arr = np.array([0.5 * m['ve'] + 0.5 * m['plv'] for m in all_metrics])
        best_mae_rel = 999
        best_frac = 0
        best_pred_rel = None
        for frac in np.arange(0.1, 0.95, 0.05):
            max_per_seg = np.max(scores_arr, axis=1, keepdims=True)
            thr_per_seg = frac * max_per_seg
            involved = scores_arr >= thr_per_seg
            pred_ext = np.sum(involved, axis=1) / 18.0
            mae = np.mean(np.abs(pred_ext - gt_spatial))
            if mae < best_mae_rel:
                best_mae_rel = mae
                best_frac = frac
                best_pred_rel = pred_ext
        r_rel = np.corrcoef(best_pred_rel, gt_spatial)[0, 1] if np.std(best_pred_rel) > 1e-8 else 0
        print(f"  {metric_name}: best frac={best_frac:.2f}, MAE={best_mae_rel:.4f}, r={r_rel:.4f}")

    # Score distribution
    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS (Best metric)")
    print("=" * 70)
    # Get the metric key for the best result
    best_key = None
    for mk_name, mk_key in [('VE', 've'), ('SNR', 'snr'), ('PLV', 'plv')]:
        if mk_name == best['metric']:
            best_key = mk_key
    if best_key:
        all_scores = np.array([m[best_key] for m in all_metrics])
        print(f"  Per-channel score stats:")
        print(f"    Mean: {all_scores.mean():.4f}")
        print(f"    Std:  {all_scores.std():.4f}")
        print(f"    Percentiles: 10%={np.percentile(all_scores, 10):.4f}, "
              f"25%={np.percentile(all_scores, 25):.4f}, "
              f"50%={np.percentile(all_scores, 50):.4f}, "
              f"75%={np.percentile(all_scores, 75):.4f}, "
              f"90%={np.percentile(all_scores, 90):.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
