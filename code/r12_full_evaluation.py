"""
Round 12: Full evaluation of all signal processing methods.

Loads ALL patients from the unified data structure:
  - data/labels/patients.csv  (patient list with gold_standard_freq, subtype)
  - data/labels/segments.csv  (segment -> .mat file mapping)
  - data/labels/annotations.csv (per-rater frequency annotations)
  - data/eeg/                 (.mat files)

Computes SP features from EEG segments and evaluates:
  1. Individual SP features (Spearman correlation, no training)
  2. Ridge regression LOPO CV on SP features
  3. Expert-expert agreement baseline

Run: conda run -n foe python code/r12_full_evaluation.py
"""

import sys
import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence as scipy_coherence
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
import scipy.io as sio

warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency, fcn_getBanana

# ── Constants ──────────────────────────────────────────────────────────
FS = 200
FREQ_LO, FREQ_HI = 0.3, 3.5
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
CACHE_DIR = DATA_DIR / 'dl_cache'


# ── Feature computation from bipolar segments (18ch, 2000 samples) ────
def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
    """FFT peak frequency in [freq_lo, freq_hi] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freqs_sub = freqs[mask]
    if np.max(fft_sub) == 0:
        return np.nan
    return freqs_sub[np.argmax(fft_sub)]


def compute_sp_features_from_bipolar(seg_bip, fs, is_gpd):
    """Compute SP features from bipolar EEG (18ch, N samples).

    Unlike compute_sp_features_from_eeg which takes raw 20ch data, this
    works on already-bipolar data (as stored in external_pd_segments.npz).
    """
    features = {}
    features['is_gpd'] = float(is_gpd)

    n_channels = seg_bip.shape[0]

    # Lowpass filter
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # f_B: ACF on lowpassed bipolar channels
    acf_freqs = np.full(n_channels, np.nan)
    for ch in range(n_channels):
        freq, score, _ = compute_acf_frequency(
            seg_lp[ch], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        acf_freqs[ch] = freq

    valid_acf = acf_freqs[np.isfinite(acf_freqs)]
    features['f_B'] = float(np.median(valid_acf)) if len(valid_acf) > 0 else np.nan
    features['n_ch'] = len(valid_acf)

    # Pointiness traces
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # f_peaks
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * PEAK_HEIGHT_FRAC, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # f_fft
    fft_freqs = []
    for ch in range(n_channels):
        f = compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    features['f_fft'] = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # f_tkeo
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh (on seg_bip, NOT seg_lp)
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = scipy_coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs,
                                          nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    return features


# ── Bootstrap CI ──────────────────────────────────────────────────────
def bootstrap_spearman(x, y, n_boot=10000, ci=0.95):
    """Compute Spearman correlation with bootstrap 95% CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan, 0

    rs, _ = spearmanr(x, y)

    rng = np.random.RandomState(42)
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            boot_rs[i], _ = spearmanr(x[idx], y[idx])
        except:
            boot_rs[i] = np.nan

    boot_rs = boot_rs[np.isfinite(boot_rs)]
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_rs, 100 * alpha)
    hi = np.percentile(boot_rs, 100 * (1 - alpha))

    return float(rs), float(lo), float(hi), int(n)


def bootstrap_mae(x, y, n_boot=10000, ci=0.95):
    """Compute MAE with bootstrap 95% CI."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, 0

    mae = float(np.mean(np.abs(x - y)))

    rng = np.random.RandomState(42)
    boot_mae = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_mae[i] = np.mean(np.abs(x[idx] - y[idx]))

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_mae, 100 * alpha)
    hi = np.percentile(boot_mae, 100 * (1 - alpha))

    return float(mae), float(lo), float(hi), int(n)


# ── Ridge LOPO ────────────────────────────────────────────────────────
def ridge_lopo(df, feature_cols, target_col='gold_standard_freq', alpha=1.0):
    """Run leave-one-patient-out Ridge regression.

    Returns array of predictions (NaN for patients that couldn't be predicted).
    """
    n = len(df)
    preds = np.full(n, np.nan)
    patient_ids = df['patient_id'].values

    unique_patients = sorted(set(patient_ids))

    for held_out in unique_patients:
        test_mask = patient_ids == held_out
        train_mask = ~test_mask

        if np.sum(train_mask) < 5:
            continue

        X_train = df.loc[train_mask, feature_cols].values.astype(float)
        y_train = np.log(df.loc[train_mask, target_col].values.astype(float))
        X_test = df.loc[test_mask, feature_cols].values.astype(float)

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            test_col = X_test[:, j]
            X_test[~np.isfinite(test_col), j] = med

        # Add intercept
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            pred_freq = np.exp(pred_log)
            test_idx = np.where(test_mask)[0]
            for i, gi in enumerate(test_idx):
                preds[gi] = pred_freq[i]
        except np.linalg.LinAlgError:
            continue

    return preds


# ── Helpers ────────────────────────────────────────────────────────────
def _load_mat_as_bipolar(mat_path, montage, n_channels):
    """Load a .mat file and return (18, N) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    return data


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    print("=" * 80)
    print("ROUND 12: FULL EVALUATION")
    print("=" * 80)

    # ================================================================
    # STEP 1: Load unified patient/segment/annotation data
    # ================================================================
    print("\n[1] Loading unified dataset...")
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_patients = df_patients[df_patients['excluded'] == False].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'].notna()].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'] > 0].copy()
    print(f"  Non-excluded patients with gold standard: {len(df_patients)}")
    print(f"  Subtypes: {df_patients['subtype'].value_counts().to_dict()}")

    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    df_annotations = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    df_annotations['patient_id'] = df_annotations['patient_id'].astype(str)

    # ================================================================
    # STEP 2: Build per-rater patient-level frequency from annotations
    # ================================================================
    print("\n[2] Building expert ratings from annotations...")
    rater_freq = (df_annotations[df_annotations['frequency_hz'].notna() &
                                  (df_annotations['no_pd'] == False) &
                                  (df_annotations['skipped'] == False)]
                  .groupby(['patient_id', 'rater'])['frequency_hz']
                  .mean().reset_index())

    expert_pivot = rater_freq[rater_freq['rater'].isin(['LB', 'PH', 'SZ'])].copy()
    expert_pivot = expert_pivot.pivot(index='patient_id', columns='rater',
                                       values='frequency_hz').reset_index()
    expert_pivot.columns = ['patient_id'] + [f'expert_{c}' for c in expert_pivot.columns[1:]]
    print(f"  Expert ratings pivoted for {len(expert_pivot)} patients")

    # ================================================================
    # STEP 3: Compute features for all patients from EEG
    # ================================================================
    print("\n[3] Computing features for all patients...")

    records = []
    n_computed = 0

    for _, pat_row in df_patients.iterrows():
        pid = str(pat_row['patient_id'])
        subtype = pat_row['subtype']
        gold_freq = float(pat_row['gold_standard_freq'])
        is_gpd = 1 if subtype == 'gpd' else 0

        rec = {
            'patient_id': pid,
            'subtype': subtype,
            'gold_standard_freq': gold_freq,
            'is_gpd': float(is_gpd),
            'f_B': np.nan,
            'f_peaks': np.nan,
            'f_fft': np.nan,
            'f_tkeo': np.nan,
            'f_coh': np.nan,
        }

        # Find segments for this patient
        pat_segs = df_segments[df_segments['patient_id'] == pid]

        # Load the best (highest variance) segment
        best_seg = None
        best_var = -1
        for _, seg_row in pat_segs.iterrows():
            mat_path = EEG_DIR / seg_row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(mat_path, seg_row['montage'], seg_row['n_channels'])
                v = np.var(seg)
                if v > best_var:
                    best_var = v
                    best_seg = seg
            except Exception:
                continue

        if best_seg is not None:
            try:
                feats = compute_sp_features_from_bipolar(best_seg, FS, is_gpd)
                rec.update({k: feats[k] for k in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']})
                n_computed += 1
            except Exception as e:
                print(f"    WARNING: Feature computation failed for {pid}: {e}")

        records.append(rec)

        if len(records) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {len(records)}/{len(df_patients)} patients ({elapsed:.0f}s)")

    df = pd.DataFrame(records)

    # Merge expert ratings
    df = df.merge(expert_pivot, on='patient_id', how='left')
    for ecol in ['expert_LB', 'expert_PH', 'expert_SZ']:
        if ecol not in df.columns:
            df[ecol] = np.nan

    print(f"\n  Total patients: {len(df)}")
    print(f"  Features computed from EEG: {n_computed}")

    # ================================================================
    # STEP 4: Feature summary
    # ================================================================
    print("\n[4] Feature availability summary:")
    for feat in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']:
        n_valid = df[feat].notna().sum()
        n_finite = np.isfinite(df[feat].values.astype(float)).sum()
        print(f"  {feat:10s}: {n_finite:>3d}/{len(df)} valid")

    print(f"\n  Subtype breakdown:")
    for st in ['lpd', 'gpd']:
        sub = df[df['subtype'] == st]
        print(f"    {st}: {len(sub)} patients")

    # ================================================================
    # STEP 5: Evaluate individual SP features (Spearman correlation)
    # ================================================================
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    gold = df['gold_standard_freq'].values.astype(float)
    subtypes = df['subtype'].values

    def evaluate_feature(name, values, gold, subtypes):
        """Evaluate a single feature against gold standard."""
        results = {}
        for group_name, mask in [('ALL', np.ones(len(gold), dtype=bool)),
                                  ('LPD', subtypes == 'lpd'),
                                  ('GPD', subtypes == 'gpd')]:
            g = gold[mask]
            v = values[mask]
            valid = np.isfinite(v) & np.isfinite(g) & (g > 0)
            g_v, v_v = g[valid], v[valid]

            rs, lo, hi, n = bootstrap_spearman(g_v, v_v)
            mae, mae_lo, mae_hi, _ = bootstrap_mae(g_v, v_v)

            results[group_name] = {
                'n': n, 'rs': rs, 'rs_lo': lo, 'rs_hi': hi,
                'mae': mae, 'mae_lo': mae_lo, 'mae_hi': mae_hi,
            }
        return results

    # Individual features
    print("\n--- Individual Signal Processing Features ---")
    sp_features = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']

    all_feature_results = {}
    header = (f"{'Method':<20s} {'N':>4s} "
              f"{'Spearman':>8s} {'95% CI':>16s} "
              f"{'MAE':>6s} {'95% CI':>14s}")
    separator = "-" * len(header)

    for group_name in ['ALL', 'LPD', 'GPD']:
        print(f"\n  {group_name}:")
        print(f"  {header}")
        print(f"  {separator}")

        for feat in sp_features:
            values = df[feat].values.astype(float)
            res = evaluate_feature(feat, values, gold, subtypes)
            r = res[group_name]
            all_feature_results[(feat, group_name)] = r

            rs_str = f"{r['rs']:.3f}" if np.isfinite(r['rs']) else "  N/A"
            ci_str = f"[{r['rs_lo']:.3f}, {r['rs_hi']:.3f}]" if np.isfinite(r['rs_lo']) else "       N/A      "
            mae_str = f"{r['mae']:.3f}" if np.isfinite(r['mae']) else " N/A"
            mae_ci_str = f"[{r['mae_lo']:.3f},{r['mae_hi']:.3f}]" if np.isfinite(r['mae_lo']) else "     N/A      "

            print(f"  {feat:<20s} {r['n']:>4d} "
                  f"{rs_str:>8s} {ci_str:>16s} "
                  f"{mae_str:>6s} {mae_ci_str:>14s}")

    # ================================================================
    # STEP 6: Ridge regression LOPO CV
    # ================================================================
    print("\n\n--- Ridge Regression LOPO CV ---")

    feature_cols_5 = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']
    feature_cols_6 = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd']

    for alpha_val in [1.0, 5.0]:
        for feat_set_name, feat_cols in [('5_SP', feature_cols_5), ('6_SP+GPD', feature_cols_6)]:
            exp_name = f"Ridge_{feat_set_name}_a{alpha_val}"
            print(f"\n  {exp_name}:")

            preds = ridge_lopo(df, feat_cols, alpha=alpha_val)

            for group_name in ['ALL', 'LPD', 'GPD']:
                if group_name == 'ALL':
                    mask = np.ones(len(df), dtype=bool)
                elif group_name == 'LPD':
                    mask = subtypes == 'lpd'
                else:
                    mask = subtypes == 'gpd'

                g = gold[mask]
                p = preds[mask]
                valid = np.isfinite(p) & np.isfinite(g) & (g > 0)

                rs, lo, hi, n = bootstrap_spearman(g[valid], p[valid])
                mae, mae_lo, mae_hi, _ = bootstrap_mae(g[valid], p[valid])

                rs_str = f"{rs:.3f}" if np.isfinite(rs) else "N/A"
                ci_str = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "N/A"
                mae_str = f"{mae:.3f}" if np.isfinite(mae) else "N/A"
                mae_ci_str = f"[{mae_lo:.3f},{mae_hi:.3f}]" if np.isfinite(mae_lo) else "N/A"

                print(f"    {group_name:>4s}: n={n:>3d}, Spearman={rs_str:>6s} {ci_str:>16s}, "
                      f"MAE={mae_str:>5s} {mae_ci_str:>14s}")

                all_feature_results[(exp_name, group_name)] = {
                    'n': n, 'rs': rs, 'rs_lo': lo, 'rs_hi': hi,
                    'mae': mae, 'mae_lo': mae_lo, 'mae_hi': mae_hi,
                }

    # ================================================================
    # STEP 7: Expert-expert agreement (original 35+ patients with 3 experts)
    # ================================================================
    print("\n\n--- Expert-Expert Agreement (Original Dataset) ---")

    # Get records that have expert ratings
    expert_cols = ['expert_LB', 'expert_PH', 'expert_SZ']
    has_experts = df.dropna(subset=expert_cols, how='all').copy()
    print(f"  Patients with expert ratings: {len(has_experts)}")

    expert_pairs = [('LB', 'PH'), ('LB', 'SZ'), ('PH', 'SZ')]
    expert_pair_labels = ['LB-PH', 'LB-SZ', 'PH-SZ']

    for group_name in ['ALL', 'LPD', 'GPD']:
        if group_name == 'ALL':
            sub = has_experts
        elif group_name == 'LPD':
            sub = has_experts[has_experts['subtype'] == 'lpd']
        else:
            sub = has_experts[has_experts['subtype'] == 'gpd']

        print(f"\n  {group_name} (n={len(sub)}):")
        print(f"  {'Pair':<12s} {'N':>4s} {'Spearman':>8s} {'95% CI':>16s} {'MAE':>6s} {'95% CI':>14s}")
        print(f"  {'-'*66}")

        for pair, pair_label in zip(expert_pairs, expert_pair_labels):
            e1 = sub[f'expert_{pair[0]}'].values.astype(float)
            e2 = sub[f'expert_{pair[1]}'].values.astype(float)

            rs, lo, hi, n = bootstrap_spearman(e1, e2)
            mae, mae_lo, mae_hi, _ = bootstrap_mae(e1, e2)

            rs_str = f"{rs:.3f}" if np.isfinite(rs) else "N/A"
            ci_str = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "N/A"
            mae_str = f"{mae:.3f}" if np.isfinite(mae) else "N/A"
            mae_ci_str = f"[{mae_lo:.3f},{mae_hi:.3f}]" if np.isfinite(mae_lo) else "N/A"

            print(f"  {pair_label:<12s} {n:>4d} {rs_str:>8s} {ci_str:>16s} {mae_str:>6s} {mae_ci_str:>14s}")

        # Also compute expert vs MW gold standard
        print(f"\n  Expert vs MW gold standard ({group_name}):")
        for expert_name in ['LB', 'PH', 'SZ']:
            e = sub[f'expert_{expert_name}'].values.astype(float)
            g = sub['gold_standard_freq'].values.astype(float)

            rs, lo, hi, n = bootstrap_spearman(e, g)
            mae, mae_lo, mae_hi, _ = bootstrap_mae(e, g)

            rs_str = f"{rs:.3f}" if np.isfinite(rs) else "N/A"
            ci_str = f"[{lo:.3f}, {hi:.3f}]" if np.isfinite(lo) else "N/A"
            mae_str = f"{mae:.3f}" if np.isfinite(mae) else "N/A"
            mae_ci_str = f"[{mae_lo:.3f},{mae_hi:.3f}]" if np.isfinite(mae_lo) else "N/A"

            print(f"  {expert_name+' vs MW':<12s} {n:>4d} {rs_str:>8s} {ci_str:>16s} {mae_str:>6s} {mae_ci_str:>14s}")

    # ================================================================
    # STEP 8: Summary comparison table
    # ================================================================
    print("\n\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Method':<25s} | {'ALL':^28s} | {'LPD':^28s} | {'GPD':^28s}")
    print(f"{'':25s} | {'rs':>6s} {'CI':>14s} {'n':>4s} | {'rs':>6s} {'CI':>14s} {'n':>4s} | {'rs':>6s} {'CI':>14s} {'n':>4s}")
    print("-" * 115)

    methods_to_show = sp_features + [
        'Ridge_5_SP_a1.0', 'Ridge_6_SP+GPD_a1.0',
        'Ridge_5_SP_a5.0', 'Ridge_6_SP+GPD_a5.0',
    ]

    for method in methods_to_show:
        parts = []
        for group in ['ALL', 'LPD', 'GPD']:
            r = all_feature_results.get((method, group), {})
            rs = r.get('rs', np.nan)
            lo = r.get('rs_lo', np.nan)
            hi = r.get('rs_hi', np.nan)
            n = r.get('n', 0)
            if np.isfinite(rs):
                parts.append(f"{rs:>6.3f} [{lo:.3f},{hi:.3f}] {n:>4d}")
            else:
                parts.append(f"{'N/A':>6s} {'N/A':>14s} {n:>4d}")
        print(f"{method:<25s} | {parts[0]} | {parts[1]} | {parts[2]}")

    elapsed = time.time() - t0
    print(f"\n\nTotal elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("Done.")


if __name__ == '__main__':
    main()
