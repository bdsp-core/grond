"""
Round 10: Expanded dataset evaluation — 120 patients (43 original + 77 new).

For each patient, 1 segment. For original data, pick the segment with best
expert agreement (lowest pairwise std). For new data, use the single annotated
segment.

Experiments:
  r10_lopo_120patients    — Ridge LOPO, alpha=1.0, all 120 patients
  r10_lopo_120patients_a5 — Ridge LOPO, alpha=5.0, all 120 patients
  r10_original_43         — Ridge LOPO, alpha=1.0, original 43 patients only
  r10_new_77              — Ridge LOPO, alpha=1.0, new 77 patients only
  r10_lpd_only_84         — Ridge LOPO, alpha=1.0, all 84 LPD patients
  r10_gpd_only_36         — Ridge LOPO, alpha=1.0, all 36 GPD patients

Run: conda run -n foe_dl python code/r10_expanded_evaluation.py
"""

import sys
import os
import re
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'dl'))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions, RUNS_DIR
from pd_detect_alternate import pd_detect_alternate, fcn_getBanana, bipolar_channels, mono_channels
from pd_pointiness_acf import pd_detect_pointiness_acf, compute_pointiness_trace
from mne.filter import notch_filter, filter_data

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
ANNOT_CSV = PROJECT_DIR / 'data' / '_archive' / 'annotation_candidates' / 'frequency_annotations.csv'
FREQ_LO, FREQ_HI = 0.3, 3.5

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


def extract_patient_id(mat_name):
    """Extract patient ID from mat_name like 'pat0103_20180322_...'."""
    m = re.match(r'^([a-zA-Z]+\d+)_', mat_name)
    if m:
        return m.group(1)
    return mat_name.split('_')[0]


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


def compute_sp_features_from_eeg(data, fs, is_gpd):
    """Compute 8 SP features from raw EEG data."""
    features = {}
    features['is_gpd'] = float(is_gpd)

    # f_B: ACF thr=0.10
    try:
        res_B = pd_detect_pointiness_acf(
            data.copy(), fs,
            method='pointiness', lowpass_hz=15,
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        f_B = res_B['event_frequency']
        if not np.isfinite(f_B):
            f_B = np.nan
        detected_channels = res_B.get('channels', [])
        if detected_channels is None or (isinstance(detected_channels, float) and np.isnan(detected_channels)):
            detected_channels = []
        n_detected = len(detected_channels)
    except:
        f_B = np.nan
        n_detected = 0
    features['f_B'] = f_B
    features['n_ch'] = n_detected

    # Preprocessing
    seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg_bip = np.array(fcn_getBanana(seg_filtered))

    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(seg_bip.shape[0]):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    n_channels = seg_lp.shape[0]
    sigma_samples = max(1, int(0.02 * fs))

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
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
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

    # f_coh
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = coherence(seg_bip[ch_a], seg_bip[ch_b], fs=fs,
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

    features['placeholder'] = 0.0
    return features


# ============================================================
# Feature preparation and ridge regression (from r9)
# ============================================================

SP_FEATURES_8 = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch', 'placeholder']


def prepare_features_lopo(feature_dicts, feature_names, train_idx, test_idx):
    """Build feature matrices with NaN->median from TRAINING set only."""
    p = len(feature_names)
    X_train = np.full((len(train_idx), p), np.nan)
    X_test = np.full((len(test_idx), p), np.nan)

    for i, gi in enumerate(train_idx):
        for j, fn in enumerate(feature_names):
            X_train[i, j] = feature_dicts[gi].get(fn, np.nan)

    for i, gi in enumerate(test_idx):
        for j, fn in enumerate(feature_names):
            X_test[i, j] = feature_dicts[gi].get(fn, np.nan)

    for j in range(p):
        col = X_train[:, j]
        finite_mask = np.isfinite(col)
        med = np.median(col[finite_mask]) if np.any(finite_mask) else 0.0
        X_train[~finite_mask, j] = med
        test_col = X_test[:, j]
        test_nan = ~np.isfinite(test_col)
        X_test[test_nan, j] = med

    return X_train, X_test


def ridge_fit(X, y, alpha=1.0):
    """Fit ridge regression."""
    p = X.shape[1]
    XtX = X.T @ X + alpha * np.eye(p)
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
    return beta


def run_lopo_ridge(all_records, feature_names, alpha=1.0, subset_mask=None):
    """
    Run LOPO ridge regression on the unified patient list.

    all_records: list of dicts with keys:
        'patient_id', 'subtype', 'target_freq', 'features',
        'expert_freqs' (list of up to 3 expert freqs, can be [single_val])

    Returns: list of predicted frequencies (same order as all_records, NaN for excluded).
    """
    n = len(all_records)
    if subset_mask is None:
        subset_mask = np.ones(n, dtype=bool)

    valid_indices = np.where(subset_mask)[0]
    patient_ids = [all_records[i]['patient_id'] for i in valid_indices]
    unique_patients = sorted(set(patient_ids))

    predictions = np.full(n, np.nan)

    for fold_i, held_out_pat in enumerate(unique_patients):
        test_idx = [i for i in valid_indices if all_records[i]['patient_id'] == held_out_pat]
        train_idx = [i for i in valid_indices if all_records[i]['patient_id'] != held_out_pat]

        if len(test_idx) == 0 or len(train_idx) < 5:
            continue

        # Build feature matrices
        feature_dicts = [all_records[i]['features'] for i in range(n)]
        X_train, X_test = prepare_features_lopo(feature_dicts, feature_names,
                                                 train_idx, test_idx)

        # For each expert column, train a separate ridge model
        # Original data has up to 3 experts; new data has 1
        max_experts = max(len(all_records[i]['expert_freqs']) for i in train_idx)
        expert_preds = []

        for e_idx in range(max_experts):
            train_targets = []
            train_mask = []
            for i_local, i_global in enumerate(train_idx):
                ef = all_records[i_global]['expert_freqs']
                if e_idx < len(ef) and np.isfinite(ef[e_idx]) and ef[e_idx] > 0:
                    train_targets.append(ef[e_idx])
                    train_mask.append(True)
                else:
                    train_targets.append(np.nan)
                    train_mask.append(False)

            train_targets = np.array(train_targets)
            train_mask = np.array(train_mask)

            if np.sum(train_mask) < 3:
                continue

            X_e = X_train[train_mask]
            y_e = np.log(train_targets[train_mask])
            beta = ridge_fit(X_e, y_e, alpha=alpha)

            preds_log = X_test @ beta
            preds_log = np.clip(preds_log, np.log(0.1), np.log(10.0))
            expert_preds.append(np.exp(preds_log))

        if len(expert_preds) == 0:
            continue

        fold_preds = np.mean(expert_preds, axis=0)

        for local_i, global_i in enumerate(test_idx):
            predictions[global_i] = fold_preds[local_i]

    return predictions


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    t0 = time.time()

    print("=" * 70)
    print("ROUND 10: EXPANDED DATASET EVALUATION — 120 PATIENTS")
    print("=" * 70)

    # ================================================================
    # STEP 1: Load original 43-patient dataset via load_dataset()
    # ================================================================
    print("\n[1] Loading original dataset...")
    dataset_full = load_dataset()  # list of dicts with mat_path etc.
    print(f"  Original dataset: {len(dataset_full)} segments")

    # ================================================================
    # STEP 2: Select best-agreement segment per patient (original data)
    # ================================================================
    print("\n[2] Selecting best segment per patient (original data)...")

    # Group dataset entries by patient
    patient_to_entries = {}
    for entry in dataset_full:
        pat = extract_patient_id(entry['mat_name'])
        if pat not in patient_to_entries:
            patient_to_entries[pat] = []
        patient_to_entries[pat].append(entry)

    original_records = []
    for pat, entries in patient_to_entries.items():
        best_entry = None
        best_std = np.inf

        for entry in entries:
            expert_vals = [entry.get(f'expert_{e}_freq', np.nan)
                           for e in ['LB', 'PH', 'SZ']]
            valid = [v for v in expert_vals if np.isfinite(v) and v > 0]
            if len(valid) < 2:
                std_val = np.inf
            else:
                std_val = np.std(valid)

            if std_val < best_std:
                best_std = std_val
                best_entry = entry

        if best_entry is None:
            best_entry = entries[0]

        expert_vals = [best_entry.get(f'expert_{e}_freq', np.nan)
                       for e in ['LB', 'PH', 'SZ']]
        valid = [v for v in expert_vals if np.isfinite(v) and v > 0]
        if len(valid) == 0:
            continue

        target_freq = float(np.median(valid))
        subtype = best_entry['subdir']

        original_records.append({
            'patient_id': pat,
            'subtype': subtype,
            'target_freq': target_freq,
            'expert_freqs': expert_vals,  # all 3 expert ratings
            'entry': best_entry,  # for loading EEG
            'source': 'original',
        })

    print(f"  Selected {len(original_records)} patients from original data")
    orig_lpd = sum(1 for r in original_records if r['subtype'] == 'lpd')
    orig_gpd = sum(1 for r in original_records if r['subtype'] == 'gpd')
    print(f"    LPD: {orig_lpd}, GPD: {orig_gpd}")

    # ================================================================
    # STEP 3: Compute SP features for selected original segments from raw EEG
    # ================================================================
    print("\n[3] Computing SP features for original segments (from .mat files)...")

    for rec_i, rec in enumerate(original_records):
        if (rec_i + 1) % 10 == 0 or rec_i == 0:
            print(f"  Processing {rec_i+1}/{len(original_records)}: {rec['patient_id']}...")

        entry = rec['entry']
        data, fs = load_eeg_data(entry)
        is_gpd = 1 if rec['subtype'] == 'gpd' else 0

        if data is not None:
            try:
                data = data.astype(np.float64)
                feats = compute_sp_features_from_eeg(data, fs, is_gpd)
            except Exception as e:
                print(f"    WARNING: Feature extraction failed for {rec['patient_id']}: {e}")
                feats = {
                    'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                    'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd),
                    'n_ch': 0, 'placeholder': 0.0
                }
        else:
            print(f"    WARNING: Could not load EEG for {rec['patient_id']}")
            feats = {
                'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd),
                'n_ch': 0, 'placeholder': 0.0
            }

        rec['features'] = feats

    # ================================================================
    # STEP 4: Load new 77-patient data from CSV
    # ================================================================
    print("\n[4] Loading new 77-patient data...")
    df_new = pd.read_csv(str(ANNOT_CSV))
    df_new = df_new[df_new['expert_annotation'] != 'skip'].copy()
    df_new['expert_annotation'] = df_new['expert_annotation'].astype(float)

    new_records = []
    for _, row in df_new.iterrows():
        patient_id = str(row['patient_id'])
        subtype = row['subtype']
        target_freq = float(row['expert_annotation'])
        is_gpd = 1 if subtype == 'gpd' else 0

        # Features are already computed in the CSV
        feats = {
            'f_B': float(row['f_B']) if pd.notna(row['f_B']) and row['f_B'] != '' else np.nan,
            'f_peaks': float(row['f_peaks']) if pd.notna(row['f_peaks']) and row['f_peaks'] != '' else np.nan,
            'f_fft': float(row['f_fft']) if pd.notna(row['f_fft']) and row['f_fft'] != '' else np.nan,
            'f_tkeo': float(row['f_tkeo']) if pd.notna(row['f_tkeo']) and row['f_tkeo'] != '' else np.nan,
            'f_coh': float(row['f_coh']) if pd.notna(row['f_coh']) and row['f_coh'] != '' else np.nan,
            'is_gpd': float(is_gpd),
            'n_ch': 0.0,  # not available for new data
            'placeholder': 0.0,
        }

        new_records.append({
            'patient_id': patient_id,
            'subtype': subtype,
            'target_freq': target_freq,
            'expert_freqs': [target_freq],  # single expert
            'features': feats,
            'source': 'new',
        })

    print(f"  New data: {len(new_records)} patients")
    new_lpd = sum(1 for r in new_records if r['subtype'] == 'lpd')
    new_gpd = sum(1 for r in new_records if r['subtype'] == 'gpd')
    print(f"    LPD: {new_lpd}, GPD: {new_gpd}")

    # ================================================================
    # STEP 5: Combine into unified dataset
    # ================================================================
    all_records = original_records + new_records
    total = len(all_records)
    total_lpd = sum(1 for r in all_records if r['subtype'] == 'lpd')
    total_gpd = sum(1 for r in all_records if r['subtype'] == 'gpd')

    print(f"\n[5] Combined dataset: {total} patients ({total_lpd} LPD + {total_gpd} GPD)")

    # ================================================================
    # STEP 6: Run experiments
    # ================================================================

    # Helper to compute Spearman, MAE from predictions
    def compute_metrics_simple(records, predictions, subset_mask=None):
        """Compute Spearman and MAE, broken down by LPD/GPD."""
        if subset_mask is None:
            subset_mask = np.ones(len(records), dtype=bool)

        results = {}
        for ptype in ['lpd', 'gpd', 'all']:
            if ptype == 'all':
                mask = subset_mask & np.isfinite(predictions)
            else:
                mask = np.array([
                    subset_mask[i] and np.isfinite(predictions[i])
                    and records[i]['subtype'] == ptype
                    for i in range(len(records))
                ])

            idx = np.where(mask)[0]
            if len(idx) < 3:
                results[f'{ptype}_n'] = len(idx)
                results[f'{ptype}_spearman'] = np.nan
                results[f'{ptype}_mae'] = np.nan
                continue

            targets = np.array([records[i]['target_freq'] for i in idx])
            preds = np.array([predictions[i] for i in idx])

            rs, _ = spearmanr(targets, preds)
            mae = float(np.mean(np.abs(targets - preds)))

            results[f'{ptype}_n'] = len(idx)
            results[f'{ptype}_spearman'] = round(float(rs), 4)
            results[f'{ptype}_mae'] = round(mae, 4)

        return results

    def save_experiment_json(experiment_name, metrics_dict):
        """Save experiment results as JSON for the dashboard."""
        out = {'experiment': experiment_name, 'timestamp': time.time()}
        out.update(metrics_dict)
        out_path = RUNS_DIR / f'{experiment_name}.json'
        with open(str(out_path), 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

    # Masks
    is_original = np.array([r['source'] == 'original' for r in all_records])
    is_new = np.array([r['source'] == 'new' for r in all_records])
    is_lpd = np.array([r['subtype'] == 'lpd' for r in all_records])
    is_gpd = np.array([r['subtype'] == 'gpd' for r in all_records])
    all_mask = np.ones(total, dtype=bool)

    experiments = [
        ('r10_lopo_120patients',    all_mask,    1.0),
        ('r10_lopo_120patients_a5', all_mask,    5.0),
        ('r10_original_43',         is_original, 1.0),
        ('r10_new_77',              is_new,      1.0),
        ('r10_lpd_only_84',         is_lpd,      1.0),
        ('r10_gpd_only_36',         is_gpd,      1.0),
    ]

    all_experiment_results = {}

    for exp_name, mask, alpha in experiments:
        n_sub = int(np.sum(mask))
        n_lpd_sub = int(np.sum(mask & is_lpd))
        n_gpd_sub = int(np.sum(mask & is_gpd))
        print(f"\n{'='*60}")
        print(f"Running: {exp_name} (n={n_sub}, LPD={n_lpd_sub}, GPD={n_gpd_sub}, alpha={alpha})")
        print(f"{'='*60}")

        preds = run_lopo_ridge(all_records, SP_FEATURES_8, alpha=alpha, subset_mask=mask)
        metrics = compute_metrics_simple(all_records, preds, subset_mask=mask)

        print(f"  Results:")
        print(f"    ALL:  n={metrics.get('all_n','?')}, Spearman={metrics.get('all_spearman','?')}, MAE={metrics.get('all_mae','?')}")
        print(f"    LPD:  n={metrics.get('lpd_n','?')}, Spearman={metrics.get('lpd_spearman','?')}, MAE={metrics.get('lpd_mae','?')}")
        print(f"    GPD:  n={metrics.get('gpd_n','?')}, Spearman={metrics.get('gpd_spearman','?')}, MAE={metrics.get('gpd_mae','?')}")

        # Also save to dashboard JSON with compatible keys
        dashboard_metrics = {
            'lpd_n': metrics.get('lpd_n', 0),
            'gpd_n': metrics.get('gpd_n', 0),
            'lpd_mae': metrics.get('lpd_mae', np.nan),
            'gpd_mae': metrics.get('gpd_mae', np.nan),
            'lpd_spearman_r': metrics.get('lpd_spearman', np.nan),
            'gpd_spearman_r': metrics.get('gpd_spearman', np.nan),
            'combined_spearman': metrics.get('all_spearman', np.nan),
            'combined_mae': metrics.get('all_mae', np.nan),
        }
        save_experiment_json(exp_name, dashboard_metrics)

        all_experiment_results[exp_name] = metrics

    # ================================================================
    # STEP 7: Also run r10_lopo_120patients through evaluate_predictions
    #         for the original 43 patients (which have 3-expert format)
    # ================================================================
    print(f"\n{'='*60}")
    print("Running: r10_lopo_120patients via evaluate_predictions (original 43 subset)")
    print(f"{'='*60}")

    # Re-run full 120-patient LOPO
    full_preds = run_lopo_ridge(all_records, SP_FEATURES_8, alpha=1.0)

    # Build predictions dict mapping mat_name -> prediction for original patients
    eval_predictions = {}
    for i, rec in enumerate(original_records):
        mat_name = rec['entry']['mat_name']
        if np.isfinite(full_preds[i]):
            eval_predictions[mat_name] = float(full_preds[i])

    print(f"  Mapped {len(eval_predictions)} predictions to mat_names")
    if len(eval_predictions) > 5:
        ep_metrics = evaluate_predictions(dataset_full, eval_predictions,
                                          'r10_lopo_120patients_eval43')

    # ================================================================
    # STEP 8: Comprehensive comparison table
    # ================================================================
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE — ROUND 10 EXPANDED EVALUATION")
    print("=" * 80)

    # Expert-expert baselines
    ee_lpd_spearman = 0.411
    ee_gpd_spearman = 0.044
    ee_combined_spearman = (ee_lpd_spearman + ee_gpd_spearman) / 2

    header = f"{'Experiment':<30s} {'N':>5s} {'ALL rs':>8s} {'ALL MAE':>8s} {'LPD n':>6s} {'LPD rs':>8s} {'LPD MAE':>8s} {'GPD n':>6s} {'GPD rs':>8s} {'GPD MAE':>8s}"
    print(header)
    print("-" * len(header))

    for exp_name, _, _ in experiments:
        m = all_experiment_results[exp_name]
        n_all = m.get('all_n', '?')
        all_rs = m.get('all_spearman', np.nan)
        all_mae = m.get('all_mae', np.nan)
        lpd_n = m.get('lpd_n', '?')
        lpd_rs = m.get('lpd_spearman', np.nan)
        lpd_mae = m.get('lpd_mae', np.nan)
        gpd_n = m.get('gpd_n', '?')
        gpd_rs = m.get('gpd_spearman', np.nan)
        gpd_mae = m.get('gpd_mae', np.nan)

        def fmt(v):
            if isinstance(v, float) and np.isfinite(v):
                return f"{v:.3f}"
            return str(v)

        print(f"  {exp_name:<28s} {str(n_all):>5s} {fmt(all_rs):>8s} {fmt(all_mae):>8s} "
              f"{str(lpd_n):>6s} {fmt(lpd_rs):>8s} {fmt(lpd_mae):>8s} "
              f"{str(gpd_n):>6s} {fmt(gpd_rs):>8s} {fmt(gpd_mae):>8s}")

    print(f"\n  {'Expert-Expert (pooled)':<28s} {'':>5s} {fmt(ee_combined_spearman):>8s} {'':>8s} "
          f"{'':>6s} {fmt(ee_lpd_spearman):>8s} {'':>8s} "
          f"{'':>6s} {fmt(ee_gpd_spearman):>8s} {'':>8s}")

    # Performance ratios
    print("\n  Performance vs Expert-Expert baseline:")
    for exp_name, _, _ in experiments:
        m = all_experiment_results[exp_name]
        lpd_rs = m.get('lpd_spearman', np.nan)
        gpd_rs = m.get('gpd_spearman', np.nan)
        if np.isfinite(lpd_rs) and ee_lpd_spearman > 0:
            lpd_pct = lpd_rs / ee_lpd_spearman * 100
        else:
            lpd_pct = np.nan
        if np.isfinite(gpd_rs) and ee_gpd_spearman > 0:
            gpd_pct = gpd_rs / ee_gpd_spearman * 100
        else:
            gpd_pct = np.nan

        lpd_s = f"{lpd_pct:.0f}%" if np.isfinite(lpd_pct) else "N/A"
        gpd_s = f"{gpd_pct:.0f}%" if np.isfinite(gpd_pct) else "N/A"
        print(f"    {exp_name:<28s} LPD: {lpd_s:>6s} of expert-expert, GPD: {gpd_s:>6s} of expert-expert")

    # Feature value summary
    print("\n  Feature value summary (all 120 patients):")
    for fname in SP_FEATURES_8:
        vals = [r['features'].get(fname, np.nan) for r in all_records]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            print(f"    {fname:<15s}: n_valid={len(vals):>3d}, "
                  f"mean={np.mean(vals):.3f}, median={np.median(vals):.3f}, "
                  f"std={np.std(vals):.3f}")
        else:
            print(f"    {fname:<15s}: all NaN")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("  Done.")
