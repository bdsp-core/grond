"""
Experiment T3: Timing benchmark for clinical deployment feasibility.

Measures per-segment processing time for:
  1. Base SP feature extraction (f_B, f_peaks, f_fft, f_tkeo, f_coh)
  2. Expanded SP feature extraction (base + f_hps, f_fft_car, n_detected, f_range, signal_variance)
  3. Individual feature components (f_B, f_fft, f_tkeo, f_peaks, f_coh separately)
  4. Model prediction times (Ridge, GBM, Random Forest)

Reports a summary table sorted by total time with performance/ms metric.

Usage:
    conda run -n foe python code/exp_t3_timing.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, compute_sp_features, _compute_fft_peak, ridge_predict_fn,
    _build_segment_level_data, FEATURE_COLS,
    FS, FREQ_LO, FREQ_HI, LOWPASS_HZ, SMOOTHING_SIGMA,
    ACF_MIN_LAG, ACF_THRESHOLD, PEAK_HEIGHT_FRAC, ADJACENT_PAIRS,
    RUNS_DIR,
)
from exp_t1_expanded_features import compute_expanded_features, EXPANDED_FEATURE_COLS
from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
from scipy.signal import butter, filtfilt, coherence as scipy_coherence, find_peaks
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ── Timing helpers ────────────────────────────────────────────────────

def time_function(fn, n_runs):
    """Time a function over n_runs calls. Returns list of elapsed times in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


# ── Individual feature timing functions ───────────────────────────────

def _preprocess_segment(segment):
    """Lowpass filter + pointiness traces (shared preprocessing)."""
    fs = FS
    seg_bip = np.asarray(segment, dtype=np.float64)
    n_channels = seg_bip.shape[0]

    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    return seg_bip, seg_lp, pointiness_traces, sigma_samples


def time_f_B(seg_lp, n_channels):
    """Time f_B: ACF on lowpassed bipolar channels."""
    fs = FS
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
    return float(np.median(valid_acf)) if len(valid_acf) > 0 else np.nan


def time_f_peaks(pointiness_traces, n_channels):
    """Time f_peaks: peak counting frequency."""
    fs = FS
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
    return float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan


def time_f_fft(pointiness_traces, n_channels):
    """Time f_fft: FFT peak on pointiness traces."""
    fs = FS
    fft_freqs = []
    for ch in range(n_channels):
        f = _compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    return float(np.median(fft_freqs)) if fft_freqs else np.nan


def time_f_tkeo(seg_lp, n_channels):
    """Time f_tkeo: TKEO-based frequency."""
    fs = FS
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1] ** 2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = _compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    return float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan


def time_f_coh(seg_bip, n_channels):
    """Time f_coh: coherence-based frequency."""
    fs = FS
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
        except Exception:
            continue
    return float(np.median(coh_freqs)) if coh_freqs else np.nan


def time_preprocessing(segment):
    """Time the shared lowpass + pointiness preprocessing."""
    return _preprocess_segment(segment)


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    N_SEGMENTS = 50

    print("=" * 72)
    print("T3: TIMING BENCHMARK")
    print("=" * 72)

    # Load dataset
    dataset = load_dataset(verbose=True)

    # Collect segments for benchmarking
    all_segs = []
    all_is_gpd = []
    df = dataset['df']
    for _, row in df.iterrows():
        pid = row['patient_id']
        subtype = row['subtype']
        is_gpd = 1 if subtype == 'gpd' else 0
        pat_segs = dataset['segments'].get(pid, [])
        for seg in pat_segs:
            if seg is not None:
                all_segs.append(seg.astype(np.float64))
                all_is_gpd.append(is_gpd)
            if len(all_segs) >= N_SEGMENTS:
                break
        if len(all_segs) >= N_SEGMENTS:
            break

    N = len(all_segs)
    print(f"\nBenchmarking on {N} segments...")

    results = {}

    # ── 1. Preprocessing time (shared by all features) ────────────────
    print("\n  Timing: Preprocessing (lowpass + pointiness)...")
    preproc_times = []
    preproc_cache = []
    for seg in all_segs:
        t0 = time.perf_counter()
        seg_bip, seg_lp, pt_traces, sigma = _preprocess_segment(seg)
        t1 = time.perf_counter()
        preproc_times.append(t1 - t0)
        preproc_cache.append((seg_bip, seg_lp, pt_traces, sigma))

    preproc_ms = np.array(preproc_times) * 1000
    print(f"    Preprocessing: {np.mean(preproc_ms):.2f} +/- {np.std(preproc_ms):.2f} ms")

    # ── 2. Individual feature timing (on preprocessed data) ───────────
    print("\n  Timing: Individual features (post-preprocessing)...")

    individual_features = {}

    # f_B
    fb_times = []
    for seg_bip, seg_lp, pt_traces, sigma in preproc_cache:
        n_ch = seg_lp.shape[0]
        t0 = time.perf_counter()
        time_f_B(seg_lp, n_ch)
        t1 = time.perf_counter()
        fb_times.append(t1 - t0)
    fb_ms = np.array(fb_times) * 1000
    individual_features['f_B'] = {'mean_ms': float(np.mean(fb_ms)), 'std_ms': float(np.std(fb_ms))}
    print(f"    f_B:     {np.mean(fb_ms):7.2f} +/- {np.std(fb_ms):.2f} ms")

    # f_peaks
    fp_times = []
    for seg_bip, seg_lp, pt_traces, sigma in preproc_cache:
        n_ch = pt_traces.shape[0]
        t0 = time.perf_counter()
        time_f_peaks(pt_traces, n_ch)
        t1 = time.perf_counter()
        fp_times.append(t1 - t0)
    fp_ms = np.array(fp_times) * 1000
    individual_features['f_peaks'] = {'mean_ms': float(np.mean(fp_ms)), 'std_ms': float(np.std(fp_ms))}
    print(f"    f_peaks: {np.mean(fp_ms):7.2f} +/- {np.std(fp_ms):.2f} ms")

    # f_fft
    ff_times = []
    for seg_bip, seg_lp, pt_traces, sigma in preproc_cache:
        n_ch = pt_traces.shape[0]
        t0 = time.perf_counter()
        time_f_fft(pt_traces, n_ch)
        t1 = time.perf_counter()
        ff_times.append(t1 - t0)
    ff_ms = np.array(ff_times) * 1000
    individual_features['f_fft'] = {'mean_ms': float(np.mean(ff_ms)), 'std_ms': float(np.std(ff_ms))}
    print(f"    f_fft:   {np.mean(ff_ms):7.2f} +/- {np.std(ff_ms):.2f} ms")

    # f_tkeo
    ft_times = []
    for seg_bip, seg_lp, pt_traces, sigma in preproc_cache:
        n_ch = seg_lp.shape[0]
        t0 = time.perf_counter()
        time_f_tkeo(seg_lp, n_ch)
        t1 = time.perf_counter()
        ft_times.append(t1 - t0)
    ft_ms = np.array(ft_times) * 1000
    individual_features['f_tkeo'] = {'mean_ms': float(np.mean(ft_ms)), 'std_ms': float(np.std(ft_ms))}
    print(f"    f_tkeo:  {np.mean(ft_ms):7.2f} +/- {np.std(ft_ms):.2f} ms")

    # f_coh
    fc_times = []
    for seg_bip, seg_lp, pt_traces, sigma in preproc_cache:
        n_ch = seg_bip.shape[0]
        t0 = time.perf_counter()
        time_f_coh(seg_bip, n_ch)
        t1 = time.perf_counter()
        fc_times.append(t1 - t0)
    fc_ms = np.array(fc_times) * 1000
    individual_features['f_coh'] = {'mean_ms': float(np.mean(fc_ms)), 'std_ms': float(np.std(fc_ms))}
    print(f"    f_coh:   {np.mean(fc_ms):7.2f} +/- {np.std(fc_ms):.2f} ms")

    # ── 3. Full feature extraction: Base SP ───────────────────────────
    print("\n  Timing: Full base SP feature extraction (compute_sp_features)...")
    base_feat_times = []
    base_features_list = []
    for i, seg in enumerate(all_segs):
        is_gpd = all_is_gpd[i]
        t0 = time.perf_counter()
        feats = compute_sp_features(seg, is_gpd)
        t1 = time.perf_counter()
        base_feat_times.append(t1 - t0)
        base_features_list.append(feats)
    base_feat_ms = np.array(base_feat_times) * 1000
    print(f"    Base SP features: {np.mean(base_feat_ms):.2f} +/- {np.std(base_feat_ms):.2f} ms")

    # ── 4. Full feature extraction: Expanded ──────────────────────────
    print("\n  Timing: Expanded feature extraction (base + expanded)...")
    expanded_feat_times = []
    for i, seg in enumerate(all_segs):
        is_gpd = all_is_gpd[i]
        t0 = time.perf_counter()
        base_feats = compute_sp_features(seg, is_gpd)
        extra_feats = compute_expanded_features(seg, base_feats)
        t1 = time.perf_counter()
        expanded_feat_times.append(t1 - t0)
    expanded_feat_ms = np.array(expanded_feat_times) * 1000
    print(f"    Expanded features: {np.mean(expanded_feat_ms):.2f} +/- {np.std(expanded_feat_ms):.2f} ms")

    # ── 5. Model prediction timing ────────────────────────────────────
    print("\n  Timing: Model predictions...")

    # Build training data from the full dataset for realistic model fitting
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_labels = np.array(seg_labels)

    # Filter out rows with all-NaN features
    valid_mask = np.any(np.isfinite(seg_features), axis=1)
    X_all = seg_features[valid_mask].copy()
    y_all = seg_labels[valid_mask]

    # Impute NaNs with median
    for j in range(X_all.shape[1]):
        col = X_all[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_all[~finite, j] = med

    y_all_log = np.log(np.clip(y_all, 0.05, 100.0))

    # Prepare a small test batch (use features from our benchmark segments)
    X_test_batch = np.array([
        [base_features_list[i].get(c, np.nan) for c in FEATURE_COLS]
        for i in range(N)
    ], dtype=float)
    for j in range(X_test_batch.shape[1]):
        col = X_test_batch[:, j]
        finite = np.isfinite(col)
        med = np.median(col[finite]) if np.any(finite) else 0.0
        X_test_batch[~finite, j] = med

    # -- Ridge --
    print("    Training Ridge...")
    X_train_b = np.column_stack([X_all, np.ones(len(X_all))])
    I_reg = np.eye(X_train_b.shape[1])
    I_reg[-1, -1] = 0
    alpha = 1.0
    w_ridge = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                               X_train_b.T @ y_all_log)
    X_test_b = np.column_stack([X_test_batch, np.ones(N)])

    ridge_times = []
    for i in range(N):
        x_single = X_test_b[i:i+1]
        t0 = time.perf_counter()
        pred = np.exp(np.clip(x_single @ w_ridge, np.log(0.1), np.log(10.0)))
        t1 = time.perf_counter()
        ridge_times.append(t1 - t0)
    ridge_ms = np.array(ridge_times) * 1000
    print(f"    Ridge predict:  {np.mean(ridge_ms):.4f} +/- {np.std(ridge_ms):.4f} ms")

    # -- GBM --
    print("    Training GBM...")
    gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                     learning_rate=0.1, random_state=42)
    gbm.fit(X_all, y_all_log)

    gbm_times = []
    for i in range(N):
        x_single = X_test_batch[i:i+1]
        t0 = time.perf_counter()
        pred = np.exp(np.clip(gbm.predict(x_single), np.log(0.1), np.log(10.0)))
        t1 = time.perf_counter()
        gbm_times.append(t1 - t0)
    gbm_ms = np.array(gbm_times) * 1000
    print(f"    GBM predict:    {np.mean(gbm_ms):.4f} +/- {np.std(gbm_ms):.4f} ms")

    # -- Random Forest --
    print("    Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=5,
                                min_samples_leaf=5, random_state=42)
    rf.fit(X_all, y_all_log)

    rf_times = []
    for i in range(N):
        x_single = X_test_batch[i:i+1]
        t0 = time.perf_counter()
        pred = np.exp(np.clip(rf.predict(x_single), np.log(0.1), np.log(10.0)))
        t1 = time.perf_counter()
        rf_times.append(t1 - t0)
    rf_ms = np.array(rf_times) * 1000
    print(f"    RF predict:     {np.mean(rf_ms):.4f} +/- {np.std(rf_ms):.4f} ms")

    # ── 6. Compute Spearman references for performance/ms metric ──────
    # Use previously computed or known Spearman values from experiments.
    # We do a quick LOPO for Ridge base to get a Spearman reference.
    print("\n  Computing Spearman references via quick LOPO...")

    from scipy.stats import spearmanr

    def quick_lopo_spearman(dataset, feature_cols, model_type='ridge', alpha=1.0,
                            gbm_model=None, rf_model=None):
        """Quick LOPO that returns combined Spearman."""
        df = dataset['df']
        features = dataset['features']

        # Build segment-level data
        seg_pids_list = []
        seg_labels_list = []
        seg_feat_rows = []
        for _, row in df.iterrows():
            pid = row['patient_id']
            gold = row['gold_standard_freq']
            pat_feats = features.get(pid, [])
            for feat_dict in pat_feats:
                seg_pids_list.append(pid)
                seg_labels_list.append(gold)
                seg_feat_rows.append([feat_dict.get(c, np.nan) for c in feature_cols])

        if not seg_feat_rows:
            return np.nan

        X = np.array(seg_feat_rows, dtype=float)
        y = np.array(seg_labels_list, dtype=float)
        pids = np.array(seg_pids_list)
        unique_pats = df['patient_id'].values

        patient_preds = {}
        patient_golds = {}

        for pat in unique_pats:
            test_mask = pids == pat
            train_mask = ~test_mask
            if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
                continue

            X_train = X[train_mask].copy()
            y_train = np.log(np.clip(y[train_mask], 0.05, 100.0))
            X_test = X[test_mask].copy()

            # Impute
            for j in range(X_train.shape[1]):
                col = X_train[:, j]
                finite = np.isfinite(col)
                med = np.median(col[finite]) if np.any(finite) else 0.0
                X_train[~finite, j] = med
                tc = X_test[:, j]
                X_test[~np.isfinite(tc), j] = med

            if model_type == 'ridge':
                X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
                X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
                I_r = np.eye(X_train_b.shape[1])
                I_r[-1, -1] = 0
                try:
                    w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_r,
                                        X_train_b.T @ y_train)
                    pred_log = X_test_b @ w
                except np.linalg.LinAlgError:
                    continue
            elif model_type == 'gbm':
                m = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                               learning_rate=0.1, random_state=42)
                m.fit(X_train, y_train)
                pred_log = m.predict(X_test)
            elif model_type == 'rf':
                m = RandomForestRegressor(n_estimators=100, max_depth=5,
                                           min_samples_leaf=5, random_state=42)
                m.fit(X_train, y_train)
                pred_log = m.predict(X_test)
            else:
                continue

            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            preds = np.exp(pred_log)
            valid = np.isfinite(preds)
            if np.any(valid):
                patient_preds[pat] = float(np.mean(preds[valid]))
                patient_golds[pat] = float(df[df['patient_id'] == pat]['gold_standard_freq'].iloc[0])

        if len(patient_preds) < 5:
            return np.nan

        gold_arr = np.array([patient_golds[p] for p in patient_preds])
        pred_arr = np.array([patient_preds[p] for p in patient_preds])

        valid = np.isfinite(gold_arr) & np.isfinite(pred_arr) & (gold_arr > 0)
        if np.sum(valid) < 5:
            return np.nan

        rs, _ = spearmanr(gold_arr[valid], pred_arr[valid])
        return float(rs)

    # Ridge + base features
    print("    Ridge + base features...")
    sp_ridge_base = quick_lopo_spearman(dataset, FEATURE_COLS, model_type='ridge', alpha=1.0)
    print(f"      Spearman = {sp_ridge_base:.4f}")

    # GBM + base features
    print("    GBM + base features...")
    sp_gbm_base = quick_lopo_spearman(dataset, FEATURE_COLS, model_type='gbm')
    print(f"      Spearman = {sp_gbm_base:.4f}")

    # RF + base features
    print("    RF + base features...")
    sp_rf_base = quick_lopo_spearman(dataset, FEATURE_COLS, model_type='rf')
    print(f"      Spearman = {sp_rf_base:.4f}")

    # Ridge + expanded features
    # First add expanded features to dataset
    from exp_t1_expanded_features import add_expanded_features
    print("    Computing expanded features for full dataset...")
    add_expanded_features(dataset)
    print("    Ridge + expanded features...")
    sp_ridge_expanded = quick_lopo_spearman(dataset, EXPANDED_FEATURE_COLS, model_type='ridge', alpha=1.0)
    print(f"      Spearman = {sp_ridge_expanded:.4f}")

    # ── 7. Build summary table ────────────────────────────────────────
    print("\n")
    print("=" * 100)
    print("TIMING BENCHMARK SUMMARY")
    print("=" * 100)

    base_feat_mean = float(np.mean(base_feat_ms))
    base_feat_std = float(np.std(base_feat_ms))
    expanded_feat_mean = float(np.mean(expanded_feat_ms))
    expanded_feat_std = float(np.std(expanded_feat_ms))
    ridge_pred_mean = float(np.mean(ridge_ms))
    ridge_pred_std = float(np.std(ridge_ms))
    gbm_pred_mean = float(np.mean(gbm_ms))
    gbm_pred_std = float(np.std(gbm_ms))
    rf_pred_mean = float(np.mean(rf_ms))
    rf_pred_std = float(np.std(rf_ms))

    # Build rows: (method, feat_time_ms, feat_std, predict_time_ms, predict_std, total_ms, spearman)
    rows = [
        ("Ridge + Base SP",     base_feat_mean, base_feat_std,
         ridge_pred_mean, ridge_pred_std,
         base_feat_mean + ridge_pred_mean, sp_ridge_base),

        ("Ridge + Expanded SP", expanded_feat_mean, expanded_feat_std,
         ridge_pred_mean, ridge_pred_std,
         expanded_feat_mean + ridge_pred_mean, sp_ridge_expanded),

        ("GBM + Base SP",       base_feat_mean, base_feat_std,
         gbm_pred_mean, gbm_pred_std,
         base_feat_mean + gbm_pred_mean, sp_gbm_base),

        ("RF + Base SP",        base_feat_mean, base_feat_std,
         rf_pred_mean, rf_pred_std,
         base_feat_mean + rf_pred_mean, sp_rf_base),
    ]

    # Sort by total time
    rows.sort(key=lambda r: r[5])

    header = f"  {'Method':<25s} {'Feat (ms)':>14s} {'Predict (ms)':>14s} {'Total (ms)':>12s} {'Spearman':>10s} {'Perf/ms':>10s}"
    print(header)
    print(f"  {'-' * (len(header.strip()) - 2)}")

    for method, ft, ft_std, pt, pt_std, total, sp in rows:
        perf_per_ms = sp / total if (total > 0 and np.isfinite(sp)) else np.nan
        ft_str = f"{ft:.2f}+/-{ft_std:.2f}"
        pt_str = f"{pt:.4f}+/-{pt_std:.4f}"
        total_str = f"{total:.2f}"
        sp_str = f"{sp:.4f}" if np.isfinite(sp) else "N/A"
        pp_str = f"{perf_per_ms:.4f}" if np.isfinite(perf_per_ms) else "N/A"
        print(f"  {method:<25s} {ft_str:>14s} {pt_str:>14s} {total_str:>12s} {sp_str:>10s} {pp_str:>10s}")

    # Individual feature breakdown
    print(f"\n  INDIVIDUAL FEATURE BREAKDOWN (post-preprocessing):")
    print(f"  {'Feature':<15s} {'Time (ms)':>14s}")
    print(f"  {'-' * 30}")
    indiv_sorted = sorted(individual_features.items(), key=lambda x: -x[1]['mean_ms'])
    for fname, fdata in indiv_sorted:
        print(f"  {fname:<15s} {fdata['mean_ms']:7.2f} +/- {fdata['std_ms']:.2f}")

    preproc_mean = float(np.mean(preproc_ms))
    preproc_std = float(np.std(preproc_ms))
    print(f"\n  Preprocessing (lowpass + pointiness): {preproc_mean:.2f} +/- {preproc_std:.2f} ms")
    print(f"  Sum of parts: preprocessing + individual features = "
          f"{preproc_mean + sum(f['mean_ms'] for f in individual_features.values()):.2f} ms")
    print(f"  Full base SP features (measured): {base_feat_mean:.2f} ms")
    print(f"    (Difference due to preprocessing being done twice in base SP: "
          f"pointiness for f_B uses same lowpass, etc.)")

    print(f"\n{'=' * 100}")

    # ── 8. Save results ───────────────────────────────────────────────
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / 't3_timing_benchmark.json'

    output = {
        'experiment': 't3_timing_benchmark',
        'n_segments_benchmarked': N,
        'feature_extraction': {
            'base_sp': {
                'mean_ms': round(base_feat_mean, 4),
                'std_ms': round(base_feat_std, 4),
            },
            'expanded_sp': {
                'mean_ms': round(expanded_feat_mean, 4),
                'std_ms': round(expanded_feat_std, 4),
            },
            'preprocessing': {
                'mean_ms': round(preproc_mean, 4),
                'std_ms': round(preproc_std, 4),
            },
            'individual_features': {
                k: {'mean_ms': round(v['mean_ms'], 4), 'std_ms': round(v['std_ms'], 4)}
                for k, v in individual_features.items()
            },
        },
        'model_prediction': {
            'ridge': {
                'mean_ms': round(ridge_pred_mean, 4),
                'std_ms': round(ridge_pred_std, 4),
            },
            'gbm': {
                'mean_ms': round(gbm_pred_mean, 4),
                'std_ms': round(gbm_pred_std, 4),
            },
            'random_forest': {
                'mean_ms': round(rf_pred_mean, 4),
                'std_ms': round(rf_pred_std, 4),
            },
        },
        'combined_pipelines': [],
        'spearman_references': {
            'ridge_base': round(sp_ridge_base, 4) if np.isfinite(sp_ridge_base) else None,
            'ridge_expanded': round(sp_ridge_expanded, 4) if np.isfinite(sp_ridge_expanded) else None,
            'gbm_base': round(sp_gbm_base, 4) if np.isfinite(sp_gbm_base) else None,
            'rf_base': round(sp_rf_base, 4) if np.isfinite(sp_rf_base) else None,
        },
    }

    # Add combined pipeline rows
    for method, ft, ft_std, pt, pt_std, total, sp in rows:
        perf_per_ms = sp / total if (total > 0 and np.isfinite(sp)) else None
        output['combined_pipelines'].append({
            'method': method,
            'feature_time_ms': round(ft, 4),
            'predict_time_ms': round(pt, 4),
            'total_time_ms': round(total, 4),
            'spearman': round(sp, 4) if np.isfinite(sp) else None,
            'performance_per_ms': round(perf_per_ms, 4) if perf_per_ms is not None else None,
        })

    with open(str(out_path), 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {out_path}")
    print("=" * 100)
