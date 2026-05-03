"""
Evaluation harness v2 for the unified patient dataset.

Uses the unified data structure:
  - data/labels/patients.csv  (patient list with gold_standard_freq, subtype)
  - data/labels/segments.csv  (segment list with .mat file references)
  - data/eeg/                 (.mat files: bipolar 18ch or monopolar 20ch)

Provides:
  - load_dataset(): preloads all segments + precomputed SP features
  - evaluate_experiment(): LOPO CV with bootstrap CIs
  - ridge_predict_fn(): Ridge regression baseline
  - compute_sp_features(): SP feature extractor for a single segment

Run baseline test:
    conda run -n foe python code/optimization_harness_v2.py
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence as scipy_coherence
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr
import scipy.io as sio

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency, fcn_getBanana

# ── Constants ─────────────────────────────────────────────────────────
FS = 200
FREQ_LO, FREQ_HI = 0.3, 3.5
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
MAX_SEGMENTS_PER_PATIENT = 5

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]

# Hemisphere channel indices (into 18-channel bipolar montage)
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]    # Fp1-F7, F7-T3, T3-T5, T5-O1, Fp1-F3, F3-C3, C3-P3, P3-O1
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]  # Fp2-F8, F8-T4, T4-T6, T6-O2, Fp2-F4, F4-C4, C4-P4, P4-O2

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
CACHE_DIR = DATA_DIR / 'dl_cache'

RESULTS_DIR = PROJECT_DIR / 'results'
RUNS_DIR = RESULTS_DIR / 'optimization_runs_v2'
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── SP Feature computation ────────────────────────────────────────────

def _compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
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


def compute_sp_features(segment, is_gpd):
    """Compute SP features from a single (18, 2000) bipolar segment.

    Returns dict with keys:
        f_B, f_peaks, f_fft, f_tkeo, f_coh, is_gpd  (frequency features)
        lat_idx, lat_energy_ratio, lat_acf_ratio      (laterality features)
    """
    fs = FS
    seg_bip = np.asarray(segment, dtype=np.float64)
    n_channels = seg_bip.shape[0]

    features = {'is_gpd': float(is_gpd)}

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
        f = _compute_fft_peak(pointiness_traces[ch], fs)
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
        f = _compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh (on raw bipolar, NOT lowpassed)
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
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # ── Laterality features ──────────────────────────────────────────
    # lat_idx: lateralization index from ACF periodicity scores (R-L)/(R+L)
    #   Follows pd_detect_alternate convention: NaN channels → 0 (no periodicity)
    #   Range [-1, +1]: negative=left, positive=right
    scores_for_lat = np.where(np.isnan(acf_freqs), 0.0, np.abs(acf_freqs))
    # Use pointiness-trace peak heights as periodicity strength per channel
    peak_strengths = np.zeros(n_channels)
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx > 0:
            pks, props = find_peaks(pt, height=mx * PEAK_HEIGHT_FRAC, distance=int(0.2 * fs))
            if len(pks) >= 2:
                peak_strengths[ch] = np.mean(props['peak_heights'])

    left_strength = np.mean(peak_strengths[LEFT_INDICES])
    right_strength = np.mean(peak_strengths[RIGHT_INDICES])
    denom = left_strength + right_strength
    features['lat_idx'] = float((right_strength - left_strength) / denom) if denom > 0 else 0.0

    # lat_energy_ratio: log ratio of right/left RMS energy
    left_energy = np.mean([np.sqrt(np.mean(seg_lp[ch] ** 2)) for ch in LEFT_INDICES])
    right_energy = np.mean([np.sqrt(np.mean(seg_lp[ch] ** 2)) for ch in RIGHT_INDICES])
    if left_energy > 0 and right_energy > 0:
        features['lat_energy_ratio'] = float(np.log(right_energy / left_energy))
    else:
        features['lat_energy_ratio'] = 0.0

    # lat_acf_ratio: asymmetry of ACF frequency estimates (R-L)/(R+L)
    left_acf = acf_freqs[LEFT_INDICES]
    right_acf = acf_freqs[RIGHT_INDICES]
    left_valid = left_acf[np.isfinite(left_acf)]
    right_valid = right_acf[np.isfinite(right_acf)]
    if len(left_valid) > 0 and len(right_valid) > 0:
        lm, rm = np.median(left_valid), np.median(right_valid)
        d = lm + rm
        features['lat_acf_ratio'] = float((rm - lm) / d) if d > 0 else 0.0
    else:
        features['lat_acf_ratio'] = 0.0

    return features


# ── Bootstrap CI ──────────────────────────────────────────────────────

def bootstrap_ci(x, y, metric_fn, n_boot=10000, ci=0.95):
    """Compute metric with bootstrap confidence interval."""
    observed = metric_fn(x, y)
    boot_vals = []
    n = len(x)
    rng = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            boot_vals.append(metric_fn(x[idx], y[idx]))
        except Exception:
            continue
    boot_vals = np.array(boot_vals)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_vals, alpha * 100)
    hi = np.percentile(boot_vals, (1 - alpha) * 100)
    return observed, lo, hi


# ── Dataset loading ───────────────────────────────────────────────────

def _load_mat_as_bipolar(mat_path, montage, n_channels):
    """Load a .mat file and return (18, N) bipolar array.

    If monopolar (20ch), convert to bipolar via fcn_getBanana.
    If already bipolar (18ch), return as-is.
    """
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    return data


def load_dataset(verbose=True):
    """Load the unified patient dataset with preloaded segments and features.

    Reads from:
      - data/labels/patients.csv  (patient list with gold_standard_freq, subtype)
      - data/labels/segments.csv  (segment -> .mat file mapping)
      - data/eeg/                 (.mat files)

    Returns a dict with:
        'df': DataFrame with columns patient_id, subtype, gold_standard_freq
        'segments': dict mapping patient_id -> list of (18, 2000) numpy arrays
        'features': dict mapping patient_id -> list of feature dicts
                     (each with f_B, f_peaks, f_fft, f_tkeo, f_coh, is_gpd)
    """
    t0 = time.time()

    # ── Load patients and segments CSVs ───────────────────────────────
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_patients = df_patients[df_patients['excluded'] != True].copy()
    # Only keep patients with a valid gold standard frequency
    df_patients = df_patients[df_patients['gold_standard_freq'].notna()].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'] > 0].copy()

    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    if verbose:
        print(f"Loading dataset: {len(df_patients)} patients with gold standard...")

    # ── Build unified DataFrame and segment/feature dicts ─────────────
    records = []
    all_segments = {}
    all_features = {}
    n_computed = 0
    n_missing = 0

    for _, pat_row in df_patients.iterrows():
        pid = str(pat_row['patient_id'])
        subtype = pat_row['subtype']
        gold = float(pat_row['gold_standard_freq'])
        is_gpd = 1 if subtype == 'gpd' else 0

        # Get segments for this patient (up to MAX_SEGMENTS_PER_PATIENT)
        pat_segs = df_segments[df_segments['patient_id'] == pid]

        # Load segments from .mat files
        loaded_segs = []
        for _, seg_row in pat_segs.iterrows():
            mat_path = EEG_DIR / seg_row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(
                    mat_path, seg_row['montage'], seg_row['n_channels'])
                loaded_segs.append(seg)
            except Exception:
                continue

        laterality = pat_row.get('laterality', '')
        if pd.isna(laterality):
            laterality = ''
        laterality = str(laterality)

        if not loaded_segs:
            n_missing += 1
            records.append({
                'patient_id': pid,
                'subtype': subtype,
                'gold_standard_freq': gold,
                'laterality': laterality,
            })
            all_segments[pid] = []
            all_features[pid] = []
            continue

        # Pick top-variance segments (up to MAX_SEGMENTS_PER_PATIENT)
        if len(loaded_segs) > MAX_SEGMENTS_PER_PATIENT:
            var_idx = sorted(range(len(loaded_segs)),
                             key=lambda i: -np.var(loaded_segs[i]))
            loaded_segs = [loaded_segs[i] for i in var_idx[:MAX_SEGMENTS_PER_PATIENT]]

        # Compute features for each segment
        seg_features = []
        for seg in loaded_segs:
            try:
                feats = compute_sp_features(seg, is_gpd)
                n_computed += 1
            except Exception:
                feats = {'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                         'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd),
                         'lat_idx': 0.0, 'lat_energy_ratio': 0.0, 'lat_acf_ratio': 0.0}
            seg_features.append(feats)

        records.append({
            'patient_id': pid,
            'subtype': subtype,
            'gold_standard_freq': gold,
            'laterality': laterality,
        })
        all_segments[pid] = loaded_segs
        all_features[pid] = seg_features

        if verbose and (len(records) % 50 == 0):
            elapsed = time.time() - t0
            print(f"  Processed {len(records)}/{len(df_patients)} patients ({elapsed:.0f}s)")

    df = pd.DataFrame(records)

    if verbose:
        elapsed = time.time() - t0
        print(f"\n  Total patients: {len(df)}")
        n_lpd = len(df[df['subtype'] == 'lpd'])
        n_gpd = len(df[df['subtype'] == 'gpd'])
        n_lat = len(df[(df['laterality'] != '') & (df['laterality'].notna())])
        print(f"  LPD: {n_lpd}, GPD: {n_gpd}")
        print(f"  With laterality labels: {n_lat}")
        print(f"  Feature sets computed: {n_computed}")
        if n_missing > 0:
            print(f"  Patients with no segments: {n_missing}")
        print(f"  Loaded in {elapsed:.1f}s")

    return {
        'df': df,
        'segments': all_segments,
        'features': all_features,
    }


# ── Feature matrix helpers ────────────────────────────────────────────

FEATURE_COLS = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd']
LATERALITY_FEATURE_COLS = ['lat_idx', 'lat_energy_ratio', 'lat_acf_ratio']
ALL_FEATURE_COLS = FEATURE_COLS + LATERALITY_FEATURE_COLS


def _build_segment_level_data(dataset):
    """Expand patient-level dataset to segment-level rows for training.

    Returns:
        seg_patient_ids: list of patient_id per segment row
        seg_labels: array of gold_standard_freq per segment row
        seg_features: array of shape (n_segments, 6) feature matrix
        seg_segments: list of (18, 2000) arrays
    """
    df = dataset['df']
    features = dataset['features']
    segments = dataset['segments']

    seg_patient_ids = []
    seg_labels = []
    seg_feat_rows = []
    seg_arrays = []

    for _, row in df.iterrows():
        pid = row['patient_id']
        gold = row['gold_standard_freq']
        pat_feats = features.get(pid, [])
        pat_segs = segments.get(pid, [])

        for i, feat_dict in enumerate(pat_feats):
            seg_patient_ids.append(pid)
            seg_labels.append(gold)
            seg_feat_rows.append([feat_dict.get(c, np.nan) for c in ALL_FEATURE_COLS])
            if i < len(pat_segs):
                seg_arrays.append(pat_segs[i])
            else:
                seg_arrays.append(None)

    n_feats = len(ALL_FEATURE_COLS)
    seg_features = np.array(seg_feat_rows, dtype=float) if seg_feat_rows else np.empty((0, n_feats))
    seg_labels = np.array(seg_labels, dtype=float)

    return seg_patient_ids, seg_labels, seg_features, seg_arrays


# ── Ridge baseline ────────────────────────────────────────────────────

def ridge_predict_fn(alpha=1.0):
    """Return a predict_fn that trains Ridge on log(freq) using the feature matrix.

    The returned function signature:
        predict_fn(train_segments, train_labels, train_features,
                   test_segments, test_features) -> predicted_frequencies
    """
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        X_train = train_features.copy()
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))
        X_test = test_features.copy()

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
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.nan)

    return _predict


# ── LOPO evaluation ───────────────────────────────────────────────────

def evaluate_experiment(dataset, experiment_name, predict_fn, eval_type='patient_lopo'):
    """Run Leave-One-Patient-Out CV and compute comprehensive metrics.

    Args:
        dataset: dict from load_dataset()
        experiment_name: string identifier
        predict_fn: callable(train_segments, train_labels, train_features,
                            test_segments, test_features) -> predicted_freqs
        eval_type: string tag for the JSON output

    Returns: metrics dict (also writes JSON)
    """
    t0 = time.time()
    print(f"\nRunning LOPO experiment: {experiment_name}")

    df = dataset['df']
    features = dataset['features']
    segments = dataset['segments']

    # Build segment-level data
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)
    unique_patients = df['patient_id'].values

    # LOPO: for each patient, hold out all their segments
    patient_preds = {}  # patient_id -> list of segment predictions
    n_processed = 0

    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask

        if np.sum(test_mask) == 0:
            continue
        if np.sum(train_mask) < 5:
            continue

        train_segs = [seg_arrays[i] for i in np.where(train_mask)[0]]
        test_segs = [seg_arrays[i] for i in np.where(test_mask)[0]]

        try:
            preds = predict_fn(
                train_segs, seg_labels[train_mask], seg_features[train_mask],
                test_segs, seg_features[test_mask]
            )
            patient_preds[pat] = preds
        except Exception as e:
            patient_preds[pat] = np.full(int(np.sum(test_mask)), np.nan)

        n_processed += 1
        if n_processed % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {n_processed}/{len(unique_patients)} patients ({elapsed:.0f}s)")

    # Average predictions per patient
    patient_level_pred = {}
    for pat, preds in patient_preds.items():
        preds = np.asarray(preds, dtype=float)
        valid = preds[np.isfinite(preds)]
        if len(valid) > 0:
            patient_level_pred[pat] = float(np.mean(valid))
        else:
            patient_level_pred[pat] = np.nan

    # Collect gold vs pred arrays by subtype
    results = {}
    for group in ['lpd', 'gpd', 'combined']:
        gold_vals = []
        pred_vals = []
        for _, row in df.iterrows():
            pid = row['patient_id']
            subtype = row['subtype']
            if group != 'combined' and subtype != group:
                continue
            if pid not in patient_level_pred:
                continue
            pred = patient_level_pred[pid]
            gold = row['gold_standard_freq']
            if np.isfinite(pred) and np.isfinite(gold) and gold > 0:
                gold_vals.append(gold)
                pred_vals.append(pred)
        results[group] = (np.array(gold_vals), np.array(pred_vals))

    # Compute metrics
    def spearman_fn(x, y):
        r, _ = spearmanr(x, y)
        return r

    def pearson_fn(x, y):
        r, _ = pearsonr(x, y)
        return r

    def mae_fn(x, y):
        return float(np.mean(np.abs(x - y)))

    metrics = {
        'experiment': experiment_name,
        'eval_type': eval_type,
        'timestamp': time.time(),
    }

    n_total = 0
    for group in ['lpd', 'gpd', 'combined']:
        gold, pred = results[group]
        n = len(gold)

        prefix = group + '_' if group != 'combined' else 'combined_'
        if group != 'combined':
            metrics[f'n_{group}'] = n
            n_total += n

        if n < 3:
            metrics[f'{prefix}spearman'] = np.nan
            metrics[f'{prefix}spearman_ci'] = [np.nan, np.nan]
            metrics[f'{prefix}pearson'] = np.nan
            metrics[f'{prefix}mae'] = np.nan
            metrics[f'{prefix}mae_ci'] = [np.nan, np.nan]
            continue

        # Spearman with CI
        rs, rs_lo, rs_hi = bootstrap_ci(gold, pred, spearman_fn)
        metrics[f'{prefix}spearman'] = round(float(rs), 4)
        metrics[f'{prefix}spearman_ci'] = [round(float(rs_lo), 4), round(float(rs_hi), 4)]

        # Pearson
        try:
            pr, _ = pearsonr(gold, pred)
            metrics[f'{prefix}pearson'] = round(float(pr), 4)
        except Exception:
            metrics[f'{prefix}pearson'] = np.nan

        # MAE with CI
        mae, mae_lo, mae_hi = bootstrap_ci(gold, pred, mae_fn)
        metrics[f'{prefix}mae'] = round(float(mae), 4)
        metrics[f'{prefix}mae_ci'] = [round(float(mae_lo), 4), round(float(mae_hi), 4)]

        # Store raw values for scatter plots
        if group != 'combined':
            metrics[f'{group}_gold_vals'] = [round(v, 4) for v in gold.tolist()]
            metrics[f'{group}_pred_vals'] = [round(v, 4) for v in pred.tolist()]

    metrics['n_patients'] = n_total

    # Write JSON
    out_path = RUNS_DIR / f'{experiment_name}.json'

    # Custom JSON serializer for numpy types
    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    # Replace NaN with None for JSON
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            metrics_json[k] = None
        elif isinstance(v, list):
            metrics_json[k] = [None if (isinstance(x, float) and np.isnan(x)) else x for x in v]
        else:
            metrics_json[k] = v

    with open(str(out_path), 'w') as f:
        json.dump(metrics_json, f, indent=2, default=default_serializer)

    elapsed = time.time() - t0

    # Print summary table
    print(f"\n{'='*72}")
    print(f"RESULTS: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    header = f"  {'Group':>10s} {'N':>5s} {'Spearman':>10s} {'95% CI':>20s} {'Pearson':>10s} {'MAE':>8s} {'95% CI':>18s}"
    print(header)
    print(f"  {'-'*len(header.strip())}")

    for group in ['lpd', 'gpd', 'combined']:
        prefix = group + '_' if group != 'combined' else 'combined_'
        n = metrics.get(f'n_{group}', metrics.get('n_patients', 0)) if group != 'combined' else metrics.get('n_patients', 0)
        rs = metrics.get(f'{prefix}spearman', np.nan)
        ci = metrics.get(f'{prefix}spearman_ci', [np.nan, np.nan])
        pr = metrics.get(f'{prefix}pearson', np.nan)
        mae = metrics.get(f'{prefix}mae', np.nan)
        mae_ci = metrics.get(f'{prefix}mae_ci', [np.nan, np.nan])

        rs_str = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if np.isfinite(ci[0]) else "N/A"
        pr_str = f"{pr:.4f}" if np.isfinite(pr) else "N/A"
        mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        mae_ci_str = f"[{mae_ci[0]:.4f}, {mae_ci[1]:.4f}]" if np.isfinite(mae_ci[0]) else "N/A"

        label = group.upper()
        print(f"  {label:>10s} {n:>5d} {rs_str:>10s} {ci_str:>20s} {pr_str:>10s} {mae_str:>8s} {mae_ci_str:>18s}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


# ── Classification: subtype (LPD vs GPD) ─────────────────────────────

def evaluate_subtype_classification(dataset, experiment_name):
    """LOPO classification of LPD vs GPD using laterality + frequency features.

    Uses logistic regression on the feature matrix (excluding is_gpd).
    Returns metrics dict (also writes JSON).
    """
    t0 = time.time()
    print(f"\nRunning subtype classification: {experiment_name}")

    df = dataset['df']
    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    # Build subtype labels per segment
    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))
    seg_subtypes = np.array([1 if pid_to_subtype.get(p) == 'gpd' else 0 for p in seg_pids])

    # Feature columns: use all EXCEPT is_gpd (that's the target)
    is_gpd_idx = ALL_FEATURE_COLS.index('is_gpd')
    feat_mask = [i for i in range(len(ALL_FEATURE_COLS)) if i != is_gpd_idx]

    unique_patients = df['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_features[train_mask][:, feat_mask].copy()
        y_train = seg_subtypes[train_mask]
        X_test = seg_features[test_mask][:, feat_mask].copy()

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        # Add intercept
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        # Ridge logistic regression via iteratively reweighted least squares (3 iters)
        w = np.zeros(X_train_b.shape[1])
        alpha = 1.0
        for _ in range(5):
            logits = X_train_b @ w
            logits = np.clip(logits, -10, 10)
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            W_diag = p * (1 - p)
            z = logits + (y_train - p) / W_diag
            W_X = X_train_b * W_diag[:, None]
            try:
                w = np.linalg.solve(W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                                    W_X.T @ z)
            except np.linalg.LinAlgError:
                break

        test_logits = X_test_b @ w
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -10, 10)))
        patient_preds[pat] = float(np.mean(test_probs))

    # Aggregate at patient level
    y_true, y_prob = [], []
    for _, row in df.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(1 if row['subtype'] == 'gpd' else 0)
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(np.mean(y_true == y_pred))
    n = len(y_true)

    # Balanced accuracy
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (sens + spec) / 2

    # AUC (trapezoidal)
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        tpr_list, fpr_list = [0.0], [0.0]
        tp_cum, fp_cum = 0, 0
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            tpr_list.append(tp_cum / n_pos)
            fpr_list.append(fp_cum / n_neg)
        auc = float(np.trapz(tpr_list, fpr_list))
    else:
        auc = np.nan

    metrics = {
        'experiment': experiment_name,
        'task': 'subtype_classification',
        'timestamp': time.time(),
        'n_patients': n,
        'n_lpd': int(np.sum(y_true == 0)),
        'n_gpd': int(np.sum(y_true == 1)),
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'auc': round(float(auc), 4) if np.isfinite(auc) else None,
        'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
    }

    # Write JSON
    out_path = RUNS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"SUBTYPE CLASSIFICATION: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"  N={n} (LPD={metrics['n_lpd']}, GPD={metrics['n_gpd']})")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Sensitivity (GPD): {sens:.3f}")
    print(f"  Specificity (LPD): {spec:.3f}")
    print(f"  AUC:               {auc:.3f}" if np.isfinite(auc) else "  AUC: N/A")
    print(f"  Confusion: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


# ── Classification: laterality (left vs right) ──────────────────────

def evaluate_laterality_classification(dataset, experiment_name):
    """LOPO classification of laterality (left/right) for LPD patients.

    Uses logistic regression on laterality features.
    Bilateral cases are excluded (only 3 patients). Patients without
    laterality labels are excluded.
    """
    t0 = time.time()
    print(f"\nRunning laterality classification: {experiment_name}")

    df = dataset['df']

    # Filter to LPD patients with left/right laterality
    lat_map = {'left': 0, 'right': 1}
    eligible = df[df['laterality'].isin(['left', 'right'])].copy()

    if len(eligible) < 10:
        print(f"  Only {len(eligible)} eligible patients — skipping.")
        return {}

    eligible_pids = set(eligible['patient_id'].values)
    pid_to_lat = dict(zip(eligible['patient_id'], eligible['laterality'].map(lat_map)))

    seg_pids, seg_labels, seg_features, seg_arrays = _build_segment_level_data(dataset)
    seg_pids = np.array(seg_pids)

    # Use laterality features + frequency features (but NOT is_gpd since all LPD)
    lat_feat_indices = [ALL_FEATURE_COLS.index(c) for c in LATERALITY_FEATURE_COLS]
    # Also include frequency features as they may differ L vs R
    freq_feat_indices = [ALL_FEATURE_COLS.index(c) for c in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']]
    feat_indices = lat_feat_indices + freq_feat_indices

    # Filter segments to eligible patients
    eligible_mask = np.array([p in eligible_pids for p in seg_pids])
    seg_pids_e = seg_pids[eligible_mask]
    seg_features_e = seg_features[eligible_mask]
    seg_lat = np.array([pid_to_lat.get(p, -1) for p in seg_pids_e])

    unique_patients = eligible['patient_id'].values
    patient_preds = {}

    for pat in unique_patients:
        test_mask = seg_pids_e == pat
        train_mask = ~test_mask
        if np.sum(test_mask) == 0 or np.sum(train_mask) < 5:
            continue

        X_train = seg_features_e[train_mask][:, feat_indices].copy()
        y_train = seg_lat[train_mask]
        X_test = seg_features_e[test_mask][:, feat_indices].copy()

        # Impute NaN
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        # Ridge logistic regression
        w = np.zeros(X_train_b.shape[1])
        alpha = 1.0
        for _ in range(5):
            logits = X_train_b @ w
            logits = np.clip(logits, -10, 10)
            p = 1.0 / (1.0 + np.exp(-logits))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            W_diag = p * (1 - p)
            z = logits + (y_train - p) / W_diag
            W_X = X_train_b * W_diag[:, None]
            try:
                w = np.linalg.solve(W_X.T @ X_train_b + alpha * np.eye(X_train_b.shape[1]),
                                    W_X.T @ z)
            except np.linalg.LinAlgError:
                break

        test_logits = X_test_b @ w
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -10, 10)))
        patient_preds[pat] = float(np.mean(test_probs))

    # Aggregate
    y_true, y_prob = [], []
    for _, row in eligible.iterrows():
        pid = row['patient_id']
        if pid not in patient_preds:
            continue
        y_true.append(pid_to_lat[pid])
        y_prob.append(patient_preds[pid])

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = float(np.mean(y_true == y_pred))
    n = len(y_true)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # right correct
    tn = np.sum((y_true == 0) & (y_pred == 0))  # left correct
    fp = np.sum((y_true == 0) & (y_pred == 1))  # left predicted right
    fn = np.sum((y_true == 1) & (y_pred == 0))  # right predicted left

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # sensitivity for right
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # specificity (= left accuracy)
    bal_acc = (sens + spec) / 2

    # AUC
    sorted_idx = np.argsort(-y_prob)
    y_sorted = y_true[sorted_idx]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos > 0 and n_neg > 0:
        tpr_list, fpr_list = [0.0], [0.0]
        tp_cum, fp_cum = 0, 0
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            tpr_list.append(tp_cum / n_pos)
            fpr_list.append(fp_cum / n_neg)
        auc = float(np.trapz(tpr_list, fpr_list))
    else:
        auc = np.nan

    metrics = {
        'experiment': experiment_name,
        'task': 'laterality_classification',
        'timestamp': time.time(),
        'n_patients': n,
        'n_left': int(np.sum(y_true == 0)),
        'n_right': int(np.sum(y_true == 1)),
        'accuracy': round(accuracy, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'sensitivity_right': round(sens, 4),
        'specificity_left': round(spec, 4),
        'auc': round(float(auc), 4) if np.isfinite(auc) else None,
        'confusion_matrix': {'tp_right': int(tp), 'tn_left': int(tn),
                             'fp': int(fp), 'fn': int(fn)},
        'pred_probs': [round(v, 4) for v in y_prob.tolist()],
        'true_labels': y_true.tolist(),
    }

    out_path = RUNS_DIR / f'{experiment_name}.json'
    with open(str(out_path), 'w') as f:
        json.dump(metrics, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"LATERALITY CLASSIFICATION: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"  N={n} (left={metrics['n_left']}, right={metrics['n_right']})")
    print(f"  Accuracy:          {accuracy:.3f}")
    print(f"  Balanced accuracy: {bal_acc:.3f}")
    print(f"  Sens (right):      {sens:.3f}")
    print(f"  Spec (left):       {spec:.3f}")
    print(f"  AUC:               {auc:.3f}" if np.isfinite(auc) else "  AUC: N/A")
    print(f"  Confusion: TP(R)={tp} TN(L)={tn} FP={fp} FN={fn}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*72}")

    return metrics


# ── Main: run all baselines ──────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_dataset(verbose=True)

    # 1. Frequency estimation (existing)
    metrics_freq = evaluate_experiment(
        dataset,
        experiment_name='ridge_baseline_v2',
        predict_fn=ridge_predict_fn(alpha=1.0),
        eval_type='patient_lopo',
    )

    # 2. Subtype classification (LPD vs GPD)
    metrics_subtype = evaluate_subtype_classification(
        dataset,
        experiment_name='subtype_baseline',
    )

    # 3. Laterality classification (left vs right, LPD only)
    metrics_lat = evaluate_laterality_classification(
        dataset,
        experiment_name='laterality_baseline',
    )
