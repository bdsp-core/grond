"""
RDA Optimization Harness — frequency estimation, channel identification,
and GRDA/LRDA classification via Leave-One-Patient-Out cross-validation.

Training data: 222 multi-expert RDA segments (GRDA + LRDA) from 24 patients.
Gold labels derived from median of expert raters (LB, PH, SZ, +MW if present).

Run baseline:
    conda run -n foe python code/rda_optimization_harness.py
"""

import sys
import json
import time
import hashlib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import detrend, find_peaks
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sio

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

# ── Constants ─────────────────────────────────────────────────────────
FS = 200
MAX_SEGMENTS_PER_PATIENT = 5

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
CACHE_DIR = DATA_DIR / 'rda_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = PROJECT_DIR / 'results'
RUNS_DIR = RESULTS_DIR / 'optimization_runs_v2'
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Bipolar channel labels (18 channels)
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',   # 0-3   left temporal
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',   # 4-7   right temporal
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',   # 8-11  left parasagittal
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',   # 12-15 right parasagittal
    'Fz-Cz', 'Cz-Pz',                       # 16-17 midline
]

LEFT_CHANNELS = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CHANNELS = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_CHANNELS = [16, 17]


# ── Preprocessing (with caching) ─────────────────────────────────────

def _preprocess_segment(seg_mono):
    """Preprocess a monopolar (20, 2000) segment into bipolar (18, 2000).

    Steps: monopolar-to-bipolar -> notch 60Hz -> bandpass 0.5-40Hz -> detrend.
    """
    from mne.filter import notch_filter, filter_data

    seg_bi = np.array(fcn_getBanana(seg_mono), dtype=np.float64)
    seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
    seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')
    return seg_bi


def _segment_cache_key(segment_id):
    """Hash-based cache path for a preprocessed segment."""
    return CACHE_DIR / f'{segment_id}.npy'


def _load_and_preprocess(segment_id, mat_path):
    """Load, preprocess, and cache a single segment. Returns (18, 2000) array."""
    cache_path = _segment_cache_key(segment_id)
    if cache_path.exists():
        return np.load(str(cache_path))

    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    seg_bi = _preprocess_segment(data)
    np.save(str(cache_path), seg_bi)
    return seg_bi


# ── Dataset loading ───────────────────────────────────────────────────

def load_rda_dataset(verbose=True):
    """Load RDA dataset with gold-standard labels from multi-expert annotations.

    Returns dict with:
        'df': DataFrame (patient_id, segment_id, subtype, gold_freq,
               gold_spatial_extent, gold_subtype)
        'segments': list of (18, 2000) preprocessed numpy arrays
                    (aligned with df rows)
    """
    t0 = time.time()

    df_ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))

    # Filter to RDA subtypes only
    rda_seg = df_seg[df_seg['subtype'].isin(['grda', 'lrda'])].copy()
    rda_seg_ids = set(rda_seg['segment_id'])

    # Filter annotations to RDA segments, non-skipped
    rda_ann = df_ann[
        (df_ann['segment_id'].isin(rda_seg_ids)) &
        (df_ann['skipped'] == False)
    ].copy()

    # Only keep segments with at least 2 core raters (LB, PH, SZ)
    core_ann = rda_ann[rda_ann['rater'].isin(['LB', 'PH', 'SZ'])]
    rater_counts = core_ann.groupby('segment_id')['rater'].nunique()
    multi_expert_ids = set(rater_counts[rater_counts >= 2].index)

    if verbose:
        print(f"RDA segments with >=2 core raters: {len(multi_expert_ids)}")

    # Build gold labels per segment
    records = []
    for seg_id in sorted(multi_expert_ids):
        seg_info = rda_seg[rda_seg['segment_id'] == seg_id].iloc[0]
        patient_id = seg_info['patient_id']
        subtype = seg_info['subtype']

        # Get all expert annotations for this segment
        seg_ann = rda_ann[rda_ann['segment_id'] == seg_id]

        # Gold frequency: median of expert frequencies where > 0
        freq_vals = []
        for _, row in seg_ann.iterrows():
            if row['rater'] in ['LB', 'PH', 'SZ', 'MW']:
                f = row['frequency_hz']
                if pd.notna(f) and f > 0:
                    freq_vals.append(f)
        gold_freq = float(np.median(freq_vals)) if freq_vals else np.nan

        # Gold spatial extent: median of expert values where available
        se_vals = []
        for _, row in seg_ann.iterrows():
            if row['rater'] in ['LB', 'PH', 'SZ', 'MW']:
                se = row['spatial_extent']
                if pd.notna(se):
                    se_vals.append(se)
        gold_spatial_extent = float(np.median(se_vals)) if se_vals else np.nan

        records.append({
            'patient_id': patient_id,
            'segment_id': seg_id,
            'subtype': subtype,
            'gold_freq': gold_freq,
            'gold_spatial_extent': gold_spatial_extent,
            'gold_subtype': subtype,
            'mat_file': seg_info['mat_file'],
        })

    df = pd.DataFrame(records)

    # Filter to segments with valid gold_freq
    df = df[df['gold_freq'].notna() & (df['gold_freq'] > 0)].copy()

    if verbose:
        print(f"Segments with valid gold_freq: {len(df)}")
        print(f"  GRDA: {(df['subtype'] == 'grda').sum()}, "
              f"LRDA: {(df['subtype'] == 'lrda').sum()}")
        print(f"  Patients: {df['patient_id'].nunique()}")

    # Cap at MAX_SEGMENTS_PER_PATIENT (pick highest variance)
    # First load all segments, then filter
    if verbose:
        print("Loading and preprocessing EEG segments...")

    all_segments = []
    valid_indices = []
    for idx, row in df.iterrows():
        mat_path = EEG_DIR / row['mat_file']
        if not mat_path.exists():
            continue
        try:
            seg_bi = _load_and_preprocess(row['segment_id'], mat_path)
            all_segments.append(seg_bi)
            valid_indices.append(idx)
        except Exception as e:
            if verbose:
                print(f"  Warning: failed to load {row['segment_id']}: {e}")
            continue

    df = df.loc[valid_indices].copy()
    df = df.reset_index(drop=True)

    # Assign segments to df rows
    segment_list = all_segments

    # Cap per patient: keep highest-variance segments
    keep_mask = np.ones(len(df), dtype=bool)
    for pid in df['patient_id'].unique():
        pid_indices = df.index[df['patient_id'] == pid].tolist()
        if len(pid_indices) > MAX_SEGMENTS_PER_PATIENT:
            variances = [np.var(segment_list[i]) for i in pid_indices]
            ranked = sorted(zip(pid_indices, variances),
                            key=lambda x: -x[1])
            keep_set = set(idx for idx, _ in ranked[:MAX_SEGMENTS_PER_PATIENT])
            for i in pid_indices:
                if i not in keep_set:
                    keep_mask[i] = False

    df = df[keep_mask].reset_index(drop=True)
    segment_list = [segment_list[i] for i in range(len(keep_mask)) if keep_mask[i]]

    if verbose:
        elapsed = time.time() - t0
        print(f"\nDataset loaded in {elapsed:.1f}s:")
        print(f"  Total segments: {len(df)}")
        print(f"  GRDA: {(df['subtype'] == 'grda').sum()}, "
              f"LRDA: {(df['subtype'] == 'lrda').sum()}")
        print(f"  Patients: {df['patient_id'].nunique()}")
        grda_patients = df[df['subtype'] == 'grda']['patient_id'].nunique()
        lrda_patients = df[df['subtype'] == 'lrda']['patient_id'].nunique()
        print(f"  GRDA patients: {grda_patients}, LRDA patients: {lrda_patients}")

    return {
        'df': df,
        'segments': segment_list,
    }


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
            val = metric_fn(x[idx], y[idx])
            if np.isfinite(val):
                boot_vals.append(val)
        except Exception:
            continue
    boot_vals = np.array(boot_vals)
    if len(boot_vals) == 0:
        return observed, np.nan, np.nan
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_vals, alpha * 100)
    hi = np.percentile(boot_vals, (1 - alpha) * 100)
    return observed, lo, hi


# ── Utility: Frequency estimation methods ─────────────────────────────

def variance_explained_search(seg_bi, freq_range=(0.5, 3.5), step=0.05, bw=0.3):
    """Direct search for best-fit frequency via variance explained.

    For each candidate frequency, build a sinusoidal template and compute
    the fraction of signal variance explained per channel.

    Args:
        seg_bi: (18, T) preprocessed bipolar segment
        freq_range: (lo, hi) Hz range to search
        step: frequency grid step in Hz
        bw: bandwidth for bandpass around candidate (Hz on each side)

    Returns:
        best_freq: float, best frequency in Hz
        per_channel_ve: (18,) array of variance-explained at best frequency
    """
    n_ch, n_samp = seg_bi.shape
    t = np.arange(n_samp) / FS

    freqs = np.arange(freq_range[0], freq_range[1] + step / 2, step)
    ve_matrix = np.zeros((len(freqs), n_ch))

    for fi, f in enumerate(freqs):
        # Build sinusoidal basis at frequency f (sin + cos for arbitrary phase)
        basis = np.column_stack([
            np.sin(2 * np.pi * f * t),
            np.cos(2 * np.pi * f * t),
            np.ones(n_samp),
        ])
        # Least-squares fit per channel
        for ch in range(n_ch):
            signal = seg_bi[ch]
            total_var = np.var(signal)
            if total_var < 1e-12:
                ve_matrix[fi, ch] = 0.0
                continue
            try:
                coeffs, _, _, _ = np.linalg.lstsq(basis, signal, rcond=None)
                fitted = basis @ coeffs
                residual_var = np.var(signal - fitted)
                ve_matrix[fi, ch] = max(0.0, 1.0 - residual_var / total_var)
            except np.linalg.LinAlgError:
                ve_matrix[fi, ch] = 0.0

    # Best frequency = one with highest mean VE across channels
    mean_ve = ve_matrix.mean(axis=1)
    best_idx = np.argmax(mean_ve)
    best_freq = float(freqs[best_idx])
    per_channel_ve = ve_matrix[best_idx]

    return best_freq, per_channel_ve


def acf_frequency(seg_bi, freq_range=(0.5, 3.5)):
    """Autocorrelation-based frequency estimation.

    Computes ACF of mean absolute signal, finds first peak
    in [1/fmax, 1/fmin] seconds.

    Args:
        seg_bi: (18, T) preprocessed bipolar segment
        freq_range: (lo, hi) Hz

    Returns:
        freq: float in Hz (or NaN if no peak found)
    """
    mean_abs = np.mean(np.abs(seg_bi), axis=0)
    mean_abs = mean_abs - np.mean(mean_abs)
    n = len(mean_abs)

    # Compute ACF via FFT
    fft_x = np.fft.fft(mean_abs, n=2 * n)
    acf_full = np.real(np.fft.ifft(fft_x * np.conj(fft_x)))[:n]
    if acf_full[0] > 0:
        acf_full = acf_full / acf_full[0]
    else:
        return np.nan

    # Search for first peak in expected lag range
    min_lag = int(FS / freq_range[1])  # 1/fmax in samples
    max_lag = int(FS / freq_range[0])  # 1/fmin in samples
    max_lag = min(max_lag, n - 1)

    if min_lag >= max_lag:
        return np.nan

    acf_segment = acf_full[min_lag:max_lag + 1]
    peaks, props = find_peaks(acf_segment, height=0.0)

    if len(peaks) == 0:
        return np.nan

    # Pick the highest peak
    best_peak = peaks[np.argmax(props['peak_heights'])]
    lag_samples = best_peak + min_lag
    if lag_samples == 0:
        return np.nan

    return float(FS / lag_samples)


def fft_peak_frequency(seg_bi, freq_range=(0.5, 3.5)):
    """Simple FFT peak in delta band averaged across channels.

    Args:
        seg_bi: (18, T) preprocessed bipolar segment
        freq_range: (lo, hi) Hz

    Returns:
        freq: float in Hz
    """
    n_ch, n_samp = seg_bi.shape
    # Average power spectrum across channels
    psd_sum = np.zeros(n_samp // 2 + 1)
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / FS)

    for ch in range(n_ch):
        fft_vals = np.abs(np.fft.rfft(seg_bi[ch] - np.mean(seg_bi[ch])))
        psd_sum += fft_vals ** 2

    psd_avg = psd_sum / n_ch
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask):
        return np.nan

    psd_sub = psd_avg[mask]
    freqs_sub = freqs[mask]
    if np.max(psd_sub) == 0:
        return np.nan

    return float(freqs_sub[np.argmax(psd_sub)])


def fooof_frequency(seg_bi):
    """FOOOF-based frequency estimation (Alexandra's method).

    Args:
        seg_bi: (18, T) preprocessed bipolar segment (already filtered)

    Returns:
        freq: float in Hz (or NaN)
        per_channel_scores: (18,) array of r_squared per channel (NaN if no peak)
    """
    from fooof import FOOOF

    n_ch, n_samp = seg_bi.shape
    freq_range_fit = [0.5, 3.0]
    thr_bw = 0.5

    freqs_fft = np.arange(0, FS / 2, FS / n_samp)
    channel_freqs = []
    per_channel_scores = np.full(n_ch, np.nan)

    for ch in range(n_ch):
        spectrum = np.abs(np.fft.fft(seg_bi[ch])[:n_samp // 2]) ** 2
        spectrum = spectrum / (FS * n_samp)
        if len(spectrum) > len(freqs_fft):
            spectrum = spectrum[:len(freqs_fft)]
        elif len(freqs_fft) > len(spectrum):
            freqs_fft_ch = freqs_fft[:len(spectrum)]
        else:
            freqs_fft_ch = freqs_fft

        fm = FOOOF(verbose=False)
        try:
            fm.fit(freqs_fft_ch, spectrum, freq_range_fit)
            peaks = fm.peak_params_
            if len(peaks) > 0 and peaks[0][2] <= thr_bw:
                channel_freqs.append(peaks[0][0])
                per_channel_scores[ch] = fm.r_squared_
        except Exception:
            continue

    if channel_freqs:
        freq = float(np.median(channel_freqs))
    else:
        freq = np.nan

    return freq, per_channel_scores


# ── Utility: Classification ───────────────────────────────────────────

def classify_laterality(per_channel_values, threshold=0.3):
    """Classify GRDA vs LRDA based on laterality index from per-channel values.

    Args:
        per_channel_values: (18,) array of per-channel scores (e.g., VE)
        threshold: |LI| above this -> LRDA

    Returns:
        subtype: 'grda' or 'lrda'
        laterality_index: float in [-1, 1]
        spatial_extent: float in [0, 1], fraction of channels with notable activity
    """
    vals = np.array(per_channel_values, dtype=float)
    vals = np.nan_to_num(vals, nan=0.0)

    left_sum = np.sum(vals[LEFT_CHANNELS])
    right_sum = np.sum(vals[RIGHT_CHANNELS])
    total = left_sum + right_sum

    if total < 1e-12:
        return 'grda', 0.0, 0.0

    li = (right_sum - left_sum) / total

    # Spatial extent: fraction of channels with VE above a threshold
    active_threshold = 0.05  # minimal VE to count as active
    n_active = np.sum(vals > active_threshold)
    spatial_extent = float(n_active / len(vals))

    if abs(li) > threshold:
        return 'lrda', float(li), spatial_extent
    else:
        return 'grda', float(li), spatial_extent


# ── LOPO Evaluation ───────────────────────────────────────────────────

def evaluate_rda_experiment(dataset, experiment_name, predict_fn):
    """Run Leave-One-Patient-Out CV for RDA tasks.

    Args:
        dataset: dict from load_rda_dataset()
        experiment_name: string identifier
        predict_fn: callable(train_segments, train_labels, test_segments,
                             test_info) -> dict with 'freq', 'spatial_extent', 'subtype'
            train_segments: list of (18, 2000) arrays
            train_labels: dict with 'freq' (array), 'spatial_extent' (array),
                          'subtype' (list of 'grda'/'lrda')
            test_segments: list of (18, 2000) arrays
            test_info: dict with 'patient_id', 'subtypes' (list), etc.

    Returns: metrics dict (also writes JSON)
    """
    t0 = time.time()
    print(f"\nRunning RDA LOPO experiment: {experiment_name}")

    df = dataset['df']
    segments = dataset['segments']
    patients = df['patient_id'].unique()

    # Collect per-segment predictions
    all_gold_freq = []
    all_pred_freq = []
    all_gold_se = []
    all_pred_se = []
    all_gold_subtype = []
    all_pred_subtype = []
    all_patient_ids = []

    for pat in patients:
        test_mask = df['patient_id'] == pat
        train_mask = ~test_mask

        test_df = df[test_mask]
        train_df = df[train_mask]

        test_indices = test_df.index.tolist()
        train_indices = train_df.index.tolist()

        if len(test_indices) == 0 or len(train_indices) < 3:
            continue

        train_segs = [segments[i] for i in train_indices]
        test_segs = [segments[i] for i in test_indices]

        train_labels = {
            'freq': train_df['gold_freq'].values,
            'spatial_extent': train_df['gold_spatial_extent'].values,
            'subtype': train_df['gold_subtype'].tolist(),
        }

        test_info = {
            'patient_id': pat,
            'subtypes': test_df['gold_subtype'].tolist(),
            'segment_ids': test_df['segment_id'].tolist(),
        }

        try:
            results = predict_fn(train_segs, train_labels, test_segs, test_info)
        except Exception as e:
            print(f"  Warning: predict_fn failed for patient {pat}: {e}")
            n_test = len(test_indices)
            results = {
                'freq': [np.nan] * n_test,
                'spatial_extent': [np.nan] * n_test,
                'subtype': ['grda'] * n_test,
            }

        # Ensure results are lists of per-segment predictions
        pred_freqs = np.atleast_1d(results.get('freq', np.nan))
        pred_ses = np.atleast_1d(results.get('spatial_extent', np.nan))
        pred_subtypes = results.get('subtype', ['grda'] * len(test_indices))
        if isinstance(pred_subtypes, str):
            pred_subtypes = [pred_subtypes] * len(test_indices)

        # Handle scalar returns (one prediction for all test segments)
        n_test = len(test_indices)
        if len(pred_freqs) == 1 and n_test > 1:
            pred_freqs = np.full(n_test, pred_freqs[0])
        if len(pred_ses) == 1 and n_test > 1:
            pred_ses = np.full(n_test, pred_ses[0])
        if len(pred_subtypes) == 1 and n_test > 1:
            pred_subtypes = pred_subtypes * n_test

        for i, row_idx in enumerate(test_indices):
            row = df.loc[row_idx]
            all_gold_freq.append(row['gold_freq'])
            all_pred_freq.append(float(pred_freqs[i]) if i < len(pred_freqs) else np.nan)
            all_gold_se.append(row['gold_spatial_extent'])
            all_pred_se.append(float(pred_ses[i]) if i < len(pred_ses) else np.nan)
            all_gold_subtype.append(row['gold_subtype'])
            all_pred_subtype.append(pred_subtypes[i] if i < len(pred_subtypes) else 'grda')
            all_patient_ids.append(pat)

    # Convert to arrays
    gold_freq = np.array(all_gold_freq)
    pred_freq = np.array(all_pred_freq)
    gold_se = np.array(all_gold_se)
    pred_se = np.array(all_pred_se)
    gold_subtype = np.array(all_gold_subtype)
    pred_subtype = np.array(all_pred_subtype)
    patient_ids = np.array(all_patient_ids)

    # ── Compute metrics ───────────────────────────────────────────────
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
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_segments': int(len(gold_freq)),
        'n_patients': int(len(np.unique(patient_ids))),
    }

    # -- Frequency metrics (overall and by subtype) --
    for group_name, group_mask_fn in [
        ('combined', lambda s: np.ones(len(s), dtype=bool)),
        ('grda', lambda s: s == 'grda'),
        ('lrda', lambda s: s == 'lrda'),
    ]:
        mask = group_mask_fn(gold_subtype)
        # Also require valid freq
        valid = mask & np.isfinite(gold_freq) & np.isfinite(pred_freq)
        gf = gold_freq[valid]
        pf = pred_freq[valid]
        n = len(gf)

        prefix = f'freq_{group_name}'
        metrics[f'{prefix}_n'] = int(n)

        if n >= 3:
            rs, rs_lo, rs_hi = bootstrap_ci(gf, pf, spearman_fn)
            metrics[f'{prefix}_spearman'] = round(float(rs), 4)
            metrics[f'{prefix}_spearman_ci'] = [round(float(rs_lo), 4),
                                                 round(float(rs_hi), 4)]
            try:
                pr, _ = pearsonr(gf, pf)
                metrics[f'{prefix}_pearson'] = round(float(pr), 4)
            except Exception:
                metrics[f'{prefix}_pearson'] = np.nan

            mae = mae_fn(gf, pf)
            metrics[f'{prefix}_mae'] = round(float(mae), 4)
        else:
            metrics[f'{prefix}_spearman'] = np.nan
            metrics[f'{prefix}_spearman_ci'] = [np.nan, np.nan]
            metrics[f'{prefix}_pearson'] = np.nan
            metrics[f'{prefix}_mae'] = np.nan

    # -- Spatial extent metrics --
    se_valid = np.isfinite(gold_se) & np.isfinite(pred_se)
    gs = gold_se[se_valid]
    ps = pred_se[se_valid]
    metrics['spatial_n'] = int(len(gs))

    if len(gs) >= 3:
        rs, rs_lo, rs_hi = bootstrap_ci(gs, ps, spearman_fn)
        metrics['spatial_spearman'] = round(float(rs), 4)
        metrics['spatial_spearman_ci'] = [round(float(rs_lo), 4),
                                           round(float(rs_hi), 4)]
        try:
            pr, _ = pearsonr(gs, ps)
            metrics['spatial_pearson'] = round(float(pr), 4)
        except Exception:
            metrics['spatial_pearson'] = np.nan
    else:
        metrics['spatial_spearman'] = np.nan
        metrics['spatial_spearman_ci'] = [np.nan, np.nan]
        metrics['spatial_pearson'] = np.nan

    # -- Classification metrics (GRDA vs LRDA) --
    # Map to binary: grda=0, lrda=1
    label_map = {'grda': 0, 'lrda': 1}
    gold_cls = np.array([label_map.get(s, 0) for s in gold_subtype])
    pred_cls = np.array([label_map.get(s, 0) for s in pred_subtype])

    metrics['classification_n'] = int(len(gold_cls))
    metrics['classification_n_grda'] = int(np.sum(gold_cls == 0))
    metrics['classification_n_lrda'] = int(np.sum(gold_cls == 1))

    if len(gold_cls) >= 3 and len(np.unique(gold_cls)) == 2:
        metrics['classification_accuracy'] = round(
            float(accuracy_score(gold_cls, pred_cls)), 4)
        metrics['classification_f1_macro'] = round(
            float(f1_score(gold_cls, pred_cls, average='macro')), 4)
        metrics['classification_f1_grda'] = round(
            float(f1_score(gold_cls, pred_cls, pos_label=0)), 4)
        metrics['classification_f1_lrda'] = round(
            float(f1_score(gold_cls, pred_cls, pos_label=1)), 4)
    else:
        metrics['classification_accuracy'] = np.nan
        metrics['classification_f1_macro'] = np.nan
        metrics['classification_f1_grda'] = np.nan
        metrics['classification_f1_lrda'] = np.nan

    # ── Store scatter data for dashboard ──────────────────────────────
    subtypes_arr = np.array(gold_subtype)
    for st_label, st_key in [('grda', 'grda'), ('lrda', 'lrda')]:
        mask = subtypes_arr == st_label
        valid = mask & np.isfinite(gold_freq) & np.isfinite(pred_freq)
        metrics[f'freq_{st_key}_gold_vals'] = [round(float(v), 3) for v in gold_freq[valid]]
        metrics[f'freq_{st_key}_pred_vals'] = [round(float(v), 3) for v in pred_freq[valid]]
    valid_all = np.isfinite(gold_freq) & np.isfinite(pred_freq)
    metrics['freq_combined_gold_vals'] = [round(float(v), 3) for v in gold_freq[valid_all]]
    metrics['freq_combined_pred_vals'] = [round(float(v), 3) for v in pred_freq[valid_all]]

    # ── Write JSON ────────────────────────────────────────────────────
    out_path = RUNS_DIR / f'rda_{experiment_name}.json'

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

    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            metrics_json[k] = None
        elif isinstance(v, list):
            metrics_json[k] = [
                None if (isinstance(x, float) and np.isnan(x)) else x
                for x in v
            ]
        else:
            metrics_json[k] = v

    with open(str(out_path), 'w') as f:
        json.dump(metrics_json, f, indent=2, default=default_serializer)

    elapsed = time.time() - t0

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"RESULTS: {experiment_name}  ({elapsed:.1f}s)")
    print(f"{'=' * 78}")

    print(f"\n  FREQUENCY ESTIMATION:")
    print(f"  {'Group':>10s} {'N':>5s} {'Spearman':>10s} {'95% CI':>20s} "
          f"{'Pearson':>10s} {'MAE':>8s}")
    print(f"  {'-' * 65}")
    for group in ['combined', 'grda', 'lrda']:
        prefix = f'freq_{group}'
        n = metrics.get(f'{prefix}_n', 0)
        rs = metrics.get(f'{prefix}_spearman', np.nan)
        ci = metrics.get(f'{prefix}_spearman_ci', [np.nan, np.nan])
        pr = metrics.get(f'{prefix}_pearson', np.nan)
        mae = metrics.get(f'{prefix}_mae', np.nan)

        rs_s = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        ci_s = (f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                if (isinstance(ci, list) and len(ci) == 2
                    and np.isfinite(ci[0])) else "N/A")
        pr_s = f"{pr:.4f}" if np.isfinite(pr) else "N/A"
        mae_s = f"{mae:.4f}" if np.isfinite(mae) else "N/A"

        print(f"  {group.upper():>10s} {n:>5d} {rs_s:>10s} {ci_s:>20s} "
              f"{pr_s:>10s} {mae_s:>8s}")

    print(f"\n  SPATIAL EXTENT:")
    n_se = metrics.get('spatial_n', 0)
    rs_se = metrics.get('spatial_spearman', np.nan)
    ci_se = metrics.get('spatial_spearman_ci', [np.nan, np.nan])
    pr_se = metrics.get('spatial_pearson', np.nan)
    rs_se_s = f"{rs_se:.4f}" if np.isfinite(rs_se) else "N/A"
    ci_se_s = (f"[{ci_se[0]:.4f}, {ci_se[1]:.4f}]"
               if (isinstance(ci_se, list) and len(ci_se) == 2
                   and np.isfinite(ci_se[0])) else "N/A")
    pr_se_s = f"{pr_se:.4f}" if np.isfinite(pr_se) else "N/A"
    print(f"  N={n_se}  Spearman={rs_se_s}  CI={ci_se_s}  Pearson={pr_se_s}")

    print(f"\n  CLASSIFICATION (GRDA vs LRDA):")
    acc = metrics.get('classification_accuracy', np.nan)
    f1m = metrics.get('classification_f1_macro', np.nan)
    f1g = metrics.get('classification_f1_grda', np.nan)
    f1l = metrics.get('classification_f1_lrda', np.nan)
    n_g = metrics.get('classification_n_grda', 0)
    n_l = metrics.get('classification_n_lrda', 0)
    acc_s = f"{acc:.4f}" if np.isfinite(acc) else "N/A"
    f1m_s = f"{f1m:.4f}" if np.isfinite(f1m) else "N/A"
    f1g_s = f"{f1g:.4f}" if np.isfinite(f1g) else "N/A"
    f1l_s = f"{f1l:.4f}" if np.isfinite(f1l) else "N/A"
    print(f"  N={n_g + n_l} (GRDA={n_g}, LRDA={n_l})")
    print(f"  Accuracy={acc_s}  F1-macro={f1m_s}  F1-GRDA={f1g_s}  F1-LRDA={f1l_s}")

    print(f"\n  Results saved to: {out_path}")
    print(f"{'=' * 78}")

    return metrics


# ── Baseline predict function ─────────────────────────────────────────

def baseline_predict_fn(train_segments, train_labels, test_segments, test_info):
    """Baseline: VE search for frequency, VE-based spatial extent and laterality.

    Per test segment:
      - Run variance_explained_search to get freq and per-channel VE
      - Spatial extent from fraction of channels with VE > threshold
      - GRDA/LRDA from laterality index
    """
    results_freq = []
    results_se = []
    results_subtype = []

    for seg in test_segments:
        try:
            freq, per_ch_ve = variance_explained_search(seg)
            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception as e:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')

    return {
        'freq': results_freq,
        'spatial_extent': results_se,
        'subtype': results_subtype,
    }


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_rda_dataset(verbose=True)

    metrics = evaluate_rda_experiment(
        dataset,
        experiment_name='baseline_ve_search',
        predict_fn=baseline_predict_fn,
    )
