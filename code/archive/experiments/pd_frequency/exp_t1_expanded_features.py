"""
Experiment T1: Expanded feature sets beyond the 6 base SP features.

New features:
  1. f_hps: Harmonic Product Spectrum on pointiness trace
  2. f_fft_car: FFT on pointiness of CAR-referenced data
  3. n_detected: Number of channels with valid ACF periodicity
  4. f_range: Range of per-channel FFT frequency estimates
  5. signal_variance: Log of median channel variance

Experiments:
  - t1_ridge_base_a1: 6 base features, alpha=1.0
  - t1_ridge_base_a5: 6 base features, alpha=5.0
  - t1_ridge_expanded_a1: 11 features, alpha=1.0
  - t1_ridge_expanded_a5: 11 features, alpha=5.0
  - t1_ridge_expanded_a10: 11 features, alpha=10.0
"""

import sys
import time
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, ridge_predict_fn,
    compute_sp_features, _compute_fft_peak, _build_segment_level_data,
    FS, FREQ_LO, FREQ_HI, LOWPASS_HZ, SMOOTHING_SIGMA,
    ACF_MIN_LAG, ACF_THRESHOLD, PEAK_HEIGHT_FRAC, FEATURE_COLS,
)
from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d


# ── Expanded feature computation ──────────────────────────────────────

def compute_expanded_features(segment, base_features):
    """Compute the 5 additional features for a single segment.

    Args:
        segment: (18, 2000) bipolar segment
        base_features: dict with the 6 base features already computed

    Returns:
        dict with keys: f_hps, f_fft_car, n_detected, f_range, signal_variance
    """
    fs = FS
    seg_bip = np.asarray(segment, dtype=np.float64)
    n_channels, n_samples = seg_bip.shape

    extra = {}

    # Lowpass filter (same as harness)
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # Pointiness traces (recompute - needed for HPS and other features)
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # ── 1. f_hps: Harmonic Product Spectrum ───────────────────────────
    # Average pointiness across channels, then HPS
    avg_pt = np.mean(pointiness_traces, axis=0)
    avg_pt = avg_pt - np.mean(avg_pt)
    n_pt = len(avg_pt)
    fft_vals = np.abs(np.fft.rfft(avg_pt))
    freqs = np.fft.rfftfreq(n_pt, d=1.0 / fs)

    # HPS: multiply downsampled spectra P(f)*P(2f)*P(3f)
    max_len = len(fft_vals)
    hps = fft_vals[:max_len // 3].copy()
    for harmonic in [2, 3]:
        downsampled = fft_vals[::harmonic][:len(hps)]
        hps[:len(downsampled)] *= downsampled

    hps_freqs = freqs[:len(hps)]
    mask = (hps_freqs >= FREQ_LO) & (hps_freqs <= FREQ_HI)
    if np.any(mask) and np.max(hps[mask]) > 0:
        extra['f_hps'] = float(hps_freqs[mask][np.argmax(hps[mask])])
    else:
        extra['f_hps'] = np.nan

    # ── 2. f_fft_car: FFT on CAR-referenced pointiness ───────────────
    # CAR = each channel minus mean of all channels
    car_mean = np.mean(seg_bip, axis=0)
    seg_car = seg_bip - car_mean[np.newaxis, :]

    # Lowpass the CAR data
    seg_car_lp = np.zeros_like(seg_car)
    for ch in range(n_channels):
        try:
            seg_car_lp[ch] = filtfilt(b_lp, a_lp, seg_car[ch])
        except ValueError:
            seg_car_lp[ch] = seg_car[ch]

    # Compute pointiness on CAR, then FFT
    car_fft_freqs = []
    for ch in range(n_channels):
        pt_car = compute_pointiness_trace(seg_car_lp[ch])
        pt_car = gaussian_filter1d(pt_car, sigma=sigma_samples)
        f = _compute_fft_peak(pt_car, fs)
        if np.isfinite(f):
            car_fft_freqs.append(f)
    extra['f_fft_car'] = float(np.median(car_fft_freqs)) if car_fft_freqs else np.nan

    # ── 3. n_detected: Number of channels with valid ACF periodicity ─
    n_valid = 0
    per_channel_freqs = []
    for ch in range(n_channels):
        freq, score, _ = compute_acf_frequency(
            seg_lp[ch], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        if np.isfinite(freq):
            n_valid += 1
            per_channel_freqs.append(freq)
    extra['n_detected'] = float(n_valid)

    # ── 4. f_range: Range of per-channel FFT frequency estimates ──────
    fft_freqs_per_ch = []
    for ch in range(n_channels):
        f = _compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs_per_ch.append(f)
    if len(fft_freqs_per_ch) >= 2:
        extra['f_range'] = float(np.max(fft_freqs_per_ch) - np.min(fft_freqs_per_ch))
    else:
        extra['f_range'] = np.nan

    # ── 5. signal_variance: Log of median channel variance ────────────
    ch_vars = [np.var(seg_bip[ch]) for ch in range(n_channels)]
    med_var = np.median(ch_vars)
    extra['signal_variance'] = float(np.log(med_var + 1e-10))

    return extra


# ── Build expanded dataset ────────────────────────────────────────────

EXPANDED_FEATURE_COLS = FEATURE_COLS + [
    'f_hps', 'f_fft_car', 'n_detected', 'f_range', 'signal_variance'
]


def add_expanded_features(dataset):
    """Compute and attach expanded features to each segment in the dataset.

    Modifies dataset['features'] in-place by adding keys to each feature dict.
    """
    features = dataset['features']
    segments = dataset['segments']

    n_computed = 0
    n_total = 0
    for pid in features:
        pat_feats = features[pid]
        pat_segs = segments.get(pid, [])
        for i, feat_dict in enumerate(pat_feats):
            n_total += 1
            if i < len(pat_segs) and pat_segs[i] is not None:
                try:
                    extra = compute_expanded_features(pat_segs[i], feat_dict)
                    feat_dict.update(extra)
                    n_computed += 1
                except Exception as e:
                    # Fill with NaN on failure
                    for key in ['f_hps', 'f_fft_car', 'n_detected', 'f_range', 'signal_variance']:
                        feat_dict[key] = np.nan
            else:
                for key in ['f_hps', 'f_fft_car', 'n_detected', 'f_range', 'signal_variance']:
                    feat_dict[key] = np.nan

    print(f"  Expanded features computed for {n_computed}/{n_total} segments")


def expanded_ridge_predict_fn(alpha=1.0):
    """Ridge predict_fn that uses the 11 expanded features."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # train_features and test_features are (N, 6) from the harness
        # But we stored expanded features in the dataset, so we need to
        # reconstruct. Instead, we'll use a wrapper approach.
        # This won't work directly since the harness only passes 6 cols.
        # We need a different approach - see below.
        raise NotImplementedError("Use custom_evaluate instead")
    return _predict


def custom_evaluate(dataset, experiment_name, alpha, feature_cols, eval_type='patient_lopo'):
    """Custom LOPO evaluation that uses arbitrary feature columns.

    Like evaluate_experiment but builds feature matrix from the specified columns.
    """
    df = dataset['df']
    features = dataset['features']
    segments = dataset['segments']

    # Build segment-level data with custom feature columns
    seg_pids = []
    seg_labels = []
    seg_feat_rows = []
    seg_arrays = []

    for _, row in df.iterrows():
        pid = row['patient_id']
        gold = row['gold_standard_freq']
        pat_feats = features.get(pid, [])
        pat_segs = segments.get(pid, [])

        for i, feat_dict in enumerate(pat_feats):
            seg_pids.append(pid)
            seg_labels.append(gold)
            seg_feat_rows.append([feat_dict.get(c, np.nan) for c in feature_cols])
            if i < len(pat_segs):
                seg_arrays.append(pat_segs[i])
            else:
                seg_arrays.append(None)

    n_feats = len(feature_cols)
    seg_features = np.array(seg_feat_rows, dtype=float) if seg_feat_rows else np.empty((0, n_feats))
    seg_labels_arr = np.array(seg_labels, dtype=float)
    seg_pids_arr = np.array(seg_pids)

    # Build the ridge predict_fn with the right dimensionality
    predict_fn = ridge_predict_fn(alpha=alpha)

    # Now call evaluate_experiment by temporarily swapping the feature matrix
    # Actually, we need to run LOPO ourselves since the harness uses FEATURE_COLS
    return evaluate_experiment(
        dataset, experiment_name,
        predict_fn=_make_ridge_for_cols(alpha, feature_cols),
        eval_type=eval_type,
    )


def _make_ridge_for_cols(alpha, feature_cols):
    """Create a predict_fn that pulls from the feature dicts using custom cols.

    The harness passes (N, 6) feature arrays built from FEATURE_COLS.
    We can't change that. Instead, we'll monkey-patch the dataset.
    Actually - the simplest approach: temporarily change FEATURE_COLS globally.
    """
    # We need a predict_fn that works with whatever feature matrix the harness builds.
    # The harness builds features using FEATURE_COLS. We'll modify the harness module
    # temporarily. Let's just do manual LOPO instead.
    return ridge_predict_fn(alpha=alpha)


def run_lopo_custom(dataset, experiment_name, alpha, feature_cols):
    """Run LOPO with custom feature columns, then use harness for metrics/output."""
    import optimization_harness_v2 as harness

    # Temporarily override FEATURE_COLS in the harness module
    original_cols = harness.FEATURE_COLS
    harness.FEATURE_COLS = list(feature_cols)

    try:
        metrics = evaluate_experiment(
            dataset,
            experiment_name=experiment_name,
            predict_fn=ridge_predict_fn(alpha=alpha),
            eval_type='patient_lopo',
        )
    finally:
        harness.FEATURE_COLS = original_cols

    return metrics


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    # Load dataset
    dataset = load_dataset(verbose=True)

    # Compute expanded features for all segments
    print("\nComputing expanded features...")
    t_feat = time.time()
    add_expanded_features(dataset)
    print(f"  Done in {time.time() - t_feat:.1f}s")

    # Print a sample of expanded features
    sample_pid = list(dataset['features'].keys())[0]
    sample_feat = dataset['features'][sample_pid][0]
    print(f"\n  Sample features for {sample_pid}:")
    for k in EXPANDED_FEATURE_COLS:
        v = sample_feat.get(k, 'MISSING')
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    all_results = {}

    # ── Experiment 1: Base features, alpha=1.0 ────────────────────────
    metrics = run_lopo_custom(dataset, 't1_ridge_base_a1', alpha=1.0,
                              feature_cols=FEATURE_COLS)
    all_results['t1_ridge_base_a1'] = metrics

    # ── Experiment 2: Base features, alpha=5.0 ────────────────────────
    metrics = run_lopo_custom(dataset, 't1_ridge_base_a5', alpha=5.0,
                              feature_cols=FEATURE_COLS)
    all_results['t1_ridge_base_a5'] = metrics

    # ── Experiment 3: Expanded features, alpha=1.0 ────────────────────
    metrics = run_lopo_custom(dataset, 't1_ridge_expanded_a1', alpha=1.0,
                              feature_cols=EXPANDED_FEATURE_COLS)
    all_results['t1_ridge_expanded_a1'] = metrics

    # ── Experiment 4: Expanded features, alpha=5.0 ────────────────────
    metrics = run_lopo_custom(dataset, 't1_ridge_expanded_a5', alpha=5.0,
                              feature_cols=EXPANDED_FEATURE_COLS)
    all_results['t1_ridge_expanded_a5'] = metrics

    # ── Experiment 5: Expanded features, alpha=10.0 ───────────────────
    metrics = run_lopo_custom(dataset, 't1_ridge_expanded_a10', alpha=10.0,
                              feature_cols=EXPANDED_FEATURE_COLS)
    all_results['t1_ridge_expanded_a10'] = metrics

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY: All T1 Experiments")
    print(f"{'='*72}")
    print(f"  {'Experiment':<30s} {'Features':>8s} {'Alpha':>6s} {'Spearman':>10s} {'MAE':>8s}")
    print(f"  {'-'*66}")

    for name, m in all_results.items():
        n_feat = '6' if 'base' in name else '11'
        alpha_str = name.split('_a')[-1]
        sp = m.get('combined_spearman', float('nan'))
        mae = m.get('combined_mae', float('nan'))
        sp_str = f"{sp:.4f}" if np.isfinite(sp) else "N/A"
        mae_str = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        print(f"  {name:<30s} {n_feat:>8s} {alpha_str:>6s} {sp_str:>10s} {mae_str:>8s}")

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'='*72}")
