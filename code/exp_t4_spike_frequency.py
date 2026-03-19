"""
T4: Spike-based frequency estimation experiments.

Tests a spike-interval frequency approach for low-frequency PDs where
FFT/Ridge fail due to only 2-5 discharges in a 10-second window.

Experiments:
  t4_spike_only           - f_spike standalone
  t4_ridge_with_spike_a5  - Ridge(9 feats + f_spike), alpha=5
  t4_ridge_with_spike_a10 - Ridge(9 feats + f_spike), alpha=10
  t4_ridge_spike_full_a5  - Ridge(9 feats + f_spike + n_spikes + spike_reg), alpha=5
  t4_adaptive_spike_ridge - Use f_spike when n_spikes<=5, else Ridge
  t4_weighted_blend       - Weighted average of f_spike and Ridge

Usage:
    conda run -n foe python code/exp_t4_spike_frequency.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    load_dataset, evaluate_experiment, ridge_predict_fn,
    FS, FEATURE_COLS, _build_segment_level_data,
)

# ── Spike feature computation ─────────────────────────────────────────

def compute_spike_features(segment, fs=FS):
    """Compute spike-interval frequency features from a single segment.

    Returns dict with:
        f_spike:          1 / median(IPI) if 2+ peaks, else NaN
        n_spikes:         number of detected spikes
        spike_regularity: CV of inter-peak intervals (std/mean), NaN if <3 peaks
    """
    seg = np.asarray(segment, dtype=np.float64)
    n_channels = seg.shape[0]

    # Lowpass filter to clean up
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg[ch])
        except ValueError:
            seg_lp[ch] = seg[ch]

    # Max absolute amplitude across channels at each time point
    max_abs = np.max(np.abs(seg_lp), axis=0)

    # Find prominent peaks
    height_threshold = np.percentile(max_abs, 75)
    min_distance = int(0.3 * fs)  # minimum 0.3s between peaks

    peaks, _ = find_peaks(max_abs, height=height_threshold, distance=min_distance)
    n_spikes = len(peaks)

    result = {
        'f_spike': np.nan,
        'n_spikes': float(n_spikes),
        'spike_regularity': np.nan,
    }

    if n_spikes <= 1:
        return result

    # Inter-peak intervals in seconds
    ipis = np.diff(peaks) / fs

    if 2 <= n_spikes <= 5:
        # Sparse case: use median IPI
        median_ipi = np.median(ipis)
        if median_ipi > 0:
            result['f_spike'] = 1.0 / median_ipi
    elif n_spikes >= 6:
        # Dense case: try both median-IPI and span-based, take the one
        # closer to the FFT band (more likely correct for higher freq)
        median_ipi = np.median(ipis)
        f_median = 1.0 / median_ipi if median_ipi > 0 else np.nan

        span_s = (peaks[-1] - peaks[0]) / fs
        f_span = (n_spikes - 1) / span_s if span_s > 0 else np.nan

        # Use median-IPI approach (more robust to outliers) but clip to [0.1, 5]
        if np.isfinite(f_median):
            result['f_spike'] = np.clip(f_median, 0.1, 5.0)
        elif np.isfinite(f_span):
            result['f_spike'] = np.clip(f_span, 0.1, 5.0)

    # Spike regularity (CV of IPIs)
    if n_spikes >= 3 and len(ipis) >= 2:
        mean_ipi = np.mean(ipis)
        if mean_ipi > 0:
            result['spike_regularity'] = float(np.std(ipis) / mean_ipi)

    return result


# ── Ridge with pre-trained model (9 features) ─────────────────────────

def _compute_9feat_vector(feat_dict_6):
    """Expand 6-feature dict to 9-feature vector matching the saved Ridge model."""
    f_fft = feat_dict_6.get('f_fft', np.nan)
    f_tkeo = feat_dict_6.get('f_tkeo', np.nan)
    f_B = feat_dict_6.get('f_B', np.nan)

    f_fft_v = f_fft if np.isfinite(f_fft) else 0
    f_tkeo_v = f_tkeo if np.isfinite(f_tkeo) else 0
    f_B_v = f_B if np.isfinite(f_B) else 0

    return np.array([
        feat_dict_6.get('f_B', np.nan),
        feat_dict_6.get('f_peaks', np.nan),
        feat_dict_6.get('f_fft', np.nan),
        feat_dict_6.get('f_tkeo', np.nan),
        feat_dict_6.get('f_coh', np.nan),
        feat_dict_6.get('is_gpd', 0.0),
        f_fft_v * f_tkeo_v,
        f_fft_v * f_B_v,
        f_tkeo_v * f_B_v,
    ])


def _predict_ridge_pretrained(test_segments, test_features_6col):
    """Use the pre-trained Ridge model (9 features) to predict frequencies.

    This doesn't need training data — it uses the saved model weights.
    """
    from discharge_timing import _load_ridge_model

    w, feature_cols, medians = _load_ridge_model()

    preds = []
    for i in range(test_features_6col.shape[0]):
        feat_dict = {c: test_features_6col[i, j] for j, c in enumerate(FEATURE_COLS)}
        x = _compute_9feat_vector(feat_dict)

        # Impute NaN with saved medians
        for j, col in enumerate(feature_cols):
            if not np.isfinite(x[j]):
                x[j] = medians.get(col, 0.0)

        x_b = np.append(x, 1.0)
        pred_log = np.clip(x_b @ w, np.log(0.1), np.log(10.0))
        preds.append(float(np.exp(pred_log)))

    return np.array(preds)


# ── Predict functions for each experiment ──────────────────────────────

def spike_only_predict_fn():
    """Experiment 1: f_spike standalone."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        preds = []
        for seg in test_segments:
            if seg is None:
                preds.append(np.nan)
                continue
            spike_feats = compute_spike_features(seg)
            f = spike_feats['f_spike']
            preds.append(f if np.isfinite(f) else np.nan)
        return np.array(preds)
    return _predict


def ridge_with_spike_predict_fn(alpha=5.0):
    """Experiment 2: Ridge on 9 SP features + f_spike (10 total)."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # Build expanded feature matrices (9 SP + f_spike = 10 features)
        def _expand(segs, feats_6col):
            rows = []
            for i in range(feats_6col.shape[0]):
                feat_dict = {c: feats_6col[i, j] for j, c in enumerate(FEATURE_COLS)}
                base = _compute_9feat_vector(feat_dict)
                seg = segs[i] if i < len(segs) else None
                if seg is not None:
                    sf = compute_spike_features(seg)
                    spike_val = sf['f_spike']
                else:
                    spike_val = np.nan
                rows.append(np.append(base, spike_val))
            return np.array(rows)

        X_train = _expand(train_segments, train_features)
        X_test = _expand(test_segments, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        # Impute NaN with training median
        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        # Ridge regression
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.nan)

    return _predict


def ridge_spike_full_predict_fn(alpha=5.0):
    """Experiment 3: Ridge on 9 SP features + f_spike + n_spikes + spike_regularity (12 total)."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        def _expand(segs, feats_6col):
            rows = []
            for i in range(feats_6col.shape[0]):
                feat_dict = {c: feats_6col[i, j] for j, c in enumerate(FEATURE_COLS)}
                base = _compute_9feat_vector(feat_dict)
                seg = segs[i] if i < len(segs) else None
                if seg is not None:
                    sf = compute_spike_features(seg)
                    extras = [sf['f_spike'], sf['n_spikes'], sf['spike_regularity']]
                else:
                    extras = [np.nan, np.nan, np.nan]
                rows.append(np.concatenate([base, extras]))
            return np.array(rows)

        X_train = _expand(train_segments, train_features)
        X_test = _expand(test_segments, test_features)
        y_train = np.log(np.clip(train_labels, 0.05, 100.0))

        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            X_test[~np.isfinite(X_test[:, j]), j] = med

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])
        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            return np.exp(pred_log)
        except np.linalg.LinAlgError:
            return np.full(X_test.shape[0], np.nan)

    return _predict


def adaptive_spike_ridge_predict_fn():
    """Experiment 4: Use f_spike when n_spikes<=5, else use pre-trained Ridge."""
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        # Get Ridge predictions for all test segments
        ridge_preds = _predict_ridge_pretrained(test_segments, test_features)

        preds = []
        for i, seg in enumerate(test_segments):
            if seg is None:
                preds.append(ridge_preds[i])
                continue
            sf = compute_spike_features(seg)
            n_spikes = sf['n_spikes']
            f_spike = sf['f_spike']

            if n_spikes <= 5 and np.isfinite(f_spike):
                preds.append(f_spike)
            else:
                preds.append(ridge_preds[i])
        return np.array(preds)
    return _predict


def weighted_blend_predict_fn():
    """Experiment 5: Weighted blend of f_spike and Ridge.

    w = max(0, 1 - n_spikes/10)
    freq = w * f_spike + (1-w) * f_ridge
    """
    def _predict(train_segments, train_labels, train_features,
                 test_segments, test_features):
        ridge_preds = _predict_ridge_pretrained(test_segments, test_features)

        preds = []
        for i, seg in enumerate(test_segments):
            if seg is None:
                preds.append(ridge_preds[i])
                continue
            sf = compute_spike_features(seg)
            n_spikes = sf['n_spikes']
            f_spike = sf['f_spike']

            if np.isfinite(f_spike):
                w = max(0.0, 1.0 - n_spikes / 10.0)
                preds.append(w * f_spike + (1.0 - w) * ridge_preds[i])
            else:
                preds.append(ridge_preds[i])
        return np.array(preds)
    return _predict


# ── Low-frequency breakdown analysis ──────────────────────────────────

def frequency_bin_analysis(dataset):
    """Print Spearman correlation breakdown by gold-standard frequency bin.

    Bins: <0.5 Hz, 0.5-1.0 Hz, >1.0 Hz
    Methods: f_spike_only, ridge_pretrained, adaptive, weighted_blend
    """
    print("\n" + "=" * 80)
    print("LOW-FREQUENCY BREAKDOWN ANALYSIS")
    print("=" * 80)

    df = dataset['df']
    segments = dataset['segments']
    features = dataset['features']

    # Collect per-patient predictions for each method
    methods = {}

    for _, row in df.iterrows():
        pid = row['patient_id']
        gold = row['gold_standard_freq']
        pat_segs = segments.get(pid, [])
        pat_feats = features.get(pid, [])

        if not pat_segs or not pat_feats:
            continue

        # Build 6-col feature matrix for this patient's segments
        feats_6 = np.array([[fd.get(c, np.nan) for c in FEATURE_COLS]
                            for fd in pat_feats])

        # f_spike predictions
        spike_preds = []
        spike_feats_list = []
        for seg in pat_segs:
            sf = compute_spike_features(seg)
            spike_feats_list.append(sf)
            if np.isfinite(sf['f_spike']):
                spike_preds.append(sf['f_spike'])

        f_spike_avg = float(np.mean(spike_preds)) if spike_preds else np.nan

        # Ridge pretrained predictions
        ridge_preds = _predict_ridge_pretrained(pat_segs, feats_6)
        ridge_avg = float(np.nanmean(ridge_preds))

        # Adaptive
        adaptive_preds = []
        for i, seg in enumerate(pat_segs):
            sf = spike_feats_list[i]
            if sf['n_spikes'] <= 5 and np.isfinite(sf['f_spike']):
                adaptive_preds.append(sf['f_spike'])
            else:
                adaptive_preds.append(ridge_preds[i])
        adaptive_avg = float(np.nanmean(adaptive_preds))

        # Weighted blend
        blend_preds = []
        for i, seg in enumerate(pat_segs):
            sf = spike_feats_list[i]
            if np.isfinite(sf['f_spike']):
                w = max(0.0, 1.0 - sf['n_spikes'] / 10.0)
                blend_preds.append(w * sf['f_spike'] + (1.0 - w) * ridge_preds[i])
            else:
                blend_preds.append(ridge_preds[i])
        blend_avg = float(np.nanmean(blend_preds))

        methods.setdefault('spike_only', []).append((gold, f_spike_avg))
        methods.setdefault('ridge_pretrained', []).append((gold, ridge_avg))
        methods.setdefault('adaptive', []).append((gold, adaptive_avg))
        methods.setdefault('weighted_blend', []).append((gold, blend_avg))

    bins = [
        ('<0.5 Hz', lambda g: g < 0.5),
        ('0.5-1.0 Hz', lambda g: 0.5 <= g <= 1.0),
        ('>1.0 Hz', lambda g: g > 1.0),
        ('All', lambda g: True),
    ]

    header = f"  {'Method':<22s}"
    for bin_name, _ in bins:
        header += f" {bin_name:>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method_name in ['spike_only', 'ridge_pretrained', 'adaptive', 'weighted_blend']:
        pairs = methods[method_name]
        row_str = f"  {method_name:<22s}"
        for bin_name, bin_fn in bins:
            bin_pairs = [(g, p) for g, p in pairs
                         if bin_fn(g) and np.isfinite(g) and np.isfinite(p)]
            if len(bin_pairs) >= 3:
                golds = np.array([g for g, p in bin_pairs])
                preds = np.array([p for g, p in bin_pairs])
                rs, _ = spearmanr(golds, preds)
                row_str += f"  {rs:.3f} (n={len(bin_pairs):>2d})"
            else:
                n = len(bin_pairs)
                row_str += f"    N/A (n={n:>2d})"
        print(row_str)

    print("=" * 80)


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()

    print("Loading dataset...")
    dataset = load_dataset(verbose=True)

    experiments = [
        ('t4_spike_only', spike_only_predict_fn()),
        ('t4_ridge_with_spike_a5', ridge_with_spike_predict_fn(alpha=5.0)),
        ('t4_ridge_with_spike_a10', ridge_with_spike_predict_fn(alpha=10.0)),
        ('t4_ridge_spike_full_a5', ridge_spike_full_predict_fn(alpha=5.0)),
        ('t4_adaptive_spike_ridge', adaptive_spike_ridge_predict_fn()),
        ('t4_weighted_blend', weighted_blend_predict_fn()),
    ]

    all_metrics = {}
    for name, pred_fn in experiments:
        metrics = evaluate_experiment(
            dataset,
            experiment_name=name,
            predict_fn=pred_fn,
            eval_type='patient_lopo',
        )
        all_metrics[name] = metrics

    # Low-frequency breakdown
    frequency_bin_analysis(dataset)

    # Summary table
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"T4 SPIKE FREQUENCY EXPERIMENTS SUMMARY  ({elapsed:.0f}s total)")
    print(f"{'=' * 80}")
    print(f"  {'Experiment':<32s} {'Spearman':>10s} {'95% CI':>20s} {'MAE':>8s}")
    print(f"  {'-' * 72}")
    for name, m in all_metrics.items():
        rs = m.get('combined_spearman', np.nan)
        ci = m.get('combined_spearman_ci', [np.nan, np.nan])
        mae = m.get('combined_mae', np.nan)
        rs_s = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if np.isfinite(ci[0]) else "N/A"
        mae_s = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        print(f"  {name:<32s} {rs_s:>10s} {ci_s:>20s} {mae_s:>8s}")
    print(f"{'=' * 80}")

    # Update dashboard
    print("\nUpdating dashboard...")
    from update_dashboard_v2 import update
    update()

    print("\nDone.")
