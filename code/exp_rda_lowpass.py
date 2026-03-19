"""
Lowpass filter experiments for RDA frequency estimation.

Key insight: RDA is 0.5-3.5 Hz. High-frequency noise (muscle artifact, EMG,
line noise harmonics) contaminates frequency estimation. A lowpass filter
applied after standard preprocessing (notch 60Hz + bandpass 0.5-40Hz + detrend)
but before frequency estimation should improve accuracy.

Experiments:
  1. rda_lp_none_fft   — No lowpass (baseline FFT peak)
  2. rda_lp_5hz_fft    — Lowpass 5 Hz + FFT peak
  3. rda_lp_8hz_fft    — Lowpass 8 Hz + FFT peak
  4. rda_lp_10hz_fft   — Lowpass 10 Hz + FFT peak
  5. rda_lp_15hz_fft   — Lowpass 15 Hz + FFT peak
  6. rda_lp_5hz_ve     — Lowpass 5 Hz + VE search
  7. rda_lp_8hz_ve     — Lowpass 8 Hz + VE search
  8. rda_lp_10hz_ve    — Lowpass 10 Hz + VE search
  9. rda_lp_8hz_ridge   — Lowpass 8 Hz + Ridge (FFT+VE+ACF features)
 10. rda_lp_10hz_ridge  — Lowpass 10 Hz + Ridge

Usage:
    conda run -n foe python code/exp_rda_lowpass.py
"""

import sys
import time
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from rda_optimization_harness import (
    load_rda_dataset,
    evaluate_rda_experiment,
    fft_peak_frequency,
    variance_explained_search,
    acf_frequency,
    classify_laterality,
    FS,
)

# ── Lowpass filter utility ────────────────────────────────────────────

def apply_lowpass(seg_bi, cutoff_hz, fs=FS, order=4):
    """Apply a Butterworth lowpass filter to all channels.

    Args:
        seg_bi: (n_ch, n_samp) preprocessed bipolar segment
        cutoff_hz: lowpass cutoff in Hz
        fs: sampling rate
        order: Butterworth filter order

    Returns:
        (n_ch, n_samp) filtered segment
    """
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    filtered = np.zeros_like(seg_bi)
    for ch in range(seg_bi.shape[0]):
        filtered[ch] = filtfilt(b, a, seg_bi[ch])
    return filtered


# ── Predict functions ─────────────────────────────────────────────────

def make_fft_predict_fn(cutoff_hz=None):
    """Create a predict_fn that optionally lowpasses then uses FFT peak."""
    def predict_fn(train_segments, train_labels, test_segments, test_info):
        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                seg_use = apply_lowpass(seg, cutoff_hz) if cutoff_hz else seg
                freq = fft_peak_frequency(seg_use)

                # Use VE for spatial/laterality (on same filtered data)
                _, per_ch_ve = variance_explained_search(seg_use)
                subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)

                results_freq.append(freq)
                results_se.append(spatial_ext)
                results_subtype.append(subtype)
            except Exception:
                results_freq.append(np.nan)
                results_se.append(np.nan)
                results_subtype.append('grda')

        return {
            'freq': results_freq,
            'spatial_extent': results_se,
            'subtype': results_subtype,
        }
    return predict_fn


def make_ve_predict_fn(cutoff_hz):
    """Create a predict_fn that lowpasses then uses VE search."""
    def predict_fn(train_segments, train_labels, test_segments, test_info):
        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                seg_use = apply_lowpass(seg, cutoff_hz)
                freq, per_ch_ve = variance_explained_search(seg_use)
                subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)

                results_freq.append(freq)
                results_se.append(spatial_ext)
                results_subtype.append(subtype)
            except Exception:
                results_freq.append(np.nan)
                results_se.append(np.nan)
                results_subtype.append('grda')

        return {
            'freq': results_freq,
            'spatial_extent': results_se,
            'subtype': results_subtype,
        }
    return predict_fn


def make_ridge_predict_fn(cutoff_hz):
    """Create a predict_fn that lowpasses then uses Ridge regression on
    multi-method features (FFT peak, VE freq, ACF freq)."""
    from sklearn.linear_model import Ridge

    def predict_fn(train_segments, train_labels, test_segments, test_info):
        # Extract features from training data
        def extract_features(seg):
            seg_lp = apply_lowpass(seg, cutoff_hz)
            fft_f = fft_peak_frequency(seg_lp)
            ve_f, per_ch_ve = variance_explained_search(seg_lp)
            acf_f = acf_frequency(seg_lp)
            mean_ve = float(np.nanmean(per_ch_ve))
            max_ve = float(np.nanmax(per_ch_ve))
            return np.array([fft_f, ve_f, acf_f, mean_ve, max_ve]), per_ch_ve

        # Build training features
        X_train = []
        y_train = []
        for seg, gold_f in zip(train_segments, train_labels['freq']):
            if not np.isfinite(gold_f):
                continue
            try:
                feats, _ = extract_features(seg)
                if np.any(~np.isfinite(feats)):
                    # Replace NaN features with column mean later; for now mark
                    pass
                X_train.append(feats)
                y_train.append(gold_f)
            except Exception:
                continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Handle NaN features: replace with column mean
        for col in range(X_train.shape[1]):
            nan_mask = ~np.isfinite(X_train[:, col])
            if nan_mask.all():
                X_train[:, col] = 0.0
            elif nan_mask.any():
                col_mean = np.nanmean(X_train[:, col])
                X_train[nan_mask, col] = col_mean

        # Fit Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Column means for imputing test NaNs
        col_means = np.nanmean(X_train, axis=0)

        # Predict on test
        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                feats, per_ch_ve = extract_features(seg)
                # Impute NaNs
                for col in range(len(feats)):
                    if not np.isfinite(feats[col]):
                        feats[col] = col_means[col]

                pred_f = float(model.predict(feats.reshape(1, -1))[0])
                # Clamp to valid range
                pred_f = max(0.5, min(3.5, pred_f))

                subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
                results_freq.append(pred_f)
                results_se.append(spatial_ext)
                results_subtype.append(subtype)
            except Exception:
                results_freq.append(np.nan)
                results_se.append(np.nan)
                results_subtype.append('grda')

        return {
            'freq': results_freq,
            'spatial_extent': results_se,
            'subtype': results_subtype,
        }
    return predict_fn


# ── Define all experiments ────────────────────────────────────────────

EXPERIMENTS = [
    # FFT peak experiments
    ("rda_lp_none_fft",   make_fft_predict_fn(cutoff_hz=None)),
    ("rda_lp_5hz_fft",    make_fft_predict_fn(cutoff_hz=5.0)),
    ("rda_lp_8hz_fft",    make_fft_predict_fn(cutoff_hz=8.0)),
    ("rda_lp_10hz_fft",   make_fft_predict_fn(cutoff_hz=10.0)),
    ("rda_lp_15hz_fft",   make_fft_predict_fn(cutoff_hz=15.0)),

    # VE search experiments
    ("rda_lp_5hz_ve",     make_ve_predict_fn(cutoff_hz=5.0)),
    ("rda_lp_8hz_ve",     make_ve_predict_fn(cutoff_hz=8.0)),
    ("rda_lp_10hz_ve",    make_ve_predict_fn(cutoff_hz=10.0)),

    # Ridge multi-feature experiments
    ("rda_lp_8hz_ridge",  make_ridge_predict_fn(cutoff_hz=8.0)),
    ("rda_lp_10hz_ridge", make_ridge_predict_fn(cutoff_hz=10.0)),
]


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    dataset = load_rda_dataset(verbose=True)

    all_metrics = []
    for exp_name, predict_fn in EXPERIMENTS:
        metrics = evaluate_rda_experiment(dataset, exp_name, predict_fn)
        all_metrics.append(metrics)

    # Print summary table
    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"LOWPASS EXPERIMENT SUMMARY  ({elapsed:.0f}s total)")
    print(f"{'=' * 90}")
    print(f"{'Experiment':<25s} {'Spearman':>10s} {'95% CI':>22s} {'MAE':>8s} {'Pearson':>10s}")
    print(f"{'-' * 78}")

    for m in all_metrics:
        name = m['experiment']
        rs = m.get('freq_combined_spearman', float('nan'))
        ci = m.get('freq_combined_spearman_ci', [float('nan'), float('nan')])
        mae = m.get('freq_combined_mae', float('nan'))
        pr = m.get('freq_combined_pearson', float('nan'))

        rs_s = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        ci_s = (f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                if (isinstance(ci, list) and len(ci) == 2
                    and ci[0] is not None and np.isfinite(ci[0]))
                else "N/A")
        mae_s = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        pr_s = f"{pr:.4f}" if np.isfinite(pr) else "N/A"

        print(f"{name:<25s} {rs_s:>10s} {ci_s:>22s} {mae_s:>8s} {pr_s:>10s}")

    print(f"{'=' * 90}")

    # Update dashboard
    print("\nUpdating dashboard...")
    from update_dashboard_v2 import update
    update()

    print("\nDone.")
