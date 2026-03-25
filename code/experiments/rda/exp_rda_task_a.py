"""
RDA Task A: Frequency estimation experiments (10 methods).

Uses rda_optimization_harness for dataset loading and LOPO evaluation.
"""

import sys
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from rda_optimization_harness import (
    load_rda_dataset, evaluate_rda_experiment,
    variance_explained_search, acf_frequency, fft_peak_frequency,
    fooof_frequency, classify_laterality,
    LEFT_CHANNELS, RIGHT_CHANNELS, FS,
)


# ── Helper: default VE-based spatial extent and subtype ──────────────

def _ve_defaults(seg):
    """Run VE search and return freq, spatial_extent, subtype."""
    freq, per_ch_ve = variance_explained_search(seg)
    subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
    return freq, per_ch_ve, subtype, spatial_ext


def _ve_spatial_subtype(seg):
    """Return only spatial_extent and subtype from VE search."""
    _, per_ch_ve = variance_explained_search(seg)
    subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
    return per_ch_ve, subtype, spatial_ext


# ── Experiment 1: VE search baseline ─────────────────────────────────

def predict_a1_ve_search(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq, per_ch_ve, subtype, spatial_ext = _ve_defaults(seg)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 2: FOOOF ──────────────────────────────────────────────

def predict_a2_fooof(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq, per_ch_scores = fooof_frequency(seg)
            per_ch_ve, subtype, spatial_ext = _ve_spatial_subtype(seg)
            if np.isnan(freq):
                # Fallback to VE search
                freq, _, _, _ = _ve_defaults(seg)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            try:
                freq, _, subtype, spatial_ext = _ve_defaults(seg)
                results_freq.append(freq)
                results_se.append(spatial_ext)
                results_subtype.append(subtype)
            except Exception:
                results_freq.append(np.nan)
                results_se.append(np.nan)
                results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 3: ACF-based ──────────────────────────────────────────

def predict_a3_acf(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq = acf_frequency(seg)
            per_ch_ve, subtype, spatial_ext = _ve_spatial_subtype(seg)
            if np.isnan(freq):
                freq, _, _, _ = _ve_defaults(seg)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 4: FFT peak ───────────────────────────────────────────

def predict_a4_fft_peak(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq = fft_peak_frequency(seg)
            per_ch_ve, subtype, spatial_ext = _ve_spatial_subtype(seg)
            if np.isnan(freq):
                freq, _, _, _ = _ve_defaults(seg)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 5: Ridge regression on multiple features ──────────────

def predict_a5_ridge(train_segments, train_labels, test_segments, test_info):
    from sklearn.linear_model import Ridge

    def _extract_features(seg, subtype_str=None):
        """Extract frequency features for a single segment."""
        ve_freq, per_ch_ve = variance_explained_search(seg)
        acf_freq = acf_frequency(seg)
        fft_freq = fft_peak_frequency(seg)

        # Try FOOOF but don't fail if unavailable
        try:
            fooof_freq, _ = fooof_frequency(seg)
        except Exception:
            fooof_freq = np.nan

        is_grda = 1.0 if (subtype_str == 'grda') else 0.0

        feats = [
            ve_freq if np.isfinite(ve_freq) else 1.5,
            acf_freq if np.isfinite(acf_freq) else 1.5,
            fft_freq if np.isfinite(fft_freq) else 1.5,
            fooof_freq if np.isfinite(fooof_freq) else 1.5,
            is_grda,
        ]
        return np.array(feats)

    # Build training feature matrix
    train_freqs = train_labels['freq']
    train_subtypes = train_labels['subtype']
    X_train, y_train = [], []
    for i, seg in enumerate(train_segments):
        try:
            feats = _extract_features(seg, train_subtypes[i])
            log_freq = np.log(train_freqs[i]) if train_freqs[i] > 0 else np.nan
            if np.isfinite(log_freq):
                X_train.append(feats)
                y_train.append(log_freq)
        except Exception:
            continue

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if len(X_train) < 3:
        # Fallback to VE search
        return predict_a1_ve_search(train_segments, train_labels, test_segments, test_info)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    # Predict test
    test_subtypes = test_info.get('subtypes', ['grda'] * len(test_segments))
    results_freq, results_se, results_subtype = [], [], []
    for i, seg in enumerate(test_segments):
        try:
            sub = test_subtypes[i] if i < len(test_subtypes) else 'grda'
            feats = _extract_features(seg, sub)
            log_pred = model.predict(feats.reshape(1, -1))[0]
            freq = np.exp(log_pred)
            # Clamp to valid range
            freq = float(np.clip(freq, 0.5, 3.5))

            per_ch_ve, subtype, spatial_ext = _ve_spatial_subtype(seg)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')

    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 6: VE search with finer grid ──────────────────────────

def predict_a6_ve_fine(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq, per_ch_ve = variance_explained_search(seg, step=0.02)
            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 7: VE search with wider bandwidth ─────────────────────

def predict_a7_ve_wide_bw(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq, per_ch_ve = variance_explained_search(seg, bw=0.5)
            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 8: VE search with narrower bandwidth ──────────────────

def predict_a8_ve_narrow_bw(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            freq, per_ch_ve = variance_explained_search(seg, bw=0.2)
            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')
    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 9: Subtype-aware VE search ────────────────────────────

def predict_a9_ve_subtype_aware(train_segments, train_labels, test_segments, test_info):
    """If GRDA, weight all channels equally. If LRDA, use only dominant hemisphere."""
    test_subtypes = test_info.get('subtypes', ['grda'] * len(test_segments))
    results_freq, results_se, results_subtype = [], [], []

    for i, seg in enumerate(test_segments):
        try:
            sub = test_subtypes[i] if i < len(test_subtypes) else 'grda'
            freq, per_ch_ve = variance_explained_search(seg)

            if sub == 'lrda':
                # Use only the hemisphere with higher total VE
                left_ve = np.sum(per_ch_ve[LEFT_CHANNELS])
                right_ve = np.sum(per_ch_ve[RIGHT_CHANNELS])
                if left_ve >= right_ve:
                    dominant_chs = LEFT_CHANNELS
                else:
                    dominant_chs = RIGHT_CHANNELS

                # Re-run VE search weighting only dominant hemisphere channels
                # Instead, we pick the best frequency using only dominant hemisphere
                n_ch, n_samp = seg.shape
                t = np.arange(n_samp) / FS
                freqs = np.arange(0.5, 3.5 + 0.025, 0.05)
                best_ve = -1.0
                best_freq = freq  # fallback

                for f in freqs:
                    basis = np.column_stack([
                        np.sin(2 * np.pi * f * t),
                        np.cos(2 * np.pi * f * t),
                        np.ones(n_samp),
                    ])
                    ve_sum = 0.0
                    for ch in dominant_chs:
                        signal = seg[ch]
                        total_var = np.var(signal)
                        if total_var < 1e-12:
                            continue
                        coeffs, _, _, _ = np.linalg.lstsq(basis, signal, rcond=None)
                        fitted = basis @ coeffs
                        residual_var = np.var(signal - fitted)
                        ve_sum += max(0.0, 1.0 - residual_var / total_var)
                    if ve_sum > best_ve:
                        best_ve = ve_sum
                        best_freq = float(f)

                freq = best_freq

            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')

    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Experiment 10: Ensemble median of VE + ACF + FFT ─────────────────

def predict_a10_ensemble_median(train_segments, train_labels, test_segments, test_info):
    results_freq, results_se, results_subtype = [], [], []
    for seg in test_segments:
        try:
            ve_freq, per_ch_ve = variance_explained_search(seg)
            acf_freq_val = acf_frequency(seg)
            fft_freq_val = fft_peak_frequency(seg)

            candidates = [f for f in [ve_freq, acf_freq_val, fft_freq_val]
                          if np.isfinite(f)]
            freq = float(np.median(candidates)) if candidates else np.nan

            subtype, li, spatial_ext = classify_laterality(per_ch_ve, threshold=0.3)
            results_freq.append(freq)
            results_se.append(spatial_ext)
            results_subtype.append(subtype)
        except Exception:
            results_freq.append(np.nan)
            results_se.append(np.nan)
            results_subtype.append('grda')

    return {'freq': results_freq, 'spatial_extent': results_se, 'subtype': results_subtype}


# ── Main ─────────────────────────────────────────────────────────────

EXPERIMENTS = [
    ('rda_a1_ve_search',         predict_a1_ve_search),
    ('rda_a2_fooof',             predict_a2_fooof),
    ('rda_a3_acf',               predict_a3_acf),
    ('rda_a4_fft_peak',          predict_a4_fft_peak),
    ('rda_a5_ridge',             predict_a5_ridge),
    ('rda_a6_ve_fine',           predict_a6_ve_fine),
    ('rda_a7_ve_wide_bw',       predict_a7_ve_wide_bw),
    ('rda_a8_ve_narrow_bw',     predict_a8_ve_narrow_bw),
    ('rda_a9_ve_subtype_aware', predict_a9_ve_subtype_aware),
    ('rda_a10_ensemble_median', predict_a10_ensemble_median),
]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='RDA Task A: Frequency estimation experiments')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only this experiment (e.g., rda_a1_ve_search)')
    args = parser.parse_args()

    dataset = load_rda_dataset(verbose=True)

    all_metrics = {}
    for name, predict_fn in EXPERIMENTS:
        if args.only and name != args.only:
            continue
        metrics = evaluate_rda_experiment(dataset, name, predict_fn)
        all_metrics[name] = metrics

    # Summary table
    print(f"\n{'=' * 90}")
    print(f"SUMMARY: RDA Task A — Frequency Estimation")
    print(f"{'=' * 90}")
    print(f"{'Experiment':<30s} {'N':>5s} {'Spearman':>10s} {'Pearson':>10s} {'MAE':>8s}")
    print(f"{'-' * 65}")
    for name, m in all_metrics.items():
        n = m.get('freq_combined_n', 0)
        rs = m.get('freq_combined_spearman', float('nan'))
        pr = m.get('freq_combined_pearson', float('nan'))
        mae = m.get('freq_combined_mae', float('nan'))
        rs_s = f"{rs:.4f}" if np.isfinite(rs) else "N/A"
        pr_s = f"{pr:.4f}" if np.isfinite(pr) else "N/A"
        mae_s = f"{mae:.4f}" if np.isfinite(mae) else "N/A"
        print(f"{name:<30s} {n:>5d} {rs_s:>10s} {pr_s:>10s} {mae_s:>8s}")
    print(f"{'=' * 90}")
