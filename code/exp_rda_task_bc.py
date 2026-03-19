"""
RDA Task B & C experiments: channel identification and LRDA/GRDA classification.

Task B: Channel identification methods (using VE-search frequency baseline).
Task C: LRDA vs GRDA classification methods (using VE-search freq + VE thresh=0.10).
Final: combined best of each.

Usage:
    conda run -n foe python code/exp_rda_task_bc.py
"""

import sys
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from rda_optimization_harness import (
    load_rda_dataset,
    evaluate_rda_experiment,
    variance_explained_search,
    classify_laterality,
    FS,
    LEFT_CHANNELS,
    RIGHT_CHANNELS,
    MIDLINE_CHANNELS,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _ve_search_freq_and_ve(seg):
    """Run VE search, return (best_freq, per_channel_ve)."""
    return variance_explained_search(seg)


def _narrowband_filter(seg, freq, bw=0.3):
    """Bandpass filter segment around freq +/- bw Hz. Returns (18, T)."""
    lo = max(freq - bw, 0.1)
    hi = min(freq + bw, FS / 2 - 0.1)
    if lo >= hi:
        return seg.copy()
    sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, seg, axis=1)


def _spatial_extent_from_mask(involved_mask):
    """Compute spatial extent as fraction of 18 channels involved."""
    return float(np.sum(involved_mask)) / 18.0


# ══════════════════════════════════════════════════════════════════════
# TASK B: Channel identification experiments
# All use VE-search frequency + laterality classification (defaults)
# ══════════════════════════════════════════════════════════════════════

def _make_task_b_predict_fn(channel_id_fn):
    """Factory: wraps a channel identification function into a predict_fn.

    channel_id_fn(seg, freq, per_ch_ve) -> involved_mask (bool array, len 18)
    Uses VE-search for frequency, laterality classification for subtype.
    """
    def predict_fn(train_segments, train_labels, test_segments, test_info):
        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                freq, per_ch_ve = _ve_search_freq_and_ve(seg)
                involved = channel_id_fn(seg, freq, per_ch_ve)
                spatial_ext = _spatial_extent_from_mask(involved)

                # Use per_ch_ve for laterality classification (default)
                subtype, li, _ = classify_laterality(per_ch_ve, threshold=0.3)

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


# B1: VE threshold > 5%
def _channel_id_ve_thresh(seg, freq, per_ch_ve, threshold=0.05):
    return per_ch_ve > threshold


# B4: Narrowband amplitude threshold
def _channel_id_nb_amplitude(seg, freq, per_ch_ve):
    nb = _narrowband_filter(seg, freq, bw=0.3)
    raw_std = np.std(seg, axis=1)
    nb_std = np.std(nb, axis=1)
    # Avoid division by zero
    ratio = np.where(raw_std > 1e-12, nb_std / raw_std, 0.0)
    return ratio > 0.20


# B5: Phase coherence with dominant channel
def _channel_id_coherence(seg, freq, per_ch_ve):
    nb = _narrowband_filter(seg, freq, bw=0.3)
    # Find dominant channel (highest VE)
    dom_ch = np.argmax(per_ch_ve)
    dom_signal = nb[dom_ch]
    dom_std = np.std(dom_signal)

    involved = np.zeros(18, dtype=bool)
    involved[dom_ch] = True

    for ch in range(18):
        if ch == dom_ch:
            continue
        ch_signal = nb[ch]
        ch_std = np.std(ch_signal)
        if dom_std < 1e-12 or ch_std < 1e-12:
            continue
        # Cross-correlation at zero lag (Pearson correlation)
        corr = np.corrcoef(dom_signal, ch_signal)[0, 1]
        if abs(corr) > 0.3:
            involved[ch] = True

    return involved


TASK_B_EXPERIMENTS = {
    'rda_b1_ve_thresh_05': _make_task_b_predict_fn(
        lambda seg, freq, ve: _channel_id_ve_thresh(seg, freq, ve, threshold=0.05)),
    'rda_b2_ve_thresh_15': _make_task_b_predict_fn(
        lambda seg, freq, ve: _channel_id_ve_thresh(seg, freq, ve, threshold=0.15)),
    'rda_b3_ve_thresh_20': _make_task_b_predict_fn(
        lambda seg, freq, ve: _channel_id_ve_thresh(seg, freq, ve, threshold=0.20)),
    'rda_b4_nb_amplitude': _make_task_b_predict_fn(_channel_id_nb_amplitude),
    'rda_b5_coherence': _make_task_b_predict_fn(_channel_id_coherence),
}


# ══════════════════════════════════════════════════════════════════════
# TASK C: LRDA vs GRDA classification experiments
# All use VE-search frequency + VE threshold=0.10 for channels (defaults)
# ══════════════════════════════════════════════════════════════════════

def _default_spatial_extent(per_ch_ve, ve_thresh=0.10):
    """Default spatial extent: fraction of channels with VE > 0.10."""
    return float(np.sum(per_ch_ve > ve_thresh)) / 18.0


def _make_task_c_predict_fn(classify_fn):
    """Factory: wraps a classification function into a predict_fn.

    classify_fn(seg, freq, per_ch_ve, spatial_ext, train_data) -> subtype str
    Uses VE-search for frequency, VE threshold=0.10 for spatial extent.

    For LOPO classifiers, train_data is a dict passed through.
    """
    def predict_fn(train_segments, train_labels, test_segments, test_info):
        # Pre-compute training VE vectors for methods that need them (e.g., logreg)
        train_data = {
            'segments': train_segments,
            'labels': train_labels,
        }

        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                freq, per_ch_ve = _ve_search_freq_and_ve(seg)
                spatial_ext = _default_spatial_extent(per_ch_ve)
                subtype = classify_fn(seg, freq, per_ch_ve, spatial_ext, train_data)

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


# C1-C3: Laterality index variations
def _classify_li_thresh(seg, freq, per_ch_ve, spatial_ext, train_data, threshold=0.1):
    vals = np.nan_to_num(per_ch_ve, nan=0.0)
    left_sum = np.sum(vals[LEFT_CHANNELS])
    right_sum = np.sum(vals[RIGHT_CHANNELS])
    total = left_sum + right_sum
    if total < 1e-12:
        return 'grda'
    li = (right_sum - left_sum) / total
    return 'lrda' if abs(li) > threshold else 'grda'


# C4-C5: Fraction-based classification
def _classify_fraction(seg, freq, per_ch_ve, spatial_ext, train_data, frac_thresh=0.80):
    return 'grda' if spatial_ext > frac_thresh else 'lrda'


# C6: Asymmetry of narrowband power
def _classify_nb_asymmetry(seg, freq, per_ch_ve, spatial_ext, train_data):
    vals = np.nan_to_num(per_ch_ve, nan=0.0)
    left_ve = np.sum(vals[LEFT_CHANNELS])
    right_ve = np.sum(vals[RIGHT_CHANNELS])
    total = left_ve + right_ve
    if total < 1e-12:
        return 'grda'
    dominant_fraction = max(left_ve, right_ve) / total
    return 'lrda' if dominant_fraction > 0.65 else 'grda'


# C7: Logistic regression on 18-element VE vector (LOPO CV)
class _LogregClassifier:
    """Stateful classifier that trains on first call, caches model."""

    def __init__(self):
        self._model = None
        self._train_id = None

    def classify(self, seg, freq, per_ch_ve, spatial_ext, train_data):
        # Train model if not yet trained for this fold
        train_labels = train_data['labels']
        train_segments = train_data['segments']

        # Use id of train_segments list as fold identifier
        fold_id = id(train_segments)
        if self._train_id != fold_id:
            # Compute VE vectors for all training segments
            X_train = []
            y_train = []
            for i, tseg in enumerate(train_segments):
                try:
                    _, t_ve = _ve_search_freq_and_ve(tseg)
                    X_train.append(t_ve)
                    y_train.append(1 if train_labels['subtype'][i] == 'lrda' else 0)
                except Exception:
                    continue

            if len(X_train) >= 5:
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                model = LogisticRegression(
                    C=1.0, max_iter=1000, solver='lbfgs',
                    class_weight='balanced', random_state=42
                )
                model.fit(X_train, y_train)
                self._model = model
            else:
                self._model = None
            self._train_id = fold_id

        if self._model is None:
            # Fallback to LI threshold
            return _classify_li_thresh(seg, freq, per_ch_ve, spatial_ext, train_data,
                                       threshold=0.3)

        pred = self._model.predict(per_ch_ve.reshape(1, -1))[0]
        return 'lrda' if pred == 1 else 'grda'


_logreg_classifier = _LogregClassifier()

TASK_C_EXPERIMENTS = {
    'rda_c1_li_thresh_01': _make_task_c_predict_fn(
        lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, threshold=0.1)),
    'rda_c2_li_thresh_02': _make_task_c_predict_fn(
        lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, threshold=0.2)),
    'rda_c3_li_thresh_03': _make_task_c_predict_fn(
        lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, threshold=0.3)),
    'rda_c4_fraction_80': _make_task_c_predict_fn(
        lambda s, f, v, se, td: _classify_fraction(s, f, v, se, td, frac_thresh=0.80)),
    'rda_c5_fraction_70': _make_task_c_predict_fn(
        lambda s, f, v, se, td: _classify_fraction(s, f, v, se, td, frac_thresh=0.70)),
    'rda_c6_nb_asymmetry': _make_task_c_predict_fn(_classify_nb_asymmetry),
    'rda_c7_logreg': _make_task_c_predict_fn(_logreg_classifier.classify),
}


# ══════════════════════════════════════════════════════════════════════
# COMBINED: best frequency + best channel ID + best classifier
# ══════════════════════════════════════════════════════════════════════

def _combined_predict_fn(best_channel_fn, best_classify_fn):
    """Build combined predict function from best channel ID and classifier."""
    def predict_fn(train_segments, train_labels, test_segments, test_info):
        train_data = {
            'segments': train_segments,
            'labels': train_labels,
        }

        results_freq = []
        results_se = []
        results_subtype = []

        for seg in test_segments:
            try:
                freq, per_ch_ve = _ve_search_freq_and_ve(seg)
                involved = best_channel_fn(seg, freq, per_ch_ve)
                spatial_ext = _spatial_extent_from_mask(involved)
                subtype = best_classify_fn(seg, freq, per_ch_ve, spatial_ext, train_data)

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


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    dataset = load_rda_dataset(verbose=True)

    all_results = {}

    # ── Task B experiments ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TASK B: Channel Identification Experiments")
    print("=" * 78)

    for name, predict_fn in TASK_B_EXPERIMENTS.items():
        metrics = evaluate_rda_experiment(dataset, name, predict_fn)
        all_results[name] = metrics

    # ── Task C experiments ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TASK C: LRDA vs GRDA Classification Experiments")
    print("=" * 78)

    for name, predict_fn in TASK_C_EXPERIMENTS.items():
        # Reset logreg classifier cache between experiments
        if 'logreg' in name:
            _logreg_classifier._model = None
            _logreg_classifier._train_id = None
        metrics = evaluate_rda_experiment(dataset, name, predict_fn)
        all_results[name] = metrics

    # ── Find best approaches ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("SELECTING BEST APPROACHES FOR COMBINED EXPERIMENT")
    print("=" * 78)

    # Best channel ID: highest spatial Spearman
    best_b_name = None
    best_b_spatial = -1.0
    for name, m in all_results.items():
        if name.startswith('rda_b'):
            val = m.get('spatial_spearman', np.nan)
            if np.isfinite(val) and val > best_b_spatial:
                best_b_spatial = val
                best_b_name = name
    print(f"  Best channel ID: {best_b_name} (spatial Spearman={best_b_spatial:.4f})")

    # Best classifier: highest classification F1-macro
    best_c_name = None
    best_c_f1 = -1.0
    for name, m in all_results.items():
        if name.startswith('rda_c'):
            val = m.get('classification_f1_macro', np.nan)
            if np.isfinite(val) and val > best_c_f1:
                best_c_f1 = val
                best_c_name = name
    print(f"  Best classifier: {best_c_name} (F1-macro={best_c_f1:.4f})")

    # Map best names to functions
    CHANNEL_ID_MAP = {
        'rda_b1_ve_thresh_05': lambda s, f, v: _channel_id_ve_thresh(s, f, v, 0.05),
        'rda_b2_ve_thresh_15': lambda s, f, v: _channel_id_ve_thresh(s, f, v, 0.15),
        'rda_b3_ve_thresh_20': lambda s, f, v: _channel_id_ve_thresh(s, f, v, 0.20),
        'rda_b4_nb_amplitude': _channel_id_nb_amplitude,
        'rda_b5_coherence': _channel_id_coherence,
    }

    CLASSIFY_MAP = {
        'rda_c1_li_thresh_01': lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, 0.1),
        'rda_c2_li_thresh_02': lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, 0.2),
        'rda_c3_li_thresh_03': lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, 0.3),
        'rda_c4_fraction_80': lambda s, f, v, se, td: _classify_fraction(s, f, v, se, td, 0.80),
        'rda_c5_fraction_70': lambda s, f, v, se, td: _classify_fraction(s, f, v, se, td, 0.70),
        'rda_c6_nb_asymmetry': _classify_nb_asymmetry,
        'rda_c7_logreg': _logreg_classifier.classify,
    }

    best_channel_fn = CHANNEL_ID_MAP.get(best_b_name,
        lambda s, f, v: _channel_id_ve_thresh(s, f, v, 0.05))
    best_classify_fn = CLASSIFY_MAP.get(best_c_name,
        lambda s, f, v, se, td: _classify_li_thresh(s, f, v, se, td, 0.3))

    # Reset logreg if it was chosen as best
    if best_c_name == 'rda_c7_logreg':
        _logreg_classifier._model = None
        _logreg_classifier._train_id = None

    # ── Combined experiment ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("COMBINED EXPERIMENT")
    print("=" * 78)

    combined_fn = _combined_predict_fn(best_channel_fn, best_classify_fn)
    metrics = evaluate_rda_experiment(dataset, 'rda_bc_combined', combined_fn)
    all_results['rda_bc_combined'] = metrics

    # ── Summary table ─────────────────────────────────────────────────
    print("\n\n" + "=" * 100)
    print("SUMMARY: All Task B & C Experiments")
    print("=" * 100)
    print(f"{'Experiment':<25s} {'Freq Spearman':>14s} {'Spatial Spearman':>17s} "
          f"{'Cls Accuracy':>13s} {'Cls F1-macro':>13s}")
    print("-" * 100)

    for name in list(TASK_B_EXPERIMENTS.keys()) + list(TASK_C_EXPERIMENTS.keys()) + ['rda_bc_combined']:
        m = all_results.get(name, {})
        fs = m.get('freq_combined_spearman', np.nan)
        ss = m.get('spatial_spearman', np.nan)
        ca = m.get('classification_accuracy', np.nan)
        cf = m.get('classification_f1_macro', np.nan)

        fs_s = f"{fs:.4f}" if np.isfinite(fs) else "N/A"
        ss_s = f"{ss:.4f}" if np.isfinite(ss) else "N/A"
        ca_s = f"{ca:.4f}" if np.isfinite(ca) else "N/A"
        cf_s = f"{cf:.4f}" if np.isfinite(cf) else "N/A"

        print(f"{name:<25s} {fs_s:>14s} {ss_s:>17s} {ca_s:>13s} {cf_s:>13s}")

    print("=" * 100)
    print("Done! All results saved to results/optimization_runs_v2/")
