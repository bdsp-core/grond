"""
ML regression experiment: predict frequency from signal processing features.

Uses leave-one-out cross-validation with Ridge, Random Forest, and Gradient Boosting.
"""

import sys
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf, compute_pointiness_trace, fcn_getBanana
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def extract_features(data, fs, entry):
    """Extract all features for one segment. Returns dict of feature values."""
    features = {}

    # --- Method A frequency ---
    try:
        result_a = pd_detect_alternate(data, fs, pk_detect='apd')
        freq_a = result_a['event_frequency']
        if freq_a is None or (isinstance(freq_a, float) and not np.isfinite(freq_a)):
            freq_a = np.nan
        else:
            freq_a = float(freq_a)
    except Exception:
        freq_a = np.nan
        result_a = {}
    features['freq_a'] = freq_a

    # --- Method B frequency ---
    try:
        result_b = pd_detect_pointiness_acf(
            data, fs,
            method='pointiness',
            lowpass_hz=15,
            smoothing_sigma=0.02,
            acf_min_lag=0.4,
            acf_peak_threshold=0.10,
            peak_height_frac=0.3,
        )
        freq_b = result_b['event_frequency']
        if freq_b is None or (isinstance(freq_b, float) and not np.isfinite(freq_b)):
            freq_b = np.nan
        else:
            freq_b = float(freq_b)
    except Exception:
        freq_b = np.nan
        result_b = {}
    features['freq_b'] = freq_b

    # --- Peak-count and IPI frequency (from pointiness trace) ---
    # We need to redo the bipolar montage + filtering to get per-channel peaks
    try:
        seg_filt = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
        seg_filt = filter_data(seg_filt, fs, 0.5, 40, n_jobs=1, verbose="ERROR")

        from pd_pointiness_acf import fcn_getBanana as getBanana_b
        seg_bp = np.array(getBanana_b(seg_filt))
        b_lp, a_lp = butter(4, 15 / (fs / 2), btype='low')
        for i in range(seg_bp.shape[0]):
            try:
                seg_bp[i] = filtfilt(b_lp, a_lp, seg_bp[i])
            except ValueError:
                pass

        n_channels = seg_bp.shape[0]
        peak_count_freqs = []
        ipi_freqs = []
        channels_with_3plus_peaks = 0

        for i in range(n_channels):
            trace = compute_pointiness_trace(seg_bp[i])
            sigma_samples = max(1, int(0.02 * fs))
            trace = gaussian_filter1d(trace, sigma=sigma_samples)
            trace_max = np.max(trace)
            peak_height = trace_max * 0.3 if trace_max > 0 else 0
            peaks, _ = find_peaks(trace, height=peak_height, distance=int(0.2 * fs))

            if len(peaks) >= 3:
                channels_with_3plus_peaks += 1

            if len(peaks) >= 2:
                time_span = (peaks[-1] - peaks[0]) / fs
                if time_span > 0:
                    pc_freq = (len(peaks) - 1) / time_span
                    peak_count_freqs.append(pc_freq)

                ipis = np.diff(peaks) / fs
                med_ipi = np.median(ipis)
                if med_ipi > 0:
                    ipi_freqs.append(1.0 / med_ipi)

        features['freq_peak_count'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan
        features['freq_ipi'] = float(np.median(ipi_freqs)) if ipi_freqs else np.nan
        features['n_channels_3plus_peaks'] = channels_with_3plus_peaks
    except Exception:
        features['freq_peak_count'] = np.nan
        features['freq_ipi'] = np.nan
        features['n_channels_3plus_peaks'] = 0

    # --- Channel-level features from Method B ---
    try:
        detected_channels = result_b.get('channels', [])
        channel_pd_scores = result_b.get('channel_pd_scores', {})
        channel_frequencies = result_b.get('channel_frequencies', {})

        features['n_acf_detected'] = len(detected_channels)

        scores = [v for v in channel_pd_scores.values() if np.isfinite(v) and v > 0]
        features['max_acf_score'] = float(np.max(scores)) if scores else 0.0

        det_scores = [channel_pd_scores.get(ch, 0) for ch in detected_channels if np.isfinite(channel_pd_scores.get(ch, 0))]
        features['mean_acf_score_detected'] = float(np.mean(det_scores)) if det_scores else 0.0

        ch_freqs = [v for v in channel_frequencies.values() if isinstance(v, (int, float)) and np.isfinite(v)]
        features['std_channel_freqs'] = float(np.std(ch_freqs)) if len(ch_freqs) >= 2 else 0.0
    except Exception:
        features['n_acf_detected'] = 0
        features['max_acf_score'] = 0.0
        features['mean_acf_score_detected'] = 0.0
        features['std_channel_freqs'] = 0.0

    # --- Pattern features ---
    features['pattern_type'] = 0.0 if entry['subdir'] == 'lpd' else 1.0

    try:
        features['spatial_extent'] = float(result_b.get('spatial_extent', 0.0))
    except Exception:
        features['spatial_extent'] = 0.0

    # --- Agreement features ---
    fa, fb, fpc = features['freq_a'], features['freq_b'], features['freq_peak_count']
    features['abs_diff_ab'] = abs(fa - fb) if (np.isfinite(fa) and np.isfinite(fb)) else np.nan

    vals = [v for v in [fa, fb, fpc] if np.isfinite(v)]
    if len(vals) >= 2:
        features['disagreement_ratio'] = max(vals) / min(vals) if min(vals) > 0 else np.nan
    else:
        features['disagreement_ratio'] = np.nan

    if len(vals) >= 1:
        features['median_abc'] = float(np.median(vals))
    else:
        features['median_abc'] = np.nan

    return features


FEATURE_NAMES = [
    'freq_a', 'freq_b', 'freq_peak_count', 'freq_ipi',
    'n_acf_detected', 'max_acf_score', 'mean_acf_score_detected',
    'std_channel_freqs', 'n_channels_3plus_peaks',
    'pattern_type', 'spatial_extent',
    'abs_diff_ab', 'disagreement_ratio', 'median_abc',
]


def features_to_array(feat_dict):
    """Convert feature dict to numpy array in canonical order."""
    return np.array([feat_dict.get(k, np.nan) for k in FEATURE_NAMES], dtype=np.float64)


def impute_nan(X, fill_value=-1):
    """Replace NaN with fill_value (column median or -1)."""
    X_out = X.copy()
    for j in range(X_out.shape[1]):
        mask = np.isnan(X_out[:, j])
        if mask.any():
            col_valid = X_out[~mask, j]
            fill = float(np.median(col_valid)) if len(col_valid) > 0 else fill_value
            X_out[mask, j] = fill
    return X_out


def loo_cv(X, y, model_factory):
    """Leave-one-out cross-validation. Returns predictions array."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_s, y_train)
        preds[i] = model.predict(X_test_s)[0]
    return preds


def loo_cv_bytype(X, y, types, model_factory):
    """LOO-CV with separate models for LPD (0) and GPD (1)."""
    n = len(y)
    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        # Train only on same-type samples
        same_type = (types == types[i]) & mask
        if same_type.sum() < 3:
            # Fall back to all training data
            same_type = mask
        X_train, y_train = X[same_type], y[same_type]
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_factory()
        model.fit(X_train_s, y_train)
        preds[i] = model.predict(X_test_s)[0]
    return preds


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset)} annotated segments")

    # Extract features
    print("\nExtracting features...")
    all_features = []
    valid_indices = []

    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  Processing segment {idx+1}/{len(dataset)}...")

        data, fs = load_eeg_data(entry)
        if data is None:
            all_features.append(None)
            continue

        try:
            feats = extract_features(data, fs, entry)
            all_features.append(feats)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  ERROR on segment {idx}: {e}")
            all_features.append(None)

    print(f"  Successfully extracted features for {len(valid_indices)} segments")

    # Build matrices
    X_list = []
    y_list = []
    mat_names = []
    pattern_types = []
    valid_dataset_indices = []

    for idx in valid_indices:
        entry = dataset[idx]
        feats = all_features[idx]
        target = entry['expert_consensus_freq']
        if not np.isfinite(target):
            continue
        X_list.append(features_to_array(feats))
        y_list.append(target)
        mat_names.append(entry['mat_name'])
        pattern_types.append(0 if entry['subdir'] == 'lpd' else 1)
        valid_dataset_indices.append(idx)

    X = np.array(X_list)
    y = np.array(y_list)
    types = np.array(pattern_types)
    print(f"  Final matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # Impute NaN
    X = impute_nan(X)

    # NaN check
    nan_per_feature = np.isnan(X).sum(axis=0)
    for i, name in enumerate(FEATURE_NAMES):
        if nan_per_feature[i] > 0:
            print(f"  WARNING: {name} still has {nan_per_feature[i]} NaN values after imputation")

    # Define models
    models = {
        'r3_ml_ridge': lambda: Ridge(alpha=1.0),
        'r3_ml_rf': lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'r3_ml_gbm': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'r3_ml_rf_deep': lambda: RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42),
    }

    best_spearman = -1
    best_model_name = None

    for name, factory in models.items():
        print(f"\n{'='*60}")
        print(f"Running LOO-CV for {name}...")
        preds = loo_cv(X, y, factory)

        predictions = {mat_names[i]: float(preds[i]) for i in range(len(preds))}
        metrics = evaluate_predictions(dataset, predictions, name)

        cs = metrics.get('combined_spearman', -1)
        if isinstance(cs, (int, float)) and np.isfinite(cs) and cs > best_spearman:
            best_spearman = cs
            best_model_name = name

    # By-type model
    print(f"\n{'='*60}")
    print("Running LOO-CV for r3_ml_rf_bytype (separate LPD/GPD models)...")
    preds_bytype = loo_cv_bytype(
        X, y, types,
        lambda: RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    )
    predictions_bytype = {mat_names[i]: float(preds_bytype[i]) for i in range(len(preds_bytype))}
    metrics_bytype = evaluate_predictions(dataset, predictions_bytype, 'r3_ml_rf_bytype')

    cs_bt = metrics_bytype.get('combined_spearman', -1)
    if isinstance(cs_bt, (int, float)) and np.isfinite(cs_bt) and cs_bt > best_spearman:
        best_spearman = cs_bt
        best_model_name = 'r3_ml_rf_bytype'

    # Feature importance for best model
    print(f"\n{'='*60}")
    print(f"Best model: {best_model_name} (combined Spearman = {best_spearman:.4f})")
    print(f"\nFeature importance (training on all data with best config):")

    # Train on all data to get feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if 'ridge' in best_model_name:
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        importances = np.abs(model.coef_)
    elif 'gbm' in best_model_name:
        model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_scaled, y)
        importances = model.feature_importances_
    else:
        # RF variant
        if 'deep' in best_model_name:
            model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_scaled, y)
        importances = model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(sorted_idx):
        print(f"  {rank+1:2d}. {FEATURE_NAMES[i]:30s}  {importances[i]:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
