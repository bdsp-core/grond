"""
Round 6: Mega Combiner — all frequency estimators + ridge regression.

Computes EVERY available frequency estimator per segment, then trains
multiple ridge regression models with LOO-CV.
"""

import sys
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    pd_detect_pointiness_acf, bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate
from scipy.signal import butter, filtfilt, find_peaks, coherence
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import eigh
from mne.filter import notch_filter, filter_data

# ── Constants ─────────────────────────────────────────────────────────
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3
FS = 200

REPO_ROOT = os.path.dirname(CODE_DIR)
TEMPLATES_C_LPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_lpd.npy')
TEMPLATES_C_GPD_PATH = os.path.join(REPO_ROOT, 'data', 'templates_C_gpd.npy')

try:
    TEMPLATES_C_LPD = np.load(TEMPLATES_C_LPD_PATH)
    TEMPLATES_C_GPD = np.load(TEMPLATES_C_GPD_PATH)
    print(f"Loaded LPD templates: {TEMPLATES_C_LPD.shape}, GPD templates: {TEMPLATES_C_GPD.shape}")
except Exception as e:
    print(f"Warning: Could not load templates: {e}")
    TEMPLATES_C_LPD = None
    TEMPLATES_C_GPD = None

# Adjacent channel pairs for spectral coherence
ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


# ── Preprocessing ─────────────────────────────────────────────────────
def preprocess_segment(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    seg = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(seg))
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    for i in range(seg.shape[0]):
        try:
            seg[i] = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            pass
    return seg


def median_finite(arr):
    valid = arr[np.isfinite(arr)] if isinstance(arr, np.ndarray) else \
        np.array([x for x in arr if np.isfinite(x)])
    return float(np.median(valid)) if len(valid) > 0 else np.nan


# ── Feature extractors ────────────────────────────────────────────────
def get_f_A(data, fs):
    """Method A: pd_detect_alternate(apd) event_frequency."""
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def get_f_B(data, fs):
    """Method B: pd_detect_pointiness_acf with acf_peak_threshold=0.10."""
    try:
        r = pd_detect_pointiness_acf(
            data, fs, method='pointiness',
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            smoothing_sigma=SMOOTHING_SIGMA,
            lowpass_hz=LOWPASS_HZ,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        f = r.get('event_frequency', np.nan)
        n_ch = 0
        try:
            ch = r.get('channels', None)
            if ch is not None and hasattr(ch, '__len__'):
                n_ch = len(ch)
        except Exception:
            pass
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan, n_ch
        return float(f), n_ch
    except Exception:
        return np.nan, 0


def compute_pointiness_traces(seg, fs):
    """Compute smoothed pointiness trace per channel. Returns (n_ch, n_samples)."""
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    traces = []
    for i in range(seg.shape[0]):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        traces.append(trace)
    return np.array(traces)


def get_f_peaks(traces, fs):
    """Peak-count on pointiness (height=max*0.3, distance=0.2*fs). Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = traces[i]
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return median_finite(freqs)


def get_f_fft(traces, fs):
    """FFT of pointiness, peak in [0.3, 3.5] Hz. Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_envelope(seg, fs, subdir):
    """Matched-filter envelope FFT (Bank C). Median across channels."""
    templates = TEMPLATES_C_LPD if subdir == 'lpd' else TEMPLATES_C_GPD
    if templates is None:
        return np.nan
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        ch = seg[i]
        n_samples = len(ch)
        envelope = np.zeros(n_samples)
        for t_idx in range(templates.shape[0]):
            tmpl = templates[t_idx]
            if len(tmpl) > n_samples:
                tmpl = tmpl[:n_samples]
            corr = np.abs(np.correlate(ch, tmpl, mode='same'))
            envelope = np.maximum(envelope, corr)
        if np.max(envelope) <= 0:
            continue
        n = len(envelope)
        fft_vals = np.abs(np.fft.rfft(envelope - np.mean(envelope)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_tkeo_fft(seg, fs):
    """FFT of |TKEO| trace. TKEO = x^2(n) - x(n-1)*x(n+1). Median across channels."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(0.02 * fs))
    for i in range(n_ch):
        x = seg[i]
        if len(x) < 3:
            continue
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
        tkeo = np.abs(tkeo)
        tkeo = gaussian_filter1d(tkeo, sigma=sigma_samples)
        n = len(tkeo)
        if n < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(tkeo - np.mean(tkeo)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        freqs[i] = freq_sub[np.argmax(fft_sub)]
    return median_finite(freqs)


def get_f_hps3(traces, fs):
    """Harmonic Product Spectrum on pointiness FFT. HPS(f)=P(f)*P(2f)*P(3f). Median across channels."""
    n_ch = traces.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        trace = traces[i]
        if np.max(trace) <= 0:
            continue
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        # HPS: downsample spectrum by factors 2 and 3
        max_idx = len(fft_vals) // 3
        if max_idx < 2:
            continue
        hps = fft_vals[:max_idx].copy()
        hps *= fft_vals[:max_idx * 2:2][:max_idx]
        hps *= fft_vals[:max_idx * 3:3][:max_idx]
        hps_freqs = fft_freqs[:max_idx]
        mask = (hps_freqs >= 0.3) & (hps_freqs <= 3.5)
        if not np.any(mask):
            continue
        hps_sub = hps[mask]
        freq_sub = hps_freqs[mask]
        freqs[i] = freq_sub[np.argmax(hps_sub)]
    return median_finite(freqs)


def get_f_spectral_coh(seg, fs):
    """Spectral coherence peak across adjacent channel pairs."""
    nperseg = min(256, seg.shape[1] // 2)
    if nperseg < 16:
        return np.nan
    coh_spectra = []
    coh_freqs = None
    for (a, b) in ADJACENT_PAIRS:
        if a >= seg.shape[0] or b >= seg.shape[0]:
            continue
        try:
            f_coh, Cxy = coherence(seg[a], seg[b], fs=fs, nperseg=nperseg)
            if coh_freqs is None:
                coh_freqs = f_coh
                coh_spectra = np.zeros_like(f_coh)
            coh_spectra = coh_spectra + Cxy
        except Exception:
            continue
    if coh_freqs is None:
        return np.nan
    coh_spectra /= len(ADJACENT_PAIRS)
    mask = (coh_freqs >= 0.3) & (coh_freqs <= 3.5)
    if not np.any(mask):
        return np.nan
    coh_sub = coh_spectra[mask]
    freq_sub = coh_freqs[mask]
    return float(freq_sub[np.argmax(coh_sub)])


def get_f_ged(seg, seg_broadband, fs, rough_freq):
    """GED-based frequency estimate."""
    try:
        if not np.isfinite(rough_freq) or rough_freq <= 0:
            rough_freq = 1.0
        lo = max(0.2, rough_freq - 0.3)
        hi = min(4.0, rough_freq + 0.3)
        if lo >= hi:
            return np.nan
        # Bandpass for narrowband
        b_bp, a_bp = butter(2, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        X_narrow = np.zeros_like(seg_broadband)
        for i in range(seg_broadband.shape[0]):
            try:
                X_narrow[i] = filtfilt(b_bp, a_bp, seg_broadband[i])
            except Exception:
                X_narrow[i] = seg_broadband[i]
        # Covariance matrices
        S_freq = X_narrow @ X_narrow.T / X_narrow.shape[1]
        S_wide = seg_broadband @ seg_broadband.T / seg_broadband.shape[1]
        reg = 0.01 * np.eye(S_wide.shape[0])
        # GED
        eigenvalues, eigenvectors = eigh(S_freq, S_wide + reg)
        # Top eigenvector (last one = largest eigenvalue)
        w = eigenvectors[:, -1]
        # Project
        filtered = w @ seg_broadband
        # 15Hz lowpass
        b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
        try:
            filtered = filtfilt(b_lp, a_lp, filtered)
        except Exception:
            pass
        # Pointiness + FFT
        trace = compute_pointiness_trace(filtered)
        sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        if np.max(trace) <= 0:
            return np.nan
        n = len(trace)
        fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= 0.3) & (fft_freqs <= 3.5)
        if not np.any(mask):
            return np.nan
        fft_sub = fft_vals[mask]
        freq_sub = fft_freqs[mask]
        return float(freq_sub[np.argmax(fft_sub)])
    except Exception:
        return np.nan


# ── Ridge regression LOO-CV ───────────────────────────────────────────
def ridge_loo_cv(X, y, alpha=1.0):
    """Leave-one-out ridge regression. Returns predictions array."""
    n = X.shape[0]
    predictions = np.full(n, np.nan)
    # Precompute: center features
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        y_train = y[mask]
        X_test = X[i:i+1]
        # Standardize
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_tr_s = (X_train - mu) / std
        X_te_s = (X_test - mu) / std
        y_mu = np.mean(y_train)
        y_c = y_train - y_mu
        # Ridge closed form: w = (X^T X + alpha I)^-1 X^T y
        p = X_tr_s.shape[1]
        A = X_tr_s.T @ X_tr_s + alpha * np.eye(p)
        try:
            w = np.linalg.solve(A, X_tr_s.T @ y_c)
        except np.linalg.LinAlgError:
            continue
        pred = X_te_s @ w + y_mu
        predictions[i] = pred[0]
    return predictions


def ridge_full_fit(X, y, alpha=1.0):
    """Fit ridge on all data, return coefficients and intercept."""
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_s = (X - mu) / std
    y_mu = np.mean(y)
    y_c = y - y_mu
    p = X_s.shape[1]
    A = X_s.T @ X_s + alpha * np.eye(p)
    w = np.linalg.solve(A, X_s.T @ y_c)
    return w, mu, std, y_mu


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Feature names
    FEATURE_NAMES = [
        'f_A', 'f_B', 'f_peaks', 'f_fft', 'f_envelope',
        'f_tkeo_fft', 'f_hps3', 'f_spectral_coh', 'f_ged',
        'is_gpd', 'n_ch',
    ]

    N = len(dataset)
    features = np.full((N, len(FEATURE_NAMES)), np.nan)
    expert_freqs = np.full(N, np.nan)
    mat_names = []
    subdirs = []

    t0 = time.time()
    for idx, entry in enumerate(dataset):
        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  Processing segment {idx+1}/{N}  ({elapsed:.0f}s elapsed)")

        mat_names.append(entry['mat_name'])
        subdirs.append(entry['subdir'])
        expert_freqs[idx] = entry['expert_consensus_freq']

        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        # ─ Method A (does its own preprocessing) ──────────────
        f_A, n_ch_A = get_f_A(data, fs)
        features[idx, 0] = f_A

        # ─ Method B (does its own preprocessing) ─────────────
        f_B, n_ch_B = get_f_B(data, fs)
        features[idx, 1] = f_B
        features[idx, 10] = n_ch_B  # n_ch

        # ─ Standard preprocessing for remaining features ──────
        seg = preprocess_segment(data, fs)

        # Also need broadband bipolar (no 15Hz LP) for GED
        seg_bb = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
        seg_bb = filter_data(seg_bb, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        seg_bb = np.array(fcn_getBanana(seg_bb))

        # Pointiness traces (reused by f_peaks, f_fft, f_hps3)
        traces = compute_pointiness_traces(seg, fs)

        # ─ f_peaks ────────────────────────────────────────────
        features[idx, 2] = get_f_peaks(traces, fs)

        # ─ f_fft ──────────────────────────────────────────────
        features[idx, 3] = get_f_fft(traces, fs)

        # ─ f_envelope ─────────────────────────────────────────
        features[idx, 4] = get_f_envelope(seg, fs, entry['subdir'])

        # ─ f_tkeo_fft ─────────────────────────────────────────
        features[idx, 5] = get_f_tkeo_fft(seg, fs)

        # ─ f_hps3 ─────────────────────────────────────────────
        features[idx, 6] = get_f_hps3(traces, fs)

        # ─ f_spectral_coh ─────────────────────────────────────
        features[idx, 7] = get_f_spectral_coh(seg_bb, fs)

        # ─ f_ged ──────────────────────────────────────────────
        rough_freq = f_B if np.isfinite(f_B) else 1.0
        features[idx, 8] = get_f_ged(seg, seg_bb, fs, rough_freq)

        # ─ is_gpd ─────────────────────────────────────────────
        features[idx, 9] = 1.0 if entry['subdir'] == 'gpd' else 0.0

    elapsed = time.time() - t0
    print(f"Feature extraction done in {elapsed:.0f}s")

    # ── Print feature coverage ────────────────────────────────────────
    print("\nFeature coverage (non-NaN counts):")
    for fi, fname in enumerate(FEATURE_NAMES):
        cnt = np.sum(np.isfinite(features[:, fi]))
        print(f"  {fname:>15s}: {cnt}/{N}")

    # ── Helper: impute NaNs with median, build valid mask ─────────────
    def prepare_features(feat_indices, require_target=True):
        """Select features, impute NaN with column median, return X, y, valid_mask."""
        X_raw = features[:, feat_indices].copy()
        y_raw = expert_freqs.copy()
        # Valid = target is finite and at least one feature is finite
        valid = np.isfinite(y_raw)
        for fi in range(X_raw.shape[1]):
            valid &= np.isfinite(X_raw[:, fi]) | True  # we impute NaN
        if require_target:
            valid &= np.isfinite(y_raw)
        # Impute NaN features with column median (over valid rows)
        X = X_raw[valid].copy()
        y = y_raw[valid].copy()
        for fi in range(X.shape[1]):
            col = X[:, fi]
            nan_mask = ~np.isfinite(col)
            if np.any(nan_mask):
                med = np.nanmedian(col)
                if not np.isfinite(med):
                    med = 1.0
                col[nan_mask] = med
                X[:, fi] = col
        return X, y, valid

    # ── Experiment definitions ────────────────────────────────────────
    experiments = {}

    # (a) r6_mega_ridge_8feat: [f_A, f_B, f_peaks, f_fft, f_envelope, f_tkeo_fft, f_hps3, is_gpd]
    idx_8 = [0, 1, 2, 3, 4, 5, 6, 9]
    experiments['r6_mega_ridge_8feat'] = {'feat_idx': idx_8, 'alpha': 1.0, 'log_target': True}

    # (b) r6_mega_ridge_10feat: add f_spectral_coh + f_ged
    idx_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    experiments['r6_mega_ridge_10feat'] = {'feat_idx': idx_10, 'alpha': 1.0, 'log_target': True}

    # (c) r6_mega_ridge_all: all features including n_ch
    idx_all = list(range(len(FEATURE_NAMES)))
    experiments['r6_mega_ridge_all'] = {'feat_idx': idx_all, 'alpha': 1.0, 'log_target': True}

    # (e) r6_mega_ridge_alpha10: same as (c) but alpha=10
    experiments['r6_mega_ridge_alpha10'] = {'feat_idx': idx_all, 'alpha': 10.0, 'log_target': True}

    # (f) r6_mega_no_A: same as (c) but drop f_A
    idx_no_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    experiments['r6_mega_no_A'] = {'feat_idx': idx_no_A, 'alpha': 1.0, 'log_target': True}

    # Run standard ridge experiments
    best_name = None
    best_mae = 999.0

    for exp_name, cfg in experiments.items():
        print(f"\n{'='*60}")
        print(f"Running {exp_name}...")
        X, y, valid = prepare_features(cfg['feat_idx'])
        if cfg['log_target']:
            y_t = np.log(y)
        else:
            y_t = y.copy()

        preds_t = ridge_loo_cv(X, y_t, alpha=cfg['alpha'])
        if cfg['log_target']:
            preds = np.exp(preds_t)
        else:
            preds = preds_t

        # Build predictions dict
        pred_dict = {}
        valid_idx = np.where(valid)[0]
        for i, vi in enumerate(valid_idx):
            if np.isfinite(preds[i]):
                pred_dict[mat_names[vi]] = float(preds[i])

        metrics = evaluate_predictions(dataset, pred_dict, exp_name)
        cmae = metrics.get('combined_mae', 999)
        if np.isfinite(cmae) and cmae < best_mae:
            best_mae = cmae
            best_name = exp_name
            best_cfg = cfg

    # (d) r6_mega_ridge_bytype: separate models for LPD and GPD
    print(f"\n{'='*60}")
    print(f"Running r6_mega_ridge_bytype...")
    X_all, y_all, valid = prepare_features(idx_all)
    y_log = np.log(y_all)
    valid_idx = np.where(valid)[0]
    subdirs_valid = [subdirs[vi] for vi in valid_idx]

    pred_dict_bytype = {}
    for ptype in ['lpd', 'gpd']:
        type_mask = np.array([s == ptype for s in subdirs_valid])
        if np.sum(type_mask) < 5:
            continue
        X_type = X_all[type_mask]
        y_type = y_log[type_mask]
        preds_type = ridge_loo_cv(X_type, y_type, alpha=1.0)
        preds_type = np.exp(preds_type)
        type_indices = np.where(type_mask)[0]
        for i, ti in enumerate(type_indices):
            vi = valid_idx[ti]
            if np.isfinite(preds_type[i]):
                pred_dict_bytype[mat_names[vi]] = float(preds_type[i])

    metrics_bytype = evaluate_predictions(dataset, pred_dict_bytype, 'r6_mega_ridge_bytype')
    cmae = metrics_bytype.get('combined_mae', 999)
    if np.isfinite(cmae) and cmae < best_mae:
        best_mae = cmae
        best_name = 'r6_mega_ridge_bytype'

    # (g) r6_tkeo_hps_avg: average of f_tkeo_fft and f_hps3
    print(f"\n{'='*60}")
    print("Running r6_tkeo_hps_avg...")
    pred_dict_th = {}
    for idx_i in range(N):
        f_tk = features[idx_i, 5]
        f_hp = features[idx_i, 6]
        vals = [v for v in [f_tk, f_hp] if np.isfinite(v)]
        if len(vals) > 0:
            pred_dict_th[mat_names[idx_i]] = float(np.mean(vals))
    evaluate_predictions(dataset, pred_dict_th, 'r6_tkeo_hps_avg')

    # (h) r6_best_six_median: median of [f_A, f_B, f_peaks, f_fft, f_tkeo_fft, f_hps3]
    print(f"\n{'='*60}")
    print("Running r6_best_six_median...")
    pred_dict_6m = {}
    six_idx = [0, 1, 2, 3, 5, 6]
    for idx_i in range(N):
        vals = [features[idx_i, fi] for fi in six_idx if np.isfinite(features[idx_i, fi])]
        if len(vals) > 0:
            pred_dict_6m[mat_names[idx_i]] = float(np.median(vals))
    evaluate_predictions(dataset, pred_dict_6m, 'r6_best_six_median')

    # ── Print feature coefficients for the best model ─────────────────
    if best_name and best_name in experiments:
        cfg = experiments[best_name]
        X, y, valid = prepare_features(cfg['feat_idx'])
        y_t = np.log(y) if cfg['log_target'] else y.copy()
        w, mu, std, y_mu = ridge_full_fit(X, y_t, alpha=cfg['alpha'])
        feat_names_used = [FEATURE_NAMES[i] for i in cfg['feat_idx']]
        print(f"\n{'='*60}")
        print(f"Best model: {best_name} (combined MAE = {best_mae:.4f})")
        print(f"Feature coefficients (standardized):")
        for fname, coef in sorted(zip(feat_names_used, w), key=lambda x: -abs(x[1])):
            print(f"  {fname:>15s}: {coef:+.4f}")
        print(f"  {'intercept':>15s}: {y_mu:.4f}")
    elif best_name == 'r6_mega_ridge_bytype':
        print(f"\nBest model: r6_mega_ridge_bytype (combined MAE = {best_mae:.4f})")
        # Print coefficients per type
        for ptype in ['lpd', 'gpd']:
            type_mask = np.array([s == ptype for s in subdirs_valid])
            if np.sum(type_mask) < 5:
                continue
            X_type = X_all[type_mask]
            y_type = y_log[type_mask]
            w, mu, std, y_mu = ridge_full_fit(X_type, y_type, alpha=1.0)
            print(f"\n  {ptype.upper()} coefficients (standardized):")
            for fname, coef in sorted(zip(FEATURE_NAMES, w), key=lambda x: -abs(x[1])):
                print(f"    {fname:>15s}: {coef:+.4f}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == '__main__':
    main()
