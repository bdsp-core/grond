"""
Round 5 experiment: Expanded template banks and better matched-filter envelopes.

Builds multiple template banks from annotated data (with more permissive
selection: expert freq std < 0.3), then evaluates matched-filter envelope FFT
frequency estimation across all banks and combinations.

Template Banks:
  Bank C16:  k-means k=16 per type (16 LPD + 16 GPD)
  Bank C24:  k-means k=24 per type (24 LPD + 24 GPD)
  Bank PCA5: first 5 PCs per type
  Bank PCA10: first 10 PCs per type
  Bank Multi: Bank C16 + Bank D (synthetic) = 24 templates per type

Variants:
  r5_bank16_fft:           Bank C16 envelope FFT, median across channels
  r5_bank24_fft:           Bank C24 envelope FFT
  r5_pca5_fft:             Bank PCA5 envelope FFT
  r5_pca10_fft:            Bank PCA10 envelope FFT
  r5_multi_fft:            Bank Multi envelope FFT
  r5_bank16_plus_peaks:    average of Bank C16 envelope FFT + peak-count freq
  r5_best_bank_ridge:      Ridge regression on log(freq) using multiple features
"""

import sys
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, os.path.join(CODE_DIR, 'pd_detector_alternate'))

from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_pointiness_acf import (
    fcn_getBanana, compute_pointiness_trace, compute_acf_frequency,
    bipolar_channels,
)
from pd_detect_alternate import pd_detect_alternate
from browse_results import detect_pd_peaks

# ── Constants ─────────────────────────────────────────────────────────
FS = 200
WINDOW_SAMPLES = 50   # 250ms at 200Hz
HALF_WIN = WINDOW_SAMPLES // 2
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3

REPO_ROOT = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, 'data')

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


# ── Step 1: Build expanded template banks ─────────────────────────────
def extract_snippets(dataset, max_std=0.3, min_score=1.0):
    """Extract discharge snippets from annotated segments.

    Select segments with expert freq std < max_std (more permissive).
    Use Method A peak detection on channels with score > min_score.
    Extract 250ms (50 sample) z-scored windows centered on peaks.
    """
    # Select segments with sufficient expert agreement
    selected = []
    for entry in dataset:
        freqs = []
        for key in ['expert_LB_freq', 'expert_PH_freq', 'expert_SZ_freq']:
            v = entry.get(key, np.nan)
            if np.isfinite(v) and v > 0:
                freqs.append(v)
        if len(freqs) < 2:
            continue
        if np.std(freqs) < max_std:
            selected.append(entry)

    print(f"Selected {len(selected)} segments with expert freq std < {max_std}")

    lpd_snippets = []
    gpd_snippets = []

    for idx, entry in enumerate(selected):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        # Preprocess: notch 60Hz, bandpass 0.5-40Hz
        data_filt = notch_filter(data.astype(np.float64), fs, 60, n_jobs=1, verbose='ERROR')
        data_filt = filter_data(data_filt, fs, 0.5, 40, n_jobs=1, verbose='ERROR')

        # Bipolar montage
        try:
            bipolar = np.array(fcn_getBanana(data_filt))
        except Exception:
            continue

        # 15Hz lowpass
        b_lp, a_lp = butter(4, 15.0 / (fs / 2.0), btype='low')
        for ch in range(bipolar.shape[0]):
            try:
                bipolar[ch] = filtfilt(b_lp, a_lp, bipolar[ch])
            except ValueError:
                pass

        # Run Method A to get channel PD scores
        try:
            result = pd_detect_alternate(data, fs)
        except Exception:
            continue

        channel_scores = result.get('channel_pd_scores', {})

        # For channels with score > min_score, detect peaks
        for ch_idx, ch_name in enumerate(bipolar_channels):
            if ch_idx >= bipolar.shape[0]:
                break
            score = channel_scores.get(ch_name, 0)
            if not np.isfinite(score) or score <= min_score:
                continue

            peaks = detect_pd_peaks(bipolar[ch_idx], fs=fs)
            if len(peaks) == 0:
                continue

            for pk in peaks:
                start = pk - HALF_WIN
                end = pk + HALF_WIN
                if start < 0 or end > bipolar.shape[1]:
                    continue
                snippet = bipolar[ch_idx, start:end].copy()
                if len(snippet) != WINDOW_SAMPLES:
                    continue
                s = np.std(snippet)
                if s < 1e-10:
                    continue
                snippet = (snippet - np.mean(snippet)) / s

                if entry['subdir'] == 'lpd':
                    lpd_snippets.append(snippet)
                else:
                    gpd_snippets.append(snippet)

        if (idx + 1) % 50 == 0:
            print(f"  Snippet extraction: {idx + 1}/{len(selected)} segments "
                  f"(LPD: {len(lpd_snippets)}, GPD: {len(gpd_snippets)})")

    print(f"Total snippets: LPD={len(lpd_snippets)}, GPD={len(gpd_snippets)}")
    return np.array(lpd_snippets) if lpd_snippets else np.zeros((0, WINDOW_SAMPLES)), \
           np.array(gpd_snippets) if gpd_snippets else np.zeros((0, WINDOW_SAMPLES))


def cluster_snippets(X, k, label):
    """Cluster snippets into k groups and return z-scored centroids."""
    if len(X) == 0:
        print(f"  WARNING: No {label} snippets to cluster, using zeros")
        return np.zeros((k, WINDOW_SAMPLES))

    print(f"  Clustering {len(X)} {label} snippets into {k} clusters...")
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(k, len(X)), n_init=10, random_state=42)
        km.fit(X)
        centroids = km.cluster_centers_
    except ImportError:
        from scipy.cluster.vq import kmeans2
        centroids, _ = kmeans2(X, min(k, len(X)), minit='points')

    # Z-score normalize each centroid
    for i in range(centroids.shape[0]):
        s = np.std(centroids[i])
        if s > 1e-10:
            centroids[i] = (centroids[i] - np.mean(centroids[i])) / s
    return centroids


def pca_templates(X, n_components, label):
    """Use first n_components principal components as templates."""
    if len(X) < n_components:
        print(f"  WARNING: Only {len(X)} {label} snippets for PCA, using all")
        n_components = max(1, len(X))

    print(f"  PCA on {len(X)} {label} snippets, keeping {n_components} components...")
    X_centered = X - np.mean(X, axis=0)
    # SVD-based PCA
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    components = Vt[:n_components]

    # Z-score normalize each component
    for i in range(components.shape[0]):
        s = np.std(components[i])
        if s > 1e-10:
            components[i] = (components[i] - np.mean(components[i])) / s
    return components


def build_all_banks(dataset):
    """Build all template banks and return dict of banks."""
    print("=" * 60)
    print("STEP 1: Extracting discharge snippets (std < 0.3)")
    print("=" * 60)

    lpd_snips, gpd_snips = extract_snippets(dataset, max_std=0.3, min_score=1.0)

    print("\n" + "=" * 60)
    print("STEP 2: Building template banks")
    print("=" * 60)

    banks = {}

    # Bank C16: k=16
    print("\n--- Bank C16 (k-means k=16) ---")
    c16_lpd = cluster_snippets(lpd_snips, 16, 'LPD')
    c16_gpd = cluster_snippets(gpd_snips, 16, 'GPD')
    banks['C16'] = {'lpd': c16_lpd, 'gpd': c16_gpd}
    print(f"  C16: LPD={c16_lpd.shape}, GPD={c16_gpd.shape}")

    # Bank C24: k=24
    print("\n--- Bank C24 (k-means k=24) ---")
    c24_lpd = cluster_snippets(lpd_snips, 24, 'LPD')
    c24_gpd = cluster_snippets(gpd_snips, 24, 'GPD')
    banks['C24'] = {'lpd': c24_lpd, 'gpd': c24_gpd}
    print(f"  C24: LPD={c24_lpd.shape}, GPD={c24_gpd.shape}")

    # Bank PCA5: first 5 PCs
    print("\n--- Bank PCA5 (first 5 PCs) ---")
    pca5_lpd = pca_templates(lpd_snips, 5, 'LPD')
    pca5_gpd = pca_templates(gpd_snips, 5, 'GPD')
    banks['PCA5'] = {'lpd': pca5_lpd, 'gpd': pca5_gpd}
    print(f"  PCA5: LPD={pca5_lpd.shape}, GPD={pca5_gpd.shape}")

    # Bank PCA10: first 10 PCs
    print("\n--- Bank PCA10 (first 10 PCs) ---")
    pca10_lpd = pca_templates(lpd_snips, 10, 'LPD')
    pca10_gpd = pca_templates(gpd_snips, 10, 'GPD')
    banks['PCA10'] = {'lpd': pca10_lpd, 'gpd': pca10_gpd}
    print(f"  PCA10: LPD={pca10_lpd.shape}, GPD={pca10_gpd.shape}")

    # Bank Multi: C16 + Bank D (synthetic)
    print("\n--- Bank Multi (C16 + Bank D) ---")
    templates_D = np.load(os.path.join(DATA_DIR, 'templates_D.npy'))
    print(f"  Loaded Bank D: {templates_D.shape}")
    multi_lpd = np.vstack([c16_lpd, templates_D])
    multi_gpd = np.vstack([c16_gpd, templates_D])
    banks['Multi'] = {'lpd': multi_lpd, 'gpd': multi_gpd}
    print(f"  Multi: LPD={multi_lpd.shape}, GPD={multi_gpd.shape}")

    return banks


# ── Matched-filter envelope ───────────────────────────────────────────
def compute_discharge_envelope(seg, templates):
    """Cross-correlate each channel with each template, max across templates."""
    n_ch, n_t = seg.shape
    n_templates, template_len = templates.shape
    envelope = np.zeros((n_ch, n_t))

    for ch in range(n_ch):
        channel = seg[ch]
        std = np.std(channel)
        if std < 1e-10:
            continue
        channel_normed = (channel - np.mean(channel)) / std

        correlations = np.zeros((n_templates, n_t))
        for ti in range(n_templates):
            tmpl = templates[ti]
            tmpl_std = np.std(tmpl)
            if tmpl_std < 1e-10:
                continue
            tmpl_normed = (tmpl - np.mean(tmpl)) / tmpl_std / template_len
            corr = np.correlate(channel_normed, tmpl_normed, mode='same')
            correlations[ti] = corr

        envelope[ch] = np.max(correlations, axis=0)
    return envelope


def envelope_fft_per_channel(envelope, fs, fmin=0.3, fmax=3.5):
    """FFT of envelope per channel, peak in [fmin, fmax] Hz."""
    n_ch, n_t = envelope.shape
    freqs_out = np.full(n_ch, np.nan)

    for ch in range(n_ch):
        env_ch = envelope[ch]
        if np.std(env_ch) < 1e-10:
            continue
        n = len(env_ch)
        fft_vals = np.abs(np.fft.rfft(env_ch - np.mean(env_ch))) ** 2
        fft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (fft_freqs >= fmin) & (fft_freqs <= fmax)
        if not np.any(mask):
            continue
        power_range = fft_vals[mask]
        freq_range = fft_freqs[mask]
        peak_idx = np.argmax(power_range)
        peak_power = power_range[peak_idx]
        mean_power = np.mean(power_range)
        if mean_power > 0 and peak_power > 2.0 * mean_power:
            freqs_out[ch] = freq_range[peak_idx]

    return freqs_out


def median_finite(arr):
    valid = [x for x in arr if np.isfinite(x)]
    return float(np.median(valid)) if valid else np.nan


# ── Peak-count frequency ──────────────────────────────────────────────
def compute_peak_count_freq_per_channel(seg, fs):
    """Peak-count frequency per channel. Require >= 3 peaks."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    min_distance = int(0.2 * fs)
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
        trace_max = np.max(trace)
        if trace_max <= 0:
            continue
        peak_height = trace_max * PEAK_HEIGHT_FRAC
        peak_locs, _ = find_peaks(trace, height=peak_height, distance=min_distance)
        if len(peak_locs) >= 3:
            freqs[i] = (len(peak_locs) - 1) / ((peak_locs[-1] - peak_locs[0]) / fs)
    return freqs


# ── Method A frequency ────────────────────────────────────────────────
def get_method_a_freq(data, fs):
    try:
        r = pd_detect_alternate(data, fs, pk_detect='apd')
        f = r.get('event_frequency', np.nan)
        if f is None or (isinstance(f, float) and np.isnan(f)):
            return np.nan
        return float(f)
    except Exception:
        return np.nan


# ── FFT of pointiness trace ──────────────────────────────────────────
def compute_fft_pointiness_per_channel(seg, fs):
    """FFT of smoothed pointiness trace per channel, peak in [0.3, 3.5] Hz."""
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))
    for i in range(n_ch):
        trace = compute_pointiness_trace(seg[i])
        trace = gaussian_filter1d(trace, sigma=sigma_samples)
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
        peak_idx = np.argmax(fft_sub)
        freqs[i] = freq_sub[peak_idx]
    return freqs


# ── ACF frequency ─────────────────────────────────────────────────────
def compute_acf_freq_per_channel(seg, fs):
    n_ch = seg.shape[0]
    freqs = np.full(n_ch, np.nan)
    for i in range(n_ch):
        freq, score, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=SMOOTHING_SIGMA,
            acf_min_lag=ACF_MIN_LAG,
            acf_peak_threshold=ACF_THRESHOLD,
            peak_height_frac=PEAK_HEIGHT_FRAC,
        )
        freqs[i] = freq
    return freqs


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Build all template banks
    banks = build_all_banks(dataset)

    # Define envelope FFT variants
    bank_variants = {
        'r5_bank16_fft': 'C16',
        'r5_bank24_fft': 'C24',
        'r5_pca5_fft':   'PCA5',
        'r5_pca10_fft':  'PCA10',
        'r5_multi_fft':  'Multi',
    }

    combo_names = list(bank_variants.keys()) + [
        'r5_bank16_plus_peaks',
        'r5_best_bank_ridge',
    ]
    all_predictions = {name: {} for name in combo_names}

    # For ridge regression: collect features
    ml_features = []

    print("\n" + "=" * 60)
    print("STEP 3: Processing all segments")
    print("=" * 60)

    n_total = len(dataset)
    for idx, entry in enumerate(dataset):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        mat_name = entry['mat_name']
        subdir = entry['subdir']
        expert_freq = entry['expert_consensus_freq']

        # Preprocess
        seg = preprocess_segment(data, fs)

        # Method A frequency
        a_freq = get_method_a_freq(data, fs)

        # Peak-count frequency
        pc_freqs = compute_peak_count_freq_per_channel(seg, fs)
        pc_freq = median_finite(pc_freqs)

        # ACF frequency (Method B, thr=0.10)
        acf_freqs = compute_acf_freq_per_channel(seg, fs)
        b_freq = median_finite(acf_freqs)

        # FFT of pointiness
        fft_pt_freqs = compute_fft_pointiness_per_channel(seg, fs)
        fft_pt_freq = median_finite(fft_pt_freqs)

        # Compute envelopes for each bank and get FFT frequencies
        bank_env_freqs = {}
        for bank_name, bank_data in banks.items():
            templates = bank_data['lpd'] if subdir == 'lpd' else bank_data['gpd']
            envelope = compute_discharge_envelope(seg, templates)
            env_freqs = envelope_fft_per_channel(envelope, fs)
            bank_env_freqs[bank_name] = median_finite(env_freqs)

        # Store predictions for pure envelope FFT variants
        for var_name, bank_name in bank_variants.items():
            all_predictions[var_name][mat_name] = bank_env_freqs[bank_name]

        # r5_bank16_plus_peaks: average of Bank C16 envelope FFT + peak-count freq
        c16_freq = bank_env_freqs['C16']
        vals_combo = [v for v in [c16_freq, pc_freq] if np.isfinite(v)]
        all_predictions['r5_bank16_plus_peaks'][mat_name] = \
            float(np.mean(vals_combo)) if vals_combo else np.nan

        # Collect features for ridge regression
        # Determine best envelope freq (will use C16 as the "best" candidate)
        type_is_gpd = 1.0 if subdir == 'gpd' else 0.0
        feature_vec = [
            a_freq,             # Method A
            b_freq,             # ACF thr=0.10
            pc_freq,            # Peak count
            fft_pt_freq,        # FFT of pointiness
            c16_freq,           # Envelope FFT (Bank C16 -- best bank TBD)
            type_is_gpd,        # Pattern type
        ]
        ml_features.append({
            'mat_name': mat_name,
            'features': feature_vec,
            'expert_freq': expert_freq,
            'subdir': subdir,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == n_total:
            print(f"  Progress: {idx + 1}/{n_total} segments")

    # ── Determine best bank and possibly swap envelope feature ──
    print("\n" + "=" * 60)
    print("STEP 4: Determining best bank for ridge regression")
    print("=" * 60)

    # Quick evaluation of each bank to find best
    best_bank_name = None
    best_bank_mae = 999.0
    for var_name, bank_name in bank_variants.items():
        preds = all_predictions[var_name]
        errors = []
        for entry in dataset:
            mn = entry['mat_name']
            if mn in preds and np.isfinite(preds[mn]) and np.isfinite(entry['expert_consensus_freq']):
                errors.append(abs(preds[mn] - entry['expert_consensus_freq']))
        if errors:
            mae = np.mean(errors)
            print(f"  {var_name} ({bank_name}): MAE = {mae:.4f} (n={len(errors)})")
            if mae < best_bank_mae:
                best_bank_mae = mae
                best_bank_name = bank_name

    print(f"\n  Best bank: {best_bank_name} (MAE={best_bank_mae:.4f})")

    # If best bank is not C16, swap the envelope feature in ml_features
    if best_bank_name and best_bank_name != 'C16':
        best_var_name = [k for k, v in bank_variants.items() if v == best_bank_name][0]
        best_preds = all_predictions[best_var_name]
        for mf in ml_features:
            mn = mf['mat_name']
            if mn in best_preds:
                mf['features'][4] = best_preds[mn]  # replace envelope freq

    # ── Ridge regression (LOO-CV) ──
    print("\n" + "=" * 60)
    print("STEP 5: Ridge regression (LOO-CV on log(freq))")
    print("=" * 60)

    valid_ml = [m for m in ml_features
                if np.isfinite(m['expert_freq']) and m['expert_freq'] > 0]
    n_ml = len(valid_ml)
    print(f"  ML samples: {n_ml}")

    X = np.array([m['features'] for m in valid_ml])
    y = np.log(np.array([m['expert_freq'] for m in valid_ml]))

    feature_names = ['MethodA', 'ACF_thr010', 'PeakCount', 'FFT_Pointiness',
                     f'EnvelopeFFT_{best_bank_name}', 'IsGPD']

    # Impute NaN features with column median
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = ~np.isfinite(col)
        if np.any(nan_mask):
            col_median = np.nanmedian(col)
            if not np.isfinite(col_median):
                col_median = 0.0
            X[nan_mask, col_idx] = col_median
            n_imputed = int(np.sum(nan_mask))
            print(f"  Imputed {n_imputed} NaN in {feature_names[col_idx]} "
                  f"with median={col_median:.3f}")

    # LOO-CV with numpy Ridge: beta = (X'X + alpha*I)^-1 X'y
    alpha = 1.0
    loo_predictions = np.full(n_ml, np.nan)

    for i in range(n_ml):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i:i + 1]

        # Add intercept column
        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(1)])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0  # Don't regularize intercept
        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = float(X_test_b @ w)
            loo_predictions[i] = np.exp(pred_log)
        except np.linalg.LinAlgError:
            loo_predictions[i] = np.nan

    # Store ML predictions
    for i, m in enumerate(valid_ml):
        all_predictions['r5_best_bank_ridge'][m['mat_name']] = float(loo_predictions[i])

    # Print feature coefficients (full model)
    X_b = np.column_stack([X, np.ones(len(X))])
    I_reg = np.eye(X_b.shape[1])
    I_reg[-1, -1] = 0
    try:
        w_full = np.linalg.solve(X_b.T @ X_b + alpha * I_reg, X_b.T @ y)
        print("\n  Feature coefficients (full model, Ridge alpha=1.0):")
        for fname, coef in zip(feature_names, w_full[:-1]):
            print(f"    {fname:>25s}: {coef:+.4f}")
        print(f"    {'intercept':>25s}: {w_full[-1]:+.4f}")
    except Exception as e:
        print(f"  Could not compute full-model coefficients: {e}")

    # ── Evaluate all variants ──
    print("\n" + "=" * 60)
    print("EVALUATING ALL ROUND 5 VARIANTS")
    print("=" * 60)

    summary_rows = []
    for var_name in combo_names:
        n_valid = sum(1 for v in all_predictions[var_name].values() if np.isfinite(v))
        print(f"\n{var_name}: {n_valid} valid predictions out of {len(all_predictions[var_name])}")
        metrics = evaluate_predictions(dataset, all_predictions[var_name], var_name)
        summary_rows.append(metrics)

    # ── Print results table ──
    print("\n" + "=" * 70)
    print("ROUND 5 EXPANDED BANKS RESULTS (sorted by combined Spearman)")
    print("=" * 70)
    header = (f"{'Experiment':<35s} {'LPD MAE':>8s} {'GPD MAE':>8s} "
              f"{'LPD Sp':>7s} {'GPD Sp':>7s} {'Comb Sp':>8s} {'Comb MAE':>9s}")
    print(header)
    print("-" * len(header))

    # Baselines
    print(f"{'Method A (baseline)':<35s} {'0.537':>8s} {'0.274':>8s} "
          f"{'0.282':>7s} {'0.309':>7s} {'0.296':>8s} {'0.406':>9s}")
    print("-" * len(header))

    sorted_rows = sorted(summary_rows,
                         key=lambda r: -(r.get('combined_spearman', -999)
                                         if np.isfinite(r.get('combined_spearman', np.nan))
                                         else -999))
    for row in sorted_rows:
        def fmt(key, default='?'):
            v = row.get(key, default)
            if isinstance(v, (int, float)) and np.isfinite(v):
                return f"{v:.4f}"
            return str(v)
        print(f"{row['experiment']:<35s} {fmt('lpd_mae'):>8s} {fmt('gpd_mae'):>8s} "
              f"{fmt('lpd_spearman_r'):>7s} {fmt('gpd_spearman_r'):>7s} "
              f"{fmt('combined_spearman'):>8s} {fmt('combined_mae'):>9s}")

    print("\nDone! Results saved to results/optimization_runs/")


if __name__ == '__main__':
    main()
