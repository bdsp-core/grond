"""
Generate publication-quality pairwise rater agreement scatterplots.

Panel A: Expert-Expert pairwise scatter (LB, PH, SZ, MW)
Panel B: Expert-Algorithm pairwise scatter (each expert vs Ridge LOPO)

Run: conda run -n foe python code/figure_pairwise_agreement.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence as scipy_coherence
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr, pearsonr
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import compute_pointiness_trace, compute_acf_frequency, fcn_getBanana

# ── Constants ──────────────────────────────────────────────────────────
FS = 200
FREQ_LO, FREQ_HI = 0.3, 3.5
LOWPASS_HZ = 15.0
SMOOTHING_SIGMA = 0.02
ACF_MIN_LAG = 0.4
ACF_THRESHOLD = 0.10
PEAK_HEIGHT_FRAC = 0.3

ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
RESULTS_DIR = PROJECT_DIR / 'results'


# ── Feature computation (from r12_full_evaluation.py) ──────────────────
def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
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


def compute_sp_features_from_bipolar(seg_bip, fs, is_gpd):
    features = {}
    features['is_gpd'] = float(is_gpd)
    n_channels = seg_bip.shape[0]

    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    sigma_samples = max(1, int(SMOOTHING_SIGMA * fs))

    # f_B
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
    features['n_ch'] = len(valid_acf)

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
        f = compute_fft_peak(pointiness_traces[ch], fs)
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
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh
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
        except:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    return features


# ── Ridge LOPO (from r12_full_evaluation.py) ───────────────────────────
def ridge_lopo(df, feature_cols, target_col='gold_standard_freq', alpha=1.0):
    n = len(df)
    preds = np.full(n, np.nan)
    patient_ids = df['patient_id'].values
    unique_patients = sorted(set(patient_ids))

    for held_out in unique_patients:
        test_mask = patient_ids == held_out
        train_mask = ~test_mask
        if np.sum(train_mask) < 5:
            continue

        X_train = df.loc[train_mask, feature_cols].values.astype(float)
        y_train = np.log(df.loc[train_mask, target_col].values.astype(float))
        X_test = df.loc[test_mask, feature_cols].values.astype(float)

        for j in range(X_train.shape[1]):
            col = X_train[:, j]
            finite = np.isfinite(col)
            med = np.median(col[finite]) if np.any(finite) else 0.0
            X_train[~finite, j] = med
            test_col = X_test[:, j]
            X_test[~np.isfinite(test_col), j] = med

        X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
        X_test_b = np.column_stack([X_test, np.ones(X_test.shape[0])])

        I_reg = np.eye(X_train_b.shape[1])
        I_reg[-1, -1] = 0

        try:
            w = np.linalg.solve(X_train_b.T @ X_train_b + alpha * I_reg,
                                X_train_b.T @ y_train)
            pred_log = X_test_b @ w
            pred_log = np.clip(pred_log, np.log(0.1), np.log(10.0))
            pred_freq = np.exp(pred_log)
            test_idx = np.where(test_mask)[0]
            for i, gi in enumerate(test_idx):
                preds[gi] = pred_freq[i]
        except np.linalg.LinAlgError:
            continue
    return preds


# ── Plotting helpers ───────────────────────────────────────────────────
def confidence_ellipse(x, y, ax, n_std=1.96, facecolor='none', **kwargs):
    """Draw a 95% confidence ellipse around a point cloud."""
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
        return None
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                       facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = np.mean(x), np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_panel(ax, scorers_data, title, jitter_scale=0.02, subtype_filter=None):
    """Plot one panel: each scorer vs gold standard (MW).

    scorers_data: list of dicts with keys:
        label, x (gold standard), y (scorer), subtypes, color
    subtype_filter: if set ('lpd' or 'gpd'), only plot points of that subtype
    """
    ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=10)

    all_vals = []
    legend_entries = []

    for pd_item in scorers_data:
        label = pd_item['label']
        x = np.array(pd_item['x'], dtype=float)
        y = np.array(pd_item['y'], dtype=float)
        subtypes = np.array(pd_item['subtypes'])
        color = pd_item['color']

        # Filter to subtype
        if subtype_filter is not None:
            mask = subtypes == subtype_filter
            x, y, subtypes = x[mask], y[mask], subtypes[mask]

        valid = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[valid], y[valid]
        n_patients = len(xv)
        if n_patients < 3:
            continue

        rho_s, _ = spearmanr(xv, yv)
        rho_p, _ = pearsonr(xv, yv)
        mad = np.mean(np.abs(xv - yv))

        # Add jitter
        rng = np.random.RandomState(hash(label) % 2**31)
        xj = xv + rng.normal(0, jitter_scale, n_patients)
        yj = yv + rng.normal(0, jitter_scale, n_patients)

        all_vals.extend(xv)
        all_vals.extend(yv)

        ax.scatter(xj, yj, c=color, marker='o',
                   s=45, alpha=0.55, edgecolors='white', linewidths=0.3, zorder=3)

        # Regression line
        coeffs = np.polyfit(xv, yv, 1)
        x_line = np.linspace(min(xv), max(xv), 100)
        y_line = np.polyval(coeffs, x_line)
        ax.plot(x_line, y_line, color=color, linewidth=2.0, alpha=0.7, zorder=4)

        legend_label = (f"{label} (N={n_patients}): "
                        f"\u03c1\u209b={rho_s:.2f}, r={rho_p:.2f}, MAD={mad:.2f}")
        legend_entries.append((color, legend_label))

    # Axis limits
    if all_vals:
        lo = max(-0.1, min(all_vals) - 0.2)
        hi = max(all_vals) + 0.2
    else:
        lo, hi = 0, 3.5

    # Identity line
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, linewidth=1.2, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # Rug plots (gold standard values on x-axis only, to avoid clutter)
    for pd_item in scorers_data:
        x_raw = np.array(pd_item['x'], dtype=float)
        st_raw = np.array(pd_item['subtypes'])
        color = pd_item['color']
        if subtype_filter is not None:
            mask = st_raw == subtype_filter
            x_raw = x_raw[mask]
        xr = x_raw[np.isfinite(x_raw)]
        rug_y = lo + (hi - lo) * 0.01
        ax.plot(xr, np.full_like(xr, rug_y), '|',
                color=color, alpha=0.4, markersize=5, zorder=5)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], linestyle='--', color='black', alpha=0.35,
                      linewidth=1.2, label='Identity (y=x)')]
    for color, label in legend_entries:
        handles.append(Line2D([0], [0], marker='o', color=color,
                              linestyle='-', linewidth=2.0,
                              markersize=5, alpha=0.7, label=label))

    ax.legend(handles=handles, fontsize=7.5, loc='upper left',
              framealpha=0.92, edgecolor='lightgray', borderpad=0.8)
    ax.set_xlabel('Gold Standard — MW (Hz)', fontsize=11)
    ax.set_ylabel('Scorer Frequency (Hz)', fontsize=11)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.15, linestyle='-')
    ax.tick_params(labelsize=9)


# ── Helpers ────────────────────────────────────────────────────────────
def _load_mat_as_bipolar(mat_path, montage, n_channels):
    """Load a .mat file and return (18, N) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    return data


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print("Loading unified dataset...")
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_patients = df_patients[df_patients['excluded'] == False].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'].notna()].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'] > 0].copy()
    print(f"  Non-excluded patients with gold standard: {len(df_patients)}")

    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    df_annotations = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    df_annotations['patient_id'] = df_annotations['patient_id'].astype(str)

    # ── Build per-rater patient-level frequency from annotations ──
    # Get mean frequency per (patient, rater) for expert ratings
    rater_freq = (df_annotations[df_annotations['frequency_hz'].notna() &
                                  (df_annotations['no_pd'] == False) &
                                  (df_annotations['skipped'] == False)]
                  .groupby(['patient_id', 'rater'])['frequency_hz']
                  .mean().reset_index())

    # Pivot to get expert_LB, expert_PH, expert_SZ columns
    expert_pivot = rater_freq[rater_freq['rater'].isin(['LB', 'PH', 'SZ'])].copy()
    expert_pivot = expert_pivot.pivot(index='patient_id', columns='rater',
                                       values='frequency_hz').reset_index()
    expert_pivot.columns = ['patient_id'] + [f'expert_{c}' for c in expert_pivot.columns[1:]]

    # ── Load best segment per patient and compute features ──
    print("Loading EEG segments and computing SP features...")
    feature_records = []
    patient_ids = []

    for _, pat_row in df_patients.iterrows():
        pid = str(pat_row['patient_id'])
        subtype = pat_row['subtype']
        is_gpd = 1 if subtype == 'gpd' else 0

        # Find segments for this patient
        pat_segs = df_segments[df_segments['patient_id'] == pid]
        if len(pat_segs) == 0:
            feature_records.append({'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                                     'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd)})
            patient_ids.append(pid)
            continue

        # Load all segments, pick the one with highest variance
        best_seg = None
        best_var = -1
        for _, seg_row in pat_segs.iterrows():
            mat_path = EEG_DIR / seg_row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(mat_path, seg_row['montage'], seg_row['n_channels'])
                v = np.var(seg)
                if v > best_var:
                    best_var = v
                    best_seg = seg
            except Exception:
                continue

        if best_seg is None:
            feature_records.append({'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                                     'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd)})
        else:
            try:
                feats = compute_sp_features_from_bipolar(best_seg, FS, is_gpd)
            except Exception:
                feats = {'f_B': np.nan, 'f_peaks': np.nan, 'f_fft': np.nan,
                         'f_tkeo': np.nan, 'f_coh': np.nan, 'is_gpd': float(is_gpd)}
            feature_records.append(feats)
        patient_ids.append(pid)

        if len(patient_ids) % 50 == 0:
            print(f"  Processed {len(patient_ids)}/{len(df_patients)}")

    # Build combined DataFrame
    df = df_patients[['patient_id', 'subtype', 'gold_standard_freq']].copy().reset_index(drop=True)
    df['patient_id'] = df['patient_id'].astype(str)
    df['reviewer_MW'] = df['gold_standard_freq']

    df_feats = pd.DataFrame(feature_records)
    for col in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd']:
        df[col] = df_feats[col].values

    # Merge expert ratings
    df = df.merge(expert_pivot, on='patient_id', how='left')
    # Ensure expert columns exist even if no data
    for ecol in ['expert_LB', 'expert_PH', 'expert_SZ']:
        if ecol not in df.columns:
            df[ecol] = np.nan

    print(f"  Combined dataset: {len(df)} patients")

    # ── Run Ridge LOPO to get algorithm predictions ──
    print("Running Ridge LOPO CV...")
    feature_cols = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd']

    # Only use patients with valid gold standard
    valid_gold = df['gold_standard_freq'].notna() & (df['gold_standard_freq'] > 0)
    df_model = df[valid_gold].copy().reset_index(drop=True)
    print(f"  Patients with valid MW rating: {len(df_model)}")

    preds = ridge_lopo(df_model, feature_cols, target_col='gold_standard_freq', alpha=1.0)
    df_model['algo_pred'] = preds

    # Map predictions back to full df
    df['algo_pred'] = np.nan
    df.loc[valid_gold, 'algo_pred'] = preds

    # Evaluate algorithm
    valid_pred = np.isfinite(preds)
    if np.sum(valid_pred) > 2:
        rho_s, _ = spearmanr(df_model.loc[valid_pred, 'gold_standard_freq'],
                             preds[valid_pred])
        print(f"  Ridge LOPO Spearman vs MW: {rho_s:.3f} (n={np.sum(valid_pred)})")

    # ── Identify patients with expert ratings (have LB, PH, SZ) ──
    has_experts = df['expert_LB'].notna() | df['expert_PH'].notna() | df['expert_SZ'].notna()
    print(f"  Patients with expert ratings (LB/PH/SZ): {has_experts.sum()}")

    # ══════════════════════════════════════════════════════════════════
    # Build scorer-vs-gold-standard data (gold standard = MW)
    # Scorers: LB, PH, SZ, Algorithm
    # ══════════════════════════════════════════════════════════════════
    scorer_cols = {
        'LB': 'expert_LB',
        'PH': 'expert_PH',
        'SZ': 'expert_SZ',
        'Algorithm': 'algo_pred',
    }

    # Colorblind-friendly palette (Wong 2011)
    colors = {'LB': '#0072B2', 'PH': '#D55E00', 'SZ': '#009E73', 'Algorithm': '#CC79A7'}

    gold_col = 'reviewer_MW'

    scorers_data = []
    for name, col in scorer_cols.items():
        vals_scorer = df[col].values.astype(float)
        vals_gold = df[gold_col].values.astype(float)
        subtypes = df['subtype'].values

        valid = np.isfinite(vals_scorer) & np.isfinite(vals_gold) & (vals_gold > 0)
        vs = vals_scorer[valid]
        vg = vals_gold[valid]
        st = subtypes[valid]

        # NOT doubled — x=gold standard, y=scorer
        scorers_data.append({
            'label': name,
            'x': vg,
            'y': vs,
            'subtypes': st,
            'color': colors[name],
        })
        print(f"  {name} vs MW: {len(vg)} patients")

    # ══════════════════════════════════════════════════════════════════
    # Create 1x2 figure: left=LPD, right=GPD
    # ══════════════════════════════════════════════════════════════════
    print("\nCreating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    n_lpd = (df['subtype'] == 'lpd').sum()
    n_gpd = (df['subtype'] == 'gpd').sum()

    plot_panel(axes[0], scorers_data,
               f'A. LPD (n={n_lpd} patients)',
               subtype_filter='lpd')
    plot_panel(axes[1], scorers_data,
               f'B. GPD (n={n_gpd} patients)',
               subtype_filter='gpd')

    fig.suptitle('Agreement with Gold Standard (MW) for Periodic Discharge Frequency',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    out_path = RESULTS_DIR / 'figure_pairwise_agreement.png'
    fig.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {out_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
