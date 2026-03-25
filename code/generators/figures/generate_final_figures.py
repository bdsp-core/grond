"""
Generate comprehensive comparison figures for the breakthrough DL embedding result.

Best model: Ridge on log(freq), per-expert training (3 experts),
  28 features (8 signal-processing + 20 PCA of CNN backbone embeddings).
  CNN backbone pretrained on 3,816 LPD/GPD classification segments.

Result: LPD Spearman 0.538, GPD Spearman 0.501, Combined 0.492 (98% of expert-expert 0.50).

Run: conda run -n foe_dl python code/generate_final_figures.py
"""

import sys
import json
import warnings
import numpy as np
import torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup paths
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
DL_DIR = CODE_DIR / 'dl'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(DL_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

CACHE_DIR = PROJECT_DIR / 'data' / 'dl_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
RUNS_DIR = RESULTS_DIR / 'optimization_runs'
RESULTS_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# LOAD THE BEST MODEL RESULTS FROM CACHED JSON
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("GENERATING FINAL COMPARISON FIGURES")
print("=" * 70)

best_path = RUNS_DIR / 'dl_emb_pca20_a1.json'
print(f"\n[1] Loading best model results from {best_path.name}...")
with open(str(best_path)) as f:
    best = json.load(f)

# Extract arrays from stored JSON
lpd_expert = np.array(best['lpd_expert_vals'])
lpd_pred = np.array(best['lpd_pred_vals'])
gpd_expert = np.array(best['gpd_expert_vals'])
gpd_pred = np.array(best['gpd_pred_vals'])

# Per-expert arrays (0 means missing)
lpd_expert_LB = np.array([v if v is not None else 0.0 for v in best['lpd_expert_LB']])
lpd_expert_PH = np.array([v if v is not None else 0.0 for v in best['lpd_expert_PH']])
lpd_expert_SZ = np.array([v if v is not None else 0.0 for v in best['lpd_expert_SZ']])
gpd_expert_LB = np.array([v if v is not None else 0.0 for v in best['gpd_expert_LB']])
gpd_expert_PH = np.array([v if v is not None else 0.0 for v in best['gpd_expert_PH']])
gpd_expert_SZ = np.array([v if v is not None else 0.0 for v in best['gpd_expert_SZ']])

print(f"  LPD: n={best['lpd_n']}, Spearman={best['lpd_spearman_r']}, pooled={best['lpd_spearman_pooled']}")
print(f"  GPD: n={best['gpd_n']}, Spearman={best['gpd_spearman_r']}, pooled={best['gpd_spearman_pooled']}")
print(f"  Combined: {best['combined_spearman']}")


# ══════════════════════════════════════════════════════════════════════════
# Helper: compute expert-expert pairwise data
# ══════════════════════════════════════════════════════════════════════════
def expert_pair_stats(e1, e2):
    """Compute stats for a pair of expert arrays. 0 means missing."""
    mask = (e1 > 0) & (e2 > 0)
    x, y = e1[mask], e2[mask]
    n = len(x)
    if n < 3:
        return {'x': x, 'y': y, 'n': n}
    rs, _ = spearmanr(x, y)
    rp, _ = pearsonr(x, y)
    mae = np.mean(np.abs(x - y))
    return {'x': x, 'y': y, 'n': n, 'spearman': rs, 'pearson': rp, 'mae': mae}


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Expert-vs-Expert and Algorithm-vs-Expert scatter (2x3 grid)
# ══════════════════════════════════════════════════════════════════════════
print("\n[2] Generating Figure 1: Scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Frequency Estimation: Expert-Expert vs Algorithm-Expert Agreement",
             fontsize=14, fontweight='bold', y=0.98)

np.random.seed(42)

for row, ptype in enumerate(['lpd', 'gpd']):
    ptype_label = ptype.upper()

    if ptype == 'lpd':
        eLB, ePH, eSZ = lpd_expert_LB, lpd_expert_PH, lpd_expert_SZ
        expert_vals, pred_vals = lpd_expert, lpd_pred
    else:
        eLB, ePH, eSZ = gpd_expert_LB, gpd_expert_PH, gpd_expert_SZ
        expert_vals, pred_vals = gpd_expert, gpd_pred

    # Col 0: LB vs PH
    ax = axes[row, 0]
    d = expert_pair_stats(eLB, ePH)
    if d['n'] > 0:
        jx = d['x'] + np.random.normal(0, 0.05, d['n'])
        jy = d['y'] + np.random.normal(0, 0.05, d['n'])
        ax.scatter(jx, jy, c='#E07020', alpha=0.6, s=30, edgecolors='none')
        lim_max = max(np.max(d['x']), np.max(d['y'])) + 0.5
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim([0, lim_max])
        ax.set_ylim([0, lim_max])
        stats_text = (f"Spearman r = {d.get('spearman', 0):.3f}\n"
                      f"Pearson r = {d.get('pearson', 0):.3f}\n"
                      f"MAE = {d.get('mae', 0):.3f} Hz\n"
                      f"n = {d['n']}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel("Expert LB (Hz)", fontsize=9)
    ax.set_ylabel("Expert PH (Hz)", fontsize=9)
    ax.set_title(f"{ptype_label}: Expert LB vs PH", fontsize=10, fontweight='bold')

    # Col 1: LB vs SZ
    ax = axes[row, 1]
    d = expert_pair_stats(eLB, eSZ)
    if d['n'] > 0:
        jx = d['x'] + np.random.normal(0, 0.05, d['n'])
        jy = d['y'] + np.random.normal(0, 0.05, d['n'])
        ax.scatter(jx, jy, c='#E07020', alpha=0.6, s=30, edgecolors='none')
        lim_max = max(np.max(d['x']), np.max(d['y'])) + 0.5
        ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim([0, lim_max])
        ax.set_ylim([0, lim_max])
        stats_text = (f"Spearman r = {d.get('spearman', 0):.3f}\n"
                      f"Pearson r = {d.get('pearson', 0):.3f}\n"
                      f"MAE = {d.get('mae', 0):.3f} Hz\n"
                      f"n = {d['n']}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
                va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
    ax.set_xlabel("Expert LB (Hz)", fontsize=9)
    ax.set_ylabel("Expert SZ (Hz)", fontsize=9)
    ax.set_title(f"{ptype_label}: Expert LB vs SZ", fontsize=10, fontweight='bold')

    # Col 2: Algorithm vs Consensus
    ax = axes[row, 2]
    algo_color = '#2E8B57' if ptype == 'lpd' else '#4169E1'
    bg_color = 'lightgreen' if ptype == 'lpd' else 'lightblue'
    n = len(expert_vals)
    jx = pred_vals + np.random.normal(0, 0.05, n)
    jy = expert_vals + np.random.normal(0, 0.05, n)
    ax.scatter(jx, jy, c=algo_color, alpha=0.6, s=30, edgecolors='none')
    lim_max = max(np.max(pred_vals), np.max(expert_vals)) + 0.5
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.4, linewidth=1)
    ax.set_xlim([0, lim_max])
    ax.set_ylim([0, lim_max])

    rs_con, _ = spearmanr(pred_vals, expert_vals)
    rp_con, _ = pearsonr(pred_vals, expert_vals)
    mae_con = np.mean(np.abs(pred_vals - expert_vals))
    stats_text = (f"Spearman r = {rs_con:.3f}\n"
                  f"Pearson r = {rp_con:.3f}\n"
                  f"MAE = {mae_con:.3f} Hz\n"
                  f"n = {n}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va='top', ha='left', bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.8))
    ax.set_xlabel("Algorithm (Hz)", fontsize=9)
    ax.set_ylabel("Expert Consensus (Hz)", fontsize=9)
    ax.set_title(f"{ptype_label}: Algorithm vs Expert Consensus", fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(str(RESULTS_DIR / 'figure_final_scatter.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved figure_final_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Progress across rounds (line plot)
# ══════════════════════════════════════════════════════════════════════════
print("\n[3] Generating Figure 2: Progress across rounds...")

round_labels = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'DL']
lpd_progress = [0.33, 0.36, 0.42, 0.42, 0.42, 0.45, 0.47, 0.49]
gpd_progress = [0.29, 0.43, 0.43, 0.38, 0.38, 0.42, 0.46, 0.54]
comb_progress = [0.45, 0.40, 0.49, 0.49, 0.49, 0.51, 0.52, 0.50]

# Use actual stored values for DL round
lpd_progress[-1] = best['lpd_spearman_pooled']
gpd_progress[-1] = best['gpd_spearman_pooled']
comb_progress[-1] = best['combined_spearman']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x = np.arange(len(round_labels))

ax.plot(x, lpd_progress, 'o-', color='#2E8B57', linewidth=2.5, markersize=9, label='LPD Spearman', zorder=5)
ax.plot(x, gpd_progress, 's-', color='#4169E1', linewidth=2.5, markersize=9, label='GPD Spearman', zorder=5)
ax.plot(x, comb_progress, '^-', color='#8B008B', linewidth=2.5, markersize=9, label='Combined Spearman', zorder=5)

# Expert-expert baselines
ax.axhline(y=0.525, color='#2E8B57', linestyle='--', alpha=0.5, linewidth=1.5, label='Expert-Expert LPD (0.525)')
ax.axhline(y=0.476, color='#4169E1', linestyle='--', alpha=0.5, linewidth=1.5, label='Expert-Expert GPD (0.476)')
ax.axhline(y=0.50, color='#8B008B', linestyle='--', alpha=0.3, linewidth=1.0, label='Expert-Expert Combined (0.50)')

# Annotations for key breakthroughs
ax.annotate('FFT + ridge\nbaseline', xy=(0, 0.33), xytext=(0.8, 0.27),
            fontsize=7, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow'))
ax.annotate('Phase coherence\n+ TKEO added', xy=(5, 0.45), xytext=(4.3, 0.37),
            fontsize=7, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.2', fc='lightyellow'))
ax.annotate(f'CNN embeddings\nBEATS EXPERTS\nGPD={gpd_progress[-1]:.3f}',
            xy=(7, gpd_progress[-1]), xytext=(6.0, 0.57),
            fontsize=8, ha='center', fontweight='bold', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='red'))

ax.set_xticks(x)
ax.set_xticklabels(round_labels, fontsize=10)
ax.set_ylabel('Spearman Correlation (pooled)', fontsize=11)
ax.set_xlabel('Optimization Round', fontsize=11)
ax.set_title('Frequency Estimation: Progress Across Optimization Rounds', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.set_ylim(0.22, 0.62)
ax.grid(True, alpha=0.3)

fig.savefig(str(RESULTS_DIR / 'figure_progress.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved figure_progress.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Method comparison bar chart
# ══════════════════════════════════════════════════════════════════════════
print("\n[4] Generating Figure 3: Method comparison bar chart...")

# Method A baseline: use consensus Spearman from r2_baseline_A (not pooled, which wasn't computed)
# LPD=0.234, GPD=-0.145 for Method A alone (very poor for GPD)
method_a_lpd, method_a_gpd = 0.234, -0.145
r2a_path = RUNS_DIR / 'r2_baseline_A.json'
if r2a_path.exists():
    with open(str(r2a_path)) as f:
        r2a = json.load(f)
    method_a_lpd = r2a.get('lpd_spearman_r', 0.234)
    method_a_gpd = r2a.get('gpd_spearman_r', -0.145)

# Load best R7 result
r7_lpd, r7_gpd = 0.47, 0.46
for fname in ['r7_no_A.json', 'r7_mega_v2.json', 'r7_mega_final.json']:
    fpath = RUNS_DIR / fname
    if fpath.exists():
        with open(str(fpath)) as f:
            r7d = json.load(f)
        r7_lpd = r7d.get('lpd_spearman_pooled', r7d.get('lpd_spearman_r', r7_lpd))
        r7_gpd = r7d.get('gpd_spearman_pooled', r7d.get('gpd_spearman_r', r7_gpd))
        print(f"  Loaded R7 values from {fname}: LPD={r7_lpd:.3f}, GPD={r7_gpd:.3f}")
        break

methods_data = {
    'Method A\nBaseline': {'lpd': method_a_lpd, 'gpd': method_a_gpd},
    'Best Signal\nProcessing (R7)': {'lpd': r7_lpd, 'gpd': r7_gpd},
    'CNN Embeddings\n+ SP (Best)': {'lpd': best['lpd_spearman_pooled'], 'gpd': best['gpd_spearman_pooled']},
    'Expert-Expert\n(ceiling)': {'lpd': 0.525, 'gpd': 0.476},
}

method_names = list(methods_data.keys())
lpd_vals = [methods_data[m]['lpd'] for m in method_names]
gpd_vals = [methods_data[m]['gpd'] for m in method_names]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
x_pos = np.arange(len(method_names))
width = 0.35

bars_lpd = ax.bar(x_pos - width/2, lpd_vals, width, label='LPD Spearman', color='#2E8B57', alpha=0.85)
bars_gpd = ax.bar(x_pos + width/2, gpd_vals, width, label='GPD Spearman', color='#4169E1', alpha=0.85)

# Add value labels
for bar in bars_lpd:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
for bar in bars_gpd:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Expert-expert lines
ax.axhline(y=0.525, color='#2E8B57', linestyle='--', alpha=0.4, linewidth=1, label='Expert-Expert LPD')
ax.axhline(y=0.476, color='#4169E1', linestyle='--', alpha=0.4, linewidth=1, label='Expert-Expert GPD')

ax.set_xticks(x_pos)
ax.set_xticklabels(method_names, fontsize=9)
ax.set_ylabel('Spearman Correlation (pooled)', fontsize=11)
ax.set_title('Method Comparison: Frequency Estimation Performance', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.set_ylim(-0.2, 0.65)
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
ax.grid(True, axis='y', alpha=0.3)

fig.savefig(str(RESULTS_DIR / 'figure_method_comparison.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved figure_method_comparison.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Feature importance (requires re-running the model to get coefs)
# ══════════════════════════════════════════════════════════════════════════
print("\n[5] Generating Figure 4: Feature importance...")
print("  Re-running ridge model to extract coefficients...")

from dl.model import EEGBackbone
from dl.data_loader import normalize_segment
from dl.evaluate import (
    preprocess_segment_r7, get_f_B, get_f_peaks, get_f_fft,
    get_f_tkeo_fft, get_f_spectral_coh, get_f_hps3,
    compute_pointiness_traces, impute_nan_median,
)
from pd_pointiness_acf import fcn_getBanana
from optimization_harness import load_dataset, load_eeg_data
import time

FS = 200

# Load annotated data
ann_data = np.load(str(CACHE_DIR / 'annotated_pd_data.npz'), allow_pickle=True)
segments = ann_data['segments']
expert_freqs = ann_data['expert_freqs']
patients_arr = ann_data['patients']
subtypes_arr = ann_data['subtypes']
N = len(segments)

# Load pretrained backbone and extract embeddings
backbone = EEGBackbone(in_channels=18, dropout=0.0)
classifier_state = torch.load(str(CACHE_DIR / 'classifier_best.pt'),
                               map_location='cpu', weights_only=False)
backbone_dict = {k.replace('backbone.', ''): v for k, v in classifier_state.items()
                 if k.startswith('backbone.')}
backbone.load_state_dict(backbone_dict, strict=True)
backbone.eval()

embeddings = np.zeros((N, 128), dtype=np.float32)
with torch.no_grad():
    batch_size = 32
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = segments[start:end].copy()
        for i in range(end - start):
            batch[i] = normalize_segment(batch[i])
        x_t = torch.from_numpy(batch)
        feats = backbone(x_t)
        pooled = feats.mean(dim=2)
        embeddings[start:end] = pooled.numpy()

pca = PCA(n_components=20)
emb_pca = pca.fit_transform(embeddings)
print(f"  Embeddings PCA: {emb_pca.shape}")

# Load dataset for SP features
dataset = load_dataset()
dataset_by_mat = {e['mat_name']: e for e in dataset}
pred_data = np.load(str(CACHE_DIR / 'phase2_predictions.npz'), allow_pickle=True)
mat_names = pred_data['mat_names']

# SP feature names
SP_FEAT_NAMES = ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh', 'is_gpd', 'n_ch', 'placeholder']
FEAT_NAMES_28 = SP_FEAT_NAMES + [f'pca_{i}' for i in range(20)]

sp_features = np.full((N, 8), np.nan)
expert_consensus = np.full(N, np.nan)
expert_LB_arr = np.full(N, np.nan)
expert_PH_arr = np.full(N, np.nan)
expert_SZ_arr = np.full(N, np.nan)

t0 = time.time()
for idx in range(N):
    mat_name = str(mat_names[idx])
    entry = dataset_by_mat.get(mat_name)
    if entry is None:
        continue
    expert_consensus[idx] = entry['expert_consensus_freq']
    expert_LB_arr[idx] = entry.get('expert_LB_freq', np.nan)
    expert_PH_arr[idx] = entry.get('expert_PH_freq', np.nan)
    expert_SZ_arr[idx] = entry.get('expert_SZ_freq', np.nan)

    try:
        data, fs = load_eeg_data(entry)
        if data is None:
            continue
        f_B = get_f_B(data, fs)
        seg = preprocess_segment_r7(data, fs)
        from mne.filter import notch_filter, filter_data
        seg_bb = notch_filter(data, fs, 60, n_jobs=1, verbose="ERROR")
        seg_bb = filter_data(seg_bb, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        seg_bb = np.array(fcn_getBanana(seg_bb))
        traces = compute_pointiness_traces(seg, fs)

        sp_features[idx, 0] = f_B
        sp_features[idx, 1] = get_f_peaks(traces, fs)
        sp_features[idx, 2] = get_f_fft(traces, fs)
        sp_features[idx, 3] = get_f_tkeo_fft(seg, fs)
        sp_features[idx, 4] = get_f_spectral_coh(seg_bb, fs)
        sp_features[idx, 5] = 1.0 if entry['subdir'] == 'gpd' else 0.0
        sp_features[idx, 6] = float(np.sum(np.std(seg, axis=1) > 1e-6))
        sp_features[idx, 7] = 0.0
    except Exception:
        continue

print(f"  SP features extracted in {time.time()-t0:.0f}s")

# Combine features
X_full = np.hstack([sp_features, emb_pca])
valid_mask = np.isfinite(expert_consensus) & (expert_consensus > 0)

# Per-expert ridge to collect coefficients
expert_names = ['LB', 'PH', 'SZ']
expert_arrays = [expert_LB_arr, expert_PH_arr, expert_SZ_arr]
all_fold_coefs = []

for ei, (ename, earr) in enumerate(zip(expert_names, expert_arrays)):
    emask = valid_mask & np.isfinite(earr) & (earr > 0)
    eidx = np.where(emask)[0]
    if len(eidx) < 10:
        continue
    X_e = impute_nan_median(X_full[eidx].copy())
    y_e = np.log(earr[eidx])
    g_e = patients_arr[eidx]

    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X_e, y_e, groups=g_e):
        X_train = X_e[train_idx]
        y_train = y_e[train_idx]
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std[std == 0] = 1.0
        X_tr_s = (X_train - mu) / std
        y_mu = np.mean(y_train)
        y_c = y_train - y_mu
        p = X_tr_s.shape[1]
        A = X_tr_s.T @ X_tr_s + 1.0 * np.eye(p)
        try:
            w = np.linalg.solve(A, X_tr_s.T @ y_c)
            all_fold_coefs.append(w / std)  # un-standardized
        except:
            pass

print(f"  Collected {len(all_fold_coefs)} fold coefficient vectors")

# Average coefficients
valid_coefs = [c for c in all_fold_coefs if len(c) == 28]
mean_coefs = np.mean(np.array(valid_coefs), axis=0) if valid_coefs else np.zeros(28)

# Sort by absolute coefficient
sort_idx = np.argsort(np.abs(mean_coefs))[::-1]
sorted_names = [FEAT_NAMES_28[i] for i in sort_idx]
sorted_coefs = mean_coefs[sort_idx]

# Color by type
colors = ['#4169E1' if name.startswith('pca_') else '#2E8B57' for name in sorted_names]

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_coefs, color=colors, alpha=0.85, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Ridge Coefficient (mean across folds and experts)', fontsize=11)
ax.set_title('Feature Importance: Ridge Coefficients (28 features)', fontsize=13, fontweight='bold')

sp_patch = mpatches.Patch(color='#2E8B57', label='Signal Processing (8)')
pca_patch = mpatches.Patch(color='#4169E1', label='CNN Embedding PCA (20)')
ax.legend(handles=[sp_patch, pca_patch], loc='lower right', fontsize=10)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, axis='x', alpha=0.3)

fig.savefig(str(RESULTS_DIR / 'figure_feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved figure_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Bland-Altman plots
# ══════════════════════════════════════════════════════════════════════════
print("\n[6] Generating Figure 5: Bland-Altman plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for col, (ptype, expert_vals, pred_vals) in enumerate([
    ('lpd', lpd_expert, lpd_pred),
    ('gpd', gpd_expert, gpd_pred),
]):
    ax = axes[col]
    mean_vals = (pred_vals + expert_vals) / 2
    diff_vals = pred_vals - expert_vals

    bias = np.mean(diff_vals)
    sd = np.std(diff_vals)
    upper_loa = bias + 1.96 * sd
    lower_loa = bias - 1.96 * sd

    color = '#2E8B57' if ptype == 'lpd' else '#4169E1'
    ax.scatter(mean_vals, diff_vals, c=color, alpha=0.5, s=25, edgecolors='none')

    xlims = [np.min(mean_vals) - 0.3, np.max(mean_vals) + 0.3]
    ax.axhline(y=bias, color='red', linestyle='-', linewidth=1.5, label=f'Bias = {bias:.3f}')
    ax.axhline(y=upper_loa, color='gray', linestyle='--', linewidth=1, label=f'+1.96 SD = {upper_loa:.3f}')
    ax.axhline(y=lower_loa, color='gray', linestyle='--', linewidth=1, label=f'-1.96 SD = {lower_loa:.3f}')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.fill_between(xlims, lower_loa, upper_loa, alpha=0.08, color='gray')

    n = len(expert_vals)
    ax.set_xlabel('Mean of Algorithm and Expert Consensus (Hz)', fontsize=10)
    ax.set_ylabel('Difference: Algorithm - Expert Consensus (Hz)', fontsize=10)
    ax.set_title(f'{ptype.upper()}: Bland-Altman Plot (n={n})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(str(RESULTS_DIR / 'figure_bland_altman_final.png'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved figure_bland_altman_final.png")


# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 70)
print(f"  1. {RESULTS_DIR / 'figure_final_scatter.png'}")
print(f"  2. {RESULTS_DIR / 'figure_progress.png'}")
print(f"  3. {RESULTS_DIR / 'figure_method_comparison.png'}")
print(f"  4. {RESULTS_DIR / 'figure_feature_importance.png'}")
print(f"  5. {RESULTS_DIR / 'figure_bland_altman_final.png'}")
print()
print(f"Best model (dl_emb_pca20_a1):")
print(f"  LPD: Spearman={best['lpd_spearman_r']} (pooled={best['lpd_spearman_pooled']}), MAE={best['lpd_mae']}")
print(f"  GPD: Spearman={best['gpd_spearman_r']} (pooled={best['gpd_spearman_pooled']}), MAE={best['gpd_mae']}")
print(f"  Combined Spearman: {best['combined_spearman']}")
