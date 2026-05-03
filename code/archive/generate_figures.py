#!/usr/bin/env python3
"""Generate comparison figures using proper patient-level CV predictions."""

import json
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_DIR = 'results'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load patient-CV predictions ──────────────────────────────────────────────
with open('results/optimization_runs/dl_proper_patient_cv_SP_only.json') as f:
    cv = json.load(f)

lpd_pred = np.array(cv['lpd_pred_vals'])
lpd_expert = np.array(cv['lpd_expert_vals'])
gpd_pred = np.array(cv['gpd_pred_vals'])
gpd_expert = np.array(cv['gpd_expert_vals'])

# ── Load expert-expert pairwise data ─────────────────────────────────────────
def load_expert_csv(pattern_type, expert_initial):
    """Load annotation CSV for a given pattern (LPDS/GPDS) and expert."""
    prefix = f'data/_archive/pd_expert_raw/{pattern_type}_{expert_initial}_'
    matches = [f for f in os.listdir('data/_archive/pd_expert_raw') if f.startswith(f'{pattern_type}_{expert_initial}_')]
    assert len(matches) == 1, f"Expected 1 file for {pattern_type}_{expert_initial}, got {matches}"
    df = pd.read_csv(f'data/_archive/pd_expert_raw/{matches[0]}')
    # Extract a patient/segment key from the filename
    def extract_key(filepath):
        basename = os.path.basename(str(filepath))
        # Remove _score.png suffix if present
        basename = basename.replace('_score.png', '').replace('.png', '')
        return basename
    df['key'] = df['files'].apply(extract_key)
    df['freq'] = pd.to_numeric(df['frequency'], errors='coerce')
    return df.set_index('key')['freq']

experts = ['LB', 'PH', 'SZ']
pairs = [('LB', 'PH'), ('LB', 'SZ'), ('PH', 'SZ')]

def get_expert_pair_data(pattern_type, e1, e2):
    """Get paired frequency ratings where both experts rated > 0."""
    s1 = load_expert_csv(pattern_type, e1)
    s2 = load_expert_csv(pattern_type, e2)
    common = s1.index.intersection(s2.index)
    v1 = s1.loc[common].values
    v2 = s2.loc[common].values
    # Filter where both > 0
    mask = (v1 > 0) & (v2 > 0)
    return v1[mask], v2[mask]


# ── Helper functions ─────────────────────────────────────────────────────────
def add_jitter(vals, sigma=0.08):
    return vals + np.random.normal(0, sigma, len(vals))

def spearman_r(x, y):
    r, _ = stats.spearmanr(x, y)
    return r

def mae(x, y):
    return np.mean(np.abs(np.array(x) - np.array(y)))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Scatter plots (2x3 grid)
# ══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for row_idx, pattern in enumerate(['LPDS', 'GPDS']):
    pattern_label = 'LPD' if pattern == 'LPDS' else 'GPD'
    pred = lpd_pred if pattern == 'LPDS' else gpd_pred
    expert_consensus = lpd_expert if pattern == 'LPDS' else gpd_expert

    for col_idx, (e1, e2) in enumerate(pairs[:2]):
        ax = axes[row_idx, col_idx]
        v1, v2 = get_expert_pair_data(pattern, e1, e2)
        r = spearman_r(v1, v2)
        m = mae(v1, v2)
        n = len(v1)
        ax.scatter(add_jitter(v1), add_jitter(v2), alpha=0.4, s=20, color='orange', edgecolors='none')
        ax.plot([0, 4], [0, 4], 'k--', alpha=0.5, linewidth=1)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xlabel(f'Expert {e1}')
        ax.set_ylabel(f'Expert {e2}')
        ax.set_title(f'{pattern_label}: {e1} vs {e2}')
        ax.text(0.05, 0.95, f'rs={r:.3f}\nMAE={m:.2f}\nn={n}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_aspect('equal')

    # Col 3: Algorithm vs expert consensus
    ax = axes[row_idx, 2]
    color = 'green' if pattern == 'LPDS' else 'royalblue'
    r = spearman_r(pred, expert_consensus)
    m = mae(pred, expert_consensus)
    n = len(pred)
    ax.scatter(add_jitter(expert_consensus), add_jitter(pred), alpha=0.4, s=20,
               color=color, edgecolors='none')
    ax.plot([0, 4], [0, 4], 'k--', alpha=0.5, linewidth=1)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_xlabel('Expert Consensus')
    ax.set_ylabel('Algorithm Prediction')
    ax.set_title(f'{pattern_label}: Algorithm vs Experts')
    ax.text(0.05, 0.95, f'rs={r:.3f}\nMAE={m:.2f}\nn={n}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')

fig.suptitle('Patient-Level CV: Algorithm vs Expert Agreement', fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT_DIR}/figure_final_scatter.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_final_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Progress across rounds
# ══════════════════════════════════════════════════════════════════════════════
rounds =    ['R1',   'R2',   'R3',   'R4',   'R5',   'R6',   'R7*',  'DL']
combined =  [ 0.33,   0.36,   0.42,   0.42,   0.42,   0.45,   0.43,   0.428]
lpd_vals =  [ 0.29,   0.43,   0.43,   0.38,   0.38,   0.42,   0.39,   0.385]
gpd_vals =  [ 0.45,   0.40,   0.49,   0.49,   0.49,   0.51,   0.52,   0.516]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(rounds))
ax.plot(x, combined, 'ko-', linewidth=2, markersize=8, label='Combined', zorder=5)
ax.plot(x, lpd_vals, 's-', color='green', linewidth=2, markersize=8, label='LPD', zorder=5)
ax.plot(x, gpd_vals, 'D-', color='royalblue', linewidth=2, markersize=8, label='GPD', zorder=5)

# Expert-expert reference lines
ax.axhline(0.525, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axhline(0.476, color='royalblue', linestyle='--', alpha=0.6, linewidth=1.5)
ax.axhline(0.50, color='gray', linestyle='--', alpha=0.6, linewidth=1.5)

ax.text(len(rounds)-0.5, 0.528, 'Expert LPD (0.525)', fontsize=8, color='green', va='bottom')
ax.text(len(rounds)-0.5, 0.479, 'Expert GPD (0.476)', fontsize=8, color='royalblue', va='bottom')
ax.text(len(rounds)-0.5, 0.503, 'Expert Mean (0.50)', fontsize=8, color='gray', va='bottom')

# Shading: LOO-CV region vs patient-CV region
ax.axvspan(-0.5, 5.5, alpha=0.05, color='red', label='_nolegend_')
ax.axvspan(5.5, 7.5, alpha=0.05, color='green', label='_nolegend_')
ax.text(2.5, 0.22, 'LOO-CV\n(optimistic)', ha='center', fontsize=9, color='red', alpha=0.7)
ax.text(6.5, 0.22, 'Patient-CV\n(honest)', ha='center', fontsize=9, color='green', alpha=0.7)

# Annotate breakthroughs
ax.annotate('Spectral features\nintroduced', xy=(2, 0.42), xytext=(2, 0.55),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
ax.annotate('Deep learning\n+ patient-CV', xy=(7, 0.428), xytext=(7, 0.56),
            fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(rounds)
ax.set_xlabel('Optimization Round')
ax.set_ylabel('Spearman Correlation')
ax.set_title('Optimization Progress: LOO-CV (R1-R6) vs Patient-CV (R7+, DL)')
ax.set_ylim(0.2, 0.6)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figure_progress.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_progress.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Method comparison bar chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Method A\nBaseline', 'Best SP\n(Patient-CV)', 'Expert-Expert']
lpd_bars = [0.29, 0.385, 0.525]   # R1 LPD as baseline
gpd_bars = [0.45, 0.516, 0.476]

bar_width = 0.3
x = np.arange(len(methods))
bars1 = ax.bar(x - bar_width/2, lpd_bars, bar_width, label='LPD', color='green', alpha=0.8)
bars2 = ax.bar(x + bar_width/2, gpd_bars, bar_width, label='GPD', color='royalblue', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Spearman Correlation')
ax.set_title('Method Comparison: Honest Patient-CV Results')
ax.set_ylim(0, 0.65)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/figure_method_comparison.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_method_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Bland-Altman plots
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (pred, expert_vals, label, color) in enumerate([
    (lpd_pred, lpd_expert, 'LPD', 'green'),
    (gpd_pred, gpd_expert, 'GPD', 'royalblue'),
]):
    ax = axes[idx]
    mean_vals = (pred + expert_vals) / 2
    diff_vals = pred - expert_vals
    bias = np.mean(diff_vals)
    sd = np.std(diff_vals, ddof=1)
    upper = bias + 1.96 * sd
    lower = bias - 1.96 * sd

    ax.scatter(mean_vals, diff_vals, alpha=0.4, s=20, color=color, edgecolors='none')
    ax.axhline(bias, color='red', linestyle='-', linewidth=1.5, label=f'Bias={bias:.3f}')
    ax.axhline(upper, color='gray', linestyle='--', linewidth=1,
               label=f'+1.96 SD={upper:.3f}')
    ax.axhline(lower, color='gray', linestyle='--', linewidth=1,
               label=f'-1.96 SD={lower:.3f}')
    ax.set_xlabel('Mean of Algorithm & Expert')
    ax.set_ylabel('Algorithm - Expert')
    ax.set_title(f'Bland-Altman: {label} (Patient-CV)')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle('Bland-Altman Analysis: Patient-Level CV Predictions', fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(f'{OUT_DIR}/figure_bland_altman_final.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved figure_bland_altman_final.png")

print("\nAll figures generated successfully.")
