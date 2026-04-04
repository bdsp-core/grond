#!/usr/bin/env python3
"""Generate Fig 6: Frequency scatter plots with quality-filtered labels.

Reads model predictions directly from segment_labels.csv (pdchar_freq_hz, tautan_freq_hz)
— no model inference needed. Regeneration takes ~2 seconds.

Quality filter includes segments where ANY of:
1. MW reviewed (rater includes 'MW' in annotations.csv)
2. LB+PH+SZ all provided frequency labels
3. >=10 IIIC expert votes with >=80% agreement on pattern type

Usage:
    conda run -n morgoth python paper_materials/generate_fig6.py
"""

import sys, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent


def load_data_and_filter():
    """Load expert freq + model predictions from segment_labels.csv, apply quality filter."""
    ann = pd.read_csv(PROJECT_DIR / 'data' / 'labels' / 'annotations.csv')
    sl = pd.read_csv(PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv')

    # Expert frequency: mean across all raters in annotations.csv
    has_freq = ann[ann.frequency_hz.notna()].copy()
    has_freq['frequency_hz'] = pd.to_numeric(has_freq['frequency_hz'], errors='coerce')
    freq_agg = has_freq.groupby('segment_id').agg(
        mean_freq=('frequency_hz', 'mean'),
        n_raters=('rater', 'nunique'),
    ).reset_index()
    expert_freq = dict(zip(freq_agg.segment_id, zip(freq_agg.mean_freq, freq_agg.n_raters)))

    # Also MW-only from segment_labels
    for _, row in sl[sl.expert_freq_rater == 'MW'].iterrows():
        sid = row['mat_file'].replace('.mat', '')
        if sid not in expert_freq and pd.notna(row.get('expert_freq_hz')):
            expert_freq[sid] = (float(row['expert_freq_hz']), 1)

    # Per-segment rater sets (for quality filter)
    freq_raters = has_freq.groupby('segment_id').agg(
        raters=('rater', lambda x: set(x)),
    ).reset_index()
    rater_info = dict(zip(freq_raters.segment_id, freq_raters.raters))

    # IIIC vote info
    vote_info = {}
    for _, row in sl.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        nv = pd.to_numeric(row.get('iiic_n_votes'), errors='coerce')
        pf = pd.to_numeric(row.get('iiic_plurality_frac'), errors='coerce')
        if np.isfinite(nv):
            vote_info[sid] = (int(nv), float(pf) if np.isfinite(pf) else 0)

    def passes_quality(sid):
        raters = rater_info.get(sid, set())
        if 'MW' in raters:
            return True
        if {'LB', 'PH', 'SZ'}.issubset(raters):
            return True
        nv, pf = vote_info.get(sid, (0, 0))
        if nv >= 10 and pf >= 0.80:
            return True
        return False

    # Build results from segment_labels columns (no inference needed!)
    results = {sub: [] for sub in ['lpd', 'gpd', 'lrda', 'grda']}
    for _, row in sl.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        subtype = row.get('subtype')
        if subtype not in results:
            continue
        if row.get('excluded') == True:
            continue
        pdchar = pd.to_numeric(row.get('pdchar_freq_hz'), errors='coerce')
        tautan = pd.to_numeric(row.get('tautan_freq_hz'), errors='coerce')
        if not np.isfinite(pdchar):
            continue
        if sid not in expert_freq:
            continue
        gt, n_raters = expert_freq[sid]
        if not np.isfinite(gt) or gt <= 0:
            continue

        results[subtype].append({
            'segment_id': sid,
            'gt': float(gt),
            'pred_cnn': float(pdchar),
            'pred_alex': float(tautan) if np.isfinite(tautan) else np.nan,
            'n_raters': int(n_raters),
            'passes_quality': passes_quality(sid),
        })

    return results


def main():
    print("Loading data from segment_labels.csv (no inference needed)...")
    results = load_data_and_filter()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')
    subtypes = [('lpd', 'LPD'), ('gpd', 'GPD'), ('lrda', 'LRDA'), ('grda', 'GRDA')]
    method_names = ['PDCharacterizer', 'Signal Proc.\n(Tautan et al.)']
    pred_keys = ['pred_cnn', 'pred_alex']

    # IIIC-standard colors matching the Jing et al. UMAP figure
    SUBTYPE_COLORS = {
        'lpd': '#F29030',    # bright warm orange
        'gpd': '#F0D020',    # warm bright yellow
        'lrda': '#7CB342',   # olive/lime green
        'grda': '#81D4FA',   # light sky blue
    }

    for row_idx, (method_name, pred_key) in enumerate(zip(method_names, pred_keys)):
        for col_idx, (sub, label) in enumerate(subtypes):
            ax = axes[row_idx][col_idx]
            d_all = [x for x in results[sub] if x['passes_quality']]
            # Filter to points with finite prediction for this method
            d = [x for x in d_all if np.isfinite(x[pred_key])]

            if not d:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            multi = [x for x in d if x['n_raters'] >= 3]
            single = [x for x in d if x['n_raters'] < 3]
            base_color = SUBTYPE_COLORS[sub]

            # Add jitter to prevent overplotting
            rng = np.random.RandomState(42)
            jitter_scale = 0.05

            # Single-rater: circles, subtype color, opaque
            if single:
                gt_j = np.array([x['gt'] for x in single]) + rng.uniform(-jitter_scale, jitter_scale, len(single))
                pr_j = np.array([x[pred_key] for x in single]) + rng.uniform(-jitter_scale, jitter_scale, len(single))
                ax.scatter(gt_j, pr_j,
                           alpha=1.0, s=25, color=base_color, edgecolors='none',
                           marker='o',
                           label=f'1-2 raters (n={len(single)})')
            # Multi-rater: stars, opaque, no border
            if multi:
                gt_j = np.array([x['gt'] for x in multi]) + rng.uniform(-jitter_scale, jitter_scale, len(multi))
                pr_j = np.array([x[pred_key] for x in multi]) + rng.uniform(-jitter_scale, jitter_scale, len(multi))
                ax.scatter(gt_j, pr_j,
                           alpha=1.0, s=60, color='black', edgecolors='none',
                           marker='*',
                           label=f'\u22653 raters (n={len(multi)})')

            gt = [x['gt'] for x in d]
            pred = [x[pred_key] for x in d]
            rho, _ = spearmanr(gt, pred)
            mae = np.mean(np.abs(np.array(gt) - np.array(pred)))

            ax.plot([0, 3], [0, 3], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect('equal')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if row_idx == 1: ax.set_xlabel('Expert Frequency (Hz)', fontsize=9)
            if col_idx == 0: ax.set_ylabel('Predicted Frequency (Hz)', fontsize=9)
            if row_idx == 0:
                ax.set_title(f'{label} (n={len(d)})\n\u03c1={rho:.3f}, MAE={mae:.3f}', fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'\u03c1={rho:.3f}, MAE={mae:.3f}', fontsize=10)
            ax.legend(fontsize=6, loc='upper left')

    fig.text(0.01, 0.73, 'PDCharacterizer\n(CNN+PLV+DP)', fontsize=12, fontweight='bold',
             va='center', ha='left', rotation=90, color='#333')
    fig.text(0.01, 0.28, 'Signal Proc.\n(Tautan et al.)', fontsize=12, fontweight='bold',
             va='center', ha='left', rotation=90, color='#333')

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    out = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig6_frequency_scatter.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')

    for sub, label in subtypes:
        d = [x for x in results[sub] if x['passes_quality']]
        if not d: continue
        gt = [x['gt'] for x in d]
        pc = [x['pred_cnn'] for x in d]
        rho, _ = spearmanr(gt, pc)
        mae = np.mean(np.abs(np.array(gt) - np.array(pc)))
        print(f'  {label}: {len(d)}/{len(results[sub])} pass filter, PDChar rho={rho:.3f} MAE={mae:.3f}')


if __name__ == '__main__':
    main()
