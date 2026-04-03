#!/usr/bin/env python3
"""Generate Fig 6: Frequency scatter plots with quality-filtered labels.

Quality filter includes segments where ANY of:
1. MW reviewed (rater includes 'MW' in annotations.csv)
2. LB+PH+SZ all provided frequency labels
3. >=10 IIIC expert votes with >=80% agreement on pattern type

Usage:
    conda run -n morgoth python paper_materials/generate_fig6.py
"""

import sys, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

def load_quality_filter():
    """Build quality filter from annotations + IIIC votes."""
    ann = pd.read_csv(PROJECT_DIR / 'data' / 'labels' / 'annotations.csv')
    sl = pd.read_csv(PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv')

    # Per-segment rater sets
    has_freq = ann[ann.frequency_hz.notna()].copy()
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

    def passes(sid):
        raters = rater_info.get(sid, set())
        # 1. MW reviewed
        if 'MW' in raters:
            return True
        # 2. LB+PH+SZ all present
        if {'LB', 'PH', 'SZ'}.issubset(raters):
            return True
        # 3. >=10 votes with >=80% agreement
        nv, pf = vote_info.get(sid, (0, 0))
        if nv >= 10 and pf >= 0.80:
            return True
        return False

    return passes


def main():
    data = json.load(open(PROJECT_DIR / 'paper_materials' / 'fig6_frequency_data.json'))
    passes_quality = load_quality_filter()

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')
    subtypes = [('lpd', 'LPD'), ('gpd', 'GPD'), ('lrda', 'LRDA'), ('grda', 'GRDA')]
    method_names = ['PDCharacterizer', 'Signal Proc.\n(Tautan et al.)']
    pred_keys = ['pred_cnn', 'pred_alex']

    for row_idx, (method_name, pred_key) in enumerate(zip(method_names, pred_keys)):
        for col_idx, (sub, label) in enumerate(subtypes):
            ax = axes[row_idx][col_idx]
            d = data[sub]
            d_filtered = [x for x in d if passes_quality(x['segment_id'])]

            if not d_filtered:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            multi = [x for x in d_filtered if x['n_raters'] >= 3]
            single = [x for x in d_filtered if x['n_raters'] < 3]

            if single:
                ax.scatter([x['gt'] for x in single], [x[pred_key] for x in single],
                           alpha=0.25, s=12, color='#aaaaaa', edgecolors='none',
                           label=f'1-2 raters (n={len(single)})')
            if multi:
                color = '#e74c3c' if sub in ('lpd', 'lrda') else '#3498db'
                ax.scatter([x['gt'] for x in multi], [x[pred_key] for x in multi],
                           alpha=0.5, s=20, color=color, edgecolors='none',
                           label=f'\u22653 raters (n={len(multi)})')

            gt = [x['gt'] for x in d_filtered]
            pred = [x[pred_key] for x in d_filtered]
            rho, _ = spearmanr(gt, pred)
            mae = np.mean(np.abs(np.array(gt) - np.array(pred)))

            ax.plot([0, 3], [0, 3], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.set_aspect('equal')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if row_idx == 1: ax.set_xlabel('Expert Frequency (Hz)', fontsize=9)
            if col_idx == 0: ax.set_ylabel('Predicted Frequency (Hz)', fontsize=9)
            if row_idx == 0:
                ax.set_title(f'{label} (n={len(d_filtered)})\n\u03c1={rho:.3f}, MAE={mae:.3f}', fontsize=11, fontweight='bold')
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
        d_f = [x for x in data[sub] if passes_quality(x['segment_id'])]
        if not d_f: continue
        gt = [x['gt'] for x in d_f]
        pc = [x['pred_cnn'] for x in d_f]
        rho, _ = spearmanr(gt, pc)
        mae = np.mean(np.abs(np.array(gt) - np.array(pc)))
        print(f'  {label}: {len(d_f)}/{len(data[sub])} pass filter, PDChar rho={rho:.3f} MAE={mae:.3f}')


if __name__ == '__main__':
    main()
