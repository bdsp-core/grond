#!/usr/bin/env python3
"""Generate spatial extent scatter plots (like Fig 6 but for spatial involvement).

2×4 layout: rows = PDCharacterizer/RDA-PLV vs Tautan et al., cols = LPD/GPD/LRDA/GRDA.
Expert GT = mean spatial_extent across available raters (LB, PH, SZ).
Algorithm predictions computed at threshold=0.62.

Usage:
    conda run -n morgoth python paper_materials/generate_fig_spatial_scatter.py
"""

import sys, numpy as np, pandas as pd, scipy.io as sio
from pathlib import Path
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'

FS = 200
PD_THRESHOLD = 0.62   # Optimized for PD (PDCharacterizer channel_probs)
RDA_THRESHOLD = 0.15  # Optimized for RDA (PLV×Amp) post SZ cleanup

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])

SUBTYPE_COLORS = {
    'lpd': '#F29030',
    'gpd': '#F0D020',
    'lrda': '#7CB342',
    'grda': '#81D4FA',
}


def load_eeg(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None, None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] != 19:
        return None, None
    mono = seg
    bipolar = mono[BIPOLAR_INDICES[:, 0]] - mono[BIPOLAR_INDICES[:, 1]]
    return mono, bipolar


def main():
    print("Loading data...")
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    ann = ann.merge(sl[['mat_file', 'subtype', 'excluded']], on='mat_file', how='left')
    ann = ann[ann.excluded != True]
    sl = sl[sl.excluded != True]

    # Build per-rater spatial extent lookup
    RATERS = ['LB', 'PH', 'SZ', 'MW']
    RATER_COLORS = {'LB': '#2196F3', 'PH': '#E91E63', 'SZ': '#4CAF50', 'MW': '#FF9800'}  # blue, pink, green, orange
    RATER_MARKERS = {'LB': 'o', 'PH': 's', 'SZ': '^', 'MW': 'D'}

    spat = ann[ann.spatial_extent.notna() & ann.rater.isin(RATERS)].copy()
    spat['spatial_extent'] = pd.to_numeric(spat['spatial_extent'], errors='coerce')

    # Per-rater per-segment lookup
    rater_lookup = {}  # mat_file -> {rater: spatial_extent}
    for _, row in spat.iterrows():
        mf = row['mat_file']
        if mf not in rater_lookup:
            rater_lookup[mf] = {}
        rater_lookup[mf][row['rater']] = float(row['spatial_extent'])

    # Require at least 2 raters
    rater_lookup = {mf: d for mf, d in rater_lookup.items() if len(d) >= 2}

    # Map subtypes
    sub_lookup = dict(zip(sl.mat_file, sl.subtype))
    freq_lookup = dict(zip(sl.mat_file, sl.pdchar_freq_hz))

    # Collect qualifying segments
    segments = []
    for mf, rater_vals in rater_lookup.items():
        sub = sub_lookup.get(mf)
        if sub not in ('lpd', 'gpd', 'lrda', 'grda'):
            continue
        segments.append({
            'mat_file': mf,
            'subtype': sub,
            'rater_vals': rater_vals,  # {rater: spatial_extent}
            'gt_mean': float(np.mean(list(rater_vals.values()))),
            'freq_hz': float(freq_lookup.get(mf, np.nan)),
        })

    print(f"  {len(segments)} segments with spatial GT")
    for sub in ['lpd', 'gpd', 'lrda', 'grda']:
        n = sum(1 for s in segments if s['subtype'] == sub)
        print(f"    {sub.upper()}: {n}")

    # Run inference
    from pd_characterizer import PDCharacterizer
    import pd_detect_alternate as pddeta
    from rda_spatial_extent import rda_spatial_extent

    pc = PDCharacterizer()

    print("\nRunning inference...")
    for i, seg in enumerate(segments):
        if (i + 1) % 50 == 0 or (i + 1) == len(segments):
            print(f"  {i+1}/{len(segments)}...")

        mono, bipolar = load_eeg(seg['mat_file'])
        if mono is None:
            seg['pred_ours'] = np.nan
            seg['pred_tautan'] = np.nan
            continue

        # Our method
        sub = seg['subtype']
        try:
            if sub in ('lpd', 'gpd'):
                result = pc.characterize(bipolar[:18], subtype=sub)
                probs = np.array(result['channel_probs'])
                seg['pred_ours'] = float(np.sum(probs > PD_THRESHOLD)) / 18.0
            else:
                freq = seg['freq_hz']
                if not np.isfinite(freq) or freq <= 0:
                    seg['pred_ours'] = np.nan
                else:
                    result = rda_spatial_extent(bipolar[:18], freq, threshold=RDA_THRESHOLD, metric='plv_amp')
                    seg['pred_ours'] = float(result['spatial_extent'])
        except Exception:
            seg['pred_ours'] = np.nan

        # Tautan
        try:
            result_t = pddeta.pd_detect_alternate(mono.copy(), FS, pk_detect='apd')
            if isinstance(result_t, dict):
                se = result_t.get('spatial_extent', np.nan)
            else:
                se = getattr(result_t, 'spatial_extent', np.nan)
            if se is not None and np.isfinite(se):
                seg['pred_tautan'] = float(se)
            else:
                seg['pred_tautan'] = np.nan
        except Exception:
            seg['pred_tautan'] = np.nan

    # Plot
    print("\nGenerating figure...")
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), facecolor='white')
    subtypes = [('lpd', 'LPD'), ('gpd', 'GPD'), ('lrda', 'LRDA'), ('grda', 'GRDA')]
    method_names = ['Ours\n(PDChar / RDA-PLV)', 'Signal Proc.\n(Tautan et al.)']
    pred_keys = ['pred_ours', 'pred_tautan']

    for row_idx, (method_name, pred_key) in enumerate(zip(method_names, pred_keys)):
        for col_idx, (sub, label) in enumerate(subtypes):
            ax = axes[row_idx][col_idx]
            d = [s for s in segments if s['subtype'] == sub and np.isfinite(s.get(pred_key, np.nan))]

            if not d:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            rng = np.random.RandomState(42)
            jitter_scale = 0.015

            # Plot per-rater dots: each segment gets up to 3 dots (one per expert)
            n_dots = 0
            for rater in RATERS:
                gt_vals, pr_vals = [], []
                for x in d:
                    if rater in x['rater_vals']:
                        gt_vals.append(x['rater_vals'][rater])
                        pr_vals.append(x[pred_key])
                if not gt_vals:
                    continue
                gt_j = np.array(gt_vals) + rng.uniform(-jitter_scale, jitter_scale, len(gt_vals))
                pr_j = np.array(pr_vals) + rng.uniform(-jitter_scale, jitter_scale, len(pr_vals))
                ax.scatter(gt_j, pr_j, alpha=0.7, s=18,
                           color=RATER_COLORS[rater], edgecolors='none',
                           marker=RATER_MARKERS[rater], zorder=2,
                           label=f'{rater} (n={len(gt_vals)})')
                n_dots += len(gt_vals)

            # Compute stats using mean expert GT vs prediction
            gt_mean = np.array([x['gt_mean'] for x in d])
            pred = np.array([x[pred_key] for x in d])
            rho, _ = spearmanr(gt_mean, pred)
            mae = np.mean(np.abs(gt_mean - pred))

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if row_idx == 1: ax.set_xlabel('Expert Spatial Extent', fontsize=9)
            if col_idx == 0: ax.set_ylabel('Predicted Spatial Extent', fontsize=9)
            if row_idx == 0:
                ax.set_title(f'{label} (n={len(d)})\n\u03c1={rho:.3f}, MAE={mae:.3f}', fontsize=11, fontweight='bold')
            else:
                ax.set_title(f'\u03c1={rho:.3f}, MAE={mae:.3f}', fontsize=10)
            ax.legend(fontsize=6, loc='upper left')

    fig.text(0.01, 0.73, 'Ours\n(PDChar / RDA-PLV)', fontsize=12, fontweight='bold',
             va='center', ha='left', rotation=90, color='#333')
    fig.text(0.01, 0.28, 'Signal Proc.\n(Tautan et al.)', fontsize=12, fontweight='bold',
             va='center', ha='left', rotation=90, color='#333')

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    out = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig_spatial_scatter.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')

    print(f'\nSummary:')
    for sub, label in subtypes:
        d = [s for s in segments if s['subtype'] == sub and np.isfinite(s.get('pred_ours', np.nan))]
        if not d: continue
        gt = np.array([x['gt_mean'] for x in d])
        pr = np.array([x['pred_ours'] for x in d])
        rho, _ = spearmanr(gt, pr)
        mae = np.mean(np.abs(gt - pr))
        print(f'  {label}: {len(d)} segs, Ours rho={rho:.3f} MAE={mae:.3f}')


if __name__ == '__main__':
    main()
