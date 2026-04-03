"""
Generate Fig 5: Spatial Agreement (4 rows: LPD, GPD, LRDA, GRDA).

LPD/GPD rows: full inter-rater analysis (3 raters + model).
LRDA/GRDA rows: model vs single rater (SZ only).

Layout: 4 rows × 3 columns
  Col 1: Jaccard matrix heatmap
  Col 2: Bar chart (rater-rater vs model-rater)
  Col 3: All pairwise agreements

Usage:
    nohup conda run -n morgoth python -u paper_materials/generate_fig5.py > /tmp/fig5_4row.log 2>&1 &
"""

import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig5_spatial_agreement.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']
SPATIAL_THRESHOLD = 0.38

# Colors
GREEN = '#2ecc71'
ORANGE = '#f39c12'
GRAY = '#95a5a6'


def parse_regions(spatial_str):
    """Parse spatial_channels string into a set of region names."""
    if pd.isna(spatial_str) or str(spatial_str).strip() == '':
        return set()
    import re
    # Handle multiple delimiters: spaces, commas, periods, semicolons
    tokens = re.split(r'[\s,;.]+', str(spatial_str).strip())
    # Normalize and keep only valid region names
    valid = set()
    for t in tokens:
        t = t.strip().upper()
        if t in ('LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO',
                 'FL', 'FR', 'FT', 'TL', 'TR'):
            # Map alternate names
            remap = {'FL': 'LF', 'FR': 'RF', 'FT': 'LT', 'TL': 'LT', 'TR': 'RT'}
            valid.add(remap.get(t, t))
    return valid


def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return 1.0
    return len(set_a & set_b) / len(union)


def load_annotations():
    """Load annotations and segment labels."""
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    # Merge subtype into annotations
    ann = ann.merge(sl[['mat_file', 'subtype']], on='mat_file', how='left')
    return ann, sl


def get_rater_regions(ann, subtype, raters):
    """Get region sets per segment per rater for a given subtype.

    Returns: dict[segment_id] -> dict[rater] -> set of regions
    """
    mask = (ann['subtype'] == subtype) & \
           (ann['spatial_channels'].notna()) & \
           (ann['spatial_channels'] != '') & \
           (ann['rater'].isin(raters))
    sub = ann[mask].copy()

    data = defaultdict(dict)
    for _, row in sub.iterrows():
        regions = parse_regions(row['spatial_channels'])
        if len(regions) > 0:
            data[row['segment_id']][row['rater']] = regions
    return data


def run_model_on_segments(segment_ids_mat, subtype, sl):
    """Run PDCharacterizer on segments, return dict[segment_id] -> set of regions."""
    from pd_characterizer import PDCharacterizer
    charzer = PDCharacterizer()

    results = {}
    total = len(segment_ids_mat)
    for i, (seg_id, mat_file) in enumerate(segment_ids_mat):
        if i % 50 == 0:
            print(f"  [{subtype.upper()}] Processing {i}/{total}...", flush=True)
        mat_path = EEG_DIR / mat_file
        if not mat_path.exists():
            continue
        try:
            mat = sio.loadmat(str(mat_path))
            dk = [k for k in mat if not k.startswith('_')][0]
            seg = mat[dk].astype(np.float64)
            if seg.shape[0] > seg.shape[1]:
                seg = seg.T
            seg = seg[:18, :2000]
            if seg.shape[1] < 2000:
                continue

            result = charzer.characterize(seg, subtype=subtype)
            # Apply threshold
            involved = set()
            for region, score in result['region_scores'].items():
                if score > SPATIAL_THRESHOLD:
                    involved.add(region)
            results[seg_id] = involved
        except Exception as e:
            print(f"  Error on {seg_id}: {e}", flush=True)
            continue
    print(f"  [{subtype.upper()}] Done: {len(results)}/{total} segments.", flush=True)
    return results


def run_alexandra_on_segments(segment_ids_mat, subtype, sl):
    """Run Tautan et al. signal processing on segments, return dict[segment_id] -> set of regions."""
    from pd_pointiness_acf import pd_detect_pointiness_acf

    results = {}
    total = len(segment_ids_mat)
    for i, (seg_id, mat_file) in enumerate(segment_ids_mat):
        if i % 50 == 0:
            print(f"  [{subtype.upper()} Tautan] Processing {i}/{total}...", flush=True)
        mat_path = EEG_DIR / mat_file
        if not mat_path.exists():
            continue
        try:
            mat = sio.loadmat(str(mat_path))
            dk = [k for k in mat if not k.startswith('_')][0]
            seg = mat[dk].astype(np.float64)
            if seg.shape[0] > seg.shape[1]:
                seg = seg.T
            if seg.shape[0] < 19 or seg.shape[1] < 2000:
                continue

            alex = pd_detect_pointiness_acf(seg[:19, :2000], fs=200)
            spatial_areas = alex.get('spatial_areas', [])
            results[seg_id] = set(spatial_areas)
        except Exception as e:
            continue
    print(f"  [{subtype.upper()} Tautan] Done: {len(results)}/{total} segments.", flush=True)
    return results


def compute_pairwise_jaccard(data, rater_a, rater_b):
    """Compute mean Jaccard between two raters across shared segments."""
    jaccards = []
    for seg_id, rater_dict in data.items():
        if rater_a in rater_dict and rater_b in rater_dict:
            j = jaccard(rater_dict[rater_a], rater_dict[rater_b])
            jaccards.append(j)
    if len(jaccards) == 0:
        return np.nan, 0
    return np.mean(jaccards), len(jaccards)


def plot_row_full(axes, subtype_label, data, raters, model_key='CNN+PLV'):
    """Plot a full inter-rater row (LPD/GPD style): matrix, bars, pairwise.
    Includes both CNN+PLV and Tautan models."""
    all_keys = raters + ['Tautan', model_key]
    n = len(all_keys)

    # ── Col 1: Jaccard matrix ──
    ax = axes[0]
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                val, cnt = compute_pairwise_jaccard(data, all_keys[i], all_keys[j])
                matrix[i, j] = val if not np.isnan(val) else 0
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(n))
    ax.set_xticklabels(all_keys, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_keys, fontsize=8)
    # Separator line between raters and models
    ax.axhline(len(raters) - 0.5, color='black', linewidth=1.5)
    ax.axvline(len(raters) - 0.5, color='black', linewidth=1.5)
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
    ax.set_title(f'{subtype_label} — Jaccard Matrix', fontsize=11, fontweight='bold')

    # ── Col 2: Bar chart (rater-rater vs Tautan-rater vs CNN+PLV-rater) ──
    ax = axes[1]
    rr_scores = []
    tautan_scores = []
    cnn_scores = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            val, cnt = compute_pairwise_jaccard(data, raters[i], raters[j])
            if not np.isnan(val):
                rr_scores.append(val)
        val_t, _ = compute_pairwise_jaccard(data, 'Tautan', raters[i])
        if not np.isnan(val_t):
            tautan_scores.append(val_t)
        val_c, _ = compute_pairwise_jaccard(data, model_key, raters[i])
        if not np.isnan(val_c):
            cnn_scores.append(val_c)

    GRAY = '#95a5a6'
    means = [np.mean(rr_scores) if rr_scores else 0,
             np.mean(tautan_scores) if tautan_scores else 0,
             np.mean(cnn_scores) if cnn_scores else 0]
    stds = [np.std(rr_scores) if len(rr_scores) > 1 else 0,
            np.std(tautan_scores) if len(tautan_scores) > 1 else 0,
            np.std(cnn_scores) if len(cnn_scores) > 1 else 0]
    bars = ax.bar([0, 1, 2], means, yerr=stds, color=[GREEN, GRAY, ORANGE],
                  edgecolor='black', linewidth=0.5, capsize=5, width=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Rater-\nRater', 'Tautan\net al.', 'CNN+\nPLV'], fontsize=8)
    ax.set_ylabel('Mean Jaccard', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=7)
    ax.set_title(f'{subtype_label} — Agreement Summary', fontsize=11, fontweight='bold')
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{m:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Col 3: All pairwise ──
    ax = axes[2]
    pairs = []
    pair_labels = []
    pair_colors = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            val, cnt = compute_pairwise_jaccard(data, raters[i], raters[j])
            if not np.isnan(val):
                pairs.append(val)
                pair_labels.append(f'{raters[i]}-{raters[j]}\n(n={cnt})')
                pair_colors.append(GREEN)
    for r in raters:
        val, cnt = compute_pairwise_jaccard(data, model_key, r)
        if not np.isnan(val):
            pairs.append(val)
            pair_labels.append(f'Model-{r}\n(n={cnt})')
            pair_colors.append(ORANGE)

    x_pos = range(len(pairs))
    ax.bar(x_pos, pairs, color=pair_colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, fontsize=7)
    ax.set_ylabel('Mean Jaccard', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{subtype_label} — All Pairwise', fontsize=11, fontweight='bold')
    for xi, val in zip(x_pos, pairs):
        ax.text(xi, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold')


def plot_row_single_rater(axes, subtype_label, data, rater):
    """Plot a single-rater row (LRDA/GRDA style): 3×3 matrix, bars, pairwise.
    Includes both CNN+PLV and Tautan models vs single rater."""
    note_text = f"Single rater ({rater}) — inter-rater analysis not available"
    all_keys = [rater, 'Tautan', 'CNN+PLV']
    n = 3

    # ── Col 1: 3×3 Jaccard matrix ──
    ax = axes[0]
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                val, cnt = compute_pairwise_jaccard(data, all_keys[i], all_keys[j])
                matrix[i, j] = val if not np.isnan(val) else 0
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlOrRd', aspect='equal')
    ax.set_xticks(range(n))
    ax.set_xticklabels(all_keys, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_keys, fontsize=9)
    ax.axhline(0.5, color='black', linewidth=1.5)
    ax.axvline(0.5, color='black', linewidth=1.5)
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            color = 'white' if v > 0.6 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')
    ax.set_title(f'{subtype_label} — Jaccard Matrix', fontsize=11, fontweight='bold')
    ax.text(0.5, -0.25, note_text, transform=ax.transAxes,
            ha='center', fontsize=8, style='italic', color='gray')

    # ── Col 2: Two bars (Tautan-rater vs CNN+PLV-rater) ──
    ax = axes[1]
    GRAY = '#95a5a6'
    val_t, cnt_t = compute_pairwise_jaccard(data, 'Tautan', rater)
    val_c, cnt_c = compute_pairwise_jaccard(data, 'CNN+PLV', rater)
    val_t = val_t if not np.isnan(val_t) else 0
    val_c = val_c if not np.isnan(val_c) else 0
    bars = ax.bar([0, 1], [val_t, val_c], color=[GRAY, ORANGE],
                  edgecolor='black', linewidth=0.5, width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Tautan\nvs {rater}', f'CNN+PLV\nvs {rater}'], fontsize=8)
    ax.set_ylabel('Mean Jaccard', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=9, fontweight='bold')
    ax.set_title(f'{subtype_label} — Agreement Summary', fontsize=11, fontweight='bold')
    ax.text(0.5, -0.25, note_text, transform=ax.transAxes,
            ha='center', fontsize=8, style='italic', color='gray')

    # ── Col 3: Single pairwise ──
    ax = axes[2]
    ax.bar([0], [val], color=[ORANGE], edgecolor='black', linewidth=0.5, width=0.4)
    ax.set_xticks([0])
    ax.set_xticklabels([f'Model-{rater}\n(n={cnt})'], fontsize=8)
    ax.set_ylabel('Mean Jaccard', fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{subtype_label} — All Pairwise', fontsize=11, fontweight='bold')
    ax.text(0, val + 0.02, f'{val:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')


def main():
    print("=" * 60)
    print("Fig 5: Spatial Agreement (4 rows)")
    print("=" * 60)

    ann, sl = load_annotations()

    # ── Collect rater spatial data ──
    # LPD/GPD: raters LB, PH, SZ
    pd_raters = ['LB', 'PH', 'SZ']

    print("\nCollecting rater spatial annotations...")
    lpd_data = get_rater_regions(ann, 'lpd', pd_raters)
    gpd_data = get_rater_regions(ann, 'gpd', pd_raters)
    lrda_data = get_rater_regions(ann, 'lrda', ['SZ'])
    grda_data = get_rater_regions(ann, 'grda', ['SZ'])
    print(f"  LPD: {len(lpd_data)} segments with spatial annotations")
    print(f"  GPD: {len(gpd_data)} segments with spatial annotations")
    print(f"  LRDA: {len(lrda_data)} segments with spatial annotations")
    print(f"  GRDA: {len(grda_data)} segments with spatial annotations")

    # ── Collect segments needing model predictions ──
    # For LPD/GPD, we need all segments that have at least one rater annotation
    # For LRDA/GRDA, we need segments with SZ annotation
    all_seg_ids = set()
    for d in [lpd_data, gpd_data, lrda_data, grda_data]:
        all_seg_ids.update(d.keys())

    # Build segment_id -> (mat_file, subtype) mapping
    seg_to_info = {}
    for _, row in sl.iterrows():
        seg_id = row['mat_file'].replace('.mat', '')
        seg_to_info[seg_id] = (row['mat_file'], row['subtype'])

    # Run both models on each subtype
    for subtype, data_dict in [('lpd', lpd_data), ('gpd', gpd_data),
                                ('lrda', lrda_data), ('grda', grda_data)]:
        seg_mat_pairs = []
        for seg_id in data_dict:
            if seg_id in seg_to_info:
                mat_file, st = seg_to_info[seg_id]
                seg_mat_pairs.append((seg_id, mat_file))
            else:
                seg_mat_pairs.append((seg_id, seg_id + '.mat'))

        # PDCharacterizer (CNN+PLV)
        print(f"\nRunning PDCharacterizer on {subtype.upper()} segments...")
        model_results = run_model_on_segments(seg_mat_pairs, subtype, sl)
        for seg_id, regions in model_results.items():
            if seg_id in data_dict:
                data_dict[seg_id]['CNN+PLV'] = regions

        # Tautan et al. (signal processing)
        print(f"Running Tautan et al. on {subtype.upper()} segments...")
        alex_results = run_alexandra_on_segments(seg_mat_pairs, subtype, sl)
        for seg_id, regions in alex_results.items():
            if seg_id in data_dict:
                data_dict[seg_id]['Tautan'] = regions

    # ── Filter to segments with both models + rater(s) ──
    def filter_with_models(data, required_raters):
        """Keep only segments that have both models + at least one required rater."""
        filtered = {}
        for seg_id, rd in data.items():
            if 'CNN+PLV' in rd and 'Tautan' in rd and any(r in rd for r in required_raters):
                filtered[seg_id] = rd
        return filtered

    lpd_data = filter_with_models(lpd_data, pd_raters)
    gpd_data = filter_with_models(gpd_data, pd_raters)
    lrda_data = filter_with_models(lrda_data, ['SZ'])
    grda_data = filter_with_models(grda_data, ['SZ'])

    print(f"\nAfter model + rater filtering:")
    print(f"  LPD: {len(lpd_data)} segments")
    print(f"  GPD: {len(gpd_data)} segments")
    print(f"  LRDA: {len(lrda_data)} segments")
    print(f"  GRDA: {len(grda_data)} segments")

    # ── Create figure ──
    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 18), facecolor='white')
    gs = gridspec.GridSpec(4, 3, hspace=0.45, wspace=0.3,
                           left=0.06, right=0.96, top=0.95, bottom=0.04)

    # Row 0: LPD (3 raters + 2 models)
    axes_lpd = [fig.add_subplot(gs[0, c]) for c in range(3)]
    plot_row_full(axes_lpd, 'LPD', lpd_data, pd_raters, model_key='CNN+PLV')

    # Row 1: GPD
    axes_gpd = [fig.add_subplot(gs[1, c]) for c in range(3)]
    plot_row_full(axes_gpd, 'GPD', gpd_data, pd_raters, model_key='CNN+PLV')

    # Row 2: LRDA (SZ + 2 models)
    axes_lrda = [fig.add_subplot(gs[2, c]) for c in range(3)]
    plot_row_single_rater(axes_lrda, 'LRDA', lrda_data, 'SZ')

    # Row 3: GRDA
    axes_grda = [fig.add_subplot(gs[3, c]) for c in range(3)]
    plot_row_single_rater(axes_grda, 'GRDA', grda_data, 'SZ')

    fig.suptitle('Figure 5: Spatial Agreement — Region-Level Jaccard Similarity',
                 fontsize=14, fontweight='bold', y=0.98)

    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()
