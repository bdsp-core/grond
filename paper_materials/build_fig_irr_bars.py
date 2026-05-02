#!/usr/bin/env python3
"""Independent-expert IRR bar-plot figure (Fig 5): paired EE vs EA with CIs + significance.

Reads results/independent_expert_v1/summary.json (the V14 canonical run) and
re-bootstraps segment-level resamples so each bar gets a proper 95% CI on
its mean. Significance comparisons (EA vs EE) reuse the paired-segment
bootstrap p-values already in summary.json.

Output:
    paper_materials/figures/fig5_irr_comparison.png
    paper_materials/figures/fig5_irr_comparison.pdf

    conda run -n morgoth python paper_materials/build_fig_irr_bars.py
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
from analyze_independent_expert_v1 import (  # type: ignore
    icc31, spearman, mae, cohen_kappa, percent_agree,
    build_label_tables, EE_PAIRS, EA_PAIRS, SUBTYPES, LAT_TASKS, PRETTY,
)

OUT_PNG = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig5_irr_comparison.png'
OUT_PDF = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig5_irr_comparison.pdf'
SUMMARY_JSON = PROJECT_DIR / 'results' / 'independent_expert_v1' / 'summary.json'

EE_COLOR = '#4a6fa5'    # muted blue
EA_COLOR = '#d97834'    # warm orange
GRID_GRAY = '#d8d8d8'

mpl.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10


def _metric_for_pair(label_a, label_b, metric):
    common = sorted(set(label_a) & set(label_b))
    if len(common) < 5:
        return np.nan
    if metric in ('icc', 'spearman', 'mae'):
        a = np.array([label_a[mf] for mf in common], dtype=float)
        b = np.array([label_b[mf] for mf in common], dtype=float)
        if metric == 'icc':
            return icc31(a, b)
        if metric == 'spearman':
            return spearman(a, b)
        return mae(a, b)
    a = [label_a[mf] for mf in common]
    b = [label_b[mf] for mf in common]
    if metric == 'kappa':
        return cohen_kappa(a, b)
    return percent_agree(a, b)


def bootstrap_means(tables, sub, mtype, metric, n_boot=2000, seed=42):
    """Returns (ee_mean_point, ee_mean_ci, ea_mean_point, ea_mean_ci, ee_pairs, ea_pairs)
    where each *_pairs is the list of 3 pair-wise point estimates."""
    tab = tables[sub][mtype]
    all_segs = sorted(set().union(*[set(tab[r]) for r in ('MW', 'SZ', 'TZ', 'ALGO')]))
    if not all_segs:
        return None
    # Point estimates per pair
    ee_vals = [_metric_for_pair(tab[a], tab[b], metric) for a, b in EE_PAIRS]
    ea_vals = [_metric_for_pair(tab[a], tab[b], metric) for a, b in EA_PAIRS]
    ee_clean = [v for v in ee_vals if not (isinstance(v, float) and np.isnan(v))]
    ea_clean = [v for v in ea_vals if not (isinstance(v, float) and np.isnan(v))]
    if not ee_clean or not ea_clean:
        return None
    ee_mean = float(np.mean(ee_clean))
    ea_mean = float(np.mean(ea_clean))

    rng = np.random.default_rng(seed)
    segs_arr = np.array(all_segs)
    n = len(segs_arr)
    ee_dist, ea_dist = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samp_segs = segs_arr[idx]
        samp = {r: {} for r in ('MW', 'SZ', 'TZ', 'ALGO')}
        for j, mf in enumerate(samp_segs):
            for r in ('MW', 'SZ', 'TZ', 'ALGO'):
                if mf in tab[r]:
                    samp[r][f'{mf}__{j}'] = tab[r][mf]
        eb = [_metric_for_pair(samp[a], samp[b], metric) for a, b in EE_PAIRS]
        ab = [_metric_for_pair(samp[a], samp[b], metric) for a, b in EA_PAIRS]
        eb = [v for v in eb if not (isinstance(v, float) and np.isnan(v))]
        ab = [v for v in ab if not (isinstance(v, float) and np.isnan(v))]
        if eb:
            ee_dist.append(float(np.mean(eb)))
        if ab:
            ea_dist.append(float(np.mean(ab)))
    ee_dist = np.array(ee_dist); ea_dist = np.array(ea_dist)
    ee_ci = (float(np.percentile(ee_dist, 2.5)), float(np.percentile(ee_dist, 97.5)))
    ea_ci = (float(np.percentile(ea_dist, 2.5)), float(np.percentile(ea_dist, 97.5)))
    return ee_mean, ee_ci, ea_mean, ea_ci, ee_clean, ea_clean


def stars_for_p(p):
    if p is None:
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def main():
    print('Loading consensus tables (V14 canonical, majority-accept)...')
    tables = build_label_tables(consensus='majority')
    # Override LRDA ALGO with V14 (V12 freq + V14 hybrid laterality)
    v14_path = PROJECT_DIR / 'data' / 'labels' / 'independent_expert_v1' / 'v14_predictions.json'
    with open(v14_path) as f:
        v14 = json.load(f)
    for sid, e in v14.items():
        mf = e.get('mat_file')
        if mf in tables['lrda']['freq']['ALGO']:
            tables['lrda']['freq']['ALGO'][mf] = float(e['v14_freq'])
        if mf in tables['lrda']['lat']['ALGO']:
            tables['lrda']['lat']['ALGO'][mf] = e['v14_laterality']

    # Read significance from canonical summary.json (already V14)
    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    sig = summary['ee_vs_ea_significance']

    # Build the rows we want to show
    # (task, metric, mtype, label) — display in this order
    panels = {
        'A. Frequency  (ICC)': [
            ('lpd',  'icc',  'freq', 'LPD'),
            ('gpd',  'icc',  'freq', 'GPD'),
            ('grda', 'icc',  'freq', 'GRDA'),
            ('lrda', 'icc',  'freq', 'LRDA'),
        ],
        'B. Frequency  (Spearman ρ)': [
            ('lpd',  'spearman', 'freq', 'LPD'),
            ('gpd',  'spearman', 'freq', 'GPD'),
            ('grda', 'spearman', 'freq', 'GRDA'),
            ('lrda', 'spearman', 'freq', 'LRDA'),
        ],
        'C. Laterality (Cohen κ)': [
            ('lpd',  'kappa', 'lat', 'LPD'),
            ('lrda', 'kappa', 'lat', 'LRDA'),
        ],
    }

    # Compute bar data for each panel
    print('Bootstrapping mean EE / mean EA per (task, metric)...')
    bar_data = {}
    for panel_title, rows in panels.items():
        bar_data[panel_title] = []
        for sub, metric, mtype, label in rows:
            res = bootstrap_means(tables, sub, mtype, metric, n_boot=2000)
            if res is None:
                continue
            ee_pt, ee_ci, ea_pt, ea_ci, ee_pairs, ea_pairs = res
            sig_entry = sig.get(sub, {}).get(mtype, {}).get(metric, {})
            p = sig_entry.get('p_two_sided') if sig_entry else None
            delta = sig_entry.get('delta_point') if sig_entry else None
            bar_data[panel_title].append({
                'label': label, 'sub': sub, 'metric': metric,
                'ee_mean': ee_pt, 'ee_ci': ee_ci, 'ee_pairs': ee_pairs,
                'ea_mean': ea_pt, 'ea_ci': ea_ci, 'ea_pairs': ea_pairs,
                'p': p, 'delta': delta,
            })

    # ---------------- Plot ----------------
    n_panels = len(bar_data)
    # Width per task: ~1.5 inch each
    panel_widths = [max(3.2, 1.6 * len(rows)) for rows in bar_data.values()]
    total_w = sum(panel_widths) + 0.5 * (n_panels - 1) + 1.0
    fig = plt.figure(figsize=(total_w, 4.6))
    gs = fig.add_gridspec(1, n_panels, width_ratios=panel_widths, wspace=0.25,
                           left=0.06, right=0.985, top=0.86, bottom=0.14)

    BAR_W = 0.36
    for pi, (panel_title, rows) in enumerate(bar_data.items()):
        ax = fig.add_subplot(gs[0, pi])
        n_tasks = len(rows)
        x = np.arange(n_tasks)
        ee_means = [r['ee_mean'] for r in rows]
        ea_means = [r['ea_mean'] for r in rows]
        ee_err = np.array([[r['ee_mean'] - r['ee_ci'][0], r['ee_ci'][1] - r['ee_mean']] for r in rows]).T
        ea_err = np.array([[r['ea_mean'] - r['ea_ci'][0], r['ea_ci'][1] - r['ea_mean']] for r in rows]).T

        bars_ee = ax.bar(x - BAR_W/2, ee_means, BAR_W, color=EE_COLOR,
                          edgecolor='#1d2d44', linewidth=0.6,
                          yerr=ee_err, capsize=3, ecolor='#1d2d44',
                          error_kw={'elinewidth': 0.9}, label='Expert–Expert')
        bars_ea = ax.bar(x + BAR_W/2, ea_means, BAR_W, color=EA_COLOR,
                          edgecolor='#7a3a10', linewidth=0.6,
                          yerr=ea_err, capsize=3, ecolor='#7a3a10',
                          error_kw={'elinewidth': 0.9}, label='Expert–Algorithm')

        # Overlay individual pair points (open circles)
        for i, r in enumerate(rows):
            for v in r['ee_pairs']:
                ax.plot(i - BAR_W/2, v, 'o', mfc='white', mec='#1d2d44',
                         mew=0.9, ms=4.5, zorder=5)
            for v in r['ea_pairs']:
                ax.plot(i + BAR_W/2, v, 'o', mfc='white', mec='#7a3a10',
                         mew=0.9, ms=4.5, zorder=5)

        # Significance brackets
        for i, r in enumerate(rows):
            top = max(r['ee_ci'][1], r['ea_ci'][1])
            bracket_y = top + 0.04
            text_y = bracket_y + 0.012
            ax.plot([i - BAR_W/2, i - BAR_W/2, i + BAR_W/2, i + BAR_W/2],
                     [bracket_y - 0.008, bracket_y, bracket_y, bracket_y - 0.008],
                     '-', color='#444444', lw=0.9)
            star = stars_for_p(r['p'])
            sign = '+' if (r['delta'] is not None and r['delta'] > 0) else '−'
            label = f'{sign}{star}' if star not in ('', 'n.s.') else 'n.s.'
            color = '#207a3a' if (r['delta'] is not None and r['delta'] > 0
                                   and r['p'] is not None and r['p'] < 0.05) else (
                    '#aa2a2a' if (r['delta'] is not None and r['delta'] < 0
                                   and r['p'] is not None and r['p'] < 0.05) else '#666666')
            ax.text(i, text_y, label, ha='center', va='bottom',
                     fontsize=10, color=color, fontweight='bold')

        # Cosmetics
        ax.set_xticks(x)
        ax.set_xticklabels([r['label'] for r in rows], fontsize=10)
        ax.set_ylim(0.0, 1.08)
        ax.set_yticks(np.arange(0.0, 1.05, 0.2))
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID_GRAY, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(panel_title, fontsize=11, loc='left', pad=8)
        if pi == 0:
            ax.set_ylabel('Inter-rater reliability', fontsize=10)
        if pi == n_panels - 1:
            ax.legend(loc='lower right', fontsize=8.5, frameon=True, framealpha=0.92,
                      edgecolor='#cccccc')

    # Subtitle / footer note
    fig.text(0.5, 0.02,
             'Stars: paired segment-level bootstrap (2000 resamples) of mean(EA) − mean(EE).  '
             '+ favors algorithm; − favors experts.  '
             '*** p<0.001  ** p<0.01  * p<0.05.  Bars: 95% CI on the mean from the same bootstrap.  '
             'Open circles: per-pair point estimates (3 EE pairs, 3 EA pairs).',
             ha='center', va='bottom', fontsize=8, color='#444444')

    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {OUT_PNG.relative_to(PROJECT_DIR)}')
    print(f'Saved {OUT_PDF.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
