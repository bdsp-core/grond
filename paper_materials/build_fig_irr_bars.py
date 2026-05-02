#!/usr/bin/env python3
"""Independent-expert IRR bar-plot figure (Fig 5): paired EE vs EA with CIs + significance.

Two-row layout:
  Row 1: Canonical 4-rater independent-expert cohort (MW, SZ, TZ, AS) under
         the canonical majority-accept consensus rule (V14 algorithm).
         Panel A: Frequency Spearman rho (LPD, GPD, GRDA, LRDA).
         Panel B: Laterality Cohen kappa (LPD, LRDA).
  Row 2: Prior 4-rater Tautan et al. (2025) cohort (PH, LB, SZ, MW), where
         PH/LB/SZ labels were produced without the present interactive
         narrowband-overlay tools (PH/LB never labeled laterality).
         Panel C: Frequency Spearman rho (LPD, GPD, GRDA, LRDA).

Each bar is a paired segment-level bootstrap (n=2000) of the mean across
the within-role pairs. Error bars are 95% CI on the mean from that
bootstrap. Open circles are individual pair-wise point estimates
(canonical: 6 EE pairs + 4 EA pairs; prior: 6 EE pairs + 4 EA pairs).
Significance brackets test mean(EA) - mean(EE), sign-corrected so + favors
algorithm; *** p<0.001, ** p<0.01, * p<0.05; n.s. otherwise.

Output:
    paper_materials/figures/fig5_irr_comparison.png
    paper_materials/figures/fig5_irr_comparison.pdf

    conda run -n morgoth python paper_materials/build_fig_irr_bars.py
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
from analyze_independent_expert_v1 import (  # type: ignore
    icc31, spearman, mae, cohen_kappa, percent_agree,
    build_label_tables, EE_PAIRS, EA_PAIRS, SUBTYPES, LAT_TASKS, PRETTY,
)

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
OUT_PNG = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig5_irr_comparison.png'
OUT_PDF = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig5_irr_comparison.pdf'
SUMMARY_JSON = PROJECT_DIR / 'results' / 'independent_expert_v1' / 'summary.json'

GRID_GRAY = '#ececec'
ANNOT_COLOR = '#333333'

# Canonical IIIC subtype palette (matches Fig 4).
SUBTYPE_COLORS = {
    'lpd':  '#F29030',    # bright warm orange
    'gpd':  '#F0D020',    # warm bright yellow
    'lrda': '#7CB342',    # olive / lime green
    'grda': '#81D4FA',    # light sky blue
}
SUBTYPE_EDGE = {
    'lpd':  '#9c5a17',
    'gpd':  '#8a7405',
    'lrda': '#3d5e22',
    'grda': '#236f8c',
}
EE_ALPHA = 0.30    # Both EE and EA bars are translucent so the per-pair
EA_ALPHA = 0.65    #   open circles read clearly on top.

mpl.rcParams['font.family'] = ['Helvetica', 'Arial', 'DejaVu Sans']
mpl.rcParams['font.size'] = 10


# ---------------- Per-pair metric helper (works for any rater set) ----------------

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


def bootstrap_means(tab, ee_pairs, ea_pairs, raters_all, metric, n_boot=2000, seed=42):
    """Paired segment-level bootstrap of mean(EE pairs) and mean(EA pairs).

    Args:
        tab: dict[rater] -> dict[mat_file] -> value (one role's table for one task).
             Must include every rater referenced in ee_pairs / ea_pairs.
        ee_pairs: list of (rater_a, rater_b) within experts.
        ea_pairs: list of (rater, 'ALGO').
        raters_all: list of rater names whose tables are bootstrapped.
        metric: 'icc' | 'spearman' | 'mae' | 'kappa' | 'percent'.

    Returns:
        (ee_mean, ee_ci, ea_mean, ea_ci, ee_pair_pts, ea_pair_pts, delta, p_two)
    """
    all_segs = sorted(set().union(*[set(tab.get(r, {})) for r in raters_all]))
    if not all_segs:
        return None
    ee_vals = [_metric_for_pair(tab.get(a, {}), tab.get(b, {}), metric) for a, b in ee_pairs]
    ea_vals = [_metric_for_pair(tab.get(a, {}), tab.get(b, {}), metric) for a, b in ea_pairs]
    ee_clean = [v for v in ee_vals if not (isinstance(v, float) and np.isnan(v))]
    ea_clean = [v for v in ea_vals if not (isinstance(v, float) and np.isnan(v))]
    if not ee_clean or not ea_clean:
        return None
    ee_mean = float(np.mean(ee_clean))
    ea_mean = float(np.mean(ea_clean))

    rng = np.random.default_rng(seed)
    segs_arr = np.array(all_segs)
    n = len(segs_arr)
    ee_dist, ea_dist, delta_dist = [], [], []
    sign = -1.0 if metric == 'mae' else 1.0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samp_segs = segs_arr[idx]
        samp = {r: {} for r in raters_all}
        for j, mf in enumerate(samp_segs):
            for r in raters_all:
                if mf in tab.get(r, {}):
                    samp[r][f'{mf}__{j}'] = tab[r][mf]
        eb = [_metric_for_pair(samp[a], samp[b], metric) for a, b in ee_pairs]
        ab = [_metric_for_pair(samp[a], samp[b], metric) for a, b in ea_pairs]
        eb = [v for v in eb if not (isinstance(v, float) and np.isnan(v))]
        ab = [v for v in ab if not (isinstance(v, float) and np.isnan(v))]
        if eb and ab:
            mee = float(np.mean(eb))
            mea = float(np.mean(ab))
            ee_dist.append(mee)
            ea_dist.append(mea)
            delta_dist.append(sign * (mea - mee))
    if not ee_dist:
        return None
    ee_ci = (float(np.percentile(ee_dist, 2.5)), float(np.percentile(ee_dist, 97.5)))
    ea_ci = (float(np.percentile(ea_dist, 2.5)), float(np.percentile(ea_dist, 97.5)))
    delta_arr = np.array(delta_dist)
    p_two = float(min(1.0, 2 * min(np.mean(delta_arr <= 0), np.mean(delta_arr >= 0))))
    delta_pt = sign * (ea_mean - ee_mean)
    return ee_mean, ee_ci, ea_mean, ea_ci, ee_clean, ea_clean, delta_pt, p_two


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


# ---------------- Build prior cohort tables (PH/LB/SZ/MW) ----------------

PRIOR_RATERS = ('PH', 'LB', 'SZ', 'MW', 'ALGO')


def build_prior_cohort_tables():
    """Return tables[sub]['freq'][rater][mat_file] = value for the prior
    PH/LB/SZ/MW Tautan cohort. Algorithm freq is `pdchar_freq_hz` for PD
    subtypes and `algo_freq_hz` for RDA subtypes (both refreshed from the
    production pipeline)."""
    labels = pd.read_csv(LABELS_DIR / 'labels.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    seg = pd.read_csv(LABELS_DIR / 'segments.csv',
                      usecols=['mat_file', 'pdchar_freq_hz'])
    sl = sl.merge(seg, on='mat_file', how='left')
    sl_idx = sl.set_index('mat_file')

    fr = labels[labels.label_type == 'frequency_hz'].copy()
    fr['value'] = pd.to_numeric(fr['value'], errors='coerce')
    fr = fr[fr.value.notna() & (fr.value > 0)]

    tables = {sub: {'freq': {r: {} for r in PRIOR_RATERS}} for sub in SUBTYPES}
    for sub in SUBTYPES:
        sub_mfs = set(sl[sl.subtype == sub].mat_file)
        for r in ('PH', 'LB', 'SZ', 'MW'):
            sub_fr = fr[(fr.rater == r) & fr.mat_file.isin(sub_mfs)]
            for _, row in sub_fr.iterrows():
                tables[sub]['freq'][r][row['mat_file']] = float(row.value)
        # Algorithm freq
        algo_col = 'pdchar_freq_hz' if sub in ('lpd', 'gpd') else 'algo_freq_hz'
        for mf in sub_mfs:
            if mf in sl_idx.index:
                v = pd.to_numeric(sl_idx.loc[mf, algo_col], errors='coerce')
                if pd.notna(v) and v > 0:
                    tables[sub]['freq']['ALGO'][mf] = float(v)
    return tables


# ---------------- Plot helper ----------------

def draw_panel(ax, rows, *, panel_title, y_label=False,
                bar_w=0.36, group_step=1.0, ylim=(0.0, 1.08), yticks=None,
                xlim=None):
    """Render a panel of paired EE/EA bars colored by IIIC pattern subtype.

    Layout uses a two-level x-axis: per-bar EE/EA tick labels, with
    subtype names placed below as group labels. Bar width is held
    constant across panels via `bar_w`; `group_step` controls the
    horizontal distance between adjacent subtype groups so 2-task and
    4-task panels render with comparable visual density.

    Significance annotations show a single asterisk symbol (*, **, ***)
    only for p < 0.05; non-significant comparisons are omitted from the
    plot and reported in the caption / text instead.
    """
    if not rows:
        ax.axis('off')
        return
    n_tasks = len(rows)
    centers = np.arange(n_tasks) * group_step

    for i, r in enumerate(rows):
        sub = r['sub']
        face = SUBTYPE_COLORS[sub]
        edge = SUBTYPE_EDGE[sub]
        ee_err = np.array([[r['ee_mean'] - r['ee_ci'][0]],
                           [r['ee_ci'][1] - r['ee_mean']]])
        ea_err = np.array([[r['ea_mean'] - r['ea_ci'][0]],
                           [r['ea_ci'][1] - r['ea_mean']]])
        ax.bar(centers[i] - bar_w/2, r['ee_mean'], bar_w,
                color=face, alpha=EE_ALPHA,
                edgecolor=edge, linewidth=0.7,
                yerr=ee_err, capsize=3, ecolor=edge,
                error_kw={'elinewidth': 0.9})
        ax.bar(centers[i] + bar_w/2, r['ea_mean'], bar_w,
                color=face, alpha=EA_ALPHA,
                edgecolor=edge, linewidth=0.7,
                yerr=ea_err, capsize=3, ecolor=edge,
                error_kw={'elinewidth': 0.9})

    # Per-pair point estimates as open circles, with horizontal jitter
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows):
        edge = SUBTYPE_EDGE[r['sub']]
        n_ee = len(r['ee_pairs'])
        n_ea = len(r['ea_pairs'])
        ee_jit = (rng.random(n_ee) - 0.5) * (bar_w * 0.55)
        ea_jit = (rng.random(n_ea) - 0.5) * (bar_w * 0.55)
        for j, v in enumerate(r['ee_pairs']):
            ax.plot(centers[i] - bar_w/2 + ee_jit[j], v, 'o',
                     mfc='white', mec=edge, mew=1.1, ms=6.0, zorder=5)
        for j, v in enumerate(r['ea_pairs']):
            ax.plot(centers[i] + bar_w/2 + ea_jit[j], v, 'o',
                     mfc='white', mec=edge, mew=1.1, ms=6.0, zorder=5)

    # Significance annotations: only significant comparisons, neutral color,
    # placed at a consistent y across all groups in this panel.
    bracket_y = ylim[1] - 0.06
    text_y = bracket_y + 0.006
    for i, r in enumerate(rows):
        if r['p'] is None or r['p'] >= 0.05:
            continue
        star = stars_for_p(r['p'])
        ax.plot([centers[i] - bar_w/2, centers[i] - bar_w/2,
                  centers[i] + bar_w/2, centers[i] + bar_w/2],
                 [bracket_y - 0.012, bracket_y, bracket_y, bracket_y - 0.012],
                 '-', color=ANNOT_COLOR, lw=0.9)
        ax.text(centers[i], text_y, star, ha='center', va='bottom',
                 fontsize=12, color=ANNOT_COLOR, fontweight='bold')

    # Two-level x-axis: EE/EA per bar, subtype group label below.
    bar_xs = []
    bar_lbls = []
    for i in range(n_tasks):
        bar_xs.extend([centers[i] - bar_w/2, centers[i] + bar_w/2])
        bar_lbls.extend(['EE', 'EA'])
    ax.set_xticks(bar_xs)
    ax.set_xticklabels(bar_lbls, fontsize=9.5, color='#333333')
    ax.tick_params(axis='x', length=0, pad=2)
    for i, r in enumerate(rows):
        ax.text(centers[i], -0.105, r['label'],
                 ha='center', va='top', fontsize=12, fontweight='bold',
                 color='#1c1c1c', transform=ax.get_xaxis_transform(),
                 clip_on=False)

    if xlim is None:
        xlim = (centers[0] - group_step / 2, centers[-1] + group_step / 2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID_GRAY, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#cfcfcf')
    ax.spines['left'].set_color('#cfcfcf')
    ax.set_title(panel_title, fontsize=13, fontweight='bold', loc='left', pad=10)
    if y_label:
        ax.set_ylabel('Inter-rater reliability', fontsize=11)
    ax.tick_params(axis='y', labelsize=10)


def collect_panel_rows(tables, ee_pairs, ea_pairs, raters_all, panel_specs,
                        sig_lookup=None):
    """For each spec (sub, metric, mtype, label) compute bootstrap means + CIs."""
    out = []
    for sub, metric, mtype, label in panel_specs:
        res = bootstrap_means(tables[sub][mtype], ee_pairs, ea_pairs, raters_all, metric)
        if res is None:
            continue
        ee_mean, ee_ci, ea_mean, ea_ci, ee_pairs_pt, ea_pairs_pt, delta, p_two = res
        # If sig_lookup is provided (canonical summary.json), prefer those numbers
        # for consistency with the manuscript text.
        if sig_lookup is not None:
            entry = sig_lookup.get(sub, {}).get(mtype, {}).get(metric)
            if entry is not None:
                p_two = entry.get('p_two_sided', p_two)
                delta = entry.get('delta_point', delta)
        out.append({
            'label': label, 'sub': sub, 'metric': metric, 'mtype': mtype,
            'ee_mean': ee_mean, 'ee_ci': ee_ci, 'ee_pairs': ee_pairs_pt,
            'ea_mean': ea_mean, 'ea_ci': ea_ci, 'ea_pairs': ea_pairs_pt,
            'delta': delta, 'p': p_two,
        })
    return out


def main():
    print('Loading canonical 4-rater consensus tables (V14)...')
    canonical_tables = build_label_tables(consensus='majority')
    v14_path = LABELS_DIR / 'independent_expert_v1' / 'v14_predictions.json'
    with open(v14_path) as f:
        v14 = json.load(f)
    for sid, e in v14.items():
        mf = e.get('mat_file')
        if mf in canonical_tables['lrda']['freq']['ALGO']:
            canonical_tables['lrda']['freq']['ALGO'][mf] = float(e['v14_freq'])
        if mf in canonical_tables['lrda']['lat']['ALGO']:
            canonical_tables['lrda']['lat']['ALGO'][mf] = e['v14_laterality']

    with open(SUMMARY_JSON) as f:
        summary = json.load(f)
    sig_lookup = summary['ee_vs_ea_significance']

    print('Loading prior 4-rater Tautan-cohort tables (PH/LB/SZ/MW)...')
    prior_tables = build_prior_cohort_tables()
    PRIOR_EE_PAIRS = list(combinations(('PH', 'LB', 'SZ', 'MW'), 2))
    PRIOR_EA_PAIRS = [(r, 'ALGO') for r in ('PH', 'LB', 'SZ', 'MW')]

    canonical_raters = ('MW', 'SZ', 'TZ', 'AS', 'ALGO')

    print('Bootstrapping means...')
    # Panel A: canonical freq Spearman
    panel_A = collect_panel_rows(
        canonical_tables, EE_PAIRS, EA_PAIRS, canonical_raters,
        [('lpd', 'spearman', 'freq', 'LPD'),
         ('gpd', 'spearman', 'freq', 'GPD'),
         ('grda', 'spearman', 'freq', 'GRDA'),
         ('lrda', 'spearman', 'freq', 'LRDA')],
        sig_lookup=sig_lookup,
    )
    # Panel B: canonical laterality kappa
    panel_B = collect_panel_rows(
        canonical_tables, EE_PAIRS, EA_PAIRS, canonical_raters,
        [('lpd', 'kappa', 'lat', 'LPD'),
         ('lrda', 'kappa', 'lat', 'LRDA')],
        sig_lookup=sig_lookup,
    )
    # Panel C: prior freq Spearman (PH/LB/SZ/MW)
    panel_C = collect_panel_rows(
        prior_tables, PRIOR_EE_PAIRS, PRIOR_EA_PAIRS, PRIOR_RATERS,
        [('lpd', 'spearman', 'freq', 'LPD'),
         ('gpd', 'spearman', 'freq', 'GPD'),
         ('grda', 'spearman', 'freq', 'GRDA'),
         ('lrda', 'spearman', 'freq', 'LRDA')],
        sig_lookup=None,  # prior cohort isn't in canonical summary.json
    )

    # ---------------- Plot ----------------
    # Stack the three panels vertically. Bar width is identical across all
    # panels; panel C (2 groups) uses a larger group_step so its bars span
    # the same physical width as the 4-group panels above. Shared x-limits
    # also ensure the bars line up vertically.
    fig = plt.figure(figsize=(10.0, 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1],
                           hspace=0.42,
                           left=0.085, right=0.985, top=0.965, bottom=0.045)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[1, 0])
    ax_C = fig.add_subplot(gs[2, 0])

    yticks_full = np.arange(0.0, 1.05, 0.2)

    # Panel A and B use group_step=1.0 with 4 groups -> centers at 0..3.
    # Panel C uses group_step=3.0 with 2 groups -> centers at 0,3, so the
    # full extent (-1.5 .. 4.5) matches A/B (-0.5 .. 3.5 -> width 4) closely.
    n4_xlim = (-0.5, 3.5)
    n2_step = 3.0
    n2_xlim = (-1.5 + 0.5, 1 * n2_step + 1.5 - 0.5)  # (-1.0, 3.5)

    draw_panel(ax_A, panel_C,  # prior cohort freq
               panel_title='A. Prior cohort: frequency (Spearman ρ)',
               y_label=True,
               ylim=(0.0, 1.08), yticks=yticks_full,
               xlim=n4_xlim)
    draw_panel(ax_B, panel_A,  # canonical cohort freq
               panel_title='B. Canonical cohort: frequency (Spearman ρ)',
               y_label=True,
               ylim=(0.0, 1.08), yticks=yticks_full,
               xlim=n4_xlim)
    draw_panel(ax_C, panel_B,  # canonical cohort laterality
               panel_title='C. Canonical cohort: laterality (Cohen κ)',
               y_label=True,
               group_step=n2_step,
               ylim=(0.0, 1.08), yticks=yticks_full,
               xlim=n2_xlim)
    # No on-figure legend or footer text -- those belong in the figure caption.

    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight', facecolor='white')
    fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {OUT_PNG.relative_to(PROJECT_DIR)}')
    print(f'Saved {OUT_PDF.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
