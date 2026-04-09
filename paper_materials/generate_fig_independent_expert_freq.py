#!/usr/bin/env python3
"""Generate figure S4: stratification-frequency histograms for the four
independent-expert annotation tasks (LPD, GPD, LRDA, GRDA).

Reads the four task manifests under
`paper_materials/independent_expert_tasks/<subtype>/manifest.csv` and writes
a 2x2 PNG to `paper_materials/figures/figS4_independent_expert_freq_distribution.png`.

The histograms document the case selection used for the planned independent
expert validation study (Reviewer Notes #1 and #2): 200 segments per subtype,
one per unique patient, stratified into 0.25-Hz bins across [0.5, 3.0) Hz.
PD bins are computed from MW's expert_freq_hz column (the only PD frequency
column populated in segment_labels.csv); RDA bins are computed from
algo_freq_hz, which is independent of any MW labels. The bin assignment is
used only to balance coverage; the viewer's pre-filled default at run time
still comes from the PDProfiler / RDA-Profiler, so the new raters never see
MW's labels.

Usage:
    conda run -n morgoth python paper_materials/generate_fig_independent_expert_freq.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'figS4_independent_expert_freq_distribution.png'

SUBS = ['lpd', 'gpd', 'lrda', 'grda']
PRETTY = {'lpd': 'LPD', 'gpd': 'GPD', 'lrda': 'LRDA', 'grda': 'GRDA'}
COLORS = {'lpd': '#cc3344', 'gpd': '#dd6622', 'lrda': '#3366cc', 'grda': '#229966'}

LO, HI, BIN_W = 0.5, 3.0, 0.25
TARGET_PER_BIN = 20  # 200 segments / 10 bins


def load_manifest_freqs(sub):
    """Return (list of strat_freq_hz floats, source label) for one subtype."""
    path = TASKS_DIR / sub / 'manifest.csv'
    freqs = []
    source = ''
    with open(path) as f:
        for row in csv.DictReader(f):
            source = row.get('strat_freq_source', '?')
            v = (row.get('strat_freq_hz') or '').strip()
            if v and v not in ('nan', 'None'):
                try:
                    freqs.append(float(v))
                except ValueError:
                    pass
    return freqs, source


def main():
    edges = np.arange(LO, HI + BIN_W / 2, BIN_W)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    fig.suptitle(
        'Stratification frequency distributions in the four independent-expert task manifests\n'
        '(n=200 segments per task, one segment per unique patient)',
        fontsize=13, fontweight='bold', y=0.995,
    )

    for ax, sub in zip(axes.flat, SUBS):
        freqs, source = load_manifest_freqs(sub)
        counts, _ = np.histogram(freqs, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts, width=BIN_W * 0.9, color=COLORS[sub],
               edgecolor='black', linewidth=0.6, alpha=0.85)
        ax.set_title(f'{PRETTY[sub]}  (n={len(freqs)}, source: {source})', fontsize=11)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Number of segments')
        ax.set_xticks(edges)
        ax.set_xticklabels([f'{e:.2f}' for e in edges], rotation=45, ha='right', fontsize=8)
        ax.set_xlim(LO - BIN_W / 2, HI + BIN_W / 2)
        ax.axhline(TARGET_PER_BIN, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.grid(axis='y', alpha=0.3)
        for c, n in zip(centers, counts):
            if n > 0:
                ax.text(c, n + 0.5, str(int(n)), ha='center', va='bottom', fontsize=8)

    axes[0, 0].text(
        0.02, 0.98,
        f'gray dashed line: {TARGET_PER_BIN} cases/bin (uniform-200 target)',
        transform=axes[0, 0].transAxes, fontsize=8, va='top',
        color='gray', style='italic',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=140, bbox_inches='tight')
    print(f"Saved {OUT_PATH.relative_to(PROJECT_DIR)}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == '__main__':
    main()
