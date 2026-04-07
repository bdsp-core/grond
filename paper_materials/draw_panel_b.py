#!/usr/bin/env python3
"""
Draw Panel B (Pipeline Architecture) for Fig 2, matching PaperBanana style.

Usage:
    conda run -n morgoth python paper_materials/draw_panel_b.py
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_panel_b(outpath='paper_materials/figures/_panel_b.png'):
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # ── Color Palette (PaperBanana-inspired) ──
    DARK_BG = '#3C3C3C'       # ChannelPD-Net and output boxes
    GREEN_CONT = '#E8F5E9'    # Laterality container
    GREEN_BOX = '#C8E6C9'     # Laterality sub-boxes
    GREEN_BD = '#81C784'
    SALMON_CONT = '#FBE9E7'   # Discharge container
    SALMON_BOX = '#FFCCBC'    # Discharge sub-boxes
    SALMON_BD = '#EF9A9A'
    BLUE_CONT = '#E3F2FD'     # Topographic container
    BLUE_BOX = '#BBDEFB'      # Topographic sub-boxes
    BLUE_BD = '#90CAF9'
    TEXT_DARK = '#2C2C2C'
    TEXT_WHITE = '#FFFFFF'
    ARROW_COLOR = '#555555'

    def rounded_box(x, y, w, h, fc, ec='#888', lw=1.0, ls='-'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                      facecolor=fc, edgecolor=ec, linewidth=lw,
                                      linestyle=ls, zorder=2)
        ax.add_patch(box)

    def arrow(x1, y1, x2, y2, lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=ARROW_COLOR, lw=lw))

    # ── Title ──
    ax.text(5, 11.6, 'B. Pipeline Architecture', fontsize=16, fontweight='bold',
            ha='center', va='center', color=TEXT_DARK)

    # ── Input label + arrow to ChannelPD-Net ──
    ax.text(1.5, 10.7, '18 Independent\nBipolar Channels', fontsize=9, ha='center',
            va='center', color='#666')
    arrow(2.5, 10.7, 3.3, 10.7, lw=1.5)

    # ── ChannelPD-Net (dark box) ──
    rounded_box(3.3, 10.1, 3.4, 1.2, fc=DARK_BG, ec='#222', lw=1.5)
    ax.text(5.0, 10.7, 'ChannelPD-Net\n(CNN+Attention)', fontsize=11, fontweight='bold',
            ha='center', va='center', color=TEXT_WHITE, zorder=3)

    # ── PD Probability / Frequency Estimate labels ──
    ax.text(3.5, 9.75, 'PD Probability', fontsize=8, ha='center', color='#666', style='italic')
    ax.text(6.5, 9.75, 'Frequency Estimate', fontsize=8, ha='center', color='#666', style='italic')

    # Horizontal connector line
    ax.plot([1.5, 8.5], [9.5, 9.5], color='#BBB', linewidth=0.8, zorder=1)
    # Vertical drops
    arrow(1.75, 9.5, 1.75, 9.0)
    arrow(5.0, 9.5, 5.0, 9.0)
    arrow(8.25, 9.5, 8.25, 9.0)

    # ══════════════════════════════════════════════
    # CONTAINER PANELS (colored backgrounds)
    # ══════════════════════════════════════════════

    # Laterality container
    rounded_box(0.3, 1.2, 2.9, 7.6, fc=GREEN_CONT, ec=GREEN_BD, lw=1.2)
    ax.text(1.75, 8.5, 'Laterality\nDetection', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#2E7D32', zorder=3)

    # Discharge container
    rounded_box(3.5, 1.2, 3.0, 7.6, fc=SALMON_CONT, ec=SALMON_BD, lw=1.2)
    ax.text(5.0, 8.5, 'Discharge\nDetection', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#C62828', zorder=3)

    # Topographic container
    rounded_box(6.8, 1.2, 2.9, 7.6, fc=BLUE_CONT, ec=BLUE_BD, lw=1.2)
    ax.text(8.25, 8.5, 'Topographic\nLocalization', fontsize=12, fontweight='bold',
            ha='center', va='center', color='#1565C0', zorder=3)

    # ══════════════════════════════════════════════
    # LATERALITY COLUMN
    # ══════════════════════════════════════════════

    rounded_box(0.6, 6.6, 2.3, 0.9, fc=GREEN_BOX, ec=GREEN_BD)
    ax.text(1.75, 7.05, 'Compare\nL vs R Mean\nProbabilities', fontsize=9, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(1.75, 6.6, 1.75, 5.8)

    ax.text(1.75, 5.3, 'Laterality\n(Side)', fontsize=10, ha='center', va='center',
            color=TEXT_DARK, fontweight='bold')

    # ══════════════════════════════════════════════
    # DISCHARGE COLUMN
    # ══════════════════════════════════════════════

    # Two top boxes side by side
    rounded_box(3.7, 6.6, 1.3, 0.9, fc=SALMON_BOX, ec=SALMON_BD)
    ax.text(4.35, 7.05, '8-channel\nCET-UNet', fontsize=8.5, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    rounded_box(5.2, 6.6, 1.3, 0.9, fc=SALMON_BOX, ec=SALMON_BD)
    ax.text(5.85, 7.05, 'CNN+ACF\nEnsemble', fontsize=8.5, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    # Labels below
    ax.text(4.35, 6.25, 'Evidence\nTrace', fontsize=8, ha='center', color='#888')
    ax.text(5.85, 6.25, 'Frequency\nPrior', fontsize=8, ha='center', color='#888')

    # Arrows converging
    arrow(4.35, 5.9, 5.0, 5.2)
    arrow(5.85, 5.9, 5.0, 5.2)

    # Dynamic Programming
    rounded_box(4.0, 4.4, 2.0, 0.8, fc=SALMON_BOX, ec=SALMON_BD)
    ax.text(5.0, 4.8, 'Dynamic\nProgramming', fontsize=9.5, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(5.0, 4.4, 5.0, 3.7)

    # EM Template Refinement
    rounded_box(3.8, 2.8, 2.4, 0.9, fc=SALMON_BOX, ec=SALMON_BD)
    ax.text(5.0, 3.25, 'EM Template\nRefinement &\nFiltering', fontsize=9, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(5.0, 2.8, 5.0, 2.2)

    ax.text(5.0, 1.7, 'Discharge Times ($t_1 \\cdots t_n$)\nFrequency', fontsize=10,
            ha='center', va='center', color=TEXT_DARK, fontweight='bold')

    # ══════════════════════════════════════════════
    # TOPOGRAPHIC COLUMN
    # ══════════════════════════════════════════════

    rounded_box(7.1, 6.6, 2.3, 0.9, fc=BLUE_BOX, ec=BLUE_BD)
    ax.text(8.25, 7.05, 'Extract\nMonopolar\nVoltage', fontsize=9, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(8.25, 6.6, 8.25, 6.0)

    rounded_box(7.1, 5.1, 2.3, 0.9, fc=BLUE_BOX, ec=BLUE_BD)
    ax.text(8.25, 5.55, 'Laplacian-GFP\nAlignment\n($\\pm$25ms)', fontsize=9, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(8.25, 5.1, 8.25, 4.5)

    rounded_box(7.1, 3.3, 2.3, 1.2, fc=BLUE_BOX, ec=BLUE_BD)
    ax.text(8.25, 3.9, 'Template\nRefinement &\nGFP$^2$-weighted\nAveraging', fontsize=9, ha='center', va='center',
            color=TEXT_DARK, zorder=3)

    arrow(8.25, 3.3, 8.25, 2.8)

    # Topoplot icon (inferno-colored circle)
    from matplotlib.colors import LinearSegmentedColormap
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.4
    cx, cy = 8.25, 2.3
    circle = plt.Circle((cx, cy), r, facecolor='#B71C1C', alpha=0.3, edgecolor='#E65100',
                         linewidth=1.0, zorder=3)
    ax.add_patch(circle)
    # Gradient effect
    inner = plt.Circle((cx-0.05, cy+0.05), r*0.5, facecolor='#FFEB3B', alpha=0.5, edgecolor='none', zorder=4)
    ax.add_patch(inner)

    arrow(8.25, 1.8, 8.25, 1.4)

    ax.text(8.25, 1.0, 'Localization', fontsize=10, ha='center', va='center',
            color=TEXT_DARK, fontweight='bold')

    # ══════════════════════════════════════════════
    # OUTPUT BOXES (dark, at very bottom)
    # ══════════════════════════════════════════════

    # These are outside the containers, in dark boxes
    for bx, label in [(1.75, 'Laterality\n(Side)'),
                       (5.0, 'Discharge Times\n($t_1 \\cdots t_n$)\nFrequency'),
                       (8.25, 'Localization')]:
        pass  # Output labels already placed inside containers above

    # ── Save ──
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    draw_panel_b()
