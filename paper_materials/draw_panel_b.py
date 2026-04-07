#!/usr/bin/env python3
"""
Draw Panel B (Pipeline Architecture) for Fig 2.
Refined version matching PaperBanana style.

Usage:
    conda run -n morgoth python paper_materials/draw_panel_b.py
"""

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def draw_panel_b(outpath='paper_materials/figures/_panel_b.png'):
    W, H = 82, 100
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Colors ──
    TOP_FC = '#3C3C3C'; TOP_EC = '#222222'  # Dark ChannelPD-Net
    L_BG = '#E8F5E9'; L_EC = '#81C784'; L_BF = '#C8E6C9'; L_BE = '#66BB6A'  # Green
    M_BG = '#FBE9E7'; M_EC = '#EF9A9A'; M_BF = '#FFCCBC'; M_BE = '#E57373'  # Salmon
    R_BG = '#E3F2FD'; R_EC = '#90CAF9'; R_BF = '#BBDEFB'; R_BE = '#64B5F6'  # Blue
    ARR = '#444444'

    def box(cx, cy, w, h, fc, ec, rad=1.5, lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle=f'round,pad=0,rounding_size={rad}',
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4, clip_on=False))

    def T(x, y, s, fs=9.5, fw='normal', ha='center', va='center', col='black'):
        ax.text(x, y, s, fontsize=fs, fontweight=fw, ha=ha, va=va,
                color=col, zorder=6, multialignment='center', linespacing=1.15, clip_on=False)

    def arr(x1, y1, x2, y2, lw=1.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle='-|>', color=ARR, lw=lw,
                            mutation_scale=9, shrinkA=0, shrinkB=0), zorder=5, clip_on=False)

    # ── Title ──
    T(2, 97, 'B. Pipeline Architecture', fs=16, fw='bold', ha='left')

    # ── Input + ChannelPD-Net ──
    T(14, 90, '18 Independent\nBipolar Channels', fs=9, ha='right', col='#555')
    arr(15, 90, 27, 90)

    TX, TY, TW, TH = 41, 90, 26, 8
    box(TX, TY, TW, TH, TOP_FC, TOP_EC, rad=2.5, lw=1.5)
    T(TX, TY + 1.5, 'ChannelPD-Net', fs=11, fw='bold', col='white')
    T(TX, TY - 1.8, '(CNN+Attention)', fs=10, col='#CCC')

    # ── Fork ──
    LX, MX, RX = 15, 41, 67
    FORK_Y = 82
    ax.plot([TX, TX], [TY - TH/2, FORK_Y], color=ARR, lw=1.0, zorder=3)
    ax.plot([LX, RX], [FORK_Y, FORK_Y], color=ARR, lw=1.0, zorder=3)
    arr(LX, FORK_Y, LX, 77)
    arr(MX, FORK_Y, MX, 77)
    arr(RX, FORK_Y, RX, 77)

    T(28, 83.5, 'PD Probability', fs=8.5, col='#666')
    T(54, 83.5, 'Frequency Estimate', fs=8.5, col='#666')

    # ── Container panels (solid borders) ──
    PW_L, PW_M, PW_R = 22, 28, 22
    PT_Y, PB_Y = 76.5, 8
    PH = PT_Y - PB_Y

    for cx, pw, bg, ec in [(LX, PW_L, L_BG, L_EC), (MX, PW_M, M_BG, M_EC), (RX, PW_R, R_BG, R_EC)]:
        ax.add_patch(FancyBboxPatch(
            (cx - pw/2, PB_Y), pw, PH,
            boxstyle='round,pad=0,rounding_size=2',
            facecolor=bg, edgecolor=ec, linewidth=1.2, zorder=2))

    # ── LEFT: Laterality Detection ──
    T(LX, 73.5, 'Laterality\nDetection', fs=11, fw='bold', col='#2E7D32')

    box(LX, 63, 19, 8, L_BF, L_BE)
    T(LX, 63, 'Compare\nL vs R Mean\nProbabilities', fs=9)

    arr(LX, 59, LX, 52)

    T(LX, 48, 'Laterality\n(Side)', fs=10, fw='bold')

    # ── MIDDLE: Discharge Detection ──
    T(MX, 73.5, 'Discharge\nDetection', fs=11, fw='bold', col='#C62828')

    CET_X, ACF_X = MX - 7, MX + 7
    box(CET_X, 64, 12, 7, M_BF, M_BE)
    T(CET_X, 64, '8-channel\nCET-UNet', fs=9)
    box(ACF_X, 64, 12, 7, M_BF, M_BE)
    T(ACF_X, 64, 'CNN+ACF\nEnsemble', fs=9)

    T(CET_X, 57.5, 'Evidence\nTrace', fs=8, col='#888')
    T(ACF_X, 57.5, 'Frequency\nPrior', fs=8, col='#888')

    arr(CET_X, 55.5, MX - 2, 49)
    arr(ACF_X, 55.5, MX + 2, 49)

    box(MX, 45, 22, 7, M_BF, M_BE)
    T(MX, 45, 'Dynamic\nProgramming', fs=9.5)

    arr(MX, 41.5, MX, 37)

    box(MX, 33, 22, 8, M_BF, M_BE)
    T(MX, 33, 'EM Template\nRefinement &\nFiltering', fs=9)

    arr(MX, 29, MX, 23)

    T(MX, 18, r'Discharge Times ($t_1 \cdots t_n$)' '\nFrequency', fs=10, fw='bold')

    # ── RIGHT: Topographic Localization ──
    T(RX, 73.5, 'Topographic\nLocalization', fs=11, fw='bold', col='#1565C0')

    box(RX, 64, 19, 7, R_BF, R_BE)
    T(RX, 64, 'Extract\nMonopolar\nVoltage', fs=9)

    arr(RX, 60.5, RX, 56)

    box(RX, 52, 19, 8, R_BF, R_BE)
    T(RX, 52, 'Laplacian-GFP\nAlignment\n(±25ms)', fs=9)

    arr(RX, 48, RX, 44)

    box(RX, 39.5, 19, 9, R_BF, R_BE)
    T(RX, 39.5, 'Template\nRefinement &\nGFP²-weighted\nAveraging', fs=9)

    arr(RX, 35, RX, 30)

    # Topoplot icon
    from matplotlib.colors import LinearSegmentedColormap
    circle_outer = plt.Circle((RX, 25), 4.5, facecolor='#1A0033', edgecolor=R_BE, lw=1.0, zorder=4)
    ax.add_artist(circle_outer)
    circle_inner = plt.Circle((RX - 0.5, 25.5), 2.0, facecolor='#FF6F00', alpha=0.7, edgecolor='none', zorder=5)
    ax.add_artist(circle_inner)
    circle_hot = plt.Circle((RX - 0.8, 25.8), 0.8, facecolor='#FFEB3B', alpha=0.8, edgecolor='none', zorder=5)
    ax.add_artist(circle_hot)

    arr(RX, 20, RX, 16)

    T(RX, 12, 'Localization', fs=10, fw='bold')

    # ── Save ──
    fig.savefig(outpath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    draw_panel_b()
