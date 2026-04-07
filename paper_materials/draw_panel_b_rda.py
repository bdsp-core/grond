#!/usr/bin/env python3
"""Draw Panel B for Fig 3 (RDA Pipeline) — matching Fig 2 style."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib as mpl

W, H = 618, 820

COLORS = {
    "top_box":    "#ddebf9",
    "left_panel": "#dfeeea",
    "center_panel":"#e8e0f0",   # Purple-tinted for spatial extent
    "right_panel":"#fbefda",
    "left_inner": "#dbe8e5",
    "center_inner":"#d4c4e8",   # Purple sub-boxes
    "right_inner":"#f9e8c8",
    "pass1_bg":   "#d6eaf8",    # Light blue for Pass 1
    "pass2_bg":   "#aed6f1",    # Slightly darker for Pass 2
    "line":       "#505050",
    "arrow":      "#6b6b6b",
    "dash":       "#cdcdcd",
}

mpl.rcParams["font.family"] = ["Liberation Sans Narrow","Nimbus Sans Narrow",
                                 "Liberation Sans","DejaVu Sans"]

def _rounded_box(ax, x, y, w, h, fc, ec=COLORS["line"], lw=1.55, radius=12,
                 dashed=False, z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=(0, (1.2, 2.1)) if dashed else "solid",
        joinstyle="round", capstyle="round", zorder=z)
    ax.add_patch(patch)

def _text(ax, x, y, s, size=12, weight="normal", ha="center", va="center",
          color="black", linespacing=0.92, z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va,
            color=color, linespacing=linespacing, zorder=z)

def _poly_arrow(ax, pts, color=COLORS["arrow"], lw=1.9, ms=13, z=4):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
    arrow = FancyArrowPatch(
        pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms,
        lw=lw, color=color, shrinkA=0, shrinkB=0,
        connectionstyle="arc3", zorder=z)
    ax.add_patch(arrow)


def draw_panel_b_rda(outpath='paper_materials/figures/_panel_b_rda.png',
                     topoplot_fn=None):
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # pixel-like (origin top-left)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title is drawn externally in build_fig3.py for font consistency
    # ax.plot([8, 610], [9, 9], color="black", lw=1.1)
    # _text(ax, 24, 52, "B. Pipeline Architecture",
    #       size=24, weight="bold", ha="left")

    # ── Input: 18 Independent Bipolar Channels ──
    _text(ax, 106, 85, "18 Independent\nBipolar Channels", size=13.2, linespacing=1.02)
    for y_off in [-16, -5.5, 5.5, 16]:
        _poly_arrow(ax, [(185, 85 + y_off), (220, 85 + y_off)], lw=1.2, ms=9)

    # ── Top: W05 Iterative Narrowband Refinement ──
    _rounded_box(ax, 220, 60, 280, 50, COLORS["top_box"], radius=15)
    _text(ax, 360, 85, "W05: Iterative\nNarrowband Refinement",
          size=13, weight="bold")

    _poly_arrow(ax, [(310, 110), (310, 155)], lw=1.9, ms=13)

    # ── Pass 1: Coarse Analysis ──
    _rounded_box(ax, 80, 155, 460, 85, COLORS["pass1_bg"],
                 ec=COLORS["line"], radius=13)
    _text(ax, 310, 172, "Pass 1: Coarse Analysis",
          size=13, weight="bold")
    _text(ax, 310, 200,
          "Bandpass 0.5\u20133.5 Hz\n"
          "Lateralization: mean variance per hemisphere\n"
          "Frequency: Hilbert from top-3 dominant channels",
          size=10.5, linespacing=1.05)

    _poly_arrow(ax, [(310, 240), (310, 265)], lw=1.9, ms=13)

    # ── Pass 2: Narrowband Refinement ──
    _rounded_box(ax, 80, 265, 460, 85, COLORS["pass2_bg"],
                 ec=COLORS["line"], radius=13)
    _text(ax, 310, 282, "Pass 2: Narrowband Refinement",
          size=13, weight="bold")
    _text(ax, 310, 310,
          "Bandpass at est_freq \u00b1 0.4 Hz\n"
          "Refined lateralization: envelope amplitude\n"
          "Refined frequency: Hilbert on dominant hemisphere",
          size=10.5, linespacing=1.05)

    # ── Fork to three branches ──
    FORK_Y = 375
    LX, MX, RX = 104, 309, 519

    ax.plot([310, 310], [350, FORK_Y], color=COLORS["arrow"], lw=1.0)
    ax.plot([LX, RX], [FORK_Y, FORK_Y], color=COLORS["arrow"], lw=1.0)
    _poly_arrow(ax, [(LX, FORK_Y), (LX, 400)])
    _poly_arrow(ax, [(MX, FORK_Y), (MX, 400)])
    _poly_arrow(ax, [(RX, FORK_Y), (RX, 400)])

    # ── Container panels (all same height) ──
    PT_Y = 400
    PANEL_H = 410  # all three panels same height

    _rounded_box(ax, 32, PT_Y, 150, PANEL_H, COLORS["left_panel"],
                 ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)
    _rounded_box(ax, 195, PT_Y, 228, PANEL_H, COLORS["center_panel"],
                 ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)
    _rounded_box(ax, 437, PT_Y, 163, PANEL_H, COLORS["right_panel"],
                 ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)

    _text(ax, LX, 430, "Laterality\nDetection",
          size=13.7, weight="bold")
    _text(ax, MX, 430, "Spatial Extent\n(PLV \u00d7 Amplitude)",
          size=13.7, weight="bold")
    _text(ax, RX, 430, "Topographic\nLocalization",
          size=13.7, weight="bold")

    # ── Left: Laterality Detection ──
    _rounded_box(ax, 42, 460, 130, 75, COLORS["left_inner"], radius=12)
    _text(ax, LX, 497, "L vs R\nnarrowband\namplitude", size=12)

    _poly_arrow(ax, [(LX, 535), (LX, 580)], lw=1.9, ms=13)
    _text(ax, LX, 605, "Output:\nLeft / Right", size=11.5)

    _poly_arrow(ax, [(LX, 630), (LX, 765)], lw=1.9, ms=13)
    _text(ax, LX, 780, "Laterality\n(Side)", size=12.7)

    # ── Center: Spatial Extent ──
    _rounded_box(ax, 210, 460, 198, 70, COLORS["center_inner"], radius=12)
    _text(ax, MX, 495,
          "Per-channel\nphase coherence with\ndominant hemisphere",
          size=11)

    _poly_arrow(ax, [(MX, 530), (MX, 560)], lw=1.9, ms=13)

    _rounded_box(ax, 220, 560, 178, 55, COLORS["center_inner"], radius=12)
    _text(ax, MX, 587, "\u00d7 narrowband\namplitude", size=11.5)

    _poly_arrow(ax, [(MX, 615), (MX, 650)], lw=1.9, ms=13)

    _rounded_box(ax, 230, 650, 158, 45, COLORS["center_inner"], radius=12)
    _text(ax, MX, 672, "Threshold \u2192\ncount / 18", size=11)

    _poly_arrow(ax, [(MX, 695), (MX, 765)], lw=1.9, ms=13)
    _text(ax, MX, 780, "Spatial Extent\n+ Frequency",
          size=12.3)

    # ── Right: Topographic Localization ──
    _rounded_box(ax, 448, 460, 140, 65, COLORS["right_inner"], radius=12)
    _text(ax, RX, 492, "Per-channel\nHilbert amplitude\nenvelope", size=11)

    _poly_arrow(ax, [(RX, 525), (RX, 555)], lw=1.9, ms=13)

    _rounded_box(ax, 448, 555, 140, 55, COLORS["right_inner"], radius=12)
    _text(ax, RX, 582, "Laplacian\ntransform", size=11.5)

    _poly_arrow(ax, [(RX, 610), (RX, 765)], lw=1.9, ms=13)

    _text(ax, RX, 780, "Localization", size=12.8)

    # ── Save ──
    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    draw_panel_b_rda()
