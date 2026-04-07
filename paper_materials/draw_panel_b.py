#!/usr/bin/env python3
"""Draw Panel B for Fig 2 — faithful recreation of PaperBanana style."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib as mpl

W, H = 618, 786

COLORS = {
    "top_box":    "#ddebf9",
    "left_panel": "#dfeeea",
    "center_panel":"#f6e8e3",
    "right_panel":"#fbefda",
    "left_inner": "#dbe8e5",
    "center_inner":"#efc7b4",
    "right_inner":"#f9e8c8",
    "line":       "#505050",
    "arrow":      "#6b6b6b",
    "dash":       "#cdcdcd",
}

mpl.rcParams["font.family"] = ["Liberation Sans Narrow","Nimbus Sans Narrow","Liberation Sans","DejaVu Sans"]

def _rounded_box(ax, x, y, w, h, fc, ec=COLORS["line"], lw=1.55, radius=12, dashed=False, z=2):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle=(0, (1.2, 2.1)) if dashed else "solid", joinstyle="round", capstyle="round", zorder=z)
    ax.add_patch(patch)

def _text(ax, x, y, s, size=12, weight="normal", ha="center", va="center", color="black", linespacing=0.92, z=5):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, linespacing=linespacing, zorder=z)

def _poly_arrow(ax, pts, color=COLORS["arrow"], lw=1.9, ms=13, z=4):
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle="round", zorder=z)
    arrow = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=ms,
        lw=lw, color=color, shrinkA=0, shrinkB=0, connectionstyle="arc3", zorder=z)
    ax.add_patch(arrow)

def draw_panel_b(outpath='paper_materials/figures/_panel_b.png', topoplot_fn=None):
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")

    ax.plot([8, 610], [9, 9], color="black", lw=1.1)
    _text(ax, 24, 52, "B. Pipeline Architecture", size=24, weight="bold", ha="left")

    _text(ax, 106, 145, "18 Independent\nBipolar Channels", size=13.2, linespacing=1.02)
    _poly_arrow(ax, [(178, 145), (220, 145)], lw=1.9, ms=14)

    _rounded_box(ax, 221, 103, 192, 84, COLORS["top_box"], radius=15)
    _text(ax, 317, 145, "ChannelPD-Net\n(CNN+Attention)", size=15, linespacing=0.95)

    _text(ax, 241, 219, "PD Probability", size=11.9)
    _text(ax, 451, 219, "Frequency Estimate", size=11.9)

    _poly_arrow(ax, [(296, 187), (296, 245), (104, 245), (104, 267)], lw=1.9, ms=13)
    _poly_arrow(ax, [(334, 187), (334, 267)], lw=1.9, ms=13)
    _poly_arrow(ax, [(355, 187), (355, 245), (519, 245), (519, 267)], lw=1.9, ms=13)

    _rounded_box(ax, 32, 267, 150, 517, COLORS["left_panel"], ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)
    _rounded_box(ax, 189, 266, 241, 418, COLORS["center_panel"], ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)
    _rounded_box(ax, 437, 267, 163, 517, COLORS["right_panel"], ec=COLORS["dash"], lw=1.1, radius=13, dashed=True, z=1)

    _text(ax, 107, 307, "Laterality\nDetection", size=13.7, weight="bold")
    _text(ax, 309, 307, "Discharge\nDetection", size=13.7, weight="bold")
    _text(ax, 519, 307, "Topographic\nLocalization", size=13.7, weight="bold")

    # Left
    _rounded_box(ax, 49, 338, 130, 84, COLORS["left_inner"], radius=12)
    _text(ax, 114, 380, "Compare\nL vs R Mean\nProbabilities", size=12.3)
    _poly_arrow(ax, [(112, 422), (112, 496)], lw=1.9, ms=13)
    _text(ax, 114, 515, "Laterality\n(Side)", size=12.7)
    _poly_arrow(ax, [(112, 542), (112, 708)], lw=1.9, ms=13)
    _text(ax, 114, 742, "Laterality\n(Side)", size=12.7)

    # Center
    _rounded_box(ax, 200, 335, 105, 62, COLORS["center_inner"], radius=12)
    _rounded_box(ax, 318, 335, 103, 62, COLORS["center_inner"], radius=12)
    _text(ax, 252.5, 366, "8-channel\nCET-UNet", size=12.3)
    _text(ax, 369.5, 366, "CNN+ACF\nEnsemble", size=12.3)
    _text(ax, 252.5, 443, "Evidence\nTrace", size=12.2)
    _text(ax, 369.5, 443, "Frequency\nPrior", size=12.2)
    _poly_arrow(ax, [(252.5, 397), (252.5, 496)], lw=1.9, ms=13)
    _poly_arrow(ax, [(369.5, 397), (369.5, 496)], lw=1.9, ms=13)
    _rounded_box(ax, 239, 495, 142, 58, COLORS["center_inner"], radius=12)
    _text(ax, 310, 524, "Dynamic\nProgramming", size=12.3)
    _poly_arrow(ax, [(310, 553), (310, 591)], lw=1.9, ms=13)
    _rounded_box(ax, 240, 591, 139, 78, COLORS["center_inner"], radius=12)
    _text(ax, 309.5, 630, "EM Template\nRefinement &\nFiltering", size=12.0)
    _poly_arrow(ax, [(309.5, 669), (309.5, 707)], lw=1.9, ms=13)
    _text(ax, 309.5, 745, "Discharge Times ($t_1\\cdots t_n$)\nFrequency", size=12.3)

    # Right
    _rounded_box(ax, 448, 335, 140, 72, COLORS["right_inner"], radius=12)
    _rounded_box(ax, 448, 436, 140, 72, COLORS["right_inner"], radius=12)
    _rounded_box(ax, 448, 538, 140, 97, COLORS["right_inner"], radius=12)
    _text(ax, 518, 371, "Extract\nMonopolar\nVoltage", size=12.2)
    _text(ax, 518, 472, "Laplacian-GFP\nAlignment\n($\\pm$25ms)", size=12.1)
    _text(ax, 518, 586, "Template\nRefinement &\nGFP-weighted\nAveraging", size=11.8)
    _poly_arrow(ax, [(518, 407), (518, 436)], lw=1.9, ms=13)
    _poly_arrow(ax, [(518, 508), (518, 538)], lw=1.9, ms=13)
    _poly_arrow(ax, [(518, 635), (518, 671)], lw=1.9, ms=13)

    if topoplot_fn is None:
        circ = Circle((518, 708), 29, facecolor="#f5f5f5", edgecolor=COLORS["line"],
                       linewidth=1.4, linestyle=(0,(2,2)), zorder=3)
        ax.add_patch(circ)
        _text(ax, 518, 708, "topoplot", size=9.5, color="#666666")
    else:
        inset = ax.inset_axes([489, 679, 58, 58], transform=ax.transData)
        inset.set_axis_off()
        topoplot_fn(inset)

    _text(ax, 518, 756, "Localization", size=12.8)

    fig.savefig(outpath, dpi=200, facecolor="white", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    draw_panel_b()
