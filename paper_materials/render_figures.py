#!/usr/bin/env python3
"""
Render publication-quality EEG characterization figures from JSON data.

Self-contained: only requires matplotlib, numpy, scipy, json.
Reads figure_*_examples_data.json and produces figure_*_examples.png at 300 DPI.
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import mne
mne.set_log_level('ERROR')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Global Matplotlib Styling ---
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']

# --- Font Sizes ---
# Base font size for general text (e.g., tick labels, general text)
# Specific elements will override this with their dedicated constants.
BASE_FONT_SIZE = 8
matplotlib.rcParams['font.size'] = BASE_FONT_SIZE
matplotlib.rcParams['axes.labelsize'] = BASE_FONT_SIZE
matplotlib.rcParams['xtick.labelsize'] = BASE_FONT_SIZE
matplotlib.rcParams['ytick.labelsize'] = BASE_FONT_SIZE

CHANNEL_LABEL_FONTSIZE = 9
TIME_AXIS_LABEL_FONTSIZE = 9
TIME_TICK_LABEL_FONTSIZE = 9
# CRITIQUE 7: Slightly increased font size for verbal descriptions
VERBAL_DESCRIPTION_FONTSIZE = 10
# CRITIQUE 1: Increased significantly for topoplot electrode labels
TOPOPLOT_ELECTRODE_LABEL_FONTSIZE = 12
# CRITIQUE 2: Increased font size for colorbar tick labels
COLORBAR_TICK_LABEL_FONTSIZE = 11
# CRITIQUE 2 & 6: Increased font size for colorbar 'Score' label
COLORBAR_LABEL_FONTSIZE = 11
# CRITIQUE 2: Increased font size for difficulty badge and agreement percentages
DIFFICULTY_BADGE_FONTSIZE = 11

FIGURE_TITLE_FONTSIZE = 12
matplotlib.rcParams['figure.titlesize'] = FIGURE_TITLE_FONTSIZE

# --- Spacing Parameters ---
# CRITIQUE 5: Increased vertical spacing between rows
ROW_VSPACE = 0.4
EEG_TO_TOPO_HSPACE = 0.03
TOPO_TO_CBAR_WSPACE = 0.05
TOPO_INFO_VSPACE = 0.01

# --- Layout Ratios ---
EEG_WIDTH_RATIO = 75
RIGHT_COL_WIDTH_RATIO = 25
TOPO_HEIGHT_RATIO = 6
INFO_HEIGHT_RATIO = 2

# --- Figure Title Placement ---
# CRITIQUE 3: Main title is already implemented via fig.suptitle, ensuring its placement.
FIGURE_TITLE_Y_POS = 0.96

# --- Difficulty Badge Styling ---
DIFF_BADGE_COLORS = {
    'Easy': {'text': '#2a7d2a', 'bg': '#e6ffe6', 'border': '#2a7d2a'},
    'Medium': {'text': '#b87700', 'bg': '#fff8e6', 'border': '#b87700'},
    'Hard': {'text': '#c03030', 'bg': '#ffe6e6', 'border': '#c03030'}
}

# --- Topoplot Electrode Label Styling ---
TOPOPLOT_ELECTRODE_LABEL_COLOR = 'white'
TOPOPLOT_ELECTRODE_LABEL_BBOX = dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.1', edgecolor='none')


# ── Channel layout ──────────────────────────────────────────────────────────

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# Display order for average reference montage:
# L parasag, L temporal, midline, R parasag, R temporal
# idx = index into MONO_CHANNELS (19 channels), -1 = gap
DISPLAY_ORDER = [
    {'idx': 0,  'name': 'Fp1', 'hemi': 'L'},
    {'idx': 1,  'name': 'F3',  'hemi': 'L'},
    {'idx': 2,  'name': 'C3',  'hemi': 'L'},
    {'idx': 3,  'name': 'P3',  'hemi': 'L'},
    {'idx': 4,  'name': 'F7',  'hemi': 'L'},
    {'idx': 5,  'name': 'T3',  'hemi': 'L'},
    {'idx': 6,  'name': 'T5',  'hemi': 'L'},
    {'idx': 7,  'name': 'O1',  'hemi': 'L'},
    {'idx': -1, 'name': '',    'hemi': ''},   # gap
    {'idx': 8,  'name': 'Fz',  'hemi': 'M'},
    {'idx': 9,  'name': 'Cz',  'hemi': 'M'},
    {'idx': 10, 'name': 'Pz',  'hemi': 'M'},
    {'idx': -1, 'name': '',    'hemi': ''},   # gap
    {'idx': 11, 'name': 'Fp2', 'hemi': 'R'},
    {'idx': 12, 'name': 'F4',  'hemi': 'R'},
    {'idx': 13, 'name': 'C4',  'hemi': 'R'},
    {'idx': 14, 'name': 'P4',  'hemi': 'R'},
    {'idx': 15, 'name': 'F8',  'hemi': 'R'},
    {'idx': 16, 'name': 'T4',  'hemi': 'R'},
    {'idx': 17, 'name': 'T6',  'hemi': 'R'},
    {'idx': 18, 'name': 'O2',  'hemi': 'R'},
]

LEFT_CH_IDX = {0, 1, 2, 3, 4, 5, 6, 7}
RIGHT_CH_IDX = {11, 12, 13, 14, 15, 16, 17, 18}

# Electrode positions (normalized to unit circle, nose at top = +y)
# These are used for manual label placement on topoplots
ELECTRODE_POS = {
    'Fp1': (-0.31, 0.95), 'Fp2': (0.31, 0.95),
    'F7': (-0.81, 0.59), 'F3': (-0.39, 0.59), 'Fz': (0.0, 0.59),
    'F4': (0.39, 0.59), 'F8': (0.81, 0.59),
    'T3': (-1.0, 0.0), 'C3': (-0.5, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.5, 0.0), 'T4': (1.0, 0.0),
    'T5': (-0.81, -0.59), 'P3': (-0.39, -0.59), 'Pz': (0.0, -0.59),
    'P4': (0.39, -0.59), 'T6': (0.81, -0.59),
    'O1': (-0.31, -0.95), 'O2': (0.31, -0.95),
}

BIPOLAR_ELECTRODES = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]

REGION_CHANNELS = {
    'LF': [0, 8],   'RF': [4, 12],
    'LT': [1, 2],   'RT': [5, 6],
    'LCP': [10, 9],  'RCP': [14, 13],
    'LO': [3, 11],  'RO': [7, 15],
    'MID': [16, 17],
}


# ── Colormap ────────────────────────────────────────────────────────────────

# Fixed color scale percentiles (computed from 500 cases each)
# PD (LPD+GPD): p5=0.459, p95=0.729
# RDA (LRDA+GRDA): p5=0.442, p95=0.697
COLOR_SCALE = {
    'pd':  {'vmin': 0.459, 'vmax': 0.729},
    'rda': {'vmin': 0.442, 'vmax': 0.697},
}

def score_to_color(t):
    """Map normalized score [0,1] -> (r, g, b) tuple in [0,1]."""
    t = np.clip(t, 0, 1)
    if t <= 0.33:
        s = t / 0.33
        return ((20 + (0 - 20) * s) / 255,
                (20 + (180 - 20) * s) / 255,
                (80 + (220 - 80) * s) / 255)
    elif t <= 0.66:
        s = (t - 0.33) / 0.33
        return ((0 + 255 * s) / 255,
                (180 + (220 - 180) * s) / 255,
                (220 + (0 - 220) * s) / 255)
    else:
        s = (t - 0.66) / 0.34
        return (1.0,
                (220 + (30 - 220) * s) / 255,
                0.0)


TOPO_CMAP = plt.cm.inferno  # perceptually uniform, colorblind-friendly


# ── EEG panel ───────────────────────────────────────────────────────────────

def draw_eeg_panel(ax, case, is_pd):
    """Draw the EEG traces on the given axes (average reference montage)."""
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, detrend

    # Use monopolar data with common average reference
    if 'mono_data' in case:
        mono = np.array(case['mono_data'])  # (19, 1000)
        # Compute CAR
        avg = np.mean(mono, axis=0)
        eeg = mono - avg[np.newaxis, :]
    else:
        # Fallback to bipolar if mono_data not available
        eeg = np.array(case['eeg_data'])
    n_ch, n_samp = eeg.shape
    fs_display = n_samp / 10.0  # 100 Hz for downsampled data

    # Filter: notch 60 Hz + bandpass 0.3-50 Hz
    # Only apply if fs is high enough for the filter
    if fs_display > 2:
        # Detrend first
        eeg = detrend(eeg, axis=1)
        # Notch 60 Hz (only if Nyquist > 60)
        if fs_display / 2 > 60:
            b_notch, a_notch = iirnotch(60.0, 30.0, fs_display)
            eeg = filtfilt(b_notch, a_notch, eeg, axis=1)
        # Bandpass 0.3 - min(50, Nyquist-1) Hz
        hi_freq = min(50.0, fs_display / 2 - 1)
        if hi_freq > 0.3:
            sos = butter(4, [0.3, hi_freq], btype='bandpass', fs=fs_display, output='sos')
            eeg = sosfiltfilt(sos, eeg, axis=1)

    # Clip EEG for display
    clip_uv = 250.0
    eeg = np.clip(eeg, -clip_uv, clip_uv)
    z_scale = 0.012

    # Time axis
    t = np.linspace(0, 10, n_samp)

    # Compute y positions for each display row (with gaps)
    n_rows = len(DISPLAY_ORDER)
    y_positions = []
    y_cursor = n_rows  # start from top
    for entry in DISPLAY_ORDER:
        if entry['idx'] == -1:
            y_cursor -= 0.5  # gap is half-height
            y_positions.append(None)
        else:
            y_positions.append(y_cursor)
            y_cursor -= 1

    # Determine y range (CRITIQUE 4: Improved Vertical Spacing within EEG Plots)
    real_ys = [y for y in y_positions if y is not None]
    y_min_plot = min(real_ys) - 1.0 # Increased padding
    y_max_plot = max(real_ys) + 1.0 # Increased padding

    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_facecolor('white')

    # Hemisphere shading
    lat = case.get('pred_lat', '') or case.get('gt_lat', '')
    subtype = case.get('subtype', '').lower()
    shade_color = (100/255, 160/255, 255/255, 0.15)

    # Determine which hemispheres to shade
    if subtype in ('gpd', 'grda'):
        shade_hemis = {'L', 'R', 'M'}  # bilateral + midline for generalized
    elif 'left' in lat:
        shade_hemis = {'L'}
    elif 'right' in lat:
        shade_hemis = {'R'}
    elif 'bilateral' in lat:
        shade_hemis = {'L', 'R'}
    elif subtype in ('lpd', 'lrda'):
        shade_hemis = {'L'}  # default left for lateralized if unknown
    else:
        shade_hemis = {'L', 'R'}

    for i, entry in enumerate(DISPLAY_ORDER):
        if entry['idx'] == -1 or y_positions[i] is None:
            continue
        yp = y_positions[i]
        if entry['hemi'] in shade_hemis:
            ax.axhspan(yp - 0.45, yp + 0.45, color=shade_color)

    # Grid lines every 1s
    for sec in range(11):
        ax.axvline(sec, color='#e0e0e0', linewidth=0.4, zorder=0)

    # Discharge markers (PD only)
    discharge_times = case.get('gt_discharge_times') or case.get('pred_discharge_times') or []
    if is_pd and discharge_times:
        for dt in discharge_times:
            if subtype == 'gpd':
                ax.axvline(dt, color='red', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
            elif subtype == 'lpd':
                # Markers on involved hemisphere only, partial height
                if lat in ('left', 'bilateral', 'bilateral, left-predominant'):
                    left_ys = [y_positions[i] for i, e in enumerate(DISPLAY_ORDER)
                               if e['hemi'] == 'L' and y_positions[i] is not None]
                    if left_ys:
                        ax.plot([dt, dt], [min(left_ys) - 0.4, max(left_ys) + 0.4],
                                color='red', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
                if lat in ('right', 'bilateral', 'bilateral, right-predominant'):
                    right_ys = [y_positions[i] for i, e in enumerate(DISPLAY_ORDER)
                                if e['hemi'] == 'R' and y_positions[i] is not None]
                    if right_ys:
                        ax.plot([dt, dt], [min(right_ys) - 0.4, max(right_ys) + 0.4],
                                color='red', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
                if lat in ('bilateral/symmetric', 'bilateral'):
                    ax.axvline(dt, color='red', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

    # Draw EEG traces
    for i, entry in enumerate(DISPLAY_ORDER):
        if entry['idx'] == -1 or y_positions[i] is None:
            continue
        ch_idx = entry['idx']
        yp = y_positions[i]
        sig = eeg[ch_idx] * z_scale + yp
        ax.plot(t, sig, color='black', linewidth=0.5, zorder=2)
        # Channel label
        ax.text(-0.1, yp, entry['name'], fontsize=CHANNEL_LABEL_FONTSIZE, va='center', ha='right',
                color='black', clip_on=False)

    # Amplitude scale bar (double-headed arrow, 100 µV) (CRITIQUE 8: Consistent EEG Y-axis Scale Bar Placement)
    from matplotlib.patches import FancyArrowPatch
    scale_uv = 100.0
    scale_height = scale_uv * z_scale  # = 1.2 plot units
    scale_x = 10.4 # Moved slightly further to the right
    scale_y_bot = y_min_plot + 0.2
    scale_y_top = scale_y_bot + scale_height
    scale_y_mid = (scale_y_bot + scale_y_top) / 2
    # Single double-headed arrow using two FancyArrowPatch
    arrow_up = FancyArrowPatch((scale_x, scale_y_mid + 0.02), (scale_x, scale_y_top),
                                arrowstyle='-|>', mutation_scale=10, color='black', lw=1.0, zorder=5, clip_on=False)
    arrow_dn = FancyArrowPatch((scale_x, scale_y_mid - 0.02), (scale_x, scale_y_bot),
                                arrowstyle='-|>', mutation_scale=10, color='black', lw=1.0, zorder=5, clip_on=False)
    ax.add_patch(arrow_up)
    ax.add_patch(arrow_dn)
    ax.text(scale_x + 0.15, scale_y_mid, f'{int(scale_uv)} µV',
            fontsize=9, va='center', ha='left', color='black', clip_on=False) # Increased font size for consistency

    # Difficulty badge (CRITIQUE 2: Increased Font Size for Agreement Percentages)
    difficulty = case.get('difficulty', '')
    agreement = case.get('agreement_pct', 0)

    dc_config = DIFF_BADGE_COLORS.get(difficulty, {'text': '#333', 'bg': '#f0f0f0', 'border': '#999'})

    diff_text = f'{difficulty.upper()} (Agreement={agreement:.0f}%)'
    ax.text(0.99, 0.99, diff_text, transform=ax.transAxes,
            fontsize=DIFFICULTY_BADGE_FONTSIZE, fontweight='bold',
            color=dc_config['text'], va='top', ha='right',
            bbox=dict(boxstyle="round,pad=0.2", fc=dc_config['bg'], ec=dc_config['border'], lw=0.5, alpha=0.7))

    # Time axis labels
    ax.set_xlabel('Time (s)', fontsize=TIME_AXIS_LABEL_FONTSIZE)
    ax.set_xticks(range(11))
    ax.set_xticklabels([str(i) for i in range(11)], fontsize=TIME_TICK_LABEL_FONTSIZE)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


# ── Topoplot panel ──────────────────────────────────────────────────────────

def draw_topoplot(ax, ax_cbar, case, is_pd):
    """Draw the topoplot from pre-rendered base64 PNG or fall back to region_scores."""
    import mne
    import io
    import base64
    from PIL import Image

    # Determine vmin, vmax for colorbar
    vmin, vmax = None, None
    if is_pd:
        vmin, vmax = COLOR_SCALE['pd']['vmin'], COLOR_SCALE['pd']['vmax']
    else:
        vmin, vmax = COLOR_SCALE['rda']['vmin'], COLOR_SCALE['rda']['vmax']

    if vmax - vmin < 1e-6: # Handle cases where range is too small
        vmin = max(0, vmin - 0.05)
        vmax = min(1, vmax + 0.05)

    # --- Draw the main topoplot content ---
    topo_b64 = case.get('topo_img_lap') or case.get('topo_img_mono')

    if topo_b64:
        # Decode base64 PNG and display as image
        img_bytes = base64.b64decode(topo_b64)
        img = Image.open(io.BytesIO(img_bytes))
        # Use extent to map image coordinates, with equal aspect to keep circle
        ax.imshow(img, extent=[-1, 1, -1, 1], origin='upper')
        ax.set_aspect('equal')
        ax.axis('off')
    else:
        # Fallback: old region_scores approach using MNE
        region_scores = case.get('region_scores', {})

        ch_names_orig = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                         'Fp2','F4','C4','P4','F8','T4','T6','O2']
        name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
        mne_names = [name_map.get(n, n) for n in ch_names_orig]

        info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        region_to_electrodes = {
            'LF': ['Fp1', 'F3', 'F7'], 'RF': ['Fp2', 'F4', 'F8'],
            'LT': ['T3', 'T5', 'F7'], 'RT': ['T4', 'T6', 'F8'],
            'LCP': ['C3', 'P3'], 'RCP': ['C4', 'P4'],
            'LO': ['O1', 'P3', 'T5'], 'RO': ['O2', 'P4', 'T6'],
            'MID': ['Fz', 'Cz', 'Pz'],
        }

        electrode_scores = {}
        electrode_counts = {}
        for reg, score in region_scores.items():
            for e in region_to_electrodes.get(reg, []):
                electrode_scores[e] = electrode_scores.get(e, 0) + score
                electrode_counts[e] = electrode_counts.get(e, 0) + 1
        for e in electrode_scores:
            electrode_scores[e] /= electrode_counts[e]

        data = np.array([electrode_scores.get(e, 0.5) for e in ch_names_orig])

        # MNE plot_topomap
        image, _ = mne.viz.plot_topomap(
            data, info, axes=ax, show=False,
            contours=6, cmap=TOPO_CMAP,
            vlim=(vmin, vmax),
            sensors=True, # Show sensor dots
            show_names=False, # Disable MNE's names to draw our own
        )
        ax.axis('off') # Ensure axis is off even if MNE doesn't fully hide it

    # Electrode labels: only add if NOT using pre-rendered topoplot
    # (pre-rendered topoplots already have electrode labels baked in)
    if not topo_b64:
        for ch_name, (x_pos, y_pos) in ELECTRODE_POS.items():
            ax.text(x_pos, y_pos, ch_name,
                    fontsize=TOPOPLOT_ELECTRODE_LABEL_FONTSIZE,
                    color=TOPOPLOT_ELECTRODE_LABEL_COLOR,
                    ha='center', va='center',
                    bbox=TOPOPLOT_ELECTRODE_LABEL_BBOX,
                    clip_on=True, zorder=3)

    # Hide colorbar axis — colorbar is not needed (pre-rendered topoplots
    # are self-explanatory, and the colorbar adds clutter)
    ax_cbar.axis('off')


# ── Right-side info panel ───────────────────────────────────────────────────

def draw_info_panel(ax, case):
    """Draw verbal description and rater info below topoplot."""
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    verbal = case.get('verbal_description', '')

    if verbal:
        # Balance text across ~2 lines by splitting near the middle
        import textwrap
        target_width = max(20, len(verbal) // 2 + 5)
        wrapped = textwrap.fill(verbal, width=target_width)
        ax.text(0.5, 1.0, wrapped, fontsize=VERBAL_DESCRIPTION_FONTSIZE, ha='center', va='top',
                linespacing=1.8, transform=ax.transAxes) # CRITIQUE 7: Increased linespacing


# ── Main figure assembly ────────────────────────────────────────────────────

def render_subtype(subtype, cases, is_pd):
    """Render a 3-row figure for one subtype."""
    n_cases = len(cases)
    # Adjust figure height for increased font sizes and better spacing
    fig_width = 18
    fig_height = 7.0 * n_cases # Adjusted height per case for better spacing due to increased ROW_VSPACE

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Main Figure Titles (CRITIQUE 3: Add Main Title to All Figures)
    # This part already exists and correctly applies a title to each subtype figure.
    fig.suptitle(f'{subtype.upper()} Characterization Examples',
                 fontsize=FIGURE_TITLE_FONTSIZE, fontweight='bold', y=FIGURE_TITLE_Y_POS)

    # 3 rows, each row: [EEG (75%) | topoplot + info (25%)]
    outer_gs = gridspec.GridSpec(n_cases, 2, figure=fig,
                                 width_ratios=[EEG_WIDTH_RATIO, RIGHT_COL_WIDTH_RATIO],
                                 hspace=ROW_VSPACE, # CRITIQUE 5: Increased vertical spacing between rows
                                 wspace=EEG_TO_TOPO_HSPACE,
                                 left=0.04, right=0.98, top=0.95, bottom=0.02)

    for row, case in enumerate(cases):
        # EEG panel (left)
        ax_eeg = fig.add_subplot(outer_gs[row, 0])
        draw_eeg_panel(ax_eeg, case, is_pd)

        # Right column: split into topoplot+colorbar (top) and info (bottom)
        inner_gs_right_col = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, 1],
                                                               height_ratios=[TOPO_HEIGHT_RATIO, INFO_HEIGHT_RATIO],
                                                               hspace=TOPO_INFO_VSPACE)

        # Topoplot and Colorbar (CRITIQUE 6: Widen Colorbar)
        topoplot_cbar_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=inner_gs_right_col[0],
                                                             width_ratios=[10, 1.5], wspace=TOPO_TO_CBAR_WSPACE) # Increased colorbar width ratio

        ax_topo = fig.add_subplot(topoplot_cbar_gs[0])
        ax_cbar = fig.add_subplot(topoplot_cbar_gs[1]) # Axis for the colorbar

        draw_topoplot(ax_topo, ax_cbar, case, is_pd)

        ax_info = fig.add_subplot(inner_gs_right_col[1])
        draw_info_panel(ax_info, case)

    out_path = os.path.join(SCRIPT_DIR, f'figure_{subtype}_examples.png')
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('subtypes', nargs='*', default=['lpd', 'gpd', 'lrda', 'grda'])
    parser.add_argument('--pick', type=str, default=None,
                        help='JSON dict of {subtype: [indices]} to select specific cases')
    args = parser.parse_args()

    pick = None
    if args.pick:
        pick = json.loads(args.pick)

    for subtype in args.subtypes:
        subtype = subtype.lower()
        json_path = os.path.join(SCRIPT_DIR, f'figure_{subtype}_examples_data.json')
        if not os.path.exists(json_path):
            print(f'  Skipping {subtype}: {json_path} not found')
            continue

        with open(json_path) as f:
            all_cases = json.load(f)

        if pick and subtype in pick:
            cases = [all_cases[i] for i in pick[subtype] if i < len(all_cases)]
            print(f'Rendering {subtype.upper()} ({len(cases)} selected from {len(all_cases)})...')
        else:
            cases = all_cases
            print(f'Rendering {subtype.upper()} ({len(cases)} cases)...')

        is_pd = subtype in ('lpd', 'gpd')
        render_subtype(subtype, cases, is_pd)


if __name__ == '__main__':
    main()