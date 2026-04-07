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
from matplotlib.colors import LinearSegmentedColormap
import mne
mne.set_log_level('ERROR')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Channel layout ──────────────────────────────────────────────────────────

BIPOLAR_NAMES = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Display order: L temporal, L parasagittal, midline, R parasagittal, R temporal
# idx=-1 means gap
DISPLAY_ORDER = [
    {'idx': 0,  'name': 'Fp1-F7', 'hemi': 'L'},
    {'idx': 1,  'name': 'F7-T3',  'hemi': 'L'},
    {'idx': 2,  'name': 'T3-T5',  'hemi': 'L'},
    {'idx': 3,  'name': 'T5-O1',  'hemi': 'L'},
    {'idx': -1, 'name': '',        'hemi': ''},   # gap
    {'idx': 8,  'name': 'Fp1-F3', 'hemi': 'L'},
    {'idx': 9,  'name': 'F3-C3',  'hemi': 'L'},
    {'idx': 10, 'name': 'C3-P3',  'hemi': 'L'},
    {'idx': 11, 'name': 'P3-O1',  'hemi': 'L'},
    {'idx': -1, 'name': '',        'hemi': ''},   # gap
    {'idx': 16, 'name': 'Fz-Cz',  'hemi': 'M'},
    {'idx': 17, 'name': 'Cz-Pz',  'hemi': 'M'},
    {'idx': -1, 'name': '',        'hemi': ''},   # gap
    {'idx': 12, 'name': 'Fp2-F4', 'hemi': 'R'},
    {'idx': 13, 'name': 'F4-C4',  'hemi': 'R'},
    {'idx': 14, 'name': 'C4-P4',  'hemi': 'R'},
    {'idx': 15, 'name': 'P4-O2',  'hemi': 'R'},
    {'idx': -1, 'name': '',        'hemi': ''},   # gap
    {'idx': 4,  'name': 'Fp2-F8', 'hemi': 'R'},
    {'idx': 5,  'name': 'F8-T4',  'hemi': 'R'},
    {'idx': 6,  'name': 'T4-T6',  'hemi': 'R'},
    {'idx': 7,  'name': 'T6-O2',  'hemi': 'R'},
]

LEFT_CH_IDX = {0, 1, 2, 3, 8, 9, 10, 11}
RIGHT_CH_IDX = {4, 5, 6, 7, 12, 13, 14, 15}

# Electrode positions (normalized to unit circle, nose at top = +y)
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
    """Draw the EEG traces on the given axes."""
    from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, detrend
    eeg = np.array(case['eeg_data'])  # (18, 1000)
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

    # Determine y range
    real_ys = [y for y in y_positions if y is not None]
    y_min_plot = min(real_ys) - 0.8
    y_max_plot = max(real_ys) + 0.8

    ax.set_xlim(-0.05, 10.05)
    ax.set_ylim(y_min_plot, y_max_plot)
    ax.set_facecolor('white')

    # Hemisphere shading — light blue on involved side(s)
    lat = case.get('gt_lat', case.get('pred_lat', ''))
    subtype = case.get('subtype', '').lower()
    shade_color = (100/255, 160/255, 255/255, 0.07)  # light blue

    # Determine which hemispheres to shade
    if subtype in ('gpd', 'grda'):
        shade_hemis = {'L', 'R', 'M'}  # bilateral + midline for generalized
    elif lat in ('left', 'bilateral, left-predominant'):
        shade_hemis = {'L'}
    elif lat in ('right', 'bilateral, right-predominant'):
        shade_hemis = {'R'}
    elif lat in ('bilateral', 'bilateral/symmetric'):
        shade_hemis = {'L', 'R'}
    else:
        shade_hemis = {'L', 'R'}  # default both if unknown

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
        ax.text(-0.15, yp, entry['name'], fontsize=6, va='center', ha='right',
                color='black', clip_on=False)

    # Title info
    freq = case.get('gt_freq') or case.get('pred_freq', 0)
    lat = case.get('gt_lat', case.get('pred_lat', '?'))
    n_discharges = len(discharge_times)
    difficulty = case.get('difficulty', '')
    jaccard = case.get('jaccard') or 0
    freq = freq or 0

    diff_colors = {'Easy': '#2a7d2a', 'Medium': '#b87700', 'Hard': '#c03030'}
    dc = diff_colors.get(difficulty, '#333')

    if is_pd:
        title_text = f'freq={freq:.2f} Hz | lat={lat} | {n_discharges} discharges'
    else:
        title_text = f'lat={lat} | regions={",".join(case.get("gt_regions", []))}'

    ax.set_title(title_text, fontsize=8, pad=2, color='#333')

    # Difficulty label in upper-left
    diff_text = f'[{difficulty.upper()}] Jaccard={jaccard:.2f}'
    ax.text(0.01, 0.98, diff_text, transform=ax.transAxes,
            fontsize=7, fontweight='bold', color=dc, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=dc, alpha=0.8))

    # Time axis
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_xticks(range(11))
    ax.set_xticklabels([str(i) for i in range(11)], fontsize=6)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)


# ── Topoplot panel ──────────────────────────────────────────────────────────

def draw_topoplot(ax, case):
    """Draw the topoplot using MNE's spherical spline interpolation."""
    import mne
    region_scores = case.get('region_scores', {})

    # 19-channel monopolar layout
    ch_names_orig = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                     'Fp2','F4','C4','P4','F8','T4','T6','O2']
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in ch_names_orig]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # Map region scores to 19 monopolar electrodes
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

    # Build 19-element data array in ch_names_orig order
    data = np.array([electrode_scores.get(e, 0.5) for e in ch_names_orig])

    # Per-case normalization: use this case's min/max for full color contrast
    # Add small padding to avoid identical vmin/vmax
    vmin = data.min()
    vmax = data.max()
    if vmax - vmin < 0.02:
        vmin = max(0, vmin - 0.05)
        vmax = min(1, vmax + 0.05)

    mne.viz.plot_topomap(
        data, info, axes=ax, show=False,
        contours=6, cmap='inferno',
        vlim=(vmin, vmax),
        sensors=True, names=ch_names_orig,
        size=3,
    )


# ── Right-side info panel ───────────────────────────────────────────────────

def draw_info_panel(ax, case):
    """Draw verbal description and rater info below topoplot."""
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    verbal = case.get('verbal_description', '')
    rater_details = case.get('rater_details', {})
    pred_regions = case.get('pred_regions', [])

    text_parts = []
    if verbal:
        text_parts.append(verbal)

    full_text = '\n'.join(text_parts)
    ax.text(0.5, 1.0, full_text, fontsize=5.5, ha='center', va='top',
            wrap=True, linespacing=1.4, family='monospace',
            transform=ax.transAxes)


# ── Main figure assembly ────────────────────────────────────────────────────

def render_subtype(subtype, cases, is_pd):
    """Render a 3-row figure for one subtype."""
    n_cases = len(cases)
    fig = plt.figure(figsize=(18, 8 * n_cases), facecolor='white')
    fig.suptitle(f'{subtype.upper()} Characterization Examples',
                 fontsize=16, fontweight='bold', y=0.995)

    # 3 rows, each row: [EEG (75%) | topoplot + info (25%)]
    outer_gs = gridspec.GridSpec(n_cases, 2, figure=fig,
                                 width_ratios=[75, 25],
                                 hspace=0.25, wspace=0.05,
                                 left=0.04, right=0.98, top=0.97, bottom=0.02)

    for row, case in enumerate(cases):
        # EEG panel (left)
        ax_eeg = fig.add_subplot(outer_gs[row, 0])
        draw_eeg_panel(ax_eeg, case, is_pd)

        # Right column: split into topoplot (top) and info (bottom)
        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[row, 1],
                                                     height_ratios=[3, 2], hspace=0.05)
        ax_topo = fig.add_subplot(inner_gs[0])
        draw_topoplot(ax_topo, case)

        ax_info = fig.add_subplot(inner_gs[1])
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
