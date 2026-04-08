#!/usr/bin/env python3
"""
Generate Fig 2: PD Characterization Pipeline composite figure.

Three-panel horizontal layout:
  Panel A (left):   Input 19-channel EEG (average reference, 10s)
  Panel B (center): Architecture flowchart (matplotlib boxes/arrows)
  Panel C (right):  Output visualization (EEG + discharge markers, topoplot, verbal)

Uses the Easy LPD case: sub-S0001114959966_20150425125519.mat

Usage:
    conda run -n morgoth python paper_materials/generate_fig2_pd_pipeline.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
# Set global font family for consistency and publication readiness
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import mne
mne.set_log_level('WARNING')

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, '/Users/mwestover/GithubRepos/morgoth-viewer')

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig2_pd_pipeline.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Constants ──
FS = 200
N_SAMPLES = 2000
MAT_FILE = 'sub-S0001114959966_20150425125519.mat'
SEGMENT_ID = MAT_FILE.replace('.mat', '')

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# Display order: L parasag, L temporal, gap, midline, gap, R parasag, R temporal
DISPLAY_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18]

# Left hemisphere channel indices (in MONO_CHANNELS)
LEFT_CH_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]   # Fp1,F3,C3,P3,F7,T3,T5,O1
RIGHT_CH_INDICES = [11, 12, 13, 14, 15, 16, 17, 18]  # Fp2,F4,C4,P4,F8,T4,T6,O2

# Laplacian neighbor map (same as generate_discharge_topo_viewer.py)
LAP_NEIGHBORS = {
    0: [1, 4, 8, 11],      # Fp1
    1: [0, 2, 4, 8],       # F3
    2: [1, 3, 5, 9],       # C3
    3: [2, 6, 7, 10],      # P3
    4: [0, 1, 5],          # F7
    5: [4, 2, 6],          # T3
    6: [5, 3, 7],          # T5
    7: [3, 6, 10],         # O1
    8: [0, 1, 9, 11, 12],  # Fz
    9: [8, 2, 10, 13],     # Cz
    10: [9, 3, 7, 14, 18], # Pz
    11: [12, 15, 8, 0],    # Fp2
    12: [11, 13, 15, 8],   # F4
    13: [12, 14, 16, 9],   # C4
    14: [13, 17, 18, 10],  # P4
    15: [11, 12, 16],      # F8
    16: [15, 13, 17],      # T4
    17: [16, 14, 18],      # T6
    18: [14, 17, 10],      # O2
}


def load_monopolar(mat_file):
    """Load raw monopolar EEG (19 channels, 2000 samples)."""
    path = EEG_DIR / mat_file
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :N_SAMPLES]
    assert seg.shape[0] == 19, f"Expected 19 channels, got {seg.shape[0]}"
    return seg


def bandpass_filter(data, lo=0.5, hi=20.0, fs=200, order=4):
    """Bandpass filter."""
    sos = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass', output='sos')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        try:
            filtered[ch] = signal.sosfiltfilt(sos, data[ch])
        except Exception:
            filtered[ch] = data[ch]
    return filtered


def compute_laplacian(mono, neighbors_map):
    """Compute Laplacian (each channel minus mean of neighbors)."""
    n_ch, n_samp = mono.shape
    lap = np.zeros_like(mono)
    for ch in range(n_ch):
        nbrs = neighbors_map.get(ch, [])
        if nbrs:
            lap[ch] = mono[ch] - np.mean(mono[nbrs], axis=0)
        else:
            lap[ch] = mono[ch]
    return lap


def gfp_align(mono_filtered, discharge_times_sec, fs=200, window_ms=25):
    """Two-pass discharge-locked topography with Laplacian-GFP alignment.

    Identical to generate_discharge_topo_viewer.py's gfp_align.
    Returns: (mean_topo_mono, mean_topo_lap) or (None, None).
    """
    window_samples = int(window_ms * fs / 1000)
    epoch_half = int(50 * fs / 1000)
    n_ch, n_total = mono_filtered.shape

    lap = compute_laplacian(mono_filtered, LAP_NEIGHBORS)

    # Pass 1: Laplacian-GFP alignment
    gfp_aligned_samples = []
    for t in discharge_times_sec:
        center = int(t * fs)
        lo = max(0, center - window_samples)
        hi = min(n_total, center + window_samples + 1)
        if hi - lo < 3:
            continue
        segment_lap = lap[:, lo:hi]
        gfp_lap = np.std(segment_lap, axis=0)
        peak_sample = lo + np.argmax(gfp_lap)
        gfp_aligned_samples.append(peak_sample)

    if len(gfp_aligned_samples) < 2:
        return None, None

    # Extract epochs
    mono_epochs = []
    lap_epochs = []
    for s in gfp_aligned_samples:
        elo = s - epoch_half
        ehi = s + epoch_half + 1
        if elo < 0 or ehi > n_total:
            continue
        mono_epochs.append(mono_filtered[:, elo:ehi])
        lap_epochs.append(lap[:, elo:ehi])

    if len(mono_epochs) < 2:
        mean_topo_mono = np.mean([mono_filtered[:, s] for s in gfp_aligned_samples], axis=0)
        mean_topo_lap = np.mean([lap[:, s] for s in gfp_aligned_samples], axis=0)
    else:
        epoch_len = mono_epochs[0].shape[1]
        lap_template = np.mean(lap_epochs, axis=0)
        template_gfp = np.std(lap_template, axis=0)
        mid = epoch_len // 2
        max_shift = window_samples

        refined_voltages = []
        for mono_epoch, lap_epoch in zip(mono_epochs, lap_epochs):
            epoch_gfp = np.std(lap_epoch, axis=0)
            best_shift = 0
            best_corr = -np.inf
            for shift in range(-max_shift, max_shift + 1):
                t_lo = max(0, -shift)
                t_hi = min(epoch_len, epoch_len - shift)
                e_lo = max(0, shift)
                e_hi = min(epoch_len, epoch_len + shift)
                if t_hi - t_lo < 5:
                    continue
                corr = np.dot(template_gfp[t_lo:t_hi], epoch_gfp[e_lo:e_hi])
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
            aligned_mid = mid + best_shift
            if 0 <= aligned_mid < epoch_len:
                refined_voltages.append(mono_epoch[:, aligned_mid])

        if len(refined_voltages) < 2:
            mean_topo_mono = np.mean([mono_filtered[:, s] for s in gfp_aligned_samples], axis=0)
            mean_topo_lap = np.mean([lap[:, s] for s in gfp_aligned_samples], axis=0)
        else:
            refined_voltages = np.array(refined_voltages)
            lap_voltages = np.array([
                compute_laplacian(v.reshape(19, 1), LAP_NEIGHBORS).ravel()
                for v in refined_voltages
            ])
            gfp_weights = np.std(lap_voltages, axis=1) ** 2
            weight_sum = np.sum(gfp_weights)
            if weight_sum > 1e-12:
                mean_topo_mono = np.average(refined_voltages, axis=0, weights=gfp_weights)
                mean_topo_lap = np.average(lap_voltages, axis=0, weights=gfp_weights)
            else:
                mean_topo_mono = np.mean(refined_voltages, axis=0)
                mean_topo_lap = np.mean(lap_voltages, axis=0)

    mean_topo_mono = np.abs(mean_topo_mono)
    mean_topo_lap = np.abs(mean_topo_lap)
    return mean_topo_mono, mean_topo_lap


def generate_topoplot_on_ax(ax, mean_topo, ch_names_orig):
    """Generate topoplot directly on given axes."""
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in ch_names_orig]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(mean_topo))
    if vmax < 1e-10:
        vmax = 1.0

    image, _ = mne.viz.plot_topomap(mean_topo, info, axes=ax, show=False,
                                     contours=6, cmap='inferno', sensors=False,
                                     vlim=(0, vmax))

    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks='eeg')

    cmap = plt.cm.inferno
    for i, (orig_name, xy) in enumerate(zip(ch_names_orig, pos)):
        val_normalized = mean_topo[i] / vmax if vmax > 0 else 0
        bg_color = cmap(val_normalized)
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = 'white' if lum < 0.45 else 'black'
        # IMPROVEMENT 3: Significantly increase font size of electrode labels
        ax.text(xy[0], xy[1], orig_name, fontsize=9, ha='center', va='center',
                fontweight='bold', color=text_color, zorder=10)
    return image # Return image for colorbar


def plot_eeg_traces(ax, eeg_data, title, discharge_times=None,
                    highlight_left=False, spacing=120.0, label_discharges=False):
    """Plot 19-channel average reference EEG traces with channel group labels."""
    n_samples = eeg_data.shape[1]
    t = np.arange(n_samples) / FS

    # Channel groups for labeling
    GROUPS = [
        ('Left\nparasagittal', [0, 1, 2, 3]),
        ('Left\ntemporal', [4, 5, 6, 7]),
        ('Midline', [8, 9, 10]),
        ('Right\nparasagittal', [11, 12, 13, 14]),
        ('Right\ntemporal', [15, 16, 17, 18]),
    ]

    y_pos = 0
    yticks = []
    yticklabels = []
    channel_y_positions = {}
    group_y_ranges = {}  # group_name -> (y_top, y_bottom)

    group_idx = 0
    channels_in_group = []

    for idx in DISPLAY_ORDER:
        if idx == -1:
            # Save group range
            if channels_in_group and group_idx < len(GROUPS):
                gname = GROUPS[group_idx][0]
                ys = [channel_y_positions[c] for c in channels_in_group]
                group_y_ranges[gname] = (max(ys) + spacing * 0.4, min(ys) - spacing * 0.4)
                group_idx += 1
                channels_in_group = []
            y_pos -= spacing * 1.5
            continue
        trace = eeg_data[idx] * 2.5
        ax.plot(t, trace + y_pos, color='black', linewidth=0.5, clip_on=True)
        yticks.append(y_pos)
        yticklabels.append(MONO_CHANNELS[idx])
        channel_y_positions[idx] = y_pos
        channels_in_group.append(idx)
        y_pos -= spacing

    # Save last group
    if channels_in_group and group_idx < len(GROUPS):
        gname = GROUPS[group_idx][0]
        ys = [channel_y_positions[c] for c in channels_in_group]
        group_y_ranges[gname] = (max(ys) + spacing * 0.4, min(ys) - spacing * 0.4)

    ax.set_xlim(0, 10)
    y_top = spacing * 0.8
    y_bottom = y_pos + spacing * 0.4
    ax.set_ylim(y_bottom, y_top)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel('(sec)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Channel group labels (rotated, left side)
    for gname, (yt, yb) in group_y_ranges.items():
        ymid = (yt + yb) / 2
        ax.text(-0.8, ymid, gname, ha='center', va='center', fontsize=6,
                fontstyle='italic', color='#555', rotation=90,
                transform=ax.get_yaxis_transform(), clip_on=False)

    # Discharge time markers
    if discharge_times is not None:
        for i, dt_val in enumerate(discharge_times):
            if 0 <= dt_val <= 10:
                ax.axvline(x=dt_val, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
                # Label discharge times (t1, t2, ... tn) at top
                if label_discharges:
                    if i < 4:
                        label = f't$_{i+1}$'
                    elif i == 4:
                        label = '...'
                    elif i == len(discharge_times) - 1:
                        label = f't$_n$'
                    else:
                        continue
                    ax.text(dt_val, y_top - spacing * 0.1, label, ha='center', va='bottom',
                            fontsize=6, color='red', fontstyle='italic', clip_on=False)

    # Light blue shading on left hemisphere channels
    if highlight_left:
        left_y_vals = [channel_y_positions[idx] for idx in LEFT_CH_INDICES
                       if idx in channel_y_positions]
        if left_y_vals:
            y_hi = max(left_y_vals) + spacing * 0.5
            y_lo = min(left_y_vals) - spacing * 0.5
            ax.axhspan(y_lo, y_hi, color='lightblue', alpha=0.15, zorder=0)

    ax.set_title(title, fontsize=11, fontweight='bold')


def draw_flowchart(ax):
    """Draw architecture flowchart — PaperBanana style with colored group panels."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors matching PaperBanana: tinted group panels, white sub-boxes
    GREEN_BG = '#E8F5E9'   # laterality group
    GREEN_BD = '#66BB6A'
    SALMON_BG = '#FBE9E7'  # discharge detection group
    SALMON_BD = '#EF9A9A'
    BLUE_BG = '#E3F2FD'    # topographic localization group
    BLUE_BD = '#90CAF9'
    BOX_BG = '#FAFAFA'     # sub-step boxes
    BOX_BD = '#BDBDBD'
    DARK_BG = '#424242'    # ChannelPD-Net

    def add_box(x, y, w, h, text, facecolor=BOX_BG, edgecolor=BOX_BD,
                fontsize=7.5, text_color='#333', linewidth=0.8, bold_first=True):
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.08",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=linewidth, zorder=3)
        ax.add_patch(box)
        lines = text.split('\n')
        n_lines = len(lines)
        line_spacing = min(fontsize * 0.2, h / (n_lines + 0.5))
        start_y = y + (n_lines - 1) * line_spacing / 2
        for i, line in enumerate(lines):
            fs = fontsize
            fw = 'bold' if (i == 0 and bold_first) else 'normal'
            if i == 0 and bold_first:
                fs = fontsize + 0.5
            ax.text(x, start_y - i * line_spacing, line, ha='center', va='center',
                    fontsize=fs, fontweight=fw, color=text_color, zorder=4)

    def add_panel(x, y, w, h, facecolor, edgecolor):
        """Add a colored background panel (group region)."""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=1.0, alpha=0.6, zorder=1)
        ax.add_patch(box)

    def add_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    # Title
    ax.text(5, 9.85, 'B. Pipeline Architecture', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

    # ── Input text + ChannelPD-Net ──
    ax.text(2.8, 9.35, '18 Independent\nBipolar Channels', ha='center', va='center',
            fontsize=7, color='#555')
    ax.annotate('', xy=(4.2, 9.35), xytext=(3.6, 9.35),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))

    add_box(6.0, 9.35, 3.5, 0.6, "ChannelPD-Net\n(CNN+Attention)",
            facecolor=DARK_BG, edgecolor='#333', text_color='white', fontsize=9, linewidth=1.5)

    # Output labels
    ax.text(4.5, 8.85, 'PD Probability', ha='center', fontsize=6.5, color='#666', style='italic')
    ax.text(7.5, 8.85, 'Frequency Estimate', ha='center', fontsize=6.5, color='#666', style='italic')

    # Horizontal line + arrows down
    ax.plot([1.5, 8.5], [8.7, 8.7], color='#999', linewidth=0.8, zorder=1)
    add_arrow(2.2, 8.7, 2.2, 8.3)
    add_arrow(5.0, 8.7, 5.0, 8.3)
    add_arrow(7.8, 8.7, 7.8, 8.3)

    # ── Colored group panels ──
    add_panel(2.2, 6.5, 3.0, 3.4, GREEN_BG, GREEN_BD)    # Laterality
    add_panel(5.0, 5.9, 3.0, 4.6, SALMON_BG, SALMON_BD)  # Discharge Detection
    add_panel(7.8, 6.3, 3.0, 3.8, BLUE_BG, BLUE_BD)      # Topographic Localization

    # ── Branch 1: Laterality Detection (green) ──
    ax.text(2.2, 8.05, 'Laterality\nDetection', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2E7D32')

    add_box(2.2, 7.15, 2.5, 0.5, "Compare\nL vs R Mean\nProbabilities",
            fontsize=7)
    add_arrow(2.2, 7.65, 2.2, 7.42)

    ax.text(2.2, 6.65, 'Laterality\n(Side)', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#333')
    add_arrow(2.2, 6.88, 2.2, 6.78)

    # ── Branch 2: Discharge Detection (salmon) ──
    ax.text(5.0, 8.05, 'Discharge\nDetection', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#C62828')

    add_box(4.3, 7.2, 1.2, 0.45, "8-channel\nCET-UNet", fontsize=6.5)
    add_box(5.7, 7.2, 1.2, 0.45, "CNN+ACF\nEnsemble", fontsize=6.5)
    add_arrow(4.6, 7.65, 4.3, 7.44)
    add_arrow(5.4, 7.65, 5.7, 7.44)

    ax.text(4.3, 6.8, 'Evidence\nTrace', ha='center', fontsize=6, color='#666')
    ax.text(5.7, 6.8, 'Frequency\nPrior', ha='center', fontsize=6, color='#666')

    add_arrow(4.6, 6.65, 5.0, 6.3)
    add_arrow(5.4, 6.65, 5.0, 6.3)

    add_box(5.0, 6.05, 2.3, 0.45, "Dynamic\nProgramming", fontsize=7)

    add_arrow(5.0, 5.82, 5.0, 5.55)
    add_box(5.0, 5.3, 2.3, 0.45, "EM Template\nRefinement & Filtering", fontsize=6.5)

    add_arrow(5.0, 5.06, 5.0, 4.8)
    ax.text(5.0, 4.55, 'Discharge Times (t\u2081 \u00b7\u00b7\u00b7 t$_n$)\nFrequency', ha='center',
            fontsize=7.5, fontweight='bold', color='#333')

    # ── Branch 3: Topographic Localization (blue) ──
    ax.text(7.8, 8.05, 'Topographic\nLocalization', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#1565C0')

    add_box(7.8, 7.2, 2.3, 0.5, "Extract\nMonopolar\nVoltage", fontsize=6.5)
    add_arrow(7.8, 7.65, 7.8, 7.47)

    add_arrow(7.8, 6.94, 7.8, 6.7)
    add_box(7.8, 6.4, 2.3, 0.55, "Laplacian-GFP\nAlignment\n(\u00b125ms)", fontsize=6.5)

    add_arrow(7.8, 6.12, 7.8, 5.9)
    add_box(7.8, 5.6, 2.3, 0.55, "Template\nRefinement &\nGFP-weighted\nAveraging", fontsize=6.5)

    add_arrow(7.8, 5.32, 7.8, 5.1)
    # Small topoplot icon placeholder
    circle = plt.Circle((7.8, 4.85), 0.2, facecolor='#FFF3E0', edgecolor='#E65100',
                         linewidth=0.8, zorder=3)
    ax.add_patch(circle)
    ax.text(7.8, 4.85, '\u2609', ha='center', va='center', fontsize=10, color='#E65100')

    ax.text(7.8, 4.45, 'Localization', ha='center',
            fontsize=7.5, fontweight='bold', color='#333')


def main():
    print("=" * 60)
    print("Fig 2: PD Characterization Pipeline")
    print("=" * 60)

    # ── Load and process EEG ──
    print("Loading EEG...", flush=True)
    mono_raw = load_monopolar(MAT_FILE)

    # Average reference
    avg = np.mean(mono_raw, axis=0)
    mono_car = mono_raw - avg[np.newaxis, :]

    # Bandpass filter
    mono_filt = bandpass_filter(mono_car, lo=0.5, hi=20.0)
    mono_filt = np.clip(mono_filt, -300, 300)

    # ── Get discharge times ──
    print("Loading discharge times...", flush=True)
    with open(LABELS_DIR / 'discharge_times.json') as f:
        dt_data = json.load(f)
    dt_entry = dt_data.get(SEGMENT_ID)
    if isinstance(dt_entry, dict):
        discharge_times = dt_entry.get('global_times', [])
    elif isinstance(dt_entry, list):
        discharge_times = dt_entry
    else:
        discharge_times = []
    print(f"  {len(discharge_times)} discharge times", flush=True)

    # ── Run PDProfiler for laterality ──
    print("Running PDProfiler...", flush=True)
    sys.path.insert(0, str(CODE_DIR))
    from pd_profiler import PDProfiler
    charzer = PDProfiler()

    # PDProfiler expects 18-ch bipolar
    bipolar_pairs = [
        ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('Fz', 'Cz'), ('Cz', 'Pz'),
    ]
    ch_idx = {ch: i for i, ch in enumerate(MONO_CHANNELS)}
    bipolar_raw = np.zeros((18, N_SAMPLES))
    for i, (a, b) in enumerate(bipolar_pairs):
        bipolar_raw[i] = mono_raw[ch_idx[a]] - mono_raw[ch_idx[b]]

    char_result = charzer.characterize(bipolar_raw, subtype='lpd')
    laterality = char_result.get('laterality', 'unknown')
    print(f"  Laterality: {laterality}", flush=True)

    # ── Compute topography ──
    print("Computing discharge topography...", flush=True)
    mono_filt_raw = bandpass_filter(mono_raw, lo=0.5, hi=20.0)  # filter raw (not CAR) for topo
    mean_topo_mono, mean_topo_lap = gfp_align(mono_filt_raw, discharge_times)
    if mean_topo_lap is None:
        print("  WARNING: gfp_align returned None, using fallback")
        mean_topo_lap = np.ones(19)
        mean_topo_mono = np.ones(19)

    # ── Generate verbal description ──
    print("Generating verbal description...", flush=True)
    sys.path.insert(0, str(PROJECT_DIR / 'paper_materials'))
    from generate_discharge_topo_viewer import generate_verbal_from_topo
    ipis = np.diff(discharge_times)
    frequency = 1.0 / np.median(ipis) if len(ipis) > 0 else np.nan
    try:
        verbal = generate_verbal_from_topo('lpd', frequency, mean_topo_mono,
                                            laterality_from_pdchar=laterality)
    except Exception as e:
        print(f"  Verbal description error: {e}")
        verbal = f"LPD, {laterality} sided, {frequency:.1f} Hz"
    print(f"  Verbal: {verbal}", flush=True)

    # ── Create figure ──
    print("Building figure...", flush=True)
    # Adjusted figure height for better balance across panels
    fig = plt.figure(figsize=(22, 7.5), facecolor='white')

    # Three panels: A (30%), B (40%), C (30%)
    # Adjusted top/bottom for more vertical margin
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.28, 0.44, 0.28],
                           left=0.04, right=0.96, top=0.94, bottom=0.06,
                           wspace=0.06)

    # ── Panel A: Input EEG ──
    ax_a = fig.add_subplot(gs[0, 0])
    plot_eeg_traces(ax_a, mono_filt,
                    title='')

    # ── Panel B: Architecture Flowchart ──
    ax_b = fig.add_subplot(gs[0, 1])
    draw_flowchart(ax_b)

    # ── Panel C: Output Visualization ──
    # Full-height EEG (same as Panel A), topoplot overlaid in lower-right
    ax_c = fig.add_subplot(gs[0, 2])
    is_left = laterality == 'left'
    plot_eeg_traces(ax_c, mono_filt,
                    title='',
                    discharge_times=discharge_times,
                    highlight_left=is_left,
                    label_discharges=True)

    # Topoplot as inset in lower-right corner of Panel C
    c_pos = ax_c.get_position()
    topo_size = 0.07
    inset_left = c_pos.x1 - topo_size - 0.01
    inset_bottom = c_pos.y0 + 0.02
    ax_topo_inset = fig.add_axes([inset_left, inset_bottom, topo_size, topo_size * (22/9)])
    generate_topoplot_on_ax(ax_topo_inset, mean_topo_lap, MONO_CHANNELS)
    for spine in ax_topo_inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
        spine.set_color('#666')

    # Verbal description below topoplot inset
    wrapped = verbal
    if len(verbal) > 45:
        words = verbal.split()
        lines = []
        current = []
        for w in words:
            current.append(w)
            if len(' '.join(current)) > 40:
                lines.append(' '.join(current))
                current = []
        if current:
            lines.append(' '.join(current))
        wrapped = '\n'.join(lines)

    # Verbal description in clean rounded box at bottom of Panel C
    c_pos2 = ax_c.get_position()
    fig.text(c_pos2.x0 + c_pos2.width / 2, c_pos2.y0 - 0.01, verbal,
             ha='center', va='top', fontsize=7, fontstyle='italic',
             fontfamily='sans-serif', color='#333',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                       edgecolor='#999', linewidth=0.8, alpha=0.95))

    # (verbal description placed via fig.text above)
    # Dummy to satisfy any remaining code expecting ax_verbal
    class _Dummy:
        def text(self, *a, **kw): pass
        def axis(self, *a, **kw): pass
    ax_verbal = _Dummy()
    ax_verbal.text(0.5, 0.5, '',
                   ha='center', va='center', fontsize=10, fontstyle='italic',
                   color='#333',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f0',
                             edgecolor='none', alpha=0.9)) # Removed edgecolor
    ax_verbal.axis('off') # Hide axes for the verbal description

    # ── Save ──
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()