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

# CRITIQUE 1: Global Font and Text Styling
# Apply a consistent sans-serif font to all text elements.
# Set a uniform dark grey color (#333333) for all text.
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
matplotlib.rcParams['text.color'] = '#333333' # Uniform dark grey color for all text

# Increase base font sizes for better readability and consistency
matplotlib.rcParams['font.size'] = 10 # Base font size, slightly larger than previous 9
matplotlib.rcParams['axes.titlesize'] = 12 # Panel titles
matplotlib.rcParams['axes.labelsize'] = 10 # X-axis label
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 10 # Channel labels in EEG plots
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['figure.titlesize'] = 14 # Not directly used for this figure, but good practice

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import mne
mne.set_log_level('WARNING')

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, '/Users/mwestover/GithubRepos/morgoth-viewer') # This path might be specific to user's setup, keeping as is.

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

# Channel group indices for Panel A labels and shading
LEFT_PARASAGITTAL_CH_INDICES = [0, 1, 2, 3] # Fp1, F3, C3, P3
LEFT_TEMPORAL_CH_INDICES = [4, 5, 6, 7] # F7, T3, T5, O1
MIDLINE_CH_INDICES = [8, 9, 10] # Fz, Cz, Pz
RIGHT_PARASAGITTAL_CH_INDICES = [11, 12, 13, 14] # Fp2, F4, C4, P4
RIGHT_TEMPORAL_CH_INDICES = [15, 16, 17, 18] # F8, T4, T6, O2

# Combined for hemisphere shading
LEFT_CH_INDICES = LEFT_PARASAGITTAL_CH_INDICES + LEFT_TEMPORAL_CH_INDICES
RIGHT_CH_INDICES = RIGHT_PARASAGITTAL_CH_INDICES + RIGHT_TEMPORAL_CH_INDICES


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

    # CRITIQUE 5: Topoplot Inset Enhancement - sensors=False to avoid default dots, we'll draw labels over them
    image, _ = mne.viz.plot_topomap(mean_topo, info, axes=ax, show=False,
                                     contours=6, cmap='inferno', sensors=False,
                                     vlim=(0, vmax))

    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks='eeg')

    cmap = plt.cm.inferno
    # CRITIQUE 5: Add electrode labels (Fp1, F3, etc.) directly onto the topoplot itself
    # CRITIQUE 1: Increase font size and make channel labels bold.
    for i, (orig_name, xy) in enumerate(zip(ch_names_orig, pos)):
        val_normalized = mean_topo[i] / vmax if vmax > 0 else 0
        bg_color = cmap(val_normalized)
        # Calculate luminance to choose white or black text for contrast
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = 'white' if lum < 0.45 else 'black'
        ax.text(xy[0], xy[1], orig_name, fontsize=9, ha='center', va='center', # Increased fontsize
                fontweight='bold', color=text_color, zorder=10) # Made bold
    return image # Return image for colorbar


def plot_eeg_traces(ax, eeg_data, title, discharge_times=None,
                    highlight_left=False, spacing=250.0): # CRITIQUE 3: Increased vertical spacing
    """Plot 19-channel average reference EEG traces.

    Args:
        ax: matplotlib axes
        eeg_data: (19, N_SAMPLES) filtered average-reference data
        title: panel title
        discharge_times: optional list of times (sec) for red dashed lines
        highlight_left: if True, shade left hemisphere channels light blue
        spacing: vertical spacing between channels in uV
    """
    n_samples = eeg_data.shape[1]
    t = np.arange(n_samples) / FS

    y_pos = 0
    yticks = []
    yticklabels = []
    channel_y_positions = {}  # idx -> y_pos

    for idx in DISPLAY_ORDER:
        if idx == -1:
            # CRITIQUE 3: Significantly increased vertical spacing between channel groups
            y_pos -= spacing * 2.0 # Adjusted multiplier for new base spacing
            continue
        # EEG traces are noticeably thinner (already linewidth=0.2)
        trace = eeg_data[idx] * 2.5 # Amplitude multiplier, kept as is
        ax.plot(t, trace + y_pos, color='black', linewidth=0.2, clip_on=True)
        yticks.append(y_pos)
        yticklabels.append(MONO_CHANNELS[idx])
        channel_y_positions[idx] = y_pos
        y_pos -= spacing

    ax.set_xlim(0, 10)
    # Adjusted top/bottom limits to account for increased spacing and amplitude
    y_top = spacing * 0.8
    y_bottom = y_pos + spacing * 0.4
    ax.set_ylim(y_bottom, y_top)

    # CRITIQUE 3: Remove any y-axis labels. The channel names are effectively labels.
    ax.set_yticks(yticks)
    # CRITIQUE 1: Increase font size and make channel labels bold in Panels A and C.
    ax.set_yticklabels(yticklabels, fontsize=10, fontweight='bold') # Increased fontsize, made bold
    
    # CRITIQUE 3: Simplify the x-axis label to "sec"
    ax.set_xlabel('sec', fontsize=matplotlib.rcParams['axes.labelsize']) # Use rcParams for consistency
    ax.tick_params(axis='x', labelsize=matplotlib.rcParams['xtick.labelsize'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False) # Keep left spine invisible
    ax.tick_params(axis='y', length=0) # No tick marks for y-axis

    # Discharge time markers (red dashed vertical lines)
    if discharge_times is not None:
        for dt in discharge_times:
            if 0 <= dt <= 10:
                # Thinner red dashed lines (already linewidth=0.5)
                ax.axvline(x=dt, color='red', linestyle='--', linewidth=0.5, alpha=0.7)

    # Light blue shading on left hemisphere channels
    if highlight_left:
        # Use combined LEFT_CH_INDICES for shading
        left_y_vals = [channel_y_positions[idx] for idx in LEFT_CH_INDICES
                       if idx in channel_y_positions]
        if left_y_vals:
            y_hi = max(left_y_vals) + spacing * 0.5
            y_lo = min(left_y_vals) - spacing * 0.5
            ax.axhspan(y_lo, y_hi, color='lightblue', alpha=0.15, zorder=0)

    # Panel title: consistent font size and weight (from rcParams)
    ax.set_title(title, fontsize=matplotlib.rcParams['axes.titlesize'], fontweight='normal', ha='center', va='bottom')
    
    return channel_y_positions # Return for grouping labels


def draw_flowchart(ax):
    """Draw the architecture flowchart in Panel B."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # CRITIQUE 2: Target colors for flowchart boxes
    COLOR_GROUPING_BLUE = '#E0F2F7' # Light blue for top-level grouping
    COLOR_PROCESSING_ORANGE = '#FFF3E0' # Light orange/yellow for processing steps
    COLOR_OUTPUT_GREEN = '#E8F5E9' # Light green for output categories
    COLOR_BORDER_ARROW = '#333333' # Dark grey for borders and arrows
    TEXT_COLOR_DARK = '#333333' # Dark grey for text inside boxes

    def add_box(x, y, w, h, text, facecolor,
                fontsize=matplotlib.rcParams['font.size'], text_color=TEXT_COLOR_DARK):
        """Add a rounded rectangle with centered text."""
        # CRITIQUE 2: Rounded rectangular boxes, thin dark grey borders
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.3", # More pronounced rounding
                             facecolor=facecolor, edgecolor=COLOR_BORDER_ARROW,
                             linewidth=0.8, alpha=1.0, zorder=2) # Thin dark grey border
        ax.add_patch(box)
        lines = text.split('\n')
        n_lines = len(lines)
        # Adjusted line spacing for better fit with new font sizes
        line_spacing = fontsize * 0.4
        start_y = y + (n_lines - 1) * line_spacing / 2

        for i, line in enumerate(lines):
            fs = fontsize
            fw = 'normal'
            if i == 0: # First line is bold and slightly larger (CRITIQUE 1)
                fw = 'bold'
                fs = fontsize + 1.5
            ax.text(x, start_y - i * line_spacing, line, ha='center', va='center',
                    fontsize=fs, fontweight=fw, color=text_color, zorder=3)

    def add_arrow(x1, y1, x2, y2, color=COLOR_BORDER_ARROW, lw=0.8, # CRITIQUE 2: Thinner, dark grey arrows
                  arrowstyle='-|>,head_width=0.25,head_length=0.5'): # Smaller, well-defined arrowhead
        """Add a straight arrow with enhanced visibility."""
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle=arrowstyle, color=color,
                                lw=lw, mutation_scale=10, zorder=1) # mutation_scale affects arrowhead size
        ax.add_patch(arrow)

    # ── Top box: ChannelPD-Net ──
    # CRITIQUE 2: Light blue for top-level grouping
    # CRITIQUE 6: Adjusted y, h, and fontsize for better readability and vertical distribution
    add_box(5, 9.0, 8.5, 1.3,
            "ChannelPD-Net\nPer-channel CNN+Attention (\u00d718)\n18 PD Probabilities + 18 Frequency Estimates",
            facecolor=COLOR_GROUPING_BLUE, fontsize=matplotlib.rcParams['font.size'])

    # Arrows from top box to three branches (adjusted y coordinates for new box positions)
    # CRITIQUE 6: Adjust arrow positions for better spacing
    add_arrow(2.5, 8.3, 2.0, 6.8)
    add_arrow(5.0, 8.3, 5.0, 6.8)
    add_arrow(7.5, 8.3, 8.0, 6.8)

    # ── Branch 1 (left): Laterality Detection ──
    # CRITIQUE 2: Light orange/yellow for processing steps
    # CRITIQUE 6: Adjusted y, h, and fontsize for better readability and vertical distribution
    add_box(2.0, 6.0, 3.2, 2.5,
            "Laterality Detection\n\nL vs R hemisphere\nmean PD probability\n\nOutput: Left / Right\nAUC = 0.963",
            facecolor=COLOR_PROCESSING_ORANGE, fontsize=matplotlib.rcParams['font.size'])

    # ── Branch 2 (center): HemiCET+DP ──
    # CRITIQUE 2: Light orange/yellow for processing steps
    # CRITIQUE 6: Adjusted y, h, and fontsize for better readability and vertical distribution
    # CRITIQUE 1: Replaced Unicode subscripts with standard text subscripts for robustness and consistency
    add_box(5.0, 5.6, 3.2, 3.3,
            "HemiCET+DP\nDischarge Detection\n\n8-ch CET-UNet -> Evidence\nCNN+ACF Frequency Prior\nDP with Periodic Prior\nEM Refinement + Filtering\n\nOutput: t_1, t_2, ..., t_n\nFreq = 1 / median(IPI)",
            facecolor=COLOR_PROCESSING_ORANGE, fontsize=matplotlib.rcParams['font.size'])

    # ── Branch 3 (right): Topographic Localization ──
    # CRITIQUE 2: Light orange/yellow for processing steps
    # CRITIQUE 6: Adjusted y, h, and fontsize for better readability and vertical distribution
    add_box(8.0, 6.0, 3.2, 2.5,
            "Discharge-Locked\nTopographic Localization\n\nLaplacian-GFP Alignment\nTwo-Pass Template Refinement\nGFP^2-Weighted Averaging\n-> Topoplot + Description",
            facecolor=COLOR_PROCESSING_ORANGE, fontsize=matplotlib.rcParams['font.size'])

    # Arrows down to output boxes (adjusted y coordinates for new box positions)
    # CRITIQUE 6: Adjust arrow positions for better spacing
    add_arrow(2.0, 4.7, 2.0, 3.75)
    add_arrow(5.0, 3.9, 5.0, 3.75)
    add_arrow(8.0, 4.7, 8.0, 3.75)

    # ── Output boxes (bottom) ──
    # CRITIQUE 2: Light green for output categories
    # CRITIQUE 6: Explicit fontsize for single-line output, adjusted y for spacing
    add_box(2.0, 3.3, 2.8, 0.9,
            "Laterality",
            facecolor=COLOR_OUTPUT_GREEN, fontsize=matplotlib.rcParams['font.size'] + 1)

    add_box(5.0, 3.3, 2.8, 0.9,
            "Timing + Frequency",
            facecolor=COLOR_OUTPUT_GREEN, fontsize=matplotlib.rcParams['font.size'] + 1)

    add_box(8.0, 3.3, 2.8, 0.9,
            "Spatial Localization",
            facecolor=COLOR_OUTPUT_GREEN, fontsize=matplotlib.rcParams['font.size'] + 1)

    # Panel B title (descriptive part, 'B.' is handled by fig.text)
    ax.set_title('Pipeline Architecture', fontsize=matplotlib.rcParams['axes.titlesize'], fontweight='normal', ha='center', va='bottom')


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
    # CRITIQUE 6: Overall Spacing and Margins - Wider figure, more generous margins
    fig = plt.figure(figsize=(22, 9), facecolor='white') # Wider and slightly taller

    # CRITIQUE 6: Adjusted top/bottom/left/right/wspace for more vertical margin and tighter horizontal spacing
    # CRITIQUE 1: Overall Layout and Panel Proportions - A (28%), B (44%), C (28%)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.28, 0.44, 0.28],
                           left=0.03, right=0.97, top=0.92, bottom=0.08, # More margin
                           wspace=0.05) # Keep wspace relatively tight as per target

    # ── Panel A: Input EEG ──
    ax_a = fig.add_subplot(gs[0, 0])
    channel_y_positions_a = plot_eeg_traces(ax_a, mono_filt,
                                            title='Input: 19-Channel EEG (10s, 200 Hz)')

    # CRITIQUE 4: Panel A Hemisphere Grouping Labels
    # Calculate average y-positions for each group
    group_labels = {
        "Left parasagittal": LEFT_PARASAGITTAL_CH_INDICES,
        "Left temporal": LEFT_TEMPORAL_CH_INDICES,
        "Midline": MIDLINE_CH_INDICES,
        "Right parasagittal": RIGHT_PARASAGITTAL_CH_INDICES,
        "Right temporal": RIGHT_TEMPORAL_CH_INDICES,
    }
    
    # Determine x-position for labels (left of the channels)
    x_label_pos = ax_a.get_xlim()[0] - 0.7 # Adjust as needed

    for label_text, indices in group_labels.items():
        valid_y_positions = [channel_y_positions_a[idx] for idx in indices if idx in channel_y_positions_a]
        if valid_y_positions:
            # Calculate the center of the group's vertical extent
            y_center = (max(valid_y_positions) + min(valid_y_positions)) / 2
            ax_a.text(x_label_pos, y_center, label_text,
                     ha='right', va='center', rotation=90,
                     fontsize=10, fontweight='bold', color='#333333',
                     transform=ax_a.get_xaxis_transform()) # Use transform to keep x-pos relative to axis

    # ── Panel B: Architecture Flowchart ──
    ax_b = fig.add_subplot(gs[0, 1])
    draw_flowchart(ax_b)

    # ── Panel C: Output Visualization ──
    ax_c = fig.add_subplot(gs[0, 2])
    is_left = laterality == 'left'
    plot_eeg_traces(ax_c, mono_filt,
                    title='Output: Characterized LPD',
                    discharge_times=discharge_times,
                    highlight_left=is_left)

    # CRITIQUE 1: Panel labels (A, B, C) are larger and bold.
    fig.text(0.015, 0.95, 'A.', fontsize=18, fontweight='bold', ha='left', va='center') # Adjusted x-pos for new left margin
    fig.text(0.30, 0.95, 'B.', fontsize=18, fontweight='bold', ha='left', va='center')
    fig.text(0.70, 0.95, 'C.', fontsize=18, fontweight='bold', ha='left', va='center')


    # Topoplot as inset in lower-right corner of Panel C
    c_pos = ax_c.get_position()
    # CRITIQUE 5: Increase the overall size of the topoplot inset in Panel C.
    topo_size_frac = 0.22 # Relative size of the inset (fraction of figure width), increased from 0.15
    inset_left = c_pos.x1 - topo_size_frac - 0.015 # Adjusted for new size and margin
    inset_bottom = c_pos.y0 + 0.02
    ax_topo_inset = fig.add_axes([inset_left, inset_bottom, topo_size_frac, topo_size_frac]) # Square aspect ratio
    generate_topoplot_on_ax(ax_topo_inset, mean_topo_lap, MONO_CHANNELS)
    ax_topo_inset.set_facecolor('none') # Transparent background

    # CRITIQUE 5: Add a thin, dark grey border around the topoplot.
    for spine in ax_topo_inset.spines.values():
        spine.set_visible(True) # Make spines visible
        spine.set_edgecolor('#333333') # Dark grey border
        spine.set_linewidth(0.8) # Thin border

    # Verbal description below topoplot inset
    wrapped = verbal
    if len(verbal) > 45: # Adjust line wrap length if needed
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

    # CRITIQUE 1: Font for verbal description - consistent dark grey sans-serif
    # CRITIQUE 6: Background for verbal description - white, no border (already implemented)
    fig.text(inset_left + topo_size_frac / 2, inset_bottom - 0.01, wrapped,
             ha='center', va='top', fontsize=matplotlib.rcParams['font.size'] - 1, # Slightly smaller than base
             fontstyle='normal',
             fontfamily='sans-serif', color='#333333', # Ensure dark grey
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='none', alpha=0.9))

    # ── Save ──
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()