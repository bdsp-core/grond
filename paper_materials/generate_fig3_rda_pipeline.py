#!/usr/bin/env python3
"""
Generate Fig 3: RDA Characterization Pipeline composite figure.

Three-panel horizontal layout:
  Panel A (left):   Input 19-channel EEG (average reference, 10s)
  Panel B (center): Architecture flowchart (W05 iterative narrowband refinement)
  Panel C (right):  Output visualization (EEG + narrowband overlay, topoplot, verbal)

Uses the Easy LRDA case: sub-S0001115633229_20190719143934.mat

Usage:
    conda run -n morgoth python paper_materials/generate_fig3_rda_pipeline.py
"""

import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # Added for colorbar

import mne
mne.set_log_level('WARNING')

# -- Paths --
PROJECT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_DIR / 'code'
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'paper_materials'))
sys.path.insert(0, '/Users/mwestover/GithubRepos/morgoth-viewer')

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig3_rda_pipeline.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# -- Constants --
FS = 200
N_SAMPLES = 2000
MAT_FILE = 'sub-S0001115633229_20190719143934.mat'
SEGMENT_ID = MAT_FILE.replace('.mat', '')

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# Display order: L parasag, L temporal, gap, midline, gap, R parasag, R temporal
DISPLAY_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18]

LEFT_CH_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
RIGHT_CH_INDICES = [11, 12, 13, 14, 15, 16, 17, 18]

LAP_NEIGHBORS = {
    0: [1, 4, 8, 11],
    1: [0, 2, 4, 8],
    2: [1, 3, 5, 9],
    3: [2, 6, 7, 10],
    4: [0, 1, 5],
    5: [4, 2, 6],
    6: [5, 3, 7],
    7: [3, 6, 10],
    8: [0, 1, 9, 11, 12],
    9: [8, 2, 10, 13],
    10: [9, 3, 7, 14, 18],
    11: [12, 15, 8, 0],
    12: [11, 13, 15, 8],
    13: [12, 14, 16, 9],
    14: [13, 17, 18, 10],
    15: [11, 12, 16],
    16: [15, 13, 17],
    17: [16, 14, 18],
    18: [14, 17, 10],
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
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass', output='sos')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        try:
            filtered[ch] = sosfiltfilt(sos, data[ch])
        except Exception:
            filtered[ch] = data[ch]
    return filtered


def compute_laplacian_vector(vec, neighbors_map):
    """Compute Laplacian of a 19-element vector."""
    lap = np.zeros_like(vec)
    for ch in range(len(vec)):
        nbrs = neighbors_map.get(ch, [])
        if nbrs:
            lap[ch] = vec[ch] - np.mean(vec[nbrs])
        else:
            lap[ch] = vec[ch]
    return lap


def compute_amplitude_envelope(mono, freq_hz, bw=0.4):
    """Compute narrowband amplitude envelope per channel.

    Returns:
        amplitude_vector: (19,) mean absolute Hilbert envelope
        narrowband: (19, N_SAMPLES) narrowband-filtered data
    """
    lo = max(freq_hz - bw, 0.1)
    hi = min(freq_hz + bw, FS / 2 - 0.1)
    if lo >= hi:
        return np.zeros(mono.shape[0]), np.zeros_like(mono)

    sos = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    narrowband = np.zeros_like(mono)
    amplitude_vector = np.zeros(mono.shape[0])

    for ch in range(mono.shape[0]):
        try:
            nb = sosfiltfilt(sos, mono[ch])
            narrowband[ch] = nb
            amplitude_vector[ch] = np.mean(np.abs(hilbert(nb)))
        except Exception:
            pass

    return amplitude_vector, narrowband


def generate_topoplot_on_ax(ax, mean_topo, ch_names_orig, title=''): # Title is now empty by default
    """Generate topoplot directly on given axes using inferno colormap."""
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
        # Improvement 4: Increase font size and add subtle background for electrode labels
        ax.text(xy[0], xy[1], orig_name, fontsize=8, ha='center', va='center', # Fontsize increased from 5 to 8
                fontweight='bold', color=text_color, zorder=10,
                bbox=dict(boxstyle="round,pad=0.05", fc=(1,1,1,0.4), ec='none')) # Added transparent background
    
    # Title is empty, but if it were present, this would apply
    ax.set_title(title, fontsize=9, fontweight='bold')

    return image # Return image for colorbar


def plot_eeg_traces(ax, eeg_data, title, narrowband=None,
                    highlight_side=None, spacing=180.0): # Improvement 1: Increased spacing from 120.0 to 180.0
    """Plot 19-channel average reference EEG traces.

    Args:
        ax: matplotlib axes
        eeg_data: (19, N_SAMPLES) filtered average-reference data
        title: panel title
        narrowband: optional (19, N_SAMPLES) narrowband overlay (green)
        highlight_side: 'left' or 'right' to shade hemisphere channels light blue
        spacing: vertical spacing between channels in uV
    """
    n_samples = eeg_data.shape[1]
    t = np.arange(n_samples) / FS

    y_pos = 0
    yticks = []
    yticklabels = []
    channel_y_positions = {}

    for idx in DISPLAY_ORDER:
        if idx == -1:
            y_pos -= spacing * 0.4
            continue
        trace = eeg_data[idx]
        ax.plot(t, trace + y_pos, color='black', linewidth=0.4, clip_on=True)

        # Green narrowband overlay
        if narrowband is not None:
            nb_trace = narrowband[idx]
            ax.plot(t, nb_trace + y_pos, color='#27AE60', linewidth=0.6,
                    alpha=0.7, clip_on=True)

        yticks.append(y_pos)
        yticklabels.append(MONO_CHANNELS[idx])
        channel_y_positions[idx] = y_pos
        y_pos -= spacing

    ax.set_xlim(0, 10)
    y_top = spacing * 0.8
    y_bottom = y_pos + spacing * 0.4
    ax.set_ylim(y_bottom, y_top)

    ax.set_yticks(yticks)
    # Improvement 6: Increased EEG Channel Label Font Size from 6 to 9
    ax.set_yticklabels(yticklabels, fontsize=9, fontfamily='sans-serif')
    ax.set_xlabel('Time (s)', fontsize=8, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Improvement 5: Hemisphere shading (already present, ensuring it's visible with new scaling)
    if highlight_side in ('left', 'right'):
        side_indices = LEFT_CH_INDICES if highlight_side == 'left' else RIGHT_CH_INDICES
        side_y_vals = [channel_y_positions[idx] for idx in side_indices
                       if idx in channel_y_positions]
        if side_y_vals:
            y_hi = max(side_y_vals) + spacing * 0.5
            y_lo = min(side_y_vals) - spacing * 0.5
            ax.axhspan(y_lo, y_hi, color='lightblue', alpha=0.15, zorder=0) # Alpha 0.15 is transparent enough

    ax.set_title(title, fontsize=10, fontweight='bold', fontfamily='sans-serif')


def draw_flowchart(ax):
    """Draw the W05 architecture flowchart in Panel B."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def add_box(x, y, w, h, text, facecolor='#E8F4FD', edgecolor='#2C3E50',
                fontsize=8, text_color='black', linewidth=1.5, alpha=0.9,
                title_fontsize_boost=1.0, base_line_spacing_factor=0.03): # Improvement 2 & 7: New params for font/spacing control
        """Add a rounded rectangle with centered text."""
        box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=facecolor, edgecolor=edgecolor,
                             linewidth=linewidth, alpha=alpha, zorder=2)
        ax.add_patch(box)
        lines = text.split('\n')
        n_lines = len(lines)
        
        # Calculate line spacing based on the base font size and a more generous factor
        # Ensure it's not too large for the box height
        line_spacing = min(fontsize * base_line_spacing_factor, h / (n_lines + 0.5))
        
        start_y = y + (n_lines - 1) * line_spacing / 2
        for i, line in enumerate(lines):
            ly = start_y - i * line_spacing
            fs = fontsize
            fw = 'normal'
            if i == 0:
                fw = 'bold'
                fs = fontsize + title_fontsize_boost # Improvement 2: Boost title font size
            ax.text(x, ly, line, ha='center', va='center',
                    fontsize=fs, fontweight=fw, color=text_color, zorder=3)

    def add_arrow(x1, y1, x2, y2, color='#2C3E50'):
        """Add a straight arrow."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, connectionstyle='arc3,rad=0'))

    # -- Top box: W05 Iterative Narrowband Refinement --
    # Improvement 2: Increased font size and box height
    add_box(5, 9.3, 8.5, 1.2, # h increased from 0.9 to 1.2
            "W05: Iterative Narrowband Refinement",
            facecolor='#D6EAF8', edgecolor='#2471A3', fontsize=9, title_fontsize_boost=1.0)

    add_arrow(5.0, 8.8, 5.0, 8.1)

    # -- Pass 1: Coarse Analysis --
    # Improvement 2: Increased font size and box height
    add_box(5, 7.3, 8.5, 2.0, # h increased from 1.4 to 2.0
            "Pass 1: Coarse Analysis\nBandpass 0.5\u20133.5 Hz\nLateralization: mean variance per hemisphere\nFrequency: Hilbert inst. freq from top-3 dominant channels",
            facecolor='#D6EAF8', edgecolor='#2980B9', fontsize=8.5, title_fontsize_boost=1.0) # Fontsize increased from 7.5 to 8.5

    add_arrow(5.0, 6.55, 5.0, 5.85)

    # -- Pass 2: Narrowband Refinement --
    # Improvement 2: Increased font size and box height
    add_box(5, 5.1, 8.5, 2.0, # h increased from 1.4 to 2.0
            "Pass 2: Narrowband Refinement\nBandpass at est_freq \u00b1 0.4 Hz\nRefined Lateralization: envelope amplitude\nRefined Frequency: Hilbert on dominant hemisphere",
            facecolor='#AED6F1', edgecolor='#2471A3', fontsize=8.5, title_fontsize_boost=1.0) # Fontsize increased from 7.5 to 8.5

    # Arrows from pass 2 to three branches
    add_arrow(2.2, 4.35, 2.0, 3.5)
    add_arrow(5.0, 4.35, 5.0, 3.5)
    add_arrow(7.8, 4.35, 8.0, 3.5)

    # -- Branch 1 (left, green): Laterality Detection --
    # Improvement 2: Increased font size and box height
    add_box(2.0, 2.5, 3.0, 2.2, # h increased from 1.8 to 2.2
            "Laterality Detection\nL vs R narrowband amplitude\nOutput: Left/Right\nAUC = 0.837",
            facecolor='#D5F5E3', edgecolor='#1E8449', fontsize=8.5, title_fontsize_boost=1.0) # Fontsize increased from 7.5 to 8.5

    # -- Branch 2 (center, purple): Spatial Extent --
    # Improvement 2 & 7: Increased font size, box width/height, and refined text layout
    add_box(5.0, 2.5, 3.3, 2.5, # w increased from 3.0 to 3.3, h increased from 1.8 to 2.5
            "Spatial Extent\nPLV \u00d7 Amplitude\nPer-channel phase coherence\nwith dominant hemisphere\n\u00d7 narrowband amplitude\nThreshold \u2192 count/18", # Text layout refined
            facecolor='#E8DAEF', edgecolor='#7D3C98', fontsize=8.0, title_fontsize_boost=1.0) # Fontsize increased from 6.8 to 8.0

    # -- Branch 3 (right, orange): Topographic Localization --
    # Improvement 2: Increased font size and box height
    add_box(8.0, 2.5, 3.0, 2.2, # h increased from 1.8 to 2.2
            "Topographic Localization\nPer-channel Hilbert envelope\nLaplacian transform\n\u2192 Topoplot + Verbal Description",
            facecolor='#FDEBD0', edgecolor='#E67E22', fontsize=8.5, title_fontsize_boost=1.0) # Fontsize increased from 7.2 to 8.5

    # Arrows down to output boxes
    add_arrow(2.0, 1.55, 2.0, 1.0)
    add_arrow(5.0, 1.55, 5.0, 1.0)
    add_arrow(8.0, 1.55, 8.0, 1.0)

    # -- Output boxes (bottom) --
    # Improvement 2: Increased font size and box height
    add_box(2.0, 0.55, 2.8, 0.9, # h increased from 0.7 to 0.9
            "Laterality",
            facecolor='#D5F5E3', edgecolor='#1E8449', fontsize=10, title_fontsize_boost=0) # Fontsize increased from 9 to 10

    add_box(5.0, 0.55, 2.8, 0.9, # h increased from 0.7 to 0.9
            "Spatial Extent + Frequency",
            facecolor='#E8DAEF', edgecolor='#7D3C98', fontsize=9.5, title_fontsize_boost=0) # Fontsize increased from 8.5 to 9.5

    add_box(8.0, 0.55, 2.8, 0.9, # h increased from 0.7 to 0.9
            "Spatial Localization",
            facecolor='#FDEBD0', edgecolor='#E67E22', fontsize=10, title_fontsize_boost=0) # Fontsize increased from 9 to 10

    # Title
    ax.text(5, 10.3, 'B. Pipeline Architecture', ha='center', va='bottom',
            fontsize=12, fontweight='bold', fontfamily='sans-serif')


def main():
    print("=" * 60)
    print("Fig 3: RDA Characterization Pipeline")
    print("=" * 60)

    # -- Load and process EEG --
    print("Loading EEG...", flush=True)
    mono_raw = load_monopolar(MAT_FILE)

    # Average reference
    avg = np.mean(mono_raw, axis=0)
    mono_car = mono_raw - avg[np.newaxis, :]

    # Broadband bandpass filter for display
    mono_filt = bandpass_filter(mono_car, lo=0.5, hi=20.0)
    mono_filt = np.clip(mono_filt, -300, 300)

    # -- Get frequency from segment_labels.csv --
    print("Loading frequency from segment_labels.csv...", flush=True)
    import csv
    freq_hz = None
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mat_file'] == MAT_FILE:
                freq_hz = float(row['pdchar_freq_hz'])
                break
    if freq_hz is None or not np.isfinite(freq_hz):
        freq_hz = 1.1  # fallback
    print(f"  Frequency: {freq_hz:.2f} Hz", flush=True)

    # -- Compute narrowband amplitude + Laplacian for topoplot --
    print("Computing narrowband amplitude envelope...", flush=True)
    # Use average-referenced data for amplitude computation
    amplitude_vector, narrowband_car = compute_amplitude_envelope(mono_car, freq_hz)

    # Laplacian of amplitude vector for topoplot
    lap_amplitude = compute_laplacian_vector(amplitude_vector, LAP_NEIGHBORS)
    lap_amplitude = np.abs(lap_amplitude)  # take absolute value
    print(f"  Max amplitude: {np.max(amplitude_vector):.2f}", flush=True)
    print(f"  Max Laplacian amplitude: {np.max(lap_amplitude):.2f}", flush=True)

    # -- Compute laterality from narrowband amplitude --
    left_amp = np.mean(amplitude_vector[LEFT_CH_INDICES])
    right_amp = np.mean(amplitude_vector[RIGHT_CH_INDICES])
    laterality = 'left' if left_amp > right_amp else 'right'
    print(f"  Laterality: {laterality} (L={left_amp:.2f}, R={right_amp:.2f})", flush=True)

    # -- Generate verbal description --
    print("Generating verbal description...", flush=True)
    from generate_discharge_topo_viewer import generate_verbal_from_topo
    try:
        verbal = generate_verbal_from_topo('lrda', freq_hz, amplitude_vector,
                                            laterality_from_pdchar=laterality)
    except Exception as e:
        print(f"  Verbal description error: {e}")
        verbal = f"LRDA, {laterality} sided, {freq_hz:.1f} Hz"
    print(f"  Verbal: {verbal}", flush=True)

    # -- Create figure --
    print("Building figure...", flush=True)
    fig = plt.figure(figsize=(22, 9), facecolor='white')

    # Three panels: A (30%), B (40%), C (30%)
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.30, 0.40, 0.30],
                           left=0.03, right=0.97, top=0.92, bottom=0.05,
                           wspace=0.08)

    # -- Panel A: Input EEG --
    ax_a = fig.add_subplot(gs[0, 0])
    plot_eeg_traces(ax_a, mono_filt,
                    title='A. Input: 19-Channel EEG (10s, 200 Hz)')

    # -- Panel B: Architecture Flowchart --
    ax_b = fig.add_subplot(gs[0, 1])
    draw_flowchart(ax_b)

    # -- Panel C: Output Visualization --
    # Full-height EEG (same as Panel A), topoplot overlaid in lower-right
    ax_c = fig.add_subplot(gs[0, 2])
    _, narrowband_display = compute_amplitude_envelope(mono_filt, freq_hz, bw=0.4)
    plot_eeg_traces(ax_c, mono_filt,
                    title='C. Output: Characterized LRDA',
                    narrowband=narrowband_display,
                    highlight_side=laterality)

    # Topoplot as inset in lower-right corner
    c_pos = ax_c.get_position()
    topo_size = 0.07
    inset_left = c_pos.x1 - topo_size - 0.01
    inset_bottom = c_pos.y0 + 0.02
    ax_topo_inset = fig.add_axes([inset_left, inset_bottom, topo_size, topo_size * (22/9)])
    generate_topoplot_on_ax(ax_topo_inset, lap_amplitude, MONO_CHANNELS, title='')
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

    fig.text(inset_left + topo_size / 2, inset_bottom - 0.01, wrapped,
             ha='center', va='top', fontsize=7, fontstyle='italic',
             fontfamily='sans-serif', color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f8f8f0',
                       edgecolor='#ccc', alpha=0.9))

    # -- Save --
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {OUT_PATH}")
    print("Done!")


if __name__ == '__main__':
    main()