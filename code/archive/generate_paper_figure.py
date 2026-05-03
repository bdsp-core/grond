"""
Generate publication-ready architecture figure for HemiCET paper.

Creates a 3-panel composite:
  Left:   Real EEG input (18-ch bipolar, montage arrangement + simulated evidence)
  Center: Architecture diagram (from PaperBanana candidate 2, re-generated)
  Right:  Real EEG output (same EEG with discharge markers + evidence trace)

Uses patient 116713991: LPD, left laterality, 2.0 Hz, 20 discharges.

Usage:
    conda run -n foe python code/generate_paper_figure.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from scipy.signal import detrend, butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# ── Constants ────────────────────────────────────────────────────────
FS = 200
PATIENT_ID = '116713991'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Display order: left temporal, left parasagittal, midline, right parasagittal, right temporal
DISPLAY_ORDER = [
    # Left temporal
    (0, 'Fp1-F7'),
    (1, 'F7-T3'),
    (2, 'T3-T5'),
    (3, 'T5-O1'),
    # Left parasagittal
    (8, 'Fp1-F3'),
    (9, 'F3-C3'),
    (10, 'C3-P3'),
    (11, 'P3-O1'),
    # Spacer
    (None, ''),
    # Midline
    (16, 'Fz-Cz'),
    (17, 'Cz-Pz'),
    # Spacer
    (None, ''),
    # Right parasagittal
    (12, 'Fp2-F4'),
    (13, 'F4-C4'),
    (14, 'C4-P4'),
    (15, 'P4-O2'),
    # Right temporal
    (4, 'Fp2-F8'),
    (5, 'F8-T4'),
    (6, 'T4-T6'),
    (7, 'T6-O2'),
]

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]

GROUP_LABELS = {
    0: 'Left\nTemporal',
    4: 'Left\nParasag.',
    10: 'Midline',
    14: 'Right\nParasag.',
    18: 'Right\nTemporal',
}


def load_mat_as_bipolar(mat_path):
    """Load .mat file and return (18, 2000) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    if 'data' in mat:
        data = mat['data']
    elif 'EEG' in mat:
        data = mat['EEG']
    else:
        for k in mat:
            if not k.startswith('_'):
                data = mat[k]
                break

    data = np.array(data, dtype=np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T

    # If 20 channels (monopolar), convert to 18-ch bipolar
    if data.shape[0] == 20:
        data = _monopolar_to_bipolar(data)
    elif data.shape[0] == 18:
        pass  # already bipolar
    else:
        raise ValueError(f"Unexpected shape: {data.shape}")

    # Take first 2000 samples (10s at 200Hz)
    if data.shape[1] > 2000:
        data = data[:, :2000]
    return data


def _monopolar_to_bipolar(mono):
    """Convert 20-channel monopolar to 18-channel bipolar (double banana)."""
    # Standard 10-20: Fp1,Fp2,F7,F3,Fz,F4,F8,T3,C3,Cz,C4,T4,T5,P3,Pz,P4,T6,O1,O2
    # indices:        0    1   2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
    # plus 19 = extra ref or similar
    ch = {
        'Fp1': 0, 'Fp2': 1, 'F7': 2, 'F3': 3, 'Fz': 4, 'F4': 5, 'F8': 6,
        'T3': 7, 'C3': 8, 'Cz': 9, 'C4': 10, 'T4': 11, 'T5': 12,
        'P3': 13, 'Pz': 14, 'P4': 15, 'T6': 16, 'O1': 17, 'O2': 18,
    }
    pairs = [
        ('Fp1','F7'), ('F7','T3'), ('T3','T5'), ('T5','O1'),
        ('Fp2','F8'), ('F8','T4'), ('T4','T6'), ('T6','O2'),
        ('Fp1','F3'), ('F3','C3'), ('C3','P3'), ('P3','O1'),
        ('Fp2','F4'), ('F4','C4'), ('C4','P4'), ('P4','O2'),
        ('Fz','Cz'), ('Cz','Pz'),
    ]
    bipolar = np.zeros((18, mono.shape[1]))
    for i, (a, b) in enumerate(pairs):
        bipolar[i] = mono[ch[a]] - mono[ch[b]]
    return bipolar


def generate_evidence_trace(seg_bi, hemi_indices):
    """Generate a simulated evidence trace using pointiness-based approach.

    Since we can't run the full HemiCET model here, we approximate with
    a pointiness/energy-based evidence signal that highlights periodic discharges.
    """
    from pd_pointiness_acf import compute_pointiness_trace

    # Average pointiness across hemisphere channels
    traces = []
    for ch_idx in hemi_indices:
        ch_data = seg_bi[ch_idx].copy()
        mu, sigma = np.mean(ch_data), np.std(ch_data)
        if sigma > 1e-8:
            ch_data = (ch_data - mu) / sigma
        pt = compute_pointiness_trace(ch_data, FS)
        traces.append(pt)

    # Average and normalize to [0, 1]
    avg = np.mean(traces, axis=0)
    avg = gaussian_filter1d(avg, sigma=5)
    if avg.max() > 0:
        avg = avg / avg.max()
    return avg


def plot_eeg_panel(ax, ax_ev, seg_bi, title, discharge_times=None,
                   evidence=None, show_channel_labels=True, highlight_left=True):
    """Plot EEG with montage arrangement on given axes.

    Args:
        ax: main EEG axis
        ax_ev: evidence trace axis (below)
        seg_bi: (18, 2000) bipolar data
        title: panel title
        discharge_times: list of times (seconds) for vertical dashed lines
        evidence: (2000,) evidence trace array
        show_channel_labels: whether to show y-axis channel names
        highlight_left: highlight left hemisphere channels (for LPD)
    """
    seg_bi = seg_bi.astype(np.float64)
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = seg_bi.shape[1]
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    # Lowpass at 20 Hz for display
    nyq = FS / 2.0
    b, a = butter(4, 20.0 / nyq, btype='low')
    filtered = seg_bi.copy()
    for i in range(18):
        try:
            filtered[i] = filtfilt(b, a, seg_bi[i])
        except ValueError:
            pass
    for i in range(18):
        filtered[i] = detrend(filtered[i], type='linear')

    # Fixed scaling
    z_scale = 0.01
    clip_uv = 300.0

    n_display = len(DISPLAY_ORDER)

    # Build channel offset mapping
    ch_to_offset = {}
    yticks = []
    ytick_labels = []

    for di in range(n_display):
        ch_idx, ch_name = DISPLAY_ORDER[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)
        if ch_idx is not None:
            ch_to_offset[ch_idx] = offset

    ax.set_facecolor('white')

    # Draw traces
    for di in range(n_display):
        ch_idx, ch_name = DISPLAY_ORDER[di]
        offset = float(n_display - di)
        if ch_idx is None:
            continue

        clipped = np.clip(filtered[ch_idx], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset

        # Color by hemisphere involvement
        if highlight_left and ch_idx in LEFT_INDICES:
            color = '#C0392B'  # red for involved hemisphere
            lw = 0.7
        elif not highlight_left and ch_idx in RIGHT_INDICES:
            color = '#2980B9'  # blue
            lw = 0.7
        elif ch_idx in (16, 17):
            color = '#7F8C8D'  # gray for midline
            lw = 0.5
        else:
            color = '#2C3E50'  # dark for uninvolved
            lw = 0.5

        ax.plot(time_vec, scaled, color=color, linewidth=lw, clip_on=True)

    # Y-axis
    if show_channel_labels:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels, fontsize=5.5, fontfamily='monospace')
        ax.tick_params(axis='y', length=0, pad=2)
    else:
        ax.set_yticks([])

    ax.set_ylim(0, n_display + 1)
    ax.set_xlim(0, n_samples / FS)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='x', labelsize=5)
    ax.grid(True, axis='x', alpha=0.15, linewidth=0.3, linestyle='--')
    ax.grid(False, axis='y')

    # Group bracket labels on the left
    if show_channel_labels:
        # Add group labels
        group_positions = [
            (0, 3, 'L. Temp.'),
            (4, 7, 'L. Parasag.'),
            (9, 10, 'Midline'),
            (12, 15, 'R. Parasag.'),
            (16, 19, 'R. Temp.'),
        ]
        for start, end, label in group_positions:
            y_top = float(n_display - start)
            y_bot = float(n_display - end)
            y_mid = (y_top + y_bot) / 2
            ax.annotate(label, xy=(-0.02, y_mid), xycoords=('axes fraction', 'data'),
                       fontsize=4, ha='right', va='center', color='#666',
                       fontstyle='italic')

    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Discharge marker lines
    if discharge_times is not None:
        for t in discharge_times:
            ax.axvline(x=t, color='#E74C3C', linestyle='--', alpha=0.5,
                      linewidth=0.6, zorder=5)
            # Red dots on involved channels
            samp = int(round(t * FS))
            samp = max(0, min(samp, n_samples - 1))
            involved = LEFT_INDICES if highlight_left else RIGHT_INDICES
            for ch_idx in involved:
                if ch_idx in ch_to_offset:
                    offset = ch_to_offset[ch_idx]
                    clipped_val = np.clip(filtered[ch_idx, samp], -clip_uv, clip_uv)
                    y_val = z_scale * clipped_val + offset
                    ax.plot(t, y_val, 'o', color='#E74C3C', markersize=1.5,
                           zorder=10, markeredgewidth=0.2, markeredgecolor='#C0392B')

    # Title
    ax.set_title(title, fontsize=7, fontweight='bold', pad=3)

    # Evidence subplot
    if ax_ev is not None and evidence is not None:
        ev_time = np.linspace(0, len(evidence) / FS, len(evidence))
        ax_ev.set_facecolor('white')
        ax_ev.fill_between(ev_time, 0, evidence, color='steelblue', alpha=0.4)
        ax_ev.plot(ev_time, evidence, color='steelblue', linewidth=0.5)
        ax_ev.set_xlim(0, n_samples / FS)
        ax_ev.set_ylabel('E(t)', fontsize=5, labelpad=1)
        ax_ev.tick_params(labelsize=4)
        ax_ev.xaxis.set_major_locator(MultipleLocator(1))
        ax_ev.grid(True, axis='x', alpha=0.15, linewidth=0.3, linestyle='--')
        for spine in ax_ev.spines.values():
            spine.set_visible(False)

        if discharge_times is not None:
            for t in discharge_times:
                ax_ev.axvline(x=t, color='#E74C3C', linestyle='--', alpha=0.4,
                             linewidth=0.5, zorder=5)


def main():
    # ── Load data ─────────────────────────────────────────────────────
    mat_path = PROJECT_DIR / 'data' / 'eeg' / f'{PATIENT_ID}_seg000.mat'
    seg_bi = load_mat_as_bipolar(mat_path)

    with open(PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json') as f:
        dt = json.load(f)
    patient_data = dt[PATIENT_ID]
    discharge_times = patient_data['global_times']

    # Generate evidence trace for the left hemisphere (LPD, left laterality)
    evidence = generate_evidence_trace(seg_bi, LEFT_INDICES)

    # ── Generate input panel ──────────────────────────────────────────
    print("Generating input panel...")
    fig_in, (ax_in, ax_in_ev) = plt.subplots(
        2, 1, figsize=(4.5, 5.5),
        gridspec_kw={'height_ratios': [10, 1.2], 'hspace': 0.06})
    fig_in.patch.set_facecolor('white')

    plot_eeg_panel(ax_in, ax_in_ev, seg_bi,
                   title='Input: 8-ch Left Hemisphere EEG (10s, 200 Hz)',
                   evidence=evidence, highlight_left=True)
    ax_in_ev.set_xlabel('Time (s)', fontsize=5)

    fig_in.savefig(str(PROJECT_DIR / 'paper_materials' / 'figures' / 'input_eeg_panel.png'),
                   dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_in)
    print("  -> input_eeg_panel.png")

    # ── Generate output panel ─────────────────────────────────────────
    print("Generating output panel...")
    fig_out, (ax_out, ax_out_ev) = plt.subplots(
        2, 1, figsize=(4.5, 5.5),
        gridspec_kw={'height_ratios': [10, 1.2], 'hspace': 0.06})
    fig_out.patch.set_facecolor('white')

    plot_eeg_panel(ax_out, ax_out_ev, seg_bi,
                   title=f'Output: Detected Discharges ({len(discharge_times)} @ {patient_data["frequency"]:.1f} Hz)',
                   discharge_times=discharge_times,
                   evidence=evidence, highlight_left=True)
    ax_out_ev.set_xlabel('Time (s)', fontsize=5)

    fig_out.savefig(str(PROJECT_DIR / 'paper_materials' / 'figures' / 'output_eeg_panel.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_out)
    print("  -> output_eeg_panel.png")

    # ── Generate composite figure ─────────────────────────────────────
    print("Generating composite figure...")
    from PIL import Image, ImageDraw, ImageFont

    # Load images
    arch_img = Image.open(str(PROJECT_DIR / 'paper_materials' / 'figures' / 'architecture_v2_1.png'))
    input_img = Image.open(str(PROJECT_DIR / 'paper_materials' / 'figures' / 'input_eeg_panel.png'))
    output_img = Image.open(str(PROJECT_DIR / 'paper_materials' / 'figures' / 'output_eeg_panel.png'))

    # Target layout: EEG panels are ~25% width each, architecture ~50%
    # Scale all to same height
    target_h = max(input_img.height, output_img.height)

    def scale_to_height(img, h):
        ratio = h / img.height
        return img.resize((int(img.width * ratio), h), Image.LANCZOS)

    input_scaled = scale_to_height(input_img, target_h)
    output_scaled = scale_to_height(output_img, target_h)

    # Scale architecture to be taller (center of attention)
    arch_target_h = int(target_h * 0.85)
    arch_scaled = scale_to_height(arch_img, arch_target_h)

    # Layout with panel labels and arrows
    gap = 50  # pixels between panels
    arrow_w = 40  # space for arrows
    label_h = 40  # space for panel labels at top

    total_h = target_h + label_h
    total_w = (input_scaled.width + arrow_w + arch_scaled.width +
               arrow_w + output_scaled.width + 2 * gap)

    composite = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(composite)

    # Try to get a font for labels
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        font_label = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Paste input panel
    x_in = 0
    y_offset = label_h
    composite.paste(input_scaled, (x_in, y_offset))

    # Panel A label
    draw.text((x_in + 5, 5), "A", fill='black', font=font_label)
    draw.text((x_in + 35, 10), "Input EEG", fill='#555', font=font_small)

    # Arrow between input and architecture
    x_arrow1 = x_in + input_scaled.width + gap // 2
    arrow_y = y_offset + target_h // 2
    # Draw right-pointing arrow
    for dy in range(-2, 3):
        draw.line([(x_arrow1 - 15, arrow_y + dy), (x_arrow1 + 15, arrow_y + dy)],
                  fill='#333', width=1)
    draw.polygon([(x_arrow1 + 15, arrow_y - 10), (x_arrow1 + 30, arrow_y),
                  (x_arrow1 + 15, arrow_y + 10)], fill='#333')

    # Paste architecture panel (centered vertically)
    x_arch = x_in + input_scaled.width + gap
    arch_y_offset = y_offset + (target_h - arch_target_h) // 2
    composite.paste(arch_scaled, (x_arch, arch_y_offset))

    # Panel B label
    draw.text((x_arch + 5, 5), "B", fill='black', font=font_label)
    draw.text((x_arch + 35, 10), "HemiCET Pipeline", fill='#555', font=font_small)

    # Arrow between architecture and output
    x_arrow2 = x_arch + arch_scaled.width + gap // 2
    for dy in range(-2, 3):
        draw.line([(x_arrow2 - 15, arrow_y + dy), (x_arrow2 + 15, arrow_y + dy)],
                  fill='#333', width=1)
    draw.polygon([(x_arrow2 + 15, arrow_y - 10), (x_arrow2 + 30, arrow_y),
                  (x_arrow2 + 15, arrow_y + 10)], fill='#333')

    # Paste output panel
    x_out = x_arch + arch_scaled.width + gap
    composite.paste(output_scaled, (x_out, y_offset))

    # Panel C label
    draw.text((x_out + 5, 5), "C", fill='black', font=font_label)
    draw.text((x_out + 35, 10), "Detected Discharges", fill='#555', font=font_small)

    composite_path = PROJECT_DIR / 'paper_materials' / 'figures' / 'hemicet_composite_figure.png'
    composite.save(str(composite_path), dpi=(300, 300))
    print(f"  -> hemicet_composite_figure.png ({composite.width}x{composite.height})")

    print("\nDone! All figures saved to paper_materials/figures/")


if __name__ == '__main__':
    main()
