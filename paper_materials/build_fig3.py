#!/usr/bin/env python3
"""
Build Fig 3: RDA Characterization Pipeline — fully reproducible from data.

Three-panel layout matching Fig 2 style:
  Panel A: Input 19-channel EEG (no shading, no markers)
  Panel B: RDA pipeline architecture (from draw_panel_b_rda.py)
  Panel C: Output EEG (hemisphere shading, topoplot inset, verbal description)

EEG panels use render_figures.draw_eeg_panel (same as figs 5-8).

To change the example segment, edit MAT_FILE and SUBTYPE below.

Usage:
    conda run -n morgoth python paper_materials/build_fig3.py
"""

import sys
import csv
import warnings
import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfiltfilt, hilbert

warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
mne.set_log_level('ERROR')

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CODE_DIR = PROJECT_DIR / 'code'
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'

OUTPUT_PATH = SCRIPT_DIR / 'figures' / 'fig3_rda_pipeline.png'
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# ── Segment to use (change this to swap examples) ──
MAT_FILE = 'sub-S0001115633229_20190719143934.mat'
SUBTYPE = 'lrda'

# ── Constants ──
FS = 200
N_SAMPLES = 2000

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

LEFT_CH_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
RIGHT_CH_INDICES = [11, 12, 13, 14, 15, 16, 17, 18]

LAP_NEIGHBORS = {
    0: [1, 4, 8, 11],      1: [0, 2, 4, 8],       2: [1, 3, 5, 9],
    3: [2, 6, 7, 10],       4: [0, 1, 5],           5: [4, 2, 6],
    6: [5, 3, 7],            7: [3, 6, 10],          8: [0, 1, 9, 11, 12],
    9: [8, 2, 10, 13],      10: [9, 3, 7, 14, 18],  11: [12, 15, 8, 0],
    12: [11, 13, 15, 8],    13: [12, 14, 16, 9],    14: [13, 17, 18, 10],
    15: [11, 12, 16],       16: [15, 13, 17],        17: [16, 14, 18],
    18: [14, 17, 10],
}


# ── Data loading and processing ──

def load_monopolar(mat_file):
    path = EEG_DIR / mat_file
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :N_SAMPLES]
    assert seg.shape[0] == 19, f"Expected 19 channels, got {seg.shape[0]}"
    return seg


def compute_amplitude_envelope(mono, freq_hz, bw=0.4):
    """Compute narrowband amplitude envelope per channel."""
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


def compute_segment_data(mat_file, subtype):
    """Load EEG, compute RDA topography, laterality, and verbal description."""
    print(f"Loading EEG: {mat_file}")
    mono_raw = load_monopolar(mat_file)

    # Average reference
    avg = np.mean(mono_raw, axis=0)
    mono_car = mono_raw - avg[np.newaxis, :]

    # Get frequency from segment_labels.csv
    print("Loading frequency from segment_labels.csv...")
    freq_hz = None
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['mat_file'] == mat_file:
                freq_hz = float(row['pdchar_freq_hz'])
                break
    if freq_hz is None or not np.isfinite(freq_hz):
        freq_hz = 1.1
    print(f"  Frequency: {freq_hz:.2f} Hz")

    # Narrowband amplitude envelope + Laplacian for topoplot
    print("Computing narrowband amplitude envelope...")
    amplitude_vector, _ = compute_amplitude_envelope(mono_car, freq_hz)
    lap_amplitude = compute_laplacian_vector(amplitude_vector, LAP_NEIGHBORS)
    lap_amplitude = np.abs(lap_amplitude)

    # Laterality from narrowband amplitude
    left_amp = np.mean(amplitude_vector[LEFT_CH_INDICES])
    right_amp = np.mean(amplitude_vector[RIGHT_CH_INDICES])
    laterality = 'left' if left_amp > right_amp else 'right'
    print(f"  Laterality: {laterality} (L={left_amp:.2f}, R={right_amp:.2f})")

    # Verbal description
    print("Generating verbal description...")
    from generate_discharge_topo_viewer import generate_verbal_from_topo
    try:
        verbal = generate_verbal_from_topo(
            subtype, freq_hz, amplitude_vector,
            laterality_from_pdchar=laterality,
        )
    except Exception as e:
        print(f"  Verbal description error: {e}")
        verbal = f"{subtype.upper()}, {laterality} sided, {freq_hz:.1f} Hz"
    print(f"  Verbal: {verbal}")

    return {
        'amplitude_vector': amplitude_vector,
        'lap_amplitude': lap_amplitude,
        'laterality': laterality,
        'frequency': freq_hz,
        'verbal': verbal,
    }


# ── Panel rendering (reused from build_fig2.py pattern) ──

def _render_eeg_panel(case, is_pd, figsize=(6, 8), show_scale_bar=True):
    """Render EEG panel using render_figures.draw_eeg_panel (same as figs 5-8)."""
    from render_figures import draw_eeg_panel
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    draw_eeg_panel(ax, case, is_pd=is_pd)

    # Remove difficulty badge and original scale bar
    for txt in list(ax.texts):
        text_content = txt.get_text()
        if 'Agreement' in text_content or any(d in text_content.upper() for d in ['EASY', 'MEDIUM', 'HARD']):
            txt.remove()
        elif 'µV' in text_content:
            txt.remove()

    for patch in list(ax.patches):
        if isinstance(patch, FancyArrowPatch):
            patch.remove()

    # Redraw scale bar inside the plot (only if requested)
    if show_scale_bar:
        z_scale = 0.012
        scale_uv = 100.0
        scale_height = scale_uv * z_scale
        scale_x = 8.8
        y_lim = ax.get_ylim()
        scale_y_bot = y_lim[0] + 1.0
        scale_y_top = scale_y_bot + scale_height
        scale_y_mid = (scale_y_bot + scale_y_top) / 2

        arrow_up = FancyArrowPatch((scale_x, scale_y_mid + 0.02), (scale_x, scale_y_top),
                                    arrowstyle='-|>', mutation_scale=10, color='black', lw=1.0, zorder=5)
        arrow_dn = FancyArrowPatch((scale_x, scale_y_mid - 0.02), (scale_x, scale_y_bot),
                                    arrowstyle='-|>', mutation_scale=10, color='black', lw=1.0, zorder=5)
        ax.add_patch(arrow_up)
        ax.add_patch(arrow_dn)
        ax.text(scale_x + 0.15, scale_y_mid, f'{int(scale_uv)} µV',
                fontsize=9, va='center', ha='left', color='black', zorder=5)

    ax.set_xlim(-0.05, 10.05)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.06)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=300, facecolor='white')
    plt.close(fig)
    img = Image.open(tmp.name)
    Path(tmp.name).unlink()
    return img


def _build_case_dict(mono_raw, laterality, subtype):
    """Build case dict for render_figures.draw_eeg_panel (RDA — no discharge times)."""
    from scipy.signal import resample
    mono_ds = resample(mono_raw, 1000, axis=1)
    return {
        'mono_data': mono_ds.tolist(),
        'subtype': subtype,
        'pred_lat': laterality,
        'gt_discharge_times': [],  # RDA has no discharge times
        'difficulty': '',
        'agreement_pct': 0,
    }


def render_panel_b():
    """Render RDA Panel B, return as PIL Image."""
    from draw_panel_b_rda import draw_panel_b_rda
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    draw_panel_b_rda(outpath=tmp.name)
    img = Image.open(tmp.name)
    Path(tmp.name).unlink()
    return img


# ── Verbal description and topoplot overlays (shared with build_fig2.py) ──

def _split_verbal_at_comma(text):
    """Split verbal description into two lines at a natural comma break."""
    commas = [i for i, c in enumerate(text) if c == ',']
    if not commas:
        mid = len(text) // 2
        spaces = [i for i, c in enumerate(text) if c == ' ']
        if spaces:
            best = min(spaces, key=lambda x: abs(x - mid))
            return text[:best] + '\n' + text[best+1:]
        return text
    mid = len(text) // 2
    best_comma = min(commas, key=lambda x: abs(x - mid))
    return text[:best_comma+1] + '\n' + text[best_comma+1:].lstrip()


def add_verbal_description(img, verbal_text, pc_left, pc_right):
    """Draw verbal description in a light blue box overlaid on bottom of Panel C."""
    draw = ImageDraw.Draw(img)

    font_size = 58
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except Exception:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', font_size)
        except Exception:
            font = ImageFont.load_default()

    wrapped = _split_verbal_at_comma(verbal_text)
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 20, 12
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y

    box_x = pc_left + 350
    max_right = pc_right - 350
    if box_x + box_w > max_right:
        box_x = max_right - box_w
    if box_x < pc_left + 10:
        box_x = pc_left + 10

    box_y = 2290

    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=10, fill='#D6EAF8', outline='#85C1E9', width=2,
    )
    draw.text((box_x + pad_x, box_y + pad_y), wrapped, fill='#1A2B3C', font=font)

    return img


def add_topoplot_inset(img, topo_data, pc_right):
    """Add a small topoplot inset in the lower-right corner of Panel C."""
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in MONO_CHANNELS]
    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(topo_data))
    if vmax < 1e-10:
        vmax = 1.0

    fig_t, ax_t = plt.subplots(figsize=(2, 2))
    mne.viz.plot_topomap(
        topo_data, info, axes=ax_t, show=False,
        contours=6, cmap='inferno', sensors=False,
        vlim=(0, vmax),
    )
    fig_t.patch.set_alpha(0)
    ax_t.set_facecolor('none')

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    fig_t.savefig(tmp.name, dpi=150, transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_t)

    topo_img = Image.open(tmp.name).convert('RGBA')
    Path(tmp.name).unlink()

    inset_size = 280
    topo_img = topo_img.resize((inset_size, inset_size), Image.LANCZOS)

    inset_x = pc_right - inset_size - 120
    inset_y = 2150

    img.paste(topo_img, (inset_x, inset_y), topo_img)
    return img


# ── Full figure composite ──

def build_full_figure(panel_a, panel_b, panel_c, verbal_text, topo_data):
    """Composite all three panels into the final figure with titles."""
    W, H = 5782, 2618
    gap = 5

    usable_w = W - 2 * gap
    b_w = int(usable_w * 0.40)
    a_w = (usable_w - b_w) // 2
    c_w = a_w  # force identical

    title_h = 150
    panel_h = H - title_h

    pa = panel_a.resize((a_w, panel_h), Image.LANCZOS)
    pb = panel_b.resize((b_w, panel_h), Image.LANCZOS)
    pc = panel_c.resize((c_w, panel_h), Image.LANCZOS)

    comp = Image.new('RGB', (W, H), 'white')
    x = 0
    comp.paste(pa, (x, title_h))
    a_end = x + a_w

    x = a_end + gap
    b_y_offset = title_h - 40
    comp.paste(pb, (x, b_y_offset))
    b_end = x + b_w

    x = b_end + gap
    comp.paste(pc, (x, title_h))

    # Draw titles — all in same Pillow font for consistency
    draw = ImageDraw.Draw(comp)
    try:
        title_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 110)
        try:
            title_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 110, index=1)
        except Exception:
            pass
    except Exception:
        title_font = ImageFont.load_default()

    title_y = 45
    draw.text((30, title_y), 'A. Input', fill='black', font=title_font)

    b_title = 'B. RDA Pipeline Architecture'
    b_bbox = draw.textbbox((0, 0), b_title, font=title_font)
    b_title_w = b_bbox[2] - b_bbox[0]
    b_center = a_end + gap + b_w // 2
    draw.text((b_center - b_title_w // 2, title_y), b_title, fill='black', font=title_font)

    draw.text((b_end + gap + 30, title_y), 'C. Output', fill='black', font=title_font)

    # Panel C overlay positions
    pc_left = b_end + gap
    pc_right = pc_left + c_w

    # Add topoplot inset
    if topo_data is not None:
        comp = comp.convert('RGBA')
        comp = add_topoplot_inset(comp, topo_data, pc_right=pc_right)
        comp = comp.convert('RGB')

    # Add verbal description
    if verbal_text:
        comp = add_verbal_description(comp, verbal_text, pc_left=pc_left, pc_right=pc_right)

    return comp


def main():
    print("=" * 50)
    print("Building Fig 3: RDA Pipeline")
    print("=" * 50)

    # Step 1: Compute all data from EEG segment
    data = compute_segment_data(MAT_FILE, subtype=SUBTYPE)

    # Step 2: Build case dict for EEG panels
    mono_raw = load_monopolar(MAT_FILE)
    case = _build_case_dict(mono_raw, data['laterality'], SUBTYPE)

    # Step 3: Render Panel A (input EEG — no shading, no markers)
    print("Rendering Panel A...")
    case_a = dict(case)
    case_a['no_shading'] = True
    panel_a = _render_eeg_panel(case_a, is_pd=False)
    print(f"  Panel A size: {panel_a.size}")

    # Step 4: Render Panel B (RDA architecture)
    print("Rendering Panel B...")
    panel_b = render_panel_b()
    print(f"  Panel B size: {panel_b.size}")

    # Step 5: Render Panel C (output EEG — with hemisphere shading)
    print("Rendering Panel C...")
    panel_c = _render_eeg_panel(case, is_pd=False, show_scale_bar=False)
    print(f"  Panel C size: {panel_c.size}")

    # Step 6: Composite everything
    print("Compositing...")
    result = build_full_figure(
        panel_a, panel_b, panel_c,
        verbal_text=data['verbal'],
        topo_data=data['lap_amplitude'],
    )
    result.save(str(OUTPUT_PATH), dpi=(300, 300))
    print(f"Saved: {OUTPUT_PATH} ({result.size})")


if __name__ == '__main__':
    main()
