#!/usr/bin/env python3
"""
Build Fig 2 composite: swap Panel B (with real topoplot) into the figure.

Renders draw_panel_b.py with a real Laplacian topoplot computed from the
EEG segment, then pastes the updated Panel B into the existing composite
figure (preserving Panels A and C untouched).

To change the example segment, edit MAT_FILE below and re-run.

Usage:
    conda run -n morgoth python paper_materials/build_fig2.py
"""

import sys
import json
import warnings
import numpy as np
import scipy.io as sio

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

COMPOSITE_PATH = SCRIPT_DIR / 'figures' / 'fig2_pd_pipeline.png'

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

# ── Segment to use (change this to swap examples) ──
MAT_FILE = 'sub-S0001114959966_20150425125519.mat'

# ── Constants ──
FS = 200
N_SAMPLES = 2000

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# Laplacian neighbor map
LAP_NEIGHBORS = {
    0: [1, 4, 8, 11],      1: [0, 2, 4, 8],       2: [1, 3, 5, 9],
    3: [2, 6, 7, 10],       4: [0, 1, 5],           5: [4, 2, 6],
    6: [5, 3, 7],            7: [3, 6, 10],          8: [0, 1, 9, 11, 12],
    9: [8, 2, 10, 13],      10: [9, 3, 7, 14, 18],  11: [12, 15, 8, 0],
    12: [11, 13, 15, 8],    13: [12, 14, 16, 9],    14: [13, 17, 18, 10],
    15: [11, 12, 16],       16: [15, 13, 17],        17: [16, 14, 18],
    18: [14, 17, 10],
}



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


def bandpass_filter(data, lo=0.5, hi=20.0, fs=200, order=4):
    import scipy.signal as signal
    sos = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass', output='sos')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        try:
            filtered[ch] = signal.sosfiltfilt(sos, data[ch])
        except Exception:
            filtered[ch] = data[ch]
    return filtered


def compute_laplacian(mono, neighbors_map):
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
    """Two-pass discharge-locked topography with Laplacian-GFP alignment."""
    window_samples = int(window_ms * fs / 1000)
    epoch_half = int(50 * fs / 1000)
    n_ch, n_total = mono_filtered.shape
    lap = compute_laplacian(mono_filtered, LAP_NEIGHBORS)

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

    mono_epochs, lap_epochs = [], []
    for s in gfp_aligned_samples:
        elo, ehi = s - epoch_half, s + epoch_half + 1
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

        refined_voltages = []
        for mono_epoch, lap_epoch in zip(mono_epochs, lap_epochs):
            epoch_gfp = np.std(lap_epoch, axis=0)
            best_shift, best_corr = 0, -np.inf
            for shift in range(-window_samples, window_samples + 1):
                t_lo, t_hi = max(0, -shift), min(epoch_len, epoch_len - shift)
                e_lo, e_hi = max(0, shift), min(epoch_len, epoch_len + shift)
                if t_hi - t_lo < 5:
                    continue
                corr = np.dot(template_gfp[t_lo:t_hi], epoch_gfp[e_lo:e_hi])
                if corr > best_corr:
                    best_corr, best_shift = corr, shift
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

    return np.abs(mean_topo_mono), np.abs(mean_topo_lap)


def make_topoplot_callback(mean_topo_lap):
    """Return a callback that draws a Laplacian topoplot (no channel labels)."""
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in MONO_CHANNELS]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(mean_topo_lap))
    if vmax < 1e-10:
        vmax = 1.0

    def topoplot_fn(ax):
        mne.viz.plot_topomap(
            mean_topo_lap, info, axes=ax, show=False,
            contours=6, cmap='inferno', sensors=False,
            vlim=(0, vmax),
        )

    return topoplot_fn


def compute_segment_data(mat_file, subtype='lpd'):
    """Load EEG, get discharge times, compute topography and verbal description.

    Returns dict with keys: mean_topo_mono, mean_topo_lap, discharge_times,
    laterality, frequency, verbal.
    """
    segment_id = mat_file.replace('.mat', '')

    print(f"Loading EEG: {mat_file}")
    mono_raw = load_monopolar(mat_file)

    print("Loading discharge times...")
    with open(LABELS_DIR / 'discharge_times.json') as f:
        dt_data = json.load(f)
    dt_entry = dt_data.get(segment_id)
    if isinstance(dt_entry, dict):
        discharge_times = dt_entry.get('global_times', [])
    elif isinstance(dt_entry, list):
        discharge_times = dt_entry
    else:
        discharge_times = []
    print(f"  {len(discharge_times)} discharge times")

    # Laterality via PDCharacterizer
    print("Computing laterality...")
    from pd_characterizer import PDCharacterizer
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
    charzer = PDCharacterizer()
    laterality = charzer.characterize(bipolar_raw, subtype=subtype).get('laterality', 'unknown')
    print(f"  Laterality: {laterality}")

    # Filter EEG for display (CAR + bandpass)
    avg = np.mean(mono_raw, axis=0)
    mono_car = mono_raw - avg[np.newaxis, :]
    mono_filt_display = bandpass_filter(mono_car, lo=0.5, hi=20.0)
    mono_filt_display = np.clip(mono_filt_display, -300, 300)

    # Topography (filter raw, not CAR)
    print("Computing topography...")
    mono_filt = bandpass_filter(mono_raw, lo=0.5, hi=20.0)
    mean_topo_mono, mean_topo_lap = gfp_align(mono_filt, discharge_times)
    if mean_topo_lap is None:
        print("  WARNING: gfp_align returned None, using fallback")
        mean_topo_lap = np.ones(19)
        mean_topo_mono = np.ones(19)

    # Frequency from IPI
    ipis = np.diff(discharge_times)
    frequency = 1.0 / np.median(ipis) if len(ipis) > 0 else np.nan

    # Verbal description (same code used for figs 5-8)
    print("Generating verbal description...")
    from generate_discharge_topo_viewer import generate_verbal_from_topo
    try:
        verbal = generate_verbal_from_topo(
            subtype, frequency, mean_topo_mono,
            laterality_from_pdchar=laterality,
        )
    except Exception as e:
        print(f"  Verbal description error: {e}")
        verbal = f"{subtype.upper()}, {laterality} sided, {frequency:.1f} Hz"
    print(f"  Verbal: {verbal}")

    return {
        'mean_topo_mono': mean_topo_mono,
        'mean_topo_lap': mean_topo_lap,
        'mono_filt_display': mono_filt_display,
        'discharge_times': discharge_times,
        'laterality': laterality,
        'frequency': frequency,
        'verbal': verbal,
    }


def render_panel_b(topoplot_fn):
    """Render Panel B with the topoplot callback, return as PIL Image."""
    from draw_panel_b import draw_panel_b
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    draw_panel_b(outpath=tmp.name, topoplot_fn=topoplot_fn)
    img = Image.open(tmp.name)
    Path(tmp.name).unlink()
    return img


def _render_eeg_panel(case, is_pd, figsize=(6, 8), show_scale_bar=True):
    """Render a single EEG panel using render_figures.draw_eeg_panel.

    Same code used for figs 5-8, ensuring visual consistency.
    Post-processes to: move scale bar inside plot, remove difficulty badge.
    """
    from render_figures import draw_eeg_panel

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    draw_eeg_panel(ax, case, is_pd=is_pd)

    # Remove difficulty badge and original scale bar
    from matplotlib.patches import FancyArrowPatch
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

    # Force identical layout so both panels produce the same pixel size
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


def _build_case_dict(mono_raw, discharge_times, laterality, subtype):
    """Build a case dict compatible with render_figures.draw_eeg_panel.

    Downsamples from 200 Hz to 100 Hz (1000 samples) to match the format
    used in figs 5-8.
    """
    from scipy.signal import resample

    # Downsample from 200 Hz (2000 samples) to 100 Hz (1000 samples)
    mono_ds = resample(mono_raw, 1000, axis=1)

    return {
        'mono_data': mono_ds.tolist(),
        'subtype': subtype,
        'pred_lat': laterality,
        'gt_discharge_times': discharge_times,
        'difficulty': '',
        'agreement_pct': 0,
    }


def _split_verbal_at_comma(text):
    """Split verbal description into two lines at a natural comma break."""
    # Find comma positions and pick the one closest to the middle
    commas = [i for i, c in enumerate(text) if c == ',']
    if not commas:
        mid = len(text) // 2
        # Fall back to splitting at a space near the middle
        spaces = [i for i, c in enumerate(text) if c == ' ']
        if spaces:
            best = min(spaces, key=lambda x: abs(x - mid))
            return text[:best] + '\n' + text[best+1:]
        return text
    mid = len(text) // 2
    best_comma = min(commas, key=lambda x: abs(x - mid))
    return text[:best_comma+1] + '\n' + text[best_comma+1:].lstrip()


def add_verbal_description(img, verbal_text, pc_left=None, pc_right=None):
    """Draw verbal description in a light blue box overlaid on bottom of Panel C."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    panel_c_left = pc_left if pc_left is not None else PANEL_B_RIGHT
    panel_c_right = pc_right if pc_right is not None else w

    font_size = 58
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', font_size)
    except Exception:
        try:
            font = ImageFont.truetype('/System/Library/Fonts/Arial.ttf', font_size)
        except Exception:
            font = ImageFont.load_default()

    # Split at a natural comma break
    wrapped = _split_verbal_at_comma(verbal_text)

    # Measure wrapped text
    bbox = draw.textbbox((0, 0), wrapped, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position: overlaid on the EEG near the bottom of Panel C
    # Keep within plot bounds (panel_c_left to panel_c_right)
    pad_x, pad_y = 20, 12
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y

    # Left edge: ensure box starts within Panel C plot area (past y-axis labels)
    box_x = panel_c_left + 350
    # Right edge: ensure box doesn't overlap topoplot (topoplot is ~320px from right)
    max_right = panel_c_right - 350
    if box_x + box_w > max_right:
        box_x = max_right - box_w
    # Clamp to panel left edge
    if box_x < panel_c_left + 10:
        box_x = panel_c_left + 10

    box_y = 2290

    # Light blue opaque rounded box
    draw.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=10, fill='#D6EAF8', outline='#85C1E9', width=2,
    )

    # Draw text
    draw.text((box_x + pad_x, box_y + pad_y), wrapped, fill='#1A2B3C', font=font)

    return img


def add_topoplot_inset(img, mean_topo_lap, pc_left=None, pc_right=None):
    """Add a small topoplot inset in the lower-right corner of Panel C."""
    w, h = img.size
    panel_c_right = pc_right if pc_right is not None else w

    # Render topoplot to a temporary figure
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in MONO_CHANNELS]
    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(mean_topo_lap))
    if vmax < 1e-10:
        vmax = 1.0

    fig_t, ax_t = plt.subplots(figsize=(2, 2))
    mne.viz.plot_topomap(
        mean_topo_lap, info, axes=ax_t, show=False,
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

    # Scale to desired inset size
    inset_size = 280
    topo_img = topo_img.resize((inset_size, inset_size), Image.LANCZOS)

    # Position: lower-right of Panel C, above the verbal description
    inset_x = panel_c_right - inset_size - 120
    inset_y = 2150

    img.paste(topo_img, (inset_x, inset_y), topo_img)
    return img


def build_full_figure(panel_a, panel_b, panel_c, verbal_text, mean_topo_lap):
    """Composite all three panels into the final figure with titles."""
    from PIL import ImageFont

    # Target size matching the backup figure dimensions
    W, H = 5782, 2618
    gap = 5  # minimal gap between panels

    # Panel widths: A=30%, B=40%, C=30% — A and C must be equal
    usable_w = W - 2 * gap
    b_w = int(usable_w * 0.40)
    a_w = (usable_w - b_w) // 2
    c_w = a_w  # force identical

    title_h = 150  # space for titles above panels
    panel_h = H - title_h

    # Scale each panel to fit its slot
    pa = panel_a.resize((a_w, panel_h), Image.LANCZOS)
    pb = panel_b.resize((b_w, panel_h), Image.LANCZOS)
    pc = panel_c.resize((c_w, panel_h), Image.LANCZOS)

    # Build composite
    comp = Image.new('RGB', (W, H), 'white')
    x = 0
    comp.paste(pa, (x, title_h))
    a_end = x + a_w

    x = a_end + gap
    # Panel B starts higher (closer to title) for better vertical centering
    b_y_offset = title_h - 40
    comp.paste(pb, (x, b_y_offset))
    b_end = x + b_w

    x = b_end + gap
    comp.paste(pc, (x, title_h))

    # Draw panel A and C titles to match Panel B's title style
    # Panel B title is "B. PDCharacterizer Architecture" at ~y=50 in the composite
    # rendered by draw_panel_b.py in bold DejaVu Sans at size 22 (in 820px canvas)
    # After scaling to 2618px, that's roughly equivalent to Pillow font size ~58
    draw = ImageDraw.Draw(comp)
    try:
        title_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 110)
        # Try to get bold variant
        try:
            title_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 110, index=1)
        except Exception:
            pass
    except Exception:
        title_font = ImageFont.load_default()

    # Match vertical position to Panel B's title (roughly y=50 in composite)
    title_y = 45
    draw.text((30, title_y), 'A. Input', fill='black', font=title_font)
    # Center Panel B title over Panel B area
    b_title = 'B. PD Pipeline Architecture'
    b_bbox = draw.textbbox((0, 0), b_title, font=title_font)
    b_title_w = b_bbox[2] - b_bbox[0]
    b_center = a_end + gap + b_w // 2
    draw.text((b_center - b_title_w // 2, title_y), b_title, fill='black', font=title_font)
    draw.text((b_end + gap + 30, title_y), 'C. Output', fill='black', font=title_font)

    # Store panel C position for overlays
    pc_left = b_end + gap
    pc_right = pc_left + c_w

    # Add topoplot inset to Panel C area
    if mean_topo_lap is not None:
        comp = comp.convert('RGBA')
        comp = add_topoplot_inset(comp, mean_topo_lap, pc_left=pc_left, pc_right=pc_right)
        comp = comp.convert('RGB')

    # Add verbal description
    if verbal_text:
        comp = add_verbal_description(comp, verbal_text, pc_left=pc_left, pc_right=pc_right)

    return comp


def main():
    print("=" * 50)
    print("Building Fig 2 composite")
    print("=" * 50)

    # Step 1: Compute all data from EEG segment
    data = compute_segment_data(MAT_FILE, subtype='lpd')

    # Step 2: Build case dict for EEG panels (same format as figs 5-8)
    mono_raw = load_monopolar(MAT_FILE)
    case = _build_case_dict(
        mono_raw, data['discharge_times'], data['laterality'], 'lpd',
    )

    # Step 3: Render Panel A (input EEG — no discharge markers)
    print("Rendering Panel A...")
    case_a = dict(case)
    case_a['gt_discharge_times'] = []  # no markers for input
    case_a['no_shading'] = True  # raw input — no laterality known yet
    panel_a = _render_eeg_panel(case_a, is_pd=False)
    print(f"  Panel A size: {panel_a.size}")

    # Step 4: Render Panel B
    print("Rendering Panel B...")
    panel_b = render_panel_b(topoplot_fn=None)
    print(f"  Panel B size: {panel_b.size}")

    # Step 5: Render Panel C (output EEG — with discharge markers + shading)
    print("Rendering Panel C...")
    panel_c = _render_eeg_panel(case, is_pd=True, show_scale_bar=False)
    print(f"  Panel C size: {panel_c.size}")

    # Step 6: Composite everything
    print("Compositing...")
    result = build_full_figure(
        panel_a, panel_b, panel_c,
        verbal_text=data['verbal'],
        mean_topo_lap=data['mean_topo_lap'],
    )
    result.save(str(COMPOSITE_PATH), dpi=(300, 300))
    print(f"Saved: {COMPOSITE_PATH} ({result.size})")


if __name__ == '__main__':
    main()
