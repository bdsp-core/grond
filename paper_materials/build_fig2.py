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
BACKUP_PATH = SCRIPT_DIR / 'figures' / 'fig2_pd_pipeline_backup.png'

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

# Panel B pixel bounds in the composite image
PANEL_B_LEFT = 1974
PANEL_B_TOP = 24
PANEL_B_RIGHT = 3840
PANEL_B_BOTTOM = 2517
PANEL_B_W = PANEL_B_RIGHT - PANEL_B_LEFT  # 1866
PANEL_B_H = PANEL_B_BOTTOM - PANEL_B_TOP   # 2493

# Panel C pixel bounds in the composite image (below "C. Output" title)
PANEL_C_LEFT = 3920
PANEL_C_TOP = 140
PANEL_C_RIGHT = 5740
PANEL_C_BOTTOM = 2590
PANEL_C_W = PANEL_C_RIGHT - PANEL_C_LEFT
PANEL_C_H = PANEL_C_BOTTOM - PANEL_C_TOP


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


def render_panel_c(mono_filt, discharge_times, laterality, subtype='lpd'):
    """Render Panel C (Output EEG with lateralized discharge markers)."""
    import matplotlib.gridspec as gridspec
    sys.path.insert(0, str(SCRIPT_DIR))
    from generate_fig2_pd_pipeline import plot_eeg_traces

    fig = plt.figure(figsize=(6, 7.5), facecolor='white')
    ax = fig.add_subplot(111)
    is_left = laterality == 'left'
    plot_eeg_traces(ax, mono_filt,
                    title='',
                    discharge_times=discharge_times,
                    highlight_left=is_left,
                    label_discharges=True)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    img = Image.open(tmp.name)
    Path(tmp.name).unlink()
    return img


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


def add_verbal_description(img, verbal_text):
    """Draw verbal description in a light blue box overlaid on bottom of Panel C."""
    draw = ImageDraw.Draw(img)
    w, h = img.size

    panel_c_left = PANEL_B_RIGHT
    panel_c_right = w

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


def add_topoplot_inset(img, mean_topo_lap):
    """Add a small topoplot inset in the lower-right corner of Panel C."""
    panel_c_left = PANEL_B_RIGHT
    w, h = img.size

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
    inset_x = w - inset_size - 40
    inset_y = 2080

    img.paste(topo_img, (inset_x, inset_y), topo_img)
    return img


def build_composite(panel_b_img, panel_c_img=None, verbal_text=None,
                    mean_topo_lap=None, base_path=None):
    """Paste Panel B (and optionally C) into the composite figure."""
    if base_path is None:
        base_path = BACKUP_PATH if BACKUP_PATH.exists() else COMPOSITE_PATH

    base = Image.open(str(base_path)).convert('RGBA')
    result = base.copy()

    # White out and paste Panel B
    draw = ImageDraw.Draw(result)
    draw.rectangle([PANEL_B_LEFT, 0, PANEL_B_RIGHT, base.size[1]], fill='white')
    pb_scaled = panel_b_img.resize((PANEL_B_W, PANEL_B_H), Image.LANCZOS)
    result.paste(pb_scaled, (PANEL_B_LEFT, PANEL_B_TOP))

    # White out and paste Panel C (if re-rendered)
    if panel_c_img is not None:
        draw = ImageDraw.Draw(result)
        # White out entire Panel C area below the title
        draw.rectangle([PANEL_C_LEFT - 20, PANEL_C_TOP, PANEL_C_RIGHT + 40, PANEL_C_BOTTOM], fill='white')
        pc_scaled = panel_c_img.resize((PANEL_C_W, PANEL_C_H), Image.LANCZOS)
        result.paste(pc_scaled, (PANEL_C_LEFT, PANEL_C_TOP))

    # Add topoplot inset to Panel C
    if mean_topo_lap is not None:
        result = add_topoplot_inset(result, mean_topo_lap)

    # Add verbal description overlaid on Panel C
    if verbal_text:
        result = add_verbal_description(result, verbal_text)

    return result.convert('RGB')


def main():
    print("=" * 50)
    print("Building Fig 2 composite")
    print("=" * 50)

    # Step 1: Compute all data from EEG segment
    data = compute_segment_data(MAT_FILE, subtype='lpd')

    # Step 2: Render Panel B (no topoplot — just "Localization" text)
    print("Rendering Panel B...")
    panel_b = render_panel_b(topoplot_fn=None)
    print(f"  Panel B size: {panel_b.size}")

    # Step 3: Composite with topoplot inset and verbal description on Panel C
    print("Compositing...")
    result = build_composite(
        panel_b,
        verbal_text=data['verbal'],
        mean_topo_lap=data['mean_topo_lap'],
    )
    result.save(str(COMPOSITE_PATH), dpi=(300, 300))
    print(f"Saved: {COMPOSITE_PATH} ({result.size})")


if __name__ == '__main__':
    main()
