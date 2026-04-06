#!/usr/bin/env python3
"""
Regenerate figure data JSONs for figures 1-3b with discharge-locked topographic
localization (PD) and narrowband amplitude topography (RDA).

Reads existing figure_*_examples_data.json to reuse the same segment_ids,
then recomputes:
  - topo_img_mono / topo_img_lap: MNE topoplot PNGs (base64, inferno)
  - verbal_description: from morgoth-viewer describe_ied_topoplot()
  - pred_lat: from discharge topo (PD) or narrowband variance (RDA)
  - discharge_times: from discharge_times.json (PD only)

Usage:
    conda run -n morgoth python paper_materials/generate_figure_data.py
"""

import sys
import json
import csv
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
mne.set_log_level('ERROR')

import io
import base64

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
PAPER_DIR = PROJECT_DIR / 'paper_materials'

FS = 200
N_SAMPLES = 2000

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])

# Left/right monopolar channel indices for laterality
LEFT_IDX = [0, 1, 2, 3, 4, 5, 6, 7]      # Fp1,F3,C3,P3,F7,T3,T5,O1
RIGHT_IDX = [11, 12, 13, 14, 15, 16, 17, 18]  # Fp2,F4,C4,P4,F8,T4,T6,O2

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


# ── I/O helpers ────────────────────────────────────────────────────────────

def load_monopolar(mat_file):
    """Load raw monopolar EEG (19 channels, 2000 samples)."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :N_SAMPLES]
    if seg.shape[0] == 19:
        return seg.astype(np.float64)
    return None


def bandpass_filter(mono, lo=0.5, hi=20.0, fs=200, order=4):
    """Bandpass filter monopolar data."""
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='bandpass', output='sos')
    filtered = np.zeros_like(mono)
    for ch in range(mono.shape[0]):
        try:
            filtered[ch] = sosfiltfilt(sos, mono[ch])
        except Exception:
            filtered[ch] = mono[ch]
    return filtered


def mono_to_bipolar(mono):
    """Convert monopolar (19,N) to bipolar (18,N)."""
    return mono[BIPOLAR_INDICES[:, 0]] - mono[BIPOLAR_INDICES[:, 1]]


def compute_laplacian(mono, neighbors_map):
    """Compute Laplacian (each channel minus mean of neighbors)."""
    n_ch = mono.shape[0]
    if mono.ndim == 1:
        lap = np.zeros_like(mono)
        for ch in range(n_ch):
            nbrs = neighbors_map.get(ch, [])
            if nbrs:
                lap[ch] = mono[ch] - np.mean(mono[nbrs])
            else:
                lap[ch] = mono[ch]
        return lap
    else:
        n_samp = mono.shape[1]
        lap = np.zeros_like(mono)
        for ch in range(n_ch):
            nbrs = neighbors_map.get(ch, [])
            if nbrs:
                lap[ch] = mono[ch] - np.mean(mono[nbrs], axis=0)
            else:
                lap[ch] = mono[ch]
        return lap


def downsample_for_display(data, target_points=1000):
    """Downsample EEG for display in the figure."""
    n_ch, n_samp = data.shape
    if n_samp <= target_points:
        return data
    indices = np.linspace(0, n_samp - 1, target_points).astype(int)
    return data[:, indices]


# ── PD: Discharge-locked topography ───────────────────────────────────────

def gfp_align(mono_filtered, discharge_times_sec, fs=200, window_ms=25):
    """Two-pass discharge-locked topography with Laplacian-GFP alignment.

    Copied from generate_discharge_topo_viewer.py.
    Returns: (mean_topo_mono, mean_topo_lap) each (19,) or (None, None).
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

        # Pass 2: Template cross-correlation refinement
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
            mean_topo_lap = np.mean([compute_laplacian(mono_filtered[:, s:s+1], LAP_NEIGHBORS).ravel()
                                     for s in gfp_aligned_samples], axis=0)
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


# ── RDA: Narrowband amplitude envelope ────────────────────────────────────

def compute_amplitude_envelope(mono, freq_hz, bw=0.4):
    """Compute narrowband amplitude envelope per channel.

    Returns:
        amplitude_vector: (19,) mean absolute Hilbert envelope
    """
    lo = max(freq_hz - bw, 0.1)
    hi = min(freq_hz + bw, FS / 2 - 0.1)
    if lo >= hi:
        return np.zeros(mono.shape[0])

    sos = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    amplitude_vector = np.zeros(mono.shape[0])

    for ch in range(mono.shape[0]):
        try:
            nb = sosfiltfilt(sos, mono[ch])
            amplitude_vector[ch] = np.mean(np.abs(hilbert(nb)))
        except Exception:
            pass

    return amplitude_vector


# ── Topoplot generation ───────────────────────────────────────────────────

def generate_topoplot_b64(data_vector, ch_names_orig, title='Topography'):
    """Generate topoplot as base64-encoded PNG with inferno colormap.

    Uses original 10-20 names with adaptive text color.
    """
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in ch_names_orig]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(data_vector))
    if vmax < 1e-10:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    image, _ = mne.viz.plot_topomap(data_vector, info, axes=ax, show=False,
                                     contours=6, cmap='inferno', sensors=False,
                                     vlim=(0, vmax))

    # Get electrode positions
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks='eeg')

    # Draw original 10-20 names with adaptive text color
    cmap = plt.cm.inferno
    for i, (orig_name, xy) in enumerate(zip(ch_names_orig, pos)):
        val_normalized = data_vector[i] / vmax if vmax > 1e-10 else 0.0
        bg_color = cmap(val_normalized)
        lum = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
        text_color = 'white' if lum < 0.45 else 'black'
        ax.text(xy[0], xy[1], orig_name, fontsize=6, ha='center', va='center',
                fontweight='bold', color=text_color, zorder=10)

    ax.set_title(title, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Verbal description ────────────────────────────────────────────────────

def generate_verbal_description(subtype, freq_hz, topo_vector, laterality):
    """Generate ACNS-style verbal description — delegates to the shared function
    in generate_discharge_topo_viewer.py for consistency across all figures/viewers.
    """
    from generate_discharge_topo_viewer import generate_verbal_from_topo
    freq = freq_hz if freq_hz and np.isfinite(freq_hz) else np.nan
    return generate_verbal_from_topo(subtype, freq, topo_vector, laterality_from_pdchar=laterality)


# ── Laterality computation ────────────────────────────────────────────────

def compute_laterality_from_topo(topo_vector, subtype):
    """Determine laterality from a 19-element topography vector.

    For LPD/LRDA: compare left vs right hemisphere amplitude.
    For GPD/GRDA: always 'bilateral' or 'generalized'.
    """
    if subtype in ('gpd', 'grda'):
        return 'bilateral'

    left_amp = np.mean(topo_vector[LEFT_IDX])
    right_amp = np.mean(topo_vector[RIGHT_IDX])

    ratio = left_amp / (right_amp + 1e-12)
    if ratio > 1.3:
        return 'left'
    elif ratio < 1 / 1.3:
        return 'right'
    else:
        return 'bilateral'


# ── Process one PD case ───────────────────────────────────────────────────

def process_pd_case(case, discharge_times_dict):
    """Recompute topoplot and verbal description for a PD case."""
    mat_file = case['mat_file']
    segment_id = case['segment_id']
    subtype = case['subtype']

    # Load EEG
    mono = load_monopolar(mat_file)
    if mono is None:
        print(f'  WARNING: Could not load {mat_file}, keeping existing data')
        return case

    # Bandpass 0.5-20 Hz
    mono_filt = bandpass_filter(mono, lo=0.5, hi=20.0)

    # Get discharge times
    key = mat_file.replace('.mat', '')
    dt_entry = discharge_times_dict.get(key, discharge_times_dict.get(segment_id, None))
    if dt_entry is None:
        print(f'  WARNING: No discharge times for {segment_id}, keeping existing data')
        return case

    # dt_entry can be a dict (with 'global_times') or a list (just times)
    if isinstance(dt_entry, dict):
        discharge_times = dt_entry.get('global_times', [])
    elif isinstance(dt_entry, list):
        discharge_times = dt_entry
    else:
        discharge_times = []
    if len(discharge_times) < 2:
        print(f'  WARNING: <2 discharge times for {segment_id}, keeping existing data')
        return case

    # GFP-aligned topography
    mean_topo_mono, mean_topo_lap = gfp_align(mono_filt, discharge_times)
    if mean_topo_mono is None:
        print(f'  WARNING: gfp_align returned None for {segment_id}, keeping existing data')
        return case

    # Generate topoplot images
    topo_img_mono = generate_topoplot_b64(mean_topo_mono, MONO_CHANNELS,
                                           title='Monopolar\ntopography')
    topo_img_lap = generate_topoplot_b64(mean_topo_lap, MONO_CHANNELS,
                                          title='Laplacian\ntopography')

    # Laterality from Laplacian topography
    laterality = compute_laterality_from_topo(mean_topo_lap, subtype)

    # Frequency from discharge times IPI
    if len(discharge_times) >= 2:
        ipis = np.diff(discharge_times)
        freq = 1.0 / np.median(ipis) if np.median(ipis) > 0 else case.get('pred_freq', 0)
    else:
        freq = case.get('gt_freq', case.get('pred_freq', 0))

    # Verbal description
    verbal = generate_verbal_description(subtype, freq, mean_topo_lap, laterality)

    # Bipolar EEG for display
    bipolar = mono_to_bipolar(mono_filt)
    bipolar_ds = downsample_for_display(bipolar, target_points=1000)

    # Update case
    case['topo_img_mono'] = topo_img_mono
    case['topo_img_lap'] = topo_img_lap
    case['verbal_description'] = verbal
    case['pred_lat'] = laterality
    case['pred_freq'] = float(freq)
    case['discharge_times'] = [float(t) for t in discharge_times]
    case['eeg_data'] = bipolar_ds.tolist()
    # Keep gt fields
    # Remove old region_scores (no longer needed)
    case.pop('region_scores', None)
    case.pop('channel_scores', None)

    return case


# ── Process one RDA case ──────────────────────────────────────────────────

def process_rda_case(case, segment_labels_lookup):
    """Recompute topoplot and verbal description for an RDA case."""
    mat_file = case['mat_file']
    segment_id = case['segment_id']
    subtype = case['subtype']

    # Load EEG
    mono = load_monopolar(mat_file)
    if mono is None:
        print(f'  WARNING: Could not load {mat_file}, keeping existing data')
        return case

    # Get frequency from segment_labels (pdchar_freq_hz)
    sl_row = segment_labels_lookup.get(mat_file, {})
    freq_str = sl_row.get('pdchar_freq_hz', '')
    if freq_str:
        try:
            freq_hz = float(freq_str)
        except (ValueError, TypeError):
            freq_hz = case.get('gt_freq', case.get('pred_freq', 1.0))
    else:
        freq_hz = case.get('gt_freq', case.get('pred_freq', 1.0))

    # Narrowband amplitude envelope
    amplitude_vector = compute_amplitude_envelope(mono, freq_hz)

    if np.max(amplitude_vector) < 1e-10:
        print(f'  WARNING: Zero amplitude for {segment_id}, keeping existing data')
        return case

    # Laplacian of amplitude vector
    lap_vector = compute_laplacian(amplitude_vector, LAP_NEIGHBORS)
    lap_vector = np.abs(lap_vector)

    # Generate topoplot images
    topo_img_mono = generate_topoplot_b64(amplitude_vector, MONO_CHANNELS,
                                           title='Amplitude\ntopography')
    topo_img_lap = generate_topoplot_b64(lap_vector, MONO_CHANNELS,
                                          title='Laplacian\ntopography')

    # Laterality from amplitude (W05 narrowband variance comparison)
    laterality = compute_laterality_from_topo(amplitude_vector, subtype)

    # Verbal description
    verbal = generate_verbal_description(subtype, freq_hz, amplitude_vector, laterality)

    # Bipolar EEG for display (broadband filtered)
    mono_filt = bandpass_filter(mono, lo=0.5, hi=20.0)
    bipolar = mono_to_bipolar(mono_filt)
    bipolar_ds = downsample_for_display(bipolar, target_points=1000)

    # Update case
    case['topo_img_mono'] = topo_img_mono
    case['topo_img_lap'] = topo_img_lap
    case['verbal_description'] = verbal
    case['pred_lat'] = laterality
    case['pred_freq'] = float(freq_hz)
    case['eeg_data'] = bipolar_ds.tolist()
    case.pop('region_scores', None)
    case.pop('channel_scores', None)

    return case


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print('Loading labels...')

    # Load discharge times
    with open(LABELS_DIR / 'discharge_times.json') as f:
        discharge_times = json.load(f)
    print(f'  {len(discharge_times)} segments with discharge times')

    # Load segment_labels.csv as lookup by mat_file
    segment_labels_lookup = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            segment_labels_lookup[row['mat_file']] = row
    print(f'  {len(segment_labels_lookup)} segments in segment_labels.csv')

    subtypes = ['lpd', 'gpd', 'lrda', 'grda']

    for subtype in subtypes:
        json_path = PAPER_DIR / f'figure_{subtype}_examples_data.json'
        if not json_path.exists():
            print(f'Skipping {subtype}: {json_path} not found')
            continue

        with open(json_path) as f:
            cases = json.load(f)

        print(f'\nProcessing {subtype.upper()} ({len(cases)} cases)...')
        is_pd = subtype in ('lpd', 'gpd')

        updated_cases = []
        for i, case in enumerate(cases):
            sid = case['segment_id']
            diff = case.get('difficulty', '?')
            print(f'  [{i+1}/{len(cases)}] {sid} ({diff})')

            if is_pd:
                updated = process_pd_case(case, discharge_times)
            else:
                updated = process_rda_case(case, segment_labels_lookup)

            updated_cases.append(updated)

        # Save updated JSON
        out_path = PAPER_DIR / f'figure_{subtype}_examples_data.json'
        with open(out_path, 'w') as f:
            json.dump(updated_cases, f)
        print(f'  Saved: {out_path} ({len(updated_cases)} cases)')


if __name__ == '__main__':
    main()
