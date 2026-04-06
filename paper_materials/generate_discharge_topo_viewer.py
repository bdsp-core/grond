"""
Discharge-locked topographic localization viewer.

For each PD segment with discharge timing labels:
  1. Load 19-channel monopolar EEG (200 Hz)
  2. Bandpass filter 0.5-20 Hz
  3. At each discharge time, extract GFP-aligned peak voltage across 19 channels
  4. Average the aligned voltage vectors -> mean discharge topography
  5. Generate MNE spherical spline topoplot
  6. Save as static PNG embedded in HTML viewer

Usage:
    conda run -n morgoth python paper_materials/generate_discharge_topo_viewer.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mne
mne.set_log_level('WARNING')

import io
import base64
import webbrowser

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'labeling_tools' / 'discharge_topo'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
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
BIPOLAR_NAMES = [f'{a}-{b}' for a, b in BIPOLAR_PAIRS]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])

DISPLAY_ORDER = [0, 1, 2, 3, -1, 8, 9, 10, 11, -1, 16, 17, -1, 12, 13, 14, 15, -1, 4, 5, 6, 7]


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
    n_ch, n_samp = mono.shape
    lap = np.zeros_like(mono)
    for ch in range(n_ch):
        nbrs = neighbors_map.get(ch, [])
        if nbrs:
            lap[ch] = mono[ch] - np.mean(mono[nbrs], axis=0)
        else:
            lap[ch] = mono[ch]
    return lap


# Neighbor map for 19-channel 10-20 system
LAP_NEIGHBORS = {
    0: [1, 4, 8, 11],      # Fp1: F3, F7, Fz, Fp2
    1: [0, 2, 4, 8],       # F3: Fp1, C3, F7, Fz
    2: [1, 3, 5, 9],       # C3: F3, P3, T3, Cz
    3: [2, 6, 7, 10],      # P3: C3, T5, O1, Pz
    4: [0, 1, 5],          # F7: Fp1, F3, T3
    5: [4, 2, 6],          # T3: F7, C3, T5
    6: [5, 3, 7],          # T5: T3, P3, O1
    7: [3, 6, 10],         # O1: P3, T5, Pz
    8: [0, 1, 9, 11, 12],  # Fz: Fp1, F3, Cz, Fp2, F4
    9: [8, 2, 10, 13],     # Cz: Fz, C3, Pz, C4
    10: [9, 3, 7, 14, 18], # Pz: Cz, P3, O1, P4, O2
    11: [12, 15, 8, 0],    # Fp2: F4, F8, Fz, Fp1
    12: [11, 13, 15, 8],   # F4: Fp2, C4, F8, Fz
    13: [12, 14, 16, 9],   # C4: F4, P4, T4, Cz
    14: [13, 17, 18, 10],  # P4: C4, T6, O2, Pz
    15: [11, 12, 16],      # F8: Fp2, F4, T4
    16: [15, 13, 17],      # T4: F8, C4, T6
    17: [16, 14, 18],      # T6: T4, P4, O2
    18: [14, 17, 10],      # O2: P4, T6, Pz
}


def gfp_align(mono_filtered, discharge_times_sec, fs=200, window_ms=25):
    """Two-pass discharge-locked topography with Laplacian-GFP alignment.

    Uses Laplacian-transformed data for alignment (better for focal sources)
    but extracts voltage from monopolar data for the topoplot.

    Pass 1: Laplacian-GFP-align each discharge, compute initial template.
    Pass 2: Cross-correlate each epoch's Laplacian-GFP with template to refine.
    Final: GFP²-weighted average of monopolar voltages at aligned times.

    Returns: mean_topo (19,) or None if <2 valid discharges.
    """
    window_samples = int(window_ms * fs / 1000)
    epoch_half = int(50 * fs / 1000)  # ±50ms epoch for template matching
    n_ch, n_total = mono_filtered.shape

    # Compute Laplacian for alignment
    lap = compute_laplacian(mono_filtered, LAP_NEIGHBORS)

    # ── Pass 1: Laplacian-GFP alignment ──
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

    # Extract ±50ms epochs around GFP-aligned peaks (both mono and Laplacian)
    mono_epochs = []
    lap_epochs = []
    valid_samples = []
    for s in gfp_aligned_samples:
        elo = s - epoch_half
        ehi = s + epoch_half + 1
        if elo < 0 or ehi > n_total:
            continue
        mono_epochs.append(mono_filtered[:, elo:ehi])
        lap_epochs.append(lap[:, elo:ehi])
        valid_samples.append(s)

    if len(mono_epochs) < 2:
        # Fall back to single-sample alignment
        mean_topo_mono = np.mean([mono_filtered[:, s] for s in gfp_aligned_samples], axis=0)
        mean_topo_lap = np.mean([lap[:, s] for s in gfp_aligned_samples], axis=0)
    else:
        epoch_len = mono_epochs[0].shape[1]
        # Template from Laplacian epochs (for alignment)
        lap_template = np.mean(lap_epochs, axis=0)

        # ── Pass 2: Template cross-correlation refinement using Laplacian GFP ──
        template_gfp = np.std(lap_template, axis=0)
        mid = epoch_len // 2
        max_shift = window_samples

        refined_voltages = []
        for mono_epoch, lap_epoch in zip(mono_epochs, lap_epochs):
            epoch_gfp = np.std(lap_epoch, axis=0)
            # Cross-correlate Laplacian GFP profiles
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
                # Extract MONOPOLAR voltage at the Laplacian-aligned time
                refined_voltages.append(mono_epoch[:, aligned_mid])

        if len(refined_voltages) < 2:
            mean_topo = np.mean([mono_filtered[:, s] for s in gfp_aligned_samples], axis=0)
        else:
            # GFP-weighted averaging using Laplacian GFP: real discharges have
            # high Laplacian GFP (focal activity), phantoms have low GFP.
            # Weighting by GFP^2 strongly suppresses phantom contributions.
            refined_voltages = np.array(refined_voltages)  # (n_discharges, 19)
            # Compute Laplacian of each discharge voltage for weighting
            lap_voltages = np.array([
                compute_laplacian(v.reshape(19, 1), LAP_NEIGHBORS).ravel()
                for v in refined_voltages
            ])
            gfp_weights = np.std(lap_voltages, axis=1)  # Laplacian GFP per discharge
            gfp_weights = gfp_weights ** 2  # square to amplify contrast
            weight_sum = np.sum(gfp_weights)
            if weight_sum > 1e-12:
                mean_topo_mono = np.average(refined_voltages, axis=0, weights=gfp_weights)
                mean_topo_lap = np.average(lap_voltages, axis=0, weights=gfp_weights)
            else:
                mean_topo_mono = np.mean(refined_voltages, axis=0)
                mean_topo_lap = np.mean(lap_voltages, axis=0)

    # Auto-flip polarity based on Laplacian (which reliably identifies the
    # discharge peak). Apply the SAME flip to both so they're consistent.
    # The Laplacian is the ground truth for where the discharge is — if
    # its max absolute is negative, flip both.
    if np.abs(np.min(mean_topo_lap)) > np.abs(np.max(mean_topo_lap)):
        mean_topo_mono = -mean_topo_mono
        mean_topo_lap = -mean_topo_lap

    return mean_topo_mono, mean_topo_lap


def generate_topoplot_b64(mean_topo, ch_names_orig, title='Mean discharge\ntopography'):
    """Generate topoplot as base64-encoded PNG."""
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in ch_names_orig]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(np.abs(mean_topo)))
    if vmax < 1e-10:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    mne.viz.plot_topomap(mean_topo, info, axes=ax, show=False,
                         contours=6, cmap='RdBu_r', sensors=True,
                         vlim=(-vmax, vmax),
                         names=mne_names, size=3)
    ax.set_title(title, fontsize=9)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def downsample_for_display(data, target_points=800):
    """Downsample EEG for display."""
    n_ch, n_samp = data.shape
    if n_samp <= target_points:
        return data
    indices = np.linspace(0, n_samp - 1, target_points).astype(int)
    return data[:, indices]


def select_cases(segment_labels_path, discharge_times_path, n_per_subtype=100):
    """Select top cases by discharge count for LPD and GPD."""
    import csv

    with open(segment_labels_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with open(discharge_times_path, 'r') as f:
        dt_data = json.load(f)

    # Build lookup: patient_id -> discharge info
    # discharge_times.json keys are patient_ids
    cases = []
    for row in rows:
        subtype = row.get('subtype', '')
        if subtype not in ('lpd', 'gpd'):
            continue
        if row.get('excluded', 'False').lower() == 'true':
            continue
        try:
            n_votes = float(row.get('iiic_n_votes', 0))
        except (ValueError, TypeError):
            continue
        if n_votes < 10:
            continue

        mat_file = row['mat_file']
        patient_id = row.get('patient_id', '')

        # Check discharge times - key is patient_id
        dt_entry = dt_data.get(patient_id)
        if dt_entry is None:
            continue
        # Handle both dict entries (with global_times) and plain list entries
        if isinstance(dt_entry, dict):
            global_times = dt_entry.get('global_times', [])
        elif isinstance(dt_entry, list):
            global_times = dt_entry
        else:
            continue
        if len(global_times) < 5:
            continue

        cases.append({
            'mat_file': mat_file,
            'patient_id': patient_id,
            'subtype': subtype,
            'discharge_times': global_times,
            'n_discharges': len(global_times),
        })

    # Sort by n_discharges descending, take top N per subtype
    lpd = sorted([c for c in cases if c['subtype'] == 'lpd'],
                 key=lambda x: -x['n_discharges'])[:n_per_subtype]
    gpd = sorted([c for c in cases if c['subtype'] == 'gpd'],
                 key=lambda x: -x['n_discharges'])[:n_per_subtype]

    print(f"Selected {len(lpd)} LPD and {len(gpd)} GPD cases")
    return lpd + gpd


def build_case_data(case_info):
    """Build display data for one case."""
    mono = load_monopolar(case_info['mat_file'])
    if mono is None:
        return None

    # Bandpass filter
    mono_filt = bandpass_filter(mono, lo=0.5, hi=20.0)

    # GFP-aligned mean topography (monopolar + Laplacian)
    result = gfp_align(mono_filt, case_info['discharge_times'])
    mean_topo_mono, mean_topo_lap = result
    if mean_topo_mono is None:
        return None

    # Generate both topoplots
    topo_img_mono = generate_topoplot_b64(mean_topo_mono, MONO_CHANNELS, title='Average Reference')
    topo_img_lap = generate_topoplot_b64(mean_topo_lap, MONO_CHANNELS, title='Laplacian')

    # Bipolar EEG for display (from filtered data)
    bipolar = mono_to_bipolar(mono_filt)
    bipolar_ds = downsample_for_display(bipolar, target_points=800)
    mono_ds = downsample_for_display(mono_filt, target_points=800)

    return {
        'segment_id': case_info['mat_file'].replace('.mat', ''),
        'mat_file': case_info['mat_file'],
        'patient_id': case_info['patient_id'],
        'subtype': case_info['subtype'],
        'n_discharges': case_info['n_discharges'],
        'discharge_times': case_info['discharge_times'],
        'eeg_data': bipolar_ds.tolist(),
        'mono_data': mono_ds.tolist(),
        'topo_img_mono': topo_img_mono,
        'topo_img_lap': topo_img_lap,
    }


def generate_html(cases_data, output_path):
    """Generate the HTML viewer."""
    cases_json = json.dumps(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Discharge-Locked Topographic Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: Consolas, Monaco, monospace; background: #f5f5f5; }}
#header {{
  background: #2c3e50; color: white; padding: 10px 20px;
  display: flex; align-items: center; justify-content: space-between;
}}
#header h1 {{ font-size: 16px; }}
#nav {{ display: flex; align-items: center; gap: 12px; }}
#nav button {{
  background: #3498db; color: white; border: none; padding: 6px 16px;
  border-radius: 4px; cursor: pointer; font-family: inherit; font-size: 13px;
}}
#nav button:hover {{ background: #2980b9; }}
#counter {{ color: #ecf0f1; font-size: 13px; }}
#info-bar {{
  background: #ecf0f1; padding: 8px 20px; font-size: 12px;
  display: flex; gap: 20px; align-items: center;
}}
.info-label {{ color: #7f8c8d; }}
.info-value {{ font-weight: bold; }}
.subtype-lpd {{ color: #e74c3c; }}
.subtype-gpd {{ color: #3498db; }}
#main {{
  display: flex; padding: 10px; gap: 10px; height: calc(100vh - 90px);
}}
#eeg-panel {{
  flex: 0 0 70%; background: white; border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden;
}}
#topo-panel {{
  flex: 0 0 28%; background: white; border-radius: 6px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 15px;
}}
#topo-panel img {{
  max-width: 100%; max-height: 80%;
}}
#topo-panel .topo-title {{
  font-size: 12px; color: #555; margin-top: 8px; text-align: center;
}}
canvas {{ display: block; }}
.kbd {{ background: #ddd; padding: 1px 6px; border-radius: 3px; font-size: 11px; }}
</style>
</head>
<body>
<div id="header">
  <h1>Discharge-Locked Topographic Viewer</h1>
  <div id="nav">
    <button onclick="prev()">&#9664; Prev</button>
    <span id="counter">1 / {len(cases_data)}</span>
    <button onclick="next()">Next &#9654;</button>
    <button id="montage-btn" onclick="cycleMontage()" style="background:#27ae60;">Bipolar</button>
    <button id="topo-btn" onclick="toggleTopoMode()" style="background:#8e44ad;">Laplacian topo</button>
    <span style="font-size:11px; color:#bdc3c7;">
      <span class="kbd">&larr;</span>/<span class="kbd">&rarr;</span> nav
      <span class="kbd">M</span>/<span class="kbd">Ctrl</span> montage
      <span class="kbd">T</span> topo mode
    </span>
  </div>
</div>
<div id="info-bar">
  <span><span class="info-label">Segment: </span><span class="info-value" id="info-seg"></span></span>
  <span><span class="info-label">Subtype: </span><span class="info-value" id="info-sub"></span></span>
  <span><span class="info-label">Patient: </span><span class="info-value" id="info-pat"></span></span>
  <span><span class="info-label">Discharges: </span><span class="info-value" id="info-nd"></span></span>
</div>
<div id="main">
  <div id="eeg-panel">
    <canvas id="eeg-canvas"></canvas>
  </div>
  <div id="topo-panel">
    <img id="topo-img" src="" alt="Topoplot">
    <div class="topo-title" id="topo-caption"></div>
  </div>
</div>

<script>
const CASES = {cases_json};
let idx = 0;
let montageMode = 'bipolar'; // 'bipolar', 'average', 'laplacian'
let topoMode = 'lap'; // 'mono' or 'lap'

function toggleTopoMode() {{
  topoMode = (topoMode === 'mono') ? 'lap' : 'mono';
  const btn = document.getElementById('topo-btn');
  btn.textContent = topoMode === 'mono' ? 'Avg Ref topo' : 'Laplacian topo';
  btn.style.background = topoMode === 'mono' ? '#2980b9' : '#8e44ad';
  updateTopo();
}}

function updateTopo() {{
  const c = CASES[idx];
  const imgSrc = topoMode === 'mono' ? c.topo_img_mono : c.topo_img_lap;
  document.getElementById('topo-img').src = 'data:image/png;base64,' + imgSrc;
  const label = topoMode === 'mono' ? 'Avg Ref' : 'Laplacian';
  document.getElementById('topo-caption').textContent =
    label + ' | ' + c.subtype.toUpperCase() + ' | ' + c.n_discharges + ' discharges';
}}

const BIPOLAR_NAMES = {json.dumps(BIPOLAR_NAMES)};
const MONO_NAMES = {json.dumps([n + '-avg' for n in MONO_CHANNELS])};
const MONO_RAW_NAMES = {json.dumps(MONO_CHANNELS)};
const BIPOLAR_DISPLAY_ORDER = {json.dumps(DISPLAY_ORDER)};
// CAR/Laplacian display: L parasag, L temporal, midline, R parasag, R temporal
const CAR_DISPLAY_ORDER = [
  0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18
];
const DURATION = {DURATION};

const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const MARGIN_LEFT = 75;
const MARGIN_RIGHT = 15;
const CLIP_UV = 300;
const Z_SCALE = 0.01;

// Laplacian neighbors (indices into 19-ch monopolar)
// Fp1=0,F3=1,C3=2,P3=3,F7=4,T3=5,T5=6,O1=7,Fz=8,Cz=9,Pz=10,
// Fp2=11,F4=12,C4=13,P4=14,F8=15,T4=16,T6=17,O2=18
const LAP_NEIGHBORS = {{
  0: [1,4,8,11],     // Fp1: F3,F7,Fz,Fp2
  1: [0,2,4,8],      // F3: Fp1,C3,F7,Fz
  2: [1,3,5,9],      // C3: F3,P3,T3,Cz
  3: [2,6,7,10],     // P3: C3,T5,O1,Pz
  4: [0,1,5],        // F7: Fp1,F3,T3
  5: [4,2,6],        // T3: F7,C3,T5
  6: [5,3,7],        // T5: T3,P3,O1
  7: [3,6,10],       // O1: P3,T5,Pz
  8: [0,1,9,11,12],  // Fz: Fp1,F3,Cz,Fp2,F4
  9: [8,2,10,13],    // Cz: Fz,C3,Pz,C4
  10:[9,3,7,14,18],  // Pz: Cz,P3,O1,P4,O2
  11:[12,15,8,0],    // Fp2: F4,F8,Fz,Fp1
  12:[11,13,15,8],   // F4: Fp2,C4,F8,Fz
  13:[12,14,16,9],   // C4: F4,P4,T4,Cz
  14:[13,17,18,10],  // P4: C4,T6,O2,Pz
  15:[11,12,16],     // F8: Fp2,F4,T4
  16:[15,13,17],     // T4: F8,C4,T6
  17:[16,14,18],     // T6: T4,P4,O2
  18:[14,17,10],     // O2: P4,T6,Pz
}};

function cycleMontage() {{
  if (montageMode === 'bipolar') montageMode = 'average';
  else if (montageMode === 'average') montageMode = 'laplacian';
  else montageMode = 'bipolar';
  const btn = document.getElementById('montage-btn');
  btn.textContent = montageMode.charAt(0).toUpperCase() + montageMode.slice(1);
  btn.style.background = montageMode === 'bipolar' ? '#27ae60' : montageMode === 'average' ? '#e67e22' : '#8e44ad';
  show();
}}

function getDisplayConfig() {{
  if (montageMode === 'bipolar') {{
    return {{ order: BIPOLAR_DISPLAY_ORDER, names: BIPOLAR_NAMES, getData: getBipolarData }};
  }} else if (montageMode === 'average') {{
    return {{ order: CAR_DISPLAY_ORDER, names: MONO_NAMES, getData: getAverageData }};
  }} else {{
    return {{ order: CAR_DISPLAY_ORDER, names: MONO_RAW_NAMES.map(n => n + '-lap'), getData: getLaplacianData }};
  }}
}}

function getBipolarData() {{ return CASES[idx].eeg_data; }}

function getAverageData() {{
  const mono = CASES[idx].mono_data;
  const nCh = mono.length, nSamp = mono[0].length;
  const avg = new Array(nSamp).fill(0);
  for (let ch = 0; ch < nCh; ch++)
    for (let s = 0; s < nSamp; s++) avg[s] += mono[ch][s];
  for (let s = 0; s < nSamp; s++) avg[s] /= nCh;
  const car = [];
  for (let ch = 0; ch < nCh; ch++) {{
    const row = new Array(nSamp);
    for (let s = 0; s < nSamp; s++) row[s] = mono[ch][s] - avg[s];
    car.push(row);
  }}
  return car;
}}

function getLaplacianData() {{
  const mono = CASES[idx].mono_data;
  const nCh = mono.length, nSamp = mono[0].length;
  const lap = [];
  for (let ch = 0; ch < nCh; ch++) {{
    const neighbors = LAP_NEIGHBORS[ch] || [];
    const row = new Array(nSamp);
    for (let s = 0; s < nSamp; s++) {{
      let nAvg = 0;
      for (const n of neighbors) nAvg += mono[n][s];
      nAvg /= neighbors.length || 1;
      row[s] = mono[ch][s] - nAvg;
    }}
    lap.push(row);
  }}
  return lap;
}}

function timeToX(t, plotLeft, plotW) {{ return plotLeft + (t / DURATION) * plotW; }}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  const panel = document.getElementById('eeg-panel');
  const W = panel.clientWidth;
  const H = panel.clientHeight;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const config = getDisplayConfig();
  const eegData = config.getData();
  const nSamples = eegData[0].length;

  // Build display channels from config
  const dispCh = [];
  for (const i of config.order) {{
    if (i < 0) dispCh.push({{ idx: -1, name: '' }});
    else dispCh.push({{ idx: i, name: config.names[i] }});
  }}
  const nDisp = dispCh.length;

  const PLOT_LEFT = MARGIN_LEFT;
  const PLOT_RIGHT = W - MARGIN_RIGHT;
  const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
  const PLOT_H = H - MARGIN_TOP - MARGIN_BOTTOM;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s, PLOT_LEFT, PLOT_W);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, H - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  const chSpacing = PLOT_H / (nDisp + 1);

  // Traces
  for (let di = 0; di < nDisp; di++) {{
    const ch = dispCh[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (let si = 0; si < nSamples; si++) {{
      const x = PLOT_LEFT + (si / (nSamples - 1)) * PLOT_W;
      let val = trace[si];
      val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
      const y = yCenter - val * Z_SCALE * chSpacing;
      if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Channel labels
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#000000';
  for (let di = 0; di < nDisp; di++) {{
    const ch = dispCh[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#000000';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', timeToX(s, PLOT_LEFT, PLOT_W), H - MARGIN_BOTTOM + 4);
  }}

  // Discharge time markers (red dashed vertical lines)
  const dtimes = c.discharge_times;
  ctx.strokeStyle = 'rgba(220, 50, 50, 0.7)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 3]);
  for (let i = 0; i < dtimes.length; i++) {{
    const x = timeToX(dtimes[i], PLOT_LEFT, PLOT_W);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    ctx.beginPath();
    ctx.moveTo(x, MARGIN_TOP);
    ctx.lineTo(x, H - MARGIN_BOTTOM);
    ctx.stroke();
  }}
  ctx.setLineDash([]);

  // Discharge time labels at top
  ctx.fillStyle = 'rgba(220, 50, 50, 0.8)';
  ctx.font = '8px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'bottom';
  for (let i = 0; i < dtimes.length; i++) {{
    const x = timeToX(dtimes[i], PLOT_LEFT, PLOT_W);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    ctx.fillText(dtimes[i].toFixed(2) + 's', x, MARGIN_TOP - 2);
  }}

  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 12px Consolas, Monaco, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(c.subtype.toUpperCase() + '  |  ' + c.patient_id + '  |  ' + c.n_discharges + ' discharges', PLOT_LEFT, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-seg').textContent = c.segment_id;
  const subEl = document.getElementById('info-sub');
  subEl.textContent = c.subtype.toUpperCase();
  subEl.className = 'info-value subtype-' + c.subtype;
  document.getElementById('info-pat').textContent = c.patient_id;
  document.getElementById('info-nd').textContent = c.n_discharges;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  updateTopo();
}}

function show() {{
  updateInfo();
  drawEEG();
}}

function prev() {{ if (idx > 0) {{ idx--; show(); }} }}
function next() {{ if (idx < CASES.length - 1) {{ idx++; show(); }} }}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); prev(); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); next(); }}
  else if (e.key === 'm' || e.key === 'M' || e.key === 'Control') {{ e.preventDefault(); cycleMontage(); }}
  else if (e.key === 't' || e.key === 'T') {{ e.preventDefault(); toggleTopoMode(); }}
}});

window.addEventListener('resize', () => show());
window.addEventListener('load', () => show());
</script>
</body>
</html>"""
    with open(output_path, 'w') as f:
        f.write(html)
    print(f"Wrote HTML viewer to {output_path}")


def main():
    segment_labels_path = LABELS_DIR / 'segment_labels.csv'
    discharge_times_path = LABELS_DIR / 'discharge_times.json'

    print("Selecting cases...")
    selected = select_cases(segment_labels_path, discharge_times_path, n_per_subtype=100)

    if not selected:
        print("No cases found!")
        return

    cases_data = []
    for i, case_info in enumerate(selected):
        result = build_case_data(case_info)
        if result is not None:
            cases_data.append(result)
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1}/{len(selected)} cases ({len(cases_data)} successful)")

    print(f"\nTotal cases built: {len(cases_data)}")

    if not cases_data:
        print("No cases could be built!")
        return

    output_path = OUT_DIR / 'discharge_topo_viewer.html'
    generate_html(cases_data, output_path)

    print(f"Opening viewer...")
    webbrowser.open(f'file://{output_path}')


if __name__ == '__main__':
    main()
