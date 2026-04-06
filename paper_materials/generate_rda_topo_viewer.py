"""
RDA Spatial Localization Viewer.

For each RDA segment (LRDA/GRDA) with W05 frequency estimates:
  1. Load 19-channel monopolar EEG (200 Hz)
  2. Bandpass filter 0.5-20 Hz for EEG display
  3. Compute narrowband filter at freq +/- 0.4 Hz
  4. Per-channel amplitude envelope via Hilbert -> 19-element vector
  5. Compute Laplacian of the amplitude vector
  6. Generate two MNE topoplots: monopolar amplitude and Laplacian amplitude
  7. Compute laterality from narrowband variance (left vs right)
  8. Generate verbal description using morgoth-viewer's describe_ied_topoplot()

Usage:
    conda run -n morgoth python paper_materials/generate_rda_topo_viewer.py
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
mne.set_log_level('WARNING')

import io
import base64
import webbrowser

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'labeling_tools' / 'rda_topo'
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

LEFT_IDX = [0, 1, 2, 3, 4, 5, 6, 7]    # monopolar left channels
RIGHT_IDX = [11, 12, 13, 14, 15, 16, 17, 18]  # monopolar right channels

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


# ---------- I/O ----------

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


def compute_laplacian_timeseries(mono, neighbors_map):
    """Compute Laplacian of timeseries (each channel minus mean of neighbors)."""
    n_ch, n_samp = mono.shape
    lap = np.zeros_like(mono)
    for ch in range(n_ch):
        nbrs = neighbors_map.get(ch, [])
        if nbrs:
            lap[ch] = mono[ch] - np.mean(mono[nbrs], axis=0)
        else:
            lap[ch] = mono[ch]
    return lap


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


def downsample_for_display(data, target_points=800):
    """Downsample EEG for display."""
    n_ch, n_samp = data.shape
    if n_samp <= target_points:
        return data
    indices = np.linspace(0, n_samp - 1, target_points).astype(int)
    return data[:, indices]


# ---------- Amplitude envelope ----------

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


# ---------- Topoplot generation ----------

def generate_topoplot_b64(amplitude_vector, ch_names_orig, title='Amplitude'):
    """Generate topoplot as base64-encoded PNG using inferno colormap."""
    name_map = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
    mne_names = [name_map.get(n, n) for n in ch_names_orig]

    info = mne.create_info(ch_names=mne_names, sfreq=200, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    vmax = float(np.max(amplitude_vector))
    if vmax < 1e-10:
        vmax = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    image, _ = mne.viz.plot_topomap(amplitude_vector, info, axes=ax, show=False,
                                     contours=6, cmap='inferno', sensors=False,
                                     vlim=(0, vmax))

    # Get electrode positions
    from mne.channels.layout import _find_topomap_coords
    pos = _find_topomap_coords(info, picks='eeg')

    # Draw original 10-20 names with adaptive text color
    cmap = plt.cm.inferno
    for i, (orig_name, xy) in enumerate(zip(ch_names_orig, pos)):
        val_normalized = amplitude_vector[i] / vmax if vmax > 1e-10 else 0.0
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


# ---------- Verbal description ----------

def generate_verbal_description(subtype, freq_hz, amplitude_vector):
    """Generate ACNS 2021 verbal description.

    Uses morgoth-viewer's describe_ied_topoplot() for regional localization
    and amplitude-based laterality from narrowband data.
    """
    # Laterality from amplitude vector
    left_amp = np.mean(amplitude_vector[LEFT_IDX])
    right_amp = np.mean(amplitude_vector[RIGHT_IDX])
    side = 'left' if left_amp > right_amp else 'right'

    # Regional localization via morgoth-viewer
    try:
        sys.path.insert(0, '/Users/mwestover/GithubRepos/morgoth-viewer')
        from morgoth_viewer_app.processing.ied_localization import describe_ied_topoplot
        result = describe_ied_topoplot(amplitude_vector)
        descriptor = result['descriptor']  # e.g., "left temporal"
    except Exception as e:
        descriptor = 'unknown region'

    type_str = subtype.upper()
    freq_str = f'at {freq_hz:.1f} Hz' if np.isfinite(freq_hz) else ''

    if subtype == 'lrda':
        parts = [type_str, f'{side} sided (unilateral)', freq_str, descriptor]
    else:  # grda
        parts = [type_str, freq_str, 'generalized', descriptor]

    return ', '.join(p for p in parts if p) + '.'


# ---------- Case selection ----------

def select_cases(n_per_subtype=100):
    """Select top RDA cases by IIIC agreement that have W05 frequency estimates."""
    sl_path = LABELS_DIR / 'segment_labels.csv'
    with open(sl_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    lrda_cases = []
    grda_cases = []

    for row in rows:
        subtype = row.get('subtype', '')
        if subtype not in ('lrda', 'grda'):
            continue
        if row.get('excluded', 'False').lower() == 'true':
            continue

        # Need iiic_n_votes >= 10
        try:
            n_votes = float(row.get('iiic_n_votes', 0))
        except (ValueError, TypeError):
            continue
        if n_votes < 10:
            continue

        # Need pdchar_freq_hz (W05 frequency estimate)
        freq_str = row.get('pdchar_freq_hz', '')
        if not freq_str or freq_str == '':
            continue
        try:
            freq_hz = float(freq_str)
            if not np.isfinite(freq_hz) or freq_hz <= 0:
                continue
        except (ValueError, TypeError):
            continue

        try:
            plurality_frac = float(row.get('iiic_plurality_frac', 0))
        except (ValueError, TypeError):
            continue

        case = {
            'mat_file': row['mat_file'],
            'patient_id': row.get('patient_id', ''),
            'subtype': subtype,
            'freq_hz': freq_hz,
            'n_votes': int(n_votes),
            'plurality_frac': plurality_frac,
        }

        if subtype == 'lrda':
            lrda_cases.append(case)
        else:
            grda_cases.append(case)

    # Sort by plurality_frac descending (highest agreement first)
    lrda_cases.sort(key=lambda x: -x['plurality_frac'])
    grda_cases.sort(key=lambda x: -x['plurality_frac'])

    selected_lrda = lrda_cases[:n_per_subtype]
    selected_grda = grda_cases[:n_per_subtype]

    print(f"Selected {len(selected_lrda)} LRDA and {len(selected_grda)} GRDA cases")
    return selected_lrda + selected_grda


# ---------- Build case data ----------

def build_case_data(case_info):
    """Build display data for one case."""
    mono = load_monopolar(case_info['mat_file'])
    if mono is None:
        return None

    freq_hz = case_info['freq_hz']

    # Bandpass filter for display
    mono_filt = bandpass_filter(mono, lo=0.5, hi=20.0)

    # Compute amplitude envelope from raw monopolar
    amplitude_vector, narrowband = compute_amplitude_envelope(mono, freq_hz, bw=0.4)

    if np.max(amplitude_vector) < 1e-10:
        return None

    # Compute Laplacian of amplitude vector
    lap_amplitude = compute_laplacian_vector(amplitude_vector, LAP_NEIGHBORS)
    # Rectify: Laplacian amplitude should be non-negative for display
    lap_amplitude = np.maximum(lap_amplitude, 0)

    # Generate topoplots
    topo_img_mono = generate_topoplot_b64(amplitude_vector, MONO_CHANNELS,
                                           title='Monopolar Amplitude')
    topo_img_lap = generate_topoplot_b64(lap_amplitude, MONO_CHANNELS,
                                          title='Laplacian Amplitude')

    # Verbal description
    try:
        verbal = generate_verbal_description(case_info['subtype'], freq_hz, amplitude_vector)
    except Exception:
        verbal = f"{case_info['subtype'].upper()}"

    # EEG display data
    bipolar_filt = mono_to_bipolar(mono_filt)
    bipolar_ds = downsample_for_display(bipolar_filt, target_points=800)
    mono_ds = downsample_for_display(mono_filt, target_points=800)

    # Raw bipolar for client-side FFT narrowband overlay
    bipolar_raw = mono_to_bipolar(mono)
    raw_bipolar_ds = downsample_for_display(bipolar_raw, target_points=400)

    return {
        'segment_id': case_info['mat_file'].replace('.mat', ''),
        'mat_file': case_info['mat_file'],
        'patient_id': case_info['patient_id'],
        'subtype': case_info['subtype'],
        'freq_hz': round(freq_hz, 2),
        'eeg_data': bipolar_ds.tolist(),
        'mono_data': mono_ds.tolist(),
        'raw_bipolar': raw_bipolar_ds.tolist(),
        'topo_img_mono': topo_img_mono,
        'topo_img_lap': topo_img_lap,
        'verbal': verbal,
    }


# ---------- HTML generation ----------

def generate_html(cases_data, output_path):
    """Generate the HTML viewer."""
    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RDA Spatial Localization Viewer ({n_cases} cases)</title>
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
.subtype-lrda {{ color: #e67e22; }}
.subtype-grda {{ color: #27ae60; }}
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
  max-width: 100%; max-height: 70%;
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
  <h1>RDA Spatial Localization Viewer</h1>
  <div id="nav">
    <button onclick="prev()">&#9664; Prev</button>
    <span id="counter">1 / {n_cases}</span>
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
  <span><span class="info-label">Frequency: </span><span class="info-value" id="info-freq"></span></span>
</div>
<div id="main">
  <div id="eeg-panel">
    <canvas id="eeg-canvas"></canvas>
  </div>
  <div id="topo-panel">
    <img id="topo-img" src="" alt="Topoplot">
    <div class="topo-title" id="topo-caption"></div>
    <div id="verbal-desc" style="font-size:12px; color:#333; margin-top:10px; padding:8px;
         background:#f0f0f0; border-radius:4px; text-align:center; line-height:1.4;"></div>
  </div>
</div>

<script>
const CASES = {cases_json};
let idx = 0;
let montageMode = 'bipolar'; // 'bipolar', 'average', 'laplacian'
let topoMode = 'lap'; // 'mono' or 'lap'

const BIPOLAR_NAMES = {json.dumps(BIPOLAR_NAMES)};
const MONO_NAMES = {json.dumps([n + '-avg' for n in MONO_CHANNELS])};
const MONO_RAW_NAMES = {json.dumps(MONO_CHANNELS)};
const BIPOLAR_DISPLAY_ORDER = {json.dumps(DISPLAY_ORDER)};
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

const LAP_NEIGHBORS = {{
  0: [1,4,8,11],
  1: [0,2,4,8],
  2: [1,3,5,9],
  3: [2,6,7,10],
  4: [0,1,5],
  5: [4,2,6],
  6: [5,3,7],
  7: [3,6,10],
  8: [0,1,9,11,12],
  9: [8,2,10,13],
  10:[9,3,7,14,18],
  11:[12,15,8,0],
  12:[11,13,15,8],
  13:[12,14,16,9],
  14:[13,17,18,10],
  15:[11,12,16],
  16:[15,13,17],
  17:[16,14,18],
  18:[14,17,10],
}};

// ---------- FFT narrowband (client-side) ----------

function fftNarrowband(signal, centerFreq, bw, fs) {{
  const N = signal.length;
  const N2 = 1 << Math.ceil(Math.log2(N));
  const re = new Float64Array(N2);
  const im = new Float64Array(N2);
  for (let i = 0; i < N; i++) re[i] = signal[i];
  fft(re, im, false);
  const lo = centerFreq - bw;
  const hi = centerFreq + bw;
  for (let k = 0; k <= N2 / 2; k++) {{
    const freq = k * fs / N2;
    if (freq < lo || freq > hi) {{
      re[k] = 0; im[k] = 0;
      if (k > 0 && k < N2 / 2) {{
        re[N2 - k] = 0; im[N2 - k] = 0;
      }}
    }}
  }}
  fft(re, im, true);
  return Array.from(re.slice(0, N));
}}

function fft(re, im, inverse) {{
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) {{
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {{
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }}
  }}
  for (let len = 2; len <= N; len *= 2) {{
    const ang = 2 * Math.PI / len * (inverse ? -1 : 1);
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {{
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {{
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        const vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len/2] = uRe - vRe;
        im[i + j + len/2] = uIm - vIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }}
    }}
  }}
  if (inverse) {{
    for (let i = 0; i < N; i++) {{ re[i] /= N; im[i] /= N; }}
  }}
}}

// Narrowband cache
const nbCache = {{}};

function getNarrowband(caseIdx) {{
  const key = caseIdx.toString();
  if (nbCache[key]) return nbCache[key];
  const c = CASES[caseIdx];
  const raw = c.raw_bipolar;
  const effectiveFs = raw[0].length / 10.0;
  const bw = 0.4;
  const result = [];
  for (let ch = 0; ch < raw.length; ch++) {{
    result.push(fftNarrowband(raw[ch], c.freq_hz, bw, effectiveFs));
  }}
  nbCache[key] = result;
  return result;
}}

// ---------- Montage helpers ----------

function cycleMontage() {{
  if (montageMode === 'bipolar') montageMode = 'average';
  else if (montageMode === 'average') montageMode = 'laplacian';
  else montageMode = 'bipolar';
  const btn = document.getElementById('montage-btn');
  btn.textContent = montageMode.charAt(0).toUpperCase() + montageMode.slice(1);
  btn.style.background = montageMode === 'bipolar' ? '#27ae60' : montageMode === 'average' ? '#e67e22' : '#8e44ad';
  show();
}}

function toggleTopoMode() {{
  topoMode = (topoMode === 'mono') ? 'lap' : 'mono';
  const btn = document.getElementById('topo-btn');
  btn.textContent = topoMode === 'mono' ? 'Monopolar topo' : 'Laplacian topo';
  btn.style.background = topoMode === 'mono' ? '#2980b9' : '#8e44ad';
  updateTopo();
}}

function updateTopo() {{
  const c = CASES[idx];
  const imgSrc = topoMode === 'mono' ? c.topo_img_mono : c.topo_img_lap;
  document.getElementById('topo-img').src = 'data:image/png;base64,' + imgSrc;
  const label = topoMode === 'mono' ? 'Monopolar' : 'Laplacian';
  document.getElementById('topo-caption').textContent =
    label + ' | ' + c.subtype.toUpperCase() + ' | ' + c.freq_hz.toFixed(1) + ' Hz';
  document.getElementById('verbal-desc').textContent = c.verbal || '';
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

  // EEG traces
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

  // Green narrowband overlay (only in bipolar mode)
  if (montageMode === 'bipolar') {{
    const nb = getNarrowband(idx);
    const nbSamples = nb[0].length;
    for (let di = 0; di < nDisp; di++) {{
      const ch = dispCh[di];
      if (ch.idx < 0 || ch.idx >= nb.length) continue;
      const yCenter = MARGIN_TOP + chSpacing * (di + 1);
      const trace = nb[ch.idx];
      ctx.strokeStyle = 'rgba(0, 180, 0, 0.6)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (let si = 0; si < nbSamples; si++) {{
        const x = PLOT_LEFT + (si / (nbSamples - 1)) * PLOT_W;
        let val = trace[si];
        val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
        const y = yCenter - val * Z_SCALE * chSpacing;
        if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
    }}
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

  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 12px Consolas, Monaco, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText(c.subtype.toUpperCase() + '  |  ' + c.patient_id + '  |  ' + c.freq_hz.toFixed(1) + ' Hz', PLOT_LEFT, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-seg').textContent = c.segment_id;
  const subEl = document.getElementById('info-sub');
  subEl.textContent = c.subtype.toUpperCase();
  subEl.className = 'info-value subtype-' + c.subtype;
  document.getElementById('info-pat').textContent = c.patient_id;
  document.getElementById('info-freq').textContent = c.freq_hz.toFixed(2) + ' Hz';
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


# ---------- Main ----------

def main():
    print("RDA Spatial Localization Viewer")
    print("=" * 50)

    print("\nSelecting cases...")
    selected = select_cases(n_per_subtype=100)

    if not selected:
        print("No cases found!")
        return

    print(f"\nProcessing {len(selected)} cases...")
    cases_data = []
    n_skipped = 0

    for i, case_info in enumerate(selected):
        result = build_case_data(case_info)
        if result is not None:
            cases_data.append(result)
        else:
            n_skipped += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(selected):
            print(f"  {i + 1}/{len(selected)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"\nTotal cases built: {len(cases_data)}")

    if not cases_data:
        print("No cases could be built!")
        return

    output_path = OUT_DIR / 'rda_topo_viewer.html'
    generate_html(cases_data, output_path)

    print(f"Opening viewer...")
    webbrowser.open(f'file://{output_path}')


if __name__ == '__main__':
    main()
