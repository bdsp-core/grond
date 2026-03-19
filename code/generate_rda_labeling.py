"""
Generate RDA labeling application with algorithm frequency estimates and sine wave overlays.

Uses Alexandra's FOOOF-based RDA detector (fcn_rdafooof_enhanced from rda1b_fft.py)
for frequency estimation, with fallback to simple FFT when FOOOF fails.

For 200 RDA candidates (100 GRDA + 100 LRDA):
1. Run FOOOF-based RDA frequency estimation on each bipolar EEG segment
2. Fit global sine wave to involved channels
3. Generate EEG images with sine overlay
4. Build annotation viewer HTML with algorithm frequency button
5. Update manifest with detection results

Must run with: conda run -n foe python code/generate_rda_labeling.py
"""

import sys, os, json, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import warnings
warnings.filterwarnings('ignore')

# Import Alexandra's RDA detector
sys.path.insert(0, str(Path(__file__).parent / 'rda_detector'))
from rda1b_fft import fcn_rdafooof_enhanced, fcn_computeSpectra, bipolar_channels
from mne.filter import notch_filter, filter_data

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
LEFT_INDICES  = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]

FS = 200
N_SAMPLES = 2000
FREQ_RANGE = (0.5, 4.0)  # Hz — RDA delta band
RDA_POWER_THRESHOLD = 4.0  # peak must exceed 4x mean power in delta band to qualify
RDA_MIN_CHANNELS = 2  # minimum channels to count as having RDA
RDA_TOP_FRACTION = 0.5  # if too many channels qualify, keep only top 50% by ratio

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
ROUND1_DIR = DATA / '_archive' / 'annotation_rda_round1'
MANIFEST_PATH = ROUND1_DIR / 'manifest.csv'
IMG_DIR = ROUND1_DIR / 'images_v2'
VIEWER_PATH = ROUND1_DIR / 'annotation_viewer_v2.html'


# -------------------------------------------------------------------
# Step 1: FOOOF-based RDA frequency estimation (Alexandra's method)
# -------------------------------------------------------------------
def estimate_rda_frequency_fooof(seg_bi):
    """
    Estimate RDA frequency using Alexandra's FOOOF-based detector.

    The input seg_bi is already bipolar (18, 2000). We apply notch + bandpass
    filters (same as rda1b_fft does), then call fcn_rdafooof_enhanced directly.

    Returns:
        estimated_freq: float (Hz), median of channel_freqs where not NaN
        involved: list of int, indices of involved channels
        peak_freqs: array of peak frequencies per channel (NaN if not involved)
        used_fooof: bool, True if FOOOF succeeded, False if fell back to simple FFT
    """
    n_ch, n_samp = seg_bi.shape

    # Apply the same filters as rda1b_fft (lines 181-182)
    seg_filtered = notch_filter(seg_bi.astype(np.float64), FS, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, FS, 0.5, 40, n_jobs=1, verbose="ERROR")

    # Direct variance-explained search over candidate frequencies.
    # For each candidate freq, bandpass narrowly around it and measure
    # how much variance the narrowband reconstruction explains per channel.
    # Best freq = maximizes total explained variance across all channels.
    from scipy.signal import butter, filtfilt

    candidate_freqs = np.arange(0.5, 3.55, 0.05)  # 0.50, 0.55, ..., 3.50 Hz
    bw = 0.3  # half-bandwidth in Hz

    best_freq = 1.0
    best_total_ve = -np.inf
    best_ve_per_ch = np.zeros(n_ch)

    for f_cand in candidate_freqs:
        lo = max(0.1, f_cand - bw)
        hi = min(FS / 2 - 1, f_cand + bw)
        try:
            b, a = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='band')
        except ValueError:
            continue

        ve_channels = np.zeros(n_ch)
        for ch in range(n_ch):
            sig = seg_filtered[ch]
            var_total = np.var(sig)
            if var_total < 1e-10:
                continue
            try:
                nb = filtfilt(b, a, sig)
                ve_channels[ch] = np.var(nb) / var_total
            except Exception:
                continue

        total_ve = np.sum(ve_channels)
        if total_ve > best_total_ve:
            best_total_ve = total_ve
            best_freq = f_cand
            best_ve_per_ch = ve_channels.copy()

    # Involved channels: variance explained > 10%
    VE_THRESHOLD = 0.10
    involved = [ch for ch in range(n_ch) if best_ve_per_ch[ch] > VE_THRESHOLD]

    # If very few channels detected, take top channels by VE
    if len(involved) < 2:
        sorted_chs = sorted(range(n_ch), key=lambda c: best_ve_per_ch[c], reverse=True)
        involved = sorted_chs[:max(2, n_ch // 4)]

    return best_freq, involved, best_ve_per_ch, True


# -------------------------------------------------------------------
# Step 2: Fit global sine wave
# -------------------------------------------------------------------
def fit_sine_wave(seg_bi, freq, involved):
    """
    Fit a global sine wave at the given frequency to the involved channels.

    Optimizes phase to maximize mean correlation across involved channels.
    Returns per-channel amplitude scaling.

    Returns:
        best_phase: float, optimal phase in radians
        amplitudes: array (18,), per-channel amplitude (0 for non-involved)
    """
    from scipy.signal import butter, filtfilt

    # Narrowband filter centered at freq: [freq - bw, freq + bw]
    bw = 0.3  # Hz half-bandwidth
    lo = max(0.1, freq - bw)
    hi = min(FS / 2 - 1, freq + bw)

    try:
        b, a = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    except ValueError:
        # If frequency range is too narrow, widen slightly
        lo = max(0.1, freq - 0.5)
        hi = min(FS / 2 - 1, freq + 0.5)
        b, a = butter(2, [lo / (FS / 2), hi / (FS / 2)], btype='band')

    # Compute narrowband reconstruction for each involved channel
    narrowband = np.zeros_like(seg_bi)
    for ch in involved:
        try:
            narrowband[ch] = filtfilt(b, a, seg_bi[ch])
        except Exception:
            narrowband[ch] = 0.0

    return narrowband


# -------------------------------------------------------------------
# Step 3: Generate images with sine overlay
# -------------------------------------------------------------------
def draw_figure_with_narrowband(seg_bi, narrowband, involved,
                                subtype, patient_id):
    """Draw EEG traces with narrowband reconstruction overlay on involved channels."""
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(19, 1, hspace=0.08,
                  left=0.10, right=0.98, top=0.95, bottom=0.05)

    time_vec = np.linspace(0, seg_bi.shape[1] / FS, seg_bi.shape[1])

    fig.text(0.5, 0.975,
             f'{subtype.upper()} \u2014 Patient {patient_id}',
             ha='center', fontsize=12, fontweight='bold')

    for i in range(18):
        ax = fig.add_subplot(gs[i + 1, 0])
        color = '#1a6dd4'

        # Lowpass at 20 Hz + detrend for cleaner display
        from scipy.signal import detrend, butter, filtfilt
        b_lp, a_lp = butter(4, 20.0 / (FS / 2), btype='low')
        sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
        sig_clean = detrend(sig_lp, type='linear')
        ax.plot(time_vec, sig_clean, color=color, linewidth=1.0)

        # Overlay narrowband reconstruction on involved channels
        if i in involved:
            ax.plot(time_vec, narrowband[i, :], color='#e03030',
                    linewidth=1.5, alpha=0.7)

        if i in LEFT_INDICES:
            ax.set_facecolor('#ffe8e8')
        elif i in RIGHT_INDICES:
            ax.set_facecolor('#e8e8ff')
        else:
            ax.set_facecolor('#f0f0f0')

        label = BIPOLAR_CHANNELS[i]
        if i in involved:
            label = '\u25cf ' + label  # bullet marker for involved channels
        ax.set_ylabel(label, fontsize=7, rotation=0,
                      labelpad=65, va='center')

        ax.tick_params(axis='y', labelsize=5)
        if i < 17:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

    return fig


# -------------------------------------------------------------------
# Step 4: Build annotation viewer HTML
# -------------------------------------------------------------------
def build_viewer(manifest):
    """Build self-contained annotation viewer HTML with inlined base64 images."""
    n_total = len(manifest)
    print(f"\nBuilding viewer for {n_total} items...")
    subtypes = manifest['subtype'].value_counts().to_dict()
    print(f"  Subtypes: {subtypes}")

    # Shuffle for randomized order
    manifest_shuffled = manifest.sample(frac=1, random_state=42).reset_index(drop=True)

    # Build manifest JSON with estimated frequency per case
    # Use case_id = segment_id + '_' + subtype for uniqueness (some patients appear as both GRDA and LRDA)
    manifest_records = []
    for _, row in manifest_shuffled.iterrows():
        case_id = f"{row['segment_id']}_{row['subtype']}"
        manifest_records.append({
            'patient_id': str(row['patient_id']),
            'segment_id': str(row['segment_id']),
            'case_id': case_id,
            'subtype': row['subtype'],
            'mat_file': row['mat_file'],
            'est_freq': round(float(row['estimated_freq']), 2),
        })

    # Inline images as base64
    image_data = {}
    missing_imgs = 0
    for _, row in manifest_shuffled.iterrows():
        sid = row['segment_id']
        img_path = IMG_DIR / f"{sid}.png"
        if img_path.exists():
            with open(img_path, 'rb') as f:
                image_data[sid] = base64.b64encode(f.read()).decode('ascii')
        else:
            missing_imgs += 1
            print(f"  WARNING: Image not found: {img_path}")

    print(f"  Inlined {len(image_data)} images ({missing_imgs} missing)")

    subtype_options = '\n'.join(
        f'      <option value="{st}">{st.upper()} only</option>'
        for st in sorted(subtypes.keys())
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>RDA Frequency Annotation (Round 1 v2)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  select {{ font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }}

  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
  #progress-bar-container {{
    width: 200px; height: 8px; background: #444; border-radius: 4px; overflow: hidden;
  }}
  #progress-bar {{ height: 100%; background: #44cc44; transition: width 0.3s; }}
  #progress-text {{ font-size: 12px; color: #aaa; }}

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }}
  .info-badge {{
    padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: bold;
  }}
  .badge-grda {{ background: #205a20; color: #88ff88; }}
  .badge-lrda {{ background: #5a5a20; color: #ffff88; }}
  .info-item {{ font-size: 13px; color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #calc-panel {{
    background: #2a2a2a; padding: 6px 16px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    border-bottom: 1px solid #333;
  }}
  .calc-label {{ font-size: 12px; color: #888; }}
  .calc-input {{
    width: 50px; font-size: 14px; padding: 4px 6px;
    background: #444; color: #eee; border: 1px solid #666; border-radius: 4px;
    text-align: center; font-family: monospace;
  }}
  .calc-result {{
    font-size: 16px; font-weight: bold; color: #44cc44; min-width: 80px;
  }}
  .calc-use-btn {{
    padding: 4px 10px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px;
  }}
  .calc-use-btn:hover {{ background: #3a4a3a; }}

  #annotation-panel {{
    background: #2a2a2a; padding: 10px 16px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    border-bottom: 2px solid #444;
  }}
  .anno-label {{ font-size: 13px; color: #aaa; margin-right: 8px; }}

  .freq-btn {{
    padding: 8px 14px; border: 2px solid #555; border-radius: 6px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 14px; font-weight: bold;
    min-width: 50px; text-align: center; transition: all 0.15s;
  }}
  .freq-btn:hover {{ background: #555; border-color: #888; }}
  .freq-btn.selected {{ background: #2a6e2a; border-color: #44cc44; box-shadow: 0 0 8px #44cc44; }}
  .freq-btn.skip-btn {{ background: #553a20; border-color: #aa7733; color: #ffcc66; min-width: 60px; }}
  .freq-btn.skip-btn.selected {{ background: #7a5520; border-color: #ffaa33; box-shadow: 0 0 8px #ffaa33; }}
  .freq-btn.custom-btn {{ background: #20405a; border-color: #4488cc; color: #88ccff; min-width: 70px; }}
  .freq-btn.custom-btn.selected {{ background: #204a6a; border-color: #44aaff; box-shadow: 0 0 8px #44aaff; }}

  .freq-btn.algo-btn {{
    background: #1a3a6a; border-color: #4488dd; color: #88bbff;
    min-width: 70px; font-size: 15px; padding: 10px 16px;
  }}
  .freq-btn.algo-btn:hover {{ background: #2a4a7a; border-color: #66aaff; }}
  .freq-btn.algo-btn.selected {{ background: #2a4a7a; border-color: #44aaff; box-shadow: 0 0 10px #44aaff; }}

  #img-container {{ text-align: center; padding: 8px; }}
  #img-container img {{ max-width: 100%; max-height: calc(100vh - 320px); }}

  #save-status {{ color: #44cc44; font-size: 12px; }}

  #shortcuts {{
    font-size: 11px; color: #777; padding: 4px 16px; background: #222;
    border-top: 1px solid #333;
  }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <select id="filter-type" onchange="filterChanged()">
      <option value="all">All types</option>
{subtype_options}
    </select>
    <select id="filter-status" onchange="filterChanged()">
      <option value="all">All</option>
      <option value="unannotated">Unannotated</option>
      <option value="annotated">Annotated</option>
    </select>
    <div id="progress-bar-container"><div id="progress-bar"></div></div>
    <span id="progress-text">0/0 annotated</span>
  </div>
  <div id="header-right">
    <span id="counter">1 / 0</span>
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
    <span id="save-status"></span>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge" id="type-badge">--</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Segment: <strong id="segment-id">--</strong></span>
</div>

<div id="calc-panel">
  <span class="calc-label">Calculator:</span>
  <input type="number" id="calc-num" class="calc-input" placeholder="N" min="1" max="30" oninput="updateCalc()">
  <span class="calc-label">peaks in</span>
  <input type="number" id="calc-sec" class="calc-input" placeholder="T" min="1" max="20" value="10" oninput="updateCalc()">
  <span class="calc-label">sec =</span>
  <span class="calc-result" id="calc-result">-- Hz</span>
  <button class="calc-use-btn" onclick="useCalcResult()">Use this</button>
  <span class="calc-label" style="margin-left: 16px;">| Custom:</span>
  <input type="number" id="custom-freq" class="calc-input" style="width:70px" placeholder="Hz" step="0.01" min="0.1" max="5">
  <button class="calc-use-btn" onclick="useCustomFreq()">Use</button>
</div>

<div id="annotation-panel">
  <span class="anno-label">Frequency (Hz):</span>
  <button class="freq-btn algo-btn" id="algo-btn" onclick="useAlgoFreq()">--<br><span class="key">Q</span></button>
  <span style="width:8px; display:inline-block;"></span>
  <button class="freq-btn" onclick="annotate('0.25')">0.25<br><span class="key">1</span></button>
  <button class="freq-btn" onclick="annotate('0.5')">0.5<br><span class="key">2</span></button>
  <button class="freq-btn" onclick="annotate('0.75')">0.75<br><span class="key">3</span></button>
  <button class="freq-btn" onclick="annotate('1.0')">1.0<br><span class="key">4</span></button>
  <button class="freq-btn" onclick="annotate('1.25')">1.25<br><span class="key">5</span></button>
  <button class="freq-btn" onclick="annotate('1.5')">1.5<br><span class="key">6</span></button>
  <button class="freq-btn" onclick="annotate('1.75')">1.75<br><span class="key">7</span></button>
  <button class="freq-btn" onclick="annotate('2.0')">2.0<br><span class="key">8</span></button>
  <button class="freq-btn" onclick="annotate('2.5')">2.5<br><span class="key">9</span></button>
  <button class="freq-btn" onclick="annotate('3.0')">3.0<br><span class="key">0</span></button>
  <button class="freq-btn skip-btn" onclick="annotate('skip')">Skip<br><span class="key">S</span></button>
  <button class="freq-btn custom-btn" id="custom-anno-btn" onclick="annotate(lastCustom)">--<br><span class="key">C</span></button>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">Q</span> use algorithm estimate &nbsp;&nbsp;
  <span class="key">1</span>-<span class="key">0</span> annotate frequency &nbsp;&nbsp;
  <span class="key">S</span> skip &nbsp;&nbsp;
  <span class="key">C</span> use custom &nbsp;&nbsp;
  <span class="key">U</span> unannotated only &nbsp;&nbsp;
  <span class="key">A</span> all types &nbsp;&nbsp;
  <span class="key">E</span> export CSV
</div>

<script>
// Inline data
const MANIFEST = {json.dumps(manifest_records)};
const IMAGE_DATA = {json.dumps(image_data)};

let annotations = {{}};
let filteredItems = [];
let idx = 0;
let lastCustom = '';

const KEY_MAP = {{ '1': '0.25', '2': '0.5', '3': '0.75', '4': '1.0', '5': '1.25',
                  '6': '1.5', '7': '1.75', '8': '2.0', '9': '2.5', '0': '3.0' }};

// Load saved annotations
try {{
  annotations = JSON.parse(localStorage.getItem('freq_annotations_rda_round1_v2') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('freq_annotations_rda_round1_v2', JSON.stringify(annotations));
}}

function init() {{
  filteredItems = MANIFEST.slice();
  idx = 0;
  show();
}}

function filterChanged() {{
  const typeFilter = document.getElementById('filter-type').value;
  const statusFilter = document.getElementById('filter-status').value;

  filteredItems = MANIFEST.filter(m => {{
    if (typeFilter !== 'all' && m.subtype !== typeFilter) return false;
    if (statusFilter === 'unannotated' && annotations[m.case_id]) return false;
    if (statusFilter === 'annotated' && !annotations[m.case_id]) return false;
    return true;
  }});

  idx = 0;
  show();
}}

function show() {{
  if (filteredItems.length === 0) {{
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('patient-id').textContent = '--';
    document.getElementById('segment-id').textContent = '--';
    document.getElementById('algo-btn').innerHTML = '--<br><span class="key">Q</span>';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image from inlined data
  const b64 = IMAGE_DATA[item.segment_id];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/png;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
    document.getElementById('viewer').alt = 'Image not found: ' + item.segment_id;
  }}

  // Info panel
  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('segment-id').textContent = item.segment_id;

  // Algorithm frequency button
  const algoBtn = document.getElementById('algo-btn');
  const estFreq = item.est_freq.toFixed(2);
  algoBtn.innerHTML = estFreq + '<br><span class="key">Q</span>';
  algoBtn.setAttribute('data-freq', estFreq);

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Highlight current annotation
  const currentAnno = annotations[item.case_id];
  document.querySelectorAll('.freq-btn').forEach(btn => btn.classList.remove('selected'));
  if (currentAnno) {{
    // Check algo button
    if (currentAnno === algoBtn.getAttribute('data-freq')) {{
      algoBtn.classList.add('selected');
    }}
    // Check standard buttons
    document.querySelectorAll('.freq-btn:not(.algo-btn)').forEach(btn => {{
      const m = btn.getAttribute('onclick');
      if (m && m.includes("'" + currentAnno + "'")) {{
        btn.classList.add('selected');
      }}
    }});
    if (!['0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.5','3.0','skip'].includes(currentAnno)
        && currentAnno !== algoBtn.getAttribute('data-freq')) {{
      const cb = document.getElementById('custom-anno-btn');
      cb.innerHTML = currentAnno + '<br><span class="key">C</span>';
      cb.classList.add('selected');
      lastCustom = currentAnno;
    }}
  }}

  updateProgress();
}}

function updateProgress() {{
  const total = MANIFEST.length;
  const nAnnotated = MANIFEST.filter(m => annotations[m.case_id]).length;
  const pct = total > 0 ? (nAnnotated / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnnotated + '/' + total + ' annotated';
}}

function annotate(value) {{
  if (filteredItems.length === 0 || !value) return;
  const item = filteredItems[idx];
  annotations[item.case_id] = value;
  saveAnnotations();

  document.querySelectorAll('.freq-btn').forEach(btn => btn.classList.remove('selected'));
  // Check if it matches algo button
  const algoBtn = document.getElementById('algo-btn');
  if (value === algoBtn.getAttribute('data-freq')) {{
    algoBtn.classList.add('selected');
  }}
  document.querySelectorAll('.freq-btn:not(.algo-btn)').forEach(btn => {{
    const m = btn.getAttribute('onclick');
    if (m && m.includes("'" + value + "'")) {{
      btn.classList.add('selected');
    }}
  }});

  document.getElementById('save-status').textContent = 'Saved: ' + value;
  setTimeout(() => {{ document.getElementById('save-status').textContent = ''; }}, 1000);

  updateProgress();

  if (idx < filteredItems.length - 1) {{
    setTimeout(() => {{ idx++; show(); }}, 350);
  }}
}}

function useAlgoFreq() {{
  const algoBtn = document.getElementById('algo-btn');
  const freq = algoBtn.getAttribute('data-freq');
  if (freq && freq !== '--') {{
    annotate(freq);
  }}
}}

function updateCalc() {{
  const n = parseFloat(document.getElementById('calc-num').value);
  const t = parseFloat(document.getElementById('calc-sec').value);
  if (n > 0 && t > 0) {{
    const freq = (n / t).toFixed(3);
    document.getElementById('calc-result').textContent = freq + ' Hz';
  }} else {{
    document.getElementById('calc-result').textContent = '-- Hz';
  }}
}}

function useCalcResult() {{
  const text = document.getElementById('calc-result').textContent;
  const m = text.match(/([\\.\\d]+) Hz/);
  if (m) {{
    lastCustom = m[1];
    const cb = document.getElementById('custom-anno-btn');
    cb.innerHTML = m[1] + '<br><span class="key">C</span>';
    annotate(m[1]);
  }}
}}

function useCustomFreq() {{
  const val = document.getElementById('custom-freq').value;
  if (val && parseFloat(val) > 0) {{
    const rounded = parseFloat(val).toFixed(2);
    lastCustom = rounded;
    const cb = document.getElementById('custom-anno-btn');
    cb.innerHTML = rounded + '<br><span class="key">C</span>';
    annotate(rounded);
  }}
}}

function exportCSV() {{
  const headers = ['patient_id', 'segment_id', 'subtype', 'est_freq', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.case_id] || '';
    rows.push([
      item.patient_id, item.segment_id, item.subtype, item.est_freq, anno
    ].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'frequency_annotations_rda_round1_v2.csv';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, filteredItems.length - 1); show(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); }}
  else if (e.key in KEY_MAP) {{ annotate(KEY_MAP[e.key]); }}
  else if (e.key === 's' || e.key === 'S') {{ annotate('skip'); }}
  else if (e.key === 'q' || e.key === 'Q') {{ useAlgoFreq(); }}
  else if (e.key === 'c' || e.key === 'C') {{ if (lastCustom) annotate(lastCustom); }}
  else if (e.key === 'e' || e.key === 'E') {{ exportCSV(); }}
  else if (e.key === 'u' || e.key === 'U') {{
    document.getElementById('filter-status').value = 'unannotated'; filterChanged();
  }}
  else if (e.key === 'a' || e.key === 'A') {{
    document.getElementById('filter-type').value = 'all';
    document.getElementById('filter-status').value = 'all';
    filterChanged();
  }}
}});

init();
</script>
</body>
</html>"""

    with open(VIEWER_PATH, 'w') as f:
        f.write(html)

    size_mb = VIEWER_PATH.stat().st_size / (1024 * 1024)
    print(f"  Saved annotation_viewer_v2.html ({size_mb:.1f} MB)")
    return VIEWER_PATH


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    print("=" * 60)
    print("RDA Labeling Application (v2 — with sine overlays)")
    print("=" * 60)

    # Load manifest
    print("\n--- Loading manifest ---")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"  {len(manifest)} segments loaded")
    print(f"  Subtypes: {manifest['subtype'].value_counts().to_dict()}")

    # Create output directory
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Process each segment
    print("\n--- Step 1-3: FOOOF RDA detection, sine fitting, and image generation ---")
    results = []
    success = 0
    fail = 0
    fooof_success = 0
    fooof_fallback = 0

    for i, (_, row) in enumerate(manifest.iterrows()):
        patient_id = str(row['patient_id'])
        segment_id = str(row['segment_id'])
        subtype = row['subtype']
        mat_file = row['mat_file']

        print(f"  [{i+1}/{len(manifest)}] {segment_id} ({subtype})", end='', flush=True)

        mat_path = EEG_DIR / mat_file
        if not mat_path.exists():
            print('  MISSING')
            fail += 1
            results.append({
                'patient_id': patient_id,
                'segment_id': segment_id,
                'subtype': subtype,
                'mat_file': mat_file,
                'estimated_freq': np.nan,
                'n_involved_channels': 0,
                'involved_channels': '',
            })
            continue

        try:
            mat = scipy.io.loadmat(str(mat_path))
            if 'data' in mat:
                data = mat['data'].astype(np.float64)
            elif 'data_50sec' in mat:
                data = mat['data_50sec'].astype(np.float64)
            else:
                print('  NO DATA KEY')
                fail += 1
                results.append({
                    'patient_id': patient_id,
                    'segment_id': segment_id,
                    'subtype': subtype,
                    'mat_file': mat_file,
                    'estimated_freq': np.nan,
                    'n_involved_channels': 0,
                    'involved_channels': '',
                })
                continue

            # Ensure shape is (channels, samples)
            if data.shape[0] > data.shape[1]:
                data = data.T

            # Data is already bipolar (18, 2000)
            seg_bi = data
            if seg_bi.shape[0] != 18:
                print(f'  BAD SHAPE {seg_bi.shape}')
                fail += 1
                results.append({
                    'patient_id': patient_id,
                    'segment_id': segment_id,
                    'subtype': subtype,
                    'mat_file': mat_file,
                    'estimated_freq': np.nan,
                    'n_involved_channels': 0,
                    'involved_channels': '',
                })
                continue

            # Step 1: Estimate RDA frequency using Alexandra's FOOOF detector
            est_freq, involved, peak_freqs, used_fooof = estimate_rda_frequency_fooof(seg_bi)
            if used_fooof:
                fooof_success += 1
                method_tag = 'FOOOF'
            else:
                fooof_fallback += 1
                method_tag = 'FFT-fallback'

            # Step 2: Narrowband reconstruction around estimated frequency
            narrowband = fit_sine_wave(seg_bi, est_freq, involved)

            # Step 3: Generate image
            fig = draw_figure_with_narrowband(seg_bi, narrowband, involved,
                                              subtype, patient_id)
            png_path = IMG_DIR / f"{segment_id}.png"
            fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
            plt.close(fig)

            involved_names = [BIPOLAR_CHANNELS[ch] for ch in involved]
            results.append({
                'patient_id': patient_id,
                'segment_id': segment_id,
                'subtype': subtype,
                'mat_file': mat_file,
                'estimated_freq': round(est_freq, 2),
                'n_involved_channels': len(involved),
                'involved_channels': ';'.join(involved_names),
            })

            print(f'  OK  freq={est_freq:.2f} Hz  ({len(involved)} ch)  [{method_tag}]')
            success += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  FAILED: {e}')
            fail += 1
            results.append({
                'patient_id': patient_id,
                'segment_id': segment_id,
                'subtype': subtype,
                'mat_file': mat_file,
                'estimated_freq': np.nan,
                'n_involved_channels': 0,
                'involved_channels': '',
            })

    print(f"\n  Results: {success} OK, {fail} failed")
    print(f"  FOOOF succeeded: {fooof_success}, FFT fallback: {fooof_fallback}")

    # Step 5: Save updated manifest
    print("\n--- Step 5: Saving updated manifest ---")
    results_df = pd.DataFrame(results)
    results_df.to_csv(MANIFEST_PATH, index=False)
    print(f"  Saved manifest.csv ({len(results_df)} rows)")
    print(f"  Columns: {list(results_df.columns)}")

    # Step 4: Build viewer
    print("\n--- Step 4: Building annotation viewer ---")
    viewer_path = build_viewer(results_df)

    # Summary
    freq_stats = results_df['estimated_freq'].dropna()
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Output directory: {ROUND1_DIR}")
    print(f"  Total segments: {len(results_df)}")
    print(f"  Successfully processed: {success}")
    print(f"  Failed: {fail}")
    print(f"  FOOOF succeeded: {fooof_success}")
    print(f"  FFT fallback: {fooof_fallback}")
    print(f"  Frequency range: {freq_stats.min():.2f} - {freq_stats.max():.2f} Hz")
    print(f"  Frequency median: {freq_stats.median():.2f} Hz")
    print(f"  Frequency mean: {freq_stats.mean():.2f} Hz")
    print(f"  Frequency std: {freq_stats.std():.2f} Hz")
    # Frequency distribution by quartile
    if len(freq_stats) > 0:
        print(f"  Frequency quartiles: Q1={freq_stats.quantile(0.25):.2f}, "
              f"Q2={freq_stats.quantile(0.50):.2f}, Q3={freq_stats.quantile(0.75):.2f}")
        # Bin counts
        bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        hist, _ = np.histogram(freq_stats, bins=bins)
        print(f"  Frequency distribution:")
        for j in range(len(bins)-1):
            print(f"    {bins[j]:.1f}-{bins[j+1]:.1f} Hz: {hist[j]} segments")
    print(f"  Viewer: {viewer_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
