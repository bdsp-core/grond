"""
Generate annotation package for Sahar: unified manifest + images + viewer.
All 336 patients that MW has labeled (202 canonical + 134 round4).

Improvements over v1:
- JPEG images (quality=70) instead of PNG — much smaller file size
- Lower DPI (100) and smaller figure (12x8) — compact but readable
- Clean EEG only: 20 Hz lowpass filter, linear detrend, no side panels
- Calculator uses n/t (not (n-1)/t)
- No algorithm estimates or MW ratings shown
- Randomized order to avoid ordering bias

Must run with: conda run -n foe python code/generate_sahar_package.py
"""

import sys, os, json, base64, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io
from scipy.signal import detrend, butter, filtfilt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
OUT_DIR = BASE / 'annotation_package_sahar'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]


def get_bipolar(segment):
    """Convert 20-channel monopolar to 18-channel bipolar."""
    bipolar_ids = np.array([
        [MONO_CHANNELS.index(bc.split('-')[0]), MONO_CHANNELS.index(bc.split('-')[1])]
        for bc in BIPOLAR_CHANNELS
    ])
    return segment[bipolar_ids[:, 0]] - segment[bipolar_ids[:, 1]]


def build_manifest():
    """Build unified manifest of all 336 patients MW has labeled."""
    rows = []

    # --- Canonical dataset (202 non-excluded patients) ---
    labels = pd.read_csv(DATA / '_archive' / 'canonical_dataset' / 'labels.csv')
    canonical = labels[labels.excluded == False].copy()

    for _, row in canonical.iterrows():
        rows.append({
            'patient_id': str(row['patient_id']),
            'subtype': row['subtype'],
            'source_round': 'canonical',
        })

    # --- Round 4 (134 patients) ---
    r4_annot = pd.read_csv(DATA / '_archive' / 'pd_round4' / 'frequency_annotations_round4.csv')
    for _, row in r4_annot.iterrows():
        rows.append({
            'patient_id': str(row['patient_id']),
            'subtype': row['subtype'],
            'source_round': 'round4',
        })

    manifest = pd.DataFrame(rows)
    # Create a unique key for each entry (patient_id + subtype for the 6 dual-subtype cases)
    manifest['case_id'] = manifest['patient_id'] + '_' + manifest['subtype']
    print(f"Unified manifest: {len(manifest)} cases ({manifest.case_id.nunique()} unique)")
    print(f"  By source: {manifest.source_round.value_counts().to_dict()}")
    print(f"  By subtype: {manifest.subtype.value_counts().to_dict()}")
    return manifest


def generate_clean_eeg_jpeg(seg_bi, fs, subtype, case_id):
    """Generate a clean EEG JPEG image with 20Hz lowpass + detrend.

    Returns JPEG bytes (not saved to disk).
    """
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T

    # Replace NaN/Inf
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)

    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / fs, n_samples)

    # Apply 20 Hz lowpass filter
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
            except ValueError:
                pass

    # Linear detrend each channel
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    # Create clean figure: 12x8, no side panels
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('white')

    # Compute uniform y-scale across all channels
    all_ranges = []
    for i in range(n_channels):
        ch_range = np.ptp(seg_bi[i, :])
        if ch_range > 0:
            all_ranges.append(ch_range)
    if all_ranges:
        median_range = np.median(all_ranges)
        y_half = max(median_range * 0.75, 10)  # at least 10 uV
    else:
        y_half = 50

    for i in range(n_channels):
        ax = axes[i]
        ch_mean = np.mean(seg_bi[i, :])

        # Color: blue for left, red-blue for right, gray for midline
        if i in LEFT_INDICES:
            color = '#cc3333'
            bg = '#fff5f5'
        elif i in RIGHT_INDICES:
            color = '#3333cc'
            bg = '#f5f5ff'
        else:
            color = '#333333'
            bg = '#f5f5f5'

        ax.set_facecolor(bg)
        ax.plot(time_vec, seg_bi[i, :], color=color, linewidth=0.8)

        # Channel label on the left
        ax.set_ylabel(BIPOLAR_CHANNELS[i], fontsize=7, rotation=0,
                      labelpad=50, va='center')

        ax.set_ylim(ch_mean - y_half, ch_mean + y_half)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3, axis='x')

        for spine in ax.spines.values():
            spine.set_visible(False)

        if i < n_channels - 1:
            ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xlabel('Time (seconds)', fontsize=9)
    axes[-1].tick_params(axis='x', labelsize=7)

    # Title: just subtype and case ID
    fig.suptitle(f'{subtype.upper()} — {case_id}', fontsize=12, fontweight='bold', y=0.98)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.04, hspace=0.05)

    # Save to JPEG bytes in memory
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 70})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_all_images(manifest):
    """Generate JPEG images for all cases, return dict of case_id -> base64 JPEG."""
    image_data = {}
    failed = 0

    for idx, row in manifest.iterrows():
        pid = row['patient_id']
        subtype = row['subtype']
        case_id = row['case_id']

        mat_path = EEG_DIR / f"{pid}_seg000.mat"
        if not mat_path.exists():
            print(f"  NO DATA: {case_id} (missing {mat_path.name})")
            failed += 1
            continue

        try:
            mat = scipy.io.loadmat(str(mat_path))
            data = mat['data'].astype(np.float64)
            fs = int(mat['Fs'].ravel()[0])
            if data.shape[0] > data.shape[1]:
                data = data.T
            # Convert monopolar (20 ch) to bipolar (18 ch)
            if data.shape[0] == 20:
                data = get_bipolar(data)
            elif data.shape[0] != 18:
                print(f"  SKIP: {case_id} has {data.shape[0]} channels (expected 18 or 20)")
                failed += 1
                continue
            # Replace NaN/Inf with 0
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            jpeg_bytes = generate_clean_eeg_jpeg(data, fs, subtype, case_id)
            image_data[case_id] = base64.b64encode(jpeg_bytes).decode('ascii')

            if (idx + 1) % 50 == 0:
                print(f"  Generated {idx + 1}/{len(manifest)} images...")
        except Exception as e:
            print(f"  FAILED: {case_id}: {e}")
            failed += 1

    print(f"\nImage summary: {len(image_data)} generated, {failed} failed")
    return image_data, failed


def build_viewer(manifest, image_data):
    """Build the annotation viewer HTML with inlined base64 JPEG images."""
    n_total = len(manifest)
    n_gpd = (manifest['subtype'] == 'gpd').sum()
    n_lpd = (manifest['subtype'] == 'lpd').sum()
    print(f"\nBuilding viewer for {n_total} items (GPD: {n_gpd}, LPD: {n_lpd})...")

    # Shuffle manifest so cases are randomized (avoids ordering bias)
    manifest_shuffled = manifest.sample(frac=1, random_state=42).reset_index(drop=True)

    manifest_json = manifest_shuffled[['patient_id', 'case_id', 'subtype', 'source_round']].to_dict('records')

    missing_imgs = sum(1 for _, r in manifest_shuffled.iterrows() if r['case_id'] not in image_data)
    print(f"  {len(image_data)} images inlined, {missing_imgs} missing")

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>PD Frequency Annotation - Sahar</title>
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
  .badge-lpd {{ background: #5a2020; color: #ff8888; }}
  .badge-gpd {{ background: #20205a; color: #8888ff; }}
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

  #img-container {{ text-align: center; padding: 8px; }}
  #img-container img {{ max-width: 100%; max-height: calc(100vh - 300px); }}

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
      <option value="gpd">GPD only</option>
      <option value="lpd">LPD only</option>
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
  <span class="info-item">Case: <strong id="case-id">--</strong></span>
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
  <span class="key">1</span>-<span class="key">0</span> annotate frequency &nbsp;&nbsp;
  <span class="key">S</span> skip &nbsp;&nbsp;
  <span class="key">C</span> use custom &nbsp;&nbsp;
  <span class="key">G</span> GPD only &nbsp;&nbsp;
  <span class="key">L</span> LPD only &nbsp;&nbsp;
  <span class="key">A</span> all types &nbsp;&nbsp;
  <span class="key">E</span> export CSV
</div>

<script>
// Inline data
const MANIFEST = {json.dumps(manifest_json)};
const IMAGE_DATA = {json.dumps(image_data)};

let annotations = {{}};
let filteredItems = [];
let idx = 0;
let lastCustom = '';

const KEY_MAP = {{ '1': '0.25', '2': '0.5', '3': '0.75', '4': '1.0', '5': '1.25',
                  '6': '1.5', '7': '1.75', '8': '2.0', '9': '2.5', '0': '3.0' }};

// Load saved annotations
try {{
  annotations = JSON.parse(localStorage.getItem('freq_annotations_sahar_v2') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('freq_annotations_sahar_v2', JSON.stringify(annotations));
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
    document.getElementById('case-id').textContent = '--';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image from inlined data (JPEG)
  const b64 = IMAGE_DATA[item.case_id];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
    document.getElementById('viewer').alt = 'Image not found: ' + item.case_id;
  }}

  // Info panel
  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('case-id').textContent = item.case_id;

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Highlight current annotation
  const currentAnno = annotations[item.case_id];
  document.querySelectorAll('.freq-btn').forEach(btn => btn.classList.remove('selected'));
  if (currentAnno) {{
    document.querySelectorAll('.freq-btn').forEach(btn => {{
      const m = btn.getAttribute('onclick');
      if (m && m.includes("'" + currentAnno + "'")) {{
        btn.classList.add('selected');
      }}
    }});
    if (!['0.25','0.5','0.75','1.0','1.25','1.5','1.75','2.0','2.5','3.0','skip'].includes(currentAnno)) {{
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
  document.querySelectorAll('.freq-btn').forEach(btn => {{
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
  const headers = ['patient_id', 'case_id', 'subtype', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.case_id] || '';
    rows.push([
      item.patient_id, item.case_id, item.subtype, anno
    ].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'frequency_annotations_sahar.csv';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, filteredItems.length - 1); show(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); }}
  else if (e.key in KEY_MAP) {{ annotate(KEY_MAP[e.key]); }}
  else if (e.key === 's' || e.key === 'S') {{ annotate('skip'); }}
  else if (e.key === 'c' || e.key === 'C') {{ if (lastCustom) annotate(lastCustom); }}
  else if (e.key === 'e' || e.key === 'E') {{ exportCSV(); }}
  else if (e.key === 'g' || e.key === 'G') {{
    document.getElementById('filter-type').value = 'gpd'; filterChanged();
  }}
  else if (e.key === 'l' || e.key === 'L') {{
    document.getElementById('filter-type').value = 'lpd'; filterChanged();
  }}
  else if (e.key === 'a' || e.key === 'A') {{
    document.getElementById('filter-type').value = 'all'; filterChanged();
  }}
}});

init();
</script>
</body>
</html>"""

    output_path = OUT_DIR / 'annotation_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved annotation_viewer.html ({size_mb:.1f} MB)")
    return output_path, size_mb


def write_readme():
    """Write instructions for the annotator."""
    readme = """PD Frequency Annotation Package
================================

What is this?
-------------
This package contains EEG recordings of periodic discharges (PD) from 336
cases. Your task is to estimate the FREQUENCY of the periodic discharges
in each recording, measured in Hz (discharges per second).

How to open
-----------
1. Open "annotation_viewer.html" in Chrome or Safari (Firefox works too).
2. No internet connection or installation is needed -- everything is self-contained.

How to use
----------
- Each case shows a 10-second EEG recording in standard bipolar montage.
- The subtype (LPD = lateralized, GPD = generalized) is shown at the top left.
- Choose the frequency that best matches the discharge repetition rate:
    - Click one of the frequency buttons (0.25 to 3.0 Hz)
    - OR use keyboard shortcuts: keys 1-0 map to 0.25-3.0 Hz
    - OR use the calculator: count the number of peaks in a time window
    - OR type a custom frequency value
- Press S (or click Skip) if no clear periodic discharges are visible.
- Use arrow keys (left/right) to navigate between cases.
- Filter by subtype: press G (GPD only), L (LPD only), A (all).
- Filter by status: use the dropdown to show only unannotated cases.

What is frequency?
------------------
Frequency = number of discharges per second (Hz).
- Count the number of discharge peaks visible in a time window.
- Frequency = number of peaks / time window in seconds.
- Example: 10 peaks in 10 seconds = 10/10 = 1.0 Hz.
- The built-in calculator does this math for you (N peaks / T seconds).

When to skip
------------
Skip a case if:
- No clear periodic pattern is visible
- The recording is too noisy to determine a frequency
- You are genuinely unsure

Important notes
---------------
- Rate each case independently. Do not look at previous ratings.
- Your annotations are automatically saved in your browser (localStorage).
- You can close and reopen the viewer -- your progress will be preserved.
- When done, click "Export CSV" (or press E) to download your annotations.
- Send the exported CSV file back when complete.

Expected time
-------------
- About 1-2 minutes per case on average.
- Total: approximately 6-10 hours for all 336 cases.
- You can do this in multiple sessions -- progress is saved automatically.

Thank you!
"""
    readme_path = OUT_DIR / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"  Saved README.txt")
    return readme_path


def main():
    print("=" * 60)
    print("Building annotation package for Sahar (v2 - compact)")
    print("=" * 60)

    # Step 1: Build manifest
    print("\n--- Step 1: Building unified manifest ---")
    manifest = build_manifest()
    manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f"  Saved manifest.csv ({len(manifest)} rows)")

    # Step 2: Generate all images as JPEG (in memory)
    print("\n--- Step 2: Generating clean EEG images (JPEG, 100 DPI) ---")
    image_data, n_failed = generate_all_images(manifest)

    # Step 3: Build viewer
    print("\n--- Step 3: Building annotation viewer ---")
    viewer_path, size_mb = build_viewer(manifest, image_data)

    # Step 4: Write instructions
    print("\n--- Step 4: Writing instructions ---")
    write_readme()

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Total cases: {len(manifest)}")
    print(f"  Images generated: {len(image_data)}")
    print(f"  Failed: {n_failed}")
    print(f"  HTML viewer size: {size_mb:.1f} MB")
    print(f"  Viewer: {viewer_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
