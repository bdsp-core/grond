"""
Generate annotation package for Sahar: unified manifest + images + viewer.
All 336 patients that MW has labeled (202 canonical + 134 round4).

Must run with: conda run -n foe python code/generate_sahar_package.py
"""

import sys, os, json, base64, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
from mne.filter import notch_filter, filter_data
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from generate_test_images import draw_figure
from browse_results import BIPOLAR_CHANNELS

BASE = CODE_DIR.parent
DATA = BASE / 'data'
OUT_DIR = BASE / 'annotation_package_sahar'
IMG_DIR = OUT_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Existing image directories (priority order)
IMAGE_DIRS = [
    DATA / '_archive' / 'annotation_round4' / 'images',
    DATA / '_archive' / 'annotation_round3' / 'images',
    DATA / '_archive' / 'annotation_round2' / 'images',
    DATA / '_archive' / 'annotation_candidates' / 'images',
]


def find_existing_image(file_name):
    """Find an existing PNG image for the given file_name."""
    for d in IMAGE_DIRS:
        p = d / f"{file_name}.png"
        if p.exists():
            return p
    return None


def build_manifest():
    """Build unified manifest of all patients MW has labeled."""
    rows = []

    # --- Canonical dataset (202 non-excluded patients) ---
    labels = pd.read_csv(DATA / '_archive' / 'canonical_dataset' / 'labels.csv')
    canonical = labels[labels.excluded == False].copy()
    canonical['file_name'] = canonical['mat_name'].str.replace('.mat', '', regex=False)

    for _, row in canonical.iterrows():
        source = row['source']
        if source == 'original_dataset':
            source_round = 'original'
        elif source == 'external_drive_round1':
            source_round = 'round1'
        elif source == 'external_drive_round2':
            source_round = 'round2'
        elif source == 'external_drive_round3':
            source_round = 'round3'
        else:
            source_round = source

        rows.append({
            'patient_id': str(row['patient_id']),
            'file_name': row['file_name'],
            'subtype': row['subtype'],
            'source_round': source_round,
        })

    # --- Round 4 (134 patients, all of them including skips) ---
    r4_annot = pd.read_csv(DATA / '_archive' / 'annotation_round4' / 'frequency_annotations_round4.csv')
    for _, row in r4_annot.iterrows():
        rows.append({
            'patient_id': str(row['patient_id']),
            'file_name': row['file_name'],
            'subtype': row['subtype'],
            'source_round': 'round4',
        })

    manifest = pd.DataFrame(rows)
    print(f"Unified manifest: {len(manifest)} patients")
    print(f"  By source: {manifest.source_round.value_counts().to_dict()}")
    print(f"  By subtype: {manifest.subtype.value_counts().to_dict()}")
    return manifest


def generate_image_from_segments(file_name, seg_bi, fs, subtype, patient_id):
    """Generate a PNG image from a bipolar EEG segment array."""
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T

    # Mock result row (no detector scores — just display EEG)
    result_row = {
        'files': file_name,
        'type_event': np.nan,
        'event_frequency': np.nan,
        'acf_frequency': np.nan,
        'spatial_extent': np.nan,
        'laterality_index': np.nan,
        'left_mean_score': np.nan,
        'right_mean_score': np.nan,
    }
    for ch in BIPOLAR_CHANNELS:
        result_row[f'score_{ch}'] = 2.0  # Mark all active for display
        result_row[f'freq_{ch}'] = np.nan

    fig = draw_figure(result_row, seg_bi, fs, subtype,
                      title_extra=f'Patient {patient_id}')
    png_path = IMG_DIR / f"{file_name}.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return png_path


def generate_missing_images(manifest):
    """Generate images for patients that don't have existing ones."""
    # Load canonical segments
    labels = pd.read_csv(DATA / '_archive' / 'canonical_dataset' / 'labels.csv')
    segments = np.load(DATA / '_archive' / 'canonical_dataset' / 'segments.npy', allow_pickle=True)
    # Build index: mat_name (without .mat) -> row index in segments.npy
    seg_index = {}
    for i, row in labels.iterrows():
        fn = row['mat_name'].replace('.mat', '')
        seg_index[fn] = i

    generated = 0
    failed = 0
    copied = 0
    already_in_out = 0

    for _, row in manifest.iterrows():
        fn = row['file_name']
        out_path = IMG_DIR / f"{fn}.png"

        # Already in output dir
        if out_path.exists():
            already_in_out += 1
            continue

        # Check existing image dirs
        existing = find_existing_image(fn)
        if existing:
            shutil.copy2(str(existing), str(out_path))
            copied += 1
            continue

        # Need to generate from EEG data
        if fn in seg_index:
            idx = seg_index[fn]
            seg_bi = segments[idx]  # (18, 2000)
            fs = 200  # canonical dataset is 200 Hz
            try:
                generate_image_from_segments(fn, seg_bi, fs,
                                             row['subtype'], row['patient_id'])
                generated += 1
                print(f"  Generated: {fn}")
            except Exception as e:
                print(f"  FAILED: {fn}: {e}")
                failed += 1
        else:
            # Try loading from .mat files in dataset_eeg
            subtype_dir = DATA / '_archive' / 'dataset_eeg' / row['subtype']
            mat_path = subtype_dir / f"{fn}.mat"
            if mat_path.exists():
                try:
                    mat = scipy.io.loadmat(str(mat_path))
                    data = mat['data']
                    fs = int(mat['Fs'].ravel()[0])
                    generate_image_from_segments(fn, data, fs,
                                                 row['subtype'], row['patient_id'])
                    generated += 1
                    print(f"  Generated (mat): {fn}")
                except Exception as e:
                    print(f"  FAILED (mat): {fn}: {e}")
                    failed += 1
            else:
                print(f"  NO DATA: {fn}")
                failed += 1

    print(f"\nImage summary:")
    print(f"  Already existed in output: {already_in_out}")
    print(f"  Copied from other rounds: {copied}")
    print(f"  Newly generated: {generated}")
    print(f"  Failed: {failed}")

    return failed


def build_viewer(manifest):
    """Build the annotation viewer HTML with inlined base64 images."""
    n_total = len(manifest)
    n_gpd = (manifest['subtype'] == 'gpd').sum()
    n_lpd = (manifest['subtype'] == 'lpd').sum()
    print(f"\nBuilding viewer for {n_total} items (GPD: {n_gpd}, LPD: {n_lpd})...")

    # Shuffle manifest so cases are randomized (avoids ordering bias)
    manifest_shuffled = manifest.sample(frac=1, random_state=42).reset_index(drop=True)

    manifest_json = manifest_shuffled[['patient_id', 'file_name', 'subtype', 'source_round']].to_dict('records')

    # Inline images as base64
    image_data = {}
    missing_imgs = 0
    for _, row in manifest_shuffled.iterrows():
        fn = row['file_name']
        img_path = IMG_DIR / f"{fn}.png"
        if img_path.exists():
            with open(img_path, 'rb') as f:
                image_data[fn] = base64.b64encode(f.read()).decode('ascii')
        else:
            missing_imgs += 1
            print(f"  WARNING: Image not found: {img_path}")

    print(f"  Inlined {len(image_data)} images ({missing_imgs} missing)")

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
  <span class="info-item">File: <strong id="file-name">--</strong></span>
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
  annotations = JSON.parse(localStorage.getItem('freq_annotations_sahar') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('freq_annotations_sahar', JSON.stringify(annotations));
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
    if (statusFilter === 'unannotated' && annotations[m.file_name]) return false;
    if (statusFilter === 'annotated' && !annotations[m.file_name]) return false;
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
    document.getElementById('file-name').textContent = '--';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image from inlined data
  const b64 = IMAGE_DATA[item.file_name];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/png;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
    document.getElementById('viewer').alt = 'Image not found: ' + item.file_name;
  }}

  // Info panel
  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('file-name').textContent = item.file_name;

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Highlight current annotation
  const currentAnno = annotations[item.file_name];
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
  const nAnnotated = MANIFEST.filter(m => annotations[m.file_name]).length;
  const pct = total > 0 ? (nAnnotated / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnnotated + '/' + total + ' annotated';
}}

function annotate(value) {{
  if (filteredItems.length === 0 || !value) return;
  const item = filteredItems[idx];
  annotations[item.file_name] = value;
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
  const headers = ['patient_id', 'file_name', 'subtype', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.file_name] || '';
    rows.push([
      item.patient_id, item.file_name, item.subtype, anno
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
    return output_path


def write_readme():
    """Write instructions for the annotator."""
    readme = """PD Frequency Annotation Package
================================

What is this?
-------------
This package contains EEG recordings of periodic discharges (PD) from 336
patients. Your task is to estimate the FREQUENCY of the periodic discharges
in each recording, measured in Hz (discharges per second).

How to open
-----------
1. Open "annotation_viewer.html" in Chrome or Safari (Firefox works too).
2. No internet connection or installation is needed — everything is self-contained.

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
- Frequency = (number of peaks - 1) / time window in seconds.
- Example: 11 peaks in 10 seconds = (11-1)/10 = 1.0 Hz.
- The built-in calculator does this math for you.

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
- You can close and reopen the viewer — your progress will be preserved.
- When done, click "Export CSV" (or press E) to download your annotations.
- Send the exported CSV file back when complete.

Expected time
-------------
- About 1-2 minutes per case on average.
- Total: approximately 6-10 hours for all 336 cases.
- You can do this in multiple sessions — progress is saved automatically.

Thank you!
"""
    readme_path = OUT_DIR / 'README.txt'
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"  Saved README.txt")
    return readme_path


def main():
    print("=" * 60)
    print("Building annotation package for Sahar")
    print("=" * 60)

    # Step 1: Build manifest
    print("\n--- Step 1: Building unified manifest ---")
    manifest = build_manifest()
    manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f"  Saved manifest.csv ({len(manifest)} rows)")

    # Step 2 & 3: Generate/copy images
    print("\n--- Step 2-3: Generating/copying images ---")
    n_failed = generate_missing_images(manifest)

    # Step 4: Build viewer
    print("\n--- Step 4: Building annotation viewer ---")
    viewer_path = build_viewer(manifest)

    # Step 5: Write instructions
    print("\n--- Step 5: Writing instructions ---")
    write_readme()

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Total patients: {len(manifest)}")
    print(f"  Images: {len(list(IMG_DIR.glob('*.png')))}")
    print(f"  Failed: {n_failed}")
    print(f"  Viewer: {viewer_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
