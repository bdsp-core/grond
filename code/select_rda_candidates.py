"""
Select 100 GRDA and 100 LRDA patients from the external drive, stratified by
estimated frequency, for annotation round 1.

Run: conda run -n foe python code/select_rda_candidates.py
"""

import sys, os, ast, json, base64, warnings, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
import scipy.io
import hdf5storage
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
LABELS_DIR = DATA / 'labels'
SEGMENTS_CSV = LABELS_DIR / 'segments.csv'
PATIENTS_CSV = LABELS_DIR / 'patients.csv'
OUT_DIR = DATA / '_archive' / 'rda_round1'
IMG_DIR = OUT_DIR / 'images'

EXTERNAL_SEG_DIR = Path('/Volumes/sanD_photos/IIIC/segments_raw')
EVENT_CSV = DATA / 'list_events_20241129.csv'

FS = 200
N_TARGET = 100  # per subtype

# ── Channel definitions ──────────────────────────────────────────────────────
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']
LEFT_INDICES  = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]

BIPOLAR_IDS = np.array([
    [MONO_CHANNELS.index(bc.split('-')[0]), MONO_CHANNELS.index(bc.split('-')[1])]
    for bc in BIPOLAR_CHANNELS
])

FREQ_BINS = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.5)]
PER_BIN = 20


def get_bipolar(mono):
    """Convert monopolar (>=19, N) to bipolar (18, N)."""
    return mono[BIPOLAR_IDS[:, 0]] - mono[BIPOLAR_IDS[:, 1]]


def load_mat_segment(mat_path):
    """Load a .mat file from external drive, return monopolar (20, 2000) at 200 Hz."""
    try:
        mat = hdf5storage.loadmat(str(mat_path))
    except Exception:
        mat = scipy.io.loadmat(str(mat_path))

    if 'data' in mat:
        data = mat['data']
    elif 'data_50sec' in mat:
        data = mat['data_50sec']
    else:
        raise ValueError(f"No data field in {mat_path}")

    data = data.astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T

    # Trim/pad to 20 channels × 2000 samples
    if data.shape[0] < 19:
        raise ValueError(f"Only {data.shape[0]} channels")
    if data.shape[0] == 19:
        data = np.vstack([data, np.zeros((1, data.shape[1]))])
    if data.shape[0] > 20:
        data = data[:20]

    if data.shape[1] < 2000:
        data = np.pad(data, ((0,0), (0, 2000 - data.shape[1])))
    elif data.shape[1] > 2000:
        # Take center 2000 samples
        start = (data.shape[1] - 2000) // 2
        data = data[:, start:start+2000]

    return data


def estimate_frequency_simple(bipolar_data, fs=200):
    """
    Simple delta-band peak frequency estimation via FFT.
    Compute mean |signal| across channels, find peak in 0.5–3.5 Hz.
    """
    # Average power across channels
    mean_signal = np.mean(np.abs(bipolar_data), axis=0)
    # Detrend
    mean_signal = mean_signal - np.mean(mean_signal)

    N = len(mean_signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    fft_vals = np.abs(np.fft.fft(mean_signal))

    # Only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]

    # Delta band mask: 0.5 – 3.5 Hz
    delta_mask = (freqs >= 0.5) & (freqs <= 3.5)
    if not np.any(delta_mask):
        return np.nan

    delta_freqs = freqs[delta_mask]
    delta_power = fft_vals[delta_mask]

    peak_idx = np.argmax(delta_power)
    return float(delta_freqs[peak_idx])


def draw_figure_clean(seg_bi, fs, subtype, title_extra=''):
    """Draw EEG traces only — no side panels."""
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(19, 1, hspace=0.08,
                  left=0.10, right=0.98, top=0.95, bottom=0.05)

    time_vec = np.linspace(0, seg_bi.shape[1] / fs, seg_bi.shape[1])

    fig.text(0.5, 0.975,
             f'{subtype.upper()} — {title_extra}',
             ha='center', fontsize=12, fontweight='bold')

    for i in range(18):
        ax = fig.add_subplot(gs[i + 1, 0])
        ax.plot(time_vec, seg_bi[i, :], color='#1a6dd4', linewidth=1.0)

        if i in LEFT_INDICES:
            ax.set_facecolor('#ffe8e8')
        elif i in RIGHT_INDICES:
            ax.set_facecolor('#e8e8ff')
        else:
            ax.set_facecolor('#f0f0f0')

        ax.set_ylabel(BIPOLAR_CHANNELS[i], fontsize=7, rotation=0,
                      labelpad=65, va='center')
        ax.tick_params(axis='y', labelsize=5)
        if i < 17:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

    return fig


def build_viewer(manifest, img_dir, out_dir, storage_key='freq_annotations_rda_round1'):
    """Build self-contained annotation viewer HTML with inlined base64 images."""
    n_total = len(manifest)
    print(f"\nBuilding viewer for {n_total} items...")

    # Shuffle for randomized order
    manifest_shuffled = manifest.sample(frac=1, random_state=42).reset_index(drop=True)
    manifest_json = manifest_shuffled[['patient_id', 'segment_id', 'subtype', 'mat_file']].to_dict('records')

    # Inline images as base64
    image_data = {}
    missing_imgs = 0
    for _, row in manifest_shuffled.iterrows():
        sid = row['segment_id']
        img_path = img_dir / f"{sid}.png"
        if img_path.exists():
            with open(img_path, 'rb') as f:
                image_data[sid] = base64.b64encode(f.read()).decode('ascii')
        else:
            missing_imgs += 1

    print(f"  Inlined {len(image_data)} images ({missing_imgs} missing)")

    subtype_options = '\n'.join(
        f'      <option value="{st}">{st.upper()} only</option>'
        for st in sorted(manifest['subtype'].unique())
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>RDA Round 1 Frequency Annotation</title>
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
  <span class="key">U</span> unannotated only &nbsp;&nbsp;
  <span class="key">A</span> all types &nbsp;&nbsp;
  <span class="key">E</span> export CSV
</div>

<script>
const MANIFEST = {json.dumps(manifest_json)};
const IMAGE_DATA = {json.dumps(image_data)};

let annotations = {{}};
let filteredItems = [];
let idx = 0;
let lastCustom = '';

const KEY_MAP = {{ '1': '0.25', '2': '0.5', '3': '0.75', '4': '1.0', '5': '1.25',
                  '6': '1.5', '7': '1.75', '8': '2.0', '9': '2.5', '0': '3.0' }};

try {{
  annotations = JSON.parse(localStorage.getItem('{storage_key}') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('{storage_key}', JSON.stringify(annotations));
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
    if (statusFilter === 'unannotated' && annotations[m.segment_id]) return false;
    if (statusFilter === 'annotated' && !annotations[m.segment_id]) return false;
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
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];
  const b64 = IMAGE_DATA[item.segment_id];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/png;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
    document.getElementById('viewer').alt = 'Image not found: ' + item.segment_id;
  }}
  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('segment-id').textContent = item.segment_id;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  const currentAnno = annotations[item.segment_id];
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
  const nAnnotated = MANIFEST.filter(m => annotations[m.segment_id]).length;
  const pct = total > 0 ? (nAnnotated / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnnotated + '/' + total + ' annotated';
}}

function annotate(value) {{
  if (filteredItems.length === 0 || !value) return;
  const item = filteredItems[idx];
  annotations[item.segment_id] = value;
  saveAnnotations();
  document.querySelectorAll('.freq-btn').forEach(btn => btn.classList.remove('selected'));
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    const m = btn.getAttribute('onclick');
    if (m && m.includes("'" + value + "'")) btn.classList.add('selected');
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
    document.getElementById('calc-result').textContent = (n / t).toFixed(3) + ' Hz';
  }} else {{
    document.getElementById('calc-result').textContent = '-- Hz';
  }}
}}

function useCalcResult() {{
  const text = document.getElementById('calc-result').textContent;
  const m = text.match(/([\\.\\d]+) Hz/);
  if (m) {{
    lastCustom = m[1];
    document.getElementById('custom-anno-btn').innerHTML = m[1] + '<br><span class="key">C</span>';
    annotate(m[1]);
  }}
}}

function useCustomFreq() {{
  const val = document.getElementById('custom-freq').value;
  if (val && parseFloat(val) > 0) {{
    const rounded = parseFloat(val).toFixed(2);
    lastCustom = rounded;
    document.getElementById('custom-anno-btn').innerHTML = rounded + '<br><span class="key">C</span>';
    annotate(rounded);
  }}
}}

function exportCSV() {{
  const headers = ['patient_id', 'segment_id', 'subtype', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.segment_id] || '';
    rows.push([item.patient_id, item.segment_id, item.subtype, anno].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'frequency_annotations_rda_round1.csv';
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

    output_path = out_dir / 'annotation_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved annotation_viewer.html ({size_mb:.1f} MB)")
    return output_path


def main():
    print("=" * 70)
    print("RDA Candidate Selection — 100 GRDA + 100 LRDA, stratified by frequency")
    print("=" * 70)

    # ── Step 1: Load event list, filter to high-certainty RDA ─────────────
    print("\n[1] Loading event list and filtering to high-certainty RDA...")
    df = pd.read_csv(str(EVENT_CSV))
    print(f"  Total rows: {len(df)}")

    df['parsed'] = df['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else None)
    df = df[df['parsed'].notna()].copy()
    df['total'] = df['parsed'].apply(sum)
    df = df[df['total'] > 0].copy()

    df['lrda_votes'] = df['parsed'].apply(lambda x: x[4])
    df['grda_votes'] = df['parsed'].apply(lambda x: x[5])
    df['lrda_frac'] = df['lrda_votes'] / df['total']
    df['grda_frac'] = df['grda_votes'] / df['total']

    df_lrda = df[df['lrda_frac'] >= 0.5].copy()
    df_grda = df[df['grda_frac'] >= 0.5].copy()
    print(f"  High-certainty LRDA events: {len(df_lrda)} ({df_lrda['bdsp_mrn'].nunique()} patients)")
    print(f"  High-certainty GRDA events: {len(df_grda)} ({df_grda['bdsp_mrn'].nunique()} patients)")

    # ── Step 2: Exclude already-annotated patients ────────────────────────
    print("\n[2] Excluding already-annotated patients...")
    existing_patients = pd.read_csv(str(PATIENTS_CSV))
    existing_ids = set(existing_patients['patient_id'].astype(str).values)
    # Also exclude by bdsp_mrn (numeric patient IDs in the event list)
    # The existing patients.csv uses bdsp_mrn as patient_id for external segments
    print(f"  Already annotated: {len(existing_ids)} patients")

    df_lrda = df_lrda[~df_lrda['bdsp_mrn'].astype(str).isin(existing_ids)].copy()
    df_grda = df_grda[~df_grda['bdsp_mrn'].astype(str).isin(existing_ids)].copy()
    print(f"  After exclusion — LRDA: {len(df_lrda)} events ({df_lrda['bdsp_mrn'].nunique()} patients)")
    print(f"  After exclusion — GRDA: {len(df_grda)} events ({df_grda['bdsp_mrn'].nunique()} patients)")

    # ── Step 3: Check .mat file existence ─────────────────────────────────
    print("\n[3] Checking .mat file existence on external drive...")
    if not EXTERNAL_SEG_DIR.exists():
        print(f"  ERROR: External drive not found at {EXTERNAL_SEG_DIR}")
        sys.exit(1)

    available = set(os.path.splitext(f)[0] for f in os.listdir(EXTERNAL_SEG_DIR) if f.endswith('.mat'))
    print(f"  Available .mat files: {len(available)}")

    df_lrda = df_lrda[df_lrda['file_name'].isin(available)].copy()
    df_grda = df_grda[df_grda['file_name'].isin(available)].copy()
    print(f"  With existing .mat — LRDA: {len(df_lrda)} events ({df_lrda['bdsp_mrn'].nunique()} patients)")
    print(f"  With existing .mat — GRDA: {len(df_grda)} events ({df_grda['bdsp_mrn'].nunique()} patients)")

    # ── Step 4: Pick 1 segment per patient (highest certainty) ────────────
    print("\n[4] Picking 1 segment per patient (highest certainty)...")

    def pick_best_per_patient(df_sub, frac_col):
        df_sorted = df_sub.sort_values(frac_col, ascending=False)
        return df_sorted.drop_duplicates(subset='bdsp_mrn', keep='first')

    lrda_best = pick_best_per_patient(df_lrda, 'lrda_frac')
    grda_best = pick_best_per_patient(df_grda, 'grda_frac')
    print(f"  LRDA: {len(lrda_best)} patients")
    print(f"  GRDA: {len(grda_best)} patients")

    # ── Step 5: Estimate frequency ────────────────────────────────────────
    print("\n[5] Estimating RDA frequency for each candidate...")
    print("    (Using simple FFT delta-band peak — fast fallback method)")

    def estimate_frequencies(df_best, label):
        freqs_list = []
        failed = 0
        for _, row in tqdm(df_best.iterrows(), total=len(df_best), desc=f"  {label}"):
            mat_path = EXTERNAL_SEG_DIR / f"{row['file_name']}.mat"
            try:
                mono = load_mat_segment(mat_path)
                bi = get_bipolar(mono)
                freq = estimate_frequency_simple(bi, FS)
                freqs_list.append(freq)
            except Exception:
                freqs_list.append(np.nan)
                failed += 1
        df_best = df_best.copy()
        df_best['est_freq'] = freqs_list
        if failed > 0:
            print(f"    {label}: {failed} failures")
        valid = df_best['est_freq'].notna()
        print(f"    {label}: {valid.sum()} with valid frequency, {(~valid).sum()} NaN")
        return df_best

    lrda_best = estimate_frequencies(lrda_best, 'LRDA')
    grda_best = estimate_frequencies(grda_best, 'GRDA')

    # ── Step 6: Stratified selection ──────────────────────────────────────
    print("\n[6] Stratified selection...")

    def stratified_select(df_sub, n_target, label):
        df_valid = df_sub[df_sub['est_freq'].notna()].copy()

        # Assign frequency bins
        def assign_bin(freq):
            for i, (lo, hi) in enumerate(FREQ_BINS):
                if lo <= freq < hi:
                    return i
            # Frequencies outside range: assign to nearest bin
            if freq < FREQ_BINS[0][0]:
                return 0
            return len(FREQ_BINS) - 1

        df_valid['freq_bin'] = df_valid['est_freq'].apply(assign_bin)

        selected = []
        remaining_quota = 0
        bin_counts = {}

        # First pass: fill each bin up to PER_BIN
        for bi in range(len(FREQ_BINS)):
            bin_df = df_valid[df_valid['freq_bin'] == bi]
            # Sort by certainty (descending)
            frac_col = 'lrda_frac' if 'lrda' in label.lower() else 'grda_frac'
            bin_df = bin_df.sort_values(frac_col, ascending=False)
            take = min(PER_BIN, len(bin_df))
            selected.extend(bin_df.head(take).index.tolist())
            bin_counts[bi] = take
            remaining_quota += max(0, PER_BIN - take)

        # Second pass: fill shortages from adjacent bins
        if remaining_quota > 0 and len(selected) < n_target:
            already = set(selected)
            available_extra = df_valid[~df_valid.index.isin(already)]
            frac_col = 'lrda_frac' if 'lrda' in label.lower() else 'grda_frac'
            available_extra = available_extra.sort_values(frac_col, ascending=False)
            need = n_target - len(selected)
            extras = available_extra.head(need).index.tolist()
            selected.extend(extras)

        result = df_valid.loc[selected[:n_target]].copy()

        print(f"  {label}: selected {len(result)} patients")
        for bi, (lo, hi) in enumerate(FREQ_BINS):
            n_in_bin = (result['freq_bin'] == bi).sum()
            print(f"    [{lo:.1f}, {hi:.1f}) Hz: {n_in_bin}")
        return result

    lrda_selected = stratified_select(lrda_best, N_TARGET, 'LRDA')
    grda_selected = stratified_select(grda_best, N_TARGET, 'GRDA')

    # ── Step 7: Save selected segments ────────────────────────────────────
    print("\n[7] Saving selected segments to data/eeg/...")
    os.makedirs(str(EEG_DIR), exist_ok=True)
    os.makedirs(str(OUT_DIR), exist_ok=True)
    os.makedirs(str(IMG_DIR), exist_ok=True)

    segments_csv = pd.read_csv(str(SEGMENTS_CSV))
    patients_csv = pd.read_csv(str(PATIENTS_CSV))

    new_segments = []
    new_patients = []
    manifest_rows = []
    save_failed = 0

    for subtype, df_sel in [('lrda', lrda_selected), ('grda', grda_selected)]:
        for _, row in tqdm(df_sel.iterrows(), total=len(df_sel), desc=f"  Saving {subtype.upper()}"):
            pid = str(int(row['bdsp_mrn']))
            seg_id = f"{pid}_seg000"
            mat_file = f"{seg_id}.mat"
            mat_path_ext = EXTERNAL_SEG_DIR / f"{row['file_name']}.mat"
            mat_path_local = EEG_DIR / mat_file

            try:
                mono = load_mat_segment(mat_path_ext)
                bi = get_bipolar(mono)

                # Save bipolar (18, 2000) with Fs
                scipy.io.savemat(str(mat_path_local), {
                    'data': bi.astype(np.float64),
                    'Fs': np.array([[FS]], dtype=np.float64),
                })

                new_segments.append({
                    'segment_id': seg_id,
                    'patient_id': pid,
                    'subtype': subtype,
                    'subtype_source': 'vote_majority',
                    'mat_file': mat_file,
                    'duration_sec': 10.0,
                    'fs': FS,
                    'n_channels': 18,
                    'montage': 'bipolar',
                    'original_source': 'external_drive',
                    'original_filename': f"{row['file_name']}.mat",
                })

                new_patients.append({
                    'patient_id': pid,
                    'subtype': subtype,
                    'n_segments': 1,
                    'n_raters': 0,
                    'raters': '',
                    'gold_standard_freq': np.nan,
                    'excluded': False,
                    'exclusion_reason': '',
                })

                manifest_rows.append({
                    'patient_id': pid,
                    'segment_id': seg_id,
                    'subtype': subtype,
                    'mat_file': mat_file,
                    'est_freq': row['est_freq'],
                    'certainty': row.get('lrda_frac', row.get('grda_frac', np.nan)),
                    'original_filename': row['file_name'],
                })

            except Exception as e:
                save_failed += 1
                if save_failed <= 5:
                    print(f"    FAILED {pid}: {e}")
                continue

    print(f"  Saved {len(new_segments)} segments ({save_failed} failed)")

    # Append to segments.csv and patients.csv
    if new_segments:
        new_seg_df = pd.DataFrame(new_segments)
        updated_seg = pd.concat([segments_csv, new_seg_df], ignore_index=True)
        updated_seg.to_csv(str(SEGMENTS_CSV), index=False)
        print(f"  Updated segments.csv: {len(segments_csv)} -> {len(updated_seg)} rows")

    if new_patients:
        new_pat_df = pd.DataFrame(new_patients)
        updated_pat = pd.concat([patients_csv, new_pat_df], ignore_index=True)
        updated_pat.to_csv(str(PATIENTS_CSV), index=False)
        print(f"  Updated patients.csv: {len(patients_csv)} -> {len(updated_pat)} rows")

    # Save manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(str(OUT_DIR / 'manifest.csv'), index=False)
    print(f"  Saved manifest.csv ({len(manifest_df)} rows)")

    # ── Step 8: Generate clean EEG images ─────────────────────────────────
    print("\n[8] Generating EEG images...")
    img_success = 0
    img_fail = 0
    for _, mrow in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="  Images"):
        seg_id = mrow['segment_id']
        pid = mrow['patient_id']
        subtype = mrow['subtype']
        mat_path = EEG_DIR / mrow['mat_file']

        try:
            mat = scipy.io.loadmat(str(mat_path))
            seg_bi = mat['data'].astype(np.float64)
            if seg_bi.shape[0] > seg_bi.shape[1]:
                seg_bi = seg_bi.T
            fs = int(mat['Fs'].ravel()[0])

            fig = draw_figure_clean(seg_bi, fs, subtype, title_extra=f'Patient {pid}')
            fig.savefig(str(IMG_DIR / f"{seg_id}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            img_success += 1
        except Exception as e:
            img_fail += 1
            if img_fail <= 3:
                traceback.print_exc()
            plt.close('all')

    print(f"  Images: {img_success} OK, {img_fail} failed")

    # ── Step 9: Build annotation viewer ───────────────────────────────────
    print("\n[9] Building annotation viewer...")
    viewer_path = build_viewer(manifest_df, IMG_DIR, OUT_DIR)

    # ── Step 10: Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    grda_count = manifest_df[manifest_df['subtype'] == 'grda'].shape[0]
    lrda_count = manifest_df[manifest_df['subtype'] == 'lrda'].shape[0]
    print(f"  GRDA selected: {grda_count}")
    print(f"  LRDA selected: {lrda_count}")
    print(f"  Total segments added to data/eeg/: {len(new_segments)}")

    print("\n  Frequency distribution:")
    for subtype in ['lrda', 'grda']:
        sub = manifest_df[manifest_df['subtype'] == subtype]
        print(f"\n  {subtype.upper()}:")
        for lo, hi in FREQ_BINS:
            n = ((sub['est_freq'] >= lo) & (sub['est_freq'] < hi)).sum()
            print(f"    [{lo:.1f}, {hi:.1f}) Hz: {n}")

    print(f"\n  Manifest: {OUT_DIR / 'manifest.csv'}")
    print(f"  Images: {IMG_DIR}")
    print(f"  Viewer: {viewer_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
