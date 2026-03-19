"""
Generate annotation package for 30 patients MW has not yet scored.
Selects 1 segment per patient (highest variance), generates EEG images,
builds annotation viewer HTML, and saves manifest.

Must run with: conda run -n foe python code/generate_mw_catchup.py
"""

import sys, os, json, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from generate_test_images import draw_figure as draw_figure_full
from browse_results import BIPOLAR_CHANNELS, LEFT_INDICES, RIGHT_INDICES, get_bipolar
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def draw_figure_clean(row, seg_bi, fs, pattern_type, title_extra=''):
    """Draw EEG traces only — no side panels, no metadata, no scores."""
    fig = plt.figure(figsize=(16, 11))

    gs = GridSpec(19, 1, hspace=0.08,
                  left=0.10, right=0.98, top=0.95, bottom=0.05)

    time_vec = np.linspace(0, seg_bi.shape[1] / fs, seg_bi.shape[1])

    fig.text(0.5, 0.975,
             f'{pattern_type.upper()} — {title_extra}',
             ha='center', fontsize=12, fontweight='bold')

    for i in range(18):
        ax = fig.add_subplot(gs[i + 1, 0])
        color = '#1a6dd4'
        ax.plot(time_vec, seg_bi[i, :], color=color, linewidth=1.0)

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

BASE = CODE_DIR.parent
DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
SEGMENTS_CSV = DATA / 'labels' / 'segments.csv'
OUT_DIR = DATA / '_archive' / 'annotation_mw_catchup'
IMG_DIR = OUT_DIR / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PATIENTS = [
    'abn10320', 'abn10424', 'abn10490', 'abn11079', 'abn11260',
    'abn13638', 'abn14115', 'abn1967', 'abn2077', 'abn2102',
    'abn3292', 'abn4064', 'abn4587', 'abn514', 'abn6074',
    'abn620', 'abn6757', 'abn6954', 'abn7509', 'abn798',
    'abn8980', 'emu141', 'emu24', 'emu268', 'emu386',
    'pat0004', 'pat0057', 'pat0081', 'pat0084', 'pat0108',
]


def select_segments(segments_df):
    """Select 1 segment per patient based on highest signal variance."""
    target_segs = segments_df[segments_df['patient_id'].isin(TARGET_PATIENTS)].copy()
    print(f"Found {len(target_segs)} total segments for {target_segs['patient_id'].nunique()} patients")

    missing = set(TARGET_PATIENTS) - set(target_segs['patient_id'].unique())
    if missing:
        print(f"  WARNING: No segments found for: {missing}")

    selected = []
    for pid in TARGET_PATIENTS:
        pat_segs = target_segs[target_segs['patient_id'] == pid]
        if len(pat_segs) == 0:
            print(f"  SKIP {pid}: no segments found")
            continue

        if len(pat_segs) == 1:
            selected.append(pat_segs.iloc[0])
            continue

        # Pick segment with highest variance
        best_var = -1
        best_row = None
        for _, row in pat_segs.iterrows():
            mat_path = EEG_DIR / row['mat_file']
            if not mat_path.exists():
                continue
            try:
                mat = scipy.io.loadmat(str(mat_path))
                # Try both field names
                if 'data' in mat:
                    data = mat['data']
                elif 'data_50sec' in mat:
                    data = mat['data_50sec']
                else:
                    continue
                var = np.var(data)
                if var > best_var:
                    best_var = var
                    best_row = row
            except Exception as e:
                print(f"  WARNING: Could not load {mat_path}: {e}")
                continue

        if best_row is not None:
            selected.append(best_row)
        else:
            # Fallback to first segment
            selected.append(pat_segs.iloc[0])

    result = pd.DataFrame(selected)
    print(f"Selected {len(result)} segments (1 per patient)")
    print(f"  Subtypes: {result['subtype'].value_counts().to_dict()}")
    return result


def generate_image(row):
    """Generate EEG image for a single segment."""
    mat_path = EEG_DIR / row['mat_file']
    pid = row['patient_id']
    subtype = row['subtype']
    seg_id = row['segment_id']

    try:
        mat = scipy.io.loadmat(str(mat_path))
        if 'data' in mat:
            data = mat['data']
        elif 'data_50sec' in mat:
            data = mat['data_50sec']
        else:
            print(f"  FAILED {seg_id}: no data field in mat file")
            return False

        fs = int(mat['Fs'].ravel()[0])

        # Data is monopolar (20, 2000) — convert to bipolar (18, 2000)
        data = data.astype(np.float64)
        if data.shape[0] > data.shape[1]:
            data = data.T

        seg_bi = get_bipolar(data)

        # Mock result row (no detector scores — just display EEG)
        result_row = {
            'files': seg_id,
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

        fig = draw_figure_clean(result_row, seg_bi, fs, subtype,
                               title_extra=f'Patient {pid}')
        png_path = IMG_DIR / f"{seg_id}.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAILED {seg_id}: {e}")
        return False


def build_viewer(manifest):
    """Build self-contained annotation viewer HTML with inlined base64 images."""
    n_total = len(manifest)
    print(f"\nBuilding viewer for {n_total} items...")
    subtypes = manifest['subtype'].value_counts().to_dict()
    print(f"  Subtypes: {subtypes}")

    # Shuffle for randomized order
    manifest_shuffled = manifest.sample(frac=1, random_state=42).reset_index(drop=True)
    manifest_json = manifest_shuffled[['patient_id', 'segment_id', 'subtype', 'mat_file']].to_dict('records')

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

    # Build filter options for subtypes present
    subtype_options = '\n'.join(
        f'      <option value="{st}">{st.upper()} only</option>'
        for st in sorted(subtypes.keys())
    )

    # Keyboard shortcut hints for subtype filters
    subtype_keys = {st: st[0].upper() for st in sorted(subtypes.keys())}

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>MW Catchup Frequency Annotation</title>
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
  annotations = JSON.parse(localStorage.getItem('freq_annotations_mw_catchup') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('freq_annotations_mw_catchup', JSON.stringify(annotations));
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

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Highlight current annotation
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
  const headers = ['patient_id', 'segment_id', 'subtype', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.segment_id] || '';
    rows.push([
      item.patient_id, item.segment_id, item.subtype, anno
    ].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'frequency_annotations_mw_catchup.csv';
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

    output_path = OUT_DIR / 'annotation_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved annotation_viewer.html ({size_mb:.1f} MB)")
    return output_path


def main():
    print("=" * 60)
    print("MW Catchup Annotation Package")
    print("=" * 60)

    # Step 1: Select segments
    print("\n--- Step 1: Selecting segments ---")
    segments = pd.read_csv(SEGMENTS_CSV)
    selected = select_segments(segments)

    # Step 2: Generate images
    print("\n--- Step 2: Generating EEG images ---")
    success = 0
    fail = 0
    for i, (_, row) in enumerate(selected.iterrows()):
        print(f"  [{i+1}/{len(selected)}] {row['segment_id']} ({row['subtype']})", end='', flush=True)
        if generate_image(row):
            print('  OK')
            success += 1
        else:
            print('  FAILED')
            fail += 1

    print(f"\n  Images: {success} OK, {fail} failed")

    # Step 3: Save manifest
    print("\n--- Step 3: Saving manifest ---")
    manifest = selected[['patient_id', 'segment_id', 'subtype', 'mat_file']].copy()
    manifest.to_csv(OUT_DIR / 'manifest.csv', index=False)
    print(f"  Saved manifest.csv ({len(manifest)} rows)")

    # Step 4: Build viewer
    print("\n--- Step 4: Building annotation viewer ---")
    viewer_path = build_viewer(manifest)

    # Summary
    print("\n" + "=" * 60)
    print("DONE!")
    print(f"  Output directory: {OUT_DIR}")
    print(f"  Total patients: {len(manifest)}")
    print(f"  Images generated: {success}")
    print(f"  Images failed: {fail}")
    print(f"  Viewer: {viewer_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
