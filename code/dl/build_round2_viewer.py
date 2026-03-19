"""
Build self-contained annotation viewer HTML for Round 2.

Inlines manifest data and base64-encodes PNGs into a single HTML file.
Includes numerator/denominator calculator and algorithm estimate buttons.

Run: conda run -n foe_dl python code/dl/build_round2_viewer.py
"""

import csv
import base64
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_DIR / 'data' / '_archive' / 'annotation_round2'
IMAGES_DIR = OUTPUT_DIR / 'images'


def main():
    print("Building Round 2 annotation viewer...")

    # Load manifest
    manifest_path = OUTPUT_DIR / 'manifest.csv'
    with open(str(manifest_path)) as f:
        manifest = list(csv.DictReader(f))
    print(f"  {len(manifest)} candidates")

    # Encode images as base64
    image_data = {}
    for row in manifest:
        fname = row['file_name']
        img_path = IMAGES_DIR / f"{fname}.png"
        if img_path.exists():
            with open(str(img_path), 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('ascii')
            image_data[fname] = b64
        else:
            print(f"  WARNING: missing image {img_path}")

    print(f"  {len(image_data)} images encoded")

    # Build HTML
    manifest_json = json.dumps(manifest, indent=None)
    image_json = json.dumps(image_data, indent=None)

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>Round 2 Frequency Annotation Viewer</title>
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

  #freq-panel {{
    background: #333; padding: 10px 16px;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  }}
  .freq-estimate {{
    padding: 4px 10px; border: 1px solid #555; border-radius: 4px;
    background: #3a3a3a; font-size: 12px; text-align: center; min-width: 80px;
    cursor: pointer; transition: all 0.15s;
  }}
  .freq-estimate:hover {{ border-color: #888; background: #454545; }}
  .freq-label {{ color: #999; font-size: 10px; display: block; }}
  .freq-value {{ color: #eee; font-size: 14px; font-weight: bold; }}
  .freq-consensus {{ border-color: #44cc44; background: #2a3a2a; }}
  .freq-consensus .freq-value {{ color: #44cc44; }}

  #annotation-panel {{
    background: #2a2a2a; padding: 10px 16px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
    border-bottom: 2px solid #444;
  }}
  .anno-label {{ font-size: 13px; color: #aaa; margin-right: 8px; }}

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
    <select id="filter" onchange="filterChanged()">
      <option value="all">All</option>
      <option value="lpd">LPD only</option>
      <option value="gpd">GPD only</option>
      <option value="unannotated">Unannotated</option>
      <option value="annotated">Annotated</option>
    </select>
    <div id="progress-bar-container"><div id="progress-bar"></div></div>
    <span id="progress-text">0/0 annotated</span>
  </div>
  <div id="header-right">
    <span id="counter">1 / 50</span>
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
    <span id="save-status"></span>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge" id="type-badge">LPD</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Disagreement: <strong id="disagreement-val">--</strong></span>
  <span class="info-item">File: <strong id="file-name">--</strong></span>
</div>

<div id="freq-panel">
  <span style="font-size:12px; color:#999;">Algorithm estimates (click to use):</span>
  <div class="freq-estimate" onclick="useEstimate('fB')"><span class="freq-label">ACF (f_B)</span><span class="freq-value" id="est-fB">--</span></div>
  <div class="freq-estimate" onclick="useEstimate('fpeaks')"><span class="freq-label">Peaks</span><span class="freq-value" id="est-fpeaks">--</span></div>
  <div class="freq-estimate" onclick="useEstimate('ffft')"><span class="freq-label">FFT</span><span class="freq-value" id="est-ffft">--</span></div>
  <div class="freq-estimate" onclick="useEstimate('ftkeo')"><span class="freq-label">TKEO</span><span class="freq-value" id="est-ftkeo">--</span></div>
  <div class="freq-estimate freq-consensus" onclick="useEstimate('consensus')"><span class="freq-label">Consensus</span><span class="freq-value" id="est-consensus">--</span></div>
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
  <span class="anno-label">Annotate (Hz):</span>
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
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> navigate &nbsp;
  <span class="key">1</span>-<span class="key">0</span> annotate &nbsp;
  <span class="key">S</span> skip &nbsp;
  <span class="key">C</span> use custom/calc value &nbsp;
  <span class="key">E</span> export CSV
</div>

<script>
// Inline data
const MANIFEST = {manifest_json};
const IMAGE_DATA = {image_json};

let annotations = {{}};
let filteredItems = [];
let idx = 0;
let lastCustom = '';

const KEY_MAP = {{ '1': '0.25', '2': '0.5', '3': '0.75', '4': '1.0', '5': '1.25',
                  '6': '1.5', '7': '1.75', '8': '2.0', '9': '2.5', '0': '3.0' }};

// Load saved annotations
try {{
  annotations = JSON.parse(localStorage.getItem('freq_annotations_round2') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('freq_annotations_round2', JSON.stringify(annotations));
}}

function init() {{
  filteredItems = MANIFEST.slice();
  updateFilterOptions();
  idx = 0;
  show();
}}

function updateFilterOptions() {{
  const nAll = MANIFEST.length;
  const nLPD = MANIFEST.filter(m => m.subtype === 'lpd').length;
  const nGPD = MANIFEST.filter(m => m.subtype === 'gpd').length;
  const nAnn = MANIFEST.filter(m => annotations[m.file_name]).length;
  const nUnann = nAll - nAnn;
  const sel = document.getElementById('filter');
  sel.options[0].text = `All (${{nAll}})`;
  sel.options[1].text = `LPD only (${{nLPD}})`;
  sel.options[2].text = `GPD only (${{nGPD}})`;
  sel.options[3].text = `Unannotated (${{nUnann}})`;
  sel.options[4].text = `Annotated (${{nAnn}})`;
}}

function filterChanged() {{
  const f = document.getElementById('filter').value;
  if (f === 'all') filteredItems = MANIFEST.slice();
  else if (f === 'lpd') filteredItems = MANIFEST.filter(m => m.subtype === 'lpd');
  else if (f === 'gpd') filteredItems = MANIFEST.filter(m => m.subtype === 'gpd');
  else if (f === 'unannotated') filteredItems = MANIFEST.filter(m => !annotations[m.file_name]);
  else if (f === 'annotated') filteredItems = MANIFEST.filter(m => annotations[m.file_name]);
  idx = 0;
  show();
}}

function show() {{
  if (filteredItems.length === 0) {{
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image from inlined data
  const b64 = IMAGE_DATA[item.file_name];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/png;base64,' + b64;
  }} else {{
    document.getElementById('viewer').alt = 'Image not found: ' + item.file_name;
  }}

  // Info panel
  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('disagreement-val').textContent = item.disagreement;
  document.getElementById('file-name').textContent = item.file_name;

  // Frequency estimates
  document.getElementById('est-fB').textContent = item.f_B || 'NaN';
  document.getElementById('est-fpeaks').textContent = item.f_peaks || 'NaN';
  document.getElementById('est-ffft').textContent = item.f_fft || 'NaN';
  document.getElementById('est-ftkeo').textContent = item.f_tkeo || 'NaN';
  document.getElementById('est-consensus').textContent = item.consensus_estimate ? item.consensus_estimate + ' Hz' : 'NaN';

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
    // Check custom button
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
  const nAnnotated = Object.keys(annotations).length;
  const pct = total > 0 ? (nAnnotated / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnnotated + '/' + total + ' annotated';
  updateFilterOptions();
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

function useEstimate(which) {{
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  let val = '';
  if (which === 'fB') val = item.f_B;
  else if (which === 'fpeaks') val = item.f_peaks;
  else if (which === 'ffft') val = item.f_fft;
  else if (which === 'ftkeo') val = item.f_tkeo;
  else if (which === 'consensus') val = item.consensus_estimate;

  if (val) {{
    const rounded = parseFloat(val).toFixed(2);
    lastCustom = rounded;
    const cb = document.getElementById('custom-anno-btn');
    cb.innerHTML = rounded + '<br><span class="key">C</span>';
    annotate(rounded);
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
  const headers = ['patient_id', 'file_name', 'subtype', 'f_B', 'f_peaks', 'f_fft',
                   'f_tkeo', 'disagreement', 'consensus_estimate', 'expert_annotation'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.file_name] || '';
    rows.push([
      item.patient_id, item.file_name, item.subtype,
      item.f_B, item.f_peaks, item.f_fft, item.f_tkeo,
      item.disagreement, item.consensus_estimate, anno
    ].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'frequency_annotations_round2.csv';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (document.activeElement.tagName === 'INPUT') return;  // Don't capture when typing
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, filteredItems.length - 1); show(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); }}
  else if (e.key in KEY_MAP) {{ annotate(KEY_MAP[e.key]); }}
  else if (e.key === 's' || e.key === 'S') {{ annotate('skip'); }}
  else if (e.key === 'c' || e.key === 'C') {{ if (lastCustom) annotate(lastCustom); }}
  else if (e.key === 'e' || e.key === 'E') {{ exportCSV(); }}
}});

init();
</script>
</body>
</html>'''

    out_path = OUTPUT_DIR / 'annotation_viewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Viewer saved to {out_path} ({size_mb:.1f} MB)")
    print(f"\n  Open: open {out_path}")


if __name__ == '__main__':
    main()
