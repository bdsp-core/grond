"""
Spatial Extent Labeling Viewer for PD and RDA EEG Segments.

Generates HTML viewers where MW can label spatial extent (% channels involved).
The algorithm predicts a default using optimized thresholds, and MW can accept
or override with a different percentage via N/18 channel buttons.

For LPD/GPD: PDCharacterizer channel_probs, threshold 0.62.
For LRDA/GRDA: rda_spatial_extent PLV channel_scores, threshold 0.30.

Targets: segments with spatial_extent from >=2 of LB/PH/SZ, non-excluded,
without MW spatial_extent in annotations.csv.

Usage:
    conda run -n morgoth python code/generators/labeling/generate_spatial_extent_labeler.py
"""

import sys
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend

LABELING_DIR = Path(__file__).resolve().parent
CODE_DIR = LABELING_DIR.parent.parent  # code/
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'labeling_tools' / 'spatial_extent_labeling'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
LOWPASS_HZ = 20.0
DISPLAY_SAMPLES = 800
PD_THRESHOLD = 0.62
RDA_THRESHOLD = 0.30

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
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

# Display order: banana montage with separators
BIPOLAR_DISPLAY_ORDER = [0, 1, 2, 3, -1, 8, 9, 10, 11, -1, 16, 17, -1, 12, 13, 14, 15, -1, 4, 5, 6, 7]


# ── Lazy-loaded algorithm modules ──

_pd_characterizer = None

def _get_pd_characterizer():
    global _pd_characterizer
    if _pd_characterizer is None:
        from pd_characterizer import PDCharacterizer
        _pd_characterizer = PDCharacterizer()
    return _pd_characterizer


# ── EEG I/O ──

def load_segment(mat_file):
    """Load EEG segment, converting monopolar to bipolar."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] == 19:
        seg = seg[BIPOLAR_INDICES[:, 0]] - seg[BIPOLAR_INDICES[:, 1]]
    elif seg.shape[0] >= 18:
        seg = seg[:18]
    else:
        return None
    return seg


def downsample(arr, target_len):
    """Downsample 2D array to target length along axis=1, rounded for compact JSON."""
    n = arr.shape[1]
    if n <= target_len:
        out = arr
    else:
        indices = np.linspace(0, n - 1, target_len).astype(int)
        out = arr[:, indices]
    # Round to 1 decimal to reduce JSON size (~40% smaller)
    return np.round(out, 1).tolist()


# ── Algorithm predictions ──

def predict_pd_channel_scores(seg_bi, subtype):
    """Use PDCharacterizer for LPD/GPD channel probabilities."""
    try:
        pc = _get_pd_characterizer()
        result = pc.characterize(seg_bi, subtype=subtype)
        channel_probs = result.get('channel_probs', [0.5] * 18)
        return [float(p) for p in channel_probs]
    except Exception as e:
        print(f"  PDCharacterizer failed: {e}")
        return [0.5] * 18


def predict_rda_channel_scores(seg_bi, freq_hz):
    """Use rda_spatial_extent for LRDA/GRDA channel scores."""
    try:
        from rda_spatial_extent import rda_spatial_extent
        result = rda_spatial_extent(seg_bi, freq_hz, threshold=RDA_THRESHOLD)
        scores = result['channel_scores']
        return [float(s) for s in scores]
    except Exception as e:
        print(f"  rda_spatial_extent failed: {e}")
        return [0.5] * 18


# ── Case selection ──

def find_target_segments():
    """Find segments with spatial_extent from >=2 of LB/PH/SZ, non-excluded, no MW label."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    # Non-excluded segments
    sl_active = sl[sl.excluded == False].copy()
    active_mats = set(sl_active['mat_file'])

    # Expert spatial_extent annotations
    expert_ann = ann[
        ann.rater.isin(['LB', 'PH', 'SZ']) &
        ann.spatial_extent.notna() &
        ann.mat_file.isin(active_mats)
    ].copy()

    # Require at least 2 expert raters with spatial_extent
    seg_rater_counts = expert_ann.groupby('segment_id').rater.nunique()
    segs_2plus = set(seg_rater_counts[seg_rater_counts >= 2].index)

    # MW already labeled
    mw_labeled = set(ann[(ann.rater == 'MW') & ann.spatial_extent.notna()].segment_id)

    # Need labeling
    need_label = segs_2plus - mw_labeled

    # Build expert values lookup: segment_id -> {rater: spatial_extent}
    expert_vals = {}
    for _, row in expert_ann.iterrows():
        sid = row['segment_id']
        if sid in need_label:
            if sid not in expert_vals:
                expert_vals[sid] = {}
            expert_vals[sid][row['rater']] = float(row['spatial_extent'])

    # Merge with segment_labels for metadata
    sl_lookup = sl_active.set_index('mat_file')
    mat_to_seg = dict(zip(
        ann[ann.segment_id.isin(need_label)].segment_id,
        ann[ann.segment_id.isin(need_label)].mat_file
    ))

    results = {'pd': [], 'rda': []}
    for sid in sorted(need_label):
        if sid not in mat_to_seg:
            continue
        mat_file = mat_to_seg[sid]
        if mat_file not in sl_lookup.index:
            continue
        row = sl_lookup.loc[mat_file]
        subtype = row['subtype']
        patient_id = str(row['patient_id'])
        is_pd = subtype in ('lpd', 'gpd')
        category = 'pd' if is_pd else 'rda'

        freq_hz = None
        if not is_pd:
            freq_hz = row.get('pdchar_freq_hz', None)
            if pd.isna(freq_hz):
                freq_hz = row.get('algo_freq_hz', None)
            if pd.isna(freq_hz):
                freq_hz = 1.5  # fallback

        results[category].append({
            'segment_id': sid,
            'mat_file': mat_file,
            'patient_id': patient_id,
            'subtype': subtype,
            'expert_vals': expert_vals.get(sid, {}),
            'freq_hz': float(freq_hz) if freq_hz is not None else None,
        })

    return results


# ── HTML builder ──

def build_html(cases_data, category, storage_key, export_filename):
    """Build self-contained HTML viewer."""
    n_cases = len(cases_data)
    title = f"{'PD' if category == 'pd' else 'RDA'} Spatial Extent Labeler"
    accent = '#F29030' if category == 'pd' else '#7CB342'

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    bipolar_names_json = json.dumps(BIPOLAR_CHANNELS)
    display_order_json = json.dumps(BIPOLAR_DISPLAY_ORDER)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title} ({n_cases} cases)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; border-bottom: 2px solid #444;
    flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}

  #progress-bar-wrap {{ width: 100%; height: 6px; background: #333; }}
  #progress-bar {{ height: 100%; background: {accent}; transition: width 0.2s; }}

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #main-container {{
    display: flex; width: 100%; min-height: 600px;
  }}

  #eeg-panel {{
    flex: 1; padding: 8px; position: relative;
  }}
  #eeg-canvas {{ display: block; width: 100%; cursor: default; }}

  #control-panel {{
    flex: 0 0 320px; max-width: 320px; padding: 8px;
    display: flex; flex-direction: column; gap: 8px;
    border-left: 1px solid #333;
  }}

  #channel-buttons {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 4px; padding: 6px;
    background: #252525; border-radius: 6px;
  }}
  .ch-btn {{
    padding: 8px 4px; border: 2px solid #555; border-radius: 4px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 12px; font-weight: bold; text-align: center;
    transition: all 0.15s; user-select: none;
  }}
  .ch-btn:hover {{ background: #444; border-color: #888; }}
  .ch-btn.active {{ background: #1a2a1a; border-color: {accent}; color: {accent}; }}
  .ch-btn .pct {{ font-size: 10px; color: #888; display: block; margin-top: 1px; }}
  .ch-btn.active .pct {{ color: {accent}80; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid {accent}; border-radius: 4px;
    background: #2a3a2a; color: {accent}; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}

  #save-status {{ color: {accent}; font-size: 13px; }}

  #expert-info {{
    background: #252525; border-radius: 6px; padding: 8px 10px;
    font-size: 12px; line-height: 1.6;
  }}
  #expert-info .label {{ color: #888; }}
  #expert-info .value {{ color: #eee; font-weight: bold; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:{accent};">{title}</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="rejectCase()" style="border-color:#ff6644; color:#ff6644; background:#3a2a2a;">Reject <span class="key">X</span></button>
    <button class="export-btn" onclick="acceptCase()">Accept <span class="key">Enter</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype" style="color:{accent};">--</strong></span>
  <span class="info-item" id="info-experts">Experts: --</span>
  <span class="info-item">Algorithm: <strong id="info-algo">--</strong></span>
  <span class="info-item">Selected: <strong id="info-selected" style="color:{accent};">--</strong></span>
</div>

<div id="main-container">
  <div id="eeg-panel">
    <canvas id="eeg-canvas"></canvas>
  </div>
  <div id="control-panel">
    <div id="expert-info"></div>
    <div style="font-size:12px; color:#888; padding:4px 6px;">Channels involved (click or use Up/Down):</div>
    <div id="channel-buttons"></div>
  </div>
</div>

<div id="shortcuts">
  <span class="key">Enter</span> Accept &amp; next &nbsp;&nbsp;
  <span class="key">X</span> Reject (not PD/RDA) &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Increase/decrease channels &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const BIPOLAR_NAMES = {bipolar_names_json};
const DISPLAY_ORDER = {display_order_json};
const DURATION = {DURATION};
const N_CHANNELS = 18;

const EEG_WIDTH = 900;
const EEG_HEIGHT = 650;
const MARGIN_LEFT = 110;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 25;
const MARGIN_BOTTOM = 20;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;
const CLIP_UV = 300.0;
const Z_SCALE = 0.01;

let idx = 0;
let selectedN = 0;     // number of channels currently selected
let labeled = new Set();

// Persistence
const STORAGE_KEY = '{storage_key}';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

// Restore already-labeled set
for (const sid of Object.keys(allLabels)) {{ labeled.add(sid); }}

// Get display channels with separators
const DISPLAY_CHANNELS = [];
for (const i of DISPLAY_ORDER) {{
  if (i < 0) DISPLAY_CHANNELS.push({{ idx: -1, name: '' }});
  else DISPLAY_CHANNELS.push({{ idx: i, name: BIPOLAR_NAMES[i] }});
}}
const N_DISPLAY = DISPLAY_CHANNELS.length;

// Build channel buttons (0/18 through 18/18)
(function() {{
  const panel = document.getElementById('channel-buttons');
  for (let n = 0; n <= N_CHANNELS; n++) {{
    const btn = document.createElement('div');
    btn.className = 'ch-btn';
    btn.id = 'ch-btn-' + n;
    const pct = Math.round(n / N_CHANNELS * 100);
    btn.innerHTML = n + '/18<span class="pct">' + pct + '%</span>';
    btn.onclick = () => setChannelCount(n);
    panel.appendChild(btn);
  }}
}})();

function setChannelCount(n) {{
  selectedN = Math.max(0, Math.min(N_CHANNELS, n));
  redraw();
}}

function getInvolvedChannels() {{
  // Sort channels by score descending, return top N indices
  const c = CASES[idx];
  const scores = c.channel_scores;
  const ranked = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const involved = new Set();
  for (let i = 0; i < selectedN && i < ranked.length; i++) {{
    involved.add(ranked[i][1]);
  }}
  return involved;
}}

function redraw() {{
  updateInfo();
  updateButtons();
  drawEEG();
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();

  // Expert values
  const ev = c.expert_vals;
  let expertStr = '';
  for (const [rater, val] of Object.entries(ev)) {{
    const pct = Math.round(val * 100);
    expertStr += rater + '=' + pct + '% ';
  }}
  document.getElementById('info-experts').innerHTML = 'Experts: <strong>' + (expertStr || 'none') + '</strong>';

  // Algorithm default
  const algoPct = Math.round(c.default_n / N_CHANNELS * 100);
  document.getElementById('info-algo').textContent = c.default_n + '/18 (' + algoPct + '%)';

  // Selected
  const selPct = Math.round(selectedN / N_CHANNELS * 100);
  document.getElementById('info-selected').textContent = selectedN + '/18 (' + selPct + '%)';

  // Progress bar
  const progress = labeled.size / CASES.length * 100;
  document.getElementById('progress-bar').style.width = progress + '%';

  // Labeled count
  document.getElementById('labeled-count').textContent = labeled.size + '/' + CASES.length + ' labeled';

  // Expert info panel
  const infoDiv = document.getElementById('expert-info');
  let html = '<div><span class="label">Segment:</span> <span class="value">' + c.segment_id + '</span></div>';
  html += '<div><span class="label">Patient:</span> <span class="value">' + c.patient_id + '</span></div>';
  html += '<div><span class="label">Subtype:</span> <span class="value">' + c.subtype.toUpperCase() + '</span></div>';
  html += '<div style="margin-top:4px"><span class="label">Expert spatial extent:</span></div>';
  for (const [rater, val] of Object.entries(ev)) {{
    const nCh = Math.round(val * N_CHANNELS);
    html += '<div>&nbsp;&nbsp;' + rater + ': ' + Math.round(val * 100) + '% (' + nCh + '/18)</div>';
  }}
  const meanVal = Object.values(ev).length > 0 ? Object.values(ev).reduce((a, b) => a + b, 0) / Object.values(ev).length : 0;
  html += '<div>&nbsp;&nbsp;Mean: ' + Math.round(meanVal * 100) + '%</div>';
  html += '<div style="margin-top:4px"><span class="label">Algorithm default:</span> <span class="value">' + c.default_n + '/18 (' + algoPct + '%)</span></div>';
  html += '<div><span class="label">Selected:</span> <span class="value" style="color:{accent};">' + selectedN + '/18 (' + selPct + '%)</span></div>';

  // Show top channels by score
  const scores = c.channel_scores;
  const ranked = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  html += '<div style="margin-top:6px"><span class="label">Channel ranking:</span></div>';
  html += '<div style="font-size:10px; max-height:160px; overflow-y:auto; padding:2px 0;">';
  for (let i = 0; i < ranked.length; i++) {{
    const [score, chIdx] = ranked[i];
    const inSet = i < selectedN;
    const color = inSet ? '{accent}' : '#666';
    html += '<div style="color:' + color + ';">' + (i + 1) + '. ' + BIPOLAR_NAMES[chIdx] + ' = ' + score.toFixed(3) + '</div>';
  }}
  html += '</div>';
  infoDiv.innerHTML = html;
}}

function updateButtons() {{
  for (let n = 0; n <= N_CHANNELS; n++) {{
    const btn = document.getElementById('ch-btn-' + n);
    btn.classList.toggle('active', n === selectedN);
  }}
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;
  const involved = getInvolvedChannels();
  const scores = c.channel_scores;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = PLOT_LEFT + (s / DURATION) * PLOT_W;
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    const isHL = involved.has(ch.idx);

    ctx.strokeStyle = isHL ? '#cc3333' : '#999999';
    ctx.lineWidth = isHL ? 1.5 : 0.6;
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

  // Channel labels with scores
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const isHL = involved.has(ch.idx);
    const score = scores[ch.idx];

    // Channel name
    ctx.font = isHL ? 'bold 10px Consolas, Monaco, monospace' : '10px Consolas, Monaco, monospace';
    ctx.textAlign = 'right';
    ctx.fillStyle = isHL ? '#cc3333' : '#aaaaaa';
    ctx.fillText(ch.name, MARGIN_LEFT - 30, yCenter);

    // Score value
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'left';
    ctx.fillStyle = isHL ? '#cc3333' : '#bbbbbb';
    ctx.fillText(score.toFixed(2), MARGIN_LEFT - 28, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    const x = PLOT_LEFT + (s / DURATION) * PLOT_W;
    ctx.fillText(s + 's', x, EEG_HEIGHT - MARGIN_BOTTOM + 4);
  }}
}}

// ── Actions ──

function acceptCase() {{
  const c = CASES[idx];
  allLabels[c.segment_id] = {{
    segment_id: c.segment_id,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
    subtype: c.subtype,
    n_channels: selectedN,
    spatial_extent: selectedN / N_CHANNELS,
    default_n: c.default_n,
    expert_vals: c.expert_vals,
    rejected: false,
    timestamp: new Date().toISOString(),
  }};
  labeled.add(c.segment_id);
  saveAll();
  showStatus('Saved: ' + selectedN + '/18');
  goNext();
}}

function rejectCase() {{
  const c = CASES[idx];
  allLabels[c.segment_id] = {{
    segment_id: c.segment_id,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
    subtype: c.subtype,
    n_channels: null,
    spatial_extent: null,
    default_n: c.default_n,
    expert_vals: c.expert_vals,
    rejected: true,
    timestamp: new Date().toISOString(),
  }};
  labeled.add(c.segment_id);
  saveAll();
  showStatus('Rejected');
  goNext();
}}

function goNext() {{
  if (idx < CASES.length - 1) {{
    idx++;
    loadCase();
  }}
}}

function goPrev() {{
  if (idx > 0) {{
    idx--;
    loadCase();
  }}
}}

function loadCase() {{
  const c = CASES[idx];
  // Restore previous label if exists
  if (allLabels[c.segment_id] && !allLabels[c.segment_id].rejected && allLabels[c.segment_id].n_channels !== null) {{
    selectedN = allLabels[c.segment_id].n_channels;
  }} else {{
    selectedN = c.default_n;
  }}
  redraw();
}}

function showStatus(msg) {{
  const el = document.getElementById('save-status');
  el.textContent = msg;
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
}}

function exportJSON() {{
  const data = JSON.stringify(allLabels, null, 2);
  const blob = new Blob([data], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = '{export_filename}';
  a.click();
  URL.revokeObjectURL(url);
  showStatus('Exported ' + Object.keys(allLabels).length + ' labels');
}}

// ── Keyboard ──

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') {{ e.preventDefault(); acceptCase(); }}
  else if (e.key === 'x' || e.key === 'X') {{ e.preventDefault(); rejectCase(); }}
  else if (e.key === 'ArrowUp') {{ e.preventDefault(); setChannelCount(selectedN + 1); }}
  else if (e.key === 'ArrowDown') {{ e.preventDefault(); setChannelCount(selectedN - 1); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); goNext(); }}
  else if (e.key === 'ArrowLeft') {{ e.preventDefault(); goPrev(); }}
  else if (e.key === 'e' || e.key === 'E') {{ e.preventDefault(); exportJSON(); }}
}});

// ── Init ──
loadCase();
</script>
</body>
</html>"""
    return html


# ── Main ──

def main():
    print("=" * 70)
    print("  Spatial Extent Labeler Generator")
    print("=" * 70)

    print("\nFinding target segments...")
    targets = find_target_segments()
    print(f"  PD segments: {len(targets['pd'])}")
    print(f"  RDA segments: {len(targets['rda'])}")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    for category, cases_info in targets.items():
        if len(cases_info) == 0:
            print(f"\n  No {category.upper()} cases to process, skipping.")
            continue

        is_pd = (category == 'pd')
        label = 'PD' if is_pd else 'RDA'
        storage_key = f'expert_spatial_{category}_v1'
        export_filename = f'expert_spatial_{category}_results.json'
        out_file = OUT_DIR / f'spatial_extent_labeler_{category}.html'

        print(f"\n{'='*50}")
        print(f"  Processing {label} ({len(cases_info)} segments)...")
        print(f"{'='*50}")

        cases_data = []
        n_skipped = 0

        for i, info in enumerate(cases_info):
            mat_file = info['mat_file']
            seg = load_segment(mat_file)
            if seg is None:
                n_skipped += 1
                continue

            # Prepare display EEG
            seg_display = np.zeros_like(seg)
            for ch in range(seg.shape[0]):
                try:
                    seg_display[ch] = filtfilt(b_lp, a_lp, detrend(seg[ch], type='linear'))
                except Exception:
                    seg_display[ch] = seg[ch]

            # Get channel scores from algorithm
            if is_pd:
                channel_scores = predict_pd_channel_scores(seg, info['subtype'])
                threshold = PD_THRESHOLD
            else:
                freq_hz = info['freq_hz'] if info['freq_hz'] else 1.5
                channel_scores = predict_rda_channel_scores(seg, freq_hz)
                threshold = RDA_THRESHOLD

            # Compute default N: count channels above threshold
            default_n = int(sum(1 for s in channel_scores if s > threshold))

            case = {
                'segment_id': info['segment_id'],
                'mat_file': mat_file,
                'patient_id': info['patient_id'],
                'subtype': info['subtype'],
                'channel_scores': [round(s, 4) for s in channel_scores],
                'default_n': default_n,
                'expert_vals': info['expert_vals'],
                'eeg_data': downsample(seg_display, DISPLAY_SAMPLES),
            }
            cases_data.append(case)

            if (i + 1) % 50 == 0 or (i + 1) == len(cases_info):
                print(f"  {i+1}/{len(cases_info)} processed ({n_skipped} skipped)")

        print(f"\n  Total {label} cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")

        if len(cases_data) == 0:
            print(f"  No {label} cases to label!")
            continue

        # Sort by segment_id for consistency
        cases_data.sort(key=lambda c: c['segment_id'])

        # Build and write HTML
        print(f"  Building HTML viewer...")
        html = build_html(cases_data, category, storage_key, export_filename)

        with open(str(out_file), 'w') as f:
            f.write(html)
        print(f"  Written to {out_file}")
        print(f"  {len(cases_data)} cases ready for review")

        # Open in browser
        import subprocess
        subprocess.run(['open', str(out_file)])

    print("\n" + "=" * 70)
    print("  Done! Both viewers opened.")
    print("=" * 70)


if __name__ == '__main__':
    main()
