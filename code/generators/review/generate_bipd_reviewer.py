"""
BIPD vs GPD Reviewer — Review model predictions and relabel cases.

Shows all GPD + BIPD cases sorted by model BIPD probability (highest first).
For each case, displays:
  - EEG with channels grouped: L-lateral, L-parasag, midline, R-parasag, R-lateral
  - Left hemisphere discharge times (red vertical lines)
  - Right hemisphere discharge times (blue vertical lines)
  - Model prediction (BIPD prob, GPD/BIPD label)
  - 1/2 buttons to label as GPD or BIPD

Uses pre-computed per-hemisphere detections from bipd_cache/.

Usage:
    conda run -n foe python code/generate_bipd_reviewer.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
CACHE_DIR = DATA_DIR / 'bipd_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
N_SAMPLES = 2000
DURATION = 10.0
LOWPASS_HZ = 20.0

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Display order: L-lateral, L-parasag, midline, R-parasag, R-lateral
DISPLAY_ORDER = [
    0, 1, 2, 3,       # left lateral
    8, 9, 10, 11,     # left parasagittal
    -1,               # gap
    16, 17,           # midline
    -1,               # gap
    12, 13, 14, 15,   # right parasagittal
    4, 5, 6, 7,       # right lateral
]


def load_segment(pid):
    """Load an EEG segment for a patient. Returns (18, 2000) or None."""
    import pandas as pd
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    rows = seg_df[seg_df['patient_id'] == pid]
    if len(rows) > 0:
        mat_file = rows.iloc[0]['mat_file']
    else:
        mat_file = None
        for suffix in ['_seg000.mat', '.mat']:
            if (EEG_DIR / f'{pid}{suffix}').exists():
                mat_file = f'{pid}{suffix}'
                break
    if mat_file is None:
        return None
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] >= 18 and seg.shape[1] >= N_SAMPLES:
        return seg[:18, :N_SAMPLES]
    return None


def downsample(arr, target_len):
    if arr.ndim == 1:
        n = len(arr)
        if n <= target_len:
            return arr.tolist()
        indices = np.linspace(0, n - 1, target_len).astype(int)
        return arr[indices].tolist()
    else:
        n = arr.shape[1]
        if n <= target_len:
            return arr.tolist()
        indices = np.linspace(0, n - 1, target_len).astype(int)
        return arr[:, indices].tolist()


def main():
    print("=" * 70)
    print("  BIPD vs GPD Reviewer")
    print("  Review model predictions, relabel cases")
    print("=" * 70)

    # Load detections from cache
    gpd_path = CACHE_DIR / 'gpd_hemi_detections.json'
    bipd_path = CACHE_DIR / 'bipd_hemi_detections.json'

    with open(str(gpd_path)) as f:
        gpd_det = json.load(f)
    with open(str(bipd_path)) as f:
        bipd_det = json.load(f)

    # Load model predictions
    results_path = RESULTS_DIR / 'bipd_detection_results.json'
    with open(str(results_path)) as f:
        model_results = json.load(f)
    predictions = model_results.get('predictions', {})

    # Combine all cases with their model predictions
    all_cases = {}
    for pid, det in gpd_det.items():
        pred = predictions.get(pid, {})
        all_cases[pid] = {
            'det': det,
            'prob': pred.get('prob', 0.0),
            'model_pred': 'BIPD' if pred.get('pred', 0) == 1 else 'GPD',
            'original_label': 'GPD',
        }
    for pid, det in bipd_det.items():
        pred = predictions.get(pid, {})
        all_cases[pid] = {
            'det': det,
            'prob': pred.get('prob', 0.0),
            'model_pred': 'BIPD' if pred.get('pred', 0) == 1 else 'GPD',
            'original_label': 'BIPD',
        }

    # Filter to candidates: prob >= 0.5 (predicted BIPD) — skip already-reviewed
    review_path = LABELS_DIR / 'bipd_review_labels_mw.json'
    already_reviewed = set()
    if review_path.exists():
        with open(str(review_path)) as f:
            already_reviewed = set(json.load(f).keys())

    candidate_pids = [pid for pid, info in all_cases.items()
                      if info['prob'] >= 0.5 and pid not in already_reviewed]
    # Sort by BIPD probability (highest first)
    sorted_pids = sorted(candidate_pids, key=lambda p: -all_cases[p]['prob'])

    n_total_flagged = sum(1 for info in all_cases.values() if info['prob'] >= 0.5)
    print(f"Total flagged (prob>=0.5): {n_total_flagged}")
    print(f"Already reviewed: {len(already_reviewed)}")
    print(f"Candidates for review: {len(sorted_pids)}")

    # Build cases data for HTML
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    cases_data = []
    n_skipped = 0

    for i, pid in enumerate(sorted_pids):
        info = all_cases[pid]
        seg = load_segment(pid)
        if seg is None:
            n_skipped += 1
            continue

        # Lowpass for display
        seg_display = np.zeros_like(seg)
        for ch in range(18):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except ValueError:
                seg_display[ch] = seg[ch]

        det = info['det']
        case = {
            'patient_id': str(pid),
            'prob': round(info['prob'], 3),
            'model_pred': info['model_pred'],
            'original_label': info['original_label'],
            'left_times': det['left']['times'],
            'right_times': det['right']['times'],
            'left_freq': round(det['left']['freq'], 2),
            'right_freq': round(det['right']['freq'], 2),
            'eeg_data': downsample(seg_display, 500),
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sorted_pids)} processed")

    print(f"  Total cases: {len(cases_data)} (skipped {n_skipped})")

    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)
    out_path = RESULTS_DIR / 'bipd_reviewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


def build_html(cases_data):
    display_order_json = json.dumps(DISPLAY_ORDER)
    channel_names_json = json.dumps(BIPOLAR_CHANNELS)
    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BIPD vs GPD Reviewer</title>
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
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}

  #prediction-panel {{
    padding: 10px 16px; display: flex; align-items: center; gap: 20px;
    flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 14px;
  }}
  .pred-gpd {{ background: #1a2a1a; }}
  .pred-bipd {{ background: #3a1a2a; }}

  #info-panel {{
    background: #2a2a2a; padding: 8px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #label-buttons {{
    display: flex; gap: 12px; padding: 8px 16px; background: #252525;
    border-bottom: 1px solid #333; align-items: center;
  }}
  #label-buttons label {{ color: #aaa; font-size: 14px; font-weight: bold; margin-right: 8px; }}
  .label-btn {{
    padding: 10px 24px; border: 2px solid #555; border-radius: 6px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 16px; font-weight: bold; transition: all 0.15s;
  }}
  .label-btn:hover {{ background: #444; }}
  .label-btn.active-gpd {{ background: #1a3a1a; border-color: #44cc88; color: #44ff66; }}
  .label-btn.active-bipd {{ background: #3a1a2a; border-color: #e040e0; color: #ff66ff; }}
  .label-btn.active-lpd {{ background: #2a2a1a; border-color: #ccaa44; color: #eebb44; }}
  .label-btn.active-reject {{ background: #2a2a2a; border-color: #888; color: #aaa; }}

  #canvas-container {{ text-align: center; padding: 8px; }}
  #eeg-canvas {{ display: block; margin: 0 auto; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}
  #save-status {{ color: #44cc44; font-size: 13px; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#e040e0;">BIPD vs GPD Reviewer</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportJSON()">Export Labels <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="prediction-panel" class="pred-gpd">
  <span>Model: <strong id="pred-label" style="font-size:18px;">--</strong></span>
  <span>BIPD prob: <strong id="pred-prob" style="font-size:18px;">--</strong></span>
  <span>Original: <strong id="pred-original">--</strong></span>
</div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item" style="color:#ff6666;">Left: <strong id="info-left">0 discharges, -- Hz</strong></span>
  <span class="info-item" style="color:#6688ff;">Right: <strong id="info-right">0 discharges, -- Hz</strong></span>
</div>

<div id="label-buttons">
  <label>Your label:</label>
  <button class="label-btn" id="btn-gpd" onclick="setLabel('GPD')">
    <span class="key">1</span> GPD
  </button>
  <button class="label-btn" id="btn-bipd" onclick="setLabel('BIPD')">
    <span class="key">2</span> BIPD
  </button>
  <button class="label-btn" id="btn-lpd" onclick="setLabel('LPD')">
    <span class="key">3</span> LPD
  </button>
  <button class="label-btn" id="btn-reject" onclick="setLabel('REJECT')">
    <span class="key">4</span> Reject
  </button>
  <span id="label-status" style="font-size:14px; color:#aaa; margin-left:20px;"></span>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">1</span> GPD &nbsp;&nbsp;
  <span class="key">2</span> BIPD &nbsp;&nbsp;
  <span class="key">3</span> LPD &nbsp;&nbsp;
  <span class="key">4</span> Reject &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const DISPLAY_ORDER = {display_order_json};
const ALL_CHANNEL_NAMES = {channel_names_json};
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1300;
const EEG_HEIGHT = 750;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

const LEFT_COLOR = 'rgba(255, 60, 60, 0.6)';
const RIGHT_COLOR = 'rgba(60, 120, 255, 0.6)';
const LEFT_CHS = [0,1,2,3,8,9,10,11];
const RIGHT_CHS = [4,5,6,7,12,13,14,15];

let idx = 0;
let userLabel = null;

const STORAGE_KEY = 'bipd_reviewer_v1';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

function getDisplayChannels() {{
  return DISPLAY_ORDER.map(i => ({{
    idx: i,
    name: i >= 0 ? ALL_CHANNEL_NAMES[i] : '',
    side: i >= 0 ? (LEFT_CHS.includes(i) ? 'left' : RIGHT_CHS.includes(i) ? 'right' : 'mid') : 'gap'
  }}));
}}
const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}

function setLabel(label) {{
  userLabel = label;
  const c = CASES[idx];
  allLabels[c.patient_id] = {{
    label: label,
    prob: c.prob,
    model_pred: c.model_pred,
    original_label: c.original_label,
  }};
  saveAll();

  document.getElementById('btn-gpd').className = 'label-btn' + (label === 'GPD' ? ' active-gpd' : '');
  document.getElementById('btn-bipd').className = 'label-btn' + (label === 'BIPD' ? ' active-bipd' : '');
  document.getElementById('btn-lpd').className = 'label-btn' + (label === 'LPD' ? ' active-lpd' : '');
  document.getElementById('btn-reject').className = 'label-btn' + (label === 'REJECT' ? ' active-reject' : '');
  document.getElementById('label-status').textContent = 'Labeled: ' + label;
  updateInfo();

  // Auto-advance after short delay
  setTimeout(() => {{
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }}, 300);
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Background shading for hemispheres
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    if (ch.side === 'left') {{
      ctx.fillStyle = 'rgba(255, 220, 220, 0.15)';
      ctx.fillRect(PLOT_LEFT, yCenter - chSpacing * 0.45, PLOT_W, chSpacing * 0.9);
    }} else if (ch.side === 'right') {{
      ctx.fillStyle = 'rgba(220, 220, 255, 0.15)';
      ctx.fillRect(PLOT_LEFT, yCenter - chSpacing * 0.45, PLOT_W, chSpacing * 0.9);
    }}
  }}

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // EEG traces
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 0.7;
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
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
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillStyle = ch.side === 'left' ? '#cc3333' : ch.side === 'right' ? '#3366cc' : '#666';
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', timeToX(s), EEG_HEIGHT - MARGIN_BOTTOM + 4);
  }}

  // Compute Y ranges for each hemisphere from display channel positions
  let leftYmin = Infinity, leftYmax = -Infinity;
  let rightYmin = Infinity, rightYmax = -Infinity;
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const yTop = yCenter - chSpacing * 0.5;
    const yBot = yCenter + chSpacing * 0.5;
    if (ch.side === 'left') {{
      leftYmin = Math.min(leftYmin, yTop);
      leftYmax = Math.max(leftYmax, yBot);
    }} else if (ch.side === 'right') {{
      rightYmin = Math.min(rightYmin, yTop);
      rightYmax = Math.max(rightYmax, yBot);
    }} else if (ch.side === 'mid') {{
      // Midline: extend both hemispheres to meet here
      leftYmax = Math.max(leftYmax, yBot);
      rightYmin = Math.min(rightYmin, yTop);
    }}
  }}

  // Left discharge markers (red) — only across left hemisphere channels
  for (const t of c.left_times) {{
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    ctx.strokeStyle = LEFT_COLOR;
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, leftYmin); ctx.lineTo(x, leftYmax); ctx.stroke();
  }}

  // Right discharge markers (blue) — only across right hemisphere channels
  for (const t of c.right_times) {{
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    ctx.strokeStyle = RIGHT_COLOR;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 3]);
    ctx.beginPath(); ctx.moveTo(x, rightYmin); ctx.lineTo(x, rightYmax); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Title
  ctx.fillStyle = '#000';
  ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const predStr = c.model_pred + ' (p=' + c.prob.toFixed(2) + ')';
  const title = c.patient_id + '  |  Model: ' + predStr + '  |  Orig: ' + c.original_label;
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Legend
  ctx.font = '11px Consolas, Monaco, monospace';
  ctx.textAlign = 'left';
  ctx.fillStyle = '#cc3333';
  ctx.fillText('--- Left discharges (' + c.left_times.length + ', ' + c.left_freq + ' Hz)', PLOT_LEFT + 10, EEG_HEIGHT - 8);
  ctx.fillStyle = '#3366cc';
  ctx.fillText('- - Right discharges (' + c.right_times.length + ', ' + c.right_freq + ' Hz)', PLOT_LEFT + 350, EEG_HEIGHT - 8);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-left').textContent = c.left_times.length + ' discharges, ' + c.left_freq + ' Hz';
  document.getElementById('info-right').textContent = c.right_times.length + ' discharges, ' + c.right_freq + ' Hz';

  document.getElementById('pred-label').textContent = c.model_pred;
  document.getElementById('pred-label').style.color = c.model_pred === 'BIPD' ? '#ff66ff' : '#44ff66';
  document.getElementById('pred-prob').textContent = c.prob.toFixed(3);
  document.getElementById('pred-original').textContent = c.original_label;
  document.getElementById('pred-original').style.color = c.original_label === 'BIPD' ? '#ff66ff' : '#44ff66';

  const panel = document.getElementById('prediction-panel');
  panel.className = c.model_pred === 'BIPD' ? 'pred-bipd' : 'pred-gpd';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  const nLabeled = Object.keys(allLabels).length;
  document.getElementById('labeled-count').textContent = nLabeled + ' labeled';

  // Show current label
  const saved = allLabels[c.patient_id];
  userLabel = saved ? saved.label : null;
  document.getElementById('btn-gpd').className = 'label-btn' + (userLabel === 'GPD' ? ' active-gpd' : '');
  document.getElementById('btn-bipd').className = 'label-btn' + (userLabel === 'BIPD' ? ' active-bipd' : '');
  document.getElementById('btn-lpd').className = 'label-btn' + (userLabel === 'LPD' ? ' active-lpd' : '');
  document.getElementById('label-status').textContent = userLabel ? 'Labeled: ' + userLabel : '';
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  drawEEG();
  updateInfo();
}}

function exportJSON() {{
  const out = {{}};
  for (const c of CASES) {{
    const pid = c.patient_id;
    if (allLabels[pid]) {{
      out[pid] = {{
        patient_id: pid,
        label: allLabels[pid].label,
        prob: c.prob,
        model_pred: c.model_pred,
        original_label: c.original_label,
        left_n: c.left_times.length,
        right_n: c.right_times.length,
        left_freq: c.left_freq,
        right_freq: c.right_freq,
        source: 'bipd_reviewer_v1',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'bipd_review_labels.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' cases';
  el.style.color = '#4f4';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === '1') {{ setLabel('GPD'); }}
  else if (e.key === '2') {{ setLabel('BIPD'); }}
  else if (e.key === '3') {{ setLabel('LPD'); }}
  else if (e.key === '4') {{ setLabel('REJECT'); }}
  else if (e.key === 'ArrowLeft') {{ e.preventDefault(); idx = Math.max(0, idx - 1); show(); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); idx = Math.min(CASES.length - 1, idx + 1); show(); }}
  else if (e.key === 'e' || e.key === 'E') {{ exportJSON(); }}
}});

show();
</script>
</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
