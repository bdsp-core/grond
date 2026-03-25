"""
HPP-Assisted BIPD Discharge Labeling Tool.

Like the LPD labeler but with separate timing for left and right hemispheres.
Channels reordered: left lateral + left parasagittal | right lateral + right parasagittal | midline.
Two sets of frequency buttons and markers (red=left, blue=right).

Usage:
    conda run -n foe python code/generate_bipd_labeler.py
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

from optimization_harness_v2 import LEFT_INDICES, RIGHT_INDICES, FS
from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
LOWPASS_HZ = 20.0

# Original channel order
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',   # 0-3 left lateral
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',   # 4-7 right lateral
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',   # 8-11 left parasagittal
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',   # 12-15 right parasagittal
    'Fz-Cz', 'Cz-Pz',                        # 16-17 midline
]

# Reordered for BIPD: left hemi together, right hemi together
DISPLAY_ORDER = [
    0, 1, 2, 3,       # left lateral
    8, 9, 10, 11,     # left parasagittal
    -1,               # gap
    4, 5, 6, 7,       # right lateral
    12, 13, 14, 15,   # right parasagittal
    -1,               # gap
    16, 17,           # midline
]

FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(1, 19)]  # 0.25 to 4.5


def hpp_with_freq(evidence, fs, freq_hz):
    """Run HPP with a given frequency prior."""
    if len(evidence) < 10 or freq_hz <= 0:
        return []
    freq_estimate = np.clip(freq_hz, 0.2, 5.0)
    active_start, active_end = _detect_active_interval(evidence, fs)
    candidates = _extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    if len(candidates) == 0:
        return []
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)
    if len(discharge_samples) == 0:
        return []
    if len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)
    times = (discharge_samples / fs).tolist()
    return [t for t in times if 0 <= t <= DURATION]


def load_segment(patient_id):
    """Load EEG segment."""
    for suffix in ['_seg000.mat', '.mat']:
        path = EEG_DIR / f'{patient_id}{suffix}'
        if path.exists():
            mat = sio.loadmat(str(path))
            data_key = [k for k in mat.keys() if not k.startswith('_')][0]
            seg = mat[data_key]
            if seg.shape[0] > seg.shape[1]:
                seg = seg.T
            return seg[:18, :2000] if seg.shape[0] >= 18 else None
    return None


def compute_hemi_evidence(seg, side):
    """Compute aggregated evidence for one hemisphere."""
    indices = LEFT_INDICES if side == 'left' else RIGHT_INDICES
    n_samples = seg.shape[1]
    evidence_all = np.zeros((len(indices), n_samples))
    for i, ch in enumerate(indices):
        evidence_all[i] = _compute_channel_evidence(seg[ch], FS)
    return np.median(evidence_all, axis=0)


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
    print("  BIPD Discharge Labeling Tool")
    print("  Separate left/right hemisphere timing")
    print("=" * 70)

    # Load BIPD manifest
    bipd_path = LABELS_DIR / 'bipd_harvest_manifest.json'
    with open(str(bipd_path)) as f:
        bipd_manifest = json.load(f)

    # Also add the BIPD case from LPD labeling
    extra_bipd = {'sub-S0002119210399_20191215002852': {'subtype': 'bipd', 'est_freq': 3.3}}
    all_bipd = {**bipd_manifest, **extra_bipd}
    print(f"Total BIPD cases: {len(all_bipd)}")

    sorted_pids = sorted(all_bipd.keys())

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    print(f"\nProcessing {len(sorted_pids)} cases...")
    cases_data = []
    n_skipped = 0

    for i, pid in enumerate(sorted_pids):
        seg = load_segment(pid)
        if seg is None:
            n_skipped += 1
            continue

        # Lowpass filter for display
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except:
                seg_display[ch] = seg[ch]

        # Compute evidence per hemisphere
        ev_left = compute_hemi_evidence(seg, 'left')
        ev_right = compute_hemi_evidence(seg, 'right')

        # Normalize for display
        ev_left_max = np.max(ev_left) if np.max(ev_left) > 0 else 1.0
        ev_right_max = np.max(ev_right) if np.max(ev_right) > 0 else 1.0

        # Precompute HPP for each freq, each hemisphere
        hpp_left = {}
        hpp_right = {}
        for freq in FREQ_BUTTONS:
            hpp_left[str(freq)] = hpp_with_freq(ev_left, FS, freq)
            hpp_right[str(freq)] = hpp_with_freq(ev_right, FS, freq)

        case = {
            'patient_id': str(pid),
            'est_freq': all_bipd[pid].get('est_freq') or 0,
            'eeg_data': downsample(seg_display, 1000),
            'evidence_left': downsample(ev_left / ev_left_max, 500),
            'evidence_right': downsample(ev_right / ev_right_max, 500),
            'hpp_left': hpp_left,
            'hpp_right': hpp_right,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sorted_pids)} processed")

    print(f"  Total cases: {len(cases_data)} (skipped {n_skipped})")

    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)
    out_path = OUT_DIR / 'bipd_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


def build_html(cases_data):
    freq_btns_json = json.dumps(FREQ_BUTTONS)
    display_order_json = json.dumps(DISPLAY_ORDER)
    channel_names_json = json.dumps(BIPOLAR_CHANNELS)
    cases_json = json.dumps(cases_data, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BIPD Discharge Labeling — Left/Right Hemispheres</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }}
  #header {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #222; border-bottom: 2px solid #444; flex-wrap: wrap; gap: 8px; }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
  #progress-bar-wrap {{ width: 100%; height: 6px; background: #333; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}
  #mode-indicator {{ font-size: 18px; font-weight: bold; padding: 6px 20px; text-align: center; letter-spacing: 2px; }}
  .mode-left {{ background: #3a1a1a; color: #ff6666; }}
  .mode-right {{ background: #1a1a3a; color: #6688ff; }}
  .mode-nav {{ background: #1a1a3a; color: #6688ff; }}
  .mode-delete {{ background: #3a1a1a; color: #ff4444; }}
  #info-panel {{ background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px; }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}
  .freq-row {{ display: flex; flex-wrap: wrap; gap: 4px; padding: 4px 16px; background: #252525; align-items: center; }}
  .freq-row label {{ font-size: 13px; margin-right: 8px; font-weight: bold; min-width: 100px; }}
  .freq-btn {{ padding: 5px 9px; border: 1px solid #555; border-radius: 4px; background: #333; color: #ccc; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; min-width: 40px; text-align: center; }}
  .freq-btn:hover {{ background: #444; border-color: #888; color: #fff; }}
  .freq-btn.active-left {{ background: #4a2020; border-color: #ff6666; color: #ff8888; }}
  .freq-btn.active-right {{ background: #20204a; border-color: #6688ff; color: #88aaff; }}
  #canvas-container {{ text-align: center; padding: 8px; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}
  #evidence-canvas {{ display: block; margin: 4px auto 0 auto; cursor: crosshair; }}
  .export-btn {{ padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px; background: #2a3a2a; color: #44cc44; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; }}
  .export-btn:hover {{ background: #3a4a3a; }}
  .skip-btn {{ padding: 6px 14px; border: 1px solid #ff6644; border-radius: 4px; background: #3a2a2a; color: #ff6644; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; }}
  #save-status {{ color: #44cc44; font-size: 13px; }}
  #shortcuts {{ font-size: 12px; color: #777; padding: 6px 16px; background: #222; border-top: 1px solid #333; line-height: 1.8; }}
  .hemi-divider {{ height: 1px; background: #444; margin: 2px 0; }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#e040e0;">BIPD Labeling — Left/Right Hemispheres</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="skip-btn" onclick="skipCase()">Skip/Reject <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item" style="color:#ff6666;">Left markers: <strong id="info-left-count">0</strong></span>
  <span class="info-item" style="color:#6688ff;">Right markers: <strong id="info-right-count">0</strong></span>
  <span class="info-item">Left IPI: <strong id="info-left-freq">--</strong></span>
  <span class="info-item">Right IPI: <strong id="info-right-freq">--</strong></span>
  <span class="info-item">Mode: <strong id="info-mode">Navigate</strong></span>
</div>

<div class="freq-row" id="freq-row-left">
  <label style="color:#ff6666;">LEFT freq:</label>
</div>
<div class="freq-row" id="freq-row-right">
  <label style="color:#6688ff;">RIGHT freq:</label>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">L</span> Add LEFT markers &nbsp;&nbsp;
  <span class="key">R</span> Add RIGHT markers &nbsp;&nbsp;
  <span class="key">D</span> Delete mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">X</span> Skip/Reject &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">C</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">E</span> Export
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};
const DISPLAY_ORDER = {display_order_json};
const ALL_CHANNEL_NAMES = {channel_names_json};
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1200;
const EEG_HEIGHT = 750;
const EV_HEIGHT = 180;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

const LEFT_COLOR = '#ff4444';
const RIGHT_COLOR = '#4488ff';

let idx = 0;
let mode = 'nav'; // 'nav','add_left','add_right','delete'
let markersLeft = [];
let markersRight = [];
let undoStack = [];
let hoverMarker = {{side: null, idx: -1}};
let selFreqLeft = null;
let selFreqRight = null;
let reviewed = new Set();

const STORAGE_KEY = 'bipd_labeler_v1';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

// Build freq buttons
(function() {{
  for (const side of ['left', 'right']) {{
    const container = document.getElementById('freq-row-' + side);
    for (const freq of FREQ_BUTTONS) {{
      const btn = document.createElement('button');
      btn.className = 'freq-btn';
      btn.textContent = freq.toFixed(2);
      btn.dataset.freq = freq;
      btn.dataset.side = side;
      btn.onclick = () => selectFreq(side, freq);
      container.appendChild(btn);
    }}
  }}
}})();

function getDisplayChannels() {{
  return DISPLAY_ORDER.map(idx => ({{
    idx: idx,
    name: idx >= 0 ? ALL_CHANNEL_NAMES[idx] : '',
    side: idx >= 0 ? ([0,1,2,3,8,9,10,11].includes(idx) ? 'left' : [4,5,6,7,12,13,14,15].includes(idx) ? 'right' : 'mid') : 'gap'
  }}));
}}
const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}
function xToTime(x) {{ return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }}

function findNearestMarker(x) {{
  let best = {{side: null, idx: -1}};
  let bestDist = Infinity;
  for (let i = 0; i < markersLeft.length; i++) {{
    const d = Math.abs(timeToX(markersLeft[i]) - x);
    if (d < bestDist) {{ bestDist = d; best = {{side:'left', idx:i}}; }}
  }}
  for (let i = 0; i < markersRight.length; i++) {{
    const d = Math.abs(timeToX(markersRight[i]) - x);
    if (d < bestDist) {{ bestDist = d; best = {{side:'right', idx:i}}; }}
  }}
  return bestDist <= 20 ? best : {{side: null, idx: -1}};
}}

function pushUndo() {{
  undoStack.push({{left: [...markersLeft], right: [...markersRight]}});
  if (undoStack.length > 100) undoStack.shift();
}}
function undo() {{
  if (undoStack.length === 0) return;
  const prev = undoStack.pop();
  markersLeft = prev.left;
  markersRight = prev.right;
  redraw();
}}

function selectFreq(side, freq) {{
  const c = CASES[idx];
  const hppKey = side === 'left' ? 'hpp_left' : 'hpp_right';
  const times = c[hppKey][String(freq)] || [];
  pushUndo();
  if (side === 'left') {{
    markersLeft = [...times];
    selFreqLeft = freq;
  }} else {{
    markersRight = [...times];
    selFreqRight = freq;
  }}
  // Highlight buttons
  const cls = side === 'left' ? 'active-left' : 'active-right';
  document.querySelectorAll('#freq-row-' + side + ' .freq-btn').forEach(btn => {{
    btn.classList.remove(cls);
    if (parseFloat(btn.dataset.freq) === freq) btn.classList.add(cls);
  }});
  redraw();
}}

function ipiFreq(markers) {{
  if (markers.length < 2) return '--';
  const sorted = [...markers].sort((a,b) => a-b);
  const ipis = [];
  for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i-1]);
  ipis.sort((a,b) => a-b);
  const med = ipis[Math.floor(ipis.length/2)];
  return med > 0 ? (1/med).toFixed(2) + ' Hz' : '--';
}}

function drawMarkers(ctx, markers, color, topY, bottomY, isHover, hoverIdx) {{
  for (let i = 0; i < markers.length; i++) {{
    const x = timeToX(markers[i]);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    const isH = isHover && hoverIdx === i;
    ctx.strokeStyle = isH ? color.replace('0.6', '0.9') : color;
    ctx.lineWidth = isH ? 4 : 2;
    ctx.beginPath(); ctx.moveTo(x, topY); ctx.lineTo(x, bottomY); ctx.stroke();
  }}
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH; canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd'; ctx.lineWidth = 0.5; ctx.setLineDash([4,4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT-MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Compute hemisphere Y boundaries from display order
  // Left channels are display indices 0-7 (8 channels), then gap at 8
  // Right channels are display indices 9-16 (8 channels), then gap at 17
  // Midline at 18-19
  const leftTopY = MARGIN_TOP;
  const leftBottomY = MARGIN_TOP + chSpacing * 9; // includes 8 channels + space before gap
  const rightTopY = leftBottomY;
  const rightBottomY = MARGIN_TOP + chSpacing * 18;
  const leftEndY = leftBottomY;
  const rightStartY = rightTopY;
  const rightEndY = rightBottomY;

  // Store globally for marker drawing
  window._leftTopY = leftTopY;
  window._leftBottomY = leftBottomY;
  window._rightTopY = rightTopY;
  window._rightBottomY = rightBottomY;
  ctx.fillStyle = 'rgba(255, 200, 200, 0.06)';
  ctx.fillRect(PLOT_LEFT, MARGIN_TOP, PLOT_W, leftEndY - MARGIN_TOP);
  ctx.fillStyle = 'rgba(200, 200, 255, 0.06)';
  ctx.fillRect(PLOT_LEFT, rightStartY, PLOT_W, rightEndY - rightStartY);

  // Traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];

    ctx.strokeStyle = ch.side === 'left' ? '#440000' : ch.side === 'right' ? '#000044' : '#000000';
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (let si = 0; si < nSamples; si++) {{
      const x = PLOT_LEFT + (si / (nSamples - 1)) * PLOT_W;
      let val = Math.max(-CLIP_UV, Math.min(CLIP_UV, trace[si]));
      const y = yCenter - val * Z_SCALE * chSpacing;
      if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Channel labels
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillStyle = ch.side === 'left' ? '#cc0000' : ch.side === 'right' ? '#0044cc' : '#000000';
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000'; ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) ctx.fillText(s+'s', timeToX(s), EEG_HEIGHT-MARGIN_BOTTOM+4);

  // Title
  ctx.fillStyle = '#000'; ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(c.patient_id + '  |  BIPD', EEG_WIDTH/2, 6);

  // Hemisphere labels
  ctx.font = 'bold 12px Consolas'; ctx.textAlign = 'left';
  ctx.fillStyle = '#cc0000'; ctx.fillText('LEFT', PLOT_LEFT + 5, MARGIN_TOP + 2);
  ctx.fillStyle = '#0044cc'; ctx.fillText('RIGHT', PLOT_LEFT + 5, rightStartY + 2);

  // Left markers (red) — only span left hemisphere channels
  drawMarkers(ctx, markersLeft, 'rgba(255,0,0,0.6)', leftTopY, leftBottomY,
    mode==='delete' && hoverMarker.side==='left', hoverMarker.idx);
  // Right markers (blue) — only span right hemisphere channels
  drawMarkers(ctx, markersRight, 'rgba(0,80,255,0.6)', rightTopY, rightBottomY,
    mode==='delete' && hoverMarker.side==='right', hoverMarker.idx);
}}

function drawEvidence() {{
  const canvas = document.getElementById('evidence-canvas');
  canvas.width = EEG_WIDTH; canvas.height = EV_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];

  const evTop = 10; const evBottom = EV_HEIGHT - 20; const evH = evBottom - evTop;
  const midY = evTop + evH / 2;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EV_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 0.5; ctx.setLineDash([4,4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // Divider
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PLOT_LEFT, midY); ctx.lineTo(PLOT_RIGHT, midY); ctx.stroke();

  // Left evidence (top half, red)
  const evL = c.evidence_left;
  if (evL && evL.length > 0) {{
    const halfH = (midY - evTop);
    ctx.fillStyle = 'rgba(255,100,100,0.15)';
    ctx.beginPath(); ctx.moveTo(PLOT_LEFT, midY);
    for (let i = 0; i < evL.length; i++) {{
      const x = PLOT_LEFT + (i/(evL.length-1))*PLOT_W;
      ctx.lineTo(x, midY - evL[i]*halfH);
    }}
    ctx.lineTo(PLOT_RIGHT, midY); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#cc3333'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < evL.length; i++) {{
      const x = PLOT_LEFT + (i/(evL.length-1))*PLOT_W;
      if (i===0) ctx.moveTo(x, midY - evL[i]*halfH); else ctx.lineTo(x, midY - evL[i]*halfH);
    }}
    ctx.stroke();
  }}

  // Right evidence (bottom half, blue, inverted)
  const evR = c.evidence_right;
  if (evR && evR.length > 0) {{
    const halfH = (evBottom - midY);
    ctx.fillStyle = 'rgba(100,100,255,0.15)';
    ctx.beginPath(); ctx.moveTo(PLOT_LEFT, midY);
    for (let i = 0; i < evR.length; i++) {{
      const x = PLOT_LEFT + (i/(evR.length-1))*PLOT_W;
      ctx.lineTo(x, midY + evR[i]*halfH);
    }}
    ctx.lineTo(PLOT_RIGHT, midY); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = '#3333cc'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < evR.length; i++) {{
      const x = PLOT_LEFT + (i/(evR.length-1))*PLOT_W;
      if (i===0) ctx.moveTo(x, midY + evR[i]*halfH); else ctx.lineTo(x, midY + evR[i]*halfH);
    }}
    ctx.stroke();
  }}

  // Labels
  ctx.font = '10px monospace'; ctx.textAlign = 'left';
  ctx.fillStyle = '#cc3333'; ctx.fillText('LEFT evidence', PLOT_LEFT+4, evTop+10);
  ctx.fillStyle = '#3333cc'; ctx.fillText('RIGHT evidence', PLOT_LEFT+4, evBottom-4);

  // Left markers on evidence
  drawMarkers(ctx, markersLeft, 'rgba(255,0,0,0.5)', evTop, midY,
    mode==='delete' && hoverMarker.side==='left', hoverMarker.idx);
  drawMarkers(ctx, markersRight, 'rgba(0,80,255,0.5)', midY, evBottom,
    mode==='delete' && hoverMarker.side==='right', hoverMarker.idx);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-left-count').textContent = markersLeft.length;
  document.getElementById('info-right-count').textContent = markersRight.length;
  document.getElementById('info-left-freq').textContent = ipiFreq(markersLeft);
  document.getElementById('info-right-freq').textContent = ipiFreq(markersRight);
  let modeStr = 'Navigate';
  if (mode === 'add_left') modeStr = 'Add LEFT';
  else if (mode === 'add_right') modeStr = 'Add RIGHT';
  else if (mode === 'delete') modeStr = 'Delete';
  document.getElementById('info-mode').textContent = modeStr;
  document.getElementById('counter').textContent = (idx+1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx+1)/CASES.length*100).toFixed(1) + '%';
  document.getElementById('labeled-count').textContent = reviewed.size + ' labeled';

  const el = document.getElementById('mode-indicator');
  if (mode === 'add_left') {{ el.textContent = 'ADD LEFT (L) — click to add red marker'; el.className = 'mode-left'; }}
  else if (mode === 'add_right') {{ el.textContent = 'ADD RIGHT (R) — click to add blue marker'; el.className = 'mode-right'; }}
  else if (mode === 'delete') {{ el.textContent = 'DELETE MODE (D) — click near marker to remove'; el.className = 'mode-delete'; }}
  else {{ el.textContent = 'NAVIGATE MODE'; el.className = 'mode-nav'; }}
}}

function redraw() {{ drawEEG(); drawEvidence(); updateInfo(); }}

function autoSave() {{
  const c = CASES[idx];
  allLabels[c.patient_id] = {{
    left_times: [...markersLeft].sort((a,b)=>a-b),
    right_times: [...markersRight].sort((a,b)=>a-b),
    sel_freq_left: selFreqLeft,
    sel_freq_right: selFreqRight,
    rejected: reviewed.has(c.patient_id) && markersLeft.length===0 && markersRight.length===0,
  }};
  if (reviewed.has(c.patient_id)) saveAll();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length-1));
  const c = CASES[idx];
  if (allLabels[c.patient_id]) {{
    markersLeft = [...(allLabels[c.patient_id].left_times || [])];
    markersRight = [...(allLabels[c.patient_id].right_times || [])];
    selFreqLeft = allLabels[c.patient_id].sel_freq_left;
    selFreqRight = allLabels[c.patient_id].sel_freq_right;
    if (markersLeft.length > 0 || markersRight.length > 0 || allLabels[c.patient_id].rejected) reviewed.add(c.patient_id);
  }} else {{
    markersLeft = []; markersRight = [];
    selFreqLeft = null; selFreqRight = null;
  }}
  undoStack = []; hoverMarker = {{side:null, idx:-1}};
  // Reset button highlights
  document.querySelectorAll('.freq-btn').forEach(btn => btn.classList.remove('active-left','active-right'));
  if (selFreqLeft) document.querySelectorAll('#freq-row-left .freq-btn').forEach(btn => {{
    if (parseFloat(btn.dataset.freq)===selFreqLeft) btn.classList.add('active-left');
  }});
  if (selFreqRight) document.querySelectorAll('#freq-row-right .freq-btn').forEach(btn => {{
    if (parseFloat(btn.dataset.freq)===selFreqRight) btn.classList.add('active-right');
  }});
  redraw();
}}

function skipCase() {{
  const c = CASES[idx];
  markersLeft = []; markersRight = [];
  reviewed.add(c.patient_id);
  autoSave(); saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED'; el.style.color = '#ff6644';
  setTimeout(() => {{ el.textContent = ''; }}, 1500);
  idx = Math.min(CASES.length-1, idx+1); show();
}}

function exportJSON() {{
  autoSave(); saveAll();
  const out = {{}};
  for (const c of CASES) {{
    const pid = c.patient_id;
    if (allLabels[pid]) {{
      const a = allLabels[pid];
      const rej = a.rejected === true;
      out[pid] = {{
        patient_id: pid, subtype: 'bipd',
        left_times: a.left_times || [],
        right_times: a.right_times || [],
        sel_freq_left: a.sel_freq_left,
        sel_freq_right: a.sel_freq_right,
        review_status: rej ? 'rejected' : 'ground_truth',
        rejected: rej, source: 'bipd_labeler',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'bipd_labeler_discharge_times.json'; a.click();
  document.getElementById('save-status').textContent = 'Exported ' + Object.keys(out).length;
}}

// Canvas events
const eegCanvas = document.getElementById('eeg-canvas');
const evCanvas = document.getElementById('evidence-canvas');

function handleClick(e) {{
  const rect = eegCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (EEG_WIDTH / rect.width);
  if (x < PLOT_LEFT || x > PLOT_RIGHT) return;
  const t = xToTime(x);
  if (mode === 'add_left' && t >= 0 && t <= DURATION) {{
    pushUndo(); markersLeft.push(t); redraw();
  }} else if (mode === 'add_right' && t >= 0 && t <= DURATION) {{
    pushUndo(); markersRight.push(t); redraw();
  }} else if (mode === 'delete') {{
    const m = findNearestMarker(x);
    if (m.side && m.idx >= 0) {{
      pushUndo();
      if (m.side === 'left') markersLeft.splice(m.idx, 1);
      else markersRight.splice(m.idx, 1);
      redraw();
    }}
  }}
}}

function handleMove(e) {{
  if (mode !== 'delete') return;
  const rect = eegCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (EEG_WIDTH / rect.width);
  const m = findNearestMarker(x);
  if (m.side !== hoverMarker.side || m.idx !== hoverMarker.idx) {{
    hoverMarker = m; redraw();
  }}
}}

eegCanvas.addEventListener('click', handleClick);
eegCanvas.addEventListener('mousemove', handleMove);
evCanvas.addEventListener('click', handleClick);
evCanvas.addEventListener('mousemove', handleMove);

document.addEventListener('keydown', (e) => {{
  if (e.key === 'l' || e.key === 'L') {{ mode = mode==='add_left' ? 'nav' : 'add_left'; redraw(); }}
  else if (e.key === 'r' || e.key === 'R') {{ mode = mode==='add_right' ? 'nav' : 'add_right'; redraw(); }}
  else if (e.key === 'd' || e.key === 'D') {{ mode = mode==='delete' ? 'nav' : 'delete'; redraw(); }}
  else if (e.key === 'Escape') {{ mode = 'nav'; redraw(); }}
  else if (e.key === 'z' || e.key === 'Z') {{ undo(); }}
  else if (e.key === 'x' || e.key === 'X') {{ skipCase(); }}
  else if (e.key === 'ArrowLeft') {{ e.preventDefault(); autoSave(); saveAll(); idx = Math.max(0, idx-1); show(); }}
  else if (e.key === 'ArrowRight' || e.key === 'c' || e.key === 'C' || e.key === 'Enter') {{
    e.preventDefault();
    if (e.key !== 'ArrowLeft') reviewed.add(CASES[idx].patient_id);
    autoSave(); saveAll();
    idx = Math.min(CASES.length-1, idx+1); show();
  }}
  else if (e.key === 'e' || e.key === 'E') {{ exportJSON(); }}
}});

show();
</script>
</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
