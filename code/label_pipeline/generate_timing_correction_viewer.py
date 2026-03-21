"""
Generate interactive HTML timing correction viewer for cases needing review.

Loads discharge_times_hpp.json and includes all cases where review_status
is NOT 'ground_truth' (i.e., auto-detected cases that haven't been manually
verified). MW can add, delete, and move discharge timing markers interactively.
EEG is rendered in HTML5 canvas for exact coordinate alignment.

Usage:
    conda run -n foe python code/label_pipeline/generate_timing_correction_viewer.py
"""

import sys, json, math, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ── Path setup ────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

GROUP_BREAKS = {4, 8, 12, 16}



def lowpass_filter(data, fs, cutoff=20.0):
    """Apply 4th-order Butterworth lowpass filter."""
    nyq = fs / 2.0
    if nyq <= cutoff:
        return data
    b, a = butter(4, cutoff / nyq, btype='low')
    out = np.copy(data)
    for i in range(out.shape[0]):
        try:
            out[i, :] = filtfilt(b, a, out[i, :])
        except ValueError:
            pass
    return out


def downsample(arr, target_len):
    """Downsample 1D or 2D array to target_len along last axis."""
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


def build_html(cases_data):
    """Build the interactive timing correction HTML viewer."""

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Discharge Timing Correction Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }

  #header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; border-bottom: 2px solid #444;
    flex-wrap: wrap; gap: 8px;
  }
  #header-left { display: flex; align-items: center; gap: 12px; }
  #header-right { display: flex; align-items: center; gap: 12px; font-size: 13px; }

  .key { background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }

  #progress-bar-wrap { width: 100%; height: 6px; background: #333; }
  #progress-bar { height: 100%; background: #44cc88; transition: width 0.2s; }

  #mode-indicator {
    font-size: 22px; font-weight: bold; padding: 8px 20px;
    text-align: center; letter-spacing: 2px;
  }
  .mode-add { background: #1a3a1a; color: #44ff66; }
  .mode-delete { background: #3a1a1a; color: #ff4444; }
  .mode-move { background: #3a3a1a; color: #ffcc00; }

  #info-panel {
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: flex-start;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }
  .info-col { display: flex; flex-direction: column; gap: 3px; }
  .info-item { color: #bbb; }
  .info-item strong { color: #eee; }
  .info-badge {
    padding: 4px 12px; border-radius: 5px; font-size: 13px; font-weight: bold;
  }
  .badge-lpd { background: #5a2020; color: #ff8888; }
  .badge-gpd { background: #20205a; color: #8888ff; }

  #canvas-container { text-align: center; padding: 8px; position: relative; }
  #eeg-canvas { cursor: crosshair; display: block; margin: 0 auto; }
  #evidence-canvas { display: block; margin: 4px auto 0 auto; cursor: crosshair; }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }

  .export-btn {
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }
  .export-btn:hover { background: #3a4a3a; }

  #save-status { color: #44cc44; font-size: 13px; }

  #original-times { font-size: 11px; color: #888; padding: 2px 16px; background: #252525; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Timing Correction Viewer</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
    <span id="progress-text" style="font-size:12px; color:#aaa;"></span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportJSON()">Export JSON <span class="key">E</span></button>
    <span id="save-status"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-add">ADD MODE (A)</div>

<div id="info-panel">
  <div class="info-col">
    <span class="info-badge" id="info-subtype-badge">--</span>
  </div>
  <div class="info-col">
    <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
    <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
    <span class="info-item">Gold freq: <strong id="info-gold-freq">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Markers: <strong id="info-n-markers">--</strong></span>
    <span class="info-item">Median IPI: <strong id="info-median-ipi">--</strong></span>
    <span class="info-item">Est. freq: <strong id="info-est-freq">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Marker times: <strong id="info-marker-times">--</strong></span>
  </div>
</div>

<div id="original-times">Original HPP times: <strong id="info-original-times">--</strong></div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">A</span> Add mode &nbsp;&nbsp;
  <span class="key">D</span> Delete mode &nbsp;&nbsp;
  <span class="key">M</span> Move mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">Esc</span> Cancel move &nbsp;&nbsp;
  <span class="key">Enter</span> Save &amp; next &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = CASES_PLACEHOLDER;

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const GROUP_BREAKS = new Set([4, 8, 12, 16]);
const DURATION = 10.0; // seconds
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

// Canvas layout constants
const EEG_WIDTH = 1200;
const EEG_HEIGHT = 800;
const EV_HEIGHT = 120;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_TOP = MARGIN_TOP;
const PLOT_BOTTOM = EEG_HEIGHT - MARGIN_BOTTOM;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = PLOT_BOTTOM - PLOT_TOP;

// State
let idx = 0;
let mode = 'add'; // 'add', 'delete', 'move'
let markers = []; // current marker times (seconds)
let undoStack = [];
let moveSelected = -1; // index of marker being moved, or -1
let hoverMarker = -1;  // for delete mode highlight
let mouseX = -1;       // for move mode follow

// Corrections stored per patient
let corrections = {};
try { corrections = JSON.parse(localStorage.getItem('timing_corrections') || '{}'); } catch(e) { corrections = {}; }

function saveCorrections() {
  localStorage.setItem('timing_corrections', JSON.stringify(corrections));
}

function timeToX(t) { return PLOT_LEFT + (t / DURATION) * PLOT_W; }
function xToTime(x) { return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }
function evTimeToX(t) { return PLOT_LEFT + (t / DURATION) * PLOT_W; }

// Build display channel list with spacers
function getDisplayChannels() {
  const dc = [];
  for (let i = 0; i < 18; i++) {
    if (GROUP_BREAKS.has(i)) dc.push({ idx: -1, name: '' }); // spacer
    dc.push({ idx: i, name: CHANNEL_NAMES[i] });
  }
  return dc;
}

const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length; // 18 + 4 spacers = 22

function drawEEG() {
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const eegData = c.eeg_data; // 18 x N array
  const nSamples = eegData[0].length;

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  // Gridlines (1-second intervals)
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, PLOT_TOP); ctx.lineTo(x, PLOT_BOTTOM); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Channel spacing
  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Draw traces
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 0.7;
  for (let di = 0; di < N_DISPLAY; di++) {
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue; // spacer

    const yCenter = PLOT_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];

    ctx.beginPath();
    for (let si = 0; si < nSamples; si++) {
      const x = PLOT_LEFT + (si / (nSamples - 1)) * PLOT_W;
      let val = trace[si];
      val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
      const y = yCenter - val * Z_SCALE * chSpacing;
      if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Channel labels
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = PLOT_TOP + chSpacing * (di + 1);
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }

  // Time axis labels
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {
    ctx.fillText(s + 's', timeToX(s), PLOT_BOTTOM + 4);
  }

  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const title = c.patient_id + '  |  ' + c.subtype.toUpperCase() + '  |  gold=' + c.gold_freq.toFixed(2) + ' Hz';
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Draw original HPP markers (thin gray dashed)
  const origTimes = c.original_times || [];
  ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  for (const t of origTimes) {
    const x = timeToX(t);
    if (x >= PLOT_LEFT && x <= PLOT_RIGHT) {
      ctx.beginPath(); ctx.moveTo(x, PLOT_TOP); ctx.lineTo(x, PLOT_BOTTOM); ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Draw current markers
  for (let mi = 0; mi < markers.length; mi++) {
    let t = markers[mi];
    // In move mode, the selected marker follows mouse
    if (mode === 'move' && moveSelected === mi && mouseX >= 0) {
      t = xToTime(mouseX);
      t = Math.max(0, Math.min(DURATION, t));
    }
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.6)';
    let lw = 2;

    if (mode === 'move' && moveSelected === mi) {
      color = 'rgba(255, 204, 0, 0.8)';
      lw = 3;
    } else if (mode === 'delete' && hoverMarker === mi) {
      color = 'rgba(255, 50, 50, 0.9)';
      lw = 4;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x, PLOT_TOP); ctx.lineTo(x, PLOT_BOTTOM); ctx.stroke();

    // Small time label at top
    ctx.fillStyle = color;
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(t.toFixed(2) + 's', x, PLOT_TOP - 2);
  }
}

function drawEvidence() {
  const canvas = document.getElementById('evidence-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EV_HEIGHT;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const ev = c.evidence_signal;
  if (!ev || ev.length === 0) {
    ctx.fillStyle = '#2a2a2a';
    ctx.fillRect(0, 0, EEG_WIDTH, EV_HEIGHT);
    ctx.fillStyle = '#888';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No evidence signal', EEG_WIDTH / 2, EV_HEIGHT / 2);
    return;
  }

  const nSamples = ev.length;
  const evTop = 10;
  const evBottom = EV_HEIGHT - 20;
  const evH = evBottom - evTop;

  // White bg
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EV_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }
  ctx.setLineDash([]);

  // Find max for scaling
  let evMax = 0;
  for (let i = 0; i < nSamples; i++) { if (ev[i] > evMax) evMax = ev[i]; }
  if (evMax < 1e-6) evMax = 1;

  // Fill
  ctx.fillStyle = 'rgba(70, 130, 180, 0.3)';
  ctx.beginPath();
  ctx.moveTo(PLOT_LEFT, evBottom);
  for (let i = 0; i < nSamples; i++) {
    const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
    const y = evBottom - (ev[i] / evMax) * evH;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(PLOT_RIGHT, evBottom);
  ctx.closePath();
  ctx.fill();

  // Line
  ctx.strokeStyle = 'steelblue';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < nSamples; i++) {
    const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
    const y = evBottom - (ev[i] / evMax) * evH;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Label
  ctx.fillStyle = '#000';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('E(t)', PLOT_LEFT - 4, (evTop + evBottom) / 2);

  // Time labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {
    ctx.fillText(s + 's', timeToX(s), evBottom + 3);
  }

  // Original markers (gray)
  const origTimes = CASES[idx].original_times || [];
  ctx.strokeStyle = 'rgba(150, 150, 150, 0.5)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  for (const t of origTimes) {
    const x = timeToX(t);
    if (x >= PLOT_LEFT && x <= PLOT_RIGHT) {
      ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
    }
  }
  ctx.setLineDash([]);

  // Current markers
  for (let mi = 0; mi < markers.length; mi++) {
    let t = markers[mi];
    if (mode === 'move' && moveSelected === mi && mouseX >= 0) {
      t = xToTime(mouseX);
      t = Math.max(0, Math.min(DURATION, t));
    }
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.6)';
    let lw = 2;
    if (mode === 'move' && moveSelected === mi) { color = 'rgba(255, 204, 0, 0.8)'; lw = 3; }
    else if (mode === 'delete' && hoverMarker === mi) { color = 'rgba(255, 50, 50, 0.9)'; lw = 4; }

    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }
}

function updateInfoPanel() {
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-gold-freq').textContent = c.gold_freq.toFixed(2) + ' Hz';

  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype.toLowerCase();

  // Markers info (live)
  const sorted = [...markers].sort((a, b) => a - b);
  document.getElementById('info-n-markers').textContent = sorted.length;
  document.getElementById('info-marker-times').textContent =
    sorted.length > 0 ? sorted.map(t => t.toFixed(2) + 's').join(', ') : 'none';

  if (sorted.length >= 2) {
    const ipis = [];
    for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i - 1]);
    ipis.sort((a, b) => a - b);
    const medianIPI = ipis[Math.floor(ipis.length / 2)];
    document.getElementById('info-median-ipi').textContent = medianIPI.toFixed(3) + ' s';
    document.getElementById('info-est-freq').textContent = (1 / medianIPI).toFixed(3) + ' Hz';
  } else {
    document.getElementById('info-median-ipi').textContent = '--';
    document.getElementById('info-est-freq').textContent = '--';
  }

  // Original times
  const orig = c.original_times || [];
  document.getElementById('info-original-times').textContent =
    orig.length > 0 ? orig.map(t => t.toFixed(2) + 's').join(', ') : 'none';
}

function updateModeIndicator() {
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') { el.textContent = 'ADD MODE (A)'; el.className = 'mode-add'; }
  else if (mode === 'delete') { el.textContent = 'DELETE MODE (D)'; el.className = 'mode-delete'; }
  else if (mode === 'move') {
    if (moveSelected >= 0) el.textContent = 'MOVE MODE (M) — click to place';
    else el.textContent = 'MOVE MODE (M) — click marker to select';
    el.className = 'mode-move';
  }
}

function updateProgress() {
  const total = CASES.length;
  let nCorrected = 0;
  for (const c of CASES) {
    if (corrections[c.patient_id] && corrections[c.patient_id].status === 'corrected') nCorrected++;
  }
  const pct = total > 0 ? (nCorrected / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-text').textContent =
    nCorrected + ' of ' + total + ' corrected';
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + total;
}

function findNearestMarker(x) {
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < markers.length; i++) {
    const mx = timeToX(markers[i]);
    const dist = Math.abs(mx - x);
    if (dist < bestDist) { bestDist = dist; best = i; }
  }
  return (bestDist <= 20) ? best : -1;
}

function pushUndo() {
  undoStack.push([...markers]);
  if (undoStack.length > 100) undoStack.shift();
}

function undo() {
  if (undoStack.length === 0) return;
  markers = undoStack.pop();
  redraw();
}

function redraw() {
  drawEEG();
  drawEvidence();
  updateInfoPanel();
  updateModeIndicator();
  updateProgress();
}

function show() {
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load markers: from corrections if saved, else from original HPP times
  if (corrections[c.patient_id] && corrections[c.patient_id].times) {
    markers = [...corrections[c.patient_id].times];
  } else {
    markers = [...(c.original_times || [])];
  }
  undoStack = [];
  moveSelected = -1;
  hoverMarker = -1;
  mouseX = -1;
  redraw();
}

function autoSave() {
  const c = CASES[idx];
  if (markers.length > 0) {
    corrections[c.patient_id] = {
      times: [...markers].sort((a, b) => a - b),
      original_times: c.original_times || [],
      status: corrections[c.patient_id]?.status || 'in_progress'
    };
    saveCorrections();
  }
}

function saveCurrent(statusOverride) {
  const c = CASES[idx];
  const edited = JSON.stringify([...markers].sort((a,b)=>a-b)) !== JSON.stringify((c.original_times||[]).sort((a,b)=>a-b));
  corrections[c.patient_id] = {
    times: [...markers].sort((a, b) => a - b),
    original_times: c.original_times || [],
    status: statusOverride || (edited ? 'corrected' : 'accepted')
  };
  saveCorrections();
  const label = corrections[c.patient_id].status === 'accepted' ? 'ACCEPTED' : 'SAVED (corrected)';
  document.getElementById('save-status').textContent = label;
  document.getElementById('save-status').style.color = '#4f4';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1000);
  updateProgress();
}

function exportJSON() {
  const out = {};
  for (const c of CASES) {
    const pid = c.patient_id;
    if (corrections[pid]) {
      out[pid] = {
        patient_id: pid,
        original_times: corrections[pid].original_times,
        corrected_times: corrections[pid].times,
        status: corrections[pid].status
      };
    }
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'timing_corrections.json';
  a.click();
}

// ── Canvas mouse events ──
const eegCanvas = document.getElementById('eeg-canvas');
const evCanvas = document.getElementById('evidence-canvas');

function handleCanvasClick(e) {
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;

  if (x < PLOT_LEFT || x > PLOT_RIGHT) return;
  const t = xToTime(x);

  if (mode === 'add') {
    if (t >= 0 && t <= DURATION) {
      pushUndo();
      markers.push(t);
      redraw();
    }
  } else if (mode === 'delete') {
    const mi = findNearestMarker(x);
    if (mi >= 0) {
      pushUndo();
      markers.splice(mi, 1);
      hoverMarker = -1;
      redraw();
    }
  } else if (mode === 'move') {
    if (moveSelected < 0) {
      // Select a marker
      const mi = findNearestMarker(x);
      if (mi >= 0) {
        moveSelected = mi;
        updateModeIndicator();
        redraw();
      }
    } else {
      // Place the marker
      pushUndo();
      const newT = Math.max(0, Math.min(DURATION, t));
      markers[moveSelected] = newT;
      moveSelected = -1;
      mouseX = -1;
      redraw();
    }
  }
}

function handleCanvasMouseMove(e) {
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;

  if (mode === 'delete') {
    const oldHover = hoverMarker;
    hoverMarker = findNearestMarker(x);
    if (hoverMarker !== oldHover) redraw();
  } else if (mode === 'move' && moveSelected >= 0) {
    mouseX = x;
    redraw();
  }
}

eegCanvas.addEventListener('click', handleCanvasClick);
eegCanvas.addEventListener('mousemove', handleCanvasMouseMove);
eegCanvas.addEventListener('mouseleave', () => {
  if (mode === 'delete' && hoverMarker >= 0) { hoverMarker = -1; redraw(); }
});

// Also allow clicks on evidence canvas
evCanvas.addEventListener('click', handleCanvasClick);
evCanvas.addEventListener('mousemove', handleCanvasMouseMove);

// ── Keyboard events ──
document.addEventListener('keydown', e => {
  if (e.key === 'a' || e.key === 'A') { mode = 'add'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'd') { mode = 'delete'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'm') { mode = 'move'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'z' || e.key === 'Z') { undo(); e.preventDefault(); }
  else if (e.key === 'Escape') { moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'c') { saveCurrent('accepted'); if (idx < CASES.length - 1) { idx++; show(); } e.preventDefault(); }
  else if (e.key === 'Enter') { saveCurrent(); if (idx < CASES.length - 1) { idx++; show(); } e.preventDefault(); }
  else if (e.key === 'ArrowRight') { autoSave(); if (idx < CASES.length - 1) { idx++; show(); } e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { autoSave(); if (idx > 0) { idx--; show(); } e.preventDefault(); }
  else if (e.key === 'e' || e.key === 'E') { exportJSON(); e.preventDefault(); }
});

// Init
show();
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate timing correction viewer')
    args = parser.parse_args()

    print("=" * 72)
    print("Timing Correction Viewer Generator")
    print("=" * 72)

    # ── Load HPP discharge times and filter to auto cases ──
    print("\n--- Loading HPP discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_results = json.load(f)
    print(f"  Loaded HPP results for {len(hpp_results)} patients")

    # Filter: include only cases where review_status is NOT 'ground_truth'
    auto_pids = set()
    for pid, entry in hpp_results.items():
        if entry.get('review_status') != 'ground_truth':
            auto_pids.add(pid)
    print(f"  Auto cases (not ground_truth): {len(auto_pids)}")
    print(f"  Ground truth cases (excluded): {len(hpp_results) - len(auto_pids)}")

    target_pids = auto_pids

    # ── Load dataset ──
    print("\n--- Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # ── Compute evidence signals ──
    print("\n--- Computing evidence signals ---")
    from label_pipeline.hpp_discharge_marking import (
        _compute_channel_evidence, _aggregate_evidence
    )

    evidence_cache = {}
    for idx_p, (_, row) in enumerate(df.iterrows()):
        pid = str(row['patient_id'])
        if pid not in target_pids:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]
        subtype = row['subtype']
        laterality = row.get('laterality', '')
        if not isinstance(laterality, str) or laterality not in ('left', 'right'):
            laterality = None

        n_channels = min(seg.shape[0], 18)
        n_samples = seg.shape[1]

        try:
            evidence_all = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                evidence_all[ch] = _compute_channel_evidence(seg[ch], FS)
            evidence = _aggregate_evidence(evidence_all, subtype, laterality)
            evidence_cache[pid] = evidence
        except Exception as e:
            print(f"  Evidence FAILED for {pid}: {e}")

    print(f"  Evidence signals computed: {len(evidence_cache)}")

    # ── Build case data for embedding ──
    print("\n--- Building case data ---")

    # Target downsample length for performance
    DS_LEN = 500

    # Lowpass filter setup
    nyq = FS / 2.0
    b_lp, a_lp = butter(4, 20.0 / nyq, btype='low')

    cases_data = []
    n_found = 0

    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        if pid not in target_pids:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            print(f"  SKIP {pid}: no segments")
            continue

        seg = pat_segs[0].astype(np.float64)
        if seg.shape[0] > seg.shape[1]:
            seg = seg.T
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        n_channels = min(seg.shape[0], 18)
        seg = seg[:n_channels, :]

        # Lowpass filter at 20 Hz
        for i in range(n_channels):
            try:
                seg[i, :] = filtfilt(b_lp, a_lp, seg[i, :])
            except ValueError:
                pass

        # Detrend
        for i in range(n_channels):
            seg[i, :] = detrend(seg[i, :], type='linear')

        subtype = row['subtype']
        gold = float(row['gold_standard_freq'])

        hpp = hpp_results.get(pid, {})
        original_times = hpp.get('global_times', [])

        # Downsample EEG data
        eeg_ds = downsample(seg, DS_LEN)

        # Downsample evidence signal
        ev = evidence_cache.get(pid, np.array([]))
        ev_ds = downsample(ev, DS_LEN) if len(ev) > 0 else []

        case = {
            'patient_id': pid,
            'subtype': subtype,
            'gold_freq': gold,
            'original_times': original_times,
            'eeg_data': eeg_ds,
            'evidence_signal': ev_ds,
        }
        cases_data.append(case)
        n_found += 1

    # Sort by gold freq
    cases_data.sort(key=lambda c: c['gold_freq'])

    print(f"  Cases prepared: {n_found}")
    not_found = target_pids - {c['patient_id'] for c in cases_data}
    if not_found:
        print(f"  WARNING: {len(not_found)} cases not found in dataset: {list(not_found)[:5]}...")

    # ── Build HTML ──
    print("\n--- Building HTML viewer ---")

    html = build_html(cases_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases_data, default=_json_default))

    output_path = OUT_DIR / 'timing_correction_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Auto cases to review: {len(target_pids)}")
    print(f"  Cases in viewer: {n_found}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Open with: open {output_path}")
    print(f"{'=' * 72}")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


if __name__ == '__main__':
    main()
