"""
Generate interactive HTML viewer for cases where gold standard frequency
and IPI-derived frequency disagree by > 0.3 Hz.

MW reviews each case and can:
  - Update the gold standard frequency
  - Edit discharge timing markers (add/delete/move)
  - Accept both as-is

Sorted by error magnitude (largest first) so worst disagreements come first.

Usage:
    conda run -n foe python code/label_pipeline/generate_freq_disagreement_viewer.py
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

FREQ_THRESHOLD = 0.3  # Hz


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
    """Build the interactive frequency disagreement HTML viewer."""

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Frequency Disagreement Viewer</title>
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
  .badge-rda { background: #205a20; color: #88ff88; }

  .error-badge {
    padding: 4px 12px; border-radius: 5px; font-size: 14px; font-weight: bold;
    background: #5a3a00; color: #ffaa00;
  }

  #freq-input-wrap {
    background: #1a2a3a; border: 2px solid #4488cc; border-radius: 6px;
    padding: 8px 14px; display: flex; align-items: center; gap: 8px;
  }
  #freq-input-wrap label { color: #88bbee; font-size: 13px; font-weight: bold; }
  #freq-input {
    width: 80px; background: #0a1a2a; border: 1px solid #4488cc; border-radius: 3px;
    color: #fff; font-family: monospace; font-size: 16px; font-weight: bold;
    padding: 4px 8px; text-align: center;
  }
  #freq-input:focus { outline: none; border-color: #66ccff; box-shadow: 0 0 6px rgba(100,180,255,0.4); }
  #freq-unit { color: #88bbee; font-size: 13px; }

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

  .status-indicator {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 4px; vertical-align: middle;
  }
  .status-pending { background: #666; }
  .status-done { background: #44cc88; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Freq Disagreement Viewer</span>
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
    <span class="error-badge" id="info-error-badge">err: --</span>
  </div>
  <div class="info-col">
    <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
    <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
    <span class="info-item">N discharges: <strong id="info-n-discharges">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Gold freq: <strong id="info-gold-freq" style="color:#ff8844;">--</strong></span>
    <span class="info-item">IPI freq: <strong id="info-ipi-freq" style="color:#4488ff;">--</strong></span>
    <span class="info-item">Error: <strong id="info-error" style="color:#ffaa00;">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Live markers: <strong id="info-n-markers">--</strong></span>
    <span class="info-item">Live IPI freq: <strong id="info-live-freq" style="color:#44ff88;">--</strong></span>
    <span class="info-item">Median IPI: <strong id="info-median-ipi">--</strong></span>
  </div>
  <div id="freq-input-wrap">
    <label for="freq-input">Corrected freq:</label>
    <input type="text" id="freq-input" value="" placeholder="Hz">
    <span id="freq-unit">Hz</span>
  </div>
  <div class="info-col">
    <span class="info-item">Marker times: <strong id="info-marker-times" style="font-size:11px;">--</strong></span>
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
  <span class="key">C</span> Accept as-is &amp; next &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">E</span> Export JSON &nbsp;&nbsp;
  <span class="key">Tab</span> Focus freq input
</div>

<script>
const CASES = CASES_PLACEHOLDER;

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const GROUP_BREAKS = new Set([4, 8, 12, 16]);
const DURATION = 10.0;
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
let mode = 'add';
let markers = [];
let undoStack = [];
let moveSelected = -1;
let hoverMarker = -1;
let mouseX = -1;

// Corrections stored per patient
let corrections = {};
try { corrections = JSON.parse(localStorage.getItem('freq_disagreement_corrections') || '{}'); } catch(e) { corrections = {}; }

function saveCorrections() {
  localStorage.setItem('freq_disagreement_corrections', JSON.stringify(corrections));
}

function timeToX(t) { return PLOT_LEFT + (t / DURATION) * PLOT_W; }
function xToTime(x) { return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }

// Build display channel list with spacers
function getDisplayChannels() {
  const dc = [];
  for (let i = 0; i < 18; i++) {
    if (GROUP_BREAKS.has(i)) dc.push({ idx: -1, name: '' });
    dc.push({ idx: i, name: CHANNEL_NAMES[i] });
  }
  return dc;
}

const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

function computeLiveFreq() {
  const sorted = [...markers].sort((a, b) => a - b);
  if (sorted.length < 2) return { freq: null, medianIPI: null };
  const ipis = [];
  for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i - 1]);
  ipis.sort((a, b) => a - b);
  const medianIPI = ipis[Math.floor(ipis.length / 2)];
  return { freq: 1.0 / medianIPI, medianIPI: medianIPI };
}

function drawEEG() {
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;

  // White background
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  // Gridlines
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
    if (ch.idx < 0) continue;

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
  const liveInfo = computeLiveFreq();
  const liveStr = liveInfo.freq !== null ? ('  |  live=' + liveInfo.freq.toFixed(2) + ' Hz') : '';
  const title = c.patient_id + '  |  ' + c.subtype.toUpperCase() +
    '  |  gold=' + c.gold_freq.toFixed(2) +
    '  |  ipi=' + c.ipi_freq.toFixed(2) +
    '  |  err=' + c.error.toFixed(2) + liveStr;
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
  document.getElementById('info-ipi-freq').textContent = c.ipi_freq.toFixed(3) + ' Hz';
  document.getElementById('info-error').textContent = c.error.toFixed(3) + ' Hz';
  document.getElementById('info-n-discharges').textContent = c.n_discharges;

  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype.toLowerCase();

  document.getElementById('info-error-badge').textContent = 'err: ' + c.error.toFixed(2) + ' Hz';

  // Markers info (live)
  const sorted = [...markers].sort((a, b) => a - b);
  document.getElementById('info-n-markers').textContent = sorted.length;
  document.getElementById('info-marker-times').textContent =
    sorted.length > 0 ? sorted.map(t => t.toFixed(2) + 's').join(', ') : 'none';

  const liveInfo = computeLiveFreq();
  if (liveInfo.freq !== null) {
    document.getElementById('info-median-ipi').textContent = liveInfo.medianIPI.toFixed(3) + ' s';
    document.getElementById('info-live-freq').textContent = liveInfo.freq.toFixed(3) + ' Hz';
  } else {
    document.getElementById('info-median-ipi').textContent = '--';
    document.getElementById('info-live-freq').textContent = '--';
  }

  // Original times
  const orig = c.original_times || [];
  document.getElementById('info-original-times').textContent =
    orig.length > 0 ? orig.map(t => t.toFixed(2) + 's').join(', ') : 'none';

  // Freq input: load from corrections if saved, else gold_standard_freq
  const freqInput = document.getElementById('freq-input');
  if (!freqInput._userEditing) {
    if (corrections[c.patient_id] && corrections[c.patient_id].corrected_freq !== null && corrections[c.patient_id].corrected_freq !== undefined) {
      freqInput.value = corrections[c.patient_id].corrected_freq;
    } else {
      freqInput.value = c.gold_freq.toFixed(2);
    }
  }
}

function updateModeIndicator() {
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') { el.textContent = 'ADD MODE (A)'; el.className = 'mode-add'; }
  else if (mode === 'delete') { el.textContent = 'DELETE MODE (D)'; el.className = 'mode-delete'; }
  else if (mode === 'move') {
    if (moveSelected >= 0) el.textContent = 'MOVE MODE (M) -- click to place';
    else el.textContent = 'MOVE MODE (M) -- click marker to select';
    el.className = 'mode-move';
  }
}

function updateProgress() {
  const total = CASES.length;
  let nDone = 0;
  for (const c of CASES) {
    if (corrections[c.patient_id] && corrections[c.patient_id].status &&
        corrections[c.patient_id].status !== 'in_progress') nDone++;
  }
  const pct = total > 0 ? (nDone / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-text').textContent =
    nDone + ' of ' + total + ' reviewed';
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

  // Reset freq input editing flag
  document.getElementById('freq-input')._userEditing = false;

  // Load markers: from corrections if saved, else from original HPP times
  if (corrections[c.patient_id] && corrections[c.patient_id].corrected_times) {
    markers = [...corrections[c.patient_id].corrected_times];
  } else {
    markers = [...(c.original_times || [])];
  }
  undoStack = [];
  moveSelected = -1;
  hoverMarker = -1;
  mouseX = -1;
  redraw();
}

function getFreqInputValue() {
  const raw = document.getElementById('freq-input').value.trim();
  const val = parseFloat(raw);
  if (isNaN(val) || val <= 0 || val > 20) return null;
  return Math.round(val * 1000) / 1000;  // 3 decimal places
}

function determineStatus(c) {
  const sortedMarkers = [...markers].sort((a, b) => a - b);
  const sortedOrig = [...(c.original_times || [])].sort((a, b) => a - b);
  const timingChanged = JSON.stringify(sortedMarkers.map(t => t.toFixed(3))) !==
                        JSON.stringify(sortedOrig.map(t => t.toFixed(3)));

  const freqVal = getFreqInputValue();
  const freqChanged = freqVal !== null && Math.abs(freqVal - c.gold_freq) > 0.001;

  if (timingChanged && freqChanged) return 'both_updated';
  if (freqChanged) return 'freq_updated';
  if (timingChanged) return 'timing_updated';
  return 'accepted';
}

function autoSave() {
  const c = CASES[idx];
  const freqVal = getFreqInputValue();
  const status = determineStatus(c);
  corrections[c.patient_id] = {
    corrected_times: [...markers].sort((a, b) => a - b),
    corrected_freq: freqVal,
    original_gold_freq: c.gold_freq,
    ipi_freq: c.ipi_freq,
    status: corrections[c.patient_id]?.status === 'accepted' ? 'accepted' :
            (corrections[c.patient_id]?.status || 'in_progress')
  };
  saveCorrections();
}

function saveCurrent(statusOverride) {
  const c = CASES[idx];
  const freqVal = getFreqInputValue();
  const status = statusOverride || determineStatus(c);

  corrections[c.patient_id] = {
    corrected_times: [...markers].sort((a, b) => a - b),
    corrected_freq: freqVal,
    original_gold_freq: c.gold_freq,
    ipi_freq: c.ipi_freq,
    status: status
  };
  saveCorrections();

  const labels = {
    'accepted': 'ACCEPTED (as-is)',
    'freq_updated': 'SAVED (freq updated)',
    'timing_updated': 'SAVED (timing updated)',
    'both_updated': 'SAVED (both updated)'
  };
  const label = labels[status] || 'SAVED';
  document.getElementById('save-status').textContent = label;
  document.getElementById('save-status').style.color = '#4f4';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1200);
  updateProgress();
}

function exportJSON() {
  const out = {};
  for (const c of CASES) {
    const pid = c.patient_id;
    if (corrections[pid]) {
      out[pid] = {
        corrected_times: corrections[pid].corrected_times,
        corrected_freq: corrections[pid].corrected_freq,
        original_gold_freq: corrections[pid].original_gold_freq,
        ipi_freq: corrections[pid].ipi_freq,
        status: corrections[pid].status
      };
    }
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'freq_disagreement_corrections.json';
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
      const mi = findNearestMarker(x);
      if (mi >= 0) {
        moveSelected = mi;
        updateModeIndicator();
        redraw();
      }
    } else {
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

evCanvas.addEventListener('click', handleCanvasClick);
evCanvas.addEventListener('mousemove', handleCanvasMouseMove);

// ── Freq input events ──
const freqInput = document.getElementById('freq-input');
freqInput.addEventListener('focus', () => { freqInput._userEditing = true; });
freqInput.addEventListener('blur', () => { freqInput._userEditing = false; });
// Prevent keyboard shortcuts when typing in freq input
freqInput.addEventListener('keydown', (e) => {
  e.stopPropagation();
  if (e.key === 'Escape') { freqInput.blur(); }
  if (e.key === 'Enter') {
    freqInput.blur();
    saveCurrent();
    if (idx < CASES.length - 1) { idx++; show(); }
  }
});

// ── Keyboard events ──
document.addEventListener('keydown', e => {
  // Skip if freq input is focused
  if (document.activeElement === freqInput) return;

  if (e.key === 'a' || e.key === 'A') { mode = 'add'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'd') { mode = 'delete'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'm') { mode = 'move'; moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'z' || e.key === 'Z') { undo(); e.preventDefault(); }
  else if (e.key === 'Escape') { moveSelected = -1; mouseX = -1; redraw(); e.preventDefault(); }
  else if (e.key === 'c') {
    saveCurrent('accepted');
    if (idx < CASES.length - 1) { idx++; show(); }
    e.preventDefault();
  }
  else if (e.key === 'Enter') {
    saveCurrent();
    if (idx < CASES.length - 1) { idx++; show(); }
    e.preventDefault();
  }
  else if (e.key === 'ArrowRight') {
    autoSave();
    if (idx < CASES.length - 1) { idx++; show(); }
    e.preventDefault();
  }
  else if (e.key === 'ArrowLeft') {
    autoSave();
    if (idx > 0) { idx--; show(); }
    e.preventDefault();
  }
  else if (e.key === 'e' || e.key === 'E') { exportJSON(); e.preventDefault(); }
  else if (e.key === 'Tab') { freqInput.focus(); freqInput.select(); e.preventDefault(); }
});

// Init
show();
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description='Generate frequency disagreement viewer')
    args = parser.parse_args()

    print("=" * 72)
    print("Frequency Disagreement Viewer Generator")
    print("=" * 72)

    # ── Load HPP discharge times ──
    print("\n--- Loading HPP discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_results = json.load(f)
    print(f"  Loaded HPP results for {len(hpp_results)} patients")

    # ── Find disagreement cases ──
    print(f"\n--- Finding cases with |gold - ipi| > {FREQ_THRESHOLD} Hz ---")
    disagreement_pids = {}
    for pid, entry in hpp_results.items():
        gold = entry.get('gold_standard_freq')
        ipi = entry.get('frequency')
        if gold is not None and ipi is not None:
            error = abs(gold - ipi)
            if error > FREQ_THRESHOLD:
                disagreement_pids[pid] = {
                    'gold_freq': gold,
                    'ipi_freq': ipi,
                    'error': error,
                    'n_discharges': entry.get('n_discharges', 0),
                }

    print(f"  Disagreement cases found: {len(disagreement_pids)}")
    if len(disagreement_pids) == 0:
        print("  No cases to review!")
        return

    # Sort by error (largest first) for display
    sorted_pids = sorted(disagreement_pids.keys(),
                         key=lambda p: -disagreement_pids[p]['error'])
    print(f"  Largest error: {disagreement_pids[sorted_pids[0]]['error']:.3f} Hz ({sorted_pids[0]})")
    print(f"  Smallest error: {disagreement_pids[sorted_pids[-1]]['error']:.3f} Hz ({sorted_pids[-1]})")

    target_pids = set(disagreement_pids.keys())

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
    for _, row in df.iterrows():
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

    DS_LEN = 500

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

        info = disagreement_pids[pid]

        # Downsample EEG data
        eeg_ds = downsample(seg, DS_LEN)

        # Downsample evidence signal
        ev = evidence_cache.get(pid, np.array([]))
        ev_ds = downsample(ev, DS_LEN) if len(ev) > 0 else []

        case = {
            'patient_id': pid,
            'subtype': subtype,
            'gold_freq': gold,
            'ipi_freq': info['ipi_freq'],
            'error': info['error'],
            'n_discharges': info['n_discharges'],
            'original_times': original_times,
            'eeg_data': eeg_ds,
            'evidence_signal': ev_ds,
        }
        cases_data.append(case)
        n_found += 1

    # Sort by error magnitude (largest first)
    cases_data.sort(key=lambda c: -c['error'])

    print(f"  Cases prepared: {n_found}")
    not_found = target_pids - {c['patient_id'] for c in cases_data}
    if not_found:
        print(f"  WARNING: {len(not_found)} cases not found in dataset: {list(not_found)[:5]}...")

    # ── Build HTML ──
    print("\n--- Building HTML viewer ---")

    html = build_html(cases_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases_data, default=_json_default))

    output_path = OUT_DIR / 'freq_disagreement_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Disagreement cases (|gold - ipi| > {FREQ_THRESHOLD} Hz): {len(disagreement_pids)}")
    print(f"  Cases in viewer: {n_found}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Open with: open {output_path}")
    print(f"  localStorage key: freq_disagreement_corrections")
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
