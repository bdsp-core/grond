"""
Generate HTML viewer comparing 3 evidence methods:
  1. HPP handcrafted (pointiness + TKEO)
  2. Old CET (CETModel encoder-decoder, no skip connections)
  3. New CET-UNet (CETUNet with skip connections)

Shows ALL 593 LPD+GPD patients (sorted by gold_standard_freq).
For each case:
  - Canvas EEG rendering (18 bipolar channels, 20Hz lowpass, white bg)
  - Interactive discharge markers (Add/Delete/Undo)
  - Evidence subplot with THREE overlaid traces:
      Blue = HPP handcrafted (pointiness+TKEO)
      Orange = Old CET (CNN ensemble)
      Green = New CET-UNet (CNN ensemble)
  - Two sets of vertical lines:
      Red solid = ground truth markers (editable)
      Blue-purple dashed = CET-UNet predicted discharge times (read-only)
  - U key: Accept CET-UNet predictions as ground truth

Usage:
    conda run -n foe_dl python code/cet_model/generate_evidence_comparison_viewer.py
"""

import sys, json, math
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend
import warnings
warnings.filterwarnings('ignore')

import torch

# -- Path setup ----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence, _aggregate_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)
from cet_model.cet import CETModel, CETUNet

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
CACHE_DIR = DATA_DIR / 'cet_cache'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use MPS if available, else CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print("Using MPS device for inference")
else:
    DEVICE = torch.device('cpu')
    print("Using CPU device for inference")

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

DURATION = 10.0  # seconds


# -- CET model loading --------------------------------------------------------

def load_cet_models(n_folds=5):
    """Load ensemble of old CET models (CETModel) from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cet_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CET model not found: {model_path}")
        model = CETModel()
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    return models


def load_cet_unet_models(n_folds=5):
    """Load ensemble of new CET-UNet models from all folds."""
    models = []
    for fold in range(n_folds):
        model_path = CACHE_DIR / f'cet_unet_fold{fold}.pt'
        if not model_path.exists():
            raise FileNotFoundError(f"CET-UNet model not found: {model_path}")
        model = CETUNet()
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        model.to(DEVICE)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def compute_cnn_evidence_single_channel(channel_data, models):
    """Run CNN ensemble on a single channel, return (2000,) evidence.

    Works for both CETModel and CETUNet since they share the same interface.
    Per-channel z-score normalized before inference.
    """
    x = channel_data.astype(np.float32).copy()
    mu = np.mean(x)
    std = np.std(x)
    if std > 1e-8:
        x = (x - mu) / std
    else:
        x = x - mu
    x_tensor = torch.from_numpy(x[np.newaxis, np.newaxis, :]).to(DEVICE)  # (1, 1, 2000)
    predictions = []
    for model in models:
        pred = model(x_tensor).squeeze().cpu().numpy()  # (2000,)
        predictions.append(pred)
    return np.mean(predictions, axis=0).astype(np.float32)


def hpp_dp_discharge_times(evidence, fs, gold_freq):
    """Run the HPP DP algorithm on evidence to get predicted discharge times.

    Uses gold_standard_freq as the period prior (the "cheating" version for
    label refinement, not deployment).

    Args:
        evidence: 1D numpy array of evidence values (raw, before normalization)
        fs: sampling rate
        gold_freq: expected frequency in Hz (used as freq_estimate)

    Returns:
        List of discharge times in seconds.
    """
    if len(evidence) < 10 or gold_freq <= 0:
        return []

    freq_estimate = np.clip(gold_freq, 0.3, 3.5)

    # A. Detect active interval
    active_start, active_end = _detect_active_interval(evidence, fs)

    # B. Extract candidate peaks
    candidates = _extract_candidates(evidence, fs, freq_estimate,
                                     active_start, active_end)

    if len(candidates) == 0:
        return []

    # C. DP best sequence
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)

    if len(discharge_samples) == 0:
        return []

    # D. EM refinement
    if len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)

    # Convert to times
    times = (discharge_samples / fs).tolist()
    times = [t for t in times if 0 <= t <= DURATION]
    return times


# -- Downsample ----------------------------------------------------------------

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


# -- JSON serializer -----------------------------------------------------------

def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# -- HTML builder --------------------------------------------------------------

def build_html(cases_data):
    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evidence Comparison: HPP vs CET vs CET-UNet</title>
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
    font-size: 18px; font-weight: bold; padding: 6px 20px;
    text-align: center; letter-spacing: 2px;
  }
  .mode-add { background: #1a3a1a; color: #44ff66; }
  .mode-delete { background: #3a1a1a; color: #ff4444; }
  .mode-nav { background: #1a1a3a; color: #6688ff; }

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

  #legend {
    display: flex; justify-content: center; gap: 24px;
    padding: 6px; background: #252525; font-size: 13px;
    flex-wrap: wrap;
  }
  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 30px; height: 4px; border-radius: 2px; }

  .export-btn {
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }
  .export-btn:hover { background: #3a4a3a; }

  #save-status { color: #44cc44; font-size: 13px; }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Evidence Comparison: HPP / CET / CET-UNet</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportJSON()">Export Markers <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <div class="info-col">
    <span class="info-badge" id="info-subtype-badge">--</span>
  </div>
  <div class="info-col">
    <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
    <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Gold freq: <strong id="info-gold-freq">--</strong></span>
    <span class="info-item">Mode: <strong id="info-mode">Navigate</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">GT markers: <strong id="info-gt-count" style="color:#ff4444;">--</strong> (red)</span>
    <span class="info-item">CET markers: <strong id="info-cet-count" style="color:#6a5acd;">--</strong> (blue-purple)</span>
  </div>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="legend">
  <div class="legend-item">
    <div class="legend-swatch" style="background: steelblue;"></div>
    <span>HPP (pointiness+TKEO)</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background: #e67e22;"></div>
    <span>Old CET (CNN)</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background: #27ae60;"></div>
    <span>New CET-UNet</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background: rgba(255,0,0,0.6); height: 12px; width: 2px;"></div>
    <span>GT markers (editable)</span>
  </div>
  <div class="legend-item">
    <div class="legend-swatch" style="background: #6a5acd; height: 12px; width: 2px; border-left: 2px dashed #6a5acd;"></div>
    <span>CET-UNet predictions</span>
  </div>
</div>

<div id="shortcuts">
  <span class="key">A</span> Add mode &nbsp;&nbsp;
  <span class="key">D</span> Delete mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">U</span> Accept CET labels &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">C</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">Enter</span> Save &amp; advance &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = CASES_PLACEHOLDER;

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const GROUP_BREAKS = new Set([4, 8, 12, 16]);
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1200;
const EEG_HEIGHT = 700;
const EV_HEIGHT = 200;
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

const CET_LINE_COLOR = '#6a5acd';  // blue-purple for CET-UNet predictions

// State
let idx = 0;
let reviewed = new Set();
let mode = 'nav'; // 'nav', 'add', 'delete'
let markers = [];  // current GT marker times (seconds) - editable
let undoStack = [];
let hoverMarker = -1;
let caseStatus = {};  // track status per patient_id: 'unchanged', 'edited', 'cet_accepted'

// Marker persistence via localStorage
const STORAGE_KEY = 'evidence_comparison_markers_v2';
let allMarkers = {};
try { allMarkers = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch(e) { allMarkers = {}; }

function saveAllMarkers() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(allMarkers));
}

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

function timeToX(t) { return PLOT_LEFT + (t / DURATION) * PLOT_W; }
function xToTime(x) { return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }

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

// Helper: draw dashed vertical lines for CET-UNet predictions
function drawCetMarkers(ctx, topY, bottomY, lineWidth) {
  const c = CASES[idx];
  const cetTimes = c.cet_unet_times || [];
  ctx.strokeStyle = CET_LINE_COLOR;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash([6, 4]);
  for (let i = 0; i < cetTimes.length; i++) {
    const t = cetTimes[i];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    ctx.beginPath(); ctx.moveTo(x, topY); ctx.lineTo(x, bottomY); ctx.stroke();
  }
  ctx.setLineDash([]);
}

function drawEEG() {
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;

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

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Traces
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

  // Time axis
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

  // CET-UNet predicted markers (blue-purple dashed, read-only) - draw first so GT is on top
  drawCetMarkers(ctx, PLOT_TOP, PLOT_BOTTOM, 1.5);

  // GT discharge markers (red solid, editable)
  for (let mi = 0; mi < markers.length; mi++) {
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.6)';
    let lw = 2;
    if (mode === 'delete' && hoverMarker === mi) {
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

  const evHpp = c.evidence_hpp;
  const evCet = c.evidence_cet;
  const evUnet = c.evidence_unet;

  const evTop = 15;
  const evBottom = EV_HEIGHT - 25;
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

  // Y-axis gridlines at 0.25, 0.5, 0.75
  ctx.strokeStyle = '#eeeeee';
  ctx.lineWidth = 0.5;
  for (let v of [0.25, 0.5, 0.75]) {
    const y = evBottom - v * evH;
    ctx.beginPath(); ctx.moveTo(PLOT_LEFT, y); ctx.lineTo(PLOT_RIGHT, y); ctx.stroke();
  }

  // Helper to draw a trace with fill
  function drawTrace(data, strokeColor, fillColor) {
    if (!data || data.length === 0) return;
    const nSamples = data.length;

    // Fill
    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.moveTo(PLOT_LEFT, evBottom);
    for (let i = 0; i < nSamples; i++) {
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - data[i] * evH;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(PLOT_RIGHT, evBottom);
    ctx.closePath();
    ctx.fill();

    // Line
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < nSamples; i++) {
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - data[i] * evH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Draw HPP (blue), then old CET (orange), then CET-UNet (green)
  drawTrace(evHpp, 'steelblue', 'rgba(70, 130, 180, 0.15)');
  drawTrace(evCet, '#e67e22', 'rgba(230, 126, 34, 0.15)');
  drawTrace(evUnet, '#27ae60', 'rgba(39, 174, 96, 0.15)');

  // Y-axis labels
  ctx.fillStyle = '#000';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('1.0', PLOT_LEFT - 4, evTop);
  ctx.fillText('0.5', PLOT_LEFT - 4, evTop + evH * 0.5);
  ctx.fillText('0.0', PLOT_LEFT - 4, evBottom);

  // Time labels
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {
    ctx.fillText(s + 's', timeToX(s), evBottom + 3);
  }

  // CET-UNet predicted markers (blue-purple dashed) - draw first
  drawCetMarkers(ctx, evTop, evBottom, 1.0);

  // GT discharge markers on evidence panel (red solid)
  for (let mi = 0; mi < markers.length; mi++) {
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.5)';
    let lw = 1.5;
    if (mode === 'delete' && hoverMarker === mi) {
      color = 'rgba(255, 50, 50, 0.9)';
      lw = 3;
    }

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

  // GT and CET marker counts
  document.getElementById('info-gt-count').textContent = markers.length;
  const cetTimes = c.cet_unet_times || [];
  document.getElementById('info-cet-count').textContent = cetTimes.length;

  // Mode
  let modeStr = 'Navigate';
  if (mode === 'add') modeStr = 'Add';
  else if (mode === 'delete') modeStr = 'Delete';
  document.getElementById('info-mode').textContent = modeStr;

  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype.toLowerCase();

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  const pct = ((idx + 1) / CASES.length * 100);
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('reviewed-count').textContent = reviewed.size + ' reviewed';
}

function updateModeIndicator() {
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') { el.textContent = 'ADD MODE (A) -- click to add marker'; el.className = 'mode-add'; }
  else if (mode === 'delete') { el.textContent = 'DELETE MODE (D) -- click near marker to remove'; el.className = 'mode-delete'; }
  else { el.textContent = 'NAVIGATE MODE'; el.className = 'mode-nav'; }
}

function getStatus(pid) {
  return caseStatus[pid] || 'unchanged';
}

function autoSave() {
  const c = CASES[idx];
  const pid = c.patient_id;
  allMarkers[pid] = {
    times: [...markers].sort((a, b) => a - b),
    original_times: c.discharge_times || [],
    status: getStatus(pid),
  };
  saveAllMarkers();
}

function saveCurrent() {
  autoSave();
  reviewed.add(CASES[idx].patient_id);
  const el = document.getElementById('save-status');
  el.textContent = 'SAVED';
  el.style.color = '#4f4';
  setTimeout(() => { el.textContent = ''; }, 1000);
}

function acceptCET() {
  const c = CASES[idx];
  const cetTimes = c.cet_unet_times || [];
  pushUndo();
  markers = [...cetTimes];
  caseStatus[c.patient_id] = 'cet_accepted';
  const el = document.getElementById('save-status');
  el.textContent = 'CET labels accepted';
  el.style.color = '#6a5acd';
  setTimeout(() => { el.textContent = ''; }, 2000);
  redraw();
}

function redraw() {
  drawEEG();
  drawEvidence();
  updateInfoPanel();
  updateModeIndicator();
}

function show() {
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load markers: from localStorage if saved, else from original discharge_times
  if (allMarkers[c.patient_id] && allMarkers[c.patient_id].times) {
    markers = [...allMarkers[c.patient_id].times];
    if (allMarkers[c.patient_id].status) {
      caseStatus[c.patient_id] = allMarkers[c.patient_id].status;
    }
  } else {
    markers = [...(c.discharge_times || [])];
  }
  undoStack = [];
  hoverMarker = -1;
  redraw();
}

function exportJSON() {
  autoSave();
  const out = {};
  for (const c of CASES) {
    const pid = c.patient_id;
    const cetTimes = c.cet_unet_times || [];
    if (allMarkers[pid]) {
      out[pid] = {
        patient_id: pid,
        original_gt_times: allMarkers[pid].original_times || c.discharge_times || [],
        updated_gt_times: allMarkers[pid].times,
        cet_unet_times: cetTimes,
        status: allMarkers[pid].status || 'unchanged',
      };
    } else {
      out[pid] = {
        patient_id: pid,
        original_gt_times: c.discharge_times || [],
        updated_gt_times: c.discharge_times || [],
        cet_unet_times: cetTimes,
        status: 'unchanged',
      };
    }
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'evidence_comparison_markers.json';
  a.click();
}

// -- Canvas mouse events --
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
      caseStatus[CASES[idx].patient_id] = caseStatus[CASES[idx].patient_id] === 'cet_accepted' ? 'cet_accepted' : 'edited';
      redraw();
    }
  } else if (mode === 'delete') {
    const mi = findNearestMarker(x);
    if (mi >= 0) {
      pushUndo();
      markers.splice(mi, 1);
      caseStatus[CASES[idx].patient_id] = caseStatus[CASES[idx].patient_id] === 'cet_accepted' ? 'cet_accepted' : 'edited';
      hoverMarker = -1;
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
  }
}

eegCanvas.addEventListener('click', handleCanvasClick);
eegCanvas.addEventListener('mousemove', handleCanvasMouseMove);
eegCanvas.addEventListener('mouseleave', () => {
  if (mode === 'delete' && hoverMarker >= 0) { hoverMarker = -1; redraw(); }
});
evCanvas.addEventListener('click', handleCanvasClick);
evCanvas.addEventListener('mousemove', handleCanvasMouseMove);

// -- Keyboard events --
document.addEventListener('keydown', e => {
  if (e.key === 'a' || e.key === 'A') {
    mode = 'add'; hoverMarker = -1; redraw(); e.preventDefault();
  }
  else if (e.key === 'd' && !e.ctrlKey && !e.metaKey) {
    mode = 'delete'; hoverMarker = -1; redraw(); e.preventDefault();
  }
  else if (e.key === 'z' || e.key === 'Z') {
    undo(); e.preventDefault();
  }
  else if (e.key === 'u' || e.key === 'U') {
    acceptCET(); e.preventDefault();
  }
  else if (e.key === 'Escape') {
    mode = 'nav'; hoverMarker = -1; redraw(); e.preventDefault();
  }
  else if (e.key === 'c' || e.key === 'C') {
    saveCurrent();
    if (idx < CASES.length - 1) { idx++; show(); }
    else { redraw(); }
    e.preventDefault();
  }
  else if (e.key === 'Enter') {
    saveCurrent();
    if (idx < CASES.length - 1) { idx++; show(); }
    else { redraw(); }
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
  else if (e.key === 'e' || e.key === 'E') {
    exportJSON(); e.preventDefault();
  }
});

// Init
show();
</script>
</body>
</html>"""
    return html


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Evidence Comparison Viewer: HPP vs CET vs CET-UNet")
    print("=" * 72)

    # Load discharge times (for MW-reviewed markers)
    print("\n--- Loading discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_results = json.load(f)
    print(f"  Total patients with discharge data: {len(hpp_results)}")

    # Load dataset
    print("\n--- Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # Load CET models (old)
    print("\n--- Loading old CET models ---")
    cet_models = load_cet_models()
    print(f"  Loaded {len(cet_models)} old CET fold models")

    # Load CET-UNet models (new)
    print("\n--- Loading new CET-UNet models ---")
    unet_models = load_cet_unet_models()
    print(f"  Loaded {len(unet_models)} CET-UNet fold models")

    # Select ALL LPD+GPD patients
    print("\n--- Selecting ALL LPD+GPD cases ---")
    candidates = []
    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        if seg.shape[0] < 18 or seg.shape[1] < 100:
            continue
        subtype = row['subtype']
        if subtype not in ('lpd', 'gpd'):
            continue
        gold = float(row['gold_standard_freq'])
        if not np.isfinite(gold) or gold <= 0:
            continue

        hpp_entry = hpp_results.get(pid, {})
        is_gt = hpp_entry.get('review_status') == 'ground_truth'

        candidates.append({
            'pid': pid,
            'subtype': subtype,
            'gold_freq': gold,
            'is_gt': is_gt,
        })

    n_lpd = sum(1 for c in candidates if c['subtype'] == 'lpd')
    n_gpd = sum(1 for c in candidates if c['subtype'] == 'gpd')
    selected = candidates
    print(f"  LPD: {n_lpd}, GPD: {n_gpd}, Total: {len(selected)}")

    # Sort final selection by gold freq
    selected.sort(key=lambda x: x['gold_freq'])
    selected_pids = {c['pid'] for c in selected}

    # Build case data
    print("\n--- Computing evidence traces (3 methods) + CET-UNet peak detection ---")
    DS_LEN = 500  # downsample for HTML size
    nyq = FS / 2.0
    b_lp, a_lp = butter(4, 20.0 / nyq, btype='low')

    cases_data = []
    for ci, cand in enumerate(selected):
        pid = cand['pid']
        row = df[df['patient_id'] == pid].iloc[0]
        seg = segments[pid][0].astype(np.float64)

        if seg.shape[0] > seg.shape[1]:
            seg = seg.T
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        n_channels = min(seg.shape[0], 18)
        seg = seg[:n_channels, :]

        subtype = cand['subtype']
        gold = cand['gold_freq']
        laterality = row.get('laterality', '')
        if not isinstance(laterality, str) or laterality not in ('left', 'right'):
            laterality = None

        # --- HPP evidence (pointiness+TKEO) ---
        try:
            hpp_ev_all = np.zeros((n_channels, seg.shape[1]))
            for ch in range(n_channels):
                hpp_ev_all[ch] = _compute_channel_evidence(seg[ch], FS)
            hpp_evidence = _aggregate_evidence(hpp_ev_all, subtype, laterality)
        except Exception as e:
            print(f"  HPP evidence FAILED for {pid}: {e}")
            hpp_evidence = np.zeros(seg.shape[1])

        # --- Old CET evidence (CETModel ensemble) ---
        try:
            cet_ev_all = np.zeros((n_channels, seg.shape[1]), dtype=np.float32)
            for ch in range(n_channels):
                ch_data = seg[ch]
                if np.all(np.isfinite(ch_data)):
                    cet_ev_all[ch] = compute_cnn_evidence_single_channel(ch_data, cet_models)
            cet_evidence = _aggregate_evidence(cet_ev_all, subtype, laterality)
        except Exception as e:
            print(f"  CET evidence FAILED for {pid}: {e}")
            cet_evidence = np.zeros(seg.shape[1])

        # --- New CET-UNet evidence (CETUNet ensemble) ---
        try:
            unet_ev_all = np.zeros((n_channels, seg.shape[1]), dtype=np.float32)
            for ch in range(n_channels):
                ch_data = seg[ch]
                if np.all(np.isfinite(ch_data)):
                    unet_ev_all[ch] = compute_cnn_evidence_single_channel(ch_data, unet_models)
            unet_evidence = _aggregate_evidence(unet_ev_all, subtype, laterality)
        except Exception as e:
            print(f"  CET-UNet evidence FAILED for {pid}: {e}")
            unet_evidence = np.zeros(seg.shape[1])

        # --- HPP DP on CET-UNet evidence to get predicted discharge times ---
        try:
            cet_unet_times = hpp_dp_discharge_times(unet_evidence, FS, gold)
        except Exception as e:
            print(f"  CET-UNet HPP DP FAILED for {pid}: {e}")
            cet_unet_times = []

        # Normalize all to [0, 1]
        def normalize_01(x):
            mn = np.min(x)
            mx = np.max(x)
            if mx - mn < 1e-10:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        hpp_evidence = normalize_01(hpp_evidence)
        cet_evidence = normalize_01(cet_evidence)
        unet_evidence = normalize_01(unet_evidence)

        # Lowpass and detrend EEG for display
        seg_display = seg.copy()
        for i in range(n_channels):
            try:
                seg_display[i, :] = filtfilt(b_lp, a_lp, seg_display[i, :])
            except ValueError:
                pass
            seg_display[i, :] = detrend(seg_display[i, :], type='linear')

        # Get discharge times
        hpp_entry = hpp_results.get(pid, {})
        discharge_times = hpp_entry.get('global_times', [])

        case = {
            'patient_id': pid,
            'subtype': subtype,
            'gold_freq': gold,
            'discharge_times': discharge_times,
            'cet_unet_times': cet_unet_times,
            'eeg_data': downsample(seg_display, DS_LEN),
            'evidence_hpp': downsample(hpp_evidence, DS_LEN),
            'evidence_cet': downsample(cet_evidence, DS_LEN),
            'evidence_unet': downsample(unet_evidence, DS_LEN),
        }
        cases_data.append(case)

        if (ci + 1) % 50 == 0:
            print(f"  Processed {ci + 1}/{len(selected)} cases")

    print(f"  Total cases prepared: {len(cases_data)}")

    # Summarize CET-UNet peak detection
    n_with_cet = sum(1 for c in cases_data if len(c['cet_unet_times']) > 0)
    avg_cet_peaks = np.mean([len(c['cet_unet_times']) for c in cases_data]) if cases_data else 0
    print(f"  CET-UNet peaks detected: {n_with_cet}/{len(cases_data)} cases, avg {avg_cet_peaks:.1f} peaks/case")

    # Build HTML
    print("\n--- Building HTML viewer ---")
    html = build_html(cases_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases_data, default=_json_default))

    output_path = OUT_DIR / 'evidence_comparison_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Cases in viewer: {len(cases_data)}")
    n_gt = sum(1 for c in selected if c['is_gt'])
    print(f"  Ground truth (MW-reviewed): {n_gt}")
    print(f"  Evidence traces: HPP (blue), CET (orange), CET-UNet (green)")
    print(f"  Marker lines: Red solid = GT (editable), Blue-purple dashed = CET-UNet (read-only)")
    print(f"  Interactive: Add (A), Delete (D), Undo (Z), Accept CET (U)")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Open with: open {output_path}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
