"""
Generate interactive HTML spatial review viewer for channel involvement predictions.

MW reviews which channels the CNN predicted as containing periodic discharges.
Channels are color-coded by PD probability (green=involved, gray=not involved).
Includes discharge timing markers and a channel probability bar chart.

Usage:
    conda run -n foe python code/label_pipeline/generate_spatial_review_viewer.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend
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

# Downsample target
DS_LEN = 500


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
    """Downsample array to target_len along last axis."""
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


def build_html(cases_data):
    """Build the interactive spatial review HTML viewer."""

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Spatial Review Viewer - Channel Involvement</title>
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

  #canvas-container { text-align: center; padding: 4px; position: relative; }
  #eeg-canvas { display: block; margin: 0 auto; }
  #bar-canvas { display: block; margin: 4px auto 0 auto; }

  #status-bar {
    padding: 6px 16px; font-size: 14px; font-weight: bold; text-align: center;
    letter-spacing: 1px;
  }
  .status-auto { background: #2a2a3a; color: #8888cc; }
  .status-correct { background: #1a3a1a; color: #44ff66; }
  .status-incorrect { background: #3a1a1a; color: #ff4444; }

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
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Spatial Review Viewer</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
    <span id="progress-text" style="font-size:12px; color:#aaa;"></span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportCSV()">Export CSV <span class="key">E</span></button>
    <span id="save-status"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="status-bar" class="status-auto">PENDING REVIEW</div>

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
    <span class="info-item">Predicted laterality: <strong id="info-laterality">--</strong></span>
    <span class="info-item">Involved channels: <strong id="info-n-involved">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Involved: <strong id="info-involved-names">--</strong></span>
  </div>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="bar-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">C</span> Accept (correct) &amp; advance &nbsp;&nbsp;
  <span class="key">I</span> Mark incorrect &amp; advance &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">E</span> Export CSV
</div>

<script>
const CASES = CASES_PLACEHOLDER;

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const LEFT_INDICES = [0,1,2,3,8,9,10,11];
const RIGHT_INDICES = [4,5,6,7,12,13,14,15];
const GROUP_BREAKS = new Set([4, 8, 12, 16]);
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;
const THRESHOLD = 0.5;

// Canvas layout
const EEG_WIDTH = 1200;
const EEG_HEIGHT = 800;
const BAR_HEIGHT = 140;
const MARGIN_LEFT = 90;
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

// Annotations stored per patient
let annotations = {};
try { annotations = JSON.parse(localStorage.getItem('spatial_review_annotations') || '{}'); } catch(e) { annotations = {}; }

function saveAnnotations() {
  localStorage.setItem('spatial_review_annotations', JSON.stringify(annotations));
}

function timeToX(t) { return PLOT_LEFT + (t / DURATION) * PLOT_W; }

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

function probToColor(prob) {
  if (prob >= THRESHOLD) {
    // Green: darker green for higher prob
    const intensity = Math.floor(100 + 155 * prob);
    return `rgb(0, ${intensity}, 0)`;
  } else {
    // Gray
    const g = Math.floor(120 + 80 * prob);
    return `rgb(${g}, ${g}, ${g})`;
  }
}

function probToColorAlpha(prob, alpha) {
  if (prob >= THRESHOLD) {
    const intensity = Math.floor(100 + 155 * prob);
    return `rgba(0, ${intensity}, 0, ${alpha})`;
  } else {
    const g = Math.floor(120 + 80 * prob);
    return `rgba(${g}, ${g}, ${g}, ${alpha})`;
  }
}

function drawEEG() {
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;
  const probs = c.channel_probs;

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

  // Draw traces color-coded by PD probability
  for (let di = 0; di < N_DISPLAY; di++) {
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;

    const yCenter = PLOT_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    const prob = probs[ch.idx];

    // Trace color based on PD probability
    ctx.strokeStyle = probToColor(prob);
    ctx.lineWidth = prob >= THRESHOLD ? 1.0 : 0.6;

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

  // Channel labels with probability
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = PLOT_TOP + chSpacing * (di + 1);
    const prob = probs[ch.idx];

    // Channel name
    ctx.fillStyle = probToColor(prob);
    ctx.font = 'bold 10px Consolas, Monaco, monospace';
    ctx.fillText(ch.name, PLOT_LEFT - 30, yCenter);

    // Probability value
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.fillText(prob.toFixed(2), PLOT_LEFT - 4, yCenter);
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
  const title = c.patient_id + '  |  ' + c.subtype.toUpperCase() +
    '  |  lat=' + c.predicted_laterality +
    '  |  gold=' + c.gold_standard_freq.toFixed(2) + ' Hz' +
    '  |  ' + c.n_involved + '/18 channels';
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Draw discharge timing markers (red dots at each channel center)
  const dischargeTimes = c.discharge_times || [];
  for (const t of dischargeTimes) {
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    // Draw small red circles at each involved channel
    for (let di = 0; di < N_DISPLAY; di++) {
      const ch = DISPLAY_CHANNELS[di];
      if (ch.idx < 0) continue;
      const prob = probs[ch.idx];
      if (prob < THRESHOLD) continue;

      const yCenter = PLOT_TOP + chSpacing * (di + 1);
      ctx.fillStyle = 'rgba(255, 40, 40, 0.7)';
      ctx.beginPath();
      ctx.arc(x, yCenter, 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Also draw a thin vertical line
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.2)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, PLOT_TOP);
    ctx.lineTo(x, PLOT_BOTTOM);
    ctx.stroke();
  }
}

function drawBarChart() {
  const canvas = document.getElementById('bar-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = BAR_HEIGHT;
  const ctx = canvas.getContext('2d');

  const c = CASES[idx];
  const probs = c.channel_probs;

  // Dark background
  ctx.fillStyle = '#2a2a2a';
  ctx.fillRect(0, 0, EEG_WIDTH, BAR_HEIGHT);

  const barTop = 20;
  const barBottom = BAR_HEIGHT - 25;
  const barH = barBottom - barTop;
  const barAreaLeft = PLOT_LEFT;
  const barAreaRight = PLOT_RIGHT;
  const barAreaW = barAreaRight - barAreaLeft;

  // Bar width for 18 channels
  const nCh = 18;
  const gap = 4;
  const barW = (barAreaW - (nCh - 1) * gap) / nCh;

  // Threshold line
  const threshY = barBottom - THRESHOLD * barH;
  ctx.strokeStyle = '#ffcc00';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(barAreaLeft, threshY);
  ctx.lineTo(barAreaRight, threshY);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#ffcc00';
  ctx.font = '9px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('0.5', barAreaLeft - 4, threshY);

  // Title
  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Channel PD Probability', EEG_WIDTH / 2, 2);

  // Draw bars
  for (let ch = 0; ch < nCh; ch++) {
    const prob = probs[ch];
    const x = barAreaLeft + ch * (barW + gap);
    const h = prob * barH;
    const y = barBottom - h;

    // Bar fill
    ctx.fillStyle = probToColorAlpha(prob, 0.8);
    ctx.fillRect(x, y, barW, h);

    // Bar outline
    ctx.strokeStyle = probToColor(prob);
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, barW, h);

    // Channel label below
    ctx.fillStyle = probToColor(prob);
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    // Abbreviated channel name
    const shortName = CHANNEL_NAMES[ch].replace('Fp', 'F\'').substring(0, 5);
    ctx.fillText(shortName, x + barW / 2, barBottom + 2);

    // Probability value on top of bar
    ctx.fillStyle = '#eee';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(prob.toFixed(2), x + barW / 2, y - 1);
  }

  // Hemisphere labels
  ctx.fillStyle = '#888';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  // Left temporal (0-3)
  const ltX = barAreaLeft + 1.5 * (barW + gap);
  ctx.fillText('L-Temp', ltX, barBottom + 13);
  // Right temporal (4-7)
  const rtX = barAreaLeft + 5.5 * (barW + gap);
  ctx.fillText('R-Temp', rtX, barBottom + 13);
  // Left parasagittal (8-11)
  const lpX = barAreaLeft + 9.5 * (barW + gap);
  ctx.fillText('L-Para', lpX, barBottom + 13);
  // Right parasagittal (12-15)
  const rpX = barAreaLeft + 13.5 * (barW + gap);
  ctx.fillText('R-Para', rpX, barBottom + 13);
  // Midline (16-17)
  const mX = barAreaLeft + 16.5 * (barW + gap);
  ctx.fillText('Mid', mX, barBottom + 13);
}

function updateInfoPanel() {
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-gold-freq').textContent = c.gold_standard_freq.toFixed(2) + ' Hz';
  document.getElementById('info-laterality').textContent = c.predicted_laterality;
  document.getElementById('info-n-involved').textContent = c.n_involved + ' / 18';

  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype.toLowerCase();

  // List involved channel names
  const involved = c.involved_channels || [];
  const names = involved.map(i => CHANNEL_NAMES[i]);
  document.getElementById('info-involved-names').textContent =
    names.length > 0 ? names.join(', ') : 'none';
}

function updateStatusBar() {
  const c = CASES[idx];
  const el = document.getElementById('status-bar');
  const ann = annotations[c.patient_id];
  if (!ann) {
    el.textContent = 'PENDING REVIEW';
    el.className = 'status-auto';
  } else if (ann.status === 'correct') {
    el.textContent = 'ACCEPTED (correct)';
    el.className = 'status-correct';
  } else if (ann.status === 'incorrect') {
    el.textContent = 'MARKED INCORRECT';
    el.className = 'status-incorrect';
  } else {
    el.textContent = 'REVIEWED';
    el.className = 'status-auto';
  }
}

function updateProgress() {
  const total = CASES.length;
  let nReviewed = 0;
  let nCorrect = 0;
  let nIncorrect = 0;
  for (const c of CASES) {
    const ann = annotations[c.patient_id];
    if (ann) {
      nReviewed++;
      if (ann.status === 'correct') nCorrect++;
      if (ann.status === 'incorrect') nIncorrect++;
    }
  }
  const pct = total > 0 ? (nReviewed / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-text').textContent =
    nReviewed + ' reviewed (' + nCorrect + ' ok, ' + nIncorrect + ' bad) of ' + total;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + total;
}

function redraw() {
  drawEEG();
  drawBarChart();
  updateInfoPanel();
  updateStatusBar();
  updateProgress();
}

function show() {
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  redraw();
}

function markStatus(status) {
  const c = CASES[idx];
  annotations[c.patient_id] = {
    status: status,
    patient_id: c.patient_id,
    subtype: c.subtype,
    n_involved: c.n_involved,
    predicted_laterality: c.predicted_laterality,
  };
  saveAnnotations();

  const el = document.getElementById('save-status');
  el.textContent = status === 'correct' ? 'ACCEPTED' : 'INCORRECT';
  el.style.color = status === 'correct' ? '#4f4' : '#f44';
  setTimeout(() => { el.textContent = ''; }, 800);

  // Advance
  if (idx < CASES.length - 1) {
    idx++;
    show();
  } else {
    updateStatusBar();
    updateProgress();
  }
}

function exportCSV() {
  let csv = 'patient_id,status,subtype,n_involved,predicted_laterality\n';
  for (const c of CASES) {
    const ann = annotations[c.patient_id];
    const status = ann ? ann.status : 'pending';
    csv += c.patient_id + ',' + status + ',' + c.subtype + ',' +
           c.n_involved + ',' + c.predicted_laterality + '\n';
  }
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'spatial_review_annotations.csv';
  a.click();
}

// ── Keyboard events ──
document.addEventListener('keydown', e => {
  if (e.key === 'c' || e.key === 'C') {
    markStatus('correct');
    e.preventDefault();
  } else if (e.key === 'i' || e.key === 'I') {
    markStatus('incorrect');
    e.preventDefault();
  } else if (e.key === 'ArrowRight') {
    if (idx < CASES.length - 1) { idx++; show(); }
    e.preventDefault();
  } else if (e.key === 'ArrowLeft') {
    if (idx > 0) { idx--; show(); }
    e.preventDefault();
  } else if (e.key === 'e' || e.key === 'E') {
    exportCSV();
    e.preventDefault();
  }
});

// Init
show();
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 72)
    print("Spatial Review Viewer Generator")
    print("=" * 72)

    # ── Load channel involvement predictions ──
    print("\n--- Loading channel involvement predictions ---")
    pred_path = LABELS_DIR / 'channel_involvement_predictions.json'
    with open(str(pred_path)) as f:
        predictions = json.load(f)
    print(f"  Loaded predictions for {len(predictions)} patients")

    # ── Load discharge times ──
    print("\n--- Loading discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    print(f"  Loaded HPP data for {len(hpp_data)} patients")

    # ── Load dataset ──
    print("\n--- Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # ── Build case data ──
    print("\n--- Building case data ---")

    # Lowpass filter setup
    nyq = FS / 2.0
    b_lp, a_lp = butter(4, 20.0 / nyq, btype='low')

    cases_data = []
    n_found = 0

    for _, row in df.iterrows():
        pid = str(row['patient_id'])

        # Must have prediction
        if pid not in predictions:
            continue

        pred = predictions[pid]
        pat_segs = segments.get(pid, [])
        if not pat_segs:
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

        subtype = pred['subtype']
        gold_freq = pred['gold_standard_freq']

        # Get discharge times from HPP
        hpp_entry = hpp_data.get(pid, {})
        discharge_times = hpp_entry.get('global_times', [])

        # Downsample EEG data
        eeg_ds = downsample(seg, DS_LEN)

        case = {
            'patient_id': pid,
            'subtype': subtype,
            'gold_standard_freq': gold_freq,
            'channel_probs': pred['channel_probs'],
            'involved_channels': pred['involved_channels'],
            'predicted_laterality': pred['predicted_laterality'],
            'n_involved': pred['n_involved'],
            'discharge_times': discharge_times,
            'eeg_data': eeg_ds,
        }
        cases_data.append(case)
        n_found += 1

    # Sort: LPD first, then GPD, then by patient_id
    subtype_order = {'lpd': 0, 'gpd': 1}
    cases_data.sort(key=lambda c: (subtype_order.get(c['subtype'], 2), c['patient_id']))

    print(f"  Cases prepared: {n_found}")

    # Count subtypes
    subtype_counts = {}
    for c in cases_data:
        st = c['subtype']
        subtype_counts[st] = subtype_counts.get(st, 0) + 1
    for st, count in sorted(subtype_counts.items()):
        print(f"    {st}: {count}")

    # ── Build HTML ──
    print("\n--- Building HTML viewer ---")

    html = build_html(cases_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases_data, default=_json_default))

    output_path = OUT_DIR / 'spatial_review_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Cases in viewer: {n_found}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Open with: open {output_path}")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
