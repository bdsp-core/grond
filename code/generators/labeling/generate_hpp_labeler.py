"""
HPP-Assisted Discharge Labeling Tool for Harvested Segments.

Semi-automated labeling: user clicks a frequency button (0.25-4.5 Hz),
HPP runs with that frequency as prior and shows inferred discharge times.
User can accept, adjust frequency, or manually add/delete markers.

The "cheating" HPP achieves Spearman >0.95 with gold standard when given
a close-to-correct frequency — so in most cases, clicking the right
frequency button gives near-perfect timing labels instantly.

Targets: 376 harvested hi-freq LPD segments (>2.5 Hz) that need annotation.

Usage:
    conda run -n foe python code/generate_hpp_labeler.py
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
    _compute_channel_evidence, _aggregate_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
LOWPASS_HZ = 20.0

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Frequency buttons: 0.25 to 4.5 Hz in 0.25 steps
FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(1, 19)]  # 0.25 to 4.5


def hpp_with_freq(evidence, fs, freq_hz):
    """Run HPP with a given frequency prior. The 'cheating' version."""
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
    """Load EEG segment for a patient."""
    # Try _seg000 naming (harvested segments)
    path = EEG_DIR / f'{patient_id}_seg000.mat'
    if not path.exists():
        path = EEG_DIR / f'{patient_id}.mat'
    if not path.exists():
        return None

    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    return seg[:18, :2000] if seg.shape[0] >= 18 else None


def compute_evidence(seg):
    """Compute aggregated HPP evidence for a segment."""
    n_ch = min(seg.shape[0], 18)
    n_samples = seg.shape[1]
    evidence_all = np.zeros((n_ch, n_samples))
    for ch in range(n_ch):
        evidence_all[ch] = _compute_channel_evidence(seg[ch], FS)
    # For LPD, use max(left, right) since we don't know laterality
    left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
    right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
    return np.maximum(left_med, right_med)


def downsample(arr, target_len):
    """Downsample array to target length."""
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


def precompute_hpp_results(evidence, freq_buttons):
    """Run HPP for each frequency button value, return dict freq -> times."""
    results = {}
    for freq in freq_buttons:
        times = hpp_with_freq(evidence, FS, freq)
        results[str(freq)] = times
    return results


def main():
    print("=" * 70)
    print("  HPP-Assisted Discharge Labeling Tool")
    print("  For harvested hi-freq LPD segments")
    print("=" * 70)

    # Load harvest manifest
    manifest_path = LABELS_DIR / 'harvest_manifest.json'
    with open(str(manifest_path)) as f:
        manifest = json.load(f)

    # Filter to hi-freq entries (>2.5 Hz)
    hf_entries = {pid: v for pid, v in manifest.items()
                  if '2.5' in v.get('bin', '') or '3.0' in v.get('bin', '') or '3.5' in v.get('bin', '')}
    print(f"Hi-freq entries in manifest: {len(hf_entries)}")

    # Also include all other harvested entries that need labeling
    all_entries = manifest
    print(f"Total manifest entries: {len(all_entries)}")

    # Check which already have discharge timing labels
    hpp_path = LABELS_DIR / 'discharge_times.json'
    existing_labels = {}
    if hpp_path.exists():
        with open(str(hpp_path)) as f:
            existing_labels = json.load(f)

    already_labeled = {pid for pid in all_entries if pid in existing_labels}
    needs_labeling = {pid: v for pid, v in all_entries.items() if pid not in already_labeled}
    print(f"Already labeled: {len(already_labeled)}")
    print(f"Needs labeling: {len(needs_labeling)}")

    # Sort by estimated frequency for efficient review (similar freqs grouped)
    sorted_pids = sorted(needs_labeling.keys(),
                         key=lambda pid: needs_labeling[pid].get('est_freq', 1.0))

    # Build cases data
    print(f"\nProcessing {len(sorted_pids)} cases...")
    cases_data = []
    n_skipped = 0

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    for i, pid in enumerate(sorted_pids):
        entry = needs_labeling[pid]
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

        # Compute evidence
        evidence = compute_evidence(seg)

        # Normalize evidence for display
        ev_max = np.max(evidence)
        ev_display = evidence / ev_max if ev_max > 0 else evidence

        # Precompute HPP results for all frequency buttons
        hpp_results = precompute_hpp_results(evidence, FREQ_BUTTONS)

        est_freq = entry.get('est_freq', 1.0)

        case = {
            'patient_id': str(pid),
            'est_freq': round(est_freq, 2),
            'bin': entry.get('bin', ''),
            'eeg_data': downsample(seg_display, 1000),
            'evidence': downsample(ev_display, 500),
            'hpp_results': hpp_results,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sorted_pids)} processed")

    print(f"  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")

    # Build HTML
    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)

    out_path = OUT_DIR / 'hpp_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")

    # Open in browser
    import subprocess
    subprocess.run(['open', str(out_path)])
    print("  Opened in browser")
    print("=" * 70)


def build_html(cases_data):
    freq_btns_json = json.dumps(FREQ_BUTTONS)

    # Serialize cases
    cases_json = json.dumps(cases_data, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HPP-Assisted Discharge Labeling</title>
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

  #mode-indicator {{
    font-size: 18px; font-weight: bold; padding: 6px 20px;
    text-align: center; letter-spacing: 2px;
  }}
  .mode-add {{ background: #1a3a1a; color: #44ff66; }}
  .mode-delete {{ background: #3a1a1a; color: #ff4444; }}
  .mode-nav {{ background: #1a1a3a; color: #6688ff; }}

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #freq-buttons {{
    display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 16px;
    background: #252525; border-bottom: 1px solid #333; align-items: center;
  }}
  #freq-buttons label {{ color: #aaa; font-size: 13px; margin-right: 8px; font-weight: bold; }}
  .freq-btn {{
    padding: 6px 10px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 13px; font-weight: bold; min-width: 42px; text-align: center;
    transition: all 0.15s;
  }}
  .freq-btn:hover {{ background: #444; border-color: #888; color: #fff; }}
  .freq-btn.active {{ background: #2a5a2a; border-color: #44cc88; color: #44ff66; }}
  .freq-btn.est {{ border-color: #ff9800; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}
  #evidence-canvas {{ display: block; margin: 4px auto 0 auto; cursor: crosshair; }}

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

  #freq-info {{
    font-size: 14px; color: #ff9800; padding: 4px 16px; background: #2a2510;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">HPP-Assisted Discharge Labeling</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="skipCase()" style="border-color:#ff6644; color:#ff6644; background:#3a2a2a;">Skip/Reject <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export Labels <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item">Est freq: <strong id="info-est-freq" style="color:#ff9800;">--</strong></span>
  <span class="info-item">Selected freq: <strong id="info-sel-freq" style="color:#44ff66;">--</strong></span>
  <span class="info-item">Markers: <strong id="info-marker-count" style="color:#ff4444;">--</strong></span>
  <span class="info-item">IPI freq: <strong id="info-ipi-freq">--</strong></span>
  <span class="info-item">Mode: <strong id="info-mode">Navigate</strong></span>
</div>

<div id="freq-buttons">
  <label>Freq prior:</label>
</div>

<div id="freq-info"></div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">A</span> Add mode &nbsp;&nbsp;
  <span class="key">D</span> Delete mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">X</span> Skip/Reject (no PDs) &nbsp;&nbsp;
  <span class="key">1-9</span> Quick freq (×0.5 Hz) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">C</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const GROUP_BREAKS = new Set([4, 8, 12, 16]);
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1200;
const EEG_HEIGHT = 700;
const EV_HEIGHT = 160;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = (EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM);

// State
let idx = 0;
let labeled = new Set();
let mode = 'nav';
let markers = [];
let undoStack = [];
let hoverMarker = -1;
let selectedFreq = null;

// Persistence
const STORAGE_KEY = 'hpp_labeler_v1';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}

function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

// Build frequency buttons
(function() {{
  const container = document.getElementById('freq-buttons');
  for (const freq of FREQ_BUTTONS) {{
    const btn = document.createElement('button');
    btn.className = 'freq-btn';
    btn.textContent = freq.toFixed(2);
    btn.dataset.freq = freq;
    btn.onclick = () => selectFreq(freq);
    container.appendChild(btn);
  }}
}})();

function getDisplayChannels() {{
  const dc = [];
  for (let i = 0; i < 18; i++) {{
    if (GROUP_BREAKS.has(i)) dc.push({{ idx: -1, name: '' }});
    dc.push({{ idx: i, name: CHANNEL_NAMES[i] }});
  }}
  return dc;
}}
const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}
function xToTime(x) {{ return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }}

function findNearestMarker(x) {{
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < markers.length; i++) {{
    const mx = timeToX(markers[i]);
    const dist = Math.abs(mx - x);
    if (dist < bestDist) {{ bestDist = dist; best = i; }}
  }}
  return (bestDist <= 20) ? best : -1;
}}

function pushUndo() {{
  undoStack.push([...markers]);
  if (undoStack.length > 100) undoStack.shift();
}}

function undo() {{
  if (undoStack.length === 0) return;
  markers = undoStack.pop();
  redraw();
}}

function selectFreq(freq) {{
  selectedFreq = freq;
  const c = CASES[idx];
  const hppTimes = c.hpp_results[String(freq)] || [];

  // Replace markers with HPP results
  pushUndo();
  markers = [...hppTimes];

  // Highlight the active button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('active');
    if (parseFloat(btn.dataset.freq) === freq) btn.classList.add('active');
  }});

  // Show info
  const infoEl = document.getElementById('freq-info');
  infoEl.textContent = `HPP with freq=${{freq.toFixed(2)}} Hz → ${{markers.length}} discharges`;

  redraw();
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

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Traces
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
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', timeToX(s), EEG_HEIGHT - MARGIN_BOTTOM + 4);
  }}

  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const freqStr = selectedFreq ? selectedFreq.toFixed(2) : c.est_freq.toFixed(2);
  const title = c.patient_id + '  |  est=' + c.est_freq.toFixed(2) + ' Hz  |  ' + c.bin;
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Discharge markers (red solid)
  for (let mi = 0; mi < markers.length; mi++) {{
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.6)';
    let lw = 2;
    if (mode === 'delete' && hoverMarker === mi) {{
      color = 'rgba(255, 50, 50, 0.9)';
      lw = 4;
    }}

    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();

    ctx.fillStyle = color;
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(t.toFixed(2) + 's', x, MARGIN_TOP - 2);
  }}
}}

function drawEvidence() {{
  const canvas = document.getElementById('evidence-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EV_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const evData = c.evidence;

  const evTop = 10;
  const evBottom = EV_HEIGHT - 20;
  const evH = evBottom - evTop;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EV_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // Evidence trace
  if (evData && evData.length > 0) {{
    const nSamples = evData.length;
    ctx.fillStyle = 'rgba(70, 130, 180, 0.15)';
    ctx.beginPath();
    ctx.moveTo(PLOT_LEFT, evBottom);
    for (let i = 0; i < nSamples; i++) {{
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - evData[i] * evH;
      ctx.lineTo(x, y);
    }}
    ctx.lineTo(PLOT_RIGHT, evBottom);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = 'steelblue';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < nSamples; i++) {{
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - evData[i] * evH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Labels
  ctx.fillStyle = '#000';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('1.0', PLOT_LEFT - 4, evTop);
  ctx.fillText('0.0', PLOT_LEFT - 4, evBottom);
  ctx.textAlign = 'left';
  ctx.fillText('Evidence (pointiness+TKEO)', PLOT_LEFT + 4, evTop - 1);

  // Discharge markers on evidence
  for (let mi = 0; mi < markers.length; mi++) {{
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    let color = 'rgba(255, 0, 0, 0.5)';
    let lw = 1.5;
    if (mode === 'delete' && hoverMarker === mi) {{ color = 'rgba(255, 50, 50, 0.9)'; lw = 3; }}
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }}
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-est-freq').textContent = c.est_freq.toFixed(2) + ' Hz';
  document.getElementById('info-sel-freq').textContent = selectedFreq ? selectedFreq.toFixed(2) + ' Hz' : '--';
  document.getElementById('info-marker-count').textContent = markers.length;

  // IPI freq
  if (markers.length >= 2) {{
    const sorted = [...markers].sort((a,b) => a-b);
    const ipis = [];
    for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i-1]);
    const medIPI = ipis.sort((a,b) => a-b)[Math.floor(ipis.length/2)];
    const ipiFreq = medIPI > 0 ? (1/medIPI).toFixed(2) : '--';
    document.getElementById('info-ipi-freq').textContent = ipiFreq + ' Hz';
  }} else {{
    document.getElementById('info-ipi-freq').textContent = '--';
  }}

  let modeStr = 'Navigate';
  if (mode === 'add') modeStr = 'Add';
  else if (mode === 'delete') modeStr = 'Delete';
  document.getElementById('info-mode').textContent = modeStr;

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';
  document.getElementById('labeled-count').textContent = labeled.size + ' labeled';

  // Highlight estimated freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('est');
    const f = parseFloat(btn.dataset.freq);
    // Find closest button to est_freq
    if (Math.abs(f - c.est_freq) < 0.13) btn.classList.add('est');
  }});
}}

function updateModeIndicator() {{
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') {{ el.textContent = 'ADD MODE (A) -- click to add marker'; el.className = 'mode-add'; }}
  else if (mode === 'delete') {{ el.textContent = 'DELETE MODE (D) -- click near marker to remove'; el.className = 'mode-delete'; }}
  else {{ el.textContent = 'NAVIGATE MODE'; el.className = 'mode-nav'; }}
}}

function autoSave() {{
  const c = CASES[idx];
  allLabels[c.patient_id] = {{
    times: [...markers].sort((a, b) => a - b),
    selected_freq: selectedFreq,
    est_freq: c.est_freq,
    rejected: markers.length === 0 && reviewed.has(c.patient_id),
  }};
  if (reviewed.has(c.patient_id)) labeled.add(c.patient_id);
  saveAll();
}}

let reviewed = new Set();

function markReviewed() {{
  const c = CASES[idx];
  reviewed.add(c.patient_id);
}}

function skipCase() {{
  const c = CASES[idx];
  pushUndo();
  markers = [];
  selectedFreq = null;
  reviewed.add(c.patient_id);
  allLabels[c.patient_id] = {{
    times: [],
    selected_freq: null,
    est_freq: c.est_freq,
    rejected: true,
  }};
  labeled.add(c.patient_id);
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (no PDs)';
  el.style.color = '#ff6644';
  setTimeout(() => {{ el.textContent = ''; }}, 1500);
  // Advance
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function redraw() {{
  drawEEG();
  drawEvidence();
  updateInfo();
  updateModeIndicator();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load from storage or start fresh
  if (allLabels[c.patient_id] && allLabels[c.patient_id].times) {{
    markers = [...allLabels[c.patient_id].times];
    selectedFreq = allLabels[c.patient_id].selected_freq || null;
    labeled.add(c.patient_id);
  }} else {{
    markers = [];
    selectedFreq = null;
  }}
  undoStack = [];
  hoverMarker = -1;

  // Highlight active freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('active');
    if (selectedFreq && parseFloat(btn.dataset.freq) === selectedFreq) btn.classList.add('active');
  }});

  document.getElementById('freq-info').textContent = '';
  redraw();
}}

function exportJSON() {{
  autoSave();
  const out = {{}};
  for (const c of CASES) {{
    const pid = c.patient_id;
    if (allLabels[pid]) {{
      const isRejected = allLabels[pid].rejected === true;
      out[pid] = {{
        patient_id: pid,
        global_times: allLabels[pid].times,
        selected_freq: allLabels[pid].selected_freq,
        est_freq: c.est_freq,
        review_status: isRejected ? 'rejected' : 'ground_truth',
        rejected: isRejected,
        source: 'hpp_labeler',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'hpp_labeler_discharge_times.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' cases';
  el.style.color = '#4f4';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Canvas click handlers
const eegCanvas = document.getElementById('eeg-canvas');
const evCanvas = document.getElementById('evidence-canvas');

function handleCanvasClick(e) {{
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;

  if (x < PLOT_LEFT || x > PLOT_RIGHT) return;
  const t = xToTime(x);

  if (mode === 'add') {{
    if (t >= 0 && t <= DURATION) {{
      pushUndo();
      markers.push(t);
      redraw();
    }}
  }} else if (mode === 'delete') {{
    const mi = findNearestMarker(x);
    if (mi >= 0) {{
      pushUndo();
      markers.splice(mi, 1);
      redraw();
    }}
  }}
}}

function handleCanvasMove(e) {{
  if (mode !== 'delete') return;
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;
  const newHover = findNearestMarker(x);
  if (newHover !== hoverMarker) {{
    hoverMarker = newHover;
    redraw();
  }}
}}

eegCanvas.addEventListener('click', handleCanvasClick);
eegCanvas.addEventListener('mousemove', handleCanvasMove);
evCanvas.addEventListener('click', handleCanvasClick);
evCanvas.addEventListener('mousemove', handleCanvasMove);

// Keyboard
document.addEventListener('keydown', (e) => {{
  if (e.key === 'a' || e.key === 'A') {{
    mode = mode === 'add' ? 'nav' : 'add';
    redraw();
  }} else if (e.key === 'd' || e.key === 'D') {{
    mode = mode === 'delete' ? 'nav' : 'delete';
    redraw();
  }} else if (e.key === 'Escape') {{
    mode = 'nav';
    redraw();
  }} else if (e.key === 'z' || e.key === 'Z') {{
    undo();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    autoSave();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight' || e.key === 'c' || e.key === 'C' || e.key === 'Enter') {{
    e.preventDefault();
    if (e.key === 'c' || e.key === 'C' || e.key === 'Enter') markReviewed();
    autoSave();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'x' || e.key === 'X') {{
    skipCase();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }} else if (e.key >= '1' && e.key <= '9') {{
    // Quick freq: key 1=0.5, 2=1.0, ..., 9=4.5
    const freqVal = parseInt(e.key) * 0.5;
    const closest = FREQ_BUTTONS.reduce((prev, curr) =>
      Math.abs(curr - freqVal) < Math.abs(prev - freqVal) ? curr : prev);
    selectFreq(closest);
  }}
}});

// Init
show();
</script>

</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
