"""
Generate HTML viewer for reviewing auto-labeled RDA wave peaks.

Features:
  - Pre-rendered EEG images with wave peak markers + filtered signal overlay
  - Accept/Reject buttons for en-bloc review
  - Interactive add/delete mode for manual peak editing (click to add, right-click to delete)
  - Filter by review tier (1=auto-accept, 2=quick review, 3=manual edit)
  - Batch-accept all Tier 1 cases with one click
  - Export results as JSON

Usage:
    conda run -n foe python code/rda_detector/generate_rda_wave_reviewer.py
"""

import sys
import json
import base64
import io
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import detrend, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

# ── Constants ────────────────────────────────────────────────────────
FS = 200
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
RDA_CACHE_DIR = DATA_DIR / 'rda_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

LEFT_CHANNELS = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CHANNELS = [4, 5, 6, 7, 12, 13, 14, 15]

# Display order: left temporal, left parasag, spacer, midline, spacer, right parasag, right temporal
DISPLAY_ORDER = [
    (0, 'Fp1-F7'), (1, 'F7-T3'), (2, 'T3-T5'), (3, 'T5-O1'),
    (8, 'Fp1-F3'), (9, 'F3-C3'), (10, 'C3-P3'), (11, 'P3-O1'),
    (None, ''),  # spacer
    (16, 'Fz-Cz'), (17, 'Cz-Pz'),
    (None, ''),  # spacer
    (12, 'Fp2-F4'), (13, 'F4-C4'), (14, 'C4-P4'), (15, 'P4-O2'),
    (4, 'Fp2-F8'), (5, 'F8-T4'), (6, 'T4-T6'), (7, 'T6-O2'),
]

GROUP_BREAKS = {4, 8, 12, 16}  # indices into BIPOLAR_CHANNELS where spacers go


def load_segment(mat_file):
    """Load a .mat file and return (18, N) bipolar array."""
    import scipy.io as sio

    segment_id = Path(mat_file).stem
    cache_path = RDA_CACHE_DIR / f'{segment_id}.npy'
    if cache_path.exists():
        seg = np.load(str(cache_path))
        if seg.shape[0] == 18:
            return seg

    mat_path = EEG_DIR / mat_file
    if not mat_path.exists():
        return None

    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)

    if data.shape[0] == 20 or (data.shape[1] == 20 and data.shape[0] != 20):
        if data.shape[1] == 20:
            data = data.T
        seg_bi = np.array(fcn_getBanana(data), dtype=np.float64)
    elif data.shape[0] == 18:
        seg_bi = data.astype(np.float64)
    elif data.shape[1] == 18:
        seg_bi = data.T.astype(np.float64)
    else:
        return None

    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')

    return seg_bi


def bandpass_filter(signal, f_center, bandwidth=0.4, fs=FS, order=4):
    """Zero-phase bandpass around f_center ± bandwidth."""
    nyq = fs / 2.0
    lo = max(f_center - bandwidth, 0.1) / nyq
    hi = min(f_center + bandwidth, nyq - 0.1) / nyq
    if lo >= hi or lo <= 0 or hi >= 1:
        return signal
    b, a = butter(order, [lo, hi], btype='band')
    try:
        return filtfilt(b, a, signal)
    except ValueError:
        return signal


def generate_eeg_image(seg_bi, wave_label, fs=FS):
    """Generate EEG JPEG with wave peak markers and filtered signal overlay.

    Returns JPEG bytes.
    """
    seg_bi = seg_bi.astype(np.float64)
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / fs, n_samples)

    # Lowpass at 20 Hz for display
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i] = filtfilt(b, a, seg_bi[i])
            except ValueError:
                pass
    for i in range(n_channels):
        seg_bi[i] = detrend(seg_bi[i], type='linear')

    z_scale = 0.01
    clip_uv = 300.0

    # Build display list with spacers
    display_channels = []
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    ch_to_offset = {}
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        if ch_idx is not None:
            ch_to_offset[ch_idx] = offset

    # Figure: EEG + filtered signal + evidence
    fig, (ax, ax_filt) = plt.subplots(
        2, 1, figsize=(14, 11),
        gridspec_kw={'height_ratios': [10, 1.5], 'hspace': 0.08})
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw EEG traces
    yticks, ytick_labels = [], []
    involved = wave_label.get('involved_channels', [])

    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)
        if ch_idx is None:
            continue

        clipped = np.clip(seg_bi[ch_idx], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset

        # Color: involved channels highlighted
        if ch_idx in involved:
            color = '#C0392B' if ch_idx in LEFT_CHANNELS else '#2980B9'
            lw = 0.7
        elif ch_idx in (16, 17):
            color = '#7F8C8D'
            lw = 0.5
        else:
            color = '#2C3E50'
            lw = 0.5

        ax.plot(time_vec, scaled, color=color, linewidth=lw, clip_on=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(0, n_display + 1)
    ax.set_xlim(0, n_samples / fs)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['bottom'].set_color('#999')

    # Wave peak markers
    wave_peaks = wave_label.get('wave_peaks', [])
    for t in wave_peaks:
        ax.axvline(x=t, color='#E74C3C', linestyle='--', alpha=0.4,
                   linewidth=0.6, zorder=5)
        samp = int(round(t * fs))
        samp = max(0, min(samp, n_samples - 1))
        for ch_idx in involved:
            if ch_idx in ch_to_offset:
                offset = ch_to_offset[ch_idx]
                clipped_val = np.clip(seg_bi[ch_idx, samp], -clip_uv, clip_uv)
                y_val = z_scale * clipped_val + offset
                ax.plot(t, y_val, 'o', color='#E74C3C', markersize=3,
                        zorder=10, markeredgewidth=0.3, markeredgecolor='#C0392B')

    # Title
    pid = wave_label.get('patient_id', '??')
    subtype = wave_label.get('subtype', '??').upper()
    freq_ann = wave_label.get('annotated_freq', 0)
    freq_det = wave_label.get('detected_freq', 0)
    n_waves = wave_label.get('n_waves', 0)
    ipi_cv = wave_label.get('ipi_cv', 0)
    tier = wave_label.get('review_tier', '?')
    lat = wave_label.get('laterality', '?')

    title = (f"{pid}  |  {subtype}  |  Annotated: {freq_ann:.2f} Hz  |  "
             f"Detected: {freq_det:.2f} Hz  |  {n_waves} waves  |  "
             f"IPI CV: {ipi_cv:.3f}  |  Tier {tier}  |  {lat}")
    fig.suptitle(title, fontsize=10, fontweight='bold', y=0.98)

    # Filtered signal subplot
    ax_filt.set_facecolor('white')
    if involved:
        bandwidth = min(0.4, freq_ann * 0.3)
        bandwidth = max(bandwidth, 0.15)
        # Reload raw (unfiltered at 20Hz) for bandpass — use the original
        # Actually we already lowpassed, so just bandpass the averaged involved channels
        filtered_chs = []
        for ch in involved:
            f = bandpass_filter(seg_bi[ch], freq_ann, bandwidth)
            std = np.std(f)
            if std > 1e-8:
                f = f / std
            filtered_chs.append(f)
        avg_filt = np.mean(filtered_chs, axis=0)
        avg_filt = gaussian_filter1d(avg_filt, sigma=2)

        filt_time = np.linspace(0, len(avg_filt) / fs, len(avg_filt))
        ax_filt.fill_between(filt_time, 0, avg_filt, where=avg_filt > 0,
                             color='steelblue', alpha=0.3)
        ax_filt.fill_between(filt_time, 0, avg_filt, where=avg_filt < 0,
                             color='#cc6666', alpha=0.3)
        ax_filt.plot(filt_time, avg_filt, color='steelblue', linewidth=0.8)

        for t in wave_peaks:
            ax_filt.axvline(x=t, color='#E74C3C', linestyle='--', alpha=0.5,
                            linewidth=0.6, zorder=5)

    ax_filt.set_xlim(0, n_samples / fs)
    ax_filt.set_ylabel('Filtered', fontsize=8)
    ax_filt.set_xlabel('Time (seconds)', fontsize=8)
    ax_filt.tick_params(labelsize=7)
    ax_filt.xaxis.set_major_locator(MultipleLocator(1))
    ax_filt.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax_filt.spines['top'].set_visible(False)
    ax_filt.spines['right'].set_visible(False)

    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.04)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 75})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_html(cases_json, image_data):
    """Build the RDA wave review HTML viewer with interactive peak editing."""

    html = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RDA Wave Peak Review</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }

  #header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
    border-bottom: 2px solid #444;
  }
  #header-left { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  #header-right { display: flex; align-items: center; gap: 12px; font-size: 13px; }

  .key { background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }

  #progress-bar-wrap {
    width: 100%; height: 6px; background: #333; border-radius: 3px; overflow: hidden;
  }
  #progress-bar { height: 100%; background: #44cc88; transition: width 0.2s; }

  #info-panel {
    background: #2a2a2a; padding: 8px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }
  .info-col { display: flex; flex-direction: column; gap: 3px; }
  .info-item { color: #bbb; }
  .info-item strong { color: #eee; }
  .info-badge {
    padding: 4px 12px; border-radius: 5px; font-size: 13px; font-weight: bold;
    display: inline-block;
  }
  .badge-lrda { background: #5a2020; color: #ff8888; }
  .badge-grda { background: #20205a; color: #8888ff; }
  .badge-tier1 { background: #1a3a1a; color: #44cc88; }
  .badge-tier2 { background: #3a3a1a; color: #cccc44; }
  .badge-tier3 { background: #5a2020; color: #ff8888; }

  #annotation-panel {
    background: #2a2a2a; padding: 10px 16px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; flex-wrap: wrap; border-bottom: 2px solid #444;
  }

  .anno-btn {
    padding: 12px 30px; border: 3px solid #555; border-radius: 10px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 16px; font-weight: bold;
    min-width: 120px; text-align: center; transition: all 0.15s;
  }
  .anno-btn:hover { filter: brightness(1.2); }
  .anno-btn.selected { box-shadow: 0 0 15px; }

  .btn-accept { background: #1a3a1a; border-color: #44cc88; color: #44cc88; }
  .btn-accept.selected { background: #2a5a2a; border-color: #66ff88; box-shadow: 0 0 15px #44cc88; }
  .btn-reject { background: #5a2020; border-color: #cc3333; color: #ff8888; }
  .btn-reject.selected { background: #8a2020; border-color: #ff4444; box-shadow: 0 0 15px #ff4444; }
  .btn-edit { background: #3a3a1a; border-color: #cccc44; color: #cccc44; }
  .btn-edit.selected { background: #5a5a2a; border-color: #ffff44; box-shadow: 0 0 15px #cccc44; }

  .batch-btn {
    padding: 8px 16px; border: 2px solid #44cc88; border-radius: 6px;
    background: #1a3a1a; color: #44cc88; cursor: pointer;
    font-family: monospace; font-size: 13px; font-weight: bold;
  }
  .batch-btn:hover { background: #2a5a2a; }

  #img-wrapper {
    position: relative; text-align: center; padding: 4px;
    cursor: default;
  }
  #img-wrapper.edit-mode { cursor: crosshair; }
  #viewer { max-width: 100%; max-height: calc(100vh - 310px); display: block; margin: 0 auto; }
  #edit-canvas {
    position: absolute; top: 0; left: 0; pointer-events: none;
  }
  #img-wrapper.edit-mode #edit-canvas { pointer-events: auto; cursor: crosshair; }

  #edit-bar {
    background: #333; padding: 6px 16px; display: none; align-items: center;
    gap: 12px; border-bottom: 1px solid #555; font-size: 13px;
  }
  #edit-bar.visible { display: flex; }

  .edit-info { color: #cccc44; font-style: italic; }

  #save-status { color: #44cc44; font-size: 13px; }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333;
  }

  .export-btn {
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }
  .export-btn:hover { background: #3a4a3a; }

  select { font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">RDA Wave Peak Review</span>
    <select id="filter-mode" onchange="filterChanged()">
      <option value="all">All cases</option>
      <option value="tier1">Tier 1 (auto-accept)</option>
      <option value="tier2">Tier 2 (quick review)</option>
      <option value="tier3">Tier 3 (manual edit)</option>
      <option value="unmarked">Unmarked</option>
      <option value="accepted">Accepted</option>
      <option value="rejected">Rejected</option>
      <option value="edited">Edited</option>
    </select>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
    <span id="progress-text" style="font-size:12px; color:#aaa;"></span>
  </div>
  <div id="header-right">
    <button class="batch-btn" onclick="batchAcceptTier1()">Accept all Tier 1</button>
    <button class="export-btn" onclick="exportJSON()">Export JSON <span class="key">E</span></button>
    <span id="save-status"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <div class="info-col">
    <span class="info-badge" id="info-subtype-badge">--</span>
    <span class="info-badge" id="info-tier-badge" style="margin-top:4px">--</span>
  </div>
  <div class="info-col">
    <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
    <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
    <span class="info-item">Laterality: <strong id="info-lat">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Annotated freq: <strong id="info-ann-freq">--</strong></span>
    <span class="info-item">Detected freq: <strong id="info-det-freq">--</strong></span>
    <span class="info-item">N waves: <strong id="info-n-waves">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">IPI CV: <strong id="info-ipi-cv">--</strong></span>
    <span class="info-item">Wave peaks: <strong id="info-peaks" style="font-size:11px; max-width:350px; word-wrap:break-word;">--</strong></span>
  </div>
</div>

<div id="annotation-panel">
  <button class="anno-btn btn-accept" onclick="annotate('accepted')">ACCEPT<br><span class="key">A</span></button>
  <button class="anno-btn btn-reject" onclick="annotate('rejected')">REJECT<br><span class="key">R</span></button>
  <button class="anno-btn btn-edit" onclick="toggleEditMode()">EDIT PEAKS<br><span class="key">W</span></button>
</div>

<div id="edit-bar">
  <span class="edit-info">EDIT MODE: Click to add peak | Right-click to delete nearest | <span class="key">Z</span> Undo | <span class="key">W</span> Exit edit</span>
  <span style="color:#888; margin-left:20px;">Peaks: <strong id="edit-peak-count">0</strong></span>
</div>

<div id="img-wrapper">
  <img id="viewer" src="" alt="Loading..." />
  <canvas id="edit-canvas" width="0" height="0"></canvas>
</div>

<div id="shortcuts">
  <span class="key">A</span> Accept &nbsp;&nbsp;
  <span class="key">R</span> Reject &nbsp;&nbsp;
  <span class="key">W</span> Toggle edit &nbsp;&nbsp;
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">Z</span> Undo (in edit mode) &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = CASES_PLACEHOLDER;
const IMAGE_DATA = IMAGE_PLACEHOLDER;

let annotations = {};   // pid -> {status, wave_peaks}
let filteredItems = [];
let idx = 0;
let editMode = false;
let editHistory = [];    // for undo

// Image geometry (set after image loads)
let imgRect = null;
const SEGMENT_DURATION = 10.0;  // seconds

// Pixel coordinates of the EEG plot area within the image
// These are approximate for the matplotlib output at dpi=100, figsize=(14,11)
// Left margin ~6.5%, right margin ~1%, top ~5%, bottom ~15% (including filtered subplot)
// The EEG subplot is roughly the top 87% of the figure
const PLOT_LEFT_FRAC = 0.065;
const PLOT_RIGHT_FRAC = 0.99;
const PLOT_TOP_FRAC = 0.05;     // top of EEG area
const PLOT_BOTTOM_FRAC = 0.87;  // bottom of EEG area (above filtered subplot)

// Load saved annotations
try {
  annotations = JSON.parse(localStorage.getItem('rda_wave_review') || '{}');
} catch(e) { annotations = {}; }

function saveAnnotations() {
  localStorage.setItem('rda_wave_review', JSON.stringify(annotations));
}

function getAnnotation(pid) {
  return annotations[pid] || null;
}

function setAnnotation(pid, status, peaks) {
  if (!annotations[pid]) annotations[pid] = {};
  annotations[pid].status = status;
  if (peaks !== undefined) annotations[pid].wave_peaks = peaks;
  saveAnnotations();
}

function getCurrentPeaks() {
  if (filteredItems.length === 0) return [];
  const item = filteredItems[idx];
  const ann = getAnnotation(item.patient_id);
  if (ann && ann.wave_peaks) return [...ann.wave_peaks];
  return [...(item.wave_peaks || [])];
}

function setCurrentPeaks(peaks) {
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  if (!annotations[item.patient_id]) annotations[item.patient_id] = {};
  annotations[item.patient_id].wave_peaks = peaks;
  annotations[item.patient_id].status = 'edited';
  saveAnnotations();
}

function updateProgress() {
  const total = CASES.length;
  let nReviewed = 0, nAccepted = 0, nRejected = 0, nEdited = 0;
  for (const c of CASES) {
    const a = getAnnotation(c.patient_id);
    if (a) {
      nReviewed++;
      if (a.status === 'accepted') nAccepted++;
      else if (a.status === 'rejected') nRejected++;
      else if (a.status === 'edited') nEdited++;
    }
  }
  const pct = total > 0 ? (nReviewed / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-text').textContent =
    nReviewed + '/' + total + ' reviewed (' + nAccepted + ' accepted, ' +
    nRejected + ' rejected, ' + nEdited + ' edited)';
}

function filterChanged() {
  const mode = document.getElementById('filter-mode').value;
  filteredItems = CASES.filter(item => {
    const a = getAnnotation(item.patient_id);
    if (mode === 'tier1') return item.review_tier === 1;
    if (mode === 'tier2') return item.review_tier === 2;
    if (mode === 'tier3') return item.review_tier === 3;
    if (mode === 'unmarked') return !a;
    if (mode === 'accepted') return a && a.status === 'accepted';
    if (mode === 'rejected') return a && a.status === 'rejected';
    if (mode === 'edited') return a && a.status === 'edited';
    return true;
  });
  idx = 0;
  exitEditMode();
  show();
}

function show() {
  updateProgress();

  if (filteredItems.length === 0) {
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('info-pid').textContent = '--';
    updateButtons(null);
    return;
  }
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image
  const b64 = IMAGE_DATA[item.patient_id];
  if (b64) {
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  }

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Info panel
  document.getElementById('info-pid').textContent = item.patient_id;
  document.getElementById('info-subtype').textContent = item.subtype.toUpperCase();
  document.getElementById('info-lat').textContent = item.laterality || '?';

  const stBadge = document.getElementById('info-subtype-badge');
  stBadge.textContent = item.subtype.toUpperCase();
  stBadge.className = 'info-badge badge-' + item.subtype;

  const tierBadge = document.getElementById('info-tier-badge');
  tierBadge.textContent = 'Tier ' + item.review_tier;
  tierBadge.className = 'info-badge badge-tier' + item.review_tier;

  document.getElementById('info-ann-freq').textContent = item.annotated_freq.toFixed(2) + ' Hz';
  document.getElementById('info-det-freq').textContent =
    (item.detected_freq && isFinite(item.detected_freq)) ? item.detected_freq.toFixed(2) + ' Hz' : 'N/A';
  document.getElementById('info-n-waves').textContent = item.n_waves;
  document.getElementById('info-ipi-cv').textContent =
    (item.ipi_cv && isFinite(item.ipi_cv)) ? item.ipi_cv.toFixed(3) : 'N/A';

  const peaks = getCurrentPeaks();
  document.getElementById('info-peaks').textContent =
    peaks.length > 0 ? peaks.map(t => t.toFixed(2) + 's').join(', ') : 'None';

  updateButtons(item);
  if (editMode) drawEditOverlay();
}

function updateButtons(item) {
  const btns = document.querySelectorAll('.anno-btn');
  btns.forEach(b => b.classList.remove('selected'));
  if (!item) return;
  const a = getAnnotation(item.patient_id);
  if (a) {
    if (a.status === 'accepted') document.querySelector('.btn-accept').classList.add('selected');
    else if (a.status === 'rejected') document.querySelector('.btn-reject').classList.add('selected');
    else if (a.status === 'edited') document.querySelector('.btn-edit').classList.add('selected');
  }
}

function annotate(status) {
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  setAnnotation(item.patient_id, status);

  document.getElementById('save-status').textContent = 'Saved: ' + status.toUpperCase();
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 800);

  updateButtons(item);
  updateProgress();

  // Auto-advance
  if (!editMode && idx < filteredItems.length - 1) {
    setTimeout(() => { idx++; show(); }, 200);
  }
}

function batchAcceptTier1() {
  let count = 0;
  for (const c of CASES) {
    if (c.review_tier === 1 && !getAnnotation(c.patient_id)) {
      setAnnotation(c.patient_id, 'accepted', c.wave_peaks);
      count++;
    }
  }
  document.getElementById('save-status').textContent = 'Batch accepted ' + count + ' Tier 1 cases';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 2000);
  updateProgress();
  show();
}

// ── Edit mode ────────────────────────────────────────────────────────

function toggleEditMode() {
  if (editMode) exitEditMode();
  else enterEditMode();
}

function enterEditMode() {
  editMode = true;
  editHistory = [];
  document.getElementById('img-wrapper').classList.add('edit-mode');
  document.getElementById('edit-bar').classList.add('visible');
  document.querySelector('.btn-edit').classList.add('selected');
  setupCanvas();
  drawEditOverlay();
}

function exitEditMode() {
  editMode = false;
  document.getElementById('img-wrapper').classList.remove('edit-mode');
  document.getElementById('edit-bar').classList.remove('visible');
  const canvas = document.getElementById('edit-canvas');
  canvas.width = 0;
  canvas.height = 0;
}

function setupCanvas() {
  const img = document.getElementById('viewer');
  const canvas = document.getElementById('edit-canvas');
  const wrapper = document.getElementById('img-wrapper');

  // Match canvas to image display size
  const rect = img.getBoundingClientRect();
  const wrapRect = wrapper.getBoundingClientRect();

  canvas.width = rect.width;
  canvas.height = rect.height;
  canvas.style.left = (rect.left - wrapRect.left) + 'px';
  canvas.style.top = (rect.top - wrapRect.top) + 'px';
  canvas.style.width = rect.width + 'px';
  canvas.style.height = rect.height + 'px';

  imgRect = rect;
}

function pixelToTime(x) {
  // Convert canvas x pixel to time in seconds
  if (!imgRect) return null;
  const canvas = document.getElementById('edit-canvas');
  const plotLeft = canvas.width * PLOT_LEFT_FRAC;
  const plotRight = canvas.width * PLOT_RIGHT_FRAC;
  const plotWidth = plotRight - plotLeft;

  const relX = (x - plotLeft) / plotWidth;
  if (relX < 0 || relX > 1) return null;
  return relX * SEGMENT_DURATION;
}

function timeToPixel(t) {
  const canvas = document.getElementById('edit-canvas');
  const plotLeft = canvas.width * PLOT_LEFT_FRAC;
  const plotRight = canvas.width * PLOT_RIGHT_FRAC;
  const plotWidth = plotRight - plotLeft;
  return plotLeft + (t / SEGMENT_DURATION) * plotWidth;
}

function drawEditOverlay() {
  const canvas = document.getElementById('edit-canvas');
  if (canvas.width === 0) return;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const peaks = getCurrentPeaks();
  document.getElementById('edit-peak-count').textContent = peaks.length;

  // Draw peak lines on canvas
  const plotTop = canvas.height * PLOT_TOP_FRAC;
  const plotBottom = canvas.height * PLOT_BOTTOM_FRAC;

  for (const t of peaks) {
    const x = timeToPixel(t);
    ctx.beginPath();
    ctx.moveTo(x, plotTop);
    ctx.lineTo(x, plotBottom);
    ctx.strokeStyle = 'rgba(0, 255, 100, 0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Small circle at top
    ctx.beginPath();
    ctx.arc(x, plotTop + 10, 5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 255, 100, 0.8)';
    ctx.fill();
  }
}

// Canvas click handlers
document.getElementById('edit-canvas').addEventListener('click', function(e) {
  if (!editMode) return;
  const rect = this.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const t = pixelToTime(x);
  if (t === null || t < 0 || t > SEGMENT_DURATION) return;

  // Add a peak
  const peaks = getCurrentPeaks();
  editHistory.push([...peaks]);
  peaks.push(Math.round(t * 10000) / 10000);
  peaks.sort((a, b) => a - b);
  setCurrentPeaks(peaks);
  drawEditOverlay();

  // Update info
  document.getElementById('info-peaks').textContent =
    peaks.map(t => t.toFixed(2) + 's').join(', ');
  document.getElementById('info-n-waves').textContent = peaks.length;
  updateButtons(filteredItems[idx]);
});

document.getElementById('edit-canvas').addEventListener('contextmenu', function(e) {
  e.preventDefault();
  if (!editMode) return;
  const rect = this.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const t = pixelToTime(x);
  if (t === null) return;

  // Delete nearest peak within 0.15s
  const peaks = getCurrentPeaks();
  if (peaks.length === 0) return;

  let minDist = Infinity, minIdx = -1;
  for (let i = 0; i < peaks.length; i++) {
    const d = Math.abs(peaks[i] - t);
    if (d < minDist) { minDist = d; minIdx = i; }
  }

  if (minDist < 0.15 && minIdx >= 0) {
    editHistory.push([...peaks]);
    peaks.splice(minIdx, 1);
    setCurrentPeaks(peaks);
    drawEditOverlay();

    document.getElementById('info-peaks').textContent =
      peaks.length > 0 ? peaks.map(t => t.toFixed(2) + 's').join(', ') : 'None';
    document.getElementById('info-n-waves').textContent = peaks.length;
    updateButtons(filteredItems[idx]);
  }
});

// Undo
function undoEdit() {
  if (editHistory.length === 0) return;
  const prev = editHistory.pop();
  setCurrentPeaks(prev);
  drawEditOverlay();
  const peaks = prev;
  document.getElementById('info-peaks').textContent =
    peaks.length > 0 ? peaks.map(t => t.toFixed(2) + 's').join(', ') : 'None';
  document.getElementById('info-n-waves').textContent = peaks.length;
}

// Resize handler
window.addEventListener('resize', () => {
  if (editMode) { setupCanvas(); drawEditOverlay(); }
});
document.getElementById('viewer').addEventListener('load', () => {
  if (editMode) { setupCanvas(); drawEditOverlay(); }
});

// Export
function exportJSON() {
  const result = {};
  for (const c of CASES) {
    const ann = getAnnotation(c.patient_id);
    if (!ann) continue;
    result[c.patient_id] = {
      patient_id: c.patient_id,
      subtype: c.subtype,
      annotated_freq: c.annotated_freq,
      review_status: ann.status,
      wave_peaks: ann.wave_peaks || c.wave_peaks,
      n_waves: (ann.wave_peaks || c.wave_peaks).length,
    };
  }
  const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rda_wave_review_results.json';
  a.click();

  document.getElementById('save-status').textContent = 'Exported ' + Object.keys(result).length + ' cases';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 2000);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;

  if (e.key === 'ArrowRight') { idx = Math.min(idx + 1, filteredItems.length - 1); exitEditMode(); show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx - 1, 0); exitEditMode(); show(); e.preventDefault(); }
  else if (e.key === 'a' || e.key === 'A') { annotate('accepted'); e.preventDefault(); }
  else if (e.key === 'r' || e.key === 'R') { annotate('rejected'); e.preventDefault(); }
  else if (e.key === 'w' || e.key === 'W') { toggleEditMode(); e.preventDefault(); }
  else if (e.key === 'z' || e.key === 'Z') { undoEdit(); e.preventDefault(); }
  else if (e.key === 'e' || e.key === 'E') { exportJSON(); e.preventDefault(); }
});

// Init
filterChanged();
</script>
</body>
</html>"""
    return html


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main():
    print("=" * 72)
    print("RDA Wave Peak Review Viewer Generator")
    print("=" * 72)

    # ── Load wave labels ──────────────────────────────────────────────
    labels_path = LABELS_DIR / 'rda_wave_labels.json'
    with open(str(labels_path)) as f:
        wave_labels = json.load(f)
    print(f"Loaded wave labels for {len(wave_labels)} cases")

    # ── Load segment info ──────────────────────────────────────────────
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    rda_segs = seg_df[seg_df['subtype'].isin(['grda', 'lrda'])].copy()
    seg_lookup = {}
    for _, row in rda_segs.iterrows():
        pid = str(row['patient_id'])
        if pid not in seg_lookup:
            seg_lookup[pid] = row['mat_file']

    # ── Sort: Tier 3 first (need most attention), then Tier 2, then Tier 1 ──
    sorted_pids = sorted(wave_labels.keys(),
                         key=lambda p: (-wave_labels[p]['review_tier'],
                                        wave_labels[p]['annotated_freq']))

    # ── Generate images ───────────────────────────────────────────────
    print("\nGenerating EEG images with wave markers...")
    cases = []
    image_data = {}
    n_generated = 0

    for pi, pid in enumerate(sorted_pids):
        wl = wave_labels[pid]
        mat_file = seg_lookup.get(pid)
        if not mat_file:
            continue

        seg_bi = load_segment(mat_file)
        if seg_bi is None:
            continue

        try:
            jpeg_bytes = generate_eeg_image(seg_bi, wl)
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
            n_generated += 1
        except Exception as e:
            print(f"  FAILED: {pid}: {e}")
            continue

        cases.append({
            'patient_id': pid,
            'subtype': wl['subtype'],
            'annotated_freq': wl['annotated_freq'],
            'detected_freq': wl['detected_freq'],
            'n_waves': wl['n_waves'],
            'ipi_cv': wl['ipi_cv'],
            'review_tier': wl['review_tier'],
            'laterality': wl['laterality'],
            'wave_peaks': wl['wave_peaks'],
        })

        if (pi + 1) % 50 == 0:
            print(f"  Generated {n_generated}/{pi+1} images...")

    print(f"Total images: {n_generated}")

    tier_counts = {1: 0, 2: 0, 3: 0}
    for c in cases:
        tier_counts[c['review_tier']] += 1

    # ── Build HTML ────────────────────────────────────────────────────
    print("\nBuilding HTML viewer...")

    html = build_html(cases, image_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases, default=_json_default))
    html = html.replace('IMAGE_PLACEHOLDER', json.dumps(image_data))

    output_path = RESULTS_DIR / 'rda_wave_review_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total cases: {len(cases)}")
    print(f"  Tier 1 (auto-accept): {tier_counts[1]}")
    print(f"  Tier 2 (quick review): {tier_counts[2]}")
    print(f"  Tier 3 (manual edit): {tier_counts[3]}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Open with: open '{output_path}'")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
