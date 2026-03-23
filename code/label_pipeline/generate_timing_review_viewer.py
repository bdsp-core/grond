"""
Generate HTML viewer for reviewing HPP-detected discharge times.

Binary review: MW marks each case as Correct (C) or Incorrect (I).
Generates EEG images with discharge timing markers overlaid.

Usage:
    conda run -n foe python code/label_pipeline/generate_timing_review_viewer.py
"""

import sys, json, base64, io, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import detrend, butter, filtfilt
from pathlib import Path
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


def generate_eeg_jpeg_with_markers(seg_bi, fs, patient_id, hpp_result,
                                    title_extra=''):
    """Generate EEG JPEG with discharge timing markers overlaid.

    Uses the EXACT morgoth-viewer style rendering from generate_misclass_reviewer.py,
    then overlays red dots and vertical dashed lines at discharge times.
    Also adds a small evidence signal subplot below.
    """
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / fs, n_samples)

    # Lowpass at 20 Hz
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
            except ValueError:
                pass

    # Detrend
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    # Fixed scaling (matching morgoth-viewer)
    z_scale = 0.01     # 100 uV = 1 unit of vertical space
    clip_uv = 300.0    # clip at +/-300 uV

    # Build display list with blank spacer channels between groups
    display_channels = []
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))  # spacer
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    # Build mapping: channel_index -> y offset for marker placement
    ch_to_offset = {}
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        if ch_idx is not None:
            ch_to_offset[ch_idx] = offset

    # --- Figure with two subplots: EEG + evidence ---
    has_evidence = (hpp_result is not None and
                    'evidence_signal' in hpp_result and
                    hpp_result['evidence_signal'] is not None)

    if has_evidence:
        fig, (ax, ax_ev) = plt.subplots(
            2, 1, figsize=(14, 11.5),
            gridspec_kw={'height_ratios': [10, 1.2], 'hspace': 0.08})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        ax_ev = None

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw each channel as an offset trace
    yticks = []
    ytick_labels = []
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)

        if ch_idx is None:
            continue  # spacer

        clipped = np.clip(seg_bi[ch_idx, :], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.6, clip_on=True)

    # Y-axis: channel labels
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)

    # Fixed Y range
    ax.set_ylim(0, n_display + 1)

    # X-axis
    ax.set_xlim(0, n_samples / fs)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

    # 1-second vertical gridlines (dashed, like morgoth)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['bottom'].set_color('#999')

    # --- Overlay discharge markers ---
    if hpp_result is not None:
        global_times = hpp_result.get('global_times', [])
        channel_times = hpp_result.get('channel_times', {})

        for t in global_times:
            # Thin vertical dashed red line across all channels
            ax.axvline(x=t, color='red', linestyle='--', alpha=0.3,
                       linewidth=0.5, zorder=5)

        # Red dots on each involved channel's trace at the per-channel time
        if channel_times:
            for ch_str, ch_time_list in channel_times.items():
                ch_idx = int(ch_str)
                if ch_idx not in ch_to_offset:
                    continue
                offset = ch_to_offset[ch_idx]
                for ct in ch_time_list:
                    # Find the sample index for this time
                    samp = int(round(ct * fs))
                    samp = max(0, min(samp, n_samples - 1))
                    # Get the trace value at this point
                    clipped_val = np.clip(seg_bi[ch_idx, samp], -clip_uv, clip_uv)
                    y_val = z_scale * clipped_val + offset
                    ax.plot(ct, y_val, 'ro', markersize=4, zorder=10,
                            markeredgewidth=0.3, markeredgecolor='darkred')
        else:
            # No per-channel times, put dots at global times on all channels
            for t in global_times:
                samp = int(round(t * fs))
                samp = max(0, min(samp, n_samples - 1))
                for ch_idx, offset in ch_to_offset.items():
                    clipped_val = np.clip(seg_bi[ch_idx, samp], -clip_uv, clip_uv)
                    y_val = z_scale * clipped_val + offset
                    ax.plot(t, y_val, 'ro', markersize=4, zorder=10,
                            markeredgewidth=0.3, markeredgecolor='darkred')

        # Active interval shading
        active = hpp_result.get('active_interval', None)
        if active and len(active) == 2:
            ax.axvspan(active[0], active[1], alpha=0.04, color='blue', zorder=0)

    # Title
    title = f'{patient_id}'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)

    # --- Evidence signal subplot ---
    if ax_ev is not None and has_evidence:
        ev_signal = hpp_result['evidence_signal']
        ev_time = np.linspace(0, len(ev_signal) / fs, len(ev_signal))
        ax_ev.set_facecolor('white')
        ax_ev.fill_between(ev_time, 0, ev_signal, color='steelblue', alpha=0.5)
        ax_ev.plot(ev_time, ev_signal, color='steelblue', linewidth=0.8)
        ax_ev.set_xlim(0, n_samples / fs)
        ax_ev.set_ylabel('E(t)', fontsize=8)
        ax_ev.set_xlabel('Time (seconds)', fontsize=8)
        ax_ev.tick_params(labelsize=7)
        ax_ev.xaxis.set_major_locator(MultipleLocator(1))
        ax_ev.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
        ax_ev.spines['top'].set_visible(False)
        ax_ev.spines['right'].set_visible(False)

        # Mark discharge times on evidence plot too
        for t in hpp_result.get('global_times', []):
            ax_ev.axvline(x=t, color='red', linestyle='--', alpha=0.4,
                          linewidth=0.5)

        fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.035)
    else:
        fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 70})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_html(cases_json, image_data):
    """Build the timing review HTML viewer."""

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HPP Discharge Timing Review</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }

  #header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
    border-bottom: 2px solid #444;
  }
  #header-left { display: flex; align-items: center; gap: 12px; }
  #header-right { display: flex; align-items: center; gap: 12px; font-size: 13px; }

  .key { background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }

  #progress-bar-wrap {
    width: 100%; height: 6px; background: #333; border-radius: 3px; overflow: hidden;
  }
  #progress-bar {
    height: 100%; background: #44cc88; transition: width 0.2s;
  }

  #info-panel {
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: flex-start;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }
  .info-col { display: flex; flex-direction: column; gap: 3px; }
  .info-item { color: #bbb; }
  .info-item strong { color: #eee; }
  .info-badge {
    padding: 4px 12px; border-radius: 5px; font-size: 13px; font-weight: bold;
    display: inline-block;
  }
  .badge-lpd { background: #5a2020; color: #ff8888; }
  .badge-gpd { background: #20205a; color: #8888ff; }
  .badge-few { background: #5a4a20; color: #ffcc44; }

  #annotation-panel {
    background: #2a2a2a; padding: 12px 16px;
    display: flex; align-items: center; justify-content: center;
    gap: 16px; flex-wrap: wrap; border-bottom: 2px solid #444;
  }

  .anno-btn {
    padding: 14px 40px; border: 3px solid #555; border-radius: 10px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 18px; font-weight: bold;
    min-width: 140px; text-align: center; transition: all 0.15s;
  }
  .anno-btn:hover { filter: brightness(1.2); }
  .anno-btn.selected { box-shadow: 0 0 15px; }

  .btn-correct { background: #1a3a1a; border-color: #44cc88; color: #44cc88; }
  .btn-correct.selected { background: #2a5a2a; border-color: #66ff88; box-shadow: 0 0 15px #44cc88; }
  .btn-incorrect { background: #5a2020; border-color: #cc3333; color: #ff8888; }
  .btn-incorrect.selected { background: #8a2020; border-color: #ff4444; box-shadow: 0 0 15px #ff4444; }
  .btn-skip { background: #3a3a3a; border-color: #888; color: #ccc; }
  .btn-skip.selected { background: #555; border-color: #aaa; box-shadow: 0 0 15px #888; }

  #img-container { text-align: center; padding: 8px; }
  #img-container img { max-width: 100%; max-height: calc(100vh - 320px); }

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

  .discharge-times {
    font-size: 11px; color: #999; max-width: 400px; word-wrap: break-word;
  }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">HPP Discharge Timing Review</span>
    <select id="filter-mode" onchange="filterChanged()">
      <option value="all">All cases</option>
      <option value="unmarked">Unmarked</option>
      <option value="correct">Correct</option>
      <option value="incorrect">Incorrect</option>
    </select>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
    <span id="progress-text" style="font-size:12px; color:#aaa;"></span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportCSV()">Export CSV <span class="key">E</span></button>
    <span id="save-status"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

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
    <span class="info-item">HPP freq (IPI): <strong id="info-hpp-freq">--</strong></span>
    <span class="info-item">N discharges: <strong id="info-n-discharges">--</strong></span>
    <span class="info-item">IPI CV: <strong id="info-ipi-cv">--</strong></span>
  </div>
  <div class="info-col">
    <span class="info-item">Active interval: <strong id="info-active">--</strong></span>
    <span class="info-item discharge-times">Discharge times: <strong id="info-times">--</strong></span>
  </div>
</div>

<div id="annotation-panel">
  <button class="anno-btn btn-correct" onclick="annotate('correct')">CORRECT<br><span class="key">C</span></button>
  <button class="anno-btn btn-incorrect" onclick="annotate('incorrect')">INCORRECT<br><span class="key">I</span></button>
  <button class="anno-btn btn-skip" onclick="annotate('skip')">SKIP<br><span class="key">S</span></button>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">C</span> Correct &nbsp;&nbsp;
  <span class="key">I</span> Incorrect &nbsp;&nbsp;
  <span class="key">S</span> Skip &nbsp;&nbsp;
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">E</span> Export CSV
</div>

<script>
const CASES = CASES_PLACEHOLDER;
const IMAGE_DATA = IMAGE_PLACEHOLDER;

let annotations = {};
let filteredItems = [];
let idx = 0;

// Load saved annotations
try {
  annotations = JSON.parse(localStorage.getItem('timing_review_annotations') || '{}');
} catch(e) { annotations = {}; }

function saveAnnotations() {
  localStorage.setItem('timing_review_annotations', JSON.stringify(annotations));
}

function updateProgress() {
  const total = CASES.length;
  let nReviewed = 0, nCorrect = 0, nIncorrect = 0;
  for (const c of CASES) {
    const a = annotations[c.patient_id];
    if (a === 'correct') { nReviewed++; nCorrect++; }
    else if (a === 'incorrect') { nReviewed++; nIncorrect++; }
    else if (a === 'skip') { nReviewed++; }
  }
  const pct = total > 0 ? (nReviewed / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-text').textContent =
    nReviewed + ' of ' + total + ' reviewed (' + nCorrect + ' correct, ' + nIncorrect + ' incorrect)';
}

function filterChanged() {
  const mode = document.getElementById('filter-mode').value;
  filteredItems = CASES.filter(item => {
    const a = annotations[item.patient_id];
    if (mode === 'unmarked') return !a;
    if (mode === 'correct') return a === 'correct';
    if (mode === 'incorrect') return a === 'incorrect';
    return true;
  });
  idx = 0;
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
  } else {
    document.getElementById('viewer').src = '';
  }

  // Counter
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Info panel
  document.getElementById('info-pid').textContent = item.patient_id;
  document.getElementById('info-subtype').textContent = item.subtype.toUpperCase();
  document.getElementById('info-gold-freq').textContent = item.gold_standard_freq.toFixed(2) + ' Hz';

  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype.toLowerCase();
  if (item.n_discharges <= 1) {
    badge.textContent += ' (few)';
    badge.className = 'info-badge badge-few';
  }

  const hppFreq = item.hpp_freq;
  document.getElementById('info-hpp-freq').textContent =
    (hppFreq !== null && isFinite(hppFreq)) ? hppFreq.toFixed(3) + ' Hz' : 'N/A';
  document.getElementById('info-n-discharges').textContent = item.n_discharges;
  document.getElementById('info-ipi-cv').textContent =
    (item.ipi_cv !== null && isFinite(item.ipi_cv)) ? item.ipi_cv.toFixed(3) : 'N/A';

  const active = item.active_interval;
  if (active && active.length === 2) {
    document.getElementById('info-active').textContent =
      active[0].toFixed(1) + 's - ' + active[1].toFixed(1) + 's';
  } else {
    document.getElementById('info-active').textContent = 'N/A';
  }

  const times = item.global_times || [];
  document.getElementById('info-times').textContent =
    times.length > 0 ? times.map(t => t.toFixed(2) + 's').join(', ') : 'None';

  updateButtons(item);
}

function updateButtons(item) {
  const btns = document.querySelectorAll('.anno-btn');
  btns.forEach(b => b.classList.remove('selected'));
  if (!item) return;
  const a = annotations[item.patient_id];
  if (a === 'correct') document.querySelector('.btn-correct').classList.add('selected');
  else if (a === 'incorrect') document.querySelector('.btn-incorrect').classList.add('selected');
  else if (a === 'skip') document.querySelector('.btn-skip').classList.add('selected');
}

function annotate(value) {
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  annotations[item.patient_id] = value;
  saveAnnotations();

  document.getElementById('save-status').textContent = 'Saved: ' + value.toUpperCase();
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 800);

  updateButtons(item);
  updateProgress();

  // Auto-advance after brief delay
  if (idx < filteredItems.length - 1) {
    setTimeout(() => { idx++; show(); }, 250);
  }
}

function exportCSV() {
  const rows = ['patient_id,status,n_discharges,ipi_freq'];
  for (const c of CASES) {
    const a = annotations[c.patient_id] || '';
    if (!a) continue;
    const freq = (c.hpp_freq !== null && isFinite(c.hpp_freq)) ? c.hpp_freq.toFixed(3) : '';
    rows.push([c.patient_id, a, c.n_discharges, freq].join(','));
  }
  const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'timing_review_annotations.csv';
  a.click();
}

document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'SELECT') return;
  if (e.key === 'ArrowRight') { idx = Math.min(idx + 1, filteredItems.length - 1); show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx - 1, 0); show(); e.preventDefault(); }
  else if (e.key === 'c' || e.key === 'C') { annotate('correct'); e.preventDefault(); }
  else if (e.key === 'i' || e.key === 'I') { annotate('incorrect'); e.preventDefault(); }
  else if (e.key === 's' || e.key === 'S') { annotate('skip'); e.preventDefault(); }
  else if (e.key === 'e' || e.key === 'E') { exportCSV(); e.preventDefault(); }
});

// Init
filterChanged();
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 72)
    print("HPP Discharge Timing Review Viewer Generator")
    print("=" * 72)

    # ── Step 1: Load dataset ──────────────────────────────────────────
    print("\n--- Step 1: Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # ── Step 2: Load HPP results ──────────────────────────────────────
    print("\n--- Step 2: Loading HPP discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_results = json.load(f)
    print(f"  Loaded HPP results for {len(hpp_results)} patients")

    # ── Step 3: Re-run HPP to get evidence signals (not stored in JSON)
    print("\n--- Step 3: Re-computing evidence signals for visualization ---")
    from label_pipeline.hpp_discharge_marking import (
        _compute_channel_evidence, _aggregate_evidence, LOWPASS_HZ
    )

    evidence_cache = {}
    for idx_p, (_, row) in enumerate(df.iterrows()):
        pid = str(row['patient_id'])
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
        except Exception:
            pass

        if (idx_p + 1) % 100 == 0:
            print(f"  Computed evidence for {idx_p + 1}/{len(df)} patients")

    print(f"  Evidence signals computed: {len(evidence_cache)}")

    # ── Step 4: Build case list and sort ──────────────────────────────
    print("\n--- Step 4: Building case list ---")

    cases = []
    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        subtype = row['subtype']
        gold = float(row['gold_standard_freq'])

        hpp = hpp_results.get(pid, None)
        if hpp:
            case = {
                'patient_id': pid,
                'subtype': subtype,
                'gold_standard_freq': gold,
                'hpp_freq': hpp.get('frequency'),
                'n_discharges': hpp.get('n_discharges', 0),
                'ipi_cv': hpp.get('ipi_cv'),
                'active_interval': hpp.get('active_interval'),
                'global_times': hpp.get('global_times', []),
                'channel_times': hpp.get('channel_times', {}),
            }
        else:
            case = {
                'patient_id': pid,
                'subtype': subtype,
                'gold_standard_freq': gold,
                'hpp_freq': None,
                'n_discharges': 0,
                'ipi_cv': None,
                'active_interval': None,
                'global_times': [],
                'channel_times': {},
            }
        cases.append(case)

    # Sort: valid IPI frequency first (by gold freq low->high),
    # then cases with too few discharges at the end
    def sort_key(c):
        has_freq = (c['hpp_freq'] is not None and
                    c['n_discharges'] >= 2 and
                    isinstance(c['hpp_freq'], (int, float)) and
                    math.isfinite(c['hpp_freq']))
        if has_freq:
            return (0, c['gold_standard_freq'])
        else:
            return (1, c['gold_standard_freq'])

    cases.sort(key=sort_key)
    print(f"  Total cases: {len(cases)}")
    n_with_freq = sum(1 for c in cases if c['n_discharges'] >= 2)
    print(f"  With valid IPI frequency: {n_with_freq}")
    print(f"  With 0-1 discharges: {len(cases) - n_with_freq}")

    # ── Step 5: Generate EEG images with markers ──────────────────────
    print("\n--- Step 5: Generating EEG images with discharge markers ---")

    image_data = {}
    n_generated = 0

    for ci, case in enumerate(cases):
        pid = case['patient_id']
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]

        # Build hpp_result dict with evidence signal for visualization
        hpp_viz = {
            'global_times': case['global_times'],
            'channel_times': case['channel_times'],
            'active_interval': case['active_interval'],
        }
        if pid in evidence_cache:
            hpp_viz['evidence_signal'] = evidence_cache[pid]
        else:
            hpp_viz['evidence_signal'] = None

        n_dis = case['n_discharges']
        hpp_freq = case['hpp_freq']
        gold = case['gold_standard_freq']

        if hpp_freq is not None and isinstance(hpp_freq, (int, float)) and math.isfinite(hpp_freq):
            title = (f"{case['subtype'].upper()}  |  "
                     f"gold={gold:.2f} Hz  |  "
                     f"HPP={hpp_freq:.2f} Hz  |  "
                     f"n={n_dis}")
        else:
            title = (f"{case['subtype'].upper()}  |  "
                     f"gold={gold:.2f} Hz  |  "
                     f"HPP=N/A  |  n={n_dis}")

        try:
            jpeg_bytes = generate_eeg_jpeg_with_markers(
                seg, FS, pid, hpp_viz, title_extra=title)
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
            n_generated += 1
            if n_generated % 50 == 0:
                print(f"  Generated {n_generated} images...")
        except Exception as e:
            print(f"  IMG FAILED: {pid}: {e}")

    print(f"  Total images generated: {n_generated}")

    # ── Step 6: Build JSON for viewer (strip channel_times to save space)
    print("\n--- Step 6: Building HTML viewer ---")

    cases_json = []
    for c in cases:
        cases_json.append({
            'patient_id': c['patient_id'],
            'subtype': c['subtype'],
            'gold_standard_freq': c['gold_standard_freq'],
            'hpp_freq': c['hpp_freq'],
            'n_discharges': c['n_discharges'],
            'ipi_cv': c['ipi_cv'],
            'active_interval': c['active_interval'],
            'global_times': c['global_times'],
        })

    html = build_html(cases_json, image_data)

    # Replace placeholders with actual JSON
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases_json, default=_json_default))
    html = html.replace('IMAGE_PLACEHOLDER', json.dumps(image_data))

    output_path = OUT_DIR / 'timing_review_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total cases: {len(cases)}")
    print(f"  Images generated: {n_generated}")
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
