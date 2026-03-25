"""
Generate frequency annotation viewer for newly harvested LPD segments.

Reads harvest_manifest.json, generates morgoth-style EEG images,
and builds an HTML viewer for MW to annotate frequency.

Must run with: conda run -n foe python code/generate_freq_annotation_viewer.py
"""

import sys, json, base64, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import detrend, butter, filtfilt
from scipy.io import loadmat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
OUT_DIR = BASE / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


# ── EEG image generation (exact copy from generate_misclass_reviewer.py) ─────

def generate_eeg_jpeg(seg_bi, fs, patient_id, title_extra=''):
    """Generate a clean EEG JPEG image using morgoth-viewer style rendering.

    Follows the morgoth_viewer.m / viewer_widget.py approach:
    - Fixed µV scaling: z_scale = 0.01 (100 µV = 1 channel unit)
    - Clip at ±300 µV before scaling
    - Uniform channel spacing (offset = channel position index)
    - Black traces on white, with L/R hemisphere coloring
    - 1-second vertical gridlines
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
    z_scale = 0.01     # 100 µV = 1 unit of vertical space
    clip_uv = 300.0    # clip at ±300 µV

    # Build display list with blank spacer channels between groups
    # Groups: temporal L [0:4], temporal R [4:8], parasagittal L [8:12],
    #         parasagittal R [12:16], midline [16:18]
    GROUP_BREAKS = {4, 8, 12, 16}  # insert spacer before these indices
    display_channels = []  # list of (channel_index_or_None, channel_name)
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))  # spacer
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw each channel as an offset trace
    yticks = []
    ytick_labels = []
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        # Position: top channel at n_display, bottom at 1
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)

        if ch_idx is None:
            continue  # spacer — no trace drawn

        # Clip then scale (morgoth style)
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

    # Title
    title = f'{patient_id}'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 70})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── HTML viewer builder ──────────────────────────────────────────────

def build_viewer_html(cases_json, image_data):
    """Build the frequency annotation HTML viewer."""

    html = """<!DOCTYPE html>
<html>
<head>
<title>Frequency Annotation Viewer</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }

  #header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
  }
  #header-left { display: flex; align-items: center; gap: 12px; }
  #header-right { display: flex; align-items: center; gap: 12px; font-size: 13px; }

  .key { background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }

  #progress-bar-container {
    width: 100%; height: 6px; background: #333; border-bottom: 1px solid #444;
  }
  #progress-bar {
    height: 100%; background: #44cc88; transition: width 0.3s;
  }

  #info-panel {
    background: #2a2a2a; padding: 12px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }
  .info-badge {
    padding: 6px 16px; border-radius: 6px; font-size: 14px; font-weight: bold;
  }
  .badge-bin { background: #20205a; color: #8888ff; }
  .info-item { font-size: 14px; color: #bbb; }
  .info-item strong { color: #eee; }

  #est-freq-banner {
    padding: 10px 16px; font-size: 20px; font-weight: bold; text-align: center;
    background: #1a2a3a; color: #44aaff; border-bottom: 2px solid #444;
  }

  #annotation-panel {
    background: #2a2a2a; padding: 14px 16px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; flex-wrap: wrap; border-bottom: 2px solid #444;
  }

  .freq-input-wrap {
    display: flex; align-items: center; gap: 6px;
  }
  .freq-input {
    width: 120px; padding: 12px 10px; font-size: 22px; font-family: monospace;
    font-weight: bold; text-align: center; border: 3px solid #cc8833;
    border-radius: 10px; background: #3a2a1a; color: #ffaa44;
  }
  .freq-input:focus { outline: none; border-color: #ffaa44; box-shadow: 0 0 10px #cc8833; }
  .freq-label { font-size: 18px; color: #cc8833; font-weight: bold; }

  .anno-btn {
    padding: 14px 32px; border: 3px solid #555; border-radius: 10px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 18px; font-weight: bold;
    min-width: 120px; text-align: center; transition: all 0.15s;
  }
  .anno-btn:hover { filter: brightness(1.2); }
  .anno-btn.selected { box-shadow: 0 0 15px; }

  .btn-skip { background: #3a3a3a; border-color: #888; color: #ccc; }
  .btn-skip.selected { background: #555; border-color: #aaa; box-shadow: 0 0 15px #888; }
  .btn-submit { background: #1a3a1a; border-color: #44cc88; color: #44cc88; }
  .btn-submit:hover { background: #2a5a2a; }

  #img-container { text-align: center; padding: 8px; }
  #img-container img { max-width: 100%; max-height: calc(100vh - 320px); }

  #save-status { color: #44cc44; font-size: 13px; }

  .export-btn {
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }
  .export-btn:hover { background: #3a4a3a; }

  select { font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333;
  }

  .reviewed-indicator {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-left: 6px;
  }
  .reviewed-yes { background: #44cc88; }
  .reviewed-no { background: #555; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">Frequency Annotation Viewer</span>
    <select id="filter-mode" onchange="filterChanged()">
      <option value="all">All</option>
      <option value="unannotated">Unannotated</option>
      <option value="annotated">Annotated</option>
    </select>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
    <span id="progress-text" style="font-size:13px; color:#44cc88;">0 / 0 annotated</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
    <span id="save-status"></span>
  </div>
</div>

<div id="progress-bar-container">
  <div id="progress-bar" style="width: 0%"></div>
</div>

<div id="info-panel">
  <span class="info-badge badge-bin" id="info-bin">--</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Est. Freq: <strong id="est-freq-val">--</strong></span>
  <span id="reviewed-dot" class="reviewed-indicator reviewed-no"></span>
</div>

<div id="est-freq-banner">Estimated Frequency: -- Hz</div>

<div id="annotation-panel">
  <div class="freq-input-wrap">
    <span class="freq-label">Freq (Hz):</span>
    <input type="text" class="freq-input" id="freq-input" placeholder="Hz" />
  </div>
  <button class="anno-btn btn-submit" onclick="submitFreq()">Submit<br><span class="key">Enter</span></button>
  <button class="anno-btn btn-skip" onclick="doSkip()">SKIP<br><span class="key">S</span></button>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">Enter</span> Submit frequency &nbsp;&nbsp;
  <span class="key">S</span> Skip &nbsp;&nbsp;
  <span class="key">E</span> Export CSV
</div>

<script>
const CASES = CASES_PLACEHOLDER;
const IMAGE_DATA = IMAGE_PLACEHOLDER;

const STORAGE_KEY = 'harvest_freq_annotations';
let annotations = {};
let filteredItems = [];
let idx = 0;

// Load saved annotations
try {
  annotations = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
} catch(e) { annotations = {}; }

function saveAnnotations() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
}

function countAnnotated() {
  let n = 0;
  for (const c of CASES) {
    if (annotations[c.patient_id] != null) n++;
  }
  return n;
}

function updateProgress() {
  const nAnn = countAnnotated();
  const total = CASES.length;
  const pct = total > 0 ? (nAnn / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnn + ' / ' + total + ' annotated';
}

function filterChanged() {
  const mode = document.getElementById('filter-mode').value;
  filteredItems = CASES.filter(c => {
    if (mode === 'unannotated') return annotations[c.patient_id] == null;
    if (mode === 'annotated') return annotations[c.patient_id] != null;
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
    document.getElementById('patient-id').textContent = '--';
    document.getElementById('est-freq-banner').textContent = 'No cases to show';
    document.getElementById('freq-input').value = '';
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

  // Info
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;
  document.getElementById('est-freq-val').textContent = item.est_freq.toFixed(2) + ' Hz';
  document.getElementById('info-bin').textContent = item.bin || 'LPD';

  // Banner
  document.getElementById('est-freq-banner').textContent =
    'Estimated Frequency: ' + item.est_freq.toFixed(2) + ' Hz';

  // Annotation state
  const ann = annotations[item.patient_id];
  const dot = document.getElementById('reviewed-dot');
  if (ann != null) {
    dot.className = 'reviewed-indicator reviewed-yes';
    if (ann === 'skip') {
      document.getElementById('freq-input').value = '';
      document.getElementById('est-freq-banner').textContent += '  [SKIPPED]';
    } else {
      document.getElementById('freq-input').value = ann;
      document.getElementById('est-freq-banner').textContent += '  [Annotated: ' + ann + ' Hz]';
    }
  } else {
    dot.className = 'reviewed-indicator reviewed-no';
    // Pre-fill with estimated frequency as suggestion
    document.getElementById('freq-input').value = item.est_freq.toFixed(1);
  }

  // Focus input for quick typing
  document.getElementById('freq-input').focus();
  document.getElementById('freq-input').select();
}

function submitFreq() {
  if (filteredItems.length === 0) return;
  const val = document.getElementById('freq-input').value.trim();
  if (!val || isNaN(parseFloat(val))) {
    document.getElementById('save-status').textContent = 'Invalid frequency!';
    setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1500);
    return;
  }
  const item = filteredItems[idx];
  annotations[item.patient_id] = val;
  saveAnnotations();

  document.getElementById('save-status').textContent = 'Saved: ' + val + ' Hz';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1000);

  // Auto-advance
  if (idx < filteredItems.length - 1) {
    setTimeout(() => { idx++; show(); }, 250);
  } else {
    show();
  }
}

function doSkip() {
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  annotations[item.patient_id] = 'skip';
  saveAnnotations();

  document.getElementById('save-status').textContent = 'Skipped';
  setTimeout(() => { document.getElementById('save-status').textContent = ''; }, 1000);

  if (idx < filteredItems.length - 1) {
    setTimeout(() => { idx++; show(); }, 250);
  } else {
    show();
  }
}

function exportCSV() {
  const rows = ['patient_id,est_freq,annotated_freq'];
  for (const c of CASES) {
    const ann = annotations[c.patient_id];
    if (ann != null && ann !== 'skip') {
      rows.push(c.patient_id + ',' + c.est_freq + ',' + ann);
    }
  }
  const blob = new Blob([rows.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'harvest_freq_annotations.csv';
  a.click();
}

document.addEventListener('keydown', e => {
  // Allow typing in the input field normally, except for special keys
  const inInput = document.activeElement.id === 'freq-input';

  if (e.key === 'ArrowRight') { idx = Math.min(idx + 1, filteredItems.length - 1); show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx - 1, 0); show(); e.preventDefault(); }
  else if (e.key === 'Enter') { submitFreq(); e.preventDefault(); }
  else if (!inInput) {
    if (e.key === 's' || e.key === 'S') { doSkip(); e.preventDefault(); }
    else if (e.key === 'e' || e.key === 'E') { exportCSV(); e.preventDefault(); }
  }
});

// Init
filterChanged();
</script>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Frequency Annotation Viewer Generator")
    print("=" * 60)

    # Step 1: Load harvest manifest
    manifest_path = DATA / 'labels' / 'harvest_manifest.json'
    print(f"\n--- Step 1: Loading manifest from {manifest_path} ---")
    with open(manifest_path) as f:
        manifest = json.load(f)
    print(f"  {len(manifest)} patients in manifest")

    # Step 2: Sort by estimated frequency (low to high)
    sorted_pids = sorted(manifest.keys(), key=lambda p: manifest[p].get('est_freq', 0))

    # Step 3: Generate EEG images
    print("\n--- Step 2: Generating EEG images ---")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    image_data = {}
    n_generated = 0
    n_skipped = 0

    for pid in sorted_pids:
        info = manifest[pid]
        seg_path = EEG_DIR / f'{pid}_seg000.mat'
        if not seg_path.exists():
            n_skipped += 1
            continue

        try:
            mat = loadmat(str(seg_path))
            data = mat['data']
            fs = float(mat['Fs'].flatten()[0])

            # Convert to bipolar if needed
            if data.shape[0] > data.shape[1]:
                data = data.T
            n_ch = data.shape[0]

            if n_ch == 20:
                # Monopolar -> bipolar via fcn_getBanana
                seg_bi = fcn_getBanana(data)
            elif n_ch == 19:
                seg_bi = fcn_getBanana(data)
            elif n_ch == 18:
                seg_bi = data  # Already bipolar
            else:
                # Try anyway
                if n_ch > 18:
                    seg_bi = fcn_getBanana(data[:20] if n_ch >= 20 else data[:19])
                else:
                    seg_bi = data

            title_extra = f'est={info.get("est_freq", "?"):.1f} Hz'
            jpeg_bytes = generate_eeg_jpeg(seg_bi, fs, pid, title_extra=title_extra)
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')

            cases.append({
                'patient_id': pid,
                'est_freq': info.get('est_freq', 0),
                'bin': info.get('bin', ''),
            })

            n_generated += 1
            if n_generated % 25 == 0:
                print(f"  Generated {n_generated} images...")

        except Exception as e:
            print(f"  FAILED {pid}: {e}")
            n_skipped += 1

    print(f"  Generated: {n_generated}, Skipped: {n_skipped}")

    # Step 4: Build HTML
    print("\n--- Step 3: Building HTML viewer ---")
    html = build_viewer_html(cases, image_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases))
    html = html.replace('IMAGE_PLACEHOLDER', json.dumps(image_data))

    output_path = OUT_DIR / 'freq_annotation_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Cases: {n_generated}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")

    # Open in browser
    import subprocess
    subprocess.Popen(['open', str(output_path)])
    print(f"  Opened in browser.")


if __name__ == '__main__':
    main()
