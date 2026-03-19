"""
Generate laterality annotation viewer for LPD cases.

For each LPD patient:
1. Estimate frequency using pre-trained Ridge model
2. Auto-detect laterality using variance-explained per hemisphere
3. Generate clean EEG image
4. Build HTML viewer sorted by confidence (ambiguous first)

Must run with: conda run -n foe python code/generate_laterality_viewer.py
"""

import sys, os, json, base64, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.io
from scipy.signal import detrend, butter, filtfilt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
OUT_DIR = BASE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from discharge_timing import estimate_frequency, detect_involved_channels

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]


def get_bipolar(segment):
    """Convert 20-channel monopolar to 18-channel bipolar."""
    bipolar_ids = np.array([
        [MONO_CHANNELS.index(bc.split('-')[0]), MONO_CHANNELS.index(bc.split('-')[1])]
        for bc in BIPOLAR_CHANNELS
    ])
    return segment[bipolar_ids[:, 0]] - segment[bipolar_ids[:, 1]]


def compute_laterality(segment, fs, freq_estimate):
    """Compute laterality using VE-based approach.

    Returns: (side, confidence, left_ve, right_ve)
        side: 'Left' or 'Right'
        confidence: |L-R| / (L+R), 0 to 1
        left_ve, right_ve: summed VE for each hemisphere
    """
    _, ve_per_channel = detect_involved_channels(
        segment, fs, freq_estimate, subtype='lpd')

    left_ve = sum(ve_per_channel[ch] for ch in LEFT_INDICES if ch < len(ve_per_channel))
    right_ve = sum(ve_per_channel[ch] for ch in RIGHT_INDICES if ch < len(ve_per_channel))

    total_ve = left_ve + right_ve
    if total_ve < 1e-10:
        return 'Left', 0.0, 0.0, 0.0

    confidence = abs(left_ve - right_ve) / total_ve
    side = 'Left' if left_ve > right_ve else 'Right'
    return side, confidence, float(left_ve), float(right_ve)


def generate_clean_eeg_jpeg(seg_bi, fs, patient_id, side_suggestion, confidence):
    """Generate a clean EEG JPEG image with 20Hz lowpass + detrend."""
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T

    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / fs, n_samples)

    # Apply 20 Hz lowpass filter
    nyq = fs / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
            except ValueError:
                pass

    # Linear detrend each channel
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    # Create clean figure
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor('white')

    # Compute uniform y-scale
    all_ranges = []
    for i in range(n_channels):
        ch_range = np.ptp(seg_bi[i, :])
        if ch_range > 0:
            all_ranges.append(ch_range)
    y_half = max(np.median(all_ranges) * 0.75, 10) if all_ranges else 50

    for i in range(n_channels):
        ax = axes[i]
        ch_mean = np.mean(seg_bi[i, :])

        if i in LEFT_INDICES:
            color = '#cc3333'
            bg = '#fff5f5'
        elif i in RIGHT_INDICES:
            color = '#3333cc'
            bg = '#f5f5ff'
        else:
            color = '#333333'
            bg = '#f5f5f5'

        ax.set_facecolor(bg)
        ax.plot(time_vec, seg_bi[i, :], color=color, linewidth=0.8)
        ax.set_ylabel(BIPOLAR_CHANNELS[i], fontsize=7, rotation=0,
                      labelpad=50, va='center')
        ax.set_ylim(ch_mean - y_half, ch_mean + y_half)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3, axis='x')
        for spine in ax.spines.values():
            spine.set_visible(False)
        if i < n_channels - 1:
            ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xlabel('Time (seconds)', fontsize=9)
    axes[-1].tick_params(axis='x', labelsize=7)
    fig.suptitle(f'LPD — {patient_id}', fontsize=12, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.04, hspace=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 70})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_viewer(cases, image_data):
    """Build the laterality annotation viewer HTML."""
    # Sort by confidence ascending (ambiguous first)
    cases_sorted = sorted(cases, key=lambda c: c['confidence'])

    manifest_json = []
    for c in cases_sorted:
        manifest_json.append({
            'patient_id': c['patient_id'],
            'freq_estimate': round(c['freq_estimate'], 2),
            'suggestion': c['suggestion'],
            'confidence': round(c['confidence'], 3),
            'left_ve': round(c['left_ve'], 4),
            'right_ve': round(c['right_ve'], 4),
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>LPD Laterality Annotation</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}

  #progress-bar-container {{
    width: 300px; height: 10px; background: #444; border-radius: 5px; overflow: hidden;
  }}
  #progress-bar {{ height: 100%; background: #44cc44; transition: width 0.3s; }}
  #progress-text {{ font-size: 13px; color: #aaa; }}

  #info-panel {{
    background: #2a2a2a; padding: 12px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }}
  .info-badge {{
    padding: 6px 16px; border-radius: 6px; font-size: 14px; font-weight: bold;
    background: #5a2020; color: #ff8888;
  }}
  .info-item {{ font-size: 14px; color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #suggestion-banner {{
    padding: 10px 16px; font-size: 18px; font-weight: bold; text-align: center;
    border-bottom: 2px solid #444;
  }}
  .suggest-left {{ background: #4a1515; color: #ff6666; }}
  .suggest-right {{ background: #15154a; color: #6666ff; }}

  #annotation-panel {{
    background: #2a2a2a; padding: 14px 16px;
    display: flex; align-items: center; justify-content: center;
    gap: 12px; flex-wrap: wrap;
    border-bottom: 2px solid #444;
  }}

  .lat-btn {{
    padding: 16px 40px; border: 3px solid #555; border-radius: 10px;
    background: #444; color: #eee; cursor: pointer;
    font-family: monospace; font-size: 20px; font-weight: bold;
    min-width: 140px; text-align: center; transition: all 0.15s;
  }}
  .lat-btn:hover {{ filter: brightness(1.2); }}

  .btn-left {{ background: #5a2020; border-color: #cc3333; color: #ff8888; }}
  .btn-left.selected {{ background: #8a2020; border-color: #ff4444; box-shadow: 0 0 15px #ff4444; }}
  .btn-left.suggested {{ border-width: 4px; border-color: #ff6666; animation: pulse-left 1.5s infinite; }}

  .btn-right {{ background: #20205a; border-color: #3333cc; color: #8888ff; }}
  .btn-right.selected {{ background: #20208a; border-color: #4444ff; box-shadow: 0 0 15px #4444ff; }}
  .btn-right.suggested {{ border-width: 4px; border-color: #6666ff; animation: pulse-right 1.5s infinite; }}

  .btn-bilateral {{ background: #3a3a20; border-color: #aaaa33; color: #dddd66; }}
  .btn-bilateral.selected {{ background: #5a5a20; border-color: #dddd44; box-shadow: 0 0 15px #dddd44; }}

  .btn-skip {{ background: #3a3a3a; border-color: #888; color: #aaa; }}
  .btn-skip.selected {{ background: #555; border-color: #aaa; box-shadow: 0 0 10px #888; }}

  @keyframes pulse-left {{
    0%, 100% {{ box-shadow: 0 0 5px #ff4444; }}
    50% {{ box-shadow: 0 0 20px #ff4444; }}
  }}
  @keyframes pulse-right {{
    0%, 100% {{ box-shadow: 0 0 5px #4444ff; }}
    50% {{ box-shadow: 0 0 20px #4444ff; }}
  }}

  #img-container {{ text-align: center; padding: 8px; }}
  #img-container img {{ max-width: 100%; max-height: calc(100vh - 340px); }}

  #save-status {{ color: #44cc44; font-size: 13px; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333;
  }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}

  select {{ font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff8888;">LPD Laterality</span>
    <select id="filter-status" onchange="filterChanged()">
      <option value="all">All cases</option>
      <option value="unannotated">Unannotated</option>
      <option value="annotated">Annotated</option>
    </select>
    <div id="progress-bar-container"><div id="progress-bar"></div></div>
    <span id="progress-text">0/0</span>
  </div>
  <div id="header-right">
    <span id="counter">1 / 0</span>
    <button class="export-btn" onclick="exportCSV()">Export CSV</button>
    <span id="save-status"></span>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge">LPD</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Frequency: <strong id="freq-est">--</strong> Hz</span>
  <span class="info-item">Left VE: <strong id="left-ve">--</strong></span>
  <span class="info-item">Right VE: <strong id="right-ve">--</strong></span>
</div>

<div id="suggestion-banner" class="suggest-left">
  Suggested: -- (confidence: --)
</div>

<div id="annotation-panel">
  <button class="lat-btn btn-left" id="btn-left" onclick="annotate('left')">LEFT<br><span class="key">L</span></button>
  <button class="lat-btn btn-right" id="btn-right" onclick="annotate('right')">RIGHT<br><span class="key">R</span></button>
  <button class="lat-btn btn-bilateral" id="btn-bilateral" onclick="annotate('bilateral')">BILATERAL<br><span class="key">B</span></button>
  <button class="lat-btn btn-skip" id="btn-skip" onclick="annotate('skip')">SKIP<br><span class="key">S</span></button>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">L</span> Left &nbsp;&nbsp;
  <span class="key">R</span> Right &nbsp;&nbsp;
  <span class="key">B</span> Bilateral &nbsp;&nbsp;
  <span class="key">S</span> Skip &nbsp;&nbsp;
  <span class="key">Enter</span> Accept suggestion &amp; next &nbsp;&nbsp;
  <span class="key">E</span> Export CSV
</div>

<script>
const MANIFEST = {json.dumps(manifest_json)};
const IMAGE_DATA = {json.dumps(image_data)};

let annotations = {{}};
let filteredItems = [];
let idx = 0;

// Load saved annotations
try {{
  annotations = JSON.parse(localStorage.getItem('lpd_laterality_annotations') || '{{}}');
}} catch(e) {{ annotations = {{}}; }}

function saveAnnotations() {{
  localStorage.setItem('lpd_laterality_annotations', JSON.stringify(annotations));
}}

function init() {{
  filteredItems = MANIFEST.slice();
  idx = 0;
  show();
}}

function filterChanged() {{
  const statusFilter = document.getElementById('filter-status').value;
  filteredItems = MANIFEST.filter(m => {{
    if (statusFilter === 'unannotated' && annotations[m.patient_id]) return false;
    if (statusFilter === 'annotated' && !annotations[m.patient_id]) return false;
    return true;
  }});
  idx = 0;
  show();
}}

function show() {{
  if (filteredItems.length === 0) {{
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('patient-id').textContent = '--';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  // Image
  const b64 = IMAGE_DATA[item.patient_id];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
    document.getElementById('viewer').alt = 'Image not found: ' + item.patient_id;
  }}

  // Info
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('freq-est').textContent = item.freq_estimate;
  document.getElementById('left-ve').textContent = item.left_ve.toFixed(4);
  document.getElementById('right-ve').textContent = item.right_ve.toFixed(4);
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;

  // Suggestion banner
  const banner = document.getElementById('suggestion-banner');
  const confPct = (item.confidence * 100).toFixed(0);
  banner.textContent = 'Suggested: ' + item.suggestion.toUpperCase() + ' (confidence: ' + item.confidence.toFixed(3) + ' = ' + confPct + '%)';
  banner.className = 'suggest-' + item.suggestion.toLowerCase();

  // Buttons: highlight suggestion and current annotation
  ['left', 'right', 'bilateral', 'skip'].forEach(s => {{
    const btn = document.getElementById('btn-' + s);
    btn.classList.remove('selected', 'suggested');
  }});

  // Mark suggested button
  const sugBtn = document.getElementById('btn-' + item.suggestion.toLowerCase());
  if (sugBtn) sugBtn.classList.add('suggested');

  // Mark annotated button
  const currentAnno = annotations[item.patient_id];
  if (currentAnno) {{
    const annoBtn = document.getElementById('btn-' + currentAnno);
    if (annoBtn) annoBtn.classList.add('selected');
  }}

  updateProgress();
}}

function updateProgress() {{
  const total = MANIFEST.length;
  const nAnnotated = MANIFEST.filter(m => annotations[m.patient_id]).length;
  const pct = total > 0 ? (nAnnotated / total * 100) : 0;
  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('progress-text').textContent = nAnnotated + '/' + total + ' annotated';
}}

function annotate(value) {{
  if (filteredItems.length === 0) return;
  const item = filteredItems[idx];
  annotations[item.patient_id] = value;
  saveAnnotations();

  // Update button states
  ['left', 'right', 'bilateral', 'skip'].forEach(s => {{
    document.getElementById('btn-' + s).classList.remove('selected');
  }});
  const btn = document.getElementById('btn-' + value);
  if (btn) btn.classList.add('selected');

  document.getElementById('save-status').textContent = 'Saved: ' + value.toUpperCase();
  setTimeout(() => {{ document.getElementById('save-status').textContent = ''; }}, 1000);

  updateProgress();

  // Auto-advance
  if (idx < filteredItems.length - 1) {{
    setTimeout(() => {{ idx++; show(); }}, 300);
  }}
}}

function exportCSV() {{
  const headers = ['patient_id', 'laterality'];
  const rows = [headers.join(',')];
  for (const item of MANIFEST) {{
    const anno = annotations[item.patient_id] || '';
    rows.push([item.patient_id, anno].join(','));
  }}
  const blob = new Blob([rows.join('\\n')], {{ type: 'text/csv' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lpd_laterality_annotations.csv';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (document.activeElement.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, filteredItems.length - 1); show(); e.preventDefault(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); e.preventDefault(); }}
  else if (e.key === 'l' || e.key === 'L') {{ annotate('left'); e.preventDefault(); }}
  else if (e.key === 'r' || e.key === 'R') {{ annotate('right'); e.preventDefault(); }}
  else if (e.key === 'b' || e.key === 'B') {{ annotate('bilateral'); e.preventDefault(); }}
  else if (e.key === 's' || e.key === 'S') {{ annotate('skip'); e.preventDefault(); }}
  else if (e.key === 'Enter') {{
    // Accept suggestion
    if (filteredItems.length > 0) {{
      const item = filteredItems[idx];
      annotate(item.suggestion.toLowerCase());
    }}
    e.preventDefault();
  }}
  else if (e.key === 'e' || e.key === 'E') {{ exportCSV(); e.preventDefault(); }}
}});

init();
</script>
</body>
</html>"""

    output_path = OUT_DIR / 'laterality_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved laterality_viewer.html ({size_mb:.1f} MB)")
    return output_path


def main():
    print("=" * 60)
    print("LPD Laterality Annotation Viewer")
    print("=" * 60)

    # Step 1: Get all LPD patients
    print("\n--- Step 1: Loading LPD patients ---")
    patients = pd.read_csv(DATA / 'labels' / 'patients.csv')
    lpd = patients[patients.subtype == 'lpd'].copy()
    print(f"  Total patients: {len(patients)}")
    print(f"  LPD patients: {len(lpd)}")

    # Step 2: Auto-detect laterality + estimate frequency
    print("\n--- Step 2: Estimating frequency & laterality ---")
    cases = []
    failed = 0

    for i, (_, row) in enumerate(lpd.iterrows()):
        pid = str(row['patient_id'])
        mat_path = EEG_DIR / f"{pid}_seg000.mat"

        if not mat_path.exists():
            print(f"  NO DATA: {pid}")
            failed += 1
            continue

        try:
            mat = scipy.io.loadmat(str(mat_path))
            data = mat['data'].astype(np.float64)
            fs = int(mat['Fs'].ravel()[0])
            if data.shape[0] > data.shape[1]:
                data = data.T
            if data.shape[0] == 20:
                data = get_bipolar(data)
            elif data.shape[0] != 18:
                print(f"  SKIP: {pid} has {data.shape[0]} channels")
                failed += 1
                continue
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            # Estimate frequency
            freq_est = estimate_frequency(data, fs, subtype='lpd')

            # Compute laterality
            side, confidence, left_ve, right_ve = compute_laterality(data, fs, freq_est)

            cases.append({
                'patient_id': pid,
                'freq_estimate': freq_est,
                'suggestion': side,
                'confidence': confidence,
                'left_ve': left_ve,
                'right_ve': right_ve,
            })

            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(lpd)}...")

        except Exception as e:
            print(f"  FAILED: {pid}: {e}")
            failed += 1

    print(f"  Successfully processed: {len(cases)}")
    print(f"  Failed: {failed}")

    # Step 3: Generate images
    print(f"\n--- Step 3: Generating EEG images ---")
    image_data = {}
    img_failed = 0

    for i, case in enumerate(cases):
        pid = case['patient_id']
        mat_path = EEG_DIR / f"{pid}_seg000.mat"

        try:
            mat = scipy.io.loadmat(str(mat_path))
            data = mat['data'].astype(np.float64)
            fs = int(mat['Fs'].ravel()[0])
            if data.shape[0] > data.shape[1]:
                data = data.T
            if data.shape[0] == 20:
                data = get_bipolar(data)
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            jpeg_bytes = generate_clean_eeg_jpeg(
                data, fs, pid, case['suggestion'], case['confidence'])
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')

            if (i + 1) % 25 == 0:
                print(f"  Generated {i + 1}/{len(cases)} images...")

        except Exception as e:
            print(f"  IMG FAILED: {pid}: {e}")
            img_failed += 1

    print(f"  Images generated: {len(image_data)}, failed: {img_failed}")

    # Step 4: Build HTML viewer
    print(f"\n--- Step 4: Building HTML viewer ---")
    viewer_path = build_viewer(cases, image_data)

    # Step 5: Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    n_left = sum(1 for c in cases if c['suggestion'] == 'Left')
    n_right = sum(1 for c in cases if c['suggestion'] == 'Right')
    print(f"  Total LPD cases: {len(cases)}")
    print(f"  Auto-detected LEFT:  {n_left} ({100*n_left/len(cases):.1f}%)")
    print(f"  Auto-detected RIGHT: {n_right} ({100*n_right/len(cases):.1f}%)")

    confidences = [c['confidence'] for c in cases]
    print(f"\n  Confidence distribution:")
    print(f"    Min:    {min(confidences):.3f}")
    print(f"    25th:   {np.percentile(confidences, 25):.3f}")
    print(f"    Median: {np.median(confidences):.3f}")
    print(f"    75th:   {np.percentile(confidences, 75):.3f}")
    print(f"    Max:    {max(confidences):.3f}")

    # Confidence buckets
    low = sum(1 for c in confidences if c < 0.2)
    med = sum(1 for c in confidences if 0.2 <= c < 0.5)
    high = sum(1 for c in confidences if c >= 0.5)
    print(f"\n  Confidence buckets:")
    print(f"    Low  (<0.2):   {low} cases (ambiguous, review first)")
    print(f"    Med  (0.2-0.5): {med} cases")
    print(f"    High (>0.5):   {high} cases (confident, quick review)")

    print(f"\n  Viewer: {viewer_path}")
    print(f"  Sorted by confidence ascending (ambiguous first)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
