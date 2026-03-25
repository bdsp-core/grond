"""
Generate PD consistency viewer.

Computes morphological consistency scores for all patients and builds
an HTML viewer showing cases ranked by consistency score (high to low).

Must run with: conda run -n foe python code/generate_consistency_viewer.py
"""

import sys, json, base64, io
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

CODE_DIR = Path(__file__).resolve().parent.parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset
from pd_consistency_score import compute_pd_consistency

DATA = BASE / 'data'
EEG_DIR = DATA / 'eeg'
OUT_DIR = BASE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


# ── EEG image generation (copied from generate_misclass_reviewer.py) ──

def generate_eeg_jpeg(seg_bi, fs, patient_id, title_extra=''):
    """Generate a clean EEG JPEG image using morgoth-viewer style rendering.

    Follows the morgoth_viewer.m / viewer_widget.py approach:
    - Fixed uV scaling: z_scale = 0.01 (100 uV = 1 channel unit)
    - Clip at +/-300 uV before scaling
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
    z_scale = 0.01     # 100 uV = 1 unit of vertical space
    clip_uv = 300.0    # clip at +/-300 uV

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
            continue  # spacer -- no trace drawn

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

def build_consistency_viewer(cases, image_data):
    """Build the consistency viewer HTML.

    Args:
        cases: list of dicts with patient_id, subtype, gold_freq,
               consistency_score, median_xcorr, shape_cv, n_discharges
        image_data: dict mapping patient_id -> base64 JPEG string
    """
    html = """<!DOCTYPE html>
<html>
<head>
<title>PD Consistency Viewer</title>
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

  #info-panel {
    background: #2a2a2a; padding: 12px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }
  .info-badge {
    padding: 6px 16px; border-radius: 6px; font-size: 14px; font-weight: bold;
  }
  .badge-lpd { background: #5a2020; color: #ff8888; }
  .badge-gpd { background: #20205a; color: #8888ff; }
  .info-item { font-size: 14px; color: #bbb; }
  .info-item strong { color: #eee; }

  #score-banner {
    padding: 10px 16px; font-size: 16px; font-weight: bold; text-align: center;
    border-bottom: 2px solid #444;
  }
  .score-green { background: #153a15; color: #44cc88; }
  .score-yellow { background: #3a3a15; color: #cccc44; }
  .score-red { background: #3a1515; color: #ff6644; }

  #img-container { text-align: center; padding: 8px; }
  #img-container img { max-width: 100%; max-height: calc(100vh - 250px); }

  #shortcuts {
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333;
  }

  .jump-input {
    width: 60px; padding: 4px 6px; font-size: 13px; font-family: monospace;
    text-align: center; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #eee;
  }
  .jump-input:focus { outline: none; border-color: #888; }
  .jump-btn {
    padding: 4px 10px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #eee; cursor: pointer; font-family: monospace; font-size: 13px;
  }
  .jump-btn:hover { background: #444; }
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#ff9800;">PD Consistency Viewer</span>
    <span id="counter" style="font-size:13px; color:#aaa;">Case 1 of 0</span>
  </div>
  <div id="header-right">
    <span style="color:#777;">Jump to:</span>
    <input type="text" class="jump-input" id="jump-input" placeholder="#" />
    <button class="jump-btn" onclick="jumpToCase()">Go</button>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge" id="info-subtype-badge">--</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Gold freq: <strong id="gold-freq">--</strong></span>
  <span class="info-item">Score: <strong id="score-val">--</strong></span>
  <span class="info-item">Xcorr: <strong id="xcorr-val">--</strong></span>
  <span class="info-item">Shape CV: <strong id="shapecv-val">--</strong></span>
  <span class="info-item">Discharges: <strong id="n-discharges">--</strong></span>
</div>

<div id="score-banner" class="score-green">--</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">Home</span> first case &nbsp;&nbsp;
  <span class="key">End</span> last case &nbsp;&nbsp;
  <span class="key">G</span> jump to case number
</div>

<script>
const CASES = CASES_PLACEHOLDER;
const IMAGE_DATA = IMAGE_PLACEHOLDER;

let idx = 0;

function show() {
  if (CASES.length === 0) {
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = 'No cases';
    return;
  }
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Image
  const b64 = IMAGE_DATA[c.patient_id];
  if (b64) {
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  } else {
    document.getElementById('viewer').src = '';
  }

  // Counter
  document.getElementById('counter').textContent = 'Case ' + (idx + 1) + ' of ' + CASES.length;

  // Info
  document.getElementById('patient-id').textContent = c.patient_id;
  document.getElementById('gold-freq').textContent = c.gold_freq.toFixed(2) + ' Hz';
  document.getElementById('score-val').textContent = c.consistency_score.toFixed(4);
  document.getElementById('xcorr-val').textContent = c.median_xcorr.toFixed(4);
  document.getElementById('shapecv-val').textContent = c.shape_cv.toFixed(4);
  document.getElementById('n-discharges').textContent = c.n_discharges;

  // Subtype badge
  const badge = document.getElementById('info-subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype.toLowerCase();

  // Score banner with color coding
  const banner = document.getElementById('score-banner');
  const score = c.consistency_score;
  if (score > 0.5) {
    banner.className = 'score-banner score-green';
    banner.textContent = 'CONSISTENCY SCORE: ' + score.toFixed(4) + '  (HIGH)';
  } else if (score >= 0.3) {
    banner.className = 'score-banner score-yellow';
    banner.textContent = 'CONSISTENCY SCORE: ' + score.toFixed(4) + '  (MODERATE)';
  } else {
    banner.className = 'score-banner score-red';
    banner.textContent = 'CONSISTENCY SCORE: ' + score.toFixed(4) + '  (LOW)';
  }
}

function jumpToCase() {
  const val = document.getElementById('jump-input').value.trim();
  const num = parseInt(val, 10);
  if (!isNaN(num) && num >= 1 && num <= CASES.length) {
    idx = num - 1;
    show();
  }
  document.getElementById('jump-input').value = '';
  document.getElementById('jump-input').blur();
}

document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'INPUT') {
    if (e.key === 'Enter') { jumpToCase(); e.preventDefault(); }
    if (e.key === 'Escape') { document.activeElement.blur(); e.preventDefault(); }
    return;
  }
  if (e.key === 'ArrowRight') { idx = Math.min(idx + 1, CASES.length - 1); show(); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { idx = Math.max(idx - 1, 0); show(); e.preventDefault(); }
  else if (e.key === 'Home') { idx = 0; show(); e.preventDefault(); }
  else if (e.key === 'End') { idx = CASES.length - 1; show(); e.preventDefault(); }
  else if (e.key === 'g' || e.key === 'G') { document.getElementById('jump-input').focus(); e.preventDefault(); }
});

// Init
show();
</script>
</body>
</html>"""

    return html


def main():
    print("=" * 60)
    print("PD Consistency Viewer Generator")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n--- Step 1: Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # Step 2: Compute consistency scores for each patient (first segment)
    print("\n--- Step 2: Computing consistency scores ---")
    cases = []
    pid_to_gold = dict(zip(df['patient_id'], df['gold_standard_freq']))
    pid_to_subtype = dict(zip(df['patient_id'], df['subtype']))

    n_processed = 0
    for pid in df['patient_id'].values:
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]  # first segment
        gold_freq = pid_to_gold.get(pid, None)

        try:
            result = compute_pd_consistency(seg, fs=200, expected_freq=gold_freq)
        except Exception as e:
            print(f"  FAILED: {pid}: {e}")
            result = {
                'consistency_score': 0.0,
                'median_xcorr': 0.0,
                'shape_cv': 1.0,
                'n_discharges': 0,
                'peak_channel': 0,
            }

        cases.append({
            'patient_id': pid,
            'subtype': pid_to_subtype.get(pid, '?'),
            'gold_freq': float(gold_freq) if gold_freq else 0.0,
            'consistency_score': result['consistency_score'],
            'median_xcorr': result['median_xcorr'],
            'shape_cv': result['shape_cv'],
            'n_discharges': result['n_discharges'],
        })

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  Processed {n_processed} patients...")

    print(f"  Total: {n_processed} patients scored")

    # Sort by consistency_score DESCENDING
    cases.sort(key=lambda c: -c['consistency_score'])

    # Print distribution summary
    scores = [c['consistency_score'] for c in cases]
    n_high = sum(1 for s in scores if s > 0.5)
    n_med = sum(1 for s in scores if 0.3 <= s <= 0.5)
    n_low = sum(1 for s in scores if s < 0.3)
    print(f"\n  Score distribution:")
    print(f"    HIGH  (>0.5):     {n_high}")
    print(f"    MODERATE (0.3-0.5): {n_med}")
    print(f"    LOW   (<0.3):     {n_low}")
    print(f"    Median score:     {np.median(scores):.4f}")
    print(f"    Mean score:       {np.mean(scores):.4f}")

    # Step 3: Generate EEG images
    print("\n--- Step 3: Generating EEG images ---")
    image_data = {}
    n_img = 0
    for case in cases:
        pid = case['patient_id']
        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        try:
            jpeg_bytes = generate_eeg_jpeg(seg, 200, pid)
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
            n_img += 1
            if n_img % 50 == 0:
                print(f"  Generated {n_img} images...")
        except Exception as e:
            print(f"  IMG FAILED: {pid}: {e}")

    print(f"  Total images: {n_img}")

    # Step 4: Build HTML
    print("\n--- Step 4: Building HTML viewer ---")
    html = build_consistency_viewer(cases, image_data)
    html = html.replace('CASES_PLACEHOLDER', json.dumps(cases))
    html = html.replace('IMAGE_PLACEHOLDER', json.dumps(image_data))

    output_path = OUT_DIR / 'consistency_viewer.html'
    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Patients scored: {n_processed}")
    print(f"  Images generated: {n_img}")
    print(f"  Viewer: {output_path} ({size_mb:.1f} MB)")
    print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
