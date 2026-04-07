#!/usr/bin/env python3
"""Generate LPD laterality review viewer.

Runs pd_detect_alternate on all LPD segments without ground-truth laterality,
generates EEG images, and builds an HTML viewer for MW to label laterality.

Controls:
  ← (left arrow)  = Left laterality
  → (right arrow)  = Right laterality
  Space            = Reject (not LPD)
  Backspace        = Go back

Cases are sorted by decreasing P(left) — i.e., laterality_index from most
negative (strongest left) to most positive (strongest right).

Usage:
    conda run -n morgoth python code/lateralization_contest/generate_lpd_laterality_reviewer.py
"""
import sys
import json
import time
import base64
import io
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt, detrend
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

import pd_detect_alternate as pddeta

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
OUT_DIR = PROJECT_DIR / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
FS = 200


def mono_to_bipolar(data_mono):
    """Convert 19-channel monopolar to 18-channel bipolar."""
    bipolar_ids = np.array([
        [MONO_CHANNELS.index(bc.split('-')[0]), MONO_CHANNELS.index(bc.split('-')[1])]
        for bc in BIPOLAR_CHANNELS
    ])
    return data_mono[bipolar_ids[:, 0]] - data_mono[bipolar_ids[:, 1]]


def load_eeg(mat_file):
    """Load EEG from .mat file. Returns (data, fs) with data shape (n_ch, n_samples)."""
    path = str(EEG_DIR / mat_file)
    try:
        mat = sio.loadmat(path)
    except Exception:
        import h5py
        with h5py.File(path, 'r') as f:
            mat = {k: f[k][()] for k in f.keys() if not k.startswith('#')}

    for key in ['data', 'data_50sec']:
        if key in mat:
            data = np.array(mat[key], dtype=np.float64)
            break
    else:
        raise KeyError(f"No data key found in {mat_file}")

    if data.shape[0] > data.shape[1]:
        data = data.T

    fs = FS
    if 'Fs' in mat:
        fs = int(np.array(mat['Fs']).flat[0])

    return data, fs


def generate_eeg_jpeg(seg_bi, segment_id, title_extra=''):
    """Generate EEG image as JPEG bytes from bipolar data."""
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    b, a = butter(4, 20.0 / (FS / 2), btype='low')
    for i in range(n_channels):
        try:
            seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
        except ValueError:
            pass
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    z_scale = 0.01
    clip_uv = 300.0
    GROUP_BREAKS = {4, 8, 12, 16}
    display_channels = []
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    yticks, ytick_labels = [], []
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)
        if ch_idx is None:
            continue
        clipped = np.clip(seg_bi[ch_idx, :], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.6, clip_on=True)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(0, n_display + 1)
    ax.set_xlim(0, n_samples / FS)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(0.3)
        ax.spines[s].set_color('#999')

    title = f'{segment_id}'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 75})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def find_unlabeled_lpd():
    """Find LPD segments on disk without ground-truth laterality."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))

    ci_path = LABELS_DIR / 'channel_involvement.json'
    ci = {}
    if ci_path.exists():
        with open(str(ci_path)) as f:
            ci = json.load(f)

    on_disk = set(seg.segment_id)
    sl_active = sl[(sl.segment_id.isin(on_disk)) &
                   (sl.excluded.fillna(0).astype(bool) == False)]
    lpd = sl_active[sl_active.subtype == 'lpd'].copy()

    # Patient IDs with existing laterality
    has_lat_sl = set(lpd[lpd.laterality.notna()].patient_id)
    has_lat_ci = {pid for pid in set(lpd.patient_id)
                  if pid in ci and ci[pid].get('laterality') in ('left', 'right', 'bilateral')}
    has_lat = has_lat_sl | has_lat_ci

    unlabeled = lpd[~lpd.patient_id.isin(has_lat)]
    # One segment per patient
    unlabeled = unlabeled.drop_duplicates(subset='patient_id')

    return unlabeled


def run_laterality_predictions(unlabeled_df):
    """Run pd_detect_alternate on each segment and return laterality predictions."""
    cases = []
    failed = 0

    for _, row in tqdm(unlabeled_df.iterrows(), total=len(unlabeled_df),
                       desc="Computing laterality"):
        mat_file = row['mat_file']
        try:
            data, fs = load_eeg(mat_file)
            result = pddeta.pd_detect_alternate(data, fs, pk_detect='apd')
            cases.append({
                'segment_id': row['segment_id'],
                'patient_id': row['patient_id'],
                'mat_file': mat_file,
                'laterality_index': float(result['laterality_index']),
                'left_mean_score': float(result['left_mean_score']),
                'right_mean_score': float(result['right_mean_score']),
                'type_event': str(result.get('type_event', '')),
            })
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Failed {mat_file}: {e}")

    if failed:
        print(f"  Total failures: {failed}/{len(unlabeled_df)}")

    # Sort by laterality_index ascending (most left-lateralized first)
    cases.sort(key=lambda c: c['laterality_index'])
    return cases


def generate_images(cases):
    """Generate EEG images for all cases."""
    images = {}
    for i, case in enumerate(tqdm(cases, desc="Generating images")):
        try:
            data, fs = load_eeg(case['mat_file'])
            seg_bi = mono_to_bipolar(data)

            lat_idx = case['laterality_index']
            predicted = 'LEFT' if lat_idx < 0 else 'RIGHT'
            confidence = abs(lat_idx)
            title = f"Predicted: {predicted} (index={lat_idx:+.3f})"

            jpeg_bytes = generate_eeg_jpeg(seg_bi, case['patient_id'], title)
            images[case['segment_id']] = base64.b64encode(jpeg_bytes).decode('ascii')
        except Exception as e:
            print(f"  Image failed {case['mat_file']}: {e}")

    return images


def build_html(cases, images):
    n_total = len(cases)

    cases_json = json.dumps(cases)
    images_json = json.dumps(images)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>LPD Laterality Review — {n_total} cases</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #111; color: #eee; font-family: 'Consolas', monospace; }}
#header {{
    position: fixed; top: 0; left: 0; right: 0; z-index: 100;
    background: #1a1a1a; border-bottom: 2px solid #333; padding: 10px 20px;
    display: flex; align-items: center; gap: 20px;
}}
#header h1 {{ font-size: 16px; color: #44cc88; white-space: nowrap; }}
.info {{ font-size: 13px; color: #aaa; }}
#progress-bar {{ flex: 1; height: 8px; background: #333; border-radius: 4px; min-width: 100px; }}
#progress-fill {{ height: 100%; background: #44cc88; border-radius: 4px; transition: width 0.3s; }}
#counter {{ font-size: 14px; color: #44cc88; font-weight: bold; white-space: nowrap; }}
#controls {{
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
    background: #1a1a1a; border-top: 2px solid #333; padding: 12px 20px;
    display: flex; align-items: center; justify-content: center; gap: 30px;
}}
.key {{ display: inline-block; padding: 4px 12px; background: #333; border-radius: 4px;
        border: 1px solid #555; font-weight: bold; margin-right: 5px; }}
.key-left {{ background: #2a2a4a; border-color: #4488ff; color: #4488ff; }}
.key-right {{ background: #4a2a2a; border-color: #ff6644; color: #ff6644; }}
.key-space {{ background: #3a3a2a; border-color: #cccc44; color: #cccc44; }}
#viewer-area {{
    margin-top: 50px; margin-bottom: 60px; display: flex;
    justify-content: center; padding: 10px;
}}
#eeg-img {{ max-width: 100%; max-height: calc(100vh - 130px); }}
#prediction-badge {{
    position: fixed; top: 55px; right: 20px; font-size: 18px; padding: 8px 16px;
    border-radius: 8px; font-weight: bold; z-index: 101;
}}
.badge-left {{ background: #4488ff33; color: #4488ff; border: 2px solid #4488ff; }}
.badge-right {{ background: #ff664433; color: #ff6644; border: 2px solid #ff6644; }}
#status-badge {{
    position: fixed; top: 55px; left: 20px; font-size: 12px; padding: 4px 10px;
    border-radius: 4px; background: #222; color: #888; z-index: 101;
}}
#decision-indicator {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 60px; font-weight: bold; z-index: 200; opacity: 0;
    transition: opacity 0.15s; pointer-events: none;
}}
.flash {{ opacity: 1 !important; }}
#summary {{
    display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    background: #1a1a1a; border: 2px solid #44cc88; border-radius: 12px; padding: 30px;
    z-index: 300; text-align: center; min-width: 400px;
}}
#summary h2 {{ color: #44cc88; margin-bottom: 15px; }}
#summary .stat {{ font-size: 16px; margin: 8px 0; }}
#download-btn {{
    margin-top: 20px; padding: 10px 24px; background: #44cc88; color: #111;
    border: none; border-radius: 6px; font-size: 14px; font-weight: bold;
    cursor: pointer; font-family: inherit;
}}
#download-btn:hover {{ background: #55dd99; }}
</style>
</head>
<body>

<div id="header">
    <h1>LPD Laterality Review</h1>
    <div class="info">
        {n_total} cases — sorted by predicted P(left), most left-lateralized first
    </div>
    <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
    <div id="counter">0 / {n_total}</div>
</div>

<div id="status-badge"></div>
<div id="prediction-badge"></div>
<div id="decision-indicator"></div>

<div id="viewer-area">
    <img id="eeg-img" src="" alt="Loading...">
</div>

<div id="controls">
    <div>
        <span class="key key-left">&larr;</span> Left
    </div>
    <div>
        <span class="key key-right">&rarr;</span> Right
    </div>
    <div>
        <span class="key key-space">Space</span> Not LPD (reject)
    </div>
    <div style="color:#666">
        <span class="key">Backspace</span> Go back
    </div>
</div>

<div id="summary">
    <h2>Review Complete!</h2>
    <div class="stat" id="stat-total"></div>
    <div class="stat" id="stat-left"></div>
    <div class="stat" id="stat-right"></div>
    <div class="stat" id="stat-rejected"></div>
    <button id="download-btn" onclick="downloadResults()">Download Results JSON</button>
</div>

<script>
const cases = {cases_json};
const images = {images_json};
let currentIdx = 0;
let decisions = {{}};  // segment_id -> 'left' | 'right' | 'not_lpd'

const STORAGE_KEY = 'lpd_laterality_review_v1';

// Load saved progress
try {{
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && saved.decisions) {{
        decisions = saved.decisions;
        for (let i = 0; i < cases.length; i++) {{
            if (!(cases[i].segment_id in decisions)) {{
                currentIdx = i;
                break;
            }}
            if (i === cases.length - 1) currentIdx = cases.length;
        }}
    }}
}} catch(e) {{}}

function saveProgress() {{
    localStorage.setItem(STORAGE_KEY, JSON.stringify({{
        decisions: decisions,
        timestamp: new Date().toISOString()
    }}));
}}

function showCase(idx) {{
    if (idx >= cases.length) {{
        showSummary();
        return;
    }}
    const c = cases[idx];
    const img = document.getElementById('eeg-img');
    if (images[c.segment_id]) {{
        img.src = 'data:image/jpeg;base64,' + images[c.segment_id];
    }} else {{
        img.src = '';
        img.alt = 'No image for ' + c.segment_id;
    }}

    // Prediction badge
    const badge = document.getElementById('prediction-badge');
    const predicted = c.laterality_index < 0 ? 'LEFT' : 'RIGHT';
    const absIdx = Math.abs(c.laterality_index).toFixed(3);
    badge.textContent = 'Model: ' + predicted + ' (|idx|=' + absIdx + ')';
    badge.className = c.laterality_index < 0 ? 'badge-left' : 'badge-right';

    // Status
    const status = document.getElementById('status-badge');
    status.textContent = 'Case ' + (idx + 1) + ' / ' + cases.length +
        '  |  Patient: ' + c.patient_id;
    if (decisions[c.segment_id]) {{
        const d = decisions[c.segment_id];
        const label = d === 'not_lpd' ? 'NOT LPD' : d.toUpperCase();
        status.textContent += '  [' + label + ']';
    }}

    // Counter
    const reviewed = Object.keys(decisions).length;
    document.getElementById('counter').textContent = reviewed + ' / ' + cases.length;
    document.getElementById('progress-fill').style.width =
        (reviewed / cases.length * 100) + '%';
}}

function decide(action) {{
    if (currentIdx >= cases.length) return;
    const c = cases[currentIdx];
    decisions[c.segment_id] = action;
    saveProgress();

    // Flash indicator
    const ind = document.getElementById('decision-indicator');
    if (action === 'left') {{
        ind.textContent = 'LEFT';
        ind.style.color = '#4488ff';
    }} else if (action === 'right') {{
        ind.textContent = 'RIGHT';
        ind.style.color = '#ff6644';
    }} else {{
        ind.textContent = 'NOT LPD';
        ind.style.color = '#cccc44';
    }}
    ind.classList.add('flash');
    setTimeout(() => ind.classList.remove('flash'), 300);

    currentIdx++;
    setTimeout(() => showCase(currentIdx), 200);
}}

function goBack() {{
    if (currentIdx > 0) {{
        currentIdx--;
        const c = cases[currentIdx];
        delete decisions[c.segment_id];
        saveProgress();
        showCase(currentIdx);
    }}
}}

function showSummary() {{
    const vals = Object.values(decisions);
    const nLeft = vals.filter(d => d === 'left').length;
    const nRight = vals.filter(d => d === 'right').length;
    const nReject = vals.filter(d => d === 'not_lpd').length;
    document.getElementById('stat-total').textContent = 'Total reviewed: ' + vals.length;
    document.getElementById('stat-left').innerHTML =
        '<span style="color:#4488ff">Left: ' + nLeft + '</span>';
    document.getElementById('stat-right').innerHTML =
        '<span style="color:#ff6644">Right: ' + nRight + '</span>';
    document.getElementById('stat-rejected').innerHTML =
        '<span style="color:#cccc44">Not LPD: ' + nReject + '</span>';
    document.getElementById('summary').style.display = 'block';
    document.getElementById('counter').textContent = 'DONE';
    document.getElementById('progress-fill').style.width = '100%';
}}

function downloadResults() {{
    const results = {{
        timestamp: new Date().toISOString(),
        total_cases: cases.length,
        decisions: {{}},
        summary: {{ left: 0, right: 0, not_lpd: 0, unreviewed: 0 }},
    }};
    for (const c of cases) {{
        const d = decisions[c.segment_id] || 'unreviewed';
        results.decisions[c.segment_id] = {{
            patient_id: c.patient_id,
            mat_file: c.mat_file,
            laterality_index: c.laterality_index,
            decision: d,
        }};
        results.summary[d] = (results.summary[d] || 0) + 1;
    }}
    const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lpd_laterality_review_results.json';
    a.click();
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        decide('left');
    }} else if (e.key === 'ArrowRight') {{
        e.preventDefault();
        decide('right');
    }} else if (e.key === ' ') {{
        e.preventDefault();
        decide('not_lpd');
    }} else if (e.key === 'Backspace') {{
        e.preventDefault();
        goBack();
    }}
}});

// Start
showCase(currentIdx);
</script>
</body>
</html>"""


def main():
    t0 = time.time()
    print("=" * 70)
    print("  LPD Laterality Review Generator")
    print("=" * 70)

    # Find unlabeled LPDs
    print("Finding unlabeled LPD segments...")
    unlabeled = find_unlabeled_lpd()
    print(f"Found {len(unlabeled)} unlabeled LPD patients")

    # Run laterality predictions
    print("\nRunning laterality predictions...")
    cases = run_laterality_predictions(unlabeled)
    print(f"Got predictions for {len(cases)} segments")

    # Save predictions
    pred_path = OUT_DIR / 'lpd_laterality_predictions.csv'
    pd.DataFrame(cases).to_csv(str(pred_path), index=False)
    print(f"Predictions saved: {pred_path}")

    # Prediction summary
    n_left = sum(1 for c in cases if c['laterality_index'] < 0)
    n_right = len(cases) - n_left
    print(f"Predicted: {n_left} left, {n_right} right")

    # Generate images
    print(f"\nGenerating {len(cases)} EEG images...")
    images = generate_images(cases)
    print(f"Generated {len(images)} images")

    # Build HTML
    print("\nBuilding HTML viewer...")
    html = build_html(cases, images)

    out_path = OUT_DIR / 'lpd_laterality_reviewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")
    print(f"\nOpen in browser to start reviewing:")
    print(f"  open {out_path}")


if __name__ == '__main__':
    main()
