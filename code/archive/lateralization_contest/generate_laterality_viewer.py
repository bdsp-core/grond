#!/usr/bin/env python3
"""Generate LRDA laterality labeling viewer.

Shows EEG with symmetric channel layout. Default side from V22 model.
Sorted by predicted laterality (most left-dominant first → most right-dominant last).

Keys:
  ← (left arrow) = Left dominant
  → (right arrow) = Right dominant
  Space = Not LRDA (reject)
  Backspace = Go back

Usage:
    conda run -n morgoth python code/lateralization_contest/generate_laterality_viewer.py
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
from scipy.signal import butter, sosfiltfilt, detrend, hilbert
from pathlib import Path

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
OUT_DIR = PROJECT_DIR / 'results'

FS = 200
LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])

DISPLAY_ORDER = [
    (0, 'Fp1-F7'), (1, 'F7-T3'), (2, 'T3-T5'), (3, 'T5-O1'),
    None,
    (8, 'Fp1-F3'), (9, 'F3-C3'), (10, 'C3-P3'), (11, 'P3-O1'),
    None,
    (16, 'Fz-Cz'), (17, 'Cz-Pz'),
    None,
    (12, 'Fp2-F4'), (13, 'F4-C4'), (14, 'C4-P4'), (15, 'P4-O2'),
    None,
    (4, 'Fp2-F8'), (5, 'F8-T4'), (6, 'T4-T6'), (7, 'T6-O2'),
]


def v22_laterality(seg_bi):
    """Compute V22 laterality index and scores."""
    sos = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg_bi, axis=1)
    ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))
    lat_idx = (rs - ls) / (ls + rs + 1e-12)
    return lat_idx, ls, rs


def load_and_preprocess(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    raw = mat[dk].astype(np.float64)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    if raw.shape[0] >= 19:
        seg_bi = np.array(fcn_getBanana(raw[:19, :2000]), dtype=np.float64)
    elif raw.shape[0] == 18:
        seg_bi = raw[:18, :2000]
    else:
        return None
    from mne.filter import notch_filter, filter_data
    seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
    seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')
    return seg_bi


def generate_eeg_jpeg(seg_bi, title=''):
    sos = butter(4, [0.5 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos, seg_bi, axis=1)
    for i in range(18):
        seg_f[i] = detrend(seg_f[i], type='linear')

    n_display = len(DISPLAY_ORDER)
    time_vec = np.linspace(0, 10, 2000)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    yticks, ytick_labels = [], []
    for di, item in enumerate(DISPLAY_ORDER):
        offset = float(n_display - di)
        if item is None:
            yticks.append(offset)
            ytick_labels.append('')
            continue
        ch_idx, ch_name = item
        yticks.append(offset)
        ytick_labels.append(ch_name)
        clipped = np.clip(seg_f[ch_idx], -300, 300)
        scaled = 0.01 * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.7)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(0, n_display + 1)
    ax.set_xlim(0, 10)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    ax.text(1.01, (n_display - 1.5) / n_display, 'L lateral',
            transform=ax.transAxes, fontsize=8, color='#666', va='center')
    ax.text(1.01, (n_display - 6.5) / n_display, 'L parasag',
            transform=ax.transAxes, fontsize=8, color='#666', va='center')
    ax.text(1.01, (n_display - 10.5) / n_display, 'midline',
            transform=ax.transAxes, fontsize=8, color='#666', va='center')
    ax.text(1.01, (n_display - 14.5) / n_display, 'R parasag',
            transform=ax.transAxes, fontsize=8, color='#666', va='center')
    ax.text(1.01, (n_display - 19.5) / n_display, 'R lateral',
            transform=ax.transAxes, fontsize=8, color='#666', va='center')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.065, right=0.94, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 75})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    t0 = time.time()
    print("=" * 70)
    print("  LRDA Laterality Labeling Viewer")
    print("=" * 70)

    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    # Select LRDA segments without left/right laterality yet
    lrda = sl[
        (sl['subtype'] == 'lrda') &
        (~sl['excluded'].fillna(False).astype(bool))
    ].copy()

    # Include those that already have laterality (for re-review) and those that don't
    print(f"Total LRDA segments: {len(lrda)}")
    print(f"  With laterality: {lrda.laterality.isin(['left','right','bilateral']).sum()}")
    print(f"  Without: {(~lrda.laterality.isin(['left','right','bilateral'])).sum()}")

    # Compute V22 laterality for each
    print("Computing V22 laterality estimates...")
    cases = []
    n_skip = 0
    for i, (_, row) in enumerate(lrda.iterrows()):
        mat_file = row['mat_file']
        seg_bi = load_and_preprocess(mat_file)
        if seg_bi is None:
            n_skip += 1
            continue

        lat_idx, ls, rs = v22_laterality(seg_bi)
        pred_side = 'left' if lat_idx < 0 else 'right'
        prob_left = max(0, min(1, 0.5 - lat_idx))  # higher = more left

        title = f"LRDA  |  V22: {pred_side} (idx={lat_idx:.3f})"
        if pd.notna(row.get('mw_freq')):
            title += f"  |  freq={row['mw_freq']:.2f} Hz"

        try:
            jpeg = generate_eeg_jpeg(seg_bi, title)
            b64 = base64.b64encode(jpeg).decode('ascii')
        except:
            n_skip += 1
            continue

        cases.append({
            'mat_file': mat_file,
            'patient_id': str(row['patient_id']),
            'lat_index': round(float(lat_idx), 4),
            'pred_side': pred_side,
            'prob_left': round(float(prob_left), 4),
            'existing_lat': row.get('laterality', ''),
            'image': b64,
        })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(lrda)} ({time.time() - t0:.0f}s)")

    print(f"  Total: {len(cases)} (skipped {n_skip})")

    # Sort by prob_left descending (most left-dominant first)
    cases.sort(key=lambda c: -c['prob_left'])

    # Build HTML
    print("Building HTML viewer...")
    BATCH_SIZE = 500
    n_batches = (len(cases) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(cases))
        batch = cases[start:end]

        cases_json = json.dumps([{k: v for k, v in c.items() if k != 'image'} for c in batch])
        images_json = json.dumps({c['mat_file']: c['image'] for c in batch})

        html = _build_html(batch, cases_json, images_json, batch_idx + 1, n_batches, len(cases))

        out_path = OUT_DIR / f'lrda_laterality_viewer_batch{batch_idx + 1}.html'
        with open(str(out_path), 'w') as f:
            f.write(html)
        print(f"  Saved: {out_path}")

    print(f"\nDone in {time.time() - t0:.0f}s")


def _build_html(batch, cases_json, images_json, batch_num, n_batches, total_cases):
    n = len(batch)
    return f"""<!DOCTYPE html>
<html><head><title>LRDA Laterality — Batch {batch_num}/{n_batches}</title>
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
#progress-bar {{ flex: 1; height: 8px; background: #333; border-radius: 4px; }}
#progress-fill {{ height: 100%; background: #44cc88; border-radius: 4px; transition: width 0.3s; }}
#counter {{ font-size: 14px; color: #44cc88; font-weight: bold; }}
#controls {{
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 100;
    background: #1a1a1a; border-top: 2px solid #333; padding: 12px 20px;
    display: flex; align-items: center; justify-content: center; gap: 40px;
}}
.key {{ display: inline-block; padding: 6px 16px; background: #333; border-radius: 6px;
        border: 2px solid #555; font-weight: bold; font-size: 16px; }}
.key-left {{ background: #1a2a4a; border-color: #4488ff; color: #4488ff; }}
.key-right {{ background: #4a1a1a; border-color: #ff4444; color: #ff4444; }}
.key-reject {{ background: #2a2a1a; border-color: #888; color: #aaa; }}
#viewer-area {{ margin-top: 50px; margin-bottom: 70px; display: flex; justify-content: center; padding: 10px; }}
#eeg-img {{ max-width: 100%; max-height: calc(100vh - 140px); }}
#flash {{ position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
          font-size: 72px; font-weight: bold; z-index: 200; opacity: 0;
          transition: opacity 0.15s; pointer-events: none; }}
.flash-show {{ opacity: 1 !important; }}
#summary {{ display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
            background: #1a1a1a; border: 2px solid #44cc88; border-radius: 12px; padding: 30px;
            z-index: 300; text-align: center; min-width: 400px; }}
#summary h2 {{ color: #44cc88; margin-bottom: 15px; }}
.stat {{ font-size: 16px; margin: 8px 0; }}
#download-btn {{ margin-top: 20px; padding: 10px 24px; background: #44cc88; color: #111;
                 border: none; border-radius: 6px; font-size: 14px; font-weight: bold;
                 cursor: pointer; font-family: inherit; }}
</style></head><body>
<div id="header">
    <h1>LRDA Laterality — Batch {batch_num}/{n_batches}</h1>
    <div class="info">{n} cases (sorted: most left-dominant → most right-dominant)</div>
    <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
    <div id="counter">0 / {n}</div>
</div>
<div id="flash"></div>
<div id="viewer-area"><img id="eeg-img" src="" alt="Loading..."></div>
<div id="controls">
    <div><span class="key key-left">←</span> LEFT</div>
    <div><span class="key key-right">→</span> RIGHT</div>
    <div><span class="key key-reject">Space</span> Not LRDA</div>
    <div style="color:#666"><span class="key" style="font-size:12px">⌫</span> Back</div>
</div>
<div id="summary">
    <h2>Complete!</h2>
    <div class="stat" id="stat-total"></div>
    <div class="stat" id="stat-left" style="color:#4488ff"></div>
    <div class="stat" id="stat-right" style="color:#ff4444"></div>
    <div class="stat" id="stat-reject" style="color:#888"></div>
    <button id="download-btn" onclick="downloadResults()">Download JSON</button>
</div>
<script>
const cases = {cases_json};
const images = {images_json};
let idx = 0;
let decisions = {{}};
const KEY = 'lrda_laterality_v1_batch{batch_num}';
try {{
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && s.decisions) {{
        decisions = s.decisions;
        for (let i = 0; i < cases.length; i++) {{
            if (!(cases[i].mat_file in decisions)) {{ idx = i; break; }}
            if (i === cases.length - 1) idx = cases.length;
        }}
    }}
}} catch(e) {{}}
function save() {{ localStorage.setItem(KEY, JSON.stringify({{decisions, ts: new Date().toISOString()}})); }}
// Preload buffer
const preloadImg = new Image();
function preload(i) {{
    if (i < cases.length && images[cases[i].mat_file]) {{
        preloadImg.src = 'data:image/jpeg;base64,' + images[cases[i].mat_file];
    }}
}}
function show(i) {{
    if (i >= cases.length) {{ showSummary(); return; }}
    const c = cases[i];
    const img = document.getElementById('eeg-img');
    if (images[c.mat_file]) {{
        img.src = 'data:image/jpeg;base64,' + images[c.mat_file];
    }}
    // Preload the next image
    preload(i + 1);
    const n = Object.keys(decisions).length;
    document.getElementById('counter').textContent = n + ' / ' + cases.length;
    document.getElementById('progress-fill').style.width = (n / cases.length * 100) + '%';
}}
function decide(label) {{
    if (idx >= cases.length) return;
    decisions[cases[idx].mat_file] = {{label, lat_index: cases[idx].lat_index, pred: cases[idx].pred_side}};
    save();
    const f = document.getElementById('flash');
    if (label === 'left') {{ f.textContent = '← LEFT'; f.style.color = '#4488ff'; }}
    else if (label === 'right') {{ f.textContent = 'RIGHT →'; f.style.color = '#ff4444'; }}
    else {{ f.textContent = '✗ NOT LRDA'; f.style.color = '#888'; }}
    f.classList.add('flash-show');
    setTimeout(() => f.classList.remove('flash-show'), 250);
    idx++;
    setTimeout(() => show(idx), 150);
}}
function goBack() {{
    if (idx > 0) {{ idx--; delete decisions[cases[idx].mat_file]; save(); show(idx); }}
}}
function showSummary() {{
    const v = Object.values(decisions);
    document.getElementById('stat-total').textContent = 'Total: ' + v.length;
    document.getElementById('stat-left').textContent = 'Left: ' + v.filter(d=>d.label==='left').length;
    document.getElementById('stat-right').textContent = 'Right: ' + v.filter(d=>d.label==='right').length;
    document.getElementById('stat-reject').textContent = 'Not LRDA: ' + v.filter(d=>d.label==='reject').length;
    document.getElementById('summary').style.display = 'block';
}}
function downloadResults() {{
    const r = {{timestamp: new Date().toISOString(), total: cases.length, decisions}};
    const b = new Blob([JSON.stringify(r, null, 2)], {{type:'application/json'}});
    const a = document.createElement('a'); a.href = URL.createObjectURL(b);
    a.download = 'lrda_laterality_batch{batch_num}.json'; a.click();
}}
document.addEventListener('keydown', function(e) {{
    if (e.key === 'ArrowLeft') {{ e.preventDefault(); decide('left'); }}
    else if (e.key === 'ArrowRight') {{ e.preventDefault(); decide('right'); }}
    else if (e.key === ' ') {{ e.preventDefault(); decide('reject'); }}
    else if (e.key === 'Backspace') {{ e.preventDefault(); goBack(); }}
}});
show(idx);
</script></body></html>"""


if __name__ == '__main__':
    main()
