#!/usr/bin/env python3
"""Generate LRDA side-labeling viewer.

Selects LRDA cases that are confirmed by our best model, plus additional IIIC
cases with strong LRDA vote majority. Generates an HTML viewer for MW to
label each case as Left or Right dominant.

Usage:
    conda run -n morgoth python code/lateralization_contest/generate_lrda_side_viewer.py
"""
import sys
import json
import time
import pickle
import base64
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from pathlib import Path

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v3'
V2_CACHE = PROJECT_DIR / 'results' / 'lateralization_contest_v2' / '_cache'
OUT_DIR = PROJECT_DIR / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
LEFT_CHS = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CHS = [4, 5, 6, 7, 12, 13, 14, 15]
FS = 200


def generate_eeg_jpeg(seg_bi, patient_id, title_extra=''):
    """Generate EEG image with left channels in blue, right in red."""
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    # Bandpass 0.5-5 Hz for cleaner display
    sos = butter(4, [0.5 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    for i in range(n_channels):
        try:
            seg_bi[i, :] = sosfiltfilt(sos, seg_bi[i, :])
        except:
            pass

    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    z_scale = 0.01
    clip_uv = 300.0

    # Channel order: left lateral, left parasag, midline, right parasag, right lateral
    # With spacers between groups for visual clarity
    DISPLAY_ORDER = [
        (0, 'Fp1-F7'),   # left lateral chain
        (1, 'F7-T3'),
        (2, 'T3-T5'),
        (3, 'T5-O1'),
        None,             # spacer
        (8, 'Fp1-F3'),   # left parasagittal
        (9, 'F3-C3'),
        (10, 'C3-P3'),
        (11, 'P3-O1'),
        None,             # spacer
        (16, 'Fz-Cz'),   # midline
        (17, 'Cz-Pz'),
        None,             # spacer
        (12, 'Fp2-F4'),  # right parasagittal
        (13, 'F4-C4'),
        (14, 'C4-P4'),
        (15, 'P4-O2'),
        None,             # spacer
        (4, 'Fp2-F8'),   # right lateral chain
        (5, 'F8-T4'),
        (6, 'T4-T6'),
        (7, 'T6-O2'),
    ]
    n_display = len(DISPLAY_ORDER)

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
        clipped = np.clip(seg_bi[ch_idx, :], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.7, clip_on=True)

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

    # Group labels on the right side
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

    title = f'{patient_id}'
    if title_extra:
        title += f'  —  {title_extra}'
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
    print("  LRDA Side Labeling Viewer")
    print("=" * 70)

    # Load corrected data
    corrected_path = V2_CACHE / 'lateral_v2_data_corrected.pkl'
    if corrected_path.exists():
        with open(corrected_path, 'rb') as f:
            data = pickle.load(f)
        print("Using corrected labels")
    else:
        with open(V2_CACHE / 'lateral_v2_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Using original labels")

    df = data['df']
    segs = data['segs']

    # Get confirmed LRDA cases from our dataset
    confirmed_lrda = df[df['subtype'] == 'lrda']['patient_id'].values
    print(f"Confirmed LRDA in balanced dataset: {len(confirmed_lrda)}")

    # Also load full patients.csv to find additional strong LRDA cases
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'), dtype={'patient_id': str})
    for col in ['n_expert_votes', 'vote_agreement', 'vote_lrda', 'vote_grda']:
        pat[col] = pd.to_numeric(pat[col], errors='coerce').fillna(0)

    # Additional IIIC cases: subtype=lrda, >=10 votes, lrda votes > 2× grda votes
    additional = pat[
        (pat['label_source'] == 'expert_majority') &
        (pat['n_expert_votes'] >= 10) &
        (pat['subtype'] == 'lrda') &
        (pat['vote_lrda'] > 2 * pat['vote_grda']) &
        (~pat['patient_id'].isin(confirmed_lrda))
    ].copy()

    # Load EEG for additional cases
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'), dtype={'patient_id': str})
    from lateralization_contest.harness_v2 import load_segment
    additional_segs = {}
    for _, row in additional.iterrows():
        pid = row['patient_id']
        if pid in segs:
            continue
        pid_segs = seg_df[seg_df['patient_id'] == pid]
        for _, sr in pid_segs.iterrows():
            seg = load_segment(sr['mat_file'])
            if seg is not None and seg.shape == (18, 2000):
                additional_segs[pid] = seg
                break
        if pid not in additional_segs:
            seg = load_segment(f'{pid}_seg000.mat')
            if seg is not None and seg.shape == (18, 2000):
                additional_segs[pid] = seg

    print(f"Additional strong LRDA cases: {len(additional_segs)}")

    # Combine all LRDA cases
    all_segs = {}
    for pid in confirmed_lrda:
        if pid in segs:
            all_segs[pid] = segs[pid]
    all_segs.update(additional_segs)

    # Get vote info for display
    pat_info = {}
    for _, row in pat.iterrows():
        pid = row['patient_id']
        if pid in all_segs:
            pat_info[pid] = {
                'vote_agreement': float(row['vote_agreement']) if row['vote_agreement'] > 0 else None,
                'n_experts': int(row['n_expert_votes']),
                'vote_lrda': int(row['vote_lrda']),
                'vote_grda': int(row['vote_grda']),
                'existing_laterality': row.get('laterality', None),
            }

    # Build case list, skip cases that already have laterality labels
    cases = []
    for pid, seg in all_segs.items():
        info = pat_info.get(pid, {})
        existing = info.get('existing_laterality')
        if existing in ('left', 'right'):
            continue  # already labeled
        n_exp = info.get('n_experts', 0)
        v_lrda = info.get('vote_lrda', 0)
        v_grda = info.get('vote_grda', 0)
        va = info.get('vote_agreement')
        source = 'confirmed' if pid in confirmed_lrda else 'strong_lrda_vote'

        cases.append({
            'patient_id': pid,
            'source': source,
            'n_experts': n_exp,
            'vote_lrda': v_lrda,
            'vote_grda': v_grda,
            'vote_agreement': round(va, 3) if va else None,
        })

    # Sort: highest LRDA confidence first
    cases.sort(key=lambda c: -(c['vote_lrda'] / max(c['n_experts'], 1)))
    print(f"Cases to label (excluding already-labeled): {len(cases)}")

    # Generate images
    print("Generating EEG images...")
    images = {}
    for i, case in enumerate(cases):
        pid = case['patient_id']
        seg = all_segs[pid]
        title = f"LRDA ({case['source']})  |  {case['vote_lrda']}/{case['n_experts']} voted LRDA"
        try:
            jpeg_bytes = generate_eeg_jpeg(seg, pid, title)
            images[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
        except Exception as e:
            print(f"  Failed {pid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(cases)} images...")
    print(f"  Generated {len(images)} images")

    # Build HTML
    print("Building HTML viewer...")
    html = _build_html(cases, images)
    out_path = OUT_DIR / 'lrda_side_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"\nSaved to: {out_path}")
    print(f"Total time: {time.time() - t0:.0f}s")


def _build_html(cases, images):
    n_total = len(cases)
    cases_json = json.dumps(cases)
    images_json = json.dumps(images)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>LRDA Side Labeler — {n_total} cases</title>
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
    display: flex; align-items: center; justify-content: center; gap: 40px;
}}
.key {{ display: inline-block; padding: 6px 16px; background: #333; border-radius: 6px;
        border: 2px solid #555; font-weight: bold; font-size: 16px; }}
.key-left {{ background: #1a2a4a; border-color: #2266cc; color: #4488ff; }}
.key-right {{ background: #4a1a1a; border-color: #cc3322; color: #ff4444; }}
.key-skip {{ background: #2a2a1a; border-color: #888; color: #aaa; }}
#viewer-area {{
    margin-top: 50px; margin-bottom: 70px; display: flex;
    justify-content: center; padding: 10px;
}}
#eeg-img {{ max-width: 100%; max-height: calc(100vh - 140px); }}
#decision-indicator {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-size: 72px; font-weight: bold; z-index: 200; opacity: 0;
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
</style>
</head>
<body>

<div id="header">
    <h1>LRDA Side Labeler</h1>
    <div class="info">{n_total} LRDA cases to label (Left or Right dominant)</div>
    <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
    <div id="counter">0 / {n_total}</div>
</div>

<div id="decision-indicator"></div>

<div id="viewer-area">
    <img id="eeg-img" src="" alt="Loading...">
</div>

<div id="controls">
    <div>
        <span class="key key-left">1</span> LEFT dominant
    </div>
    <div>
        <span class="key key-right">2</span> RIGHT dominant
    </div>
    <div>
        <span class="key key-skip">S</span> Skip / Unsure
    </div>
    <div style="color:#666">
        <span class="key" style="font-size:12px">←</span> Back
    </div>
</div>

<div id="summary">
    <h2>Labeling Complete!</h2>
    <div class="stat" id="stat-total"></div>
    <div class="stat" id="stat-left" style="color:#4488ff"></div>
    <div class="stat" id="stat-right" style="color:#ff4444"></div>
    <div class="stat" id="stat-skip" style="color:#888"></div>
    <button id="download-btn" onclick="downloadResults()">Download Results JSON</button>
</div>

<script>
const cases = {cases_json};
const images = {images_json};
let currentIdx = 0;
let decisions = {{}};

const STORAGE_KEY = 'lrda_side_labeler_v1';

try {{
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && saved.decisions) {{
        decisions = saved.decisions;
        for (let i = 0; i < cases.length; i++) {{
            if (!(cases[i].patient_id in decisions)) {{
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
    if (idx >= cases.length) {{ showSummary(); return; }}
    const c = cases[idx];
    const img = document.getElementById('eeg-img');
    img.src = images[c.patient_id] ? 'data:image/jpeg;base64,' + images[c.patient_id] : '';

    const reviewed = Object.keys(decisions).length;
    document.getElementById('counter').textContent = reviewed + ' / ' + cases.length;
    document.getElementById('progress-fill').style.width = (reviewed / cases.length * 100) + '%';
}}

function decide(side) {{
    if (currentIdx >= cases.length) return;
    const c = cases[currentIdx];
    decisions[c.patient_id] = side;
    saveProgress();

    const ind = document.getElementById('decision-indicator');
    if (side === 'left') {{
        ind.textContent = '← LEFT'; ind.style.color = '#4488ff';
    }} else if (side === 'right') {{
        ind.textContent = 'RIGHT →'; ind.style.color = '#ff4444';
    }} else {{
        ind.textContent = '⏭ SKIP'; ind.style.color = '#888';
    }}
    ind.classList.add('flash');
    setTimeout(() => ind.classList.remove('flash'), 300);

    currentIdx++;
    setTimeout(() => showCase(currentIdx), 200);
}}

function goBack() {{
    if (currentIdx > 0) {{
        currentIdx--;
        delete decisions[cases[currentIdx].patient_id];
        saveProgress();
        showCase(currentIdx);
    }}
}}

function showSummary() {{
    const vals = Object.values(decisions);
    document.getElementById('stat-total').textContent = 'Total: ' + vals.length;
    document.getElementById('stat-left').textContent = 'Left: ' + vals.filter(d => d === 'left').length;
    document.getElementById('stat-right').textContent = 'Right: ' + vals.filter(d => d === 'right').length;
    document.getElementById('stat-skip').textContent = 'Skipped: ' + vals.filter(d => d === 'skip').length;
    document.getElementById('summary').style.display = 'block';
    document.getElementById('counter').textContent = 'DONE';
    document.getElementById('progress-fill').style.width = '100%';
}}

function downloadResults() {{
    const results = {{
        timestamp: new Date().toISOString(),
        total_cases: cases.length,
        labels: {{}},
        summary: {{ left: 0, right: 0, skip: 0 }},
    }};
    for (const c of cases) {{
        const d = decisions[c.patient_id] || 'unlabeled';
        results.labels[c.patient_id] = {{
            side: d,
            source: c.source,
            n_experts: c.n_experts,
            vote_lrda: c.vote_lrda,
        }};
        if (d in results.summary) results.summary[d]++;
    }}
    const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'lrda_side_labels.json';
    a.click();
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === '1') {{ e.preventDefault(); decide('left'); }}
    else if (e.key === '2') {{ e.preventDefault(); decide('right'); }}
    else if (e.key === 's' || e.key === 'S') {{ e.preventDefault(); decide('skip'); }}
    else if (e.key === 'ArrowLeft') {{ e.preventDefault(); goBack(); }}
}});

showCase(currentIdx);
</script>
</body>
</html>"""


if __name__ == '__main__':
    main()
