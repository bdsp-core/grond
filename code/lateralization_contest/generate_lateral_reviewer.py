#!/usr/bin/env python3
"""Generate LRDA vs GRDA misclassification review viewer.

Trains the best model (spatial_all + GBM_deep), finds cases where the model
disagrees with the IIIC crowd label, generates EEG images, and builds an
HTML viewer where MW can review and relabel.

Predicted LRDA cases first, then predicted GRDA.
- Enter: accept model prediction (relabel)
- Space: keep original label (model was wrong)

Usage:
    conda run -n morgoth python code/lateralization_contest/generate_lateral_reviewer.py
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
from scipy.signal import butter, filtfilt, detrend
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
FEAT_CACHE = RESULTS_DIR / '_feat_cache.pkl'
OUT_DIR = PROJECT_DIR / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
FS = 200


def generate_eeg_jpeg(seg_bi, patient_id, title_extra=''):
    """Generate EEG image as JPEG bytes."""
    seg_bi = seg_bi.astype(np.float64)
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    # Lowpass at 20 Hz
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

    title = f'{patient_id}'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.95, bottom=0.045)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=100, pil_kwargs={'quality': 75})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def main():
    t0 = time.time()
    print("=" * 70)
    print("  LRDA vs GRDA Misclassification Reviewer")
    print("=" * 70)

    # Load data
    print("Loading data...")
    with open(V2_CACHE / 'lateral_v2_data.pkl', 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    segs = data['segs']

    # Balance
    lrda_pids = df[df['subtype'] == 'lrda']['patient_id'].values
    grda_pids = df[df['subtype'] == 'grda']['patient_id'].values
    np.random.seed(42)
    grda_sub = np.random.choice(grda_pids, size=len(lrda_pids), replace=False)
    use_pids = set(list(lrda_pids) + list(grda_sub))
    df_bal = df[df['patient_id'].isin(use_pids)].copy().reset_index(drop=True)
    labels = (df_bal['subtype'] == 'lrda').astype(int).values
    print(f"Balanced: {(labels==1).sum()} LRDA + {(labels==0).sum()} GRDA = {len(labels)}")

    # Load features
    print("Loading features...")
    with open(FEAT_CACHE, 'rb') as f:
        feat_cache = pickle.load(f)
    X, keys = feat_cache['spatial_all']
    print(f"Features: {X.shape[1]} (spatial_all)")

    # Train model and get CV predictions
    print("Running 10-fold CV to get predictions...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=42)
    y_prob = cross_val_predict(clf, X, labels, cv=cv, method='predict_proba', n_jobs=-1)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Find disagreements
    disagree_mask = y_pred != labels
    n_disagree = disagree_mask.sum()
    print(f"Disagreements: {n_disagree}/{len(labels)} ({100*n_disagree/len(labels):.1f}%)")
    print(f"  Model says LRDA but label is GRDA: {((y_pred==1) & (labels==0)).sum()}")
    print(f"  Model says GRDA but label is LRDA: {((y_pred==0) & (labels==1)).sum()}")

    # Build case list: predicted LRDA first (sorted by confidence), then predicted GRDA
    cases = []
    for i in range(len(df_bal)):
        if not disagree_mask[i]:
            continue
        row = df_bal.iloc[i]
        pid = row['patient_id']
        cases.append({
            'patient_id': pid,
            'iiic_label': row['subtype'],
            'model_pred': 'lrda' if y_pred[i] == 1 else 'grda',
            'model_prob_lrda': round(float(y_prob[i]), 4),
            'vote_agreement': round(float(row['vote_agreement']), 3),
        })

    # Sort: predicted LRDA first (high prob), then predicted GRDA (low prob)
    pred_lrda = [c for c in cases if c['model_pred'] == 'lrda']
    pred_grda = [c for c in cases if c['model_pred'] == 'grda']
    pred_lrda.sort(key=lambda c: -c['model_prob_lrda'])
    pred_grda.sort(key=lambda c: c['model_prob_lrda'])
    cases = pred_lrda + pred_grda
    print(f"Review order: {len(pred_lrda)} predicted LRDA, then {len(pred_grda)} predicted GRDA")

    # Generate EEG images
    print("Generating EEG images...")
    images = {}
    for i, case in enumerate(cases):
        pid = case['patient_id']
        seg = segs[pid]
        title = f"Model: {case['model_pred'].upper()} ({case['model_prob_lrda']:.0%})  |  IIIC: {case['iiic_label'].upper()} ({case['vote_agreement']:.0%} agree)"
        try:
            jpeg_bytes = generate_eeg_jpeg(seg, pid, title)
            images[pid] = base64.b64encode(jpeg_bytes).decode('ascii')
        except Exception as e:
            print(f"  Failed {pid}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(cases)} images...")
    print(f"  Generated {len(images)} images")

    # Build HTML
    print("Building HTML viewer...")
    html = _build_html(cases, images)

    out_path = OUT_DIR / 'lateral_misclass_reviewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"\nSaved to: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")

    # Also save the case data as JSON for later use
    json_path = RESULTS_DIR / 'misclass_cases.json'
    with open(str(json_path), 'w') as f:
        json.dump(cases, f, indent=2)
    print(f"Case data: {json_path}")


def _build_html(cases, images):
    n_total = len(cases)
    n_pred_lrda = sum(1 for c in cases if c['model_pred'] == 'lrda')
    n_pred_grda = n_total - n_pred_lrda

    cases_json = json.dumps(cases)
    images_json = json.dumps(images)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>LRDA vs GRDA Review — {n_total} cases</title>
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
.info .lrda {{ color: #ff6644; font-weight: bold; }}
.info .grda {{ color: #4488ff; font-weight: bold; }}
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
.key-enter {{ background: #2a4a2a; border-color: #44cc88; color: #44cc88; }}
.key-space {{ background: #4a2a2a; border-color: #cc4444; color: #cc4444; }}
#viewer-area {{
    margin-top: 50px; margin-bottom: 60px; display: flex;
    justify-content: center; padding: 10px;
}}
#eeg-img {{ max-width: 100%; max-height: calc(100vh - 130px); }}
#label-badge {{
    position: fixed; top: 55px; right: 20px; font-size: 20px; padding: 8px 16px;
    border-radius: 8px; font-weight: bold; z-index: 101;
}}
.badge-lrda {{ background: #ff664433; color: #ff6644; border: 2px solid #ff6644; }}
.badge-grda {{ background: #4488ff33; color: #4488ff; border: 2px solid #4488ff; }}
#section-label {{
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
    <h1>LRDA vs GRDA Review</h1>
    <div class="info">
        <span class="lrda">Pred LRDA: {n_pred_lrda}</span> then
        <span class="grda">Pred GRDA: {n_pred_grda}</span>
        = {n_total} disagreements
    </div>
    <div id="progress-bar"><div id="progress-fill" style="width:0%"></div></div>
    <div id="counter">0 / {n_total}</div>
</div>

<div id="section-label"></div>
<div id="label-badge"></div>
<div id="decision-indicator"></div>

<div id="viewer-area">
    <img id="eeg-img" src="" alt="Loading...">
</div>

<div id="controls">
    <div>
        <span class="key key-enter">Enter</span> Accept model prediction (relabel)
    </div>
    <div>
        <span class="key key-space">Space</span> Keep original label (model wrong)
    </div>
    <div style="color:#666">
        <span class="key">←</span> Go back
    </div>
</div>

<div id="summary">
    <h2>Review Complete!</h2>
    <div class="stat" id="stat-total"></div>
    <div class="stat" id="stat-accepted"></div>
    <div class="stat" id="stat-rejected"></div>
    <button id="download-btn" onclick="downloadResults()">Download Results JSON</button>
</div>

<script>
const cases = {cases_json};
const images = {images_json};
let currentIdx = 0;
let decisions = {{}};  // pid -> 'accept' or 'reject'

const STORAGE_KEY = 'lateral_misclass_review_v1';

// Load saved progress
try {{
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && saved.decisions) {{
        decisions = saved.decisions;
        // Find first unreviewed
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
    if (idx >= cases.length) {{
        showSummary();
        return;
    }}
    const c = cases[idx];
    const img = document.getElementById('eeg-img');
    if (images[c.patient_id]) {{
        img.src = 'data:image/jpeg;base64,' + images[c.patient_id];
    }} else {{
        img.src = '';
        img.alt = 'No image for ' + c.patient_id;
    }}

    // Badge
    const badge = document.getElementById('label-badge');
    badge.textContent = 'Model: ' + c.model_pred.toUpperCase() +
        ' (' + (c.model_prob_lrda * 100).toFixed(0) + '%)';
    badge.className = c.model_pred === 'lrda' ? 'badge-lrda' : 'badge-grda';

    // Section label
    const section = document.getElementById('section-label');
    const predLrdaCount = cases.filter(x => x.model_pred === 'lrda').length;
    if (idx < predLrdaCount) {{
        section.textContent = 'Section: Predicted LRDA (' + (idx+1) + '/' + predLrdaCount + ')';
        section.style.borderLeft = '3px solid #ff6644';
    }} else {{
        const gIdx = idx - predLrdaCount;
        const gTotal = cases.length - predLrdaCount;
        section.textContent = 'Section: Predicted GRDA (' + (gIdx+1) + '/' + gTotal + ')';
        section.style.borderLeft = '3px solid #4488ff';
    }}

    // Counter
    const reviewed = Object.keys(decisions).length;
    document.getElementById('counter').textContent = reviewed + ' / ' + cases.length;
    document.getElementById('progress-fill').style.width =
        (reviewed / cases.length * 100) + '%';

    // Show if already decided
    if (decisions[c.patient_id]) {{
        const d = decisions[c.patient_id];
        section.textContent += (d === 'accept' ? '  ✓ RELABELED' : '  ✗ KEPT');
    }}
}}

function decide(action) {{
    if (currentIdx >= cases.length) return;
    const c = cases[currentIdx];
    decisions[c.patient_id] = action;
    saveProgress();

    // Flash indicator
    const ind = document.getElementById('decision-indicator');
    if (action === 'accept') {{
        ind.textContent = '✓ RELABEL → ' + c.model_pred.toUpperCase();
        ind.style.color = '#44cc88';
    }} else {{
        ind.textContent = '✗ KEEP ' + c.iiic_label.toUpperCase();
        ind.style.color = '#cc4444';
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
        delete decisions[c.patient_id];
        saveProgress();
        showCase(currentIdx);
    }}
}}

function showSummary() {{
    const accepted = Object.values(decisions).filter(d => d === 'accept').length;
    const rejected = Object.values(decisions).filter(d => d === 'reject').length;
    document.getElementById('stat-total').textContent = 'Total reviewed: ' + (accepted + rejected);
    document.getElementById('stat-accepted').innerHTML =
        '<span style="color:#44cc88">Relabeled (model correct): ' + accepted + '</span>';
    document.getElementById('stat-rejected').innerHTML =
        '<span style="color:#cc4444">Kept original (model wrong): ' + rejected + '</span>';
    document.getElementById('summary').style.display = 'block';

    document.getElementById('counter').textContent = 'DONE';
    document.getElementById('progress-fill').style.width = '100%';
}}

function downloadResults() {{
    const results = {{
        timestamp: new Date().toISOString(),
        total_cases: cases.length,
        decisions: {{}},
        relabeled: [],
        kept: [],
    }};
    for (const c of cases) {{
        const d = decisions[c.patient_id] || 'unreviewed';
        results.decisions[c.patient_id] = {{
            iiic_label: c.iiic_label,
            model_pred: c.model_pred,
            model_prob_lrda: c.model_prob_lrda,
            vote_agreement: c.vote_agreement,
            decision: d,
            new_label: d === 'accept' ? c.model_pred : c.iiic_label,
        }};
        if (d === 'accept') results.relabeled.push(c.patient_id);
        if (d === 'reject') results.kept.push(c.patient_id);
    }}
    const blob = new Blob([JSON.stringify(results, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'lateral_review_results.json';
    a.click();
}}

document.addEventListener('keydown', function(e) {{
    if (e.key === 'Enter') {{
        e.preventDefault();
        decide('accept');
    }} else if (e.key === ' ') {{
        e.preventDefault();
        decide('reject');
    }} else if (e.key === 'ArrowLeft') {{
        e.preventDefault();
        goBack();
    }}
}});

// Start
showCase(currentIdx);
</script>
</body>
</html>"""


if __name__ == '__main__':
    main()
