"""
LPD Laterality Labeling Tool.

For each unlabeled LPD case:
  - Runs CNN+Attention PD detector to predict laterality
  - Shows EEG with predicted side in blue, other side in black
  - Channel layout: left hemi | right hemi | midline (same as BIPD viewer)

Controls:
  Enter = accept model prediction, advance
  Space = reject, choose other side, advance
  N = not LPDs (mark as excluded)
  Arrow keys = navigate without saving

Usage:
    conda run -n foe_dl python code/generate_laterality_labeler.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from cet_model.auto_pipeline import load_cnn_attn_models, DEVICE

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOWPASS_HZ = 20.0

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

DISPLAY_ORDER = [
    0, 1, 2, 3, 8, 9, 10, 11,   # left
    -1,                           # gap
    16, 17,                       # midline
    -1,                           # gap
    4, 5, 6, 7, 12, 13, 14, 15, # right
]


@torch.no_grad()
def predict_laterality(seg, cnn_models, device=DEVICE):
    """Predict laterality using CNN+Attention PD probabilities per channel.

    Returns: 'left' or 'right', and the confidence (left_score, right_score).
    """
    n_channels = min(seg.shape[0], 18)
    pd_probs = []

    for ch in range(n_channels):
        ch_data = seg[ch].astype(np.float32).copy()
        if not np.all(np.isfinite(ch_data)):
            pd_probs.append(0.0)
            continue
        mu, std = np.mean(ch_data), np.std(ch_data)
        ch_data = (ch_data - mu) / std if std > 1e-8 else ch_data - mu
        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(device)

        probs = []
        for model in cnn_models:
            pd_prob, _, _ = model(x)
            probs.append(pd_prob.item())
        pd_probs.append(np.mean(probs))

    pd_probs = np.array(pd_probs)
    left_score = float(np.mean(pd_probs[LEFT_INDICES]))
    right_score = float(np.mean(pd_probs[RIGHT_INDICES]))

    predicted = 'left' if left_score > right_score else 'right'
    confidence = max(left_score, right_score) / (left_score + right_score + 1e-8)

    return predicted, left_score, right_score, confidence


def downsample_2d(arr, target_len):
    n = arr.shape[1]
    if n <= target_len:
        return arr.tolist()
    indices = np.linspace(0, n - 1, target_len).astype(int)
    return arr[:, indices].tolist()


def main():
    import pandas as pd

    print("=" * 70)
    print("  LPD Laterality Labeling Tool")
    print("=" * 70)

    # Load data
    patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Find LPD cases needing laterality
    lpd = patients[patients['subtype'] == 'lpd']
    need_lat = lpd[
        ~lpd['laterality'].isin(['left', 'right', 'bilateral']) &
        (lpd['excluded'] != True)
    ]
    print(f"LPD cases needing laterality: {len(need_lat)}")

    # Load CNN models for laterality prediction
    print("Loading CNN+Attention models...")
    cnn_models = load_cnn_attn_models(device=DEVICE)
    print(f"  {len(cnn_models)} models on {DEVICE}")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    cases = []
    n_skipped = 0

    # Sort by patient_id for consistent ordering
    pids = sorted(need_lat['patient_id'].values)

    print(f"\nProcessing {len(pids)} cases...")
    for i, pid in enumerate(pids):
        # Try segments dict first, then direct file load
        pat_segs = segments.get(pid, []) or segments.get(str(pid), [])
        if pat_segs:
            seg = pat_segs[0]
        else:
            seg = None
            for suffix in ['_seg000.mat', '.mat']:
                path = DATA_DIR / 'eeg' / f'{pid}{suffix}'
                if path.exists():
                    import scipy.io as sio
                    mat = sio.loadmat(str(path))
                    key = [k for k in mat.keys() if not k.startswith('_')][0]
                    seg = mat[key]
                    if seg.shape[0] > seg.shape[1]:
                        seg = seg.T
                    seg = seg[:18, :2000]
                    break
            if seg is None:
                n_skipped += 1
                continue

        # Lowpass for display
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except:
                seg_display[ch] = seg[ch]

        # Predict laterality
        predicted, left_score, right_score, confidence = predict_laterality(
            seg, cnn_models, DEVICE)

        row = patients[patients['patient_id'] == pid].iloc[0]
        gold_freq = row.get('gold_standard_freq', None)
        gold_freq = float(gold_freq) if pd.notna(gold_freq) else None

        cases.append({
            'patient_id': str(pid),
            'predicted_lat': predicted,
            'left_score': round(left_score, 3),
            'right_score': round(right_score, 3),
            'confidence': round(confidence, 3),
            'gold_freq': gold_freq,
            'eeg_data': downsample_2d(seg_display, 1000),
        })

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(pids)} ({n_skipped} skipped)")

    print(f"  Total: {len(cases)} cases ({n_skipped} skipped)")

    # Build HTML
    print("\nBuilding HTML...")
    html = build_html(cases)
    out_path = OUT_DIR / 'laterality_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


def build_html(cases_data):
    cases_json = json.dumps(cases_data,
        default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    display_order_json = json.dumps(DISPLAY_ORDER)
    channel_names_json = json.dumps(BIPOLAR_CHANNELS)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LPD Laterality Labeling</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; }}
  #header {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #222; border-bottom: 2px solid #444; flex-wrap: wrap; gap: 8px; }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
  #progress-bar-wrap {{ width: 100%; height: 6px; background: #333; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}
  #verdict {{ font-size: 22px; font-weight: bold; padding: 8px 20px; text-align: center; letter-spacing: 2px; }}
  .verdict-left {{ background: #1a1a3a; color: #4488ff; }}
  .verdict-right {{ background: #3a1a1a; color: #ff6666; }}
  .verdict-none {{ background: #222; color: #888; }}
  .verdict-not-lpd {{ background: #3a3a1a; color: #ffaa44; }}
  .verdict-gpd {{ background: #20205a; color: #8888ff; }}
  #info-panel {{ background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center; gap: 20px; font-size: 13px; border-bottom: 1px solid #333; flex-wrap: wrap; }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}
  #canvas-container {{ text-align: center; padding: 8px; }}
  #eeg-canvas {{ display: block; margin: 0 auto; }}
  .export-btn {{ padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px; background: #2a3a2a; color: #44cc44; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; }}
  #shortcuts {{ font-size: 12px; color: #777; padding: 6px 16px; background: #222; border-top: 1px solid #333; line-height: 1.8; }}
  #stats {{ font-size: 13px; color: #aaa; padding: 4px 16px; background: #252525; }}
  .pred-box {{ padding: 6px 16px; font-size: 15px; font-weight: bold; text-align: center; }}
  .pred-left {{ background: #1a1a3a; color: #4488ff; }}
  .pred-right {{ background: #3a1a1a; color: #ff6666; }}
</style>
</head>
<body>

<div id="header">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:16px;font-weight:bold;color:#44aaff;">LPD Laterality Labeling</span>
    <span id="counter" style="font-size:13px;color:#aaa;">1 / 0</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status" style="color:#44cc44;font-size:13px;"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
<div id="verdict" class="verdict-none">— not yet reviewed —</div>
<div class="pred-box" id="pred-box">Model prediction: --</div>
<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item">Left PD score: <strong id="info-left" style="color:#4488ff;">--</strong></span>
  <span class="info-item">Right PD score: <strong id="info-right" style="color:#ff6666;">--</strong></span>
  <span class="info-item">Confidence: <strong id="info-conf">--</strong></span>
  <span class="info-item">Gold freq: <strong id="info-freq">--</strong></span>
</div>
<div id="stats"></div>
<div id="canvas-container"><canvas id="eeg-canvas"></canvas></div>
<div id="shortcuts">
  <span class="key">Enter</span> Accept model prediction &amp; advance &nbsp;&nbsp;
  <span class="key">Space</span> Choose OTHER side &amp; advance &nbsp;&nbsp;
  <span class="key">G</span> GPD (reclassify) &nbsp;&nbsp;
  <span class="key">N</span> Not LPDs (exclude) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (no save) &nbsp;&nbsp;
  <span class="key">E</span> Export
</div>

<script>
const CASES = {cases_json};
const DISPLAY_ORDER = {display_order_json};
const ALL_CH = {channel_names_json};
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;
const W = 1200, H = 750;
const ML = 70, MR = 20, MT = 30, MB = 25;
const PL = ML, PR = W-MR, PW = PR-PL, PH = H-MT-MB;

const LEFT_SET = new Set([0,1,2,3,8,9,10,11]);
const RIGHT_SET = new Set([4,5,6,7,12,13,14,15]);

const DISP = DISPLAY_ORDER.map(i => ({{
  idx: i,
  name: i >= 0 ? ALL_CH[i] : '',
  side: i >= 0 ? (LEFT_SET.has(i) ? 'left' : RIGHT_SET.has(i) ? 'right' : 'mid') : 'gap'
}}));
const ND = DISP.length;

function timeToX(t) {{ return PL + (t/DURATION)*PW; }}

let idx = 0;
const SKEY = 'lat_labeler_v1';
let verdicts = {{}};
try {{ verdicts = JSON.parse(localStorage.getItem(SKEY) || '{{}}'); }} catch(e) {{ verdicts = {{}}; }}
function save() {{ localStorage.setItem(SKEY, JSON.stringify(verdicts)); }}

function countStats() {{
  let left=0, right=0, notlpd=0;
  for (const v of Object.values(verdicts)) {{
    if (v==='left') left++;
    else if (v==='right') right++;
    else if (v==='not_lpd') notlpd++;
  }}
  return {{left, right, notlpd, total: left+right+notlpd}};
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eeg = c.eeg_data;
  const ns = eeg[0].length;
  const sp = PH / (ND + 1);
  const predicted = c.predicted_lat;

  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  // Hemisphere backgrounds
  // Left channels: display indices 0-7
  // Midline: 9-10 (with gaps at 8 and 11)
  // Right channels: 12-19
  const leftEndY = MT + sp * 9;
  const rightStartY = MT + sp * 13;

  if (predicted === 'left') {{
    ctx.fillStyle = 'rgba(68,136,255,0.06)';
    ctx.fillRect(PL, MT, PW, leftEndY - MT);
  }} else {{
    ctx.fillStyle = 'rgba(255,100,100,0.06)';
    ctx.fillRect(PL, rightStartY, PW, H - MB - rightStartY);
  }}

  // Grid
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 0.5; ctx.setLineDash([4,4]);
  for (let s=0; s<=10; s++) {{ const x=timeToX(s); ctx.beginPath(); ctx.moveTo(x,MT); ctx.lineTo(x,H-MB); ctx.stroke(); }}
  ctx.setLineDash([]);

  // Traces
  for (let di=0; di<ND; di++) {{
    const ch = DISP[di];
    if (ch.idx < 0) continue;
    const yc = MT + sp*(di+1);

    // Color: predicted side = blue, other = black
    let traceColor = '#000';
    if (ch.side === predicted) {{
      traceColor = '#0055cc';
    }} else if (ch.side !== 'mid' && ch.side !== 'gap') {{
      traceColor = '#333';
    }}

    ctx.strokeStyle = traceColor;
    ctx.lineWidth = ch.side === predicted ? 1.0 : 0.6;
    ctx.beginPath();
    for (let si=0; si<ns; si++) {{
      const x = PL + (si/(ns-1))*PW;
      const v = Math.max(-CLIP_UV, Math.min(CLIP_UV, eeg[ch.idx][si]));
      const y = yc - v*Z_SCALE*sp;
      si===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }}
    ctx.stroke();
  }}

  // Channel labels
  ctx.font = '10px Consolas,Monaco,monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let di=0; di<ND; di++) {{
    const ch = DISP[di]; if (ch.idx<0) continue;
    const yc = MT + sp*(di+1);
    if (ch.side === predicted) {{
      ctx.fillStyle = '#0055cc';
      ctx.font = 'bold 10px Consolas,Monaco,monospace';
    }} else if (ch.side === 'mid') {{
      ctx.fillStyle = '#666';
      ctx.font = '10px Consolas,Monaco,monospace';
    }} else {{
      ctx.fillStyle = '#999';
      ctx.font = '10px Consolas,Monaco,monospace';
    }}
    ctx.fillText(ch.name, PL-4, yc);
  }}

  // Time
  ctx.fillStyle='#000'; ctx.font='10px Consolas'; ctx.textAlign='center'; ctx.textBaseline='top';
  for (let s=0;s<=10;s++) ctx.fillText(s+'s', timeToX(s), H-MB+4);

  // Title
  ctx.fillStyle='#000'; ctx.font='bold 13px Consolas'; ctx.textAlign='center'; ctx.textBaseline='top';
  const freqStr = c.gold_freq ? c.gold_freq.toFixed(2) + ' Hz' : 'N/A';
  ctx.fillText(c.patient_id + '  |  LPD  |  ' + freqStr, W/2, 6);

  // Hemisphere labels
  ctx.font = 'bold 12px Consolas'; ctx.textAlign = 'left';
  ctx.fillStyle = predicted === 'left' ? '#0055cc' : '#999';
  ctx.fillText('LEFT', PL+5, MT+2);
  ctx.fillStyle = predicted === 'right' ? '#0055cc' : '#999';
  ctx.fillText('RIGHT', PL+5, rightStartY+2);
}}

function updateUI() {{
  const c = CASES[idx];
  const v = verdicts[c.patient_id];

  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-left').textContent = c.left_score.toFixed(3);
  document.getElementById('info-right').textContent = c.right_score.toFixed(3);
  document.getElementById('info-conf').textContent = (c.confidence * 100).toFixed(0) + '%';
  document.getElementById('info-freq').textContent = c.gold_freq ? c.gold_freq.toFixed(2) + ' Hz' : 'N/A';
  document.getElementById('counter').textContent = (idx+1)+' / '+CASES.length;
  document.getElementById('progress-bar').style.width = ((idx+1)/CASES.length*100).toFixed(1)+'%';

  // Prediction box
  const pb = document.getElementById('pred-box');
  pb.textContent = 'Model prediction: ' + c.predicted_lat.toUpperCase() +
    ' (L=' + c.left_score.toFixed(3) + ' R=' + c.right_score.toFixed(3) + ')';
  pb.className = 'pred-box pred-' + c.predicted_lat;

  // Verdict
  const el = document.getElementById('verdict');
  if (v === 'left') {{ el.textContent = 'LABELED: LEFT'; el.className = 'verdict-left'; }}
  else if (v === 'right') {{ el.textContent = 'LABELED: RIGHT'; el.className = 'verdict-right'; }}
  else if (v === 'gpd') {{ el.textContent = 'GPD'; el.className = 'verdict-gpd'; }}
  else if (v === 'not_lpd') {{ el.textContent = 'NOT LPD (excluded)'; el.className = 'verdict-not-lpd'; }}
  else {{ el.textContent = '— not yet reviewed —'; el.className = 'verdict-none'; }}

  const s = countStats();
  document.getElementById('stats').textContent =
    'Left: ' + s.left + '  |  Right: ' + s.right + '  |  Not LPD: ' + s.notlpd + '  |  Total: ' + s.total + ' / ' + CASES.length;
}}

function show() {{
  idx = Math.max(0, Math.min(idx, CASES.length-1));
  drawEEG(); updateUI();
}}

function setVerdict(v) {{
  verdicts[CASES[idx].patient_id] = v;
  save(); updateUI();
  if (idx < CASES.length-1) {{ idx++; show(); }}
}}

function exportJSON() {{
  const out = {{}};
  for (const c of CASES) {{
    const v = verdicts[c.patient_id];
    if (v) {{
      out[c.patient_id] = {{
        patient_id: c.patient_id,
        laterality: (v === 'not_lpd' || v === 'gpd') ? null : v,
        is_gpd: v === 'gpd',
        predicted_lat: c.predicted_lat,
        accepted_prediction: v === c.predicted_lat,
        not_lpd: v === 'not_lpd',
        left_score: c.left_score,
        right_score: c.right_score,
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'laterality_labels.json'; a.click();
  document.getElementById('save-status').textContent = 'Exported '+Object.keys(out).length;
  setTimeout(()=>{{ document.getElementById('save-status').textContent=''; }}, 3000);
}}

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    setVerdict(CASES[idx].predicted_lat);  // Accept model prediction
  }}
  else if (e.key === ' ') {{
    e.preventDefault();
    const other = CASES[idx].predicted_lat === 'left' ? 'right' : 'left';
    setVerdict(other);  // Choose other side
  }}
  else if (e.key === 'n' || e.key === 'N') {{
    setVerdict('not_lpd');
  }}
  else if (e.key === 'g' || e.key === 'G') {{
    setVerdict('gpd');
  }}
  else if (e.key === 'ArrowLeft') {{ e.preventDefault(); idx=Math.max(0,idx-1); show(); }}
  else if (e.key === 'ArrowRight') {{ e.preventDefault(); idx=Math.min(CASES.length-1,idx+1); show(); }}
  else if (e.key === 'e' || e.key === 'E') exportJSON();
}});

show();
</script>
</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
