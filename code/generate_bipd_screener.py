"""
BIPD Screening Tool — Phase 1: Yes/No classification.

Quick review of 198 BIPD candidates. For each case, user presses:
  Y = yes, this is BIPD
  N = no, reject
  Arrow keys to navigate

Channels reordered: left hemi | right hemi | midline (same as BIPD labeler).

Usage:
    conda run -n foe python code/generate_bipd_screener.py
"""

import sys
import json
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import FS

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
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
    4, 5, 6, 7, 12, 13, 14, 15, # right
    -1,                           # gap
    16, 17,                       # midline
]


def load_segment(patient_id):
    for suffix in ['_seg000.mat', '.mat']:
        path = EEG_DIR / f'{patient_id}{suffix}'
        if path.exists():
            mat = sio.loadmat(str(path))
            data_key = [k for k in mat.keys() if not k.startswith('_')][0]
            seg = mat[data_key]
            if seg.shape[0] > seg.shape[1]:
                seg = seg.T
            return seg[:18, :2000] if seg.shape[0] >= 18 else None
    return None


def downsample(arr, target_len):
    n = arr.shape[1]
    if n <= target_len:
        return arr.tolist()
    indices = np.linspace(0, n - 1, target_len).astype(int)
    return arr[:, indices].tolist()


def main():
    print("=" * 70)
    print("  BIPD Screening Tool — Yes/No")
    print("=" * 70)

    with open(str(LABELS_DIR / 'bipd_harvest_manifest.json')) as f:
        bipd_manifest = json.load(f)

    extra = {'sub-S0002119210399_20191215002852': {'subtype': 'bipd'}}
    all_bipd = {**bipd_manifest, **extra}
    print(f"Total candidates: {len(all_bipd)}")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    cases = []
    for i, pid in enumerate(sorted(all_bipd.keys())):
        seg = load_segment(pid)
        if seg is None:
            continue
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except:
                seg_display[ch] = seg[ch]
        cases.append({
            'patient_id': str(pid),
            'eeg_data': downsample(seg_display, 1000),
        })
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(all_bipd)}")

    print(f"  Loaded {len(cases)} cases")

    cases_json = json.dumps(cases, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
    display_order_json = json.dumps(DISPLAY_ORDER)
    channel_names_json = json.dumps(BIPOLAR_CHANNELS)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>BIPD Screening — Yes / No</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; }}
  #header {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #222; border-bottom: 2px solid #444; flex-wrap: wrap; gap: 8px; }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}
  #progress-bar-wrap {{ width: 100%; height: 6px; background: #333; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}
  #verdict {{ font-size: 24px; font-weight: bold; padding: 10px 20px; text-align: center; letter-spacing: 3px; }}
  .verdict-none {{ background: #222; color: #888; }}
  .verdict-yes {{ background: #1a3a1a; color: #44ff66; }}
  .verdict-no {{ background: #3a1a1a; color: #ff4444; }}
  #info-panel {{ background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center; gap: 20px; font-size: 13px; border-bottom: 1px solid #333; }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}
  #canvas-container {{ text-align: center; padding: 8px; }}
  #eeg-canvas {{ display: block; margin: 0 auto; }}
  .export-btn {{ padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px; background: #2a3a2a; color: #44cc44; cursor: pointer; font-family: monospace; font-size: 12px; font-weight: bold; }}
  #shortcuts {{ font-size: 12px; color: #777; padding: 6px 16px; background: #222; border-top: 1px solid #333; line-height: 1.8; }}
  #stats {{ font-size: 13px; color: #aaa; padding: 4px 16px; background: #252525; }}
</style>
</head>
<body>

<div id="header">
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="font-size:16px;font-weight:bold;color:#e040e0;">BIPD Screening — Yes / No</span>
    <span id="counter" style="font-size:13px;color:#aaa;">1 / 0</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status" style="color:#44cc44;font-size:13px;"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
<div id="verdict" class="verdict-none">— not yet reviewed —</div>
<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
</div>
<div id="stats"></div>
<div id="canvas-container"><canvas id="eeg-canvas"></canvas></div>
<div id="shortcuts">
  <span class="key">Y</span> Yes (BIPD) &nbsp;&nbsp;
  <span class="key">N</span> No (reject) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
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
const PL = ML, PR = W - MR, PW = PR - PL, PH = H - MT - MB;

const DISP = DISPLAY_ORDER.map(i => ({{
  idx: i,
  name: i >= 0 ? ALL_CH[i] : '',
  side: i >= 0 ? ([0,1,2,3,8,9,10,11].includes(i) ? 'left' : [4,5,6,7,12,13,14,15].includes(i) ? 'right' : 'mid') : 'gap'
}}));
const ND = DISP.length;

function timeToX(t) {{ return PL + (t/DURATION)*PW; }}

let idx = 0;
const SKEY = 'bipd_screener_v1';
let verdicts = {{}};
try {{ verdicts = JSON.parse(localStorage.getItem(SKEY) || '{{}}'); }} catch(e) {{ verdicts = {{}}; }}
function save() {{ localStorage.setItem(SKEY, JSON.stringify(verdicts)); }}

function countStats() {{
  let y=0, n=0;
  for (const v of Object.values(verdicts)) {{ if(v==='yes') y++; if(v==='no') n++; }}
  return {{yes:y, no:n, total:y+n}};
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eeg = c.eeg_data;
  const ns = eeg[0].length;
  const sp = PH / (ND + 1);

  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  // Hemi backgrounds
  const leftEnd = MT + sp * 9;
  const rightEnd = MT + sp * 18;
  ctx.fillStyle = 'rgba(255,200,200,0.08)';
  ctx.fillRect(PL, MT, PW, leftEnd - MT);
  ctx.fillStyle = 'rgba(200,200,255,0.08)';
  ctx.fillRect(PL, leftEnd, PW, rightEnd - leftEnd);

  // Grid
  ctx.strokeStyle = '#ddd'; ctx.lineWidth = 0.5; ctx.setLineDash([4,4]);
  for (let s=0; s<=10; s++) {{ const x=timeToX(s); ctx.beginPath(); ctx.moveTo(x,MT); ctx.lineTo(x,H-MB); ctx.stroke(); }}
  ctx.setLineDash([]);

  // Traces
  for (let di=0; di<ND; di++) {{
    const ch = DISP[di];
    if (ch.idx < 0) continue;
    const yc = MT + sp*(di+1);
    ctx.strokeStyle = ch.side==='left' ? '#440000' : ch.side==='right' ? '#000044' : '#000';
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    for (let si=0; si<ns; si++) {{
      const x = PL + (si/(ns-1))*PW;
      const v = Math.max(-CLIP_UV, Math.min(CLIP_UV, eeg[ch.idx][si]));
      const y = yc - v*Z_SCALE*sp;
      si===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    }}
    ctx.stroke();
  }}

  // Labels
  ctx.font = '10px Consolas,Monaco,monospace';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let di=0; di<ND; di++) {{
    const ch = DISP[di]; if (ch.idx<0) continue;
    ctx.fillStyle = ch.side==='left' ? '#cc0000' : ch.side==='right' ? '#0044cc' : '#000';
    ctx.fillText(ch.name, PL-4, MT+sp*(di+1));
  }}

  // Time
  ctx.fillStyle='#000'; ctx.font='10px Consolas'; ctx.textAlign='center'; ctx.textBaseline='top';
  for (let s=0;s<=10;s++) ctx.fillText(s+'s', timeToX(s), H-MB+4);

  // Title + hemi labels
  ctx.fillStyle='#000'; ctx.font='bold 13px Consolas'; ctx.textAlign='center'; ctx.textBaseline='top';
  ctx.fillText(c.patient_id + '  |  BIPD candidate', W/2, 6);
  ctx.font='bold 12px Consolas'; ctx.textAlign='left';
  ctx.fillStyle='#cc0000'; ctx.fillText('LEFT', PL+5, MT+2);
  ctx.fillStyle='#0044cc'; ctx.fillText('RIGHT', PL+5, leftEnd+2);
}}

function updateUI() {{
  const c = CASES[idx];
  const v = verdicts[c.patient_id];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('counter').textContent = (idx+1)+' / '+CASES.length;
  document.getElementById('progress-bar').style.width = ((idx+1)/CASES.length*100).toFixed(1)+'%';

  const el = document.getElementById('verdict');
  if (v === 'yes') {{ el.textContent = 'YES — BIPD'; el.className = 'verdict-yes'; }}
  else if (v === 'no') {{ el.textContent = 'NO — REJECTED'; el.className = 'verdict-no'; }}
  else {{ el.textContent = '— not yet reviewed —'; el.className = 'verdict-none'; }}

  const s = countStats();
  document.getElementById('stats').textContent =
    'Yes: ' + s.yes + '  |  No: ' + s.no + '  |  Reviewed: ' + s.total + ' / ' + CASES.length;
}}

function show() {{
  idx = Math.max(0, Math.min(idx, CASES.length-1));
  drawEEG(); updateUI();
}}

function setVerdict(v) {{
  verdicts[CASES[idx].patient_id] = v;
  save(); updateUI();
  // Auto-advance
  if (idx < CASES.length - 1) {{ idx++; show(); }}
}}

function exportJSON() {{
  const out = {{}};
  for (const c of CASES) {{
    const v = verdicts[c.patient_id];
    if (v) out[c.patient_id] = {{ patient_id: c.patient_id, is_bipd: v === 'yes', verdict: v }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{type:'application/json'}});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'bipd_screening_results.json'; a.click();
  document.getElementById('save-status').textContent = 'Exported '+Object.keys(out).length;
  setTimeout(()=>{{ document.getElementById('save-status').textContent=''; }}, 3000);
}}

document.addEventListener('keydown', (e) => {{
  if (e.key==='y'||e.key==='Y') setVerdict('yes');
  else if (e.key==='n'||e.key==='N') setVerdict('no');
  else if (e.key==='ArrowLeft') {{ e.preventDefault(); idx=Math.max(0,idx-1); show(); }}
  else if (e.key==='ArrowRight') {{ e.preventDefault(); idx=Math.min(CASES.length-1,idx+1); show(); }}
  else if (e.key==='e'||e.key==='E') exportJSON();
}});

show();
</script>
</body>
</html>"""

    out_path = OUT_DIR / 'bipd_screener.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


if __name__ == '__main__':
    main()
