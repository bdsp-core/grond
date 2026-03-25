"""
Review rejected LPD cases: GPD or Not PD?

49 cases that were labeled "not LPD" during laterality review.
User categorizes each as: (G) GPD, or (N) Not PD.

Usage:
    conda run -n foe_dl python code/generate_rejected_lpd_reviewer.py
"""

import sys, json, numpy as np, scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
from optimization_harness_v2 import load_dataset, FS

DATA_DIR = PROJECT_DIR / 'data'
OUT_DIR = PROJECT_DIR / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7','F7-T3','T3-T5','T5-O1',
    'Fp2-F8','F8-T4','T4-T6','T6-O2',
    'Fp1-F3','F3-C3','C3-P3','P3-O1',
    'Fp2-F4','F4-C4','C4-P4','P4-O2',
    'Fz-Cz','Cz-Pz',
]

def downsample_2d(arr, n):
    if arr.shape[1] <= n: return arr.tolist()
    idx = np.linspace(0, arr.shape[1]-1, n).astype(int)
    return arr[:, idx].tolist()

def main():
    # Load rejected cases
    with open('/Users/mwestover/Downloads/laterality_labels_lpd.json') as f:
        all_labels = json.load(f)
    rejected = {pid: v for pid, v in all_labels.items() if v['not_lpd']}
    print(f"Rejected cases to review: {len(rejected)}")

    b_lp, a_lp = butter(4, 20.0/(FS/2), btype='low')

    cases = []
    for pid in sorted(rejected.keys()):
        # Load EEG directly (these are excluded from load_dataset)
        seg = None
        for suffix in ['_seg000.mat', '.mat']:
            path = DATA_DIR / 'eeg' / f'{pid}{suffix}'
            if path.exists():
                mat = sio.loadmat(str(path))
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                seg = mat[key]
                if seg.shape[0] > seg.shape[1]:
                    seg = seg.T
                seg = seg[:18, :2000]
                break
        if seg is None:
            continue
        seg_d = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try: seg_d[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except: seg_d[ch] = seg[ch]
        info = rejected[pid]
        cases.append({
            'patient_id': str(pid),
            'left_score': info['left_score'],
            'right_score': info['right_score'],
            'eeg_data': downsample_2d(seg_d[:18,:2000], 1000),
        })
    print(f"Loaded {len(cases)} cases with EEG")

    cases_json = json.dumps(cases, default=lambda o: float(o) if isinstance(o,(np.floating,)) else o)
    ch_json = json.dumps(BIPOLAR_CHANNELS)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Rejected LPD Review: GPD or Not PD</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a1a1a;color:#eee;font-family:'Consolas','Monaco',monospace}}
  #header{{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:#222;border-bottom:2px solid #444;flex-wrap:wrap;gap:8px}}
  .key{{background:#444;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold}}
  #progress-bar-wrap{{width:100%;height:6px;background:#333}}
  #progress-bar{{height:100%;background:#44cc88;transition:width 0.2s}}
  #verdict{{font-size:24px;font-weight:bold;padding:10px 20px;text-align:center;letter-spacing:3px}}
  .v-none{{background:#222;color:#888}}
  .v-gpd{{background:#20205a;color:#8888ff}}
  .v-lpd{{background:#1a3a1a;color:#44ff66}}
  .v-notpd{{background:#3a3a1a;color:#ffaa44}}
  #info{{background:#2a2a2a;padding:10px 16px;font-size:13px;border-bottom:1px solid #333}}
  #canvas-container{{text-align:center;padding:8px}}
  #eeg-canvas{{display:block;margin:0 auto}}
  .export-btn{{padding:6px 14px;border:1px solid #44cc44;border-radius:4px;background:#2a3a2a;color:#44cc44;cursor:pointer;font-family:monospace;font-size:12px;font-weight:bold}}
  #shortcuts{{font-size:12px;color:#777;padding:6px 16px;background:#222;border-top:1px solid #333;line-height:1.8}}
  #stats{{font-size:13px;color:#aaa;padding:4px 16px;background:#252525}}
</style></head><body>
<div id="header">
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:16px;font-weight:bold;color:#ffaa44">Rejected LPD Review: GPD or Not PD</span>
    <span id="counter" style="font-size:13px;color:#aaa">1/0</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status" style="color:#44cc44;font-size:13px"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
<div id="verdict" class="v-none">— not reviewed —</div>
<div id="info">Patient: <strong id="info-pid">--</strong> &nbsp; L=<strong id="info-l">--</strong> R=<strong id="info-r">--</strong></div>
<div id="stats"></div>
<div id="canvas-container"><canvas id="eeg-canvas"></canvas></div>
<div id="shortcuts">
  <span class="key">G</span> = GPD &nbsp;&nbsp;
  <span class="key">L</span> = LPD (keep as LPD, needs laterality) &nbsp;&nbsp;
  <span class="key">N</span> = Not PD (exclude) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">E</span> Export
</div>
<script>
const CASES={cases_json};
const CH={ch_json};
const D=10,Z=0.01,CL=300,W=1200,H=700;
const ML=70,MR=20,MT=30,MB=25,PL=ML,PR=W-MR,PW=PR-PL,PH=H-MT-MB;
const GB=new Set([4,8,12,16]);
let idx=0;
const SK='rejected_lpd_v1';
let vd={{}};
try{{vd=JSON.parse(localStorage.getItem(SK)||'{{}}')}}catch(e){{vd={{}}}}
function sv(){{localStorage.setItem(SK,JSON.stringify(vd))}}
function tx(t){{return PL+(t/D)*PW}}

function getDC(){{
  const dc=[];
  for(let i=0;i<18;i++){{if(GB.has(i))dc.push({{idx:-1,name:''}});dc.push({{idx:i,name:CH[i]}})}}
  return dc;
}}
const DC=getDC(),ND=DC.length;

function draw(){{
  const cv=document.getElementById('eeg-canvas');cv.width=W;cv.height=H;
  const ctx=cv.getContext('2d'),c=CASES[idx],eeg=c.eeg_data,ns=eeg[0].length,sp=PH/(ND+1);
  ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#ddd';ctx.lineWidth=0.5;ctx.setLineDash([4,4]);
  for(let s=0;s<=10;s++){{const x=tx(s);ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke()}}
  ctx.setLineDash([]);
  for(let di=0;di<ND;di++){{
    const ch=DC[di];if(ch.idx<0)continue;
    const yc=MT+sp*(di+1);
    ctx.strokeStyle='#000';ctx.lineWidth=0.7;ctx.beginPath();
    for(let si=0;si<ns;si++){{
      const x=PL+(si/(ns-1))*PW,v=Math.max(-CL,Math.min(CL,eeg[ch.idx][si])),y=yc-v*Z*sp;
      si===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }}ctx.stroke();
  }}
  ctx.fillStyle='#000';ctx.font='10px Consolas';ctx.textAlign='right';ctx.textBaseline='middle';
  for(let di=0;di<ND;di++){{const ch=DC[di];if(ch.idx<0)continue;ctx.fillText(ch.name,PL-4,MT+sp*(di+1))}}
  ctx.textAlign='center';ctx.textBaseline='top';
  for(let s=0;s<=10;s++)ctx.fillText(s+'s',tx(s),H-MB+4);
  ctx.font='bold 13px Consolas';ctx.textBaseline='top';
  ctx.fillText(c.patient_id,W/2,6);
}}

function stats(){{
  let g=0,n=0,l=0;for(const v of Object.values(vd)){{if(v==='gpd')g++;if(v==='not_pd')n++;if(v==='lpd')l++}}
  document.getElementById('stats').textContent='GPD: '+g+'  |  LPD: '+l+'  |  Not PD: '+n+'  |  Total: '+(g+n+l)+' / '+CASES.length;
}}

function ui(){{
  const c=CASES[idx],v=vd[c.patient_id];
  document.getElementById('info-pid').textContent=c.patient_id;
  document.getElementById('info-l').textContent=c.left_score.toFixed(3);
  document.getElementById('info-r').textContent=c.right_score.toFixed(3);
  document.getElementById('counter').textContent=(idx+1)+' / '+CASES.length;
  document.getElementById('progress-bar').style.width=((idx+1)/CASES.length*100).toFixed(1)+'%';
  const el=document.getElementById('verdict');
  if(v==='gpd'){{el.textContent='GPD';el.className='v-gpd'}}
  else if(v==='lpd'){{el.textContent='LPD (needs laterality)';el.className='v-lpd'}}
  else if(v==='not_pd'){{el.textContent='NOT PD';el.className='v-notpd'}}
  else{{el.textContent='— not reviewed —';el.className='v-none'}}
  stats();
}}

function show(){{idx=Math.max(0,Math.min(idx,CASES.length-1));draw();ui()}}
function set(v){{vd[CASES[idx].patient_id]=v;sv();ui();if(idx<CASES.length-1){{idx++;show()}}}}

function exportJSON(){{
  const out={{}};
  for(const c of CASES){{const v=vd[c.patient_id];if(v)out[c.patient_id]={{patient_id:c.patient_id,verdict:v}}}}
  const b=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='rejected_lpd_review.json';a.click();
  document.getElementById('save-status').textContent='Exported '+Object.keys(out).length;
}}

document.addEventListener('keydown',(e)=>{{
  if(e.key==='g'||e.key==='G')set('gpd');
  else if(e.key==='l'||e.key==='L')set('lpd');
  else if(e.key==='n'||e.key==='N')set('not_pd');
  else if(e.key==='ArrowLeft'){{e.preventDefault();idx=Math.max(0,idx-1);show()}}
  else if(e.key==='ArrowRight'){{e.preventDefault();idx=Math.min(CASES.length-1,idx+1);show()}}
  else if(e.key==='e'||e.key==='E')exportJSON();
}});
show();
</script></body></html>"""

    out_path = OUT_DIR / 'rejected_lpd_reviewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"Written to {out_path}")
    import subprocess
    subprocess.run(['open', str(out_path)])

if __name__ == '__main__':
    main()
