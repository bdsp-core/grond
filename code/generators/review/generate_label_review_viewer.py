"""
Generate HTML viewer for reviewing cases where HemiCET disagrees with GT labels.

Prioritizes high-yield cases:
1. Cases where HemiCET finds many more discharges than GT (possible label FNs)
2. Cases where HemiCET finds fewer discharges (possible label FPs)
3. Cases with large frequency disagreement
4. hpp_labeler source labels (less reviewed)

Shows EEG with:
- Red solid lines: GT discharge times (editable)
- Blue dashed lines: HemiCET predictions (read-only)
- U key: accept HemiCET predictions as GT
- A/D: add/delete GT markers manually
- C/Enter: accept current & advance

Usage:
    conda run -n foe_dl python code/generate_label_review_viewer.py
"""

import sys, json, numpy as np, time
from pathlib import Path
from scipy.signal import butter, filtfilt

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, combine_evidence, detect_active_interval,
    extract_candidates, dp_best_sequence, em_refine, posthoc_filter,
    compute_channel_evidence, estimate_frequency_acf,
)
from hemi_detector.hemi_cet import HemiCET

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
OUT_DIR = PROJECT_DIR / 'results'

BIPOLAR_CHANNELS = [
    'Fp1-F7','F7-T3','T3-T5','T5-O1',
    'Fp2-F8','F8-T4','T4-T6','T6-O2',
    'Fp1-F3','F3-C3','C3-P3','P3-O1',
    'Fp2-F4','F4-C4','C4-P4','P4-O2',
    'Fz-Cz','Cz-Pz',
]
DURATION = 10.0
LOWPASS_HZ = 20.0
TOLERANCE_S = 0.1


@torch.no_grad()
def run_hemi_cet(seg, subtype, laterality, detector, hemi_cet_models):
    """Run HemiCET + DP and return discharge times."""
    def _run(indices):
        # Freq
        all_pd, all_lf = [], []
        for ci in indices:
            ch = seg[ci].astype(np.float32).copy()
            if not np.all(np.isfinite(ch)):
                all_pd.append(0.0); all_lf.append(0.0); continue
            mu, std = np.mean(ch), np.std(ch)
            ch = (ch - mu) / std if std > 1e-8 else ch - mu
            x = torch.from_numpy(ch[None, None, :]).to(detector.device)
            pp, lf = [], []
            for m in detector.cnn_models:
                p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
            all_pd.append(np.mean(pp)); all_lf.append(np.mean(lf))
        pdw = np.array(all_pd); lfs = np.array(all_lf); ws = pdw.sum()
        cnn_freq = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
        b, a = butter(4, 20.0 / (FS / 2), btype='low')
        acfs = []
        for ci in indices:
            try: sig = filtfilt(b, a, seg[ci])
            except: sig = seg[ci]
            f = estimate_frequency_acf(sig, FS)
            if np.isfinite(f): acfs.append(f)
        acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn_freq
        freq = float(np.clip(0.8 * cnn_freq + 0.2 * acf, 0.3, 3.5))

        hs = seg[indices].astype(np.float32).copy()
        for i in range(len(indices)):
            mu, std = np.mean(hs[i]), np.std(hs[i])
            hs[i] = (hs[i] - mu) / std if std > 1e-8 else hs[i] - mu
        x = torch.from_numpy(hs[None]).to(detector.device)
        preds = [m(x).squeeze().cpu().numpy() for m in hemi_cet_models]
        ev = np.mean(preds, axis=0)
        a0, a1 = detect_active_interval(ev, FS)
        cands = extract_candidates(ev, FS, freq, a0, a1)
        ds = dp_best_sequence(cands, ev, FS, freq)
        if len(ds) >= 3: ds = em_refine(ev, ds, FS, freq)
        ds = posthoc_filter(ds, ev)
        return (ds / FS).tolist() if len(ds) > 0 else []

    if subtype == 'gpd' or laterality not in ('left', 'right'):
        tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
        return tl if len(tl) >= len(tr) else tr
    return _run(LEFT_INDICES if laterality == 'left' else RIGHT_INDICES)


def compute_disagreement(gt_times, algo_times):
    """Compute per-case disagreement metrics."""
    gt = sorted(gt_times)
    alg = sorted(algo_times)

    # Match
    gt_matched = [False] * len(gt)
    alg_matched = [False] * len(alg)
    for gi, g in enumerate(gt):
        best_d, best_a = np.inf, -1
        for ai, a in enumerate(alg):
            if not alg_matched[ai]:
                d = abs(g - a)
                if d < best_d: best_d, best_a = d, ai
        if best_d <= TOLERANCE_S and best_a >= 0:
            gt_matched[gi] = True
            alg_matched[best_a] = True

    tp = sum(gt_matched)
    fn = len(gt) - tp  # GT discharges missed by algo
    fp = len(alg) - sum(alg_matched)  # algo detections not in GT

    # Freq disagreement
    if len(gt) >= 2 and len(alg) >= 2:
        gt_freq = 1.0 / np.median(np.diff(gt))
        alg_freq = 1.0 / np.median(np.diff(alg))
        freq_diff = abs(gt_freq - alg_freq)
    else:
        freq_diff = 0

    # Priority score: higher = more valuable to review
    # Weight FPs heavily (algo found things GT missed)
    # Weight freq disagreement
    # Weight new labels (hpp_labeler source)
    score = fp * 2 + fn + freq_diff * 3

    return dict(tp=tp, fn=fn, fp=fp, freq_diff=round(freq_diff, 2),
                n_gt=len(gt), n_algo=len(alg), score=round(score, 1))


def downsample(arr, n):
    if arr.ndim == 1:
        if len(arr) <= n: return arr.tolist()
        return arr[np.linspace(0, len(arr)-1, n).astype(int)].tolist()
    if arr.shape[1] <= n: return arr.tolist()
    return arr[:, np.linspace(0, arr.shape[1]-1, n).astype(int)].tolist()


def main():
    print("=" * 70)
    print("  Label Review Viewer — HemiCET vs GT")
    print("=" * 70)

    dataset = load_dataset(verbose=False)
    df = dataset['df']; segments = dataset['segments']

    with open(str(LABELS_DIR / 'discharge_times.json')) as f:
        dt = json.load(f)
    gt_cases = {pid: v for pid, v in dt.items()
                if v.get('review_status') == 'ground_truth' and len(v.get('global_times', [])) >= 2}
    print(f"GT cases: {len(gt_cases)}")

    print("Loading models...")
    detector = DischargeDetector()
    hcm = []
    for fold in range(5):
        p = DATA_DIR / f'hemi_cache/hemi_cet/hemi_cet_fold{fold}.pt'
        if p.exists():
            m = HemiCET()
            m.load_state_dict(torch.load(str(p), map_location=detector.device, weights_only=True))
            m.to(detector.device); m.eval(); hcm.append(m)
    print(f"  HemiCET: {len(hcm)} models")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    # Run HemiCET on all cases and compute disagreement
    print(f"\nRunning HemiCET on {len(gt_cases)} cases...")
    cases_with_scores = []

    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        row = df[df['patient_id'] == pid]
        pat_segs = segments.get(pid, [])
        if not pat_segs or len(row) == 0: continue
        row = row.iloc[0]; seg = pat_segs[0]
        st = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'): lat = None

        try:
            algo_times = run_hemi_cet(seg, st, lat, detector, hcm)
        except:
            continue

        dis = compute_disagreement(gt_data['global_times'], algo_times)

        # Skip cases with no disagreement
        if dis['fp'] == 0 and dis['fn'] == 0:
            continue

        # Filter: only include cases with >0.5 Hz freq disagreement
        if dis['freq_diff'] < 0.5:
            continue

        # Boost score for hpp_labeler source (less reviewed)
        source = gt_data.get('source', 'original')
        if source == 'hpp_labeler':
            dis['score'] *= 1.5

        # Lowpass display
        seg_d = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try: seg_d[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except: seg_d[ch] = seg[ch]

        cases_with_scores.append({
            'patient_id': str(pid),
            'subtype': st,
            'laterality': lat or '',
            'source': source,
            'gold_freq': float(row.get('gold_standard_freq', 0)) if not np.isnan(row.get('gold_standard_freq', np.nan)) else 0,
            'gt_times': sorted(gt_data['global_times']),
            'algo_times': sorted(algo_times),
            'disagreement': dis,
            'eeg_data': downsample(seg_d[:18, :2000], 1000),
        })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(gt_cases)} ({len(cases_with_scores)} with disagreement)")

    # Sort by review priority (highest score first)
    cases_with_scores.sort(key=lambda c: c['disagreement']['score'], reverse=True)
    print(f"\nTotal cases with disagreement: {len(cases_with_scores)}")
    print(f"Top 5 by priority score:")
    for c in cases_with_scores[:5]:
        d = c['disagreement']
        print(f"  {c['patient_id']}: score={d['score']} FP={d['fp']} FN={d['fn']} freq_diff={d['freq_diff']} src={c['source']}")

    # Build HTML
    print("\nBuilding HTML...")
    html = build_html(cases_with_scores)
    out_path = OUT_DIR / 'label_review_viewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"Written to {out_path}")

    import subprocess
    subprocess.run(['open', str(out_path)])


def build_html(cases_data):
    cases_json = json.dumps(cases_data,
        default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Label Review: HemiCET vs GT</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#1a1a1a;color:#eee;font-family:'Consolas','Monaco',monospace}}
  #header{{display:flex;justify-content:space-between;align-items:center;padding:8px 16px;background:#222;border-bottom:2px solid #444;flex-wrap:wrap;gap:8px}}
  .key{{background:#444;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:bold}}
  #progress-bar-wrap{{width:100%;height:6px;background:#333}}
  #progress-bar{{height:100%;background:#44cc88;transition:width 0.2s}}
  #mode-indicator{{font-size:18px;font-weight:bold;padding:6px 20px;text-align:center;letter-spacing:2px}}
  .mode-add{{background:#1a3a1a;color:#44ff66}}
  .mode-delete{{background:#3a1a1a;color:#ff4444}}
  .mode-nav{{background:#1a1a3a;color:#6688ff}}
  #info{{background:#2a2a2a;padding:10px 16px;display:flex;gap:20px;flex-wrap:wrap;font-size:13px;border-bottom:1px solid #333}}
  .info-item{{color:#bbb}} .info-item strong{{color:#eee}}
  .dis-item{{padding:3px 8px;border-radius:4px;font-size:12px;font-weight:bold}}
  .dis-fp{{background:#3a1a1a;color:#ff6666}}
  .dis-fn{{background:#1a1a3a;color:#6688ff}}
  #eeg-canvas{{display:block;margin:8px auto;cursor:crosshair}}
  .export-btn{{padding:6px 14px;border:1px solid #44cc44;border-radius:4px;background:#2a3a2a;color:#44cc44;cursor:pointer;font-family:monospace;font-size:12px;font-weight:bold}}
  .accept-btn{{padding:6px 14px;border:1px solid #6a5acd;border-radius:4px;background:#2a2a3a;color:#6a5acd;cursor:pointer;font-family:monospace;font-size:12px;font-weight:bold}}
  #shortcuts{{font-size:12px;color:#777;padding:6px 16px;background:#222;border-top:1px solid #333;line-height:1.8}}
  #save-status{{color:#44cc44;font-size:13px}}
  #stats{{font-size:13px;color:#aaa;padding:4px 16px;background:#252525}}
</style></head><body>
<div id="header">
  <div style="display:flex;align-items:center;gap:12px">
    <span style="font-size:16px;font-weight:bold;color:#ff9800">Label Review: HemiCET vs GT</span>
    <span id="counter" style="font-size:13px;color:#aaa">1/0</span>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <button class="accept-btn" onclick="acceptAlgo()">Accept HemiCET <span class="key">U</span></button>
    <button style="padding:6px 14px;border:1px solid #ff6644;border-radius:4px;background:#3a2a2a;color:#ff6644;cursor:pointer;font-family:monospace;font-size:12px;font-weight:bold" onclick="rejectCase()">Not PDs <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px;color:#aaa"></span>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>
<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>
<div id="info">
  <span class="info-item">Patient: <strong id="i-pid">--</strong></span>
  <span class="info-item">Type: <strong id="i-type">--</strong></span>
  <span class="info-item">Source: <strong id="i-source">--</strong></span>
  <span class="info-item">GT: <strong id="i-ngt" style="color:#ff4444">--</strong></span>
  <span class="info-item">HemiCET: <strong id="i-nalgo" style="color:#6a5acd">--</strong></span>
  <span class="dis-item dis-fp" id="i-fp">FP: --</span>
  <span class="dis-item dis-fn" id="i-fn">FN: --</span>
  <span class="info-item">Freq diff: <strong id="i-fdiff">--</strong></span>
  <span class="info-item">Score: <strong id="i-score" style="color:#ff9800">--</strong></span>
</div>
<div id="stats"></div>
<div style="text-align:center;padding:8px"><canvas id="eeg-canvas"></canvas></div>
<div id="shortcuts">
  <span class="key">U</span> Accept HemiCET labels &nbsp;
  <span class="key">X</span> Reject (not PDs) &nbsp;
  <span class="key">A</span> Add mode &nbsp;
  <span class="key">D</span> Delete mode &nbsp;
  <span class="key">Esc</span> Navigate &nbsp;
  <span class="key">Z</span> Undo &nbsp;
  <span class="key">C</span>/<span class="key">Enter</span> Accept &amp; advance &nbsp;
  <span class="key">&larr;/&rarr;</span> Navigate &nbsp;
  <span class="key">E</span> Export
</div>
<script>
const CASES={cases_json};
const CH=['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2','Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const GB=new Set([4,8,12,16]);
const D=10,Z=0.01,CL=300,W=1200,H=700,ML=70,MR=20,MT=30,MB=25;
const PL=ML,PR=W-MR,PW=PR-PL,PH=H-MT-MB;
function getDC(){{const dc=[];for(let i=0;i<18;i++){{if(GB.has(i))dc.push({{idx:-1,name:''}});dc.push({{idx:i,name:CH[i]}})}}return dc}}
const DC=getDC(),ND=DC.length;
function tx(t){{return PL+(t/D)*PW}}
function xt(x){{return((x-PL)/PW)*D}}

let idx=0,mode='nav',markers=[],undoStack=[],hoverMarker=-1;
const SK='label_review_v1';
let allM={{}};
try{{allM=JSON.parse(localStorage.getItem(SK)||'{{}}')}}catch(e){{allM={{}}}}
function sv(){{localStorage.setItem(SK,JSON.stringify(allM))}}
let reviewed=new Set();

function findNearest(x){{let b=-1,bd=Infinity;for(let i=0;i<markers.length;i++){{const d=Math.abs(tx(markers[i])-x);if(d<bd){{bd=d;b=i}}}}return bd<=20?b:-1}}
function pushUndo(){{undoStack.push([...markers]);if(undoStack.length>100)undoStack.shift()}}

function draw(){{
  const cv=document.getElementById('eeg-canvas');cv.width=W;cv.height=H;
  const ctx=cv.getContext('2d'),c=CASES[idx],eeg=c.eeg_data,ns=eeg[0].length,sp=PH/(ND+1);
  ctx.fillStyle='#fff';ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#ddd';ctx.lineWidth=0.5;ctx.setLineDash([4,4]);
  for(let s=0;s<=10;s++){{const x=tx(s);ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke()}}
  ctx.setLineDash([]);
  for(let di=0;di<ND;di++){{const ch=DC[di];if(ch.idx<0)continue;const yc=MT+sp*(di+1);
    ctx.strokeStyle='#000';ctx.lineWidth=0.7;ctx.beginPath();
    for(let si=0;si<ns;si++){{const x=PL+(si/(ns-1))*PW,v=Math.max(-CL,Math.min(CL,eeg[ch.idx][si])),y=yc-v*Z*sp;
      si===0?ctx.moveTo(x,y):ctx.lineTo(x,y)}}ctx.stroke()}}
  ctx.font='10px Consolas';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillStyle='#000';
  for(let di=0;di<ND;di++){{const ch=DC[di];if(ch.idx<0)continue;ctx.fillText(ch.name,PL-4,MT+sp*(di+1))}}
  ctx.textAlign='center';ctx.textBaseline='top';
  for(let s=0;s<=10;s++)ctx.fillText(s+'s',tx(s),H-MB+4);
  ctx.font='bold 13px Consolas';ctx.textBaseline='top';
  ctx.fillText(c.patient_id+'  |  '+c.subtype.toUpperCase()+'  |  score='+c.disagreement.score,W/2,6);
  // HemiCET predictions (blue-purple dashed) — draw first
  ctx.strokeStyle='#6a5acd';ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
  for(const t of c.algo_times){{const x=tx(t);if(x<PL||x>PR)continue;ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke()}}
  ctx.setLineDash([]);
  // GT markers (red solid, editable)
  for(let mi=0;mi<markers.length;mi++){{const t=markers[mi],x=tx(t);if(x<PL||x>PR)continue;
    const isH=mode==='delete'&&hoverMarker===mi;
    ctx.strokeStyle=isH?'rgba(255,50,50,0.9)':'rgba(255,0,0,0.6)';ctx.lineWidth=isH?4:2;
    ctx.beginPath();ctx.moveTo(x,MT);ctx.lineTo(x,H-MB);ctx.stroke();
    ctx.fillStyle=ctx.strokeStyle;ctx.font='9px Consolas';ctx.textAlign='center';ctx.textBaseline='bottom';
    ctx.fillText(t.toFixed(2)+'s',x,MT-2)}}
}}

function updateUI(){{
  const c=CASES[idx],d=c.disagreement;
  document.getElementById('i-pid').textContent=c.patient_id;
  document.getElementById('i-type').textContent=c.subtype.toUpperCase();
  document.getElementById('i-source').textContent=c.source;
  document.getElementById('i-ngt').textContent=markers.length;
  document.getElementById('i-nalgo').textContent=c.algo_times.length;
  document.getElementById('i-fp').textContent='FP: '+d.fp;
  document.getElementById('i-fn').textContent='FN: '+d.fn;
  document.getElementById('i-fdiff').textContent=d.freq_diff.toFixed(2)+' Hz';
  document.getElementById('i-score').textContent=d.score;
  document.getElementById('counter').textContent=(idx+1)+'/'+CASES.length;
  document.getElementById('progress-bar').style.width=((idx+1)/CASES.length*100)+'%';
  document.getElementById('reviewed-count').textContent=reviewed.size+' reviewed';
  const el=document.getElementById('mode-indicator');
  if(mode==='add'){{el.textContent='ADD MODE (A)';el.className='mode-add'}}
  else if(mode==='delete'){{el.textContent='DELETE MODE (D)';el.className='mode-delete'}}
  else{{el.textContent='NAVIGATE MODE';el.className='mode-nav'}}
  // Stats
  let acc=0,rej=0,cet=0,notpd=0;
  for(const v of Object.values(allM)){{if(v.status==='accepted')acc++;if(v.status==='cet_accepted')cet++;if(v.status==='rejected_not_pd')notpd++;}}
  document.getElementById('stats').textContent='Reviewed: '+reviewed.size+' | Accepted: '+acc+' | CET: '+cet+' | Rejected: '+notpd;
}}

function redraw(){{draw();updateUI()}}

function autoSave(){{
  const c=CASES[idx];
  allM[c.patient_id]={{times:[...markers].sort((a,b)=>a-b),original:c.gt_times,algo:c.algo_times,
    status:allM[c.patient_id]?allM[c.patient_id].status:'unchanged'}};
  sv();
}}

function show(){{
  idx=Math.max(0,Math.min(idx,CASES.length-1));
  const c=CASES[idx];
  if(allM[c.patient_id]&&allM[c.patient_id].times){{markers=[...allM[c.patient_id].times]}}
  else{{markers=[...c.gt_times]}}
  undoStack=[];hoverMarker=-1;redraw()}}

function acceptAlgo(){{
  const c=CASES[idx];pushUndo();markers=[...c.algo_times];
  allM[c.patient_id]={{times:[...markers].sort((a,b)=>a-b),original:c.gt_times,algo:c.algo_times,status:'cet_accepted'}};
  reviewed.add(c.patient_id);sv();
  document.getElementById('save-status').textContent='HemiCET accepted';
  setTimeout(()=>document.getElementById('save-status').textContent='',1500);
  redraw()}}

function rejectCase(){{
  const c=CASES[idx];
  allM[c.patient_id]={{times:[],original:c.gt_times,algo:c.algo_times,status:'rejected_not_pd'}};
  reviewed.add(c.patient_id);sv();
  document.getElementById('save-status').textContent='REJECTED — not PDs';
  document.getElementById('save-status').style.color='#ff6644';
  setTimeout(()=>{{document.getElementById('save-status').textContent='';document.getElementById('save-status').style.color='#44cc44'}},1500);
  if(idx<CASES.length-1){{idx++;show()}}
  else redraw();
}}

function exportJSON(){{
  autoSave();
  const out={{}};for(const c of CASES){{const pid=c.patient_id;
    if(allM[pid])out[pid]={{patient_id:pid,updated_times:allM[pid].times,original_times:allM[pid].original,
      algo_times:allM[pid].algo,status:allM[pid].status}}}}
  const b=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='label_review_results.json';a.click();
  document.getElementById('save-status').textContent='Exported '+Object.keys(out).length;
}}

const cv=document.getElementById('eeg-canvas');
cv.addEventListener('click',(e)=>{{
  const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*(W/r.width);
  if(x<PL||x>PR)return;const t=xt(x);
  if(mode==='add'&&t>=0&&t<=D){{pushUndo();markers.push(t);
    allM[CASES[idx].patient_id]={{...allM[CASES[idx].patient_id],status:'edited'}};redraw()}}
  else if(mode==='delete'){{const mi=findNearest(x);if(mi>=0){{pushUndo();markers.splice(mi,1);
    allM[CASES[idx].patient_id]={{...allM[CASES[idx].patient_id],status:'edited'}};redraw()}}}}
}});
cv.addEventListener('mousemove',(e)=>{{if(mode!=='delete')return;
  const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)*(W/r.width),m=findNearest(x);
  if(m!==hoverMarker){{hoverMarker=m;redraw()}}
}});

document.addEventListener('keydown',(e)=>{{
  if(e.key==='x'||e.key==='X')rejectCase();
  else if(e.key==='u'||e.key==='U')acceptAlgo();
  else if(e.key==='a'||e.key==='A'){{mode=mode==='add'?'nav':'add';redraw()}}
  else if(e.key==='d'||e.key==='D'){{mode=mode==='delete'?'nav':'delete';redraw()}}
  else if(e.key==='Escape'){{mode='nav';redraw()}}
  else if(e.key==='z'||e.key==='Z'){{if(undoStack.length){{markers=undoStack.pop();redraw()}}}}
  else if(e.key==='ArrowLeft'){{e.preventDefault();autoSave();idx=Math.max(0,idx-1);show()}}
  else if(e.key==='ArrowRight'||e.key==='c'||e.key==='C'||e.key==='Enter'){{
    e.preventDefault();reviewed.add(CASES[idx].patient_id);
    if(!allM[CASES[idx].patient_id]||allM[CASES[idx].patient_id].status==='unchanged')
      allM[CASES[idx].patient_id]={{times:[...markers].sort((a,b)=>a-b),original:CASES[idx].gt_times,
        algo:CASES[idx].algo_times,status:'accepted'}};
    autoSave();idx=Math.min(CASES.length-1,idx+1);show()}}
  else if(e.key==='e'||e.key==='E')exportJSON();
}});
show();
</script></body></html>"""
    return html


if __name__ == '__main__':
    main()
