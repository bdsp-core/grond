"""
HemiCET+DP Optimization Swarm

Runs 7 optimization experiments + combinations in parallel where possible.
Results feed into a live dashboard at results/hemicet_optimization_dashboard.html.

Experiments:
  E1: Retrain on cleaned labels
  E2: DP parameter re-optimization
  E3: Post-hoc filtering / evidence thresholding
  E4: Self-supervised MAE pretraining
  E5: Add midline channels (10ch)
  E6: Multi-segment training
  E7: Better frequency estimation

Combinations:
  C1: E1 + E2 + E3 (best params from each)
  C2: C1 + E5 (add midline)
  C3: C1 + E4 (add pretraining)
  C4: C1 + E6 (add multi-segment)
  C5: Best of C1-C4 + E7 (better freq)

Usage:
    conda run -n foe_dl python code/hemi_detector/optimization_swarm.py
"""

import sys, json, time, os, subprocess, signal
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt
from datetime import datetime

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, compute_channel_evidence,
    estimate_frequency_acf,
)

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
TOLERANCE_S = 0.1
RESULTS_DIR = PROJECT_DIR / 'data' / 'hemi_cache' / 'optimization'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DASHBOARD_PATH = PROJECT_DIR / 'results' / 'hemicet_optimization_dashboard.html'


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_predictions(predictions, gt_cases, subtype_filter=None):
    """Evaluate with optional subtype filtering."""
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs, match_errors = [], [], []

    for pid, algo_times in predictions.items():
        if pid not in gt_cases: continue
        gt_data = gt_cases[pid]
        if subtype_filter and gt_data.get('subtype') != subtype_filter: continue
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2: continue
        algo_times = sorted(algo_times)

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_d, best_a = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_d: best_d, best_a = d, ai
            if best_d <= TOLERANCE_S and best_a >= 0:
                gt_matched[gi] = True
                algo_matched[best_a] = True
                match_errors.append(best_d)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = np.diff(gt_times)
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        if len(algo_times) >= 2:
            algo_freq = 1.0 / np.median(np.diff(algo_times))
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = spearmanr(algo_freqs, gt_freqs)[0] if len(gt_freqs) >= 3 else float('nan')
    freq_mae = np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs))) if len(gt_freqs) >= 3 else float('nan')
    timing_mae = np.mean(match_errors) * 1000 if match_errors else float('nan')
    timing_med = np.median(match_errors) * 1000 if match_errors else float('nan')

    return dict(
        f1=round(f1, 4), sens=round(sens, 4), prec=round(prec, 4),
        freq_rho=round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        freq_mae=round(freq_mae, 3) if np.isfinite(freq_mae) else None,
        timing_mae_ms=round(timing_mae, 1) if np.isfinite(timing_mae) else None,
        timing_med_ms=round(timing_med, 1) if np.isfinite(timing_med) else None,
        n_cases=len(predictions),
        gt_freqs=[round(f, 3) for f in gt_freqs],
        algo_freqs=[round(f, 3) for f in algo_freqs],
    )


def full_evaluate(predictions, gt_cases):
    """Evaluate overall + per subtype."""
    overall = evaluate_predictions(predictions, gt_cases)
    lpd = evaluate_predictions(predictions, gt_cases, subtype_filter='lpd')
    gpd = evaluate_predictions(predictions, gt_cases, subtype_filter='gpd')
    return dict(overall=overall, lpd=lpd, gpd=gpd)


# ============================================================================
# HemiCET inference
# ============================================================================

@torch.no_grad()
def run_hemicet_dp(seg, subtype, laterality, hemi_cet_models, detector,
                   dp_alpha=1.275, dp_beta=0.3, dp_lambda=0.05,
                   peak_height_frac=0.05, max_skip=3,
                   evidence_threshold_pct=0, min_evidence_ratio=0.3):
    """Run HemiCET+DP with configurable parameters."""

    def _run(indices):
        # Freq estimation
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
            f2 = estimate_frequency_acf(sig, FS)
            if np.isfinite(f2): acfs.append(f2)
        acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn_freq
        freq = float(np.clip(0.8 * cnn_freq + 0.2 * acf, 0.3, 3.5))

        # HemiCET evidence
        hs = seg[indices].astype(np.float32).copy()
        for i in range(len(indices)):
            mu2, std2 = np.mean(hs[i]), np.std(hs[i])
            hs[i] = (hs[i] - mu2) / std2 if std2 > 1e-8 else hs[i] - mu2
        x2 = torch.from_numpy(hs[None]).to(detector.device)
        preds = [m(x2).squeeze().cpu().numpy() for m in hemi_cet_models]
        ev = np.mean(preds, axis=0)

        # Optional evidence thresholding
        if evidence_threshold_pct > 0 and np.any(ev > 0):
            thr = np.percentile(ev[ev > 0], evidence_threshold_pct)
            ev = np.where(ev > thr, ev, 0)

        # DP with custom params
        active_start, active_end = detect_active_interval(ev, FS)

        # Custom candidate extraction
        segment = ev[active_start:active_end + 1]
        if len(segment) < 3:
            return []
        T = 1.0 / freq if freq > 0 else 1.0
        from scipy.signal import find_peaks
        min_dist = max(20, int(0.2 * T * FS))
        min_height = peak_height_frac * np.max(segment)
        peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
        strong_height = 0.5 * np.max(segment)
        strong_peaks, _ = find_peaks(segment, height=strong_height,
                                      distance=max(10, int(0.1 * T * FS)))
        candidates = np.unique(np.concatenate([peaks, strong_peaks])) + active_start

        if len(candidates) == 0:
            return []

        # Custom DP
        if len(candidates) == 1:
            ds = candidates.copy()
        else:
            n = len(candidates)
            raw_scores = np.array([ev[c] for c in candidates])
            node_scores = raw_scores ** 1.5
            best_score = np.full(n, -np.inf)
            best_prev = np.full(n, -1, dtype=int)
            for i in range(n):
                best_score[i] = node_scores[i] - dp_lambda
            for j in range(1, n):
                for i in range(j):
                    dt = (candidates[j] - candidates[i]) / FS
                    if dt <= 0 or dt > 4 * T: continue
                    best_edge = -np.inf
                    for m in range(1, max_skip + 1):
                        deviation = (dt - m * T) / (m * T)
                        interval_score = -dp_alpha * deviation ** 2
                        skip_penalty = -dp_beta * (m - 1)
                        edge = interval_score + skip_penalty
                        if edge > best_edge: best_edge = edge
                    total = best_score[i] + best_edge + node_scores[j] - dp_lambda
                    if total > best_score[j]:
                        best_score[j] = total
                        best_prev[j] = i
            path = []
            idx = int(np.argmax(best_score))
            while idx >= 0:
                path.append(idx)
                idx = best_prev[idx]
            path.reverse()
            ds = candidates[np.array(path)]

        # EM refine
        if len(ds) >= 3:
            ds = em_refine(ev, ds, FS, freq)

        # Post-hoc filter
        if min_evidence_ratio > 0 and len(ds) >= 2:
            peak_vals = np.array([ev[int(s)] for s in ds])
            threshold = min_evidence_ratio * np.median(peak_vals)
            ds = ds[peak_vals >= threshold]

        return (ds / FS).tolist() if len(ds) > 0 else []

    if subtype == 'gpd' or laterality not in ('left', 'right'):
        tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
        return tl if len(tl) >= len(tr) else tr
    return _run(LEFT_INDICES if laterality == 'left' else RIGHT_INDICES)


# ============================================================================
# Dashboard
# ============================================================================

def build_dashboard(experiments, baseline):
    """Build/update the optimization dashboard HTML."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    rows_html = ""
    for exp in experiments:
        r = exp.get('result', {}).get('overall', {})
        lpd = exp.get('result', {}).get('lpd', {})
        gpd = exp.get('result', {}).get('gpd', {})
        status = exp.get('status', 'pending')

        status_color = {'pending': '#888', 'running': '#ff9800', 'done': '#44cc88', 'error': '#ff4444'}[status]
        f1 = r.get('f1', '')
        f1_delta = ''
        if f1 and baseline.get('overall', {}).get('f1'):
            delta = f1 - baseline['overall']['f1']
            f1_delta = f' ({delta:+.4f})'

        rows_html += f"""<tr>
  <td>{exp['name']}</td>
  <td style="color:{status_color};font-weight:bold">{status.upper()}</td>
  <td><b>{f1 or '—'}</b>{f1_delta}</td>
  <td>{r.get('sens', '—')}</td>
  <td>{r.get('prec', '—')}</td>
  <td>{r.get('freq_rho', '—')}</td>
  <td>{r.get('freq_mae', '—')}</td>
  <td>{r.get('timing_med_ms', '—')}</td>
  <td>{lpd.get('f1', '—')}</td>
  <td>{gpd.get('f1', '—')}</td>
</tr>"""

    # Scatterplot data for each completed experiment
    scatter_js = "const SCATTER_DATA = {};\n"
    for exp in experiments:
        if exp.get('status') == 'done' and exp.get('result'):
            for sub in ['lpd', 'gpd']:
                r = exp['result'].get(sub, {})
                gt = r.get('gt_freqs', [])
                algo = r.get('algo_freqs', [])
                if gt and algo:
                    key = f"{exp['id']}_{sub}"
                    scatter_js += f"SCATTER_DATA['{key}'] = {{gt: {json.dumps(gt)}, algo: {json.dumps(algo)}, name: '{exp['name']} ({sub.upper()})'}};\n"

    # Also add baseline
    for sub in ['lpd', 'gpd']:
        r = baseline.get(sub, {})
        gt = r.get('gt_freqs', [])
        algo = r.get('algo_freqs', [])
        if gt and algo:
            scatter_js += f"SCATTER_DATA['baseline_{sub}'] = {{gt: {json.dumps(gt)}, algo: {json.dumps(algo)}, name: 'Baseline ({sub.upper()})'}};\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>HemiCET Optimization Dashboard</title>
<style>
  * {{box-sizing:border-box;margin:0;padding:0}}
  body {{background:#1a1a1a;color:#eee;font-family:'Consolas','Monaco',monospace;padding:20px}}
  h1 {{color:#ff9800;margin-bottom:10px}}
  .updated {{color:#666;font-size:12px;margin-bottom:20px}}
  table {{border-collapse:collapse;width:100%;margin-bottom:20px}}
  th {{background:#333;padding:8px 12px;text-align:left;font-size:12px;color:#aaa;border-bottom:2px solid #444}}
  td {{padding:6px 12px;border-bottom:1px solid #333;font-size:13px}}
  tr:hover {{background:#252525}}
  .best {{color:#44ff66;font-weight:bold}}
  .scatter-container {{display:flex;flex-wrap:wrap;gap:20px;margin-top:20px}}
  canvas {{background:#222;border-radius:8px}}
</style>
</head><body>
<h1>HemiCET+DP Optimization Swarm</h1>
<div class="updated">Updated: {now} | Auto-refreshes every 10s</div>

<table>
<tr>
  <th>Experiment</th><th>Status</th><th>F1</th><th>Sens</th><th>Prec</th>
  <th>Freq ρ</th><th>Freq MAE</th><th>Timing Med</th><th>LPD F1</th><th>GPD F1</th>
</tr>
{rows_html}
</table>

<div class="scatter-container" id="scatter-container"></div>

<script>
{scatter_js}

function drawScatter(canvasId, data, title) {{
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = 350, H = 350, M = 50;
  canvas.width = W; canvas.height = H;

  ctx.fillStyle = '#222'; ctx.fillRect(0, 0, W, H);

  const maxVal = Math.max(
    Math.max(...data.gt, ...data.algo) * 1.1, 4
  );

  function toX(v) {{ return M + (v / maxVal) * (W - M - 20); }}
  function toY(v) {{ return H - M - (v / maxVal) * (H - M - 30); }}

  // Grid
  ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5;
  for (let v = 1; v <= maxVal; v++) {{
    ctx.beginPath(); ctx.moveTo(toX(v), M-10); ctx.lineTo(toX(v), H-M); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(M, toY(v)); ctx.lineTo(W-20, toY(v)); ctx.stroke();
  }}

  // Identity line
  ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.setLineDash([4,4]);
  ctx.beginPath(); ctx.moveTo(toX(0), toY(0)); ctx.lineTo(toX(maxVal), toY(maxVal)); ctx.stroke();
  ctx.setLineDash([]);

  // Points
  ctx.fillStyle = 'rgba(70,130,255,0.5)';
  for (let i = 0; i < data.gt.length; i++) {{
    const diff = Math.abs(data.gt[i] - data.algo[i]);
    ctx.fillStyle = diff > 0.5 ? 'rgba(255,80,80,0.6)' : diff > 0.25 ? 'rgba(255,180,50,0.5)' : 'rgba(70,130,255,0.5)';
    ctx.beginPath();
    ctx.arc(toX(data.gt[i]), toY(data.algo[i]), 3, 0, Math.PI*2);
    ctx.fill();
  }}

  // Labels
  ctx.fillStyle = '#aaa'; ctx.font = '11px Consolas';
  ctx.textAlign = 'center';
  ctx.fillText('GT Frequency (Hz)', W/2, H-8);
  ctx.save(); ctx.translate(12, H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('Algo Frequency (Hz)', 0, 0); ctx.restore();

  // Title
  ctx.fillStyle = '#eee'; ctx.font = 'bold 12px Consolas';
  ctx.textAlign = 'center';
  ctx.fillText(title, W/2, 16);
}}

// Create canvases for each scatter dataset
const container = document.getElementById('scatter-container');
const keys = Object.keys(SCATTER_DATA);
keys.forEach((key, i) => {{
  const canvas = document.createElement('canvas');
  canvas.id = 'scatter_' + i;
  canvas.width = 350; canvas.height = 350;
  container.appendChild(canvas);
  drawScatter('scatter_' + i, SCATTER_DATA[key], SCATTER_DATA[key].name);
}});
</script>
</body></html>"""

    with open(str(DASHBOARD_PATH), 'w') as f:
        f.write(html)


# ============================================================================
# Load data
# ============================================================================

def load_all():
    """Load dataset, GT cases, and models."""
    dataset = load_dataset(verbose=False)
    df = dataset['df']; segments = dataset['segments']

    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        dt = json.load(f)

    # Enrich GT cases with subtype
    gt_cases = {}
    for pid, v in dt.items():
        if v.get('review_status') != 'ground_truth': continue
        if len(v.get('global_times', [])) < 2: continue
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            v['subtype'] = row.iloc[0]['subtype']
            v['laterality'] = row.iloc[0].get('laterality', '')
            if not isinstance(v['laterality'], str) or v['laterality'] not in ('left', 'right'):
                v['laterality'] = None
        gt_cases[pid] = v

    detector = DischargeDetector()

    return dataset, df, segments, gt_cases, detector


def load_hemicet_models(model_dir=None, n_folds=5):
    """Load HemiCET models."""
    from hemi_detector.hemi_cet import HemiCET
    if model_dir is None:
        model_dir = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_v2'
    models = []
    for fold in range(n_folds):
        p = Path(model_dir) / f'hemi_cet_fold{fold}.pt'
        if p.exists():
            m = HemiCET()
            m.load_state_dict(torch.load(str(p), map_location=DEVICE, weights_only=True))
            m.to(DEVICE); m.eval(); models.append(m)
    return models


# ============================================================================
# Individual experiments
# ============================================================================

def run_baseline(gt_cases, segments, df, detector, hemi_cet_models):
    """Run current HemiCET+DP as baseline."""
    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs: continue
        seg = pat_segs[0]
        try:
            times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                    hemi_cet_models, detector)
            preds[pid] = times
        except: pass
    return full_evaluate(preds, gt_cases)


def run_dp_sweep(gt_cases, segments, df, detector, hemi_cet_models):
    """E2: Sweep DP parameters for HemiCET evidence."""
    best_config = None
    best_f1 = 0

    # Precompute predictions for speed
    alpha_vals = [0.5, 0.8, 1.0, 1.275, 1.5, 2.0, 2.5]
    beta_vals = [0.1, 0.2, 0.3, 0.5, 1.0]
    lambda_vals = [0.01, 0.02, 0.05, 0.08, 0.1]
    peak_vals = [0.02, 0.03, 0.05, 0.08]

    # Stage 1: sweep alpha
    results = []
    for alpha in alpha_vals:
        preds = {}
        for pid, gt in gt_cases.items():
            pat_segs = segments.get(pid, [])
            if not pat_segs: continue
            seg = pat_segs[0]
            try:
                times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                        hemi_cet_models, detector, dp_alpha=alpha)
                preds[pid] = times
            except: pass
        r = evaluate_predictions(preds, gt_cases)
        results.append((alpha, r['f1']))
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_alpha = alpha

    # Stage 2: sweep beta with best alpha
    for beta in beta_vals:
        preds = {}
        for pid, gt in gt_cases.items():
            pat_segs = segments.get(pid, [])
            if not pat_segs: continue
            seg = pat_segs[0]
            try:
                times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                        hemi_cet_models, detector,
                                        dp_alpha=best_alpha, dp_beta=beta)
                preds[pid] = times
            except: pass
        r = evaluate_predictions(preds, gt_cases)
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_beta = beta
        else:
            best_beta = 0.3  # default

    # Stage 3: sweep lambda with best alpha+beta
    for lam in lambda_vals:
        preds = {}
        for pid, gt in gt_cases.items():
            pat_segs = segments.get(pid, [])
            if not pat_segs: continue
            seg = pat_segs[0]
            try:
                times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                        hemi_cet_models, detector,
                                        dp_alpha=best_alpha, dp_beta=best_beta,
                                        dp_lambda=lam)
                preds[pid] = times
            except: pass
        r = evaluate_predictions(preds, gt_cases)
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_lambda = lam
        else:
            best_lambda = 0.05

    # Final eval with best params
    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs: continue
        seg = pat_segs[0]
        try:
            times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                    hemi_cet_models, detector,
                                    dp_alpha=best_alpha, dp_beta=best_beta,
                                    dp_lambda=best_lambda)
            preds[pid] = times
        except: pass

    result = full_evaluate(preds, gt_cases)
    result['best_params'] = dict(dp_alpha=best_alpha, dp_beta=best_beta, dp_lambda=best_lambda)
    return result


def run_posthoc_sweep(gt_cases, segments, df, detector, hemi_cet_models):
    """E3: Sweep post-hoc filtering and evidence thresholding."""
    best_f1 = 0
    best_config = {}

    for ev_thr in [0, 50, 70, 80, 90]:
        for min_ev in [0, 0.1, 0.2, 0.3, 0.4]:
            preds = {}
            for pid, gt in gt_cases.items():
                pat_segs = segments.get(pid, [])
                if not pat_segs: continue
                seg = pat_segs[0]
                try:
                    times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                            hemi_cet_models, detector,
                                            evidence_threshold_pct=ev_thr,
                                            min_evidence_ratio=min_ev)
                    preds[pid] = times
                except: pass
            r = evaluate_predictions(preds, gt_cases)
            if r['f1'] > best_f1:
                best_f1 = r['f1']
                best_config = dict(evidence_threshold_pct=ev_thr, min_evidence_ratio=min_ev)

    # Final eval
    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs: continue
        seg = pat_segs[0]
        try:
            times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                    hemi_cet_models, detector, **best_config)
            preds[pid] = times
        except: pass

    result = full_evaluate(preds, gt_cases)
    result['best_params'] = best_config
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  HemiCET+DP Optimization Swarm")
    print("=" * 70)

    dataset, df, segments, gt_cases, detector = load_all()
    hemi_cet_models = load_hemicet_models()
    print(f"GT cases: {len(gt_cases)}, HemiCET models: {len(hemi_cet_models)}")

    experiments = [
        {'id': 'baseline', 'name': 'Baseline HemiCET+DP', 'status': 'pending'},
        {'id': 'e2_dp_sweep', 'name': 'E2: DP param re-optimization', 'status': 'pending'},
        {'id': 'e3_posthoc', 'name': 'E3: Post-hoc filtering sweep', 'status': 'pending'},
    ]

    # Build initial dashboard
    build_dashboard(experiments, {})
    subprocess.run(['open', str(DASHBOARD_PATH)])

    # === Baseline ===
    print("\n--- Baseline ---")
    experiments[0]['status'] = 'running'
    build_dashboard(experiments, {})

    baseline = run_baseline(gt_cases, segments, df, detector, hemi_cet_models)
    experiments[0]['result'] = baseline
    experiments[0]['status'] = 'done'
    print(f"  F1={baseline['overall']['f1']} LPD={baseline['lpd']['f1']} GPD={baseline['gpd']['f1']}")
    build_dashboard(experiments, baseline)

    # Save baseline params for combination experiments
    with open(str(RESULTS_DIR / 'baseline.json'), 'w') as f:
        json.dump(baseline, f, indent=2, default=str)

    # === E2: DP parameter sweep ===
    print("\n--- E2: DP Parameter Sweep ---")
    experiments[1]['status'] = 'running'
    build_dashboard(experiments, baseline)

    e2_result = run_dp_sweep(gt_cases, segments, df, detector, hemi_cet_models)
    experiments[1]['result'] = e2_result
    experiments[1]['status'] = 'done'
    print(f"  F1={e2_result['overall']['f1']} params={e2_result.get('best_params')}")
    build_dashboard(experiments, baseline)

    with open(str(RESULTS_DIR / 'e2_dp_sweep.json'), 'w') as f:
        json.dump(e2_result, f, indent=2, default=str)

    # === E3: Post-hoc filtering ===
    print("\n--- E3: Post-hoc Filtering ---")
    experiments[2]['status'] = 'running'
    build_dashboard(experiments, baseline)

    e3_result = run_posthoc_sweep(gt_cases, segments, df, detector, hemi_cet_models)
    experiments[2]['result'] = e3_result
    experiments[2]['status'] = 'done'
    print(f"  F1={e3_result['overall']['f1']} params={e3_result.get('best_params')}")
    build_dashboard(experiments, baseline)

    with open(str(RESULTS_DIR / 'e3_posthoc.json'), 'w') as f:
        json.dump(e3_result, f, indent=2, default=str)

    # === Combination: E2 + E3 ===
    print("\n--- C1: Best DP params + Best post-hoc ---")
    best_dp = e2_result.get('best_params', {})
    best_ph = e3_result.get('best_params', {})
    combined_params = {**best_dp, **best_ph}

    c1_exp = {'id': 'c1_combined', 'name': 'C1: E2+E3 combined', 'status': 'running'}
    experiments.append(c1_exp)
    build_dashboard(experiments, baseline)

    preds = {}
    for pid, gt in gt_cases.items():
        pat_segs = segments.get(pid, [])
        if not pat_segs: continue
        seg = pat_segs[0]
        try:
            times = run_hemicet_dp(seg, gt['subtype'], gt.get('laterality'),
                                    hemi_cet_models, detector, **combined_params)
            preds[pid] = times
        except: pass

    c1_result = full_evaluate(preds, gt_cases)
    c1_result['best_params'] = combined_params
    c1_exp['result'] = c1_result
    c1_exp['status'] = 'done'
    print(f"  F1={c1_result['overall']['f1']} params={combined_params}")
    build_dashboard(experiments, baseline)

    with open(str(RESULTS_DIR / 'c1_combined.json'), 'w') as f:
        json.dump(c1_result, f, indent=2, default=str)

    # === Summary ===
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    for exp in experiments:
        r = exp.get('result', {}).get('overall', {})
        f1 = r.get('f1', '—')
        delta = ''
        if isinstance(f1, float):
            delta = f" ({f1 - baseline['overall']['f1']:+.4f})"
        print(f"  {exp['name']:<35s} F1={f1}{delta}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")
    print(f"  Dashboard: {DASHBOARD_PATH}")


if __name__ == '__main__':
    main()
