"""
Generate 3-panel scatter plot comparing gold standard frequency vs three
EEG-only frequency estimation methods for all LPD+GPD patients.

Panel 1: Alexandra's method (FFT peak on pointiness trace)
Panel 2: CNN+Attention direct frequency estimate
Panel 3: IPI from HPP+CNN_freq (best deployable system)

Output: results/freq_estimation_comparison.html

Usage:
    conda run -n foe_dl python code/generate_freq_comparison_scatter.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, FS
from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from cet_model.auto_pipeline import (
    detect_discharges_auto, load_cnn_attn_models, estimate_frequency_cnn,
    DEVICE,
)

RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Frequency Estimation Comparison: 3 EEG-Only Methods")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # ── Load dataset ──────────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']
    features = dataset['features']

    # ── Load CNN+Attention models (needed for panels 2 and 3) ─────────
    print("\nLoading CNN+Attention models...")
    cnn_models = load_cnn_attn_models(device=DEVICE)
    print(f"  Loaded {len(cnn_models)} fold models on {DEVICE}")

    # ── Iterate over all patients ─────────────────────────────────────
    n_patients = len(df)
    print(f"\nProcessing {n_patients} patients...")

    gold_freqs = []
    subtypes = []
    fft_freqs = []       # Panel 1
    cnn_freqs = []       # Panel 2
    ipi_freqs = []       # Panel 3

    for i, (_, row) in enumerate(df.iterrows()):
        pid = str(row['patient_id'])
        gold = float(row['gold_standard_freq'])
        subtype = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        pat_segs = segments.get(pid, [])
        pat_feats = features.get(pid, [])
        if not pat_segs:
            gold_freqs.append(gold)
            subtypes.append(subtype)
            fft_freqs.append(np.nan)
            cnn_freqs.append(np.nan)
            ipi_freqs.append(np.nan)
            continue

        # ── Panel 1: FFT peak on pointiness trace ────────────────────
        seg_fft_vals = []
        for feat_dict in pat_feats:
            f = feat_dict.get('f_fft', np.nan)
            if np.isfinite(f):
                seg_fft_vals.append(f)
        panel1_freq = float(np.mean(seg_fft_vals)) if seg_fft_vals else np.nan

        # ── Panel 2: CNN+Attention direct frequency ──────────────────
        seg_cnn_vals = []
        for seg in pat_segs:
            try:
                f = estimate_frequency_cnn(seg, cnn_models, device=DEVICE, fs=FS)
                seg_cnn_vals.append(f)
            except Exception:
                pass
        panel2_freq = float(np.mean(seg_cnn_vals)) if seg_cnn_vals else np.nan

        # ── Panel 3: IPI from HPP + CNN freq ─────────────────────────
        seg_ipi_vals = []
        for seg in pat_segs:
            try:
                result = detect_discharges_auto(
                    seg, subtype=subtype, laterality=lat,
                    evidence_type='hpp', freq_method='cnn',
                    cnn_models=cnn_models, fs=FS, refine=True)
                f = result.get('frequency', np.nan)
                if np.isfinite(f) and f > 0:
                    seg_ipi_vals.append(f)
            except Exception:
                pass
        panel3_freq = float(np.mean(seg_ipi_vals)) if seg_ipi_vals else np.nan

        gold_freqs.append(gold)
        subtypes.append(subtype)
        fft_freqs.append(panel1_freq)
        cnn_freqs.append(panel2_freq)
        ipi_freqs.append(panel3_freq)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_patients} patients ({elapsed:.0f}s)")

    print(f"\nAll {n_patients} patients processed in {time.time()-t0:.0f}s")

    # ── Convert to arrays ─────────────────────────────────────────────
    gold_freqs = np.array(gold_freqs)
    fft_freqs = np.array(fft_freqs)
    cnn_freqs = np.array(cnn_freqs)
    ipi_freqs = np.array(ipi_freqs)
    subtypes = np.array(subtypes)

    # ── Compute stats per panel ───────────────────────────────────────
    panels = [
        ("FFT Peak on Pointiness Trace<br>(Alexandra's method)", fft_freqs),
        ("CNN+Attention Direct Frequency<br>(Phase 3b ensemble)", cnn_freqs),
        ("IPI from HPP+CNN_freq<br>(best deployable system)", ipi_freqs),
    ]

    panel_stats = []
    for name, pred in panels:
        mask = np.isfinite(gold_freqs) & np.isfinite(pred)
        n_valid = int(mask.sum())
        if n_valid >= 3:
            rho, _ = spearmanr(gold_freqs[mask], pred[mask])
            mae = float(np.mean(np.abs(gold_freqs[mask] - pred[mask])))
        else:
            rho, mae = np.nan, np.nan
        panel_stats.append({'name': name, 'n': n_valid, 'rho': rho, 'mae': mae})
        print(f"  {name[:40]:40s}  N={n_valid}  rho={rho:.3f}  MAE={mae:.3f}")

    # ── Build HTML ────────────────────────────────────────────────────
    print("\nGenerating HTML...")
    html = _build_html(gold_freqs, subtypes, panels, panel_stats)
    out_path = RESULTS_DIR / 'freq_estimation_comparison.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"Saved: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


def _build_html(gold, subtypes, panels, stats):
    """Build a 3-panel scatter plot HTML page."""

    # Prepare JSON data for each panel
    panel_json_blocks = []
    for (name, pred), st in zip(panels, stats):
        points = []
        for j in range(len(gold)):
            if np.isfinite(gold[j]) and np.isfinite(pred[j]):
                points.append({
                    'x': round(float(gold[j]), 4),
                    'y': round(float(pred[j]), 4),
                    'sub': subtypes[j],
                })
        panel_json_blocks.append({
            'name': st['name'],
            'n': st['n'],
            'rho': round(st['rho'], 3) if np.isfinite(st['rho']) else None,
            'mae': round(st['mae'], 3) if np.isfinite(st['mae']) else None,
            'points': points,
        })

    panels_json = json.dumps(panel_json_blocks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Frequency Estimation Comparison</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
       margin: 20px; background: #fafafa; color: #222; }}
h1 {{ text-align: center; margin-bottom: 5px; }}
.subtitle {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
.panels {{ display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; }}
.panel {{ background: #fff; border: 1px solid #ddd; border-radius: 6px;
          padding: 15px; text-align: center; }}
.panel canvas {{ display: block; margin: 0 auto; }}
.panel-title {{ font-size: 13px; font-weight: 600; margin-bottom: 4px; line-height: 1.3; }}
.panel-stats {{ font-size: 12px; color: #555; margin-top: 6px; }}
.legend {{ text-align: center; margin-top: 16px; font-size: 13px; }}
.legend span {{ margin: 0 12px; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>
<h1>Gold Standard Frequency vs EEG-Only Estimates</h1>
<div class="subtitle">All LPD + GPD patients with gold standard frequency</div>
<div class="panels" id="panels"></div>
<div class="legend">
  <span><span class="dot" style="background:#2ca02c;"></span>LPD</span>
  <span><span class="dot" style="background:#1f77b4;"></span>GPD</span>
  <span style="color:#888;">--- y = x</span>
</div>
<script>
const PANELS = {panels_json};
const SIZE = 380;
const PAD = 50;
const XMAX = 4.5;
const YMAX = 4.5;

function draw(panel, idx) {{
    const container = document.getElementById('panels');
    const div = document.createElement('div');
    div.className = 'panel';

    const titleDiv = document.createElement('div');
    titleDiv.className = 'panel-title';
    titleDiv.innerHTML = panel.name;
    div.appendChild(titleDiv);

    const canvas = document.createElement('canvas');
    canvas.width = SIZE + PAD + 20;
    canvas.height = SIZE + PAD + 20;
    div.appendChild(canvas);

    const statsDiv = document.createElement('div');
    statsDiv.className = 'panel-stats';
    const rhoStr = panel.rho !== null ? panel.rho.toFixed(3) : 'N/A';
    const maeStr = panel.mae !== null ? panel.mae.toFixed(3) : 'N/A';
    statsDiv.innerHTML = 'Spearman &rho; = ' + rhoStr + ' &nbsp;|&nbsp; MAE = ' + maeStr + ' Hz &nbsp;|&nbsp; N = ' + panel.n;
    div.appendChild(statsDiv);

    container.appendChild(div);

    const ctx = canvas.getContext('2d');
    const left = PAD;
    const bottom = SIZE + 5;
    const top = 5;
    const right = SIZE + 5;
    const plotW = right - left;
    const plotH = bottom - top;

    function toX(v) {{ return left + (v / XMAX) * plotW; }}
    function toY(v) {{ return bottom - (v / YMAX) * plotH; }}

    // Axes
    ctx.strokeStyle = '#999';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left, top); ctx.lineTo(left, bottom); ctx.lineTo(right, bottom);
    ctx.stroke();

    // Ticks + labels
    ctx.fillStyle = '#555';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    for (let v = 0; v <= XMAX; v += 1) {{
        const x = toX(v);
        ctx.beginPath(); ctx.moveTo(x, bottom); ctx.lineTo(x, bottom + 5); ctx.stroke();
        ctx.fillText(v.toString(), x, bottom + 16);
    }}
    ctx.textAlign = 'right';
    for (let v = 0; v <= YMAX; v += 1) {{
        const y = toY(v);
        ctx.beginPath(); ctx.moveTo(left - 5, y); ctx.lineTo(left, y); ctx.stroke();
        ctx.fillText(v.toString(), left - 8, y + 4);
    }}

    // Axis labels
    ctx.save();
    ctx.textAlign = 'center';
    ctx.fillStyle = '#333';
    ctx.font = '12px sans-serif';
    ctx.fillText('Gold Standard Frequency (Hz)', (left + right) / 2, bottom + 32);
    ctx.save();
    ctx.translate(14, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Predicted Frequency (Hz)', 0, 0);
    ctx.restore();
    ctx.restore();

    // y=x dashed line
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = '#aaa';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(toX(0), toY(0));
    ctx.lineTo(toX(XMAX), toY(YMAX));
    ctx.stroke();
    ctx.setLineDash([]);

    // Dots
    for (const pt of panel.points) {{
        const cx = toX(pt.x);
        const cy = toY(pt.y);
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = pt.sub === 'lpd' ? '#2ca02c' : '#1f77b4';
        ctx.beginPath();
        ctx.arc(cx, cy, 3.5, 0, 2 * Math.PI);
        ctx.fill();
    }}
    ctx.globalAlpha = 1.0;
}}

PANELS.forEach((p, i) => draw(p, i));
</script>
</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
