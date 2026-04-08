#!/usr/bin/env python3
"""
Auto-generate Table 5 (Frequency Estimation Performance) from primary data.

Replicates the EXACT quality filtering used in generate_fig6.py:
  - MW reviewed, OR
  - LB+PH+SZ consensus, OR
  - IIIC ≥10 votes with ≥80% agreement

Expert frequency = mean across all raters in annotations.csv.

Usage:
    conda run -n morgoth python paper_materials/tables/generate_table5.py
"""

import csv
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
CONTEST_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v4'
OUTPUT_PATH = Path(__file__).resolve().parent / 'table5_frequency.md'

SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']


def load_data_and_filter():
    """Load expert freq + model predictions, apply quality filter.

    Same logic as generate_fig6.py for consistency.
    """
    ann = pd.read_csv(LABELS_DIR / 'annotations.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    # Expert frequency: mean across all raters in annotations.csv
    has_freq = ann[ann.frequency_hz.notna()].copy()
    has_freq['frequency_hz'] = pd.to_numeric(has_freq['frequency_hz'], errors='coerce')
    freq_agg = has_freq.groupby('segment_id').agg(
        mean_freq=('frequency_hz', 'mean'),
        n_raters=('rater', 'nunique'),
    ).reset_index()
    expert_freq = dict(zip(freq_agg.segment_id, zip(freq_agg.mean_freq, freq_agg.n_raters)))

    # Also MW-only from segment_labels
    for _, row in sl[sl.expert_freq_rater == 'MW'].iterrows():
        sid = row['mat_file'].replace('.mat', '')
        if sid not in expert_freq and pd.notna(row.get('expert_freq_hz')):
            expert_freq[sid] = (float(row['expert_freq_hz']), 1)

    # Per-segment rater sets (for quality filter)
    freq_raters = has_freq.groupby('segment_id').agg(
        raters=('rater', lambda x: set(x)),
    ).reset_index()
    rater_info = dict(zip(freq_raters.segment_id, freq_raters.raters))

    # IIIC vote info
    vote_info = {}
    for _, row in sl.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        nv = pd.to_numeric(row.get('iiic_n_votes'), errors='coerce')
        pf = pd.to_numeric(row.get('iiic_plurality_frac'), errors='coerce')
        if np.isfinite(nv):
            vote_info[sid] = (int(nv), float(pf) if np.isfinite(pf) else 0)

    def passes_quality(sid):
        raters = rater_info.get(sid, set())
        if 'MW' in raters:
            return True
        if {'LB', 'PH', 'SZ'}.issubset(raters):
            return True
        nv, pf = vote_info.get(sid, (0, 0))
        if nv >= 10 and pf >= 0.80:
            return True
        return False

    # Build results
    results = {sub: [] for sub in SUBTYPES}
    for _, row in sl.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        subtype = row.get('subtype')
        if subtype not in results:
            continue
        if row.get('excluded') == True:
            continue
        pdchar = pd.to_numeric(row.get('pdchar_freq_hz'), errors='coerce')
        tautan = pd.to_numeric(row.get('tautan_freq_hz'), errors='coerce')
        if not np.isfinite(pdchar):
            continue
        if sid not in expert_freq:
            continue
        gt, n_raters = expert_freq[sid]
        if not np.isfinite(gt) or gt <= 0:
            continue

        results[subtype].append({
            'gt': float(gt),
            'pdchar': float(pdchar),
            'tautan': float(tautan) if np.isfinite(tautan) else np.nan,
            'passes_quality': passes_quality(sid),
        })

    return results


def compute_metrics(items, pred_key, quality_filtered=True):
    """Compute Spearman ρ and MAE."""
    if quality_filtered:
        items = [x for x in items if x['passes_quality']]
    gt = np.array([x['gt'] for x in items])
    pred = np.array([x[pred_key] for x in items])
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum() < 5:
        return None, None, 0
    rho, _ = spearmanr(gt[valid], pred[valid])
    mae = np.mean(np.abs(gt[valid] - pred[valid]))
    return rho, mae, int(valid.sum())


def main():
    print("Generating Table 5 from segment_labels.csv + annotations.csv...")
    data = load_data_and_filter()

    lines = []
    lines.append("# Table 5: Frequency Estimation Performance")
    lines.append("")
    lines.append("*Auto-generated from `segment_labels.csv` + `annotations.csv` with quality filtering.*")
    lines.append("*Quality filter: MW reviewed OR LB+PH+SZ consensus OR IIIC ≥10 votes with ≥80% agreement.*")
    lines.append("*Expert frequency = mean across raters. Same logic as generate_fig6.py.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table5.py`*")
    lines.append("")

    lines.append("## Per-Subtype Performance (Quality-Filtered)")
    lines.append("")
    lines.append("| Subtype | N | PDChar/W05 ρ | MAE (Hz) | N (Tautan) | Tautan ρ | Tautan MAE |")
    lines.append("|---|---:|---|---|---:|---|---|")

    for sub in SUBTYPES:
        rho_p, mae_p, n_p = compute_metrics(data[sub], 'pdchar', quality_filtered=True)
        rho_t, mae_t, n_t = compute_metrics(data[sub], 'tautan', quality_filtered=True)

        rho_p_s = f"**{rho_p:.3f}**" if rho_p is not None else "—"
        mae_p_s = f"{mae_p:.3f}" if mae_p is not None else "—"
        rho_t_s = f"{rho_t:.3f}" if rho_t is not None else "—"
        mae_t_s = f"{mae_t:.3f}" if mae_t is not None else "—"

        print(f"  {sub.upper()}: n={n_p}, PDChar ρ={rho_p_s}, Tautan n={n_t} ρ={rho_t_s}")
        lines.append(f"| {sub.upper()} | {n_p:,} | {rho_p_s} | {mae_p_s} | {n_t:,} | {rho_t_s} | {mae_t_s} |")

    lines.append("")

    # Also show unfiltered for reference
    lines.append("## Per-Subtype Performance (All Segments, Unfiltered)")
    lines.append("")
    lines.append("| Subtype | N | PDChar/W05 ρ | MAE (Hz) |")
    lines.append("|---|---:|---|---|")
    for sub in SUBTYPES:
        rho_p, mae_p, n_p = compute_metrics(data[sub], 'pdchar', quality_filtered=False)
        rho_p_s = f"{rho_p:.3f}" if rho_p is not None else "—"
        mae_p_s = f"{mae_p:.3f}" if mae_p is not None else "—"
        lines.append(f"| {sub.upper()} | {n_p:,} | {rho_p_s} | {mae_p_s} |")
    lines.append("")

    # RDA contest
    if CONTEST_DIR.exists():
        methods = []
        for jf in sorted(CONTEST_DIR.glob('*.json')):
            try:
                with open(jf) as f:
                    r = json.load(f)
                if isinstance(r, dict) and r.get('freq_rho') and r['freq_rho'] > 0:
                    methods.append({'name': jf.stem, 'auc': r['primary_auc'], 'freq_rho': r['freq_rho']})
            except Exception:
                pass
        if methods:
            methods.sort(key=lambda x: x['freq_rho'], reverse=True)
            lines.append("## RDA Frequency — Top Methods (V5 Contest)")
            lines.append("")
            lines.append("| Rank | Method | Lat AUC | Freq ρ |")
            lines.append("|---|---|---|---|")
            for i, m in enumerate(methods[:10]):
                lines.append(f"| {i+1} | {m['name']} | {m['auc']:.3f} | {m['freq_rho']:.3f} |")
            lines.append("")

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
