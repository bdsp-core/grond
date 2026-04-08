#!/usr/bin/env python3
"""
Auto-generate Table 3 (Lateralization Performance) from primary data.

PD lateralization AUC: computed from per-channel predictions in predictions.json
  vs laterality ground truth in segment_labels.csv.
RDA lateralization: from V5 contest result JSONs.

Usage:
    conda run -n morgoth python paper_materials/tables/generate_table3.py
"""

import csv
import json
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
CONTEST_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v4'
PREDICTIONS_PATH = LABELS_DIR / 'predictions.json'
COMPARISON_PATH = PROJECT_DIR / 'paper_materials' / 'method_comparison_table.json'
OUTPUT_PATH = Path(__file__).resolve().parent / 'table3_lateralization.md'

LEFT_CH = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CH = [4, 5, 6, 7, 12, 13, 14, 15]


def compute_pd_lateralization_auc():
    """Compute PD lateralization AUC from predictions.json + segment_labels.csv.

    For each LPD segment with a laterality label, compare:
      - predicted: mean(channel_probs[left]) vs mean(channel_probs[right])
      - ground truth: laterality label (left/right)
    """
    if not PREDICTIONS_PATH.exists():
        return None, None, 0

    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    # Load ground truth laterality for LPD segments
    gt_labels = {}  # mat_file → 'left' or 'right'
    subtypes = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get('subtype', '').lower()
            if sub != 'lpd':
                continue
            if row.get('excluded', '').lower() in ('true', '1', 'yes'):
                continue
            lat = row.get('laterality', '').strip().lower()
            if lat in ('left', 'right'):
                gt_labels[row['mat_file']] = lat
                subtypes[row['mat_file']] = sub

    # Compute laterality scores
    scores = []  # (pred_left_score, gt_is_left)
    for mat_file, gt_lat in gt_labels.items():
        pred = predictions.get(mat_file, {})
        probs = pred.get('channel_probs')
        if probs is None or len(probs) != 18:
            continue

        probs = np.array(probs, dtype=float)
        if np.any(np.isnan(probs)):
            continue  # corrupt EEG file
        left_mean = float(np.mean(probs[LEFT_CH]))
        right_mean = float(np.mean(probs[RIGHT_CH]))
        # Score: how "left" is this segment (higher = more left)
        lat_score = left_mean - right_mean
        gt_is_left = 1 if gt_lat == 'left' else 0
        scores.append((lat_score, gt_is_left))

    if len(scores) < 10:
        return None, None, len(scores)

    # Compute AUC
    from sklearn.metrics import roc_auc_score
    lat_scores = np.array([s[0] for s in scores])
    gt_labels_binary = np.array([s[1] for s in scores])
    auc = roc_auc_score(gt_labels_binary, lat_scores)

    return auc, len(scores), len(scores)


def compute_lpd_vs_gpd_auc():
    """Compute LPD vs GPD classification AUC from channel probabilities.

    Uses mean channel probability as the classifier (higher → more lateralized → LPD).
    """
    if not PREDICTIONS_PATH.exists():
        return None, 0

    with open(PREDICTIONS_PATH) as f:
        predictions = json.load(f)

    # Load subtype labels
    segments = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get('subtype', '').lower()
            if sub not in ('lpd', 'gpd'):
                continue
            if row.get('excluded', '').lower() in ('true', '1', 'yes'):
                continue
            segments[row['mat_file']] = sub

    scores = []
    for mat_file, subtype in segments.items():
        pred = predictions.get(mat_file, {})
        probs = pred.get('channel_probs')
        if probs is None or len(probs) != 18:
            continue

        probs = np.array(probs, dtype=float)
        if np.any(np.isnan(probs)):
            continue
        # Asymmetry score: LPD should be more asymmetric than GPD
        left_mean = float(np.mean(probs[LEFT_CH]))
        right_mean = float(np.mean(probs[RIGHT_CH]))
        asymmetry = abs(left_mean - right_mean)
        is_lpd = 1 if subtype == 'lpd' else 0
        scores.append((asymmetry, is_lpd))

    if len(scores) < 10:
        return None, len(scores)

    from sklearn.metrics import roc_auc_score
    asym_scores = np.array([s[0] for s in scores])
    gt = np.array([s[1] for s in scores])
    auc = roc_auc_score(gt, asym_scores)
    return auc, len(scores)


def load_rda_contest_results():
    """Load RDA lateralization contest results."""
    if not CONTEST_DIR.exists():
        return []
    methods = []
    for jf in sorted(CONTEST_DIR.glob('*.json')):
        try:
            with open(jf) as f:
                r = json.load(f)
            if isinstance(r, dict) and 'primary_auc' in r:
                methods.append({
                    'name': jf.stem,
                    'auc': r['primary_auc'],
                    'freq_rho': r.get('freq_rho'),
                    'n_total': r.get('n_total', 0),
                    'n_lrda': r.get('n_lrda', 0),
                    'n_grda': r.get('n_grda', 0),
                })
        except Exception:
            pass
    methods.sort(key=lambda x: x['auc'], reverse=True)
    return methods


def main():
    print("Generating Table 3 from predictions + labels...")

    lines = []
    lines.append("# Table 3: Lateralization Performance")
    lines.append("")
    lines.append("*Auto-generated from `predictions.json` + `segment_labels.csv` + contest results.*")
    lines.append("*AUC computed on demand from stored per-channel predictions.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table3.py`*")
    lines.append("")

    # PD lateralization
    lines.append("## PD Lateralization (ChannelPD-Net V1)")
    lines.append("")

    lat_auc, lat_n, _ = compute_pd_lateralization_auc()
    lpd_gpd_auc, lpd_gpd_n = compute_lpd_vs_gpd_auc()

    lines.append("| Metric | Value | N | Method |")
    lines.append("|---|---|---:|---|")

    if lat_auc is not None:
        lines.append(f"| Hemisphere AUC (L vs R) | **{lat_auc:.3f}** | {lat_n:,} | L vs R hemisphere mean PD probability |")
        print(f"  PD Hemisphere AUC: {lat_auc:.3f} (n={lat_n})")
    else:
        lines.append("| Hemisphere AUC (L vs R) | *run inference first* | — | L vs R hemisphere mean PD probability |")
        print("  PD Hemisphere AUC: predictions.json not found — run code/evaluation/run_all_inference.py")

    # LPD vs GPD classification from RF evaluation
    subtype_result_path = PROJECT_DIR / 'data' / 'evaluation_results' / 'subtype_classification.json'
    if subtype_result_path.exists():
        with open(subtype_result_path) as f:
            sub_result = json.load(f)
        sub_auc = sub_result.get('auc', '—')
        sub_n = sub_result.get('n_patients', '—')
        lines.append(f"| LPD vs GPD AUC | **{sub_auc}** | {sub_n:,} | {sub_result.get('method', 'RF 300 trees')} |")
        print(f"  LPD vs GPD AUC: {sub_auc} (n={sub_n})")
    else:
        lines.append("| LPD vs GPD AUC | *run eval_subtype_classification.py* | — | RF 300 trees |")

    # Production model metrics
    if Path(COMPARISON_PATH).exists():
        with open(COMPARISON_PATH) as f:
            comp = json.load(f)
        prod = comp.get('results', {}).get('HemiCET v2 + DP (C1)', {})
        if prod:
            lines.append(f"| Timing F1 (production) | {prod['f1']:.3f} | {prod.get('n', '—')} | HemiCET v2 + DP |")
            lines.append(f"| Frequency ρ (IPI) | {prod['freq_rho']:.3f} | {prod.get('n', '—')} | IPI-derived from detected discharges |")
    lines.append("")

    # 3-way LPD/GPD/BIPD classification
    threeway_path = PROJECT_DIR / 'data' / 'evaluation_results' / 'three_way_classification.json'
    if threeway_path.exists():
        with open(threeway_path) as f:
            tw = json.load(f)
        lines.append("## 3-Way Classification (LPD vs GPD vs BIPD)")
        lines.append("")
        lines.append("| Metric | Value | N | Method |")
        lines.append("|---|---|---:|---|")
        lines.append(f"| Macro AUC (3-way OVR) | **{tw['macro_auc']}** | {tw['n_patients']:,} | {tw['method']} |")
        per_auc = tw.get('per_class_auc', {})
        for cls in ['LPD', 'GPD', 'BIPD']:
            auc = per_auc.get(cls)
            lines.append(f"| {cls} AUC (OVR) | {auc} | — | |")
        lines.append(f"| BIPD vs GPD AUC | **{tw['bipd_vs_gpd_auc']}** | {tw['n_gpd'] + tw['n_bipd']:,} | Binary BIPD detection from GPD |")
        lines.append(f"| 3-way accuracy | {tw['accuracy']} | {tw['n_patients']:,} | |")
        lines.append("")
        lines.append(f"Dataset: {tw['n_lpd']:,} LPD + {tw['n_gpd']:,} GPD + {tw['n_bipd']} BIPD.")
        lines.append("")
        print(f"  3-way macro AUC: {tw['macro_auc']}, BIPD vs GPD: {tw['bipd_vs_gpd_auc']}")

    # RDA lateralization
    rda_methods = load_rda_contest_results()
    lines.append("## RDA Lateralization (LRDA vs GRDA) — V5 Contest")
    lines.append("")
    if rda_methods:
        n_total = rda_methods[0].get('n_total', 0)
        n_lrda = rda_methods[0].get('n_lrda', 0)
        n_grda = rda_methods[0].get('n_grda', 0)
        lines.append(f"Dataset: {n_lrda:,} LRDA + {n_grda:,} GRDA = {n_total:,} segments.")
        lines.append("")

        unified = [m for m in rda_methods if m['freq_rho'] and m['freq_rho'] > 0]
        if unified:
            unified.sort(key=lambda x: x['auc'], reverse=True)
            lines.append("### Top Unified Methods (lateralization + frequency)")
            lines.append("")
            lines.append("| Rank | Method | AUC | Freq ρ |")
            lines.append("|---|---|---|---|")
            for i, m in enumerate(unified[:10]):
                lines.append(f"| {i+1} | {m['name']} | {m['auc']:.3f} | {m['freq_rho']:.3f} |")
            lines.append("")

        lat_only = [m for m in rda_methods if not m['freq_rho'] or m['freq_rho'] <= 0]
        if lat_only:
            lat_only.sort(key=lambda x: x['auc'], reverse=True)
            lines.append("### Top Lateralization-Only Methods")
            lines.append("")
            lines.append("| Rank | Method | AUC |")
            lines.append("|---|---|---|")
            for i, m in enumerate(lat_only[:5]):
                lines.append(f"| {i+1} | {m['name']} | {m['auc']:.3f} |")
            lines.append("")

        lines.append(f"{len(rda_methods)} methods evaluated in V5 lateralization contest.")
    lines.append("")

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
