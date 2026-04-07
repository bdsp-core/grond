#!/usr/bin/env python3
"""
Auto-generate Table 3 (Lateralization Performance) from primary data.

PD lateralization: computed from segment_labels.csv (ChannelPD-Net probs).
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
OUTPUT_PATH = Path(__file__).resolve().parent / 'table3_lateralization.md'


def load_rda_contest_results():
    """Load RDA lateralization contest results from JSON files."""
    if not CONTEST_DIR.exists():
        return []

    methods = []
    for jf in sorted(CONTEST_DIR.glob('*.json')):
        try:
            with open(jf) as f:
                r = json.load(f)
            if isinstance(r, dict) and 'primary_auc' in r:
                entry = {
                    'name': jf.stem,
                    'auc': r['primary_auc'],
                    'freq_rho': r.get('freq_rho'),
                    'n_total': r.get('n_total', 0),
                    'n_lrda': r.get('n_lrda', 0),
                    'n_grda': r.get('n_grda', 0),
                }
                methods.append(entry)
        except Exception:
            pass

    methods.sort(key=lambda x: x['auc'], reverse=True)
    return methods


def load_pd_lateralization():
    """Load PD lateralization data from pdnet_v2 evaluation results."""
    eval_path = PROJECT_DIR / 'data' / 'pdnet_v2_cache' / 'evaluation_results.json'
    if eval_path.exists():
        with open(eval_path) as f:
            return json.load(f)
    return None


def main():
    print("Generating Table 3 from primary data...")

    rda_methods = load_rda_contest_results()
    pd_eval = load_pd_lateralization()

    lines = []
    lines.append("# Table 3: Lateralization Performance")
    lines.append("")
    lines.append("*Auto-generated from contest result JSONs and evaluation results.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table3.py`*")
    lines.append("")

    # PD lateralization
    lines.append("## PD Lateralization (ChannelPD-Net)")
    lines.append("")
    if pd_eval:
        lines.append("| Metric | Value | N | Method |")
        lines.append("|---|---|---:|---|")
        auc = pd_eval.get('auc', pd_eval.get('lat_auc', '—'))
        n = pd_eval.get('n_patients', pd_eval.get('n', '—'))
        lines.append(f"| AUC (LPD vs GPD) | {auc} | {n} | RF on ChannelPD-Net features |")
        freq_rho = pd_eval.get('freq_spearman', '—')
        lines.append(f"| Frequency Spearman ρ | {freq_rho} | {n} | ChannelPD-Net ensemble |")
    else:
        lines.append("*PD evaluation results not found at data/pdnet_v2_cache/evaluation_results.json*")
    lines.append("")

    # RDA lateralization
    lines.append("## RDA Lateralization (LRDA vs GRDA) — V5 Contest")
    lines.append("")
    if rda_methods:
        n_total = rda_methods[0].get('n_total', 0)
        n_lrda = rda_methods[0].get('n_lrda', 0)
        n_grda = rda_methods[0].get('n_grda', 0)
        lines.append(f"Dataset: {n_lrda:,} LRDA + {n_grda:,} GRDA = {n_total:,} segments.")
        lines.append("")

        # Top unified methods (with frequency)
        unified = [m for m in rda_methods if m['freq_rho'] is not None and m['freq_rho'] > 0]
        if unified:
            unified.sort(key=lambda x: x['auc'], reverse=True)
            lines.append("### Top Unified Methods (lateralization + frequency)")
            lines.append("")
            lines.append("| Rank | Method | AUC | Freq ρ |")
            lines.append("|---|---|---|---|")
            for i, m in enumerate(unified[:10]):
                lines.append(f"| {i+1} | {m['name']} | {m['auc']:.3f} | {m['freq_rho']:.3f} |")
            lines.append("")

        # Top lateralization-only
        lat_only = [m for m in rda_methods if m['freq_rho'] is None or m['freq_rho'] <= 0]
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
    else:
        lines.append("*Contest results not found at results/lateralization_contest_v4/*")
    lines.append("")

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
