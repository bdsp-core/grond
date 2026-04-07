#!/usr/bin/env python3
"""
Auto-generate Table 5 (Frequency Estimation Performance) from segment_labels.csv.

Computes Spearman ρ and MAE between expert and predicted frequencies for
PDCharacterizer/W05 and Tautan et al., per subtype.

Usage:
    conda run -n morgoth python paper_materials/tables/generate_table5.py
"""

import csv
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
OUTPUT_PATH = Path(__file__).resolve().parent / 'table5_frequency.md'

SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']


def load_freq_data():
    """Load frequency data per subtype from segment_labels.csv."""
    data = {s: {'expert': [], 'pdchar': [], 'tautan': []} for s in SUBTYPES}

    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get('subtype', '').lower()
            if sub not in SUBTYPES:
                continue
            if row.get('excluded', '').lower() in ('true', '1', 'yes'):
                continue

            expert = row.get('expert_freq_hz', '').strip()
            pdchar = row.get('pdchar_freq_hz', '').strip()
            tautan = row.get('tautan_freq_hz', '').strip()

            try:
                ef = float(expert)
            except (ValueError, TypeError):
                continue  # no expert freq → skip

            if not np.isfinite(ef) or ef <= 0:
                continue

            try:
                pf = float(pdchar) if pdchar and pdchar not in ('nan', 'NaN', '') else np.nan
            except (ValueError, TypeError):
                pf = np.nan

            try:
                tf = float(tautan) if tautan and tautan not in ('nan', 'NaN', '') else np.nan
            except (ValueError, TypeError):
                tf = np.nan

            data[sub]['expert'].append(ef)
            data[sub]['pdchar'].append(pf)
            data[sub]['tautan'].append(tf)

    return data


def compute_metrics(expert, predicted):
    """Compute Spearman ρ and MAE for valid pairs."""
    expert = np.array(expert)
    predicted = np.array(predicted)
    valid = np.isfinite(expert) & np.isfinite(predicted) & (expert > 0) & (predicted > 0)
    if valid.sum() < 5:
        return None, None, 0
    e, p = expert[valid], predicted[valid]
    rho, _ = spearmanr(e, p)
    mae = np.mean(np.abs(e - p))
    return rho, mae, int(valid.sum())


def main():
    print("Generating Table 5 from segment_labels.csv...")

    data = load_freq_data()

    lines = []
    lines.append("# Table 5: Frequency Estimation Performance")
    lines.append("")
    lines.append("*Auto-generated from `data/labels/segment_labels.csv`.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table5.py`*")
    lines.append("")

    # Per-subtype performance
    lines.append("## Per-Subtype Performance (Quality-Filtered)")
    lines.append("")
    lines.append("| Subtype | N (PDChar) | PDChar/W05 ρ | PDChar MAE (Hz) | N (Tautan) | Tautan ρ | Tautan MAE (Hz) |")
    lines.append("|---|---:|---|---|---:|---|---|")

    for sub in SUBTYPES:
        d = data[sub]
        rho_p, mae_p, n_p = compute_metrics(d['expert'], d['pdchar'])
        rho_t, mae_t, n_t = compute_metrics(d['expert'], d['tautan'])

        rho_p_str = f"**{rho_p:.3f}**" if rho_p is not None else "—"
        mae_p_str = f"{mae_p:.3f}" if mae_p is not None else "—"
        rho_t_str = f"{rho_t:.3f}" if rho_t is not None else "—"
        mae_t_str = f"{mae_t:.3f}" if mae_t is not None else "—"

        print(f"  {sub.upper()}: PDChar n={n_p} ρ={rho_p_str}, Tautan n={n_t} ρ={rho_t_str}")

        lines.append(f"| {sub.upper()} | {n_p:,} | {rho_p_str} | {mae_p_str} | {n_t:,} | {rho_t_str} | {mae_t_str} |")

    lines.append("")
    lines.append("Quality filter: segments with expert-reviewed frequency (expert_freq_hz) and valid algorithm prediction.")
    lines.append("")

    # RDA frequency top methods (from contest results)
    contest_dir = PROJECT_DIR / 'results' / 'lateralization_contest_v4'
    if contest_dir.exists():
        import json
        methods = []
        for jf in sorted(contest_dir.glob('*.json')):
            try:
                with open(jf) as f:
                    r = json.load(f)
                if isinstance(r, dict) and 'primary_auc' in r and 'freq_rho' in r:
                    if r['freq_rho'] is not None and r['freq_rho'] > 0:
                        methods.append({
                            'name': jf.stem,
                            'auc': r['primary_auc'],
                            'freq_rho': r['freq_rho'],
                        })
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
            lines.append(f"{len(methods)} methods evaluated on LRDA vs GRDA classification + frequency estimation.")
            lines.append("")

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
