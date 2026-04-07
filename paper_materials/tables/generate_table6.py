#!/usr/bin/env python3
"""
Auto-generate Table 6 (Discharge Timing Performance) from primary data.

Reads evaluation results from:
  - data/cet_cache/consolidated_results.json (main CET+DP results)
  - paper_materials/method_comparison_table.json (method comparison)

Usage:
    conda run -n morgoth python paper_materials/tables/generate_table6.py
"""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CET_CACHE = PROJECT_DIR / 'data' / 'cet_cache'
COMPARISON_PATH = PROJECT_DIR / 'paper_materials' / 'method_comparison_table.json'
OUTPUT_PATH = Path(__file__).resolve().parent / 'table6_timing.md'


def load_cet_results():
    """Load CET evaluation results."""
    results = {}

    # Main consolidated results
    consolidated = CET_CACHE / 'consolidated_results.json'
    if consolidated.exists():
        with open(consolidated) as f:
            results['consolidated'] = json.load(f)

    # CET-UNet CV results
    unet_cv = CET_CACHE / 'cet_unet_cv_results.json'
    if unet_cv.exists():
        with open(unet_cv) as f:
            results['unet_cv'] = json.load(f)

    # Post-hoc improvement results
    posthoc = CET_CACHE / 'improvement_posthoc_results.json'
    if posthoc.exists():
        with open(posthoc) as f:
            results['posthoc'] = json.load(f)

    return results


def load_method_comparison():
    """Load method comparison table."""
    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH) as f:
            return json.load(f)
    return None


def fmt_metric(val, fmt_str=".3f"):
    """Format a metric value."""
    if val is None:
        return "—"
    try:
        return f"{float(val):{fmt_str}}"
    except (ValueError, TypeError):
        return str(val)


def main():
    print("Generating Table 6 from primary data...")

    cet = load_cet_results()
    comparison = load_method_comparison()

    lines = []
    lines.append("# Table 6: PD Discharge Timing Performance")
    lines.append("")
    lines.append("*Auto-generated from `data/cet_cache/` evaluation results.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table6.py`*")
    lines.append("")

    # Main CET results
    if 'consolidated' in cet:
        c = cet['consolidated']
        lines.append("## CET-UNet + DP (Consolidated Results)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for key in ['f1', 'sensitivity', 'precision', 'freq_spearman', 'timing_mae_ms']:
            if key in c:
                label = key.replace('_', ' ').title()
                lines.append(f"| {label} | {fmt_metric(c[key])} |")
        if 'n_cases' in c:
            lines.append(f"| N cases | {c['n_cases']} |")
        lines.append("")

    # Method comparison
    if comparison:
        lines.append("## Method Comparison")
        lines.append("")

        if isinstance(comparison, list):
            # List of methods
            lines.append("| Method | F1 | Timing MAE (ms) | Freq ρ | Notes |")
            lines.append("|---|---|---|---|---|")
            for m in comparison:
                name = m.get('name', m.get('method', '—'))
                f1 = fmt_metric(m.get('f1'))
                mae = fmt_metric(m.get('timing_mae_ms', m.get('timing_mae')))
                rho = fmt_metric(m.get('freq_spearman', m.get('freq_rho')))
                notes = m.get('notes', '')
                lines.append(f"| {name} | {f1} | {mae} | {rho} | {notes} |")
        elif isinstance(comparison, dict):
            # Dict with sections
            for section_name, methods in comparison.items():
                lines.append(f"### {section_name}")
                lines.append("")
                if isinstance(methods, list):
                    lines.append("| Method | F1 | Freq ρ | Notes |")
                    lines.append("|---|---|---|---|")
                    for m in methods:
                        name = m.get('name', m.get('method', '—'))
                        f1 = fmt_metric(m.get('f1'))
                        rho = fmt_metric(m.get('freq_spearman', m.get('freq_rho')))
                        notes = m.get('notes', '')
                        lines.append(f"| {name} | {f1} | {rho} | {notes} |")
                lines.append("")
    else:
        # Fallback: read from individual result files
        lines.append("## Available Evaluation Results")
        lines.append("")
        for name, data in cet.items():
            lines.append(f"### {name}")
            lines.append("")
            if isinstance(data, dict):
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for k, v in sorted(data.items()):
                    if isinstance(v, (int, float)):
                        lines.append(f"| {k} | {fmt_metric(v)} |")
            lines.append("")

    # CET-UNet CV results
    if 'unet_cv' in cet:
        cv = cet['unet_cv']
        lines.append("## Cross-Validation Results (CET-UNet)")
        lines.append("")
        if isinstance(cv, dict):
            # Per-fold results
            if 'folds' in cv:
                lines.append("| Fold | F1 | Sens | Prec |")
                lines.append("|---|---|---|---|")
                for i, fold in enumerate(cv['folds']):
                    lines.append(f"| {i} | {fmt_metric(fold.get('f1'))} | {fmt_metric(fold.get('sensitivity'))} | {fmt_metric(fold.get('precision'))} |")
            else:
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for k, v in sorted(cv.items()):
                    if isinstance(v, (int, float)):
                        lines.append(f"| {k} | {fmt_metric(v)} |")
        lines.append("")

    OUTPUT_PATH.write_text('\n'.join(lines) + '\n')
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
