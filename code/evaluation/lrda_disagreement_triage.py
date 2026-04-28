#!/usr/bin/env python3
"""LRDA disagreement triage tool.

Identifies the LRDA cases where MW disagrees with the algorithm AND with
at least one other expert (SZ or TZ), on either frequency or laterality,
then produces:

  1. A sub-manifest CSV listing those segments
     (paper_materials/independent_expert_tasks/lrda/disagreement_manifest.csv)

  2. A markdown summary table with the 4-way labels per case
     (paper_materials/independent_expert_tasks/lrda/disagreement_summary.md)

The companion LRDA viewer can then be regenerated against the sub-manifest:

    conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py \\
        --subtype lrda \\
        --manifest paper_materials/independent_expert_tasks/lrda/disagreement_manifest.csv \\
        --output paper_materials/independent_expert_tasks/lrda/disagreement_viewer.html \\
        --no-open

Usage:
    conda run -n morgoth python code/evaluation/lrda_disagreement_triage.py
    conda run -n morgoth python code/evaluation/lrda_disagreement_triage.py --freq-threshold 0.25
"""

import argparse
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
TASK_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'

ROUND = 'independent_expert_v1'


def load_manifest():
    with open(TASK_DIR / 'manifest.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows  # list of dicts with mat_file, patient_id, ...


def load_rater_labels():
    """Load LRDA labels for all 4 raters (MW, SZ, TZ, ALGO).

    Returns: freq[rater][mat_file] = float, lat[rater][mat_file] = 'left'/'right'
    """
    manifest_mfs = {r['mat_file'] for r in load_manifest()}

    freq = {r: {} for r in ['MW', 'SZ', 'TZ', 'ALGO']}
    lat = {r: {} for r in ['MW', 'SZ', 'TZ', 'ALGO']}

    # MW, SZ, TZ from labels.csv
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            rater = row.get('rater', '')
            if rater not in ('MW', 'SZ', 'TZ'):
                continue
            mf = row.get('mat_file', '')
            if mf not in manifest_mfs:
                continue
            lt = row.get('label_type', '')
            v = row.get('value', '')
            if lt == 'frequency_hz':
                try:
                    freq[rater][mf] = float(v)
                except ValueError:
                    pass
            elif lt == 'laterality':
                v = v.strip().lower()
                if v in ('left', 'right'):
                    lat[rater][mf] = v

    # ALGO from rater export JSONs (algo predictions are deterministic per segment)
    algo_files = [
        RAW_DIR / 'TZ' / 'lrda_freq_labeling_results_TZ.json',
        RAW_DIR / 'SZ' / 'rda_freq_labeling_results-2.json',
        RAW_DIR / 'MW' / 'rda_freq_labeling_results-mbw.json',
    ]
    for fp in algo_files:
        if not fp.exists():
            continue
        with open(fp) as f:
            d = json.load(f)
        for entry in d.values():
            mf = entry.get('mat_file')
            sub = (entry.get('subtype') or '').lower()
            if mf not in manifest_mfs or sub != 'lrda':
                continue
            if mf not in freq['ALGO']:
                v = entry.get('w05_freq')
                if v is not None:
                    freq['ALGO'][mf] = float(v)
            if mf not in lat['ALGO']:
                v = entry.get('w05_laterality')
                if v in ('left', 'right'):
                    lat['ALGO'][mf] = v

    return freq, lat


def find_disagreements(freq, lat, freq_thresh=0.5):
    """Return list of disagreement cases.

    A case qualifies if MW disagrees with ALGO on either frequency or
    laterality, AND MW also disagrees with at least one of (SZ, TZ) on
    the SAME metric.

    Returns: list of {mat_file, why, labels} dicts, sorted by severity.
    """
    cases = []
    for mf in {*freq['MW'].keys(), *lat['MW'].keys()}:
        m_f, s_f, t_f, a_f = (freq[r].get(mf) for r in ['MW', 'SZ', 'TZ', 'ALGO'])
        m_l, s_l, t_l, a_l = (lat[r].get(mf) for r in ['MW', 'SZ', 'TZ', 'ALGO'])

        fr_dis_a = m_f is not None and a_f is not None and abs(m_f - a_f) > freq_thresh
        fr_dis_s = m_f is not None and s_f is not None and abs(m_f - s_f) > freq_thresh
        fr_dis_t = m_f is not None and t_f is not None and abs(m_f - t_f) > freq_thresh
        la_dis_a = m_l is not None and a_l is not None and m_l != a_l
        la_dis_s = m_l is not None and s_l is not None and m_l != s_l
        la_dis_t = m_l is not None and t_l is not None and m_l != t_l

        why = []
        if fr_dis_a and (fr_dis_s or fr_dis_t):
            why.append('freq')
        if la_dis_a and (la_dis_s or la_dis_t):
            why.append('lat')
        if not why:
            continue

        # Severity: max abs delta from MW
        sev = 0.0
        if 'freq' in why:
            for other in (a_f, s_f, t_f):
                if other is not None and m_f is not None:
                    sev = max(sev, abs(m_f - other))
        if 'lat' in why:
            sev = max(sev, 1.0)  # lateralization disagreement is high-severity

        cases.append({
            'mat_file': mf,
            'why': why,
            'severity': sev,
            'labels': {
                'MW':   {'freq': m_f, 'lat': m_l},
                'SZ':   {'freq': s_f, 'lat': s_l},
                'TZ':   {'freq': t_f, 'lat': t_l},
                'ALGO': {'freq': a_f, 'lat': a_l},
            },
        })

    cases.sort(key=lambda c: -c['severity'])
    return cases


def write_sub_manifest(cases, manifest_rows):
    """Write a sub-manifest CSV with the same schema as manifest.csv."""
    out_path = TASK_DIR / 'disagreement_manifest.csv'
    indexed = {r['mat_file']: r for r in manifest_rows}
    fieldnames = list(manifest_rows[0].keys())
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for c in cases:
            row = indexed.get(c['mat_file'])
            if row:
                w.writerow(row)
    return out_path


def write_summary_md(cases):
    out_path = TASK_DIR / 'disagreement_summary.md'
    lines = [
        '# LRDA disagreement triage',
        '',
        f'**{len(cases)} segments** where MW disagrees with the algorithm AND with at least one other expert (SZ or TZ), on either frequency (>0.5 Hz) or laterality. Sorted by severity (largest disagreement first).',
        '',
        'For each case, all four labels are shown side-by-side. **Bold** marks any value that differs from MW by more than the disagreement threshold; *italics* marks missing labels. The "Why" column shows which metric triggered the disagreement.',
        '',
        '| # | mat_file | Why | MW freq | SZ freq | TZ freq | ALGO freq | MW lat | SZ lat | TZ lat | ALGO lat |',
        '|---|---|---|---:|---:|---:|---:|---|---|---|---|',
    ]
    for i, c in enumerate(cases, 1):
        L = c['labels']
        mf = c['mat_file']

        def fmt_freq(rater):
            v = L[rater]['freq']
            if v is None:
                return '*--*'
            mw_v = L['MW']['freq']
            if rater != 'MW' and mw_v is not None and abs(v - mw_v) > 0.5:
                return f'**{v:.2f}**'
            return f'{v:.2f}'

        def fmt_lat(rater):
            v = L[rater]['lat']
            if v is None:
                return '*--*'
            mw_v = L['MW']['lat']
            if rater != 'MW' and mw_v is not None and v != mw_v:
                return f'**{v}**'
            return v

        why_str = '+'.join(c['why'])
        # Truncate mat_file for table readability
        mf_short = mf.replace('.mat', '')
        if len(mf_short) > 40:
            mf_short = mf_short[:37] + '...'
        lines.append(
            f"| {i} | `{mf_short}` | {why_str} | "
            f"{fmt_freq('MW')} | {fmt_freq('SZ')} | {fmt_freq('TZ')} | {fmt_freq('ALGO')} | "
            f"{fmt_lat('MW')} | {fmt_lat('SZ')} | {fmt_lat('TZ')} | {fmt_lat('ALGO')} |"
        )

    lines.extend([
        '',
        '## How to review',
        '',
        '1. Generate the focused viewer for these cases:',
        '',
        '   ```bash',
        '   conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py \\',
        '       --subtype lrda \\',
        '       --manifest paper_materials/independent_expert_tasks/lrda/disagreement_manifest.csv \\',
        '       --output paper_materials/independent_expert_tasks/lrda/disagreement_viewer.html \\',
        '       --no-open',
        '   ```',
        '',
        '2. Open `disagreement_viewer.html` and step through the cases (← / → arrows). For each case, the table above tells you exactly which raters disagreed.',
        '',
        '3. For each case, decide:',
        '   - **MW labeling error**: change the row in `labels.csv` (rater=MW, label_type=frequency_hz or laterality, mat_file=...).',
        '   - **Genuine ambiguity**: leave as-is, note in the manuscript that some LRDA segments are inherently ambiguous.',
        '   - **Algorithm bug**: file an issue, design an error-analysis fix.',
        '',
        '4. After any label changes, re-run:',
        '',
        '   ```bash',
        '   conda run -n morgoth python code/data_management/build_segment_labels.py',
        '   conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py',
        '   cp results/independent_expert_v1/forest_plot.png paper_materials/figures/figS5_independent_expert_v1_irr.png',
        '   ```',
    ])
    out_path.write_text('\n'.join(lines) + '\n')
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freq-threshold', type=float, default=0.5,
                        help='Frequency disagreement threshold in Hz (default 0.5)')
    args = parser.parse_args()

    print(f"LRDA disagreement triage")
    print(f"  freq threshold: {args.freq_threshold} Hz")
    print()

    manifest_rows = load_manifest()
    freq, lat = load_rater_labels()

    print(f"Coverage on LRDA manifest:")
    for r in ['MW', 'SZ', 'TZ', 'ALGO']:
        print(f"  {r}:  freq={len(freq[r]):>3d}  lat={len(lat[r]):>3d}")
    print()

    cases = find_disagreements(freq, lat, freq_thresh=args.freq_threshold)
    print(f"Disagreement cases: {len(cases)}")
    for i, c in enumerate(cases, 1):
        L = c['labels']
        why = '+'.join(c['why'])
        mw_f = L['MW']['freq']
        a_f = L['ALGO']['freq']
        mw_l = L['MW']['lat']
        a_l = L['ALGO']['lat']
        print(f"  {i}. {c['mat_file']}  (severity={c['severity']:.2f})")
        print(f"     why={why}, MW={mw_f}/{mw_l}, ALGO={a_f}/{a_l}, "
              f"SZ={L['SZ']['freq']}/{L['SZ']['lat']}, "
              f"TZ={L['TZ']['freq']}/{L['TZ']['lat']}")

    sub_manifest_path = write_sub_manifest(cases, manifest_rows)
    summary_path = write_summary_md(cases)
    print()
    print(f"Wrote {sub_manifest_path.relative_to(PROJECT_DIR)}")
    print(f"Wrote {summary_path.relative_to(PROJECT_DIR)}")


if __name__ == '__main__':
    main()
