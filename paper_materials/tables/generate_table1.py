#!/usr/bin/env python3
"""
Auto-generate Table 1 (Dataset Statistics) from canonical label files.

Reads:
  - data/labels/segment_labels.csv  (consolidated per-segment summary)
  - data/labels/annotations.csv     (per-rater annotations)
  - data/labels/discharge_times.json
  - data/labels/rda_wave_labels.json
  - data/labels/channel_involvement.json

Writes:
  - paper_materials/tables/table1_dataset.md

Usage:
    conda run -n morgoth python paper_materials/tables/generate_table1.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
OUTPUT_PATH = Path(__file__).resolve().parent / 'table1_dataset.md'

SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']


def is_valid(val):
    """Check if a CSV value is non-empty and not NaN."""
    if val is None:
        return False
    val = str(val).strip()
    return val not in ('', 'nan', 'NaN', 'None', '0', '0.0')


def count_segment_labels():
    """Count labels per subtype from segment_labels.csv."""
    counts = defaultdict(lambda: defaultdict(int))
    patients = defaultdict(set)

    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sub = row.get('subtype', '').lower()
            if sub not in SUBTYPES:
                continue

            excluded = row.get('excluded', '').lower() in ('true', '1', 'yes')

            if excluded:
                counts[sub]['excluded'] += 1
                continue

            counts[sub]['total'] += 1
            patients[sub].add(row.get('patient_id', ''))

            # Expert frequency
            if is_valid(row.get('expert_freq_hz')):
                counts[sub]['expert_freq'] += 1

            # Algorithm frequency
            if is_valid(row.get('algo_freq_hz')):
                counts[sub]['algo_freq'] += 1

            # Any frequency (expert, algo, pdchar, or tautan)
            has_any_freq = any(is_valid(row.get(col)) for col in
                               ['expert_freq_hz', 'algo_freq_hz', 'pdchar_freq_hz', 'tautan_freq_hz'])
            if has_any_freq:
                counts[sub]['any_freq'] += 1

            # Discharge timing
            if row.get('has_discharge_timing', '').lower() in ('true', '1', 'yes'):
                counts[sub]['discharge_timing'] += 1

            # Wave timing
            if row.get('has_wave_timing', '').lower() in ('true', '1', 'yes'):
                counts[sub]['wave_timing'] += 1

            # Channel involvement
            if row.get('has_channel_involvement', '').lower() in ('true', '1', 'yes'):
                counts[sub]['channel_involvement'] += 1

            # Laterality
            if is_valid(row.get('laterality')):
                counts[sub]['laterality'] += 1

            # IIIC crowd votes
            try:
                nv = float(row.get('iiic_n_votes', 0) or 0)
                if nv >= 10:
                    counts[sub]['iiic_10plus'] += 1
            except (ValueError, TypeError):
                pass

    for sub in SUBTYPES:
        counts[sub]['patients'] = len(patients[sub])

    return counts


def count_rater_annotations():
    """Count per-rater per-subtype annotations from annotations.csv."""
    # First build mat_file → subtype lookup from segment_labels
    subtype_map = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subtype_map[row['mat_file']] = row.get('subtype', '').lower()

    rater_counts = defaultdict(lambda: defaultdict(int))
    with open(LABELS_DIR / 'annotations.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rater = row.get('rater', 'unknown')
            mat_file = row.get('mat_file', '')
            sub = subtype_map.get(mat_file, 'unknown')
            rater_counts[rater][sub] += 1
            rater_counts[rater]['total'] += 1

    return rater_counts


def fmt(n):
    """Format number with commas."""
    return f"{n:,}"


def fmt_or_dash(n):
    """Format number or show dash if zero."""
    return fmt(n) if n > 0 else '—'


def generate_table(counts, rater_counts):
    """Generate the markdown table."""
    lines = []
    lines.append("# Table 1: Dataset Statistics")
    lines.append("")
    lines.append("*Auto-generated from `data/labels/segment_labels.csv` and `annotations.csv`.*")
    lines.append("*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table1.py`*")
    lines.append("")

    # Segment counts
    lines.append("## EEG Segments by Subtype")
    lines.append("")
    total_segs = sum(counts[s]['total'] for s in SUBTYPES)
    total_excl = sum(counts[s]['excluded'] for s in SUBTYPES)
    lines.append("| | LPD | GPD | LRDA | GRDA | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| Segments (non-excluded) | {fmt(counts['lpd']['total'])} | {fmt(counts['gpd']['total'])} | {fmt(counts['lrda']['total'])} | {fmt(counts['grda']['total'])} | {fmt(total_segs)} |")
    lines.append(f"| Excluded | {fmt(counts['lpd']['excluded'])} | {fmt(counts['gpd']['excluded'])} | {fmt(counts['lrda']['excluded'])} | {fmt(counts['grda']['excluded'])} | {fmt(total_excl)} |")
    lines.append(f"| Unique patients | {fmt(counts['lpd']['patients'])} | {fmt(counts['gpd']['patients'])} | {fmt(counts['lrda']['patients'])} | {fmt(counts['grda']['patients'])} | — |")
    lines.append("")
    lines.append("All segments: 19-channel monopolar EEG, 10 seconds at 200 Hz (2,000 samples). Common average reference montage.")
    lines.append("")

    # Label coverage
    lines.append("## Label Coverage by Subtype and Task")
    lines.append("")
    lines.append("| Label type | LPD | GPD | LRDA | GRDA | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    label_rows = [
        ('Expert-reviewed frequency', 'expert_freq'),
        ('Algorithm frequency', 'algo_freq'),
        ('Any frequency', 'any_freq'),
        ('Discharge timing', 'discharge_timing'),
        ('Wave timing', 'wave_timing'),
        ('Channel involvement / spatial', 'channel_involvement'),
        ('Laterality', 'laterality'),
        ('IIIC crowd votes (≥10 raters)', 'iiic_10plus'),
    ]
    for label_name, key in label_rows:
        vals = [counts[s][key] for s in SUBTYPES]
        total = sum(vals)
        lines.append(f"| {label_name} | {fmt_or_dash(vals[0])} | {fmt_or_dash(vals[1])} | {fmt_or_dash(vals[2])} | {fmt_or_dash(vals[3])} | {fmt(total)} |")
    lines.append("")

    # Rater contributions
    lines.append("## Rater Contributions (annotations.csv)")
    lines.append("")
    lines.append("| Rater | LPD | GPD | LRDA | GRDA | Total |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    grand_total = 0
    for rater in sorted(rater_counts.keys()):
        rc = rater_counts[rater]
        vals = [rc.get(s, 0) for s in SUBTYPES]
        total = rc['total']
        grand_total += total
        lines.append(f"| {rater} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} | {fmt(vals[3])} | {fmt(total)} |")
    lines.append(f"| **Total annotations** | | | | | **{fmt(grand_total)}** |")
    lines.append("")

    # Model training data
    lines.append("## Model Training Data")
    lines.append("")
    lines.append("| Model | Segments | Selection criteria |")
    lines.append("|---|---:|---|")
    # ChannelPD-Net: LPD/GPD with expert freq
    cpdnet = counts['lpd']['expert_freq'] + counts['gpd']['expert_freq']
    lines.append(f"| ChannelPD-Net | {fmt(cpdnet)} | LPD/GPD with expert frequency |")
    # HemiCET: ground-truth discharge timing (from discharge_times.json GT entries)
    try:
        with open(LABELS_DIR / 'discharge_times.json') as f:
            dt = json.load(f)
        gt_count = sum(1 for v in dt.values()
                       if isinstance(v, dict) and v.get('review_status') == 'ground_truth')
        lines.append(f"| HemiCET-UNet | {fmt(gt_count)} | Ground-truth discharge timing |")
        lines.append(f"| CET-UNet | {fmt(gt_count)} | Ground-truth discharge timing |")
    except Exception:
        lines.append("| HemiCET-UNet | — | Ground-truth discharge timing |")
        lines.append("| CET-UNet | — | Ground-truth discharge timing |")
    lines.append("")

    # Data provenance
    lines.append("## Data Provenance")
    lines.append("")
    lines.append("| Source | Description |")
    lines.append("|---|---|")
    lines.append("| IIIC crowd-labeled | 10-min recordings, center 10s extracted; ≥10 expert votes per segment |")
    lines.append("| MW-labeled | From pattern-specific S3 folders; single-rater classification |")
    lines.append("| Expert dataset | 38-patient dataset; 4 raters (LB, PH, SZ, MW) |")

    return '\n'.join(lines) + '\n'


def main():
    print("Generating Table 1 from label files...")

    print("  Reading segment_labels.csv...")
    counts = count_segment_labels()

    print("  Reading annotations.csv...")
    rater_counts = count_rater_annotations()

    # Print summary
    for sub in SUBTYPES:
        c = counts[sub]
        print(f"  {sub.upper()}: {c['total']} segments, {c['expert_freq']} expert freq, "
              f"{c['discharge_timing']} discharge timing, {c['laterality']} laterality")

    table_md = generate_table(counts, rater_counts)

    OUTPUT_PATH.write_text(table_md)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
