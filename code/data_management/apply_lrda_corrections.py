#!/usr/bin/env python3
"""Apply MW LRDA corrections from the disagreement-review pass.

Diffs the corrections JSON (mbw2) against the original MW export (mbw)
and, for each disagreement-manifest segment that changed, updates the
MW row in labels.csv in place (no new rows are appended; old values
are overwritten). The original MW JSON is preserved for audit trail.

Sources:
    data/labels/raw_inputs/independent_expert_v1/MW/rda_freq_labeling_results-mbw.json    (original)
    data/labels/raw_inputs/independent_expert_v1/MW/rda_freq_labeling_results-mbw2.json   (corrections)
    paper_materials/independent_expert_tasks/lrda/disagreement_manifest.csv               (scope)

Sink:
    data/labels/labels.csv  (in-place edit, only MW rows for changed segments)

The corrections file is expected to be the raw export from
disagreement_viewer.html and may contain the full ~744 entries from the
shared LRDA-viewer localStorage; only segments in the disagreement
manifest are considered. Likewise, only the (segment, label_type) tuples
where the value actually changed between the two JSONs are updated -- a
re-run with no changes is a no-op.

Usage:
    conda run -n morgoth python code/data_management/apply_lrda_corrections.py
    conda run -n morgoth python code/data_management/apply_lrda_corrections.py --dry-run
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1' / 'MW'
TASK_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'

LABELS_CSV = LABELS_DIR / 'labels.csv'
ORIGINAL_JSON = RAW_DIR / 'rda_freq_labeling_results-mbw.json'
CORRECTIONS_JSON = RAW_DIR / 'rda_freq_labeling_results-mbw2.json'
DISAGREEMENT_MANIFEST = TASK_DIR / 'disagreement_manifest.csv'


def load_disagreement_set():
    with open(DISAGREEMENT_MANIFEST) as f:
        return {r['mat_file'] for r in csv.DictReader(f)}


def value_for(entry, field):
    """Return the canonical value the ingester would have used for a given
    field, or None if the entry doesn't represent an accepted label."""
    if entry is None:
        return None
    if entry.get('action') != 'accept':
        return None
    if field == 'frequency_hz':
        v = entry.get('freq')
        return None if v is None else float(v)
    if field == 'laterality':
        v = entry.get('laterality')
        return v if v in ('left', 'right') else None
    return None


def compute_diffs(old, new, disagreement_set):
    """Return [(mat_file, label_type, old_val, new_val), ...] for changes
    in disagreement-manifest segments only.

    A change includes value -> value, value -> None (rejection now), and
    None -> value (accept now).
    """
    diffs = []
    for sid, new_entry in new.items():
        mf = new_entry.get('mat_file')
        if mf not in disagreement_set:
            continue
        old_entry = old.get(sid)
        # Only consider LRDA entries (the JSON also has GRDA + residue).
        if (new_entry.get('subtype') or '').lower() != 'lrda':
            continue
        if old_entry is not None and (old_entry.get('subtype') or '').lower() != 'lrda':
            old_entry = None
        for field in ('frequency_hz', 'laterality'):
            ov = value_for(old_entry, field)
            nv = value_for(new_entry, field)
            if ov != nv:
                diffs.append((mf, field, ov, nv))
    return diffs


def apply_diffs(diffs, dry_run=False):
    """Update labels.csv in place. For each (mat_file, label_type, _, new):
      - delete any existing row(s) with rater=MW, label_type=label_type, mat_file=mf
      - if new is not None, insert a fresh row tagged with today's date and
        round=independent_expert_v1 (the original ingest's round)
    """
    from datetime import date
    today = date.today().isoformat()

    # Read entire CSV
    with open(LABELS_CSV, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    print(f"Loaded {len(rows)} rows from labels.csv")

    # Index header
    h = {col: i for i, col in enumerate(header)}

    # Apply each diff
    summary = []
    for mf, label_type, old_val, new_val in diffs:
        seg_id = mf.replace('.mat', '')
        # Find any existing rows
        keep = []
        deleted = 0
        for row in rows:
            if (row[h['rater']] == 'MW'
                    and row[h['mat_file']] == mf
                    and row[h['label_type']] == label_type):
                deleted += 1
                continue
            keep.append(row)
        rows = keep

        # Insert new row if the new value is not None
        if new_val is not None:
            new_row = [''] * len(header)
            new_row[h['segment_id']] = seg_id
            new_row[h['mat_file']] = mf
            new_row[h['rater']] = 'MW'
            new_row[h['label_type']] = label_type
            new_row[h['value']] = (
                f"{float(new_val):.4f}".rstrip('0').rstrip('.')
                if label_type == 'frequency_hz' else str(new_val)
            )
            new_row[h['metadata']] = json.dumps({
                'correction_round': 'lrda_disagreement_review',
                'previous_value': old_val,
            }, separators=(',', ':'))
            new_row[h['date']] = today
            new_row[h['round']] = 'independent_expert_v1'
            rows.append(new_row)

        summary.append({
            'mat_file': mf,
            'label_type': label_type,
            'old': old_val,
            'new': new_val,
            'rows_deleted': deleted,
            'row_inserted': new_val is not None,
        })

    # Print summary
    print()
    print(f"Diffs to apply: {len(diffs)}")
    for s in summary:
        print(f"  {s['mat_file']}  {s['label_type']}: "
              f"old={s['old']} -> new={s['new']}  "
              f"(deleted {s['rows_deleted']} existing row(s); "
              f"{'inserted 1 new' if s['row_inserted'] else 'no new row (rejected)'})")

    if dry_run:
        print("\nDRY RUN -- labels.csv unchanged.")
        return

    # Backup before writing
    backup = LABELS_CSV.with_suffix('.csv.bak_pre_lrda_corrections')
    shutil.copy(LABELS_CSV, backup)
    print(f"\nBacked up to {backup.relative_to(PROJECT_DIR)}")

    # Write back
    with open(LABELS_CSV, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {LABELS_CSV.relative_to(PROJECT_DIR)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    disagreement_set = load_disagreement_set()
    print(f"Disagreement manifest: {len(disagreement_set)} segments")

    with open(ORIGINAL_JSON) as f:
        old = json.load(f)
    with open(CORRECTIONS_JSON) as f:
        new = json.load(f)
    print(f"Original JSON entries: {len(old)}")
    print(f"Corrections JSON entries: {len(new)}")

    diffs = compute_diffs(old, new, disagreement_set)
    if not diffs:
        print("\nNo diffs detected on the disagreement-manifest segments. labels.csv unchanged.")
        return
    apply_diffs(diffs, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
