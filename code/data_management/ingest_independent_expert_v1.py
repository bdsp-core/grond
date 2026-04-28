#!/usr/bin/env python3
"""Ingest the independent expert v1 annotation results into labels.csv.

Reads the raw JSON exports under

    data/labels/raw_inputs/independent_expert_v1/{SZ,TZ}/

(one set of four files per rater, plus a 400-entry combined RDA file
for SZ -- see the README in that folder) and appends rows to
data/labels/labels.csv with round='independent_expert_v1'. Then run
build_segment_labels.py separately to regenerate segment_labels.csv.

Mapping rules:
  - frequency_hz   <- 'selected_freq' (TZ PD viewer) or 'freq' (everywhere else),
                     skipped when the entry was rejected/not accepted
  - laterality     <- 'laterality' (LPD and LRDA only); skipped when null,
                     'bilateral', or rejected (the user can flag GRDA-on-LRDA
                     reclassifications via rejected entries)
  - discharge_times<- 'global_times' (LPD and GPD only) as JSON-encoded list

The script is idempotent: it refuses to add a row whose
(segment_id, rater, label_type, round) tuple already exists.

Usage:
    conda run -n morgoth python code/data_management/ingest_independent_expert_v1.py
    conda run -n morgoth python code/data_management/ingest_independent_expert_v1.py --dry-run
"""

import argparse
import csv
import json
import sys
from datetime import date as _date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
LABELS_CSV = LABELS_DIR / 'labels.csv'

ROUND = 'independent_expert_v1'

# Files per rater. The order matters only for logging.
RATER_FILES = {
    'TZ': [
        ('lpd_freq_timing_results_TZ.json',     'lpd',  'pd'),
        ('gpd_freq_timing_results_TZ.json',     'gpd',  'pd'),
        ('lrda_freq_labeling_results_TZ.json',  'lrda', 'rda'),
        ('grda_freq_labeling_results_TZ.json',  'grda', 'rda'),
    ],
    'SZ': [
        ('lpd_freq_timing_batch1_results.json',  'lpd',  'pd'),
        ('gpd_freq_timing_batch1_results.json',  'gpd',  'pd'),
        # SZ exported a combined LRDA+GRDA file (400 entries); use it.
        # The earlier 200-entry rda_freq_labeling_results.json was a partial
        # export and is superseded.
        ('rda_freq_labeling_results-2.json',     'rda',  'rda_combined'),
    ],
    'MW': [
        # MW only labels LRDA + GRDA frequency in v1; LPD/GPD frequency,
        # laterality, and discharge timing already exist in labels.csv from
        # earlier rounds. Add an LPD or GPD entry here if MW does a catch-up
        # pass on those subsets later.
        #
        # MW's actual export is a single combined file (rda_freq_labeling_results-mbw.json)
        # containing both LRDA-tagged and GRDA-tagged entries because the RDA
        # viewer's localStorage key is shared across file:// URLs, so opening
        # both lrda_task_MW.html and grda_task_MW.html accumulates labels in
        # one shared localStorage namespace. We mark it 'rda_combined' (same
        # treatment as SZ's combined export) and rely on the manifest filter
        # added below to discard out-of-manifest entries that cross-pollute
        # from earlier no-manifest viewer runs on the same browser profile.
        ('rda_freq_labeling_results-mbw.json',   None,   'rda_combined'),
    ],
}


def load_all_manifests():
    """Return {subtype: set_of_mat_files} for the four task manifests."""
    out = {}
    for sub in ('lpd', 'gpd', 'lrda', 'grda'):
        manifest_csv = (PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'
                        / sub / 'manifest.csv')
        with open(manifest_csv) as f:
            out[sub] = {row['mat_file'] for row in csv.DictReader(f)}
    return out


def is_pd_accepted(entry):
    """A PD viewer entry counts as accepted iff not flagged rejected."""
    if entry.get('rejected'):
        return False
    if entry.get('review_status') == 'rejected':
        return False
    return True


def is_rda_accepted(entry):
    """An RDA viewer entry counts as accepted iff action=='accept'."""
    return entry.get('action') == 'accept'


def freq_from_pd_entry(entry):
    """PD viewer prefers selected_freq, falls back to est_freq."""
    v = entry.get('selected_freq')
    if v is None:
        v = entry.get('est_freq')
    return v


def freq_from_rda_entry(entry):
    return entry.get('freq')


def laterality_from(entry):
    """Normalize laterality to 'left' / 'right' or None if not informative."""
    lat = entry.get('laterality')
    if lat is None or lat == '':
        return None
    lat = str(lat).strip().lower()
    if lat in ('left', 'right'):
        return lat
    # 'bilateral' / 'generalized' / anything else is not a per-side label
    return None


def collect_rater_rows(rater, files, today, existing_keys, manifests):
    """Yield labels.csv rows from one rater's set of JSON files.

    `files` is a list of (filename, subtype_tag, viewer_kind) tuples.
    `existing_keys` is a set of (segment_id, rater, label_type, round)
    tuples already present in labels.csv that we must not duplicate.
    `manifests` is {subtype: set_of_mat_files} for the canonical 200-segment
    task subsets; entries whose mat_file is not in the appropriate manifest
    are discarded as out-of-scope localStorage residue.
    """
    rows = []
    accept_counts = {'lpd':0, 'gpd':0, 'lrda':0, 'grda':0}
    reject_counts = {'lpd':0, 'gpd':0, 'lrda':0, 'grda':0}
    out_of_manifest = {'lpd':0, 'gpd':0, 'lrda':0, 'grda':0}

    for filename, subtype_tag, viewer_kind in files:
        path = RAW_DIR / rater / filename
        if not path.exists():
            print(f"  WARNING: {path} not found; skipping", file=sys.stderr)
            continue
        with open(path) as f:
            d = json.load(f)
        print(f"  [{rater}] {filename}: {len(d)} entries")

        for key, entry in d.items():
            # Identify the segment.
            mat_file = entry.get('mat_file')
            seg_id = entry.get('segment_id')
            if not mat_file:
                # In TZ's PD files the patient_id is also the segment_id base;
                # mat_file may not be present, but segment_id is.
                if seg_id:
                    mat_file = seg_id + '.mat'
            if not seg_id:
                if mat_file:
                    seg_id = mat_file.replace('.mat', '')
            if not seg_id or not mat_file:
                continue  # malformed

            # Determine subtype for this entry.
            if viewer_kind == 'rda_combined':
                # SZ's combined file: entry-level subtype tells us LRDA vs GRDA.
                ent_sub = (entry.get('subtype') or '').lower()
                if ent_sub not in ('lrda', 'grda'):
                    continue
                this_sub = ent_sub
            else:
                this_sub = subtype_tag

            # Manifest filter: discard segments outside the canonical
            # 200-segment subsets (defends against localStorage residue from
            # other RDA viewer runs that share the file:// origin).
            if mat_file not in manifests.get(this_sub, set()):
                out_of_manifest[this_sub] += 1
                continue

            # Acceptance check.
            if viewer_kind == 'pd':
                if not is_pd_accepted(entry):
                    reject_counts[this_sub] += 1
                    continue
            else:
                if not is_rda_accepted(entry):
                    reject_counts[this_sub] += 1
                    continue
            accept_counts[this_sub] += 1

            # Frequency
            if viewer_kind == 'pd':
                f_val = freq_from_pd_entry(entry)
            else:
                f_val = freq_from_rda_entry(entry)
            if f_val is not None:
                k = (seg_id, rater, 'frequency_hz', ROUND)
                if k not in existing_keys:
                    rows.append({
                        'segment_id': seg_id, 'mat_file': mat_file,
                        'rater': rater, 'label_type': 'frequency_hz',
                        'value': str(round(float(f_val), 4)),
                        'metadata': '', 'date': today, 'round': ROUND,
                    })

            # Laterality (only meaningful for LPD and LRDA; GPD and GRDA are
            # bilateral/generalized by definition and the viewer omits the
            # laterality input).
            if this_sub in ('lpd', 'lrda'):
                lat = laterality_from(entry)
                if lat is not None:
                    k = (seg_id, rater, 'laterality', ROUND)
                    if k not in existing_keys:
                        rows.append({
                            'segment_id': seg_id, 'mat_file': mat_file,
                            'rater': rater, 'label_type': 'laterality',
                            'value': lat,
                            'metadata': '', 'date': today, 'round': ROUND,
                        })

            # Discharge times (PD only).
            if viewer_kind == 'pd':
                times = entry.get('global_times')
                if times is not None and len(times) > 0:
                    k = (seg_id, rater, 'discharge_times', ROUND)
                    if k not in existing_keys:
                        # Match MW's format: JSON-encoded list with no spaces
                        # after commas (the existing CSV escaping handles
                        # quoting; we use json.dumps with default separators).
                        val = json.dumps([round(float(t), 4) for t in times], separators=(',', ':'))
                        rows.append({
                            'segment_id': seg_id, 'mat_file': mat_file,
                            'rater': rater, 'label_type': 'discharge_times',
                            'value': val,
                            'metadata': '', 'date': today, 'round': ROUND,
                        })

    return rows, accept_counts, reject_counts, out_of_manifest


def existing_round_keys():
    """Return the set of (segment_id, rater, label_type, round) tuples
    already in labels.csv. Used to enforce idempotency."""
    keys = set()
    if not LABELS_CSV.exists():
        return keys
    with open(LABELS_CSV) as f:
        for row in csv.DictReader(f):
            keys.add((
                row.get('segment_id', ''),
                row.get('rater', ''),
                row.get('label_type', ''),
                row.get('round', ''),
            ))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be added without modifying labels.csv')
    args = parser.parse_args()

    today = _date.today().isoformat()
    print(f"Ingesting independent expert v1 results")
    print(f"  Round tag: {ROUND}")
    print(f"  Date stamp: {today}")
    print()

    existing_keys = existing_round_keys()
    print(f"  labels.csv currently has {len(existing_keys)} (seg, rater, type, round) keys")

    manifests = load_all_manifests()
    print(f"  Loaded {sum(len(v) for v in manifests.values())} manifest segments across "
          f"{', '.join(f'{k}={len(v)}' for k,v in manifests.items())}")

    new_rows = []
    summary = {}
    for rater, files in RATER_FILES.items():
        print(f"\n[{rater}]")
        rows, accept_counts, reject_counts, oom_counts = collect_rater_rows(
            rater, files, today, existing_keys, manifests
        )
        new_rows.extend(rows)
        # Add to existing_keys so subsequent raters don't double-count
        for r in rows:
            existing_keys.add((r['segment_id'], r['rater'], r['label_type'], r['round']))
        summary[rater] = {
            'rows_to_add': len(rows),
            'accepted': accept_counts,
            'rejected': reject_counts,
            'out_of_manifest': oom_counts,
        }

    print(f"\n=== Summary ===")
    for rater, s in summary.items():
        print(f"  {rater}: {s['rows_to_add']} new rows")
        print(f"     accepted by subtype:        {s['accepted']}")
        print(f"     rejected by subtype:        {s['rejected']}")
        print(f"     out-of-manifest (skipped):  {s['out_of_manifest']}")
    print(f"  TOTAL new rows: {len(new_rows)}")

    # Per-label-type breakdown
    from collections import Counter
    by_lbl = Counter((r['rater'], r['label_type']) for r in new_rows)
    print(f"\n  Breakdown by (rater, label_type):")
    for k, n in sorted(by_lbl.items()):
        print(f"     {k}: {n}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    if not new_rows:
        print("\nNo new rows to add; labels.csv unchanged.")
        return

    # Append to labels.csv preserving the existing column order.
    with open(LABELS_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)

    with open(LABELS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
        for r in new_rows:
            # Pad missing columns to match header
            for col in header:
                r.setdefault(col, '')
            writer.writerow({k: r.get(k, '') for k in header})

    print(f"\nAppended {len(new_rows)} rows to {LABELS_CSV.relative_to(PROJECT_DIR)}")
    print(f"Run code/data_management/build_segment_labels.py to regenerate segment_labels.csv.")


if __name__ == '__main__':
    main()
