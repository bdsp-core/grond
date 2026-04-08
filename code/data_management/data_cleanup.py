#!/usr/bin/env python3
"""
Data cleanup script for IIIC Frequency Functions project.
Steps A-G as specified.
"""

import os
import json
import shutil
import pandas as pd
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELS = os.path.join(BASE, 'data', 'labels')
EEG_DIR = os.path.join(BASE, 'data', 'eeg')
ARCHIVE = os.path.join(BASE, 'data', '_archive')

PATIENTS_CSV = os.path.join(LABELS, 'patients.csv')
SEGMENTS_CSV = os.path.join(LABELS, 'segments.csv')
DISCHARGE_TIMES_HPP = os.path.join(LABELS, 'discharge_times_hpp.json')
DISCHARGE_TIMES = os.path.join(LABELS, 'discharge_times.json')
BIPD_SCREENING = os.path.join(ARCHIVE, 'bipd', 'bipd_screening_results.json')

MANIFEST_FILES = {
    'harvest_manifest': (os.path.join(LABELS, 'harvest_manifest.json'), 'lpd'),
    'bipd_harvest_manifest': (os.path.join(LABELS, 'bipd_harvest_manifest.json'), 'bipd'),
    'rda_harvest_manifest': (os.path.join(LABELS, 'rda_harvest_manifest.json'), None),  # subtype in data
    'other_harvest_manifest': (os.path.join(LABELS, 'other_harvest_manifest.json'), 'other'),
}

print("=" * 70)
print("DATA CLEANUP SCRIPT")
print("=" * 70)

# ─── STEP A: Add harvested patients to segments.csv ────────────────────────

print("\n[STEP A] Adding harvest patients to segments.csv")

patients_df = pd.read_csv(PATIENTS_CSV, dtype={'patient_id': str})
segments_df = pd.read_csv(SEGMENTS_CSV, dtype={'patient_id': str})

# Backup
shutil.copy(SEGMENTS_CSV, SEGMENTS_CSV + '.bak')
print(f"  Backed up segments.csv → segments.csv.bak")

harvest_patients = patients_df[patients_df['subtype_rater'] == 'harvest'].copy()
existing_seg_pids = set(segments_df['patient_id'].astype(str))

eeg_files_set = set(os.listdir(EEG_DIR))

new_seg_rows = []
skipped = []
no_eeg = []

for _, row in harvest_patients.iterrows():
    pid = str(row['patient_id'])
    if pid in existing_seg_pids:
        skipped.append(pid)
        continue
    mat_file = f"{pid}_seg000.mat"
    if mat_file not in eeg_files_set:
        no_eeg.append(pid)
        print(f"  WARNING: No EEG file for harvest patient {pid}")
        continue
    new_seg_rows.append({
        'segment_id': f"{pid}_seg000",
        'patient_id': pid,
        'subtype': row['subtype'],
        'subtype_source': 'harvest',
        'mat_file': mat_file,
        'duration_sec': 10,
        'fs': 200,
        'n_channels': 18,
        'montage': 'bipolar',
        'original_source': 'harvest',
        'original_filename': mat_file,
    })

if new_seg_rows:
    new_segs_df = pd.DataFrame(new_seg_rows)
    segments_df = pd.concat([segments_df, new_segs_df], ignore_index=True)
    segments_df.to_csv(SEGMENTS_CSV, index=False)
    print(f"  Added {len(new_seg_rows)} new segment rows to segments.csv")
else:
    print("  No new rows to add.")

print(f"  Already in segments.csv (skipped): {len(skipped)}")
print(f"  Missing EEG file (skipped): {len(no_eeg)}")
print(f"  segments.csv now has {len(segments_df)} rows")


# ─── STEP B: Copy discharge_times_hpp.json → discharge_times.json ─────────

print("\n[STEP B] Copying discharge_times_hpp.json → discharge_times.json")

shutil.copy(DISCHARGE_TIMES_HPP, DISCHARGE_TIMES)
print(f"  Copied: {DISCHARGE_TIMES_HPP}")
print(f"      to: {DISCHARGE_TIMES}")
print("  (Original discharge_times_hpp.json preserved)")


# ─── STEP C: Migrate timing data — add 'subtype' field where missing ───────

print("\n[STEP C] Migrating timing data (adding subtype field)")

with open(DISCHARGE_TIMES) as f:
    dt = json.load(f)

# Build pid→subtype map from patients.csv
pid_to_subtype = dict(zip(patients_df['patient_id'].astype(str), patients_df['subtype'].astype(str)))

updated_count = 0
already_count = 0
missing_pid_count = 0

for pid, entry in dt.items():
    if 'subtype' not in entry or entry['subtype'] is None:
        subtype = pid_to_subtype.get(pid)
        if subtype:
            entry['subtype'] = subtype
            updated_count += 1
        else:
            missing_pid_count += 1
    else:
        already_count += 1

print(f"  Entries already had subtype: {already_count}")
print(f"  Entries updated with subtype: {updated_count}")
print(f"  Entries where pid not found in patients.csv: {missing_pid_count}")

# Rename 'source' → keep as-is; rename 'review_source' to 'source' if no 'source'
# The target schema has 'source', not 'review_source'. Handle gracefully.
source_fixed = 0
for pid, entry in dt.items():
    if 'source' not in entry and 'review_source' in entry:
        entry['source'] = entry['review_source']
        source_fixed += 1

if source_fixed:
    print(f"  Fixed 'source' field (from review_source) for {source_fixed} entries")

# Rename 'selected_freq' — check if it's stored as 'gold_standard_freq'
# The target schema shows 'selected_freq'. Map gold_standard_freq → selected_freq if missing.
freq_fixed = 0
for pid, entry in dt.items():
    if 'selected_freq' not in entry and 'gold_standard_freq' in entry:
        entry['selected_freq'] = entry['gold_standard_freq']
        freq_fixed += 1

if freq_fixed:
    print(f"  Added 'selected_freq' (from gold_standard_freq) for {freq_fixed} entries")

with open(DISCHARGE_TIMES, 'w') as f:
    json.dump(dt, f, indent=2)
print(f"  Saved discharge_times.json with {len(dt)} entries")


# ─── STEP D: Merge BIPD screening results ──────────────────────────────────

print("\n[STEP D] Merging BIPD screening results")

with open(BIPD_SCREENING) as f:
    bipd_data = json.load(f)

with open(DISCHARGE_TIMES) as f:
    dt = json.load(f)

# Backup patients.csv
shutil.copy(PATIENTS_CSV, PATIENTS_CSV + '.bak')
print(f"  Backed up patients.csv → patients.csv.bak")

patients_df = pd.read_csv(PATIENTS_CSV, dtype={'patient_id': str})
existing_pids = set(patients_df['patient_id'].astype(str))

confirmed_bipd = [(k, v) for k, v in bipd_data.items() if v.get('is_bipd')]
print(f"  Confirmed BIPD entries: {len(confirmed_bipd)}")

new_patients_rows = []
added_to_dt = 0
already_in_dt = 0

for pid, info in confirmed_bipd:
    # Add to discharge_times.json if not present
    if pid not in dt:
        dt[pid] = {
            'review_status': 'awaiting_timing',
            'source': 'bipd_screener',
            'subtype': 'bipd',
            'left_times': [],
            'right_times': [],
        }
        added_to_dt += 1
    else:
        already_in_dt += 1
        # Ensure subtype is set
        if dt[pid].get('subtype') != 'bipd':
            dt[pid]['subtype'] = 'bipd'

    # Add to patients.csv if not present
    if pid not in existing_pids:
        new_patients_rows.append({
            'patient_id': pid,
            'subtype': 'bipd',
            'subtype_original': 'bipd',
            'n_segments': 1,
            'n_raters': None,
            'raters': None,
            'gold_standard_freq': None,
            'gold_standard_freq_original': None,
            'excluded': False,
            'exclusion_reason': None,
            'laterality': None,
            'laterality_original': None,
            'subtype_rater': 'bipd_screener',
            'laterality_rater': None,
        })
        existing_pids.add(pid)

with open(DISCHARGE_TIMES, 'w') as f:
    json.dump(dt, f, indent=2)

print(f"  Added {added_to_dt} new BIPD entries to discharge_times.json")
print(f"  Already in discharge_times.json: {already_in_dt}")
print(f"  New patients to add to patients.csv: {len(new_patients_rows)}")

if new_patients_rows:
    new_pt_df = pd.DataFrame(new_patients_rows)
    patients_df = pd.concat([patients_df, new_pt_df], ignore_index=True)
    patients_df.to_csv(PATIENTS_CSV, index=False)
    print(f"  patients.csv now has {len(patients_df)} rows")
else:
    print("  No new patients added (all already present)")


# ─── STEP E: Add harvest-manifest patients to patients.csv ────────────────

print("\n[STEP E] Adding harvest-manifest patients to patients.csv")

patients_df = pd.read_csv(PATIENTS_CSV, dtype={'patient_id': str})
segments_df = pd.read_csv(SEGMENTS_CSV, dtype={'patient_id': str})
existing_pids = set(patients_df['patient_id'].astype(str))
existing_seg_pids = set(segments_df['patient_id'].astype(str))

# Collect all manifest entries
all_manifest_entries = {}  # pid_key → dict
for mname, (mpath, default_subtype) in MANIFEST_FILES.items():
    with open(mpath) as f:
        mdata = json.load(f)
    for pid_key, val in mdata.items():
        pid = val.get('patient_id', pid_key)
        subtype = val.get('subtype', default_subtype)
        # Store first occurrence only to avoid dupes
        if pid_key not in all_manifest_entries:
            all_manifest_entries[pid_key] = {
                'patient_id': pid,
                'pid_key': pid_key,
                'subtype': subtype,
                'manifest': mname,
                'val': val,
            }

# Deduplicate by patient_id (pid_key may differ from patient_id for some entries)
# Use pid_key as patient_id for those entries (BIPD, RDA, other use long IDs as keys)
new_patients_rows = []
new_seg_rows = []

added_patients = 0
added_segs = 0

# Track which patient IDs we've seen (to avoid double-adding)
seen_pids_this_step = set()

for pid_key, info in all_manifest_entries.items():
    pid = info['patient_id']
    # Use pid_key as the canonical ID (some manifests use long IDs as keys)
    # For harvest_manifest, pid_key == patient_id (numeric)
    # For others, pid_key is the long ID and patient_id may be the same
    canonical_pid = pid_key  # Use the key as canonical

    if canonical_pid in existing_pids or canonical_pid in seen_pids_this_step:
        continue
    seen_pids_this_step.add(canonical_pid)

    val = info['val']
    subtype = info['subtype']
    mname = info['manifest']

    # Determine gold_standard_freq: only if they have an est_freq AND s3_file (i.e., were accepted)
    est_freq = val.get('est_freq')
    s3_file = val.get('s3_file', '')
    # For harvest_manifest: accepted means s3_file is non-empty
    # For others: all entries are considered accepted (they're in the manifest)
    if mname == 'harvest_manifest':
        gold_freq = est_freq if (est_freq is not None and s3_file) else None
    else:
        gold_freq = est_freq if est_freq is not None else None

    new_patients_rows.append({
        'patient_id': canonical_pid,
        'subtype': subtype,
        'subtype_original': subtype,
        'n_segments': None,
        'n_raters': None,
        'raters': None,
        'gold_standard_freq': gold_freq,
        'gold_standard_freq_original': gold_freq,
        'excluded': True,
        'exclusion_reason': 'harvest_unreviewed',
        'laterality': None,
        'laterality_original': None,
        'subtype_rater': 'harvest',
        'laterality_rater': None,
    })
    added_patients += 1
    existing_pids.add(canonical_pid)

    # Check if EEG file exists for this patient
    mat_file = f"{canonical_pid}_seg000.mat"
    if mat_file in eeg_files_set and canonical_pid not in existing_seg_pids:
        new_seg_rows.append({
            'segment_id': f"{canonical_pid}_seg000",
            'patient_id': canonical_pid,
            'subtype': subtype,
            'subtype_source': 'harvest',
            'mat_file': mat_file,
            'duration_sec': 10,
            'fs': 200,
            'n_channels': 18,
            'montage': 'bipolar',
            'original_source': mname,
            'original_filename': mat_file,
        })
        added_segs += 1
        existing_seg_pids.add(canonical_pid)

if new_patients_rows:
    new_pt_df = pd.DataFrame(new_patients_rows)
    patients_df = pd.concat([patients_df, new_pt_df], ignore_index=True)
    patients_df.to_csv(PATIENTS_CSV, index=False)

if new_seg_rows:
    new_segs_df2 = pd.DataFrame(new_seg_rows)
    segments_df = pd.concat([segments_df, new_segs_df2], ignore_index=True)
    segments_df.to_csv(SEGMENTS_CSV, index=False)

print(f"  Added {added_patients} new patients to patients.csv")
print(f"  Added {added_segs} new segment rows to segments.csv (those with local EEG files)")
print(f"  patients.csv now has {len(patients_df)} rows")
print(f"  segments.csv now has {len(segments_df)} rows")


# ─── STEP F: Catalog orphan EEG files ─────────────────────────────────────

print("\n[STEP F] Cataloging orphan EEG files")

# Reload fresh data
patients_df = pd.read_csv(PATIENTS_CSV, dtype={'patient_id': str})
segments_df = pd.read_csv(SEGMENTS_CSV, dtype={'patient_id': str})

all_patient_ids = set(patients_df['patient_id'].astype(str))
seg_mat_files = set(segments_df['mat_file'].astype(str))

# Collect all manifest keys
all_manifest_keys = set()
for mname, (mpath, _) in MANIFEST_FILES.items():
    with open(mpath) as f:
        mdata = json.load(f)
    all_manifest_keys.update(mdata.keys())
    # Also collect patient_id values
    for k, v in mdata.items():
        pid = v.get('patient_id')
        if pid:
            all_manifest_keys.add(pid)

eeg_files_list = os.listdir(EEG_DIR)
orphan_catalog = {}

for fname in eeg_files_list:
    if fname in seg_mat_files:
        continue  # already cataloged in segments.csv
    # Extract patient ID
    parts = fname.rsplit('_seg', 1)
    if len(parts) != 2:
        continue
    pid = parts[0]
    if pid not in orphan_catalog:
        orphan_catalog[pid] = {
            'mat_files': [],
            'n_segments': 0,
            'in_patients_csv': pid in all_patient_ids,
            'in_harvest_manifest': pid in all_manifest_keys,
        }
    orphan_catalog[pid]['mat_files'].append(fname)
    orphan_catalog[pid]['n_segments'] += 1

# Sort mat_files within each entry
for pid in orphan_catalog:
    orphan_catalog[pid]['mat_files'].sort()

catalog_path = os.path.join(LABELS, 'orphan_eeg_catalog.json')
with open(catalog_path, 'w') as f:
    json.dump(orphan_catalog, f, indent=2, sort_keys=True)

print(f"  Orphan patients: {len(orphan_catalog)}")
print(f"  Orphan EEG files: {sum(v['n_segments'] for v in orphan_catalog.values())}")
print(f"  Of orphan pids — in patients.csv: {sum(1 for v in orphan_catalog.values() if v['in_patients_csv'])}")
print(f"  Of orphan pids — in manifest: {sum(1 for v in orphan_catalog.values() if v['in_harvest_manifest'])}")
print(f"  Written to: {catalog_path}")


# ─── STEP G: Validate consistency ─────────────────────────────────────────

print("\n[STEP G] Validating consistency")

patients_df = pd.read_csv(PATIENTS_CSV, dtype={'patient_id': str})
segments_df = pd.read_csv(SEGMENTS_CSV, dtype={'patient_id': str})

with open(DISCHARGE_TIMES) as f:
    dt = json.load(f)

all_pids = set(patients_df['patient_id'].astype(str))
active_pids = set(patients_df[patients_df['excluded'] == False]['patient_id'].astype(str))
seg_pids = set(segments_df['patient_id'].astype(str))
dt_pids = set(dt.keys())
eeg_files_set2 = set(os.listdir(EEG_DIR))

# Check 1: Every active patient has at least one EEG file
active_no_eeg = []
for pid in active_pids:
    # Check if any EEG file matches this pid
    has_eeg = any(f.startswith(f"{pid}_seg") for f in eeg_files_set2)
    if not has_eeg:
        active_no_eeg.append(pid)

# Check 2: Every patient in segments.csv exists in patients.csv
seg_pids_not_in_patients = seg_pids - all_pids

# Check 3: Every patient in discharge_times.json exists in patients.csv
dt_pids_not_in_patients = dt_pids - all_pids

print(f"\n  Validation Results:")
print(f"  {'─'*50}")
print(f"  Total patients in patients.csv:       {len(patients_df):>6}")
print(f"  Active (non-excluded) patients:       {len(active_pids):>6}")
print(f"  Patients with segments:               {len(seg_pids):>6}")
print(f"  Patients in discharge_times.json:     {len(dt_pids):>6}")
print(f"  {'─'*50}")
print(f"  [CHECK 1] Active patients without EEG file: {len(active_no_eeg)}")
if active_no_eeg:
    for pid in active_no_eeg[:10]:
        print(f"    - {pid}")
    if len(active_no_eeg) > 10:
        print(f"    ... and {len(active_no_eeg)-10} more")
else:
    print(f"    PASS - all active patients have EEG files")

print(f"  [CHECK 2] segments.csv patients not in patients.csv: {len(seg_pids_not_in_patients)}")
if seg_pids_not_in_patients:
    for pid in list(seg_pids_not_in_patients)[:10]:
        print(f"    - {pid}")
else:
    print(f"    PASS")

print(f"  [CHECK 3] discharge_times.json pids not in patients.csv: {len(dt_pids_not_in_patients)}")
if dt_pids_not_in_patients:
    for pid in list(dt_pids_not_in_patients)[:10]:
        print(f"    - {pid}")
else:
    print(f"    PASS")

print(f"\n  Final Summary Table:")
print(f"  {'─'*50}")

# Subtype breakdown for patients
subtype_counts = patients_df.groupby(['subtype', 'excluded']).size().unstack(fill_value=0)
print(f"  Patients by subtype (active/excluded):")
for subtype in sorted(patients_df['subtype'].dropna().unique()):
    sub_df = patients_df[patients_df['subtype'] == subtype]
    active = (sub_df['excluded'] == False).sum()
    excluded = (sub_df['excluded'] == True).sum()
    print(f"    {subtype:<12}  active={active:>4}  excluded={excluded:>4}  total={len(sub_df):>4}")

print(f"\n  Segments by subtype:")
seg_subtype = segments_df['subtype'].value_counts()
for st, cnt in seg_subtype.items():
    print(f"    {st:<12}  {cnt:>5} segments")

print(f"\n  Discharge times by review_status:")
status_counts = defaultdict(int)
for pid, entry in dt.items():
    status_counts[entry.get('review_status', 'unknown')] += 1
for status, cnt in sorted(status_counts.items()):
    print(f"    {status:<25}  {cnt:>4} entries")

print("\n" + "=" * 70)
print("ALL STEPS COMPLETE")
print("=" * 70)
