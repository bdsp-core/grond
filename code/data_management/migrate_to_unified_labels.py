#!/usr/bin/env python3
"""
Migrate the current label system to a unified two-file structure.

Input files (NOT modified):
  - data/labels/annotations.csv
  - data/labels/segment_labels.csv
  - data/labels/discharge_times.json
  - data/labels/archive_labels/rda_wave_labels.json

Output files:
  - data/labels/labels.csv    — one row per (segment, rater, label_type)
  - data/labels/segments.csv  — one row per segment, no human labels

Usage:
  conda run -n morgoth python code/data_management/migrate_to_unified_labels.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]
LABELS_DIR = BASE / "data" / "labels"

ANNOTATIONS_CSV = LABELS_DIR / "annotations.csv"
SEGMENT_LABELS_CSV = LABELS_DIR / "segment_labels.csv"
DISCHARGE_TIMES_JSON = LABELS_DIR / "discharge_times.json"
RDA_WAVE_LABELS_JSON = LABELS_DIR / "archive_labels" / "rda_wave_labels.json"

OUT_LABELS = LABELS_DIR / "labels.csv"
OUT_SEGMENTS = LABELS_DIR / "segments.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seg_id_from_mat(mat_file: str) -> str:
    """Strip .mat to get canonical segment_id."""
    return mat_file.replace(".mat", "") if pd.notna(mat_file) else ""


def safe_json(obj) -> str:
    """Serialize to compact JSON string."""
    return json.dumps(obj, separators=(",", ":"))


def is_valid(val) -> bool:
    """Check if a value is non-null and non-empty."""
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str) and val.strip() == "":
        return False
    return True


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------
print("=" * 70)
print("MIGRATION: Current labels -> unified two-file structure")
print("=" * 70)

print("\n[1/6] Loading source files...")

ann = pd.read_csv(ANNOTATIONS_CSV)
sl = pd.read_csv(SEGMENT_LABELS_CSV)

with open(DISCHARGE_TIMES_JSON) as f:
    discharge_times = json.load(f)

if RDA_WAVE_LABELS_JSON.exists():
    with open(RDA_WAVE_LABELS_JSON) as f:
        rda_wave_labels = json.load(f)
else:
    rda_wave_labels = {}

print(f"  annotations.csv:      {len(ann):,} rows")
print(f"  segment_labels.csv:   {len(sl):,} rows")
print(f"  discharge_times.json: {len(discharge_times):,} entries")
print(f"  rda_wave_labels.json: {len(rda_wave_labels):,} entries")

# Build segment_id for segment_labels
sl["segment_id"] = sl["mat_file"].apply(seg_id_from_mat)
seg_id_set_sl = set(sl["segment_id"])
patient_ids_sl = set(sl["patient_id"].astype(str))

# Ensure segment_id column exists in annotations and normalize bare patient_ids
if "segment_id" not in ann.columns:
    ann["segment_id"] = ann["mat_file"].apply(seg_id_from_mat)

# Normalize: if segment_id is a bare patient_id (not in segment_labels),
# map to {patient_id}_seg000
def normalize_segment_id(sid):
    if sid in seg_id_set_sl:
        return sid
    if sid in patient_ids_sl and (sid + "_seg000") in seg_id_set_sl:
        return sid + "_seg000"
    return sid

ann["segment_id"] = ann["segment_id"].apply(normalize_segment_id)
normalized_count = sum(1 for s in ann["segment_id"] if s.endswith("_seg000"))
print(f"  Normalized {sum(1 for _ in ann['segment_id'])} annotation segment_ids"
      f" ({len(ann[ann['segment_id'].apply(lambda s: s in seg_id_set_sl)])} match segment_labels)")

# ---------------------------------------------------------------------------
# Build labels.csv rows
# ---------------------------------------------------------------------------
print("\n[2/6] Extracting labels from annotations.csv...")

label_rows = []


def add_label(segment_id, mat_file, rater, label_type, value,
              metadata="", date="", round_name=""):
    label_rows.append({
        "segment_id": segment_id,
        "mat_file": mat_file if is_valid(mat_file) else segment_id + ".mat",
        "rater": rater,
        "label_type": label_type,
        "value": str(value),
        "metadata": metadata,
        "date": date if is_valid(date) else "",
        "round": round_name if is_valid(round_name) else "",
    })


# --- From annotations.csv ---
ann_freq_count = 0
ann_spatial_extent_count = 0
ann_spatial_channels_count = 0

for _, row in ann.iterrows():
    sid = row["segment_id"]
    mat = sid + ".mat"  # Use normalized segment_id for consistent mat_file
    rater = row["rater"]
    date = row.get("annotation_date", "")
    rnd = row.get("annotation_round", "")

    # frequency_hz
    if is_valid(row.get("frequency_hz")):
        add_label(sid, mat, rater, "frequency_hz", row["frequency_hz"],
                  date=date, round_name=rnd)
        ann_freq_count += 1

    # spatial_extent
    if is_valid(row.get("spatial_extent")):
        add_label(sid, mat, rater, "spatial_extent", row["spatial_extent"],
                  date=date, round_name=rnd)
        ann_spatial_extent_count += 1

    # spatial_channels
    if is_valid(row.get("spatial_channels")):
        add_label(sid, mat, rater, "spatial_channels", row["spatial_channels"],
                  date=date, round_name=rnd)
        ann_spatial_channels_count += 1

print(f"  frequency_hz:      {ann_freq_count:,}")
print(f"  spatial_extent:    {ann_spatial_extent_count:,}")
print(f"  spatial_channels:  {ann_spatial_channels_count:,}")

# --- From segment_labels.csv: IIIC crowd votes ---
print("\n[3/6] Extracting IIIC crowd votes + MW labels from segment_labels.csv...")

iiic_count = 0
mw_class_count = 0
mw_freq_extra_count = 0
mw_lat_count = 0

# Track MW freq segment_ids already captured from annotations.csv
mw_ann_freq_segs = set(
    ann.loc[(ann["rater"] == "MW") & ann["frequency_hz"].notna(), "segment_id"]
)

vote_cols = ["iiic_vote_other", "iiic_vote_seizure", "iiic_vote_lpd",
             "iiic_vote_gpd", "iiic_vote_lrda", "iiic_vote_grda"]

for _, row in sl.iterrows():
    sid = row["segment_id"]
    mat = row["mat_file"]

    # IIIC crowd pattern_class
    n_votes = row.get("iiic_n_votes")
    if is_valid(n_votes) and float(n_votes) >= 1:
        votes = [
            int(row[c]) if is_valid(row.get(c)) else 0
            for c in vote_cols
        ]
        plurality = row.get("iiic_plurality", "")
        plurality_frac = row.get("iiic_plurality_frac", 0)
        meta = safe_json({
            "n_votes": int(float(n_votes)),
            "votes": votes,
            "plurality_frac": round(float(plurality_frac), 4)
                           if is_valid(plurality_frac) else 0,
        })
        add_label(sid, mat, "IIIC_crowd", "pattern_class",
                  plurality if is_valid(plurality) else "",
                  metadata=meta)
        iiic_count += 1

    # MW pattern_class (subtype assigned by MW via folder)
    subtype = row.get("subtype", "")
    if is_valid(subtype):
        add_label(sid, mat, "MW", "pattern_class", subtype,
                  metadata=safe_json({"source": "folder_assignment"}))
        mw_class_count += 1

    # MW expert_freq_hz not already in annotations
    if (is_valid(row.get("expert_freq_hz"))
            and str(row.get("expert_freq_rater", "")).strip() == "MW"
            and sid not in mw_ann_freq_segs):
        add_label(sid, mat, "MW", "frequency_hz", row["expert_freq_hz"],
                  metadata=safe_json({"source": "segment_labels_expert"}))
        mw_freq_extra_count += 1

    # MW laterality
    lat = row.get("laterality")
    if is_valid(lat):
        lat_rater = row.get("laterality_rater", "MW")
        rater = lat_rater if is_valid(lat_rater) else "MW"
        add_label(sid, mat, rater, "laterality", lat)
        mw_lat_count += 1

print(f"  IIIC_crowd pattern_class: {iiic_count:,}")
print(f"  MW pattern_class:         {mw_class_count:,}")
print(f"  MW frequency_hz (extra):  {mw_freq_extra_count:,}")
print(f"  laterality:               {mw_lat_count:,}")

# --- From discharge_times.json ---
print("\n[4/6] Extracting discharge times...")

# Build lookup maps for resolving discharge_times keys
seg_id_set = set(sl["segment_id"])
# Map: source_filename (without .mat) -> list of segment_ids
src_to_segs = {}
for _, row in sl.dropna(subset=["source_filename"]).iterrows():
    src_key = row["source_filename"].replace(".mat", "")
    src_to_segs.setdefault(src_key, []).append(row["segment_id"])

# Map: patient_id -> list of segment_ids
pat_to_segs = {}
for _, row in sl.iterrows():
    pat_to_segs.setdefault(str(row["patient_id"]), []).append(row["segment_id"])

# Segments flagged as having discharge timing
has_dt_segs = set(sl.loc[sl["has_discharge_timing"] == True, "segment_id"])

dt_count = 0
dt_skipped = 0

for key, data in discharge_times.items():
    # Extract just the global_times list
    if isinstance(data, dict):
        times = data.get("global_times", [])
    elif isinstance(data, list):
        times = data
    else:
        dt_skipped += 1
        continue

    if not times:
        dt_skipped += 1
        continue

    # Resolve key -> segment_id(s)
    resolved_segs = []

    if key in seg_id_set:
        # Key is already a segment_id
        resolved_segs = [key]
    elif key in src_to_segs:
        # Key is a source_filename stem
        resolved_segs = src_to_segs[key]
    elif key in pat_to_segs:
        # Key is a patient_id — assign to all segments with has_discharge_timing
        candidates = pat_to_segs[key]
        resolved_segs = [s for s in candidates if s in has_dt_segs]
        if not resolved_segs:
            # Fall back to all segments of that patient
            resolved_segs = candidates

    if not resolved_segs:
        dt_skipped += 1
        continue

    # For each resolved segment, add a discharge_times label
    for sid in resolved_segs:
        mat = sid + ".mat"
        add_label(sid, mat, "MW", "discharge_times", safe_json(times))
        dt_count += 1

print(f"  discharge_times labels: {dt_count:,}")
print(f"  skipped DT entries:     {dt_skipped:,}")

# --- From rda_wave_labels.json ---
print("\n[5/6] Extracting RDA wave times...")

wt_count = 0
wt_skipped = 0

# rda_wave_labels keys are source_filenames (sub-S...) that also appear as
# segment_ids in segment_labels
for key, data in rda_wave_labels.items():
    if not data:
        wt_skipped += 1
        continue

    resolved_segs = []

    if key in seg_id_set:
        resolved_segs = [key]
    elif key in src_to_segs:
        resolved_segs = src_to_segs[key]

    if not resolved_segs:
        wt_skipped += 1
        continue

    for sid in resolved_segs:
        mat = sid + ".mat"
        add_label(sid, mat, "MW", "wave_times", safe_json(data))
        wt_count += 1

print(f"  wave_times labels:  {wt_count:,}")
print(f"  skipped WT entries: {wt_skipped:,}")

# ---------------------------------------------------------------------------
# Build labels DataFrame
# ---------------------------------------------------------------------------
print("\n[6/6] Building output files...")

labels_df = pd.DataFrame(label_rows)
labels_df = labels_df.sort_values(
    ["segment_id", "label_type", "rater"]
).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Build segments.csv
# ---------------------------------------------------------------------------

# Columns that are algorithm predictions or segment metadata (NOT human labels)
ALGO_COLS = ["pdchar_freq_hz", "tautan_freq_hz", "algo_freq_hz"]
META_COLS = [
    "mat_file", "patient_id", "subtype", "excluded", "exclusion_reason",
    "original_source", "source_filename", "montage",
    "duration_sec", "fs", "n_channels",
    "subtype_source", "subtype_original",
    "has_discharge_timing", "has_wave_timing", "has_channel_involvement",
]

# Human label columns to EXCLUDE from segments.csv
HUMAN_COLS = {
    "expert_freq_hz", "expert_freq_rater", "spatial_channels",
    "spatial_raters", "laterality", "laterality_rater",
    "freq_original", "laterality_original", "annotators",
    "iiic_vote_other", "iiic_vote_seizure", "iiic_vote_lpd",
    "iiic_vote_gpd", "iiic_vote_lrda", "iiic_vote_grda",
    "iiic_n_votes", "iiic_plurality", "iiic_plurality_frac",
}

keep_cols = []
for c in sl.columns:
    if c == "segment_id":
        continue  # we'll rebuild it
    if c in HUMAN_COLS:
        continue
    if c in META_COLS or c in ALGO_COLS:
        keep_cols.append(c)

# Also grab any remaining columns not in HUMAN_COLS or META_COLS+ALGO_COLS
for c in sl.columns:
    if c not in keep_cols and c != "segment_id" and c not in HUMAN_COLS:
        keep_cols.append(c)

# Deduplicate while preserving order
seen = set()
unique_keep = []
for c in keep_cols:
    if c not in seen:
        unique_keep.append(c)
        seen.add(c)

segments_df = sl[unique_keep].copy()

# Rename for clarity
col_renames = {}
if "duration_sec" in segments_df.columns:
    col_renames["duration_sec"] = "duration_s"
if "original_source" in segments_df.columns:
    col_renames["original_source"] = "eeg_source"
if "source_filename" in segments_df.columns:
    col_renames["source_filename"] = "eeg_file"
segments_df = segments_df.rename(columns=col_renames)

# Ensure excluded is boolean
if "excluded" in segments_df.columns:
    segments_df["excluded"] = segments_df["excluded"].fillna(False).astype(bool)

segments_df = segments_df.sort_values("mat_file").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
labels_df.to_csv(OUT_LABELS, index=False)
segments_df.to_csv(OUT_SEGMENTS, index=False)

print(f"\n  Wrote {OUT_LABELS}  ({len(labels_df):,} rows)")
print(f"  Wrote {OUT_SEGMENTS} ({len(segments_df):,} rows)")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# 1. Count labels by type and rater
print("\n--- Labels by (label_type, rater) ---")
counts = labels_df.groupby(["label_type", "rater"]).size().reset_index(name="count")
for _, r in counts.iterrows():
    print(f"  {r['label_type']:25s}  {r['rater']:12s}  {r['count']:,}")
print(f"\n  Total label rows: {len(labels_df):,}")

# 2. Verify every segment in segments.csv has at least one label
seg_ids_in_segments = set(segments_df["mat_file"].apply(seg_id_from_mat))
seg_ids_in_labels = set(labels_df["segment_id"])
no_labels = seg_ids_in_segments - seg_ids_in_labels
if no_labels:
    print(f"\n  WARNING: {len(no_labels)} segments have NO labels in labels.csv")
    if len(no_labels) <= 10:
        for s in sorted(no_labels):
            print(f"    - {s}")
else:
    print("\n  OK: Every segment in segments.csv has at least one label.")

# 3. Verify no human labels leaked into segments.csv
human_leak_cols = [c for c in segments_df.columns
                   if c in HUMAN_COLS]
if human_leak_cols:
    print(f"\n  WARNING: Human label columns in segments.csv: {human_leak_cols}")
else:
    print("  OK: No human label columns in segments.csv.")

# 4. Comparison: old vs new counts
print("\n--- Old vs New Comparison ---")
old_ann = len(ann)
old_segs = len(sl)

new_labels = len(labels_df)
new_segs = len(segments_df)

print(f"  Old annotations.csv rows:     {old_ann:,}")
print(f"  Old segment_labels.csv rows:  {old_segs:,}")
print(f"  New labels.csv rows:          {new_labels:,}")
print(f"  New segments.csv rows:        {new_segs:,}")

# Label type breakdown
print("\n--- Label Type Summary ---")
for lt in sorted(labels_df["label_type"].unique()):
    n = (labels_df["label_type"] == lt).sum()
    print(f"  {lt:25s}  {n:,}")

# Rater breakdown
print("\n--- Rater Summary ---")
for r in sorted(labels_df["rater"].unique()):
    n = (labels_df["rater"] == r).sum()
    print(f"  {r:12s}  {n:,}")

print("\n" + "=" * 70)
print("Migration complete. Old files are untouched.")
print("=" * 70)
