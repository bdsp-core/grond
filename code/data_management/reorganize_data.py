#!/usr/bin/env python
"""
reorganize_data.py — Reorganize ALL data into a clean unified structure.

Target:
    data/
    ├── eeg/                    ← ALL .mat files, uniform format
    ├── labels/
    │   ├── segments.csv        ← One row per segment
    │   ├── annotations.csv     ← One row per (segment, rater) rating
    │   └── patients.csv        ← One row per patient (derived)
"""

import os
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio

warnings.filterwarnings("ignore")

ROOT = Path("/Users/mwestover/GithubRepos/IIIC-Frequency-Functions-For-Morgoth")
DATA = ROOT / "data"
EEG_OUT = DATA / "eeg"
LABELS_OUT = DATA / "labels"

# Create output dirs
EEG_OUT.mkdir(parents=True, exist_ok=True)
LABELS_OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: Build segment registry and copy/export .mat files
# ─────────────────────────────────────────────────────────────────────────────

def extract_patient_id(filename):
    """Extract patient_id from dataset_eeg filename.
    e.g. 'pat0103_20180322_120800_1008_seg_3.mat' → 'pat0103'
         'abn1047_20141222_080922_1858_seg_3.mat' → 'abn1047'
         'emu386_20160426_131624_3874_seg_1.mat' → 'emu386'
         'sub-S0001112591683_20160911141146_seg_0-10s.mat' → 'sub-S0001112591683'
    """
    # Standard format: letters+digits before first 8-digit date
    m = re.match(r'^([a-zA-Z]+\d+)_\d{8}_', filename)
    if m:
        return m.group(1)
    # sub-SXXX format
    m = re.match(r'^(sub-S\d+)_\d+_', filename)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract patient_id from {filename}")


segments = []  # list of dicts for segments.csv

# --- Source A: Original 43 patients (data/dataset_eeg/) ---
print("=" * 70)
print("PART 1A: Processing dataset_eeg (original patients)")
print("=" * 70)

patient_seg_counter = defaultdict(int)  # patient_id → next segment number

for subtype in ["lpd", "gpd", "grda", "lrda"]:
    src_dir = DATA / "dataset_eeg" / subtype
    mat_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".mat")])
    print(f"  {subtype}: {len(mat_files)} files")

    for orig_fname in mat_files:
        patient_id = extract_patient_id(orig_fname)
        seg_num = patient_seg_counter[patient_id]
        patient_seg_counter[patient_id] += 1

        new_fname = f"{patient_id}_seg{seg_num:03d}.mat"
        segment_id = f"{patient_id}_seg{seg_num:03d}"

        # Read original .mat — could be v7.3 (HDF5) or older v5
        src_path = src_dir / orig_fname
        try:
            with h5py.File(src_path, "r") as f:
                # HDF5: stored as (2000, 20); we want (20, 2000)
                key = "data_50sec" if "data_50sec" in f else "data"
                data = f[key][:].T  # → (20, 2000)
        except OSError:
            # Older .mat format
            m = sio.loadmat(str(src_path))
            key = "data_50sec" if "data_50sec" in m else "data"
            data = m[key]
            # Ensure (20, 2000) shape
            if data.shape == (2000, 20):
                data = data.T

        out_path = EEG_OUT / new_fname
        sio.savemat(str(out_path), {"data": data, "Fs": 200}, do_compression=True)

        segments.append({
            "segment_id": segment_id,
            "patient_id": patient_id,
            "subtype": subtype,
            "subtype_source": "folder_structure",
            "mat_file": new_fname,
            "duration_sec": 10.0,
            "fs": 200,
            "n_channels": 20,
            "montage": "monopolar",
            "original_source": "dataset_eeg",
            "original_filename": orig_fname,
        })

print(f"  Total Source A segments: {len(segments)}")

# --- Source B: External patients (data/dl_cache/external_pd_segments.npz) ---
print("\n" + "=" * 70)
print("PART 1B: Processing external_pd_segments.npz")
print("=" * 70)

npz = np.load(DATA / "dl_cache" / "external_pd_segments.npz")
ext_segments = npz["segments"]  # (3816, 18, 2000)
ext_labels = npz["labels"]  # 0=LPD, 1=GPD
ext_patients = npz["patients"]  # patient IDs as strings

label_map = {0: "lpd", 1: "gpd"}

# Group by patient and select top 5 by variance
patient_indices = defaultdict(list)
for i, pid in enumerate(ext_patients):
    patient_indices[pid].append(i)

print(f"  Total external segments: {len(ext_segments)}")
print(f"  Unique external patients: {len(patient_indices)}")

n_ext_before = 0
n_ext_after = 0
ext_seg_counter = defaultdict(int)

for pid in sorted(patient_indices.keys()):
    indices = patient_indices[pid]
    n_ext_before += len(indices)

    # Compute variance for each segment to pick top 5
    if len(indices) > 5:
        variances = [ext_segments[i].var() for i in indices]
        top_indices = [indices[j] for j in np.argsort(variances)[-5:]]
    else:
        top_indices = indices

    n_ext_after += len(top_indices)

    for idx in sorted(top_indices):
        seg_num = ext_seg_counter[pid]
        ext_seg_counter[pid] += 1

        new_fname = f"{pid}_seg{seg_num:03d}.mat"
        segment_id = f"{pid}_seg{seg_num:03d}"
        subtype = label_map[ext_labels[idx]]

        data = ext_segments[idx]  # (18, 2000)
        out_path = EEG_OUT / new_fname
        sio.savemat(str(out_path), {"data": data, "Fs": 200}, do_compression=True)

        segments.append({
            "segment_id": segment_id,
            "patient_id": pid,
            "subtype": subtype,
            "subtype_source": "iiic_majority_vote",
            "mat_file": new_fname,
            "duration_sec": 10.0,
            "fs": 200,
            "n_channels": 18,
            "montage": "bipolar",
            "original_source": "external_drive",
            "original_filename": f"external_pd_segments.npz[{idx}]",
        })

print(f"  Before cap: {n_ext_before} segments, After cap (5/patient): {n_ext_after}")
print(f"  Total segments so far: {len(segments)}")

# Build segments DataFrame
seg_df = pd.DataFrame(segments)
seg_df.to_csv(LABELS_OUT / "segments.csv", index=False)
print(f"\n  segments.csv written: {len(seg_df)} rows")

# Build lookup from original_filename → segment_id (for Source A)
orig_to_segid = {}
for _, row in seg_df[seg_df["original_source"] == "dataset_eeg"].iterrows():
    orig_to_segid[row["original_filename"]] = row["segment_id"]

# Build lookup from patient_id → list of segment_ids (for external)
ext_patient_segments = defaultdict(list)
for _, row in seg_df[seg_df["original_source"] == "external_drive"].iterrows():
    ext_patient_segments[row["patient_id"]].append(row["segment_id"])


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: Build annotations.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: Building annotations.csv")
print("=" * 70)

annotations = []  # list of dicts

# --- Source 1: Raw expert annotations ---
print("\n  Source 1: Raw expert annotations")

annotation_dir = DATA / "annotations"
annotation_files = sorted(os.listdir(annotation_dir))
print(f"    Files: {annotation_files}")

# Map rater codes from filenames
def get_rater_from_filename(fname):
    if "_LB_" in fname:
        return "LB"
    elif "_PH_" in fname:
        return "PH"
    elif "_SZ_" in fname:
        return "SZ"
    raise ValueError(f"Unknown rater in {fname}")

def get_subtype_from_filename(fname):
    fname_upper = fname.upper()
    if fname_upper.startswith("LPDS"):
        return "lpd"
    elif fname_upper.startswith("GPDS"):
        return "gpd"
    elif fname_upper.startswith("GRDA"):
        return "grda"
    elif fname_upper.startswith("LRDA"):
        return "lrda"
    raise ValueError(f"Unknown subtype in {fname}")

def extract_mat_from_files_col(files_path):
    """Extract original .mat filename from annotation files column.
    e.g. '.../lpd/pat0103_20180322_120800_1008_seg_3_score.png'
    → 'pat0103_20180322_120800_1008_seg_3.mat'
    """
    basename = os.path.basename(files_path)
    # Remove _score.png suffix
    mat_name = basename.replace("_score.png", ".mat")
    return mat_name

src1_count = 0
src1_unmapped = 0

for ann_file in annotation_files:
    if not ann_file.endswith(".csv"):
        continue

    rater = get_rater_from_filename(ann_file)
    subtype = get_subtype_from_filename(ann_file)

    df = pd.read_csv(annotation_dir / ann_file, encoding="utf-8", encoding_errors="replace")

    for _, row in df.iterrows():
        files_val = str(row.get("files", ""))
        if not files_val or files_val == "nan":
            continue

        orig_mat = extract_mat_from_files_col(files_val)
        segment_id = orig_to_segid.get(orig_mat)

        if segment_id is None:
            src1_unmapped += 1
            continue

        patient_id = segment_id.rsplit("_seg", 1)[0]

        freq = row.get("frequency", None)
        try:
            freq = float(freq)
        except (ValueError, TypeError):
            freq = None

        no_pd = (freq is not None and freq == 0.0)
        freq_hz = None if (freq is None or freq == 0.0) else freq

        # Spatial extent
        spatial = row.get("spatial", None)
        try:
            spatial = float(spatial)
        except (ValueError, TypeError):
            spatial = None

        # Spatial area/channels — use canonical columns (not _original)
        spatial_area = row.get("spatial_area", None)
        if pd.isna(spatial_area) or str(spatial_area).strip() in ("", "0", "nan"):
            spatial_channels = None
        else:
            spatial_channels = str(spatial_area).strip()

        # Notes for PH's spatial_origin
        notes = None
        if rater == "PH" and "spatial_origin" in row.index:
            so = row.get("spatial_origin", None)
            if pd.notna(so) and str(so).strip() not in ("", "nan"):
                notes = f"spatial_origin={so}"

        annotations.append({
            "segment_id": segment_id,
            "patient_id": patient_id,
            "rater": rater,
            "frequency_hz": freq_hz,
            "no_pd": no_pd,
            "skipped": False,
            "spatial_extent": spatial,
            "spatial_channels": spatial_channels,
            "annotation_date": None,
            "annotation_round": "original",
            "notes": notes,
        })
        src1_count += 1

print(f"    Added {src1_count} annotations, {src1_unmapped} unmapped")

# --- Source 2: Canonical dataset labels (MW ratings) ---
print("\n  Source 2: Canonical dataset labels (MW)")

canonical = pd.read_csv(DATA / "canonical_dataset" / "labels.csv")
src2_count = 0
src2_unmapped = 0

for _, row in canonical.iterrows():
    if row.get("excluded", False):
        continue

    mat_name = row["mat_name"]
    mw_val = row.get("reviewer_MW", None)

    if pd.isna(mw_val):
        continue

    patient_id = str(row["patient_id"])

    # Find segment_id: first try original filename (dataset_eeg patients)
    segment_id = orig_to_segid.get(mat_name)

    # If not found, try external patients by patient_id
    if segment_id is None:
        patient_segs = ext_patient_segments.get(patient_id, [])
        if patient_segs:
            segment_id = patient_segs[0]

    if segment_id is None:
        src2_unmapped += 1
        continue

    freq = float(mw_val)
    no_pd = (freq == 0.0)
    freq_hz = None if freq == 0.0 else freq

    annotations.append({
        "segment_id": segment_id,
        "patient_id": patient_id,
        "rater": "MW",
        "frequency_hz": freq_hz,
        "no_pd": no_pd,
        "skipped": False,
        "spatial_extent": None,
        "spatial_channels": None,
        "annotation_date": None,
        "annotation_round": "canonical",
        "notes": None,
    })
    src2_count += 1

print(f"    Added {src2_count} annotations, {src2_unmapped} unmapped")

# Track which patients already have MW annotations (from canonical)
mw_annotated_patients = set()
for a in annotations:
    if a["rater"] == "MW" and not a.get("skipped", False):
        mw_annotated_patients.add(a["patient_id"])

# --- Sources 3-5: Round 2/3/4 annotations (MW ratings on external patients) ---
# Only add if patient doesn't already have a canonical MW rating
for round_num in [2, 3, 4]:
    print(f"\n  Source {round_num + 1}: Round {round_num} annotations (MW)")

    round_path = DATA / f"annotation_round{round_num}" / f"frequency_annotations_round{round_num}.csv"
    df = pd.read_csv(round_path)
    src_count = 0
    src_unmapped = 0
    src_skipped = 0

    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        expert_val = row.get("expert_annotation", None)

        # Check for skip
        skipped = False
        if pd.isna(expert_val) or str(expert_val).strip().lower() == "skip":
            skipped = True
            freq_hz = None
            no_pd = False
        else:
            try:
                freq = float(expert_val)
                no_pd = (freq == 0.0)
                freq_hz = None if freq == 0.0 else freq
            except (ValueError, TypeError):
                skipped = True
                freq_hz = None
                no_pd = False

        # Skip if this patient already has a canonical MW annotation
        if pid in mw_annotated_patients:
            continue

        # Find segment for this patient — use the first segment
        patient_segs = ext_patient_segments.get(pid, [])
        if not patient_segs:
            src_unmapped += 1
            continue

        # Use the first segment (these are 1-per-patient annotations)
        segment_id = patient_segs[0]

        if skipped:
            src_skipped += 1

        annotations.append({
            "segment_id": segment_id,
            "patient_id": pid,
            "rater": "MW",
            "frequency_hz": freq_hz,
            "no_pd": no_pd,
            "skipped": skipped,
            "spatial_extent": None,
            "spatial_channels": None,
            "annotation_date": None,
            "annotation_round": f"round{round_num}",
            "notes": None,
        })
        src_count += 1

    print(f"    Added {src_count} annotations ({src_skipped} skipped), {src_unmapped} unmapped")

# Build annotations DataFrame
ann_df = pd.DataFrame(annotations)
ann_df.to_csv(LABELS_OUT / "annotations.csv", index=False)
print(f"\n  annotations.csv written: {len(ann_df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: Build patients.csv
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 3: Building patients.csv")
print("=" * 70)

# Load exclusion info from canonical
canonical = pd.read_csv(DATA / "canonical_dataset" / "labels.csv")
excluded_info = {}
for _, row in canonical.iterrows():
    pid = row["patient_id"]
    excluded_info[pid] = {
        "excluded": row.get("excluded", False),
        "exclusion_reason": row.get("exclusion_reason", None),
    }

patients = []

all_patient_ids = seg_df["patient_id"].unique()
print(f"  Total unique patients: {len(all_patient_ids)}")

for pid in sorted(all_patient_ids):
    pid_segs = seg_df[seg_df["patient_id"] == pid]
    pid_anns = ann_df[ann_df["patient_id"] == pid]

    subtype = pid_segs["subtype"].iloc[0]
    n_segments = len(pid_segs)

    raters = sorted(pid_anns["rater"].unique().tolist()) if len(pid_anns) > 0 else []
    n_raters = len(raters)

    # Gold standard frequency
    mw_anns = pid_anns[(pid_anns["rater"] == "MW") & (~pid_anns["skipped"]) & (pid_anns["frequency_hz"].notna())]
    if len(mw_anns) > 0:
        gold_freq = mw_anns["frequency_hz"].mean()
    else:
        other_anns = pid_anns[(pid_anns["rater"] != "MW") & (~pid_anns["no_pd"]) & (pid_anns["frequency_hz"].notna())]
        if len(other_anns) > 0:
            gold_freq = other_anns["frequency_hz"].mean()
        else:
            gold_freq = None

    excl = excluded_info.get(pid, {})
    excluded = excl.get("excluded", False)
    excl_reason = excl.get("exclusion_reason", None)
    if pd.isna(excl_reason):
        excl_reason = None

    patients.append({
        "patient_id": pid,
        "subtype": subtype,
        "n_segments": n_segments,
        "n_raters": n_raters,
        "raters": ",".join(raters) if raters else None,
        "gold_standard_freq": gold_freq,
        "excluded": excluded,
        "exclusion_reason": excl_reason,
    })

pat_df = pd.DataFrame(patients)
pat_df.to_csv(LABELS_OUT / "patients.csv", index=False)
print(f"  patients.csv written: {len(pat_df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4: Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 4: Summary Statistics")
print("=" * 70)

# Count .mat files in eeg/
n_mat_files = len([f for f in os.listdir(EEG_OUT) if f.endswith(".mat")])
print(f"\n  Total .mat files in data/eeg/: {n_mat_files}")
print(f"  Total rows in segments.csv:    {len(seg_df)}")
print(f"  Total rows in annotations.csv: {len(ann_df)}")
print(f"  Total rows in patients.csv:    {len(pat_df)}")

print(f"\n  --- Segments by subtype ---")
print(seg_df["subtype"].value_counts().to_string())

print(f"\n  --- Segments by source ---")
print(seg_df["original_source"].value_counts().to_string())

print(f"\n  --- Annotations by rater ---")
print(ann_df["rater"].value_counts().to_string())

print(f"\n  --- Annotations by round ---")
print(ann_df["annotation_round"].value_counts().to_string())

print(f"\n  --- Patients by number of raters ---")
rater_counts = pat_df["n_raters"].value_counts().sort_index()
for n, count in rater_counts.items():
    print(f"    {n} rater(s): {count} patients")

print(f"\n  --- Segments by number of raters ---")
seg_rater_counts = ann_df.groupby("segment_id")["rater"].nunique()
# Also include segments with 0 raters
all_seg_ids = set(seg_df["segment_id"])
annotated_seg_ids = set(seg_rater_counts.index)
n_zero_raters = len(all_seg_ids - annotated_seg_ids)
print(f"    0 rater(s): {n_zero_raters} segments")
for n in sorted(seg_rater_counts.unique()):
    count = (seg_rater_counts == n).sum()
    print(f"    {n} rater(s): {count} segments")

print(f"\n  --- Patients with gold standard frequency ---")
has_gold = pat_df["gold_standard_freq"].notna().sum()
print(f"    {has_gold} / {len(pat_df)} patients have a gold standard frequency")

print(f"\n  --- Excluded patients ---")
print(f"    {pat_df['excluded'].sum()} excluded")

print("\n" + "=" * 70)
print("DONE! Data reorganization complete.")
print("=" * 70)
