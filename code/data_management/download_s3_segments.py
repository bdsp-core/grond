"""
Download all EEG segments from S3 for GPD, LPD, LRDA, GRDA folders.
Integrates expert vote data from list_events_20241129.xlsx.
Updates patients.csv and segments.csv.
Generates a live HTML dashboard.
"""

import os
import re
import ast
import time
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io as sio

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EEG_DIR = os.path.join(PROJECT, "data", "eeg")
LABELS_DIR = os.path.join(PROJECT, "data", "labels")
RESULTS_DIR = os.path.join(PROJECT, "results")
PATIENTS_CSV = os.path.join(LABELS_DIR, "patients.csv")
SEGMENTS_CSV = os.path.join(LABELS_DIR, "segments.csv")
XLSX_PATH = os.path.join(LABELS_DIR, "list_events_20241129.xlsx")
DASHBOARD_HTML = os.path.join(RESULTS_DIR, "download_dashboard.html")

S3_BASE = "s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset"
AWS_PROFILE = "opendata"
MAX_WORKERS = 10

# ─── S3 file listing paths ────────────────────────────────────────────────────
S3_LISTINGS = {
    "lpd": "/tmp/s3_LPD.txt",
    "gpd": "/tmp/s3_GPD.txt",
    "lrda": "/tmp/s3_LRDA.txt",
    "grda": "/tmp/s3_GRDA.txt",
}
S3_FOLDER_MAP = {
    "lpd": "LPD",
    "gpd": "GPD",
    "lrda": "LRDA",
    "grda": "GRDA",
}

os.makedirs(EEG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_mrn(filename: str):
    """Extract MRN integer from sub-S0001{MRN}_ or sub-I0002{MRN}_ filenames."""
    fname = filename.replace(".mat", "")
    m = re.match(r"sub-[SI](\d+)_", fname)
    if m:
        digits = m.group(1)
        # First 4 digits are site prefix (0001 or 0002), rest is MRN
        try:
            return int(digits[4:])
        except ValueError:
            return None
    return None


def parse_vote_label(label_str: str):
    """Parse '[2,0,0,0,0,0]' -> list of 6 ints."""
    try:
        val = ast.literal_eval(str(label_str).strip())
        if isinstance(val, (list, tuple)) and len(val) == 6:
            return [int(x) for x in val]
    except Exception:
        pass
    return None


def aggregate_votes_for_mrn(mrn: int, xlsx_df: pd.DataFrame):
    """Sum votes across all rows for this MRN."""
    rows = xlsx_df[xlsx_df["bdsp_mrn"] == mrn]
    if rows.empty:
        return None
    totals = [0, 0, 0, 0, 0, 0]
    for lbl in rows["label ([other,seizure,lpd,gpd,lrda,grda])"]:
        parsed = parse_vote_label(lbl)
        if parsed:
            totals = [totals[i] + parsed[i] for i in range(6)]
    n_votes = sum(totals)
    if n_votes == 0:
        return None
    return totals, n_votes


def vote_dict_from_totals(totals, n_votes):
    keys = ["vote_other", "vote_seizure", "vote_lpd", "vote_gpd", "vote_lrda", "vote_grda"]
    d = {keys[i]: totals[i] for i in range(6)}
    d["n_expert_votes"] = n_votes
    d["vote_agreement"] = max(totals) / n_votes if n_votes > 0 else 0.0
    return d


def load_mat_data(filepath: str):
    """Load data and Fs from .mat file (handles both v5 and v7.3/HDF5 formats).
    Returns (data_2d_array, fs_float) or raises.
    """
    import h5py

    # Try scipy first (handles v5/v6 mat files)
    try:
        mat = sio.loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith("_")]
        data_key = None
        fs_key = None
        for k in keys:
            kl = k.lower()
            if kl in ("data", "eeg", "x"):
                data_key = k
            if kl in ("fs", "srate", "sample_rate", "sr"):
                fs_key = k
        if data_key is None:
            for k in keys:
                if isinstance(mat[k], np.ndarray) and mat[k].ndim >= 2:
                    data_key = k
                    break
        if data_key is None:
            raise ValueError(f"No data key found; keys={keys}")
        data = np.array(mat[data_key], dtype=np.float64)
        fs = float(mat[fs_key].flat[0]) if fs_key else 200.0
        return data, fs
    except NotImplementedError:
        pass  # v7.3 HDF5, fall through to h5py

    # Handle HDF5 (MATLAB v7.3) files
    with h5py.File(filepath, "r") as hf:
        keys = list(hf.keys())
        # Look for data key
        data_key = None
        fs_key = None
        for k in keys:
            kl = k.lower()
            if kl in ("data", "eeg", "x") and isinstance(hf[k], h5py.Dataset):
                data_key = k
            if kl in ("fs", "srate", "sample_rate", "sr") and isinstance(hf[k], h5py.Dataset):
                fs_key = k
        if data_key is None:
            # Try first 2D dataset
            for k in keys:
                if isinstance(hf[k], h5py.Dataset) and hf[k].ndim >= 2:
                    data_key = k
                    break
        if data_key is None:
            raise ValueError(f"No data dataset in HDF5 file; keys={keys}")
        data = np.array(hf[data_key], dtype=np.float64)
        fs = float(np.array(hf[fs_key]).flat[0]) if fs_key else 200.0
    return data, fs


def standardize_mat(filepath: str) -> bool:
    """Load mat (v5 or v7.3), extract (18, 2000) channels-first window, save with {data, Fs}.

    For long recordings (>10s), extracts a 10s window centered at 5 minutes (300s)
    into the recording, which corresponds to the labeled event time for the IIIC dataset.
    Returns True if successful.
    """
    try:
        data, fs = load_mat_data(filepath)

        # Ensure 2D
        if data.ndim != 2:
            print(f"  WARNING: unexpected ndim={data.ndim} in {filepath}")
            return False

        # Ensure channels-first: shape should be (channels, samples)
        # HDF5 MATLAB files store as (samples, channels), scipy as (channels, samples)
        # Detect: if larger dim is first → samples-first → transpose
        if data.shape[0] > data.shape[1]:
            data = data.T  # now (channels, samples)

        n_channels, n_samples = data.shape
        target_samples = 2000
        target_fs = 200.0

        # Resample if needed (shouldn't be needed for IIIC data at 200 Hz)
        if abs(fs - target_fs) > 1.0:
            # Simple integer resampling
            ratio = int(round(fs / target_fs))
            if ratio > 1:
                data = data[:, ::ratio]
                fs = target_fs
                n_samples = data.shape[1]

        # If recording is longer than 10s, extract 10s around the 5-minute mark
        # (300s into the recording = where the labeled event is in the IIIC dataset)
        if n_samples > target_samples:
            # Target center: 300s at target_fs = 60000 samples from start
            center_sample = int(300 * target_fs)
            half = target_samples // 2
            start = center_sample - half
            end = center_sample + half
            # Clamp to valid range
            if start < 0:
                start = 0
                end = target_samples
            if end > n_samples:
                end = n_samples
                start = max(0, end - target_samples)
            data = data[:, start:end]
            n_samples = data.shape[1]

        # Trim/pad channels to 18
        if n_channels > 18:
            data = data[:18, :]
        elif n_channels < 18:
            pad = np.zeros((18 - n_channels, data.shape[1]), dtype=data.dtype)
            data = np.vstack([data, pad])

        # Trim/pad samples to 2000
        if data.shape[1] > target_samples:
            data = data[:, :target_samples]
        elif data.shape[1] < target_samples:
            pad = np.zeros((18, target_samples - data.shape[1]), dtype=data.dtype)
            data = np.hstack([data, pad])

        assert data.shape == (18, 2000), f"Shape mismatch: {data.shape}"

        # Save as standard mat v5 with {data, Fs}
        sio.savemat(filepath, {"data": data, "Fs": np.array([[target_fs]])})
        return True
    except Exception as e:
        print(f"  ERROR standardizing {filepath}: {e}")
        return False


# ─── Dashboard ────────────────────────────────────────────────────────────────

def render_dashboard(state: dict) -> str:
    folders = ["lpd", "gpd", "lrda", "grda"]
    colors = {"lpd": "#4e79a7", "gpd": "#f28e2b", "lrda": "#59a14f", "grda": "#e15759"}
    rows = ""
    total_done = sum(state["done"][f] for f in folders)
    total_all = sum(state["total"][f] for f in folders)
    total_rem = total_all - total_done
    elapsed = time.time() - state["start_time"]
    speed = total_done / elapsed if elapsed > 0 else 0
    eta_sec = total_rem / speed if speed > 0 else 0
    eta_str = f"{int(eta_sec//3600)}h {int((eta_sec%3600)//60)}m" if eta_sec > 0 else "—"

    for f in folders:
        done = state["done"][f]
        total = state["total"][f]
        pct = 100 * done / total if total > 0 else 0
        c = colors[f]
        rows += f"""
        <tr>
          <td style="font-weight:bold;color:{c}">{f.upper()}</td>
          <td>{done:,}</td><td>{total:,}</td><td>{total-done:,}</td>
          <td>
            <div style="background:#eee;border-radius:4px;width:300px;height:18px">
              <div style="background:{c};width:{pct:.1f}%;height:18px;border-radius:4px"></div>
            </div>
            {pct:.1f}%
          </td>
          <td>{state['errors'][f]:,}</td>
        </tr>"""

    status_color = "#2a9d8f" if "complete" in state["status"].lower() else "#e9c46a"
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="5">
  <title>S3 EEG Download Dashboard</title>
  <style>
    body {{font-family:monospace;background:#1a1a2e;color:#eee;padding:20px}}
    h1 {{color:#a8dadc}}
    table {{border-collapse:collapse;width:100%}}
    th,td {{padding:8px 14px;text-align:left;border-bottom:1px solid #333}}
    th {{color:#a8dadc;background:#16213e}}
    .status {{background:#16213e;padding:12px;border-left:4px solid {status_color};margin:10px 0;font-size:1.1em}}
    .stat {{display:inline-block;background:#16213e;padding:10px 20px;margin:6px;border-radius:6px}}
    .stat-val {{font-size:1.6em;color:#a8dadc}}
  </style>
</head>
<body>
  <h1>S3 EEG Download Dashboard</h1>
  <div class="status">{state['status']}</div>
  <div>
    <span class="stat">Total Downloaded<br><span class="stat-val">{total_done:,}</span></span>
    <span class="stat">Total Files<br><span class="stat-val">{total_all:,}</span></span>
    <span class="stat">Remaining<br><span class="stat-val">{total_rem:,}</span></span>
    <span class="stat">Speed<br><span class="stat-val">{speed:.1f} files/s</span></span>
    <span class="stat">ETA<br><span class="stat-val">{eta_str}</span></span>
    <span class="stat">Elapsed<br><span class="stat-val">{int(elapsed//60)}m {int(elapsed%60)}s</span></span>
  </div>
  <table>
    <tr><th>Folder</th><th>Done</th><th>Total</th><th>Remaining</th><th>Progress</th><th>Errors</th></tr>
    {rows}
  </table>
  <p style="color:#888;font-size:0.85em">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Auto-refreshes every 5s</p>
</body>
</html>"""
    return html


def save_dashboard(state: dict):
    try:
        with open(DASHBOARD_HTML, "w") as f:
            f.write(render_dashboard(state))
    except Exception as e:
        print(f"  Dashboard save error: {e}")


# ─── Download one file ────────────────────────────────────────────────────────

def is_already_standardized(filepath: str) -> bool:
    """Check if file is already in (18, 2000) format with {data, Fs} keys."""
    try:
        mat = sio.loadmat(filepath)
        keys = [k for k in mat.keys() if not k.startswith("_")]
        if "data" in keys and "Fs" in keys:
            data = mat["data"]
            if data.shape == (18, 2000):
                return True
    except Exception:
        pass
    return False


def download_file(subtype: str, filename: str):
    """Download one file from S3. Returns (filename, success, message)."""
    dest = os.path.join(EEG_DIR, filename)
    if os.path.exists(dest):
        # File exists — check if already standardized, if not standardize it
        if is_already_standardized(dest):
            return filename, True, "skipped"
        # Needs standardization (downloaded in a previous run that failed to standardize)
        ok = standardize_mat(dest)
        if ok:
            return filename, True, "standardized_existing"
        else:
            return filename, False, "standardize_failed_existing"

    s3_folder = S3_FOLDER_MAP[subtype]
    s3_uri = f"{S3_BASE}/{s3_folder}/segments_raw/{filename}"
    cmd = ["aws", "s3", "cp", "--profile", AWS_PROFILE, s3_uri, dest]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(dest):
            # Standardize
            ok = standardize_mat(dest)
            if not ok:
                return filename, False, "standardize_failed"
            return filename, True, "downloaded"
        else:
            return filename, False, result.stderr.strip()[:120]
    except subprocess.TimeoutExpired:
        return filename, False, "timeout"
    except Exception as e:
        return filename, False, str(e)[:120]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== S3 EEG Download Script ===")
    print(f"EEG dir: {EEG_DIR}")
    print(f"Dashboard: {DASHBOARD_HTML}")

    # Load xlsx vote data
    print("\nLoading expert vote data...")
    xlsx_df = pd.read_excel(XLSX_PATH)
    xlsx_df["bdsp_mrn"] = xlsx_df["bdsp_mrn"].astype("Int64")
    # Build MRN -> aggregated votes mapping (pre-computed for speed)
    mrn_votes = {}
    for mrn, grp in xlsx_df.groupby("bdsp_mrn"):
        totals = [0, 0, 0, 0, 0, 0]
        for lbl in grp["label ([other,seizure,lpd,gpd,lrda,grda])"]:
            parsed = parse_vote_label(lbl)
            if parsed:
                totals = [totals[i] + parsed[i] for i in range(6)]
        n_votes = sum(totals)
        if n_votes > 0:
            mrn_votes[int(mrn)] = vote_dict_from_totals(totals, n_votes)
    print(f"  {len(mrn_votes):,} MRNs with expert votes")

    # Load S3 file listings
    all_files = {}  # subtype -> list of filenames
    for subtype, listing_path in S3_LISTINGS.items():
        files = [l.strip() for l in open(listing_path).readlines() if l.strip()]
        all_files[subtype] = files
        print(f"  {subtype.upper()}: {len(files):,} files in listing")

    # Check already downloaded
    existing = set(os.listdir(EEG_DIR))
    print(f"\nExisting files in data/eeg/: {len(existing):,}")

    # Build state for dashboard
    state = {
        "total": {s: len(f) for s, f in all_files.items()},
        "done": {s: sum(1 for f in files if f in existing) for s, files in all_files.items()},
        "errors": {s: 0 for s in all_files},
        "status": "Starting downloads...",
        "start_time": time.time(),
    }
    save_dashboard(state)

    # ── Download phase ─────────────────────────────────────────────────────────
    download_lock = threading.Lock()
    file_counter = [0]  # mutable for closure

    def on_result(subtype, filename, success, msg):
        with download_lock:
            if success:
                state["done"][subtype] += 1
            else:
                state["errors"][subtype] += 1
                if msg != "skipped":
                    print(f"  ERROR [{subtype}] {filename}: {msg}")
            file_counter[0] += 1
            if file_counter[0] % 50 == 0:
                total_done = sum(state["done"].values())
                total_all = sum(state["total"].values())
                elapsed = time.time() - state["start_time"]
                speed = total_done / elapsed if elapsed > 0 else 0
                print(f"  [{file_counter[0]:,}/{total_all:,}] done={total_done:,}  speed={speed:.1f} f/s")
            if file_counter[0] % 10 == 0:
                state["status"] = (
                    f"Downloading... {sum(state['done'].values()):,}/{sum(state['total'].values()):,} files"
                )
                save_dashboard(state)

    print("\nStarting concurrent downloads (up to 10 parallel)...")
    for subtype, files in all_files.items():
        to_download = [f for f in files if f not in existing]
        skipped = len(files) - len(to_download)
        print(f"  {subtype.upper()}: {len(to_download):,} to download ({skipped:,} already present)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for subtype, files in all_files.items():
            for filename in files:
                future = executor.submit(download_file, subtype, filename)
                futures[future] = (subtype, filename)

        for future in as_completed(futures):
            subtype, filename = futures[future]
            try:
                fname, success, msg = future.result()
                on_result(subtype, fname, success, msg)
            except Exception as e:
                on_result(subtype, filename, False, str(e))

    state["status"] = "Downloads complete! Updating patients.csv and segments.csv..."
    save_dashboard(state)
    print("\nDownloads complete.")

    # ── Update patients.csv ───────────────────────────────────────────────────
    print("\nUpdating patients.csv...")
    patients_df = pd.read_csv(PATIENTS_CSV, dtype={"patient_id": str})

    # Add new vote columns if missing
    vote_cols = ["n_expert_votes", "vote_other", "vote_seizure", "vote_lpd",
                 "vote_gpd", "vote_lrda", "vote_grda", "vote_agreement", "label_source"]
    for col in vote_cols:
        if col not in patients_df.columns:
            patients_df[col] = None

    # Fill vote data for existing patients where we can match
    # Existing patient_id may be numeric MRN or sub-* style
    existing_ids = set(patients_df["patient_id"].astype(str))

    # Build set of all sub-* patient_ids already in CSV
    existing_sub_ids = {pid for pid in existing_ids if pid.startswith("sub-")}
    existing_numeric_ids = {pid for pid in existing_ids if not pid.startswith("sub-")}

    # Fill votes for existing patients with numeric IDs
    for idx, row in patients_df.iterrows():
        pid = str(row["patient_id"])
        mrn = None
        if re.match(r"^\d+$", pid):
            try:
                mrn = int(pid)
            except ValueError:
                pass
        elif pid.startswith("sub-"):
            mrn = extract_mrn(pid)

        if mrn and mrn in mrn_votes:
            vd = mrn_votes[mrn]
            for k, v in vd.items():
                if pd.isna(patients_df.at[idx, k]) or patients_df.at[idx, k] is None:
                    patients_df.at[idx, k] = v
            if pd.isna(patients_df.at[idx, "label_source"]) or patients_df.at[idx, "label_source"] is None:
                patients_df.at[idx, "label_source"] = "expert_majority"

    # Build new patient rows for S3 files not yet in patients.csv
    new_patient_rows = []
    for subtype, files in all_files.items():
        for filename in files:
            pid = filename.replace(".mat", "")
            if pid in existing_sub_ids:
                continue  # already exists

            mrn = extract_mrn(filename)
            vd = mrn_votes.get(mrn, {}) if mrn else {}
            n_votes = vd.get("n_expert_votes", 0)
            label_src = "expert_majority" if n_votes > 0 else "folder_assignment"

            row = {
                "patient_id": pid,
                "subtype": subtype,
                "subtype_original": subtype,
                "n_segments": 1,
                "n_raters": None,
                "raters": None,
                "gold_standard_freq": None,
                "gold_standard_freq_original": None,
                "excluded": False,
                "exclusion_reason": None,
                "laterality": None,
                "laterality_original": None,
                "subtype_rater": "iiic_dataset",
                "laterality_rater": None,
                "n_expert_votes": n_votes,
                "vote_other": vd.get("vote_other", 0),
                "vote_seizure": vd.get("vote_seizure", 0),
                "vote_lpd": vd.get("vote_lpd", 0),
                "vote_gpd": vd.get("vote_gpd", 0),
                "vote_lrda": vd.get("vote_lrda", 0),
                "vote_grda": vd.get("vote_grda", 0),
                "vote_agreement": vd.get("vote_agreement", 0.0),
                "label_source": label_src,
            }
            new_patient_rows.append(row)

    if new_patient_rows:
        new_df = pd.DataFrame(new_patient_rows)
        patients_df = pd.concat([patients_df, new_df], ignore_index=True)
        print(f"  Added {len(new_patient_rows):,} new patient rows")
    else:
        print("  No new patients to add")

    patients_df.to_csv(PATIENTS_CSV, index=False)
    print(f"  Saved {len(patients_df):,} patients to {PATIENTS_CSV}")

    # ── Update segments.csv ───────────────────────────────────────────────────
    print("\nUpdating segments.csv...")
    segments_df = pd.read_csv(SEGMENTS_CSV, dtype={"segment_id": str, "patient_id": str})
    existing_seg_ids = set(segments_df["segment_id"].astype(str))
    existing_seg_matfiles = set(segments_df["mat_file"].astype(str))

    new_seg_rows = []
    for subtype, files in all_files.items():
        for filename in files:
            pid = filename.replace(".mat", "")
            mat_filename = filename
            seg_id = pid  # one segment per S3 file, no _seg000 suffix

            if seg_id in existing_seg_ids or mat_filename in existing_seg_matfiles:
                continue

            # Try to load metadata from file
            filepath = os.path.join(EEG_DIR, filename)
            fs = 200.0
            n_channels = 18
            duration = 10.0
            if os.path.exists(filepath):
                try:
                    mat = sio.loadmat(filepath)
                    keys = [k for k in mat.keys() if not k.startswith("_")]
                    for k in keys:
                        if k.lower() in ("fs", "srate"):
                            fs = float(mat[k].flat[0])
                        if k.lower() in ("data", "eeg", "x"):
                            arr = mat[k]
                            if arr.ndim == 2:
                                n_channels = min(arr.shape)
                                n_samp = max(arr.shape)
                                duration = n_samp / fs
                except Exception:
                    pass

            row = {
                "segment_id": seg_id,
                "patient_id": pid,
                "subtype": subtype,
                "subtype_source": "s3_folder",
                "mat_file": mat_filename,
                "duration_sec": duration,
                "fs": fs,
                "n_channels": n_channels,
                "montage": "monopolar",
                "original_source": "s3_morgoth",
                "original_filename": filename,
            }
            new_seg_rows.append(row)

    if new_seg_rows:
        new_seg_df = pd.DataFrame(new_seg_rows)
        segments_df = pd.concat([segments_df, new_seg_df], ignore_index=True)
        print(f"  Added {len(new_seg_rows):,} new segment rows")
    else:
        print("  No new segments to add")

    segments_df.to_csv(SEGMENTS_CSV, index=False)
    print(f"  Saved {len(segments_df):,} segments to {SEGMENTS_CSV}")

    # ── Final dashboard ───────────────────────────────────────────────────────
    state["status"] = (
        f"COMPLETE! {sum(state['done'].values()):,} files downloaded/verified. "
        f"patients.csv: {len(patients_df):,} rows. segments.csv: {len(segments_df):,} rows."
    )
    save_dashboard(state)
    print(f"\nDone! Dashboard: {DASHBOARD_HTML}")
    print(f"Errors: { {s: state['errors'][s] for s in state['errors']} }")


if __name__ == "__main__":
    main()
