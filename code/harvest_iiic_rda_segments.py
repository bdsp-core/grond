"""
Harvest LRDA and GRDA segments from the morgoth1 IIIC S3 dataset.

Steps:
1. Load the IIIC events spreadsheet
2. Parse expert vote distributions to find LRDA/GRDA-dominant segments
3. Filter to segments with >= 3 expert votes
4. LRDA: up to 5 segments per patient, up to 300 patients (~1000 segments)
5. GRDA: up to 3 segments per patient, up to 300 patients (~900 segments)
6. Download, extract central 10s, convert to bipolar, save

Usage:
    conda run -n foe python code/harvest_iiic_rda_segments.py

Requires AWS credentials in profile 'opendata'.
"""

import sys
import os
import json
import ast
import subprocess
import h5py
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

DATA_DIR = BASE / 'data'
EEG_DIR = DATA_DIR / 'eeg'
STAGING_DIR = Path('/tmp/rda_staging')
STAGING_DIR.mkdir(exist_ok=True)

DASHBOARD_PATH = BASE / 'results' / 'rda_harvest_dashboard.html'
HARVEST_MANIFEST = DATA_DIR / 'labels' / 'rda_harvest_manifest.json'

S3_IIIC_PREFIX = 's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/IIIC/'
S3_RAW_PREFIX = S3_IIIC_PREFIX + 'segments_raw/'
S3_EVENTS_FILE = S3_IIIC_PREFIX + 'list_events_20241129.xlsx'
LOCAL_EVENTS = Path('/tmp/iiic_events.xlsx')
AWS_PROFILE = 'opendata'

FS = 200

# Label order in the vote vector: [other, seizure, lpd, gpd, lrda, grda]
LABEL_NAMES = ['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']
LRDA_IDX = 4
GRDA_IDX = 5

# Harvest limits
MAX_SEGS_PER_PATIENT_LRDA = 5
MAX_SEGS_PER_PATIENT_GRDA = 3
MAX_PATIENTS_LRDA = 300
MAX_PATIENTS_GRDA = 300


def load_manifest():
    """Load the harvest manifest."""
    if HARVEST_MANIFEST.exists():
        with open(HARVEST_MANIFEST) as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    """Save the harvest manifest."""
    HARVEST_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(HARVEST_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)


def download_events_spreadsheet():
    """Download the IIIC events spreadsheet from S3."""
    if LOCAL_EVENTS.exists():
        print(f"  Events spreadsheet already at {LOCAL_EVENTS}")
        return True

    print(f"  Downloading events spreadsheet from S3...")
    result = subprocess.run(
        ['aws', 's3', 'cp', S3_EVENTS_FILE, str(LOCAL_EVENTS),
         '--profile', AWS_PROFILE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR downloading events: {result.stderr}")
        return False
    return LOCAL_EVENTS.exists()


def parse_votes(vote_str):
    """Parse a vote string like '[0, 0, 0, 0, 5, 1]' into a list of ints."""
    try:
        if isinstance(vote_str, (list, np.ndarray)):
            return [int(v) for v in vote_str]
        vote_str = str(vote_str).strip()
        votes = ast.literal_eval(vote_str)
        return [int(v) for v in votes]
    except Exception:
        return None


def get_dominant_label(votes):
    """Return the index of the dominant label (highest vote count)."""
    if votes is None or len(votes) < 6:
        return -1
    return int(np.argmax(votes))


def extract_patient_id_from_filename(filename):
    """Extract patient ID from filename like sub-S0001111201556_20150823212448.mat."""
    name = str(filename).replace('.mat', '')
    if name.startswith('sub-S0001'):
        pid = name[9:].split('_')[0]
        return pid
    # Try other formats
    if name.startswith('sub-'):
        pid = name[4:].split('_')[0]
        return pid
    return name


def download_file(filename):
    """Download a file from S3 to staging. Returns local path."""
    local_path = STAGING_DIR / filename
    subprocess.run(
        ['aws', 's3', 'cp', S3_RAW_PREFIX + filename, str(local_path),
         '--profile', AWS_PROFILE, '--quiet'],
        capture_output=True
    )
    return local_path if local_path.exists() else None


def extract_central_10s(mat_path):
    """Load a 10-min .mat file and extract the central 10-second segment.

    Returns (bipolar_segment, fs) where bipolar_segment is (18, 2000).
    Returns (None, None) on failure.
    """
    try:
        with h5py.File(str(mat_path), 'r') as f:
            data = np.array(f['data'], dtype=np.float64)
            fs = int(np.array(f['Fs']).ravel()[0])
    except Exception:
        try:
            mat = sio.loadmat(str(mat_path))
            data = mat['data'].astype(np.float64)
            fs = int(mat['Fs'].ravel()[0])
        except Exception:
            return None, None

    # Ensure shape is (channels, samples)
    if data.shape[0] > data.shape[1]:
        data = data.T

    n_channels, n_samples = data.shape

    # Extract central 10 seconds
    center = n_samples // 2
    half_win = 5 * fs
    start = max(0, center - half_win)
    end = min(n_samples, center + half_win)
    seg = data[:, start:end]

    # Convert to bipolar if monopolar (20 channels)
    if n_channels == 20:
        seg = np.array(fcn_getBanana(seg), dtype=np.float64)
    elif n_channels == 19:
        try:
            seg = np.array(fcn_getBanana(seg), dtype=np.float64)
        except Exception:
            return None, None
    elif n_channels != 18:
        return None, None

    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    return seg, fs


def save_segment(seg, fs, patient_id, seg_idx=0):
    """Save a 10-second bipolar segment as a .mat file in data/eeg/."""
    out_path = EEG_DIR / f'{patient_id}_seg{seg_idx:03d}.mat'
    sio.savemat(str(out_path), {
        'data': seg,
        'Fs': np.array([[fs]], dtype=np.float64),
    })
    return out_path


def build_dashboard(lrda_patients, grda_patients, lrda_segs, grda_segs,
                    processed, failed, total_selected):
    """Build a simple progress dashboard HTML."""
    pct = 100 * processed / max(total_selected, 1)
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>RDA Harvest Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px; background: #1a1a1a; color: #eee; font-family: 'Menlo', 'Consolas', monospace; }}
  h1 {{ color: #4488cc; margin-bottom: 5px; font-size: 22px; }}
  .subtitle {{ color: #888; margin-bottom: 20px; font-size: 13px; }}

  .status {{ padding: 12px 16px; background: #222; border-radius: 8px; margin-bottom: 18px;
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
  .status-item {{ text-align: center; min-width: 100px; }}
  .status-val {{ font-size: 22px; font-weight: bold; }}
  .status-label {{ font-size: 10px; color: #777; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}
  .green {{ color: #44cc88; }}
  .orange {{ color: #ff9800; }}
  .cyan {{ color: #44cccc; }}
  .red {{ color: #ff4444; }}
  .blue {{ color: #4488cc; }}
  .purple {{ color: #aa66cc; }}
  .dim {{ color: #666; }}

  .progress-container {{
    background: #333; border-radius: 6px; height: 20px; margin-bottom: 18px;
    overflow: hidden; position: relative;
  }}
  .progress-fill {{
    height: 100%; background: linear-gradient(90deg, #4488cc, #44cc88);
    border-radius: 6px; transition: width 0.5s;
  }}
  .progress-label {{
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; color: #eee; text-shadow: 0 0 4px #000;
  }}

  .type-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 18px; }}
  .type-box {{ background: #222; border-radius: 8px; padding: 16px; }}
  .type-box h3 {{ margin: 0 0 10px 0; font-size: 16px; }}
  .type-stat {{ display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #2a2a2a; }}
  .type-stat-label {{ color: #888; font-size: 12px; }}
  .type-stat-val {{ font-weight: bold; font-size: 14px; }}
</style>
</head>
<body>
<h1>RDA Harvest Dashboard</h1>
<p class="subtitle">Pulling LRDA/GRDA segments from morgoth1 IIIC folder</p>

<div class="status">
  <div class="status-item">
    <div class="status-val orange">{processed}</div>
    <div class="status-label">Processed</div>
  </div>
  <div class="status-item">
    <div class="status-val green">{lrda_segs + grda_segs}</div>
    <div class="status-label">Total Kept</div>
  </div>
  <div class="status-item">
    <div class="status-val red">{failed}</div>
    <div class="status-label">Failed</div>
  </div>
  <div class="status-item">
    <div class="status-val cyan">{total_selected}</div>
    <div class="status-label">Selected</div>
  </div>
</div>

<div class="progress-container">
  <div class="progress-fill" style="width:{pct:.1f}%;"></div>
  <div class="progress-label">{processed} / {total_selected} segments ({pct:.0f}%)</div>
</div>

<div class="type-grid">
  <div class="type-box">
    <h3 style="color: #4488cc;">LRDA</h3>
    <div class="type-stat">
      <span class="type-stat-label">Patients</span>
      <span class="type-stat-val blue">{lrda_patients} / {MAX_PATIENTS_LRDA}</span>
    </div>
    <div class="type-stat">
      <span class="type-stat-label">Segments</span>
      <span class="type-stat-val blue">{lrda_segs}</span>
    </div>
    <div class="type-stat">
      <span class="type-stat-label">Max per patient</span>
      <span class="type-stat-val dim">{MAX_SEGS_PER_PATIENT_LRDA}</span>
    </div>
  </div>
  <div class="type-box">
    <h3 style="color: #aa66cc;">GRDA</h3>
    <div class="type-stat">
      <span class="type-stat-label">Patients</span>
      <span class="type-stat-val purple">{grda_patients} / {MAX_PATIENTS_GRDA}</span>
    </div>
    <div class="type-stat">
      <span class="type-stat-label">Segments</span>
      <span class="type-stat-val purple">{grda_segs}</span>
    </div>
    <div class="type-stat">
      <span class="type-stat-label">Max per patient</span>
      <span class="type-stat-val dim">{MAX_SEGS_PER_PATIENT_GRDA}</span>
    </div>
  </div>
</div>

</body>
</html>"""

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(html)


def main():
    print("=" * 60)
    print("RDA Segment Harvester (morgoth1 IIIC - LRDA/GRDA)")
    print("=" * 60)

    # Ensure directories exist
    EEG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or download events spreadsheet
    print("\nLoading events spreadsheet...")
    if not download_events_spreadsheet():
        print("ERROR: Cannot get events spreadsheet. Exiting.")
        return

    df = pd.read_excel(str(LOCAL_EVENTS))
    print(f"  Loaded {len(df)} events")
    print(f"  Columns: {list(df.columns)}")

    # Load harvest manifest
    manifest = load_manifest()
    print(f"  Manifest: {len(manifest)} previously harvested RDA segments")

    # Get existing patient IDs in data/eeg/ to skip
    existing_eeg_files = set(p.stem.split('_seg')[0] for p in EEG_DIR.glob('*.mat'))
    manifest_pids = set(manifest.keys())
    print(f"  Already have {len(existing_eeg_files)} segment files locally")

    # Step 2: Parse votes and find LRDA/GRDA dominant segments
    print("\nParsing expert votes...")

    # Try to find the label/votes column
    label_col = None
    for col_name in ['label', 'labels', 'expert_consensus', 'votes', 'label_votes']:
        if col_name in df.columns:
            label_col = col_name
            break

    if label_col is None:
        # Try to find any column that looks like vote data
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ''
            if '[' in sample and ']' in sample:
                label_col = col
                break

    if label_col is None:
        print(f"  ERROR: Cannot find label/votes column in spreadsheet.")
        print(f"  Available columns: {list(df.columns)}")
        print(f"  First row sample:")
        if len(df) > 0:
            for col in df.columns:
                print(f"    {col}: {df[col].iloc[0]}")
        return

    print(f"  Using label column: '{label_col}'")

    # Try to find the filename column
    file_col = None
    for col_name in ['file_name', 'filename', 'seg_file', 'file', 'segment_file', 'eeg_file']:
        if col_name in df.columns:
            file_col = col_name
            break

    if file_col is None:
        # Fallback: any column with .mat in values
        for col in df.columns:
            sample = str(df[col].iloc[0]) if len(df) > 0 else ''
            if '.mat' in sample:
                file_col = col
                break

    if file_col is None:
        print(f"  ERROR: Cannot find filename column in spreadsheet.")
        return

    print(f"  Using filename column: '{file_col}'")

    # Try to find patient ID column
    pid_col = None
    for col_name in ['patient_id', 'subject_id', 'patient', 'subject', 'sub_id']:
        if col_name in df.columns:
            pid_col = col_name
            break

    if pid_col:
        print(f"  Using patient ID column: '{pid_col}'")

    # Parse all votes and classify
    lrda_by_patient = defaultdict(list)  # pid -> [(row_idx, votes, filename), ...]
    grda_by_patient = defaultdict(list)

    for idx, row in df.iterrows():
        votes = parse_votes(row[label_col])
        if votes is None or len(votes) < 6:
            continue

        n_experts = sum(votes)
        if n_experts < 3:
            continue

        dominant = get_dominant_label(votes)
        filename = str(row[file_col])
        if not filename.endswith('.mat'):
            filename = filename + '.mat'

        # Extract patient ID
        if pid_col and pd.notna(row.get(pid_col, None)):
            pid = str(row[pid_col])
        else:
            pid = extract_patient_id_from_filename(filename)

        # Skip patients already in our dataset
        if pid in existing_eeg_files or pid in manifest_pids:
            continue

        if dominant == LRDA_IDX:
            lrda_by_patient[pid].append((idx, votes, filename))
        elif dominant == GRDA_IDX:
            grda_by_patient[pid].append((idx, votes, filename))

    print(f"\n  LRDA candidates: {sum(len(v) for v in lrda_by_patient.values())} segments from {len(lrda_by_patient)} patients")
    print(f"  GRDA candidates: {sum(len(v) for v in grda_by_patient.values())} segments from {len(grda_by_patient)} patients")

    # Step 4-5: Select segments (limit per patient and total patients)
    selected = []  # (pid, votes, filename, subtype, seg_idx)

    # LRDA: up to 5 segments per patient, up to 300 patients
    lrda_pids = sorted(lrda_by_patient.keys())
    np.random.seed(42)
    np.random.shuffle(lrda_pids)
    lrda_pids = lrda_pids[:MAX_PATIENTS_LRDA]
    for pid in lrda_pids:
        segs = lrda_by_patient[pid][:MAX_SEGS_PER_PATIENT_LRDA]
        for seg_idx, (row_idx, votes, filename) in enumerate(segs):
            selected.append((pid, votes, filename, 'lrda', seg_idx))

    # GRDA: up to 3 segments per patient, up to 300 patients
    grda_pids = sorted(grda_by_patient.keys())
    np.random.shuffle(grda_pids)
    grda_pids = grda_pids[:MAX_PATIENTS_GRDA]
    for pid in grda_pids:
        segs = grda_by_patient[pid][:MAX_SEGS_PER_PATIENT_GRDA]
        for seg_idx, (row_idx, votes, filename) in enumerate(segs):
            selected.append((pid, votes, filename, 'grda', seg_idx))

    print(f"\n  Total selected: {len(selected)} segments")
    print(f"    LRDA: {sum(1 for s in selected if s[3] == 'lrda')} segments from {len(lrda_pids)} patients")
    print(f"    GRDA: {sum(1 for s in selected if s[3] == 'grda')} segments from {len(grda_pids)} patients")

    if not selected:
        print("\nNo segments to harvest! Exiting.")
        return

    # Step 6-7: Download and process
    print("\nBeginning harvest...")
    processed = 0
    failed = 0
    lrda_patients_done = set()
    grda_patients_done = set()
    lrda_segs_done = 0
    grda_segs_done = 0

    build_dashboard(0, 0, 0, 0, 0, 0, len(selected))

    for si, (pid, votes, filename, subtype, seg_idx) in enumerate(selected):
        # Download
        local_path = download_file(filename)
        if local_path is None:
            failed += 1
            processed += 1
            if (processed) % 20 == 0:
                print(f"  [{processed}/{len(selected)}] FAILED download: {filename}")
            continue

        try:
            # Extract central 10s
            seg, fs = extract_central_10s(local_path)
            if seg is None:
                failed += 1
                processed += 1
                local_path.unlink(missing_ok=True)
                continue

            # Save the 10s segment
            save_segment(seg, fs, pid, seg_idx)

            # Create manifest key that's unique per segment
            manifest_key = f"{pid}_seg{seg_idx:03d}" if seg_idx > 0 else pid
            manifest[manifest_key] = {
                'patient_id': pid,
                'subtype': subtype,
                'expert_votes': votes,
                'n_experts': sum(votes),
                's3_file': filename,
                'est_freq': None,
            }

            if subtype == 'lrda':
                lrda_patients_done.add(pid)
                lrda_segs_done += 1
            else:
                grda_patients_done.add(pid)
                grda_segs_done += 1

            # Save manifest periodically
            if (processed + 1) % 10 == 0:
                save_manifest(manifest)

            if (processed + 1) % 20 == 0:
                print(f"  [{processed+1}/{len(selected)}] KEPT {pid} ({subtype}) votes={votes} "
                      f"[LRDA: {lrda_segs_done} segs/{len(lrda_patients_done)} pts | "
                      f"GRDA: {grda_segs_done} segs/{len(grda_patients_done)} pts]")

        except Exception as e:
            failed += 1
            print(f"  FAILED {pid}: {e}")

        finally:
            if local_path and local_path.exists():
                local_path.unlink(missing_ok=True)

        processed += 1

        # Update dashboard every 20 files
        if processed % 20 == 0:
            build_dashboard(len(lrda_patients_done), len(grda_patients_done),
                            lrda_segs_done, grda_segs_done,
                            processed, failed, len(selected))

    # Final save and dashboard
    save_manifest(manifest)
    build_dashboard(len(lrda_patients_done), len(grda_patients_done),
                    lrda_segs_done, grda_segs_done,
                    processed, failed, len(selected))

    print(f"\n{'=' * 60}")
    print("RDA HARVEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {processed}")
    print(f"  Failed: {failed}")
    print(f"  LRDA: {lrda_segs_done} segments from {len(lrda_patients_done)} patients")
    print(f"  GRDA: {grda_segs_done} segments from {len(grda_patients_done)} patients")
    print(f"  Manifest entries: {len(manifest)}")
    print(f"  Dashboard: {DASHBOARD_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
