"""
Harvest BIPD segments from the morgoth2 S3 dataset.

For each 10-minute BIPD file on S3:
1. Download it
2. Extract the central 10-second segment
3. Convert to 18-channel bipolar montage
4. Save to data/eeg/
5. Track in manifest: data/labels/bipd_harvest_manifest.json
6. Delete the 10-minute file locally

Usage:
    conda run -n foe python code/harvest_bipd_segments.py

Requires AWS credentials in profile 'opendata'.
"""

import sys
import os
import json
import subprocess
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

DATA_DIR = BASE / 'data'
EEG_DIR = DATA_DIR / 'eeg'
STAGING_DIR = Path('/tmp/bipd_staging')
STAGING_DIR.mkdir(exist_ok=True)

DASHBOARD_PATH = BASE / 'results' / 'bipd_harvest_dashboard.html'
HARVEST_MANIFEST = DATA_DIR / 'labels' / 'bipd_harvest_manifest.json'
BIPD_LIST_CSV = DATA_DIR / '_archive' / 'bipd' / 'list_bipds_20250613.csv'

S3_BUCKET = 's3://bdsp-opendata-credentialed/morgoth2/data/internal_dataset/BIPD/'
S3_RAW_PREFIX = S3_BUCKET + 'segments_raw/'
S3_BIPD_LIST = S3_BUCKET + 'list_bipds_20250613.csv'
AWS_PROFILE = 'opendata'

FS = 200


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


def download_bipd_list():
    """Download the BIPD list CSV from S3."""
    BIPD_LIST_CSV.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ['aws', 's3', 'cp', S3_BIPD_LIST, str(BIPD_LIST_CSV),
         '--profile', AWS_PROFILE],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Warning: failed to download BIPD list: {result.stderr}")
    return BIPD_LIST_CSV.exists()


def list_s3_files():
    """List all .mat files in the S3 BIPD segments_raw directory."""
    result = subprocess.run(
        ['aws', 's3', 'ls', S3_RAW_PREFIX, '--profile', AWS_PROFILE],
        capture_output=True, text=True
    )
    files = []
    for line in result.stdout.strip().split('\n'):
        if line.strip() and line.strip().endswith('.mat'):
            parts = line.strip().split()
            fname = parts[-1]
            files.append(fname)
    return files


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
        # Try HDF5 first (v7.3 .mat files)
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

    # Clean up NaN/Inf
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

    return seg, fs


def save_segment(seg, fs, patient_id):
    """Save a 10-second bipolar segment as a .mat file in data/eeg/."""
    out_path = EEG_DIR / f'{patient_id}_seg000.mat'
    sio.savemat(str(out_path), {
        'data': seg,
        'Fs': np.array([[fs]], dtype=np.float64),
    })
    return out_path


def extract_patient_id(filename):
    """Extract patient ID from S3 filename like sub-S0001111201556_20150823212448.mat."""
    name = filename.replace('.mat', '')
    if name.startswith('sub-S0001'):
        pid = name[9:].split('_')[0]
        return pid
    return name


def build_dashboard(processed, kept, skipped, failed, total_files):
    """Build a simple progress dashboard HTML."""
    pct = 100 * processed / max(total_files, 1)
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>BIPD Harvest Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px; background: #1a1a1a; color: #eee; font-family: 'Menlo', 'Consolas', monospace; }}
  h1 {{ color: #cc44cc; margin-bottom: 5px; font-size: 22px; }}
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
  .dim {{ color: #666; }}

  .progress-container {{
    background: #333; border-radius: 6px; height: 20px; margin-bottom: 18px;
    overflow: hidden; position: relative;
  }}
  .progress-fill {{
    height: 100%; background: linear-gradient(90deg, #cc44cc, #44cc88);
    border-radius: 6px; transition: width 0.5s;
  }}
  .progress-label {{
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; color: #eee; text-shadow: 0 0 4px #000;
  }}
</style>
</head>
<body>
<h1>BIPD Harvest Dashboard</h1>
<p class="subtitle">Pulling BIPD segments from morgoth2 S3</p>

<div class="status">
  <div class="status-item">
    <div class="status-val orange">{processed}</div>
    <div class="status-label">Processed</div>
  </div>
  <div class="status-item">
    <div class="status-val green">{kept}</div>
    <div class="status-label">Kept</div>
  </div>
  <div class="status-item">
    <div class="status-val dim">{skipped}</div>
    <div class="status-label">Skipped (existing)</div>
  </div>
  <div class="status-item">
    <div class="status-val red">{failed}</div>
    <div class="status-label">Failed</div>
  </div>
  <div class="status-item">
    <div class="status-val cyan">{total_files}</div>
    <div class="status-label">Total on S3</div>
  </div>
</div>

<div class="progress-container">
  <div class="progress-fill" style="width:{pct:.1f}%;"></div>
  <div class="progress-label">{processed} / {total_files} files ({pct:.0f}%)</div>
</div>

</body>
</html>"""

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(html)


def main():
    print("=" * 60)
    print("BIPD Segment Harvester (morgoth2)")
    print("=" * 60)

    # Ensure directories exist
    EEG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Download the BIPD list CSV
    print("\nDownloading BIPD list CSV...")
    download_bipd_list()
    if BIPD_LIST_CSV.exists():
        print(f"  Saved to {BIPD_LIST_CSV}")
    else:
        print("  Warning: BIPD list CSV not downloaded (continuing anyway)")

    # Load harvest manifest
    manifest = load_manifest()
    print(f"  Manifest: {len(manifest)} previously harvested BIPDs")

    # Step 2: List all .mat files on S3
    print("\nListing S3 files...")
    s3_files = list_s3_files()
    print(f"  Found {len(s3_files)} files on S3")

    if not s3_files:
        print("No files found! Check S3 access.")
        return

    # Get existing patient IDs to skip
    existing_eeg_files = set(p.stem.split('_seg')[0] for p in EEG_DIR.glob('*.mat'))
    print(f"  Already have {len(existing_eeg_files)} segment files locally")

    processed = 0
    kept = 0
    skipped = 0
    failed = 0

    # Build initial dashboard
    build_dashboard(0, 0, 0, 0, len(s3_files))

    for fi, filename in enumerate(s3_files):
        patient_id = extract_patient_id(filename)

        # Skip if we already have this patient (in eeg dir or manifest)
        if patient_id in existing_eeg_files or patient_id in manifest:
            skipped += 1
            processed += 1
            continue

        # Download
        local_path = download_file(filename)
        if local_path is None:
            failed += 1
            processed += 1
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
            save_segment(seg, fs, patient_id)
            kept += 1
            existing_eeg_files.add(patient_id)

            # Record in manifest
            manifest[patient_id] = {
                'subtype': 'bipd',
                's3_file': filename,
                'est_freq': None,
            }
            save_manifest(manifest)

            if (processed + 1) % 20 == 0 or kept <= 5:
                print(f"  [{processed+1}/{len(s3_files)}] KEPT {patient_id} (total kept: {kept})")

        except Exception as e:
            failed += 1
            print(f"  FAILED {patient_id}: {e}")

        finally:
            # Always delete the 10-minute file locally
            if local_path and local_path.exists():
                local_path.unlink(missing_ok=True)

        processed += 1

        # Update dashboard every 20 files
        if processed % 20 == 0:
            build_dashboard(processed, kept, skipped, failed, len(s3_files))
            print(f"  Progress: {processed}/{len(s3_files)} processed, {kept} kept, {failed} failed")

    # Final dashboard update
    build_dashboard(processed, kept, skipped, failed, len(s3_files))

    print(f"\n{'=' * 60}")
    print("BIPD HARVEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {processed}")
    print(f"  Kept: {kept}")
    print(f"  Skipped (already had): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Dashboard: {DASHBOARD_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
