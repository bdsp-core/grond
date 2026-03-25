"""
Harvest LPD segments from the morgoth1 S3 dataset to balance the frequency distribution.

For each 10-minute LPD file on S3:
1. Download it
2. Extract the central 10-second segment
3. Convert to bipolar montage
4. Estimate frequency using our pre-trained Ridge model
5. If the frequency bin needs more cases, keep the 10s segment
6. Update a live HTML dashboard showing histogram progress
7. Delete the 10-minute file locally (it stays on S3)

Usage:
    conda run -n foe python code/harvest_lpd_segments.py

Requires AWS credentials in profile 'opendata'.
"""

import sys
import os
import json
import time
import base64
import io
import subprocess
import h5py
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana, compute_pointiness_trace
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

DATA_DIR = BASE / 'data'
EEG_DIR = DATA_DIR / 'eeg'
STAGING_DIR = Path('/tmp/lpd_staging')
STAGING_DIR.mkdir(exist_ok=True)

DASHBOARD_PATH = BASE / 'results' / 'harvest_dashboard.html'

S3_PREFIX = 's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/LPD/segments_raw/'
AWS_PROFILE = 'opendata'

FS = 200
FREQ_LO, FREQ_HI = 0.3, 3.5
LOWPASS_HZ = 15.0


def estimate_frequency_fft(segment, fs=FS):
    """Estimate frequency using raw FFT peak on pointiness traces.

    Unlike the Ridge model, this does not shrink toward the mean —
    it just finds the dominant periodic frequency in each channel
    and returns the median. Unbiased estimator for triage/binning.
    """
    n_channels = segment.shape[0]

    # Lowpass filter
    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(segment)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, segment[ch])
        except ValueError:
            seg_lp[ch] = segment[ch]

    # Compute pointiness traces and take FFT peak per channel
    fft_freqs = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=max(1, int(0.02 * fs)))

        n = len(pt)
        if n < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(pt - np.mean(pt)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freqs_sub = freqs[mask]
        if np.max(fft_sub) > 0:
            fft_freqs.append(freqs_sub[np.argmax(fft_sub)])

    if fft_freqs:
        return float(np.median(fft_freqs))
    return np.nan


# Frequency bins and targets
BINS = [
    (0.25, 0.5,  '0.25-0.5 Hz'),
    (0.5,  1.0,  '0.5-1.0 Hz'),
    (1.0,  1.5,  '1.0-1.5 Hz'),
    (1.5,  2.0,  '1.5-2.0 Hz'),
    (2.0,  2.5,  '2.0-2.5 Hz'),
    (2.5,  3.0,  '2.5-3.0 Hz'),
    (3.0,  5.0,  '3.0+ Hz'),
]
TARGET_PER_BIN = 100


def get_bin_index(freq):
    """Return the bin index for a frequency, or -1 if outside all bins."""
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= freq < hi:
            return i
    return -1


HARVEST_MANIFEST = DATA_DIR / 'labels' / 'harvest_manifest.json'


def load_harvest_manifest():
    """Load the harvest manifest (tracks previously harvested files and their freq estimates)."""
    if HARVEST_MANIFEST.exists():
        with open(HARVEST_MANIFEST) as f:
            return json.load(f)
    return {}


def save_harvest_manifest(manifest):
    """Save the harvest manifest."""
    with open(HARVEST_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)


def count_existing(harvest_manifest):
    """Count existing LPD patients per frequency bin from patients.csv + harvest manifest."""
    import csv
    counts = [0] * len(BINS)
    seen_pids = set()

    # Count from patients.csv
    with open(DATA_DIR / 'labels' / 'patients.csv') as f:
        for row in csv.DictReader(f):
            if row['subtype'] != 'lpd' or row['excluded'] == 'True':
                continue
            freq_str = row.get('gold_standard_freq', '')
            if not freq_str:
                continue
            try:
                freq = float(freq_str)
            except ValueError:
                continue
            if freq <= 0:
                continue
            bi = get_bin_index(freq)
            if bi >= 0:
                counts[bi] += 1
                seen_pids.add(row['patient_id'])

    # Count from harvest manifest (previously harvested but not yet in patients.csv)
    for pid, info in harvest_manifest.items():
        if pid in seen_pids:
            continue
        freq = info.get('est_freq', 0)
        bi = get_bin_index(freq)
        if bi >= 0:
            counts[bi] += 1
            seen_pids.add(pid)

    return counts


def list_s3_files():
    """List all .mat files in the S3 LPD directory."""
    result = subprocess.run(
        ['aws', 's3', 'ls', S3_PREFIX, '--profile', AWS_PROFILE],
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
        ['aws', 's3', 'cp', S3_PREFIX + filename, str(local_path),
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
        # Might be monopolar without EKG — try anyway
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
    # Remove .mat extension
    name = filename.replace('.mat', '')
    # Extract the numeric ID from sub-S0001XXXXXXXXXXX
    if name.startswith('sub-S0001'):
        pid = name[9:].split('_')[0]  # Get digits after sub-S0001, before _timestamp
        return pid
    return name


def build_dashboard(existing_counts, new_counts, processed, kept, skipped, failed,
                    total_files, recent_cases=None):
    """Build/update the harvest dashboard HTML."""
    if recent_cases is None:
        recent_cases = []

    bin_labels = [b[2] for b in BINS]
    total_counts = [e + n for e, n in zip(existing_counts, new_counts)]

    # Build histogram data
    hist_data = json.dumps({
        'labels': bin_labels,
        'existing': existing_counts,
        'new': new_counts,
        'total': total_counts,
        'target': TARGET_PER_BIN,
    })

    recent_json = json.dumps(recent_cases[-20:])  # Keep last 20

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>LPD Harvest Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px; background: #1a1a1a; color: #eee; font-family: 'Menlo', 'Consolas', monospace; }}
  h1 {{ color: #ff9800; margin-bottom: 5px; font-size: 22px; }}
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
    height: 100%; background: linear-gradient(90deg, #ff9800, #44cc88);
    border-radius: 6px; transition: width 0.5s;
  }}
  .progress-label {{
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; color: #eee; text-shadow: 0 0 4px #000;
  }}

  .hist-container {{ background: #222; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
  .hist-container h3 {{ margin: 0 0 12px 0; color: #aaa; font-size: 14px; }}
  .bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .bar-label {{ width: 120px; font-size: 12px; color: #aaa; text-align: right; padding-right: 10px; }}
  .bar-track {{ flex: 1; height: 28px; background: #333; border-radius: 4px; position: relative; overflow: hidden; }}
  .bar-existing {{ height: 100%; background: #2a5a2a; position: absolute; left: 0; }}
  .bar-new {{ height: 100%; background: #44cc88; position: absolute; }}
  .bar-target {{ position: absolute; height: 100%; border-right: 2px dashed #ff9800; }}
  .bar-count {{ position: absolute; right: 8px; top: 4px; font-size: 12px; font-weight: bold; color: #eee; z-index: 1; }}
  .bar-count-dim {{ color: #888; }}

  .recent {{ background: #222; border-radius: 8px; padding: 16px; }}
  .recent h3 {{ margin: 0 0 12px 0; color: #aaa; font-size: 14px; }}
  .recent-item {{ display: flex; gap: 12px; align-items: center; padding: 4px 0; border-bottom: 1px solid #2a2a2a; font-size: 12px; }}
  .recent-freq {{ font-weight: bold; min-width: 60px; }}
  .recent-kept {{ color: #44cc88; }}
  .recent-skipped {{ color: #666; }}
  .recent-img {{ max-height: 80px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>LPD Harvest Dashboard</h1>
<p class="subtitle">Pulling LPD segments from morgoth1 S3 to balance frequency distribution | Target: {TARGET_PER_BIN} per bin</p>

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
    <div class="status-label">Skipped (bin full)</div>
  </div>
  <div class="status-item">
    <div class="status-val red">{failed}</div>
    <div class="status-label">Failed</div>
  </div>
  <div class="status-item">
    <div class="status-val cyan">{total_files}</div>
    <div class="status-label">Total on S3</div>
  </div>
  <div class="status-item">
    <div class="status-val {'green' if all(t >= TARGET_PER_BIN for t in total_counts) else 'orange'}">{sum(total_counts)}</div>
    <div class="status-label">Total LPD</div>
  </div>
  <div class="status-item">
    <div class="status-val dim">{sum(1 for t in total_counts if t >= TARGET_PER_BIN)}/{len(BINS)}</div>
    <div class="status-label">Bins Complete</div>
  </div>
</div>

<div class="progress-container">
  <div class="progress-fill" style="width:{100*processed/max(total_files,1):.1f}%;"></div>
  <div class="progress-label">{processed} / {total_files} files tried ({100*processed/max(total_files,1):.0f}%)</div>
</div>

<div class="hist-container">
  <h3>Frequency Distribution (existing + new | target = {TARGET_PER_BIN})</h3>
"""

    max_count = max(max(total_counts), TARGET_PER_BIN) * 1.1
    for i, (label, existing, new, total) in enumerate(
            zip(bin_labels, existing_counts, new_counts, total_counts)):
        ex_pct = existing / max_count * 100
        new_pct = new / max_count * 100
        target_pct = TARGET_PER_BIN / max_count * 100
        complete = '✓' if total >= TARGET_PER_BIN else ''
        count_cls = '' if total >= TARGET_PER_BIN else 'bar-count-dim'

        html += f"""  <div class="bar-row">
    <span class="bar-label">{label}</span>
    <div class="bar-track">
      <div class="bar-existing" style="width:{ex_pct:.1f}%;"></div>
      <div class="bar-new" style="left:{ex_pct:.1f}%; width:{new_pct:.1f}%;"></div>
      <div class="bar-target" style="left:{target_pct:.1f}%;"></div>
      <span class="bar-count {count_cls}">{total} {complete}</span>
    </div>
  </div>
"""

    html += """</div>

<div class="recent">
  <h3>Recent cases</h3>
"""

    for case in reversed(recent_cases[-20:]):
        status_cls = 'recent-kept' if case.get('kept') else 'recent-skipped'
        status_txt = 'KEPT' if case.get('kept') else 'SKIP'
        html += f"""  <div class="recent-item">
    <span class="{status_cls}">{status_txt}</span>
    <span>{case.get('patient_id', '?')}</span>
    <span class="recent-freq" style="color:{'#44cc88' if case.get('kept') else '#666'};">{case.get('freq', 0):.2f} Hz</span>
    <span style="color:#888;">{case.get('bin_label', '?')}</span>
  </div>
"""

    html += """</div>
</body>
</html>"""

    with open(DASHBOARD_PATH, 'w') as f:
        f.write(html)


def main():
    print("=" * 60)
    print("LPD Segment Harvester")
    print("=" * 60)

    # Load harvest manifest (tracks previously harvested files)
    harvest_manifest = load_harvest_manifest()
    print(f"  Harvest manifest: {len(harvest_manifest)} previously harvested")

    # Count what we already have (patients.csv + manifest)
    existing_counts = count_existing(harvest_manifest)
    new_counts = [0] * len(BINS)
    print("\nExisting LPD frequency distribution:")
    for i, (lo, hi, label) in enumerate(BINS):
        need = max(0, TARGET_PER_BIN - existing_counts[i])
        print(f"  {label}: {existing_counts[i]} (need {need} more)")

    # Check if already done
    if all(e >= TARGET_PER_BIN for e in existing_counts):
        print("\nAll bins already at target! Nothing to do.")
        return

    # List S3 files
    print("\nListing S3 files...")
    s3_files = list_s3_files()
    print(f"  Found {len(s3_files)} files on S3")

    # Filter out files we already have
    existing_eeg_files = set(p.stem.split('_seg')[0] for p in EEG_DIR.glob('*.mat'))
    print(f"  Already have {len(existing_eeg_files)} segment files locally")

    # Shuffle to avoid processing in order (which might cluster similar frequencies)
    np.random.seed(42)
    np.random.shuffle(s3_files)

    processed = 0
    kept = 0
    skipped = 0
    failed = 0
    recent_cases = []

    # Open dashboard
    build_dashboard(existing_counts, new_counts, 0, 0, 0, 0, len(s3_files))
    subprocess.run(['open', str(DASHBOARD_PATH)])

    for fi, filename in enumerate(s3_files):
        # Check if all bins are full
        total_counts = [e + n for e, n in zip(existing_counts, new_counts)]
        if all(t >= TARGET_PER_BIN for t in total_counts):
            print(f"\nAll bins at target! Stopping after {processed} files.")
            break

        patient_id = extract_patient_id(filename)

        # Skip if we already have this patient (in eeg dir or manifest)
        if patient_id in existing_eeg_files or patient_id in harvest_manifest:
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

            # Estimate frequency using unbiased FFT (no mean-shrinkage)
            freq = estimate_frequency_fft(seg, fs)

            # Determine bin
            bi = get_bin_index(freq)
            bin_label = BINS[bi][2] if bi >= 0 else 'out of range'

            # Check if this bin needs more
            total_in_bin = (existing_counts[bi] + new_counts[bi]) if bi >= 0 else TARGET_PER_BIN
            needs_more = bi >= 0 and total_in_bin < TARGET_PER_BIN

            case_info = {
                'patient_id': patient_id,
                'freq': float(freq),
                'bin_label': bin_label,
                'kept': needs_more,
            }
            recent_cases.append(case_info)

            if needs_more:
                # Save the 10s segment
                save_segment(seg, fs, patient_id)
                new_counts[bi] += 1
                kept += 1
                existing_eeg_files.add(patient_id)
                # Record in manifest
                harvest_manifest[patient_id] = {
                    'est_freq': round(float(freq), 3),
                    'bin': bin_label,
                    's3_file': filename,
                }
                save_harvest_manifest(harvest_manifest)
                print(f"  [{processed+1}/{len(s3_files)}] KEPT {patient_id}: {freq:.2f} Hz ({bin_label})")
            else:
                skipped += 1
                if (processed + 1) % 20 == 0:
                    print(f"  [{processed+1}/{len(s3_files)}] skip {patient_id}: {freq:.2f} Hz ({bin_label}) - bin full")

        except Exception as e:
            failed += 1
            print(f"  FAILED {patient_id}: {e}")

        finally:
            # Always delete the 10-minute file locally
            local_path.unlink(missing_ok=True)

        processed += 1

        # Update dashboard every 5 files
        if processed % 5 == 0 or needs_more:
            build_dashboard(existing_counts, new_counts, processed, kept,
                           skipped, failed, len(s3_files), recent_cases)

    # Final dashboard update
    build_dashboard(existing_counts, new_counts, processed, kept,
                   skipped, failed, len(s3_files), recent_cases)

    print(f"\n{'=' * 60}")
    print("HARVEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {processed}")
    print(f"  Kept: {kept}")
    print(f"  Skipped (bin full): {skipped}")
    print(f"  Failed: {failed}")
    print(f"\n  Final distribution:")
    for i, (lo, hi, label) in enumerate(BINS):
        total = existing_counts[i] + new_counts[i]
        status = '✓' if total >= TARGET_PER_BIN else f'need {TARGET_PER_BIN - total} more'
        print(f"    {label}: {existing_counts[i]} existing + {new_counts[i]} new = {total} ({status})")
    print(f"\n  Dashboard: {DASHBOARD_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
