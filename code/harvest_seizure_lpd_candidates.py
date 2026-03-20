"""
Harvest fast-LPD candidates from seizure folders on S3.

Searches SEIZURE, IIIC, and SEIZURE_BCH folders for segments that look like
fast LPDs (>2.5 Hz periodic discharges) rather than true evolving seizures.

Filtering criteria (ALL must pass):
  a. FFT peak frequency on pointiness traces: 2.0-5.0 Hz
  b. Inter-peak interval CV < 0.35 (regular periodicity)
  c. Laterality index |R-L|/(R+L) > 0.15
  d. Frequency stability: |freq_first5s - freq_last5s| < 0.5 Hz

Usage:
    conda run -n foe python code/harvest_seizure_lpd_candidates.py
"""

import sys
import os
import json
import time
import subprocess
import h5py
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
BASE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana, compute_pointiness_trace

DATA_DIR = BASE / 'data'
EEG_DIR = DATA_DIR / 'eeg'
STAGING_DIR = Path('/tmp/seizure_lpd_staging')
STAGING_DIR.mkdir(exist_ok=True)

DASHBOARD_PATH = BASE / 'results' / 'harvest_seizure_dashboard.html'

S3_PREFIXES = [
    's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/SEIZURE/segments_raw/',
    's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/IIIC/segments_raw/',
    's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/SEIZURE_BCH/segments_raw/',
]
AWS_PROFILE = 'opendata'

FS = 200
LOWPASS_HZ = 15.0

# Target bins for fast LPDs
BINS = [
    (2.5, 3.0, '2.5-3.0 Hz'),
    (3.0, 3.5, '3.0-3.5 Hz'),
    (3.5, 5.0, '3.5+ Hz'),
]
TARGET_PER_BIN = 100

HARVEST_MANIFEST = DATA_DIR / 'labels' / 'harvest_manifest.json'


def load_harvest_manifest():
    """Load the harvest manifest."""
    if HARVEST_MANIFEST.exists():
        with open(HARVEST_MANIFEST) as f:
            return json.load(f)
    return {}


def save_harvest_manifest(manifest):
    """Save the harvest manifest."""
    with open(HARVEST_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=2)


def get_bin_index(freq):
    """Return the bin index for a frequency, or -1 if outside all bins."""
    for i, (lo, hi, _) in enumerate(BINS):
        if lo <= freq < hi:
            return i
    return -1


def count_existing(harvest_manifest):
    """Count existing cases per frequency bin from the harvest manifest."""
    counts = [0] * len(BINS)
    for pid, info in harvest_manifest.items():
        freq = info.get('est_freq', 0)
        bi = get_bin_index(freq)
        if bi >= 0:
            counts[bi] += 1
    return counts


def estimate_frequency_fft(segment, fs=FS, freq_lo=2.0, freq_hi=5.0):
    """Estimate frequency using FFT peak on pointiness traces.

    Returns median peak frequency across channels.
    """
    n_channels = segment.shape[0]

    b_lp, a_lp = butter(4, LOWPASS_HZ / (fs / 2), btype='low')
    seg_lp = np.zeros_like(segment)
    for ch in range(n_channels):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, segment[ch])
        except ValueError:
            seg_lp[ch] = segment[ch]

    fft_freqs = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=max(1, int(0.02 * fs)))

        n = len(pt)
        if n < 10:
            continue
        fft_vals = np.abs(np.fft.rfft(pt - np.mean(pt)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= freq_lo) & (freqs <= freq_hi)
        if not np.any(mask):
            continue
        fft_sub = fft_vals[mask]
        freqs_sub = freqs[mask]
        if np.max(fft_sub) > 0:
            fft_freqs.append(freqs_sub[np.argmax(fft_sub)])

    if fft_freqs:
        return float(np.median(fft_freqs))
    return np.nan


def list_s3_files(prefix):
    """List all .mat files in an S3 prefix."""
    result = subprocess.run(
        ['aws', 's3', 'ls', prefix, '--profile', AWS_PROFILE],
        capture_output=True, text=True
    )
    files = []
    for line in result.stdout.strip().split('\n'):
        if line.strip() and line.strip().endswith('.mat'):
            parts = line.strip().split()
            fname = parts[-1]
            files.append((prefix, fname))
    return files


def download_file(prefix, filename):
    """Download a file from S3 to staging. Returns local path."""
    local_path = STAGING_DIR / filename
    subprocess.run(
        ['aws', 's3', 'cp', prefix + filename, str(local_path),
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


def save_segment(seg, fs, patient_id):
    """Save a 10-second bipolar segment as a .mat file in data/eeg/."""
    out_path = EEG_DIR / f'{patient_id}_seg000.mat'
    sio.savemat(str(out_path), {
        'data': seg,
        'Fs': np.array([[fs]], dtype=np.float64),
    })
    return out_path


def extract_patient_id(filename):
    """Extract patient ID from filename like sub-S0001111201556_20150823212448.mat."""
    name = filename.replace('.mat', '')
    if name.startswith('sub-S0001'):
        pid = name[9:].split('_')[0]
        return pid
    return name


# ── Fast LPD candidate filters ──────────────────────────────────────────────

LEFT_CH = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CH = [4, 5, 6, 7, 12, 13, 14, 15]


def check_fft_peak_frequency(seg, fs):
    """Check FFT peak freq on pointiness traces (median across channels) is 2.0-5.0 Hz.

    Returns (pass, freq).
    """
    freq = estimate_frequency_fft(seg, fs, freq_lo=2.0, freq_hi=5.0)
    if np.isnan(freq):
        return False, freq, 'fft_nan'
    if freq < 2.0 or freq >= 5.0:
        return False, freq, f'fft_freq={freq:.2f}'
    return True, freq, None


def check_interval_regularity(seg, fs):
    """Check inter-peak interval CV < 0.35 on max-abs-envelope.

    Returns (pass, cv, reason).
    """
    # Max absolute envelope across channels
    envelope = np.max(np.abs(seg), axis=0)
    # Smooth slightly
    envelope = gaussian_filter1d(envelope, sigma=max(1, int(0.01 * fs)))

    # Find peaks with minimum distance ~0.1s
    min_dist = max(1, int(0.1 * fs))
    peaks, props = find_peaks(envelope, distance=min_dist, height=np.percentile(envelope, 50))

    if len(peaks) < 4:
        return False, np.nan, 'too_few_peaks'

    intervals = np.diff(peaks) / fs
    cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else np.inf

    if cv >= 0.35:
        return False, cv, f'cv={cv:.3f}'
    return True, cv, None


def check_laterality(seg):
    """Check laterality index |R-L|/(R+L) > 0.15.

    Returns (pass, lat_index, reason).
    """
    left_rms = np.sqrt(np.mean(seg[LEFT_CH] ** 2))
    right_rms = np.sqrt(np.mean(seg[RIGHT_CH] ** 2))
    total = left_rms + right_rms

    if total < 1e-10:
        return False, 0.0, 'flat_signal'

    lat = abs(right_rms - left_rms) / total

    if lat <= 0.15:
        return False, lat, f'lat={lat:.3f}'
    return True, lat, None


def check_frequency_stability(seg, fs):
    """Check freq in first 5s vs last 5s — difference < 0.5 Hz.

    Returns (pass, freq_diff, reason).
    """
    half = seg.shape[1] // 2
    first_half = seg[:, :half]
    second_half = seg[:, half:]

    freq1 = estimate_frequency_fft(first_half, fs, freq_lo=2.0, freq_hi=5.0)
    freq2 = estimate_frequency_fft(second_half, fs, freq_lo=2.0, freq_hi=5.0)

    if np.isnan(freq1) or np.isnan(freq2):
        return False, np.nan, 'stability_nan'

    diff = abs(freq1 - freq2)
    if diff >= 0.5:
        return False, diff, f'freq_drift={diff:.2f}'
    return True, diff, None


def filter_candidate(seg, fs):
    """Apply all fast-LPD filters. Returns (pass, freq, reject_reason, details)."""
    details = {}

    # a. FFT peak frequency
    ok, freq, reason = check_fft_peak_frequency(seg, fs)
    details['freq'] = freq
    if not ok:
        return False, freq, reason or 'fft_freq', details

    # b. Interval regularity
    ok, cv, reason = check_interval_regularity(seg, fs)
    details['cv'] = cv
    if not ok:
        return False, freq, reason or 'regularity', details

    # c. Laterality
    ok, lat, reason = check_laterality(seg)
    details['laterality'] = lat
    if not ok:
        return False, freq, reason or 'laterality', details

    # d. Frequency stability
    ok, fdiff, reason = check_frequency_stability(seg, fs)
    details['freq_diff'] = fdiff
    if not ok:
        return False, freq, reason or 'stability', details

    return True, freq, None, details


# ── Dashboard ────────────────────────────────────────────────────────────────

def build_dashboard(existing_counts, new_counts, processed, kept,
                    reject_stats, failed, total_files, recent_cases=None):
    """Build/update the harvest seizure dashboard HTML."""
    if recent_cases is None:
        recent_cases = []

    bin_labels = [b[2] for b in BINS]
    total_counts = [e + n for e, n in zip(existing_counts, new_counts)]
    total_rejected = sum(reject_stats.values())

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Seizure-to-LPD Harvest Dashboard</title>
<meta http-equiv="refresh" content="5">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px; background: #1a1a1a; color: #eee; font-family: 'Menlo', 'Consolas', monospace; }}
  h1 {{ color: #e06030; margin-bottom: 5px; font-size: 22px; }}
  .subtitle {{ color: #888; margin-bottom: 20px; font-size: 13px; }}

  .status {{ padding: 12px 16px; background: #222; border-radius: 8px; margin-bottom: 18px;
            display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; }}
  .status-item {{ text-align: center; min-width: 90px; }}
  .status-val {{ font-size: 22px; font-weight: bold; }}
  .status-label {{ font-size: 10px; color: #777; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }}
  .green {{ color: #44cc88; }}
  .orange {{ color: #ff9800; }}
  .cyan {{ color: #44cccc; }}
  .red {{ color: #ff4444; }}
  .dim {{ color: #666; }}
  .yellow {{ color: #cccc44; }}

  .progress-container {{
    background: #333; border-radius: 6px; height: 20px; margin-bottom: 18px;
    overflow: hidden; position: relative;
  }}
  .progress-fill {{
    height: 100%; background: linear-gradient(90deg, #e06030, #44cc88);
    border-radius: 6px; transition: width 0.5s;
  }}
  .progress-label {{
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: bold; color: #eee; text-shadow: 0 0 4px #000;
  }}

  .reject-stats {{ background: #222; border-radius: 8px; padding: 12px 16px; margin-bottom: 18px;
                   display: flex; gap: 16px; flex-wrap: wrap; }}
  .reject-item {{ font-size: 12px; }}
  .reject-count {{ font-weight: bold; color: #cc6644; }}
  .reject-label {{ color: #888; }}

  .hist-container {{ background: #222; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
  .hist-container h3 {{ margin: 0 0 12px 0; color: #aaa; font-size: 14px; }}
  .bar-row {{ display: flex; align-items: center; margin: 6px 0; }}
  .bar-label {{ width: 120px; font-size: 12px; color: #aaa; text-align: right; padding-right: 10px; }}
  .bar-track {{ flex: 1; height: 28px; background: #333; border-radius: 4px; position: relative; overflow: hidden; }}
  .bar-existing {{ height: 100%; background: #2a5a2a; position: absolute; left: 0; }}
  .bar-new {{ height: 100%; background: #44cc88; position: absolute; }}
  .bar-target {{ position: absolute; height: 100%; border-right: 2px dashed #e06030; }}
  .bar-count {{ position: absolute; right: 8px; top: 4px; font-size: 12px; font-weight: bold; color: #eee; z-index: 1; }}

  .recent {{ background: #222; border-radius: 8px; padding: 16px; }}
  .recent h3 {{ margin: 0 0 12px 0; color: #aaa; font-size: 14px; }}
  .recent-item {{ display: flex; gap: 12px; align-items: center; padding: 4px 0; border-bottom: 1px solid #2a2a2a; font-size: 12px; }}
  .recent-kept {{ color: #44cc88; font-weight: bold; }}
  .recent-reject {{ color: #cc6644; }}
</style>
</head>
<body>
<h1>Seizure-to-LPD Harvest Dashboard</h1>
<p class="subtitle">Searching seizure folders for fast LPD candidates (>2.5 Hz periodic discharges) | Target: {TARGET_PER_BIN} per bin</p>

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
    <div class="status-val yellow">{total_rejected}</div>
    <div class="status-label">Rejected</div>
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
    <div class="status-label">Total kept</div>
  </div>
  <div class="status-item">
    <div class="status-val dim">{sum(1 for t in total_counts if t >= TARGET_PER_BIN)}/{len(BINS)}</div>
    <div class="status-label">Bins complete</div>
  </div>
</div>

<div class="progress-container">
  <div class="progress-fill" style="width:{100*processed/max(total_files,1):.1f}%;"></div>
  <div class="progress-label">{processed} / {total_files} files tried ({100*processed/max(total_files,1):.0f}%)</div>
</div>

<div class="reject-stats">
  <span style="color:#aaa; font-size:12px; font-weight:bold;">Reject reasons:</span>"""

    for reason in ['fft_freq', 'fft_nan', 'regularity', 'laterality', 'stability',
                   'too_few_peaks', 'stability_nan', 'bin_full']:
        count = reject_stats.get(reason, 0)
        # Also check partial-match keys
        for k, v in reject_stats.items():
            if k.startswith(reason.split('_')[0]) and k != reason and reason in ['fft_freq', 'regularity', 'laterality', 'stability']:
                pass  # handled by exact keys below
        html += f"""
  <span class="reject-item"><span class="reject-count">{count}</span> <span class="reject-label">{reason}</span></span>"""

    html += f"""
</div>

<div class="hist-container">
  <h3>Fast LPD Frequency Distribution (existing + new | target = {TARGET_PER_BIN})</h3>
"""

    max_count = max(max(total_counts) if total_counts else 1, TARGET_PER_BIN) * 1.1
    for i, (label, existing, new, total) in enumerate(
            zip(bin_labels, existing_counts, new_counts, total_counts)):
        ex_pct = existing / max_count * 100
        new_pct = new / max_count * 100
        target_pct = TARGET_PER_BIN / max_count * 100

        html += f"""  <div class="bar-row">
    <span class="bar-label">{label}</span>
    <div class="bar-track">
      <div class="bar-existing" style="width:{ex_pct:.1f}%;"></div>
      <div class="bar-new" style="left:{ex_pct:.1f}%; width:{new_pct:.1f}%;"></div>
      <div class="bar-target" style="left:{target_pct:.1f}%;"></div>
      <span class="bar-count">{total} / {TARGET_PER_BIN}</span>
    </div>
  </div>
"""

    html += """</div>

<div class="recent">
  <h3>Recent cases</h3>
"""

    for case in reversed(recent_cases[-20:]):
        if case.get('kept'):
            status_cls = 'recent-kept'
            status_txt = 'KEPT'
        else:
            status_cls = 'recent-reject'
            status_txt = 'REJECT'
        freq_val = case.get('freq', 0)
        freq_str = f'{freq_val:.2f} Hz' if not np.isnan(freq_val) else 'N/A'
        reason_str = case.get('reject_reason', '') or ''
        html += f"""  <div class="recent-item">
    <span class="{status_cls}" style="min-width:55px;">{status_txt}</span>
    <span style="min-width:100px;">{case.get('patient_id', '?')}</span>
    <span style="min-width:70px; font-weight:bold; color:{'#44cc88' if case.get('kept') else '#888'};">{freq_str}</span>
    <span style="color:#666;">{reason_str}</span>
  </div>
"""

    html += """</div>
</body>
</html>"""

    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DASHBOARD_PATH, 'w') as f:
        f.write(html)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Seizure-to-LPD Candidate Harvester")
    print("=" * 60)

    # Ensure directories exist
    EEG_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / 'labels').mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load harvest manifest
    harvest_manifest = load_harvest_manifest()
    print(f"  Harvest manifest: {len(harvest_manifest)} previously harvested")

    # Count existing in target bins
    existing_counts = count_existing(harvest_manifest)
    new_counts = [0] * len(BINS)
    print("\nExisting fast-LPD frequency distribution (from manifest):")
    for i, (lo, hi, label) in enumerate(BINS):
        need = max(0, TARGET_PER_BIN - existing_counts[i])
        print(f"  {label}: {existing_counts[i]} (need {need} more)")

    if all(e >= TARGET_PER_BIN for e in existing_counts):
        print("\nAll bins already at target! Nothing to do.")
        return

    # List S3 files from all seizure folders
    print("\nListing S3 files from seizure folders...")
    all_files = []
    for prefix in S3_PREFIXES:
        folder_name = prefix.split('/')[-3]
        files = list_s3_files(prefix)
        print(f"  {folder_name}: {len(files)} files")
        all_files.extend(files)
    print(f"  Total: {len(all_files)} files")

    if not all_files:
        print("No files found! Check S3 access.")
        return

    # Shuffle with seed=123
    np.random.seed(123)
    np.random.shuffle(all_files)

    # Get existing patient IDs to skip
    existing_eeg_files = set(p.stem.split('_seg')[0] for p in EEG_DIR.glob('*.mat'))
    manifest_pids = set(harvest_manifest.keys())
    print(f"  Skipping {len(existing_eeg_files)} existing EEG files, {len(manifest_pids)} manifest entries")

    processed = 0
    kept = 0
    failed = 0
    reject_stats = {}  # reason -> count
    recent_cases = []

    # Build initial dashboard and open it
    build_dashboard(existing_counts, new_counts, 0, 0, reject_stats, 0,
                    len(all_files))
    subprocess.run(['open', str(DASHBOARD_PATH)])

    for fi, (prefix, filename) in enumerate(all_files):
        # Check if all bins are full
        total_counts = [e + n for e, n in zip(existing_counts, new_counts)]
        if all(t >= TARGET_PER_BIN for t in total_counts):
            print(f"\nAll bins at target! Stopping after {processed} files.")
            break

        patient_id = extract_patient_id(filename)

        # Skip if already processed
        if patient_id in existing_eeg_files or patient_id in manifest_pids:
            continue

        # Download
        local_path = download_file(prefix, filename)
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

            # Apply all fast-LPD filters
            passed, freq, reject_reason, details = filter_candidate(seg, fs)

            if passed:
                bi = get_bin_index(freq)
                if bi >= 0 and (existing_counts[bi] + new_counts[bi]) < TARGET_PER_BIN:
                    # Keep it
                    save_segment(seg, fs, patient_id)
                    new_counts[bi] += 1
                    kept += 1
                    existing_eeg_files.add(patient_id)
                    manifest_pids.add(patient_id)

                    bin_label = BINS[bi][2]
                    harvest_manifest[patient_id] = {
                        'est_freq': round(float(freq), 3),
                        'bin': bin_label,
                        's3_file': filename,
                        's3_prefix': prefix.split('/')[-3],
                        'source': 'seizure_harvest',
                        'cv': round(float(details.get('cv', 0)), 3),
                        'laterality': round(float(details.get('laterality', 0)), 3),
                        'freq_diff': round(float(details.get('freq_diff', 0)), 3),
                    }
                    save_harvest_manifest(harvest_manifest)

                    case_info = {'patient_id': patient_id, 'freq': freq,
                                 'kept': True, 'reject_reason': None}
                    recent_cases.append(case_info)
                    print(f"  [{processed+1}/{len(all_files)}] KEPT {patient_id}: {freq:.2f} Hz ({bin_label}) "
                          f"cv={details.get('cv',0):.2f} lat={details.get('laterality',0):.2f}")
                else:
                    # Bin full
                    reject_stats['bin_full'] = reject_stats.get('bin_full', 0) + 1
                    case_info = {'patient_id': patient_id, 'freq': freq,
                                 'kept': False, 'reject_reason': 'bin_full'}
                    recent_cases.append(case_info)
            else:
                # Categorize rejection reason
                if reject_reason:
                    # Normalize to a simple category
                    if 'fft' in reject_reason or 'freq=' in reject_reason:
                        cat = 'fft_freq' if 'nan' not in reject_reason else 'fft_nan'
                    elif 'cv' in reject_reason or 'few_peaks' in reject_reason:
                        cat = 'regularity' if 'cv' in reject_reason else 'too_few_peaks'
                    elif 'lat' in reject_reason:
                        cat = 'laterality'
                    elif 'drift' in reject_reason or 'stability' in reject_reason:
                        cat = 'stability' if 'nan' not in reject_reason else 'stability_nan'
                    else:
                        cat = reject_reason
                    reject_stats[cat] = reject_stats.get(cat, 0) + 1
                else:
                    reject_stats['unknown'] = reject_stats.get('unknown', 0) + 1

                case_info = {'patient_id': patient_id, 'freq': float(freq) if not np.isnan(freq) else 0.0,
                             'kept': False, 'reject_reason': reject_reason}
                recent_cases.append(case_info)

                if (processed + 1) % 50 == 0:
                    print(f"  [{processed+1}/{len(all_files)}] reject {patient_id}: {reject_reason}")

        except Exception as e:
            failed += 1
            print(f"  FAILED {patient_id}: {e}")

        finally:
            local_path.unlink(missing_ok=True)

        processed += 1

        # Update dashboard every 5 files or on keep
        if processed % 5 == 0 or (case_info and case_info.get('kept')):
            build_dashboard(existing_counts, new_counts, processed, kept,
                            reject_stats, failed, len(all_files), recent_cases)

    # Final dashboard update
    build_dashboard(existing_counts, new_counts, processed, kept,
                    reject_stats, failed, len(all_files), recent_cases)

    print(f"\n{'=' * 60}")
    print("HARVEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed: {processed}")
    print(f"  Kept: {kept}")
    print(f"  Rejected: {sum(reject_stats.values())}")
    for reason, count in sorted(reject_stats.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count}")
    print(f"  Failed: {failed}")
    print(f"\n  Final distribution:")
    for i, (lo, hi, label) in enumerate(BINS):
        total = existing_counts[i] + new_counts[i]
        status = 'DONE' if total >= TARGET_PER_BIN else f'need {TARGET_PER_BIN - total} more'
        print(f"    {label}: {existing_counts[i]} existing + {new_counts[i]} new = {total} ({status})")
    print(f"\n  Dashboard: {DASHBOARD_PATH}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
