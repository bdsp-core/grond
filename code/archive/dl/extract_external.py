"""
Extract and cache 10-second segments from the external drive for Phase 1 pretraining.
Run: conda run -n foe_dl python code/dl/extract_external.py
"""
import sys, os, ast, warnings
import numpy as np
import pandas as pd
import hdf5storage
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import preprocess_segment

EXTERNAL_DIR = Path('/Volumes/sanD_photos/IIIC')
EXCEL_PATH = EXTERNAL_DIR / 'list_events_20241129.xlsx'
SEG_DIR = EXTERNAL_DIR / 'segments_raw'
CACHE_DIR = Path(__file__).parent.parent.parent / 'data' / 'dl_cache'

MAX_PER_TYPE = 2000  # Max segments per type to keep runtime reasonable
FS = 200

def parse_timestamp_from_filename(filename):
    """Extract start datetime from filename like 'sub-S0001117208459_20141122125953'."""
    parts = filename.split('_')
    ts_str = parts[-1]  # '20141122125953'
    try:
        return datetime.strptime(ts_str, '%Y%m%d%H%M%S')
    except:
        return None

def parse_event_time(event_time_str):
    """Parse event time string like '2014-11-22 13:04:53' or '22-Nov-2014 13:04:53'."""
    event_time_str = str(event_time_str).strip("'\"")
    for fmt in ['%Y-%m-%d %H:%M:%S', '%d-%b-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
        try:
            return datetime.strptime(event_time_str, fmt)
        except:
            continue
    return None

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"[1] Loading Excel file: {EXCEL_PATH}")
    df = pd.read_excel(str(EXCEL_PATH))
    print(f"  Total rows: {len(df)}")

    # Parse labels
    df['parsed'] = df['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else None)
    df = df[df['parsed'].notna()].copy()
    df['total'] = df['parsed'].apply(sum)
    df['lpd'] = df['parsed'].apply(lambda x: x[2])
    df['gpd'] = df['parsed'].apply(lambda x: x[3])

    # Check which files exist on the drive
    available = set(os.path.splitext(f)[0] for f in os.listdir(SEG_DIR)) if SEG_DIR.exists() else set()
    print(f"  Available .mat files: {len(available)}")

    # Select majority LPD and GPD segments that exist on drive
    segments_to_extract = []
    for ptype, col_idx in [('lpd', 'lpd'), ('gpd', 'gpd')]:
        df_type = df[(df[col_idx] / df['total'] > 0.5) & (df['file_name'].isin(available))].copy()
        df_type = df_type.sort_values(col_idx, ascending=False)
        # Diversify by patient
        seen_patients = set()
        picks = []
        for _, row in df_type.iterrows():
            mrn = row['bdsp_mrn']
            if mrn not in seen_patients or len(picks) < MAX_PER_TYPE:
                seen_patients.add(mrn)
                picks.append(row)
                if len(picks) >= MAX_PER_TYPE:
                    break
        for row in picks:
            segments_to_extract.append({
                'file_name': row['file_name'],
                'event_time': row['event_time'],
                'ptype': ptype,
                'patient': str(row['bdsp_mrn']),
                'votes': row[col_idx],
                'total': row['total'],
            })
        print(f"  {ptype.upper()}: selected {len(picks)} segments from {len(seen_patients)} patients")

    print(f"\n[2] Extracting {len(segments_to_extract)} segments...")

    all_segments = []
    all_labels = []
    all_patients = []
    failed = 0

    for i, seg_info in enumerate(tqdm(segments_to_extract, desc="Extracting")):
        try:
            mat_path = SEG_DIR / f"{seg_info['file_name']}.mat"
            mat = hdf5storage.loadmat(str(mat_path))
            data = mat['data']
            if data.shape[0] < data.shape[1]:
                pass  # Already (channels, samples)
            else:
                data = data.T  # Convert to (channels, samples)

            fs_mat = int(mat.get('Fs', np.array([[FS]]))[0, 0]) if 'Fs' in mat else FS

            # Compute event offset
            file_start = parse_timestamp_from_filename(seg_info['file_name'])
            event_time = parse_event_time(seg_info['event_time'])

            if file_start and event_time:
                offset_sec = (event_time - file_start).total_seconds()
                offset_sample = int(offset_sec * fs_mat)
            else:
                # Default to center of recording
                offset_sample = data.shape[1] // 2

            # Extract 10s window centered on event
            half_win = int(5 * fs_mat)
            start = max(0, offset_sample - half_win)
            end = min(data.shape[1], offset_sample + half_win)

            window = data[:, start:end]

            # Ensure exactly 2000 samples (10s at 200Hz)
            target_len = 10 * FS
            if window.shape[1] < target_len:
                # Pad with zeros
                pad = target_len - window.shape[1]
                window = np.pad(window, ((0, 0), (0, pad)), mode='constant')
            elif window.shape[1] > target_len:
                window = window[:, :target_len]

            # Ensure 20 channels (or at least 19 for bipolar)
            if window.shape[0] < 19:
                failed += 1
                continue
            if window.shape[0] > 20:
                window = window[:20]
            elif window.shape[0] == 19:
                # Add a zero EKG channel
                window = np.vstack([window, np.zeros((1, target_len))])

            # Preprocess: bipolar montage + filtering
            seg_processed = preprocess_segment(window, FS)  # (18, 2000)

            all_segments.append(seg_processed)
            all_labels.append(0 if seg_info['ptype'] == 'lpd' else 1)
            all_patients.append(seg_info['patient'])

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Warning: failed on {seg_info['file_name']}: {e}")
            continue

    print(f"\n[3] Successfully extracted {len(all_segments)} segments ({failed} failed)")

    segments_arr = np.array(all_segments, dtype=np.float32)  # (N, 18, 2000)
    labels_arr = np.array(all_labels, dtype=np.int64)
    patients_arr = np.array(all_patients)

    n_lpd = np.sum(labels_arr == 0)
    n_gpd = np.sum(labels_arr == 1)
    n_patients = len(set(all_patients))
    print(f"  LPD: {n_lpd}, GPD: {n_gpd}, Patients: {n_patients}")

    # Save cache
    cache_path = CACHE_DIR / 'external_pd_segments.npz'
    np.savez_compressed(str(cache_path),
                        segments=segments_arr,
                        labels=labels_arr,
                        patients=patients_arr)
    print(f"\n[4] Saved cache: {cache_path} ({segments_arr.nbytes / 1e6:.1f} MB)")
    print("Done!")

if __name__ == '__main__':
    main()
