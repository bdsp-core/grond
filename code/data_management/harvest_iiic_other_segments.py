"""Harvest 'Other' (control) segments from morgoth1 IIIC dataset.

These are EEG segments that experts agreed do NOT contain LPD, GPD, LRDA, GRDA, or seizure.
Useful as true negatives for channel-level PD/RDA detection training.

Usage: conda run -n foe python code/harvest_iiic_other_segments.py
"""
import json, sys, os, ast, subprocess, tempfile
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pd_pointiness_acf import fcn_getBanana

PROJECT_DIR = Path(__file__).resolve().parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
MANIFEST_PATH = PROJECT_DIR / 'data' / 'labels' / 'other_harvest_manifest.json'

S3_BASE = 's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/IIIC/segments_raw/'
EVENTS_PATH = '/tmp/iiic_events.xlsx'

MAX_PATIENTS = 500
MAX_SEGS_PER_PATIENT = 2
TARGET_TOTAL = 1000

def load_manifest():
    if MANIFEST_PATH.exists():
        with open(str(MANIFEST_PATH)) as f:
            return json.load(f)
    return {}

def save_manifest(manifest):
    with open(str(MANIFEST_PATH), 'w') as f:
        json.dump(manifest, f)

def extract_patient_id(filename):
    name = filename.replace('.mat', '')
    if 'sub-S' in name:
        parts = name.split('_')
        mrn = parts[0].replace('sub-S0001', '').replace('sub-S0002', '')
        return mrn
    return name

def main():
    print("=" * 60)
    print("Harvesting 'Other' (control) segments from IIIC")
    print("=" * 60)
    
    # Load events
    if not os.path.exists(EVENTS_PATH):
        print("Downloading events spreadsheet...")
        subprocess.run(['aws', 's3', 'cp',
            's3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/IIIC/list_events_20241129.xlsx',
            EVENTS_PATH], env={**os.environ, 'AWS_PROFILE': 'opendata'})
    
    df = pd.read_excel(EVENTS_PATH)
    
    def get_dominant(label_str):
        try:
            votes = ast.literal_eval(str(label_str))
            names = ['other','seizure','lpd','gpd','lrda','grda']
            return names[np.argmax(votes)]
        except:
            return 'unknown'
    
    def get_votes(label_str):
        try: return ast.literal_eval(str(label_str))
        except: return [0]*6
    
    df['dominant'] = df['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(get_dominant)
    df['votes'] = df['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(get_votes)
    df['n_votes'] = df['votes'].apply(sum)
    
    # Filter to "other" with >=3 votes
    other = df[(df['dominant'] == 'other') & (df['n_votes'] >= 3)].copy()
    other = other.sample(frac=1, random_state=42)  # shuffle
    
    print(f"Available: {len(other)} Other segments from {other['bdsp_mrn'].nunique()} patients")
    
    manifest = load_manifest()
    existing_pids = set(manifest.keys())
    n_kept = len(manifest)
    n_processed = 0
    
    patients_seen = Counter()
    
    for _, row in other.iterrows():
        if n_kept >= TARGET_TOTAL:
            break
        
        pid = str(row['bdsp_mrn'])
        if pid in existing_pids:
            continue
        if patients_seen[pid] >= MAX_SEGS_PER_PATIENT:
            continue
        if len(set(patients_seen.keys())) >= MAX_PATIENTS and pid not in patients_seen:
            continue
        
        filename = str(row['file_name']) + '.mat'
        s3_path = S3_BASE + filename
        
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            result = subprocess.run(
                ['aws', 's3', 'cp', s3_path, tmp_path],
                env={**os.environ, 'AWS_PROFILE': 'opendata'},
                capture_output=True, timeout=120)
            
            if result.returncode != 0:
                continue
            
            # Load HDF5
            import h5py
            with h5py.File(tmp_path, 'r') as f:
                data = np.array(f['data'], dtype=np.float64)
                fs = float(np.array(f['Fs']).flat[0])
            
            if data.ndim == 2 and data.shape[0] == 20:
                data = data.T
            
            n_samples = data.shape[0]
            center = n_samples // 2
            half = int(5 * fs)
            seg = data[center - half:center + half, :]
            
            if seg.shape[1] == 20:
                seg_bipolar = fcn_getBanana(seg)
            else:
                continue
            
            seg_id = f"{pid}_other_seg{patients_seen[pid]:03d}"
            out_path = EEG_DIR / f"{seg_id}.mat"
            sio.savemat(str(out_path), {'data': seg_bipolar, 'Fs': fs})
            
            manifest[seg_id] = {
                'patient_id': pid,
                'subtype': 'other',
                'expert_votes': row['votes'],
                'n_experts': int(row['n_votes']),
                's3_file': filename,
            }
            
            patients_seen[pid] += 1
            n_kept += 1
            n_processed += 1
            
            if n_processed % 20 == 0:
                save_manifest(manifest)
                print(f"  [{n_processed}] Kept: {n_kept}/{TARGET_TOTAL}, Patients: {len(set(patients_seen.keys()))}")
        
        except Exception as e:
            pass
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    save_manifest(manifest)
    print(f"\nDone: {n_kept} Other segments from {len(set(patients_seen.keys()))} patients")

if __name__ == '__main__':
    main()
