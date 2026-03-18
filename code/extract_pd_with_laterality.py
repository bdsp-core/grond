"""
Extract PD frequency, spatial extent, and lateralization from EEG dataset.

Runs the pd_detect_alternate detector (with per-channel scores) on all LPD/GPD segments
and saves results including laterality index to CSV.
"""

import pandas as pd
import numpy as np
import warnings
import hdf5storage
import h5py
from pathlib import Path
from tqdm import tqdm

import pd_detector_alternate as pddeta

warnings.filterwarnings('ignore')

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']


def load_mat_file(filepath):
    """Load MATLAB file, handling both v7.3 and earlier versions."""
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError as e:
        if 'HDF reader for matlab v7.3 files' in str(e):
            with h5py.File(filepath, 'r') as f:
                return {key: f[key][()] for key in f.keys()}
        else:
            raise


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent if script_dir.name == 'code' else script_dir

    data_dir = repo_root / 'data' / 'dataset_eeg'
    results_dir = repo_root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please download the dataset from BDSP and place it in {repo_root / 'data'}\n"
            f"See DATASET_INFO.md for instructions."
        )

    for event, pk_detect in [('lpd', 'apd'), ('gpd', 'apd')]:
        print('*' * 70)
        print(event.upper())
        print('*' * 70)

        event_dir = data_dir / event
        if not event_dir.exists():
            print(f"Warning: Directory {event_dir} not found. Skipping {event}.")
            continue

        files_4analysis = sorted([f for f in event_dir.iterdir() if f.suffix == '.mat'])
        rows = []

        for file_path in tqdm(files_4analysis, desc=f"Processing {event.upper()}"):
            fs = 200

            try:
                mat = load_mat_file(str(file_path))
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")
                continue

            try:
                segment = mat['data_50sec']
            except (KeyError, Exception):
                segment = mat['data']

            data_obj = pddeta.pd_detect_alternate(segment, fs, pk_detect=pk_detect)

            row = {
                'files': file_path.stem,
                'type_event': data_obj['type_event'],
                'event_frequency': data_obj['event_frequency'],
                'spatial_extent': data_obj['spatial_extent'],
                'spatial_areas': data_obj['spatial_areas'],
                'laterality_index': data_obj['laterality_index'],
                'left_mean_score': data_obj['left_mean_score'],
                'right_mean_score': data_obj['right_mean_score'],
            }

            # Per-channel PD scores
            for ch in BIPOLAR_CHANNELS:
                row[f'score_{ch}'] = data_obj['channel_pd_scores'][ch]

            # Per-channel frequencies
            for ch in BIPOLAR_CHANNELS:
                row[f'freq_{ch}'] = data_obj['channel_frequencies'][ch]

            # Per-region scores
            for reg in REGIONS:
                row[f'region_{reg}'] = data_obj['region_scores'][reg]

            rows.append(row)

        df = pd.DataFrame(rows)
        output_file = results_dir / f'{event}_laterality_results.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
