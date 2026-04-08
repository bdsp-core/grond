#!/usr/bin/env python3
"""Re-download existing bipolar files as monopolar from S3.

Finds files that are currently 18-channel (bipolar), locates them on S3,
downloads the 10-min raw, extracts center 10-sec, saves as 19-channel monopolar.
Does NOT modify any labels.

S3 sources:
- morgoth1/data/internal_dataset/{LRDA,GRDA,LPD,GPD,SEIZURE,...}/segments_raw/
- morgoth1/data/internal_dataset/IIIC/segments_raw/

Usage:
    python code/data_management/redownload_monopolar.py [--dry-run] [--max N]
"""
import os
import sys
import argparse
import tempfile
import subprocess
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'

S3_BUCKET = 's3://bdsp-opendata-credentialed'
AWS_REGION = 'us-east-1'

# Path to a CSV file with the columns "Access key ID" and "Secret access key"
# (the format produced by AWS IAM when you download a new access key). Override
# by exporting AWS_KEYS_PATH or by setting AWS_ACCESS_KEY_ID and
# AWS_SECRET_ACCESS_KEY directly in the environment.
AWS_KEYS_PATH = Path(
    os.environ.get(
        'AWS_KEYS_PATH',
        Path.home() / '.aws' / 'bdsp_opendata_write_accessKeys.csv',
    )
)

FS = 200
CENTER_OFFSET = 300  # seconds


def load_aws_credentials():
    """Load AWS credentials, preferring the standard environment variables."""
    if 'AWS_ACCESS_KEY_ID' in os.environ and 'AWS_SECRET_ACCESS_KEY' in os.environ:
        return os.environ['AWS_ACCESS_KEY_ID'], os.environ['AWS_SECRET_ACCESS_KEY']
    if not AWS_KEYS_PATH.exists():
        raise FileNotFoundError(
            f"AWS credentials not found. Either set AWS_ACCESS_KEY_ID and "
            f"AWS_SECRET_ACCESS_KEY in your environment, or place a credentials "
            f"CSV at {AWS_KEYS_PATH} (or set AWS_KEYS_PATH to point at one)."
        )
    keys = pd.read_csv(str(AWS_KEYS_PATH))
    return keys.iloc[0]['Access key ID'], keys.iloc[0]['Secret access key']


# Mapping from original_source/filename patterns to S3 paths
S3_SEARCH_PATHS = [
    'morgoth1/data/internal_dataset/LRDA/segments_raw',
    'morgoth1/data/internal_dataset/GRDA/segments_raw',
    'morgoth1/data/internal_dataset/LPD/segments_raw',
    'morgoth1/data/internal_dataset/GPD/segments_raw',
    'morgoth1/data/internal_dataset/SEIZURE/segments_raw',
    'morgoth1/data/internal_dataset/SEIZURE_BCH/segments_raw',
    'morgoth1/data/internal_dataset/BIPD/segments_raw',
    'morgoth1/data/internal_dataset/IIIC/segments_raw',
    'morgoth1/data/pretrain',
    # Note: iiic-freq3/data/eeg has pre-processed bipolar files, not raw monopolar
]


def find_on_s3(original_filename, aws_key, aws_secret):
    """Try to find a file across known S3 paths. Returns full S3 path or None."""
    env = os.environ.copy()
    env['AWS_ACCESS_KEY_ID'] = aws_key
    env['AWS_SECRET_ACCESS_KEY'] = aws_secret

    for prefix in S3_SEARCH_PATHS:
        s3_path = f'{S3_BUCKET}/{prefix}/{original_filename}'
        result = subprocess.run(
            ['aws', 's3', 'ls', s3_path, '--region', AWS_REGION],
            capture_output=True, text=True, env=env, timeout=15
        )
        if result.stdout.strip():
            return s3_path
    return None


def download_and_extract(s3_path, aws_key, aws_secret):
    """Download 10-min file, extract center 10-sec, return monopolar array."""
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        env = os.environ.copy()
        env['AWS_ACCESS_KEY_ID'] = aws_key
        env['AWS_SECRET_ACCESS_KEY'] = aws_secret

        result = subprocess.run(
            ['aws', 's3', 'cp', s3_path, tmp_path, '--region', AWS_REGION, '--quiet'],
            capture_output=True, text=True, env=env, timeout=120
        )
        if result.returncode != 0:
            return None, f'Download failed: {result.stderr.strip()}'

        # Try HDF5 first (v7.3), then scipy
        try:
            import h5py
            with h5py.File(tmp_path, 'r') as f:
                data = f['data'][:]
                fs = float(f['Fs'][0, 0])
        except:
            mat = sio.loadmat(tmp_path)
            dk = [k for k in mat if not k.startswith('_')][0]
            data = mat[dk].astype(np.float64)
            fs = FS

        # data shape: (n_samples, n_channels) or (n_channels, n_samples)
        if data.shape[0] < data.shape[1]:
            data = data.T  # make (samples, channels)

        n_samples = data.shape[0]
        n_channels = data.shape[1]

        if n_samples > 10000:
            # 10-minute file — extract center 10 seconds
            start = int(CENTER_OFFSET * fs)
            end = int((CENTER_OFFSET + 10) * fs)
            if end > n_samples:
                start = n_samples - int(10 * fs)
                end = n_samples
            segment = data[start:end, :]
        else:
            # Already a 10-second segment (shouldn't happen but handle it)
            segment = data

        # Keep first 19 channels (monopolar, drop EKG if 20th)
        if segment.shape[1] > 19:
            segment = segment[:, :19]

        # Transpose to (channels, samples) for storage
        segment = segment.T

        return segment, None

    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max', type=int, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  Re-download existing files as monopolar")
    print("=" * 70)

    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    # Find files that are currently bipolar (18ch) and have an original_filename
    # that could be on S3
    candidates = sl[
        (sl['original_filename'].notna()) &
        (sl['original_filename'] != '') &
        (sl['original_source'].isin(['s3_morgoth', 'morgoth1_s3',
                                      'rda_harvest_manifest', 'bipd_harvest_manifest',
                                      'harvest_manifest', 'harvest', 'external_drive']))
    ].copy()

    # Check which are currently bipolar
    bipolar = []
    for _, row in candidates.iterrows():
        mat_path = EEG_DIR / row['mat_file']
        if not mat_path.exists():
            continue
        try:
            mat = sio.loadmat(str(mat_path))
            dk = [k for k in mat if not k.startswith('_')][0]
            n_ch = min(mat[dk].shape)
            if n_ch == 18:
                bipolar.append(row)
        except:
            continue

    print(f"Bipolar files with S3 original_filename: {len(bipolar)}")

    if args.max:
        bipolar = bipolar[:args.max]
        print(f"Limited to {args.max}")

    if args.dry_run:
        print("(dry run)")
        return

    aws_key, aws_secret = load_aws_credentials()

    # Build S3 path cache: list all files in each search path
    print("Building S3 file index...")
    s3_index = {}
    env = os.environ.copy()
    env['AWS_ACCESS_KEY_ID'] = aws_key
    env['AWS_SECRET_ACCESS_KEY'] = aws_secret

    for prefix in S3_SEARCH_PATHS:
        result = subprocess.run(
            ['aws', 's3', 'ls', f'{S3_BUCKET}/{prefix}/', '--region', AWS_REGION],
            capture_output=True, text=True, env=env, timeout=60
        )
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                parts = line.split()
                if len(parts) >= 4:
                    fname = parts[-1]
                    s3_index[fname] = f'{S3_BUCKET}/{prefix}/{fname}'
            print(f"  {prefix}: {len([l for l in result.stdout.strip().split(chr(10)) if l])} files")

    print(f"Total S3 index: {len(s3_index)} files")

    # Download loop
    n_success = 0
    n_fail = 0
    n_not_found = 0

    for i, row in enumerate(bipolar):
        orig = str(row['original_filename'])
        mat_file = str(row['mat_file'])

        # Try multiple lookup keys
        s3_path = None
        for key in [orig, mat_file]:
            if key and key != 'nan':
                s3_path = s3_index.get(key)
                if s3_path:
                    break

        if not s3_path:
            n_not_found += 1
            continue

        segment, error = download_and_extract(s3_path, aws_key, aws_secret)

        if segment is not None and min(segment.shape) >= 19:
            out_path = EEG_DIR / mat_file
            sio.savemat(str(out_path), {'data': segment, 'Fs': FS})
            n_success += 1
        elif segment is not None:
            n_fail += 1
            if n_fail <= 5:
                print(f"  BAD SHAPE: {mat_file}: {segment.shape}")
        else:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAILED: {mat_file}: {error}")

        if (i + 1) % 100 == 0 or (i + 1) == len(bipolar):
            print(f"  [{i + 1}/{len(bipolar)}] success={n_success} fail={n_fail} not_found={n_not_found}")

    print(f"\nDone: {n_success} replaced, {n_fail} failed, {n_not_found} not found on S3")


if __name__ == '__main__':
    main()
