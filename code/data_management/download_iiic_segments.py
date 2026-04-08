#!/usr/bin/env python3
"""Download IIIC segments from S3 and extract 10-second windows.

Each file on S3 is a 10-minute recording (120,000 samples @ 200 Hz, 20 monopolar channels).
The labeled 10-second segment is at the center (samples 60,000-62,000).

Usage:
    python code/data_management/download_iiic_segments.py [--min-votes 10] [--patterns lrda,grda,lpd,gpd,seizure]
"""
import os
import sys
import ast
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
S3_PREFIX = 'morgoth1/data/internal_dataset/IIIC/segments_raw'
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
SEGMENT_DURATION = 10  # seconds
CENTER_OFFSET = 300  # seconds into the 10-min recording


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


def download_and_extract(file_name, aws_key, aws_secret):
    """Download a 10-min file from S3, extract the center 10-sec segment, return as numpy array."""
    s3_path = f'{S3_BUCKET}/{S3_PREFIX}/{file_name}.mat'

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
            return None, f'S3 download failed: {result.stderr.strip()}'

        # Read the file (HDF5 / v7.3 format)
        import h5py
        with h5py.File(tmp_path, 'r') as f:
            data = f['data'][:]  # (120000, 20)
            fs = float(f['Fs'][0, 0])

        # Extract center 10-second window
        start_sample = int(CENTER_OFFSET * fs)
        end_sample = int((CENTER_OFFSET + SEGMENT_DURATION) * fs)
        segment = data[start_sample:end_sample, :]  # (2000, 20)

        # Transpose to (channels, samples) and drop EKG (channel 20) if present
        if segment.shape[1] > 19:
            segment = segment[:, :19].T  # (19, 2000) — 19 monopolar channels (no EKG)
        else:
            segment = segment.T

        return segment, None

    except Exception as e:
        return None, str(e)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-votes', type=int, default=10)
    parser.add_argument('--patterns', type=str, default='lrda,grda,lpd,gpd,seizure')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-downloads', type=int, default=None)
    args = parser.parse_args()

    patterns = set(args.patterns.split(','))

    print("=" * 70)
    print(f"  IIIC Segment Downloader (min votes: {args.min_votes})")
    print("=" * 70)

    # Load IIIC events
    events_path = LABELS_DIR / 'archive_labels' / 'list_events_20241129.csv'
    events = pd.read_csv(str(events_path))
    events['votes'] = events['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(
        lambda s: ast.literal_eval(str(s)))
    events['n_votes'] = events['votes'].apply(sum)
    events['plurality'] = events['votes'].apply(
        lambda v: ['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda'][max(range(6), key=lambda i: v[i])])

    # Filter
    target = events[
        (events['n_votes'] >= args.min_votes) &
        (events['plurality'].isin(patterns))
    ].copy()
    print(f"Target segments: {len(target)}")
    for p in sorted(patterns):
        print(f"  {p}: {(target.plurality == p).sum()}")

    # Check which we already have
    existing = set(os.listdir(str(EEG_DIR)))
    target['mat_name'] = target['file_name'] + '.mat'
    target['already_have'] = target['mat_name'].isin(existing)
    to_download = target[~target['already_have']]
    print(f"\nAlready on disk: {target.already_have.sum()}")
    print(f"To download: {len(to_download)}")

    if args.dry_run:
        print("(dry run — not downloading)")
        return

    if len(to_download) == 0:
        print("Nothing to download!")
        return

    if args.max_downloads:
        to_download = to_download.head(args.max_downloads)
        print(f"Limited to {args.max_downloads} downloads")

    # Load AWS credentials
    aws_key, aws_secret = load_aws_credentials()

    # Download loop
    n_success = 0
    n_fail = 0
    for i, (_, row) in enumerate(to_download.iterrows()):
        file_name = row['file_name']
        mat_name = row['mat_name']
        votes = row['votes']

        segment, error = download_and_extract(file_name, aws_key, aws_secret)

        if segment is not None:
            # Save as .mat (v5 format, compatible with scipy.io.loadmat)
            out_path = EEG_DIR / mat_name
            sio.savemat(str(out_path), {'data': segment, 'Fs': FS})
            n_success += 1
        else:
            n_fail += 1
            if n_fail <= 5:
                print(f"  FAILED: {file_name}: {error}")

        if (i + 1) % 50 == 0 or (i + 1) == len(to_download):
            print(f"  [{i + 1}/{len(to_download)}] success={n_success} fail={n_fail}")

    print(f"\nDone: {n_success} downloaded, {n_fail} failed")

    # Update segment_labels.csv
    if n_success > 0:
        print("\nUpdating segment_labels.csv...")
        _update_labels(target)


def _update_labels(target):
    """Add newly downloaded segments to segment_labels.csv."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    existing_mats = set(sl['mat_file'])

    new_rows = []
    for _, row in target.iterrows():
        mat_name = row['mat_name']
        if mat_name in existing_mats:
            continue
        if not (EEG_DIR / mat_name).exists():
            continue

        votes = row['votes']
        cats = ['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']
        total = sum(votes)
        winner_idx = max(range(6), key=lambda i: votes[i])

        new_rows.append({
            'mat_file': mat_name,
            'segment_id': mat_name.replace('.mat', ''),
            'patient_id': str(row['bdsp_mrn']),
            'subtype': cats[winner_idx],
            'subtype_source': 'iiic_segment_vote',
            'vote_other': votes[0],
            'vote_seizure': votes[1],
            'vote_lpd': votes[2],
            'vote_gpd': votes[3],
            'vote_lrda': votes[4],
            'vote_grda': votes[5],
            'n_votes': total,
            'plurality': cats[winner_idx],
            'plurality_frac': round(votes[winner_idx] / total, 3) if total > 0 else '',
            'original_source': 'iiic_s3_download',
            'original_filename': f'{row["file_name"]}.mat',
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Fill missing columns with empty
        for col in sl.columns:
            if col not in new_df.columns:
                new_df[col] = ''
        new_df = new_df[sl.columns]
        sl = pd.concat([sl, new_df], ignore_index=True)
        sl.to_csv(str(LABELS_DIR / 'segment_labels.csv'), index=False)
        print(f"Added {len(new_rows)} new rows to segment_labels.csv (total: {len(sl)})")
    else:
        print("No new rows to add")


if __name__ == '__main__':
    main()
