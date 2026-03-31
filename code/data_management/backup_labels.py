#!/usr/bin/env python3
"""Backup all label files to a timestamped directory.

Usage:
    python code/data_management/backup_labels.py
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'

FILES_TO_BACKUP = [
    'segment_labels.csv',
    'annotations.csv',
    'segments.csv',
    'channel_involvement.json',
    'channel_pseudolabels.json',
    'discharge_times.json',
    'rda_wave_labels.json',
]


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = LABELS_DIR / f'backup_{timestamp}'
    backup_dir.mkdir(parents=True, exist_ok=True)

    backed_up = []
    for fname in FILES_TO_BACKUP:
        src = LABELS_DIR / fname
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / fname))
            backed_up.append(fname)

    # Also backup batch JSONs from archive_labels
    archive_dir = LABELS_DIR / 'archive_labels'
    if archive_dir.exists():
        for bf in sorted(archive_dir.glob('*_laterality_batch*.json')):
            shutil.copy2(str(bf), str(backup_dir / bf.name))
            backed_up.append(f'archive_labels/{bf.name}')

    print(f"Backup created: {backup_dir}")
    print(f"Files: {len(backed_up)}")
    for f in backed_up:
        print(f"  {f}")


if __name__ == '__main__':
    main()
