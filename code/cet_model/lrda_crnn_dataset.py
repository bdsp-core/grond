"""LRDA frequency dataset for the CRNN. 5-fold patient-stratified CV.

For each fold, the 200-segment manifest is the evaluation set; LRDA
segments with at least one expert frequency label and not in the
evaluation patient set are training data.

Targets are computed as the median of available expert frequencies
(MW, SZ, TZ on the 200-segment manifest; MW only on legacy LRDA).
"""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'


def _load_segment(mat_file: str) -> np.ndarray | None:
    """Load 18-channel bipolar EEG (2000 samples). Returns None on failure."""
    import scipy.io as sio
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] == 19:
        # Convert monopolar to bipolar
        MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                'Fp2','F4','C4','P4','F8','T4','T6','O2']
        PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
                 ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
                 ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
                 ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
                 ('Fz','Cz'),('Cz','Pz')]
        idx = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])
        seg = seg[idx[:, 0]] - seg[idx[:, 1]]
    if seg.shape[0] != 18:
        return None
    return seg.astype(np.float32)


def _build_label_map() -> dict:
    """Return {mat_file: {'patient_id': str, 'targets': [freqs from each rater]}}.

    Includes:
    - 200-segment manifest LRDA segments with expert labels
    - Legacy LRDA segments (subtype=='lrda', any rater MW/SZ/TZ/PH/LB)
    """
    m = defaultdict(lambda: {'patient_id': None, 'targets': [], 'in_manifest': False, 'subtype': None})

    # Manifest segments
    with open(TASKS_DIR / 'manifest.csv') as f:
        for row in csv.DictReader(f):
            m[row['mat_file']]['patient_id'] = row['patient_id']
            m[row['mat_file']]['in_manifest'] = True

    # Pull all rater frequency labels for LRDA segments
    # First identify which mat_files are LRDA via segment_labels.csv
    lrda_mfs = set()
    pid_lookup = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        for row in csv.DictReader(f):
            if row.get('subtype', '').lower() == 'lrda':
                if row.get('excluded', '').strip().lower() in ('true', '1', 'yes'):
                    continue
                lrda_mfs.add(row['mat_file'])
                pid_lookup[row['mat_file']] = row.get('patient_id', '')

    # Frequency labels from labels.csv
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            if row['rater'] not in ('MW', 'SZ', 'TZ', 'PH', 'LB'):
                continue
            if row['label_type'] != 'frequency_hz':
                continue
            mf = row['mat_file']
            if mf not in lrda_mfs:
                continue
            try:
                freq = float(row['value'])
            except ValueError:
                continue
            if freq <= 0 or freq > 5:
                continue
            m[mf]['patient_id'] = m[mf]['patient_id'] or pid_lookup.get(mf, '')
            m[mf]['targets'].append(freq)
            m[mf]['subtype'] = 'lrda'

    # Drop entries without targets or patient_id
    out = {mf: v for mf, v in m.items() if v['targets'] and v['patient_id']}
    return out


class LRDADataset(Dataset):
    """One sample per LRDA mat_file. Target = median of available expert
    frequencies (clipped to [0.5, 4]); input = (18, 2000) bipolar EEG."""

    def __init__(self, items: list[dict], augment: bool = False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        mf = it['mat_file']
        seg = _load_segment(mf)
        if seg is None:
            # Return zero-filled sample if EEG missing (shouldn't happen post-filter)
            seg = np.zeros((18, 2000), dtype=np.float32)
        target_freq = float(np.median(it['targets']))
        target_freq = max(0.5, min(4.0, target_freq))

        if self.augment:
            # Amplitude scale (0.8-1.2x)
            seg = seg * np.random.uniform(0.8, 1.2)
            # Additive Gaussian noise (20-40 dB SNR)
            sig_pow = np.mean(seg ** 2) + 1e-9
            snr_db = np.random.uniform(20.0, 40.0)
            noise_pow = sig_pow / (10 ** (snr_db / 10))
            seg = seg + np.random.randn(*seg.shape).astype(np.float32) * np.sqrt(noise_pow)
            # Channel dropout (p=0.15 per channel)
            mask = (np.random.rand(seg.shape[0]) > 0.15).astype(np.float32)
            seg = seg * mask[:, None]
            # Time shift (+/- 100 samples), zero-pad
            shift = int(np.random.uniform(-100, 100))
            if shift != 0:
                seg = np.roll(seg, shift, axis=1)
                if shift > 0:
                    seg[:, :shift] = 0
                else:
                    seg[:, shift:] = 0
            # Hemisphere swap with 50% prob (target frequency unchanged; this is a label-preserving aug)
            if np.random.rand() < 0.5:
                # swap left and right channel groups in the bipolar montage
                # Left  bipolar idx: 0,1,2,3, 8,9,10,11
                # Right bipolar idx: 4,5,6,7, 12,13,14,15
                # Midline: 16, 17
                left = np.array([0, 1, 2, 3, 8, 9, 10, 11])
                right = np.array([4, 5, 6, 7, 12, 13, 14, 15])
                tmp = seg[left].copy()
                seg[left] = seg[right]
                seg[right] = tmp

        # Normalize per channel (z-score). Replace NaN/Inf and use a stable
        # standard deviation to avoid blowing up flat channels.
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        std = seg.std(axis=1, keepdims=True)
        std = np.where(std < 1.0, 1.0, std)  # don't amplify near-zero channels
        seg = (seg - seg.mean(axis=1, keepdims=True)) / std
        seg = np.clip(seg, -10, 10)
        # Final NaN guard
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(seg), torch.tensor(np.log(target_freq), dtype=torch.float32), mf


def make_folds(label_map: dict, n_folds: int = 5, seed: int = 42) -> list[tuple[list, list]]:
    """Return list of (train_items, val_items) splits.

    Folds are patient-stratified across the 200-segment manifest. Legacy
    (non-manifest) LRDA segments are added to every training fold.
    """
    manifest_items = [
        {'mat_file': mf, 'patient_id': v['patient_id'], 'targets': v['targets']}
        for mf, v in label_map.items() if v['in_manifest']
    ]
    legacy_items = [
        {'mat_file': mf, 'patient_id': v['patient_id'], 'targets': v['targets']}
        for mf, v in label_map.items() if not v['in_manifest']
    ]

    # Patient-stratified split of manifest
    rng = np.random.default_rng(seed)
    pids = sorted({it['patient_id'] for it in manifest_items})
    rng.shuffle(pids)
    fold_assign = {pid: i % n_folds for i, pid in enumerate(pids)}

    folds = []
    for k in range(n_folds):
        train, val = [], []
        for it in manifest_items:
            if fold_assign[it['patient_id']] == k:
                val.append(it)
            else:
                train.append(it)
        # Add legacy items to training (none are in val)
        train.extend(legacy_items)
        folds.append((train, val))
    return folds
