"""
E2E Dataset — Load EEG segments with discharge timing, frequency, and laterality labels.

Handles:
  - Loading 19-channel monopolar EEG from .mat files
  - Converting to 18-channel bipolar montage via fcn_getBanana
  - Loading discharge times from discharge_times.json
  - Loading laterality and frequency from segment_labels.csv
  - Data augmentation: amplitude scaling, gaussian noise, channel dropout
"""

import json
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from pathlib import Path
from torch.utils.data import Dataset

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from pd_pointiness_acf import fcn_getBanana

EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
FS = 200
DURATION_S = 10.0
N_SAMPLES = int(FS * DURATION_S)  # 2000


def _resolve_eeg_path(key, eeg_files_set):
    """Try to resolve a discharge_times key to an EEG .mat file path."""
    candidates = [
        f'{key}_seg000.mat',
        f'{key}.mat',
    ]
    for c in candidates:
        if c in eeg_files_set:
            return EEG_DIR / c
    return None


def build_sample_list():
    """Build list of (key, eeg_path, discharge_times, freq, laterality) samples.

    Only includes samples that have both discharge timing data and an EEG file.
    """
    # Load discharge times
    dt_path = LABELS_DIR / 'discharge_times.json'
    with open(dt_path) as f:
        discharge_times = json.load(f)

    # Load segment labels
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')

    # Build lookups from segment_labels
    # patient_id -> row for frequency and laterality
    pid_to_row = {}
    mat_to_row = {}
    for _, row in sl.iterrows():
        pid_to_row[str(row['patient_id'])] = row
        mat_to_row[row['mat_file']] = row

    # Get available EEG files
    eeg_files = set(f.name for f in EEG_DIR.iterdir() if f.suffix == '.mat')

    samples = []
    for key, dt_entry in discharge_times.items():
        # Must have global_times
        gt_times = dt_entry.get('global_times', [])
        if not gt_times or len(gt_times) < 2:
            continue

        # Resolve EEG file
        eeg_path = _resolve_eeg_path(key, eeg_files)
        if eeg_path is None:
            continue

        # Get frequency and laterality from discharge_times or segment_labels
        freq = dt_entry.get('gold_standard_freq') or dt_entry.get('selected_freq')
        laterality = dt_entry.get('laterality', None)

        # Try segment_labels if missing
        mat_name = eeg_path.name
        if mat_name in mat_to_row:
            row = mat_to_row[mat_name]
            if freq is None and pd.notna(row.get('expert_freq_hz')):
                freq = float(row['expert_freq_hz'])
            if laterality is None and pd.notna(row.get('laterality')):
                laterality = row['laterality']

        # Also try patient_id lookup
        if key in pid_to_row:
            row = pid_to_row[key]
            if freq is None and pd.notna(row.get('expert_freq_hz')):
                freq = float(row['expert_freq_hz'])
            if laterality is None and pd.notna(row.get('laterality')):
                laterality = row['laterality']

        # Encode laterality: left=0, right=1, bilateral/None -> skip for lat loss
        lat_label = None
        if laterality == 'left':
            lat_label = 0.0
        elif laterality == 'right':
            lat_label = 1.0
        elif laterality == 'bilateral':
            lat_label = 0.5

        # Get patient_id for stratified CV
        patient_id = dt_entry.get('subtype', key)  # fallback
        # Try to extract from key or mat file
        if key.startswith('sub-'):
            # Extract patient ID from sub-S0001XXXXXXXXX_TIMESTAMP format
            parts = key.split('_')
            patient_id = parts[0]  # sub-S0001...
        else:
            patient_id = key

        samples.append({
            'key': key,
            'eeg_path': str(eeg_path),
            'gt_times': np.array(gt_times, dtype=np.float32),
            'freq': float(freq) if freq is not None else None,
            'lat_label': lat_label,
            'patient_id': patient_id,
        })

    return samples


class E2EDataset(Dataset):
    """Dataset for E2E discharge detector training."""

    def __init__(self, samples, augment=True, hpp_cache=None):
        """
        Args:
            samples: list of dicts from build_sample_list()
            augment: whether to apply data augmentation
            hpp_cache: dict of key -> (18, 125) numpy arrays, or None
        """
        self.samples = samples
        self.augment = augment
        self.hpp_cache = hpp_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load EEG
        mat = sio.loadmat(sample['eeg_path'])
        eeg_mono = mat['data'].astype(np.float32)  # (19, 2000)

        # Convert to bipolar montage
        eeg_bipolar = fcn_getBanana(eeg_mono).astype(np.float32)  # (18, 2000)

        # Ensure correct length
        if eeg_bipolar.shape[1] < N_SAMPLES:
            pad = N_SAMPLES - eeg_bipolar.shape[1]
            eeg_bipolar = np.pad(eeg_bipolar, ((0, 0), (0, pad)))
        elif eeg_bipolar.shape[1] > N_SAMPLES:
            eeg_bipolar = eeg_bipolar[:, :N_SAMPLES]

        # Augmentation
        if self.augment:
            eeg_bipolar = self._augment(eeg_bipolar)

        # Convert to tensors
        eeg_tensor = torch.from_numpy(eeg_bipolar)  # (18, 2000)

        # Ground truth discharge times (variable length, pad to 30)
        gt_times = sample['gt_times']
        n_gt = len(gt_times)
        gt_times_padded = np.zeros(30, dtype=np.float32)
        gt_times_padded[:min(n_gt, 30)] = gt_times[:30]
        gt_mask = np.zeros(30, dtype=np.float32)
        gt_mask[:min(n_gt, 30)] = 1.0

        # Frequency (-1 if unknown)
        freq = sample['freq'] if sample['freq'] is not None else -1.0

        # Laterality (-1 if unknown)
        lat = sample['lat_label'] if sample['lat_label'] is not None else -1.0

        # HPP features (from cache or placeholder)
        if self.hpp_cache is not None and sample['key'] in self.hpp_cache:
            hpp = self.hpp_cache[sample['key']].astype(np.float32)  # (18, 125)
        else:
            hpp = np.zeros((18, 125), dtype=np.float32)

        return {
            'eeg': eeg_tensor,
            'hpp': torch.from_numpy(hpp),
            'gt_times': torch.from_numpy(gt_times_padded),
            'gt_mask': torch.from_numpy(gt_mask),
            'n_gt': n_gt,
            'freq': torch.tensor(freq, dtype=torch.float32),
            'lat': torch.tensor(lat, dtype=torch.float32),
            'key': sample['key'],
            'patient_id': sample['patient_id'],
        }

    def _augment(self, eeg):
        """Apply data augmentation to bipolar EEG."""
        # 1. Amplitude scaling (0.8-1.2)
        scale = np.random.uniform(0.8, 1.2)
        eeg = eeg * scale

        # 2. Gaussian noise (SNR ~25 dB)
        if np.random.random() < 0.5:
            signal_power = np.mean(eeg ** 2, axis=1, keepdims=True) + 1e-10
            snr_db = np.random.uniform(20, 30)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.randn(*eeg.shape).astype(np.float32) * np.sqrt(noise_power)
            eeg = eeg + noise

        # 3. Channel dropout (10% of channels)
        if np.random.random() < 0.3:
            n_drop = max(1, int(0.1 * eeg.shape[0]))
            drop_idx = np.random.choice(eeg.shape[0], n_drop, replace=False)
            eeg[drop_idx] = 0.0

        return eeg


def custom_collate(batch):
    """Custom collate that handles HPP tensor."""
    eeg = torch.stack([b['eeg'] for b in batch])
    hpp = torch.stack([b['hpp'] for b in batch])
    gt_times = torch.stack([b['gt_times'] for b in batch])
    gt_mask = torch.stack([b['gt_mask'] for b in batch])
    n_gt = [b['n_gt'] for b in batch]
    freq = torch.stack([b['freq'] for b in batch])
    lat = torch.stack([b['lat'] for b in batch])
    keys = [b['key'] for b in batch]
    patient_ids = [b['patient_id'] for b in batch]

    return {
        'eeg': eeg,
        'hpp': hpp,
        'gt_times': gt_times,
        'gt_mask': gt_mask,
        'n_gt': n_gt,
        'freq': freq,
        'lat': lat,
        'key': keys,
        'patient_id': patient_ids,
    }


def load_hpp_cache():
    """Load precomputed HPP cache if available."""
    cache_path = PROJECT_DIR / 'data' / 'e2e_cache' / 'hpp_cache.npz'
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        # Convert to dict
        cache = {k: data[k] for k in data.files}
        print(f"Loaded HPP cache with {len(cache)} entries")
        return cache
    return None
