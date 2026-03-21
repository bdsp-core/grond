"""
Dataset class for the unified multi-task PD/RDA model.

Loads:
  - EEG data (18 channels x 2000 samples) from data/eeg/
  - Subtype label (0=LPD, 1=GPD, 2=LRDA, 3=GRDA)
  - Frequency label (float Hz, or NaN if not available)
  - Per-channel PD labels (18 values: 0, 1, or -1 for null/masked)
  - Per-channel RDA labels (18 values: 0, 1, or -1 for null/masked)
  - Per-channel confidence weights (18 values from pseudolabels)

Sources:
  - data/labels/channel_pseudolabels.json (per-channel PD+RDA labels)
  - data/labels/patients.csv (subtype, gold_standard_freq)
  - data/labels/segments.csv (segment -> .mat file mapping)
  - data/eeg/ (.mat files)
"""

import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path

import torch
from torch.utils.data import Dataset

import sys
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
from pd_pointiness_acf import fcn_getBanana

PROJECT_DIR = CODE_DIR.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'

SUBTYPE_TO_IDX = {'lpd': 0, 'gpd': 1, 'lrda': 2, 'grda': 3}

CONFIDENCE_WEIGHTS = {
    'ground_truth': 2.0,
    'high': 1.0,
    'medium-high': 0.8,
    'medium': 0.5,
    'low': 0.3,
}


def _load_mat_as_bipolar(mat_path, montage, n_channels):
    """Load a .mat file and return (18, N) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    return data


def load_unified_dataset(verbose=True):
    """Load the full unified dataset for multi-task training.

    Returns a list of sample dicts, each with:
        'patient_id': str
        'segment_id': str
        'eeg': (18, 2000) numpy array
        'subtype': int (0=LPD, 1=GPD, 2=LRDA, 3=GRDA)
        'freq': float (Hz, or NaN)
        'pd_labels': (18,) array (-1=null, 0=no PD, 1=PD)
        'rda_labels': (18,) array (-1=null, 0=no RDA, 1=RDA)
        'confidence_weights': (18,) array of floats
    """
    # Load patients CSV (all non-excluded patients, not just those with freq)
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_patients = df_patients[df_patients['excluded'] != True].copy()

    # Load segments CSV
    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    # Load channel pseudolabels
    with open(str(LABELS_DIR / 'channel_pseudolabels.json')) as f:
        pseudolabels = json.load(f)

    # Build patient info lookup
    pid_info = {}
    for _, row in df_patients.iterrows():
        pid = str(row['patient_id'])
        subtype = row['subtype']
        if subtype not in SUBTYPE_TO_IDX:
            continue
        freq = row['gold_standard_freq']
        if pd.isna(freq) or freq <= 0:
            freq = float('nan')
        else:
            freq = float(freq)
        pid_info[pid] = {
            'subtype': SUBTYPE_TO_IDX[subtype],
            'subtype_str': subtype,
            'freq': freq,
        }

    # Only keep patients that are in pseudolabels (have channel labels)
    valid_pids = set(pid_info.keys()) & set(pseudolabels.keys())
    if verbose:
        print(f"Patients with both CSV info and pseudolabels: {len(valid_pids)}")

    # Build per-patient channel labels
    pid_channel_info = {}
    for pid in valid_pids:
        pl = pseudolabels[pid]
        pd_labels = np.full(18, -1, dtype=np.float32)
        rda_labels = np.full(18, -1, dtype=np.float32)
        conf_weights = np.ones(18, dtype=np.float32)

        for ch_str, ch_info in pl['channels'].items():
            ch_idx = int(ch_str)
            if ch_idx >= 18:
                continue

            pd_val = ch_info.get('pd_label')
            if pd_val is not None:
                pd_labels[ch_idx] = float(pd_val)
            else:
                pd_labels[ch_idx] = -1.0  # null/masked

            rda_val = ch_info.get('rda_label')
            if rda_val is not None:
                rda_labels[ch_idx] = float(rda_val)
            else:
                rda_labels[ch_idx] = -1.0

            conf = ch_info.get('confidence', 'medium')
            if conf is None:
                conf = 'medium'
            conf_weights[ch_idx] = CONFIDENCE_WEIGHTS.get(conf, 0.5)

        pid_channel_info[pid] = {
            'pd_labels': pd_labels,
            'rda_labels': rda_labels,
            'confidence_weights': conf_weights,
        }

    # Load EEG segments
    samples = []
    n_loaded = 0
    n_skipped = 0
    max_segs_per_patient = 5

    for pid in sorted(valid_pids):
        pat_segs = df_segments[df_segments['patient_id'] == pid]
        if len(pat_segs) == 0:
            n_skipped += 1
            continue

        loaded_segs = []
        seg_ids = []
        for _, seg_row in pat_segs.iterrows():
            mat_path = EEG_DIR / seg_row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(
                    mat_path, seg_row['montage'], seg_row['n_channels'])
                # Ensure shape is (18, 2000)
                if seg.shape[0] != 18 or seg.shape[1] < 2000:
                    continue
                seg = seg[:, :2000]
                loaded_segs.append(seg.astype(np.float32))
                seg_ids.append(seg_row['segment_id'])
            except Exception:
                continue

        if not loaded_segs:
            n_skipped += 1
            continue

        # Pick top-variance segments (up to max_segs_per_patient)
        if len(loaded_segs) > max_segs_per_patient:
            var_idx = sorted(range(len(loaded_segs)),
                             key=lambda i: -np.var(loaded_segs[i]))
            loaded_segs = [loaded_segs[i] for i in var_idx[:max_segs_per_patient]]
            seg_ids = [seg_ids[i] for i in var_idx[:max_segs_per_patient]]

        info = pid_info[pid]
        ch_info = pid_channel_info[pid]

        for seg, sid in zip(loaded_segs, seg_ids):
            samples.append({
                'patient_id': pid,
                'segment_id': sid,
                'eeg': seg,
                'subtype': info['subtype'],
                'subtype_str': info['subtype_str'],
                'freq': info['freq'],
                'pd_labels': ch_info['pd_labels'].copy(),
                'rda_labels': ch_info['rda_labels'].copy(),
                'confidence_weights': ch_info['confidence_weights'].copy(),
            })
            n_loaded += 1

    if verbose:
        print(f"Loaded {n_loaded} segments from {len(valid_pids) - n_skipped} patients")
        print(f"Skipped {n_skipped} patients (no valid EEG files)")
        # Count subtypes
        from collections import Counter
        st_counts = Counter(s['subtype'] for s in samples)
        for idx, name in [(0, 'LPD'), (1, 'GPD'), (2, 'LRDA'), (3, 'GRDA')]:
            print(f"  {name}: {st_counts.get(idx, 0)} segments")
        n_with_freq = sum(1 for s in samples if np.isfinite(s['freq']))
        print(f"  With frequency labels: {n_with_freq}")

    return samples


class UnifiedPDDataset(Dataset):
    """PyTorch dataset for the unified multi-task model.

    Each sample is one 10-second EEG segment (18 channels x 2000 samples).

    Returns:
        eeg: (18, 2000) tensor, per-channel z-score normalized
        subtype: int (0-3)
        freq: float (log Hz, or NaN)
        freq_mask: float (1.0 if freq available, else 0.0)
        pd_labels: (18,) tensor (-1=masked, 0=no PD, 1=PD)
        rda_labels: (18,) tensor (-1=masked, 0=no RDA, 1=RDA)
        pd_mask: (18,) tensor (1.0 where pd_label != -1, else 0.0)
        rda_mask: (18,) tensor (1.0 where rda_label != -1, else 0.0)
        confidence_weights: (18,) tensor
    """

    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        eeg = s['eeg'].copy()  # (18, 2000)

        # Per-channel z-score normalization
        for ch in range(18):
            # Replace NaN/Inf with 0
            if not np.all(np.isfinite(eeg[ch])):
                eeg[ch] = np.nan_to_num(eeg[ch], nan=0.0, posinf=0.0, neginf=0.0)
            mu = np.mean(eeg[ch])
            std = np.std(eeg[ch])
            if std > 1e-8:
                eeg[ch] = (eeg[ch] - mu) / std
            else:
                eeg[ch] = eeg[ch] - mu

        if self.augment:
            eeg = self._augment(eeg)

        # Frequency: convert to log Hz
        freq = s['freq']
        if np.isfinite(freq) and freq > 0:
            log_freq = np.log(freq)
            freq_mask = 1.0
        else:
            log_freq = 0.0  # placeholder, masked out
            freq_mask = 0.0

        # Channel labels and masks
        pd_labels = s['pd_labels']  # (18,) with -1 for null
        rda_labels = s['rda_labels']
        confidence_weights = s['confidence_weights']

        pd_mask = (pd_labels >= 0).astype(np.float32)  # 1 where not null
        rda_mask = (rda_labels >= 0).astype(np.float32)

        # Replace -1 with 0 for the actual label tensors (masked out anyway)
        pd_labels_clean = np.clip(pd_labels, 0, 1)
        rda_labels_clean = np.clip(rda_labels, 0, 1)

        return (
            torch.from_numpy(eeg),                         # (18, 2000)
            torch.tensor(s['subtype'], dtype=torch.long),   # int
            torch.tensor(log_freq, dtype=torch.float32),    # float
            torch.tensor(freq_mask, dtype=torch.float32),   # float
            torch.from_numpy(pd_labels_clean),              # (18,)
            torch.from_numpy(rda_labels_clean),             # (18,)
            torch.from_numpy(pd_mask),                      # (18,)
            torch.from_numpy(rda_mask),                     # (18,)
            torch.from_numpy(confidence_weights),           # (18,)
        )

    def _augment(self, eeg):
        """Apply data augmentation to all 18 channels.

        - Random amplitude scaling (0.8-1.2x) -- same scale for all channels
        - Random Gaussian noise (SNR 20-40 dB) -- independent per channel
        - Random time shift (+/-50 samples circular) -- same shift for all channels
        """
        # Random amplitude scaling (same for all channels)
        scale = np.random.uniform(0.8, 1.2)
        eeg = eeg * scale

        # Random Gaussian noise (independent per channel)
        snr_db = np.random.uniform(20, 40)
        for ch in range(18):
            signal_power = np.mean(eeg[ch] ** 2)
            if signal_power > 1e-10:
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.random.randn(eeg.shape[1]).astype(np.float32) * np.sqrt(noise_power)
                eeg[ch] = eeg[ch] + noise

        # Random circular time shift (same for all channels)
        shift = np.random.randint(-50, 51)
        if shift != 0:
            eeg = np.roll(eeg, shift, axis=1)

        return eeg
