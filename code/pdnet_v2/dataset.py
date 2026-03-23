"""
PDNetV2 Dataset.

Builds a PyTorch Dataset from:
  - EEG segments (18, 2000) at 200 Hz (10 seconds)
  - Discharge timing labels from discharge_times.json

For each patient with review_status='ground_truth' and >=2 discharge times,
we use one 10-second EEG segment aligned to the discharge times.

Targets (all at 100 Hz = 1000 bins for 10 seconds):
  - y_event  (1000,) float: Gaussian bumps at discharge centers (sigma=2 bins)
  - y_active (1000,) float: 1.0 in active regions (padded ±0.25s, smooth edges)
  - y_freq   (1000,) float: local log-frequency from IPI, interpolated
  - y_freq_mask (1000,) float: 1.0 where y_freq is defined

Segment labels:
  - y_subtype int: 0=lpd, 1=gpd
  - y_lat int: 0=left, 1=right, 2=unknown
"""

import sys
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import scipy.io as sio

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'

FS = 200          # EEG sample rate
TARGET_FS = 100   # Target/output sample rate (100 Hz = 1000 bins for 10s)
N_BINS = 1000     # Number of output bins (10s at 100 Hz)
DURATION = 10.0   # Segment duration in seconds

SIGMA_BINS = 2.0           # Gaussian bump sigma for event target (2 bins = 20ms)
GAP_THRESH = 1.8           # Seconds gap to split active regions
ACTIVE_PAD = 0.25          # Seconds of padding around active regions
ACTIVE_SMOOTH_BINS = 5     # Bins for sigmoid-like edge smoothing

SUBTYPE_MAP = {'lpd': 0, 'gpd': 1}
LAT_MAP = {'left': 0, 'right': 1, 'bilateral': 2, 'unknown': 2, '': 2}


def _load_segment(mat_path, montage, n_channels):
    """Load a .mat segment and return (18, 2000) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float32)
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float32)
    # Ensure shape (18, 2000)
    if data.shape[1] < 2000:
        data = np.pad(data, ((0, 0), (0, 2000 - data.shape[1])))
    elif data.shape[1] > 2000:
        data = data[:, :2000]
    return data  # (18, 2000)


def _make_event_target(discharge_times, n_bins=N_BINS, duration=DURATION, sigma=SIGMA_BINS):
    """
    Create Gaussian bump target at each discharge time.
    Returns (n_bins,) float32 array in [0, 1].
    """
    target = np.zeros(n_bins, dtype=np.float32)
    bin_times = np.arange(n_bins) / TARGET_FS  # seconds
    for t in discharge_times:
        if 0 <= t <= duration:
            bin_idx = t * TARGET_FS
            # Gaussian centered at bin_idx
            bump = np.exp(-0.5 * ((np.arange(n_bins) - bin_idx) / sigma) ** 2).astype(np.float32)
            target = np.maximum(target, bump)
    return target


def _make_active_target(discharge_times, n_bins=N_BINS, duration=DURATION,
                         gap_thresh=GAP_THRESH, pad=ACTIVE_PAD, smooth=ACTIVE_SMOOTH_BINS):
    """
    Create binary active region target.
    Returns (n_bins,) float32 array in [0, 1] with smooth edges.
    """
    if len(discharge_times) == 0:
        return np.zeros(n_bins, dtype=np.float32)

    times = sorted(discharge_times)
    target = np.zeros(n_bins, dtype=np.float32)

    # Group into runs by gap_thresh
    runs = []
    run_start = times[0]
    run_end = times[0]
    for t in times[1:]:
        if t - run_end > gap_thresh:
            runs.append((run_start, run_end))
            run_start = t
        run_end = t
    runs.append((run_start, run_end))

    for (start, end) in runs:
        s_bin = int(max(0, (start - pad) * TARGET_FS))
        e_bin = int(min(n_bins - 1, (end + pad) * TARGET_FS))
        target[s_bin:e_bin + 1] = 1.0

    # Smooth edges with Gaussian
    if smooth > 0:
        target = gaussian_filter1d(target, sigma=smooth)
        target = np.clip(target, 0, 1)

    return target.astype(np.float32)


def _make_freq_target(discharge_times, active_target, n_bins=N_BINS, duration=DURATION):
    """
    Create log-frequency target from inter-pulse intervals (IPI).
    Returns:
        y_freq      (n_bins,) float32 - log(frequency) where defined, 0 elsewhere
        y_freq_mask (n_bins,) float32 - 1.0 where frequency is defined
    """
    y_freq = np.zeros(n_bins, dtype=np.float32)
    y_freq_mask = np.zeros(n_bins, dtype=np.float32)

    if len(discharge_times) < 2:
        return y_freq, y_freq_mask

    times = sorted(discharge_times)
    # IPI = inter-pulse interval in seconds (skip zero-gap pairs)
    ipi_times = []
    ipi_freqs = []
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt > 1e-6:  # skip duplicates
            ipi_times.append((times[i] + times[i + 1]) / 2.0)
            ipi_freqs.append(1.0 / dt)

    if len(ipi_times) < 1:
        return y_freq, y_freq_mask

    # Interpolate log-frequency over time
    bin_times = np.arange(n_bins) / TARGET_FS

    # Active mask (binary, thresholded)
    active_binary = active_target > 0.5

    # Only define freq where active
    if not np.any(active_binary):
        return y_freq, y_freq_mask

    # Linear interpolation of log-freq at each active bin
    ipi_times_arr = np.array(ipi_times)
    log_freqs = np.log(np.clip(ipi_freqs, 0.1, 10.0))

    for i in range(n_bins):
        if not active_binary[i]:
            continue
        t = bin_times[i]
        # Find nearest IPI measurements
        if len(ipi_times_arr) == 1:
            y_freq[i] = log_freqs[0]
        else:
            # Linear interpolation / extrapolation
            y_freq[i] = float(np.interp(t, ipi_times_arr, log_freqs))
        y_freq_mask[i] = 1.0

    return y_freq, y_freq_mask


def _z_score(segment):
    """Z-score normalize each channel independently. Returns float32.
    NaN values are replaced with 0 after normalization.
    """
    seg = segment.astype(np.float32)
    # Replace NaN in raw data with 0
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    means = seg.mean(axis=1, keepdims=True)
    stds = seg.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-6, 1.0, stds)
    result = (seg - means) / stds
    # Final safety check
    result = np.nan_to_num(result, nan=0.0, posinf=5.0, neginf=-5.0)
    return result


class PDNetDataset(Dataset):
    """
    Dataset for PDNetV2.

    Each item is one patient's EEG segment with associated temporal targets.
    """

    def __init__(self, patient_ids, segments_by_patient, hpp_data,
                 df_patients, augment=False):
        """
        Args:
            patient_ids: list of patient IDs to include
            segments_by_patient: dict {pid: [array(18,2000), ...]}
            hpp_data: dict from discharge_times.json
            df_patients: DataFrame with subtype, laterality columns
            augment: bool, whether to apply data augmentation
        """
        self.augment = augment
        self.items = []  # list of (segment, y_event, y_active, y_freq, y_freq_mask, y_subtype, y_lat)

        df_patients = df_patients.set_index('patient_id')

        for pid in patient_ids:
            pid_str = str(pid)

            # Get discharge times
            if pid_str not in hpp_data:
                continue
            hpp = hpp_data[pid_str]
            if hpp.get('review_status') != 'ground_truth':
                continue
            discharge_times = hpp.get('global_times', [])
            if len(discharge_times) < 2:
                continue

            # Get EEG segments
            segs = segments_by_patient.get(pid_str, [])
            if not segs:
                continue

            # Use the highest-variance segment (most informative)
            if len(segs) == 1:
                seg = segs[0]
            else:
                vars_ = [np.var(s) for s in segs]
                seg = segs[np.argmax(vars_)]

            # Get labels from patients df
            if pid_str not in df_patients.index:
                continue
            pat_row = df_patients.loc[pid_str]
            subtype_str = str(pat_row['subtype']).lower()
            lat_str = str(pat_row.get('laterality', '')).lower()
            if pd.isna_str(subtype_str) or subtype_str not in SUBTYPE_MAP:
                # Use hpp subtype as fallback
                subtype_str = str(hpp.get('subtype', '')).lower()
            if subtype_str not in SUBTYPE_MAP:
                continue  # skip if unknown subtype

            y_subtype = SUBTYPE_MAP[subtype_str]
            y_lat = LAT_MAP.get(lat_str, 2)

            # Build targets
            y_event = _make_event_target(discharge_times)
            y_active = _make_active_target(discharge_times)
            y_freq, y_freq_mask = _make_freq_target(discharge_times, y_active)

            self.items.append({
                'seg': seg.astype(np.float32),
                'y_event': y_event,
                'y_active': y_active,
                'y_freq': y_freq,
                'y_freq_mask': y_freq_mask,
                'y_subtype': y_subtype,
                'y_lat': y_lat,
                'pid': pid_str,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        seg = _z_score(item['seg'])  # (18, 2000) float32

        if self.augment:
            seg = self._augment(seg)

        return {
            'eeg': torch.from_numpy(seg),
            'y_event': torch.from_numpy(item['y_event']),
            'y_active': torch.from_numpy(item['y_active']),
            'y_freq': torch.from_numpy(item['y_freq']),
            'y_freq_mask': torch.from_numpy(item['y_freq_mask']),
            'y_subtype': torch.tensor(item['y_subtype'], dtype=torch.long),
            'y_lat': torch.tensor(item['y_lat'], dtype=torch.long),
            'pid': item['pid'],
        }

    def _augment(self, seg):
        """Apply training augmentations."""
        rng = np.random

        # Amplitude scaling: uniform [0.7, 1.3]
        scale = rng.uniform(0.7, 1.3)
        seg = seg * scale

        # Gaussian noise: std=0.05 of signal std
        sig_std = seg.std()
        noise_std = 0.05 * sig_std
        seg = seg + rng.randn(*seg.shape).astype(np.float32) * noise_std

        # Channel dropout: zero 1-2 channels with prob=0.2
        if rng.random() < 0.2:
            n_drop = rng.randint(1, 3)
            drop_channels = rng.choice(18, size=n_drop, replace=False)
            seg[drop_channels] = 0.0

        return seg


def pd_isna_str(s):
    """Check if a string value is NA."""
    return s in ('nan', 'none', 'na', '')


# Monkey-patch the 'pd' reference in the module
import builtins
builtins.__dict__  # no-op

# Fix the pd.isna_str reference in the class
def _fix_import():
    import pandas as pd
    global _pd_isna
    _pd_isna = pd.isna

_fix_import()


# Fix the class method that uses pd.isna_str (patching it properly)
def _patched_isna_str(s):
    return s in ('nan', 'none', 'na', '', 'NaN', 'None')


# Override what was incorrectly written in the Dataset.__init__
# We need to use the module-level function
import pandas as _pd_module


def build_dataset(patient_ids, segments_by_patient, hpp_data, df_patients, augment=False):
    """
    Build a PDNetDataset with proper pandas handling.
    This is the preferred entry point.
    """
    df_patients = df_patients.copy()
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_idx = df_patients.set_index('patient_id')

    items = []
    skipped = 0

    for pid in patient_ids:
        pid_str = str(pid)

        # Get discharge times
        if pid_str not in hpp_data:
            skipped += 1
            continue
        hpp = hpp_data[pid_str]
        if hpp.get('review_status') != 'ground_truth':
            skipped += 1
            continue
        discharge_times = hpp.get('global_times', [])
        if len(discharge_times) < 2:
            skipped += 1
            continue

        # Get EEG segments
        segs = segments_by_patient.get(pid_str, [])
        if not segs:
            skipped += 1
            continue

        # Filter out NaN segments
        segs = [s for s in segs if not np.any(np.isnan(s))]
        if not segs:
            skipped += 1
            continue

        # Use the highest-variance segment
        if len(segs) == 1:
            seg = segs[0]
        else:
            vars_ = [np.var(s) for s in segs]
            seg = segs[int(np.argmax(vars_))]

        # Get labels from patients df
        if pid_str not in df_idx.index:
            skipped += 1
            continue
        pat_row = df_idx.loc[pid_str]
        subtype_str = str(pat_row['subtype']).lower().strip()
        lat_val = pat_row.get('laterality', '')
        lat_str = '' if _pd_module.isna(lat_val) else str(lat_val).lower().strip()

        if subtype_str not in SUBTYPE_MAP:
            # Fallback to hpp subtype
            subtype_str = str(hpp.get('subtype', '')).lower().strip()
        if subtype_str not in SUBTYPE_MAP:
            skipped += 1
            continue

        y_subtype = SUBTYPE_MAP[subtype_str]
        y_lat = LAT_MAP.get(lat_str, 2)

        # Build targets
        y_event = _make_event_target(discharge_times)
        y_active = _make_active_target(discharge_times)
        y_freq, y_freq_mask = _make_freq_target(discharge_times, y_active)

        items.append({
            'seg': seg.astype(np.float32),
            'y_event': y_event,
            'y_active': y_active,
            'y_freq': y_freq,
            'y_freq_mask': y_freq_mask,
            'y_subtype': y_subtype,
            'y_lat': y_lat,
            'pid': pid_str,
        })

    return _InternalDataset(items, augment=augment)


class _InternalDataset(Dataset):
    """Internal dataset that holds pre-built items."""

    def __init__(self, items, augment=False):
        self.items = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        seg = _z_score(item['seg'])  # (18, 2000) float32

        if self.augment:
            seg = self._augment(seg)

        return {
            'eeg': torch.from_numpy(seg),
            'y_event': torch.from_numpy(item['y_event']),
            'y_active': torch.from_numpy(item['y_active']),
            'y_freq': torch.from_numpy(item['y_freq']),
            'y_freq_mask': torch.from_numpy(item['y_freq_mask']),
            'y_subtype': torch.tensor(item['y_subtype'], dtype=torch.long),
            'y_lat': torch.tensor(item['y_lat'], dtype=torch.long),
            'pid': item['pid'],
        }

    def _augment(self, seg):
        """Apply training augmentations."""
        rng = np.random

        # Amplitude scaling: uniform [0.7, 1.3]
        scale = rng.uniform(0.7, 1.3)
        seg = seg * scale

        # Gaussian noise: std=0.05 of signal std
        sig_std = float(seg.std())
        if sig_std > 0:
            noise_std = 0.05 * sig_std
            seg = seg + rng.randn(*seg.shape).astype(np.float32) * float(noise_std)

        # Channel dropout: zero 1-2 channels with prob=0.2
        if rng.random() < 0.2:
            n_drop = rng.randint(1, 3)
            drop_channels = rng.choice(18, size=n_drop, replace=False)
            seg = seg.copy()
            seg[drop_channels] = 0.0

        return seg


if __name__ == '__main__':
    import json
    import sys
    sys.path.insert(0, str(CODE_DIR))

    # Quick test
    with open(str(DATA_DIR / 'labels' / 'discharge_times.json')) as f:
        hpp_data = json.load(f)

    import pandas as pd
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_seg['patient_id'] = df_seg['patient_id'].astype(str)

    # Load a few segments
    from optimization_harness_v2 import _load_mat_as_bipolar
    segments_by_patient = {}
    for pid, group in df_seg.groupby('patient_id'):
        segs = []
        for _, row in group.iterrows():
            mat_path = EEG_DIR / row['mat_file']
            if mat_path.exists():
                try:
                    seg = _load_mat_as_bipolar(mat_path, row['montage'], row['n_channels'])
                    segs.append(seg)
                except Exception:
                    pass
        if segs:
            segments_by_patient[pid] = segs[:3]

    patient_ids = list(hpp_data.keys())[:20]
    ds = build_dataset(patient_ids, segments_by_patient, hpp_data, df_patients, augment=False)
    print(f"Dataset size: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        print("EEG shape:", item['eeg'].shape)
        print("y_event shape:", item['y_event'].shape)
        print("y_active shape:", item['y_active'].shape)
        print("y_freq shape:", item['y_freq'].shape)
        print("y_freq_mask shape:", item['y_freq_mask'].shape)
        print("y_subtype:", item['y_subtype'])
        print("y_lat:", item['y_lat'])
