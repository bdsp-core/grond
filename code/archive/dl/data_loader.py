"""
EEG Data Loaders for classification (Phase 1) and frequency estimation (Phase 2).
"""
import torch, numpy as np, os, sys
from torch.utils.data import Dataset
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Add parent code dir for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from pd_pointiness_acf import fcn_getBanana

# Bipolar montage channel order (from pd_pointiness_acf.py)
MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2']

def preprocess_segment(data, fs=200):
    """Preprocess raw 20-channel data: bipolar montage + bandpass + lowpass.
    Input: (20, N) raw referential. Output: (18, N) preprocessed bipolar.
    """
    from mne.filter import notch_filter, filter_data
    seg = notch_filter(data.astype(np.float64), fs, 60, n_jobs=1, verbose='ERROR')
    seg = filter_data(seg, fs, 0.5, 40, n_jobs=1, verbose='ERROR')
    seg_bi = np.array(fcn_getBanana(seg))  # (18, N)
    # 15Hz lowpass
    b, a = butter(4, 15.0 / (fs/2), btype='low')
    for i in range(seg_bi.shape[0]):
        try:
            seg_bi[i] = filtfilt(b, a, seg_bi[i])
        except:
            pass
    return seg_bi.astype(np.float32)

def normalize_segment(seg):
    """Per-channel z-score normalize, clip to [-10, 10]. Handles NaN."""
    # Replace NaN with 0
    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
    mean = seg.mean(axis=1, keepdims=True)
    std = seg.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    seg = (seg - mean) / std
    return np.clip(seg, -10, 10).astype(np.float32)


class IIICClassificationDataset(Dataset):
    """Phase 1: LPD vs GPD classification from cached .npz file."""

    def __init__(self, npz_path, patient_ids=None, augment=False, fs=200):
        """
        Args:
            npz_path: path to cached .npz with keys 'segments', 'labels', 'patients'
            patient_ids: if provided, only include segments from these patients
            augment: whether to apply data augmentation
        """
        data = np.load(npz_path, allow_pickle=True)
        self.segments = data['segments']    # (N, 18, 2000) preprocessed bipolar
        self.labels = data['labels']        # (N,) 0=LPD, 1=GPD
        self.patients = data['patients']    # (N,) patient IDs
        self.augment = augment
        self.fs = fs

        if patient_ids is not None:
            mask = np.isin(self.patients, patient_ids)
            self.segments = self.segments[mask]
            self.labels = self.labels[mask]
            self.patients = self.patients[mask]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx].copy()  # (18, 2000)
        label = self.labels[idx]

        if self.augment:
            seg = self._augment(seg)

        seg = normalize_segment(seg)
        return torch.from_numpy(seg), torch.tensor(label, dtype=torch.float32)

    def _augment(self, seg):
        """Apply random augmentations that don't change the class label."""
        # Gaussian noise
        if np.random.random() < 0.5:
            noise_scale = 0.1 * np.std(seg, axis=1, keepdims=True)
            seg = seg + np.random.randn(*seg.shape).astype(np.float32) * noise_scale
        # Random amplitude scaling per channel
        if np.random.random() < 0.5:
            scales = np.random.uniform(0.8, 1.2, (18, 1)).astype(np.float32)
            seg = seg * scales
        # Random channel dropout (zero out 1-2 channels)
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, 3)
            drop_idx = np.random.choice(18, n_drop, replace=False)
            seg[drop_idx] = 0.0
        return seg


class IIICFrequencyDataset(Dataset):
    """Phase 2: Frequency estimation with per-expert targets and weak eventness labels."""

    def __init__(self, segments, expert_freqs, weak_eventness=None,
                 patient_ids_filter=None, patients=None, augment=False, fs=200):
        """
        Args:
            segments: (N, 18, 2000) preprocessed bipolar
            expert_freqs: (N, 3) frequencies from [LB, PH, SZ], NaN if unavailable
            weak_eventness: (N, 2000) pseudo-eventness labels, or None
            patient_ids_filter: if provided, only include these patients
            patients: (N,) patient IDs
            augment: whether to apply augmentation (including time-stretch)
        """
        self.segments = segments
        self.expert_freqs = expert_freqs
        self.weak_eventness = weak_eventness
        self.patients = patients
        self.augment = augment
        self.fs = fs

        if patient_ids_filter is not None and patients is not None:
            mask = np.isin(patients, patient_ids_filter)
            self.segments = self.segments[mask]
            self.expert_freqs = self.expert_freqs[mask]
            if self.weak_eventness is not None:
                self.weak_eventness = self.weak_eventness[mask]
            if self.patients is not None:
                self.patients = self.patients[mask]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx].copy()  # (18, 2000)
        freqs = self.expert_freqs[idx].copy()  # (3,)
        eventness = self.weak_eventness[idx].copy() if self.weak_eventness is not None else np.zeros(2000, dtype=np.float32)

        stretch_factor = 1.0
        if self.augment:
            seg, freqs, eventness, stretch_factor = self._augment(seg, freqs, eventness)

        seg = normalize_segment(seg)

        # Log-frequency targets (NaN stays NaN)
        log_freqs = np.full(3, np.nan, dtype=np.float32)
        for i in range(3):
            if np.isfinite(freqs[i]) and freqs[i] > 0:
                log_freqs[i] = np.log(freqs[i])

        return (torch.from_numpy(seg),
                torch.from_numpy(log_freqs),
                torch.from_numpy(eventness.astype(np.float32)))

    def _augment(self, seg, freqs, eventness):
        """Apply augmentations including time-stretch (adjusts target freq)."""
        stretch_factor = 1.0

        # Time-stretch (changes frequency proportionally)
        if np.random.random() < 0.5:
            stretch_factor = np.random.uniform(0.85, 1.15)
            new_len = int(2000 * stretch_factor)
            from scipy.signal import resample
            seg_stretched = np.zeros((18, new_len), dtype=np.float32)
            for i in range(18):
                seg_stretched[i] = resample(seg[i], new_len)
            # Crop or pad to 2000
            if new_len >= 2000:
                start = (new_len - 2000) // 2
                seg = seg_stretched[:, start:start+2000]
            else:
                pad = 2000 - new_len
                seg = np.pad(seg_stretched, ((0,0), (pad//2, pad-pad//2)), mode='constant')
            # Adjust frequency targets
            freqs = freqs * stretch_factor
            # Adjust eventness (resample)
            if eventness is not None and np.any(eventness > 0):
                eventness = resample(eventness, 2000).astype(np.float32)
                eventness = np.clip(eventness, 0, 1)

        # Gaussian noise
        if np.random.random() < 0.5:
            noise_scale = 0.1 * np.std(seg, axis=1, keepdims=True)
            noise_scale[noise_scale < 1e-6] = 0.01
            seg = seg + np.random.randn(*seg.shape).astype(np.float32) * noise_scale

        # Random amplitude scaling
        if np.random.random() < 0.5:
            scales = np.random.uniform(0.8, 1.2, (18, 1)).astype(np.float32)
            seg = seg * scales

        # Channel dropout
        if np.random.random() < 0.2:
            n_drop = np.random.randint(1, 3)
            drop_idx = np.random.choice(18, n_drop, replace=False)
            seg[drop_idx] = 0.0

        return seg, freqs, eventness, stretch_factor
