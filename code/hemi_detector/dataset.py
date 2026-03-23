"""
HemiNet Dataset (Experiment 1.1)

For each ground-truth case:
  - LPD with laterality  → 1 example (affected hemisphere)
  - LPD without laterality → 1 example (hemisphere with stronger PD signal)
  - GPD                  → 2 examples (left hemisphere + right hemisphere)
  - GRDA / LRDA          → 1 example (treated as LPD for affected side)

Targets (all at 100 Hz = 1000 bins for 10s):
  - event_target  : Gaussian bumps σ=2 bins at discharge times (float32, [0,1])
  - active_target : 1.0 in active regime (float32 binary)
  - freq_target   : log(gold_standard_freq) or log(IPI-median freq)

Augmentation (training only):
  - Amplitude scale × Uniform(0.7, 1.3)
  - Gaussian noise σ = 0.05 × channel_std
  - Channel dropout: zero 1 random channel with p=0.15
  - Discharge time jitter: N(0, σ=1 sample = 5ms) on event target construction
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import scipy.io as sio
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# ── Constants ─────────────────────────────────────────────────────────────────
FS_EEG = 200          # EEG sample rate Hz
FS_TARGET = 100       # Target resolution Hz (1000 bins for 10s)
N_SAMPLES = 2000      # EEG samples (10s @ 200Hz)
N_BINS = 1000         # Target bins (10s @ 100Hz)
GAUSSIAN_SIGMA = 2.0  # bins (20ms at 100Hz)
ACTIVE_PAD_S = 0.25   # seconds padding around active regions
ACTIVE_MIN_GAP_S = 1.8  # seconds — events closer than this → same active region
JITTER_SIGMA = 1.0    # sample jitter std (1 sample = 5ms at 200Hz → 0.5 bin at 100Hz)

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]    # into 18-channel bipolar
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_segment(mat_path: Path) -> np.ndarray:
    """Load .mat EEG, handle both monopolar (20ch) and bipolar (18ch).

    Returns (18, 2000) float32 bipolar array.
    """
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    n_ch = data.shape[0]

    if n_ch == 20:
        # Convert monopolar → bipolar using banana montage
        # Matches fcn_getBanana order in optimization_harness_v2
        # Channels: [Fp1,F7,T3,T5,O1,Fp2,F8,T4,T6,O2,Fp1,F3,C3,P3,O1,Fp2,F4,C4,P4,O2]
        # Bipolar pairs (0-indexed monopolar):
        #   0-1, 1-2, 2-3, 3-4,  → Fp1-F7, F7-T3, T3-T5, T5-O1
        #   5-6, 6-7, 7-8, 8-9,  → Fp2-F8, F8-T4, T4-T6, T6-O2
        #   10-11, 11-12, 12-13, 13-14, → Fp1-F3, F3-C3, C3-P3, P3-O1
        #   15-16, 16-17, 17-18, 18-19, → Fp2-F4, F4-C4, C4-P4, P4-O2
        #   ? → 2 more channels for 18ch total (Fz-Cz, Cz-Pz use midline ch)
        # Use the fcn_getBanana function from pd_pointiness_acf if available
        try:
            import sys
            code_dir = Path(__file__).resolve().parent.parent
            if str(code_dir) not in sys.path:
                sys.path.insert(0, str(code_dir))
            from pd_pointiness_acf import fcn_getBanana
            bipolar = np.array(fcn_getBanana(data)).astype(np.float64)
        except Exception:
            # Fallback: simple consecutive differences for first 18
            bipolar = np.zeros((18, data.shape[1]))
            pairs = [
                (0,1),(1,2),(2,3),(3,4),
                (5,6),(6,7),(7,8),(8,9),
                (10,11),(11,12),(12,13),(13,14),
                (15,16),(16,17),(17,18),(18,19),
                (0,5),(4,9)  # rough approximation for last 2
            ]
            for i, (a, b) in enumerate(pairs[:18]):
                bipolar[i] = data[a] - data[b]
        data = bipolar
    elif n_ch == 18:
        pass  # Already bipolar
    else:
        # Unexpected — try to use as-is, pad/trim to 18 channels
        if n_ch < 18:
            pad = np.zeros((18 - n_ch, data.shape[1]))
            data = np.vstack([data, pad])
        else:
            data = data[:18]

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Ensure exactly N_SAMPLES time points
    if data.shape[1] < N_SAMPLES:
        pad = np.zeros((18, N_SAMPLES - data.shape[1]), dtype=np.float32)
        data = np.hstack([data, pad])
    elif data.shape[1] > N_SAMPLES:
        data = data[:, :N_SAMPLES]

    return data  # (18, 2000) float32


def _hemisphere_evidence(seg18: np.ndarray, hemi_indices: List[int]) -> float:
    """Compute total PD evidence for a hemisphere.

    Uses RMS energy in the low-frequency band (0.3–3.5 Hz) relative to
    broadband energy — a simple proxy for PD signal strength.
    """
    try:
        b_lo, a_lo = butter(4, [0.3, 3.5], btype='bandpass', fs=FS_EEG)
        strengths = []
        for ch in hemi_indices:
            x = seg18[ch].astype(np.float64)
            if np.all(x == 0) or np.std(x) < 1e-10:
                continue
            lp = filtfilt(b_lo, a_lo, x)
            ratio = np.sqrt(np.mean(lp**2)) / (np.std(x) + 1e-8)
            strengths.append(ratio)
        return float(np.mean(strengths)) if strengths else 0.0
    except Exception:
        return 0.0


def _make_event_target(
    discharge_times_s: List[float],
    n_bins: int = N_BINS,
    fs_target: float = FS_TARGET,
    sigma: float = GAUSSIAN_SIGMA,
    jitter_sigma_samples: float = JITTER_SIGMA,
    augment: bool = False,
) -> np.ndarray:
    """Create Gaussian-bump event target at 100 Hz.

    Parameters
    ----------
    discharge_times_s : discharge times in seconds
    n_bins : number of output bins (default 1000)
    fs_target : target resolution Hz (default 100)
    sigma : Gaussian std in bins (default 2)
    jitter_sigma_samples : std of per-event jitter in EEG samples (5ms = 1 sample)
    augment : if True, apply per-event jitter

    Returns
    -------
    target : (n_bins,) float32 in [0, 1]
    """
    target = np.zeros(n_bins, dtype=np.float64)
    bins = np.arange(n_bins)
    for t_s in discharge_times_s:
        if augment:
            # Jitter in seconds (jitter_sigma_samples / FS_EEG)
            t_s = t_s + np.random.randn() * (jitter_sigma_samples / FS_EEG)
        center_bin = t_s * fs_target
        if center_bin < 0 or center_bin >= n_bins:
            continue
        gauss = np.exp(-0.5 * ((bins - center_bin) / sigma) ** 2)
        target = np.maximum(target, gauss)
    return target.astype(np.float32)


def _make_active_target(
    discharge_times_s: List[float],
    n_bins: int = N_BINS,
    fs_target: float = FS_TARGET,
    min_gap_s: float = ACTIVE_MIN_GAP_S,
    pad_s: float = ACTIVE_PAD_S,
    explicit_interval: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Create binary active-regime target at 100 Hz.

    Active regions: runs where adjacent events are < min_gap_s apart,
    padded by ±pad_s on each side. If explicit_interval is provided,
    use that directly.
    """
    target = np.zeros(n_bins, dtype=np.float32)
    if explicit_interval is not None:
        t_start, t_end = explicit_interval
        b_start = max(0, int((t_start - pad_s) * fs_target))
        b_end = min(n_bins, int((t_end + pad_s) * fs_target) + 1)
        target[b_start:b_end] = 1.0
        return target

    if len(discharge_times_s) < 2:
        if len(discharge_times_s) == 1:
            t = discharge_times_s[0]
            b_start = max(0, int((t - pad_s) * fs_target))
            b_end = min(n_bins, int((t + pad_s) * fs_target) + 1)
            target[b_start:b_end] = 1.0
        return target

    times = sorted(discharge_times_s)
    # Group into runs
    runs = []
    run_start = times[0]
    run_end = times[0]
    for t in times[1:]:
        if t - run_end < min_gap_s:
            run_end = t
        else:
            runs.append((run_start, run_end))
            run_start = t
            run_end = t
    runs.append((run_start, run_end))

    for (r_start, r_end) in runs:
        b_start = max(0, int((r_start - pad_s) * fs_target))
        b_end = min(n_bins, int((r_end + pad_s) * fs_target) + 1)
        target[b_start:b_end] = 1.0

    return target


def _zscore_segment(seg8: np.ndarray) -> np.ndarray:
    """Z-score each channel independently. Handles NaN/zero-std channels."""
    out = seg8.astype(np.float32)
    for ch in range(out.shape[0]):
        mu = np.mean(out[ch])
        std = np.std(out[ch])
        if std > 1e-8:
            out[ch] = (out[ch] - mu) / std
        else:
            out[ch] = 0.0
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

class HemiDataset(Dataset):
    """Dataset of single-hemisphere EEG examples for discharge detection.

    Each item:
        eeg      : (8, 2000) float32 — z-scored hemisphere channels
        event_t  : (1000,) float32   — Gaussian bump discharge targets
        active_t : (1000,) float32   — binary active-regime target
        freq_t   : (1,) float32      — log(gold_freq) or log(IPI freq)
        pid      : str               — patient id (for CV splitting)
        hemi     : str               — 'left' or 'right'
    """

    def __init__(
        self,
        hpp_data: Dict,
        eeg_dir: Path,
        patients_df=None,  # optional DataFrame with laterality/subtype
        augment: bool = False,
        amp_scale_range: Tuple[float, float] = (0.7, 1.3),
        noise_sigma: float = 0.05,
        ch_dropout_p: float = 0.15,
    ):
        self.augment = augment
        self.amp_scale_range = amp_scale_range
        self.noise_sigma = noise_sigma
        self.ch_dropout_p = ch_dropout_p
        self.eeg_dir = Path(eeg_dir)

        # Build patient lookup from patients_df
        self.pat_info: Dict[str, Dict] = {}
        if patients_df is not None:
            for _, row in patients_df.iterrows():
                pid = str(row['patient_id'])
                self.pat_info[pid] = {
                    'subtype': str(row.get('subtype', 'lpd')).lower(),
                    'laterality': str(row.get('laterality', '')).lower()
                    if isinstance(row.get('laterality'), str) else '',
                }

        # Build examples list
        self.examples: List[Dict] = []
        self._build_examples(hpp_data)

    def _find_eeg_file(self, pid: str) -> Optional[Path]:
        """Find the first available EEG segment file for a patient."""
        # Try seg000 first (primary segment)
        for suffix in ['_seg000.mat', '_seg001.mat', '_seg002.mat']:
            p = self.eeg_dir / f'{pid}{suffix}'
            if p.exists():
                return p
        # Glob fallback
        matches = sorted(self.eeg_dir.glob(f'{pid}_seg*.mat'))
        if matches:
            return matches[0]
        return None

    def _build_examples(self, hpp_data: Dict):
        """Build the list of (pid, hemisphere, targets) examples."""
        n_added = 0
        n_skipped = 0

        for pid, record in hpp_data.items():
            # Only use ground truth cases
            if record.get('review_status') != 'ground_truth':
                continue

            discharge_times = record.get('global_times', [])
            if len(discharge_times) < 2:
                continue

            gold_freq = record.get('gold_standard_freq')
            ipi_freq = record.get('frequency')  # IPI-derived

            # Determine log-frequency target
            if gold_freq and gold_freq > 0:
                log_freq = float(np.log(gold_freq))
            elif ipi_freq and ipi_freq > 0:
                log_freq = float(np.log(ipi_freq))
            else:
                if len(discharge_times) >= 2:
                    ipis = np.diff(sorted(discharge_times))
                    ipi_med = np.median(ipis)
                    log_freq = float(np.log(1.0 / ipi_med)) if ipi_med > 0 else float(np.log(1.0))
                else:
                    log_freq = float(np.log(1.0))

            log_freq = float(np.clip(log_freq, np.log(0.2), np.log(5.0)))

            subtype = str(record.get('subtype', 'lpd')).lower()
            laterality = str(record.get('laterality', '') or '').lower()

            # Override with patients_df laterality if available and more reliable
            if pid in self.pat_info:
                pat_lat = self.pat_info[pid]['laterality']
                if pat_lat in ('left', 'right'):
                    laterality = pat_lat
                pat_sub = self.pat_info[pid]['subtype']
                if pat_sub:
                    subtype = pat_sub

            # Find EEG file
            eeg_path = self._find_eeg_file(pid)
            if eeg_path is None:
                n_skipped += 1
                continue

            # Active interval from record
            active_interval = record.get('active_interval')
            if active_interval and len(active_interval) == 2:
                active_explicit = tuple(active_interval)
            else:
                active_explicit = None

            # ── Determine hemispheres for this case ───────────────────
            if subtype == 'gpd':
                # GPD: both hemispheres (2 examples)
                for hemi in ('left', 'right'):
                    self.examples.append({
                        'pid': pid,
                        'hemi': hemi,
                        'eeg_path': eeg_path,
                        'discharge_times': discharge_times,
                        'log_freq': log_freq,
                        'active_explicit': active_explicit,
                        'laterality_mode': 'known',
                    })
                    n_added += 1
            elif subtype in ('lpd', 'lrda', 'grda'):
                if laterality in ('left', 'right'):
                    # Known laterality
                    self.examples.append({
                        'pid': pid,
                        'hemi': laterality,
                        'eeg_path': eeg_path,
                        'discharge_times': discharge_times,
                        'log_freq': log_freq,
                        'active_explicit': active_explicit,
                        'laterality_mode': 'known',
                    })
                    n_added += 1
                else:
                    # Unknown laterality — will resolve at load time
                    # using hemisphere signal strength comparison
                    self.examples.append({
                        'pid': pid,
                        'hemi': 'auto',  # resolved in __getitem__
                        'eeg_path': eeg_path,
                        'discharge_times': discharge_times,
                        'log_freq': log_freq,
                        'active_explicit': active_explicit,
                        'laterality_mode': 'auto',
                    })
                    n_added += 1
            else:
                # Unknown/other subtype — skip or treat as LPD auto
                if laterality in ('left', 'right'):
                    self.examples.append({
                        'pid': pid,
                        'hemi': laterality,
                        'eeg_path': eeg_path,
                        'discharge_times': discharge_times,
                        'log_freq': log_freq,
                        'active_explicit': active_explicit,
                        'laterality_mode': 'known',
                    })
                    n_added += 1
                # else skip

        print(f"HemiDataset: {n_added} examples built, {n_skipped} skipped (no EEG)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]

        # ── Load EEG ─────────────────────────────────────────────────
        seg18 = _load_segment(ex['eeg_path'])  # (18, 2000) float32

        # ── Determine hemisphere ──────────────────────────────────────
        hemi = ex['hemi']
        if hemi == 'auto':
            # Choose hemisphere with stronger PD signal
            ev_left = _hemisphere_evidence(seg18, LEFT_INDICES)
            ev_right = _hemisphere_evidence(seg18, RIGHT_INDICES)
            hemi = 'left' if ev_left >= ev_right else 'right'

        hemi_idx = LEFT_INDICES if hemi == 'left' else RIGHT_INDICES
        seg8 = seg18[hemi_idx, :]  # (8, 2000) float32

        # ── Z-score ──────────────────────────────────────────────────
        seg8 = _zscore_segment(seg8)  # (8, 2000)

        # ── Augmentation ─────────────────────────────────────────────
        if self.augment:
            # Amplitude scale
            scale = np.random.uniform(self.amp_scale_range[0], self.amp_scale_range[1])
            seg8 = seg8 * scale

            # Gaussian noise proportional to channel std
            for ch in range(8):
                ch_std = np.std(seg8[ch])
                noise = np.random.randn(*seg8[ch].shape).astype(np.float32)
                seg8[ch] = seg8[ch] + noise * self.noise_sigma * ch_std

            # Channel dropout
            if np.random.rand() < self.ch_dropout_p:
                drop_ch = np.random.randint(0, 8)
                seg8[drop_ch] = 0.0

        # ── Targets ──────────────────────────────────────────────────
        event_t = _make_event_target(
            ex['discharge_times'],
            augment=self.augment,
        )
        active_t = _make_active_target(
            ex['discharge_times'],
            explicit_interval=ex['active_explicit'],
        )
        freq_t = np.array([ex['log_freq']], dtype=np.float32)

        return {
            'eeg': torch.from_numpy(seg8),             # (8, 2000)
            'event_t': torch.from_numpy(event_t),      # (1000,)
            'active_t': torch.from_numpy(active_t),    # (1000,)
            'freq_t': torch.from_numpy(freq_t),        # (1,)
            'pid': ex['pid'],
            'hemi': hemi,
        }


def get_patient_ids(dataset: HemiDataset) -> List[str]:
    """Return ordered list of patient IDs (one per example, for CV splitting)."""
    return [ex['pid'] for ex in dataset.examples]


def get_patient_subtypes(dataset: HemiDataset, hpp_data: Dict) -> List[str]:
    """Return subtype label for each example (for stratified CV)."""
    subtypes = []
    for ex in dataset.examples:
        pid = ex['pid']
        record = hpp_data.get(pid, {})
        subtypes.append(str(record.get('subtype', 'lpd')).lower())
    return subtypes


if __name__ == '__main__':
    # Quick smoke test
    import json
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent.parent
    hpp_path = project_dir / 'data' / 'labels' / 'discharge_times.json'
    eeg_dir = project_dir / 'data' / 'eeg'

    with open(hpp_path) as f:
        hpp_data = json.load(f)

    ds = HemiDataset(hpp_data, eeg_dir, augment=True)
    print(f"Dataset size: {len(ds)}")

    item = ds[0]
    print(f"eeg shape: {item['eeg'].shape}")
    print(f"event_t shape: {item['event_t'].shape}, max={item['event_t'].max():.3f}")
    print(f"active_t shape: {item['active_t'].shape}, mean={item['active_t'].mean():.3f}")
    print(f"freq_t: {item['freq_t']}, exp={float(np.exp(item['freq_t'].numpy()[0])):.3f} Hz")
    print(f"pid: {item['pid']}, hemi: {item['hemi']}")
