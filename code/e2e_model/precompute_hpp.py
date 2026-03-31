#!/usr/bin/env python
"""
Precompute HPP (pointiness + TKEO) features for all EEG segments.
Saves to data/e2e_cache/hpp_cache.npz for fast loading during training.
"""

import sys
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from pd_pointiness_acf import fcn_getBanana
from discharge_detector import compute_channel_evidence
from e2e_model.dataset import build_sample_list

CACHE_DIR = PROJECT_DIR / 'data' / 'e2e_cache'
CACHE_DIR.mkdir(exist_ok=True)
HPP_CACHE_PATH = CACHE_DIR / 'hpp_cache.npz'


def precompute_all():
    """Precompute HPP features for all samples."""
    samples = build_sample_list()
    print(f"Precomputing HPP for {len(samples)} samples...")

    hpp_dict = {}
    pool_size = 16  # 2000 / 125

    for i, sample in enumerate(samples):
        key = sample['key']
        if i % 50 == 0:
            print(f"  [{i}/{len(samples)}] {key}")

        try:
            mat = sio.loadmat(sample['eeg_path'])
            eeg_mono = mat['data'].astype(np.float32)
            eeg_bipolar = fcn_getBanana(eeg_mono).astype(np.float32)

            # Ensure correct length
            if eeg_bipolar.shape[1] < 2000:
                pad = 2000 - eeg_bipolar.shape[1]
                eeg_bipolar = np.pad(eeg_bipolar, ((0, 0), (0, pad)))
            elif eeg_bipolar.shape[1] > 2000:
                eeg_bipolar = eeg_bipolar[:, :2000]

            hpp = np.zeros((18, 125), dtype=np.float32)
            for c in range(18):
                evidence = compute_channel_evidence(eeg_bipolar[c])
                n = len(evidence)
                if n % pool_size != 0:
                    pad_n = pool_size - (n % pool_size)
                    evidence = np.pad(evidence, (0, pad_n), mode='constant')
                hpp[c] = evidence[:pool_size * 125].reshape(125, pool_size).mean(axis=1)

            hpp_dict[key] = hpp

        except Exception as e:
            print(f"  ERROR on {key}: {e}")
            hpp_dict[key] = np.zeros((18, 125), dtype=np.float32)

    # Save as npz
    np.savez_compressed(HPP_CACHE_PATH, **hpp_dict)
    print(f"\nSaved HPP cache with {len(hpp_dict)} entries to {HPP_CACHE_PATH}")


if __name__ == '__main__':
    precompute_all()
