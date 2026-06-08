#!/usr/bin/env python3
"""One-time pass: precompute filtered envelopes for all (20, 6000) bank entries.

Result: /Volumes/Extreme SSD/.eeg_cache/eeg-test/bank_envelopes.npz
  containing:
    keys:    (N,) array of seg_id strings
    envs:    (N, 6000) float32 — 20Hz low-passed channel-summed |bipolar| envelope
    bip0:    (N, 6000) float32 — first bipolar channel (filtered) for content sanity-checking
"""
import os
import sys
import time
import h5py
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt

H5_PATH = '/Volumes/Extreme SSD/.eeg_cache/eeg-test/eeg_bank_spec.h5'
OUT_PATH = '/Volumes/Extreme SSD/.eeg_cache/eeg-test/bank_envelopes.npz'

MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
        'Fp2','F4','C4','P4','F8','T4','T6','O2']
PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
         ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
         ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
         ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
         ('Fz','Cz'),('Cz','Pz')]
BIP_IDX = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])

b, a = butter(4, 20, fs=200, btype='low')


def main():
    print(f'Opening {H5_PATH}...', flush=True)
    with h5py.File(H5_PATH, 'r') as f:
        seg = f['segments']
        all_keys = list(seg.keys())
        print(f'  total entries: {len(all_keys)}', flush=True)

        # Pre-filter to (20, 6000) entries
        print('Pre-filtering to (20, 6000) entries...', flush=True)
        t0 = time.time()
        candidates = []
        for k in all_keys:
            obj = seg[k]
            if 'data30s' in obj and obj['data30s'].shape == (20, 6000):
                candidates.append(k)
        print(f'  {len(candidates)} candidates ({time.time()-t0:.0f}s)', flush=True)

        # Precompute envelopes
        N = len(candidates)
        envs = np.zeros((N, 6000), dtype=np.float32)
        keys_out = []
        t0 = time.time()
        skipped = 0
        for i, k in enumerate(candidates):
            try:
                d = np.array(seg[k]['data30s']).astype(np.float64)
                if not np.isfinite(d).all():
                    skipped += 1
                    continue
                mono = d[:19]
                bip = mono[BIP_IDX[:, 0]] - mono[BIP_IDX[:, 1]]
                bip_lp = filtfilt(b, a, bip, axis=-1)
                env = np.sum(np.abs(bip_lp), axis=0)
                envs[len(keys_out)] = env.astype(np.float32)
                keys_out.append(k)
            except Exception:
                skipped += 1
                continue
            if (i + 1) % 2000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (N - i - 1) / rate
                print(f'  [{i+1}/{N}] {elapsed:.0f}s, rate={rate:.1f}/s, ETA={eta/60:.1f} min, '
                      f'skipped={skipped}', flush=True)

        envs = envs[:len(keys_out)]
        keys_arr = np.array(keys_out, dtype='S16')
        print(f'\nProcessed {len(keys_out)} entries ({skipped} skipped) in {time.time()-t0:.0f}s', flush=True)

        print(f'Saving {OUT_PATH}...', flush=True)
        np.savez(OUT_PATH, keys=keys_arr, envs=envs)
        print(f'Done. Size: {Path(OUT_PATH).stat().st_size / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
