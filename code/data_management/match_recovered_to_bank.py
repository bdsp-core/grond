#!/usr/bin/env python3
"""Match all recovered (filtered) patients to the unfiltered bank.

Algorithm:
  1. Build query envelope (filtered, channel-summed |bipolar|) for each
     recovered patient → QZ matrix (N_queries, 2000).
  2. For each of the ~68K bank entries:
       - Compute sliding-window NCC (4001 windows × 2000 samples)
         vs ALL queries simultaneously via one matmul.
       - Track best (NCC, window_offset) per query.
  3. Save (best_bank_seg_id, window_offset, NCC) per query.

Then a separate extract step pulls the unfiltered 10-second slice for each
high-confidence match.

Output: results/c1_repro/bank_match_results.csv
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from numpy.lib.stride_tricks import sliding_window_view
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'c1_repro'
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENV_PATH = '/Volumes/Extreme SSD/.eeg_cache/eeg-test/bank_envelopes.npz'

RECOVERY_SOURCES = ('alex_recovery', 's3_iiic_freq3_recovery', 'pathC_morgoth1_s3')

b_coef, a_coef = butter(4, 20, fs=200, btype='low')


def build_query_envelope(mat_file):
    """Load a recovered .mat file → returns (2000,) float32 envelope or None."""
    p = EEG_DIR / mat_file
    if not p.exists():
        return None
    try:
        m = sio.loadmat(str(p))
    except Exception:
        return None
    data = m.get('data')
    if data is None or data.shape[0] not in (18, 19):
        return None
    if data.shape[0] == 18:
        bip = data.astype(np.float64)
    else:
        # 19 monopolar -> 18 bipolar
        MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                'Fp2','F4','C4','P4','F8','T4','T6','O2']
        PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
                 ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
                 ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
                 ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
                 ('Fz','Cz'),('Cz','Pz')]
        IDX = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])
        bip = data[IDX[:, 0]].astype(np.float64) - data[IDX[:, 1]].astype(np.float64)
    if bip.shape[1] != 2000:
        return None
    bip_lp = filtfilt(b_coef, a_coef, bip, axis=-1)
    env = np.sum(np.abs(bip_lp), axis=0).astype(np.float32)
    return env


def main():
    print('Loading recovery list...', flush=True)
    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    recov = seg_df[seg_df['eeg_source'].isin(RECOVERY_SOURCES)].copy()
    recov = recov.drop_duplicates(subset=['mat_file'])
    print(f'  {len(recov)} recovered rows', flush=True)

    # Build query envelopes
    print('Building query envelopes (filtered, channel-summed |bipolar|)...', flush=True)
    t0 = time.time()
    queries = []  # list of (pid, mat_file, source, env)
    for _, row in recov.iterrows():
        env = build_query_envelope(row['mat_file'])
        if env is None or len(env) != 2000:
            continue
        queries.append({'pid': row['patient_id'], 'mat_file': row['mat_file'],
                        'source': row['eeg_source'], 'env': env})
    print(f'  {len(queries)} valid queries in {time.time()-t0:.0f}s', flush=True)

    # Build QZ matrix: each query is z-scored to mean=0, std=1
    QZ = np.zeros((len(queries), 2000), dtype=np.float32)
    for i, q in enumerate(queries):
        e = q['env'].astype(np.float64)
        z = (e - e.mean()) / (e.std() + 1e-12)
        QZ[i] = z.astype(np.float32)
    print(f'  QZ shape: {QZ.shape}', flush=True)
    QZ_T = QZ.T.astype(np.float32)  # (2000, N_queries)

    # Load bank envelopes
    print(f'Loading bank envelopes from {ENV_PATH}...', flush=True)
    z = np.load(ENV_PATH)
    bank_keys = z['keys']  # bytes
    envs = z['envs']  # (N_bank, 6000) float32
    n_bank = len(bank_keys)
    print(f'  {n_bank} bank entries, envs shape {envs.shape}', flush=True)

    # Per-query state: best NCC, best bank index, best window offset
    n_q = len(queries)
    best_ncc = np.full(n_q, -np.inf, dtype=np.float32)
    best_bank_idx = np.zeros(n_q, dtype=np.int64)
    best_offset = np.zeros(n_q, dtype=np.int32)

    nq = 2000
    inv_nq = 1.0 / nq

    print(f'\nMatching {n_q} queries × {n_bank} bank entries × 4001 windows...', flush=True)
    t0 = time.time()
    for bi in range(n_bank):
        env = envs[bi].astype(np.float64)  # (6000,)
        # Sliding window normalization
        windows = sliding_window_view(env, nq)  # (4001, 2000)
        means = windows.mean(axis=1)  # (4001,)
        stds = windows.std(axis=1) + 1e-12  # (4001,)
        # z_w[i,j] = (windows[i,j] - means[i]) / stds[i]
        z_w = (windows - means[:, None]) / stds[:, None]  # (4001, 2000) float64
        # NCC matrix: z_w @ QZ_T = (4001, N_queries)
        # Then divide by nq -> NCC
        ncc = z_w.astype(np.float32) @ QZ_T * inv_nq  # (4001, N_queries)
        # Per-query best
        best_offs = ncc.argmax(axis=0)  # (N_queries,)
        best_vals = ncc[best_offs, np.arange(n_q)]  # (N_queries,)
        # Update global bests
        better = best_vals > best_ncc
        if better.any():
            best_ncc[better] = best_vals[better]
            best_bank_idx[better] = bi
            best_offset[better] = best_offs[better]

        if (bi + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (bi + 1) / elapsed
            eta = (n_bank - bi - 1) / rate
            print(f'  [{bi+1}/{n_bank}] {elapsed:.0f}s, rate={rate:.0f}/s, ETA={eta/60:.1f} min, '
                  f'best NCC: p50={np.percentile(best_ncc, 50):.3f}, p99={np.percentile(best_ncc, 99):.3f}, '
                  f'max={best_ncc.max():.3f}', flush=True)

    print(f'\nMatching done in {time.time()-t0:.0f}s', flush=True)

    # Build result DataFrame
    rows = []
    for i, q in enumerate(queries):
        bk = bank_keys[best_bank_idx[i]]
        if isinstance(bk, bytes): bk = bk.decode()
        rows.append({
            'pid': q['pid'],
            'mat_file': q['mat_file'],
            'source': q['source'],
            'bank_seg_id': str(bk),
            'bank_window_start_samples': int(best_offset[i]),
            'bank_window_start_s': float(best_offset[i] / 200.0),
            'best_ncc': float(best_ncc[i]),
        })
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / 'bank_match_results.csv'
    df.to_csv(out_csv, index=False)

    print(f'\n=== SUMMARY ===', flush=True)
    print(f'Saved: {out_csv}', flush=True)
    print(f'\nNCC distribution (per query):', flush=True)
    for p in [50, 75, 90, 95, 99]:
        print(f'  p{p}: {np.percentile(best_ncc, p):.4f}', flush=True)
    print(f'  max: {best_ncc.max():.4f}, min: {best_ncc.min():.4f}', flush=True)

    print(f'\nBy source:', flush=True)
    for src in df['source'].unique():
        s = df[df['source'] == src]
        print(f'  {src}: n={len(s)}, NCC mean={s["best_ncc"].mean():.4f}, '
              f'p50={s["best_ncc"].median():.4f}, '
              f'>0.9={sum(s["best_ncc"] > 0.9)}, '
              f'>0.95={sum(s["best_ncc"] > 0.95)}', flush=True)


if __name__ == '__main__':
    main()
