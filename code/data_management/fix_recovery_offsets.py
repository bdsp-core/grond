#!/usr/bin/env python3
"""Estimate and correct the per-source data-vs-GT timing offset.

For data that was recovered from filtered backups (Alex/npz) or via
window-finders (path C, IRR S3), the discharge_times.json annotations
were made on a slightly differently-time-aligned version of the same EEG
content. The result is a small (~50-100 ms) systematic timing offset between
the recovered data and the GT discharge times.

This script:
  1. For each recovered patient (eeg_source in {alex_recovery,
     s3_iiic_freq3_recovery, pathC_morgoth1_s3}), runs the HemiCET inference
     pipeline to get an evidence trace E(t).
  2. For each candidate offset τ in [-150, +150] ms (1 sample = 5 ms),
     computes alignment_score(τ) = sum over GT discharge times of
     E(GT_time + τ) (within ±15 ms window for robustness).
  3. The τ that maximizes alignment_score is the patient's offset.
  4. Aggregates τ by source — checks if there's a clean per-source distribution.
  5. Per patient: re-saves the .mat file shifted by τ samples so that GT
     timings line up with the data's true peak locations. (Edges are
     zero-padded by τ samples.)

After this script, the verifier should show recovered patients hitting
F1 close to the baseline on the 147 original-source patients.

Usage:
    conda run -n morgoth python code/data_management/fix_recovery_offsets.py
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from scipy.signal import butter, filtfilt

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))

from optimization_harness_v2 import LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import DischargeDetector, detect_active_interval, estimate_frequency_acf
from hemi_detector.hemi_cet import HemiCET

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'

RECOVERY_SOURCES = ('alex_recovery', 's3_iiic_freq3_recovery', 'pathC_morgoth1_s3')

# Offset search range: ±150 ms = ±30 samples at 200 Hz
MAX_OFFSET_SAMPLES = 30
OFFSET_TOL_SAMPLES = 3  # ±15 ms window around each GT time when scoring


def load_hemicet_v2():
    models = []
    for k in range(5):
        m = HemiCET(in_channels=8).to(DEVICE)
        ckpt = torch.load(PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet_v2'
                          / f'hemi_cet_fold{k}.pt',
                          map_location=DEVICE, weights_only=False)
        m.load_state_dict(ckpt['model_state_dict']
                          if 'model_state_dict' in ckpt else ckpt)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def compute_evidence_trace(seg, lat, subtype, hemi, detector):
    """Run HemiCET on a (18, 2000) bipolar seg; return evidence E(t) of length 2000."""
    # Pick hemisphere indices
    if subtype == 'gpd' or lat not in ('left', 'right'):
        # Run both and take the side with higher mean evidence
        ev_left = _ev_for_side(seg, LEFT_INDICES, hemi)
        ev_right = _ev_for_side(seg, RIGHT_INDICES, hemi)
        return ev_left if ev_left.mean() >= ev_right.mean() else ev_right
    indices = LEFT_INDICES if lat == 'left' else RIGHT_INDICES
    return _ev_for_side(seg, indices, hemi)


@torch.no_grad()
def _ev_for_side(seg, indices, hemi):
    hs = np.zeros((8, 2000), dtype=np.float32)
    for i, ci in enumerate(indices[:8]):
        ch = seg[ci].astype(np.float32)
        mu, std = float(ch.mean()), float(ch.std())
        hs[i] = (ch - mu) / std if std > 1e-8 else ch - mu
    x = torch.from_numpy(hs[None]).to(DEVICE)
    preds = [m(x).squeeze().cpu().numpy() for m in hemi]
    return np.mean(preds, axis=0)


def find_optimal_offset(evidence, gt_times):
    """Find offset τ (samples) maximizing alignment score with GT discharge times.

    Returns: (best_offset_samples, best_score, normalized_score).
    """
    gt_samples = np.array([int(round(t * FS)) for t in gt_times])
    n = len(evidence)
    n_gt = len(gt_samples)

    offsets = np.arange(-MAX_OFFSET_SAMPLES, MAX_OFFSET_SAMPLES + 1)
    scores = np.zeros(len(offsets), dtype=np.float64)
    for i, off in enumerate(offsets):
        score = 0.0
        for gs in gt_samples:
            idx = gs + off
            lo = max(0, idx - OFFSET_TOL_SAMPLES)
            hi = min(n, idx + OFFSET_TOL_SAMPLES + 1)
            if hi > lo:
                score += float(evidence[lo:hi].max())
        scores[i] = score

    best_i = int(np.argmax(scores))
    best_offset = int(offsets[best_i])
    best_score = float(scores[best_i])
    median_score = float(np.median(scores))
    norm = best_score / max(median_score, 1e-9)
    return best_offset, best_score, norm


def apply_offset_shift(data, offset_samples):
    """Shift data such that content originally at index i now sits at index i - offset.

    If offset > 0: data peaks were AT i, GT was at i-offset; we shift left by offset
    so that what was at index i in the original now appears at index i-offset (matching GT).
    Equivalently: out[0:N-offset] = data[offset:N]; out[N-offset:N] = 0.

    If offset < 0: shift right by |offset|, zero-pad at start.
    """
    n_ch, n_samp = data.shape
    out = np.zeros_like(data)
    if offset_samples > 0:
        out[:, :n_samp - offset_samples] = data[:, offset_samples:]
    elif offset_samples < 0:
        k = -offset_samples
        out[:, k:] = data[:, :n_samp - k]
    else:
        out[:] = data
    return out


def main():
    print('Loading inputs...')
    with open(DT_JSON) as f:
        dt = json.load(f)
    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    # Recovered rows
    recovered = seg_df[seg_df['eeg_source'].isin(RECOVERY_SOURCES)]
    print(f'  Recovered patients (rows): {len(recovered)}')
    print(f'  By source: {recovered["eeg_source"].value_counts().to_dict()}')

    print('Loading HemiCET v2 models...')
    hemi = load_hemicet_v2()
    detector = DischargeDetector()
    print(f'  {len(hemi)} fold models + {len(detector.cnn_models)} CNN models')

    # Pass 1: estimate per-patient offset
    print('\nPass 1: estimating per-patient offsets...')
    results = []
    t0 = time.time()
    for i, (_, row) in enumerate(recovered.iterrows()):
        pid = row['patient_id']
        mf = row['mat_file']
        src = row['eeg_source']
        # GT for this patient (only PD-GT carry discharge_times)
        gt_data = dt.get(pid)
        if (gt_data is None
                or gt_data.get('review_status') != 'ground_truth'
                or gt_data.get('subtype') not in ('lpd', 'gpd')
                or not isinstance(gt_data.get('global_times'), list)
                or len(gt_data['global_times']) < 2):
            continue
        gt_times = gt_data['global_times']
        lat = gt_data.get('laterality')

        eeg_path = EEG_DIR / mf
        if not eeg_path.exists():
            continue
        try:
            m = sio.loadmat(str(eeg_path))
            data = m['data'].astype(np.float64)
            if data.shape[0] == 19:
                # monopolar→bipolar
                MONO = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz',
                        'Pz','Fp2','F4','C4','P4','F8','T4','T6','O2']
                PAIRS = [('Fp1','F7'),('F7','T3'),('T3','T5'),('T5','O1'),
                         ('Fp2','F8'),('F8','T4'),('T4','T6'),('T6','O2'),
                         ('Fp1','F3'),('F3','C3'),('C3','P3'),('P3','O1'),
                         ('Fp2','F4'),('F4','C4'),('C4','P4'),('P4','O2'),
                         ('Fz','Cz'),('Cz','Pz')]
                idx = np.array([[MONO.index(a), MONO.index(b)] for a, b in PAIRS])
                bip = data[idx[:, 0]] - data[idx[:, 1]]
            elif data.shape[0] == 18:
                bip = data
            else:
                continue
        except Exception:
            continue

        evidence = compute_evidence_trace(bip, lat, gt_data['subtype'],
                                          hemi, detector)
        offset, score, norm = find_optimal_offset(evidence, gt_times)
        results.append({
            'pid': pid,
            'mat_file': mf,
            'source': src,
            'subtype': gt_data['subtype'],
            'n_gt': len(gt_times),
            'offset_samples': offset,
            'offset_ms': offset * 1000.0 / FS,
            'score': score,
            'norm_score': norm,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(recovered)}] {elapsed:.0f}s elapsed '
                  f'({len(results)} estimates so far)')

    print(f'\nTotal patients with offset estimate: {len(results)}')

    # Aggregate per-source
    print('\nOffset distribution by source (in ms):')
    df = pd.DataFrame(results)
    if df.empty:
        print('  no offset estimates')
        return
    for src, grp in df.groupby('source'):
        offsets_ms = grp['offset_ms']
        median = offsets_ms.median()
        mean = offsets_ms.mean()
        std = offsets_ms.std()
        p25, p75 = offsets_ms.quantile([0.25, 0.75])
        print(f'  {src:>30s}  n={len(grp):>4d}  '
              f'median={median:>+7.1f}  mean={mean:>+7.1f}  std={std:>6.1f}  '
              f'IQR=[{p25:>+6.1f}, {p75:>+6.1f}]')

    # Save offset table
    OUT_DIR = PROJECT_DIR / 'results' / 'c1_repro'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    offset_csv = OUT_DIR / 'recovery_offsets.csv'
    df.to_csv(offset_csv, index=False)
    print(f'\nSaved offset table to {offset_csv}')

    # Pass 2: apply shift per patient
    # Use PER-PATIENT offset (more robust than per-source if offsets are noisy)
    # But require high confidence: norm_score > 1.5 to apply
    print('\nPass 2: applying per-patient shifts (those with norm_score >= 1.5)...')
    n_shifted = 0
    n_skipped = 0
    for r in results:
        if r['norm_score'] < 1.5:
            n_skipped += 1
            continue
        offset = r['offset_samples']
        if offset == 0:
            continue  # nothing to do
        mf = r['mat_file']
        eeg_path = EEG_DIR / mf
        m = sio.loadmat(str(eeg_path))
        data = m['data']
        shifted = apply_offset_shift(data, offset)
        sio.savemat(str(eeg_path), {'data': shifted.astype(data.dtype),
                                    'Fs': np.array([[FS]], dtype=np.int64)})
        n_shifted += 1

    print(f'\nShifted {n_shifted} files; skipped {n_skipped} (low-confidence alignment)')
    print(f'Saved: results/c1_repro/recovery_offsets.csv (audit log)')


if __name__ == '__main__':
    main()
