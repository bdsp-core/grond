#!/usr/bin/env python3
"""V12: retuned W05/NB-Hilbert LRDA frequency estimator.

Re-uses the W05 architecture but with hyperparameters retuned against
the canonical majority-accept consensus dataset (155 segments). Best
combo from the 192-point sweep:

    p1_hi (pass-1 bandpass upper)  : 4.5 Hz  (was 3.5)
    p2_bw (pass-2 narrowband half) : 0.5 Hz  (was 0.4)
    top_k (channels averaged)      : 3      (unchanged)
    freq_cap (search upper limit)  : 4.5 Hz  (was 4.0)

Sweep results (mean ICC vs each rater on majority-accept set):
                  V1 baseline   V12 retuned
        MW            0.704         0.764
        SZ            0.890         0.860   (still above SZ EE ceiling 0.866)
        TZ            0.750         0.806
        mean          0.781         0.810

Writes data/labels/independent_expert_v1/v12_predictions.json with the
same shape as v9_predictions.json so it slots into analyze_*.py.

    conda run -n morgoth python code/evaluation/generate_v12_predictions.py
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
OUT_PATH = LABELS_DIR / 'independent_expert_v1' / 'v12_predictions.json'

# Frozen V12 hyperparameters
P1_HI = 4.5
P2_BW = 0.5
TOP_K = 3
FREQ_CAP = 4.5


def _hilbert_freq(sig, freq_min, freq_max):
    if np.std(sig) < 1e-10:
        return float('nan')
    analytic = hilbert(sig)
    inst = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
    mask = (inst > freq_min) & (inst < freq_max)
    valid = inst[mask]
    if len(valid) < 20:
        return float('nan')
    return float(np.median(valid))


def w05_v12(seg_bi):
    """Returns (freq_hz, laterality)."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos_pre, seg_bi, axis=1)
    sos1 = butter(4, [0.5 / (FS / 2), P1_HI / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_f, axis=1)
    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top = dom_chs[np.argsort(powers)[::-1][:TOP_K]]
    sig_p1 = np.mean(seg_n[top], axis=0)
    f1 = _hilbert_freq(sig_p1, freq_min=0.3, freq_max=FREQ_CAP)
    if not np.isfinite(f1):
        f1 = 1.5
    lo = max(f1 - P2_BW, 0.1)
    hi = min(f1 + P2_BW, FS / 2 - 0.1)
    if lo < hi:
        sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
    else:
        seg_nb = seg_n
    ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
    dom_chs2 = LEFT_CHS if ls >= rs else RIGHT_CHS
    side = 'left' if ls >= rs else 'right'
    powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs2])
    top2 = dom_chs2[np.argsort(powers2)[::-1][:TOP_K]]
    sig_p2 = np.mean(seg_nb[top2], axis=0)
    f2 = _hilbert_freq(sig_p2, freq_min=0.3, freq_max=FREQ_CAP)
    final = f2 if np.isfinite(f2) else f1
    return float(np.clip(final, 0.25, FREQ_CAP)), side


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_DIR / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    print(f'Generating V12 predictions on {len(rows)} LRDA segments...')

    out = {}
    n_done = 0
    for r in rows:
        mf = r['mat_file']
        seg = load_segment(mf)
        if seg is None:
            print(f'  WARNING: {mf} unloadable; skipping')
            continue
        f, lat = w05_v12(seg)
        sid = mf[:-4] if mf.endswith('.mat') else mf
        out[sid] = {
            'mat_file': mf,
            'patient_id': r['patient_id'],
            'subtype': 'lrda',
            'v12_freq': f,
            'v12_laterality': lat,
            'hyperparams': {'p1_hi': P1_HI, 'p2_bw': P2_BW, 'top_k': TOP_K, 'freq_cap': FREQ_CAP},
        }
        n_done += 1
        if n_done % 50 == 0:
            print(f'  {n_done}/{len(rows)}')

    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT_PATH}  ({len(out)} segments)')


if __name__ == '__main__':
    main()
