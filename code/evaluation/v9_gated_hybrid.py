#!/usr/bin/env python3
"""V9 = V1 (W05/NB-Hilbert) + V8 (active-window spectral peak) gated by the
hard-case classifier from train_lrda_hard_case_classifier.py.

Per LRDA segment:
    if classifier predicts hard:  use V8
    else:                         use V1

Saves predictions to data/labels/independent_expert_v1/v9_predictions.json
which the IRR analysis script consumes via --algo v9.

    conda run -n morgoth python code/evaluation/v9_gated_hybrid.py
    conda run -n morgoth python code/evaluation/v9_gated_hybrid.py --threshold 0.5
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, find_peaks

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))

from generate_rda_freq_labeler import (  # type: ignore
    load_segment, FS, LEFT_CHS, RIGHT_CHS, w05_estimate_freq,
)
from lrda_features import featurize  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
MODEL_PATH = LABELS_DIR / 'independent_expert_v1' / 'hard_case_classifier.pkl'
OUT_PATH = LABELS_DIR / 'independent_expert_v1' / 'v9_predictions.json'


def v8_active_window_freq(seg_bi, side, win_sec=3.0, hop_sec=0.5, top_k=3):
    """Active-window spectral-peak frequency search restricted to the given side."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)
    hem_chs = LEFT_CHS if side == 'left' else RIGHT_CHS
    win_samp = int(win_sec * FS)
    hop_samp = int(hop_sec * FS)
    n_samp = seg_pre.shape[1]
    best_score = -1.0
    best_freq = None
    for ws in range(0, n_samp - win_samp + 1, hop_samp):
        we = ws + win_samp
        powers = np.array([np.var(seg_pre[ch, ws:we]) for ch in hem_chs])
        top_chs = hem_chs[np.argsort(powers)[::-1][:top_k]]
        sig = np.mean(seg_pre[top_chs, ws:we], axis=0)
        nperseg = min(len(sig), 256)
        f, pxx = welch(sig, fs=FS, nperseg=nperseg)
        mask = (f >= 0.5) & (f <= 4.0)
        f_m = f[mask]
        pxx_m = pxx[mask]
        if not len(pxx_m) or np.max(pxx_m) <= 0:
            continue
        peaks, props = find_peaks(pxx_m, prominence=np.max(pxx_m) * 0.05)
        if not len(peaks):
            continue
        bp = int(np.argmax(props['prominences']))
        score = float(props['prominences'][bp]) / np.max(pxx_m)
        if score > best_score:
            best_score = score
            best_freq = float(f_m[peaks[bp]])
    return best_freq if best_freq is not None else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--threshold', type=float, default=None,
                    help='Classifier probability threshold for routing to V8. '
                         'Default: pick threshold from saved model bundle.')
    args = ap.parse_args()

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
    clf = bundle['model']
    feature_names = bundle['feature_names']
    threshold = args.threshold if args.threshold is not None else bundle['threshold']
    if threshold is None:
        threshold = 0.5
    print(f'Loaded classifier (threshold={threshold:.2f})')

    with open(TASKS_DIR / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    print(f'Running V9 on {len(rows)} LRDA manifest segments...')

    out = {}
    n_v8 = 0
    n_v1 = 0
    for i, r in enumerate(rows):
        mf = r['mat_file']
        seg = load_segment(mf)
        if seg is None:
            continue
        v1_freq, v1_side = w05_estimate_freq(seg)
        try:
            iiic_p = float(r.get('iiic_plurality_frac') or 0)
        except ValueError:
            iiic_p = 0.0
        try:
            iiic_n = float(r.get('iiic_n_votes') or 0)
        except ValueError:
            iiic_n = 0.0
        feats = featurize(seg, iiic_plurality_frac=iiic_p, iiic_n_votes=iiic_n)
        feat_vec = np.array([[feats[fn] for fn in feature_names]], dtype=np.float64)
        proba_hard = float(clf.predict_proba(feat_vec)[0, 1])
        if proba_hard >= threshold:
            v8_f = v8_active_window_freq(seg, v1_side)
            if v8_f is not None:
                used_freq = float(np.clip(round(v8_f * 4) / 4.0, 0.5, 3.5))
                used = 'v8'
                n_v8 += 1
            else:
                used_freq = float(np.clip(round(v1_freq * 4) / 4.0, 0.5, 3.5))
                used = 'v1_fallback'
                n_v1 += 1
        else:
            used_freq = float(np.clip(round(v1_freq * 4) / 4.0, 0.5, 3.5))
            used = 'v1'
            n_v1 += 1
        out[mf.replace('.mat', '')] = {
            'mat_file': mf,
            'patient_id': r['patient_id'],
            'subtype': 'lrda',
            'v9_freq': used_freq,
            'v9_laterality': v1_side,   # V9 inherits V1's lat
            'v1_freq': float(v1_freq),
            'v1_laterality': v1_side,
            'classifier_proba_hard': proba_hard,
            'used': used,
        }
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(rows)}  (n_v8={n_v8}, n_v1={n_v1})')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote {OUT_PATH}  (n_v8={n_v8}, n_v1={n_v1})')


if __name__ == '__main__':
    main()
