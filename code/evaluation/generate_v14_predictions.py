#!/usr/bin/env python3
"""V14: V12 frequency + V14_unanimous laterality.

V14_unanimous rule (laterality only):
    Default to V12 laterality (V12's pass-2 envelope amplitude rule).
    Override with the opposite call only when all 4 V13 rhythmicity
    features (Q-factor, PLV, peak-CV-inv, peak prominence) unanimously
    disagree with V12 -- the strongest possible evidence that amplitude
    is misleading us.

V12 frequency is unchanged. So V14 = V12 freq + amplitude/rhythmicity
hybrid laterality.

Reads:
    data/labels/independent_expert_v1/v12_predictions.json
    data/labels/independent_expert_v1/lrda_laterality_v13_features.csv

Writes:
    data/labels/independent_expert_v1/v14_predictions.json

    conda run -n morgoth python code/evaluation/generate_v14_predictions.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
V12_PRED = LABELS_DIR / 'independent_expert_v1' / 'v12_predictions.json'
V13_FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v13_features.csv'
OUT = LABELS_DIR / 'independent_expert_v1' / 'v14_predictions.json'

V13_KEYS = ['q_log_ratio', 'plv_log_ratio', 'peak_cv_inv_log_ratio', 'prom_log_ratio']


def main():
    with open(V12_PRED) as f:
        v12 = json.load(f)
    v13 = {r['mat_file']: r for r in csv.DictReader(open(V13_FEAT_CSV))}
    print(f'V12 predictions: {len(v12)} segments. V13 features: {len(v13)} segments.')

    out = {}
    n_flipped = 0
    flips = []
    for sid, e in v12.items():
        mf = e['mat_file']
        v12_lat = e['v12_laterality']  # 'left' or 'right'
        v14_lat = v12_lat
        unan = False

        if mf in v13:
            v13r = v13[mf]
            v12_sign = 1 if v12_lat == 'left' else -1
            v13_signs = [(1 if float(v13r[k]) > 0 else -1) for k in V13_KEYS]
            if all(s != v12_sign for s in v13_signs):
                v14_lat = 'right' if v12_lat == 'left' else 'left'
                unan = True
                n_flipped += 1
                flips.append((mf, v12_lat, v14_lat, [float(v13r[k]) for k in V13_KEYS]))

        out[sid] = {
            'mat_file': mf,
            'patient_id': e.get('patient_id', ''),
            'subtype': 'lrda',
            'v14_freq': float(e['v12_freq']),  # V14 reuses V12's frequency
            'v14_laterality': v14_lat,
            'v12_laterality': v12_lat,
            'v14_unanimous_flip': unan,
            'v12_hyperparams': e.get('hyperparams', {}),
            'v14_rule': 'flip V12 laterality only when all 4 V13 rhythmicity features unanimously disagree',
        }

    print(f'V14_unanimous flipped {n_flipped} of {len(v12)} segments.')
    for mf, v12_lat, v14_lat, scores in flips:
        print(f'  FLIP {mf}: V12={v12_lat} -> V14={v14_lat}  (V13 scores: '
              f'q={scores[0]:+.2f}, plv={scores[1]:+.2f}, '
              f'peakCV={scores[2]:+.2f}, prom={scores[3]:+.2f})')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nWrote {OUT}  ({len(out)} segments)')


if __name__ == '__main__':
    main()
