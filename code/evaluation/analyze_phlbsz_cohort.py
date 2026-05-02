#!/usr/bin/env python3
"""EE vs EA frequency-IRR on the PH/LB/SZ Tautan cohort, per subtype.

This is the second (independent) validation cohort the manuscript should
surface alongside the canonical 200-segment-per-subtype MW/SZ/TZ cohort.
Raters PH, LB, SZ here are the same three experts who labeled the
Tautan et al. (2025) dataset, which was annotated WITHOUT the
narrowband-overlay tools developed for the present work; their labels
are therefore noisier by construction.

For each (subtype, rater pair) the script reports:
  - n (segments where both raters labeled the segment + algorithm has a freq)
  - Spearman rho, ICC(3,1), MAE (Hz)

For each (subtype, rater) it also reports the rater-vs-algorithm pair.

Algorithm freq sources:
  - LPD/GPD: pdchar_freq_hz from segments.csv (PD-Profiler IPI-derived,
    refreshed by code/evaluation/refresh_pdchar_freq.py)
  - LRDA/GRDA: algo_freq_hz from segment_labels.csv (V12 NB-Hilbert,
    refreshed by code/evaluation/refresh_algo_freq_rda.py)
  - Tautan baseline: tautan_freq_hz from segments.csv (refreshed by
    code/evaluation/refresh_tautan_freq.py)

    conda run -n morgoth python code/evaluation/analyze_phlbsz_cohort.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'

RATERS = ('PH', 'LB', 'SZ', 'MW')


def icc_3_1(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); k = 2
    if n < 3:
        return float('nan')
    M = np.column_stack([x, y]); g = M.mean()
    bms = k * np.sum((M.mean(1) - g) ** 2) / (n - 1)
    ems = np.sum((M - M.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    if (bms + (k - 1) * ems) == 0:
        return float('nan')
    return float((bms - ems) / (bms + (k - 1) * ems))


def main():
    labels = pd.read_csv(LABELS_DIR / 'labels.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    seg = pd.read_csv(LABELS_DIR / 'segments.csv',
                      usecols=['mat_file', 'pdchar_freq_hz', 'tautan_freq_hz'])
    sl = sl.merge(seg, on='mat_file', how='left')

    freq = labels[labels.label_type == 'frequency_hz'].copy()
    freq['value'] = pd.to_numeric(freq['value'], errors='coerce')
    freq = freq[freq.value.notna() & (freq.value > 0)]
    sl_idx = sl.set_index('mat_file')

    print(f'{"task":>5} | {"pair":<10} | {"n":>4} | {"rho":>6} | {"ICC":>6} | {"MAE":>6}')
    print('-' * 60)
    summary = {}
    for sub in ('lpd', 'gpd', 'lrda', 'grda'):
        sub_mfs = set(sl[sl.subtype == sub].mat_file)
        rater_d = {r: dict(zip(freq[(freq.rater == r) & freq.mat_file.isin(sub_mfs)].mat_file,
                                freq[(freq.rater == r) & freq.mat_file.isin(sub_mfs)].value))
                   for r in RATERS}

        algo_col = 'pdchar_freq_hz' if sub in ('lpd', 'gpd') else 'algo_freq_hz'
        algo_d = {}
        tautan_d = {}
        for mf in sub_mfs:
            if mf in sl_idx.index:
                v = pd.to_numeric(sl_idx.loc[mf, algo_col], errors='coerce')
                if pd.notna(v) and v > 0:
                    algo_d[mf] = float(v)
                t = pd.to_numeric(sl_idx.loc[mf, 'tautan_freq_hz'], errors='coerce')
                if pd.notna(t) and t > 0:
                    tautan_d[mf] = float(t)

        ee_means = {'rho': [], 'icc': [], 'mae': []}
        ea_means = {'rho': [], 'icc': [], 'mae': []}
        et_means = {'rho': [], 'icc': [], 'mae': []}  # expert vs Tautan baseline

        from itertools import combinations as _combos
        for a, b in _combos(RATERS, 2):
            common = sorted(set(rater_d[a]) & set(rater_d[b]))
            if len(common) < 5:
                continue
            x = [rater_d[a][m] for m in common]
            y = [rater_d[b][m] for m in common]
            rho, _ = spearmanr(x, y)
            icc = icc_3_1(x, y)
            mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
            print(f'{sub.upper():>5} | EE {a}-{b:<3s}| {len(common):>4d} | {rho:>6.3f} | {icc:>6.3f} | {mae:>6.3f}')
            ee_means['rho'].append(rho); ee_means['icc'].append(icc); ee_means['mae'].append(mae)

        for r in RATERS:
            common = sorted(set(rater_d[r]) & set(algo_d))
            if len(common) < 5:
                continue
            x = [rater_d[r][m] for m in common]
            y = [algo_d[m] for m in common]
            rho, _ = spearmanr(x, y)
            icc = icc_3_1(x, y)
            mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
            print(f'{sub.upper():>5} | EA {r}-ALG  | {len(common):>4d} | {rho:>6.3f} | {icc:>6.3f} | {mae:>6.3f}')
            ea_means['rho'].append(rho); ea_means['icc'].append(icc); ea_means['mae'].append(mae)

        for r in RATERS:
            common = sorted(set(rater_d[r]) & set(tautan_d))
            if len(common) < 5:
                continue
            x = [rater_d[r][m] for m in common]
            y = [tautan_d[m] for m in common]
            rho, _ = spearmanr(x, y)
            icc = icc_3_1(x, y)
            mae = float(np.mean(np.abs(np.array(x) - np.array(y))))
            print(f'{sub.upper():>5} | ET {r}-TAU  | {len(common):>4d} | {rho:>6.3f} | {icc:>6.3f} | {mae:>6.3f}')
            et_means['rho'].append(rho); et_means['icc'].append(icc); et_means['mae'].append(mae)

        # Mean across pairs
        def _m(d): return {k: float(np.mean(v)) if v else float('nan') for k, v in d.items()}
        summary[sub] = {'EE': _m(ee_means), 'EA': _m(ea_means), 'ET': _m(et_means)}
        print()

    print('=' * 70)
    print('Mean across 3 pairs (EE = expert-expert; EA = expert-algorithm; ET = expert-Tautan)')
    print('=' * 70)
    print(f'{"task":>5} | {"role":<3s} | {"rho":>6} | {"ICC":>6} | {"MAE":>6}')
    for sub in ('lpd', 'gpd', 'lrda', 'grda'):
        for role in ('EE', 'EA', 'ET'):
            m = summary[sub][role]
            print(f'{sub.upper():>5} | {role:<3s} | {m["rho"]:>6.3f} | {m["icc"]:>6.3f} | {m["mae"]:>6.3f}')
        print()


if __name__ == '__main__':
    main()
