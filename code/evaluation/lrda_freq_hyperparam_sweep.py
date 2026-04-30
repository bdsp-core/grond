#!/usr/bin/env python3
"""LRDA frequency hyperparameter sweep over W05/NB-Hilbert.

Re-tunes the design parameters that were chosen during the original
contest of 76 RDA-frequency variants, now using the canonical
majority-accept consensus dataset (155 LRDA segments with >=2 of 3
raters accepting + at least 2 raters providing freq labels).

Searches over:
    - pass-1 bandpass upper limit:  3.5, 4.0, 4.5, 5.0 Hz
    - pass-2 narrowband half-width: 0.3, 0.4, 0.5, 0.6 Hz
    - top-K channel selection:      2, 3, 4, 5 (was 3)
    - frequency search upper cap:   3.5, 4.0, 4.5 Hz (was 4.0)

Total combos: 4 * 4 * 4 * 3 = 192. For each combo, runs the modified
W05 on the 155-segment consensus set and computes:
    - mean abs error vs consensus-median expert frequency
    - per-rater MW / SZ / TZ MAE and ICC

Reports the best combos and saves results to:
    data/labels/independent_expert_v1/lrda_freq_hyperparam_sweep.csv

    conda run -n morgoth python code/evaluation/lrda_freq_hyperparam_sweep.py
"""
import csv
import json
import sys
from itertools import product
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
OUT_PATH = LABELS_DIR / 'independent_expert_v1' / 'lrda_freq_hyperparam_sweep.csv'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'


def _hilbert_freq(sig: np.ndarray, freq_min: float, freq_max: float) -> float:
    """Median Hilbert IF in [freq_min, freq_max]."""
    if np.std(sig) < 1e-10:
        return float('nan')
    analytic = hilbert(sig)
    inst = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
    mask = (inst > freq_min) & (inst < freq_max)
    valid = inst[mask]
    if len(valid) < 20:
        return float('nan')
    return float(np.median(valid))


def w05_param(seg_bi: np.ndarray, p1_hi: float, p2_bw: float, top_k: int, freq_cap: float) -> tuple[float, str]:
    """Parameterized W05 frequency estimator.

    p1_hi: pass-1 bandpass upper limit (Hz)
    p2_bw: pass-2 narrowband half-width (Hz)
    top_k: number of dominant-hemisphere channels averaged
    freq_cap: upper limit of the Hilbert IF search range (Hz)
    """
    # Pre-filter 0.3-5 Hz
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos_pre, seg_bi, axis=1)
    # Pass 1: coarse bandpass 0.5 -- p1_hi
    sos1 = butter(4, [0.5 / (FS / 2), p1_hi / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_f, axis=1)
    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
    dom_side = 'left' if ls1 >= rs1 else 'right'
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top = dom_chs[np.argsort(powers)[::-1][:top_k]]
    sig_p1 = np.mean(seg_n[top], axis=0)
    f1 = _hilbert_freq(sig_p1, freq_min=0.3, freq_max=freq_cap)
    if not np.isfinite(f1):
        f1 = 1.5
    # Pass 2 narrowband
    lo = max(f1 - p2_bw, 0.1)
    hi = min(f1 + p2_bw, FS / 2 - 0.1)
    if lo < hi:
        sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
    else:
        seg_nb = seg_n
    ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
    dom_chs2 = LEFT_CHS if ls >= rs else RIGHT_CHS
    dom_side = 'left' if ls >= rs else 'right'
    powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs2])
    top2 = dom_chs2[np.argsort(powers2)[::-1][:top_k]]
    sig_p2 = np.mean(seg_nb[top2], axis=0)
    f2 = _hilbert_freq(sig_p2, freq_min=0.3, freq_max=freq_cap)
    final = f2 if np.isfinite(f2) else f1
    return float(np.clip(final, 0.25, freq_cap)), dom_side


def icc_3_1(x: list, y: list) -> float:
    if len(x) < 3:
        return float('nan')
    x = np.array(x); y = np.array(y)
    n = len(x); k = 2
    M = np.column_stack([x, y])
    grand = M.mean()
    BMS = k * np.sum((M.mean(1) - grand) ** 2) / (n - 1)
    EMS = np.sum((M - M.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return float((BMS - EMS) / (BMS + (k - 1) * EMS))


def main():
    # Load consensus-eligible LRDA segments and rater frequencies
    with open(TASKS_DIR / 'manifest.csv') as f:
        all_lrda = [r['mat_file'] for r in csv.DictReader(f)]

    # Per-rater accept/reject status
    status = {r: {} for r in ('MW', 'SZ', 'TZ')}
    for rel, rater in [
        ('TZ/lrda_freq_labeling_results_TZ.json', 'TZ'),
        ('SZ/rda_freq_labeling_results-2.json', 'SZ'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', 'MW'),
    ]:
        with open(RAW_DIR / rel) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf and sub == 'lrda':
                status[rater][mf] = v.get('action') or 'unknown'

    freq = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'frequency_hz':
                continue
            try:
                freq[r][row['mat_file']] = float(row['value'])
            except ValueError:
                pass

    # Filter to majority-accept consensus segments
    eligible = []
    for mf in all_lrda:
        accepts = sum(1 for r in ('MW', 'SZ', 'TZ') if status[r].get(mf) == 'accept')
        if accepts >= 2:
            eligible.append(mf)
    print(f'Eligible majority-accept LRDA segments: {len(eligible)}')

    # Pre-load EEG segments once (avoid reloading per combo)
    print(f'Pre-loading {len(eligible)} segments...')
    segs = {}
    for mf in eligible:
        seg = load_segment(mf)
        if seg is not None:
            segs[mf] = seg
    print(f'  Loaded {len(segs)} segments\n')

    # Hyperparameter grid
    p1_his = [3.5, 4.0, 4.5, 5.0]
    p2_bws = [0.3, 0.4, 0.5, 0.6]
    top_ks = [2, 3, 4, 5]
    freq_caps = [3.5, 4.0, 4.5]

    print(f'Sweeping {len(p1_his)*len(p2_bws)*len(top_ks)*len(freq_caps)} combos...')
    results = []
    for i, (p1_hi, p2_bw, top_k, freq_cap) in enumerate(product(p1_his, p2_bws, top_ks, freq_caps)):
        # Run on every segment
        algo_preds = {}
        for mf, seg in segs.items():
            f, _ = w05_param(seg, p1_hi=p1_hi, p2_bw=p2_bw, top_k=top_k, freq_cap=freq_cap)
            algo_preds[mf] = f

        # Compute MAE + ICC vs each rater (only segments where rater accepted + provided freq)
        per_rater = {}
        for r in ('MW', 'SZ', 'TZ'):
            common = [mf for mf in eligible
                      if mf in algo_preds and mf in freq[r]
                      and status[r].get(mf) == 'accept']
            if not common:
                continue
            mae = float(np.mean([abs(freq[r][mf] - algo_preds[mf]) for mf in common]))
            icc = icc_3_1([freq[r][mf] for mf in common], [algo_preds[mf] for mf in common])
            per_rater[r] = {'n': len(common), 'mae': mae, 'icc': icc}

        ea_mean_mae = np.mean([per_rater[r]['mae'] for r in ('MW', 'SZ', 'TZ') if r in per_rater])
        ea_mean_icc = np.mean([per_rater[r]['icc'] for r in ('MW', 'SZ', 'TZ') if r in per_rater])

        results.append({
            'p1_hi': p1_hi,
            'p2_bw': p2_bw,
            'top_k': top_k,
            'freq_cap': freq_cap,
            'mw_mae': per_rater['MW']['mae'],
            'sz_mae': per_rater['SZ']['mae'],
            'tz_mae': per_rater['TZ']['mae'],
            'mw_icc': per_rater['MW']['icc'],
            'sz_icc': per_rater['SZ']['icc'],
            'tz_icc': per_rater['TZ']['icc'],
            'ea_mean_mae': ea_mean_mae,
            'ea_mean_icc': ea_mean_icc,
        })
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{len(p1_his)*len(p2_bws)*len(top_ks)*len(freq_caps)} combos done')

    # Sort by ea_mean_icc
    results.sort(key=lambda r: -r['ea_mean_icc'])

    # Identify the BASELINE (current production: p1=3.5, p2=0.4, top_k=3, freq_cap=4.0)
    baseline = next((r for r in results
                     if r['p1_hi'] == 3.5 and r['p2_bw'] == 0.4
                     and r['top_k'] == 3 and r['freq_cap'] == 4.0), None)
    print()
    print(f'Current baseline (p1_hi=3.5, p2_bw=0.4, top_k=3, freq_cap=4.0):')
    if baseline:
        print(f'  MAE: MW={baseline["mw_mae"]:.3f}  SZ={baseline["sz_mae"]:.3f}  TZ={baseline["tz_mae"]:.3f}  | mean={baseline["ea_mean_mae"]:.3f}')
        print(f'  ICC: MW={baseline["mw_icc"]:.3f}  SZ={baseline["sz_icc"]:.3f}  TZ={baseline["tz_icc"]:.3f}  | mean={baseline["ea_mean_icc"]:.3f}')

    print()
    print('Top 10 combos by ea_mean_icc:')
    print(f'{"p1_hi":>6s} {"p2_bw":>6s} {"top_k":>5s} {"f_cap":>6s} {"MW_mae":>7s} {"SZ_mae":>7s} {"TZ_mae":>7s} {"mean_mae":>8s} {"MW_icc":>7s} {"SZ_icc":>7s} {"TZ_icc":>7s} {"mean_icc":>8s}')
    for r in results[:10]:
        print(f'{r["p1_hi"]:>6.1f} {r["p2_bw"]:>6.1f} {r["top_k"]:>5d} {r["freq_cap"]:>6.1f} '
              f'{r["mw_mae"]:>7.3f} {r["sz_mae"]:>7.3f} {r["tz_mae"]:>7.3f} {r["ea_mean_mae"]:>8.3f} '
              f'{r["mw_icc"]:>7.3f} {r["sz_icc"]:>7.3f} {r["tz_icc"]:>7.3f} {r["ea_mean_icc"]:>8.3f}')

    # Save full results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(results[0].keys())
    with open(OUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        w.writerows(results)
    print(f'\nSaved full sweep results to {OUT_PATH}')


if __name__ == '__main__':
    main()
