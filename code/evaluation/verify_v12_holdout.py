#!/usr/bin/env python3
"""Patient-stratified held-out retune for V12 (NB-Hilbert) LRDA frequency.

The published V12 hyperparameters were selected by
code/evaluation/lrda_freq_hyperparam_sweep.py via a 192-combo sweep
**on the same 155-segment canonical-cohort majority-accept LRDA set**
that is used for the EE-vs-EA evaluation in §3.2 (figure irr_main, panel
B). That is in-sample evaluation: hyperparameters were chosen to
maximize the very metric reported as the production-pipeline result.

This script splits the canonical-cohort LRDA segments **by patient**
into a tuning half and a held-out reporting half (50/50, fixed seed),
re-runs the 192-combo sweep on the tuning half only, picks the best
config by mean expert-algorithm ICC across MW/SZ/TZ, and re-evaluates
that *fixed* config on the held-out half.

Outputs are saved to:
    results/v12_holdout/v12_holdout_results.json
    results/v12_holdout/sweep_train_half.csv

We also recompute the *current* published V12 config
(p1_hi=4.5, p2_bw=0.5, top_k=3, freq_cap=4.5) on the held-out half so we
can report what the published config does on out-of-sample patients.

Usage:
    conda run -n morgoth python code/evaluation/verify_v12_holdout.py
"""
from __future__ import annotations
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
OUT_DIR = PROJECT_DIR / 'results' / 'v12_holdout'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42


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


def w05_param(seg_bi, p1_hi, p2_bw, top_k, freq_cap):
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos_pre, seg_bi, axis=1)
    sos1 = butter(4, [0.5 / (FS / 2), p1_hi / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_f, axis=1)
    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top = dom_chs[np.argsort(powers)[::-1][:top_k]]
    sig_p1 = np.mean(seg_n[top], axis=0)
    f1 = _hilbert_freq(sig_p1, 0.3, freq_cap)
    if not np.isfinite(f1):
        f1 = 1.5
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
    powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs2])
    top2 = dom_chs2[np.argsort(powers2)[::-1][:top_k]]
    sig_p2 = np.mean(seg_nb[top2], axis=0)
    f2 = _hilbert_freq(sig_p2, 0.3, freq_cap)
    final = f2 if np.isfinite(f2) else f1
    return float(np.clip(final, 0.25, freq_cap))


def icc_3_1(x, y):
    if len(x) < 3:
        return float('nan')
    x = np.array(x); y = np.array(y)
    n = len(x); k = 2
    M = np.column_stack([x, y])
    grand = M.mean()
    BMS = k * np.sum((M.mean(1) - grand) ** 2) / (n - 1)
    EMS = np.sum((M - M.mean(1, keepdims=True)) ** 2) / (n * (k - 1))
    return float((BMS - EMS) / (BMS + (k - 1) * EMS))


def evaluate_config(algo_preds, freq, status, eligible_subset):
    """Evaluate a fixed algo prediction map on a fixed segment subset."""
    per_rater = {}
    for r in ('MW', 'SZ', 'TZ'):
        common = [mf for mf in eligible_subset
                  if mf in algo_preds and mf in freq[r]
                  and status[r].get(mf) == 'accept']
        if len(common) < 3:
            continue
        mae = float(np.mean([abs(freq[r][mf] - algo_preds[mf]) for mf in common]))
        icc = icc_3_1([freq[r][mf] for mf in common], [algo_preds[mf] for mf in common])
        per_rater[r] = {'n': len(common), 'mae': mae, 'icc': icc}
    if not per_rater:
        return None
    ea_mean_mae = float(np.mean([v['mae'] for v in per_rater.values()]))
    ea_mean_icc = float(np.mean([v['icc'] for v in per_rater.values()]))
    return {'per_rater': per_rater, 'ea_mean_mae': ea_mean_mae, 'ea_mean_icc': ea_mean_icc,
            'n_segments': len(eligible_subset)}


def main():
    t0 = time.time()
    # 1. Load segment list + rater accept/reject status + rater frequencies
    with open(TASKS_DIR / 'manifest.csv') as f:
        all_lrda = [r['mat_file'] for r in csv.DictReader(f)]

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

    eligible = [mf for mf in all_lrda
                if sum(1 for r in ('MW', 'SZ', 'TZ') if status[r].get(mf) == 'accept') >= 2]
    print(f'Eligible majority-accept LRDA segments: {len(eligible)}')

    # 2. Patient-stratified 50/50 split
    pid_for_mf = {}
    with open(LABELS_DIR / 'segments.csv') as f:
        for row in csv.DictReader(f):
            mf = row['mat_file']
            if mf in set(eligible):
                pid_for_mf[mf] = row['patient_id']
    eligible_known = [mf for mf in eligible if mf in pid_for_mf]
    pids = sorted(set(pid_for_mf[mf] for mf in eligible_known))
    print(f'Unique patients in eligible cohort: {len(pids)}  '
          f'(segments with known patient: {len(eligible_known)}/{len(eligible)})')

    rng = np.random.default_rng(SEED)
    pid_perm = list(pids); rng.shuffle(pid_perm)
    half = len(pid_perm) // 2
    train_pids = set(pid_perm[:half])
    test_pids = set(pid_perm[half:])
    train_mfs = sorted([mf for mf in eligible_known if pid_for_mf[mf] in train_pids])
    test_mfs = sorted([mf for mf in eligible_known if pid_for_mf[mf] in test_pids])
    print(f'Train half: {len(train_mfs)} segments / {len(train_pids)} patients')
    print(f'Test  half: {len(test_mfs)} segments / {len(test_pids)} patients\n')

    # 3. Pre-load all eligible EEG segments
    print(f'Pre-loading {len(eligible_known)} segments...')
    segs = {}
    for mf in eligible_known:
        seg = load_segment(mf)
        if seg is not None:
            segs[mf] = seg
    print(f'  Loaded {len(segs)} segments.\n')

    # 4. Run 192-combo sweep on TRAIN HALF only
    p1_his = [3.5, 4.0, 4.5, 5.0]
    p2_bws = [0.3, 0.4, 0.5, 0.6]
    top_ks = [2, 3, 4, 5]
    freq_caps = [3.5, 4.0, 4.5]
    n_combos = len(p1_his) * len(p2_bws) * len(top_ks) * len(freq_caps)
    print(f'Sweeping {n_combos} combos on train half ({len(train_mfs)} segments)...')

    sweep_rows = []
    for i, (p1_hi, p2_bw, top_k, freq_cap) in enumerate(product(p1_his, p2_bws, top_ks, freq_caps)):
        algo = {}
        for mf in train_mfs:
            if mf not in segs:
                continue
            try:
                algo[mf] = w05_param(segs[mf], p1_hi, p2_bw, top_k, freq_cap)
            except Exception:
                pass
        m = evaluate_config(algo, freq, status, train_mfs)
        if m is None:
            continue
        sweep_rows.append({
            'p1_hi': p1_hi, 'p2_bw': p2_bw, 'top_k': top_k, 'freq_cap': freq_cap,
            'train_ea_mean_mae': round(m['ea_mean_mae'], 4),
            'train_ea_mean_icc': round(m['ea_mean_icc'], 4),
        })
        if (i + 1) % 20 == 0:
            print(f'  {i+1}/{n_combos} combos done  ({time.time() - t0:.0f}s)')

    sweep_rows.sort(key=lambda r: -r['train_ea_mean_icc'])
    best_train = sweep_rows[0]
    print(f'\nBest combo on TRAIN HALF (max train_ea_mean_icc):')
    print(f'  p1_hi={best_train["p1_hi"]}  p2_bw={best_train["p2_bw"]}  '
          f'top_k={best_train["top_k"]}  freq_cap={best_train["freq_cap"]}  '
          f'train_icc={best_train["train_ea_mean_icc"]:.3f}  '
          f'train_mae={best_train["train_ea_mean_mae"]:.3f}\n')

    # 5. Re-evaluate best-on-train + published V12 on TEST HALF
    PUBLISHED_V12 = dict(p1_hi=4.5, p2_bw=0.5, top_k=3, freq_cap=4.5)
    PUBLISHED_V12_NOTE = ('Published V12 hyperparameters from manuscript '
                          '(fig irr_main caption / §3.2)')

    print('Evaluating fixed configs on TEST HALF...')
    held_out_results = {}
    for cfg_name, cfg in [
        ('best_on_train_half', dict(p1_hi=best_train['p1_hi'], p2_bw=best_train['p2_bw'],
                                    top_k=best_train['top_k'], freq_cap=best_train['freq_cap'])),
        ('published_V12', PUBLISHED_V12),
    ]:
        algo_test = {}
        for mf in test_mfs:
            if mf not in segs:
                continue
            try:
                algo_test[mf] = w05_param(segs[mf], **cfg)
            except Exception:
                pass
        m = evaluate_config(algo_test, freq, status, test_mfs)
        held_out_results[cfg_name] = {'config': cfg, 'metrics': m}
        if m:
            print(f'  {cfg_name}: ICC={m["ea_mean_icc"]:.3f}  MAE={m["ea_mean_mae"]:.3f}  '
                  f'per-rater={[(r, m["per_rater"][r]["n"], round(m["per_rater"][r]["icc"], 3)) for r in sorted(m["per_rater"])]}')

    # 6. Also report best_on_train on the *full canonical cohort* (matches the
    # in-sample published metric) so we can cross-check against the manuscript
    # figure's reported numbers.
    algo_all = {}
    for mf in eligible_known:
        if mf not in segs:
            continue
        try:
            algo_all[mf] = w05_param(segs[mf], **PUBLISHED_V12)
        except Exception:
            pass
    full = evaluate_config(algo_all, freq, status, eligible_known)
    print(f'\nPublished V12 on FULL canonical cohort (in-sample baseline check):')
    if full:
        print(f'  ICC={full["ea_mean_icc"]:.3f}  MAE={full["ea_mean_mae"]:.3f}  '
              f'per-rater={[(r, full["per_rater"][r]["n"], round(full["per_rater"][r]["icc"], 3)) for r in sorted(full["per_rater"])]}')

    # 7. Save
    with open(OUT_DIR / 'v12_holdout_results.json', 'w') as f:
        json.dump({
            'seed': SEED,
            'n_eligible': len(eligible),
            'n_eligible_with_pid': len(eligible_known),
            'n_train_segments': len(train_mfs),
            'n_test_segments': len(test_mfs),
            'n_train_patients': len(train_pids),
            'n_test_patients': len(test_pids),
            'held_out_results': held_out_results,
            'published_V12_on_full_cohort_in_sample': full,
            'best_train_combo': best_train,
            'top_5_train_combos': sweep_rows[:5],
            'total_time_s': round(time.time() - t0, 1),
        }, f, indent=2)
    print(f'\nSaved {(OUT_DIR / "v12_holdout_results.json").relative_to(PROJECT_DIR)}')

    with open(OUT_DIR / 'sweep_train_half.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()), lineterminator='\n')
        w.writeheader()
        w.writerows(sweep_rows)
    print(f'Saved {(OUT_DIR / "sweep_train_half.csv").relative_to(PROJECT_DIR)}')
    print(f'Total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
