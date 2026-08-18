#!/usr/bin/env python3
"""Controlled montage ablation for the RDA-Profiler (NB-Hilbert / V12).

Motivation (JNE review, referee 1, point 2): bipolar derivations emphasise
local gradients but subtract common-mode activity, which could attenuate
broad, synchronous GRDA/GPD.  This script holds the algorithm completely
fixed (V12 NB-Hilbert, frozen hyperparameters) and varies ONLY the input
montage derived from the same canonical 19-channel monopolar recording:

    bipolar        18-channel longitudinal bipolar ("banana") -- production
    car            19-channel common-average reference (monopolar)
    laplacian      19-channel surface Laplacian (spherical-spline CSD)
    bipolar+car    fusion: 18 bipolar + 19 CAR channels, per-block scaled
    bipolar+lap    fusion: 18 bipolar + 19 Laplacian channels, per-block scaled

Evaluation is the canonical 4-rater independent-expert cohort (MW, SZ, TZ,
AS) under the same majority-accept inclusion rule used for the manuscript's
headline IRR analysis, so the ablation numbers are directly comparable to
figure `fig:irr_main`.

    conda run -n morgoth python code/evaluation/montage_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))

from analyze_independent_expert_v1 import (  # type: ignore
    build_label_tables, icc31, spearman, mae, cohen_kappa,
    EE_PAIRS, EA_PAIRS,
)
import generate_v12_predictions as v12mod  # type: ignore

EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'montage_ablation'

FS = 200

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_INDICES = np.array([[MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)]
                            for a, b in BIPOLAR_PAIRS])

# Hemisphere channel groups (midline excluded in every montage, matching the
# production bipolar convention where Fz-Cz / Cz-Pz are not scored).
BIP_LEFT = np.array([0, 1, 2, 3, 8, 9, 10, 11])
BIP_RIGHT = np.array([4, 5, 6, 7, 12, 13, 14, 15])
MONO_LEFT = np.array([0, 1, 2, 3, 4, 5, 6, 7])            # Fp1 F3 C3 P3 F7 T3 T5 O1
MONO_RIGHT = np.array([11, 12, 13, 14, 15, 16, 17, 18])   # Fp2 F4 C4 P4 F8 T4 T6 O2


# ---------------------------------------------------------------- montages --

def _csd_operator():
    """19x19 linear operator for MNE's spherical-spline surface Laplacian.

    CSD is applied independently at each time sample, so the transform is a
    fixed linear map on channels; we recover it by pushing the identity
    matrix through MNE and reuse it for every segment (exact, and ~10^4x
    faster than constructing an Epochs object per segment).
    """
    import mne
    from mne.preprocessing import compute_current_source_density
    mne.set_log_level('ERROR')
    info = mne.create_info(list(MONO_CHANNELS), FS, ch_types='eeg')
    info.set_montage('standard_1020')
    eye = np.eye(len(MONO_CHANNELS))[np.newaxis, :, :]  # (1, 19, 19)
    ep = mne.EpochsArray(eye, info, verbose='ERROR')
    return compute_current_source_density(ep).get_data()[0]  # (19, 19)


_CSD = None


def to_bipolar(mono):
    return mono[BIPOLAR_INDICES[:, 0]] - mono[BIPOLAR_INDICES[:, 1]]


def to_car(mono):
    return mono - mono.mean(axis=0, keepdims=True)


def to_laplacian(mono):
    global _CSD
    if _CSD is None:
        _CSD = _csd_operator()
    return _CSD @ mono


def _block_scale(x):
    """Normalise a montage block to unit median per-channel SD.

    Needed only for the fusion conditions: bipolar and monopolar/Laplacian
    channels live on different amplitude scales, and NB-Hilbert selects its
    dominant hemisphere and top-k channels by variance, so an unscaled
    concatenation would simply hand the decision to whichever block happens
    to have the larger units.
    """
    s = float(np.median(np.std(x, axis=1)))
    return x / s if s > 1e-12 else x


MONTAGES = {
    'bipolar':     dict(build=lambda m: to_bipolar(m),
                        left=BIP_LEFT, right=BIP_RIGHT,
                        label='Bipolar (18 ch, production)'),
    'car':         dict(build=lambda m: to_car(m),
                        left=MONO_LEFT, right=MONO_RIGHT,
                        label='Common-average monopolar (19 ch)'),
    'laplacian':   dict(build=lambda m: to_laplacian(m),
                        left=MONO_LEFT, right=MONO_RIGHT,
                        label='Surface Laplacian (19 ch, spherical spline)'),
    'bipolar+car': dict(build=lambda m: np.vstack([_block_scale(to_bipolar(m)),
                                                   _block_scale(to_car(m))]),
                        left=np.concatenate([BIP_LEFT, MONO_LEFT + 18]),
                        right=np.concatenate([BIP_RIGHT, MONO_RIGHT + 18]),
                        label='Bipolar + monopolar fusion (37 ch)'),
    'bipolar+lap': dict(build=lambda m: np.vstack([_block_scale(to_bipolar(m)),
                                                   _block_scale(to_laplacian(m))]),
                        left=np.concatenate([BIP_LEFT, MONO_LEFT + 18]),
                        right=np.concatenate([BIP_RIGHT, MONO_RIGHT + 18]),
                        label='Bipolar + Laplacian fusion (37 ch)'),
}


def load_monopolar(mat_file):
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    return seg if seg.shape[0] == 19 else None


def run_v12(seg, left, right):
    """V12 NB-Hilbert with the hemisphere index sets swapped for this montage.

    v12mod.w05_v12 reads LEFT_CHS/RIGHT_CHS from module scope, so we rebind
    them around the call rather than duplicating the (frozen) algorithm.
    """
    old_l, old_r = v12mod.LEFT_CHS, v12mod.RIGHT_CHS
    v12mod.LEFT_CHS, v12mod.RIGHT_CHS = left, right
    try:
        return v12mod.w05_v12(seg)
    finally:
        v12mod.LEFT_CHS, v12mod.RIGHT_CHS = old_l, old_r


# -------------------------------------------------------------- evaluation --

RDA_SUBTYPES = ['lrda', 'grda']
EXPERTS = ['MW', 'SZ', 'TZ', 'AS']


def predict_all(tables, verbose=True):
    """{montage: {subtype: {mat_file: (freq_hz, laterality)}}}."""
    wanted = {}
    for sub in RDA_SUBTYPES:
        wanted[sub] = sorted(set(tables[sub]['freq']['ALGO'])
                             | set(tables[sub]['lat']['ALGO']))
    preds = {m: {s: {} for s in RDA_SUBTYPES} for m in MONTAGES}
    n_missing = 0
    for sub in RDA_SUBTYPES:
        mats = wanted[sub]
        if verbose:
            print(f'  {sub.upper()}: {len(mats)} segments x {len(MONTAGES)} montages')
        for i, mf in enumerate(mats):
            mono = load_monopolar(mf)
            if mono is None:
                n_missing += 1
                continue
            for name, cfg in MONTAGES.items():
                seg = cfg['build'](mono)
                f, lat = run_v12(seg, cfg['left'], cfg['right'])
                preds[name][sub][mf] = (f, lat)
            if verbose and (i + 1) % 25 == 0:
                print(f'    {i + 1}/{len(mats)}')
    if n_missing and verbose:
        print(f'  NOTE: {n_missing} segments had no recoverable 19-ch monopolar file')
    return preds


def _paired(a_map, b_map):
    keys = sorted(set(a_map) & set(b_map))
    return keys, [a_map[k] for k in keys], [b_map[k] for k in keys]


def freq_pair_metrics(expert_map, algo_map):
    keys, x, y = _paired(expert_map, algo_map)
    if len(keys) < 3:
        return None
    return dict(n=len(keys), icc=float(icc31(x, y)),
                rho=float(spearman(x, y)), mae=float(mae(x, y)), keys=keys)


def eval_condition(tables, sub, algo_freq, algo_lat):
    """Mean expert-algorithm metrics over the 4 EA pairs, + the EE reference."""
    out = {}
    ea = [freq_pair_metrics(tables[sub]['freq'][r], algo_freq) for r in EXPERTS]
    ea = [e for e in ea if e]
    out['freq_ea'] = dict(
        icc=float(np.mean([e['icc'] for e in ea])),
        rho=float(np.mean([e['rho'] for e in ea])),
        mae=float(np.mean([e['mae'] for e in ea])),
        n_pairs=len(ea), n_mean=float(np.mean([e['n'] for e in ea])),
        per_rater={r: e['icc'] for r, e in zip(EXPERTS, ea)},
    )
    ee = [freq_pair_metrics(tables[sub]['freq'][a], tables[sub]['freq'][b])
          for a, b in EE_PAIRS]
    ee = [e for e in ee if e]
    out['freq_ee'] = dict(icc=float(np.mean([e['icc'] for e in ee])),
                          rho=float(np.mean([e['rho'] for e in ee])),
                          mae=float(np.mean([e['mae'] for e in ee])),
                          n_pairs=len(ee))

    if sub == 'lrda':
        ks = []
        for r in EXPERTS:
            keys, x, y = _paired(tables[sub]['lat'][r], algo_lat)
            if len(keys) >= 3:
                ks.append(float(cohen_kappa(x, y)))
        kes = []
        for a, b in EE_PAIRS:
            keys, x, y = _paired(tables[sub]['lat'][a], tables[sub]['lat'][b])
            if len(keys) >= 3:
                kes.append(float(cohen_kappa(x, y)))
        out['lat_ea_kappa'] = float(np.mean(ks)) if ks else None
        out['lat_ee_kappa'] = float(np.mean(kes)) if kes else None
    return out


def paired_bootstrap_vs_baseline(tables, sub, base_freq, alt_freq,
                                 n_boot=2000, seed=42):
    """Two-sided paired segment-level bootstrap on mean-EA-ICC difference.

    Resamples segments (not pairs), recomputing both conditions on the same
    resample, so the comparison is paired on the exact same EEG.
    """
    rng = np.random.default_rng(seed)
    segs = sorted(set(base_freq) & set(alt_freq)
                  & set().union(*[set(tables[sub]['freq'][r]) for r in EXPERTS]))
    if len(segs) < 5:
        return None

    def mean_icc(freq_map, sample):
        vals = []
        for r in EXPERTS:
            em = tables[sub]['freq'][r]
            x = [em[s] for s in sample if s in em and s in freq_map]
            y = [freq_map[s] for s in sample if s in em and s in freq_map]
            if len(x) >= 3:
                v = icc31(x, y)
                if np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else np.nan

    obs = mean_icc(alt_freq, segs) - mean_icc(base_freq, segs)
    deltas = []
    for _ in range(n_boot):
        sample = list(rng.choice(segs, size=len(segs), replace=True))
        d = mean_icc(alt_freq, sample) - mean_icc(base_freq, sample)
        if np.isfinite(d):
            deltas.append(d)
    if len(deltas) < 100:
        return None
    deltas = np.array(deltas)
    p = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return dict(delta=float(obs), ci=[float(np.percentile(deltas, 2.5)),
                                      float(np.percentile(deltas, 97.5))],
                p=min(1.0, p), n_segments=len(segs))


def paired_bootstrap_lat_vs_baseline(tables, sub, base_lat, alt_lat,
                                     n_boot=2000, seed=42):
    """Paired segment-level bootstrap on the mean expert-algorithm Cohen kappa."""
    rng = np.random.default_rng(seed)
    segs = sorted(set(base_lat) & set(alt_lat)
                  & set().union(*[set(tables[sub]['lat'][r]) for r in EXPERTS]))
    if len(segs) < 5:
        return None

    def mean_kappa(lat_map, sample):
        vals = []
        for r in EXPERTS:
            em = tables[sub]['lat'][r]
            x = [em[s] for s in sample if s in em and s in lat_map]
            y = [lat_map[s] for s in sample if s in em and s in lat_map]
            if len(x) >= 3:
                v = cohen_kappa(x, y)
                if np.isfinite(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else np.nan

    obs = mean_kappa(alt_lat, segs) - mean_kappa(base_lat, segs)
    deltas = []
    for _ in range(n_boot):
        sample = list(rng.choice(segs, size=len(segs), replace=True))
        d = mean_kappa(alt_lat, sample) - mean_kappa(base_lat, sample)
        if np.isfinite(d):
            deltas.append(d)
    if len(deltas) < 100:
        return None
    deltas = np.array(deltas)
    p = 2 * min(float(np.mean(deltas <= 0)), float(np.mean(deltas >= 0)))
    return dict(delta=float(obs), ci=[float(np.percentile(deltas, 2.5)),
                                      float(np.percentile(deltas, 97.5))],
                p=min(1.0, p), n_segments=len(segs))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Loading canonical 4-rater majority-accept tables...')
    tables = build_label_tables(consensus='majority')

    print('Running V12 NB-Hilbert under every montage...')
    preds = predict_all(tables)

    results = {'montages': {k: v['label'] for k, v in MONTAGES.items()},
               'conditions': {}}
    for name in MONTAGES:
        results['conditions'][name] = {}
        for sub in RDA_SUBTYPES:
            af = {k: v[0] for k, v in preds[name][sub].items()}
            al = {k: v[1] for k, v in preds[name][sub].items()}
            results['conditions'][name][sub] = eval_condition(tables, sub, af, al)

    print('Paired bootstrap vs. the production bipolar montage...')
    results['vs_bipolar'] = {}
    for name in MONTAGES:
        if name == 'bipolar':
            continue
        results['vs_bipolar'][name] = {}
        for sub in RDA_SUBTYPES:
            base = {k: v[0] for k, v in preds['bipolar'][sub].items()}
            alt = {k: v[0] for k, v in preds[name][sub].items()}
            results['vs_bipolar'][name][sub] = paired_bootstrap_vs_baseline(
                tables, sub, base, alt)
        base_l = {k: v[1] for k, v in preds['bipolar']['lrda'].items()}
        alt_l = {k: v[1] for k, v in preds[name]['lrda'].items()}
        results['vs_bipolar'][name]['lrda_lat'] = paired_bootstrap_lat_vs_baseline(
            tables, 'lrda', base_l, alt_l)

    with open(OUT_DIR / 'montage_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ---- console report ----
    print()
    print('=' * 96)
    print('MONTAGE ABLATION -- RDA-Profiler (V12 NB-Hilbert), canonical 4-rater cohort')
    print('=' * 96)
    for sub in RDA_SUBTYPES:
        ee = results['conditions']['bipolar'][sub]['freq_ee']
        print(f'\n{sub.upper()} frequency  (expert-expert ceiling: '
              f'ICC {ee["icc"]:.3f}, rho {ee["rho"]:.3f}, MAE {ee["mae"]:.3f} Hz)')
        print(f'  {"montage":<44} {"EA ICC":>7} {"rho":>7} {"MAE":>7}  '
              f'{"dICC vs bipolar":>16} {"95% CI":>20} {"p":>7}')
        for name, cfg in MONTAGES.items():
            c = results['conditions'][name][sub]['freq_ea']
            if name == 'bipolar':
                print(f'  {cfg["label"]:<44} {c["icc"]:>7.3f} {c["rho"]:>7.3f} '
                      f'{c["mae"]:>7.3f}  {"(reference)":>16}')
            else:
                b = results['vs_bipolar'][name][sub]
                ci = f'[{b["ci"][0]:+.3f}, {b["ci"][1]:+.3f}]' if b else '--'
                p = f'{b["p"]:.3f}' if b else '--'
                d = f'{b["delta"]:+.3f}' if b else '--'
                print(f'  {cfg["label"]:<44} {c["icc"]:>7.3f} {c["rho"]:>7.3f} '
                      f'{c["mae"]:>7.3f}  {d:>16} {ci:>20} {p:>7}')

    print(f'\nLRDA laterality (Cohen kappa; expert-expert ceiling '
          f'{results["conditions"]["bipolar"]["lrda"]["lat_ee_kappa"]:.3f})')
    for name, cfg in MONTAGES.items():
        k = results['conditions'][name]['lrda']['lat_ea_kappa']
        if name == 'bipolar':
            print(f'  {cfg["label"]:<44} {k:>7.3f}  (reference)')
        else:
            b = results['vs_bipolar'][name].get('lrda_lat')
            if b:
                print(f'  {cfg["label"]:<44} {k:>7.3f}  {b["delta"]:+.3f}  '
                      f'[{b["ci"][0]:+.3f}, {b["ci"][1]:+.3f}]  p={b["p"]:.3f}')
            else:
                print(f'  {cfg["label"]:<44} {k:>7.3f}')
    print(f'\nWrote {OUT_DIR / "montage_ablation.json"}')


if __name__ == '__main__':
    main()
