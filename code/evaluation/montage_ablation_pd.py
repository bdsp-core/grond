#!/usr/bin/env python3
"""Controlled montage ablation for the PD-Profiler (ChannelPD-Net).

Companion to montage_ablation.py (which covers the RDA-Profiler).  Referee 1
asked whether the production bipolar derivation attenuates broad, synchronous
GPD activity by removing the common mode, and whether fusing bipolar with
monopolar or Laplacian features helps.

ChannelPD-Net is a *per-channel* model, so the montage can be swapped without
touching the architecture.  Everything else is held fixed -- the same segments,
the same patient-stratified 5-fold splits, the same per-channel label rule,
the same hyperparameters, the same number of epochs -- and only the channel
derivation varies:

    bipolar        18-ch longitudinal bipolar ("banana")      -- production
    car            19-ch common-average reference
    laplacian      19-ch surface Laplacian (spherical-spline CSD)
    bipolar+car    37-ch fusion (per-block amplitude-normalised)
    bipolar+lap    37-ch fusion (per-block amplitude-normalised)

Per-channel PD labels follow build_channel_dataset_v5.py exactly: GPD -> all
channels positive; LPD -> ipsilateral-hemisphere channels positive, all others
(including midline) negative.  The rule is defined on hemisphere membership,
so it transfers to every montage unchanged.

Reported out-of-fold, per montage:
  * channel-level PD detection AUC
  * segment-level PD hemisphere laterality AUC (mean left vs right probability)
  * CNN segment frequency vs expert frequency (Spearman rho, MAE), LPD and GPD

    conda run -n morgoth python code/evaluation/montage_ablation_pd.py --build
    conda run -n morgoth python code/evaluation/montage_ablation_pd.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention  # type: ignore
from montage_ablation import (  # type: ignore
    MONO_CHANNELS, BIPOLAR_INDICES, to_bipolar, to_car, to_laplacian,
    _block_scale, BIP_LEFT, BIP_RIGHT, MONO_LEFT, MONO_RIGHT,
)

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
CACHE = PROJECT_DIR / 'data' / 'montage_ablation_cache'
OUT_DIR = PROJECT_DIR / 'results' / 'montage_ablation'

FS = 200
N_FOLDS = 5
N_EPOCHS = 30
BATCH = 256
LR = 1e-3
SEED = 42

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# Midline channels get PD-negative labels under the v5 rule (they belong to
# neither hemisphere set), in every montage.
MONTAGES = {
    'bipolar':     dict(n=18, left=BIP_LEFT, right=BIP_RIGHT,
                        label='Bipolar (18 ch, production)'),
    'car':         dict(n=19, left=MONO_LEFT, right=MONO_RIGHT,
                        label='Common-average monopolar (19 ch)'),
    'laplacian':   dict(n=19, left=MONO_LEFT, right=MONO_RIGHT,
                        label='Surface Laplacian (19 ch)'),
    'bipolar+car': dict(n=37, left=np.concatenate([BIP_LEFT, MONO_LEFT + 18]),
                        right=np.concatenate([BIP_RIGHT, MONO_RIGHT + 18]),
                        label='Bipolar + monopolar fusion (37 ch)'),
    'bipolar+lap': dict(n=37, left=np.concatenate([BIP_LEFT, MONO_LEFT + 18]),
                        right=np.concatenate([BIP_RIGHT, MONO_RIGHT + 18]),
                        label='Bipolar + Laplacian fusion (37 ch)'),
}


def build_montage(mono, name):
    if name == 'bipolar':
        return to_bipolar(mono)
    if name == 'car':
        return to_car(mono)
    if name == 'laplacian':
        return to_laplacian(mono)
    if name == 'bipolar+car':
        return np.vstack([_block_scale(to_bipolar(mono)), _block_scale(to_car(mono))])
    if name == 'bipolar+lap':
        return np.vstack([_block_scale(to_bipolar(mono)), _block_scale(to_laplacian(mono))])
    raise ValueError(name)


# ------------------------------------------------------------ segment cache --

def build_cache():
    """Load every PD segment once as 19-ch monopolar; montages derive from this."""
    CACHE.mkdir(parents=True, exist_ok=True)
    import os
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    eeg_files = set(os.listdir(EEG_DIR))
    segs = sl[(sl.subtype.isin(['lpd', 'gpd'])) & (sl.excluded != True)
              & (sl.expert_freq_hz.notna()) & (sl.mat_file.isin(eeg_files))].copy()
    print(f'{len(segs)} candidate PD segments')

    data, meta = [], []
    for i, (_, row) in enumerate(segs.iterrows()):
        try:
            mat = sio.loadmat(str(EEG_DIR / row['mat_file']))
        except Exception:
            continue
        key = [k for k in mat if not k.startswith('_')][0]
        x = mat[key]
        if x.shape[0] > x.shape[1]:
            x = x.T
        if x.shape[0] != 19 or x.shape[1] < 2000:
            continue
        x = x[:, :2000].astype(np.float32)
        if not np.all(np.isfinite(x)):
            continue
        data.append(x)
        lat = str(row.get('laterality', '')).lower().strip()
        meta.append(dict(mat_file=row['mat_file'], patient_id=str(row['patient_id']),
                         subtype=row['subtype'], freq=float(row['expert_freq_hz']),
                         laterality='' if lat in ('nan', '') else lat))
        if (i + 1) % 250 == 0:
            print(f'  {i + 1}/{len(segs)} ({len(data)} kept)')

    arr = np.stack(data)
    np.save(CACHE / 'mono_segments.npy', arr)
    with open(CACHE / 'mono_meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'Cached {arr.shape} to {CACHE / "mono_segments.npy"}')


def load_cache():
    arr = np.load(CACHE / 'mono_segments.npy', mmap_mode='r')
    with open(CACHE / 'mono_meta.json') as f:
        meta = json.load(f)
    return arr, meta


# ---------------------------------------------------------- dataset assembly --

def channel_labels(subtype, laterality, left, right, n_ch):
    """Per-channel PD +/- labels, following build_channel_dataset_v5.py."""
    lab = np.zeros(n_ch, dtype=np.float32)
    if subtype == 'gpd' and laterality not in ('left', 'right'):
        lab[:] = 1.0
        return lab
    if subtype == 'lpd' and laterality == 'bilateral':
        lab[:] = 1.0
        return lab
    if subtype == 'lpd' and laterality == '':
        lab[:] = 1.0  # unknown laterality: conservative positive, as in v5
        return lab
    side = left if laterality == 'left' else right
    lab[side] = 1.0
    return lab


def assemble(montage, arr, meta):
    """Build the flat channel-level training tensors for one montage."""
    cfg = MONTAGES[montage]
    n_ch = cfg['n']
    X = np.empty((len(meta) * n_ch, 2000), dtype=np.float32)
    y = np.empty(len(meta) * n_ch, dtype=np.float32)
    logf = np.empty(len(meta) * n_ch, dtype=np.float32)
    seg_idx = np.empty(len(meta) * n_ch, dtype=np.int32)
    ch_idx = np.empty(len(meta) * n_ch, dtype=np.int16)

    k = 0
    for si, m in enumerate(meta):
        seg = build_montage(np.asarray(arr[si]), montage)
        lab = channel_labels(m['subtype'], m['laterality'], cfg['left'], cfg['right'], n_ch)
        lf = np.log(m['freq']) if m['freq'] > 0 else np.nan
        for c in range(n_ch):
            v = seg[c]
            s = v.std()
            X[k] = (v - v.mean()) / (s if s > 1e-8 else 1.0)   # per-channel z-score
            y[k] = lab[c]
            logf[k] = lf if lab[c] > 0 and np.isfinite(lf) else np.nan
            seg_idx[k] = si
            ch_idx[k] = c
            k += 1
    return X[:k], y[:k], logf[:k], seg_idx[:k], ch_idx[:k]


def fold_assignment(meta):
    """Patient-stratified 5-fold assignment, identical across montages."""
    pids = sorted({m['patient_id'] for m in meta})
    rng = np.random.default_rng(SEED)
    rng.shuffle(pids)
    pid_fold = {p: i % N_FOLDS for i, p in enumerate(pids)}
    return np.array([pid_fold[m['patient_id']] for m in meta], dtype=np.int8)


# --------------------------------------------------------------- train / eval --

def train_fold(Xtr, ytr, ftr, Xva, yva, fva, seed):
    torch.manual_seed(seed)
    model = ChannelPDNetAttention().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    bce = nn.BCELoss()

    def to_t(a):
        return torch.from_numpy(np.ascontiguousarray(a))

    ds = TensorDataset(to_t(Xtr), to_t(ytr), to_t(np.nan_to_num(ftr, nan=0.0)),
                       to_t(np.isfinite(ftr).astype(np.float32)))
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    best, best_state, bad = np.inf, None, 0
    for ep in range(N_EPOCHS):
        model.train()
        for xb, yb, fb, mb in dl:
            xb = xb.unsqueeze(1).to(DEVICE)
            yb, fb, mb = yb.to(DEVICE), fb.to(DEVICE), mb.to(DEVICE)
            # augmentation: amplitude scale + additive noise (as in training)
            xb = xb * (0.8 + 0.4 * torch.rand(xb.shape[0], 1, 1, device=DEVICE))
            xb = xb + 0.05 * torch.randn_like(xb)
            opt.zero_grad()
            prob, freq, _ = model(xb)
            loss = bce(prob.squeeze(1).clamp(1e-6, 1 - 1e-6), yb)
            if mb.sum() > 0:
                loss = loss + 0.5 * (((freq.squeeze(1) - fb) ** 2) * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl, n = 0.0, 0
            for i in range(0, len(Xva), 4096):
                xb = to_t(Xva[i:i + 4096]).unsqueeze(1).to(DEVICE)
                yb = to_t(yva[i:i + 4096]).to(DEVICE)
                prob, _, _ = model(xb)
                vl += float(bce(prob.squeeze(1).clamp(1e-6, 1 - 1e-6), yb)) * len(yb)
                n += len(yb)
            vl /= max(n, 1)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    probs = np.empty(len(X), dtype=np.float32)
    freqs = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + 4096])).unsqueeze(1).to(DEVICE)
            prob, f, _ = model(xb)
            probs[i:i + 4096] = prob.squeeze(1).cpu().numpy()
            freqs[i:i + 4096] = f.squeeze(1).cpu().numpy()
    return probs, freqs


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    r = np.argsort(np.argsort(s)) + 1
    return float((r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float); rx -= rx.mean()
    ry = np.argsort(np.argsort(y)).astype(float); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float('nan')


def run_montage(name, arr, meta, folds):
    cfg = MONTAGES[name]
    t0 = time.time()
    print(f'\n=== {cfg["label"]} ===')
    X, y, logf, seg_idx, ch_idx = assemble(name, arr, meta)
    print(f'  {len(X):,} channel examples ({y.mean() * 100:.1f}% PD+), '
          f'assembled in {time.time() - t0:.0f}s')

    seg_fold = folds[seg_idx]
    oof_prob = np.zeros(len(X), dtype=np.float32)
    oof_freq = np.zeros(len(X), dtype=np.float32)
    for f in range(N_FOLDS):
        tr, va = seg_fold != f, seg_fold == f
        m = train_fold(X[tr], y[tr], logf[tr], X[va], y[va], logf[va], seed=SEED + f)
        oof_prob[va], oof_freq[va] = predict(m, X[va])
        print(f'  fold {f + 1}/{N_FOLDS} done ({time.time() - t0:.0f}s elapsed)')

    res = {'label': cfg['label'], 'n_channels': int(cfg['n']),
           'channel_auc': auc(y.astype(int), oof_prob)}

    # ---- segment-level aggregation ----
    left, right = cfg['left'], cfg['right']
    lat_y, lat_s, per_sub = [], [], {'lpd': ([], []), 'gpd': ([], [])}
    # Per-segment out-of-fold records. Summary metrics alone cannot support a
    # paired bootstrap between montages, so keep the raw per-segment estimates:
    # patient_id lets the bootstrap be patient-stratified, matching the
    # manuscript's other significance tests.
    per_seg = {}
    for si, m in enumerate(meta):
        sel = seg_idx == si
        p = oof_prob[sel][np.argsort(ch_idx[sel])]
        fq = oof_freq[sel][np.argsort(ch_idx[sel])]

        # hemisphere laterality: mean probability left vs right
        if m['subtype'] == 'lpd' and m['laterality'] in ('left', 'right'):
            lat_y.append(1 if m['laterality'] == 'right' else 0)
            lat_s.append(float(p[right].mean() - p[left].mean()))

        # CNN segment frequency: PD-probability-weighted mean of per-channel log-freq
        w = p / p.sum() if p.sum() > 0 else np.full_like(p, 1 / len(p))
        est = float(np.clip(np.exp((w * fq).sum()), 0.3, 3.5))
        if m['subtype'] in per_sub:
            per_sub[m['subtype']][0].append(m['freq'])
            per_sub[m['subtype']][1].append(est)

        per_seg[m['mat_file']] = {
            'patient_id': m['patient_id'],
            'subtype': m['subtype'],
            'expert_freq': m['freq'],
            'cnn_freq': est,
            'lat_true': m['laterality'],
            'lat_score': float(p[right].mean() - p[left].mean()),
            'mean_prob': float(p.mean()),
        }

    res['laterality_auc'] = auc(np.array(lat_y), np.array(lat_s))
    res['n_laterality'] = len(lat_y)
    res['per_segment'] = per_seg
    for sub, (gt, pr) in per_sub.items():
        res[f'{sub}_freq_rho'] = spearman(gt, pr)
        res[f'{sub}_freq_mae'] = float(np.mean(np.abs(np.array(gt) - np.array(pr))))
        res[f'n_{sub}'] = len(gt)
    res['minutes'] = round((time.time() - t0) / 60, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true', help='rebuild the monopolar segment cache')
    ap.add_argument('--montages', default=','.join(MONTAGES))
    args = ap.parse_args()

    if args.build or not (CACHE / 'mono_segments.npy').exists():
        build_cache()

    arr, meta = load_cache()
    folds = fold_assignment(meta)
    print(f'{len(meta)} segments, {len({m["patient_id"] for m in meta})} patients, '
          f'device={DEVICE}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'montage_ablation_pd.json'
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for name in args.montages.split(','):
        results[name] = run_montage(name, arr, meta, folds)
        out_path.write_text(json.dumps(results, indent=2))

    print('\n' + '=' * 104)
    print('MONTAGE ABLATION -- PD-Profiler (ChannelPD-Net), out-of-fold, patient-stratified 5-fold')
    print('=' * 104)
    hdr = (f'  {"montage":<36} {"chan AUC":>9} {"lat AUC":>9} '
           f'{"LPD rho":>9} {"LPD MAE":>9} {"GPD rho":>9} {"GPD MAE":>9}')
    print(hdr)
    for name in MONTAGES:
        if name not in results:
            continue
        r = results[name]
        print(f'  {r["label"]:<36} {r["channel_auc"]:>9.3f} {r["laterality_auc"]:>9.3f} '
              f'{r["lpd_freq_rho"]:>9.3f} {r["lpd_freq_mae"]:>9.3f} '
              f'{r["gpd_freq_rho"]:>9.3f} {r["gpd_freq_mae"]:>9.3f}')
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
