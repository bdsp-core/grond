#!/usr/bin/env python3
"""Train the LRDA-frequency CRNN with 5-fold patient-stratified CV.

Saves per-fold checkpoints and an aggregated out-of-fold predictions JSON
that the IRR analysis script can consume via --algo crnn.

    conda run -n morgoth python code/cet_model/train_lrda_crnn.py --epochs 80
    conda run -n morgoth python code/cet_model/train_lrda_crnn.py --fold 0   # single fold

Outputs:
    data/lrda_crnn_cache/fold_{0..4}/best.pt          per-fold checkpoint
    data/lrda_crnn_cache/fold_{0..4}/loss_curve.csv   per-epoch train/val loss
    data/labels/independent_expert_v1/lrda_crnn_predictions.json   OOF predictions
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'cet_model'))
from lrda_crnn import LRDAFreqCRNN, num_parameters  # type: ignore
from lrda_crnn_dataset import LRDADataset, _build_label_map, make_folds  # type: ignore

CACHE_DIR = PROJECT_DIR / 'data' / 'lrda_crnn_cache'
PRED_OUT = PROJECT_DIR / 'data' / 'labels' / 'independent_expert_v1' / 'lrda_crnn_predictions.json'


def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_fold(fold_idx, train_items, val_items, args, device):
    print(f'\n=== Fold {fold_idx} ===')
    print(f'  Train: {len(train_items)} segments  ({sum(1 for it in train_items if len(it["targets"]) > 1)} multi-rater)')
    print(f'  Val:   {len(val_items)} segments')

    train_ds = LRDADataset(train_items, augment=True)
    val_ds = LRDADataset(val_items, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0)

    model = LRDAFreqCRNN().to(device)
    if fold_idx == 0:
        print(f'  Model parameters: {num_parameters(model):,}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()

    fold_dir = CACHE_DIR / f'fold_{fold_idx}'
    fold_dir.mkdir(parents=True, exist_ok=True)
    loss_path = fold_dir / 'loss_curve.csv'
    ckpt_path = fold_dir / 'best.pt'
    with open(loss_path, 'w', newline='') as f:
        csv.writer(f, lineterminator='\n').writerow(['epoch', 'train_loss', 'val_loss', 'val_mae_hz'])

    best_val = float('inf')
    patience = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        t0 = time.time()
        for seg, target, _ in train_loader:
            # Skip batches with non-finite inputs (corrupt segments)
            if not torch.isfinite(seg).all() or not torch.isfinite(target).all():
                continue
            seg = seg.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad()
            pred = model(seg)
            loss = crit(pred, target)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(float(loss.item()))
        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')

        # Validate
        model.eval()
        val_losses = []
        val_preds_log = []
        val_targets_log = []
        with torch.no_grad():
            for seg, target, _ in val_loader:
                seg = seg.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                pred = model(seg)
                val_losses.append(float(crit(pred, target).item()))
                val_preds_log.extend(pred.cpu().numpy().tolist())
                val_targets_log.extend(target.cpu().numpy().tolist())
        val_loss = float(np.mean(val_losses)) if val_losses else float('nan')
        val_preds_hz = np.exp(val_preds_log)
        val_targets_hz = np.exp(val_targets_log)
        val_mae = float(np.mean(np.abs(val_preds_hz - val_targets_hz))) if len(val_preds_log) else float('nan')

        sched.step()
        with open(loss_path, 'a', newline='') as f:
            csv.writer(f, lineterminator='\n').writerow([epoch, train_loss, val_loss, val_mae])

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1

        elapsed = time.time() - t0
        marker = '*' if improved else ' '
        if epoch % 5 == 0 or improved or epoch == args.epochs - 1:
            print(f'  ep{epoch:>3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_MAE={val_mae:.3f} Hz  ({elapsed:.1f}s)  {marker}')

        if patience >= args.patience:
            print(f'  Early stopping at epoch {epoch} (best val_loss={best_val:.4f})')
            break

    # Predict on val with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    preds = {}
    with torch.no_grad():
        for seg, target, mf_batch in val_loader:
            seg = seg.to(device, non_blocking=True)
            pred_log = model(seg).cpu().numpy()
            for i, mf in enumerate(mf_batch):
                preds[mf] = float(np.exp(pred_log[i]))
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--fold', type=int, default=-1, help='Only train this fold (default -1 = all 5)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force-cpu', action='store_true',
                    help='Force CPU training (avoids MPS numerical instability).')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(force_cpu=args.force_cpu)
    print(f'Device: {device}')
    print(f'Building label map...')
    label_map = _build_label_map()
    print(f'  Total LRDA segments with at least one rater freq: {len(label_map)}')
    n_manifest = sum(1 for v in label_map.values() if v['in_manifest'])
    print(f'  In 200-segment manifest: {n_manifest}')
    print(f'  Legacy LRDA: {len(label_map) - n_manifest}')

    folds = make_folds(label_map, n_folds=5, seed=args.seed)

    fold_indices = [args.fold] if args.fold >= 0 else list(range(len(folds)))
    all_preds = {}
    for k in fold_indices:
        train_items, val_items = folds[k]
        preds = train_one_fold(k, train_items, val_items, args, device)
        all_preds.update(preds)

    if args.fold == -1:
        # Save aggregated OOF predictions
        out = {}
        for mf, freq in all_preds.items():
            sid = mf.replace('.mat', '')
            out[sid] = {
                'mat_file': mf,
                'subtype': 'lrda',
                'crnn_freq': float(np.clip(round(freq * 4) / 4.0, 0.5, 3.5)),
                'crnn_freq_raw': freq,
            }
        PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(PRED_OUT, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved {len(out)} OOF predictions to {PRED_OUT}')
        print(f'Run: conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py --algo crnn')


if __name__ == '__main__':
    main()
