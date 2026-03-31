#!/usr/bin/env python
"""
Evaluation script for E2E Discharge Detector.

Loads trained model, runs on all labeled segments, computes metrics,
and saves results.

Usage:
    python code/e2e_model/evaluate_e2e.py
    python code/e2e_model/evaluate_e2e.py --phase 2
    python code/e2e_model/evaluate_e2e.py --test-case sub-S0001116940915_20140820235019
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from e2e_model.model import E2EDischargeDetector
from e2e_model.dataset import E2EDataset, build_sample_list, custom_collate, load_hpp_cache
from e2e_model.train_e2e import compute_timing_metrics, get_device
from torch.utils.data import DataLoader

CACHE_DIR = PROJECT_DIR / 'data' / 'e2e_cache'
RESULTS_DIR = PROJECT_DIR / 'results' / 'hemi_retrain_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PRETRAINED_PATH = PROJECT_DIR / 'data' / 'pd_channel_cache' / 'cnn_attn_fold0.pt'


def load_best_model(phase=2, fold=0, device='cpu'):
    """Load best model from a given phase and fold."""
    model = E2EDischargeDetector(
        pretrained_path=str(PRETRAINED_PATH) if PRETRAINED_PATH.exists() else None
    )

    ckpt_path = CACHE_DIR / f'e2e_phase{phase}_fold{fold}.pt'
    if not ckpt_path.exists():
        # Try other phase
        alt_phase = 1 if phase == 2 else 2
        ckpt_path = CACHE_DIR / f'e2e_phase{alt_phase}_fold{fold}.pt'

    if ckpt_path.exists():
        print(f"Loading model from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print(f"WARNING: No checkpoint found, using randomly initialized model")

    model.to(device)
    model.eval()
    return model


def evaluate_all(model, samples, device, batch_size=4, hpp_cache=None):
    """Run evaluation on all samples."""
    dataset = E2EDataset(samples, augment=False, hpp_cache=hpp_cache)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=custom_collate
    )

    all_results = []
    timing_metrics = []
    freq_preds = []
    freq_gts = []
    lat_preds = []
    lat_gts = []

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            hpp = batch['hpp'].to(device)
            outputs = model(eeg, hpp)

            pred_t = outputs['pred_times'].cpu().numpy()
            pred_c = outputs['pred_confs'].cpu().numpy()
            gt_t = batch['gt_times'].numpy()
            gt_m = batch['gt_mask'].numpy()

            for b in range(eeg.shape[0]):
                # Timing metrics
                tm = compute_timing_metrics(pred_t[b], pred_c[b], gt_t[b], gt_m[b])
                timing_metrics.append(tm)

                # Active predictions
                active = pred_c[b] > 0.5
                active_times = sorted(pred_t[b][active].tolist())
                n_gt = int(gt_m[b].sum())

                result = {
                    'key': batch['key'][b],
                    'patient_id': batch['patient_id'][b],
                    'pred_times': active_times,
                    'pred_freq': outputs['pred_freq'][b].cpu().item(),
                    'lat_logit': outputs['lat_logit'][b].cpu().item(),
                    'gt_times': gt_t[b][:n_gt].tolist(),
                    'gt_freq': batch['freq'][b].item(),
                    'gt_lat': batch['lat'][b].item(),
                    'timing_f1': tm['f1'],
                    'timing_mae': tm['mae'] if not np.isnan(tm['mae']) else None,
                }
                all_results.append(result)

                # Frequency
                if batch['freq'][b].item() > 0:
                    freq_preds.append(outputs['pred_freq'][b].cpu().item())
                    freq_gts.append(batch['freq'][b].item())

                # Laterality
                if batch['lat'][b].item() >= 0:
                    lat_val = torch.sigmoid(outputs['lat_logit'][b] * 5.0).cpu().item()
                    lat_preds.append(lat_val)
                    lat_gts.append(batch['lat'][b].item())

    # Aggregate metrics
    from scipy.stats import spearmanr

    avg_f1 = np.mean([m['f1'] for m in timing_metrics])
    avg_sens = np.mean([m['sensitivity'] for m in timing_metrics])
    avg_prec = np.mean([m['precision'] for m in timing_metrics])
    mae_vals = [m['mae'] for m in timing_metrics if not np.isnan(m['mae'])]
    avg_mae = np.mean(mae_vals) if mae_vals else None

    freq_rho = 0.0
    freq_mae = None
    if len(freq_preds) >= 5:
        rho, _ = spearmanr(freq_preds, freq_gts)
        freq_rho = float(rho) if not np.isnan(rho) else 0.0
        freq_mae = float(np.mean(np.abs(
            np.array(freq_preds) - np.array(freq_gts)
        )))

    lat_auc = 0.0
    if len(lat_preds) >= 5:
        from sklearn.metrics import roc_auc_score
        try:
            lat_binary = [1 if g > 0.5 else 0 for g in lat_gts]
            if len(set(lat_binary)) > 1:
                lat_auc = roc_auc_score(lat_binary, lat_preds)
        except Exception:
            pass

    summary = {
        'n_samples': len(all_results),
        'timing_f1': avg_f1,
        'timing_sensitivity': avg_sens,
        'timing_precision': avg_prec,
        'timing_mae': avg_mae,
        'freq_rho': freq_rho,
        'freq_mae': freq_mae,
        'lat_auc': lat_auc,
        'n_freq_samples': len(freq_preds),
        'n_lat_samples': len(lat_preds),
    }

    return summary, all_results


def evaluate_single(model, samples, key, device, hpp_cache=None):
    """Evaluate a single test case by key."""
    matching = [s for s in samples if s['key'] == key]
    if not matching:
        print(f"Key '{key}' not found in samples")
        return None

    dataset = E2EDataset(matching, augment=False, hpp_cache=hpp_cache)
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate)

    with torch.no_grad():
        batch = next(iter(loader))
        eeg = batch['eeg'].to(device)
        hpp = batch['hpp'].to(device)
        outputs = model(eeg, hpp)

    pred_t = outputs['pred_times'][0].cpu().numpy()
    pred_c = outputs['pred_confs'][0].cpu().numpy()
    gt_t = batch['gt_times'][0].numpy()
    gt_m = batch['gt_mask'][0].numpy()
    n_gt = int(gt_m.sum())

    active = pred_c > 0.5
    active_times = sorted(pred_t[active].tolist())

    print(f"\nTest case: {key}")
    print(f"  GT times ({n_gt}): {gt_t[:n_gt].tolist()}")
    print(f"  Pred times ({len(active_times)}): {[f'{t:.3f}' for t in active_times]}")
    print(f"  GT freq: {batch['freq'][0].item():.3f}")
    print(f"  Pred freq: {outputs['pred_freq'][0].cpu().item():.3f}")
    print(f"  Lat logit: {outputs['lat_logit'][0].cpu().item():.3f}")

    tm = compute_timing_metrics(pred_t, pred_c, gt_t, gt_m)
    print(f"  Timing F1: {tm['f1']:.3f}")
    print(f"  Timing MAE: {tm['mae']:.4f}" if not np.isnan(tm['mae']) else "  Timing MAE: N/A")

    # Show all query predictions
    order = np.argsort(pred_t)
    print(f"\n  All queries (sorted by time):")
    for i in order:
        marker = '*' if pred_c[i] > 0.5 else ' '
        print(f"    [{marker}] t={pred_t[i]:.3f}s  conf={pred_c[i]:.3f}")

    return {
        'key': key,
        'pred_times': active_times,
        'gt_times': gt_t[:n_gt].tolist(),
        'timing_f1': tm['f1'],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate E2E Discharge Detector')
    parser.add_argument('--phase', type=int, default=2,
                        help='Which phase model to load (1 or 2)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold model to load')
    parser.add_argument('--test-case', type=str, default=None,
                        help='Evaluate single test case by key')
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Build samples
    samples = build_sample_list()
    print(f"Total samples: {len(samples)}")

    # Load HPP cache
    hpp_cache = load_hpp_cache()

    # Load model
    model = load_best_model(phase=args.phase, fold=args.fold, device=device)

    if args.test_case:
        evaluate_single(model, samples, args.test_case, device, hpp_cache=hpp_cache)
    else:
        print(f"\nEvaluating on all {len(samples)} samples...")
        summary, all_results = evaluate_all(model, samples, device, args.batch_size,
                                            hpp_cache=hpp_cache)

        print(f"\n{'='*60}")
        print("E2E Discharge Detector Results")
        print(f"{'='*60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Save results
        output = {
            'summary': summary,
            'per_sample': all_results,
        }
        results_path = RESULTS_DIR / 'e2e_results.json'

        # Make JSON serializable
        def make_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        with open(results_path, 'w') as f:
            json.dump(make_serializable(output), f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
