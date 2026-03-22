"""
PDNetV2 Evaluation - Decode predictions and compute metrics.

Usage:
    conda run -n foe_dl python code/pdnet_v2/evaluate.py

Loads best models from each fold, evaluates on held-out patients,
and reports:
  - Discharge timing F1, precision, recall (tolerance ±100ms)
  - Frequency Spearman correlation
  - Comparison to current best F1=0.7395
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.stats import spearmanr

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pdnet_v2.model import PDNetV2
from pdnet_v2.dataset import build_dataset, TARGET_FS, N_BINS
from pdnet_v2.train import (load_all_segments, decode_predictions,
                             _match_events, BATCH_SIZE)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
CACHE_DIR = DATA_DIR / 'pdnet_v2_cache'

TOLERANCE = 0.1   # seconds (±100ms)
CURRENT_BEST_F1 = 0.7395


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def decode_batch_predictions(model, loader, device):
    """
    Run model on all batches and return per-patient predictions.

    Returns:
        dict: {pid: {'pred_times': [...], 'gt_times': [...],
                     'pred_freq': float, 'gt_freq': float}}
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            outputs = model(eeg)
            event_logits, active_logits, freq_loghz = outputs[0], outputs[1], outputs[2]

            for i in range(len(eeg)):
                pid = batch['pid'][i]

                # Decode discharge times
                pred_times = decode_predictions(event_logits[i], active_logits[i])

                # Ground truth times from y_event
                y_event = batch['y_event'][i].cpu().numpy()
                gt_peaks, _ = find_peaks(y_event, height=0.5)
                gt_times = [pk / TARGET_FS for pk in gt_peaks]

                # Predicted frequency from active + freq head
                p_active = torch.sigmoid(active_logits[i]).cpu().numpy()
                freq_logvals = freq_loghz[i].cpu().numpy()
                active_mask = p_active > 0.4
                if active_mask.sum() > 5:
                    pred_log_freq = float(np.median(freq_logvals[active_mask]))
                    pred_freq = float(np.exp(pred_log_freq))
                elif pred_times and len(pred_times) >= 2:
                    ipis = np.diff(sorted(pred_times))
                    pred_freq = float(1.0 / np.median(ipis))
                else:
                    pred_freq = float('nan')

                # Ground truth frequency (from y_freq if available)
                y_freq = batch['y_freq'][i].cpu().numpy()
                y_freq_mask_np = batch['y_freq_mask'][i].cpu().numpy()
                if y_freq_mask_np.sum() > 5:
                    gt_log_freq = float(np.median(y_freq[y_freq_mask_np > 0.5]))
                    gt_freq = float(np.exp(gt_log_freq))
                elif len(gt_times) >= 2:
                    ipis = np.diff(sorted(gt_times))
                    gt_freq = float(1.0 / np.median(ipis))
                else:
                    gt_freq = float('nan')

                results[pid] = {
                    'pred_times': pred_times,
                    'gt_times': gt_times,
                    'pred_freq': pred_freq,
                    'gt_freq': gt_freq,
                }

    return results


def evaluate_predictions(results, tol=TOLERANCE):
    """
    Compute aggregate metrics from predictions dict.

    Returns dict with F1, precision, recall, freq_spearman
    """
    all_tp = 0
    all_fp = 0
    all_fn = 0
    pred_freqs = []
    gt_freqs = []

    for pid, res in results.items():
        gt_times = res['gt_times']
        pred_times = res['pred_times']

        tp, fp, fn = _match_events(gt_times, pred_times, tol)
        all_tp += tp
        all_fp += fp
        all_fn += fn

        if not np.isnan(res['pred_freq']) and not np.isnan(res['gt_freq']):
            pred_freqs.append(res['pred_freq'])
            gt_freqs.append(res['gt_freq'])

    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_tp + all_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    if len(pred_freqs) >= 3:
        freq_spearman, _ = spearmanr(pred_freqs, gt_freqs)
    else:
        freq_spearman = float('nan')

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn,
        'n_patients': len(results),
        'freq_spearman': freq_spearman,
        'n_freq_pairs': len(pred_freqs),
    }


def load_fold_pids(fold_path):
    """Load train/val PIDs from a saved fold checkpoint."""
    ckpt = torch.load(str(fold_path), map_location='cpu')
    return ckpt.get('train_pids', []), ckpt.get('val_pids', [])


def main():
    print("=" * 70)
    print("PDNetV2 Evaluation")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # Check for saved models
    fold_models = sorted(CACHE_DIR.glob('fold*_best.pt'))
    if not fold_models:
        print(f"ERROR: No saved models found in {CACHE_DIR}")
        print("Run training first: conda run -n foe_dl python code/pdnet_v2/train.py")
        return

    print(f"Found {len(fold_models)} fold models: {[f.name for f in fold_models]}")

    # Load data
    print("\nLoading patients and labels...")
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)

    with open(str(DATA_DIR / 'labels' / 'discharge_times_hpp.json')) as f:
        hpp_data = json.load(f)

    print("\nLoading EEG segments...")
    t0 = time.time()
    segments_by_patient = load_all_segments(verbose=True)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Evaluate each fold
    all_results = {}  # pid -> results (from the fold where it was validation)
    fold_metrics = []

    for fold_path in fold_models:
        fold_idx = int(fold_path.stem.split('fold')[1].split('_')[0])
        print(f"\n--- Evaluating Fold {fold_idx} ---")

        # Load model
        ckpt = torch.load(str(fold_path), map_location='cpu')
        model = PDNetV2()
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()

        val_pids = ckpt.get('val_pids', [])
        print(f"  Validation patients: {len(val_pids)}")
        print(f"  Best epoch: {ckpt.get('epoch', '?')}, "
              f"Val F1 (at training): {ckpt.get('val_f1', '?'):.4f}")

        # Build validation dataset
        val_ds = build_dataset(val_pids, segments_by_patient, hpp_data, df_patients, augment=False)
        print(f"  Val dataset size: {len(val_ds)}")

        if len(val_ds) == 0:
            print("  WARNING: empty validation dataset, skipping")
            continue

        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=False)

        # Run predictions
        fold_results = decode_batch_predictions(model, val_loader, device)
        metrics = evaluate_predictions(fold_results)

        print(f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}")
        print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")
        print(f"  Freq Spearman={metrics['freq_spearman']:.4f} "
              f"(n={metrics['n_freq_pairs']})")

        all_results.update(fold_results)
        fold_metrics.append(metrics)

    # Overall metrics (across all val patients from all folds)
    print("\n" + "=" * 70)
    print("OVERALL EVALUATION (all folds combined)")
    print("=" * 70)

    overall_metrics = evaluate_predictions(all_results)
    print(f"Total patients evaluated: {overall_metrics['n_patients']}")
    print(f"F1          = {overall_metrics['f1']:.4f}   (current best: {CURRENT_BEST_F1})")
    print(f"Precision   = {overall_metrics['precision']:.4f}")
    print(f"Recall      = {overall_metrics['recall']:.4f}")
    print(f"TP={overall_metrics['tp']}  FP={overall_metrics['fp']}  FN={overall_metrics['fn']}")
    print(f"Freq Spearman = {overall_metrics['freq_spearman']:.4f} "
          f"(n={overall_metrics['n_freq_pairs']})")

    improvement = overall_metrics['f1'] - CURRENT_BEST_F1
    print(f"\nDelta vs current best: {improvement:+.4f} "
          f"({'IMPROVEMENT' if improvement > 0 else 'degradation'})")

    # Per-fold summary
    if fold_metrics:
        f1_list = [round(m['f1'], 4) for m in fold_metrics]
        print(f"\nPer-fold F1: {f1_list}")
        mean_fold_f1 = np.mean([m['f1'] for m in fold_metrics])
        print(f"Mean fold F1: {mean_fold_f1:.4f}")

    # Save results
    results_path = CACHE_DIR / 'evaluation_results.json'
    save_data = {
        'overall_metrics': overall_metrics,
        'fold_metrics': fold_metrics,
        'current_best_f1': CURRENT_BEST_F1,
        'improvement': improvement,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(str(results_path), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    return overall_metrics


if __name__ == '__main__':
    main()
