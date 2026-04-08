#!/usr/bin/env python3
"""Evaluate PDProfiler with v3 weights on the retrain contest dataset.

Runs baseline (entry A) from retrain_contest.py but with v3 model weights:
  - CNN Attention: v3_cnn_attn_fold{0-4}.pt
  - CET-UNet: v3_cet_unet_fold{0-4}.pt
  - HemiCET: v3_hemi_cet_fold{0-4}.pt (used via DischargeDetector)

Saves results to results/hemi_retrain_contest/v3_results.json
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import scipy.io as sio

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

FS = 200
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
RESULTS_DIR = ROOT_DIR / 'results' / 'hemi_retrain_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_v3_characterizer():
    """Create PDProfiler with v3 weight paths.

    Monkey-patches the weight loading to use v3_ prefixed files.
    """
    import torch
    from pd_profiler import PDProfiler
    from pd_channel_detector.channel_cnn import ChannelPDNetAttention
    from cet_model.cet import CETUNet

    pc = PDProfiler()

    # Override CNN attention loading with v3 weights
    cnn_dir = ROOT_DIR / 'data' / 'pd_channel_cache'
    cnn_models = []
    for fold in range(5):
        path = cnn_dir / f'v3_cnn_attn_fold{fold}.pt'
        if path.exists():
            m = ChannelPDNetAttention()
            m.load_state_dict(torch.load(str(path), map_location='cpu', weights_only=True))
            m.to(pc.device)
            m.eval()
            cnn_models.append(m)
        else:
            print(f"WARNING: Missing v3 CNN weight: {path}")
    pc._channel_cnn_models = cnn_models
    print(f"Loaded {len(cnn_models)} v3 CNN attention models")

    # Override CET loading with v3 weights via DischargeDetector
    # The DischargeDetector loads CET and CNN models.
    # We need to patch it to use v3 prefixes.
    from discharge_detector import DischargeDetector

    class V3DischargeDetector(DischargeDetector):
        def __init__(self):
            cet_model_dir = ROOT_DIR / 'data' / 'cet_cache'
            cnn_model_dir = ROOT_DIR / 'data' / 'pd_channel_cache'

            if torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            self.device = device

            self.cet_models = self._load_models(
                CETUNet, cet_model_dir, 'v3_cet_unet_fold', 5)
            self.cnn_models = self._load_models(
                ChannelPDNetAttention, cnn_model_dir, 'v3_cnn_attn_fold', 5)

    det = V3DischargeDetector()
    pc._cet_models = det.cet_models
    pc._cet_compute = det.compute_cet_evidence_channel
    pc._freq_estimator = det
    print(f"Loaded {len(det.cet_models)} v3 CET-UNet models")
    print(f"Loaded {len(det.cnn_models)} v3 CNN models (for DischargeDetector)")

    return pc


def load_evaluation_data():
    """Load all labeled LPD segments and ground truth."""
    import os
    sl = pd.read_csv(ROOT_DIR / 'data' / 'labels' / 'segment_labels.csv')
    lpd = sl[(sl.subtype == 'lpd') & (sl.excluded != True)].copy()

    with open(ROOT_DIR / 'data' / 'labels' / 'discharge_times.json') as f:
        dt_json = json.load(f)

    eeg_files = set(os.listdir(ROOT_DIR / 'data' / 'eeg'))
    dt_by_matfile = {}
    for key, val in dt_json.items():
        if isinstance(val, dict):
            times = val.get('discharge_times', val.get('times', val.get('global_times', [])))
        elif isinstance(val, list):
            times = val
        else:
            continue
        if not times:
            continue
        if key + '.mat' in eeg_files:
            dt_by_matfile[key + '.mat'] = times
        elif not key.startswith('sub-'):
            for f in eeg_files:
                if key in f and f.endswith('.mat'):
                    dt_by_matfile[f] = times
                    break

    print(f"LPD segments: {len(lpd)}")
    print(f"  With laterality: {lpd.laterality.isin(['left','right']).sum()}")
    print(f"  With expert freq: {lpd.expert_freq_hz.notna().sum()}")
    print(f"  With discharge times: {sum(1 for m in lpd.mat_file if m in dt_by_matfile)}")

    return lpd, dt_by_matfile


def main():
    t0 = time.time()
    print("=" * 70)
    print("V3 PDProfiler Evaluation (Curated dataset + expert labels)")
    print("=" * 70)

    print("\nLoading v3 models...")
    pc = create_v3_characterizer()

    print("\nLoading evaluation data...")
    lpd, dt_by_matfile = load_evaluation_data()

    # Evaluation accumulators
    gt_labels, lat_scores = [], []
    lat_correct = 0
    freq_gt, freq_pred = [], []
    tp_total, fp_total, fn_total = 0, 0, 0
    timing_errors = []
    n_processed = 0
    test_case_result = None

    for _, row in lpd.iterrows():
        mat_file = row['mat_file']
        try:
            mat = sio.loadmat(str(ROOT_DIR / 'data' / 'eeg' / mat_file))
            data = mat['data']
            if data.shape[0] > data.shape[1]:
                data = data.T
            seg_bi = fcn_getBanana(data[:, :2000])

            result = pc.characterize(seg_bi, subtype='lpd')

        except Exception as e:
            continue

        n_processed += 1

        # --- Laterality ---
        gt_lat = row.get('laterality')
        if gt_lat in ('left', 'right'):
            cp = result['channel_probs']
            score = np.mean([cp[i] for i in RIGHT_INDICES]) - np.mean([cp[i] for i in LEFT_INDICES])
            gt_labels.append(1 if gt_lat == 'right' else 0)
            lat_scores.append(score)
            if result['laterality'] == gt_lat:
                lat_correct += 1

        # --- Frequency ---
        gt_freq = float(row['expert_freq_hz']) if pd.notna(row.get('expert_freq_hz')) else None
        pred_freq = result.get('frequency')
        if gt_freq and pred_freq and np.isfinite(pred_freq):
            freq_gt.append(gt_freq)
            freq_pred.append(pred_freq)

        # --- Discharge timing ---
        gt_times = dt_by_matfile.get(mat_file)
        if gt_times and len(gt_times) > 0:
            pred_times = sorted(result.get('discharge_times', []))
            gt_sorted = sorted(gt_times)
            tol = 0.1  # 100ms

            matched_gt = set()
            for pt in pred_times:
                best_j, best_d = -1, tol + 1
                for j, gt_t in enumerate(gt_sorted):
                    if j not in matched_gt:
                        d = abs(pt - gt_t)
                        if d < best_d:
                            best_d = d
                            best_j = j
                if best_j >= 0 and best_d <= tol:
                    matched_gt.add(best_j)
                    timing_errors.append(best_d)

            tp = len(matched_gt)
            fp = len(pred_times) - tp
            fn = len(gt_sorted) - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

        # --- Test case ---
        if 'S0001116940915_20140820235019' in mat_file:
            test_case_result = {
                'pred_lat': result.get('laterality', '?'),
                'pred_freq': round(result.get('frequency', 0), 2),
                'n_det': len(result.get('discharge_times', [])),
                'gt_lat': 'right',
                'gt_freq': 1.75,
            }

        if n_processed % 200 == 0:
            print(f"  {n_processed} segments processed...")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_processed} segments in {elapsed:.0f}s")

    # --- Compute metrics ---
    metrics = {
        'name': 'V3_Curated_ExpertLabels',
        'description': 'v3 weights: curated 815-patient dataset + expanded expert labels',
        'version': 'v3',
        'n_processed': n_processed,
        'time_s': round(elapsed, 1),
    }

    # Laterality
    if gt_labels and len(set(gt_labels)) > 1:
        metrics['lat_auc'] = round(roc_auc_score(gt_labels, lat_scores), 3)
        metrics['lat_acc'] = round(100 * lat_correct / len(gt_labels), 1)
        metrics['lat_n'] = len(gt_labels)
    else:
        metrics['lat_auc'] = '--'
        metrics['lat_acc'] = '--'

    # Frequency
    if freq_gt:
        rho, _ = spearmanr(freq_gt, freq_pred)
        metrics['freq_rho'] = round(rho, 3)
        metrics['freq_mae'] = round(np.mean(np.abs(np.array(freq_gt) - np.array(freq_pred))), 3)
        metrics['freq_n'] = len(freq_gt)
    else:
        metrics['freq_rho'] = '--'
        metrics['freq_mae'] = '--'

    # Timing
    if tp_total > 0:
        prec = tp_total / (tp_total + fp_total)
        sens = tp_total / (tp_total + fn_total)
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        metrics['timing_f1'] = round(f1, 3)
        metrics['timing_sens'] = round(sens, 3)
        metrics['timing_prec'] = round(prec, 3)
        metrics['timing_mae_ms'] = round(np.mean(timing_errors) * 1000, 1)
        metrics['timing_tp'] = tp_total
        metrics['timing_fp'] = fp_total
        metrics['timing_fn'] = fn_total
    else:
        metrics['timing_f1'] = '--'
        metrics['timing_sens'] = '--'
        metrics['timing_prec'] = '--'
        metrics['timing_mae_ms'] = '--'

    metrics['test_case'] = test_case_result or {}

    # Print summary
    print(f"\n{'='*70}")
    print("V3 EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"  Laterality AUC:  {metrics.get('lat_auc', '--')}")
    print(f"  Laterality Acc:  {metrics.get('lat_acc', '--')}%")
    print(f"  Freq Spearman:   {metrics.get('freq_rho', '--')}")
    print(f"  Freq MAE:        {metrics.get('freq_mae', '--')}")
    print(f"  Timing F1:       {metrics.get('timing_f1', '--')}")
    print(f"  Timing Sens:     {metrics.get('timing_sens', '--')}")
    print(f"  Timing Prec:     {metrics.get('timing_prec', '--')}")
    print(f"  Timing MAE (ms): {metrics.get('timing_mae_ms', '--')}")
    if test_case_result:
        print(f"  Test case: lat={test_case_result['pred_lat']}, "
              f"freq={test_case_result['pred_freq']}, det={test_case_result['n_det']}")

    # Save results
    result_path = RESULTS_DIR / 'v3_results.json'
    with open(result_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to {result_path}")
    print(f"  Total time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
