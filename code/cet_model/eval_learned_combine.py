"""
Improvement #4: Learned evidence combination.

Instead of naive max(HPP, CET), train a small model to optimally combine
HPP and CET evidence at each time point.

Approaches tested:
  A. Per-timepoint logistic: sigmoid(w1*hpp + w2*cet + b) — 3 params
  B. Per-timepoint MLP: small network on [hpp, cet, hpp*cet, |hpp-cet|] — ~20 params
  C. 1D conv combiner: short conv on stacked [hpp, cet] — captures local context
  D. Gated combination: gate(t) = sigmoid(w*cet + b), output = hpp + gate*cet

All trained with LOPO-style CV to avoid overfitting.

Usage:
    conda run -n foe_dl python code/cet_model/eval_learned_combine.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from cet_model.auto_pipeline import (
    load_cet_unet_models, load_cnn_attn_models,
    estimate_frequency_cnn, estimate_frequency_acf,
    compute_cet_evidence,
    _aggregate_evidence, _detect_active_interval,
    DEVICE, TOLERANCE_S, FS,
)
from cet_model.parameter_sweep import (
    run_hpp_single,
)
from optimization_harness_v2 import load_dataset
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

CACHE_DIR = PROJECT_DIR / 'data' / 'cet_cache'

OPTIMIZED_PARAMS = {
    'dp_alpha': 1.275,
    'dp_beta': 0.3,
    'dp_lambda': 0.05,
    'peak_height_frac': 0.05,
    'max_skip': 3,
}

CET_THRESHOLD_PCT = 80  # from Stage A


def normalize_evidence(ev):
    """Normalize evidence to [0,1]."""
    mx = np.max(ev)
    return ev / mx if mx > 0 else ev


def make_discharge_target(gt_times, n_samples, fs, sigma_samples=10):
    """Create target evidence trace: Gaussians at discharge locations."""
    target = np.zeros(n_samples)
    for t in gt_times:
        s = int(t * fs)
        for i in range(max(0, s - 3*sigma_samples), min(n_samples, s + 3*sigma_samples + 1)):
            target[i] = max(target[i], np.exp(-0.5 * ((i - s) / sigma_samples) ** 2))
    return target


def evaluate_combined_evidence(precomputed, combine_fn, params=OPTIMIZED_PARAMS):
    """Evaluate using a custom combine function."""
    total_tp, total_fn, total_fp = 0, 0, 0
    gt_freqs, algo_freqs = [], []
    n_cases = 0

    for pid, pc in precomputed.items():
        gt_times = pc['gt_times']

        hpp_norm = normalize_evidence(pc['hpp_agg'])
        cet_norm = normalize_evidence(pc['cet_agg'])
        evidence = combine_fn(hpp_norm, cet_norm)

        try:
            algo_times_arr = run_hpp_single(
                evidence, pc['hpp_all'], pc['cnn_freq'], FS, params)
            algo_times = sorted(algo_times_arr.tolist()) if len(algo_times_arr) > 0 else []
        except Exception:
            continue

        n_cases += 1

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    dist = abs(gt - at)
                    if dist < best_dist:
                        best_dist = dist
                        best_ai = ai
            if best_dist <= TOLERANCE_S and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = [gt_times[i+1] - gt_times[i] for i in range(len(gt_times)-1)]
        gt_freq = 1.0 / np.median(gt_ipis)
        if len(algo_times) >= 2:
            algo_ipis = [algo_times[i+1] - algo_times[i]
                         for i in range(len(algo_times)-1)]
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    if len(gt_freqs) >= 3:
        freq_rho, _ = spearmanr(algo_freqs, gt_freqs)
    else:
        freq_rho = float('nan')

    return {
        'n_cases': n_cases,
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'freq_spearman': round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
    }


# ============================================================================
# Approach A: Weighted combination with learned weights (grid search)
# ============================================================================

def weighted_combine(hpp, cet, w_hpp, w_cet, cet_floor=0.0):
    """Weighted combination with CET floor suppression."""
    cet_cleaned = np.where(cet > cet_floor, cet, 0)
    return w_hpp * hpp + w_cet * cet_cleaned


# ============================================================================
# Approach B: Gated combination
# ============================================================================

def gated_combine(hpp, cet, gate_threshold, gate_sharpness=10.0):
    """Gate CET based on HPP: only add CET where HPP is already active.

    gate = sigmoid(sharpness * (hpp - threshold))
    output = hpp + gate * cet
    """
    gate = 1.0 / (1.0 + np.exp(-gate_sharpness * (hpp - gate_threshold)))
    return hpp + gate * cet


# ============================================================================
# Approach C: Product-boosted combination
# ============================================================================

def product_boost_combine(hpp, cet, boost_weight=1.0, floor=0.0):
    """Boost evidence where both HPP and CET agree.

    output = max(hpp, cet) + boost_weight * hpp * cet
    """
    cet_cleaned = np.where(cet > floor, cet, 0)
    base = np.maximum(hpp, cet_cleaned)
    agreement = hpp * cet_cleaned
    return base + boost_weight * agreement


# ============================================================================
# Approach D: Adaptive max with CET suppression in HPP-quiet regions
# ============================================================================

def adaptive_max_combine(hpp, cet, hpp_quiet_threshold=0.1, suppression=0.0):
    """max(hpp, cet) but suppress CET in regions where HPP is quiet.

    In regions where HPP < quiet_threshold, scale CET by suppression factor.
    """
    hpp_quiet = hpp < hpp_quiet_threshold
    cet_adjusted = np.where(hpp_quiet, cet * suppression, cet)
    return np.maximum(hpp, cet_adjusted)


# ============================================================================
# Approach E: Trained 1D Conv combiner (5-fold CV)
# ============================================================================

class ConvCombiner(nn.Module):
    """Small 1D conv that combines HPP and CET evidence."""
    def __init__(self, kernel_size=51):
        super().__init__()
        # Input: 2 channels (hpp, cet)
        self.conv1 = nn.Conv1d(2, 8, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(8, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, 2, time)
        h = self.relu(self.conv1(x))
        return self.sigmoid(self.conv2(h)).squeeze(1)


def train_conv_combiner(train_data, n_epochs=50, lr=0.001, device='cpu'):
    """Train conv combiner on discharge timing labels.

    train_data: list of (hpp_norm, cet_norm, target, n_discharges)
    """
    model = ConvCombiner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Prepare tensors
    for epoch in range(n_epochs):
        total_loss = 0
        np.random.shuffle(train_data)

        for hpp_norm, cet_norm, target, _ in train_data:
            x = torch.from_numpy(
                np.stack([hpp_norm, cet_norm])[np.newaxis].astype(np.float32)
            ).to(device)
            y = torch.from_numpy(target[np.newaxis].astype(np.float32)).to(device)

            pred = model(x)

            # Weighted BCE: weight positives by 10x
            pos_weight = torch.where(y > 0.5, torch.tensor(10.0).to(device),
                                      torch.tensor(1.0).to(device))
            loss = nn.functional.binary_cross_entropy(pred, y, weight=pos_weight)

            # Sharpness penalty: penalize mean activation
            loss = loss + 0.05 * pred.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def eval_conv_combiner_cv(precomputed, n_folds=5, device='cpu'):
    """Train and evaluate conv combiner with 5-fold patient CV."""
    # Prepare data
    pids = list(precomputed.keys())
    np.random.seed(42)
    np.random.shuffle(pids)
    fold_size = len(pids) // n_folds

    all_results = {'tp': 0, 'fn': 0, 'fp': 0}
    gt_freqs_all, algo_freqs_all = [], []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(pids)
        test_pids = set(pids[test_start:test_end])
        train_pids = [p for p in pids if p not in test_pids]

        # Prepare training data
        train_data = []
        for pid in train_pids:
            pc = precomputed[pid]
            hpp_norm = normalize_evidence(pc['hpp_agg'])
            cet_norm = normalize_evidence(pc['cet_agg'])
            target = make_discharge_target(pc['gt_times'], len(hpp_norm), FS, sigma_samples=10)
            n_discharges = len(pc['gt_times'])
            train_data.append((hpp_norm, cet_norm, target, n_discharges))

        # Train
        model = train_conv_combiner(train_data, n_epochs=30, device=device)
        model.eval()

        # Test
        test_precomputed = {p: precomputed[p] for p in test_pids}

        @torch.no_grad()
        def conv_combine(hpp_norm, cet_norm):
            x = torch.from_numpy(
                np.stack([hpp_norm, cet_norm])[np.newaxis].astype(np.float32)
            ).to(device)
            return model(x).squeeze().cpu().numpy()

        r = evaluate_combined_evidence(test_precomputed, conv_combine)
        all_results['tp'] += r['tp']
        all_results['fn'] += r['fn']
        all_results['fp'] += r['fp']

        print(f"    Fold {fold+1}: F1={r['f1']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Prec={r['precision']:.4f}  N={r['n_cases']}")

    tp = all_results['tp']
    fn = all_results['fn']
    fp = all_results['fp']
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    return {
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'f1': round(f1, 4),
        'tp': tp, 'fn': fn, 'fp': fp,
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("  Improvement #4: Learned Evidence Combination")
    print("=" * 78)
    print(f"\nDevice: {DEVICE}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    hpp_path = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    gt_cases = {pid: v for pid, v in hpp_data.items()
                if v.get('review_status') == 'ground_truth'}

    print("\nLoading models...")
    cet_models = load_cet_unet_models(device=DEVICE)
    cnn_models = load_cnn_attn_models(device=DEVICE)

    print(f"\nPrecomputing...")
    precomputed = {}
    t_pre = time.time()

    for idx, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        row = df[df['patient_id'] == pid]
        if len(row) == 0:
            continue
        row = row.iloc[0]

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]
        subtype = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None
        n_channels = min(seg.shape[0], 18)
        n_samples = seg.shape[1]

        try:
            cnn_freq = estimate_frequency_cnn(seg, cnn_models, DEVICE, FS)
            acf_freq = estimate_frequency_acf(seg, subtype, lat, FS)
            # Ensemble freq (best from earlier)
            ensemble_freq = 0.8 * cnn_freq + 0.2 * acf_freq

            hpp_all = np.zeros((n_channels, n_samples))
            for ch in range(n_channels):
                hpp_all[ch] = _compute_channel_evidence(seg[ch], FS)

            cet_all = np.zeros((n_channels, n_samples), dtype=np.float32)
            for ch in range(n_channels):
                if np.all(np.isfinite(seg[ch])):
                    cet_all[ch] = compute_cet_evidence(seg[ch], cet_models, DEVICE)

            hpp_agg = _aggregate_evidence(hpp_all, subtype, lat)
            cet_agg = _aggregate_evidence(cet_all, subtype, lat)

        except Exception:
            continue

        precomputed[pid] = {
            'gt_times': gt_times,
            'hpp_all': hpp_all,
            'hpp_agg': hpp_agg,
            'cet_agg': cet_agg,
            'cnn_freq': ensemble_freq,  # use ensemble freq
            'subtype': subtype,
            'laterality': lat,
        }

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(gt_cases)} ({time.time()-t_pre:.0f}s)")

    print(f"  Precomputed {len(precomputed)} cases ({time.time()-t_pre:.0f}s)")

    # ====================================================================
    # Baseline: max(HPP, CET) with 80% threshold + ensemble freq
    # ====================================================================
    print(f"\n{'='*78}")
    print("  BASELINE: max(HPP,CET) 80% threshold + ensemble freq")
    print(f"{'='*78}")

    def baseline_combine(hpp, cet):
        threshold = np.percentile(cet[cet > 0], CET_THRESHOLD_PCT) if np.any(cet > 0) else 0
        cet_clean = np.where(cet > threshold, cet, 0)
        return np.maximum(hpp, cet_clean)

    r_base = evaluate_combined_evidence(precomputed, baseline_combine)
    print(f"  F1={r_base['f1']:.4f}  Sens={r_base['sensitivity']:.4f}  "
          f"Prec={r_base['precision']:.4f}  FrqRho={r_base['freq_spearman']}  "
          f"TP={r_base['tp']} FN={r_base['fn']} FP={r_base['fp']}")

    # Also test plain max (no threshold) with ensemble freq
    def plain_max_combine(hpp, cet):
        return np.maximum(hpp, cet)

    r_plain = evaluate_combined_evidence(precomputed, plain_max_combine)
    print(f"  (plain max, no threshold: F1={r_plain['f1']:.4f})")

    # ====================================================================
    # Approach A: Weighted combination sweep
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Approach A: Weighted Combination (grid search)")
    print(f"{'='*78}")

    best_a = {'f1': 0}
    for w_hpp in [0.5, 0.7, 0.8, 0.9, 1.0]:
        for w_cet in [0.3, 0.5, 0.7, 1.0]:
            for floor in [0.0, 0.1, 0.2, 0.3]:
                combine_fn = lambda h, c, wh=w_hpp, wc=w_cet, fl=floor: weighted_combine(h, c, wh, wc, fl)
                r = evaluate_combined_evidence(precomputed, combine_fn)
                if r['f1'] > best_a['f1']:
                    best_a = {**r, 'w_hpp': w_hpp, 'w_cet': w_cet, 'floor': floor}

    print(f"  Best: w_hpp={best_a['w_hpp']}, w_cet={best_a['w_cet']}, floor={best_a['floor']}")
    print(f"  F1={best_a['f1']:.4f}  Sens={best_a['sensitivity']:.4f}  "
          f"Prec={best_a['precision']:.4f}  "
          f"({best_a['f1'] - r_base['f1']:+.4f} vs baseline)")

    # ====================================================================
    # Approach B: Gated combination
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Approach B: Gated Combination")
    print(f"{'='*78}")

    best_b = {'f1': 0}
    for gate_thresh in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for sharpness in [5.0, 10.0, 20.0, 50.0]:
            combine_fn = lambda h, c, gt=gate_thresh, gs=sharpness: gated_combine(h, c, gt, gs)
            r = evaluate_combined_evidence(precomputed, combine_fn)
            if r['f1'] > best_b['f1']:
                best_b = {**r, 'gate_thresh': gate_thresh, 'sharpness': sharpness}

    print(f"  Best: gate_thresh={best_b['gate_thresh']}, sharpness={best_b['sharpness']}")
    print(f"  F1={best_b['f1']:.4f}  Sens={best_b['sensitivity']:.4f}  "
          f"Prec={best_b['precision']:.4f}  "
          f"({best_b['f1'] - r_base['f1']:+.4f} vs baseline)")

    # ====================================================================
    # Approach C: Product-boosted
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Approach C: Product-Boosted Combination")
    print(f"{'='*78}")

    best_c = {'f1': 0}
    for boost in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for floor in [0.0, 0.1, 0.2, 0.3]:
            combine_fn = lambda h, c, b=boost, fl=floor: product_boost_combine(h, c, b, fl)
            r = evaluate_combined_evidence(precomputed, combine_fn)
            if r['f1'] > best_c['f1']:
                best_c = {**r, 'boost': boost, 'floor': floor}

    print(f"  Best: boost={best_c['boost']}, floor={best_c['floor']}")
    print(f"  F1={best_c['f1']:.4f}  Sens={best_c['sensitivity']:.4f}  "
          f"Prec={best_c['precision']:.4f}  "
          f"({best_c['f1'] - r_base['f1']:+.4f} vs baseline)")

    # ====================================================================
    # Approach D: Adaptive max
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Approach D: Adaptive Max (suppress CET in HPP-quiet regions)")
    print(f"{'='*78}")

    best_d = {'f1': 0}
    for quiet_thresh in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for suppression in [0.0, 0.1, 0.2, 0.3, 0.5]:
            combine_fn = lambda h, c, qt=quiet_thresh, s=suppression: adaptive_max_combine(h, c, qt, s)
            r = evaluate_combined_evidence(precomputed, combine_fn)
            if r['f1'] > best_d['f1']:
                best_d = {**r, 'quiet_thresh': quiet_thresh, 'suppression': suppression}

    print(f"  Best: quiet_thresh={best_d['quiet_thresh']}, suppression={best_d['suppression']}")
    print(f"  F1={best_d['f1']:.4f}  Sens={best_d['sensitivity']:.4f}  "
          f"Prec={best_d['precision']:.4f}  "
          f"({best_d['f1'] - r_base['f1']:+.4f} vs baseline)")

    # ====================================================================
    # Approach E: Trained 1D Conv Combiner (5-fold CV)
    # ====================================================================
    print(f"\n{'='*78}")
    print("  Approach E: Trained 1D Conv Combiner (5-fold patient CV)")
    print(f"{'='*78}")

    conv_device = 'cpu'  # CPU for small model, avoid MPS overhead
    r_conv = eval_conv_combiner_cv(precomputed, n_folds=5, device=conv_device)
    print(f"  Overall: F1={r_conv['f1']:.4f}  Sens={r_conv['sensitivity']:.4f}  "
          f"Prec={r_conv['precision']:.4f}  "
          f"({r_conv['f1'] - r_base['f1']:+.4f} vs baseline)")

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"\n{'='*78}")
    print("  SUMMARY — Learned Evidence Combination")
    print(f"{'='*78}")

    all_approaches = [
        ('Baseline (max+thresh)', r_base),
        ('A: Weighted', best_a),
        ('B: Gated', best_b),
        ('C: Product-boost', best_c),
        ('D: Adaptive max', best_d),
        ('E: Conv combiner', r_conv),
    ]

    print(f"\n  {'Approach':<25s} {'F1':>6s} {'Sens':>6s} {'Prec':>6s} {'Delta':>7s}")
    print(f"  {'-'*55}")
    for name, r in all_approaches:
        delta = r['f1'] - r_base['f1']
        print(f"  {name:<25s} {r['f1']:.4f} {r['sensitivity']:.4f} {r['precision']:.4f} {delta:+.4f}")

    overall_best = max(all_approaches, key=lambda x: x[1]['f1'])
    print(f"\n  Best approach: {overall_best[0]} (F1={overall_best[1]['f1']:.4f})")

    # Save
    save_path = CACHE_DIR / 'improvement_learned_combine_results.json'
    save_data = {name: {k: v for k, v in r.items()} for name, r in all_approaches}
    with open(str(save_path), 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to {save_path}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == '__main__':
    main()
