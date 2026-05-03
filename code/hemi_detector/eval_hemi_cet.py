"""
Evaluate HemiCET (Experiment 5.3) — 8-channel hemisphere CET-UNet.

Three evaluation configurations:
  Config 1: HemiCET evidence only + DP
  Config 2: max(HPP, HemiCET) + DP  [product-boost]
  Config 3: Product-boost(HPP, HemiCET) + DP + all improvements

Comparisons:
  - Per-hemisphere baseline (exp0): F1≈0.672
  - Full 18ch pipeline (reference): F1≈0.740

Usage:
    conda run -n foe_dl python code/hemi_detector/eval_hemi_cet.py
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from hemi_detector.hemi_cet import HemiCET
from discharge_detector import (
    FS, LEFT_INDICES, RIGHT_INDICES,
    compute_channel_evidence,
    combine_evidence,
    detect_active_interval,
    extract_candidates,
    dp_best_sequence,
    em_refine,
    posthoc_filter,
    estimate_frequency_acf,
)
from optimization_harness_v2 import load_dataset
from label_pipeline.hpp_discharge_marking import _compute_channel_evidence
from pd_channel_detector.channel_cnn import ChannelPDNetAttention

SAVE_DIR = PROJECT_DIR / 'data' / 'hemi_cache' / 'hemi_cet'
CNN_CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {DEVICE}")

TOLERANCE_S = 0.1   # ±100ms for discharge matching
N_SAMPLES = 2000
N_FOLDS = 5


# ── Model loading ──────────────────────────────────────────────────────────

def load_hemi_cet_models(device=DEVICE):
    """Load all 5 fold HemiCET models."""
    models = []
    for fold in range(N_FOLDS):
        path = SAVE_DIR / f'hemi_cet_fold{fold}.pt'
        if not path.exists():
            raise FileNotFoundError(f"HemiCET model not found: {path}")
        m = HemiCET()
        m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        models.append(m)
    return models


def load_cnn_attn_models(device=DEVICE):
    """Load all 5 fold CNN+Attention models for frequency estimation."""
    models = []
    for fold in range(N_FOLDS):
        path = CNN_CACHE_DIR / f'cnn_attn_fold{fold}.pt'
        if not path.exists():
            raise FileNotFoundError(f"CNN+Attn model not found: {path}")
        m = ChannelPDNetAttention()
        m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        m.to(device)
        m.eval()
        models.append(m)
    return models


# ── Evidence computation ───────────────────────────────────────────────────

@torch.no_grad()
def compute_hemi_cet_evidence(seg_8ch, hemi_cet_models, device=DEVICE):
    """Run HemiCET ensemble on 8-channel hemisphere segment.

    Args:
        seg_8ch: (8, 2000) numpy array (raw EEG)
        hemi_cet_models: list of HemiCET models

    Returns:
        evidence: (2000,) float32 — ensemble-mean evidence trace
    """
    # Z-score per channel
    x = seg_8ch.astype(np.float32).copy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    x = (x - mu) / std

    x_tensor = torch.from_numpy(x[np.newaxis, :, :]).to(device)  # (1, 8, 2000)

    preds = []
    for model in hemi_cet_models:
        pred = model(x_tensor).squeeze().cpu().numpy()  # (2000,)
        preds.append(pred)

    return np.mean(preds, axis=0).astype(np.float32)


def compute_hpp_evidence_hemisphere(seg_8ch, fs=FS):
    """Compute HPP evidence for each of 8 channels, return median.

    Args:
        seg_8ch: (8, 2000) numpy array

    Returns:
        hpp_evidence: (2000,) float32 — median HPP evidence across channels
    """
    n_ch = seg_8ch.shape[0]
    evidence_ch = np.zeros((n_ch, N_SAMPLES), dtype=np.float32)
    for i in range(n_ch):
        ch = seg_8ch[i].astype(np.float32)
        ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        ev = _compute_channel_evidence(ch, fs).astype(np.float32)
        mx = ev.max()
        if mx > 1e-8:
            ev = ev / mx
        evidence_ch[i] = ev
    return np.median(evidence_ch, axis=0)


# ── Frequency estimation (CNN+ACF on hemisphere channels) ──────────────────

@torch.no_grad()
def estimate_freq_hemi(seg_8ch, hemi_indices, cnn_models, fs=FS):
    """Estimate frequency from 8 hemisphere channels using CNN+ACF ensemble.

    Mirrors the logic in exp0_baseline.run_hemisphere.

    Args:
        seg_8ch: (8, 2000) — the hemisphere segment (already extracted)
        hemi_indices: original channel indices (not used for data; kept for clarity)
        cnn_models: list of ChannelPDNetAttention models

    Returns:
        freq_estimate: float, clipped to [0.3, 3.5]
    """
    all_pd_probs = []
    all_log_freqs = []

    for i in range(seg_8ch.shape[0]):
        ch_data = seg_8ch[i].astype(np.float32).copy()
        ch_data = np.nan_to_num(ch_data, nan=0.0, posinf=0.0, neginf=0.0)
        mu, std = np.mean(ch_data), np.std(ch_data)
        if std > 1e-8:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu

        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :]).to(DEVICE)
        pd_probs, log_freqs = [], []
        for model in cnn_models:
            pd_prob, freq_pred, _ = model(x)
            pd_probs.append(pd_prob.item())
            log_freqs.append(freq_pred.item())
        all_pd_probs.append(float(np.mean(pd_probs)))
        all_log_freqs.append(float(np.mean(log_freqs)))

    pd_w = np.array(all_pd_probs)
    lf = np.array(all_log_freqs)
    ws = pd_w.sum()
    if ws > 1e-6:
        wlf = np.sum(pd_w * lf) / ws
    else:
        wlf = np.mean(lf)
    cnn_freq = float(np.clip(np.exp(wlf), 0.3, 3.5))

    # ACF freq
    b_lp, a_lp = butter(4, 20.0 / (fs / 2), btype='low')
    acf_freqs = []
    for i in range(seg_8ch.shape[0]):
        ch = seg_8ch[i]
        ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            sig = filtfilt(b_lp, a_lp, ch)
        except Exception:
            sig = ch
        f = estimate_frequency_acf(sig, fs)
        if np.isfinite(f):
            acf_freqs.append(f)
    acf_freq = float(np.clip(np.median(acf_freqs), 0.3, 3.5)) if acf_freqs else cnn_freq

    freq_estimate = 0.8 * cnn_freq + 0.2 * acf_freq
    return float(np.clip(freq_estimate, 0.3, 3.5))


# ── Per-hemisphere pipeline ────────────────────────────────────────────────

def run_hemi_pipeline(seg, hemi_indices, hemi_cet_models, cnn_models,
                      config, fs=FS):
    """Run the full pipeline on one hemisphere.

    Args:
        seg: (18, 2000) full EEG segment
        hemi_indices: LEFT_INDICES or RIGHT_INDICES
        hemi_cet_models: list of HemiCET models
        cnn_models: list of ChannelPDNetAttention models
        config: 1, 2, or 3 (evidence configuration)
        fs: sampling rate

    Returns:
        (global_times, freq_estimate)
    """
    n_samples = seg.shape[1]

    # Extract 8-channel hemisphere segment
    seg_8ch = np.zeros((8, n_samples), dtype=np.float32)
    for i, ch_idx in enumerate(hemi_indices):
        if ch_idx < seg.shape[0]:
            ch = seg[ch_idx].astype(np.float32)
            seg_8ch[i] = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)

    # Frequency estimation (same for all configs)
    freq_estimate = estimate_freq_hemi(seg_8ch, hemi_indices, cnn_models, fs)

    # Evidence depends on config
    if config == 1:
        # HemiCET evidence only
        evidence = compute_hemi_cet_evidence(seg_8ch, hemi_cet_models)

    elif config == 2:
        # max(HPP, HemiCET) with product-boost
        hpp_ev = compute_hpp_evidence_hemisphere(seg_8ch, fs)
        cet_ev = compute_hemi_cet_evidence(seg_8ch, hemi_cet_models)
        evidence = combine_evidence(hpp_ev, cet_ev)

    elif config == 3:
        # Same as Config 2 — product-boost with CET threshold + post-hoc filter
        hpp_ev = compute_hpp_evidence_hemisphere(seg_8ch, fs)
        cet_ev = compute_hemi_cet_evidence(seg_8ch, hemi_cet_models)
        evidence = combine_evidence(hpp_ev, cet_ev,
                                    cet_threshold_pct=80,
                                    boost_weight=3.0,
                                    cet_floor=0.3)
    else:
        raise ValueError(f"Unknown config: {config}")

    # DP pipeline
    active_start, active_end = detect_active_interval(evidence, fs)
    candidates = extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    discharge_samples = dp_best_sequence(candidates, evidence, fs, freq_estimate)

    # EM refine
    if len(discharge_samples) >= 3:
        discharge_samples = em_refine(evidence, discharge_samples, fs, freq_estimate)

    # Post-hoc filter (config 2 and 3)
    if config in (2, 3):
        discharge_samples = posthoc_filter(discharge_samples, evidence)

    global_times = (discharge_samples / fs).tolist() if len(discharge_samples) > 0 else []
    return global_times, freq_estimate


# ── Evaluation helper ──────────────────────────────────────────────────────

def evaluate(predictions, gt_cases, tolerance=TOLERANCE_S):
    """Compute F1, sensitivity, precision, freq Spearman."""
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs = [], []

    for pid, algo_times in predictions.items():
        if pid not in gt_cases:
            continue
        gt_times = sorted(gt_cases[pid]['global_times'])
        if len(gt_times) < 2:
            continue
        algo_times = sorted(algo_times)

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_dist:
                        best_dist, best_ai = d, ai
            if best_dist <= tolerance and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = np.diff(sorted(gt_times))
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        if len(algo_times) >= 2:
            algo_ipis = np.diff(sorted(algo_times))
            algo_freq = 1.0 / np.median(algo_ipis)
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = spearmanr(algo_freqs, gt_freqs)[0] if len(gt_freqs) >= 3 else float('nan')

    return {
        'f1': round(f1, 4),
        'sensitivity': round(sens, 4),
        'precision': round(prec, 4),
        'freq_spearman': round(freq_rho, 4) if np.isfinite(freq_rho) else None,
        'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
        'n_cases': len(predictions),
    }


def run_config(config_id, config_name, gt_cases, df, segments,
               hemi_cet_models, cnn_models):
    """Run one evaluation configuration across all GT cases."""
    print(f"\n{'='*70}")
    print(f"  Config {config_id}: {config_name}")
    print(f"{'='*70}")

    predictions = {}
    t0 = time.time()

    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2:
            continue

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue
        seg = pat_segs[0]
        if seg.shape[1] != N_SAMPLES:
            continue

        # Determine laterality
        row = df[df['patient_id'] == pid]
        if len(row) > 0:
            row = row.iloc[0]
            subtype = row['subtype']
            lat = row.get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'):
                lat = 'unknown'
        else:
            subtype = gt_data.get('subtype', 'unknown')
            lat = 'unknown'

        try:
            if subtype == 'gpd':
                # GPD: run both hemispheres, take more detections
                times_l, _ = run_hemi_pipeline(seg, LEFT_INDICES, hemi_cet_models,
                                                cnn_models, config=config_id)
                times_r, _ = run_hemi_pipeline(seg, RIGHT_INDICES, hemi_cet_models,
                                                cnn_models, config=config_id)
                predictions[pid] = times_l if len(times_l) >= len(times_r) else times_r
            elif lat == 'left':
                times, _ = run_hemi_pipeline(seg, LEFT_INDICES, hemi_cet_models,
                                              cnn_models, config=config_id)
                predictions[pid] = times
            elif lat == 'right':
                times, _ = run_hemi_pipeline(seg, RIGHT_INDICES, hemi_cet_models,
                                              cnn_models, config=config_id)
                predictions[pid] = times
            else:
                # Unknown: run both, pick more detections
                times_l, _ = run_hemi_pipeline(seg, LEFT_INDICES, hemi_cet_models,
                                                cnn_models, config=config_id)
                times_r, _ = run_hemi_pipeline(seg, RIGHT_INDICES, hemi_cet_models,
                                                cnn_models, config=config_id)
                predictions[pid] = times_l if len(times_l) >= len(times_r) else times_r
        except Exception as e:
            print(f"  Warning: {pid} failed — {e}")
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(gt_cases)} cases ({elapsed:.0f}s)")

    result = evaluate(predictions, gt_cases)
    elapsed = time.time() - t0
    print(f"\n  F1={result['f1']:.4f}  Sens={result['sensitivity']:.4f}  "
          f"Prec={result['precision']:.4f}  FreqRho={result['freq_spearman']}  "
          f"N={result['n_cases']}  ({elapsed:.0f}s)")
    return result, predictions


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  Experiment 5.3: HemiCET Evaluation")
    print("=" * 70)

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
    print(f"Ground truth cases: {len(gt_cases)}")

    # Load models
    print("\nLoading HemiCET models...")
    hemi_cet_models = load_hemi_cet_models()
    print(f"  Loaded {len(hemi_cet_models)} HemiCET fold models")

    print("Loading CNN+Attention models...")
    cnn_models = load_cnn_attn_models()
    print(f"  Loaded {len(cnn_models)} CNN+Attention fold models")

    # Run all three configs
    results = {}

    r1, _ = run_config(1, "HemiCET evidence only + DP",
                       gt_cases, df, segments, hemi_cet_models, cnn_models)
    results['config1_hemi_cet_only'] = r1

    r2, _ = run_config(2, "max(HPP, HemiCET) product-boost + DP",
                       gt_cases, df, segments, hemi_cet_models, cnn_models)
    results['config2_hpp_hemi_max'] = r2

    r3, _ = run_config(3, "Product-boost(HPP, HemiCET) + DP + all improvements",
                       gt_cases, df, segments, hemi_cet_models, cnn_models)
    results['config3_full'] = r3

    # Print comparison table
    print(f"\n{'='*70}")
    print("  EXPERIMENT 5.3 RESULTS — HemiCET vs Baselines")
    print(f"{'='*70}")
    print(f"\n  {'Method':<42s}  {'F1':>6s}  {'Sens':>6s}  {'Prec':>6s}  {'FrqRho':>7s}")
    print("  " + "-" * 70)

    rows = [
        ("Per-hemisphere baseline (exp0)",        0.672, None,  None,  None),
        ("Full 18ch pipeline (reference)",         0.740, None,  None,  None),
        ("Config 1: HemiCET only + DP",            r1['f1'], r1['sensitivity'], r1['precision'], r1['freq_spearman']),
        ("Config 2: max(HPP,HemiCET) + DP",        r2['f1'], r2['sensitivity'], r2['precision'], r2['freq_spearman']),
        ("Config 3: Product-boost + all improv",   r3['f1'], r3['sensitivity'], r3['precision'], r3['freq_spearman']),
    ]

    def fmt(v, d=4):
        if v is None:
            return '  N/A '
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:.{d}f}"
        return '  N/A '

    for name, f1, sens, prec, frho in rows:
        print(f"  {name:<42s}  {fmt(f1):>6s}  {fmt(sens):>6s}  {fmt(prec):>6s}  {fmt(frho):>7s}")

    print(f"\n  Note: baseline values are from previous experiments")
    print(f"\n  Discharge counts (new configs):")
    for cname, r in [("Config 1", r1), ("Config 2", r2), ("Config 3", r3)]:
        print(f"    {cname}: TP={r['tp']}, FN={r['fn']}, FP={r['fp']}")

    # Save results
    results['meta'] = {
        'total_time_s': round(time.time() - t0, 1),
        'n_gt_cases': len(gt_cases),
        'tolerance_s': TOLERANCE_S,
        'baselines': {
            'hemi_baseline_f1': 0.672,
            'full_18ch_f1': 0.740,
        },
    }
    save_path = SAVE_DIR / 'eval_hemi_cet_results.json'
    with open(str(save_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {save_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")
    print('=' * 70)


if __name__ == '__main__':
    main()
