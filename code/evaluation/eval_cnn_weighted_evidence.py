"""Evaluate CNN-weighted evidence aggregation vs baseline median.

Compares:
1. Baseline: HemiCET+DP with median aggregation (current method)
2. CNN-weighted: HemiCET+DP with ChannelPDNetAttention-weighted aggregation

Uses the 593 patients with ground truth discharge times.
"""
import sys, time, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from discharge_detector import (
    DischargeDetector, FS, LEFT_INDICES, RIGHT_INDICES,
    compute_channel_evidence, combine_evidence,
    detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, per_channel_times,
)
from pd_channel_detector.channel_cnn import ChannelPDNetAttention

PROJECT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'


def load_cnn_channel_models(device='cpu'):
    """Load ChannelPDNetAttention 5-fold ensemble."""
    model_dir = DATA_DIR / 'pd_channel_cache'
    models = []
    for fold in range(5):
        path = model_dir / f'cnn_attn_fold{fold}.pt'
        if path.exists():
            m = ChannelPDNetAttention()
            m.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
            m.to(device)
            m.eval()
            models.append(m)
    return models


def get_channel_pd_probs(segment_18ch, cnn_models, device='cpu'):
    """Get per-channel PD probability from CNN ensemble."""
    n_ch = min(18, segment_18ch.shape[0])
    probs = np.zeros(n_ch)
    for ch in range(n_ch):
        ch_sig = segment_18ch[ch:ch+1, :].astype(np.float32)
        std = np.std(ch_sig)
        if std > 1e-8:
            ch_sig = (ch_sig - np.mean(ch_sig)) / std
        x = torch.tensor(ch_sig[np.newaxis, :, :], dtype=torch.float32).to(device)
        fold_probs = []
        with torch.no_grad():
            for m in cnn_models:
                out = m(x)
                if isinstance(out, tuple):
                    p = torch.sigmoid(out[0]).item()
                else:
                    p = torch.sigmoid(out).item()
                fold_probs.append(p)
        probs[ch] = np.mean(fold_probs)
    return probs


def aggregate_evidence_weighted(evidence_all, weights, subtype, laterality=None):
    """Weighted aggregation using CNN channel probabilities."""
    if subtype == 'gpd':
        ch_idx = np.arange(min(18, evidence_all.shape[0]))
    elif laterality == 'left':
        ch_idx = LEFT_INDICES
    elif laterality == 'right':
        ch_idx = RIGHT_INDICES
    else:
        # LPD without laterality: weight both sides, take max
        left_w = weights[LEFT_INDICES]
        right_w = weights[RIGHT_INDICES]
        left_w_sum = left_w.sum()
        right_w_sum = right_w.sum()
        if left_w_sum > 0:
            left_agg = np.average(evidence_all[LEFT_INDICES], weights=left_w, axis=0)
        else:
            left_agg = np.median(evidence_all[LEFT_INDICES], axis=0)
        if right_w_sum > 0:
            right_agg = np.average(evidence_all[RIGHT_INDICES], weights=right_w, axis=0)
        else:
            right_agg = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_agg, right_agg)

    w = weights[ch_idx]
    w_sum = w.sum()
    if w_sum > 0:
        return np.average(evidence_all[ch_idx], weights=w, axis=0)
    else:
        return np.median(evidence_all[ch_idx], axis=0)


def run_detection(detector, segment, subtype, laterality, cnn_models=None, use_weighted=False):
    """Run detection with either baseline or CNN-weighted aggregation."""
    n_ch = min(segment.shape[0], 18)
    n_samp = segment.shape[1]

    # Frequency estimate
    cnn_freq = detector.estimate_frequency(segment)
    acf_freq = detector.estimate_frequency_acf_multichannel(segment, subtype, laterality)
    if np.isfinite(acf_freq):
        freq_est = 0.8 * cnn_freq + 0.2 * acf_freq
    else:
        freq_est = cnn_freq
    freq_est = float(np.clip(freq_est, 0.3, 3.5))

    # Per-channel evidence
    hpp_all = np.zeros((n_ch, n_samp))
    cet_all = np.zeros((n_ch, n_samp), dtype=np.float32)
    for ch in range(n_ch):
        hpp_all[ch] = compute_channel_evidence(segment[ch], FS)
        if np.all(np.isfinite(segment[ch])):
            cet_all[ch] = detector.compute_cet_evidence_channel(segment[ch])

    # Aggregation
    if use_weighted and cnn_models:
        weights = get_channel_pd_probs(segment, cnn_models)
        # Ensure minimum weight so no channel is completely silenced
        weights = np.clip(weights, 0.05, None)
        hpp_agg = aggregate_evidence_weighted(hpp_all, weights, subtype, laterality)
        cet_agg = aggregate_evidence_weighted(cet_all, weights, subtype, laterality)
    else:
        from discharge_detector import aggregate_evidence
        hpp_agg = aggregate_evidence(hpp_all, subtype, laterality)
        cet_agg = aggregate_evidence(cet_all, subtype, laterality)

    evidence = combine_evidence(hpp_agg, cet_agg)

    # DP inference
    active_start, active_end = detect_active_interval(evidence, FS)
    candidates = extract_candidates(evidence, FS, freq_est, active_start, active_end)
    discharge_samples = dp_best_sequence(candidates, evidence, FS, freq_est)
    if len(discharge_samples) >= 3:
        discharge_samples = em_refine(evidence, discharge_samples, FS, freq_est)
    discharge_samples = posthoc_filter(discharge_samples, evidence)

    times = (discharge_samples / FS).tolist() if len(discharge_samples) > 0 else []
    return times


def compute_f1(pred_times, gold_times, tolerance_s=0.1):
    """Compute F1 between predicted and gold discharge times."""
    if len(pred_times) == 0 and len(gold_times) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_times) == 0 or len(gold_times) == 0:
        return 0.0, 0.0, 0.0

    pred = np.array(pred_times)
    gold = np.array(gold_times)

    # Precision: fraction of predictions matched to a gold
    matched_pred = 0
    gold_used = set()
    for p in pred:
        dists = np.abs(gold - p)
        best = np.argmin(dists)
        if dists[best] <= tolerance_s and best not in gold_used:
            matched_pred += 1
            gold_used.add(best)
    precision = matched_pred / len(pred) if len(pred) > 0 else 0

    # Recall: fraction of golds matched to a prediction
    matched_gold = 0
    pred_used = set()
    for g in gold:
        dists = np.abs(pred - g)
        best = np.argmin(dists)
        if dists[best] <= tolerance_s and best not in pred_used:
            matched_gold += 1
            pred_used.add(best)
    recall = matched_gold / len(gold) if len(gold) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def main():
    print("=" * 70)
    print("  CNN-Weighted Evidence Aggregation Evaluation")
    print("=" * 70)

    # Load ground truth
    with open(str(LABELS_DIR / 'discharge_times.json')) as f:
        gt = json.load(f)
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    print(f"Ground truth: {len(gt)} patients with discharge times")

    # Load models
    device = torch.device('cpu')  # MPS produces corrupted results for this model
    print(f"Device: {device}")

    detector = DischargeDetector()
    cnn_models = load_cnn_channel_models(device)
    print(f"Loaded {len(cnn_models)} ChannelPDNetAttention models")

    # Evaluate both methods
    import scipy.io as sio

    results = {'baseline': [], 'weighted': []}
    n_eval = 0

    for pid, gt_entry in gt.items():
        gold_times = gt_entry.get('times', gt_entry.get('global_times', []))
        if len(gold_times) < 2:
            continue

        row = pat[pat['patient_id'] == pid]
        if len(row) == 0:
            continue
        subtype = row.iloc[0]['subtype']
        laterality = row.iloc[0].get('laterality', None)
        if pd.isna(laterality):
            laterality = None

        # Find EEG
        mat_file = None
        for _, sr in seg_df[seg_df['patient_id'] == pid].iterrows():
            if (EEG_DIR / sr['mat_file']).exists():
                mat_file = sr['mat_file']
                break
        if not mat_file:
            for sx in ['_seg000.mat', '.mat']:
                if (EEG_DIR / f'{pid}{sx}').exists():
                    mat_file = f'{pid}{sx}'
                    break
        if not mat_file:
            continue

        try:
            mat = sio.loadmat(str(EEG_DIR / mat_file))
            dk = [k for k in mat if not k.startswith('_')][0]
            seg = mat[dk].astype(np.float64)
            if seg.shape[0] > seg.shape[1]:
                seg = seg.T
            if seg.shape[0] < 18:
                continue
            seg = seg[:18, :2000]
        except:
            continue

        # Baseline
        try:
            pred_base = run_detection(detector, seg, subtype, laterality,
                                       use_weighted=False)
            _, _, f1_base = compute_f1(pred_base, gold_times)
            results['baseline'].append(f1_base)
        except:
            results['baseline'].append(0.0)

        # CNN-weighted
        try:
            pred_weighted = run_detection(detector, seg, subtype, laterality,
                                           cnn_models=cnn_models, use_weighted=True)
            _, _, f1_weighted = compute_f1(pred_weighted, gold_times)
            results['weighted'].append(f1_weighted)
        except:
            results['weighted'].append(0.0)

        n_eval += 1
        if n_eval % 100 == 0:
            base_mean = np.mean(results['baseline'])
            wt_mean = np.mean(results['weighted'])
            print(f"  {n_eval} patients: baseline F1={base_mean:.4f}, weighted F1={wt_mean:.4f}")

    print(f"\n{'='*70}")
    print(f"  RESULTS ({n_eval} patients)")
    print(f"{'='*70}")

    base_f1s = np.array(results['baseline'])
    wt_f1s = np.array(results['weighted'])

    print(f"  Baseline (median agg):      F1 = {np.mean(base_f1s):.4f} ± {np.std(base_f1s):.4f}")
    print(f"  CNN-weighted agg:           F1 = {np.mean(wt_f1s):.4f} ± {np.std(wt_f1s):.4f}")
    print(f"  Improvement:                ΔF1 = {np.mean(wt_f1s) - np.mean(base_f1s):+.4f}")

    # Per-subtype
    # Need to track subtypes
    print(f"\n  Wins: weighted better in {(wt_f1s > base_f1s).sum()}/{n_eval} cases")
    print(f"  Ties: {(wt_f1s == base_f1s).sum()}/{n_eval}")
    print(f"  Losses: {(wt_f1s < base_f1s).sum()}/{n_eval}")

    # Paired test
    from scipy.stats import wilcoxon
    diff = wt_f1s - base_f1s
    nonzero = diff[diff != 0]
    if len(nonzero) > 10:
        stat, pval = wilcoxon(nonzero)
        print(f"\n  Wilcoxon signed-rank test: p = {pval:.4f}")


if __name__ == '__main__':
    main()
