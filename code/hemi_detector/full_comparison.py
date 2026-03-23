"""Full comparison of all methods with F1, freq Spearman, freq MAE, timing MAE."""

import sys, json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, combine_evidence, detect_active_interval,
    extract_candidates, dp_best_sequence, em_refine, posthoc_filter,
    compute_channel_evidence, estimate_frequency_acf,
)
from scipy.signal import butter, filtfilt

TOLERANCE_S = 0.1


def evaluate_predictions(predictions, gt_cases):
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs, match_errors = [], [], []

    for pid, algo_times in predictions.items():
        if pid not in gt_cases: continue
        gt_times = sorted(gt_cases[pid]['global_times'])
        if len(gt_times) < 2: continue
        algo_times = sorted(algo_times)

        gt_matched = [False] * len(gt_times)
        algo_matched = [False] * len(algo_times)
        for gi, gt in enumerate(gt_times):
            best_dist, best_ai = np.inf, -1
            for ai, at in enumerate(algo_times):
                if not algo_matched[ai]:
                    d = abs(gt - at)
                    if d < best_dist: best_dist, best_ai = d, ai
            if best_dist <= TOLERANCE_S and best_ai >= 0:
                gt_matched[gi] = True
                algo_matched[best_ai] = True
                match_errors.append(best_dist)

        total_tp += sum(gt_matched)
        total_fn += len(gt_times) - sum(gt_matched)
        total_fp += len(algo_times) - sum(algo_matched)

        gt_ipis = np.diff(gt_times)
        gt_freq = 1.0 / np.median(gt_ipis) if len(gt_ipis) > 0 else np.nan
        if len(algo_times) >= 2:
            algo_freq = 1.0 / np.median(np.diff(algo_times))
        else:
            algo_freq = np.nan
        if np.isfinite(gt_freq) and np.isfinite(algo_freq):
            gt_freqs.append(gt_freq)
            algo_freqs.append(algo_freq)

    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
    freq_rho = spearmanr(algo_freqs, gt_freqs)[0] if len(gt_freqs) >= 3 else float('nan')
    freq_mae = np.mean(np.abs(np.array(gt_freqs) - np.array(algo_freqs))) if len(gt_freqs) >= 3 else float('nan')
    timing_mae = np.mean(match_errors) * 1000 if match_errors else float('nan')
    timing_median = np.median(match_errors) * 1000 if match_errors else float('nan')

    return dict(f1=round(f1, 4), sens=round(sens, 4), prec=round(prec, 4),
                freq_rho=round(freq_rho, 4) if np.isfinite(freq_rho) else None,
                freq_mae=round(freq_mae, 3) if np.isfinite(freq_mae) else None,
                timing_mae_ms=round(timing_mae, 1) if np.isfinite(timing_mae) else None,
                timing_median_ms=round(timing_median, 1) if np.isfinite(timing_median) else None,
                n_cases=len(predictions), tp=total_tp, fn=total_fn, fp=total_fp)


def hemi_freq(seg, hemi_indices, detector):
    all_pd, all_lf = [], []
    for ch_idx in hemi_indices:
        ch = seg[ch_idx].astype(np.float32).copy()
        if not np.all(np.isfinite(ch)): all_pd.append(0.0); all_lf.append(0.0); continue
        mu, std = np.mean(ch), np.std(ch)
        ch = (ch - mu) / std if std > 1e-8 else ch - mu
        x = torch.from_numpy(ch[None, None, :]).to(detector.device)
        pp, lf = [], []
        for m in detector.cnn_models:
            p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
        all_pd.append(np.mean(pp)); all_lf.append(np.mean(lf))
    pdw = np.array(all_pd); lfs = np.array(all_lf); ws = pdw.sum()
    cnn = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
    b, a = butter(4, 20.0 / (FS / 2), btype='low')
    acfs = []
    for ci in hemi_indices:
        try: sig = filtfilt(b, a, seg[ci])
        except: sig = seg[ci]
        f = estimate_frequency_acf(sig, FS)
        if np.isfinite(f): acfs.append(f)
    acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn
    return float(np.clip(0.8 * cnn + 0.2 * acf, 0.3, 3.5))


def run_hemi_baseline(seg, st, lat, detector):
    """Per-hemisphere baseline using existing per-channel CET + HPP."""
    def _run(indices):
        freq = hemi_freq(seg, indices, detector)
        n_ch = len(indices); n_s = seg.shape[1]
        hpp = np.zeros((n_ch, n_s)); cet = np.zeros((n_ch, n_s), dtype=np.float32)
        for i, ci in enumerate(indices):
            hpp[i] = compute_channel_evidence(seg[ci], FS)
            if np.all(np.isfinite(seg[ci])): cet[i] = detector.compute_cet_evidence_channel(seg[ci])
        ev = combine_evidence(np.median(hpp, axis=0), np.median(cet, axis=0))
        a0, a1 = detect_active_interval(ev, FS)
        cands = extract_candidates(ev, FS, freq, a0, a1)
        ds = dp_best_sequence(cands, ev, FS, freq)
        if len(ds) >= 3: ds = em_refine(ev, ds, FS, freq)
        ds = posthoc_filter(ds, ev)
        return (ds / FS).tolist() if len(ds) > 0 else []

    if st == 'gpd' or lat not in ('left', 'right'):
        tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
        return tl if len(tl) >= len(tr) else tr
    return _run(LEFT_INDICES if lat == 'left' else RIGHT_INDICES)


@torch.no_grad()
def run_hemi_cet(seg, st, lat, detector, hemi_cet_models):
    """HemiCET + DP."""
    def _run(indices):
        freq = hemi_freq(seg, indices, detector)
        hs = seg[indices].astype(np.float32).copy()
        for i in range(len(indices)):
            mu, std = np.mean(hs[i]), np.std(hs[i])
            hs[i] = (hs[i] - mu) / std if std > 1e-8 else hs[i] - mu
        x = torch.from_numpy(hs[None]).to(detector.device)
        preds = [m(x).squeeze().cpu().numpy() for m in hemi_cet_models]
        ev = np.mean(preds, axis=0)
        a0, a1 = detect_active_interval(ev, FS)
        cands = extract_candidates(ev, FS, freq, a0, a1)
        ds = dp_best_sequence(cands, ev, FS, freq)
        if len(ds) >= 3: ds = em_refine(ev, ds, FS, freq)
        ds = posthoc_filter(ds, ev)
        return (ds / FS).tolist() if len(ds) > 0 else []

    if st == 'gpd' or lat not in ('left', 'right'):
        tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
        return tl if len(tl) >= len(tr) else tr
    return _run(LEFT_INDICES if lat == 'left' else RIGHT_INDICES)


def main():
    print("Loading data...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']; segments = dataset['segments']

    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        dt = json.load(f)
    gt_cases = {pid: v for pid, v in dt.items() if v.get('review_status') == 'ground_truth'}
    print(f"GT cases: {len(gt_cases)}")

    detector = DischargeDetector()

    # Load HemiCET
    from hemi_detector.hemi_cet import HemiCET
    hcm = []
    for fold in range(5):
        p = PROJECT_DIR / f'data/hemi_cache/hemi_cet/hemi_cet_fold{fold}.pt'
        if p.exists():
            m = HemiCET()
            m.load_state_dict(torch.load(str(p), map_location=detector.device, weights_only=True))
            m.to(detector.device); m.eval(); hcm.append(m)
    print(f"HemiCET models: {len(hcm)}")

    methods = {}

    for method_name, method_fn in [
        ('Full 18ch pipeline', lambda seg, st, lat: detector.detect(seg, subtype=st, laterality=lat)['global_times']),
        ('Per-hemi baseline', lambda seg, st, lat: run_hemi_baseline(seg, st, lat, detector)),
        ('HemiCET + DP', lambda seg, st, lat: run_hemi_cet(seg, st, lat, detector, hcm)),
    ]:
        print(f"\n--- {method_name} ---")
        preds = {}
        for i, (pid, gt) in enumerate(gt_cases.items()):
            if len(gt.get('global_times', [])) < 2: continue
            row = df[df['patient_id'] == pid]
            pat_segs = segments.get(pid, [])
            if not pat_segs or len(row) == 0: continue
            row = row.iloc[0]; seg = pat_segs[0]
            st = row['subtype']; lat = row.get('laterality', '')
            if not isinstance(lat, str) or lat not in ('left', 'right'): lat = None
            try:
                preds[pid] = method_fn(seg, st, lat)
            except: pass
            if (i + 1) % 100 == 0: print(f"  {i+1}/{len(gt_cases)}")
        methods[method_name] = evaluate_predictions(preds, gt_cases)
        r = methods[method_name]
        print(f"  F1={r['f1']:.3f} Sens={r['sens']:.3f} Prec={r['prec']:.3f} "
              f"FrqRho={r['freq_rho']} FrqMAE={r['freq_mae']} TmgMAE={r['timing_mae_ms']}ms N={r['n_cases']}")

    print(f"\n{'='*95}")
    print(f"  FULL COMPARISON TABLE")
    print(f"{'='*95}")
    h = f"{'Method':<25s} {'F1':>6s} {'Sens':>6s} {'Prec':>6s} {'FrqRho':>7s} {'FrqMAE':>8s} {'TmgMAE':>8s} {'TmgMed':>8s}"
    print(f"\n{h}")
    print("-" * 95)
    for name, r in methods.items():
        rho = f"{r['freq_rho']:.3f}" if r['freq_rho'] else "N/A"
        fm = f"{r['freq_mae']:.3f}" if r['freq_mae'] else "N/A"
        tm = f"{r['timing_mae_ms']:.1f}ms" if r['timing_mae_ms'] else "N/A"
        tmed = f"{r['timing_median_ms']:.1f}ms" if r['timing_median_ms'] else "N/A"
        print(f"{name:<25s} {r['f1']:>6.3f} {r['sens']:>6.3f} {r['prec']:>6.3f} {rho:>7s} {fm:>8s} {tm:>8s} {tmed:>8s}")

    with open(str(PROJECT_DIR / 'data/hemi_cache/full_comparison.json'), 'w') as f:
        json.dump(methods, f, indent=2)
    print(f"\nSaved to data/hemi_cache/full_comparison.json")


if __name__ == '__main__':
    main()
