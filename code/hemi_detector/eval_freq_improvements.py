"""
Evaluate frequency estimation improvements for HemiCET+DP.

Tests:
  F1: Current CNN+ACF freq → HemiCET+DP (baseline, F1=0.891)
  F2: Two-pass: CNN freq → HemiCET+DP → IPI freq → re-run DP
  F3: FFT of HemiCET evidence trace as freq estimate
  F4: ACF of HemiCET evidence trace as freq estimate
  F5: HemiCET evidence FFT + CNN ensemble
  F6: Two-pass with evidence-derived freq

Usage:
    conda run -n foe_dl python code/hemi_detector/eval_freq_improvements.py
"""

import sys, json, time, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.signal import butter, filtfilt, find_peaks
import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, estimate_frequency_acf,
)
from hemi_detector.hemi_cet import HemiCET

DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
TOLERANCE_S = 0.1

# Best C1 params
C1 = dict(dp_alpha=1.5, dp_beta=0.3, dp_lambda=0.05, peak_height_frac=0.05,
           max_skip=3, evidence_threshold_pct=50, min_evidence_ratio=0.4)


def load_models():
    detector = DischargeDetector()
    hcm = []
    for fold in range(5):
        p = PROJECT_DIR / f'data/hemi_cache/hemi_cet_v2/hemi_cet_fold{fold}.pt'
        if p.exists():
            m = HemiCET()
            m.load_state_dict(torch.load(str(p), map_location=DEVICE, weights_only=True))
            m.to(DEVICE); m.eval(); hcm.append(m)
    return detector, hcm


@torch.no_grad()
def get_hemicet_evidence(seg, indices, hcm):
    """Get HemiCET evidence trace for one hemisphere."""
    hs = seg[indices].astype(np.float32).copy()
    for i in range(len(indices)):
        mu, std = np.mean(hs[i]), np.std(hs[i])
        hs[i] = (hs[i] - mu) / std if std > 1e-8 else hs[i] - mu
    x = torch.from_numpy(hs[None]).to(DEVICE)
    preds = [m(x).squeeze().cpu().numpy() for m in hcm]
    return np.mean(preds, axis=0)


def cnn_freq_hemi(seg, indices, detector):
    """CNN+ACF freq for hemisphere channels."""
    all_pd, all_lf = [], []
    for ci in indices:
        ch = seg[ci].astype(np.float32).copy()
        if not np.all(np.isfinite(ch)):
            all_pd.append(0.0); all_lf.append(0.0); continue
        mu, std = np.mean(ch), np.std(ch)
        ch = (ch - mu) / std if std > 1e-8 else ch - mu
        x = torch.from_numpy(ch[None, None, :]).to(DEVICE)
        pp, lf = [], []
        for m in detector.cnn_models:
            p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
        all_pd.append(np.mean(pp)); all_lf.append(np.mean(lf))
    pdw = np.array(all_pd); lfs = np.array(all_lf); ws = pdw.sum()
    cnn = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
    b, a = butter(4, 20.0 / (FS / 2), btype='low')
    acfs = []
    for ci in indices:
        try: sig = filtfilt(b, a, seg[ci])
        except: sig = seg[ci]
        f2 = estimate_frequency_acf(sig, FS)
        if np.isfinite(f2): acfs.append(f2)
    acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn
    return float(np.clip(0.8 * cnn + 0.2 * acf, 0.3, 3.5))


def evidence_fft_freq(evidence, fs=FS):
    """Estimate frequency from FFT of evidence trace."""
    # The evidence trace has peaks at discharge times
    # Its FFT should have a peak at the discharge frequency
    ev = evidence - np.mean(evidence)
    n = len(ev)
    fft = np.abs(np.fft.rfft(ev))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    # Only look in 0.3-5 Hz range
    mask = (freqs >= 0.3) & (freqs <= 5.0)
    if not np.any(mask):
        return 1.0
    fft_masked = fft[mask]
    freqs_masked = freqs[mask]
    peak_idx = np.argmax(fft_masked)
    return float(freqs_masked[peak_idx])


def evidence_acf_freq(evidence, fs=FS):
    """Estimate frequency from ACF of evidence trace."""
    ev = evidence - np.mean(evidence)
    n = len(ev)
    acf = np.correlate(ev, ev, mode='full')[n-1:]
    if acf[0] > 0:
        acf = acf / acf[0]
    min_lag = int(0.2 * fs)  # max 5 Hz
    max_lag = min(int(3.0 * fs), len(acf) - 1)  # min 0.33 Hz
    if max_lag <= min_lag:
        return 1.0
    segment = acf[min_lag:max_lag + 1]
    peaks, _ = find_peaks(segment, height=0.1)
    if len(peaks) == 0:
        return 1.0
    best_lag = peaks[0] + min_lag
    return float(fs / best_lag)


def run_dp_with_freq(evidence, freq, fs=FS):
    """Run DP pipeline with given frequency and C1 params."""
    if C1['evidence_threshold_pct'] > 0 and np.any(evidence > 0):
        thr = np.percentile(evidence[evidence > 0], C1['evidence_threshold_pct'])
        evidence = np.where(evidence > thr, evidence, 0)

    active_start, active_end = detect_active_interval(evidence, fs)
    T = 1.0 / freq if freq > 0 else 1.0
    segment = evidence[active_start:active_end + 1]
    if len(segment) < 3:
        return np.array([])

    min_dist = max(20, int(0.2 * T * fs))
    min_height = C1['peak_height_frac'] * np.max(segment)
    peaks, _ = find_peaks(segment, height=min_height, distance=min_dist)
    strong_height = 0.5 * np.max(segment)
    strong_peaks, _ = find_peaks(segment, height=strong_height,
                                  distance=max(10, int(0.1 * T * fs)))
    candidates = np.unique(np.concatenate([peaks, strong_peaks])) + active_start

    if len(candidates) == 0:
        return np.array([])
    if len(candidates) == 1:
        return candidates

    n = len(candidates)
    raw_scores = np.array([evidence[c] for c in candidates])
    node_scores = raw_scores ** 1.5
    best_score = np.full(n, -np.inf)
    best_prev = np.full(n, -1, dtype=int)
    for i in range(n):
        best_score[i] = node_scores[i] - C1['dp_lambda']
    for j in range(1, n):
        for i in range(j):
            dt = (candidates[j] - candidates[i]) / fs
            if dt <= 0 or dt > 4 * T: continue
            best_edge = -np.inf
            for m in range(1, C1['max_skip'] + 1):
                deviation = (dt - m * T) / (m * T)
                interval_score = -C1['dp_alpha'] * deviation ** 2
                skip_penalty = -C1['dp_beta'] * (m - 1)
                edge = interval_score + skip_penalty
                if edge > best_edge: best_edge = edge
            total = best_score[i] + best_edge + node_scores[j] - C1['dp_lambda']
            if total > best_score[j]:
                best_score[j] = total
                best_prev[j] = i
    path = []
    idx = int(np.argmax(best_score))
    while idx >= 0:
        path.append(idx)
        idx = best_prev[idx]
    path.reverse()
    ds = candidates[np.array(path)]

    if len(ds) >= 3:
        ds = em_refine(evidence, ds, fs, freq)

    if C1['min_evidence_ratio'] > 0 and len(ds) >= 2:
        peak_vals = np.array([evidence[int(s)] for s in ds])
        threshold = C1['min_evidence_ratio'] * np.median(peak_vals)
        ds = ds[peak_vals >= threshold]

    return ds


def evaluate(predictions, gt_cases):
    total_tp = total_fn = total_fp = 0
    gt_freqs, algo_freqs, match_errors = [], [], []
    for pid, at in predictions.items():
        if pid not in gt_cases: continue
        gt = sorted(gt_cases[pid]['global_times'])
        if len(gt) < 2: continue
        at = sorted(at)
        gm = [False]*len(gt); am = [False]*len(at)
        for gi, g in enumerate(gt):
            bd, ba = np.inf, -1
            for ai, a in enumerate(at):
                if not am[ai]:
                    d = abs(g-a)
                    if d < bd: bd, ba = d, ai
            if bd <= TOLERANCE_S and ba >= 0:
                gm[gi] = True; am[ba] = True; match_errors.append(bd)
        total_tp += sum(gm); total_fn += len(gt)-sum(gm); total_fp += len(at)-sum(am)
        gf = 1/np.median(np.diff(gt))
        af = 1/np.median(np.diff(at)) if len(at) >= 2 else np.nan
        if np.isfinite(gf) and np.isfinite(af):
            gt_freqs.append(gf); algo_freqs.append(af)
    s = total_tp/(total_tp+total_fn) if total_tp+total_fn > 0 else 0
    p = total_tp/(total_tp+total_fp) if total_tp+total_fp > 0 else 0
    f1 = 2*p*s/(p+s) if p+s > 0 else 0
    rho = spearmanr(gt_freqs, algo_freqs)[0] if len(gt_freqs) >= 3 else 0
    mae = np.mean(np.abs(np.array(gt_freqs)-np.array(algo_freqs))) if gt_freqs else 0
    tmed = np.median(match_errors)*1000 if match_errors else 0
    return dict(f1=round(f1,4), sens=round(s,4), prec=round(p,4),
                freq_rho=round(rho,4), freq_mae=round(mae,3), timing_med=round(tmed,1),
                n=len(predictions))


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Frequency Estimation Improvements for HemiCET+DP")
    print("=" * 70)

    dataset = load_dataset(verbose=False)
    df = dataset['df']; segments = dataset['segments']
    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        dt = json.load(f)
    gt_cases = {}
    for pid, v in dt.items():
        if v.get('review_status') != 'ground_truth' or len(v.get('global_times', [])) < 2: continue
        row = df[df['patient_id'] == pid]
        if len(row) == 0: continue
        v['subtype'] = row.iloc[0]['subtype']
        v['laterality'] = row.iloc[0].get('laterality', '')
        if not isinstance(v['laterality'], str) or v['laterality'] not in ('left', 'right'):
            v['laterality'] = None
        gt_cases[pid] = v
    print(f"GT cases: {len(gt_cases)}")

    detector, hcm = load_models()
    print(f"Models: {len(hcm)} HemiCET, {len(detector.cnn_models)} CNN")

    results = {}

    for exp_name, freq_method in [
        ('F1: CNN+ACF (baseline)', 'cnn_acf'),
        ('F2: Two-pass (CNN→IPI→re-DP)', 'two_pass'),
        ('F3: Evidence FFT', 'ev_fft'),
        ('F4: Evidence ACF', 'ev_acf'),
        ('F5: Evidence ACF + CNN ensemble', 'ev_acf_cnn'),
        ('F6: Two-pass with ev_acf freq', 'two_pass_ev'),
    ]:
        print(f"\n--- {exp_name} ---")
        preds = {}
        freq_estimates = []

        for i, (pid, gt) in enumerate(gt_cases.items()):
            pat_segs = segments.get(pid, [])
            if not pat_segs: continue
            seg = pat_segs[0]
            st = gt['subtype']; lat = gt.get('laterality')

            def _run(indices):
                ev = get_hemicet_evidence(seg, indices, hcm)

                if freq_method == 'cnn_acf':
                    freq = cnn_freq_hemi(seg, indices, detector)
                elif freq_method == 'ev_fft':
                    freq = evidence_fft_freq(ev)
                elif freq_method == 'ev_acf':
                    freq = evidence_acf_freq(ev)
                elif freq_method == 'ev_acf_cnn':
                    ev_freq = evidence_acf_freq(ev)
                    cnn_freq = cnn_freq_hemi(seg, indices, detector)
                    freq = 0.5 * ev_freq + 0.5 * cnn_freq
                elif freq_method == 'two_pass':
                    # First pass with CNN freq
                    freq1 = cnn_freq_hemi(seg, indices, detector)
                    ds1 = run_dp_with_freq(ev.copy(), freq1)
                    if len(ds1) >= 2:
                        ipi_freq = FS / np.median(np.diff(ds1))
                        freq = float(np.clip(ipi_freq, 0.3, 5.0))
                    else:
                        freq = freq1
                elif freq_method == 'two_pass_ev':
                    # First pass with evidence ACF freq
                    freq1 = evidence_acf_freq(ev)
                    ds1 = run_dp_with_freq(ev.copy(), freq1)
                    if len(ds1) >= 2:
                        ipi_freq = FS / np.median(np.diff(ds1))
                        freq = float(np.clip(ipi_freq, 0.3, 5.0))
                    else:
                        freq = freq1
                else:
                    freq = 1.0

                freq = float(np.clip(freq, 0.3, 5.0))
                freq_estimates.append(freq)
                ds = run_dp_with_freq(ev, freq)
                return (ds / FS).tolist() if len(ds) > 0 else []

            try:
                if st == 'gpd' or lat not in ('left', 'right'):
                    tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
                    preds[pid] = tl if len(tl) >= len(tr) else tr
                else:
                    preds[pid] = _run(LEFT_INDICES if lat == 'left' else RIGHT_INDICES)
            except:
                pass

            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(gt_cases)}")

        r = evaluate(preds, gt_cases)
        results[exp_name] = r
        print(f"  F1={r['f1']} Sens={r['sens']} Prec={r['prec']} "
              f"FrqRho={r['freq_rho']} FrqMAE={r['freq_mae']} TmgMed={r['timing_med']}ms N={r['n']}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  FREQUENCY IMPROVEMENT RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Experiment':<40s} {'F1':>7s} {'FrqRho':>8s} {'FrqMAE':>8s}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<40s} {r['f1']:>7.4f} {r['freq_rho']:>8.4f} {r['freq_mae']:>8.3f}")

    # Save
    save_path = PROJECT_DIR / 'data' / 'hemi_cache' / 'optimization' / 'freq_improvements.json'
    with open(str(save_path), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
