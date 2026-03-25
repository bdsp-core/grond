"""Generate frequency scatterplot: GT vs HemiCET+DP estimates."""

import sys, json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from scipy.signal import butter, filtfilt
import torch
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from discharge_detector import (
    DischargeDetector, detect_active_interval, extract_candidates,
    dp_best_sequence, em_refine, posthoc_filter, estimate_frequency_acf,
)
from hemi_detector.hemi_cet import HemiCET

TOLERANCE_S = 0.1


def main():
    dataset = load_dataset(verbose=False)
    df = dataset['df']; segments = dataset['segments']

    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        dt = json.load(f)
    gt_cases = {pid: v for pid, v in dt.items()
                if v.get('review_status') == 'ground_truth' and len(v.get('global_times', [])) >= 2}

    detector = DischargeDetector()
    hcm = []
    for fold in range(5):
        p = PROJECT_DIR / f'data/hemi_cache/hemi_cet/hemi_cet_fold{fold}.pt'
        if p.exists():
            m = HemiCET()
            m.load_state_dict(torch.load(str(p), map_location=detector.device, weights_only=True))
            m.to(detector.device); m.eval(); hcm.append(m)

    gt_freqs, algo_freqs, pids_out = [], [], []

    print(f"Computing frequencies for {len(gt_cases)} cases...")
    for i, (pid, gt_data) in enumerate(gt_cases.items()):
        gt_times = sorted(gt_data['global_times'])
        if len(gt_times) < 2: continue
        row = df[df['patient_id'] == pid]
        pat_segs = segments.get(pid, [])
        if not pat_segs or len(row) == 0: continue
        row = row.iloc[0]; seg = pat_segs[0]
        st = row['subtype']; lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'): lat = None

        gt_freq = 1.0 / np.median(np.diff(gt_times))

        # Run HemiCET
        try:
            def _run(indices):
                all_pd, all_lf = [], []
                with torch.no_grad():
                    for ci in indices:
                        ch = seg[ci].astype(np.float32).copy()
                        if not np.all(np.isfinite(ch)):
                            all_pd.append(0.0); all_lf.append(0.0); continue
                        mu, std = np.mean(ch), np.std(ch)
                        ch = (ch - mu) / std if std > 1e-8 else ch - mu
                        x = torch.from_numpy(ch[None, None, :]).to(detector.device)
                        pp, lf = [], []
                        for m in detector.cnn_models:
                            p, f, _ = m(x); pp.append(p.item()); lf.append(f.item())
                        all_pd.append(np.mean(pp)); all_lf.append(np.mean(lf))
                pdw = np.array(all_pd); lfs = np.array(all_lf); ws = pdw.sum()
                cnn_freq = float(np.clip(np.exp(np.sum(pdw * lfs) / ws if ws > 1e-6 else np.mean(lfs)), 0.3, 3.5))
                b, a = butter(4, 20.0 / (FS / 2), btype='low')
                acfs = []
                for ci in indices:
                    try: sig2 = filtfilt(b, a, seg[ci])
                    except: sig2 = seg[ci]
                    f2 = estimate_frequency_acf(sig2, FS)
                    if np.isfinite(f2): acfs.append(f2)
                acf = float(np.clip(np.median(acfs), 0.3, 3.5)) if acfs else cnn_freq
                freq = float(np.clip(0.8 * cnn_freq + 0.2 * acf, 0.3, 3.5))
                hs = seg[indices].astype(np.float32).copy()
                for j in range(len(indices)):
                    mu2, std2 = np.mean(hs[j]), np.std(hs[j])
                    hs[j] = (hs[j] - mu2) / std2 if std2 > 1e-8 else hs[j] - mu2
                with torch.no_grad():
                    x2 = torch.from_numpy(hs[None]).to(detector.device)
                    preds = [m2(x2).squeeze().cpu().numpy() for m2 in hcm]
                ev = np.mean(preds, axis=0)
                a0, a1 = detect_active_interval(ev, FS)
                cands = extract_candidates(ev, FS, freq, a0, a1)
                ds = dp_best_sequence(cands, ev, FS, freq)
                if len(ds) >= 3: ds = em_refine(ev, ds, FS, freq)
                ds = posthoc_filter(ds, ev)
                return (ds / FS).tolist() if len(ds) > 0 else []

            if st == 'gpd' or lat not in ('left', 'right'):
                tl, tr = _run(LEFT_INDICES), _run(RIGHT_INDICES)
                algo_times = tl if len(tl) >= len(tr) else tr
            else:
                algo_times = _run(LEFT_INDICES if lat == 'left' else RIGHT_INDICES)

            if len(algo_times) >= 2:
                algo_freq = 1.0 / np.median(np.diff(sorted(algo_times)))
                gt_freqs.append(gt_freq)
                algo_freqs.append(algo_freq)
                pids_out.append(pid)
        except:
            pass
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(gt_cases)}")

    gt_freqs = np.array(gt_freqs)
    algo_freqs = np.array(algo_freqs)
    rho, _ = spearmanr(gt_freqs, algo_freqs)
    mae = np.mean(np.abs(gt_freqs - algo_freqs))

    print(f"\nSpearman rho: {rho:.3f}")
    print(f"MAE: {mae:.3f} Hz")
    print(f"N: {len(gt_freqs)}")

    # Scatterplot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Color by disagreement
    diff = np.abs(gt_freqs - algo_freqs)
    colors = np.where(diff > 0.5, 'red', np.where(diff > 0.25, 'orange', 'steelblue'))

    ax.scatter(gt_freqs, algo_freqs, c=colors, alpha=0.5, s=20, edgecolors='none')

    # Identity line
    lims = [0, max(gt_freqs.max(), algo_freqs.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    # ±0.5 Hz lines
    ax.plot(lims, [l + 0.5 for l in lims], 'r--', alpha=0.2, linewidth=0.5)
    ax.plot(lims, [max(0, l - 0.5) for l in lims], 'r--', alpha=0.2, linewidth=0.5)

    ax.set_xlabel('Ground Truth Frequency (Hz)', fontsize=12)
    ax.set_ylabel('HemiCET + DP Frequency (Hz)', fontsize=12)
    ax.set_title(f'Frequency: GT vs HemiCET+DP\nSpearman ρ={rho:.3f}, MAE={mae:.3f} Hz, N={len(gt_freqs)}', fontsize=13)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.5, label=f'<0.25 Hz diff ({(diff<0.25).sum()})'),
        Patch(facecolor='orange', alpha=0.5, label=f'0.25-0.5 Hz diff ({((diff>=0.25)&(diff<0.5)).sum()})'),
        Patch(facecolor='red', alpha=0.5, label=f'>0.5 Hz diff ({(diff>=0.5).sum()})'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    out_path = PROJECT_DIR / 'results' / 'freq_scatter_hemicet.png'
    plt.savefig(str(out_path), dpi=150)
    print(f"\nScatterplot saved to {out_path}")

    # Save the freq data for the reviewer
    freq_data = []
    for pid, gf, af in zip(pids_out, gt_freqs, algo_freqs):
        freq_data.append({'pid': pid, 'gt_freq': round(float(gf), 3), 'algo_freq': round(float(af), 3),
                          'diff': round(float(abs(gf - af)), 3)})
    with open(str(PROJECT_DIR / 'data/hemi_cache/freq_comparison.json'), 'w') as f:
        json.dump(freq_data, f, indent=2)

    # Count cases with >0.5 Hz disagreement
    big_diff = [d for d in freq_data if d['diff'] > 0.5]
    print(f"\nCases with >0.5 Hz freq disagreement: {len(big_diff)}")


if __name__ == '__main__':
    main()
