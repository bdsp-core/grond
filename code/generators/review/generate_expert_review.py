"""
Generate expert disagreement review data:
1. Select top 50 high-disagreement cases (1 per patient)
2. Compute algorithm features (f_B, f_peaks, f_fft, f_tkeo, f_coh, consensus)
3. Generate EEG PNGs via draw_figure
4. Save manifest.csv
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt, coherence
from scipy.ndimage import gaussian_filter1d
from mne.filter import notch_filter, filter_data
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from optimization_harness import load_dataset, load_eeg_data
from pd_detect_alternate import pd_detect_alternate, fcn_getBanana, bipolar_channels, mono_channels
from pd_pointiness_acf import pd_detect_pointiness_acf, compute_pointiness_trace
from generate_test_images import draw_figure, run_detector
from browse_results import BIPOLAR_CHANNELS, get_bipolar

OUTPUT_DIR = CODE_DIR.parent / 'data' / '_archive' / 'pd_expert_review'
IMAGES_DIR = OUTPUT_DIR / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

FREQ_LO, FREQ_HI = 0.3, 3.5

# Adjacent channel pairs for spectral coherence
ADJACENT_PAIRS = [
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17),
]


def compute_fft_peak(trace, fs, freq_lo=FREQ_LO, freq_hi=FREQ_HI):
    """FFT of a 1D trace, return peak frequency in [freq_lo, freq_hi] Hz."""
    n = len(trace)
    if n < 10:
        return np.nan
    fft_vals = np.abs(np.fft.rfft(trace - np.mean(trace)))
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    mask = (freqs >= freq_lo) & (freqs <= freq_hi)
    if not np.any(mask):
        return np.nan
    fft_sub = fft_vals[mask]
    freqs_sub = freqs[mask]
    if np.max(fft_sub) == 0:
        return np.nan
    return freqs_sub[np.argmax(fft_sub)]


def extract_patient_id(mat_name):
    """Extract patient ID from mat filename like 'abn1047_20141222_...'"""
    parts = mat_name.split('_')
    return parts[0]


def compute_algorithm_features(data, fs, entry):
    """Compute algorithm frequency estimates for one segment."""
    features = {}

    # Method B: pd_detect_pointiness_acf (thr=0.10)
    try:
        res_B = pd_detect_pointiness_acf(
            data.copy(), fs,
            method='pointiness', lowpass_hz=15,
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3
        )
        f_B = res_B['event_frequency']
        if not np.isfinite(f_B):
            f_B = np.nan
    except:
        f_B = np.nan
    features['f_B'] = f_B

    # Preprocessing
    seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
    seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg_bip = np.array(fcn_getBanana(seg_filtered))

    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    seg_lp = np.zeros_like(seg_bip)
    for ch in range(seg_bip.shape[0]):
        try:
            seg_lp[ch] = filtfilt(b_lp, a_lp, seg_bip[ch])
        except ValueError:
            seg_lp[ch] = seg_bip[ch]

    n_channels = seg_lp.shape[0]
    sigma_samples = max(1, int(0.02 * fs))

    # Pointiness traces
    pointiness_traces = []
    for ch in range(n_channels):
        pt = compute_pointiness_trace(seg_lp[ch])
        pt = gaussian_filter1d(pt, sigma=sigma_samples)
        pointiness_traces.append(pt)
    pointiness_traces = np.array(pointiness_traces)

    # f_peaks: peak-count on pointiness
    peak_count_freqs = []
    for ch in range(n_channels):
        pt = pointiness_traces[ch]
        mx = np.max(pt)
        if mx == 0:
            continue
        pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) >= 3:
            span = (pks[-1] - pks[0]) / fs
            if span > 0:
                peak_count_freqs.append((len(pks) - 1) / span)
    features['f_peaks'] = float(np.median(peak_count_freqs)) if peak_count_freqs else np.nan

    # f_fft: FFT of pointiness
    fft_freqs = []
    for ch in range(n_channels):
        f = compute_fft_peak(pointiness_traces[ch], fs)
        if np.isfinite(f):
            fft_freqs.append(f)
    features['f_fft'] = float(np.median(fft_freqs)) if fft_freqs else np.nan

    # f_tkeo: TKEO on bipolar signal
    tkeo_freqs = []
    for ch in range(n_channels):
        x = seg_lp[ch]
        if len(x) < 3:
            continue
        tkeo = np.abs(x[1:-1]**2 - x[:-2] * x[2:])
        tkeo_smooth = gaussian_filter1d(tkeo, sigma=sigma_samples)
        f = compute_fft_peak(tkeo_smooth, fs)
        if np.isfinite(f):
            tkeo_freqs.append(f)
    features['f_tkeo'] = float(np.median(tkeo_freqs)) if tkeo_freqs else np.nan

    # f_coh: spectral coherence
    coh_freqs = []
    for (ch_a, ch_b) in ADJACENT_PAIRS:
        if ch_a >= n_channels or ch_b >= n_channels:
            continue
        try:
            f_coh, Cxy = coherence(seg_bip[ch_a], seg_bip[ch_b],
                                   fs=fs, nperseg=min(256, seg_bip.shape[1]))
            mask = (f_coh >= FREQ_LO) & (f_coh <= FREQ_HI)
            if np.any(mask):
                Cxy_sub = Cxy[mask]
                f_coh_sub = f_coh[mask]
                if np.max(Cxy_sub) > 0:
                    coh_freqs.append(f_coh_sub[np.argmax(Cxy_sub)])
        except:
            continue
    features['f_coh'] = float(np.median(coh_freqs)) if coh_freqs else np.nan

    # Consensus: median of available algorithm estimates
    algo_vals = [features[k] for k in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']
                 if np.isfinite(features.get(k, np.nan))]
    features['consensus_estimate'] = float(np.median(algo_vals)) if algo_vals else np.nan

    return features


def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Loaded {len(dataset)} segments")

    # Step 1: Compute expert disagreement (std) for each segment
    print("\nComputing expert disagreement...")
    records = []
    for entry in dataset:
        freqs = []
        for key in ['expert_LB_freq', 'expert_PH_freq', 'expert_SZ_freq']:
            val = entry.get(key, np.nan)
            if np.isfinite(val) and val > 0:
                freqs.append(val)
        if len(freqs) >= 2:
            std = float(np.std(freqs))
        else:
            std = 0.0

        records.append({
            'mat_name': entry['mat_name'],
            'subdir': entry['subdir'],
            'mat_path': entry['mat_path'],
            'patient_id': extract_patient_id(entry['mat_name']),
            'expert_LB': entry.get('expert_LB_freq', np.nan),
            'expert_PH': entry.get('expert_PH_freq', np.nan),
            'expert_SZ': entry.get('expert_SZ_freq', np.nan),
            'expert_mean': float(np.mean(freqs)) if freqs else np.nan,
            'expert_median': float(np.median(freqs)) if freqs else np.nan,
            'expert_std': std,
        })

    # Sort by std descending
    records.sort(key=lambda x: -x['expert_std'])

    # Select top 50, 1 per patient
    seen_patients = set()
    selected = []
    for rec in records:
        pid = rec['patient_id']
        if pid not in seen_patients:
            seen_patients.add(pid)
            selected.append(rec)
            if len(selected) == 50:
                break

    print(f"Selected {len(selected)} high-disagreement cases from {len(records)} total")
    print(f"  Std range: {selected[0]['expert_std']:.3f} - {selected[-1]['expert_std']:.3f}")

    # Step 2: Compute algorithm features and generate images
    print("\nProcessing selected cases...")
    manifest_rows = []

    for i, rec in enumerate(selected):
        mat_name = rec['mat_name']
        subdir = rec['subdir']
        subtype = 'LPD' if subdir == 'lpd' else 'GPD'
        print(f"  [{i+1}/{len(selected)}] {mat_name} ({subtype}, std={rec['expert_std']:.3f})")

        # Find the entry in dataset
        entry = None
        for e in dataset:
            if e['mat_name'] == mat_name:
                entry = e
                break
        if entry is None:
            print(f"    WARNING: entry not found, skipping")
            continue

        # Load EEG
        data, fs = load_eeg_data(entry)
        if data is None:
            print(f"    WARNING: could not load EEG, skipping")
            continue

        # Compute algorithm features
        algo_feats = compute_algorithm_features(data, fs, entry)
        print(f"    f_B={algo_feats['f_B']:.2f}, f_peaks={algo_feats['f_peaks']:.2f}, "
              f"f_fft={algo_feats['f_fft']:.2f}, f_tkeo={algo_feats['f_tkeo']:.2f}, "
              f"f_coh={algo_feats['f_coh']:.2f}, consensus={algo_feats['consensus_estimate']:.2f}"
              if all(np.isfinite(algo_feats.get(k, np.nan)) for k in ['f_B','f_peaks','f_fft','f_tkeo','f_coh','consensus_estimate'])
              else f"    features (some NaN): {algo_feats}")

        # Run detector for drawing
        try:
            # Filter + bipolar for the figure
            seg_filtered = notch_filter(data.copy(), fs, 60, n_jobs=1, verbose="ERROR")
            seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg_bi = np.array(get_bipolar(seg_filtered))

            row = run_detector(data.copy(), fs, subdir)

            # Title with expert ratings and algorithm
            lb_str = f"LB:{rec['expert_LB']:.2f}" if np.isfinite(rec['expert_LB']) else "LB:--"
            ph_str = f"PH:{rec['expert_PH']:.2f}" if np.isfinite(rec['expert_PH']) else "PH:--"
            sz_str = f"SZ:{rec['expert_SZ']:.2f}" if np.isfinite(rec['expert_SZ']) else "SZ:--"
            cons_str = f"Algo:{algo_feats['consensus_estimate']:.2f}" if np.isfinite(algo_feats['consensus_estimate']) else "Algo:--"
            title_extra = f"{mat_name}  |  {lb_str}  {ph_str}  {sz_str}  std={rec['expert_std']:.2f}  |  {cons_str}"

            fig = draw_figure(row, seg_bi, fs, subdir, title_extra=title_extra)

            # Save PNG
            img_name = mat_name.replace('.mat', '.png')
            img_path = IMAGES_DIR / img_name
            fig.savefig(str(img_path), dpi=120, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved {img_name}")
        except Exception as e:
            print(f"    WARNING: could not generate image: {e}")
            img_name = ''

        # Build manifest row
        manifest_rows.append({
            'mat_name': mat_name,
            'subtype': subtype,
            'patient_id': rec['patient_id'],
            'expert_LB': round(rec['expert_LB'], 4) if np.isfinite(rec['expert_LB']) else '',
            'expert_PH': round(rec['expert_PH'], 4) if np.isfinite(rec['expert_PH']) else '',
            'expert_SZ': round(rec['expert_SZ'], 4) if np.isfinite(rec['expert_SZ']) else '',
            'expert_mean': round(rec['expert_mean'], 4) if np.isfinite(rec['expert_mean']) else '',
            'expert_median': round(rec['expert_median'], 4) if np.isfinite(rec['expert_median']) else '',
            'expert_std': round(rec['expert_std'], 4),
            'f_B': round(algo_feats['f_B'], 4) if np.isfinite(algo_feats['f_B']) else '',
            'f_peaks': round(algo_feats['f_peaks'], 4) if np.isfinite(algo_feats['f_peaks']) else '',
            'f_fft': round(algo_feats['f_fft'], 4) if np.isfinite(algo_feats['f_fft']) else '',
            'f_tkeo': round(algo_feats['f_tkeo'], 4) if np.isfinite(algo_feats['f_tkeo']) else '',
            'f_coh': round(algo_feats['f_coh'], 4) if np.isfinite(algo_feats['f_coh']) else '',
            'consensus_estimate': round(algo_feats['consensus_estimate'], 4) if np.isfinite(algo_feats['consensus_estimate']) else '',
        })

    # Save manifest
    df = pd.DataFrame(manifest_rows)
    csv_path = OUTPUT_DIR / 'manifest.csv'
    df.to_csv(str(csv_path), index=False)
    print(f"\nSaved manifest to {csv_path}")
    print(f"  {len(df)} cases, columns: {list(df.columns)}")
    print(f"\nImages saved to {IMAGES_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
