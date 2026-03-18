"""
Build template banks C (data-driven) and D (synthetic) for matched-filter
discharge detection and save to disk.

Bank C: cluster centroids from annotated EEG segments with high expert agreement.
Bank D: synthetic discharge templates with parameterized sharp+slow wave shapes.
"""

import sys
import os
from pathlib import Path

# Ensure code/ and code/pd_detector_alternate/ are on sys.path
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import warnings
warnings.filterwarnings('ignore')

from mne.filter import notch_filter, filter_data
from scipy.signal import butter, filtfilt, detrend, savgol_filter

from optimization_harness import load_dataset, load_eeg_data
from pd_detect_alternate import pd_detect_alternate, fcn_getBanana, bipolar_channels
from browse_results import detect_pd_peaks

# Paths
REPO_ROOT = CODE_DIR.parent
DATA_DIR = REPO_ROOT / 'data'
RESULTS_DIR = REPO_ROOT / 'results'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FS = 200  # sampling frequency
WINDOW_SAMPLES = 50  # 250ms at 200Hz
HALF_WIN = WINDOW_SAMPLES // 2  # 25 samples


def build_bank_c():
    """Build Bank C: data-driven templates from annotated segments."""
    print("=" * 60)
    print("BANK C: Data-driven templates from annotated segments")
    print("=" * 60)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} annotated segments")

    # Select segments with high expert agreement (std < 0.2 Hz)
    selected = []
    for entry in dataset:
        freqs = []
        for key in ['expert_LB_freq', 'expert_PH_freq', 'expert_SZ_freq']:
            v = entry.get(key, np.nan)
            if np.isfinite(v) and v > 0:
                freqs.append(v)
        if len(freqs) < 2:
            continue
        if np.std(freqs) < 0.2:
            selected.append(entry)

    print(f"Selected {len(selected)} segments with expert freq std < 0.2 Hz")

    # Extract snippets from each selected segment
    lpd_snippets = []
    gpd_snippets = []

    for idx, entry in enumerate(selected):
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        # Preprocess: notch 60Hz, bandpass 0.5-40Hz
        data = notch_filter(data.astype(np.float64), fs, 60, n_jobs=1, verbose='ERROR')
        data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose='ERROR')

        # Bipolar montage
        try:
            bipolar = fcn_getBanana(data)
        except Exception:
            continue
        bipolar = np.array(bipolar)

        # 15Hz lowpass
        b_lp, a_lp = butter(4, 15.0 / (fs / 2.0), btype='low')
        for ch in range(bipolar.shape[0]):
            bipolar[ch] = filtfilt(b_lp, a_lp, bipolar[ch])

        # Run Method A to get channel PD scores
        try:
            result = pd_detect_alternate(data, fs)
        except Exception:
            continue

        channel_scores = result.get('channel_pd_scores', {})

        # For channels with score > 1.0, detect peaks
        for ch_idx, ch_name in enumerate(bipolar_channels):
            score = channel_scores.get(ch_name, 0)
            if not np.isfinite(score) or score <= 1.0:
                continue

            # Use detect_pd_peaks on the preprocessed bipolar channel
            peaks = detect_pd_peaks(bipolar[ch_idx], fs=fs)
            if len(peaks) == 0:
                continue

            # Extract 250ms windows centered on each peak
            for pk in peaks:
                start = pk - HALF_WIN
                end = pk + HALF_WIN
                if start < 0 or end > bipolar.shape[1]:
                    continue
                snippet = bipolar[ch_idx, start:end].copy()
                if len(snippet) != WINDOW_SAMPLES:
                    continue
                # Z-score normalize
                s = np.std(snippet)
                if s < 1e-10:
                    continue
                snippet = (snippet - np.mean(snippet)) / s

                if entry['subdir'] == 'lpd':
                    lpd_snippets.append(snippet)
                else:
                    gpd_snippets.append(snippet)

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(selected)} segments "
                  f"(LPD: {len(lpd_snippets)}, GPD: {len(gpd_snippets)} snippets)")

    print(f"\nTotal snippets: LPD={len(lpd_snippets)}, GPD={len(gpd_snippets)}")

    # Cluster each type with KMeans (k=8)
    K = 8
    templates_lpd = _cluster_snippets(lpd_snippets, K, 'LPD')
    templates_gpd = _cluster_snippets(gpd_snippets, K, 'GPD')

    # Save
    np.save(str(DATA_DIR / 'templates_C_lpd.npy'), templates_lpd)
    np.save(str(DATA_DIR / 'templates_C_gpd.npy'), templates_gpd)
    print(f"Saved templates_C_lpd.npy shape={templates_lpd.shape}")
    print(f"Saved templates_C_gpd.npy shape={templates_gpd.shape}")

    return templates_lpd, templates_gpd


def _cluster_snippets(snippets, k, label):
    """Cluster snippets into k groups and return centroids."""
    if len(snippets) == 0:
        print(f"  WARNING: No {label} snippets to cluster, using zeros")
        return np.zeros((k, WINDOW_SAMPLES))

    X = np.array(snippets)
    print(f"  Clustering {len(X)} {label} snippets into {k} clusters...")

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        centroids = km.cluster_centers_
    except ImportError:
        from scipy.cluster.vq import kmeans2
        centroids, _ = kmeans2(X, k, minit='points')

    # Z-score normalize each centroid
    for i in range(centroids.shape[0]):
        s = np.std(centroids[i])
        if s > 1e-10:
            centroids[i] = (centroids[i] - np.mean(centroids[i])) / s

    return centroids


def build_bank_d():
    """Build Bank D: synthetic discharge templates."""
    print("\n" + "=" * 60)
    print("BANK D: Synthetic discharge templates")
    print("=" * 60)

    t = np.linspace(0, 0.25, WINDOW_SAMPLES)
    t_peak = 0.10  # peak at 40% of window

    sigma_sharp_vals = [0.008, 0.015, 0.025, 0.040]
    slow_ratio_vals = [0.5, 0.8]

    templates = []
    for sigma_sharp in sigma_sharp_vals:
        for slow_ratio in slow_ratio_vals:
            t_slow = t_peak + 2 * sigma_sharp + 0.03
            sigma_slow = 3 * sigma_sharp

            # Sharp negative + slow positive return
            template = (-np.exp(-(t - t_peak) ** 2 / (2 * sigma_sharp ** 2))
                        + slow_ratio * np.exp(-(t - t_slow) ** 2 / (2 * sigma_slow ** 2)))

            # Z-score normalize
            s = np.std(template)
            if s > 1e-10:
                template = (template - np.mean(template)) / s

            templates.append(template)

    templates = np.array(templates)
    np.save(str(DATA_DIR / 'templates_D.npy'), templates)
    print(f"Saved templates_D.npy shape={templates.shape}")

    return templates


def plot_templates(templates_lpd, templates_gpd, templates_d):
    """Plot all template banks as a 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    t_ms = np.linspace(0, 250, WINDOW_SAMPLES)

    for i in range(templates_lpd.shape[0]):
        axes[0].plot(t_ms, templates_lpd[i], label=f'C-LPD-{i}')
    axes[0].set_title('Bank C: LPD Templates')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel('Amplitude (z-scored)')
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for i in range(templates_gpd.shape[0]):
        axes[1].plot(t_ms, templates_gpd[i], label=f'C-GPD-{i}')
    axes[1].set_title('Bank C: GPD Templates')
    axes[1].set_xlabel('Time (ms)')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    for i in range(templates_d.shape[0]):
        axes[2].plot(t_ms, templates_d[i], label=f'D-{i}')
    axes[2].set_title('Bank D: Synthetic Templates')
    axes[2].set_xlabel('Time (ms)')
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / 'template_banks.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\nSaved figure to {out_path}")


if __name__ == '__main__':
    templates_lpd, templates_gpd = build_bank_c()
    templates_d = build_bank_d()
    plot_templates(templates_lpd, templates_gpd, templates_d)
    print("\nDone.")
