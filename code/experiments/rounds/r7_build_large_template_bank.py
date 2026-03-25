"""
Build a large template bank from the ~10K unlabeled IIIC PD segments.

Extracts discharge snippets from majority-vote LPD and GPD segments,
clusters them with KMeans, and saves centroid templates for both
250ms (50-sample) and 500ms (100-sample) windows.

Outputs:
  data/templates_large_lpd.npy       (50, 50)
  data/templates_large_gpd.npy       (50, 50)
  data/templates_large_lpd_500ms.npy (30, 100)
  data/templates_large_gpd_500ms.npy (30, 100)
  results/large_template_banks.png   (4-panel figure)
"""

import sys
import os
import ast
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Paths
CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_pointiness_acf import fcn_getBanana, compute_pointiness_trace

import hdf5storage
from mne.filter import notch_filter, filter_data

IIIC_DIR = Path('/Volumes/sanD_photos/IIIC')
SEGMENTS_DIR = IIIC_DIR / 'segments_raw'
EXCEL_PATH = IIIC_DIR / 'list_events_20241129.xlsx'

DATA_DIR = REPO_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = REPO_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

FS = 200
MAX_SEGMENTS_PER_TYPE = 2000
MAX_SNIPPETS_PER_SEGMENT = 50
HALF_WIN_250 = 25   # 250ms = 50 samples, half = 25
HALF_WIN_500 = 50   # 500ms = 100 samples, half = 50


def load_segment(file_name):
    """Load a .mat segment from the external drive (HDF5 format)."""
    filepath = SEGMENTS_DIR / f'{file_name}.mat'
    mat = hdf5storage.loadmat(str(filepath))
    data = mat['data']
    # Ensure shape is (channels, samples)
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data


def extract_central_10s(data, fs):
    """Extract central 10s from segment."""
    total = data.shape[1]
    n_samples = 10 * fs
    if total <= n_samples:
        return data
    center = total // 2
    half = 5 * fs
    start = max(0, center - half)
    end = start + n_samples
    return data[:, start:end]


def preprocess(data, fs):
    """Notch 60Hz, bandpass 0.5-40Hz, bipolar montage, 15Hz lowpass."""
    data = notch_filter(data.astype(np.float64), fs, 60, n_jobs=1, verbose="ERROR")
    data = filter_data(data, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    bip = np.array(fcn_getBanana(data))
    # 15 Hz lowpass
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    for i in range(bip.shape[0]):
        try:
            bip[i] = filtfilt(b_lp, a_lp, bip[i])
        except ValueError:
            pass
    return bip


def extract_snippets_from_segment(bip, fs, half_win):
    """Extract z-scored snippets centered on pointiness peaks from all channels."""
    snippets = []
    sigma_samples = max(1, int(0.02 * fs))
    win_len = 2 * half_win

    for ch in range(bip.shape[0]):
        sig = bip[ch]
        n = len(sig)
        if n < win_len + 10:
            continue

        # Compute pointiness trace
        trace = compute_pointiness_trace(sig)
        trace = gaussian_filter1d(trace, sigma=sigma_samples)

        mx = np.max(trace)
        if mx <= 0:
            continue

        # Find peaks
        pks, _ = find_peaks(trace, height=mx * 0.3, distance=int(0.2 * fs))
        if len(pks) < 3:
            continue

        for pk in pks:
            start = pk - half_win
            end = pk + half_win
            if start < 0 or end > n:
                continue
            snippet = sig[start:end].copy()
            # z-score normalize
            std = np.std(snippet)
            if std < 1e-10:
                continue
            snippet = (snippet - np.mean(snippet)) / std
            snippets.append(snippet)

    return snippets


def collect_snippets(df_subset, max_segments, half_win, label_name):
    """Collect all snippets from a subset of segments."""
    all_snippets = []
    n_processed = 0
    n_failed = 0

    indices = df_subset.index.tolist()
    if len(indices) > max_segments:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, size=max_segments, replace=False).tolist()

    total = len(indices)
    for count, idx in enumerate(indices):
        row = df_subset.loc[idx]
        fname = row['file_name']

        if (count + 1) % 200 == 0:
            print(f"  [{label_name}] Processed {count+1}/{total}, "
                  f"snippets so far: {len(all_snippets)}, failed: {n_failed}")

        try:
            data = load_segment(fname)
            data = extract_central_10s(data, FS)

            # Need at least 19 channels for bipolar montage
            if data.shape[0] < 19:
                n_failed += 1
                continue

            bip = preprocess(data, FS)
            snippets = extract_snippets_from_segment(bip, FS, half_win)

            if len(snippets) > MAX_SNIPPETS_PER_SEGMENT:
                snippets = snippets[:MAX_SNIPPETS_PER_SEGMENT]

            all_snippets.extend(snippets)
            n_processed += 1

        except Exception as e:
            n_failed += 1
            continue

    print(f"  [{label_name}] Done. Processed: {n_processed}, Failed: {n_failed}, "
          f"Total snippets: {len(all_snippets)}")
    return all_snippets


def cluster_templates(snippets_list, k, label):
    """Cluster snippets and return centroids."""
    X = np.array(snippets_list)
    print(f"  Clustering {label}: {X.shape[0]} snippets -> {k} clusters")

    if X.shape[0] < k:
        print(f"  WARNING: fewer snippets ({X.shape[0]}) than clusters ({k}), "
              f"using all snippets as templates")
        # Pad with zeros if needed
        templates = np.zeros((k, X.shape[1]))
        templates[:X.shape[0]] = X
        return templates

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        km.fit(X)
        centroids = km.cluster_centers_
    except ImportError:
        from scipy.cluster.vq import kmeans2, whiten
        X_w = whiten(X)
        centroids_w, labels = kmeans2(X_w, k, minit='points', iter=20)
        # Un-whiten: compute per-feature std
        stds = np.std(X, axis=0)
        stds[stds < 1e-10] = 1.0
        centroids = centroids_w * stds

    print(f"  {label} centroids shape: {centroids.shape}")
    return centroids


def plot_templates(templates_dict, save_path):
    """Plot 4-panel figure of template banks."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Large Template Banks from ~10K IIIC Segments', fontsize=14, fontweight='bold')

    panels = [
        ('LPD 250ms', 'templates_lpd_250'),
        ('GPD 250ms', 'templates_gpd_250'),
        ('LPD 500ms', 'templates_lpd_500'),
        ('GPD 500ms', 'templates_gpd_500'),
    ]

    for ax, (title, key) in zip(axes.flat, panels):
        templates = templates_dict[key]
        k, n_samples = templates.shape
        t_ms = np.arange(n_samples) / FS * 1000  # time in ms

        for i in range(k):
            ax.plot(t_ms, templates[i] + i * 3, linewidth=0.7, color='steelblue', alpha=0.8)

        ax.set_title(f'{title} ({k} templates)', fontsize=12)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Template index (offset)')
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def main():
    print("=" * 70)
    print("Building large template bank from IIIC segments")
    print("=" * 70)

    # 1. Load and parse labels
    print("\n[1] Loading Excel file...")
    df = pd.read_excel(str(EXCEL_PATH))
    print(f"  Total segments in Excel: {len(df)}")

    label_col = 'label ([other,seizure,lpd,gpd,lrda,grda])'
    df['parsed'] = df[label_col].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else None)
    df = df[df['parsed'].notna()].copy()
    df['total'] = df['parsed'].apply(sum)
    df = df[df['total'] > 0].copy()

    for i, name in enumerate(['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']):
        df[name] = df['parsed'].apply(lambda x, idx=i: x[idx])

    df['lpd_frac'] = df['lpd'] / df['total']
    df['gpd_frac'] = df['gpd'] / df['total']

    # Check which files actually exist
    available = set()
    if SEGMENTS_DIR.exists():
        available = set(os.path.splitext(f)[0] for f in os.listdir(str(SEGMENTS_DIR))
                       if f.endswith('.mat'))
    print(f"  Available .mat files on drive: {len(available)}")

    df = df[df['file_name'].isin(available)].copy()

    df_lpd = df[df['lpd_frac'] > 0.5].copy()
    df_gpd = df[df['gpd_frac'] > 0.5].copy()
    print(f"  LPD segments (majority vote): {len(df_lpd)}")
    print(f"  GPD segments (majority vote): {len(df_gpd)}")

    # 2. Extract snippets (250ms and 500ms)
    print("\n[2] Extracting 250ms snippets...")
    print("  Processing LPD...")
    lpd_snippets_250 = collect_snippets(df_lpd, MAX_SEGMENTS_PER_TYPE, HALF_WIN_250, "LPD-250ms")
    print("  Processing GPD...")
    gpd_snippets_250 = collect_snippets(df_gpd, MAX_SEGMENTS_PER_TYPE, HALF_WIN_250, "GPD-250ms")

    print("\n[3] Extracting 500ms snippets...")
    print("  Processing LPD...")
    lpd_snippets_500 = collect_snippets(df_lpd, MAX_SEGMENTS_PER_TYPE, HALF_WIN_500, "LPD-500ms")
    print("  Processing GPD...")
    gpd_snippets_500 = collect_snippets(df_gpd, MAX_SEGMENTS_PER_TYPE, HALF_WIN_500, "GPD-500ms")

    # 3. Cluster
    print("\n[4] Clustering 250ms templates (k=50)...")
    templates_lpd_250 = cluster_templates(lpd_snippets_250, 50, "LPD-250ms")
    templates_gpd_250 = cluster_templates(gpd_snippets_250, 50, "GPD-250ms")

    print("\n[5] Clustering 500ms templates (k=30)...")
    templates_lpd_500 = cluster_templates(lpd_snippets_500, 30, "LPD-500ms")
    templates_gpd_500 = cluster_templates(gpd_snippets_500, 30, "GPD-500ms")

    # 4. Save
    print("\n[6] Saving templates...")
    np.save(str(DATA_DIR / 'templates_large_lpd.npy'), templates_lpd_250)
    np.save(str(DATA_DIR / 'templates_large_gpd.npy'), templates_gpd_250)
    np.save(str(DATA_DIR / 'templates_large_lpd_500ms.npy'), templates_lpd_500)
    np.save(str(DATA_DIR / 'templates_large_gpd_500ms.npy'), templates_gpd_500)

    print(f"  templates_large_lpd.npy:        {templates_lpd_250.shape}")
    print(f"  templates_large_gpd.npy:        {templates_gpd_250.shape}")
    print(f"  templates_large_lpd_500ms.npy:  {templates_lpd_500.shape}")
    print(f"  templates_large_gpd_500ms.npy:  {templates_gpd_500.shape}")

    # 5. Plot
    print("\n[7] Plotting templates...")
    plot_templates({
        'templates_lpd_250': templates_lpd_250,
        'templates_gpd_250': templates_gpd_250,
        'templates_lpd_500': templates_lpd_500,
        'templates_gpd_500': templates_gpd_500,
    }, RESULTS_DIR / 'large_template_banks.png')

    print("\nDone!")


if __name__ == '__main__':
    main()
