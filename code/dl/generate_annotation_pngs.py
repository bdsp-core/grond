"""
Generate EEG trace PNGs for annotation candidates.

Loads each candidate segment from data/annotation_candidates/{lpd,gpd}/*.mat,
plots 18-channel bipolar EEG traces, and saves as PNG.

Run: conda run -n foe_dl python code/dl/generate_annotation_pngs.py
"""

import sys
import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

warnings.filterwarnings('ignore')

DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import bipolar_channels

OUTPUT_DIR = PROJECT_DIR / 'data' / '_archive' / 'annotation_candidates'
IMAGES_DIR = OUTPUT_DIR / 'images'
FS = 200

# Channel layout: left/right coloring
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_INDICES = [16, 17]


def plot_eeg_traces(seg, fs, channel_names, title, out_path):
    """Plot 18-channel bipolar EEG traces in a single figure.

    Args:
        seg: (18, N) bipolar EEG data
        fs: sampling rate
        channel_names: list of 18 channel names
        title: figure title (includes freq estimates)
        out_path: where to save PNG
    """
    n_ch, n_samples = seg.shape
    time_vec = np.arange(n_samples) / fs

    fig, axes = plt.subplots(n_ch, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=0.98)

    # Compute a global scale for consistent amplitude across channels
    global_std = np.nanstd(seg)
    if global_std < 1e-10:
        global_std = 1.0
    scale = 4.0 * global_std  # signals clipped visually at +/- 4 std

    for i in range(n_ch):
        ax = axes[i]
        signal = seg[i]

        # Color by hemisphere
        if i in LEFT_INDICES:
            color = '#cc3333'
            bg = '#fff0f0'
        elif i in RIGHT_INDICES:
            color = '#3333cc'
            bg = '#f0f0ff'
        else:
            color = '#333333'
            bg = '#f5f5f5'

        ax.set_facecolor(bg)
        ax.plot(time_vec, signal, color=color, linewidth=0.6)
        ax.set_ylim(-scale, scale)
        ax.set_ylabel(channel_names[i], fontsize=7, rotation=0, ha='right', va='center')
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='x', labelsize=7)

        # Light gridlines at 1-second intervals
        for t in range(0, int(n_samples / fs) + 1):
            ax.axvline(t, color='#cccccc', linewidth=0.3, zorder=0)

        if i < n_ch - 1:
            ax.tick_params(axis='x', labelbottom=False)

    axes[-1].set_xlabel('Time (s)', fontsize=9)
    axes[-1].set_xlim(0, n_samples / fs)

    plt.tight_layout(rect=[0.06, 0.01, 1.0, 0.96])
    fig.savefig(str(out_path), dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)


def main():
    print("=" * 60)
    print("Generate Annotation PNGs")
    print("=" * 60)

    # Load manifest
    manifest_path = OUTPUT_DIR / 'manifest.csv'
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run select_for_annotation.py first.")
        sys.exit(1)

    with open(str(manifest_path), 'r') as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    print(f"  {len(manifest)} candidates in manifest")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(manifest):
        file_name = row['file_name']
        subtype = row['subtype']
        patient_id = row['patient_id']

        mat_path = OUTPUT_DIR / subtype / f"{file_name}.mat"
        if not mat_path.exists():
            print(f"  WARNING: {mat_path} not found, skipping")
            continue

        mat = loadmat(str(mat_path))
        seg = mat['data']  # (18, 2000)

        # Build title with frequency estimates
        freqs_str = []
        for key in ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']:
            val = row.get(key, '')
            if val:
                label = key.replace('f_', '')
                freqs_str.append(f"{label}={val}")
            else:
                label = key.replace('f_', '')
                freqs_str.append(f"{label}=NaN")

        consensus = row.get('consensus_estimate', '?')
        disag = row.get('disagreement', '?')

        title = (f"{subtype.upper()} | Patient {patient_id} | "
                 f"Consensus: {consensus} Hz | Disagreement: {disag}\n"
                 f"{' | '.join(freqs_str)}")

        out_path = IMAGES_DIR / f"{file_name}.png"
        plot_eeg_traces(seg, FS, bipolar_channels, title, out_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(manifest):
            print(f"  {i + 1}/{len(manifest)} PNGs generated")

    print(f"\nDone! PNGs saved to {IMAGES_DIR}")


if __name__ == '__main__':
    main()
