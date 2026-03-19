"""
Generate EEG trace PNGs for Round 2 annotation candidates.

Loads each candidate from data/annotation_round2/{lpd,gpd}/*.mat (already bipolar),
plots 18-channel traces with frequency annotations and peak markers.

Run: conda run -n foe python code/dl/generate_round2_pngs.py
"""

import sys
import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

DL_DIR = Path(__file__).resolve().parent
CODE_DIR = DL_DIR.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from pd_pointiness_acf import bipolar_channels, compute_pointiness_trace, compute_acf_frequency

OUTPUT_DIR = PROJECT_DIR / 'data' / '_archive' / 'annotation_round2'
IMAGES_DIR = OUTPUT_DIR / 'images'
FS = 200

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_INDICES = [16, 17]


def draw_round2_figure(seg, fs, channel_names, manifest_row):
    """Draw 18-channel bipolar EEG with ACF mini-plots and peak markers.

    Args:
        seg: (18, N) bipolar EEG data (float64)
        fs: sampling rate
        channel_names: list of 18 channel names
        manifest_row: dict with file_name, subtype, patient_id, freq estimates
    Returns:
        matplotlib figure
    """
    n_ch, n_samples = seg.shape
    time_vec = np.arange(n_samples) / fs

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(n_ch + 2, 5, width_ratios=[3, 0.4, 0.05, 0.8, 0.01],
                  hspace=0.08, wspace=0.25,
                  left=0.06, right=0.98, top=0.90, bottom=0.05)

    # Lowpass for pointiness
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')

    # Pre-compute pointiness peaks for all channels
    channel_point_peaks = {}
    channel_acf_curves = {}
    channel_acf_freqs = {}

    for i in range(n_ch):
        try:
            sig_lp = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            sig_lp = seg[i]

        pt = compute_pointiness_trace(sig_lp)
        pt = gaussian_filter1d(pt, sigma=fs * 0.02)
        mx = np.max(pt)
        if mx > 0:
            pks, _ = find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
            if len(pks) > 0:
                channel_point_peaks[i] = pks

        # ACF
        ptm = pt - np.mean(pt)
        max_lag = min(4 * fs, len(ptm) - 1)
        acf = np.correlate(ptm, ptm, mode='full')
        acf = acf[len(ptm) - 1:][:max_lag + 1]
        if acf[0] > 0:
            acf = acf / acf[0]
        channel_acf_curves[i] = acf

        # ACF frequency
        freq, _, _ = compute_acf_frequency(
            seg[i], fs, method='pointiness',
            smoothing_sigma=0.02, acf_min_lag=0.4,
            acf_peak_threshold=0.10, peak_height_frac=0.3)
        channel_acf_freqs[i] = freq

    # Global scale
    global_std = np.nanstd(seg)
    if global_std < 1e-10:
        global_std = 1.0
    scale = 4.0 * global_std

    for i in range(n_ch):
        # Main EEG trace
        ax = fig.add_subplot(gs[i + 1, 0])
        signal = seg[i]

        if i in LEFT_INDICES:
            color = '#cc3333'
            bg = '#ffe8e8'
        elif i in RIGHT_INDICES:
            color = '#3333cc'
            bg = '#e8e8ff'
        else:
            color = '#333333'
            bg = '#f0f0f0'

        ax.set_facecolor(bg)
        ax.plot(time_vec, signal, color=color, linewidth=0.7)

        # Mark pointiness peaks
        if i in channel_point_peaks:
            pk_idx = channel_point_peaks[i]
            pk_idx = pk_idx[pk_idx < len(time_vec)]
            ax.plot(time_vec[pk_idx], signal[pk_idx], 'v',
                    color='#22aa22', markersize=4, alpha=0.7)

        ax.set_ylim(-scale, scale)

        freq_str = f'{channel_acf_freqs[i]:.1f}Hz' if np.isfinite(channel_acf_freqs[i]) else ''
        label = f'{channel_names[i]}  {freq_str}'
        ax.set_ylabel(label, fontsize=7, rotation=0, ha='right', va='center',
                       labelpad=65)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax.tick_params(axis='x', labelsize=6)

        for t in range(0, int(n_samples / fs) + 1):
            ax.axvline(t, color='#cccccc', linewidth=0.3, zorder=0)

        if i < n_ch - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_xlim(0, n_samples / fs)

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Mini ACF plot
        ac_ax = fig.add_subplot(gs[i + 1, 1])
        acf = channel_acf_curves[i]
        lag = np.arange(len(acf)) / fs
        ac_ax.plot(lag, acf, color='#ee7722', linewidth=0.7)
        ac_ax.axhline(0, color='gray', linewidth=0.3)

        # Mark ACF peak
        min_lag_samples = int(0.4 * fs)
        for k in range(min_lag_samples + 1, len(acf) - 1):
            if acf[k] > acf[k - 1] and acf[k] > acf[k + 1] and acf[k] > 0.15:
                ac_ax.axvline(k / fs, color='#ee7722', linewidth=0.7,
                              linestyle='--', alpha=0.8)
                break

        ac_ax.set_xlim(0, 4)
        ac_ax.set_ylim(-0.5, 1)
        ac_ax.set_yticks([])
        ac_ax.tick_params(axis='x', labelsize=4, pad=1)
        if i < n_ch - 1:
            ac_ax.set_xticklabels([])
        else:
            ac_ax.set_xlabel('Lag (s)', fontsize=5)
        for spine in ac_ax.spines.values():
            spine.set_visible(False)

    # ── Title and info panel ─────────────────────────────────────────
    subtype = manifest_row.get('subtype', '?').upper()
    patient_id = manifest_row.get('patient_id', '?')
    consensus = manifest_row.get('consensus_estimate', '?')
    disagreement = manifest_row.get('disagreement', '?')

    fig.text(0.5, 0.95,
             f'{subtype} | Patient {patient_id} | '
             f'Consensus: {consensus} Hz | Disagreement: {disagreement}',
             ha='center', fontsize=12, fontweight='bold')

    # Frequency estimates summary
    f_B = manifest_row.get('f_B', '')
    f_peaks = manifest_row.get('f_peaks', '')
    f_fft = manifest_row.get('f_fft', '')
    f_tkeo = manifest_row.get('f_tkeo', '')

    est_str = (f'ACF={f_B or "NaN"} | Peaks={f_peaks or "NaN"} | '
               f'FFT={f_fft or "NaN"} | TKEO={f_tkeo or "NaN"}')
    fig.text(0.5, 0.92, est_str,
             ha='center', fontsize=9, color='#444488',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffff0',
                       edgecolor='#aaaaaa', alpha=0.9))

    # ── Right panel: per-channel frequency summary ───────────────────
    info_ax = fig.add_subplot(gs[1:10, 3])
    info_ax.axis('off')
    lines = ['Per-ch ACF freq (Hz):', '']
    for i in range(n_ch):
        f = channel_acf_freqs[i]
        f_str = f'{f:.2f}' if np.isfinite(f) else ' NaN'
        hemi = 'L' if i in LEFT_INDICES else ('R' if i in RIGHT_INDICES else 'M')
        lines.append(f'{hemi} {channel_names[i]:>8s}: {f_str}')
    info_ax.text(0.0, 1.0, '\n'.join(lines),
                 fontsize=7.5, fontfamily='monospace',
                 va='top', ha='left', transform=info_ax.transAxes)

    # Frequency histogram of channel estimates
    hist_ax = fig.add_subplot(gs[11:16, 3])
    valid_freqs = [channel_acf_freqs[i] for i in range(n_ch)
                   if np.isfinite(channel_acf_freqs[i])]
    if valid_freqs:
        hist_ax.hist(valid_freqs, bins=15, range=(0.2, 3.5),
                     color='#ee7722', alpha=0.7, edgecolor='white')
        if consensus and consensus != '?':
            try:
                hist_ax.axvline(float(consensus), color='#44cc44',
                                linewidth=2, linestyle='--', label='consensus')
                hist_ax.legend(fontsize=7)
            except ValueError:
                pass
    hist_ax.set_xlabel('Frequency (Hz)', fontsize=8)
    hist_ax.set_ylabel('Channels', fontsize=8)
    hist_ax.set_title('Channel freq distribution', fontsize=8)
    hist_ax.tick_params(labelsize=7)

    # Legend
    fig.text(0.5, 0.01,
             'Green triangles: pointiness peaks | '
             'Orange dashed: ACF period | '
             'Left=red bg, Right=blue bg',
             ha='center', fontsize=8, color='gray')

    return fig


def main():
    print("=" * 60)
    print("Generate Round 2 Annotation PNGs")
    print("=" * 60)

    manifest_path = OUTPUT_DIR / 'manifest.csv'
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(str(manifest_path), 'r') as f:
        reader = csv.DictReader(f)
        manifest = list(reader)

    print(f"  {len(manifest)} candidates in manifest")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    for i, mrow in enumerate(manifest):
        file_name = mrow['file_name']
        subtype = mrow['subtype']

        mat_path = OUTPUT_DIR / subtype / f"{file_name}.mat"
        if not mat_path.exists():
            print(f"  WARNING: {mat_path} not found, skipping")
            continue

        mat = loadmat(str(mat_path))
        seg = mat['data'].astype(np.float64)

        fig = draw_round2_figure(seg, FS, bipolar_channels, mrow)
        out_path = IMAGES_DIR / f"{file_name}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (i + 1) % 5 == 0 or (i + 1) == len(manifest):
            print(f"  {i + 1}/{len(manifest)} PNGs generated")

    print(f"\nDone! PNGs saved to {IMAGES_DIR}")


if __name__ == '__main__':
    main()
