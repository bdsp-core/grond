"""
Generate Fig 0 (Figure 1 in paper): Six raw EEG examples for the introduction.

A 3x2 grid showing clear and ambiguous IIIC patterns:
  Row 1: Clear LPD (A), Clear GPD (B)
  Row 2: Clear LRDA (C), Clear GRDA (D)
  Row 3: Ambiguous LPD (E), Ambiguous mixed (F)

Each panel shows 10 seconds of raw EEG in bipolar banana montage.
No algorithm markup — just clean traces for visual inspection.

Usage:
    conda run -n morgoth python paper_materials/generate_fig0_examples.py
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ── Paths ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_CSV = DATA_DIR / 'labels' / 'segment_labels.csv'
OUT_PATH = PROJECT_DIR / 'paper_materials' / 'figures' / 'fig0_eeg_examples.png'
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Channel definitions ──
MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),   # 0-3: L temporal
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),   # 4-7: R temporal
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),   # 8-11: L parasag
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),   # 12-15: R parasag
    ('Fz', 'Cz'), ('Cz', 'Pz'),                                 # 16-17: Midline
]

# Display order: L temporal, L parasag, midline, R parasag, R temporal
# -1 = blank separator row
DISPLAY_ORDER = [0, 1, 2, 3, -1, 8, 9, 10, 11, -1, 16, 17, -1, 12, 13, 14, 15, -1, 4, 5, 6, 7]

BIPOLAR_LABELS = [f'{a}-{b}' for a, b in BIPOLAR_PAIRS]

FS = 200  # sampling rate

# ── Selected cases ──
# All have >=10 votes and EEG files verified
PANELS = {
    'A': {
        'file': 'sub-S0001114959966_20150425125519.mat',
        'title': 'LPD',
        'desc': 'Clear LPD',
    },
    'B': {
        'file': 'sub-S0001114037812_20130112160354.mat',
        'title': 'GPD',
        'desc': 'Clear GPD',
    },
    'C': {
        'file': 'sub-S0001116263695_20180305213732.mat',
        'title': 'LRDA',
        'desc': 'Clear LRDA',
    },
    'D': {
        'file': 'sub-S0001112827622_20151223154257.mat',
        'title': 'GRDA',
        'desc': 'Clear GRDA',
    },
    'E': {
        'file': 'sub-S0001119806448_20121222234610.mat',
        'title': 'Ambiguous LPD',
        'desc': 'Ambiguous LPD',
    },
    'F': {
        'file': 'sub-S0001114433509_20111031010306.mat',
        'title': 'Ambiguous mixed',
        'desc': 'Ambiguous pattern',
    },
}


def load_eeg(filepath):
    """Load 19-channel monopolar EEG from .mat file."""
    d = sio.loadmat(str(filepath))
    data = d['data'].astype(np.float64)  # (19, 2000)
    fs = int(d['Fs'].item())
    assert data.shape[0] == 19, f"Expected 19 channels, got {data.shape[0]}"
    return data, fs


def mono_to_bipolar(mono_data):
    """Convert 19-channel monopolar to 18-channel bipolar banana montage."""
    ch_idx = {ch: i for i, ch in enumerate(MONO_CHANNELS)}
    bipolar = np.zeros((len(BIPOLAR_PAIRS), mono_data.shape[1]))
    for i, (a, b) in enumerate(BIPOLAR_PAIRS):
        bipolar[i] = mono_data[ch_idx[a]] - mono_data[ch_idx[b]]
    return bipolar


def filter_eeg(data, fs):
    """Apply bandpass 0.5-20 Hz, notch 60 Hz, and detrend."""
    # Detrend each channel
    data = signal.detrend(data, axis=1)

    # Notch filter at 60 Hz
    b_notch, a_notch = signal.iirnotch(60.0, Q=30.0, fs=fs)
    data = signal.filtfilt(b_notch, a_notch, data, axis=1)

    # Bandpass 0.5-20 Hz (4th order Butterworth)
    sos = signal.butter(4, [0.5, 20.0], btype='bandpass', fs=fs, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)

    return data


def clip_data(data, clip_uv=300.0):
    """Clip data to +/- clip_uv."""
    return np.clip(data, -clip_uv, clip_uv)


def get_vote_info(row):
    """Extract vote information from a label row."""
    votes = {
        'lpd': int(row['iiic_vote_lpd']),
        'gpd': int(row['iiic_vote_gpd']),
        'lrda': int(row['iiic_vote_lrda']),
        'grda': int(row['iiic_vote_grda']),
        'seizure': int(row['iiic_vote_seizure']),
        'other': int(row['iiic_vote_other']),
    }
    n_votes = int(row['iiic_n_votes'])
    plurality = row['iiic_plurality']
    frac = row['iiic_plurality_frac']
    pct = int(round(frac * 100))
    return votes, n_votes, plurality, pct


def plot_eeg_panel(ax, bipolar_data, fs, panel_label, subtitle_text):
    """Plot a single EEG panel with bipolar banana montage."""
    n_samples = bipolar_data.shape[1]
    t = np.arange(n_samples) / fs

    # Spacing between channels (in uV)
    spacing = 120.0
    n_display = len(DISPLAY_ORDER)

    y_pos = 0
    yticks = []
    yticklabels = []

    for idx in DISPLAY_ORDER:
        if idx == -1:
            # Separator - just skip a half-space
            y_pos -= spacing * 0.4
            continue
        trace = bipolar_data[idx]
        ax.plot(t, trace + y_pos, color='black', linewidth=0.4, clip_on=True)
        yticks.append(y_pos)
        yticklabels.append(BIPOLAR_LABELS[idx])
        y_pos -= spacing

    # Set axis properties
    ax.set_xlim(0, 10)
    y_top = spacing * 0.8
    y_bottom = y_pos + spacing * 0.4
    ax.set_ylim(y_bottom, y_top)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=7, fontfamily='sans-serif')

    ax.set_xlabel('Time (s)', fontsize=9, fontfamily='sans-serif')
    ax.tick_params(axis='x', labelsize=8)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    # Panel label (A., B., etc.) in top-left
    ax.text(0.02, 0.97, f'{panel_label}.', transform=ax.transAxes,
            fontsize=14, fontweight='bold', fontfamily='sans-serif',
            verticalalignment='top', horizontalalignment='left')

    # Subtitle below panel label
    ax.text(0.02, 0.90, subtitle_text, transform=ax.transAxes,
            fontsize=8.5, fontfamily='sans-serif', fontstyle='italic',
            verticalalignment='top', horizontalalignment='left',
            color='#444444')

    # Scale bar: 100 uV, 1 second — in bottom-right
    bar_x = 8.5
    bar_y = y_bottom + spacing * 0.7
    # Horizontal bar (1 second)
    ax.plot([bar_x, bar_x + 1], [bar_y, bar_y], color='black', linewidth=1.5, clip_on=False)
    # Vertical bar (100 uV)
    ax.plot([bar_x, bar_x], [bar_y, bar_y + 100], color='black', linewidth=1.5, clip_on=False)
    ax.text(bar_x + 0.5, bar_y - spacing * 0.15, '1 s', fontsize=7,
            ha='center', va='top', fontfamily='sans-serif')
    ax.text(bar_x - 0.15, bar_y + 50, '100 \u00b5V', fontsize=7,
            ha='right', va='center', fontfamily='sans-serif', rotation=90)


def main():
    # Load labels
    labels = pd.read_csv(LABELS_CSV)

    # Build figure: 3 rows x 2 columns
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.25, wspace=0.22, left=0.07, right=0.97, top=0.97, bottom=0.03)

    panel_order = ['A', 'B', 'C', 'D', 'E', 'F']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    caption_parts = []
    caption_parts.append("Figure 1. Representative EEG examples of IIIC patterns.\n")

    for panel_label, (row_idx, col_idx) in zip(panel_order, positions):
        info = PANELS[panel_label]
        mat_file = info['file']

        # Get label info
        label_row = labels[labels['mat_file'] == mat_file]
        if len(label_row) == 0:
            print(f"WARNING: {mat_file} not found in labels!")
            continue
        label_row = label_row.iloc[0]
        votes, n_votes, plurality, pct = get_vote_info(label_row)
        expert_freq = label_row.get('expert_freq_hz', None)
        if pd.isna(expert_freq):
            expert_freq = None

        # Subtitle for panel
        subtitle = f"{info['title']} ({pct}% agreement, {n_votes} votes)"

        # Load and process EEG
        eeg_path = EEG_DIR / mat_file
        mono_data, fs = load_eeg(eeg_path)

        # Check if data might already be bipolar (18 channels)
        if mono_data.shape[0] == 18:
            bipolar = mono_data
        else:
            bipolar = mono_to_bipolar(mono_data)

        bipolar = filter_eeg(bipolar, fs)
        bipolar = clip_data(bipolar, clip_uv=300.0)

        # Plot
        ax = axes[row_idx, col_idx]
        plot_eeg_panel(ax, bipolar, fs, panel_label, subtitle)

        # Build caption text
        vote_str = (f"{n_votes} experts: {votes['lpd']} LPD, {votes['gpd']} GPD, "
                    f"{votes['lrda']} LRDA, {votes['grda']} GRDA, "
                    f"{votes['seizure']} seizure, {votes['other']} other")
        freq_str = f" Expert-labeled frequency: {expert_freq:.2f} Hz." if expert_freq else ""

        if panel_label == 'A':
            caption_parts.append(
                f"(A) Clear lateralized periodic discharges (LPD). "
                f"High inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Sharp periodic discharges are visible with clear lateralization, "
                f"most prominent in the left temporal and parasagittal chains."
            )
        elif panel_label == 'B':
            caption_parts.append(
                f"(B) Clear generalized periodic discharges (GPD). "
                f"High inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Bilateral synchronous periodic discharges are visible across all channels "
                f"with a generalized distribution."
            )
        elif panel_label == 'C':
            caption_parts.append(
                f"(C) Clear lateralized rhythmic delta activity (LRDA). "
                f"High inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Rhythmic delta waves are visible with clear lateralization, "
                f"showing a sinusoidal morphology that distinguishes LRDA from the sharper waveforms of LPD."
            )
        elif panel_label == 'D':
            caption_parts.append(
                f"(D) Clear generalized rhythmic delta activity (GRDA). "
                f"High inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Bilateral rhythmic delta activity is present across all channels "
                f"with a generalized, relatively symmetric distribution."
            )
        elif panel_label == 'E':
            caption_parts.append(
                f"(E) Ambiguous lateralized periodic discharges. "
                f"Moderate inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Although classified as LPD by plurality vote, substantial disagreement exists, "
                f"with some experts labeling this as seizure or other, "
                f"illustrating the challenge of distinguishing LPD from ictal patterns."
            )
        elif panel_label == 'F':
            caption_parts.append(
                f"(F) Ambiguous pattern with mixed classification. "
                f"Low inter-rater agreement ({pct}%, {vote_str}).{freq_str} "
                f"Votes are distributed across multiple IIIC categories (GPD, LPD, LRDA, GRDA), "
                f"demonstrating that some EEG patterns do not fit neatly into a single IIIC category "
                f"and that classification difficulty varies substantially across segments."
            )

    # Save
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nFigure saved to: {OUT_PATH}")
    print(f"File size: {OUT_PATH.stat().st_size / 1024:.0f} KB\n")

    # Print caption
    print("=" * 80)
    print("FIGURE CAPTION")
    print("=" * 80)
    for part in caption_parts:
        print(part)
        print()


if __name__ == '__main__':
    main()
