"""
Generate 100 test-case images (25 per pattern: LPD, GPD, LRDA, GRDA)
from expert-labeled segments in /Volumes/sanD_photos/IIIC/.

Each image shows the central 10 seconds of the recording (around the
expert-annotated event time), run through the appropriate detector,
and displayed with the verbal description banner.

Raw EEG segments are also saved alongside the images for quick re-use.
"""

import sys, os, ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mne.filter import notch_filter, filter_data
from scipy.signal import detrend, savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
from datetime import datetime
import warnings
import hdf5storage
import scipy.io

warnings.filterwarnings('ignore')

# Add code dirs to path
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from pd_detect_alternate import pd_detect_alternate
from rda1b_fft import rda1b_fft
from browse_results import (
    BIPOLAR_CHANNELS, MONO_CHANNELS, LEFT_INDICES, RIGHT_INDICES,
    MIDLINE_INDICES, REGION_META, REGION_ORDER, REGION_BARE,
    LEFT_REGIONS, RIGHT_REGIONS, REGION_ACTIVE_THRESHOLD,
    generate_verbal_description, get_bipolar, detect_pd_peaks,
    compute_pointiness_trace,
)

IIIC_DIR = Path('/Volumes/sanD_photos/IIIC')
SEGMENTS_DIR = IIIC_DIR / 'segments_raw'
OUTPUT_DIR = CODE_DIR.parent / 'test_case_images'
OUTPUT_DIR.mkdir(exist_ok=True)


def select_examples(n=25):
    """Select n expert-labeled examples per pattern type from diverse patients."""
    df = pd.read_excel(str(IIIC_DIR / 'list_events_20241129.xlsx'))
    df['parsed'] = df['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else None)
    df = df[df['parsed'].notna()]
    df['total'] = df['parsed'].apply(sum)
    for i, name in enumerate(['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']):
        df[name] = df['parsed'].apply(lambda x, idx=i: x[idx])

    available = set(os.path.splitext(f)[0] for f in os.listdir(str(SEGMENTS_DIR)))

    selections = {}
    for ptype in ['lpd', 'gpd', 'lrda', 'grda']:
        df[ptype + '_frac'] = df[ptype] / df['total']
        subset = df[(df[ptype + '_frac'] > 0.5) & (df['file_name'].isin(available))].sort_values(
            ptype, ascending=False)
        seen_mrn = set()
        picks = []
        for _, row in subset.iterrows():
            if row['bdsp_mrn'] not in seen_mrn:
                seen_mrn.add(row['bdsp_mrn'])
                picks.append(row)
                if len(picks) == n:
                    break
        selections[ptype] = picks
        print(f'{ptype.upper()}: selected {len(picks)} examples from {len(subset)} available')
    return selections


def load_segment(file_name):
    """Load a .mat file from IIIC segments_raw."""
    filepath = SEGMENTS_DIR / f'{file_name}.mat'
    mat = hdf5storage.loadmat(str(filepath))
    data = mat['data']
    fs = int(mat['Fs'].ravel()[0])
    # data may be (channels, samples) or (samples, channels)
    if data.shape[0] > data.shape[1]:
        data = data.T
    return data, fs


def extract_central_10s(data, fs):
    """Extract the central 10 seconds of a recording."""
    total_samples = data.shape[1]
    center = total_samples // 2
    half = 5 * fs  # 5 seconds each side
    start = max(0, center - half)
    end = min(total_samples, center + half)
    return data[:, start:end]


def run_detector(segment, fs, pattern_type):
    """Run the appropriate detector and return a results dict matching browse_results CSV format."""
    if pattern_type in ('lpd', 'gpd'):
        result = pd_detect_alternate(segment, fs, pk_detect='apd')
        score_key = 'channel_pd_scores'
    else:
        result, _, _ = rda1b_fft(segment, fs, channel_filter=0)
        score_key = 'channel_rda_scores'

    row = {
        'files': 'temp',
        'type_event': result.get('type_event', np.nan),
        'event_frequency': result.get('event_frequency', np.nan),
        'acf_frequency': result.get('acf_frequency', np.nan),
        'spatial_extent': result.get('spatial_extent', np.nan),
        'laterality_index': result.get('laterality_index', np.nan),
        'left_mean_score': result.get('left_mean_score', np.nan),
        'right_mean_score': result.get('right_mean_score', np.nan),
    }

    ch_scores = result.get(score_key, {})
    ch_freqs = result.get('channel_frequencies', {})
    for ch in BIPOLAR_CHANNELS:
        row[f'score_{ch}'] = ch_scores.get(ch, np.nan)
        row[f'freq_{ch}'] = ch_freqs.get(ch, np.nan)

    for reg, score in result.get('region_scores', {}).items():
        row[f'region_{reg}'] = score

    return row


def draw_figure(row, seg_bi, fs, pattern_type, title_extra=''):
    """Draw the EEG browser figure and return the figure object."""
    fig = plt.figure(figsize=(20, 11))

    scores = np.array([row.get(f'score_{ch}', np.nan) for ch in BIPOLAR_CHANNELS])
    ch_freqs = np.array([row.get(f'freq_{ch}', np.nan) for ch in BIPOLAR_CHANNELS])
    detected = scores > 1.0

    gs = GridSpec(21, 6, width_ratios=[1, 1, 1, 0.3, 0.05, 0.65],
                  hspace=0.08, wspace=0.3,
                  left=0.06, right=0.98, top=0.90, bottom=0.07)

    time_vec = np.linspace(0, seg_bi.shape[1] / fs, seg_bi.shape[1])

    # Detect peaks for PD types
    is_pd = pattern_type in ('lpd', 'gpd')
    channel_peaks = {}  # Method A peaks (red)
    pointiness_peaks = {}  # Method B peaks (green)
    if is_pd:
        b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
        for i in range(18):
            # Method A peaks
            pks = detect_pd_peaks(seg_bi[i, :], fs)
            if len(pks) > 0:
                channel_peaks[i] = pks
            # Method B: pointiness peaks (15Hz lowpass)
            try:
                sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
            except ValueError:
                sig_lp = seg_bi[i, :]
            pt = compute_pointiness_trace(sig_lp)
            pt = gaussian_filter1d(pt, sigma=fs * 0.02)
            mx = np.max(pt)
            if mx > 0:
                from scipy.signal import find_peaks as sp_find_peaks
                pk_b, _ = sp_find_peaks(pt, height=mx * 0.3, distance=int(0.2 * fs))
                if len(pk_b) > 0:
                    pointiness_peaks[i] = pk_b

    for i in range(18):
        ax = fig.add_subplot(gs[i + 1, 0:3])
        color = '#1a6dd4' if detected[i] else '#555555'
        lw = 1.2 if detected[i] else 0.7
        ax.plot(time_vec, seg_bi[i, :], color=color, linewidth=lw)

        if i in channel_peaks:
            pk_idx = channel_peaks[i]
            pk_idx = pk_idx[pk_idx < len(time_vec)]
            ax.plot(time_vec[pk_idx], seg_bi[i, pk_idx], 'v',
                    color='#e03030', markersize=4, alpha=0.7)

        if i in pointiness_peaks:
            pk_b_idx = pointiness_peaks[i]
            pk_b_idx = pk_b_idx[pk_b_idx < len(time_vec)]
            ax.plot(time_vec[pk_b_idx], seg_bi[i, pk_b_idx], '^',
                    color='#22aa22', markersize=4, alpha=0.7)

        if i in LEFT_INDICES:
            ax.set_facecolor('#ffe8e8')
        elif i in RIGHT_INDICES:
            ax.set_facecolor('#e8e8ff')
        else:
            ax.set_facecolor('#f0f0f0')

        score_str = f'{scores[i]:.2f}' if np.isfinite(scores[i]) else '—'
        freq_str = f'{ch_freqs[i]:.1f}Hz' if np.isfinite(ch_freqs[i]) else ''
        label = f'{BIPOLAR_CHANNELS[i]}  [{score_str}] {freq_str}'
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=75, va='center')

        ax.tick_params(axis='y', labelsize=5)
        if i < 17:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Mini ACF of pointiness trace
        ac_ax = fig.add_subplot(gs[i + 1, 3])
        b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
        try:
            sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
        except ValueError:
            sig_lp = seg_bi[i, :]

        pt = compute_pointiness_trace(sig_lp)
        pt = gaussian_filter1d(pt, sigma=fs * 0.02)
        ptm = pt - np.mean(pt)
        max_lag_pt = min(4 * fs, len(ptm) - 1)
        acf_pt = np.correlate(ptm, ptm, mode='full')
        acf_pt = acf_pt[len(ptm) - 1:][:max_lag_pt + 1]
        if acf_pt[0] > 0:
            acf_pt = acf_pt / acf_pt[0]

        lag_pt = np.arange(len(acf_pt)) / fs
        ac_ax.plot(lag_pt, acf_pt, color='#ee7722', linewidth=0.7)
        ac_ax.axhline(0, color='gray', linewidth=0.3)

        # Red dashed: Method A frequency
        if np.isfinite(ch_freqs[i]) and ch_freqs[i] > 0:
            period = 1.0 / ch_freqs[i]
            if period <= 4.0:
                ac_ax.axvline(period, color='#e03030', linewidth=0.7,
                              linestyle='--', alpha=0.8)

        # Orange dashed: ACF peak (Method B frequency)
        min_lag_samples = int(0.4 * fs)
        for k in range(min_lag_samples + 1, len(acf_pt) - 1):
            if acf_pt[k] > acf_pt[k - 1] and acf_pt[k] > acf_pt[k + 1] and acf_pt[k] > 0.2:
                ac_ax.axvline(k / fs, color='#ee7722', linewidth=0.7,
                              linestyle='--', alpha=0.8)
                break
        ac_ax.set_xlim(0, 4)
        ac_ax.set_ylim(-0.5, 1)
        ac_ax.set_yticks([])
        ac_ax.tick_params(axis='x', labelsize=4, pad=1)
        if i < 17:
            ac_ax.set_xticklabels([])
        else:
            ac_ax.set_xlabel('Lag (s)', fontsize=5)
        for spine in ac_ax.spines.values():
            spine.set_visible(False)

    # Region-based laterality index
    region_vals = {reg: row.get(f'region_{reg}', np.nan) for reg in REGION_ORDER}
    lrv = [region_vals[r] for r in LEFT_REGIONS if np.isfinite(region_vals[r])]
    rrv = [region_vals[r] for r in RIGHT_REGIONS if np.isfinite(region_vals[r])]
    if lrv and rrv:
        lr_mean, rr_mean = np.mean(lrv), np.mean(rrv)
        lat_idx = (rr_mean - lr_mean) / (rr_mean + lr_mean)
    else:
        lat_idx = row.get('laterality_index', np.nan)
        lr_mean = row.get('left_mean_score', np.nan)
        rr_mean = row.get('right_mean_score', np.nan)
    lat_str = f'{lat_idx:+.3f}' if np.isfinite(lat_idx) else 'N/A'
    type_event = row.get('type_event', 'N/A')

    fig.text(0.5, 0.95,
             f'{pattern_type.upper()} — {title_extra}    '
             f'Type: {type_event}    Laterality: {lat_str}',
             ha='center', fontsize=11, fontweight='bold')

    desc = generate_verbal_description(row)
    fig.text(0.5, 0.92, desc,
             ha='center', fontsize=9, color='#222244',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffff0',
                       edgecolor='#aaaaaa', alpha=0.9))

    # Right panel: summary stats
    info_ax = fig.add_subplot(gs[1:8, 5])
    info_ax.axis('off')
    freq = row.get('event_frequency', np.nan)
    acf_freq = row.get('acf_frequency', np.nan)
    spat = row.get('spatial_extent', np.nan)
    freq_a_str = f'{freq:.2f} Hz' if np.isfinite(freq) else 'N/A'
    freq_b_str = f'{acf_freq:.2f} Hz' if np.isfinite(acf_freq) else 'N/A'
    info_lines = [
        f'Type:      {type_event}',
        f'Freq (A):  {freq_a_str}  [peak det]',
        f'Freq (B):  {freq_b_str}  [ACF]',
        f'Spatial:   {spat:.3f} ({int(round(spat * 18))}/18 ch)' if np.isfinite(spat) else 'Spatial:   N/A',
        f'Lat. idx:  {lat_str}  (region-based)',
        f'Left mean: {lr_mean:.3f}' if np.isfinite(lr_mean) else 'Left mean: N/A',
        f'Right mean:{rr_mean:.3f}' if np.isfinite(rr_mean) else 'Right mean:N/A',
    ]
    info_ax.text(0.0, 1.0, '\n'.join(info_lines),
                 fontsize=8.5, fontfamily='monospace',
                 va='top', ha='left', transform=info_ax.transAxes)

    # Region scores
    reg_ax = fig.add_subplot(gs[8:15, 5])
    reg_ax.axis('off')
    reg_lines = ['Regions (mean score / threshold=2.0):']
    for reg in REGION_ORDER:
        val = row.get(f'region_{reg}', np.nan)
        full, _, ch_list = REGION_META[reg]
        n_ch = len(ch_list)
        val_str = f'{val:.2f}' if np.isfinite(val) else '—'
        marker = ' ◀' if (np.isfinite(val) and val > REGION_ACTIVE_THRESHOLD) else ''
        reg_lines.append(f'  {reg:4s} {val_str:>5}  {full} ({n_ch}ch){marker}')
    reg_ax.text(0.0, 1.0, '\n'.join(reg_lines),
                fontsize=7.5, fontfamily='monospace',
                va='top', ha='left', transform=reg_ax.transAxes)

    # Laterality bar
    bar_ax = fig.add_subplot(gs[15:17, 5])
    bar_ax.set_title('Laterality Index', fontsize=8)
    if np.isfinite(lat_idx):
        bar_color = '#cc3333' if lat_idx < 0 else '#3333cc'
        bar_ax.barh(0, lat_idx, color=bar_color, height=0.6, alpha=0.7)
        bar_ax.axvline(0, color='black', linewidth=0.8)
        for t in [-0.3, -0.1, 0.1, 0.3]:
            bar_ax.axvline(t, color='gray', linewidth=0.5, linestyle=':')
        bar_ax.set_xlim(-1, 1)
        bar_ax.set_yticks([])
        bar_ax.set_xlabel('← Left    Right →', fontsize=7)
    else:
        bar_ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=bar_ax.transAxes)
        bar_ax.set_yticks([])

    # Per-channel score bars
    score_ax = fig.add_subplot(gs[17:21, 5])
    score_ax.set_title('Per-Channel Scores', fontsize=8)
    valid_scores = np.where(np.isfinite(scores), scores, 0)
    bar_colors = []
    for i in range(18):
        if i in LEFT_INDICES:
            bar_colors.append('#cc5555')
        elif i in RIGHT_INDICES:
            bar_colors.append('#5555cc')
        else:
            bar_colors.append('#888888')
    score_ax.bar(range(18), valid_scores, color=bar_colors, alpha=0.7, width=0.8)
    score_ax.axhline(REGION_ACTIVE_THRESHOLD, color='green', linewidth=1,
                     linestyle='--', alpha=0.7, label=f'active ({REGION_ACTIVE_THRESHOLD})')
    score_ax.axhline(1.0, color='orange', linewidth=0.8,
                     linestyle=':', alpha=0.7, label='baseline (1.0)')
    score_ax.set_xticks(range(18))
    score_ax.set_xticklabels(
        [ch.split('-')[1][:2] for ch in BIPOLAR_CHANNELS],
        fontsize=5, rotation=45)
    score_ax.tick_params(axis='y', labelsize=6)
    score_ax.legend(fontsize=6, loc='upper right')

    return fig


def draw_pointiness_figure(row, seg_bi, fs, pattern_type, title_extra=''):
    """Draw the same layout but showing |d²x/dt²| pointiness signal instead of raw EEG."""
    fig = plt.figure(figsize=(20, 11))

    scores = np.array([row.get(f'score_{ch}', np.nan) for ch in BIPOLAR_CHANNELS])
    ch_freqs = np.array([row.get(f'freq_{ch}', np.nan) for ch in BIPOLAR_CHANNELS])
    detected = scores > 1.0

    gs = GridSpec(21, 6, width_ratios=[1, 1, 1, 0.3, 0.05, 0.65],
                  hspace=0.08, wspace=0.3,
                  left=0.06, right=0.98, top=0.90, bottom=0.07)

    # Pre-compute pointiness traces for shared scaling
    # Apply 15 Hz lowpass before feature extraction (optimal from grid search)
    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    pt_traces = []
    pt_norm_traces = []
    for i in range(18):
        try:
            sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
        except ValueError:
            sig_lp = seg_bi[i, :]
        pt = compute_pointiness_trace(sig_lp)
        pt = gaussian_filter1d(pt, sigma=fs * 0.02)
        pt_traces.append(pt)
        # Percentile normalization (window=10s i.e. full segment, pct=90)
        n = len(pt)
        win_samples = max(3, int(10.0 * fs))
        half_win = win_samples // 2
        step = max(1, win_samples // 20)
        sample_pts = np.arange(0, n, step)
        pvals = np.array([
            np.percentile(pt[max(0, p - half_win):min(n, p + half_win)], 90)
            for p in sample_pts
        ])
        running_pct = np.interp(np.arange(n), sample_pts, pvals)
        running_pct = np.maximum(running_pct, 1e-10)
        pt_norm_traces.append(pt / running_pct)

    pt_maxes = [np.max(t) for t in pt_traces]
    global_pt_ymax = max(pt_maxes) * 1.05 if max(pt_maxes) > 0 else 1.0
    pn_maxes = [np.max(t) for t in pt_norm_traces]
    global_pn_ymax = max(pn_maxes) * 1.05 if max(pn_maxes) > 0 else 1.0

    for i in range(18):
        ax = fig.add_subplot(gs[i + 1, 0:3])
        color = '#1a6dd4' if detected[i] else '#555555'

        # Plot raw pointiness trace (orange, faded)
        pt = pt_traces[i]
        time_pt = np.linspace(0, len(pt) / fs, len(pt))
        ax.plot(time_pt, pt, color='#ee7722', linewidth=0.6, alpha=0.35)
        ax.fill_between(time_pt, 0, pt, color='#ee7722', alpha=0.08)
        ax.set_ylim(0, global_pt_ymax)

        # Overlay percentile-normalized trace (blue, on secondary y-axis)
        ax2 = ax.twinx()
        pt_n = pt_norm_traces[i]
        ax2.plot(time_pt[:len(pt_n)], pt_n, color='#2288dd', linewidth=0.9, alpha=0.9)
        ax2.set_ylim(0, global_pn_ymax)
        ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_visible(False)

        if i in LEFT_INDICES:
            ax.set_facecolor('#ffe8e8')
        elif i in RIGHT_INDICES:
            ax.set_facecolor('#e8e8ff')
        else:
            ax.set_facecolor('#f0f0f0')

        score_str = f'{scores[i]:.2f}' if np.isfinite(scores[i]) else '—'
        freq_str = f'{ch_freqs[i]:.1f}Hz' if np.isfinite(ch_freqs[i]) else ''
        label = f'{BIPOLAR_CHANNELS[i]}  [{score_str}] {freq_str}'
        ax.set_ylabel(label, fontsize=7, rotation=0, labelpad=75, va='center')
        ax.tick_params(axis='y', labelsize=5)
        if i < 17:
            ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # ACF of pointiness (same as main figure)
        # ACF of pointiness trace (uses 15Hz lowpass signal)
        ac_ax = fig.add_subplot(gs[i + 1, 3])
        b_lp2, a_lp2 = butter(4, 15.0 / (fs / 2), btype='low')
        try:
            sig_lp2 = filtfilt(b_lp2, a_lp2, seg_bi[i, :])
        except ValueError:
            sig_lp2 = seg_bi[i, :]

        pt_acf = compute_pointiness_trace(sig_lp2)
        pt_acf = gaussian_filter1d(pt_acf, sigma=fs * 0.02)
        ptm = pt_acf - np.mean(pt_acf)
        max_lag_pt = min(4 * fs, len(ptm) - 1)
        acf_pt = np.correlate(ptm, ptm, mode='full')
        acf_pt = acf_pt[len(ptm) - 1:][:max_lag_pt + 1]
        if acf_pt[0] > 0:
            acf_pt = acf_pt / acf_pt[0]

        lag_pt = np.arange(len(acf_pt)) / fs
        ac_ax.plot(lag_pt, acf_pt, color='#ee7722', linewidth=0.7)
        ac_ax.axhline(0, color='gray', linewidth=0.3)

        if np.isfinite(ch_freqs[i]) and ch_freqs[i] > 0:
            period = 1.0 / ch_freqs[i]
            if period <= 4.0:
                ac_ax.axvline(period, color='#e03030', linewidth=0.7,
                              linestyle='--', alpha=0.8)

        min_lag_samples = int(0.4 * fs)
        for k in range(min_lag_samples + 1, len(acf_pt) - 1):
            if acf_pt[k] > acf_pt[k - 1] and acf_pt[k] > acf_pt[k + 1] and acf_pt[k] > 0.2:
                ac_ax.axvline(k / fs, color='#ee7722', linewidth=0.7,
                              linestyle='--', alpha=0.8)
                break
        ac_ax.set_xlim(0, 4)
        ac_ax.set_ylim(-0.5, 1)
        ac_ax.set_yticks([])
        ac_ax.tick_params(axis='x', labelsize=4, pad=1)
        if i < 17:
            ac_ax.set_xticklabels([])
        else:
            ac_ax.set_xlabel('Lag (s)', fontsize=5)
        for spine in ac_ax.spines.values():
            spine.set_visible(False)

    # Title
    region_vals = {reg: row.get(f'region_{reg}', np.nan) for reg in REGION_ORDER}
    lrv = [region_vals[r] for r in LEFT_REGIONS if np.isfinite(region_vals[r])]
    rrv = [region_vals[r] for r in RIGHT_REGIONS if np.isfinite(region_vals[r])]
    if lrv and rrv:
        lat_idx = (np.mean(rrv) - np.mean(lrv)) / (np.mean(rrv) + np.mean(lrv))
    else:
        lat_idx = row.get('laterality_index', np.nan)
    lat_str = f'{lat_idx:+.3f}' if np.isfinite(lat_idx) else 'N/A'
    type_event = row.get('type_event', 'N/A')

    fig.text(0.5, 0.95,
             f'{pattern_type.upper()} — {title_extra}    '
             f'Type: {type_event}    Laterality: {lat_str}    '
             f'[POINTINESS: |d\u00b2x/dt\u00b2|]',
             ha='center', fontsize=11, fontweight='bold')

    desc = generate_verbal_description(row)
    fig.text(0.5, 0.92, desc,
             ha='center', fontsize=9, color='#222244',
             style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffff0',
                       edgecolor='#aaaaaa', alpha=0.9))

    # Legend for dashed lines
    fig.text(0.5, 0.01,
             'ACF of |d\u00b2x/dt\u00b2|  —  Red dashed: peak detector period  |  Green dashed: ACF peak period',
             ha='center', fontsize=9, color='gray')

    return fig


def main():
    selections = select_examples(n=25)

    for ptype, examples in selections.items():
        print(f'\nProcessing {ptype.upper()} ({len(examples)} examples)...')
        for i, ex_row in enumerate(examples):
            file_name = ex_row['file_name']
            votes = int(ex_row[ptype])
            total = int(ex_row['total'])
            print(f'  [{i + 1}/25] {file_name}  ({votes}/{total} votes)', end='', flush=True)

            try:
                data, fs = load_segment(file_name)
                seg_10s = extract_central_10s(data, fs)

                # Save raw 10s segment
                seg_filename = f'{ptype}_{i + 1:02d}_{file_name}'
                raw_eeg_dir = OUTPUT_DIR / 'raw_eeg'
                raw_eeg_dir.mkdir(exist_ok=True)
                raw_path = raw_eeg_dir / f'{seg_filename}.mat'
                scipy.io.savemat(str(raw_path), {'data': seg_10s, 'Fs': fs})

                # Filter for display
                seg_filtered = notch_filter(seg_10s.astype(float), fs, 60, n_jobs=1, verbose="ERROR")
                seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
                seg_bi = get_bipolar(seg_filtered)

                # Run detector on filtered monopolar data
                result_row = run_detector(seg_10s, fs, ptype)
                result_row['files'] = seg_filename

                # Draw and save main image
                fig = draw_figure(result_row, seg_bi, fs, ptype,
                                  title_extra=f'{file_name} ({votes}/{total} votes)')
                png_path = OUTPUT_DIR / f'{seg_filename}.png'
                fig.savefig(str(png_path), dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Draw and save pointiness image
                fig2 = draw_pointiness_figure(result_row, seg_bi, fs, ptype,
                                              title_extra=f'{file_name} ({votes}/{total} votes)')
                png_path2 = OUTPUT_DIR / f'{seg_filename}_pointiness.png'
                fig2.savefig(str(png_path2), dpi=150, bbox_inches='tight')
                plt.close(fig2)
                print('  ✓')

            except Exception as e:
                print(f'  FAILED: {e}')
                continue

    print(f'\nDone! Images saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
