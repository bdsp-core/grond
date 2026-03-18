"""
Interactive EEG segment browser with laterality analysis results.

Usage:
    python browse_results.py [--event lrda|grda|lpd|gpd] [--start N]

Controls:
    Right arrow / N  → Next segment
    Left arrow / P   → Previous segment
    Q / Escape       → Quit
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from mne.filter import notch_filter, filter_data
from scipy.signal import detrend, savgol_filter, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path
import warnings
import hdf5storage
import h5py

warnings.filterwarnings('ignore')

# Channel definitions
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
MONO_CHANNELS = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                 'Fp2','F4','C4','P4','F8','T4','T6','O2','EKG']

LEFT_INDICES  = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_INDICES = [16, 17]

# Region metadata: code → (full name, channel indices, channel labels)
REGION_META = {
    'LF':  ('Left Frontal',          [0, 8, 9, 1],   ['Fp1-F7','Fp1-F3','F3-C3','F7-T3']),
    'RF':  ('Right Frontal',         [4, 5, 12, 13],  ['Fp2-F8','F8-T4','Fp2-F4','F4-C4']),
    'LT':  ('Left Temporal',         [2, 3],          ['T3-T5','T5-O1']),
    'RT':  ('Right Temporal',        [6, 7],          ['T4-T6','T6-O2']),
    'LCP': ('Left Centro-Parietal',  [10],            ['C3-P3']),
    'RCP': ('Right Centro-Parietal', [14],            ['C4-P4']),
    'LO':  ('Left Occipital',        [11],            ['P3-O1']),
    'RO':  ('Right Occipital',       [15],            ['P4-O2']),
}

REGION_ORDER = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']

# Side-agnostic region names (used when side is already stated in laterality)
REGION_BARE = {
    'LF': 'frontal', 'RF': 'frontal',
    'LT': 'temporal', 'RT': 'temporal',
    'LCP': 'centro-parietal', 'RCP': 'centro-parietal',
    'LO': 'occipital', 'RO': 'occipital',
}
LEFT_REGIONS  = ['LF', 'LT', 'LCP', 'LO']
RIGHT_REGIONS = ['RF', 'RT', 'RCP', 'RO']

# Score threshold above which a region is considered "active"
REGION_ACTIVE_THRESHOLD = 2.0


def compute_pointiness_trace(signal_1d, half_win=8):
    """Compute a continuous pointiness trace (prominence^2 / width) at each local max.

    Matches the morgoth-viewer spike_peaks.py metric but computed densely.
    Returns an array the same length as signal_1d (zero where not a local max).
    """
    from scipy.signal import find_peaks
    n = len(signal_1d)
    trace = np.zeros(n)
    peaks, _ = find_peaks(signal_1d)
    for loc in peaks:
        if loc < half_win or loc >= n - half_win:
            continue
        peak_val = signal_1d[loc]
        left_valley = np.min(signal_1d[loc - half_win:loc])
        right_valley = np.min(signal_1d[loc + 1:loc + half_win + 1])
        prom = peak_val - max(left_valley, right_valley)
        if prom <= 0:
            continue
        half_prom_level = peak_val - 0.5 * prom
        width = 0
        for j in range(1, half_win + 1):
            if signal_1d[loc - j] > half_prom_level:
                width += 1
            else:
                break
        for j in range(1, half_win + 1):
            if loc + j < n and signal_1d[loc + j] > half_prom_level:
                width += 1
            else:
                break
        if width > 0:
            trace[loc] = prom ** 2 / width
    return trace


def detect_pd_peaks(signal_1d, fs=200, window_size=200, n_std=3):
    """Re-run adaptive peak detection on one channel, matching pd_detect_alternate logic.

    Only returns peaks if they pass the regularity filter (std of inter-peak
    intervals < 1 second), consistent with what the detector actually uses
    for frequency estimation.  Returns empty array for channels that fail.
    """
    sig_range = np.max(np.abs(signal_1d)) - np.min(np.abs(signal_1d))
    if np.var(signal_1d) > 50 * sig_range:
        return np.array([])
    x = detrend(signal_1d - np.mean(signal_1d))
    x = savgol_filter(x, window_length=10, polyorder=2)
    d_x = np.diff(x)
    # Adaptive peak detection
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(d_x, (window_size // 2, window_size // 2), mode='reflect'),
        window_size
    )
    local_mean = np.mean(windows, axis=1)[:len(d_x)]
    local_std = np.std(windows, axis=1)[:len(d_x)]
    threshold = local_mean + n_std * local_std
    peak_indices = []
    for i in range(1, len(d_x) - 1):
        if d_x[i] > d_x[i - 1] and d_x[i] > d_x[i + 1] and d_x[i] > threshold[i]:
            peak_indices.append(i)
    peaks = np.array(peak_indices)
    # Regularity filter: match pd_detect_alternate's std(intervals) < 1 rule
    if len(peaks) < 2:
        return np.array([])
    intervals = np.diff(peaks / fs)
    if len(intervals) < 2 or np.std(intervals, ddof=1) >= 1.0:
        return np.array([])
    return peaks


def load_mat_file(filepath):
    try:
        return hdf5storage.loadmat(filepath)
    except NotImplementedError:
        with h5py.File(filepath, 'r') as f:
            return {key: f[key][()] for key in f.keys()}


def get_bipolar(segment):
    bipolar_ids = np.array([
        [MONO_CHANNELS.index(bc.split('-')[0]), MONO_CHANNELS.index(bc.split('-')[1])]
        for bc in BIPOLAR_CHANNELS
    ])
    return segment[bipolar_ids[:, 0]] - segment[bipolar_ids[:, 1]]


def freq_band_name(hz):
    if hz < 0.5:
        return 'infra-slow'
    elif hz < 4.0:
        return 'delta'
    elif hz < 8.0:
        return 'theta'
    elif hz < 13.0:
        return 'alpha'
    else:
        return 'beta'


def generate_verbal_description(row):
    """
    Generate a concise clinical verbal description from one results row.
    See DESCRIPTION_RULES.md for full rationale.
    """
    event_type = str(row.get('type_event', '')).upper()
    freq       = row.get('event_frequency', np.nan)
    lat_idx    = row.get('laterality_index', np.nan)
    is_lateralized = event_type in ('LRDA', 'LPD')
    is_generalized = event_type in ('GRDA', 'GPD')

    type_str = event_type if event_type else 'Unknown'
    freq_str = f'at {freq:.1f} Hz' if np.isfinite(freq) else 'at unknown frequency'

    region_scores = {reg: row.get(f'region_{reg}', np.nan) for reg in REGION_ORDER}

    # Recompute LI from region means (equal weight per region) instead of
    # channel-level means, which over-weight frontal (4ch) vs occipital (1ch).
    left_region_vals  = [region_scores[r] for r in LEFT_REGIONS  if np.isfinite(region_scores[r])]
    right_region_vals = [region_scores[r] for r in RIGHT_REGIONS if np.isfinite(region_scores[r])]
    if left_region_vals and right_region_vals:
        left_rmean  = np.mean(left_region_vals)
        right_rmean = np.mean(right_region_vals)
        lat_idx = (right_rmean - left_rmean) / (right_rmean + left_rmean)
    # else lat_idx stays as the CSV value (fallback)

    if is_lateralized:
        # --- Laterality (LRDA / LPD) ---
        # Thresholds (ACNS 2021 qualitative; no numerical cutoff given):
        #   |LI| > 0.15 → unilateral
        #   0.10 < |LI| ≤ 0.15 → bilateral asymmetric
        #   |LI| ≤ 0.10 → bilateral/symmetric
        UNILATERAL_THRESHOLD = 0.15
        BILATERAL_THRESHOLD  = 0.10

        if np.isfinite(lat_idx):
            if lat_idx < -UNILATERAL_THRESHOLD:
                dom_regs = LEFT_REGIONS
                lat_str  = 'unilateral left'
            elif lat_idx > UNILATERAL_THRESHOLD:
                dom_regs = RIGHT_REGIONS
                lat_str  = 'unilateral right'
            elif lat_idx < -BILATERAL_THRESHOLD:
                dom_regs = LEFT_REGIONS
                lat_str  = 'bilateral asymmetric, left-predominant'
            elif lat_idx > BILATERAL_THRESHOLD:
                dom_regs = RIGHT_REGIONS
                lat_str  = 'bilateral asymmetric, right-predominant'
            else:
                dom_regs = LEFT_REGIONS + RIGHT_REGIONS
                lat_str  = 'bilateral/symmetric'
        else:
            dom_regs = LEFT_REGIONS + RIGHT_REGIONS
            lat_str  = 'laterality unknown'

        # Dominant-side regions only, bare names (side already stated in lat_str)
        active = [(reg, scr) for reg, scr in region_scores.items()
                  if np.isfinite(scr) and scr > REGION_ACTIVE_THRESHOLD and reg in dom_regs]
        active.sort(key=lambda x: -x[1])
        top_names = list(dict.fromkeys(REGION_BARE[reg] for reg, _ in active[:2]))

        region_str = ('maximal in the ' + ' and '.join(top_names) + ' regions'
                      if top_names else 'no region clearly dominant')
        return f'{type_str} {freq_str}, {lat_str}; {region_str}.'

    elif is_generalized:
        # --- Regional predominance (GRDA / GPD) — ACNS 2021 ---
        # Groups: frontal, occipital, midline (central/vertex)
        frontal_score  = np.nanmean([region_scores.get('LF', np.nan),
                                     region_scores.get('RF', np.nan)])
        occipital_score = np.nanmean([region_scores.get('LO', np.nan),
                                      region_scores.get('RO', np.nan)])
        mid_fz = row.get('score_Fz-Cz', np.nan)
        mid_cz = row.get('score_Cz-Pz', np.nan)
        mid_fz = 1.0 if not np.isfinite(mid_fz) else mid_fz
        mid_cz = 1.0 if not np.isfinite(mid_cz) else mid_cz
        midline_score = np.mean([mid_fz, mid_cz])

        groups = {
            'frontally predominant':  frontal_score,
            'occipitally predominant': occipital_score,
            'midline predominant':    midline_score,
        }
        best_label = max(groups, key=groups.get)
        best_score = groups[best_label]

        region_str = best_label if best_score > REGION_ACTIVE_THRESHOLD else 'no regional predominance'
        return f'{type_str} {freq_str}, {region_str}.'

    else:
        return f'{type_str} {freq_str}, laterality unknown.'


EVENT_TYPES = ['lrda', 'grda', 'lpd', 'gpd']


class EEGBrowser:
    def __init__(self, event_type, start_idx=0):
        self.fs = 200

        script_dir = Path(__file__).resolve().parent
        self.repo_root = script_dir.parent if script_dir.name == 'code' else script_dir

        # Pre-load all available datasets
        self.datasets = {}
        for et in EVENT_TYPES:
            results_file = self.repo_root / 'results' / f'{et}_laterality_results.csv'
            data_dir = self.repo_root / 'data' / 'dataset_eeg' / et
            if results_file.exists() and data_dir.exists():
                self.datasets[et] = {
                    'df': pd.read_csv(results_file),
                    'data_dir': data_dir,
                }

        if not self.datasets:
            print("Error: no results files found.")
            sys.exit(1)

        if event_type not in self.datasets:
            event_type = list(self.datasets.keys())[0]
            print(f"Requested type not available, using {event_type}")

        self.event_type = event_type
        self.idx = max(0, min(start_idx, len(self.datasets[event_type]['df']) - 1))

        self.fig = plt.figure(figsize=(20, 11))
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._update_title()
        self._draw()
        plt.show()

    def _update_title(self):
        available = '  |  '.join(
            f'[{i+1}] {et.upper()}' for i, et in enumerate(EVENT_TYPES)
            if et in self.datasets
        )
        self.fig.canvas.manager.set_window_title(
            f'{self.event_type.upper()} Browser — ← → navigate  |  {available}  |  Q quit'
        )

    def _switch_event_type(self, new_type):
        if new_type in self.datasets and new_type != self.event_type:
            self.event_type = new_type
            self.idx = 0
            self._update_title()
            self._draw()

    def _on_key(self, event):
        if event.key in ('right', 'n'):
            self.idx = min(self.idx + 1, len(self.datasets[self.event_type]['df']) - 1)
            self._draw()
        elif event.key in ('left', 'p'):
            self.idx = max(self.idx - 1, 0)
            self._draw()
        elif event.key == 'tab':
            available = [et for et in EVENT_TYPES if et in self.datasets]
            next_idx = (available.index(self.event_type) + 1) % len(available)
            self._switch_event_type(available[next_idx])
        elif event.key == '1':
            self._switch_event_type('lrda')
        elif event.key == '2':
            self._switch_event_type('grda')
        elif event.key == '3':
            self._switch_event_type('lpd')
        elif event.key == '4':
            self._switch_event_type('gpd')
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    def _load_segment(self, filename):
        mat_file = self.datasets[self.event_type]['data_dir'] / f'{filename}.mat'
        if not mat_file.exists():
            return None, None
        mat = load_mat_file(str(mat_file))
        try:
            segment = mat['data_50sec']
        except (KeyError, Exception):
            segment = mat['data']
        segment = notch_filter(segment, self.fs, 60, n_jobs=1, verbose="ERROR")
        segment = filter_data(segment, self.fs, 0.5, 40, n_jobs=1, verbose="ERROR")
        seg_bi = get_bipolar(segment)
        return segment, seg_bi

    def _draw(self):
        self.fig.clf()

        row = self.datasets[self.event_type]['df'].iloc[self.idx]
        filename = row['files']

        _, seg_bi = self._load_segment(filename)
        if seg_bi is None:
            self.fig.text(0.5, 0.5, f"Could not load: {filename}", ha='center', fontsize=14)
            self.fig.canvas.draw_idle()
            return

        scores   = np.array([row.get(f'score_{ch}', np.nan) for ch in BIPOLAR_CHANNELS])
        ch_freqs = np.array([row.get(f'freq_{ch}', np.nan)  for ch in BIPOLAR_CHANNELS])
        detected = scores > 1.0

        gs = GridSpec(21, 6, width_ratios=[1, 1, 1, 0.3, 0.05, 0.65],
                      hspace=0.08, wspace=0.3,
                      left=0.06, right=0.98, top=0.90, bottom=0.07)

        time_vec = np.linspace(0, seg_bi.shape[1] / self.fs, seg_bi.shape[1])

        # Detect peaks per channel for PD types
        is_pd = self.event_type in ('lpd', 'gpd')
        channel_peaks = {}
        if is_pd:
            for i in range(18):
                pks = detect_pd_peaks(seg_bi[i, :], self.fs)
                if len(pks) > 0:
                    channel_peaks[i] = pks

        for i in range(18):
            ax = self.fig.add_subplot(gs[i + 1, 0:3])
            color = '#1a6dd4' if detected[i] else '#555555'
            lw    = 1.2      if detected[i] else 0.7
            ax.plot(time_vec, seg_bi[i, :], color=color, linewidth=lw)

            # Draw peak markers (triangles)
            if i in channel_peaks:
                pk_idx = channel_peaks[i]
                pk_idx = pk_idx[pk_idx < len(time_vec)]
                ax.plot(time_vec[pk_idx], seg_bi[i, pk_idx], 'v',
                        color='#e03030', markersize=4, alpha=0.7)

            if i in LEFT_INDICES:
                ax.set_facecolor('#ffe8e8')
            elif i in RIGHT_INDICES:
                ax.set_facecolor('#e8e8ff')
            else:
                ax.set_facecolor('#f0f0f0')

            score_str = f'{scores[i]:.2f}'  if np.isfinite(scores[i]) else '—'
            freq_str  = f'{ch_freqs[i]:.1f}Hz' if np.isfinite(ch_freqs[i]) else ''
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
            ac_ax = self.fig.add_subplot(gs[i + 1, 3])
            b_lp, a_lp = butter(4, 15.0 / (self.fs / 2), btype='low')
            try:
                sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
            except ValueError:
                sig_lp = seg_bi[i, :]

            pt = compute_pointiness_trace(sig_lp)
            pt = gaussian_filter1d(pt, sigma=self.fs * 0.02)
            ptm = pt - np.mean(pt)
            max_lag_pt = min(4 * self.fs, len(ptm) - 1)
            acf_pt = np.correlate(ptm, ptm, mode='full')
            acf_pt = acf_pt[len(ptm) - 1:][:max_lag_pt + 1]
            if acf_pt[0] > 0:
                acf_pt = acf_pt / acf_pt[0]

            lag_pt = np.arange(len(acf_pt)) / self.fs
            ac_ax.plot(lag_pt, acf_pt, color='#ee7722', linewidth=0.7)
            ac_ax.axhline(0, color='gray', linewidth=0.3)

            # Red dashed: Method A frequency
            if np.isfinite(ch_freqs[i]) and ch_freqs[i] > 0:
                period = 1.0 / ch_freqs[i]
                if period <= 4.0:
                    ac_ax.axvline(period, color='#e03030', linewidth=0.7,
                                  linestyle='--', alpha=0.8)

            # Orange dashed: ACF peak (Method B frequency)
            min_lag_samples = int(0.4 * self.fs)
            for k in range(min_lag_samples + 1, len(acf_pt) - 1):
                if acf_pt[k] > acf_pt[k - 1] and acf_pt[k] > acf_pt[k + 1] and acf_pt[k] > 0.2:
                    ac_ax.axvline(k / self.fs, color='#ee7722', linewidth=0.7,
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

        # Region-based laterality index (equal weight per region)
        region_vals = {reg: row.get(f'region_{reg}', np.nan) for reg in REGION_ORDER}
        lrv = [region_vals[r] for r in LEFT_REGIONS  if np.isfinite(region_vals[r])]
        rrv = [region_vals[r] for r in RIGHT_REGIONS if np.isfinite(region_vals[r])]
        if lrv and rrv:
            lr_mean, rr_mean = np.mean(lrv), np.mean(rrv)
            lat_idx = (rr_mean - lr_mean) / (rr_mean + lr_mean)
        else:
            lat_idx = row.get('laterality_index', np.nan)
            lr_mean = row.get('left_mean_score', np.nan)
            rr_mean = row.get('right_mean_score', np.nan)
        lat_str   = f'{lat_idx:+.3f}' if np.isfinite(lat_idx) else 'N/A'
        type_event = row.get('type_event', 'N/A')
        self.fig.text(0.5, 0.95,
                      f'{self.event_type.upper()} — {filename}  '
                      f'({self.idx + 1}/{len(self.datasets[self.event_type]["df"])})    '
                      f'Type: {type_event}    Laterality: {lat_str}',
                      ha='center', fontsize=11, fontweight='bold')

        # Verbal description banner
        desc = generate_verbal_description(row)
        self.fig.text(0.5, 0.92, desc,
                      ha='center', fontsize=9, color='#222244',
                      style='italic',
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffff0',
                                edgecolor='#aaaaaa', alpha=0.9))

        # ---- Right panel: summary stats ----
        info_ax = self.fig.add_subplot(gs[1:8, 5])
        info_ax.axis('off')

        freq  = row.get('event_frequency', np.nan)
        spat  = row.get('spatial_extent', np.nan)

        info_lines = [
            f'Type:      {type_event}',
            f'Frequency: {freq:.3f} Hz' if np.isfinite(freq) else 'Frequency: N/A',
            f'Spatial:   {spat:.3f} ({int(round(spat*18))}/18 ch)' if np.isfinite(spat) else 'Spatial:   N/A',
            f'Lat. idx:  {lat_str}  (region-based)',
            f'Left mean: {lr_mean:.3f}' if np.isfinite(lr_mean) else 'Left mean: N/A',
            f'Right mean:{rr_mean:.3f}' if np.isfinite(rr_mean) else 'Right mean:N/A',
        ]
        info_ax.text(0.0, 1.0, '\n'.join(info_lines),
                     fontsize=8.5, fontfamily='monospace',
                     va='top', ha='left', transform=info_ax.transAxes)

        # ---- Region scores with full names ----
        reg_ax = self.fig.add_subplot(gs[8:15, 5])
        reg_ax.axis('off')

        reg_lines = ['Regions (mean score / threshold=2.0):']
        region_scores_vals = {}
        for reg in REGION_ORDER:
            val = row.get(f'region_{reg}', np.nan)
            region_scores_vals[reg] = val
            full, _, ch_list = REGION_META[reg]
            n_ch = len(ch_list)
            val_str = f'{val:.2f}' if np.isfinite(val) else '—'
            marker = ' ◀' if (np.isfinite(val) and val > REGION_ACTIVE_THRESHOLD) else ''
            reg_lines.append(f'  {reg:4s} {val_str:>5}  {full} ({n_ch}ch){marker}')

        reg_ax.text(0.0, 1.0, '\n'.join(reg_lines),
                    fontsize=7.5, fontfamily='monospace',
                    va='top', ha='left', transform=reg_ax.transAxes)

        # ---- Laterality bar ----
        bar_ax = self.fig.add_subplot(gs[15:17, 5])
        bar_ax.set_title('Laterality Index', fontsize=8)
        if np.isfinite(lat_idx):
            bar_color = '#cc3333' if lat_idx < 0 else '#3333cc'
            bar_ax.barh(0, lat_idx, color=bar_color, height=0.6, alpha=0.7)
            bar_ax.axvline(0, color='black', linewidth=0.8)
            # threshold lines
            for t in [-0.3, -0.1, 0.1, 0.3]:
                bar_ax.axvline(t, color='gray', linewidth=0.5, linestyle=':')
            bar_ax.set_xlim(-1, 1)
            bar_ax.set_yticks([])
            bar_ax.set_xlabel('← Left    Right →', fontsize=7)
        else:
            bar_ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=bar_ax.transAxes)
            bar_ax.set_yticks([])

        # ---- Per-channel score bars ----
        score_ax = self.fig.add_subplot(gs[17:21, 5])
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

        # Navigation hint
        self.fig.text(0.5, 0.01,
                      '← → navigate  |  Tab cycle type  |  1 LRDA  2 GRDA  3 LPD  4 GPD  |  Q quit',
                      ha='center', fontsize=9, color='gray')

        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description='Browse EEG segments with laterality results')
    parser.add_argument('--event', type=str, default='lrda',
                        choices=['lrda', 'grda', 'lpd', 'gpd'],
                        help='Event type to browse (default: lrda)')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting segment index (default: 0)')
    args = parser.parse_args()
    EEGBrowser(args.event, args.start)


if __name__ == '__main__':
    main()
