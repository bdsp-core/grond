"""
LRDA Labeling Tool — frequency, laterality, and wave morphology annotation.

For each LRDA segment, pre-computes NVO (Narrowband Variance Optimization)
across a grid of candidate frequencies.  The HTML viewer lets MW:
  1. Pick a frequency (buttons or slider) — see per-channel VE heatmap update
  2. Label laterality (Left / Right / Bilateral)
  3. Mark wave triplets: onset → peak → offset of each repeating wave
  4. View the bandpass-filtered overlay at the selected frequency

Uses variance_explained_search() from rda_optimization_harness.py as the
core algorithm for identifying LRDA channels and estimating frequency.

Usage:
    conda run -n foe python code/generate_lrda_labeler.py
"""

import sys
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

# ── Constants ─────────────────────────────────────────────────────────
FS = 200
DURATION = 10.0
N_SAMPLES = 2000
LOWPASS_HZ = 20.0

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',   # 0-3   left temporal
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',   # 4-7   right temporal
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',   # 8-11  left parasagittal
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',   # 12-15 right parasagittal
    'Fz-Cz', 'Cz-Pz',                       # 16-17 midline
]

LEFT_CHANNELS = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_CHANNELS = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_CHANNELS = [16, 17]

# Fine frequency grid for VE scoring (0.05 Hz resolution)
FREQ_MIN, FREQ_MAX, FREQ_STEP = 0.5, 3.5, 0.05
FREQ_GRID = [round(FREQ_MIN + i * FREQ_STEP, 2)
             for i in range(int((FREQ_MAX - FREQ_MIN) / FREQ_STEP) + 1)]

# Frequency buttons (coarser — these are the ones we store filtered signals for)
FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(2, 15)]  # 0.5 to 3.5

# Bandpass half-width for narrowband filtering
NB_BW = 0.3  # Hz on each side


# ── Narrowband Bandpass Variance Optimization ─────────────────────────

def narrowband_filter(seg_bi, freq, bw=NB_BW):
    """Bandpass filter around freq ± bw Hz. Returns (18, T) filtered signal."""
    from scipy.signal import sosfiltfilt
    lo = max(freq - bw, 0.1)
    hi = min(freq + bw, FS / 2 - 0.1)
    if lo >= hi:
        return np.zeros_like(seg_bi)
    sos = butter(4, [lo, hi], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, seg_bi, axis=1)


def nvo_bandpass_grid(seg_bi):
    """Narrowband bandpass VE search with sliding window.

    For each candidate frequency f, bandpass filters at f ± bw, then
    finds the best contiguous window (≥ 2 sec) where the narrowband
    explains the most variance. This handles intermittent LRDA that
    may only be present for part of the 10-second segment.

    The whole-segment VE is also stored for display purposes.

    Returns:
        ve_matrix: (n_freqs, 18) — whole-segment VE per channel per freq
        best_freq: float — best frequency (from sliding-window scoring)
        best_ve: (18,) — whole-segment VE at best frequency
    """
    n_ch, n_samp = seg_bi.shape
    total_var = np.array([np.var(seg_bi[ch]) for ch in range(n_ch)])
    total_var[total_var < 1e-12] = 1e-12

    # Sliding window parameters
    win_sec = 3.0  # window length in seconds
    win_samp = int(win_sec * FS)
    hop_samp = int(0.5 * FS)  # 0.5 sec hop
    n_wins = max(1, (n_samp - win_samp) // hop_samp + 1)

    ve_matrix = np.zeros((len(FREQ_GRID), n_ch))
    scores = np.zeros(len(FREQ_GRID))

    for fi, f in enumerate(FREQ_GRID):
        filtered = narrowband_filter(seg_bi, f)

        # Whole-segment VE (for display)
        for ch in range(n_ch):
            ve_matrix[fi, ch] = np.var(filtered[ch]) / total_var[ch]

        # Sliding-window VE: find the window where the narrowband
        # best explains the signal on the top channels
        best_win_score = 0.0
        for wi in range(n_wins):
            s = wi * hop_samp
            e = s + win_samp
            if e > n_samp:
                break

            # Per-channel windowed VE
            win_ve = np.zeros(n_ch)
            for ch in range(n_ch):
                seg_var = np.var(seg_bi[ch, s:e])
                if seg_var < 1e-12:
                    continue
                win_ve[ch] = np.var(filtered[ch, s:e]) / seg_var

            # Laterality-aware: top-3 per hemisphere
            left_ve = np.sort(win_ve[LEFT_CHANNELS])[::-1]
            right_ve = np.sort(win_ve[RIGHT_CHANNELS])[::-1]
            win_score = max(np.mean(left_ve[:3]), np.mean(right_ve[:3]))
            if win_score > best_win_score:
                best_win_score = win_score

        scores[fi] = best_win_score

    best_idx = int(np.argmax(scores))
    best_freq = FREQ_GRID[best_idx]
    best_ve = ve_matrix[best_idx]

    return ve_matrix, best_freq, best_ve


def compute_button_filtered(seg_bi):
    """Precompute narrowband filtered signals for each button frequency.

    Uses narrow bandpass at f ± 0.3Hz. Only stores channels where
    the filtered amplitude is meaningful (>= 15% of strongest channel).

    Returns dict: freq_str -> sparse channel dict of arrays.
    """
    n_ds = 400
    indices = np.linspace(0, seg_bi.shape[1] - 1, n_ds).astype(int)

    result = {}
    for freq in FREQ_BUTTONS:
        filtered = narrowband_filter(seg_bi, freq)
        # Only store channels with meaningful amplitude
        ch_amp = np.array([np.std(filtered[ch]) for ch in range(18)])
        amp_threshold = np.max(ch_amp) * 0.15
        ds = {}
        for ch in range(18):
            if ch_amp[ch] >= amp_threshold:
                ds[str(ch)] = filtered[ch, indices]
        result[str(freq)] = ds
    return result


# ── Data loading ──────────────────────────────────────────────────────

def preprocess_segment(seg_mono):
    """Monopolar (20, 2000) → preprocessed bipolar (18, 2000)."""
    from mne.filter import notch_filter, filter_data
    seg_bi = np.array(fcn_getBanana(seg_mono), dtype=np.float64)
    seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
    seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')
    return seg_bi


def load_segment(mat_file, montage='monopolar'):
    """Load an EEG segment from a .mat file."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if montage == 'monopolar':
        seg = seg[:20, :N_SAMPLES]
        seg = preprocess_segment(seg)
    else:
        seg = seg[:18, :N_SAMPLES]
    return seg


def downsample(arr, target_len):
    """Downsample for JSON embedding."""
    if arr.ndim == 1:
        n = len(arr)
        if n <= target_len:
            return arr.tolist()
        indices = np.linspace(0, n - 1, target_len).astype(int)
        return arr[indices].tolist()
    else:
        n = arr.shape[1]
        if n <= target_len:
            return arr.tolist()
        indices = np.linspace(0, n - 1, target_len).astype(int)
        return arr[:, indices].tolist()


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  LRDA Labeling Tool Generator")
    print("  NVO-assisted frequency, laterality, and wave morphology labeling")
    print("=" * 70)

    # Load segments database
    df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    # Get LRDA segments
    lrda_segs = df_seg[df_seg['subtype'] == 'lrda'].copy()
    print(f"Total LRDA segments: {len(lrda_segs)} from {lrda_segs['patient_id'].nunique()} patients")

    # Find which patients already have frequency annotations
    lrda_seg_ids = set(lrda_segs['segment_id'])
    annotated = df_ann[
        (df_ann['segment_id'].isin(lrda_seg_ids)) &
        (df_ann['frequency_hz'] > 0)
    ]['patient_id'].unique()

    print(f"Already annotated: {len(annotated)} patients")

    # Take one segment per patient (prefer first), prioritize unannotated
    lrda_segs_dedup = lrda_segs.sort_values('segment_id').drop_duplicates(
        subset='patient_id', keep='first')

    # Sort: unannotated first, then by patient_id
    lrda_segs_dedup = lrda_segs_dedup.copy()
    lrda_segs_dedup['annotated'] = lrda_segs_dedup['patient_id'].isin(annotated)
    lrda_segs_dedup = lrda_segs_dedup.sort_values(
        ['annotated', 'patient_id'], ascending=[True, True])

    n_unannotated = (~lrda_segs_dedup['annotated']).sum()
    print(f"Cases to label: {n_unannotated} unannotated + {len(lrda_segs_dedup) - n_unannotated} already annotated")

    # Build cases
    print(f"\nProcessing {len(lrda_segs_dedup)} cases (NVO + bandpass)...")
    cases_data = []
    n_skipped = 0

    # Lowpass at 5Hz for delta-focused display (LRDA is 0.5-3.5Hz)
    DISPLAY_LP = 5.0
    b_lp5, a_lp5 = butter(4, DISPLAY_LP / (FS / 2), btype='low')

    for i, (_, row) in enumerate(lrda_segs_dedup.iterrows()):
        pid = row['patient_id']
        seg_id = row['segment_id']
        mat_file = row['mat_file']
        montage = row.get('montage', 'monopolar')

        seg_bi = load_segment(mat_file, montage)
        if seg_bi is None or seg_bi.shape != (18, N_SAMPLES):
            n_skipped += 1
            continue

        # Lowpass at 5Hz for display — shows delta waves clearly
        seg_display = np.zeros_like(seg_bi)
        for ch in range(18):
            try:
                seg_display[ch] = filtfilt(b_lp5, a_lp5, seg_bi[ch])
                seg_display[ch] = detrend(seg_display[ch], type='linear')
            except ValueError:
                seg_display[ch] = seg_bi[ch]

        # Run narrowband bandpass VE search on the 5Hz-lowpassed signal
        # so VE denominator is delta-band energy only (not theta/alpha/beta)
        ve_matrix, best_freq, best_ve = nvo_bandpass_grid(seg_display)

        # Precompute narrowband filtered signals on the 5Hz-lowpassed signal
        btn_filtered = compute_button_filtered(seg_display)

        # Serialize filtered signals
        nb_signals = {}
        for freq_str, ch_dict in btn_filtered.items():
            ch_data = {}
            for ch_str, arr in ch_dict.items():
                ch_data[ch_str] = [round(float(v), 1) for v in arr]
            nb_signals[freq_str] = ch_data

        # Laterality estimate from VE
        left_ve = np.mean(best_ve[LEFT_CHANNELS])
        right_ve = np.mean(best_ve[RIGHT_CHANNELS])
        if left_ve + right_ve > 0:
            lat_index = (right_ve - left_ve) / (left_ve + right_ve)
        else:
            lat_index = 0.0
        est_laterality = 'left' if lat_index < -0.15 else ('right' if lat_index > 0.15 else 'bilateral')

        case = {
            'patient_id': str(pid),
            'segment_id': str(seg_id),
            'est_freq': round(float(best_freq), 2),
            'est_laterality': est_laterality,
            'lat_index': round(float(lat_index), 3),
            'annotated': bool(row['annotated']),
            'eeg_data': downsample(seg_display, 500),
            've_matrix': [[round(float(v), 3) for v in ve_matrix[fi]]
                          for fi in range(len(FREQ_GRID))],
            'freq_grid': [round(f, 2) for f in FREQ_GRID],
            'best_ve': [round(float(v), 4) for v in best_ve],
            'nb_signals': nb_signals,
        }
        cases_data.append(case)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(lrda_segs_dedup)} processed")

    print(f"  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")

    # Build HTML
    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)

    out_path = OUT_DIR / 'lrda_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")

    # Open in browser
    import subprocess
    subprocess.run(['open', str(out_path)])
    print("  Opened in browser")
    print("=" * 70)


# ── HTML Builder ──────────────────────────────────────────────────────

def build_html(cases_data):
    freq_btns_json = json.dumps(FREQ_BUTTONS)
    freq_grid_json = json.dumps([round(f, 2) for f in FREQ_GRID])
    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LRDA Labeling Tool — Frequency, Laterality &amp; Wave Morphology</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a1a; color: #eee; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; border-bottom: 2px solid #444;
    flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; }}

  #progress-bar-wrap {{ width: 100%; height: 6px; background: #333; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}

  #mode-indicator {{
    font-size: 18px; font-weight: bold; padding: 6px 20px;
    text-align: center; letter-spacing: 2px;
  }}
  .mode-onset {{ background: #1a2a3a; color: #66aaff; }}
  .mode-peak {{ background: #3a2a1a; color: #ffaa44; }}
  .mode-offset {{ background: #2a1a3a; color: #cc66ff; }}
  .mode-nav {{ background: #1a1a3a; color: #6688ff; }}
  .mode-delete {{ background: #3a1a1a; color: #ff4444; }}

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #freq-buttons {{
    display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 16px;
    background: #252525; border-bottom: 1px solid #333; align-items: center;
  }}
  #freq-buttons label {{ color: #aaa; font-size: 13px; margin-right: 8px; font-weight: bold; }}
  .freq-btn {{
    padding: 6px 10px; border: 1px solid #555; border-radius: 4px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 13px; font-weight: bold; min-width: 42px; text-align: center;
    transition: all 0.15s;
  }}
  .freq-btn:hover {{ background: #444; border-color: #888; color: #fff; }}
  .freq-btn.active {{ background: #2a5a2a; border-color: #44cc88; color: #44ff66; }}
  .freq-btn.est {{ border-color: #ff9800; box-shadow: 0 0 4px #ff9800; }}

  #laterality-buttons {{
    display: flex; gap: 6px; padding: 6px 16px;
    background: #252525; border-bottom: 1px solid #333; align-items: center;
  }}
  #laterality-buttons label {{ color: #aaa; font-size: 13px; margin-right: 8px; font-weight: bold; }}
  .lat-btn {{
    padding: 6px 16px; border: 2px solid #555; border-radius: 4px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 14px; font-weight: bold; transition: all 0.15s;
  }}
  .lat-btn:hover {{ background: #444; }}
  .lat-btn.active-left {{ background: #2a2a5a; border-color: #6688ff; color: #88aaff; }}
  .lat-btn.active-right {{ background: #5a2a2a; border-color: #ff6644; color: #ff8866; }}
  .lat-btn.active-bilateral {{ background: #3a3a1a; border-color: #ccaa44; color: #eebb44; }}
  .lat-btn.est {{ box-shadow: 0 0 4px #ff9800; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}
  #ve-canvas {{ display: block; margin: 4px auto 0 auto; cursor: default; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}

  #save-status {{ color: #44cc44; font-size: 13px; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }}

  #freq-info {{
    font-size: 14px; color: #ff9800; padding: 4px 16px; background: #2a2510;
  }}

  #wave-info {{
    font-size: 13px; color: #aaa; padding: 4px 16px; background: #222;
    border-bottom: 1px solid #333;
  }}
  #wave-info .triplet-count {{ color: #44ff66; font-weight: bold; }}
  #wave-info .pending {{ color: #ffaa44; }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#66aaff;">LRDA Labeling Tool</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / 0</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="skipCase()" style="border-color:#ff6644; color:#ff6644; background:#3a2a2a;">Skip/Reject <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export Labels <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item">NVO freq: <strong id="info-est-freq" style="color:#ff9800;">--</strong></span>
  <span class="info-item">Selected freq: <strong id="info-sel-freq" style="color:#44ff66;">--</strong></span>
  <span class="info-item">Laterality: <strong id="info-laterality" style="color:#88aaff;">--</strong></span>
  <span class="info-item">Lat index: <strong id="info-lat-index">--</strong></span>
  <span class="info-item">Waves: <strong id="info-wave-count" style="color:#cc66ff;">0</strong></span>
  <span class="info-item">IPI freq: <strong id="info-ipi-freq">--</strong></span>
</div>

<div id="freq-buttons">
  <label>Freq:</label>
</div>

<div id="laterality-buttons">
  <label>Side:</label>
  <button class="lat-btn" data-lat="left" onclick="setLaterality('left')">LEFT (L)</button>
  <button class="lat-btn" data-lat="right" onclick="setLaterality('right')">RIGHT (R)</button>
  <button class="lat-btn" data-lat="bilateral" onclick="setLaterality('bilateral')">BILATERAL (B)</button>
  <span id="lat-est-hint" style="font-size:12px; color:#777; margin-left:12px;"></span>
</div>

<div id="freq-info"></div>

<div id="wave-info">
  Waves: <span class="triplet-count" id="wi-complete">0</span> complete |
  <span class="pending" id="wi-pending">--</span>
  &nbsp;&nbsp; Click sequence: <span class="key" style="background:#2a4a6a;">Onset</span> →
  <span class="key" style="background:#5a3a1a;">Peak</span> →
  <span class="key" style="background:#3a1a5a;">Offset</span> (auto-cycles)
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="ve-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">W</span> Wave-mark mode (onset→peak→offset cycle) &nbsp;&nbsp;
  <span class="key">D</span> Delete mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">X</span> Skip/Reject &nbsp;&nbsp;
  <span class="key">L</span> Left &nbsp;
  <span class="key">R</span> Right &nbsp;
  <span class="key">B</span> Bilateral &nbsp;&nbsp;
  <span class="key">F</span> Set freq from IPI &nbsp;&nbsp;
  <span class="key">1-7</span> Quick freq &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate (auto-save) &nbsp;&nbsp;
  <span class="key">C</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};
const FREQ_GRID = {freq_grid_json};

const CHANNEL_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const LEFT_CHS = [0,1,2,3,8,9,10,11];
const RIGHT_CHS = [4,5,6,7,12,13,14,15];
const MIDLINE_CHS = [16,17];
// Display order: L-lateral, L-parasag, midline, R-parasag, R-lateral
const DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1200;
const EEG_HEIGHT = 750;
const VE_HEIGHT = 100;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 40;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = (EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM);

// ── State ──
let idx = 0;
let labeled = new Set();
let mode = 'nav';  // nav, wave, delete
let wavePhase = 0; // 0=onset, 1=peak, 2=offset
let selectedFreq = null;
let selectedLaterality = null;
let hoverMarker = -1;

// Wave triplets: array of objects with onset, peak, offset times
let waveTriplets = [];
let pendingTriplet = {{}};  // partially filled triplet
let undoStack = [];

// ── Persistence ──
const STORAGE_KEY = 'lrda_labeler_v1';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

let reviewed = new Set();

// ── Build frequency buttons ──
(function() {{
  const container = document.getElementById('freq-buttons');
  for (const freq of FREQ_BUTTONS) {{
    const btn = document.createElement('button');
    btn.className = 'freq-btn';
    btn.textContent = freq.toFixed(1);
    btn.dataset.freq = freq;
    btn.onclick = () => selectFreq(freq);
    container.appendChild(btn);
  }}
}})();

// ── Display channel layout: L-lat, L-para, mid, R-para, R-lat ──
function getDisplayChannels() {{
  return DISPLAY_ORDER.map(i => i < 0 ? {{ idx: -1, name: '' }} : {{ idx: i, name: CHANNEL_NAMES[i] }});
}}
const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

// ── Coordinate transforms ──
function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}
function xToTime(x) {{ return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }}

// ── Get per-channel VE at current freq ──
function getCurrentVE() {{
  const c = CASES[idx];
  if (!selectedFreq) return c.best_ve;
  // Find closest grid index
  let bestGi = 0, bestDist = Infinity;
  for (let gi = 0; gi < FREQ_GRID.length; gi++) {{
    const d = Math.abs(FREQ_GRID[gi] - selectedFreq);
    if (d < bestDist) {{ bestDist = d; bestGi = gi; }}
  }}
  return c.ve_matrix[bestGi];
}}

// ── Push/pop undo ──
function pushUndo() {{
  undoStack.push({{
    triplets: JSON.parse(JSON.stringify(waveTriplets)),
    pending: JSON.parse(JSON.stringify(pendingTriplet)),
    phase: wavePhase,
  }});
  if (undoStack.length > 100) undoStack.shift();
}}
function undo() {{
  if (undoStack.length === 0) return;
  const state = undoStack.pop();
  waveTriplets = state.triplets;
  pendingTriplet = state.pending;
  wavePhase = state.phase;
  redraw();
}}

// ── Select frequency ──
function selectFreq(freq) {{
  selectedFreq = freq;
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('active');
    if (Math.abs(parseFloat(btn.dataset.freq) - freq) < 0.01) btn.classList.add('active');
  }});
  document.getElementById('freq-info').textContent =
    'Selected freq=' + freq.toFixed(1) + ' Hz — channel VE heatmap updated';
  redraw();
}}

// ── Set laterality ──
function setLaterality(lat) {{
  selectedLaterality = lat;
  document.querySelectorAll('.lat-btn').forEach(btn => {{
    btn.classList.remove('active-left', 'active-right', 'active-bilateral');
    if (btn.dataset.lat === lat) btn.classList.add('active-' + lat);
  }});
  updateInfo();
}}

// ── Marker finding (for delete mode) ──
function findNearestWavePoint(x) {{
  // Returns {{tripletIdx, pointType, dist}} or null
  let best = null;
  for (let ti = 0; ti < waveTriplets.length; ti++) {{
    for (const pt of ['onset', 'peak', 'offset']) {{
      if (waveTriplets[ti][pt] == null) continue;
      const mx = timeToX(waveTriplets[ti][pt]);
      const dist = Math.abs(mx - x);
      if (dist < 20 && (!best || dist < best.dist)) {{
        best = {{ tripletIdx: ti, pointType: pt, dist: dist }};
      }}
    }}
  }}
  // Also check pending
  for (const pt of ['onset', 'peak', 'offset']) {{
    if (pendingTriplet[pt] == null) continue;
    const mx = timeToX(pendingTriplet[pt]);
    const dist = Math.abs(mx - x);
    if (dist < 20 && (!best || dist < best.dist)) {{
      best = {{ tripletIdx: -1, pointType: pt, dist: dist }};
    }}
  }}
  return best;
}}

// ── Drawing ──
function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = c.eeg_data;
  const nSamples = eegData[0].length;
  const ve = getCurrentVE();

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Draw per-channel VE indicator bars on the right margin
  const VE_BAR_W = 14;
  const veBarX = PLOT_RIGHT + 2;
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const veVal = ve[ch.idx];
    const barH = chSpacing * 0.8;
    const fillH = barH * Math.min(veVal / 0.4, 1.0);  // 40% VE = full bar

    // Background
    ctx.fillStyle = '#eee';
    ctx.fillRect(veBarX, yCenter - barH / 2, VE_BAR_W, barH);

    // Filled portion (bottom-up)
    let color;
    if (LEFT_CHS.includes(ch.idx)) color = 'rgba(50, 100, 255, 0.7)';
    else if (RIGHT_CHS.includes(ch.idx)) color = 'rgba(255, 80, 50, 0.7)';
    else color = 'rgba(100, 200, 100, 0.7)';
    ctx.fillStyle = color;
    ctx.fillRect(veBarX, yCenter + barH / 2 - fillH, VE_BAR_W, fillH);
  }}

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // EEG traces
  ctx.strokeStyle = '#000000';
  ctx.lineWidth = 0.7;
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    ctx.beginPath();
    for (let si = 0; si < nSamples; si++) {{
      const x = PLOT_LEFT + (si / (nSamples - 1)) * PLOT_W;
      let val = trace[si];
      val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
      const y = yCenter - val * Z_SCALE * chSpacing;
      if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Draw per-channel narrowband filtered signal overlay
  if (selectedFreq && c.nb_signals) {{
    // Find closest button frequency
    let closestBtn = FREQ_BUTTONS[0];
    let closestDist = Infinity;
    for (const fb of FREQ_BUTTONS) {{
      const d = Math.abs(fb - selectedFreq);
      if (d < closestDist) {{ closestDist = d; closestBtn = fb; }}
    }}
    const nbData = c.nb_signals[String(closestBtn)];
    if (nbData) {{
      // Get VE at closest grid freq for alpha scaling
      let bestGi = 0, bestGiDist = Infinity;
      for (let gi = 0; gi < FREQ_GRID.length; gi++) {{
        const d = Math.abs(FREQ_GRID[gi] - selectedFreq);
        if (d < bestGiDist) {{ bestGiDist = d; bestGi = gi; }}
      }}
      const veAtFreq = c.ve_matrix[bestGi];

      for (let di = 0; di < N_DISPLAY; di++) {{
        const ch = DISPLAY_CHANNELS[di];
        if (ch.idx < 0) continue;
        // nbData is sparse: keys are channel index strings
        const trace = nbData[String(ch.idx)];
        if (!trace) continue;  // no data for this channel (VE too low)

        const veVal = veAtFreq[ch.idx];
        const yCenter = MARGIN_TOP + chSpacing * (di + 1);
        const nSamp = trace.length;
        const alpha = Math.min(0.4 + veVal * 3.0, 0.9);

        let color;
        if (LEFT_CHS.includes(ch.idx)) color = `rgba(50, 120, 255, ${{alpha.toFixed(2)}})`;
        else if (RIGHT_CHS.includes(ch.idx)) color = `rgba(255, 80, 50, ${{alpha.toFixed(2)}})`;
        else color = `rgba(255, 180, 0, ${{alpha.toFixed(2)}})`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        for (let si = 0; si < nSamp; si++) {{
          const x = PLOT_LEFT + (si / (nSamp - 1)) * PLOT_W;
          let val = trace[si];
          val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
          const y = yCenter - val * Z_SCALE * chSpacing;
          if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
      }}
    }}
  }}

  // Channel labels with VE values
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const veVal = ve[ch.idx];
    ctx.fillStyle = veVal > 0.1 ? '#000' : '#888';
    ctx.fillText(ch.name + ' ' + (veVal * 100).toFixed(0) + '%', PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', timeToX(s), EEG_HEIGHT - MARGIN_BOTTOM + 4);
  }}

  // Title
  ctx.fillStyle = '#000';
  ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const latStr = selectedLaterality || c.est_laterality;
  const title = c.patient_id + '  |  NVO=' + c.est_freq.toFixed(1) + ' Hz  |  ' + latStr.toUpperCase() +
    '  |  LI=' + c.lat_index.toFixed(2);
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Draw wave triplets
  drawWaveTriplets(ctx);
}}

function drawWaveTriplets(ctx) {{
  const COLORS = {{
    onset: 'rgba(100, 170, 255, 0.8)',
    peak: 'rgba(255, 170, 50, 0.9)',
    offset: 'rgba(200, 100, 255, 0.8)',
  }};
  const DASH = {{
    onset: [6, 3],
    peak: [],
    offset: [3, 3],
  }};

  // Draw complete triplets with shaded regions
  for (let ti = 0; ti < waveTriplets.length; ti++) {{
    const trip = waveTriplets[ti];
    if (trip.onset != null && trip.offset != null) {{
      // Shaded region onset → offset
      const x1 = timeToX(trip.onset);
      const x2 = timeToX(trip.offset);
      ctx.fillStyle = 'rgba(180, 140, 255, 0.08)';
      ctx.fillRect(x1, MARGIN_TOP, x2 - x1, EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM);
    }}

    for (const pt of ['onset', 'peak', 'offset']) {{
      if (trip[pt] == null) continue;
      const x = timeToX(trip[pt]);
      if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

      ctx.setLineDash(DASH[pt]);
      ctx.strokeStyle = COLORS[pt];
      ctx.lineWidth = pt === 'peak' ? 2.5 : 1.5;
      ctx.beginPath();
      ctx.moveTo(x, MARGIN_TOP);
      ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM);
      ctx.stroke();
      ctx.setLineDash([]);

      // Label
      ctx.fillStyle = COLORS[pt];
      ctx.font = '9px Consolas, Monaco, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      const label = pt[0].toUpperCase() + (ti + 1);
      ctx.fillText(label + ' ' + trip[pt].toFixed(2) + 's', x, MARGIN_TOP - 2);
    }}
  }}

  // Draw pending triplet
  for (const pt of ['onset', 'peak', 'offset']) {{
    if (pendingTriplet[pt] == null) continue;
    const x = timeToX(pendingTriplet[pt]);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    ctx.setLineDash(DASH[pt]);
    ctx.strokeStyle = COLORS[pt];
    ctx.lineWidth = pt === 'peak' ? 2.5 : 1.5;
    ctx.globalAlpha = 0.6;
    ctx.beginPath();
    ctx.moveTo(x, MARGIN_TOP);
    ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1.0;

    ctx.fillStyle = COLORS[pt];
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(pt[0].toUpperCase() + '? ' + pendingTriplet[pt].toFixed(2) + 's', x, MARGIN_TOP - 2);
  }}
}}

function drawVE() {{
  const canvas = document.getElementById('ve-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = VE_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const ve = getCurrentVE();

  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, EEG_WIDTH, VE_HEIGHT);

  const barH = 14;
  const barY = 8;
  const barW = (PLOT_W - 20) / 18;

  // Title
  ctx.fillStyle = '#aaa';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('Per-channel Variance Explained (VE%)', PLOT_LEFT, barY - 2);

  // Draw bars
  for (let ch = 0; ch < 18; ch++) {{
    const x = PLOT_LEFT + ch * barW + 1;
    const veVal = ve[ch];
    const barFill = Math.min(veVal / 0.4, 1.0);  // scale: 40% VE = full bar

    // Background
    ctx.fillStyle = '#333';
    ctx.fillRect(x, barY + 14, barW - 2, barH);

    // Fill
    let color;
    if (LEFT_CHS.includes(ch)) color = `rgba(80, 130, 255, ${{0.3 + barFill * 0.7}})`;
    else if (RIGHT_CHS.includes(ch)) color = `rgba(255, 100, 80, ${{0.3 + barFill * 0.7}})`;
    else color = `rgba(100, 200, 100, ${{0.3 + barFill * 0.7}})`;
    ctx.fillStyle = color;
    ctx.fillRect(x, barY + 14, (barW - 2) * barFill, barH);

    // Label
    ctx.fillStyle = '#ccc';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const shortName = CHANNEL_NAMES[ch].replace('Fp', 'F').substring(0, 5);
    ctx.fillText(shortName, x + barW / 2 - 1, barY + 30);
    ctx.fillText((veVal * 100).toFixed(0) + '%', x + barW / 2 - 1, barY + 40);
  }}

  // Left/Right/Midline labels
  ctx.font = '10px monospace';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#6688ff';
  ctx.textAlign = 'left';
  ctx.fillText('L-Temp', PLOT_LEFT, barY + 54);
  ctx.fillStyle = '#ff6644';
  ctx.fillText('R-Temp', PLOT_LEFT + 4 * barW, barY + 54);
  ctx.fillStyle = '#6688ff';
  ctx.fillText('L-Para', PLOT_LEFT + 8 * barW, barY + 54);
  ctx.fillStyle = '#ff6644';
  ctx.fillText('R-Para', PLOT_LEFT + 12 * barW, barY + 54);
  ctx.fillStyle = '#88cc88';
  ctx.fillText('Mid', PLOT_LEFT + 16 * barW, barY + 54);

  // Summary stats
  const leftMean = LEFT_CHS.reduce((s, ch) => s + ve[ch], 0) / LEFT_CHS.length;
  const rightMean = RIGHT_CHS.reduce((s, ch) => s + ve[ch], 0) / RIGHT_CHS.length;
  ctx.fillStyle = '#aaa';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(
    'L=' + (leftMean * 100).toFixed(1) + '%  R=' + (rightMean * 100).toFixed(1) + '%  ' +
    'LI=' + ((rightMean - leftMean) / Math.max(leftMean + rightMean, 0.001)).toFixed(2),
    PLOT_RIGHT, barY + 54
  );
}}

// ── Update info panel ──
function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-est-freq').textContent = c.est_freq.toFixed(1) + ' Hz';
  document.getElementById('info-sel-freq').textContent = selectedFreq ? selectedFreq.toFixed(1) + ' Hz' : '--';
  document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;
  document.getElementById('info-lat-index').textContent = c.lat_index.toFixed(2);
  document.getElementById('info-wave-count').textContent = waveTriplets.length;

  // IPI from wave peaks
  if (waveTriplets.length >= 2) {{
    const peaks = waveTriplets.filter(t => t.peak != null).map(t => t.peak).sort((a, b) => a - b);
    if (peaks.length >= 2) {{
      const ipis = [];
      for (let i = 1; i < peaks.length; i++) ipis.push(peaks[i] - peaks[i - 1]);
      const medIPI = ipis.sort((a, b) => a - b)[Math.floor(ipis.length / 2)];
      const ipiFreq = medIPI > 0 ? (1 / medIPI).toFixed(2) : '--';
      document.getElementById('info-ipi-freq').textContent = ipiFreq + ' Hz';
    }} else {{
      document.getElementById('info-ipi-freq').textContent = '--';
    }}
  }} else {{
    document.getElementById('info-ipi-freq').textContent = '--';
  }}

  // Counter and progress
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';
  document.getElementById('labeled-count').textContent = labeled.size + ' labeled';

  // Highlight est freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('est');
    if (Math.abs(parseFloat(btn.dataset.freq) - c.est_freq) < 0.13) btn.classList.add('est');
  }});

  // Laterality estimate hint
  document.getElementById('lat-est-hint').textContent =
    'NVO suggests: ' + c.est_laterality + ' (LI=' + c.lat_index.toFixed(2) + ')';
  document.querySelectorAll('.lat-btn').forEach(btn => {{
    btn.classList.remove('est');
    if (btn.dataset.lat === c.est_laterality) btn.classList.add('est');
  }});

  // Wave info
  document.getElementById('wi-complete').textContent = waveTriplets.length;
  const phases = ['onset', 'peak', 'offset'];
  const hasPending = Object.keys(pendingTriplet).length > 0 &&
    (pendingTriplet.onset != null || pendingTriplet.peak != null);
  if (mode === 'wave') {{
    document.getElementById('wi-pending').textContent =
      hasPending ? 'Next click: ' + phases[wavePhase] : 'Click to mark onset';
  }} else {{
    document.getElementById('wi-pending').textContent = '';
  }}
}}

function updateModeIndicator() {{
  const el = document.getElementById('mode-indicator');
  const phases = ['onset', 'peak', 'offset'];
  if (mode === 'wave') {{
    const p = phases[wavePhase];
    el.textContent = 'WAVE MODE — click to place ' + p.toUpperCase();
    el.className = 'mode-' + p;
  }} else if (mode === 'delete') {{
    el.textContent = 'DELETE MODE — click near marker to remove wave';
    el.className = 'mode-delete';
  }} else {{
    el.textContent = 'NAVIGATE MODE';
    el.className = 'mode-nav';
  }}
}}

// ── Auto-save ──
function autoSave() {{
  const c = CASES[idx];
  allLabels[c.patient_id] = {{
    triplets: waveTriplets.map(t => ({{ onset: t.onset, peak: t.peak, offset: t.offset }})),
    selected_freq: selectedFreq,
    est_freq: c.est_freq,
    laterality: selectedLaterality || c.est_laterality,
    est_laterality: c.est_laterality,
    lat_index: c.lat_index,
    rejected: waveTriplets.length === 0 && reviewed.has(c.patient_id) && !selectedFreq,
  }};
  if (reviewed.has(c.patient_id)) labeled.add(c.patient_id);
  saveAll();
}}

function markReviewed() {{
  const c = CASES[idx];
  reviewed.add(c.patient_id);
}}

function skipCase() {{
  const c = CASES[idx];
  pushUndo();
  waveTriplets = [];
  pendingTriplet = {{}};
  wavePhase = 0;
  selectedFreq = null;
  selectedLaterality = null;
  reviewed.add(c.patient_id);
  allLabels[c.patient_id] = {{
    triplets: [],
    selected_freq: null,
    est_freq: c.est_freq,
    laterality: null,
    rejected: true,
  }};
  labeled.add(c.patient_id);
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (no LRDA)';
  el.style.color = '#ff6644';
  setTimeout(() => {{ el.textContent = ''; }}, 1500);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

// ── Redraw ──
function redraw() {{
  drawEEG();
  drawVE();
  updateInfo();
  updateModeIndicator();
}}

// ── Show case ──
function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load from storage or start fresh
  if (allLabels[c.patient_id] && allLabels[c.patient_id].triplets) {{
    waveTriplets = allLabels[c.patient_id].triplets.map(t => ({{...t}}));
    selectedFreq = allLabels[c.patient_id].selected_freq || null;
    selectedLaterality = allLabels[c.patient_id].laterality || null;
    labeled.add(c.patient_id);
  }} else {{
    waveTriplets = [];
    // Default to NVO estimate so overlay shows immediately
    selectedFreq = c.est_freq;
    selectedLaterality = c.est_laterality;
  }}
  pendingTriplet = {{}};
  wavePhase = 0;
  undoStack = [];
  hoverMarker = -1;

  // Highlight active freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('active');
    if (selectedFreq && Math.abs(parseFloat(btn.dataset.freq) - selectedFreq) < 0.01)
      btn.classList.add('active');
  }});

  // Highlight laterality
  document.querySelectorAll('.lat-btn').forEach(btn => {{
    btn.classList.remove('active-left', 'active-right', 'active-bilateral');
    if (selectedLaterality && btn.dataset.lat === selectedLaterality)
      btn.classList.add('active-' + selectedLaterality);
  }});

  document.getElementById('freq-info').textContent = '';
  redraw();
}}

// ── Export ──
function exportJSON() {{
  autoSave();
  const out = {{}};
  for (const c of CASES) {{
    const pid = c.patient_id;
    if (allLabels[pid]) {{
      const lab = allLabels[pid];
      const isRejected = lab.rejected === true;
      out[pid] = {{
        patient_id: pid,
        segment_id: c.segment_id,
        triplets: lab.triplets || [],
        selected_freq: lab.selected_freq,
        est_freq: c.est_freq,
        laterality: lab.laterality,
        est_laterality: c.est_laterality,
        lat_index: c.lat_index,
        review_status: isRejected ? 'rejected' : 'ground_truth',
        rejected: isRejected,
        source: 'lrda_labeler_v1',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lrda_labels.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' cases';
  el.style.color = '#4f4';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// ── Canvas click handler ──
const eegCanvas = document.getElementById('eeg-canvas');

function handleCanvasClick(e) {{
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;

  if (x < PLOT_LEFT || x > PLOT_RIGHT) return;
  const t = parseFloat(xToTime(x).toFixed(3));

  if (mode === 'wave') {{
    if (t < 0 || t > DURATION) return;
    pushUndo();

    const phases = ['onset', 'peak', 'offset'];
    pendingTriplet[phases[wavePhase]] = t;
    wavePhase++;

    if (wavePhase >= 3) {{
      // Complete triplet — add and reset
      waveTriplets.push({{ ...pendingTriplet }});
      // Sort by onset time
      waveTriplets.sort((a, b) => (a.onset || 0) - (b.onset || 0));
      pendingTriplet = {{}};
      wavePhase = 0;
    }}
    redraw();

  }} else if (mode === 'delete') {{
    const nearest = findNearestWavePoint(x);
    if (nearest) {{
      pushUndo();
      if (nearest.tripletIdx >= 0) {{
        // Delete entire triplet
        waveTriplets.splice(nearest.tripletIdx, 1);
      }} else {{
        // Delete from pending
        pendingTriplet = {{}};
        wavePhase = 0;
      }}
      redraw();
    }}
  }}
}}

eegCanvas.addEventListener('click', handleCanvasClick);

// ── Keyboard ──
document.addEventListener('keydown', (e) => {{
  // Ignore if focused on an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === 'w' || e.key === 'W') {{
    mode = mode === 'wave' ? 'nav' : 'wave';
    if (mode === 'wave') {{ wavePhase = 0; pendingTriplet = {{}}; }}
    redraw();
  }} else if (e.key === 'd' || e.key === 'D') {{
    mode = mode === 'delete' ? 'nav' : 'delete';
    redraw();
  }} else if (e.key === 'Escape') {{
    mode = 'nav';
    redraw();
  }} else if (e.key === 'z' || e.key === 'Z') {{
    undo();
  }} else if (e.key === 'l' || e.key === 'L') {{
    setLaterality('left');
  }} else if (e.key === 'r') {{
    setLaterality('right');
  }} else if (e.key === 'b' || e.key === 'B') {{
    setLaterality('bilateral');
  }} else if (e.key === 'f' || e.key === 'F') {{
    // Set freq from IPI of marked waves
    if (waveTriplets.length >= 2) {{
      const peaks = waveTriplets.filter(t => t.peak != null).map(t => t.peak).sort((a, b) => a - b);
      if (peaks.length >= 2) {{
        const ipis = [];
        for (let i = 1; i < peaks.length; i++) ipis.push(peaks[i] - peaks[i - 1]);
        const medIPI = ipis.sort((a, b) => a - b)[Math.floor(ipis.length / 2)];
        if (medIPI > 0) {{
          const ipiFreq = 1 / medIPI;
          // Find closest button
          const closest = FREQ_BUTTONS.reduce((prev, curr) =>
            Math.abs(curr - ipiFreq) < Math.abs(prev - ipiFreq) ? curr : prev);
          selectFreq(closest);
          document.getElementById('freq-info').textContent =
            'Freq from IPI: ' + ipiFreq.toFixed(2) + ' Hz → button ' + closest.toFixed(1) + ' Hz';
        }}
      }}
    }}
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    autoSave();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight' || e.key === 'c' || e.key === 'C' || e.key === 'Enter') {{
    e.preventDefault();
    if (e.key === 'c' || e.key === 'C' || e.key === 'Enter') markReviewed();
    autoSave();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'x' || e.key === 'X') {{
    skipCase();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }} else if (e.key >= '1' && e.key <= '7') {{
    // Quick freq: 1=0.5, 2=1.0, ..., 7=3.5
    const freqVal = parseInt(e.key) * 0.5;
    const closest = FREQ_BUTTONS.reduce((prev, curr) =>
      Math.abs(curr - freqVal) < Math.abs(prev - freqVal) ? curr : prev);
    selectFreq(closest);
  }} else if (e.key === 'R') {{
    setLaterality('right');
  }}
}});

// Init
show();
</script>

</body>
</html>"""
    return html


if __name__ == '__main__':
    main()
