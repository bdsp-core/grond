"""
LPD Laterality + Timing Labeler.

Adapted from HPP labeler. For each unlabeled LPD segment:
  1. Runs HemiCET+DP for default laterality prediction
  2. Precomputes HPP discharge times at each frequency button
  3. Generates an HTML viewer where MW can:
     - Label laterality (1=left, 2=right)
     - Select frequency via up/down arrows (or click), Enter to accept
     - View/edit discharge timing markers

Channels on the predicted/selected hemisphere are drawn in crimson.

Usage:
    conda run -n foe python code/generators/labeling/generate_lpd_laterality_timing_labeler.py [--batch N] [--batch-size S]
"""

import sys
import json
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt

LABELING_DIR = Path(__file__).resolve().parent
CODE_DIR = LABELING_DIR.parent.parent  # code/
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import LEFT_INDICES, RIGHT_INDICES, FS
from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence, _aggregate_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)

sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
import pd_detect_alternate as pddeta

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'labeling_tools' / 'lpd_laterality_timing'
OUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
LOWPASS_HZ = 20.0
BATCH_SIZE = 500

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

# Frequency buttons: 0.25 to 4.5 Hz in 0.25 steps
FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(1, 19)]  # 0.25 to 4.5


def hpp_with_freq(evidence, fs, freq_hz):
    """Run HPP with a given frequency prior. The 'cheating' version."""
    if len(evidence) < 10 or freq_hz <= 0:
        return []
    freq_estimate = np.clip(freq_hz, 0.2, 5.0)
    active_start, active_end = _detect_active_interval(evidence, fs)
    candidates = _extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    if len(candidates) == 0:
        return []
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)
    if len(discharge_samples) == 0:
        return []
    if len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)
    times = (discharge_samples / fs).tolist()
    return [t for t in times if 0 <= t <= DURATION]


MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_PAIRS = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]
BIPOLAR_INDICES = np.array([
    [MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)] for a, b in BIPOLAR_PAIRS
])


def _load_monopolar(mat_file):
    """Load raw monopolar EEG (19 channels) without bipolar conversion."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] == 19:
        return seg
    return None


def load_segment(mat_file):
    """Load EEG segment from mat file, converting monopolar to bipolar if needed."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    data_key = [k for k in mat.keys() if not k.startswith('_')][0]
    seg = mat[data_key]
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    seg = seg[:, :2000]
    if seg.shape[0] == 19:
        # Monopolar → bipolar conversion
        seg = seg[BIPOLAR_INDICES[:, 0]] - seg[BIPOLAR_INDICES[:, 1]]
    elif seg.shape[0] >= 18:
        seg = seg[:18]
    else:
        return None
    return seg


def compute_evidence(seg, laterality=None):
    """Compute aggregated HPP evidence for a segment.

    Preprocesses with notch (60 Hz) + bandpass (0.5-20 Hz) before evidence
    computation to remove drift, HF artifact, and line noise.
    """
    from scipy.signal import detrend as dt_func
    n_ch = min(seg.shape[0], 18)
    n_samples = seg.shape[1]

    # Preprocess: detrend + bandpass 0.5-20 Hz
    sos_bp = butter(4, [0.5 / (FS / 2), 20.0 / (FS / 2)], btype='bandpass', output='sos')
    from scipy.signal import sosfiltfilt
    seg_clean = np.zeros_like(seg[:n_ch])
    for ch in range(n_ch):
        try:
            seg_clean[ch] = sosfiltfilt(sos_bp, dt_func(seg[ch], type='linear'))
        except Exception:
            seg_clean[ch] = seg[ch]

    evidence_all = np.zeros((n_ch, n_samples))
    for ch in range(n_ch):
        evidence_all[ch] = _compute_channel_evidence(seg_clean[ch], FS)

    if laterality == 'left':
        return np.median(evidence_all[LEFT_INDICES], axis=0)
    elif laterality == 'right':
        return np.median(evidence_all[RIGHT_INDICES], axis=0)
    else:
        # Unknown laterality: use max of left/right
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)


_pd_characterizer = None

def _get_pd_characterizer():
    """Lazy-load PDCharacterizer (loads CNN weights once)."""
    global _pd_characterizer
    if _pd_characterizer is None:
        from pd_characterizer import PDCharacterizer
        _pd_characterizer = PDCharacterizer()
    return _pd_characterizer


def predict_laterality(seg):
    """Predict laterality, frequency, and discharge times using PDCharacterizer.

    Uses the full CNN pipeline (ChannelPD-Net + HemiCET+DP) — the best PD model
    we have (laterality AUC 0.963, timing F1 0.684, freq ρ 0.681).

    Returns dict with laterality_index, predicted_side, left_score, right_score, est_freq.
    """
    try:
        pc = _get_pd_characterizer()
        result = pc.characterize(seg, subtype='lpd')

        # Extract laterality from channel_probs
        channel_probs = result.get('channel_probs', [0.5] * 18)
        left_score = float(np.mean([channel_probs[i] for i in LEFT_INDICES]))
        right_score = float(np.mean([channel_probs[i] for i in RIGHT_INDICES]))
        total = left_score + right_score
        if total > 0:
            lat_index = (right_score - left_score) / total
        else:
            lat_index = 0.0

        laterality = result.get('laterality', 'left')
        est_freq = result.get('frequency', 1.0)
        if est_freq is None or not np.isfinite(est_freq):
            est_freq = 1.0

        return {
            'laterality_index': round(lat_index, 4),
            'predicted_side': laterality,
            'left_score': round(left_score, 4),
            'right_score': round(right_score, 4),
            'est_freq': round(max(0.25, min(4.5, est_freq)), 2),
        }
    except Exception as e:
        print(f"  PDCharacterizer failed: {e}")
        return {
            'laterality_index': 0.0,
            'predicted_side': 'left',
            'left_score': 0.0,
            'right_score': 0.0,
            'est_freq': 1.0,
        }


def downsample(arr, target_len):
    """Downsample array to target length."""
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


def precompute_hpp_results(evidence, freq_buttons):
    """Run HPP for each frequency button value, return dict freq -> times."""
    results = {}
    for freq in freq_buttons:
        times = hpp_with_freq(evidence, FS, freq)
        results[str(freq)] = times
    return results


def find_unlabeled_lpd(min_votes=0):
    """Find LPD segments without ground-truth laterality, one per patient.

    Args:
        min_votes: Minimum IIIC crowd votes required (0 = all segments).
    """
    import pandas as pd
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    sl_active = sl[(sl.excluded.fillna(False).astype(bool) == False)]
    lpd = sl_active[sl_active.subtype == 'lpd'].copy()

    # Filter by minimum votes if specified
    if min_votes > 0:
        nv = pd.to_numeric(lpd.iiic_n_votes, errors='coerce').fillna(0)
        lpd = lpd[nv >= min_votes].copy()

    # Missing laterality
    missing = lpd[lpd.laterality.isna()].copy()
    # One segment per patient
    missing = missing.drop_duplicates(subset='patient_id')

    print(f"  Total LPD (>={min_votes} votes): {len(lpd)}")
    print(f"  Missing laterality: {len(missing)} patients")
    return missing


def main():
    parser = argparse.ArgumentParser(description='LPD Laterality + Timing Labeler')
    parser.add_argument('--batch', type=int, default=1, help='Batch number (1-indexed)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Cases per batch')
    parser.add_argument('--min-votes', type=int, default=0, help='Minimum IIIC crowd votes (0=all)')
    args = parser.parse_args()

    print("=" * 70)
    print("  LPD Laterality + Timing Labeler")
    print("=" * 70)

    # Find unlabeled LPDs
    print("\nFinding unlabeled LPD segments...")
    unlabeled = find_unlabeled_lpd(min_votes=args.min_votes)

    # Sort by patient_id for deterministic batching
    unlabeled = unlabeled.sort_values('patient_id').reset_index(drop=True)

    # Batch
    total = len(unlabeled)
    n_batches = (total + args.batch_size - 1) // args.batch_size
    batch_start = (args.batch - 1) * args.batch_size
    batch_end = min(batch_start + args.batch_size, total)

    if batch_start >= total:
        print(f"  Batch {args.batch} is out of range (only {n_batches} batches)")
        return

    batch = unlabeled.iloc[batch_start:batch_end]
    print(f"\n  Batch {args.batch}/{n_batches}: cases {batch_start+1}–{batch_end} of {total}")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')

    cases_data = []
    n_skipped = 0

    for i, (_, row) in enumerate(batch.iterrows()):
        mat_file = row['mat_file']
        seg = load_segment(mat_file)
        if seg is None:
            n_skipped += 1
            continue

        # Also load raw monopolar for CAR montage toggle
        mono_raw = _load_monopolar(mat_file)

        # Detrend + lowpass filter for display
        from scipy.signal import detrend
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, detrend(seg[ch], type='linear'))
            except Exception:
                seg_display[ch] = seg[ch]

        # Detrend + lowpass filter monopolar for CAR display
        from scipy.signal import detrend
        mono_display = None
        if mono_raw is not None:
            mono_display = np.zeros_like(mono_raw)
            for ch in range(mono_raw.shape[0]):
                try:
                    mono_display[ch] = filtfilt(b_lp, a_lp, detrend(mono_raw[ch], type='linear'))
                except Exception:
                    mono_display[ch] = mono_raw[ch]

        # Predict laterality
        lat_pred = predict_laterality(seg)

        # Compute evidence using predicted laterality
        evidence = compute_evidence(seg, laterality=lat_pred['predicted_side'])

        # Normalize evidence for display
        ev_max = np.max(evidence)
        ev_display = evidence / ev_max if ev_max > 0 else evidence

        # Precompute HPP results for all frequency buttons
        hpp_results = precompute_hpp_results(evidence, FREQ_BUTTONS)

        # Find closest frequency button to estimated freq
        est_freq = lat_pred['est_freq']
        closest_btn = min(FREQ_BUTTONS, key=lambda f: abs(f - est_freq))

        case = {
            'patient_id': str(row['patient_id']),
            'segment_id': str(mat_file).replace('.mat', ''),
            'mat_file': str(mat_file),
            'est_freq': round(est_freq, 2),
            'closest_btn_freq': closest_btn,
            'predicted_side': lat_pred['predicted_side'],
            'laterality_index': lat_pred['laterality_index'],
            'left_score': lat_pred['left_score'],
            'right_score': lat_pred['right_score'],
            'eeg_data': downsample(seg_display, 1000),
            'mono_data': downsample(mono_display, 1000) if mono_display is not None else None,
            'evidence': downsample(ev_display, 500),
            'hpp_results': hpp_results,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(batch)} processed")

    print(f"  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")

    if len(cases_data) == 0:
        print("  No cases to label!")
        return

    # Build HTML
    print("\nBuilding HTML viewer...")
    html = build_html(cases_data, args.batch, n_batches)

    out_path = OUT_DIR / f'lpd_lat_timing_batch{args.batch}.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases ready for review")

    # Open in browser
    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


def build_html(cases_data, batch_num, n_batches):
    freq_btns_json = json.dumps(FREQ_BUTTONS)
    left_indices_json = json.dumps(LEFT_INDICES)
    right_indices_json = json.dumps(RIGHT_INDICES)

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LPD Laterality + Timing — Batch {batch_num}/{n_batches} ({n_cases} cases)</title>
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
  .mode-add {{ background: #1a3a1a; color: #44ff66; }}
  .mode-delete {{ background: #3a1a1a; color: #ff4444; }}
  .mode-nav {{ background: #1a1a3a; color: #6688ff; }}

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #laterality-panel {{
    background: #252525; padding: 8px 16px; display: flex; align-items: center;
    gap: 16px; border-bottom: 1px solid #333;
  }}
  #laterality-panel label {{ color: #aaa; font-size: 13px; font-weight: bold; margin-right: 4px; }}
  .lat-badge {{
    font-size: 16px; font-weight: bold; padding: 6px 18px; border-radius: 6px;
    letter-spacing: 1px; border: 2px solid transparent;
  }}
  .lat-left {{ background: #3a1a1a; color: #dc143c; border-color: #dc143c; }}
  .lat-right {{ background: #1a1a3a; color: #4488ff; border-color: #4488ff; }}
  .lat-none {{ background: #2a2a2a; color: #666; border-color: #444; }}

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
  .freq-btn.est {{ border-color: #ff9800; }}
  .freq-btn.focused {{ outline: 2px solid #fff; outline-offset: 1px; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}
  #evidence-canvas {{ display: block; margin: 4px auto 0 auto; cursor: crosshair; }}

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
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#dc143c;">LPD Laterality + Timing</span>
    <span style="font-size:12px; color:#888;">Batch {batch_num}/{n_batches}</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / {n_cases}</span>
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
  <span class="info-item">Est freq: <strong id="info-est-freq" style="color:#ff9800;">--</strong></span>
  <span class="info-item">Selected freq: <strong id="info-sel-freq" style="color:#44ff66;">--</strong></span>
  <span class="info-item">Markers: <strong id="info-marker-count" style="color:#ff4444;">--</strong></span>
  <span class="info-item">IPI freq: <strong id="info-ipi-freq">--</strong></span>
  <span class="info-item">Mode: <strong id="info-mode">Navigate</strong></span>
</div>

<div id="laterality-panel">
  <label>Laterality:</label>
  <span id="lat-badge" class="lat-badge lat-none">--</span>
  <span style="color:#666; font-size:12px; margin-left:8px;">
    <span class="key">1</span> Left &nbsp;
    <span class="key">2</span> Right
  </span>
  <span style="color:#666; font-size:12px; margin-left:12px;">
    Lat index: <strong id="info-lat-index" style="color:#ccc;">--</strong>
  </span>
</div>

<div id="freq-buttons">
  <label>Freq prior:</label>
</div>

<div id="freq-info"></div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">1</span> Left &nbsp;&nbsp;
  <span class="key">2</span> Right &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Change freq &nbsp;&nbsp;
  <span class="key">Enter</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">A</span> Add marker mode &nbsp;&nbsp;
  <span class="key">D</span> Delete marker mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">X</span> Skip/Reject &nbsp;&nbsp;
  <span class="key">E</span> Export JSON &nbsp;&nbsp;
  <span class="key">Ctrl</span> Toggle montage (<span id="montage-label">bipolar</span>)
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};
const MIDLINE_INDICES = [16, 17];

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const MONO_NAMES = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg',
  'Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg'];
// CAR display order: left parasag, left temporal, midline, right parasag, right temporal
const CAR_DISPLAY_ORDER = [
  0, 1, 2, 3,    // Fp1, F3, C3, P3 (left parasag)
  4, 5, 6, 7,    // F7, T3, T5, O1 (left temporal)
  -1,
  8, 9, 10,      // Fz, Cz, Pz (midline)
  -1,
  11, 12, 13, 14, // Fp2, F4, C4, P4 (right parasag)
  15, 16, 17, 18  // F8, T4, T6, O2 (right temporal)
];
const CAR_LEFT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7];
const CAR_RIGHT_INDICES = [11, 12, 13, 14, 15, 16, 17, 18];
let CHANNEL_NAMES = BIPOLAR_NAMES;
const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 1200;
const EEG_HEIGHT = 700;
const EV_HEIGHT = 160;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = (EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM);

// State
let idx = 0;
let labeled = new Set();
let mode = 'nav';
let markers = [];
let undoStack = [];
let hoverMarker = -1;
let selectedFreq = null;
let freqBtnIdx = -1;  // index into FREQ_BUTTONS for arrow key nav
let selectedLaterality = null;  // 'left' | 'right' | null
let montage = 'bipolar';  // 'bipolar' | 'car'

// Helper: look up HPP results by freq, trying multiple key formats
function hppLookup(hppResults, freq) {{
  // Python json.dumps uses repr-like formatting: 1.0 -> "1.0", 0.25 -> "0.25"
  // JS String(1.0) -> "1", toFixed(2) -> "1.00" — neither matches "1.0"
  const keys = [String(freq), freq.toFixed(1), freq.toFixed(2)];
  for (const k of keys) {{
    if (hppResults[k]) return hppResults[k];
  }}
  return [];
}}

// Persistence
const STORAGE_KEY = 'lpd_lat_timing_batch{batch_num}_v2';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}

function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

// Build frequency buttons
(function() {{
  const container = document.getElementById('freq-buttons');
  for (let fi = 0; fi < FREQ_BUTTONS.length; fi++) {{
    const freq = FREQ_BUTTONS[fi];
    const btn = document.createElement('button');
    btn.className = 'freq-btn';
    btn.textContent = freq.toFixed(2);
    btn.dataset.freq = freq;
    btn.dataset.fi = fi;
    btn.id = 'freq-btn-' + fi;
    btn.onclick = () => selectFreqByIndex(fi);
    container.appendChild(btn);
  }}
}})();

/// Channel display order: L-lateral, L-parasag, midline, R-parasag, R-lateral
const BIPOLAR_DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];

function getDisplayChannels() {{
  const order = (montage === 'car') ? CAR_DISPLAY_ORDER : BIPOLAR_DISPLAY_ORDER;
  const names = (montage === 'car') ? MONO_NAMES : BIPOLAR_NAMES;
  const dc = [];
  for (const i of order) {{
    if (i < 0) dc.push({{ idx: -1, name: '' }});
    else dc.push({{ idx: i, name: names[i] }});
  }}
  return dc;
}}
let DISPLAY_CHANNELS = getDisplayChannels();
let N_DISPLAY = DISPLAY_CHANNELS.length;

function getEEGData() {{
  const c = CASES[idx];
  if (montage === 'car' && c.mono_data) {{
    // Compute common average reference on-the-fly
    const mono = c.mono_data;
    const nCh = mono.length;
    const nSamp = mono[0].length;
    const avg = new Array(nSamp).fill(0);
    for (let ch = 0; ch < nCh; ch++) {{
      for (let s = 0; s < nSamp; s++) avg[s] += mono[ch][s];
    }}
    for (let s = 0; s < nSamp; s++) avg[s] /= nCh;
    const car = [];
    for (let ch = 0; ch < nCh; ch++) {{
      const row = new Array(nSamp);
      for (let s = 0; s < nSamp; s++) row[s] = mono[ch][s] - avg[s];
      car.push(row);
    }}
    return car;
  }}
  return c.eeg_data;
}}

function toggleMontage() {{
  montage = (montage === 'bipolar') ? 'car' : 'bipolar';
  CHANNEL_NAMES = (montage === 'car') ? MONO_NAMES : BIPOLAR_NAMES;
  DISPLAY_CHANNELS = getDisplayChannels();
  N_DISPLAY = DISPLAY_CHANNELS.length;
  drawEEG();
}}

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}
function xToTime(x) {{ return ((x - PLOT_LEFT) / PLOT_W) * DURATION; }}

function findNearestMarker(x) {{
  let best = -1, bestDist = Infinity;
  for (let i = 0; i < markers.length; i++) {{
    const mx = timeToX(markers[i]);
    const dist = Math.abs(mx - x);
    if (dist < bestDist) {{ bestDist = dist; best = i; }}
  }}
  return (bestDist <= 20) ? best : -1;
}}

function pushUndo() {{
  undoStack.push({{ markers: [...markers], freq: selectedFreq, lat: selectedLaterality }});
  if (undoStack.length > 100) undoStack.shift();
}}

function undo() {{
  if (undoStack.length === 0) return;
  const state = undoStack.pop();
  markers = state.markers;
  selectedFreq = state.freq;
  selectedLaterality = state.lat;
  updateFreqHighlight();
  redraw();
}}

function selectFreqByIndex(fi) {{
  freqBtnIdx = fi;
  const freq = FREQ_BUTTONS[fi];
  selectedFreq = freq;
  const c = CASES[idx];
  const hppTimes = hppLookup(c.hpp_results, freq);

  pushUndo();
  markers = [...hppTimes];

  updateFreqHighlight();

  const infoEl = document.getElementById('freq-info');
  infoEl.textContent = 'HPP with freq=' + freq.toFixed(2) + ' Hz -> ' + markers.length + ' discharges';

  redraw();
}}

function updateFreqHighlight() {{
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('active', 'focused');
    const f = parseFloat(btn.dataset.freq);
    if (selectedFreq && f === selectedFreq) {{
      btn.classList.add('active');
      btn.classList.add('focused');
    }}
  }});
}}

function setLaterality(side) {{
  pushUndo();
  selectedLaterality = side;
  updateLatBadge();
  redraw();
}}

function updateLatBadge() {{
  const badge = document.getElementById('lat-badge');
  if (selectedLaterality === 'left') {{
    badge.textContent = 'LEFT';
    badge.className = 'lat-badge lat-left';
  }} else if (selectedLaterality === 'right') {{
    badge.textContent = 'RIGHT';
    badge.className = 'lat-badge lat-right';
  }} else {{
    badge.textContent = '--';
    badge.className = 'lat-badge lat-none';
  }}
}}

function channelColor(chIdx) {{
  // Crimson for selected hemisphere, black for others
  if (montage === 'car') {{
    if (selectedLaterality === 'left' && CAR_LEFT_INDICES.includes(chIdx)) return '#dc143c';
    if (selectedLaterality === 'right' && CAR_RIGHT_INDICES.includes(chIdx)) return '#dc143c';
  }} else {{
    if (selectedLaterality === 'left' && LEFT_INDICES.includes(chIdx)) return '#dc143c';
    if (selectedLaterality === 'right' && RIGHT_INDICES.includes(chIdx)) return '#dc143c';
  }}
  return '#000000';
}}

function channelLabelColor(chIdx) {{
  if (montage === 'car') {{
    if (selectedLaterality === 'left' && CAR_LEFT_INDICES.includes(chIdx)) return '#dc143c';
    if (selectedLaterality === 'right' && CAR_RIGHT_INDICES.includes(chIdx)) return '#dc143c';
  }} else {{
    if (selectedLaterality === 'left' && LEFT_INDICES.includes(chIdx)) return '#dc143c';
    if (selectedLaterality === 'right' && RIGHT_INDICES.includes(chIdx)) return '#dc143c';
  }}
  return '#000000';
}}

function drawEEG() {{
  const canvas = document.getElementById('eeg-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const eegData = getEEGData();
  const nSamples = eegData[0].length;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Traces — color by hemisphere
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    ctx.strokeStyle = channelColor(ch.idx);
    ctx.lineWidth = 0.7;
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

  // Channel labels — color by hemisphere
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillStyle = channelLabelColor(ch.idx);
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#000000';
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', timeToX(s), EEG_HEIGHT - MARGIN_BOTTOM + 4);
  }}

  // Title
  ctx.fillStyle = '#000000';
  ctx.font = 'bold 13px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const latStr = selectedLaterality ? selectedLaterality.toUpperCase() : '??';
  const freqStr = selectedFreq ? selectedFreq.toFixed(2) : c.est_freq.toFixed(2);
  const title = c.patient_id + '  |  lat=' + latStr + '  |  freq=' + freqStr + ' Hz  |  lat_idx=' + c.laterality_index.toFixed(3);
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Discharge markers (dashed, extend full height for visual continuity with evidence canvas)
  for (let mi = 0; mi < markers.length; mi++) {{
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

    let color = 'rgba(255, 0, 0, 0.6)';
    let lw = 2;
    if (mode === 'delete' && hoverMarker === mi) {{
      color = 'rgba(255, 50, 50, 0.9)';
      lw = 4;
    }}

    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.setLineDash([6, 3]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, EEG_HEIGHT); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = color;
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(t.toFixed(2) + 's', x, MARGIN_TOP - 2);
  }}
}}

function drawEvidence() {{
  const canvas = document.getElementById('evidence-canvas');
  canvas.width = EEG_WIDTH;
  canvas.height = EV_HEIGHT;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];
  const evData = c.evidence;

  const evTop = 10;
  const evBottom = EV_HEIGHT - 20;
  const evH = evBottom - evTop;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EV_HEIGHT);

  // Gridlines
  ctx.strokeStyle = '#dddddd';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  for (let s = 0; s <= 10; s++) {{
    const x = timeToX(s);
    ctx.beginPath(); ctx.moveTo(x, evTop); ctx.lineTo(x, evBottom); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // Evidence trace
  if (evData && evData.length > 0) {{
    const nSamples = evData.length;
    ctx.fillStyle = 'rgba(70, 130, 180, 0.15)';
    ctx.beginPath();
    ctx.moveTo(PLOT_LEFT, evBottom);
    for (let i = 0; i < nSamples; i++) {{
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - evData[i] * evH;
      ctx.lineTo(x, y);
    }}
    ctx.lineTo(PLOT_RIGHT, evBottom);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = 'steelblue';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < nSamples; i++) {{
      const x = PLOT_LEFT + (i / (nSamples - 1)) * PLOT_W;
      const y = evBottom - evData[i] * evH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
    ctx.stroke();
  }}

  // Labels
  ctx.fillStyle = '#000';
  ctx.font = '10px monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('1.0', PLOT_LEFT - 4, evTop);
  ctx.fillText('0.0', PLOT_LEFT - 4, evBottom);
  ctx.textAlign = 'left';
  ctx.fillText('Evidence (pointiness+TKEO)', PLOT_LEFT + 4, evTop - 1);

  // Discharge markers on evidence (dashed, full height)
  for (let mi = 0; mi < markers.length; mi++) {{
    const t = markers[mi];
    const x = timeToX(t);
    if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;
    let color = 'rgba(255, 0, 0, 0.5)';
    let lw = 1.5;
    if (mode === 'delete' && hoverMarker === mi) {{ color = 'rgba(255, 50, 50, 0.9)'; lw = 3; }}
    ctx.strokeStyle = color;
    ctx.lineWidth = lw;
    ctx.setLineDash([6, 3]);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, EV_HEIGHT); ctx.stroke();
    ctx.setLineDash([]);
  }}
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-est-freq').textContent = c.est_freq.toFixed(2) + ' Hz';
  document.getElementById('info-sel-freq').textContent = selectedFreq ? selectedFreq.toFixed(2) + ' Hz' : '--';
  document.getElementById('info-marker-count').textContent = markers.length;
  document.getElementById('info-lat-index').textContent = c.laterality_index.toFixed(3);

  // IPI freq
  if (markers.length >= 2) {{
    const sorted = [...markers].sort((a,b) => a-b);
    const ipis = [];
    for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i-1]);
    const medIPI = ipis.sort((a,b) => a-b)[Math.floor(ipis.length/2)];
    const ipiFreq = medIPI > 0 ? (1/medIPI).toFixed(2) : '--';
    document.getElementById('info-ipi-freq').textContent = ipiFreq + ' Hz';
  }} else {{
    document.getElementById('info-ipi-freq').textContent = '--';
  }}

  let modeStr = 'Navigate';
  if (mode === 'add') modeStr = 'Add';
  else if (mode === 'delete') modeStr = 'Delete';
  document.getElementById('info-mode').textContent = modeStr;

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';
  document.getElementById('labeled-count').textContent = labeled.size + ' labeled';

  // Highlight estimated freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('est');
    const f = parseFloat(btn.dataset.freq);
    if (Math.abs(f - c.est_freq) < 0.13) btn.classList.add('est');
  }});
}}

function updateModeIndicator() {{
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') {{ el.textContent = 'ADD MODE (A) -- click to add marker'; el.className = 'mode-add'; }}
  else if (mode === 'delete') {{ el.textContent = 'DELETE MODE (D) -- click near marker to remove'; el.className = 'mode-delete'; }}
  else {{ el.textContent = 'NAVIGATE MODE'; el.className = 'mode-nav'; }}
}}

let reviewed = new Set();

function autoSave() {{
  const c = CASES[idx];
  allLabels[c.patient_id] = {{
    segment_id: c.segment_id,
    times: [...markers].sort((a, b) => a - b),
    selected_freq: selectedFreq,
    est_freq: c.est_freq,
    laterality: selectedLaterality,
    laterality_index: c.laterality_index,
    rejected: markers.length === 0 && reviewed.has(c.patient_id),
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
  markers = [];
  selectedFreq = null;
  selectedLaterality = null;
  reviewed.add(c.patient_id);
  allLabels[c.patient_id] = {{
    segment_id: c.segment_id,
    times: [],
    selected_freq: null,
    est_freq: c.est_freq,
    laterality: null,
    laterality_index: c.laterality_index,
    rejected: true,
  }};
  labeled.add(c.patient_id);
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED';
  el.style.color = '#ff6644';
  setTimeout(() => {{ el.textContent = ''; }}, 1500);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function redraw() {{
  drawEEG();
  drawEvidence();
  updateInfo();
  updateModeIndicator();
  updateLatBadge();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load from storage or initialize from predictions
  if (allLabels[c.patient_id] && allLabels[c.patient_id].times) {{
    markers = [...allLabels[c.patient_id].times];
    selectedFreq = allLabels[c.patient_id].selected_freq || c.closest_btn_freq;
    selectedLaterality = allLabels[c.patient_id].laterality || c.predicted_side;
    labeled.add(c.patient_id);
  }} else {{
    // Initialize with model predictions
    selectedLaterality = c.predicted_side;
    selectedFreq = c.closest_btn_freq;
    // Auto-select the closest frequency button and load HPP results
    markers = [...hppLookup(c.hpp_results, c.closest_btn_freq)];
  }}
  undoStack = [];
  hoverMarker = -1;

  // Set freqBtnIdx
  if (selectedFreq) {{
    freqBtnIdx = FREQ_BUTTONS.indexOf(selectedFreq);
    if (freqBtnIdx < 0) {{
      // Find closest
      let best = 0, bestD = Infinity;
      for (let i = 0; i < FREQ_BUTTONS.length; i++) {{
        const d = Math.abs(FREQ_BUTTONS[i] - selectedFreq);
        if (d < bestD) {{ bestD = d; best = i; }}
      }}
      freqBtnIdx = best;
    }}
  }} else {{
    freqBtnIdx = -1;
  }}

  updateFreqHighlight();
  document.getElementById('freq-info').textContent = '';
  redraw();
}}

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
        segment_id: lab.segment_id || c.segment_id,
        global_times: lab.times,
        selected_freq: lab.selected_freq,
        est_freq: c.est_freq,
        laterality: lab.laterality,
        laterality_index: c.laterality_index,
        review_status: isRejected ? 'rejected' : 'ground_truth',
        rejected: isRejected,
        source: 'lpd_lat_timing_labeler',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'lpd_lat_timing_batch{batch_num}_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' cases';
  el.style.color = '#4f4';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Canvas click handlers
const eegCanvas = document.getElementById('eeg-canvas');
const evCanvas = document.getElementById('evidence-canvas');

function handleCanvasClick(e) {{
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;

  if (x < PLOT_LEFT || x > PLOT_RIGHT) return;
  const t = xToTime(x);

  if (mode === 'add') {{
    if (t >= 0 && t <= DURATION) {{
      pushUndo();
      markers.push(t);
      redraw();
    }}
  }} else if (mode === 'delete') {{
    const mi = findNearestMarker(x);
    if (mi >= 0) {{
      pushUndo();
      markers.splice(mi, 1);
      redraw();
    }}
  }}
}}

function handleCanvasMove(e) {{
  if (mode !== 'delete') return;
  const rect = eegCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const x = (e.clientX - rect.left) * scaleX;
  const newHover = findNearestMarker(x);
  if (newHover !== hoverMarker) {{
    hoverMarker = newHover;
    redraw();
  }}
}}

eegCanvas.addEventListener('click', handleCanvasClick);
eegCanvas.addEventListener('mousemove', handleCanvasMove);
evCanvas.addEventListener('click', handleCanvasClick);
evCanvas.addEventListener('mousemove', handleCanvasMove);

// Keyboard
document.addEventListener('keydown', (e) => {{
  if (e.key === 'a' || e.key === 'A') {{
    mode = mode === 'add' ? 'nav' : 'add';
    redraw();
  }} else if (e.key === 'd' || e.key === 'D') {{
    mode = mode === 'delete' ? 'nav' : 'delete';
    redraw();
  }} else if (e.key === 'Escape') {{
    mode = 'nav';
    redraw();
  }} else if (e.key === 'z' || e.key === 'Z') {{
    undo();
  }} else if (e.key === '1') {{
    // Left laterality
    setLaterality('left');
  }} else if (e.key === '2') {{
    // Right laterality
    setLaterality('right');
  }} else if (e.key === 'ArrowUp') {{
    // Move frequency selection up (lower freq)
    e.preventDefault();
    if (freqBtnIdx > 0) {{
      selectFreqByIndex(freqBtnIdx - 1);
    }} else if (freqBtnIdx < 0 && FREQ_BUTTONS.length > 0) {{
      selectFreqByIndex(0);
    }}
  }} else if (e.key === 'ArrowDown') {{
    // Move frequency selection down (higher freq)
    e.preventDefault();
    if (freqBtnIdx < FREQ_BUTTONS.length - 1) {{
      selectFreqByIndex(freqBtnIdx + 1);
    }} else if (freqBtnIdx < 0 && FREQ_BUTTONS.length > 0) {{
      selectFreqByIndex(0);
    }}
  }} else if (e.key === 'Enter') {{
    // Accept and advance
    e.preventDefault();
    markReviewed();
    autoSave();
    const el = document.getElementById('save-status');
    el.textContent = 'Saved';
    el.style.color = '#44cc44';
    setTimeout(() => {{ el.textContent = ''; }}, 800);
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'ArrowLeft') {{
    // Navigate to previous case
    e.preventDefault();
    autoSave();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
    // Navigate to next case (without marking reviewed)
    e.preventDefault();
    autoSave();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'x' || e.key === 'X') {{
    skipCase();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }} else if (e.key === 'Control') {{
    toggleMontage();
    const label = document.getElementById('montage-label');
    if (label) label.textContent = montage;
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
