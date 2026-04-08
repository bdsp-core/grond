"""
Frequency Discrepancy Reviewer: MW label vs model predictions.

Two modes:
  --type pd   PD (LPD+GPD) with timing-based IPI frequency, threshold-click,
              evidence trace, markers, add/delete modes.
  --type rda  RDA (LRDA+GRDA) with narrowband overlays (green=MW, red=model).

Usage:
    python code/generators/labeling/generate_freq_discrepancy_reviewer.py --type pd
    python code/generators/labeling/generate_freq_discrepancy_reviewer.py --type rda
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, filtfilt, detrend, iirnotch

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
OUT_BASE = PROJECT_DIR / 'results' / 'labeling_tools' / 'freq_discrepancy'
OUT_BASE.mkdir(parents=True, exist_ok=True)

DURATION = 10.0
LOWPASS_HZ = 20.0
NOTCH_HZ = 60.0
DISC_THRESHOLD = 0.5  # Hz

MONO_CHANNELS = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
    'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
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

FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(1, 19)]  # 0.25 to 4.5


# ---------- I/O helpers ----------

def _load_monopolar(mat_file):
    """Load raw monopolar EEG (19 channels)."""
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
    """Load EEG segment, converting monopolar to bipolar if needed."""
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
        seg = seg[BIPOLAR_INDICES[:, 0]] - seg[BIPOLAR_INDICES[:, 1]]
    elif seg.shape[0] >= 18:
        seg = seg[:18]
    else:
        return None
    return seg


# ---------- Signal processing ----------

def compute_narrowband(seg_bi, freq_hz, fs=200):
    """Compute narrowband-filtered signal at freq +/- 0.3 Hz."""
    lo = max(freq_hz - 0.3, 0.1)
    hi = min(freq_hz + 0.3, 99.0)
    if lo >= hi or lo >= fs / 2 or hi >= fs / 2:
        return np.zeros_like(seg_bi)
    try:
        sos = butter(4, [lo, hi], btype='bandpass', fs=fs, output='sos')
        return sosfiltfilt(sos, seg_bi, axis=1)
    except Exception:
        return np.zeros_like(seg_bi)


def compute_evidence(seg, laterality=None):
    """Compute aggregated HPP evidence for a segment."""
    n_ch = min(seg.shape[0], 18)
    n_samples = seg.shape[1]

    sos_bp = butter(4, [0.5 / (FS / 2), 20.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_clean = np.zeros_like(seg[:n_ch])
    for ch in range(n_ch):
        try:
            seg_clean[ch] = sosfiltfilt(sos_bp, detrend(seg[ch], type='linear'))
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
        left_med = np.median(evidence_all[LEFT_INDICES], axis=0)
        right_med = np.median(evidence_all[RIGHT_INDICES], axis=0)
        return np.maximum(left_med, right_med)


def hpp_with_freq(evidence, fs, freq_hz):
    """Run HPP with a given frequency prior."""
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


def precompute_hpp_results(evidence, freq_buttons):
    """Run HPP for each frequency button value."""
    results = {}
    for freq in freq_buttons:
        times = hpp_with_freq(evidence, FS, freq)
        results[str(freq)] = times
    return results


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


# ---------- PDProfiler ----------

_pd_profiler = None

def _get_pd_profiler():
    global _pd_profiler
    if _pd_profiler is None:
        from pd_profiler import PDProfiler
        _pd_profiler = PDProfiler()
    return _pd_profiler


def predict_laterality(seg, subtype='lpd'):
    """Predict laterality, frequency, and discharge times using PDProfiler."""
    try:
        pc = _get_pd_profiler()
        result = pc.characterize(seg, subtype=subtype)
        channel_probs = result.get('channel_probs', [0.5] * 18)
        left_score = float(np.mean([channel_probs[i] for i in LEFT_INDICES]))
        right_score = float(np.mean([channel_probs[i] for i in RIGHT_INDICES]))
        total = left_score + right_score
        lat_index = (right_score - left_score) / total if total > 0 else 0.0
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
        print(f"  PDProfiler failed: {e}")
        return {
            'laterality_index': 0.0,
            'predicted_side': 'left',
            'left_score': 0.0,
            'right_score': 0.0,
            'est_freq': 1.0,
        }


# ---------- Case finding ----------

def find_discrepancy_cases(subtypes, max_cases=0):
    """Find segments where model freq differs from MW label by >0.5 Hz."""
    import pandas as pd

    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    mw_segs = sl[
        (sl.expert_freq_rater == 'MW') &
        (sl.subtype.isin(subtypes)) &
        (sl.excluded.fillna(False).astype(bool) == False)
    ].copy()
    mw_segs['segment_id'] = mw_segs.mat_file.str.replace('.mat', '', regex=False)

    type_label = '/'.join(s.upper() for s in subtypes)
    print(f"  MW-labeled {type_label} segments: {len(mw_segs)}")
    for st in subtypes:
        print(f"    {st.upper()}: {(mw_segs.subtype == st).sum()}")

    fig6_path = PROJECT_DIR / 'paper_materials' / 'fig6_frequency_data.json'
    if not fig6_path.exists():
        print("  ERROR: fig6_frequency_data.json not found")
        return pd.DataFrame()

    with open(str(fig6_path)) as f:
        fig6 = json.load(f)

    model_preds = {}
    for subtype in subtypes:
        for entry in fig6.get(subtype, []):
            model_preds[entry['segment_id']] = {
                'pred_cnn': entry['pred_cnn'],
                'gt': entry['gt'],
                'n_raters': entry['n_raters'],
            }

    matched = mw_segs[mw_segs.segment_id.isin(model_preds)].copy()
    matched['model_freq'] = matched.segment_id.map(lambda x: model_preds[x]['pred_cnn'])
    matched['consensus_freq'] = matched.segment_id.map(lambda x: model_preds[x]['gt'])
    matched['n_raters'] = matched.segment_id.map(lambda x: model_preds[x]['n_raters'])
    matched['diff'] = abs(matched.expert_freq_hz - matched.model_freq)

    disc = matched[matched['diff'] > DISC_THRESHOLD].copy()
    disc = disc.sort_values('diff', ascending=False).reset_index(drop=True)

    if max_cases > 0 and len(disc) > max_cases:
        disc = disc.head(max_cases).reset_index(drop=True)

    print(f"  Matched with model predictions: {len(matched)}")
    print(f"  Discrepancies (>{DISC_THRESHOLD} Hz): {len(disc)}")
    for st in subtypes:
        print(f"    {st.upper()}: {(disc.subtype == st).sum()}")

    ann_path = LABELS_DIR / 'annotations.csv'
    ann_freqs = {}
    if ann_path.exists():
        ann = pd.read_csv(str(ann_path))
        ann_with_freq = ann[ann.frequency_hz.notna() & (ann.frequency_hz > 0)]
        for _, row in ann_with_freq.iterrows():
            sid = str(row.mat_file).replace('.mat', '')
            if sid not in ann_freqs:
                ann_freqs[sid] = []
            ann_freqs[sid].append({
                'rater': row.rater,
                'freq': float(row.frequency_hz),
            })

    disc['ann_raters'] = disc.segment_id.map(lambda x: ann_freqs.get(x, []))
    return disc


# ---------- PD data preparation ----------

def prepare_pd_cases(disc, sl=None):
    """Prepare PD cases: EEG + evidence + HPP markers at multiple freqs."""
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, (_, row) in enumerate(disc.iterrows()):
        mat_file = row['mat_file']
        seg = load_segment(mat_file)
        if seg is None:
            n_skipped += 1
            continue

        mono_raw = _load_monopolar(mat_file)

        # Detrend + notch + lowpass for display
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                s = detrend(seg[ch], type='linear')
                s = filtfilt(b_notch, a_notch, s)
                s = filtfilt(b_lp, a_lp, s)
                seg_display[ch] = s
            except Exception:
                seg_display[ch] = seg[ch]

        mono_display = None
        if mono_raw is not None:
            mono_display = np.zeros_like(mono_raw)
            for ch in range(mono_raw.shape[0]):
                try:
                    s = detrend(mono_raw[ch], type='linear')
                    s = filtfilt(b_notch, a_notch, s)
                    s = filtfilt(b_lp, a_lp, s)
                    mono_display[ch] = s
                except Exception:
                    mono_display[ch] = mono_raw[ch]

        # Compute evidence
        subtype = row['subtype']
        evidence = compute_evidence(seg, laterality=None)

        ev_max = np.max(evidence)
        ev_display = evidence / ev_max if ev_max > 0 else evidence

        model_freq = float(row['model_freq'])
        mw_freq = float(row['expert_freq_hz'])

        # Model discharge markers
        model_discharges = hpp_with_freq(evidence, FS, model_freq)

        # HPP results at all frequency buttons
        hpp_results = precompute_hpp_results(evidence, FREQ_BUTTONS)

        # Get IIIC vote count
        sl_row = sl[sl.mat_file == mat_file]
        n_votes = 0
        if len(sl_row) > 0:
            nv = pd.to_numeric(sl_row.iloc[0].get('iiic_n_votes'), errors='coerce')
            if np.isfinite(nv):
                n_votes = int(nv)

        case = {
            'segment_id': str(row['segment_id']),
            'mat_file': str(mat_file),
            'patient_id': str(row['patient_id']),
            'subtype': subtype,
            'mw_freq': round(mw_freq, 4),
            'model_freq': round(model_freq, 4),
            'consensus_freq': round(float(row['consensus_freq']), 4),
            'n_raters': int(row['n_raters']),
            'n_votes': n_votes,
            'diff': round(float(row['diff']), 4),
            'ann_raters': row['ann_raters'],
            'eeg_data': downsample(seg_display, 1000),
            'mono_data': downsample(mono_display, 1000) if mono_display is not None else None,
            'evidence': downsample(ev_display, 500),
            'model_discharges': model_discharges,
            'hpp_results': hpp_results,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(disc):
            print(f"  {i+1}/{len(disc)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"\n  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ---------- RDA data preparation ----------

def prepare_rda_cases(disc):
    """Prepare RDA cases: EEG + narrowband overlays."""
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, (_, row) in enumerate(disc.iterrows()):
        mat_file = row['mat_file']
        seg = load_segment(mat_file)
        if seg is None:
            n_skipped += 1
            continue

        mono_raw = _load_monopolar(mat_file)

        # Detrend + notch + lowpass for display
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                s = detrend(seg[ch], type='linear')
                s = filtfilt(b_notch, a_notch, s)
                s = filtfilt(b_lp, a_lp, s)
                seg_display[ch] = s
            except Exception:
                seg_display[ch] = seg[ch]

        mono_display = None
        if mono_raw is not None:
            mono_display = np.zeros_like(mono_raw)
            for ch in range(mono_raw.shape[0]):
                try:
                    s = detrend(mono_raw[ch], type='linear')
                    s = filtfilt(b_notch, a_notch, s)
                    s = filtfilt(b_lp, a_lp, s)
                    mono_display[ch] = s
                except Exception:
                    mono_display[ch] = mono_raw[ch]

        mw_freq = float(row['expert_freq_hz'])
        model_freq = float(row['model_freq'])

        nb_mw = compute_narrowband(seg, mw_freq, FS)
        nb_model = compute_narrowband(seg, model_freq, FS)

        case = {
            'segment_id': str(row['segment_id']),
            'mat_file': str(mat_file),
            'patient_id': str(row['patient_id']),
            'subtype': row['subtype'],
            'mw_freq': round(mw_freq, 4),
            'model_freq': round(model_freq, 4),
            'consensus_freq': round(float(row['consensus_freq']), 4),
            'n_raters': int(row['n_raters']),
            'diff': round(float(row['diff']), 4),
            'ann_raters': row['ann_raters'],
            'eeg_data': downsample(seg_display, 1000),
            'mono_data': downsample(mono_display, 1000) if mono_display is not None else None,
            'nb_mw': downsample(nb_mw, 1000),
            'nb_model': downsample(nb_model, 1000),
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(disc):
            print(f"  {i+1}/{len(disc)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"\n  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ========================================================================
#  HTML BUILDERS
# ========================================================================

def build_pd_html(cases_data):
    """Build PD frequency discrepancy review viewer with threshold-click."""
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
<title>PD Freq Discrepancy Review ({n_cases} cases)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #f5f5f5; color: #222; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #fff; border-bottom: 2px solid #ccc;
    flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  .key {{ background: #ddd; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; color: #333; }}

  #progress-bar-wrap {{ width: 100%; height: 6px; background: #ddd; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}

  #mode-indicator {{
    font-size: 18px; font-weight: bold; padding: 6px 20px;
    text-align: center; letter-spacing: 2px;
  }}
  .mode-add {{ background: #e8f5e8; color: #228822; }}
  .mode-delete {{ background: #fce8e8; color: #cc2222; }}
  .mode-nav {{ background: #e8eafc; color: #4455bb; }}

  #info-panel {{
    background: #fff; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #ccc; font-size: 13px;
  }}
  .info-item {{ color: #555; }}
  .info-item strong {{ color: #222; }}
  .diff-high {{ color: #cc0000; font-weight: bold; }}
  .diff-med {{ color: #cc6600; font-weight: bold; }}

  #decision-panel {{
    padding: 10px 16px; background: #fafafa; border-bottom: 2px solid #ccc;
    display: flex; align-items: center; gap: 16px;
  }}
  #decision-status {{
    font-size: 16px; font-weight: bold; padding: 6px 16px;
    border-radius: 6px; letter-spacing: 1px;
  }}
  .decision-none {{ background: #eee; color: #888; }}
  .decision-accept {{ background: #ffe0e0; color: #cc0000; }}
  .decision-keep {{ background: #e0e8ff; color: #0044cc; }}
  .decision-custom {{ background: #e8ffe0; color: #226622; }}

  #freq-buttons {{
    display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 16px;
    background: #fafafa; border-bottom: 1px solid #ccc; align-items: center;
  }}
  #freq-buttons label {{ color: #666; font-size: 13px; margin-right: 8px; font-weight: bold; }}
  .freq-btn {{
    padding: 6px 10px; border: 1px solid #aaa; border-radius: 4px;
    background: #f0f0f0; color: #444; cursor: pointer; font-family: monospace;
    font-size: 13px; font-weight: bold; min-width: 42px; text-align: center;
    transition: all 0.15s;
  }}
  .freq-btn:hover {{ background: #e0e0e0; border-color: #666; color: #000; }}
  .freq-btn.active {{ background: #d4edd4; border-color: #44cc88; color: #226622; }}
  .freq-btn.focused {{ outline: 2px solid #333; outline-offset: 1px; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}
  #evidence-canvas {{ display: block; margin: 4px auto 0 auto; cursor: crosshair; }}

  .action-btn {{
    padding: 8px 18px; border: 2px solid; border-radius: 6px;
    cursor: pointer; font-family: monospace; font-size: 14px; font-weight: bold;
    transition: all 0.15s;
  }}
  .action-btn:hover {{ opacity: 0.85; }}
  .btn-accept {{ border-color: #cc0000; background: #fff0f0; color: #cc0000; }}
  .btn-keep {{ border-color: #0066cc; background: #f0f5ff; color: #0066cc; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #f0fff0; color: #228822; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #e0ffe0; }}

  #save-status {{ color: #228822; font-size: 13px; }}
  #freq-info {{ font-size: 14px; color: #996600; padding: 4px 16px; background: #fffde8; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #fff;
    border-top: 1px solid #ccc; line-height: 1.8;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#cc0000;">PD Freq Discrepancy Review</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptMarkers()">Accept Markers <span class="key">Enter</span></button>
    <button class="action-btn btn-keep" onclick="keepMW()">Keep MW <span class="key">Space</span></button>
    <button class="action-btn" style="border-color:#ff4444; color:#ff4444; background:#3a1a1a;" onclick="rejectCase()">Not PD <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">MW freq: <strong id="info-mw-freq" style="color:#008800;">--</strong></span>
  <span class="info-item">Model freq: <strong id="info-model-freq" style="color:#cc0000;">--</strong></span>
  <span class="info-item">Consensus: <strong id="info-consensus">--</strong></span>
  <span class="info-item">Diff: <strong id="info-diff">--</strong></span>
  <span class="info-item">Raters: <strong id="info-raters">--</strong></span>
  <span class="info-item">Markers: <strong id="info-marker-count" style="color:#cc0000;">--</strong></span>
  <span class="info-item">IPI freq: <strong id="info-ipi-freq" style="color:#996600;">--</strong></span>
</div>

<div id="decision-panel">
  <span>Decision:</span>
  <span id="decision-status" class="decision-none">NOT REVIEWED</span>
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
  <span class="key">Enter</span> Accept markers/IPI freq &nbsp;&nbsp;
  <span class="key">Space</span> Keep MW freq &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Change freq prior &nbsp;&nbsp;
  <span class="key">A</span> Add marker mode &nbsp;&nbsp;
  <span class="key">D</span> Delete marker mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">Ctrl</span> Toggle montage (<span id="montage-label">bipolar</span>) &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const MONO_NAMES = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg',
  'Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg'];
const CAR_DISPLAY_ORDER = [
  0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18
];
const BIPOLAR_DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];

const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;
const EEG_WIDTH = 1400;
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
let montage = 'bipolar';
let mode = 'nav';
let markers = [];
let undoStack = [];
let hoverMarker = -1;
let selectedFreq = null;
let freqBtnIdx = -1;

// Persistence
const STORAGE_KEY = 'pd_freq_discrepancy_review_v1';
let allDecisions = {{}};
try {{ allDecisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allDecisions = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allDecisions)); }}

// Build freq buttons
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

function hppLookup(hppResults, freq) {{
  const keys = [String(freq), freq.toFixed(1), freq.toFixed(2)];
  for (const k of keys) {{
    if (hppResults[k]) return hppResults[k];
  }}
  return [];
}}

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
    const mono = c.mono_data;
    const nCh = mono.length;
    const nSamp = mono[0].length;
    const avg = new Array(nSamp).fill(0);
    for (let ch = 0; ch < nCh; ch++)
      for (let s = 0; s < nSamp; s++) avg[s] += mono[ch][s];
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
  DISPLAY_CHANNELS = getDisplayChannels();
  N_DISPLAY = DISPLAY_CHANNELS.length;
  document.getElementById('montage-label').textContent = montage;
  redraw();
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
  undoStack.push({{ markers: [...markers], freq: selectedFreq }});
  if (undoStack.length > 100) undoStack.shift();
}}

function undo() {{
  if (undoStack.length === 0) return;
  const state = undoStack.pop();
  markers = state.markers;
  selectedFreq = state.freq;
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
  document.getElementById('freq-info').textContent =
    'HPP with freq=' + freq.toFixed(2) + ' Hz -> ' + markers.length + ' discharges';
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

function computeIPIFreq() {{
  if (markers.length < 2) return null;
  const sorted = [...markers].sort((a,b) => a-b);
  const ipis = [];
  for (let i = 1; i < sorted.length; i++) ipis.push(sorted[i] - sorted[i-1]);
  const medIPI = ipis.sort((a,b) => a-b)[Math.floor(ipis.length/2)];
  return medIPI > 0 ? 1.0 / medIPI : null;
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

  // EEG traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    ctx.strokeStyle = '#000000';
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

  // Channel labels
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillStyle = '#000000';
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
  const ipiF = computeIPIFreq();
  const ipiStr = ipiF ? ipiF.toFixed(2) : '??';
  const title = c.segment_id + '  |  ' + c.subtype.toUpperCase() +
    '  |  MW=' + c.mw_freq.toFixed(2) + '  |  Model=' + c.model_freq.toFixed(2) +
    '  |  IPI=' + ipiStr + ' Hz  |  Diff=' + c.diff.toFixed(2);
  ctx.fillText(title, EEG_WIDTH / 2, 6);

  // Discharge markers (red dashed)
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

  // Discharge markers on evidence
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
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-mw-freq').textContent = c.mw_freq.toFixed(2) + ' Hz';
  document.getElementById('info-model-freq').textContent = c.model_freq.toFixed(2) + ' Hz';
  document.getElementById('info-consensus').textContent = c.consensus_freq.toFixed(2) + ' Hz (' + c.n_raters + ' raters)';

  const diffEl = document.getElementById('info-diff');
  diffEl.textContent = c.diff.toFixed(2) + ' Hz';
  diffEl.className = c.diff > 1.0 ? 'diff-high' : 'diff-med';

  const raterStrs = c.ann_raters.map(r => r.rater + '=' + r.freq.toFixed(2));
  document.getElementById('info-raters').textContent = raterStrs.length > 0 ? raterStrs.join(', ') : '--';

  document.getElementById('info-marker-count').textContent = markers.length;

  const ipiF = computeIPIFreq();
  document.getElementById('info-ipi-freq').textContent = ipiF ? ipiF.toFixed(2) + ' Hz' : '--';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  // Decision status
  const decision = allDecisions[c.segment_id];
  const statusEl = document.getElementById('decision-status');
  if (decision) {{
    if (decision.action === 'accept_model') {{
      statusEl.textContent = 'ACCEPT (' + decision.new_freq.toFixed(2) + ' Hz)';
      statusEl.className = 'decision-accept';
    }} else if (decision.action === 'keep_mw') {{
      statusEl.textContent = 'KEEP MW';
      statusEl.className = 'decision-keep';
    }} else {{
      statusEl.textContent = 'CUSTOM (' + decision.new_freq.toFixed(2) + ' Hz)';
      statusEl.className = 'decision-custom';
    }}
  }} else {{
    statusEl.textContent = 'NOT REVIEWED';
    statusEl.className = 'decision-none';
  }}

  const reviewed = Object.keys(allDecisions).length;
  document.getElementById('reviewed-count').textContent = reviewed + ' / ' + CASES.length + ' reviewed';

  // Highlight model freq button
  document.querySelectorAll('.freq-btn').forEach(btn => {{
    btn.classList.remove('est');
  }});
}}

function updateModeIndicator() {{
  const el = document.getElementById('mode-indicator');
  if (mode === 'add') {{ el.textContent = 'ADD MODE (A) -- click to add marker'; el.className = 'mode-add'; }}
  else if (mode === 'delete') {{ el.textContent = 'DELETE MODE (D) -- click near marker to remove'; el.className = 'mode-delete'; }}
  else {{ el.textContent = 'NAVIGATE MODE -- click evidence to set threshold'; el.className = 'mode-nav'; }}
}}

function redraw() {{
  drawEEG();
  drawEvidence();
  updateInfo();
  updateModeIndicator();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load from storage or initialize from model predictions
  const decision = allDecisions[c.segment_id];
  if (decision && decision._markers) {{
    markers = [...decision._markers];
    selectedFreq = decision._selectedFreq || null;
  }} else {{
    // Initialize with model's discharge markers
    markers = [...c.model_discharges];
    selectedFreq = null;
    // Find closest freq button to model freq
    let bestBtn = 0, bestDiff = 999;
    for (let bi = 0; bi < FREQ_BUTTONS.length; bi++) {{
      const diff = Math.abs(FREQ_BUTTONS[bi] - c.model_freq);
      if (diff < bestDiff) {{ bestDiff = diff; bestBtn = bi; }}
    }}
    freqBtnIdx = bestBtn;
    selectedFreq = FREQ_BUTTONS[bestBtn];
  }}
  undoStack = [];
  hoverMarker = -1;

  if (selectedFreq) {{
    freqBtnIdx = FREQ_BUTTONS.indexOf(selectedFreq);
    if (freqBtnIdx < 0) {{
      let best = 0, bestD = Infinity;
      for (let i = 0; i < FREQ_BUTTONS.length; i++) {{
        const d = Math.abs(FREQ_BUTTONS[i] - selectedFreq);
        if (d < bestD) {{ bestD = d; best = i; }}
      }}
      freqBtnIdx = best;
    }}
  }}

  updateFreqHighlight();
  document.getElementById('freq-info').textContent = '';
  redraw();
}}

function acceptMarkers() {{
  // Accept current markers and IPI-derived freq
  const c = CASES[idx];
  const ipiF = computeIPIFreq();
  const newFreq = ipiF || c.model_freq;
  allDecisions[c.segment_id] = {{
    action: 'accept_model',
    new_freq: newFreq,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
    _markers: [...markers],
    _selectedFreq: selectedFreq,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'ACCEPTED (IPI=' + newFreq.toFixed(2) + ' Hz)';
  el.style.color = '#cc0000';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function keepMW() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'keep_mw',
    new_freq: c.mw_freq,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
    _markers: [...markers],
    _selectedFreq: selectedFreq,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'KEPT MW (' + c.mw_freq.toFixed(2) + ' Hz)';
  el.style.color = '#0066cc';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function rejectCase() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'reject_not_pd',
    new_freq: null,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
    n_votes: c.n_votes || 0,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (not PD)';
  el.style.color = '#ff4444';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function exportJSON() {{
  // Build clean export (no internal state)
  const out = {{}};
  for (const sid in allDecisions) {{
    const d = allDecisions[sid];
    out[sid] = {{
      action: d.action,
      new_freq: d.new_freq,
      mw_freq: d.mw_freq,
      model_freq: d.model_freq,
      subtype: d.subtype,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'pd_freq_discrepancy_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' decisions';
  el.style.color = '#228822';
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

// Evidence canvas: click sets threshold in nav mode, or add/delete in those modes
evCanvas.addEventListener('click', function(e) {{
  const rect = evCanvas.getBoundingClientRect();
  const scaleX = EEG_WIDTH / rect.width;
  const scaleY = EV_HEIGHT / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  const evTop = 10;
  const evBottom = EV_HEIGHT - 20;
  const evH = evBottom - evTop;

  if (mode === 'add' || mode === 'delete') {{
    handleCanvasClick(e);
    return;
  }}

  // Nav mode: threshold click -> auto-detect peaks
  const c = CASES[idx];
  const evData = c.evidence;
  if (!evData || evData.length === 0) return;

  const threshold = Math.max(0, Math.min(1, (evBottom - y) / evH));
  if (threshold < 0.01) return;

  const nSamples = evData.length;
  const minDistSamples = Math.max(5, Math.round(nSamples / (DURATION * 4)));

  const peaks = [];
  for (let i = 1; i < nSamples - 1; i++) {{
    if (evData[i] >= threshold && evData[i] > evData[i-1] && evData[i] >= evData[i+1]) {{
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= minDistSamples) {{
        peaks.push(i);
      }} else if (evData[i] > evData[peaks[peaks.length - 1]]) {{
        peaks[peaks.length - 1] = i;
      }}
    }}
  }}

  const newMarkers = peaks.map(i => (i / (nSamples - 1)) * DURATION);

  if (newMarkers.length > 0) {{
    pushUndo();
    markers.length = 0;
    for (const t of newMarkers) markers.push(t);

    // Update freq button to match IPI
    if (markers.length >= 2) {{
      const ipis = [];
      for (let j = 1; j < markers.length; j++) ipis.push(markers[j] - markers[j-1]);
      ipis.sort((a, b) => a - b);
      const medIPI = ipis[Math.floor(ipis.length / 2)];
      const ipiFreq = 1.0 / medIPI;
      let bestBtn = 0, bestDiff = 999;
      for (let bi = 0; bi < FREQ_BUTTONS.length; bi++) {{
        const diff = Math.abs(FREQ_BUTTONS[bi] - ipiFreq);
        if (diff < bestDiff) {{ bestDiff = diff; bestBtn = bi; }}
      }}
      freqBtnIdx = bestBtn;
      selectedFreq = FREQ_BUTTONS[bestBtn];
      updateFreqHighlight();
    }}

    redraw();

    const el = document.getElementById('save-status');
    el.textContent = 'Threshold: ' + threshold.toFixed(2) + ' -> ' + newMarkers.length + ' peaks';
    el.style.color = '#008888';
    setTimeout(() => {{ el.textContent = ''; }}, 2000);
  }}
}});

evCanvas.addEventListener('mousemove', function(e) {{
  if (mode === 'add' || mode === 'delete') {{
    handleCanvasMove(e);
    return;
  }}
  // Nav mode: draw cyan threshold guide line
  const rect = evCanvas.getBoundingClientRect();
  const scaleY = EV_HEIGHT / rect.height;
  const y = (e.clientY - rect.top) * scaleY;
  const evTop = 10;
  const evBottom = EV_HEIGHT - 20;
  const evH = evBottom - evTop;
  const threshold = Math.max(0, Math.min(1, (evBottom - y) / evH));

  drawEvidence();
  if (threshold > 0.01 && threshold < 0.99) {{
    const ctx = evCanvas.getContext('2d');
    const lineY = evBottom - threshold * evH;
    ctx.strokeStyle = 'rgba(0, 200, 200, 0.7)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(PLOT_LEFT, lineY);
    ctx.lineTo(PLOT_RIGHT, lineY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0, 200, 200, 0.9)';
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.fillText(threshold.toFixed(2), PLOT_LEFT - 4, lineY);
  }}
}});

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
  }} else if (e.key === 'ArrowUp') {{
    e.preventDefault();
    if (freqBtnIdx > 0) selectFreqByIndex(freqBtnIdx - 1);
    else if (freqBtnIdx < 0 && FREQ_BUTTONS.length > 0) selectFreqByIndex(0);
  }} else if (e.key === 'ArrowDown') {{
    e.preventDefault();
    if (freqBtnIdx < FREQ_BUTTONS.length - 1) selectFreqByIndex(freqBtnIdx + 1);
    else if (freqBtnIdx < 0 && FREQ_BUTTONS.length > 0) selectFreqByIndex(0);
  }} else if (e.key === 'Enter') {{
    e.preventDefault();
    acceptMarkers();
  }} else if (e.key === ' ') {{
    e.preventDefault();
    keepMW();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
    e.preventDefault();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'x' || e.key === 'X') {{
    rejectCase();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }} else if (e.key === 'Control') {{
    toggleMontage();
  }}
}});

document.addEventListener('keyup', function(e) {{
  if (e.key === 'Control') {{
    toggleMontage();
  }}
}});

// Init
show();
</script>
</body>
</html>"""
    return html


def build_rda_html(cases_data):
    """Build RDA frequency discrepancy review viewer with narrowband overlays."""
    left_indices_json = json.dumps(LEFT_INDICES)
    right_indices_json = json.dumps(RIGHT_INDICES)

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>RDA Freq Discrepancy Review ({n_cases} cases)</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #f5f5f5; color: #222; font-family: 'Consolas','Monaco',monospace; overflow-x: hidden; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #fff; border-bottom: 2px solid #ccc;
    flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}

  .key {{ background: #ddd; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: bold; color: #333; }}

  #progress-bar-wrap {{ width: 100%; height: 6px; background: #ddd; }}
  #progress-bar {{ height: 100%; background: #44cc88; transition: width 0.2s; }}

  #info-panel {{
    background: #fff; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #ccc; font-size: 13px;
  }}
  .info-item {{ color: #555; }}
  .info-item strong {{ color: #222; }}
  .diff-high {{ color: #cc0000; font-weight: bold; }}
  .diff-med {{ color: #cc6600; font-weight: bold; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}

  .action-btn {{
    padding: 8px 18px; border: 2px solid; border-radius: 6px;
    cursor: pointer; font-family: monospace; font-size: 14px; font-weight: bold;
    transition: all 0.15s;
  }}
  .action-btn:hover {{ opacity: 0.85; }}
  .btn-accept {{ border-color: #cc0000; background: #fff0f0; color: #cc0000; }}
  .btn-keep {{ border-color: #0066cc; background: #f0f5ff; color: #0066cc; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #f0fff0; color: #228822; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #e0ffe0; }}

  #save-status {{ color: #228822; font-size: 13px; }}

  #decision-panel {{
    padding: 10px 16px; background: #fafafa; border-bottom: 2px solid #ccc;
    display: flex; align-items: center; gap: 16px;
  }}
  #decision-status {{
    font-size: 16px; font-weight: bold; padding: 6px 16px;
    border-radius: 6px; letter-spacing: 1px;
  }}
  .decision-none {{ background: #eee; color: #888; }}
  .decision-accept {{ background: #ffe0e0; color: #cc0000; }}
  .decision-keep {{ background: #e0e8ff; color: #0044cc; }}

  #legend {{
    padding: 8px 16px; background: #fff; border-bottom: 1px solid #ccc;
    display: flex; gap: 24px; align-items: center; font-size: 13px;
  }}
  .legend-swatch {{
    display: inline-block; width: 30px; height: 4px; vertical-align: middle;
    margin-right: 6px; border-radius: 2px;
  }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #fff;
    border-top: 1px solid #ccc; line-height: 1.8;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#cc0000;">RDA Freq Discrepancy Review</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptModel()">Accept Model <span class="key">Enter</span></button>
    <button class="action-btn btn-keep" onclick="keepMW()">Keep MW <span class="key">Space</span></button>
    <button class="action-btn" style="border-color:#ff4444; color:#ff4444; background:#3a1a1a;" onclick="rejectCase()">Not RDA <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">MW freq: <strong id="info-mw-freq" style="color:#008800;">--</strong></span>
  <span class="info-item">Model freq: <strong id="info-model-freq" style="color:#cc0000;">--</strong></span>
  <span class="info-item">Consensus: <strong id="info-consensus">--</strong></span>
  <span class="info-item">Diff: <strong id="info-diff">--</strong></span>
  <span class="info-item">Raters: <strong id="info-raters">--</strong></span>
</div>

<div id="decision-panel">
  <span>Decision:</span>
  <span id="decision-status" class="decision-none">NOT REVIEWED</span>
</div>

<div id="legend">
  <span><span class="legend-swatch" style="background:rgba(0,136,0,0.6);"></span>MW narrowband</span>
  <span><span class="legend-swatch" style="background:rgba(204,0,0,0.6);"></span>Model narrowband</span>
  <span><span class="legend-swatch" style="background:#000;"></span>EEG trace</span>
  <span style="color:#888; font-size:11px;">Toggle narrowband: <span class="key">N</span> &nbsp; Toggle montage: <span class="key">Ctrl</span></span>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">Enter</span> Accept model freq &nbsp;&nbsp;
  <span class="key">Space</span> Keep MW freq &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">N</span> Toggle narrowband overlay &nbsp;&nbsp;
  <span class="key">Ctrl</span> Toggle montage (<span id="montage-label">bipolar</span>) &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const MONO_NAMES = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg',
  'Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg'];
const CAR_DISPLAY_ORDER = [
  0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, 18
];
const BIPOLAR_DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];

const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;
const EEG_WIDTH = 1400;
const EEG_HEIGHT = 700;
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
let montage = 'bipolar';
let showNarrowband = true;

// Persistence
const STORAGE_KEY = 'rda_freq_discrepancy_review_v1';
let allDecisions = {{}};
try {{ allDecisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allDecisions = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allDecisions)); }}

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
    const mono = c.mono_data;
    const nCh = mono.length;
    const nSamp = mono[0].length;
    const avg = new Array(nSamp).fill(0);
    for (let ch = 0; ch < nCh; ch++)
      for (let s = 0; s < nSamp; s++) avg[s] += mono[ch][s];
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
  DISPLAY_CHANNELS = getDisplayChannels();
  N_DISPLAY = DISPLAY_CHANNELS.length;
  document.getElementById('montage-label').textContent = montage;
  redraw();
}}

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}

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

  // Narrowband overlays (behind EEG traces)
  if (showNarrowband && montage === 'bipolar') {{
    const nbMW = c.nb_mw;
    const nbModel = c.nb_model;
    const nbSamples = nbMW[0].length;

    for (let di = 0; di < N_DISPLAY; di++) {{
      const ch = DISPLAY_CHANNELS[di];
      if (ch.idx < 0 || ch.idx >= nbMW.length) continue;
      const yCenter = MARGIN_TOP + chSpacing * (di + 1);

      // MW narrowband (green)
      ctx.strokeStyle = 'rgba(0, 136, 0, 0.45)';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let si = 0; si < nbSamples; si++) {{
        const x = PLOT_LEFT + (si / (nbSamples - 1)) * PLOT_W;
        let val = nbMW[ch.idx][si];
        val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
        const y = yCenter - val * Z_SCALE * chSpacing;
        if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();

      // Model narrowband (red)
      ctx.strokeStyle = 'rgba(204, 0, 0, 0.45)';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let si = 0; si < nbSamples; si++) {{
        const x = PLOT_LEFT + (si / (nbSamples - 1)) * PLOT_W;
        let val = nbModel[ch.idx][si];
        val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
        const y = yCenter - val * Z_SCALE * chSpacing;
        if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }}
      ctx.stroke();
    }}
  }}

  // EEG traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    ctx.strokeStyle = '#000000';
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

  // Channel labels
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    ctx.fillStyle = '#000000';
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
  const title = c.segment_id + '  |  ' + c.subtype.toUpperCase() +
    '  |  MW=' + c.mw_freq.toFixed(2) + ' Hz  |  Model=' + c.model_freq.toFixed(2) +
    ' Hz  |  Diff=' + c.diff.toFixed(2) + ' Hz';
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-mw-freq').textContent = c.mw_freq.toFixed(2) + ' Hz';
  document.getElementById('info-model-freq').textContent = c.model_freq.toFixed(2) + ' Hz';
  document.getElementById('info-consensus').textContent = c.consensus_freq.toFixed(2) + ' Hz (' + c.n_raters + ' raters)';

  const diffEl = document.getElementById('info-diff');
  diffEl.textContent = c.diff.toFixed(2) + ' Hz';
  diffEl.className = c.diff > 1.0 ? 'diff-high' : 'diff-med';

  const raterStrs = c.ann_raters.map(r => r.rater + '=' + r.freq.toFixed(2));
  document.getElementById('info-raters').textContent = raterStrs.length > 0 ? raterStrs.join(', ') : '--';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  // Decision status
  const decision = allDecisions[c.segment_id];
  const statusEl = document.getElementById('decision-status');
  if (decision) {{
    if (decision.action === 'accept_model') {{
      statusEl.textContent = 'ACCEPT MODEL';
      statusEl.className = 'decision-accept';
    }} else {{
      statusEl.textContent = 'KEEP MW';
      statusEl.className = 'decision-keep';
    }}
  }} else {{
    statusEl.textContent = 'NOT REVIEWED';
    statusEl.className = 'decision-none';
  }}

  const reviewed = Object.keys(allDecisions).length;
  document.getElementById('reviewed-count').textContent = reviewed + ' / ' + CASES.length + ' reviewed';
}}

function redraw() {{
  drawEEG();
  updateInfo();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  redraw();
}}

function acceptModel() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'accept_model',
    new_freq: c.model_freq,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'ACCEPTED MODEL (' + c.model_freq.toFixed(2) + ' Hz)';
  el.style.color = '#cc0000';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function keepMW() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'keep_mw',
    new_freq: c.mw_freq,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'KEPT MW (' + c.mw_freq.toFixed(2) + ' Hz)';
  el.style.color = '#0066cc';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function rejectCase() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'reject_not_rda',
    new_freq: null,
    mw_freq: c.mw_freq,
    model_freq: c.model_freq,
    consensus_freq: c.consensus_freq,
    subtype: c.subtype,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (not RDA)';
  el.style.color = '#ff4444';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function exportJSON() {{
  const out = {{}};
  for (const sid in allDecisions) {{
    const d = allDecisions[sid];
    out[sid] = {{
      action: d.action,
      new_freq: d.new_freq,
      mw_freq: d.mw_freq,
      model_freq: d.model_freq,
      subtype: d.subtype,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rda_freq_discrepancy_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' decisions';
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Keyboard
let ctrlHeld = false;
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Control') {{
    ctrlHeld = true;
    toggleMontage();
    e.preventDefault();
    return;
  }}
  if (e.key === 'Enter') {{
    e.preventDefault();
    acceptModel();
  }} else if (e.key === ' ') {{
    e.preventDefault();
    keepMW();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
    e.preventDefault();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'x' || e.key === 'X') {{
    rejectCase();
  }} else if (e.key === 'n' || e.key === 'N') {{
    showNarrowband = !showNarrowband;
    redraw();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }}
}});

document.addEventListener('keyup', function(e) {{
  if (e.key === 'Control') {{
    ctrlHeld = false;
    toggleMontage();
  }}
}});

// Init
show();
</script>
</body>
</html>"""
    return html


# ========================================================================
#  MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Frequency Discrepancy Reviewer')
    parser.add_argument('--type', type=str, required=True, choices=['pd', 'rda'],
                        help='Type: pd (LPD+GPD) or rda (LRDA+GRDA)')
    parser.add_argument('--max-cases', type=int, default=0,
                        help='Max cases to include (0=all)')
    args = parser.parse_args()

    viewer_type = args.type
    max_cases = args.max_cases

    if viewer_type == 'pd':
        subtypes = ['lpd', 'gpd']
        out_file = 'pd_freq_review.html'
        label = 'PD (LPD+GPD)'
    else:
        subtypes = ['lrda', 'grda']
        out_file = 'rda_freq_review.html'
        label = 'RDA (LRDA+GRDA)'

    print("=" * 70)
    print(f"  {label} Frequency Discrepancy Reviewer")
    print("=" * 70)

    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    print(f"\nFinding discrepancy cases for {label}...")
    disc = find_discrepancy_cases(subtypes, max_cases=max_cases)

    if len(disc) == 0:
        print("  No discrepancy cases found!")
        return

    print(f"\nProcessing {len(disc)} cases...")
    if viewer_type == 'pd':
        cases_data = prepare_pd_cases(disc, sl=sl)
    else:
        cases_data = prepare_rda_cases(disc)

    if len(cases_data) == 0:
        print("  No cases to review!")
        return

    print("\nBuilding HTML viewer...")
    if viewer_type == 'pd':
        html = build_pd_html(cases_data)
    else:
        html = build_rda_html(cases_data)

    out_path = OUT_BASE / out_file
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases ready for review")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


if __name__ == '__main__':
    main()
