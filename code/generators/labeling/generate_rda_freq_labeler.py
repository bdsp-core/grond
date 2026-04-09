"""
RDA Frequency Labeler: Label frequency for unlabeled LRDA/GRDA segments.

Targets segments with >=50% IIIC agreement that lack frequency labels.
Uses W05_DomOnly_IterRefine (Hilbert + iterative narrowband) for default
frequency estimate. Also runs Tautan et al. for comparison.

UI: EEG with narrowband overlay at selected frequency. Frequency buttons
to accept default or pick a custom value. Enter=accept, X=not RDA, arrows=navigate.

Usage:
    conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py
    conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py --max-cases 100
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, filtfilt, detrend, iirnotch, hilbert, welch

LABELING_DIR = Path(__file__).resolve().parent
CODE_DIR = LABELING_DIR.parent.parent  # code/
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_BASE = PROJECT_DIR / 'results' / 'labeling_tools' / 'rda_freq_labeling'
OUT_BASE.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
LOWPASS_HZ = 20.0
NOTCH_HZ = 60.0
MIN_IIIC_AGREE = 0.50

LEFT_CHS = np.array([0, 1, 2, 3, 8, 9, 10, 11])
RIGHT_CHS = np.array([4, 5, 6, 7, 12, 13, 14, 15])

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

FREQ_BUTTONS = [round(0.25 * i, 2) for i in range(1, 17)]  # 0.25 to 4.0


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


# ---------- W05 frequency estimation ----------

def _hilbert_freq_cv(sig):
    """Hilbert instantaneous frequency with CV quality metric."""
    if np.std(sig) < 1e-10:
        return np.nan, 1.0
    analytic = hilbert(sig)
    inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
    mask = (inst_freq > 0.3) & (inst_freq < 4.0)
    valid = inst_freq[mask]
    if len(valid) < 20:
        return np.nan, 1.0
    return float(np.median(valid)), float(np.std(valid) / max(np.median(valid), 1e-6))


def _spectral_peak(sig):
    """Welch spectral peak in delta band."""
    f, pxx = welch(sig, fs=FS, nperseg=400)
    delta = (f >= 0.5) & (f <= 3.5)
    if not delta.any() or pxx[delta].sum() == 0:
        return np.nan
    return float(f[delta][np.argmax(pxx[delta])])


def w05_estimate_freq(seg_bi):
    """W05_DomOnly_IterRefine: two-pass Hilbert frequency from dominant hemisphere.

    Returns: (freq_hz, dom_side)
    """
    # Prefilter 0.3-5 Hz
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos_pre, seg_bi, axis=1)

    # Pass 1: coarse bandpass 0.5-3.5 Hz
    sos1 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_f, axis=1)

    # Coarse lateralization via variance
    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
    dom_side = 'left' if ls1 >= rs1 else 'right'

    # Estimate frequency from top-3 channels of dominant hemisphere
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top3 = dom_chs[np.argsort(powers)[::-1][:3]]
    dom_sig = np.mean(seg_n[top3], axis=0)
    est_freq, _ = _hilbert_freq_cv(dom_sig)
    if not np.isfinite(est_freq):
        est_freq = _spectral_peak(dom_sig)
    if not np.isfinite(est_freq):
        est_freq = 1.5

    # Pass 2: narrowband at estimated freq ± 0.4 Hz
    bw = 0.4
    lo = max(est_freq - bw, 0.1)
    hi = min(est_freq + bw, FS / 2 - 0.1)
    if lo < hi:
        sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
    else:
        seg_nb = seg_n

    # Refined lateralization
    ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
    dom_chs2 = LEFT_CHS if ls >= rs else RIGHT_CHS
    dom_side = 'left' if ls >= rs else 'right'

    # Refined frequency from dominant side in narrowband
    powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs2])
    top3_2 = dom_chs2[np.argsort(powers2)[::-1][:3]]
    dom_sig2 = np.mean(seg_nb[top3_2], axis=0)
    refined_freq, _ = _hilbert_freq_cv(dom_sig2)
    dom_freq = refined_freq if np.isfinite(refined_freq) else est_freq

    # Clamp to valid range
    dom_freq = float(np.clip(dom_freq, 0.25, 4.0))
    return dom_freq, dom_side


# ---------- Tautan et al. frequency estimation ----------

def tautan_estimate_freq(mat_file):
    """Run Tautan et al. (pd_detect_alternate) for frequency."""
    try:
        sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
        import pd_detect_alternate as pddeta
        mono = _load_monopolar(mat_file)
        if mono is None:
            return np.nan
        result = pddeta.pd_detect_alternate(mono.copy(), FS, pk_detect='apd')
        freq = getattr(result, 'event_frequency', np.nan)
        if freq is not None and np.isfinite(freq) and freq > 0:
            return float(freq)
        return np.nan
    except Exception as e:
        return np.nan


# ---------- Narrowband computation ----------

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


# ---------- Case selection ----------

def find_unlabeled_cases(max_cases=0):
    """Find LRDA/GRDA segments with >=50% IIIC agreement that lack frequency labels."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    # Get segments that already have expert frequency labels
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    has_freq_sids = set(ann[ann.frequency_hz.notna()]['segment_id'].unique())

    # Also check segment_labels for MW freq
    mw_freq_sids = set()
    for _, row in sl[sl.expert_freq_hz.notna()].iterrows():
        mw_freq_sids.add(row['mat_file'].replace('.mat', ''))

    labeled_sids = has_freq_sids | mw_freq_sids

    candidates = []
    for _, row in sl.iterrows():
        subtype = row.get('subtype')
        if subtype not in ('lrda', 'grda'):
            continue
        if row.get('excluded') == True:
            continue

        sid = row['mat_file'].replace('.mat', '')

        # Check IIIC agreement
        nv = pd.to_numeric(row.get('iiic_n_votes'), errors='coerce')
        pf = pd.to_numeric(row.get('iiic_plurality_frac'), errors='coerce')
        if not np.isfinite(nv) or nv < 10:
            continue
        if not np.isfinite(pf) or pf < MIN_IIIC_AGREE:
            continue

        # Skip if already labeled
        if sid in labeled_sids:
            continue

        candidates.append({
            'mat_file': str(row['mat_file']),
            'segment_id': sid,
            'patient_id': str(row['patient_id']),
            'subtype': subtype,
            'n_votes': int(nv),
            'plurality_frac': float(pf),
        })

    # Sort by agreement (highest first), then by subtype
    candidates.sort(key=lambda x: (-x['plurality_frac'], x['subtype'], x['segment_id']))

    if max_cases > 0 and len(candidates) > max_cases:
        candidates = candidates[:max_cases]

    # Summary
    lrda_n = sum(1 for c in candidates if c['subtype'] == 'lrda')
    grda_n = sum(1 for c in candidates if c['subtype'] == 'grda')
    print(f"  Found {len(candidates)} unlabeled cases (LRDA: {lrda_n}, GRDA: {grda_n})")
    print(f"  Agreement range: {candidates[-1]['plurality_frac']:.0%} - {candidates[0]['plurality_frac']:.0%}")

    return candidates


# ---------- Data preparation ----------

def prepare_cases(candidates):
    """Load EEG, estimate frequency, compute narrowband for each case."""
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, cand in enumerate(candidates):
        mat_file = cand['mat_file']
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

        # W05 frequency estimate (default)
        w05_freq, dom_side = w05_estimate_freq(seg)

        # Tautan frequency estimate
        tautan_freq = tautan_estimate_freq(mat_file)

        # Snap default to nearest button
        default_freq = min(FREQ_BUTTONS, key=lambda f: abs(f - w05_freq))

        # Store raw bipolar (for JS-side narrowband filtering) at full resolution
        # and display-filtered EEG separately
        case = {
            'segment_id': cand['segment_id'],
            'mat_file': mat_file,
            'patient_id': cand['patient_id'],
            'subtype': cand['subtype'],
            'n_votes': cand['n_votes'],
            'plurality_frac': round(cand['plurality_frac'], 3),
            'w05_freq': round(w05_freq, 3),
            'tautan_freq': round(tautan_freq, 3) if np.isfinite(tautan_freq) else None,
            'default_freq': default_freq,
            'dom_side': dom_side,
            'eeg_data': downsample(seg_display, 800),
            'raw_bipolar': downsample(seg, 400),
        }
        cases_data.append(case)

        if (i + 1) % 25 == 0 or (i + 1) == len(candidates):
            print(f"  {i+1}/{len(candidates)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"\n  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ========================================================================
#  HTML BUILDER
# ========================================================================

def build_html(cases_data, laterality_mode=False, subtype_arg='rda'):
    """Build RDA frequency labeling viewer.

    Parameters
    ----------
    cases_data : list of dict
        Per-segment payloads (eeg + W05/Tautan defaults + dom_side).
    laterality_mode : bool
        If True, render the L/R laterality buttons (LRDA task). The JS export
        will include a `laterality` field on every saved decision.
    subtype_arg : str
        Used only for the page title.
    """
    left_indices_json = json.dumps(LEFT_CHS.tolist())
    right_indices_json = json.dumps(RIGHT_CHS.tolist())
    freq_buttons_json = json.dumps(FREQ_BUTTONS)
    laterality_mode_json = 'true' if laterality_mode else 'false'

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)
    title_subtype = subtype_arg.upper()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title_subtype} Frequency Labeler ({n_cases} cases)</title>
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

  #decision-panel {{
    padding: 10px 16px; background: #fafafa; border-bottom: 2px solid #ccc;
    display: flex; align-items: center; gap: 16px;
  }}
  #decision-status {{
    font-size: 16px; font-weight: bold; padding: 6px 16px;
    border-radius: 6px; letter-spacing: 1px;
  }}
  .decision-none {{ background: #eee; color: #888; }}
  .decision-accept {{ background: #d4edd4; color: #226622; }}
  .decision-reject {{ background: #fce8e8; color: #cc2222; }}

  /* Laterality panel (LRDA mode only) */
  #laterality-panel {{
    background: #f5f5f5; padding: 8px 16px; display: flex; align-items: center;
    gap: 16px; border-bottom: 1px solid #ccc;
  }}
  #laterality-panel label {{ color: #555; font-size: 13px; font-weight: bold; margin-right: 4px; }}
  .lat-badge {{
    font-size: 16px; font-weight: bold; padding: 6px 18px; border-radius: 6px;
    letter-spacing: 1px; border: 2px solid transparent; cursor: pointer;
  }}
  .lat-left  {{ background: #fce8e8; color: #aa1133; border-color: #aa1133; }}
  .lat-right {{ background: #e8eefc; color: #1144aa; border-color: #1144aa; }}
  .lat-none  {{ background: #eee; color: #888; border-color: #ccc; }}
  .lat-badge.selected {{ box-shadow: 0 0 0 3px #44cc88; }}

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
  .freq-btn.default-marker {{ border-bottom: 3px solid #0066cc; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: crosshair; display: block; margin: 0 auto; }}

  .action-btn {{
    padding: 8px 18px; border: 2px solid; border-radius: 6px;
    cursor: pointer; font-family: monospace; font-size: 14px; font-weight: bold;
    transition: all 0.15s;
  }}
  .action-btn:hover {{ opacity: 0.85; }}
  .btn-accept {{ border-color: #228822; background: #f0fff0; color: #228822; }}
  .btn-reject {{ border-color: #cc2222; background: #fff0f0; color: #cc2222; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #f0fff0; color: #228822; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #e0ffe0; }}

  #save-status {{ color: #228822; font-size: 13px; }}

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
    <span style="font-size:16px; font-weight:bold; color:#226622;">{title_subtype} Frequency Labeler</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptFreq()">Accept <span class="key">Enter</span></button>
    <button class="action-btn btn-reject" onclick="rejectCase()">Not {title_subtype} <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">IIIC agreement: <strong id="info-agree">--</strong></span>
  <span class="info-item">W05 estimate: <strong id="info-w05" style="color:#0066cc;">--</strong></span>
  <span class="info-item">Tautan estimate: <strong id="info-tautan" style="color:#996600;">--</strong></span>
  <span class="info-item">Selected: <strong id="info-selected" style="color:#228822;">--</strong></span>
</div>

<div id="decision-panel">
  <span>Decision:</span>
  <span id="decision-status" class="decision-none">NOT REVIEWED</span>
</div>

<div id="laterality-panel" style="display: {('flex' if laterality_mode else 'none')};">
  <label>Laterality:</label>
  <span id="lat-badge-left"  class="lat-badge lat-left  lat-none" onclick="setLaterality('left')">Left  <span class="key">1</span></span>
  <span id="lat-badge-right" class="lat-badge lat-right lat-none" onclick="setLaterality('right')">Right <span class="key">2</span></span>
  <span style="color:#888; font-size:12px;">(W05 default highlighted; press 1/2 to override)</span>
</div>

<div id="freq-buttons">
  <label>Frequency:</label>
</div>

<div id="legend">
  <span><span class="legend-swatch" style="background:rgba(0,136,0,0.6);"></span>Narrowband at selected freq</span>
  <span><span class="legend-swatch" style="background:#000;"></span>EEG trace</span>
  <span style="color:#888; font-size:11px;">Toggle narrowband: <span class="key">N</span></span>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">Enter</span> Accept selected freq &nbsp;&nbsp;
  <span class="key">X</span> Not RDA (reject) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Change frequency &nbsp;&nbsp;
  <span class="key">N</span> Toggle narrowband overlay &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};
const FREQ_BUTTONS = {freq_buttons_json};
const LATERALITY_MODE = {laterality_mode_json};
let selectedLaterality = null;

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

// FFT-based narrowband filter (computed client-side)
function fftNarrowband(signal, centerFreq, bw, fs) {{
  // Zero-pad to power of 2
  const N = signal.length;
  const N2 = 1 << Math.ceil(Math.log2(N));
  // Real FFT via DFT
  const re = new Float64Array(N2);
  const im = new Float64Array(N2);
  for (let i = 0; i < N; i++) re[i] = signal[i];
  fft(re, im, false);
  // Zero out frequencies outside [centerFreq-bw, centerFreq+bw]
  const lo = centerFreq - bw;
  const hi = centerFreq + bw;
  for (let k = 0; k <= N2 / 2; k++) {{
    const freq = k * fs / N2;
    if (freq < lo || freq > hi) {{
      re[k] = 0; im[k] = 0;
      if (k > 0 && k < N2 / 2) {{
        re[N2 - k] = 0; im[N2 - k] = 0;
      }}
    }}
  }}
  fft(re, im, true);
  return Array.from(re.slice(0, N));
}}

function fft(re, im, inverse) {{
  const N = re.length;
  // Bit-reversal
  for (let i = 1, j = 0; i < N; i++) {{
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {{
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }}
  }}
  for (let len = 2; len <= N; len *= 2) {{
    const ang = 2 * Math.PI / len * (inverse ? -1 : 1);
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {{
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {{
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        const vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j] = uRe + vRe;
        im[i + j] = uIm + vIm;
        re[i + j + len/2] = uRe - vRe;
        im[i + j + len/2] = uIm - vIm;
        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }}
    }}
  }}
  if (inverse) {{
    for (let i = 0; i < N; i++) {{ re[i] /= N; im[i] /= N; }}
  }}
}}

// Cache narrowband results per case+freq to avoid recomputation
const nbCache = {{}};

function getNarrowband(caseIdx, freqHz) {{
  const key = caseIdx + '_' + freqHz.toFixed(2);
  if (nbCache[key]) return nbCache[key];
  const c = CASES[caseIdx];
  const raw = c.raw_bipolar;
  const fs = 200;  // downsampled to 1000 pts over 10s = 100 Hz effective
  const effectiveFs = raw[0].length / 10.0;
  const bw = 0.3;
  const result = [];
  for (let ch = 0; ch < raw.length; ch++) {{
    result.push(fftNarrowband(raw[ch], freqHz, bw, effectiveFs));
  }}
  nbCache[key] = result;
  return result;
}}

// State
let idx = 0;
let montage = 'bipolar';
let showNarrowband = true;
let selectedFreqIdx = 0;  // index into FREQ_BUTTONS

// Persistence
const STORAGE_KEY = 'rda_freq_labeling_v1';
let allDecisions = {{}};
try {{ allDecisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allDecisions = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allDecisions)); }}

function getSelectedFreq() {{ return FREQ_BUTTONS[selectedFreqIdx]; }}

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
  return CASES[idx].eeg_data;
}}

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}

function buildFreqButtons() {{
  const container = document.getElementById('freq-buttons');
  // Keep the label
  const label = container.querySelector('label');
  container.innerHTML = '';
  container.appendChild(label);

  const c = CASES[idx];
  const defaultFreq = c.default_freq;

  for (let i = 0; i < FREQ_BUTTONS.length; i++) {{
    const btn = document.createElement('span');
    btn.className = 'freq-btn';
    btn.textContent = FREQ_BUTTONS[i].toFixed(2);
    if (i === selectedFreqIdx) btn.classList.add('active');
    if (FREQ_BUTTONS[i] === defaultFreq) btn.classList.add('default-marker');
    btn.onclick = () => {{
      selectedFreqIdx = i;
      updateFreqButtons();
      redraw();
    }};
    container.appendChild(btn);
  }}
}}

function updateFreqButtons() {{
  const btns = document.querySelectorAll('.freq-btn');
  const c = CASES[idx];
  btns.forEach((btn, i) => {{
    btn.classList.toggle('active', i === selectedFreqIdx);
    btn.classList.toggle('default-marker', FREQ_BUTTONS[i] === c.default_freq);
  }});
  document.getElementById('info-selected').textContent = getSelectedFreq().toFixed(2) + ' Hz';
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

  // Narrowband overlay at selected frequency (computed client-side via FFT)
  if (showNarrowband && montage === 'bipolar') {{
    const nbData = getNarrowband(idx, getSelectedFreq());
    if (nbData) {{
      const nbSamples = nbData[0].length;
      for (let di = 0; di < N_DISPLAY; di++) {{
        const ch = DISPLAY_CHANNELS[di];
        if (ch.idx < 0 || ch.idx >= nbData.length) continue;
        const yCenter = MARGIN_TOP + chSpacing * (di + 1);

        ctx.strokeStyle = 'rgba(0, 136, 0, 0.5)';
        ctx.lineWidth = 2.0;
        ctx.beginPath();
        for (let si = 0; si < nbSamples; si++) {{
          const x = PLOT_LEFT + (si / (nbSamples - 1)) * PLOT_W;
          let val = nbData[ch.idx][si];
          val = Math.max(-CLIP_UV, Math.min(CLIP_UV, val));
          const y = yCenter - val * Z_SCALE * chSpacing;
          if (si === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }}
        ctx.stroke();
      }}
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
    '  |  IIIC ' + (c.plurality_frac * 100).toFixed(0) + '% (' + c.n_votes + ' votes)' +
    '  |  W05=' + c.w05_freq.toFixed(2) + ' Hz' +
    (c.tautan_freq ? '  |  Tautan=' + c.tautan_freq.toFixed(2) + ' Hz' : '');
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-agree').textContent =
    (c.plurality_frac * 100).toFixed(0) + '% (' + c.n_votes + ' votes)';
  document.getElementById('info-w05').textContent = c.w05_freq.toFixed(2) + ' Hz';
  document.getElementById('info-tautan').textContent =
    c.tautan_freq ? c.tautan_freq.toFixed(2) + ' Hz' : '--';
  document.getElementById('info-selected').textContent = getSelectedFreq().toFixed(2) + ' Hz';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  // Decision status
  const decision = allDecisions[c.segment_id];
  const statusEl = document.getElementById('decision-status');
  if (decision) {{
    if (decision.action === 'reject_not_rda') {{
      statusEl.textContent = 'REJECTED (not RDA)';
      statusEl.className = 'decision-reject';
    }} else {{
      statusEl.textContent = 'LABELED: ' + decision.freq.toFixed(2) + ' Hz';
      statusEl.className = 'decision-accept';
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
  updateFreqButtons();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Restore previous decision or use default
  const prev = allDecisions[c.segment_id];
  if (prev && prev.action === 'accept') {{
    // Find the button index for saved freq
    const savedIdx = FREQ_BUTTONS.indexOf(prev.freq);
    selectedFreqIdx = savedIdx >= 0 ? savedIdx : FREQ_BUTTONS.indexOf(c.default_freq);
  }} else {{
    selectedFreqIdx = FREQ_BUTTONS.indexOf(c.default_freq);
    if (selectedFreqIdx < 0) selectedFreqIdx = 3;  // fallback to 1.0 Hz
  }}

  if (LATERALITY_MODE) {{
    if (prev && prev.laterality) {{
      selectedLaterality = prev.laterality;
    }} else {{
      // Default to W05 dom_side; valid values are 'left' or 'right'
      selectedLaterality = (c.dom_side === 'left' || c.dom_side === 'right') ? c.dom_side : 'left';
    }}
    updateLateralityBadges();
  }}

  buildFreqButtons();
  redraw();
}}

function setLaterality(side) {{
  if (!LATERALITY_MODE) return;
  if (side !== 'left' && side !== 'right') return;
  selectedLaterality = side;
  updateLateralityBadges();
}}

function updateLateralityBadges() {{
  if (!LATERALITY_MODE) return;
  const lEl = document.getElementById('lat-badge-left');
  const rEl = document.getElementById('lat-badge-right');
  if (!lEl || !rEl) return;
  lEl.classList.remove('selected', 'lat-none');
  rEl.classList.remove('selected', 'lat-none');
  if (selectedLaterality === 'left') {{
    lEl.classList.add('selected');
    rEl.classList.add('lat-none');
  }} else if (selectedLaterality === 'right') {{
    rEl.classList.add('selected');
    lEl.classList.add('lat-none');
  }} else {{
    lEl.classList.add('lat-none');
    rEl.classList.add('lat-none');
  }}
}}

function acceptFreq() {{
  const c = CASES[idx];
  const freq = getSelectedFreq();
  allDecisions[c.segment_id] = {{
    action: 'accept',
    freq: freq,
    laterality: LATERALITY_MODE ? selectedLaterality : null,
    w05_freq: c.w05_freq,
    w05_laterality: c.dom_side,
    tautan_freq: c.tautan_freq,
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  const latStr = LATERALITY_MODE && selectedLaterality ? ' [' + selectedLaterality + ']' : '';
  el.textContent = 'LABELED: ' + freq.toFixed(2) + ' Hz' + latStr;
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function rejectCase() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'reject_not_rda',
    freq: null,
    laterality: null,
    w05_freq: c.w05_freq,
    w05_laterality: c.dom_side,
    tautan_freq: c.tautan_freq,
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (not RDA)';
  el.style.color = '#cc2222';
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
      freq: d.freq,
      laterality: d.laterality || null,
      w05_freq: d.w05_freq,
      w05_laterality: d.w05_laterality || null,
      tautan_freq: d.tautan_freq,
      subtype: d.subtype,
      mat_file: d.mat_file,
      patient_id: d.patient_id,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rda_freq_labeling_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' decisions';
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Keyboard
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    acceptFreq();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
    e.preventDefault();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'ArrowUp') {{
    e.preventDefault();
    if (selectedFreqIdx < FREQ_BUTTONS.length - 1) {{
      selectedFreqIdx++;
      updateFreqButtons();
      redraw();
    }}
  }} else if (e.key === 'ArrowDown') {{
    e.preventDefault();
    if (selectedFreqIdx > 0) {{
      selectedFreqIdx--;
      updateFreqButtons();
      redraw();
    }}
  }} else if (e.key === 'x' || e.key === 'X') {{
    rejectCase();
  }} else if (e.key === 'n' || e.key === 'N') {{
    showNarrowband = !showNarrowband;
    redraw();
  }} else if (e.key === 'e' || e.key === 'E') {{
    exportJSON();
  }} else if (LATERALITY_MODE && e.key === '1') {{
    e.preventDefault();
    setLaterality('left');
  }} else if (LATERALITY_MODE && e.key === '2') {{
    e.preventDefault();
    setLaterality('right');
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

def _candidates_from_manifest(manifest_path, subtype_filter=None):
    """Build candidates list from a manifest CSV (independent-expert mode).

    The manifest must have a `mat_file` column. Other fields (patient_id,
    subtype, IIIC stats) are looked up from segment_labels.csv when present.
    """
    manifest_df = pd.read_csv(manifest_path)
    if 'mat_file' not in manifest_df.columns:
        raise ValueError(f"Manifest must contain mat_file column; got {list(manifest_df.columns)}")

    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    sl_indexed = sl.set_index('mat_file')
    candidates = []
    for _, mrow in manifest_df.iterrows():
        mf = str(mrow['mat_file'])
        if mf not in sl_indexed.index:
            print(f"  WARNING: {mf} not in segment_labels.csv; skipping")
            continue
        srow = sl_indexed.loc[mf]
        sub = str(srow.get('subtype', '')).lower()
        if subtype_filter and sub != subtype_filter:
            continue
        try:
            nv = float(srow.get('iiic_n_votes') or 0)
        except (TypeError, ValueError):
            nv = 0.0
        try:
            pf = float(srow.get('iiic_plurality_frac') or 0)
        except (TypeError, ValueError):
            pf = 0.0
        candidates.append({
            'mat_file': mf,
            'segment_id': mf.replace('.mat', ''),
            'patient_id': str(srow.get('patient_id', mrow.get('patient_id', ''))),
            'subtype': sub,
            'n_votes': int(nv) if np.isfinite(nv) else 0,
            'plurality_frac': float(pf) if np.isfinite(pf) else 0.0,
        })
    return candidates


def main():
    parser = argparse.ArgumentParser(description='RDA Frequency Labeler')
    parser.add_argument('--max-cases', type=int, default=0,
                        help='Max cases to include (0=all). Used only when --manifest is not given.')
    parser.add_argument('--subtype', type=str, choices=['lrda', 'grda', 'rda'], default='rda',
                        help='Filter to one subtype. lrda enables the laterality (left/right) input UI; '
                             'grda omits it; rda (default) keeps the legacy mixed-pool behavior.')
    parser.add_argument('--manifest', type=str, default=None,
                        help='CSV with mat_file column; bypasses auto-discovery '
                             '(used for the independent-expert annotation tasks).')
    parser.add_argument('--output', type=str, default=None,
                        help='Output HTML path; default: results/labeling_tools/rda_freq_labeling/rda_freq_labeler.html')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not open the generated HTML in a browser.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit cases (for testing); 0=all.')
    args = parser.parse_args()

    # In LRDA mode the UI shows left/right laterality buttons; in GRDA mode it
    # does not (laterality is generalized by definition). The legacy 'rda' mode
    # also omits the buttons.
    laterality_mode = (args.subtype == 'lrda')
    subtype_filter = args.subtype if args.subtype in ('lrda', 'grda') else None

    print("=" * 70)
    print(f"  RDA Frequency Labeler — subtype={args.subtype}, laterality_mode={laterality_mode}")
    print("  Default freq from W05_DomOnly_IterRefine (Hilbert + narrowband)")
    print("=" * 70)

    if args.manifest:
        print(f"\nLoading cases from manifest: {args.manifest}")
        candidates = _candidates_from_manifest(args.manifest, subtype_filter=subtype_filter)
        print(f"  {len(candidates)} segments in manifest after filtering to subtype={subtype_filter}")
    else:
        print(f"\nFinding unlabeled LRDA/GRDA cases...")
        candidates = find_unlabeled_cases(max_cases=args.max_cases)
        if subtype_filter:
            candidates = [c for c in candidates if c['subtype'] == subtype_filter]
            print(f"  Filtered to subtype={subtype_filter}: {len(candidates)}")

    if args.limit > 0 and len(candidates) > args.limit:
        candidates = candidates[:args.limit]
        print(f"  Limited to first {args.limit} cases")

    if len(candidates) == 0:
        print("  No cases to label!")
        return

    print(f"\nProcessing {len(candidates)} cases (loading EEG + estimating freq)...")
    cases_data = prepare_cases(candidates)

    if len(cases_data) == 0:
        print("  No cases to review!")
        return

    print("\nBuilding HTML viewer...")
    html = build_html(cases_data, laterality_mode=laterality_mode, subtype_arg=args.subtype)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = OUT_BASE / 'rda_freq_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases ready for labeling")

    if not args.no_open:
        import subprocess
        subprocess.run(['open', str(out_path)])
    print("=" * 70)


if __name__ == '__main__':
    main()
