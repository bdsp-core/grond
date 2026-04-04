"""
Expert Frequency Labeler: MW frequency labels for original 38-patient expert dataset.

Targets 690 PD/RDA segments (abn*, pat*, emu*) where LB/PH/SZ have frequency
labels but MW does not. Computes default frequency via PDCharacterizer (PD)
or W05 (RDA). Shows existing expert labels (LB, PH, SZ) in info panel.

UI: EEG with bipolar montage + narrowband overlay. Frequency buttons 0.25-4.0 Hz.
Enter=accept, X=reject, arrows=navigate/change freq.

Usage:
    conda run -n morgoth python code/generators/labeling/generate_expert_freq_labeler.py
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
OUT_BASE = PROJECT_DIR / 'results' / 'labeling_tools' / 'expert_freq_labeling'
OUT_BASE.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
LOWPASS_HZ = 20.0
NOTCH_HZ = 60.0

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


# ---------- W05 frequency estimation (for RDA) ----------

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

    # Pass 2: narrowband at estimated freq +/- 0.4 Hz
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


# ---------- PDCharacterizer frequency estimation (for PD) ----------

_pd_char = None

def _get_pd_characterizer():
    global _pd_char
    if _pd_char is None:
        from pd_characterizer import PDCharacterizer
        _pd_char = PDCharacterizer()
    return _pd_char


def pd_estimate_freq(seg_bi, subtype):
    """Use PDCharacterizer for LPD/GPD frequency estimation."""
    try:
        pc = _get_pd_characterizer()
        result = pc.characterize(seg_bi, subtype=subtype)
        freq = result['frequency']
        if freq is not None and np.isfinite(freq) and freq > 0:
            return float(freq)
        return 1.0  # fallback
    except Exception as e:
        print(f"    PDCharacterizer error: {e}")
        return 1.0


# ---------- Case selection ----------

def find_cases():
    """Find expert dataset segments where LB/PH/SZ have freq but MW does not."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    # Expert dataset: abn*, pat*, emu* prefixes
    expert_mask = sl['mat_file'].str.match(r'^(abn|pat|emu)')
    expert_segs = sl[expert_mask]
    not_excluded = expert_segs[expert_segs['excluded'] == False]
    not_excluded_mats = set(not_excluded['mat_file'])

    # Build segment_id -> mat_file map and subtype map from segment_labels
    mat_to_subtype = dict(zip(not_excluded['mat_file'], not_excluded['subtype']))
    mat_to_patient = dict(zip(not_excluded['mat_file'], not_excluded['patient_id']))

    # Filter annotations to expert dataset
    ann_expert = ann[ann['mat_file'].isin(not_excluded_mats)]

    # Segments with LB/PH/SZ freq
    expert_raters = ann_expert[ann_expert['rater'].isin(['LB', 'PH', 'SZ'])]
    has_expert_freq = set(expert_raters[expert_raters['frequency_hz'].notna()]['segment_id'].unique())

    # Segments with MW freq
    mw_ann = ann_expert[ann_expert['rater'] == 'MW']
    mw_freq_sids = set(mw_ann[mw_ann['frequency_hz'].notna()]['segment_id'].unique())

    # Build expert label lookup: segment_id -> {rater: freq}
    expert_labels = {}
    for _, row in expert_raters.iterrows():
        sid = row['segment_id']
        rater = row['rater']
        freq = row['frequency_hz']
        no_pd = row.get('no_pd', 0)
        if sid not in expert_labels:
            expert_labels[sid] = {}
        if pd.notna(freq):
            expert_labels[sid][rater] = float(freq)
        elif pd.notna(no_pd) and float(no_pd) == 1.0:
            expert_labels[sid][rater] = 'no_pd'

    # Candidates
    candidates = []
    for sid in has_expert_freq:
        if sid in mw_freq_sids:
            continue  # MW already labeled
        mat_file = sid + '.mat'
        if mat_file not in not_excluded_mats:
            continue
        subtype = mat_to_subtype.get(mat_file, '')
        if subtype not in ('lpd', 'gpd', 'lrda', 'grda'):
            continue
        patient_id = str(mat_to_patient.get(mat_file, ''))
        labels = expert_labels.get(sid, {})

        candidates.append({
            'mat_file': mat_file,
            'segment_id': sid,
            'patient_id': patient_id,
            'subtype': subtype,
            'lb_freq': labels.get('LB'),
            'ph_freq': labels.get('PH'),
            'sz_freq': labels.get('SZ'),
        })

    # Sort by subtype (lpd, gpd, lrda, grda), then segment_id
    subtype_order = {'lpd': 0, 'gpd': 1, 'lrda': 2, 'grda': 3}
    candidates.sort(key=lambda x: (subtype_order.get(x['subtype'], 9), x['segment_id']))

    # Summary
    from collections import Counter
    subtypes = Counter(c['subtype'] for c in candidates)
    print(f"  Found {len(candidates)} cases needing MW frequency labels")
    for st in ['lpd', 'gpd', 'lrda', 'grda']:
        print(f"    {st.upper()}: {subtypes.get(st, 0)}")

    return candidates


# ---------- Data preparation ----------

def prepare_cases(candidates):
    """Load EEG, estimate frequency, compute data for each case."""
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, cand in enumerate(candidates):
        mat_file = cand['mat_file']
        subtype = cand['subtype']
        seg = load_segment(mat_file)
        if seg is None:
            n_skipped += 1
            continue

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

        # Frequency estimate depends on subtype
        if subtype in ('lpd', 'gpd'):
            model_freq = pd_estimate_freq(seg, subtype)
            model_name = 'PDChar'
        else:
            model_freq, _ = w05_estimate_freq(seg)
            model_name = 'W05'

        # Snap default to nearest button
        default_freq = min(FREQ_BUTTONS, key=lambda f: abs(f - model_freq))

        case = {
            'segment_id': cand['segment_id'],
            'mat_file': mat_file,
            'patient_id': cand['patient_id'],
            'subtype': subtype,
            'lb_freq': cand['lb_freq'],
            'ph_freq': cand['ph_freq'],
            'sz_freq': cand['sz_freq'],
            'model_freq': round(model_freq, 3),
            'model_name': model_name,
            'default_freq': default_freq,
            'eeg_data': downsample(seg_display, 800),
            'raw_bipolar': downsample(seg, 400),
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(candidates):
            print(f"  {i+1}/{len(candidates)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"\n  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ========================================================================
#  HTML BUILDER
# ========================================================================

def build_html(cases_data):
    """Build expert frequency labeling viewer."""
    left_indices_json = json.dumps(LEFT_CHS.tolist())
    right_indices_json = json.dumps(RIGHT_CHS.tolist())
    freq_buttons_json = json.dumps(FREQ_BUTTONS)

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Expert Frequency Labeler ({n_cases} cases)</title>
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

  #expert-panel {{
    background: #fffff0; padding: 8px 16px; border-bottom: 2px solid #e8e8c0;
    display: flex; align-items: center; gap: 24px; font-size: 14px;
  }}
  #expert-panel .expert-label {{ font-weight: bold; color: #666; }}
  .expert-freq {{ font-size: 16px; font-weight: bold; padding: 3px 10px; border-radius: 4px; }}
  .expert-freq-val {{ background: #e8f4e8; color: #226622; }}
  .expert-freq-nopd {{ background: #fce8e8; color: #cc2222; }}
  .expert-freq-na {{ background: #eee; color: #999; }}

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
    <span style="font-size:16px; font-weight:bold; color:#226622;">Expert Frequency Labeler (MW)</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptFreq()">Accept Freq <span class="key">Enter</span></button>
    <button class="action-btn btn-reject" onclick="rejectCase()">Not PD/RDA <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Patient: <strong id="info-patient">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">Model (<span id="info-model-name">--</span>): <strong id="info-model-freq" style="color:#0066cc;">--</strong></span>
  <span class="info-item">Selected: <strong id="info-selected" style="color:#228822;">--</strong></span>
</div>

<div id="expert-panel">
  <span class="expert-label">Expert labels:</span>
  <span>LB: <span id="expert-lb" class="expert-freq expert-freq-na">--</span></span>
  <span>PH: <span id="expert-ph" class="expert-freq expert-freq-na">--</span></span>
  <span>SZ: <span id="expert-sz" class="expert-freq expert-freq-na">--</span></span>
  <span style="color:#888; font-size:12px; margin-left:16px;" id="expert-summary"></span>
</div>

<div id="decision-panel">
  <span>Decision:</span>
  <span id="decision-status" class="decision-none">NOT REVIEWED</span>
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
  <span class="key">X</span> Not PD/RDA (reject) &nbsp;&nbsp;
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

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
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
  const N = signal.length;
  const N2 = 1 << Math.ceil(Math.log2(N));
  const re = new Float64Array(N2);
  const im = new Float64Array(N2);
  for (let i = 0; i < N; i++) re[i] = signal[i];
  fft(re, im, false);
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

// Cache narrowband results per case+freq
const nbCache = {{}};

function getNarrowband(caseIdx, freqHz) {{
  const key = caseIdx + '_' + freqHz.toFixed(2);
  if (nbCache[key]) return nbCache[key];
  const c = CASES[caseIdx];
  const raw = c.raw_bipolar;
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
let showNarrowband = true;
let selectedFreqIdx = 0;

// Persistence
const STORAGE_KEY = 'expert_freq_labeling_v1';
let allDecisions = {{}};
try {{ allDecisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allDecisions = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allDecisions)); }}

function getSelectedFreq() {{ return FREQ_BUTTONS[selectedFreqIdx]; }}

let DISPLAY_CHANNELS = [];
let N_DISPLAY = 0;

function initDisplayChannels() {{
  DISPLAY_CHANNELS = [];
  for (const i of BIPOLAR_DISPLAY_ORDER) {{
    if (i < 0) DISPLAY_CHANNELS.push({{ idx: -1, name: '' }});
    else DISPLAY_CHANNELS.push({{ idx: i, name: BIPOLAR_NAMES[i] }});
  }}
  N_DISPLAY = DISPLAY_CHANNELS.length;
}}

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}

function formatExpertFreq(val) {{
  if (val === null || val === undefined) return ['--', 'expert-freq-na'];
  if (val === 'no_pd') return ['no PD', 'expert-freq-nopd'];
  return [parseFloat(val).toFixed(2) + ' Hz', 'expert-freq-val'];
}}

function buildFreqButtons() {{
  const container = document.getElementById('freq-buttons');
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
  const eegData = c.eeg_data;
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

  // Narrowband overlay
  if (showNarrowband) {{
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
    '  |  ' + c.model_name + '=' + c.model_freq.toFixed(2) + ' Hz';
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-patient').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-model-name').textContent = c.model_name;
  document.getElementById('info-model-freq').textContent = c.model_freq.toFixed(2) + ' Hz';
  document.getElementById('info-selected').textContent = getSelectedFreq().toFixed(2) + ' Hz';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  // Expert labels
  const lbInfo = formatExpertFreq(c.lb_freq);
  const phInfo = formatExpertFreq(c.ph_freq);
  const szInfo = formatExpertFreq(c.sz_freq);
  const lbEl = document.getElementById('expert-lb');
  const phEl = document.getElementById('expert-ph');
  const szEl = document.getElementById('expert-sz');
  lbEl.textContent = lbInfo[0]; lbEl.className = 'expert-freq ' + lbInfo[1];
  phEl.textContent = phInfo[0]; phEl.className = 'expert-freq ' + phInfo[1];
  szEl.textContent = szInfo[0]; szEl.className = 'expert-freq ' + szInfo[1];

  // Expert summary: compute mean of numeric values
  const vals = [c.lb_freq, c.ph_freq, c.sz_freq].filter(v => v !== null && v !== 'no_pd' && v !== undefined);
  const numVals = vals.map(v => parseFloat(v)).filter(v => !isNaN(v));
  const summaryEl = document.getElementById('expert-summary');
  if (numVals.length > 0) {{
    const mean = numVals.reduce((a,b) => a+b, 0) / numVals.length;
    const range = numVals.length > 1 ? (Math.max(...numVals) - Math.min(...numVals)).toFixed(2) : '0.00';
    summaryEl.textContent = 'Mean: ' + mean.toFixed(2) + ' Hz  |  Range: ' + range + ' Hz  |  N=' + numVals.length;
  }} else {{
    summaryEl.textContent = '';
  }}

  // Decision status
  const decision = allDecisions[c.segment_id];
  const statusEl = document.getElementById('decision-status');
  if (decision) {{
    if (decision.action === 'reject') {{
      statusEl.textContent = 'REJECTED (not PD/RDA)';
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
    const savedIdx = FREQ_BUTTONS.indexOf(prev.freq);
    selectedFreqIdx = savedIdx >= 0 ? savedIdx : FREQ_BUTTONS.indexOf(c.default_freq);
  }} else {{
    selectedFreqIdx = FREQ_BUTTONS.indexOf(c.default_freq);
    if (selectedFreqIdx < 0) selectedFreqIdx = 3;  // fallback to 1.0 Hz
  }}

  buildFreqButtons();
  redraw();
}}

function acceptFreq() {{
  const c = CASES[idx];
  const freq = getSelectedFreq();
  allDecisions[c.segment_id] = {{
    action: 'accept',
    freq: freq,
    model_freq: c.model_freq,
    model_name: c.model_name,
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'LABELED: ' + freq.toFixed(2) + ' Hz';
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function rejectCase() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    action: 'reject',
    freq: null,
    model_freq: c.model_freq,
    model_name: c.model_name,
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (not PD/RDA)';
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
      model_freq: d.model_freq,
      model_name: d.model_name,
      subtype: d.subtype,
      mat_file: d.mat_file,
      patient_id: d.patient_id,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'expert_freq_labeling_results.json';
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
  }}
}});

// Init
initDisplayChannels();
show();
</script>
</body>
</html>"""
    return html


# ========================================================================
#  MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='Expert Frequency Labeler')
    parser.add_argument('--max-cases', type=int, default=0,
                        help='Max cases to include (0=all)')
    args = parser.parse_args()

    print("=" * 70)
    print("  Expert Frequency Labeler (MW)")
    print("  690 PD/RDA segments from original expert dataset")
    print("  LB/PH/SZ have freq labels; MW needs to add freq labels")
    print("  PD default: PDCharacterizer | RDA default: W05")
    print("=" * 70)

    print(f"\nFinding cases needing MW frequency labels...")
    candidates = find_cases()

    if args.max_cases > 0 and len(candidates) > args.max_cases:
        candidates = candidates[:args.max_cases]
        print(f"  (limited to {args.max_cases} cases)")

    if len(candidates) == 0:
        print("  No cases to label!")
        return

    print(f"\nProcessing {len(candidates)} cases (loading EEG + estimating freq)...")
    cases_data = prepare_cases(candidates)

    if len(cases_data) == 0:
        print("  No cases to review!")
        return

    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)

    out_path = OUT_BASE / 'expert_freq_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases, {size_mb:.1f} MB")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


if __name__ == '__main__':
    main()
