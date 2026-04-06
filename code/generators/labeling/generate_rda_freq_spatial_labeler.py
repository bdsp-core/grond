"""
Combined RDA Frequency + Spatial Extent Labeler for LRDA/GRDA segments.

Labels both frequency and spatial extent (per-channel involvement) in a single
UI. Targets segments with >=10 IIIC votes that need either frequency OR spatial
extent labels from MW.

UI: EEG with narrowband overlay + channel involvement toggling + frequency buttons.

Usage:
    conda run -n morgoth python code/generators/labeling/generate_rda_freq_spatial_labeler.py
    conda run -n morgoth python code/generators/labeling/generate_rda_freq_spatial_labeler.py --max-cases 50
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
OUT_BASE = PROJECT_DIR / 'results' / 'labeling_tools' / 'rda_freq_spatial'
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

SPATIAL_THRESHOLD = 0.15  # PLV×Amp threshold for default channel involvement


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


# ---------- Spatial scoring ----------

def compute_spatial_scores(seg_bi, freq_hz):
    """Compute PLV x Amp channel scores using rda_spatial_extent."""
    try:
        from rda_spatial_extent import rda_spatial_extent
        result = rda_spatial_extent(seg_bi, freq_hz, threshold=SPATIAL_THRESHOLD,
                                    metric='plv_amp')
        scores = result['channel_scores']
        return [round(float(s), 4) for s in scores]
    except Exception as e:
        print(f"  rda_spatial_extent failed: {e}")
        return [0.0] * 18


# ---------- Case selection ----------

def find_target_cases(max_cases=0):
    """Find LRDA/GRDA segments with >=10 IIIC votes needing freq or spatial labels."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    # MW freq labels from annotations.csv
    mw_ann = ann[ann.rater == 'MW']
    mw_freq_sids = set(mw_ann[mw_ann.frequency_hz.notna()]['segment_id'].unique())

    # MW freq labels from segment_labels.csv (expert_freq_rater='MW')
    sl_mw_freq = sl[
        (sl.expert_freq_hz.notna()) &
        (sl.expert_freq_rater == 'MW')
    ]
    for _, row in sl_mw_freq.iterrows():
        mw_freq_sids.add(row['mat_file'].replace('.mat', ''))

    # MW spatial labels from annotations.csv
    mw_spatial_sids = set(
        mw_ann[mw_ann.spatial_extent.notna()]['segment_id'].unique()
    )

    # Build MW freq lookup (segment_id -> freq value)
    mw_freq_lookup = {}
    for _, row in mw_ann[mw_ann.frequency_hz.notna()].iterrows():
        mw_freq_lookup[row['segment_id']] = float(row['frequency_hz'])
    for _, row in sl_mw_freq.iterrows():
        sid = row['mat_file'].replace('.mat', '')
        if sid not in mw_freq_lookup:
            mw_freq_lookup[sid] = float(row['expert_freq_hz'])

    candidates = []
    for _, row in sl.iterrows():
        subtype = row.get('subtype')
        if subtype not in ('lrda', 'grda'):
            continue
        if row.get('excluded') == True or (
            isinstance(row.get('excluded'), str) and row.get('excluded').lower() == 'true'
        ):
            continue

        nv = pd.to_numeric(row.get('iiic_n_votes'), errors='coerce')
        if not np.isfinite(nv) or nv < 10:
            continue

        sid = row['mat_file'].replace('.mat', '')
        pf = pd.to_numeric(row.get('iiic_plurality_frac'), errors='coerce')
        if not np.isfinite(pf):
            pf = 0.0

        has_mw_freq = sid in mw_freq_sids
        has_mw_spatial = sid in mw_spatial_sids
        needs_freq = not has_mw_freq
        needs_spatial = not has_mw_spatial

        if not needs_freq and not needs_spatial:
            continue

        existing_freq = mw_freq_lookup.get(sid, None)

        candidates.append({
            'mat_file': str(row['mat_file']),
            'segment_id': sid,
            'patient_id': str(row['patient_id']),
            'subtype': subtype,
            'n_votes': int(nv),
            'plurality_frac': float(pf),
            'has_freq': has_mw_freq,
            'existing_freq': existing_freq,
            'needs_freq': needs_freq,
            'needs_spatial': needs_spatial,
        })

    # Sort: highest agreement first, then subtype
    candidates.sort(key=lambda x: (-x['plurality_frac'], x['subtype'], x['segment_id']))

    if max_cases > 0 and len(candidates) > max_cases:
        candidates = candidates[:max_cases]

    lrda_n = sum(1 for c in candidates if c['subtype'] == 'lrda')
    grda_n = sum(1 for c in candidates if c['subtype'] == 'grda')
    need_freq = sum(1 for c in candidates if c['needs_freq'])
    need_spatial = sum(1 for c in candidates if c['needs_spatial'])
    print(f"  Found {len(candidates)} cases (LRDA: {lrda_n}, GRDA: {grda_n})")
    print(f"  Need freq: {need_freq}, need spatial: {need_spatial}")
    if candidates:
        print(f"  Agreement range: {candidates[-1]['plurality_frac']:.0%} - {candidates[0]['plurality_frac']:.0%}")

    return candidates


# ---------- Data preparation ----------

def prepare_cases(candidates):
    """Load EEG, estimate frequency, compute spatial scores for each case."""
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

        # W05 frequency estimate
        w05_freq, dom_side = w05_estimate_freq(seg)

        # Default freq: use existing MW freq if available, else snap W05 to button
        if cand['has_freq'] and cand['existing_freq'] is not None:
            default_freq = min(FREQ_BUTTONS, key=lambda f: abs(f - cand['existing_freq']))
        else:
            default_freq = min(FREQ_BUTTONS, key=lambda f: abs(f - w05_freq))

        # Spatial scores at default frequency
        channel_scores = compute_spatial_scores(seg, default_freq)

        # Default channel count: number of channels above threshold
        default_n = sum(1 for s in channel_scores if s > SPATIAL_THRESHOLD)

        case = {
            'segment_id': cand['segment_id'],
            'mat_file': mat_file,
            'patient_id': cand['patient_id'],
            'subtype': cand['subtype'],
            'n_votes': cand['n_votes'],
            'plurality_frac': round(cand['plurality_frac'], 3),
            'w05_freq': round(w05_freq, 3),
            'default_freq': default_freq,
            'has_freq': cand['has_freq'],
            'existing_freq': cand['existing_freq'],
            'dom_side': dom_side,
            'channel_scores': channel_scores,
            'default_n': default_n,
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

def build_html(cases_data):
    """Build combined RDA frequency + spatial extent labeling viewer."""
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
<title>RDA Freq+Spatial Labeler ({n_cases} cases)</title>
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
  .freq-btn.existing-marker {{ border-top: 3px solid #ff8800; }}

  #channel-buttons {{
    display: flex; flex-wrap: wrap; gap: 4px; padding: 8px 16px;
    background: #fafafa; border-bottom: 1px solid #ccc; align-items: center;
  }}
  #channel-buttons label {{ color: #666; font-size: 13px; margin-right: 8px; font-weight: bold; }}
  .ch-btn {{
    padding: 6px 10px; border: 1px solid #aaa; border-radius: 4px;
    background: #f0f0f0; color: #444; cursor: pointer; font-family: monospace;
    font-size: 13px; font-weight: bold; min-width: 42px; text-align: center;
    transition: all 0.15s;
  }}
  .ch-btn:hover {{ background: #e0e0e0; border-color: #666; color: #000; }}
  .ch-btn.active {{ background: #d4edd4; border-color: #44cc88; color: #226622; }}
  .ch-btn.default-marker {{ border-bottom: 3px solid #0066cc; }}

  #canvas-container {{ text-align: center; padding: 8px; position: relative; }}
  #eeg-canvas {{ cursor: pointer; display: block; margin: 0 auto; }}

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
    <span style="font-size:16px; font-weight:bold; color:#226622;">RDA Freq+Spatial Labeler</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptCase()">Accept <span class="key">Enter</span></button>
    <button class="action-btn btn-reject" onclick="rejectCase()">Not RDA <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">IIIC: <strong id="info-agree">--</strong></span>
  <span class="info-item">W05: <strong id="info-w05" style="color:#0066cc;">--</strong></span>
  <span class="info-item">MW freq: <strong id="info-existing" style="color:#ff8800;">--</strong></span>
  <span class="info-item">Freq: <strong id="info-selected" style="color:#228822;">--</strong></span>
  <span class="info-item">Channels: <strong id="info-channels" style="color:#228822;">--</strong></span>
</div>

<div id="decision-panel">
  <span>Decision:</span>
  <span id="decision-status" class="decision-none">NOT REVIEWED</span>
</div>

<div id="freq-buttons">
  <label>Frequency:</label>
</div>

<div id="channel-buttons">
  <label>Channels:</label>
</div>

<div id="legend">
  <span><span class="legend-swatch" style="background:rgba(0,136,0,0.6);"></span>Narrowband overlay</span>
  <span><span class="legend-swatch" style="background:#cc0000;"></span>Involved channel</span>
  <span><span class="legend-swatch" style="background:#888;"></span>Uninvolved channel</span>
  <span style="color:#0066cc; font-size:11px;">Blue underline = W05 default</span>
  <span style="color:#ff8800; font-size:11px;">Orange top = existing MW freq</span>
</div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">Enter</span> Accept freq+spatial &nbsp;&nbsp;
  <span class="key">X</span> Not RDA (reject) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Change frequency &nbsp;&nbsp;
  <span class="key">Shift+&uarr;</span>/<span class="key">Shift+&darr;</span> Change channel count &nbsp;&nbsp;
  Click channel trace to toggle involvement &nbsp;&nbsp;
  <span class="key">N</span> Toggle narrowband &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};
const FREQ_BUTTONS = {freq_buttons_json};
const SPATIAL_THRESHOLD = {SPATIAL_THRESHOLD};

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const BIPOLAR_DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];

const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;
const EEG_WIDTH = 1400;
const EEG_HEIGHT = 750;
const MARGIN_LEFT = 120;
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
let selectedChannels = new Set();  // involved bipolar channel indices (0-17)
let useManualMode = false;  // once user clicks a channel, switch to manual
let manualOverrides = new Set();  // channels manually toggled

// Persistence
const STORAGE_KEY = 'iiic_rda_freq_spatial_v1';
let allDecisions = {{}};
try {{ allDecisions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allDecisions = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allDecisions)); }}

function getSelectedFreq() {{ return FREQ_BUTTONS[selectedFreqIdx]; }}

function getDisplayChannels() {{
  const dc = [];
  for (const i of BIPOLAR_DISPLAY_ORDER) {{
    if (i < 0) dc.push({{ idx: -1, name: '' }});
    else dc.push({{ idx: i, name: BIPOLAR_NAMES[i] }});
  }}
  return dc;
}}

const DISPLAY_CHANNELS = getDisplayChannels();
const N_DISPLAY = DISPLAY_CHANNELS.length;

function timeToX(t) {{ return PLOT_LEFT + (t / DURATION) * PLOT_W; }}

// Compute default channels from scores using threshold
function computeDefaultChannels(scores, threshold) {{
  const channels = new Set();
  for (let ch = 0; ch < 18; ch++) {{
    if (scores[ch] > threshold) channels.add(ch);
  }}
  return channels;
}}

// Set selected channels based on channel count (top N by score)
function setChannelsByCount(n) {{
  const c = CASES[idx];
  const scores = c.channel_scores;
  // Sort channels by score descending
  const ranked = scores.map((s, i) => [i, s]).sort((a, b) => b[1] - a[1]);
  selectedChannels = new Set();
  for (let i = 0; i < Math.min(n, 18); i++) {{
    selectedChannels.add(ranked[i][0]);
  }}
  useManualMode = false;
  manualOverrides = new Set();
}}

function buildFreqButtons() {{
  const container = document.getElementById('freq-buttons');
  const label = container.querySelector('label');
  container.innerHTML = '';
  container.appendChild(label);

  const c = CASES[idx];
  const w05Snap = findNearestFreqIdx(c.w05_freq);
  const existingSnap = c.existing_freq != null ? findNearestFreqIdx(c.existing_freq) : -1;

  for (let i = 0; i < FREQ_BUTTONS.length; i++) {{
    const btn = document.createElement('span');
    btn.className = 'freq-btn';
    btn.textContent = FREQ_BUTTONS[i].toFixed(2);
    if (i === selectedFreqIdx) btn.classList.add('active');
    if (i === w05Snap) btn.classList.add('default-marker');
    if (i === existingSnap) btn.classList.add('existing-marker');
    btn.onclick = () => {{
      selectedFreqIdx = i;
      updateFreqButtons();
      redraw();
    }};
    container.appendChild(btn);
  }}
}}

function findNearestFreqIdx(freq) {{
  let best = 0, bestDist = 999;
  for (let i = 0; i < FREQ_BUTTONS.length; i++) {{
    const d = Math.abs(FREQ_BUTTONS[i] - freq);
    if (d < bestDist) {{ bestDist = d; best = i; }}
  }}
  return best;
}}

function updateFreqButtons() {{
  const btns = document.querySelectorAll('.freq-btn');
  const c = CASES[idx];
  const w05Snap = findNearestFreqIdx(c.w05_freq);
  const existingSnap = c.existing_freq != null ? findNearestFreqIdx(c.existing_freq) : -1;
  btns.forEach((btn, i) => {{
    btn.classList.toggle('active', i === selectedFreqIdx);
    btn.classList.toggle('default-marker', i === w05Snap);
    btn.classList.toggle('existing-marker', i === existingSnap);
  }});
  document.getElementById('info-selected').textContent = getSelectedFreq().toFixed(2) + ' Hz';
}}

function buildChannelButtons() {{
  const container = document.getElementById('channel-buttons');
  const label = container.querySelector('label');
  container.innerHTML = '';
  container.appendChild(label);

  const c = CASES[idx];

  for (let n = 0; n <= 18; n++) {{
    const btn = document.createElement('span');
    btn.className = 'ch-btn';
    btn.textContent = n + '/18';
    if (selectedChannels.size === n && !useManualMode) btn.classList.add('active');
    if (n === c.default_n) btn.classList.add('default-marker');
    btn.onclick = () => {{
      setChannelsByCount(n);
      updateChannelButtons();
      redraw();
    }};
    container.appendChild(btn);
  }}
}}

function updateChannelButtons() {{
  const btns = document.querySelectorAll('.ch-btn');
  const c = CASES[idx];
  const nSel = selectedChannels.size;
  btns.forEach((btn, n) => {{
    btn.classList.toggle('active', nSel === n && !useManualMode);
    btn.classList.toggle('default-marker', n === c.default_n);
  }});
  document.getElementById('info-channels').textContent = nSel + '/18';
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

  // Narrowband overlay at selected frequency (computed client-side via FFT)
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

  // EEG traces — color by involvement
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    const isInvolved = selectedChannels.has(ch.idx);
    ctx.strokeStyle = isInvolved ? '#cc0000' : '#888888';
    ctx.lineWidth = isInvolved ? 1.0 : 0.6;
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

  // Channel labels + PLV*Amp scores
  ctx.font = '10px Consolas, Monaco, monospace';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const isInvolved = selectedChannels.has(ch.idx);
    const score = c.channel_scores[ch.idx] || 0;

    // Channel name (right-aligned before plot)
    ctx.textAlign = 'right';
    ctx.fillStyle = isInvolved ? '#cc0000' : '#888888';
    ctx.font = (isInvolved ? 'bold ' : '') + '10px Consolas, Monaco, monospace';
    ctx.fillText(ch.name, PLOT_LEFT - 30, yCenter);

    // Score (right-aligned, before channel name)
    ctx.textAlign = 'right';
    ctx.fillStyle = score > SPATIAL_THRESHOLD ? '#0066cc' : '#bbb';
    ctx.font = '9px Consolas, Monaco, monospace';
    ctx.fillText(score.toFixed(2), PLOT_LEFT - 4, yCenter);
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
  const freqInfo = c.has_freq ? ('MW=' + c.existing_freq.toFixed(2) + ' Hz') : ('W05=' + c.w05_freq.toFixed(2) + ' Hz');
  const title = c.segment_id + '  |  ' + c.subtype.toUpperCase() +
    '  |  IIIC ' + (c.plurality_frac * 100).toFixed(0) + '% (' + c.n_votes + ' votes)' +
    '  |  ' + freqInfo +
    '  |  ' + selectedChannels.size + '/18 channels';
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-agree').textContent =
    (c.plurality_frac * 100).toFixed(0) + '% (' + c.n_votes + ' votes)';
  document.getElementById('info-w05').textContent = c.w05_freq.toFixed(2) + ' Hz';
  document.getElementById('info-existing').textContent =
    c.has_freq && c.existing_freq != null ? c.existing_freq.toFixed(2) + ' Hz' : '--';
  document.getElementById('info-selected').textContent = getSelectedFreq().toFixed(2) + ' Hz';
  document.getElementById('info-channels').textContent = selectedChannels.size + '/18';

  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';

  // Decision status
  const decision = allDecisions[c.segment_id];
  const statusEl = document.getElementById('decision-status');
  if (decision) {{
    if (decision.rejected) {{
      statusEl.textContent = 'REJECTED (not RDA)';
      statusEl.className = 'decision-reject';
    }} else {{
      statusEl.textContent = 'LABELED: ' + decision.freq.toFixed(2) + ' Hz, ' + decision.n_channels + '/18 ch';
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
  updateChannelButtons();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Restore previous decision or use defaults
  const prev = allDecisions[c.segment_id];
  if (prev && !prev.rejected) {{
    // Restore freq
    const savedFreqIdx = FREQ_BUTTONS.indexOf(prev.freq);
    selectedFreqIdx = savedFreqIdx >= 0 ? savedFreqIdx : findNearestFreqIdx(c.default_freq);
    // Restore channels
    if (prev.selected_channels) {{
      selectedChannels = new Set(prev.selected_channels);
      useManualMode = true;
    }} else {{
      setChannelsByCount(prev.n_channels || c.default_n);
    }}
  }} else if (prev && prev.rejected) {{
    selectedFreqIdx = findNearestFreqIdx(c.default_freq);
    selectedChannels = new Set();
    useManualMode = false;
  }} else {{
    // Fresh case: use defaults
    selectedFreqIdx = findNearestFreqIdx(c.default_freq);
    selectedChannels = computeDefaultChannels(c.channel_scores, SPATIAL_THRESHOLD);
    useManualMode = false;
    manualOverrides = new Set();
  }}

  buildFreqButtons();
  buildChannelButtons();
  redraw();
}}

function acceptCase() {{
  const c = CASES[idx];
  const freq = getSelectedFreq();
  const nChannels = selectedChannels.size;
  const freqChanged = c.has_freq
    ? (freq !== c.existing_freq)
    : (freq !== c.default_freq);

  allDecisions[c.segment_id] = {{
    segment_id: c.segment_id,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
    subtype: c.subtype,
    freq: freq,
    n_channels: nChannels,
    spatial_extent: Math.round(nChannels / 18.0 * 1000) / 1000,
    selected_channels: [...selectedChannels],
    freq_changed: freqChanged,
    rejected: false,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'LABELED: ' + freq.toFixed(2) + ' Hz, ' + nChannels + '/18 ch';
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 2000);
  idx = Math.min(CASES.length - 1, idx + 1);
  show();
}}

function rejectCase() {{
  const c = CASES[idx];
  allDecisions[c.segment_id] = {{
    segment_id: c.segment_id,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
    subtype: c.subtype,
    freq: null,
    n_channels: 0,
    spatial_extent: 0,
    selected_channels: [],
    freq_changed: false,
    rejected: true,
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
      segment_id: d.segment_id,
      mat_file: d.mat_file,
      patient_id: d.patient_id,
      subtype: d.subtype,
      freq: d.freq,
      n_channels: d.n_channels,
      spatial_extent: d.spatial_extent,
      freq_changed: d.freq_changed,
      rejected: d.rejected,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'iiic_rda_freq_spatial_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' decisions';
  el.style.color = '#228822';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Click on EEG canvas to toggle channel involvement
document.getElementById('eeg-canvas').addEventListener('click', function(e) {{
  const rect = this.getBoundingClientRect();
  const clickY = (e.clientY - rect.top) * (EEG_HEIGHT / rect.height);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);
  let bestDi = -1;
  let bestDist = 999;
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const dist = Math.abs(clickY - yCenter);
    if (dist < bestDist) {{
      bestDist = dist;
      bestDi = di;
    }}
  }}
  if (bestDi < 0 || bestDist > chSpacing * 0.6) return;

  const clickedChIdx = DISPLAY_CHANNELS[bestDi].idx;
  useManualMode = true;
  manualOverrides.add(clickedChIdx);

  if (selectedChannels.has(clickedChIdx)) {{
    selectedChannels.delete(clickedChIdx);
  }} else {{
    selectedChannels.add(clickedChIdx);
  }}
  redraw();
}});

// Keyboard
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Enter') {{
    e.preventDefault();
    acceptCase();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
    e.preventDefault();
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'ArrowUp' && e.shiftKey) {{
    // Shift+Up: increase channel count
    e.preventDefault();
    const newN = Math.min(18, selectedChannels.size + 1);
    setChannelsByCount(newN);
    redraw();
  }} else if (e.key === 'ArrowDown' && e.shiftKey) {{
    // Shift+Down: decrease channel count
    e.preventDefault();
    const newN = Math.max(0, selectedChannels.size - 1);
    setChannelsByCount(newN);
    redraw();
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
show();
</script>
</body>
</html>"""
    return html


# ========================================================================
#  MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description='RDA Freq+Spatial Labeler')
    parser.add_argument('--max-cases', type=int, default=0,
                        help='Max cases to include (0=all)')
    args = parser.parse_args()

    print("=" * 70)
    print("  RDA Freq+Spatial Labeler")
    print("  Combined frequency + spatial extent labeling for LRDA/GRDA")
    print("  Targets segments with >=10 IIIC votes needing freq or spatial")
    print("=" * 70)

    print(f"\nFinding target LRDA/GRDA cases...")
    candidates = find_target_cases(max_cases=args.max_cases)

    if len(candidates) == 0:
        print("  No cases to label!")
        return

    print(f"\nProcessing {len(candidates)} cases (loading EEG + freq + spatial)...")
    cases_data = prepare_cases(candidates)

    if len(cases_data) == 0:
        print("  No cases to review!")
        return

    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)

    out_path = OUT_BASE / 'rda_freq_spatial_labeler.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases ready for labeling")

    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


if __name__ == '__main__':
    main()
