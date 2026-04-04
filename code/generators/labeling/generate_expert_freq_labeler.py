"""
Expert Frequency Labeler: MW frequency labels for original 38-patient expert dataset.

Generates TWO HTML files:
  1. expert_freq_labeler_pd.html  — PD (LPD+GPD) with evidence trace + HPP discharge markers
  2. expert_freq_labeler_rda.html — RDA (LRDA+GRDA) with client-side FFT narrowband overlay

Targets segments where LB/PH/SZ have frequency labels but MW does not.

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

from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence, _aggregate_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_BASE = PROJECT_DIR / 'results' / 'labeling_tools' / 'expert_freq_labeling'
OUT_BASE.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
LOWPASS_HZ = 20.0
NOTCH_HZ = 60.0

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
LEFT_CHS = np.array(LEFT_INDICES)
RIGHT_CHS = np.array(RIGHT_INDICES)

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


# ---------- HPP evidence and discharge marking (for PD) ----------

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
    """W05_DomOnly_IterRefine: two-pass Hilbert frequency from dominant hemisphere."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_f = sosfiltfilt(sos_pre, seg_bi, axis=1)

    sos1 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_f, axis=1)

    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls1 >= rs1 else RIGHT_CHS
    dom_side = 'left' if ls1 >= rs1 else 'right'

    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top3 = dom_chs[np.argsort(powers)[::-1][:3]]
    dom_sig = np.mean(seg_n[top3], axis=0)
    est_freq, _ = _hilbert_freq_cv(dom_sig)
    if not np.isfinite(est_freq):
        est_freq = _spectral_peak(dom_sig)
    if not np.isfinite(est_freq):
        est_freq = 1.5

    bw = 0.4
    lo = max(est_freq - bw, 0.1)
    hi = min(est_freq + bw, FS / 2 - 0.1)
    if lo < hi:
        sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos2, seg_f, axis=1)
    else:
        seg_nb = seg_n

    ls = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
    dom_chs2 = LEFT_CHS if ls >= rs else RIGHT_CHS
    dom_side = 'left' if ls >= rs else 'right'

    powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs2])
    top3_2 = dom_chs2[np.argsort(powers2)[::-1][:3]]
    dom_sig2 = np.mean(seg_nb[top3_2], axis=0)
    refined_freq, _ = _hilbert_freq_cv(dom_sig2)
    dom_freq = refined_freq if np.isfinite(refined_freq) else est_freq

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
        return 1.0
    except Exception as e:
        print(f"    PDCharacterizer error: {e}")
        return 1.0


# ---------- Case selection ----------

def find_cases():
    """Find expert dataset segments where LB/PH/SZ have freq but MW does not."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    expert_mask = sl['mat_file'].str.match(r'^(abn|pat|emu)')
    expert_segs = sl[expert_mask]
    not_excluded = expert_segs[expert_segs['excluded'] == False]
    not_excluded_mats = set(not_excluded['mat_file'])

    mat_to_subtype = dict(zip(not_excluded['mat_file'], not_excluded['subtype']))
    mat_to_patient = dict(zip(not_excluded['mat_file'], not_excluded['patient_id']))

    ann_expert = ann[ann['mat_file'].isin(not_excluded_mats)]

    expert_raters = ann_expert[ann_expert['rater'].isin(['LB', 'PH', 'SZ'])]
    has_expert_freq = set(expert_raters[expert_raters['frequency_hz'].notna()]['segment_id'].unique())

    mw_ann = ann_expert[ann_expert['rater'] == 'MW']
    mw_freq_sids = set(mw_ann[mw_ann['frequency_hz'].notna()]['segment_id'].unique())

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

    candidates = []
    for sid in has_expert_freq:
        if sid in mw_freq_sids:
            continue
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

    subtype_order = {'lpd': 0, 'gpd': 1, 'lrda': 2, 'grda': 3}
    candidates.sort(key=lambda x: (subtype_order.get(x['subtype'], 9), x['segment_id']))

    from collections import Counter
    subtypes = Counter(c['subtype'] for c in candidates)
    print(f"  Found {len(candidates)} cases needing MW frequency labels")
    for st in ['lpd', 'gpd', 'lrda', 'grda']:
        print(f"    {st.upper()}: {subtypes.get(st, 0)}")

    return candidates


# ---------- PD data preparation ----------

def prepare_pd_cases(candidates):
    """Prepare PD (LPD+GPD) cases: EEG + evidence + HPP markers at multiple freqs."""
    pd_candidates = [c for c in candidates if c['subtype'] in ('lpd', 'gpd')]
    print(f"\n  Preparing {len(pd_candidates)} PD cases...")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, cand in enumerate(pd_candidates):
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

        # Compute evidence
        evidence = compute_evidence(seg, laterality=None)
        ev_max = np.max(evidence)
        ev_display = evidence / ev_max if ev_max > 0 else evidence

        # HPP results at all frequency buttons
        hpp_results = precompute_hpp_results(evidence, FREQ_BUTTONS)

        # PDCharacterizer default frequency
        model_freq = pd_estimate_freq(seg, subtype)
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
            'default_freq': default_freq,
            'eeg_data': downsample(seg_display, 800),
            'evidence': downsample(ev_display, 500),
            'hpp_results': hpp_results,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(pd_candidates):
            print(f"    {i+1}/{len(pd_candidates)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"  PD total: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ---------- RDA data preparation ----------

def prepare_rda_cases(candidates):
    """Prepare RDA (LRDA+GRDA) cases: EEG + raw bipolar for client-side FFT narrowband."""
    rda_candidates = [c for c in candidates if c['subtype'] in ('lrda', 'grda')]
    print(f"\n  Preparing {len(rda_candidates)} RDA cases...")

    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    b_notch, a_notch = iirnotch(NOTCH_HZ, 30.0, FS)

    cases_data = []
    n_skipped = 0

    for i, cand in enumerate(rda_candidates):
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

        # W05 default frequency
        model_freq, _ = w05_estimate_freq(seg)
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
            'default_freq': default_freq,
            'eeg_data': downsample(seg_display, 800),
            'raw_bipolar': downsample(seg, 400),
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(rda_candidates):
            print(f"    {i+1}/{len(rda_candidates)} processed ({len(cases_data)} valid, {n_skipped} skipped)")

    print(f"  RDA total: {len(cases_data)} (skipped {n_skipped} missing EEG)")
    return cases_data


# ========================================================================
#  PD HTML BUILDER
# ========================================================================

def build_pd_html(cases_data):
    """Build PD frequency labeling viewer with evidence trace + HPP markers."""
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
<title>Expert PD Freq Labeler ({n_cases} cases)</title>
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
  .btn-accept {{ border-color: #228822; background: #f0fff0; color: #228822; }}
  .btn-reject {{ border-color: #cc2222; background: #fff0f0; color: #cc2222; }}

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
    <span style="font-size:16px; font-weight:bold; color:#226622;">Expert PD Freq Labeler (MW)</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptFreq()">Accept Freq <span class="key">Enter</span></button>
    <button class="action-btn btn-reject" onclick="rejectCase()">Not PD <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="reviewed-count" style="font-size:12px; color:#888;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="mode-indicator" class="mode-nav">NAVIGATE MODE</div>

<div id="info-panel">
  <span class="info-item">Segment: <strong id="info-sid">--</strong></span>
  <span class="info-item">Patient: <strong id="info-patient">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype">--</strong></span>
  <span class="info-item">PDChar: <strong id="info-model-freq" style="color:#0066cc;">--</strong></span>
  <span class="info-item">Markers: <strong id="info-marker-count" style="color:#cc0000;">--</strong></span>
  <span class="info-item">IPI freq: <strong id="info-ipi-freq" style="color:#996600;">--</strong></span>
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
  <label>Freq prior:</label>
</div>
<div id="freq-info"></div>

<div id="canvas-container">
  <canvas id="eeg-canvas"></canvas>
  <canvas id="evidence-canvas"></canvas>
</div>

<div id="shortcuts">
  <span class="key">Enter</span> Accept IPI freq &nbsp;&nbsp;
  <span class="key">X</span> Not PD (reject) &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">&uarr;</span>/<span class="key">&darr;</span> Change freq prior &nbsp;&nbsp;
  <span class="key">A</span> Add marker mode &nbsp;&nbsp;
  <span class="key">D</span> Delete marker mode &nbsp;&nbsp;
  <span class="key">Esc</span> Navigate mode &nbsp;&nbsp;
  <span class="key">Z</span> Undo &nbsp;&nbsp;
  <span class="key">E</span> Export JSON
</div>

<script>
const CASES = {cases_json};
const FREQ_BUTTONS = {freq_btns_json};
const LEFT_INDICES = {left_indices_json};
const RIGHT_INDICES = {right_indices_json};

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
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
let mode = 'nav';
let markers = [];
let undoStack = [];
let hoverMarker = -1;
let selectedFreq = null;
let freqBtnIdx = -1;

// Persistence
const STORAGE_KEY = 'expert_freq_pd_v1';
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

function formatExpertFreq(val) {{
  if (val === null || val === undefined) return ['--', 'expert-freq-na'];
  if (val === 'no_pd') return ['no PD', 'expert-freq-nopd'];
  return [parseFloat(val).toFixed(2) + ' Hz', 'expert-freq-val'];
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
    '  |  PDChar=' + c.model_freq.toFixed(2) + '  |  IPI=' + ipiStr + ' Hz';
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
  document.getElementById('info-patient').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-model-freq').textContent = c.model_freq.toFixed(2) + ' Hz';
  document.getElementById('info-marker-count').textContent = markers.length;

  const ipiF = computeIPIFreq();
  document.getElementById('info-ipi-freq').textContent = ipiF ? ipiF.toFixed(2) + ' Hz' : '--';

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
      statusEl.textContent = 'REJECTED (not PD)';
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

  // Load from storage or initialize
  const decision = allDecisions[c.segment_id];
  if (decision && decision._markers) {{
    markers = [...decision._markers];
    selectedFreq = decision._selectedFreq || null;
  }} else {{
    // Initialize with HPP markers at default freq
    let bestBtn = 0, bestDiff = 999;
    for (let bi = 0; bi < FREQ_BUTTONS.length; bi++) {{
      const diff = Math.abs(FREQ_BUTTONS[bi] - c.default_freq);
      if (diff < bestDiff) {{ bestDiff = diff; bestBtn = bi; }}
    }}
    freqBtnIdx = bestBtn;
    selectedFreq = FREQ_BUTTONS[bestBtn];
    markers = [...hppLookup(c.hpp_results, selectedFreq)];
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

function acceptFreq() {{
  const c = CASES[idx];
  const ipiF = computeIPIFreq();
  const freq = ipiF || (selectedFreq || c.default_freq);
  allDecisions[c.segment_id] = {{
    action: 'accept',
    freq: freq,
    model_freq: c.model_freq,
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
    _markers: [...markers],
    _selectedFreq: selectedFreq,
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
    subtype: c.subtype,
    mat_file: c.mat_file,
    patient_id: c.patient_id,
  }};
  saveAll();
  const el = document.getElementById('save-status');
  el.textContent = 'REJECTED (not PD)';
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
      subtype: d.subtype,
      mat_file: d.mat_file,
      patient_id: d.patient_id,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'expert_freq_pd_results.json';
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
    acceptFreq();
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
#  RDA HTML BUILDER
# ========================================================================

def build_rda_html(cases_data):
    """Build RDA frequency labeling viewer with client-side FFT narrowband overlay."""
    left_indices_json = json.dumps(LEFT_INDICES)
    right_indices_json = json.dumps(RIGHT_INDICES)
    freq_buttons_json = json.dumps(FREQ_BUTTONS)

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Expert RDA Freq Labeler ({n_cases} cases)</title>
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
    <span style="font-size:16px; font-weight:bold; color:#226622;">Expert RDA Freq Labeler (MW)</span>
    <span id="counter" style="font-size:13px; color:#888;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="action-btn btn-accept" onclick="acceptFreq()">Accept Freq <span class="key">Enter</span></button>
    <button class="action-btn btn-reject" onclick="rejectCase()">Not RDA <span class="key">X</span></button>
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
  <span class="info-item">W05: <strong id="info-model-freq" style="color:#0066cc;">--</strong></span>
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
const STORAGE_KEY = 'expert_freq_rda_v1';
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
    '  |  W05=' + c.model_freq.toFixed(2) + ' Hz';
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-sid').textContent = c.segment_id;
  document.getElementById('info-patient').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
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
      model_freq: d.model_freq,
      subtype: d.subtype,
      mat_file: d.mat_file,
      patient_id: d.patient_id,
    }};
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'expert_freq_rda_results.json';
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
    print("  PD (LPD+GPD): evidence trace + HPP discharge markers")
    print("  RDA (LRDA+GRDA): client-side FFT narrowband overlay")
    print("  LB/PH/SZ have freq labels; MW needs to add freq labels")
    print("=" * 70)

    print(f"\nFinding cases needing MW frequency labels...")
    candidates = find_cases()

    if args.max_cases > 0 and len(candidates) > args.max_cases:
        candidates = candidates[:args.max_cases]
        print(f"  (limited to {args.max_cases} cases)")

    if len(candidates) == 0:
        print("  No cases to label!")
        return

    # Prepare PD cases
    pd_cases = prepare_pd_cases(candidates)
    # Prepare RDA cases
    rda_cases = prepare_rda_cases(candidates)

    import subprocess

    # Build and write PD viewer
    if len(pd_cases) > 0:
        print(f"\nBuilding PD HTML viewer ({len(pd_cases)} cases)...")
        pd_html = build_pd_html(pd_cases)
        pd_path = OUT_BASE / 'expert_freq_labeler_pd.html'
        with open(str(pd_path), 'w') as f:
            f.write(pd_html)
        size_mb = pd_path.stat().st_size / (1024 * 1024)
        print(f"  Written to {pd_path}")
        print(f"  {len(pd_cases)} PD cases, {size_mb:.1f} MB")
        subprocess.run(['open', str(pd_path)])
    else:
        print("\n  No PD cases to review.")

    # Build and write RDA viewer
    if len(rda_cases) > 0:
        print(f"\nBuilding RDA HTML viewer ({len(rda_cases)} cases)...")
        rda_html = build_rda_html(rda_cases)
        rda_path = OUT_BASE / 'expert_freq_labeler_rda.html'
        with open(str(rda_path), 'w') as f:
            f.write(rda_html)
        size_mb = rda_path.stat().st_size / (1024 * 1024)
        print(f"  Written to {rda_path}")
        print(f"  {len(rda_cases)} RDA cases, {size_mb:.1f} MB")
        subprocess.run(['open', str(rda_path)])
    else:
        print("\n  No RDA cases to review.")

    print("=" * 70)


if __name__ == '__main__':
    main()
