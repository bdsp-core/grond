"""
Generate publication-quality EEG characterization figures for the paper.

Creates 4 HTML files in paper_materials/:
  - figure_lpd_examples.html
  - figure_gpd_examples.html
  - figure_lrda_examples.html
  - figure_grda_examples.html

Each shows 3 examples (easy, medium, hard) with:
  - 18-channel bipolar EEG with discharge timing markers (PD only)
  - Topoplot with inverse-distance weighted heatmap
  - Verbal description and annotation details

Usage:
    conda run -n morgoth python code/generators/figures/generate_paper_figures.py
"""

import sys
import json
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt, hilbert

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent.parent  # code/
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'paper_materials'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
DURATION = 10.0
LOWPASS_HZ = 20.0

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

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

REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO', 'MID']
REGION_TO_CHANNELS = {
    'LF': [0, 8],     'RF': [4, 12],
    'LT': [1, 2],     'RT': [5, 6],
    'LCP': [9, 10],   'RCP': [13, 14],
    'LO': [3, 11],    'RO': [7, 15],
    'MID': [16, 17],
}

# Left hemisphere channel indices (left temporal + left parasagittal + left occipital + left frontal)
LEFT_CHANNELS = set([0, 1, 2, 3, 8, 9, 10, 11])  # Fp1-F7, F7-T3, T3-T5, T5-O1, Fp1-F3, F3-C3, C3-P3, P3-O1
RIGHT_CHANNELS = set([4, 5, 6, 7, 12, 13, 14, 15])  # Fp2-F8, F8-T4, T4-T6, T6-O2, Fp2-F4, F4-C4, C4-P4, P4-O2
MID_CHANNELS = set([16, 17])  # Fz-Cz, Cz-Pz

# ── PDCharacterizer (lazy loaded) ──
_pd_characterizer = None

def _get_pd_characterizer():
    global _pd_characterizer
    if _pd_characterizer is None:
        from pd_characterizer import PDCharacterizer
        _pd_characterizer = PDCharacterizer()
    return _pd_characterizer


def parse_regions(s):
    """Parse spatial_channels string, handling various formats.

    Handles: 'LF RF LT', 'LF, RF, LT', 'LF,RF,LT',
             'LB:LF RF; PH:LF, RF', 'SZ:LF,LCP; SZ:LF,LCP'
    """
    if pd.isna(s):
        return set()
    all_regions = set()
    valid = set(REGIONS)
    for part in s.split(';'):
        part = part.strip()
        if ':' in part:
            part = part.split(':', 1)[1]
        for r in part.replace(',', ' ').split():
            r = r.strip()
            if r in valid:
                all_regions.add(r)
    return all_regions


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


def predict_spatial_pd(seg, subtype):
    """Use PDCharacterizer for LPD/GPD spatial prediction."""
    try:
        pc = _get_pd_characterizer()
        result = pc.characterize(seg, subtype=subtype)
        channel_probs = result.get('channel_probs', [0.5] * 18)
        region_scores = result.get('region_scores', {r: 0.5 for r in REGIONS})
        predicted_regions = result.get('regions', [])
        discharge_times = result.get('discharge_times', [])
        frequency = result.get('frequency', 0)
        laterality = result.get('laterality', '?')
        confidence = float(np.mean([region_scores[r] for r in predicted_regions])) if predicted_regions else 0.0
        return channel_probs, region_scores, predicted_regions, confidence, discharge_times, frequency, laterality
    except Exception as e:
        print(f"  PDCharacterizer failed: {e}")
        return [0.5] * 18, {r: 0.5 for r in REGIONS}, [], 0.0, [], 0, '?'


def predict_spatial_rda(seg):
    """Use PLV-based scoring for LRDA/GRDA spatial prediction."""
    try:
        n_ch = min(18, seg.shape[0])
        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
        seg_f = sosfiltfilt(sos, seg[:n_ch], axis=1)
        variances = np.var(seg_f, axis=1)
        ref_ch = int(np.argmax(variances))
        phases = np.zeros_like(seg_f)
        for ch in range(n_ch):
            phases[ch] = np.angle(hilbert(seg_f[ch]))
        ref_phase = phases[ref_ch]
        channel_scores = np.zeros(n_ch)
        for ch in range(n_ch):
            phase_diff = phases[ch] - ref_phase
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            channel_scores[ch] = plv

        region_scores = {}
        for region, chs in REGION_TO_CHANNELS.items():
            valid_chs = [c for c in chs if c < n_ch]
            if valid_chs:
                region_scores[region] = float(max(channel_scores[c] for c in valid_chs))
            else:
                region_scores[region] = 0.0

        predicted_regions = [r for r, s in region_scores.items() if s > 0.38]
        confidence = float(np.mean([region_scores[r] for r in predicted_regions])) if predicted_regions else 0.0
        return channel_scores.tolist(), region_scores, predicted_regions, confidence
    except Exception as e:
        print(f"  PLV spatial failed: {e}")
        return [0.5] * 18, {r: 0.5 for r in REGIONS}, [], 0.0


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


def load_discharge_times():
    """Load discharge_times.json and build mat_file lookup."""
    import os
    dt_path = LABELS_DIR / 'discharge_times.json'
    if not dt_path.exists():
        return {}
    with open(str(dt_path)) as f:
        dt_json = json.load(f)

    eeg_files = set(os.listdir(str(EEG_DIR)))
    dt_by_matfile = {}
    for key, val in dt_json.items():
        if isinstance(val, dict):
            times = val.get('discharge_times', val.get('times', val.get('global_times', [])))
        elif isinstance(val, list):
            times = val
        else:
            continue
        if not times:
            continue
        if key + '.mat' in eeg_files:
            dt_by_matfile[key + '.mat'] = times
        elif not key.startswith('sub-'):
            for f in eeg_files:
                if key in f and f.endswith('.mat'):
                    dt_by_matfile[f] = times
                    break
    return dt_by_matfile


def select_pd_cases(subtype, ann, sl, dt_by_matfile):
    """Select easy/medium/hard cases for LPD/GPD based on inter-rater Jaccard."""
    sub_sl = sl[(sl.subtype == subtype) & (sl.excluded != True)]
    spatial = ann[ann.spatial_channels.notna()].copy()
    spatial['regions_set'] = spatial.spatial_channels.apply(parse_regions)
    spatial = spatial[spatial.regions_set.apply(len) > 0]

    sub_ann = spatial[spatial.mat_file.isin(sub_sl.mat_file)]
    multi = sub_ann.groupby('segment_id').filter(lambda g: len(g) >= 2)
    seg_ids = multi.segment_id.unique()

    # Compute Jaccard for each segment
    jaccards = {}
    rater_details = {}
    for seg_id in seg_ids:
        raters = multi[multi.segment_id == seg_id]
        rsets = {r.rater: r.regions_set for _, r in raters.iterrows()}
        all_sets = list(rsets.values())
        union = set.union(*all_sets)
        inter = set.intersection(*all_sets)
        jaccards[seg_id] = len(inter) / len(union) if union else 0
        rater_details[seg_id] = {k: sorted(list(v)) for k, v in rsets.items()}

    # Filter to those with discharge times + freq + laterality
    candidates = {}
    for seg_id, j in jaccards.items():
        mat = seg_id + '.mat'
        row = sub_sl[sub_sl.mat_file == mat]
        if row.empty:
            continue
        r = row.iloc[0]
        has_dt = mat in dt_by_matfile
        has_freq = pd.notna(r.get('expert_freq_hz'))
        has_lat = pd.notna(r.get('laterality'))
        if has_dt and has_freq and has_lat:
            candidates[seg_id] = {
                'jaccard': j,
                'raters': rater_details[seg_id],
                'row': r,
            }

    # Also include candidates with freq+lat but no discharge times
    for seg_id, j in jaccards.items():
        if seg_id in candidates:
            continue
        mat = seg_id + '.mat'
        row = sub_sl[sub_sl.mat_file == mat]
        if row.empty:
            continue
        r = row.iloc[0]
        has_freq = pd.notna(r.get('expert_freq_hz'))
        has_lat = pd.notna(r.get('laterality'))
        if has_freq and has_lat:
            candidates[seg_id] = {
                'jaccard': j,
                'raters': rater_details[seg_id],
                'row': r,
            }

    # Sort all candidates by Jaccard
    sorted_cands = sorted(candidates.items(), key=lambda x: x[1]['jaccard'])

    # Split into 3 difficulty tiers using terciles of the actual distribution
    n = len(sorted_cands)
    if n < 3:
        print(f"  WARNING: only {n} candidates for {subtype}")
        return [(sid, c, 'Easy') for sid, c in sorted_cands]

    # Tercile boundaries
    t1 = sorted_cands[n // 3][1]['jaccard']
    t2 = sorted_cands[2 * n // 3][1]['jaccard']

    # Check if all Jaccards are the same (common for GPD)
    all_same = t1 == t2 == sorted_cands[0][1]['jaccard'] == sorted_cands[-1][1]['jaccard']

    if all_same:
        # All agreement is the same — pick 3 from different patients with varied freq
        # Label as "Example 1/2/3" instead of difficulty
        hard = []
        medium = []
        easy = list(sorted_cands)
        # Will pick 3 diverse from 'Easy' pool below
    else:
        hard = [(sid, c) for sid, c in sorted_cands if c['jaccard'] <= t1]
        easy = [(sid, c) for sid, c in sorted_cands if c['jaccard'] >= t2]
        medium = [(sid, c) for sid, c in sorted_cands if t1 < c['jaccard'] < t2]

    # If medium is empty (e.g., many ties), split differently
    if not medium and not all_same:
        medium = [(sid, c) for sid, c in sorted_cands
                  if t1 < c['jaccard'] <= t2 and (sid, c) not in easy]
    if not medium and len(easy) > 1 and not all_same:
        medium = [easy.pop(0)]

    selected = []
    used_patients = set()

    if all_same:
        # All agreement is the same — pick 3 from different patients, label by frequency
        pool = list(easy)
        pool.sort(key=lambda x: float(x[1]['row'].get('expert_freq_hz', 1.0) or 1.0))
        labels = ['Low frequency', 'Medium frequency', 'High frequency']
        indices = [0, len(pool) // 2, len(pool) - 1]
        for idx_i, label in zip(indices, labels):
            if idx_i < len(pool):
                sid, c = pool[idx_i]
                pid = c['row']['patient_id']
                if pid in used_patients:
                    for sid2, c2 in pool:
                        if c2['row']['patient_id'] not in used_patients:
                            sid, c = sid2, c2
                            break
                used_patients.add(c['row']['patient_id'])
                selected.append((sid, c, label))
    else:
        for difficulty, pool in [('Easy', easy), ('Medium', medium), ('Hard', hard)]:
            pool.sort(key=lambda x: x[1]['jaccard'], reverse=(difficulty == 'Easy'))
            chosen = None
            # First try different patient
            for sid, c in pool:
                pid = c['row']['patient_id']
                if pid not in used_patients:
                    chosen = (sid, c, difficulty)
                    used_patients.add(pid)
                    break
            # Fall back to same patient if needed
            if chosen is None and pool:
                chosen = (pool[0][0], pool[0][1], difficulty)
                used_patients.add(pool[0][1]['row']['patient_id'])
            if chosen:
                selected.append(chosen)

    return selected


def select_rda_cases(subtype, sl):
    """Select easy/medium/hard cases for LRDA/GRDA based on spatial complexity."""
    sub = sl[(sl.subtype == subtype) & (sl.excluded != True)].copy()
    has_all = sub[sub.expert_freq_hz.notna() & sub.laterality.notna() & sub.spatial_channels.notna()]

    if len(has_all) == 0:
        # Fallback: just need freq + laterality
        has_all = sub[sub.expert_freq_hz.notna() & sub.laterality.notna()]

    # Parse spatial channels from segment_labels (may have complex format)
    def parse_sl_spatial(s):
        if pd.isna(s):
            return set()
        # Handle "SZ:LF,LCP; SZ:LF,LCP" format — extract unique regions
        all_regions = set()
        for part in s.split(';'):
            part = part.strip()
            if ':' in part:
                part = part.split(':', 1)[1]
            for r in part.replace(',', ' ').split():
                if r in set(REGIONS):
                    all_regions.add(r)
        return all_regions

    cases = []
    for _, r in has_all.iterrows():
        regs = parse_sl_spatial(r.get('spatial_channels'))
        lat = r.get('laterality')
        n_regs = len(regs)

        # Complexity score: more regions + bilateral = harder
        complexity = n_regs / 9.0  # normalize by max regions
        if lat == 'bilateral':
            complexity += 0.3
        elif lat not in ('left', 'right'):
            complexity += 0.15

        cases.append({
            'mat_file': r['mat_file'],
            'patient_id': r['patient_id'],
            'row': r,
            'regions': regs,
            'complexity': complexity,
        })

    if len(cases) < 3:
        print(f"  WARNING: only {len(cases)} candidates for {subtype}")
        for c in cases:
            c['difficulty'] = 'Easy'
        return cases

    # Sort by complexity, split into terciles
    cases.sort(key=lambda c: c['complexity'])
    n = len(cases)
    for i, c in enumerate(cases):
        if i < n // 3:
            c['difficulty'] = 'Easy'
        elif i < 2 * n // 3:
            c['difficulty'] = 'Medium'
        else:
            c['difficulty'] = 'Hard'

    selected = []
    used_patients = set()
    np.random.seed(42)

    for difficulty in ['Easy', 'Medium', 'Hard']:
        pool = [c for c in cases if c['difficulty'] == difficulty]
        np.random.shuffle(pool)
        chosen = None
        for c in pool:
            pid = c['patient_id']
            if pid not in used_patients:
                chosen = c
                used_patients.add(pid)
                break
        if chosen is None and pool:
            chosen = pool[0]
        if chosen:
            selected.append(chosen)

    return selected


def process_case(case_info, subtype, dt_by_matfile):
    """Load EEG, compute predictions, prepare case data dict."""
    is_pd = subtype in ('lpd', 'gpd')

    if is_pd:
        seg_id, cand, difficulty = case_info
        mat_file = seg_id + '.mat'
        row = cand['row']
        jaccard = cand['jaccard']
        rater_details = cand['raters']
    else:
        mat_file = case_info['mat_file']
        row = case_info['row']
        difficulty = case_info['difficulty']
        jaccard = None
        rater_details = None

    seg = load_segment(mat_file)
    if seg is None:
        print(f"  SKIP: {mat_file} — EEG not found")
        return None

    # Detrend + lowpass filter
    b_lp, a_lp = butter(4, LOWPASS_HZ / (FS / 2), btype='low')
    seg_display = np.zeros_like(seg)
    for ch in range(seg.shape[0]):
        try:
            seg_display[ch] = filtfilt(b_lp, a_lp, detrend(seg[ch], type='linear'))
        except Exception:
            seg_display[ch] = seg[ch]

    # Compute spatial predictions
    if is_pd:
        channel_scores, region_scores, predicted_regions, confidence, \
            pred_discharge_times, pred_freq, pred_lat = predict_spatial_pd(seg, subtype)
    else:
        channel_scores, region_scores, predicted_regions, confidence = predict_spatial_rda(seg)
        pred_discharge_times = []
        pred_freq = 0
        pred_lat = '?'

    # Ground truth discharge times
    gt_discharge_times = dt_by_matfile.get(mat_file, [])

    # Ground truth labels
    gt_freq = float(row['expert_freq_hz']) if pd.notna(row.get('expert_freq_hz')) else None
    gt_lat = str(row['laterality']) if pd.notna(row.get('laterality')) else None
    gt_spatial = str(row.get('spatial_channels', '')) if pd.notna(row.get('spatial_channels')) else ''

    # Involved channels from ground truth spatial
    gt_regions = parse_regions(gt_spatial) if is_pd else set()
    # For RDA, parse from segment_labels spatial_channels
    if not is_pd and gt_spatial:
        gt_regions_rda = set()
        for part in gt_spatial.split(';'):
            part = part.strip()
            if ':' in part:
                part = part.split(':', 1)[1]
            for r in part.replace(',', ' ').split():
                if r in set(REGIONS):
                    gt_regions_rda.add(r)
        gt_regions = gt_regions_rda

    # Build involved channel set from regions
    involved_channels = set()
    for reg in gt_regions:
        if reg in REGION_TO_CHANNELS:
            for ch in REGION_TO_CHANNELS[reg]:
                involved_channels.add(ch)

    case = {
        'mat_file': mat_file,
        'patient_id': str(row['patient_id']),
        'segment_id': mat_file.replace('.mat', ''),
        'subtype': subtype,
        'difficulty': difficulty,
        'jaccard': round(jaccard, 3) if jaccard is not None else None,
        'rater_details': rater_details,
        'gt_freq': gt_freq,
        'gt_lat': gt_lat,
        'gt_regions': sorted(list(gt_regions)),
        'gt_discharge_times': sorted(gt_discharge_times) if gt_discharge_times else [],
        'pred_discharge_times': sorted(pred_discharge_times) if pred_discharge_times else [],
        'pred_freq': round(float(pred_freq), 2) if pred_freq else 0,
        'pred_lat': str(pred_lat),
        'pred_regions': predicted_regions,
        'pred_confidence': round(confidence, 3),
        'channel_scores': [round(float(s), 4) for s in channel_scores],
        'region_scores': {r: round(float(s), 4) for r, s in region_scores.items()},
        'involved_channels': sorted(list(involved_channels)),
        'eeg_data': downsample(seg_display, 1000),
    }
    return case


def build_html(cases, subtype):
    """Build the HTML figure file for a given subtype."""
    is_pd = subtype in ('lpd', 'gpd')
    cases_json = json.dumps(cases, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{subtype.upper()} EEG Characterization Examples</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #ffffff; color: #222; font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 20px; max-width: 1250px; margin: 0 auto;
  }}
  h1 {{
    font-size: 20px; font-weight: 700; text-align: center; margin-bottom: 6px;
    color: #111;
  }}
  .subtitle {{
    font-size: 13px; color: #666; text-align: center; margin-bottom: 20px;
  }}
  .example-panel {{
    border: 1px solid #ccc; border-radius: 6px; margin-bottom: 24px;
    page-break-inside: avoid;
  }}
  .title-bar {{
    background: #f0f0f0; padding: 8px 14px; border-bottom: 1px solid #ccc;
    font-size: 14px; font-weight: 600; color: #333; border-radius: 6px 6px 0 0;
  }}
  .title-bar .difficulty-easy {{ color: #2a7d2a; }}
  .title-bar .difficulty-medium {{ color: #b87700; }}
  .title-bar .difficulty-hard {{ color: #c03030; }}
  .main-row {{
    display: flex; padding: 10px; gap: 12px;
  }}
  .eeg-container {{
    flex: 0 0 78%; position: relative;
  }}
  .topo-container {{
    flex: 0 0 20%; display: flex; flex-direction: column; align-items: center; gap: 6px;
  }}
  .verbal-desc {{
    font-size: 12px; color: #444; text-align: center; font-style: italic;
    line-height: 1.4; max-width: 220px; margin-top: 4px;
  }}
  .annotation-row {{
    padding: 8px 14px; border-top: 1px solid #ddd; font-size: 12px; color: #555;
    line-height: 1.6; background: #fafafa; border-radius: 0 0 6px 6px;
  }}
  .annotation-row strong {{ color: #333; }}
  .rater-label {{ font-weight: 600; color: #2255aa; }}
  .pred-label {{ font-weight: 600; color: #aa5522; }}
  hr.separator {{
    border: none; border-top: 2px solid #ddd; margin: 16px 0;
  }}
  @media print {{
    .example-panel {{ page-break-inside: avoid; }}
    body {{ padding: 10px; }}
  }}
</style>
</head>
<body>

<h1>{subtype.upper()} EEG Characterization Examples</h1>
<div class="subtitle">
  {"Inter-rater agreement defines difficulty: Easy (Jaccard >= 0.9), Medium (0.6-0.9), Hard (< 0.6)" if is_pd else "Difficulty defined by spatial complexity and laterality clarity"}
</div>

<div id="panels"></div>

<script>
const CASES = {cases_json};
const IS_PD = {'true' if is_pd else 'false'};
const SUBTYPE = '{subtype}';

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];

// Standard display order: L temporal, L parasagittal, midline, R parasagittal, R temporal
const DISPLAY_ORDER = [
  {{idx: 0, name: 'Fp1-F7', hemi: 'L'}},
  {{idx: 1, name: 'F7-T3', hemi: 'L'}},
  {{idx: 2, name: 'T3-T5', hemi: 'L'}},
  {{idx: 3, name: 'T5-O1', hemi: 'L'}},
  {{idx: -1, name: '', hemi: ''}},
  {{idx: 8, name: 'Fp1-F3', hemi: 'L'}},
  {{idx: 9, name: 'F3-C3', hemi: 'L'}},
  {{idx: 10, name: 'C3-P3', hemi: 'L'}},
  {{idx: 11, name: 'P3-O1', hemi: 'L'}},
  {{idx: -1, name: '', hemi: ''}},
  {{idx: 16, name: 'Fz-Cz', hemi: 'M'}},
  {{idx: 17, name: 'Cz-Pz', hemi: 'M'}},
  {{idx: -1, name: '', hemi: ''}},
  {{idx: 12, name: 'Fp2-F4', hemi: 'R'}},
  {{idx: 13, name: 'F4-C4', hemi: 'R'}},
  {{idx: 14, name: 'C4-P4', hemi: 'R'}},
  {{idx: 15, name: 'P4-O2', hemi: 'R'}},
  {{idx: -1, name: '', hemi: ''}},
  {{idx: 4, name: 'Fp2-F8', hemi: 'R'}},
  {{idx: 5, name: 'F8-T4', hemi: 'R'}},
  {{idx: 6, name: 'T4-T6', hemi: 'R'}},
  {{idx: 7, name: 'T6-O2', hemi: 'R'}},
];

const N_DISPLAY = DISPLAY_ORDER.length;
const EEG_WIDTH = 880;
const EEG_HEIGHT = 520;
const MARGIN_LEFT = 62;
const MARGIN_RIGHT = 14;
const MARGIN_TOP = 28;
const MARGIN_BOTTOM = 22;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

const TOPO_SIZE = 220;
const CLIP_UV = 250.0;
const Z_SCALE = 0.012;

// Electrode positions (normalized to unit circle, nose at top)
const ELECTRODE_POS = {{
  'Fp1': [-0.31, 0.95], 'Fp2': [0.31, 0.95],
  'F7': [-0.81, 0.59], 'F3': [-0.39, 0.59], 'Fz': [0.0, 0.59], 'F4': [0.39, 0.59], 'F8': [0.81, 0.59],
  'T3': [-1.0, 0.0], 'C3': [-0.5, 0.0], 'Cz': [0.0, 0.0], 'C4': [0.5, 0.0], 'T4': [1.0, 0.0],
  'T5': [-0.81, -0.59], 'P3': [-0.39, -0.59], 'Pz': [0.0, -0.59], 'P4': [0.39, -0.59], 'T6': [0.81, -0.59],
  'O1': [-0.31, -0.95], 'O2': [0.31, -0.95]
}};

const BIPOLAR_ELECTRODES = [
  ['Fp1','F7'], ['F7','T3'], ['T3','T5'], ['T5','O1'],
  ['Fp2','F8'], ['F8','T4'], ['T4','T6'], ['T6','O2'],
  ['Fp1','F3'], ['F3','C3'], ['C3','P3'], ['P3','O1'],
  ['Fp2','F4'], ['F4','C4'], ['C4','P4'], ['P4','O2'],
  ['Fz','Cz'], ['Cz','Pz']
];

const REGION_BARE = {{
  'LF': 'frontal', 'RF': 'frontal',
  'LT': 'temporal', 'RT': 'temporal',
  'LCP': 'centro-parietal', 'RCP': 'centro-parietal',
  'LO': 'occipital', 'RO': 'occipital',
  'MID': 'midline'
}};
const LEFT_REGS = ['LF', 'LT', 'LCP', 'LO'];
const RIGHT_REGS = ['RF', 'RT', 'RCP', 'RO'];

// Left hemisphere channel indices
const LEFT_CH = new Set([0, 1, 2, 3, 8, 9, 10, 11]);
const RIGHT_CH = new Set([4, 5, 6, 7, 12, 13, 14, 15]);

function scoreToColor(t) {{
  t = Math.max(0, Math.min(1, t));
  if (t <= 0.33) {{
    const s = t / 0.33;
    return [Math.round(20+(0-20)*s), Math.round(20+(180-20)*s), Math.round(80+(220-80)*s)];
  }} else if (t <= 0.66) {{
    const s = (t - 0.33) / 0.33;
    return [Math.round(0+(255-0)*s), Math.round(180+(220-180)*s), Math.round(220+(0-220)*s)];
  }} else {{
    const s = (t - 0.66) / 0.34;
    return [255, Math.round(220+(30-220)*s), 0];
  }}
}}

function generateVerbalDescription(c) {{
  const st = c.subtype.toUpperCase();
  const regs = c.gt_regions || [];
  if (regs.length === 0) return st + ' -- no spatial regions labeled.';

  const leftSel = regs.filter(r => LEFT_REGS.includes(r));
  const rightSel = regs.filter(r => RIGHT_REGS.includes(r));
  const midSel = regs.includes('MID');
  const isGen = st === 'GPD' || st === 'GRDA';

  let latStr = '';
  if (leftSel.length > 0 && rightSel.length === 0) latStr = 'unilateral left';
  else if (rightSel.length > 0 && leftSel.length === 0) latStr = 'unilateral right';
  else if (leftSel.length > rightSel.length) latStr = 'bilateral, left-predominant';
  else if (rightSel.length > leftSel.length) latStr = 'bilateral, right-predominant';
  else if (leftSel.length > 0 && rightSel.length > 0) latStr = 'bilateral/symmetric';
  else if (midSel) latStr = 'midline';

  if (isGen) {{
    const rs = c.region_scores || {{}};
    const frontal = ((rs['LF']||0) + (rs['RF']||0)) / 2;
    const occipital = ((rs['LO']||0) + (rs['RO']||0)) / 2;
    const temporal = ((rs['LT']||0) + (rs['RT']||0)) / 2;
    const scores = {{'frontally': frontal, 'occipitally': occipital, 'temporally': temporal}};
    const best = Object.entries(scores).sort((a,b) => b[1] - a[1])[0][0];
    const range = Math.max(frontal, occipital, temporal) - Math.min(frontal, occipital, temporal);
    const predom = range > 0.1 ? best + ' predominant' : 'no regional predominance';
    return st + ', ' + predom + '.';
  }}

  const domRegs = (leftSel.length >= rightSel.length) ? leftSel : rightSel;
  if (domRegs.length === 0 && midSel) domRegs.push('MID');
  const scored = domRegs.map(r => [r, (c.region_scores||{{}})[r] || 0]).sort((a,b) => b[1] - a[1]);
  const topNames = [];
  for (const [r, s] of scored.slice(0, 2)) {{
    const bare = REGION_BARE[r];
    if (bare && !topNames.includes(bare)) topNames.push(bare);
  }}
  const regionStr = topNames.length > 0
    ? 'maximal in the ' + topNames.join(' and ') + ' region' + (topNames.length > 1 ? 's' : '')
    : 'no region clearly dominant';

  return st + ', ' + latStr + '; ' + regionStr + '.';
}}

function drawEEG(canvas, c) {{
  canvas.width = EEG_WIDTH;
  canvas.height = EEG_HEIGHT;
  const ctx = canvas.getContext('2d');
  const eeg = c.eeg_data;
  const nSamples = eeg[0].length;
  const involvedSet = new Set(c.involved_channels || []);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, EEG_WIDTH, EEG_HEIGHT);

  const chSpacing = PLOT_H / (N_DISPLAY + 1);

  // Hemisphere shading
  const lat = c.gt_lat || '';
  if (lat === 'left' || lat === 'bilateral') {{
    // Find Y range of left channels
    let yMin = EEG_HEIGHT, yMax = 0;
    for (let di = 0; di < N_DISPLAY; di++) {{
      const ch = DISPLAY_ORDER[di];
      if (ch.idx >= 0 && LEFT_CH.has(ch.idx)) {{
        const y = MARGIN_TOP + chSpacing * (di + 1);
        yMin = Math.min(yMin, y - chSpacing * 0.5);
        yMax = Math.max(yMax, y + chSpacing * 0.5);
      }}
    }}
    ctx.fillStyle = 'rgba(100,150,255,0.06)';
    ctx.fillRect(PLOT_LEFT, yMin, PLOT_W, yMax - yMin);
  }}
  if (lat === 'right' || lat === 'bilateral') {{
    let yMin = EEG_HEIGHT, yMax = 0;
    for (let di = 0; di < N_DISPLAY; di++) {{
      const ch = DISPLAY_ORDER[di];
      if (ch.idx >= 0 && RIGHT_CH.has(ch.idx)) {{
        const y = MARGIN_TOP + chSpacing * (di + 1);
        yMin = Math.min(yMin, y - chSpacing * 0.5);
        yMax = Math.max(yMax, y + chSpacing * 0.5);
      }}
    }}
    ctx.fillStyle = 'rgba(255,150,100,0.06)';
    ctx.fillRect(PLOT_LEFT, yMin, PLOT_W, yMax - yMin);
  }}

  // Gridlines
  ctx.strokeStyle = '#e0e0e0';
  ctx.lineWidth = 0.5;
  ctx.setLineDash([3, 3]);
  for (let s = 0; s <= 10; s++) {{
    const x = PLOT_LEFT + (s / 10.0) * PLOT_W;
    ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
  }}
  ctx.setLineDash([]);

  // Discharge timing markers (PD only)
  if (IS_PD && c.gt_discharge_times && c.gt_discharge_times.length > 0) {{
    ctx.strokeStyle = '#cc3333';
    ctx.lineWidth = 1.0;
    ctx.setLineDash([5, 3]);

    const isLPD = SUBTYPE === 'lpd';
    // For LPD: markers on involved hemisphere only (half screen)
    // For GPD: full height
    for (const t of c.gt_discharge_times) {{
      const x = PLOT_LEFT + (t / 10.0) * PLOT_W;
      if (x < PLOT_LEFT || x > PLOT_RIGHT) continue;

      if (isLPD) {{
        // Find Y range of involved channels
        let yStart = MARGIN_TOP;
        let yEnd = MARGIN_TOP + PLOT_H * 0.5;
        // Use laterality to determine which half
        if (lat === 'left') {{
          // Left channels are at top in our display order
          yEnd = MARGIN_TOP + PLOT_H * 0.45;
        }} else if (lat === 'right') {{
          yStart = MARGIN_TOP + PLOT_H * 0.55;
          yEnd = EEG_HEIGHT - MARGIN_BOTTOM;
        }} else {{
          yEnd = EEG_HEIGHT - MARGIN_BOTTOM;
        }}
        ctx.beginPath(); ctx.moveTo(x, yStart); ctx.lineTo(x, yEnd); ctx.stroke();
      }} else {{
        ctx.beginPath(); ctx.moveTo(x, MARGIN_TOP); ctx.lineTo(x, EEG_HEIGHT - MARGIN_BOTTOM); ctx.stroke();
      }}
    }}
    ctx.setLineDash([]);
  }}

  // EEG traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_ORDER[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eeg[ch.idx];
    const isInvolved = involvedSet.has(ch.idx);
    ctx.strokeStyle = isInvolved ? '#228833' : '#333333';
    ctx.lineWidth = isInvolved ? 1.2 : 0.7;
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
  ctx.font = '10px Helvetica, Arial, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_ORDER[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const isInvolved = involvedSet.has(ch.idx);
    ctx.fillStyle = isInvolved ? '#228833' : '#666666';
    ctx.font = isInvolved ? 'bold 10px Helvetica, Arial, sans-serif' : '10px Helvetica, Arial, sans-serif';
    ctx.fillText(ch.name, PLOT_LEFT - 4, yCenter);
  }}

  // Time axis
  ctx.fillStyle = '#333';
  ctx.font = '10px Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let s = 0; s <= 10; s++) {{
    ctx.fillText(s + 's', PLOT_LEFT + (s / 10.0) * PLOT_W, EEG_HEIGHT - MARGIN_BOTTOM + 3);
  }}

  // Title on EEG
  ctx.fillStyle = '#333';
  ctx.font = 'bold 12px Helvetica, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  let titleParts = [];
  if (c.gt_freq) titleParts.push('freq=' + c.gt_freq.toFixed(2) + ' Hz');
  if (c.gt_lat) titleParts.push('lat=' + c.gt_lat);
  if (IS_PD && c.gt_discharge_times) titleParts.push(c.gt_discharge_times.length + ' discharges');
  ctx.fillText(titleParts.join(' | '), EEG_WIDTH / 2, 6);
}}

function drawTopoplot(canvas, c) {{
  canvas.width = TOPO_SIZE;
  canvas.height = TOPO_SIZE;
  const ctx = canvas.getContext('2d');
  const cx = TOPO_SIZE / 2;
  const cy = TOPO_SIZE / 2;
  const headR = TOPO_SIZE * 0.40;

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, TOPO_SIZE, TOPO_SIZE);

  // Per-electrode scores from bipolar channel scores
  const electrodeNames = Object.keys(ELECTRODE_POS);
  const electrodeScores = {{}};
  for (const ename of electrodeNames) {{
    let maxScore = 0;
    for (let bi = 0; bi < BIPOLAR_ELECTRODES.length; bi++) {{
      if (BIPOLAR_ELECTRODES[bi].includes(ename)) {{
        maxScore = Math.max(maxScore, c.channel_scores[bi] || 0);
      }}
    }}
    electrodeScores[ename] = maxScore;
  }}

  const allScores = Object.values(electrodeScores);
  let topoMin = Math.min(...allScores);
  let topoMax = Math.max(...allScores);
  if (topoMax - topoMin < 0.05) {{
    topoMin = Math.max(0, topoMin - 0.1);
    topoMax = Math.min(1, topoMax + 0.1);
  }}
  function norm(v) {{ return topoMax <= topoMin ? 0.5 : (v - topoMin) / (topoMax - topoMin); }}

  // Heatmap via inverse-distance weighting
  const imgData = ctx.createImageData(TOPO_SIZE, TOPO_SIZE);
  const elecList = electrodeNames.map(name => ({{
    x: cx + ELECTRODE_POS[name][0] * headR,
    y: cy - ELECTRODE_POS[name][1] * headR,
    score: electrodeScores[name]
  }}));

  for (let py = 0; py < TOPO_SIZE; py++) {{
    for (let px = 0; px < TOPO_SIZE; px++) {{
      const dx = px - cx, dy = py - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > headR + 2) continue;
      let wSum = 0, vSum = 0;
      for (const e of elecList) {{
        const d = Math.sqrt((px - e.x) ** 2 + (py - e.y) ** 2);
        const w = 1.0 / (d * d * d + 1);
        wSum += w;
        vSum += w * e.score;
      }}
      const val = vSum / wSum;
      const rgb = scoreToColor(norm(val));
      let alpha = 200;
      if (dist > headR - 5) alpha = Math.max(0, Math.round(200 * (1 - (dist - headR + 5) / 7)));
      const pidx = (py * TOPO_SIZE + px) * 4;
      imgData.data[pidx] = rgb[0];
      imgData.data[pidx + 1] = rgb[1];
      imgData.data[pidx + 2] = rgb[2];
      imgData.data[pidx + 3] = alpha;
    }}
  }}
  ctx.putImageData(imgData, 0, 0);

  // Head outline
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, headR, 0, 2 * Math.PI);
  ctx.stroke();

  // Nose
  ctx.beginPath();
  ctx.moveTo(cx - 8, cy - headR);
  ctx.lineTo(cx, cy - headR - 12);
  ctx.lineTo(cx + 8, cy - headR);
  ctx.strokeStyle = '#888';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Ears
  ctx.beginPath();
  ctx.ellipse(cx - headR - 5, cy, 5, 12, 0, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(cx + headR + 5, cy, 5, 12, 0, 0, 2 * Math.PI);
  ctx.stroke();

  // Electrode dots
  const involvedSet = new Set(c.involved_channels || []);
  const involvedElecs = new Set();
  for (const chIdx of involvedSet) {{
    if (chIdx < BIPOLAR_ELECTRODES.length) {{
      for (const e of BIPOLAR_ELECTRODES[chIdx]) involvedElecs.add(e);
    }}
  }}

  for (const ename of electrodeNames) {{
    const ex = cx + ELECTRODE_POS[ename][0] * headR;
    const ey = cy - ELECTRODE_POS[ename][1] * headR;
    const score = electrodeScores[ename];
    const rgb = scoreToColor(norm(score));
    const isInv = involvedElecs.has(ename);
    const r = 3 + 5 * norm(score);
    ctx.beginPath();
    ctx.arc(ex, ey, r, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    ctx.fill();
    ctx.strokeStyle = isInv ? '#228833' : '#555';
    ctx.lineWidth = isInv ? 2 : 0.8;
    ctx.stroke();

    ctx.fillStyle = '#333';
    ctx.font = '8px Helvetica, Arial, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(ename, ex, ey + r + 1);
  }}
}}

// Render all panels
function render() {{
  const container = document.getElementById('panels');
  container.innerHTML = '';

  for (let i = 0; i < CASES.length; i++) {{
    const c = CASES[i];
    const panel = document.createElement('div');
    panel.className = 'example-panel';

    // Title bar
    const diffClass = c.difficulty.includes('Easy') || c.difficulty.includes('Low') ? 'difficulty-easy' :
                      c.difficulty.includes('Medium') ? 'difficulty-medium' : 'difficulty-hard';
    let titleText = c.subtype.toUpperCase() + ' -- ';
    titleText += '<span class="' + diffClass + '">' + c.difficulty + ' case</span>';
    titleText += ' -- Patient: ' + c.patient_id;
    if (c.jaccard !== null && c.jaccard !== undefined) {{
      titleText += ' -- Inter-rater Jaccard: ' + c.jaccard.toFixed(3);
    }}
    const titleBar = document.createElement('div');
    titleBar.className = 'title-bar';
    titleBar.innerHTML = titleText;
    panel.appendChild(titleBar);

    // Main row: EEG + topo
    const mainRow = document.createElement('div');
    mainRow.className = 'main-row';

    const eegDiv = document.createElement('div');
    eegDiv.className = 'eeg-container';
    const eegCanvas = document.createElement('canvas');
    eegDiv.appendChild(eegCanvas);
    mainRow.appendChild(eegDiv);

    const topoDiv = document.createElement('div');
    topoDiv.className = 'topo-container';
    const topoCanvas = document.createElement('canvas');
    topoDiv.appendChild(topoCanvas);

    const verbal = document.createElement('div');
    verbal.className = 'verbal-desc';
    verbal.textContent = generateVerbalDescription(c);
    topoDiv.appendChild(verbal);

    mainRow.appendChild(topoDiv);
    panel.appendChild(mainRow);

    // Annotation row
    const annoRow = document.createElement('div');
    annoRow.className = 'annotation-row';
    let annoHTML = '';

    if (c.rater_details) {{
      annoHTML += '<strong>Rater agreement:</strong> ';
      const parts = [];
      for (const [rater, regs] of Object.entries(c.rater_details)) {{
        parts.push('<span class="rater-label">' + rater + '</span>=[' + regs.join(', ') + ']');
      }}
      annoHTML += parts.join(' &nbsp; ');
      if (c.jaccard !== null) annoHTML += ' &nbsp; | &nbsp; Jaccard=' + c.jaccard.toFixed(3);
      annoHTML += '<br>';
    }} else {{
      annoHTML += '<strong>Spatial regions:</strong> ' + (c.gt_regions.length > 0 ? c.gt_regions.join(', ') : 'none labeled') + '<br>';
    }}

    annoHTML += '<span class="pred-label">Model prediction:</span> ';
    annoHTML += '[' + (c.pred_regions || []).join(', ') + ']';
    annoHTML += ' &nbsp; | &nbsp; Confidence: ' + (c.pred_confidence || 0).toFixed(3);
    if (IS_PD) {{
      annoHTML += ' &nbsp; | &nbsp; Pred freq: ' + (c.pred_freq || 0).toFixed(2) + ' Hz';
      annoHTML += ' &nbsp; | &nbsp; Pred lat: ' + (c.pred_lat || '?');
    }}

    annoRow.innerHTML = annoHTML;
    panel.appendChild(annoRow);

    container.appendChild(panel);

    // Draw after DOM insertion
    drawEEG(eegCanvas, c);
    drawTopoplot(topoCanvas, c);
  }}
}}

render();
</script>
</body>
</html>"""
    return html


def main():
    print("=" * 70)
    print("  Paper Figure Generator — EEG Characterization Examples")
    print("=" * 70)

    # Load data
    print("\nLoading labels...")
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    dt_by_matfile = load_discharge_times()
    print(f"  {len(sl)} segments, {len(ann)} annotations, {len(dt_by_matfile)} discharge time entries")

    for subtype in ['lpd', 'gpd', 'lrda', 'grda']:
        is_pd = subtype in ('lpd', 'gpd')
        print(f"\n{'─' * 50}")
        print(f"  Processing {subtype.upper()}")
        print(f"{'─' * 50}")

        # Select cases
        if is_pd:
            selected = select_pd_cases(subtype, ann, sl, dt_by_matfile)
        else:
            selected = select_rda_cases(subtype, sl)

        if not selected:
            print(f"  No cases found for {subtype.upper()}!")
            continue

        print(f"  Selected {len(selected)} cases:")
        for s in selected:
            if is_pd:
                sid, cand, diff = s
                print(f"    {diff}: {sid} (Jaccard={cand['jaccard']:.3f})")
            else:
                print(f"    {s['difficulty']}: {s['mat_file']} (lat={s['row'].get('laterality')}, regs={s.get('regions')})")

        # Process cases
        cases = []
        for case_info in selected:
            case = process_case(case_info, subtype, dt_by_matfile)
            if case:
                cases.append(case)

        if not cases:
            print(f"  No valid cases after processing!")
            continue

        # Build HTML
        print(f"  Building HTML ({len(cases)} cases)...")
        html = build_html(cases, subtype)

        out_path = OUT_DIR / f'figure_{subtype}_examples.html'
        with open(str(out_path), 'w') as f:
            f.write(html)
        print(f"  Written: {out_path}")

    print(f"\n{'=' * 70}")
    print("  Done! Figures in paper_materials/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
