"""
Spatial Labeling Viewer for PD and RDA EEG Segments.

Generates HTML viewers where MW can review and correct model-predicted
spatial localization (which brain regions are involved with periodic/rhythmic
discharges).

For LPD/GPD: uses PDProfiler for channel_probs and region_scores.
For LRDA/GRDA: uses PLV-based scoring (bandpass 0.5-3.5 Hz, reference channel,
PLV per channel, map to regions).

Usage:
    conda run -n morgoth python code/generators/labeling/generate_spatial_labeler.py \
        --subtype lpd --batch 1 --batch-size 500
    # Test mode (5 segments):
    conda run -n morgoth python code/generators/labeling/generate_spatial_labeler.py \
        --subtype lpd --batch 1 --batch-size 5 --test
"""

import sys
import json
import argparse
import numpy as np
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend, sosfiltfilt, hilbert

LABELING_DIR = Path(__file__).resolve().parent
CODE_DIR = LABELING_DIR.parent.parent  # code/
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results' / 'labeling_tools' / 'spatial'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
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
    'LF': [0, 8],     # Fp1-F7, Fp1-F3
    'RF': [4, 12],    # Fp2-F8, Fp2-F4
    'LT': [1, 2],     # F7-T3, T3-T5
    'RT': [5, 6],     # F8-T4, T4-T6
    'LCP': [9, 10],   # F3-C3, C3-P3
    'RCP': [13, 14],  # F4-C4, C4-P4
    'LO': [3, 11],    # T5-O1, P3-O1
    'RO': [7, 15],    # T6-O2, P4-O2
    'MID': [16, 17],  # Fz-Cz, Cz-Pz
}

# ── PDProfiler (lazy loaded) ──
_pd_profiler = None

def _get_pd_profiler():
    global _pd_profiler
    if _pd_profiler is None:
        from pd_profiler import PDProfiler
        _pd_profiler = PDProfiler()
    return _pd_profiler


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
    """Use PDProfiler for LPD/GPD spatial prediction.

    Returns:
        channel_probs: list of 18 floats
        region_scores: dict region -> score
        predicted_regions: list of region names above threshold
        confidence: float (mean of top region scores)
    """
    try:
        pc = _get_pd_profiler()
        result = pc.characterize(seg, subtype=subtype)
        channel_probs = result.get('channel_probs', [0.5] * 18)
        region_scores = result.get('region_scores', {r: 0.5 for r in REGIONS})
        predicted_regions = result.get('regions', [])
        # Confidence: mean of predicted region scores
        if predicted_regions:
            confidence = float(np.mean([region_scores[r] for r in predicted_regions]))
        else:
            confidence = 0.0
        return channel_probs, region_scores, predicted_regions, confidence
    except Exception as e:
        print(f"  PDProfiler failed: {e}")
        return [0.5] * 18, {r: 0.5 for r in REGIONS}, [], 0.0


def predict_spatial_rda(seg):
    """Use PLV-based scoring for LRDA/GRDA spatial prediction.

    Bandpass 0.5-3.5 Hz, find reference channel (highest variance in band),
    compute PLV for each channel vs reference, map to regions.

    Returns:
        channel_scores: list of 18 floats (PLV-based)
        region_scores: dict region -> score
        predicted_regions: list of region names above threshold
        confidence: float
    """
    try:
        n_ch = min(18, seg.shape[0])
        # Bandpass 0.5-3.5 Hz
        sos = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)],
                     btype='bandpass', output='sos')
        seg_f = sosfiltfilt(sos, seg[:n_ch], axis=1)

        # Find reference channel (highest variance in band)
        variances = np.var(seg_f, axis=1)
        ref_ch = int(np.argmax(variances))

        # Phase of each channel
        phases = np.zeros_like(seg_f)
        for ch in range(n_ch):
            phases[ch] = np.angle(hilbert(seg_f[ch]))

        ref_phase = phases[ref_ch]

        # PLV for each channel vs reference
        channel_scores = np.zeros(n_ch)
        for ch in range(n_ch):
            phase_diff = phases[ch] - ref_phase
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            channel_scores[ch] = plv

        # Map to regions (max aggregation)
        region_scores = {}
        for region, chs in REGION_TO_CHANNELS.items():
            valid_chs = [c for c in chs if c < n_ch]
            if valid_chs:
                region_scores[region] = float(max(channel_scores[c] for c in valid_chs))
            else:
                region_scores[region] = 0.0

        # Threshold at 0.38 (optimized for avg(LB,PH)=0.767, exceeds LB-PH agreement 0.762)
        predicted_regions = [r for r, s in region_scores.items() if s > 0.38]

        if predicted_regions:
            confidence = float(np.mean([region_scores[r] for r in predicted_regions]))
        else:
            confidence = 0.0

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


def find_unlabeled_segments(subtype, min_votes=10):
    """Find segments without spatial labels, one per patient.

    Prioritizes segments with higher IIIC votes.
    """
    import pandas as pd
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))
    sl_active = sl[sl.excluded.fillna(False).astype(bool) == False]
    sub = sl_active[sl_active.subtype == subtype].copy()

    # Missing spatial labels
    missing = sub[sub.spatial_channels.isna()].copy()

    # Add vote count for sorting
    missing['n_votes'] = pd.to_numeric(missing['iiic_n_votes'], errors='coerce').fillna(0)

    # One segment per patient (keep the one with most votes)
    missing = missing.sort_values('n_votes', ascending=False)
    missing = missing.drop_duplicates(subset='patient_id')

    # Filter by minimum votes if specified
    if min_votes > 0:
        missing = missing[missing.n_votes >= min_votes].copy()

    print(f"  Total {subtype.upper()} active: {len(sub)}")
    print(f"  Missing spatial labels: {len(missing)} unique patients")
    if min_votes > 0:
        print(f"  With >= {min_votes} votes: {len(missing)}")
    return missing


def main():
    parser = argparse.ArgumentParser(description='Spatial Labeling Viewer')
    parser.add_argument('--subtype', required=True,
                        choices=['lpd', 'gpd', 'lrda', 'grda'],
                        help='EEG pattern subtype')
    parser.add_argument('--batch', type=int, default=1, help='Batch number (1-indexed)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Cases per batch')
    parser.add_argument('--min-votes', type=int, default=10,
                        help='Minimum IIIC crowd votes (default: 10)')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (overrides batch-size to 5)')
    args = parser.parse_args()

    if args.test:
        args.batch_size = 5

    subtype = args.subtype
    is_pd = subtype in ('lpd', 'gpd')

    print("=" * 70)
    print(f"  Spatial Labeling Viewer — {subtype.upper()}")
    print("=" * 70)

    # Find unlabeled segments
    print(f"\nFinding unlabeled {subtype.upper()} segments...")
    unlabeled = find_unlabeled_segments(subtype, min_votes=args.min_votes)

    # Sort by model confidence (will be computed during processing)
    # For now sort by n_votes descending (more votes = more certain subtype)
    unlabeled = unlabeled.sort_values('n_votes', ascending=False).reset_index(drop=True)

    # Batch
    total = len(unlabeled)
    n_batches = max(1, (total + args.batch_size - 1) // args.batch_size)
    batch_start = (args.batch - 1) * args.batch_size
    batch_end = min(batch_start + args.batch_size, total)

    if batch_start >= total:
        print(f"  Batch {args.batch} is out of range (only {n_batches} batches)")
        return

    batch = unlabeled.iloc[batch_start:batch_end]
    print(f"\n  Batch {args.batch}/{n_batches}: cases {batch_start+1}-{batch_end} of {total}")

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
        seg_display = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try:
                seg_display[ch] = filtfilt(b_lp, a_lp, detrend(seg[ch], type='linear'))
            except Exception:
                seg_display[ch] = seg[ch]

        # Detrend + lowpass filter monopolar for CAR display
        mono_display = None
        if mono_raw is not None:
            mono_display = np.zeros_like(mono_raw)
            for ch in range(mono_raw.shape[0]):
                try:
                    mono_display[ch] = filtfilt(b_lp, a_lp, detrend(mono_raw[ch], type='linear'))
                except Exception:
                    mono_display[ch] = mono_raw[ch]

        # Predict spatial localization
        if is_pd:
            channel_scores, region_scores, predicted_regions, confidence = \
                predict_spatial_pd(seg, subtype)
        else:
            channel_scores, region_scores, predicted_regions, confidence = \
                predict_spatial_rda(seg)

        # Compute CSD topography at discharge peaks (for PD only)
        csd_electrode_scores = None
        if is_pd and mono_raw is not None:
            try:
                import mne
                mne.set_log_level('ERROR')
                pc = _get_pd_profiler()
                result = pc.characterize(seg, subtype=subtype)
                discharge_times = result.get('discharge_times', [])

                # Extract peak-to-trough voltage at each discharge
                # This captures the discharge waveform amplitude, matching what bipolar shows
                mono_19 = mono_raw[:19, :2000].astype(float)
                # Re-reference to average first
                mono_19 = mono_19 - np.mean(mono_19, axis=0, keepdims=True)

                peak_amplitudes = []
                half_win = int(0.15 * FS)  # 150ms window for peak-to-trough
                for t in discharge_times:
                    s = int(t * FS)
                    s_start = max(0, s - half_win)
                    s_end = min(mono_19.shape[1], s + half_win)
                    if s_end - s_start < 10:
                        continue
                    window = mono_19[:, s_start:s_end]
                    # Peak-to-trough amplitude per channel
                    amp = np.max(window, axis=1) - np.min(window, axis=1)
                    peak_amplitudes.append(amp)

                if len(peak_amplitudes) >= 2:
                    mean_topo = np.mean(peak_amplitudes, axis=0)  # (19,)
                else:
                    # Fallback: use full-segment peak-to-trough
                    from scipy.signal import detrend as sp_detrend
                    mono_det = sp_detrend(mono_19, axis=1)
                    mean_topo = np.max(mono_det, axis=1) - np.min(mono_det, axis=1)

                # Apply CSD via MNE
                ch_names_orig = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz',
                                 'Fp2','F4','C4','P4','F8','T4','T6','O2']
                name_map_mne = {'T3':'T7','T4':'T8','T5':'P7','T6':'P8'}
                mne_names = [name_map_mne.get(n, n) for n in ch_names_orig]
                info_mne = mne.create_info(ch_names=mne_names, sfreq=FS, ch_types='eeg')
                montage_mne = mne.channels.make_standard_montage('standard_1020')
                info_mne.set_montage(montage_mne)

                evoked = mne.EvokedArray(mean_topo.reshape(19, 1), info_mne, tmin=0)
                evoked_csd = mne.preprocessing.compute_current_source_density(evoked)
                csd_vals = np.abs(evoked_csd.data[:, 0])  # absolute CSD

                # Normalize to [0, 1]
                if csd_vals.max() > 0:
                    csd_norm = (csd_vals / csd_vals.max()).tolist()
                else:
                    csd_norm = [0.0] * 19
                csd_electrode_scores = {ch_names_orig[i]: round(csd_norm[i], 4) for i in range(19)}
            except Exception as e:
                print(f"  CSD failed: {e}")
                csd_electrode_scores = None

        case = {
            'patient_id': str(row['patient_id']),
            'segment_id': str(mat_file).replace('.mat', ''),
            'mat_file': str(mat_file),
            'subtype': subtype,
            'channel_scores': [round(float(s), 4) for s in channel_scores],
            'region_scores': {r: round(float(s), 4) for r, s in region_scores.items()},
            'predicted_regions': predicted_regions,
            'confidence': round(confidence, 4),
            'eeg_data': downsample(seg_display, 1000),
            'mono_data': downsample(mono_display, 1000) if mono_display is not None else None,
            'csd_electrode_scores': csd_electrode_scores,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0 or (i + 1) == len(batch):
            print(f"  {i+1}/{len(batch)} processed")

    print(f"\n  Total cases: {len(cases_data)} (skipped {n_skipped} missing EEG)")

    if len(cases_data) == 0:
        print("  No cases to label!")
        return

    # Sort by confidence descending (easiest cases first)
    cases_data.sort(key=lambda c: c['confidence'], reverse=True)

    # Build HTML
    print("\nBuilding HTML viewer...")
    html = build_html(cases_data, subtype, args.batch, n_batches)

    out_path = OUT_DIR / f'{subtype}_spatial_batch{args.batch}.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"  Written to {out_path}")
    print(f"  {len(cases_data)} cases ready for review")

    # Open in browser
    import subprocess
    subprocess.run(['open', str(out_path)])
    print("=" * 70)


def build_html(cases_data, subtype, batch_num, n_batches):
    regions_json = json.dumps(REGIONS)
    region_to_channels_json = json.dumps(REGION_TO_CHANNELS)

    cases_json = json.dumps(cases_data,
                            default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)

    n_cases = len(cases_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{subtype.upper()} Spatial Labeler — Batch {batch_num}/{n_batches} ({n_cases} cases)</title>
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

  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 20px; flex-wrap: wrap; border-bottom: 1px solid #333; font-size: 13px;
  }}
  .info-item {{ color: #bbb; }}
  .info-item strong {{ color: #eee; }}

  #main-container {{
    display: flex; width: 100%; min-height: 700px;
  }}

  #eeg-panel {{
    flex: 0 0 60%; max-width: 60%; padding: 8px; position: relative;
  }}
  #eeg-canvas {{ cursor: default; display: block; width: 100%; }}

  #spatial-panel {{
    flex: 0 0 40%; max-width: 40%; padding: 8px;
    display: flex; flex-direction: column; gap: 8px;
  }}

  #topo-container {{ text-align: center; position: relative; }}
  #topo-canvas {{ display: block; margin: 0 auto; }}

  #region-panel {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 6px; padding: 8px;
    background: #252525; border-radius: 6px;
  }}
  .region-btn {{
    padding: 10px 8px; border: 2px solid #555; border-radius: 6px;
    background: #333; color: #ccc; cursor: pointer; font-family: monospace;
    font-size: 14px; font-weight: bold; text-align: center;
    transition: all 0.15s; user-select: none;
  }}
  .region-btn:hover {{ background: #444; border-color: #888; }}
  .region-btn.active {{ background: #1a3a1a; border-color: #44cc88; color: #44ff66; }}
  .region-btn .score {{ font-size: 11px; color: #888; display: block; margin-top: 2px; }}
  .region-btn.active .score {{ color: #88cc88; }}

  .export-btn {{
    padding: 6px 14px; border: 1px solid #44cc44; border-radius: 4px;
    background: #2a3a2a; color: #44cc44; cursor: pointer;
    font-family: monospace; font-size: 12px; font-weight: bold;
  }}
  .export-btn:hover {{ background: #3a4a3a; }}

  #save-status {{ color: #44cc44; font-size: 13px; }}
  #eeg-canvas {{ cursor: pointer; }}
  #topo-canvas {{ cursor: pointer; }}
  #threshold-container {{ background: #222; border-radius: 4px; }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 6px 16px; background: #222;
    border-top: 1px solid #333; line-height: 1.8;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <span style="font-size:16px; font-weight:bold; color:#44cc88;">{subtype.upper()} Spatial Labeler</span>
    <span style="font-size:12px; color:#888;">Batch {batch_num}/{n_batches}</span>
    <span id="counter" style="font-size:13px; color:#aaa;">1 / {n_cases}</span>
  </div>
  <div id="header-right">
    <button class="export-btn" onclick="skipCase()" style="border-color:#ff6644; color:#ff6644; background:#3a2a2a;">Reject <span class="key">X</span></button>
    <button class="export-btn" onclick="exportJSON()">Export <span class="key">E</span></button>
    <span id="save-status"></span>
    <span id="labeled-count" style="font-size:12px; color:#aaa;"></span>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar" style="width:0%"></div></div>

<div id="info-panel">
  <span class="info-item">Patient: <strong id="info-pid">--</strong></span>
  <span class="info-item">Subtype: <strong id="info-subtype" style="color:#44cc88;">--</strong></span>
  <span class="info-item">Confidence: <strong id="info-confidence" style="color:#ff9800;">--</strong></span>
  <span class="info-item">Regions: <strong id="info-regions" style="color:#44ff66;">--</strong></span>
  <span class="info-item">Montage: <strong id="montage-label">bipolar</strong></span>
</div>
<div id="verbal-panel" style="padding:6px 16px; background:#1a2a1a; border-bottom:1px solid #333; font-size:14px;">
  <strong id="verbal-description" style="color:#44ff66;">--</strong>
</div>

<div id="main-container">
  <div id="eeg-panel">
    <canvas id="eeg-canvas"></canvas>
  </div>
  <div id="spatial-panel">
    <div id="topo-toggle" style="text-align:center; padding:4px; background:#222; border-bottom:1px solid #333;">
      <button id="btn-cnn-plv" onclick="setTopoMode('cnn')" style="padding:3px 10px; font-size:11px; font-family:monospace; cursor:pointer; background:#1a3a1a; color:#44cc88; border:1px solid #44cc88; border-radius:3px; margin-right:4px;">CNN+PLV</button>
      <button id="btn-csd" onclick="setTopoMode('csd')" style="padding:3px 10px; font-size:11px; font-family:monospace; cursor:pointer; background:#222; color:#888; border:1px solid #555; border-radius:3px;">CSD at Peak</button>
    </div>
    <div id="topo-container">
      <canvas id="topo-canvas"></canvas>
    </div>
    <div id="threshold-container" style="padding:4px 10px; display:flex; align-items:center; gap:8px;">
      <span style="font-size:11px; color:#888;">Threshold:</span>
      <input type="range" id="threshold-slider" min="0" max="100" value="38"
        style="flex:1; accent-color:#44cc88; cursor:pointer;"
        oninput="onThresholdChange(this.value)">
      <span id="threshold-value" style="font-size:12px; color:#44cc88; min-width:35px;">0.38</span>
    </div>
    <div id="region-panel">
      <!-- Region buttons built by JS -->
    </div>
  </div>
</div>

<div id="shortcuts">
  <span class="key">1</span>-<span class="key">8</span> Toggle regions &nbsp;&nbsp;
  <span class="key">Enter</span> Accept &amp; advance &nbsp;&nbsp;
  <span class="key">X</span> Reject/skip &nbsp;&nbsp;
  <span class="key">&larr;</span>/<span class="key">&rarr;</span> Navigate &nbsp;&nbsp;
  <span class="key">Ctrl</span> Toggle montage &nbsp;&nbsp;
  <span class="key">E</span> Export &nbsp;&nbsp;
  Click <strong>channels</strong>, <strong>topoplot electrodes</strong>, or <strong>region buttons</strong> to toggle involvement. Use <strong>slider</strong> to adjust threshold.
</div>

<script>
const CASES = {cases_json};
const REGIONS = {regions_json};
const REGION_TO_CHANNELS = {region_to_channels_json};

const BIPOLAR_NAMES = ['Fp1-F7','F7-T3','T3-T5','T5-O1','Fp2-F8','F8-T4','T4-T6','T6-O2',
  'Fp1-F3','F3-C3','C3-P3','P3-O1','Fp2-F4','F4-C4','C4-P4','P4-O2','Fz-Cz','Cz-Pz'];
const MONO_NAMES = ['Fp1-avg','F3-avg','C3-avg','P3-avg','F7-avg','T3-avg','T5-avg','O1-avg','Fz-avg','Cz-avg',
  'Pz-avg','Fp2-avg','F4-avg','C4-avg','P4-avg','F8-avg','T4-avg','T6-avg','O2-avg'];

const CAR_DISPLAY_ORDER = [
  0, 1, 2, 3,    // Fp1, F3, C3, P3 (left parasag)
  4, 5, 6, 7,    // F7, T3, T5, O1 (left temporal)
  -1,
  8, 9, 10,      // Fz, Cz, Pz (midline)
  -1,
  11, 12, 13, 14, // Fp2, F4, C4, P4 (right parasag)
  15, 16, 17, 18  // F8, T4, T6, O2 (right temporal)
];

const BIPOLAR_DISPLAY_ORDER = [0,1,2,3, -1, 8,9,10,11, -1, 16,17, -1, 12,13,14,15, -1, 4,5,6,7];

const DURATION = 10.0;
const Z_SCALE = 0.01;
const CLIP_UV = 300.0;

const EEG_WIDTH = 900;
const EEG_HEIGHT = 700;
const MARGIN_LEFT = 70;
const MARGIN_RIGHT = 20;
const MARGIN_TOP = 30;
const MARGIN_BOTTOM = 25;
const PLOT_LEFT = MARGIN_LEFT;
const PLOT_RIGHT = EEG_WIDTH - MARGIN_RIGHT;
const PLOT_W = PLOT_RIGHT - PLOT_LEFT;
const PLOT_H = EEG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

const TOPO_SIZE = 380;

// Electrode positions (normalized to unit circle, nose at top)
const ELECTRODE_POS = {{
  'Fp1': [-0.31, 0.95], 'Fp2': [0.31, 0.95],
  'F7': [-0.81, 0.59], 'F3': [-0.39, 0.59], 'Fz': [0.0, 0.59], 'F4': [0.39, 0.59], 'F8': [0.81, 0.59],
  'T3': [-1.0, 0.0], 'C3': [-0.5, 0.0], 'Cz': [0.0, 0.0], 'C4': [0.5, 0.0], 'T4': [1.0, 0.0],
  'T5': [-0.81, -0.59], 'P3': [-0.39, -0.59], 'Pz': [0.0, -0.59], 'P4': [0.39, -0.59], 'T6': [0.81, -0.59],
  'O1': [-0.31, -0.95], 'O2': [0.31, -0.95]
}};

// Map bipolar channel index to contributing electrode names
const BIPOLAR_ELECTRODES = [
  ['Fp1','F7'], ['F7','T3'], ['T3','T5'], ['T5','O1'],
  ['Fp2','F8'], ['F8','T4'], ['T4','T6'], ['T6','O2'],
  ['Fp1','F3'], ['F3','C3'], ['C3','P3'], ['P3','O1'],
  ['Fp2','F4'], ['F4','C4'], ['C4','P4'], ['P4','O2'],
  ['Fz','Cz'], ['Cz','Pz']
];

// State — per-channel tracking (18 bipolar channels)
let idx = 0;
let labeled = new Set();
let selectedChannels = new Set(); // bipolar channel indices (0-17)
let montage = 'bipolar';
let CHANNEL_NAMES = BIPOLAR_NAMES;

// Derive regions from selected channels
function getSelectedRegions() {{
  const regions = [];
  for (const [region, chs] of Object.entries(REGION_TO_CHANNELS)) {{
    if (chs.some(ch => selectedChannels.has(ch))) {{
      regions.push(region);
    }}
  }}
  return regions;
}}

// Derive monopolar electrode involvement from selected bipolar channels
function getInvolvedElectrodes() {{
  const involved = new Set();
  for (const chIdx of selectedChannels) {{
    for (const elec of BIPOLAR_ELECTRODES[chIdx]) {{
      involved.add(elec);
    }}
  }}
  return involved;
}}

// Persistence
const STORAGE_KEY = '{subtype}_spatial_batch{batch_num}';
let allLabels = {{}};
try {{ allLabels = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}'); }} catch(e) {{ allLabels = {{}}; }}
function saveAll() {{ localStorage.setItem(STORAGE_KEY, JSON.stringify(allLabels)); }}

// Build region buttons
(function() {{
  const panel = document.getElementById('region-panel');
  for (let ri = 0; ri < REGIONS.length; ri++) {{
    const btn = document.createElement('div');
    btn.className = 'region-btn';
    btn.id = 'region-btn-' + ri;
    btn.innerHTML = REGIONS[ri] + '<span class="score" id="region-score-' + ri + '">0.00</span>';
    btn.dataset.ri = ri;
    btn.onclick = () => toggleRegion(ri);
    panel.appendChild(btn);
  }}
}})();

function toggleRegion(ri) {{
  // Toggle ALL channels belonging to this region
  const region = REGIONS[ri];
  const chs = REGION_TO_CHANNELS[region];
  const allSelected = chs.every(ch => selectedChannels.has(ch));
  for (const ch of chs) {{
    if (allSelected) {{
      selectedChannels.delete(ch);
    }} else {{
      selectedChannels.add(ch);
    }}
  }}
  redraw();
}}

function toggleChannel(chIdx) {{
  // Toggle a single bipolar channel
  if (selectedChannels.has(chIdx)) {{
    selectedChannels.delete(chIdx);
  }} else {{
    selectedChannels.add(chIdx);
  }}
  redraw();
}}

function updateRegionButtons() {{
  const selRegions = getSelectedRegions();
  for (let ri = 0; ri < REGIONS.length; ri++) {{
    const btn = document.getElementById('region-btn-' + ri);
    const region = REGIONS[ri];
    const chs = REGION_TO_CHANNELS[region];
    const anySelected = chs.some(ch => selectedChannels.has(ch));
    const allSelected = chs.every(ch => selectedChannels.has(ch));
    btn.classList.remove('active', 'partial');
    if (allSelected) {{
      btn.classList.add('active');
    }} else if (anySelected) {{
      btn.classList.add('active');
      btn.style.opacity = '0.7';
    }} else {{
      btn.style.opacity = '1.0';
    }}
    if (allSelected || !anySelected) btn.style.opacity = '1.0';
    const c = CASES[idx];
    const score = c.region_scores[REGIONS[ri]] || 0;
    document.getElementById('region-score-' + ri).textContent = score.toFixed(2);
  }}
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
  document.getElementById('montage-label').textContent = montage;
}}

// Channel color based on per-channel selection
function channelColor(chIdx) {{
  if (selectedChannels.has(chIdx)) {{
    const c = CASES[idx];
    const score = c.channel_scores[chIdx] || 0;
    const g = Math.round(100 + 155 * score);
    return 'rgb(0,' + g + ',0)';
  }}
  return '#555555';
}}

function channelLabelColor(chIdx) {{
  if (selectedChannels.has(chIdx)) return '#44ff66';
  return '#888888';
}}

// For CAR montage: monopolar electrode is highlighted if any of its bipolar channels are selected
function monoChannelColor(monoIdx) {{
  const monoName = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2'][monoIdx];
  const involved = getInvolvedElectrodes();
  if (involved.has(monoName)) return '#44ff66';
  return '#555555';
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

  // Traces
  for (let di = 0; di < N_DISPLAY; di++) {{
    const ch = DISPLAY_CHANNELS[di];
    if (ch.idx < 0) continue;
    const yCenter = MARGIN_TOP + chSpacing * (di + 1);
    const trace = eegData[ch.idx];
    let isHL;
    if (montage === 'bipolar') {{
      isHL = selectedChannels.has(ch.idx);
    }} else {{
      // CAR: highlight if this monopolar electrode is involved
      isHL = monoChannelColor(ch.idx) === '#44ff66';
    }}
    ctx.strokeStyle = isHL ? (montage === 'bipolar' ? channelColor(ch.idx) : '#44ff66') : '#555555';
    ctx.lineWidth = isHL ? 1.2 : 0.6;
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
    let isHL2;
    if (montage === 'bipolar') {{
      isHL2 = selectedChannels.has(ch.idx);
    }} else {{
      isHL2 = monoChannelColor(ch.idx) === '#44ff66';
    }}
    ctx.fillStyle = isHL2 ? (montage === 'bipolar' ? channelLabelColor(ch.idx) : '#44ff66') : '#888888';
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
  const selRegs = getSelectedRegions();
  const regStr = selRegs.length > 0 ? selRegs.join(',') : 'none';
  const title = c.patient_id + '  |  ' + c.subtype.toUpperCase() + '  |  regions=' + regStr + '  |  conf=' + c.confidence.toFixed(3);
  ctx.fillText(title, EEG_WIDTH / 2, 6);
}}

// ── Topoplot ──
// Colormap: normalized score 0→1 maps dark blue → cyan → yellow → red
function scoreToColor(normScore) {{
  const t = Math.max(0, Math.min(1, normScore));
  if (t <= 0.33) {{
    const s = t / 0.33;
    return [
      Math.round(20 + (0 - 20) * s),
      Math.round(20 + (180 - 20) * s),
      Math.round(80 + (220 - 80) * s)
    ];
  }} else if (t <= 0.66) {{
    const s = (t - 0.33) / 0.33;
    return [
      Math.round(0 + (255 - 0) * s),
      Math.round(180 + (220 - 180) * s),
      Math.round(220 + (0 - 220) * s)
    ];
  }} else {{
    const s = (t - 0.66) / 0.34;
    return [
      Math.round(255),
      Math.round(220 + (30 - 220) * s),
      Math.round(0)
    ];
  }}
}}

// Normalize score to [0,1] relative to current case min/max
let topoMin = 0, topoMax = 1;
function normalizeScore(raw) {{
  if (topoMax <= topoMin) return 0.5;
  return (raw - topoMin) / (topoMax - topoMin);
}}

// ── Verbal description (ACNS 2021 style) ──
const REGION_BARE = {{
  'LF': 'frontal', 'RF': 'frontal',
  'LT': 'temporal', 'RT': 'temporal',
  'LCP': 'centro-parietal', 'RCP': 'centro-parietal',
  'LO': 'occipital', 'RO': 'occipital',
  'MID': 'midline'
}};
const LEFT_REGS = ['LF', 'LT', 'LCP', 'LO'];
const RIGHT_REGS = ['RF', 'RT', 'RCP', 'RO'];

function generateVerbalDescription() {{
  const c = CASES[idx];
  const subtype = c.subtype.toUpperCase();
  const selRegs = getSelectedRegions();

  if (selRegs.length === 0) return subtype + ' — no regions selected';

  // Laterality from selected regions
  const leftSel = selRegs.filter(r => LEFT_REGS.includes(r));
  const rightSel = selRegs.filter(r => RIGHT_REGS.includes(r));
  const midSel = selRegs.includes('MID');
  const isLateralized = subtype === 'LPD' || subtype === 'LRDA';
  const isGeneralized = subtype === 'GPD' || subtype === 'GRDA';

  // Determine laterality string
  let latStr = '';
  if (leftSel.length > 0 && rightSel.length === 0) {{
    latStr = 'unilateral left';
  }} else if (rightSel.length > 0 && leftSel.length === 0) {{
    latStr = 'unilateral right';
  }} else if (leftSel.length > rightSel.length) {{
    latStr = 'bilateral, left-predominant';
  }} else if (rightSel.length > leftSel.length) {{
    latStr = 'bilateral, right-predominant';
  }} else if (leftSel.length > 0 && rightSel.length > 0) {{
    latStr = 'bilateral/symmetric';
  }} else if (midSel) {{
    latStr = 'midline';
  }}

  // Determine dominant regions (by score)
  const domRegs = (leftSel.length >= rightSel.length) ? leftSel : rightSel;
  if (domRegs.length === 0 && midSel) domRegs.push('MID');
  const scored = domRegs.map(r => [r, c.region_scores[r] || 0]).sort((a,b) => b[1] - a[1]);
  const topNames = [];
  for (const [r, s] of scored.slice(0, 2)) {{
    const bare = REGION_BARE[r];
    if (bare && !topNames.includes(bare)) topNames.push(bare);
  }}

  // For generalized: check predominance
  if (isGeneralized) {{
    const frontalScore = ((c.region_scores['LF'] || 0) + (c.region_scores['RF'] || 0)) / 2;
    const occipitalScore = ((c.region_scores['LO'] || 0) + (c.region_scores['RO'] || 0)) / 2;
    const temporalScore = ((c.region_scores['LT'] || 0) + (c.region_scores['RT'] || 0)) / 2;
    const allScores = {{'frontally': frontalScore, 'occipitally': occipitalScore, 'temporally': temporalScore}};
    const bestLabel = Object.entries(allScores).sort((a,b) => b[1] - a[1])[0][0];
    const range = Math.max(frontalScore, occipitalScore, temporalScore) - Math.min(frontalScore, occipitalScore, temporalScore);
    const predom = range > 0.1 ? bestLabel + ' predominant' : 'no regional predominance';
    return subtype + ', ' + predom + '.';
  }}

  // For lateralized
  const regionStr = topNames.length > 0
    ? 'maximal in the ' + topNames.join(' and ') + ' region' + (topNames.length > 1 ? 's' : '')
    : 'no region clearly dominant';

  return subtype + ', ' + latStr + '; ' + regionStr + '.';
}}

function updateVerbalDescription() {{
  document.getElementById('verbal-description').textContent = generateVerbalDescription();
}}

function drawTopoplot() {{
  const canvas = document.getElementById('topo-canvas');
  canvas.width = TOPO_SIZE;
  canvas.height = TOPO_SIZE;
  const ctx = canvas.getContext('2d');
  const c = CASES[idx];

  const cx = TOPO_SIZE / 2;
  const cy = TOPO_SIZE / 2;
  const headR = TOPO_SIZE * 0.40;

  // Compute per-electrode scores — either from CNN+PLV (bipolar) or CSD (monopolar)
  const electrodeScores = {{}};
  const electrodeNames = Object.keys(ELECTRODE_POS);

  if (topoMode === 'csd' && c.csd_electrode_scores) {{
    // CSD mode: use precomputed CSD electrode scores (already 19-channel monopolar)
    for (const ename of electrodeNames) {{
      electrodeScores[ename] = c.csd_electrode_scores[ename] || 0;
    }}
  }} else {{
    // CNN+PLV mode: derive from bipolar channel scores
    for (const ename of electrodeNames) {{
      let maxScore = 0;
      for (let bi = 0; bi < BIPOLAR_ELECTRODES.length; bi++) {{
        if (BIPOLAR_ELECTRODES[bi].includes(ename)) {{
          maxScore = Math.max(maxScore, c.channel_scores[bi] || 0);
        }}
      }}
      electrodeScores[ename] = maxScore;
    }}
  }}

  // Compute min/max for normalization (use full range of this case's scores)
  const allScores = Object.values(electrodeScores);
  topoMin = Math.min(...allScores);
  topoMax = Math.max(...allScores);
  // Ensure some contrast even if all scores are similar
  if (topoMax - topoMin < 0.05) {{
    topoMin = Math.max(0, topoMin - 0.1);
    topoMax = Math.min(1, topoMax + 0.1);
  }}

  // Draw heatmap via inverse-distance weighting, clipped to head circle
  const imgData = ctx.createImageData(TOPO_SIZE, TOPO_SIZE);
  const elecList = electrodeNames.map(name => ({{
    x: cx + ELECTRODE_POS[name][0] * headR,
    y: cy - ELECTRODE_POS[name][1] * headR,
    score: electrodeScores[name]
  }}));

  for (let py = 0; py < TOPO_SIZE; py++) {{
    for (let px = 0; px < TOPO_SIZE; px++) {{
      const dx = px - cx;
      const dy = py - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > headR + 2) continue;

      // Inverse-distance weighting
      let wSum = 0, vSum = 0;
      for (const e of elecList) {{
        const d = Math.sqrt((px - e.x) ** 2 + (py - e.y) ** 2);
        const w = 1.0 / (d * d * d + 1);  // cubic falloff for sharper gradients
        wSum += w;
        vSum += w * e.score;
      }}
      const val = vSum / wSum;
      const rgb = scoreToColor(normalizeScore(val));

      // Fade at edges
      let alpha = 230;
      if (dist > headR - 5) {{
        alpha = Math.max(0, Math.round(230 * (1 - (dist - headR + 5) / 7)));
      }}

      const pidx = (py * TOPO_SIZE + px) * 4;
      imgData.data[pidx] = rgb[0];
      imgData.data[pidx + 1] = rgb[1];
      imgData.data[pidx + 2] = rgb[2];
      imgData.data[pidx + 3] = alpha;
    }}
  }}
  ctx.putImageData(imgData, 0, 0);

  // Head outline
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx, cy, headR, 0, 2 * Math.PI);
  ctx.stroke();

  // Nose (triangle at top)
  ctx.beginPath();
  ctx.moveTo(cx - 10, cy - headR);
  ctx.lineTo(cx, cy - headR - 14);
  ctx.lineTo(cx + 10, cy - headR);
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth = 2;
  ctx.stroke();

  // Ears
  ctx.beginPath();
  ctx.ellipse(cx - headR - 6, cy, 6, 14, 0, 0, 2 * Math.PI);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(cx + headR + 6, cy, 6, 14, 0, 0, 2 * Math.PI);
  ctx.stroke();

  // Electrode dots and labels — highlight involved electrodes
  const involvedElecs = getInvolvedElectrodes();
  for (const ename of electrodeNames) {{
    const ex = cx + ELECTRODE_POS[ename][0] * headR;
    const ey = cy - ELECTRODE_POS[ename][1] * headR;
    const score = electrodeScores[ename];
    const rgb = scoreToColor(normalizeScore(score));
    const isInvolved = involvedElecs.has(ename);

    // Filled circle, size proportional to normalized score
    const r = 4 + 8 * normalizeScore(score);
    ctx.beginPath();
    ctx.arc(ex, ey, r, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')';
    ctx.fill();
    // Bright ring for involved electrodes
    ctx.strokeStyle = isInvolved ? '#44ff66' : '#fff';
    ctx.lineWidth = isInvolved ? 3 : 1;
    ctx.stroke();

    // Label
    ctx.fillStyle = isInvolved ? '#44ff66' : '#fff';
    ctx.font = (isInvolved ? 'bold ' : '') + '9px Consolas, Monaco, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(ename, ex, ey + r + 2);
  }}

  // Title
  ctx.fillStyle = '#ccc';
  ctx.font = 'bold 12px Consolas, Monaco, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  const topoTitle = topoMode === 'csd' ? 'CSD at Discharge Peak' : 'CNN+PLV Involvement';
  ctx.fillText(topoTitle, cx, 4);
}}

function updateInfo() {{
  const c = CASES[idx];
  document.getElementById('info-pid').textContent = c.patient_id;
  document.getElementById('info-subtype').textContent = c.subtype.toUpperCase();
  document.getElementById('info-confidence').textContent = c.confidence.toFixed(3);
  const selRegions = getSelectedRegions();
  document.getElementById('info-regions').textContent = selRegions.length > 0 ? selRegions.join(', ') + ' (' + selectedChannels.size + ' ch)' : 'none';
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + CASES.length;
  document.getElementById('progress-bar').style.width = ((idx + 1) / CASES.length * 100).toFixed(1) + '%';
  document.getElementById('labeled-count').textContent = labeled.size + ' labeled';
}}

function redraw() {{
  drawEEG();
  drawTopoplot();
  updateRegionButtons();
  updateInfo();
  updateVerbalDescription();
}}

// ── Topoplot mode toggle (CNN+PLV vs CSD) ──
let topoMode = 'cnn';  // 'cnn' or 'csd'

function setTopoMode(mode) {{
  topoMode = mode;
  document.getElementById('btn-cnn-plv').style.background = mode === 'cnn' ? '#1a3a1a' : '#222';
  document.getElementById('btn-cnn-plv').style.color = mode === 'cnn' ? '#44cc88' : '#888';
  document.getElementById('btn-cnn-plv').style.borderColor = mode === 'cnn' ? '#44cc88' : '#555';
  document.getElementById('btn-csd').style.background = mode === 'csd' ? '#1a2a3a' : '#222';
  document.getElementById('btn-csd').style.color = mode === 'csd' ? '#4488cc' : '#888';
  document.getElementById('btn-csd').style.borderColor = mode === 'csd' ? '#4488cc' : '#555';
  drawTopoplot();
}}

// ── Threshold slider ──
let currentThreshold = 0.38;

function onThresholdChange(val) {{
  currentThreshold = val / 100.0;
  document.getElementById('threshold-value').textContent = currentThreshold.toFixed(2);
  // Auto-select channels whose score >= threshold
  const c = CASES[idx];
  selectedChannels = new Set();
  for (let ch = 0; ch < 18; ch++) {{
    const score = c.channel_scores[ch] || 0;
    if (score >= currentThreshold) {{
      selectedChannels.add(ch);
    }}
  }}
  redraw();
}}

// ── Click on EEG canvas to toggle individual channels ──
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

  if (montage === 'bipolar') {{
    // Toggle single bipolar channel
    toggleChannel(clickedChIdx);
  }} else {{
    // CAR mode: toggle all bipolar channels involving this monopolar electrode
    const monoNames = ['Fp1','F3','C3','P3','F7','T3','T5','O1','Fz','Cz','Pz','Fp2','F4','C4','P4','F8','T4','T6','O2'];
    const clickedElec = monoNames[clickedChIdx];
    if (!clickedElec) return;
    // Find bipolar channels involving this electrode
    const affectedChs = [];
    for (let bi = 0; bi < BIPOLAR_ELECTRODES.length; bi++) {{
      if (BIPOLAR_ELECTRODES[bi].includes(clickedElec)) affectedChs.push(bi);
    }}
    const allSel = affectedChs.every(ch => selectedChannels.has(ch));
    for (const ch of affectedChs) {{
      if (allSel) selectedChannels.delete(ch);
      else selectedChannels.add(ch);
    }}
    redraw();
  }}
}});

// ── Click on topoplot to toggle electrode → its bipolar channels ──
document.getElementById('topo-canvas').addEventListener('click', function(e) {{
  const rect = this.getBoundingClientRect();
  const clickX = (e.clientX - rect.left) * (TOPO_SIZE / rect.width);
  const clickY = (e.clientY - rect.top) * (TOPO_SIZE / rect.height);

  const cx = TOPO_SIZE / 2;
  const cy = TOPO_SIZE / 2;
  const headR = TOPO_SIZE * 0.40;

  // Find nearest electrode
  let bestElec = null;
  let bestDist = 999;
  for (const [ename, pos] of Object.entries(ELECTRODE_POS)) {{
    const ex = cx + pos[0] * headR;
    const ey = cy - pos[1] * headR;
    const d = Math.sqrt((clickX - ex) ** 2 + (clickY - ey) ** 2);
    if (d < bestDist) {{
      bestDist = d;
      bestElec = ename;
    }}
  }}

  if (!bestElec || bestDist > 30) return;

  // Find bipolar channels involving this electrode
  const affectedChs = [];
  for (let bi = 0; bi < BIPOLAR_ELECTRODES.length; bi++) {{
    if (BIPOLAR_ELECTRODES[bi].includes(bestElec)) affectedChs.push(bi);
  }}

  // Toggle: if all selected, deselect; otherwise select
  const allSel = affectedChs.every(ch => selectedChannels.has(ch));
  for (const ch of affectedChs) {{
    if (allSel) selectedChannels.delete(ch);
    else selectedChannels.add(ch);
  }}
  redraw();
}});

let reviewed = new Set();

function autoSave() {{
  const c = CASES[idx];
  const selRegions = getSelectedRegions();
  allLabels[c.patient_id] = {{
    segment_id: c.segment_id,
    subtype: c.subtype,
    predicted_regions: c.predicted_regions,
    predicted_scores: c.region_scores,
    selected_channels: [...selectedChannels],
    selected_regions: selRegions,
    verbal_description: generateVerbalDescription(),
    rejected: reviewed.has(c.patient_id) && selectedChannels.size === 0,
    source: 'spatial_labeler',
  }};
  if (reviewed.has(c.patient_id)) labeled.add(c.patient_id);
  saveAll();
}}

function show() {{
  if (CASES.length === 0) return;
  idx = Math.max(0, Math.min(idx, CASES.length - 1));
  const c = CASES[idx];

  // Load from storage or initialize from predictions
  if (allLabels[c.patient_id] && allLabels[c.patient_id].selected_channels) {{
    selectedChannels = new Set(allLabels[c.patient_id].selected_channels);
    labeled.add(c.patient_id);
  }} else {{
    // Initialize from predicted regions → channels
    selectedChannels = new Set();
    for (const region of c.predicted_regions) {{
      const chs = REGION_TO_CHANNELS[region];
      if (chs) for (const ch of chs) selectedChannels.add(ch);
    }}
  }}

  // Set threshold slider to match current state
  document.getElementById('threshold-slider').value = Math.round(currentThreshold * 100);
  document.getElementById('threshold-value').textContent = currentThreshold.toFixed(2);

  redraw();
}}

function skipCase() {{
  const c = CASES[idx];
  selectedChannels = new Set();
  reviewed.add(c.patient_id);
  allLabels[c.patient_id] = {{
    segment_id: c.segment_id,
    subtype: c.subtype,
    predicted_regions: c.predicted_regions,
    predicted_scores: c.region_scores,
    selected_channels: [],
    selected_regions: [],
    rejected: true,
    source: 'spatial_labeler',
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

function exportJSON() {{
  autoSave();
  const out = {{}};
  for (const c of CASES) {{
    const pid = c.patient_id;
    if (allLabels[pid]) {{
      out[pid] = {{
        patient_id: pid,
        segment_id: allLabels[pid].segment_id || c.segment_id,
        subtype: c.subtype,
        predicted_regions: c.predicted_regions,
        predicted_scores: c.region_scores,
        selected_regions: allLabels[pid].selected_regions,
        rejected: allLabels[pid].rejected === true,
        source: 'spatial_labeler',
      }};
    }}
  }}
  const blob = new Blob([JSON.stringify(out, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = '{subtype}_spatial_batch{batch_num}_results.json';
  a.click();
  const el = document.getElementById('save-status');
  el.textContent = 'Exported ' + Object.keys(out).length + ' cases';
  el.style.color = '#4f4';
  setTimeout(() => {{ el.textContent = ''; }}, 3000);
}}

// Keyboard
document.addEventListener('keydown', (e) => {{
  // 1-8: toggle regions
  const regionKey = parseInt(e.key);
  if (regionKey >= 1 && regionKey <= 8) {{
    toggleRegion(regionKey - 1);
    return;
  }}

  if (e.key === 'Enter') {{
    e.preventDefault();
    reviewed.add(CASES[idx].patient_id);
    autoSave();
    const el = document.getElementById('save-status');
    el.textContent = 'Saved';
    el.style.color = '#44cc44';
    setTimeout(() => {{ el.textContent = ''; }}, 800);
    idx = Math.min(CASES.length - 1, idx + 1);
    show();
  }} else if (e.key === 'ArrowLeft') {{
    e.preventDefault();
    autoSave();
    idx = Math.max(0, idx - 1);
    show();
  }} else if (e.key === 'ArrowRight') {{
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
