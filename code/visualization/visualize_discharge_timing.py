"""
Visualize discharge timing detection overlaid on EEG.

Generates JPEG images of EEG with detected discharge times marked as circles,
then builds a self-contained HTML viewer.

Usage:
    conda run -n foe python code/visualize_discharge_timing.py
"""

import sys
import json
import base64
import warnings
import traceback
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from scipy.signal import butter, filtfilt, detrend
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana
from discharge_timing import detect_discharge_times, estimate_frequency, detect_involved_channels

# Constants
FS = 200
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
RESULTS_DIR = PROJECT_DIR / 'results'


def load_segment_as_bipolar(mat_path, montage, n_channels):
    """Load .mat file and return (18, N) bipolar array."""
    mat = sio.loadmat(str(mat_path))
    data = mat['data'].astype(np.float64)
    if data.shape[0] > data.shape[1]:
        data = data.T
    if montage == 'monopolar' and n_channels == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    elif data.shape[0] == 20:
        data = np.array(fcn_getBanana(data)).astype(np.float64)
    return data


def draw_discharge_figure(seg_bi, result, subtype, patient_id, freq_estimate):
    """Draw EEG with discharge timing circles overlaid on involved channels."""
    n_channels = min(seg_bi.shape[0], 18)
    n_samples = seg_bi.shape[1]
    duration = n_samples / FS
    time_vec = np.linspace(0, duration, n_samples)

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 8),
                             gridspec_kw={'hspace': 0.08})
    if n_channels == 1:
        axes = [axes]

    # 20 Hz lowpass for display
    b_lp, a_lp = butter(4, 20.0 / (FS / 2), btype='low')

    involved = set(result['involved_channels'])
    n_events = len(result['global_times'])

    fig.suptitle(
        f"{subtype.upper()} \u2014 Patient {patient_id} \u2014 "
        f"Est: {freq_estimate:.2f} Hz, Detected: {n_events} events",
        fontsize=10, fontweight='bold', y=0.98
    )

    for i in range(n_channels):
        ax = axes[i]

        # Clean display signal
        try:
            sig_lp = filtfilt(b_lp, a_lp, seg_bi[i, :])
        except ValueError:
            sig_lp = seg_bi[i, :]
        sig_clean = detrend(sig_lp, type='linear')

        # Color by hemisphere
        if i in LEFT_INDICES:
            color = '#cc3333'
            ax.set_facecolor('#fff0f0')
        elif i in RIGHT_INDICES:
            color = '#3333cc'
            ax.set_facecolor('#f0f0ff')
        else:
            color = '#333333'
            ax.set_facecolor('#f5f5f5')

        ax.plot(time_vec, sig_clean, color=color, linewidth=0.6, alpha=0.8)

        # Overlay discharge markers on involved channels
        if i in involved and i in result['channel_times']:
            ch_times = result['channel_times'][i]
            if len(ch_times) > 0:
                # Get y-values at discharge times for positioning
                ch_time_samples = np.clip(
                    (ch_times * FS).astype(int), 0, n_samples - 1)
                y_vals = sig_clean[ch_time_samples]
                ax.plot(ch_times, y_vals, 'o', color='red',
                        markersize=5, markerfacecolor='none',
                        markeredgewidth=1.5, zorder=5)

        # Channel label
        label = BIPOLAR_CHANNELS[i] if i < len(BIPOLAR_CHANNELS) else f'Ch{i}'
        ax.set_ylabel(label, fontsize=6, rotation=0, labelpad=50, va='center')
        ax.tick_params(axis='y', labelsize=4, length=2)

        if i < n_channels - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('Time (s)', fontsize=7)
            ax.tick_params(axis='x', labelsize=6)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.94, bottom=0.04)
    return fig


def build_html_viewer(cases, image_data_dict, output_path):
    """Build self-contained HTML viewer with inlined JPEG images."""
    manifest_records = []
    for c in cases:
        manifest_records.append({
            'patient_id': c['patient_id'],
            'subtype': c['subtype'],
            'freq_estimate': round(c['freq_estimate'], 2),
            'n_events': c['n_events'],
            'case_id': c['case_id'],
        })

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>Discharge Timing Viewer</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }}
  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px;
  }}
  #header-left {{ display: flex; align-items: center; gap: 12px; }}
  #header-right {{ display: flex; align-items: center; gap: 12px; font-size: 13px; }}
  select {{ font-size: 13px; padding: 3px 6px; background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}
  #info-panel {{
    background: #2a2a2a; padding: 10px 16px; display: flex; align-items: center;
    gap: 16px; flex-wrap: wrap; border-bottom: 1px solid #333;
  }}
  .info-badge {{
    padding: 4px 12px; border-radius: 4px; font-size: 13px; font-weight: bold;
  }}
  .badge-lpd {{ background: #5a2020; color: #ff8888; }}
  .badge-gpd {{ background: #20205a; color: #8888ff; }}
  .info-item {{ font-size: 13px; color: #bbb; }}
  .info-item strong {{ color: #eee; }}
  #img-container {{ text-align: center; padding: 8px; }}
  #img-container img {{ max-width: 100%; max-height: calc(100vh - 140px); }}
  #shortcuts {{
    font-size: 11px; color: #777; padding: 4px 16px; background: #222;
    border-top: 1px solid #333;
  }}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <select id="filter-type" onchange="filterChanged()">
      <option value="all">All types</option>
      <option value="lpd">LPD only</option>
      <option value="gpd">GPD only</option>
    </select>
    <span id="counter">1 / 0</span>
  </div>
  <div id="header-right">
    <span>Discharge Timing Viewer</span>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge" id="type-badge">--</span>
  <span class="info-item">Patient: <strong id="patient-id">--</strong></span>
  <span class="info-item">Freq: <strong id="freq-est">--</strong> Hz</span>
  <span class="info-item">Events: <strong id="n-events">--</strong></span>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate &nbsp;&nbsp;
  <span class="key">G</span> GPD only &nbsp;&nbsp;
  <span class="key">L</span> LPD only &nbsp;&nbsp;
  <span class="key">A</span> all types
</div>

<script>
const MANIFEST = {json.dumps(manifest_records)};
const IMAGE_DATA = {json.dumps(image_data_dict)};

let filteredItems = [];
let idx = 0;

function init() {{
  filteredItems = MANIFEST.slice();
  idx = 0;
  show();
}}

function filterChanged() {{
  const typeFilter = document.getElementById('filter-type').value;
  filteredItems = MANIFEST.filter(m => {{
    if (typeFilter !== 'all' && m.subtype !== typeFilter) return false;
    return true;
  }});
  idx = 0;
  show();
}}

function show() {{
  if (filteredItems.length === 0) {{
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }}
  idx = Math.max(0, Math.min(idx, filteredItems.length - 1));
  const item = filteredItems[idx];

  const b64 = IMAGE_DATA[item.case_id];
  if (b64) {{
    document.getElementById('viewer').src = 'data:image/jpeg;base64,' + b64;
  }} else {{
    document.getElementById('viewer').src = '';
  }}

  const badge = document.getElementById('type-badge');
  badge.textContent = item.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + item.subtype;
  document.getElementById('patient-id').textContent = item.patient_id;
  document.getElementById('freq-est').textContent = item.freq_estimate.toFixed(2);
  document.getElementById('n-events').textContent = item.n_events;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + filteredItems.length;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, filteredItems.length - 1); show(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); }}
  else if (e.key === 'g' || e.key === 'G') {{
    document.getElementById('filter-type').value = 'gpd'; filterChanged();
  }}
  else if (e.key === 'l' || e.key === 'L') {{
    document.getElementById('filter-type').value = 'lpd'; filterChanged();
  }}
  else if (e.key === 'a' || e.key === 'A') {{
    document.getElementById('filter-type').value = 'all'; filterChanged();
  }}
}});

init();
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved viewer: {output_path} ({size_mb:.1f} MB)")


def main():
    print("=" * 60)
    print("Discharge Timing Detection + Viewer")
    print("=" * 60)

    # Load data
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)
    df_patients = df_patients[df_patients['excluded'] == False].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'].notna()].copy()
    df_patients = df_patients[df_patients['gold_standard_freq'] > 0].copy()

    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    print(f"  {len(df_patients)} patients with gold standard labels")

    # Sort by patient_id, get first 50 LPD + first 50 GPD
    df_patients = df_patients.sort_values('patient_id')

    lpd_patients = df_patients[df_patients['subtype'] == 'lpd'].head(50)
    gpd_patients = df_patients[df_patients['subtype'] == 'gpd'].head(50)
    selected = pd.concat([lpd_patients, gpd_patients]).sort_values('patient_id')

    print(f"  Selected: {len(lpd_patients)} LPD + {len(gpd_patients)} GPD = {len(selected)} cases")

    # Process each case
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cases = []
    image_data = {}
    n_success = 0
    n_fail = 0
    all_n_events = []

    for i, (_, pat_row) in enumerate(selected.iterrows()):
        pid = str(pat_row['patient_id'])
        subtype = pat_row['subtype']
        gold_freq = float(pat_row['gold_standard_freq'])

        # Get first segment for this patient
        pat_segs = df_segments[df_segments['patient_id'] == pid]
        if len(pat_segs) == 0:
            print(f"  [{i+1}/{len(selected)}] {pid} ({subtype}) — NO SEGMENTS")
            n_fail += 1
            continue

        seg_row = pat_segs.iloc[0]
        mat_path = EEG_DIR / seg_row['mat_file']
        if not mat_path.exists():
            print(f"  [{i+1}/{len(selected)}] {pid} ({subtype}) — FILE MISSING")
            n_fail += 1
            continue

        try:
            seg = load_segment_as_bipolar(
                mat_path, seg_row['montage'], seg_row['n_channels'])

            # Ensure 18 channels
            if seg.shape[0] < 18:
                print(f"  [{i+1}/{len(selected)}] {pid} ({subtype}) — BAD SHAPE {seg.shape}")
                n_fail += 1
                continue
            if seg.shape[0] > 18:
                seg = seg[:18, :]

            # Step 1: Estimate frequency using pre-trained Ridge model
            freq_est = estimate_frequency(seg, FS, subtype=subtype)

            # Step 2: Detect involved channels using VE method (subtype-aware)
            involved, ve_per_ch = detect_involved_channels(seg, FS, freq_est,
                                                            subtype=subtype)

            # Step 3: Detect discharge times (frequency + channels as priors)
            result = detect_discharge_times(seg, FS, freq_est, subtype,
                                            involved_channels=involved)

            n_events = len(result['global_times'])
            all_n_events.append(n_events)

            # Generate figure
            fig = draw_discharge_figure(seg, result, subtype, pid, freq_est)

            # Save to buffer as JPEG
            buf = BytesIO()
            fig.savefig(buf, format='jpeg', dpi=100, pil_kwargs={'quality': 70})
            plt.close(fig)
            buf.seek(0)

            case_id = f"{pid}_{subtype}"
            image_data[case_id] = base64.b64encode(buf.read()).decode('ascii')

            cases.append({
                'patient_id': pid,
                'subtype': subtype,
                'freq_estimate': freq_est,
                'n_events': n_events,
                'case_id': case_id,
            })

            n_success += 1
            print(f"  [{i+1}/{len(selected)}] {pid} ({subtype}) — "
                  f"freq={freq_est:.2f} Hz, {n_events} events")

        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {pid} ({subtype}) — FAILED: {e}")
            traceback.print_exc()
            n_fail += 1

    # Build viewer
    print(f"\n--- Building HTML viewer ---")
    viewer_path = RESULTS_DIR / 'discharge_timing_viewer.html'
    build_html_viewer(cases, image_data, viewer_path)

    # Report
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"  Cases processed: {n_success}")
    print(f"  Failures: {n_fail}")
    if all_n_events:
        print(f"  Average events per case: {np.mean(all_n_events):.1f}")
        print(f"  Median events per case: {np.median(all_n_events):.1f}")
        print(f"  Min/Max events: {np.min(all_n_events)} / {np.max(all_n_events)}")
    print(f"  Viewer: {viewer_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
