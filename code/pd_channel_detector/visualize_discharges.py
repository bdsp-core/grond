"""
Visualize frame-level discharge detections on EEG plots.

For each patient:
  1. Load segment(s), run each channel through DischargeDetector
  2. Peak-pick discharge times per channel
  3. Generate EEG plot (morgoth style) with overlays:
     - Standard black EEG traces
     - RED DOTS at detected discharge times per channel
     - Thin vertical DASHED RED LINES connecting simultaneous discharges
       across channels (within +/-25ms tolerance)
     - HEAT STRIP below each channel: discharge probability gradient (white->red)
     - Summary text: estimated frequency, IPI mean/CV, n_discharges

Build HTML viewer at results/discharge_timing_viewer.html with ~50 example
patients, arrow key navigation.

Run: conda run -n foe_dl python code/pd_channel_detector/visualize_discharges.py
"""

import sys
import io
import json
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks, detrend, butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.discharge_detector import DischargeDetector
from optimization_harness_v2 import load_dataset

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
RESULTS_DIR = PROJECT_DIR / 'results'
DEVICE = torch.device('cpu')

FS = 200

BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

GROUP_BREAKS = {4, 8, 12, 16}


def load_ensemble_models():
    """Load all 5 fold models, or whichever exist."""
    models = []
    for fold in range(5):
        path = CACHE_DIR / f'discharge_fold{fold}.pt'
        if path.exists():
            model = DischargeDetector().to(DEVICE)
            model.load_state_dict(torch.load(str(path), map_location=DEVICE))
            model.eval()
            models.append(model)
    if not models:
        raise FileNotFoundError("No discharge detector models found. Run train_discharge_detector.py first.")
    return models


@torch.no_grad()
def run_detector_on_segment(models, segment):
    """Run discharge detector ensemble on all channels of a segment.

    Args:
        models: list of DischargeDetector models
        segment: (18, 2000) bipolar segment

    Returns:
        probs: (18, 2000) discharge probability per channel (ensemble avg)
        peaks_per_channel: list of 18 arrays, each containing peak indices
        freqs_per_channel: (18,) estimated frequency per channel
        ipis_per_channel: list of 18 arrays, each containing IPIs in seconds
    """
    n_channels = segment.shape[0]
    n_samples = segment.shape[1]
    all_probs = np.zeros((n_channels, n_samples), dtype=np.float32)

    for ch in range(n_channels):
        ch_data = segment[ch].astype(np.float32).copy()

        if not np.all(np.isfinite(ch_data)):
            continue

        mu = np.mean(ch_data)
        std = np.std(ch_data)
        if std > 1e-8:
            ch_data = (ch_data - mu) / std
        else:
            ch_data = ch_data - mu

        x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :])

        # Ensemble average
        prob_sum = np.zeros(n_samples, dtype=np.float32)
        for model in models:
            prob = model(x)
            prob_sum += prob[0, 0].numpy()
        all_probs[ch] = prob_sum / len(models)

    # Peak-pick each channel
    peaks_per_channel = []
    freqs_per_channel = np.full(n_channels, np.nan)
    ipis_per_channel = []

    for ch in range(n_channels):
        pks, _ = find_peaks(all_probs[ch], height=0.3, distance=40)
        peaks_per_channel.append(pks)

        if len(pks) >= 3:
            ipis = np.diff(pks) / FS
            ipis_per_channel.append(ipis)
            median_ipi = np.median(ipis)
            if median_ipi > 0:
                freqs_per_channel[ch] = 1.0 / median_ipi
            else:
                ipis_per_channel.append(np.array([]))
        else:
            ipis_per_channel.append(np.array([]))

    return all_probs, peaks_per_channel, freqs_per_channel, ipis_per_channel


def find_synchronous_times(peaks_per_channel, tolerance_samples=5):
    """Find times where multiple channels discharge simultaneously.

    Args:
        peaks_per_channel: list of 18 peak index arrays
        tolerance_samples: +/-tolerance in samples (5 = 25ms at 200Hz)

    Returns:
        sync_times: list of (time_sample, list_of_channel_indices)
    """
    # Collect all peaks with their channel
    all_peaks = []
    for ch, pks in enumerate(peaks_per_channel):
        for pk in pks:
            all_peaks.append((pk, ch))

    if not all_peaks:
        return []

    all_peaks.sort(key=lambda x: x[0])

    # Group peaks within tolerance
    sync_events = []
    used = set()

    for i, (t, ch) in enumerate(all_peaks):
        if i in used:
            continue

        group_channels = [ch]
        group_time = t
        used.add(i)

        for j in range(i + 1, len(all_peaks)):
            if j in used:
                continue
            t2, ch2 = all_peaks[j]
            if abs(t2 - group_time) <= tolerance_samples:
                if ch2 not in group_channels:
                    group_channels.append(ch2)
                    used.add(j)
            elif t2 - group_time > tolerance_samples:
                break

        if len(group_channels) >= 2:
            sync_events.append((int(group_time), sorted(group_channels)))

    return sync_events


def generate_discharge_eeg_image(segment, probs, peaks_per_channel,
                                  freqs_per_channel, ipis_per_channel,
                                  sync_events, patient_id, subtype,
                                  gold_freq, title_extra=''):
    """Generate EEG image with discharge overlays.

    Returns JPEG bytes.
    """
    seg_bi = segment.astype(np.float64).copy()
    if seg_bi.shape[0] > seg_bi.shape[1]:
        seg_bi = seg_bi.T
    seg_bi = np.nan_to_num(seg_bi, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels, n_samples = seg_bi.shape
    time_vec = np.linspace(0, n_samples / FS, n_samples)

    # Lowpass at 20 Hz
    nyq = FS / 2.0
    if nyq > 20:
        b, a = butter(4, 20.0 / nyq, btype='low')
        for i in range(n_channels):
            try:
                seg_bi[i, :] = filtfilt(b, a, seg_bi[i, :])
            except ValueError:
                pass

    # Detrend
    for i in range(n_channels):
        seg_bi[i, :] = detrend(seg_bi[i, :], type='linear')

    # Fixed scaling
    z_scale = 0.01
    clip_uv = 300.0

    # Build display list with spacers
    display_channels = []
    for i in range(n_channels):
        if i in GROUP_BREAKS:
            display_channels.append((None, ''))
        display_channels.append((i, BIPOLAR_CHANNELS[i]))
    n_display = len(display_channels)

    # Heat strip height (fraction of channel spacing)
    heat_height = 0.15

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # White-to-red colormap for heat strips
    cmap = LinearSegmentedColormap.from_list('wr', ['white', '#ff4444'])

    yticks = []
    ytick_labels = []

    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        offset = float(n_display - di)
        yticks.append(offset)
        ytick_labels.append(ch_name)

        if ch_idx is None:
            continue

        # Draw heat strip below the trace
        prob_signal = probs[ch_idx]
        # Create a 2D image (1 row x n_samples cols) for the heat strip
        heat_bottom = offset - 0.5 - heat_height
        heat_top = offset - 0.5
        ax.imshow(prob_signal[np.newaxis, :],
                  extent=[0, n_samples / FS, heat_bottom, heat_top],
                  aspect='auto', cmap=cmap, vmin=0, vmax=1,
                  interpolation='bilinear', alpha=0.7)

        # Draw EEG trace
        clipped = np.clip(seg_bi[ch_idx, :], -clip_uv, clip_uv)
        scaled = z_scale * clipped + offset
        ax.plot(time_vec, scaled, color='black', linewidth=0.6, clip_on=True)

        # Draw RED DOTS at detected peaks
        pks = peaks_per_channel[ch_idx]
        if len(pks) > 0:
            pk_times = pks / FS
            pk_vals = z_scale * np.clip(seg_bi[ch_idx, pks], -clip_uv, clip_uv) + offset
            ax.scatter(pk_times, pk_vals, color='red', s=18, zorder=5,
                       edgecolors='darkred', linewidths=0.5, alpha=0.8)

    # Draw synchronous discharge lines
    ch_to_offset = {}
    for di in range(n_display):
        ch_idx, ch_name = display_channels[di]
        if ch_idx is not None:
            ch_to_offset[ch_idx] = float(n_display - di)

    for sync_time, sync_channels in sync_events:
        if len(sync_channels) < 2:
            continue
        t = sync_time / FS
        offsets = [ch_to_offset[ch] for ch in sync_channels if ch in ch_to_offset]
        if len(offsets) >= 2:
            y_min = min(offsets) - 0.3
            y_max = max(offsets) + 0.3
            ax.plot([t, t], [y_min, y_max], color='red', linewidth=0.5,
                    linestyle='--', alpha=0.4, zorder=2)

    # Y-axis
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels, fontsize=7.5, fontfamily='monospace')
    ax.tick_params(axis='y', length=0, pad=4)
    ax.set_ylim(-0.5, n_display + 1.5)

    # X-axis
    ax.set_xlim(0, n_samples / FS)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xlabel('Time (seconds)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)

    # Grid
    ax.grid(True, axis='x', alpha=0.25, linewidth=0.5, linestyle='--')
    ax.grid(False, axis='y')

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['left'].set_color('#999')
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['bottom'].set_color('#999')

    # Summary statistics
    valid_freqs = freqs_per_channel[np.isfinite(freqs_per_channel)]
    est_freq = float(np.median(valid_freqs)) if len(valid_freqs) > 0 else float('nan')
    total_peaks = sum(len(p) for p in peaks_per_channel)

    all_ipis = np.concatenate([ip for ip in ipis_per_channel if len(ip) > 0])
    ipi_mean = float(np.mean(all_ipis)) if len(all_ipis) > 0 else float('nan')
    ipi_cv = float(np.std(all_ipis) / np.mean(all_ipis)) if len(all_ipis) > 0 and np.mean(all_ipis) > 0 else float('nan')

    n_sync = len(sync_events)

    # Title
    title = f'{patient_id}  [{subtype.upper()}]'
    if title_extra:
        title += f'  {title_extra}'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.99)

    # Summary text box
    freq_str = f'{est_freq:.2f}' if np.isfinite(est_freq) else 'N/A'
    gold_str = f'{gold_freq:.2f}' if np.isfinite(gold_freq) else 'N/A'
    ipi_mean_str = f'{ipi_mean:.3f}' if np.isfinite(ipi_mean) else 'N/A'
    ipi_cv_str = f'{ipi_cv:.2f}' if np.isfinite(ipi_cv) else 'N/A'

    summary = (f'Est freq: {freq_str} Hz  |  Gold: {gold_str} Hz  |  '
               f'Discharges: {total_peaks}  |  Sync events: {n_sync}  |  '
               f'IPI mean: {ipi_mean_str}s  |  IPI CV: {ipi_cv_str}')

    fig.text(0.5, 0.01, summary, ha='center', fontsize=9,
             fontfamily='monospace', color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                       edgecolor='#ccc', alpha=0.9))

    fig.subplots_adjust(left=0.065, right=0.99, top=0.96, bottom=0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format='jpg', dpi=110, pil_kwargs={'quality': 75})
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_discharge_viewer_html(cases, image_data):
    """Build the discharge timing viewer HTML.

    Args:
        cases: list of dicts with patient info and metrics
        image_data: dict patient_id -> base64 JPEG string
    """
    cases_json = json.dumps(cases)
    images_json = json.dumps(image_data)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discharge Timing Viewer</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: 'Consolas', 'Monaco', monospace; }}

  #header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 20px; background: #222; border-bottom: 2px solid #cc3333;
  }}
  #header h1 {{ margin: 0; font-size: 18px; color: #ff6644; }}
  #counter {{ font-size: 14px; color: #aaa; }}

  #info-panel {{
    background: #2a2a2a; padding: 12px 20px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
    border-bottom: 1px solid #333;
  }}
  .info-badge {{
    padding: 6px 16px; border-radius: 6px; font-size: 14px; font-weight: bold;
  }}
  .badge-lpd {{ background: #5a2020; color: #ff8888; }}
  .badge-gpd {{ background: #20205a; color: #8888ff; }}
  .info-item {{ font-size: 13px; color: #bbb; }}
  .info-item strong {{ color: #eee; }}
  .info-item .val-good {{ color: #44cc88; }}
  .info-item .val-warn {{ color: #ffaa44; }}
  .info-item .val-bad {{ color: #ff4444; }}

  #img-container {{
    text-align: center; padding: 8px; overflow: auto;
  }}
  #img-container img {{
    max-width: 100%; max-height: calc(100vh - 180px);
  }}

  #shortcuts {{
    font-size: 12px; color: #777; padding: 8px 20px; background: #222;
    border-top: 1px solid #333; text-align: center;
  }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 11px; }}

  select {{
    font-size: 13px; padding: 4px 8px; background: #333; color: #eee;
    border: 1px solid #555; border-radius: 4px;
  }}
</style>
</head>
<body>

<div id="header">
  <h1>Discharge Timing Viewer</h1>
  <div style="display:flex; align-items:center; gap:12px;">
    <label>Sort:
      <select id="sort-mode" onchange="sortChanged()">
        <option value="gold_freq">Gold Frequency</option>
        <option value="est_freq">Estimated Frequency</option>
        <option value="error">Frequency Error</option>
        <option value="n_discharges">N Discharges</option>
        <option value="ipi_cv">IPI CV</option>
      </select>
    </label>
    <label>Filter:
      <select id="filter-subtype" onchange="sortChanged()">
        <option value="all">All</option>
        <option value="lpd">LPD only</option>
        <option value="gpd">GPD only</option>
      </select>
    </label>
    <span id="counter">1 / 0</span>
  </div>
</div>

<div id="info-panel">
  <span class="info-badge" id="subtype-badge">--</span>
  <span class="info-item">Patient: <strong id="pid">--</strong></span>
  <span class="info-item">Gold freq: <strong id="gold-freq">--</strong></span>
  <span class="info-item">Est freq: <strong id="est-freq">--</strong></span>
  <span class="info-item">N discharges: <strong id="n-discharges">--</strong></span>
  <span class="info-item">IPI mean: <strong id="ipi-mean">--</strong></span>
  <span class="info-item">IPI CV: <strong id="ipi-cv">--</strong></span>
  <span class="info-item">N sync: <strong id="n-sync">--</strong></span>
</div>

<div id="img-container">
  <img id="viewer" src="" alt="Loading..." />
</div>

<div id="shortcuts">
  <span class="key">&larr;</span> / <span class="key">&rarr;</span> navigate
  &nbsp;&nbsp;
  <span class="key">Home</span> first &nbsp;
  <span class="key">End</span> last
</div>

<script>
const ALL_CASES = {cases_json};
const IMAGE_DATA = {images_json};

let displayList = [...ALL_CASES];
let idx = 0;

function sortChanged() {{
  const sortMode = document.getElementById('sort-mode').value;
  const filterSubtype = document.getElementById('filter-subtype').value;

  displayList = ALL_CASES.filter(c => {{
    if (filterSubtype === 'all') return true;
    return c.subtype === filterSubtype;
  }});

  displayList.sort((a, b) => {{
    if (sortMode === 'gold_freq') return a.gold_freq - b.gold_freq;
    if (sortMode === 'est_freq') return (a.est_freq || 0) - (b.est_freq || 0);
    if (sortMode === 'error') return Math.abs(b.freq_error || 0) - Math.abs(a.freq_error || 0);
    if (sortMode === 'n_discharges') return (b.n_discharges || 0) - (a.n_discharges || 0);
    if (sortMode === 'ipi_cv') return (b.ipi_cv || 0) - (a.ipi_cv || 0);
    return 0;
  }});

  idx = 0;
  show();
}}

function show() {{
  if (displayList.length === 0) {{
    document.getElementById('viewer').src = '';
    document.getElementById('counter').textContent = '0 / 0';
    return;
  }}
  idx = Math.max(0, Math.min(idx, displayList.length - 1));
  const c = displayList[idx];

  // Image
  const b64 = IMAGE_DATA[c.patient_id];
  document.getElementById('viewer').src = b64 ? 'data:image/jpeg;base64,' + b64 : '';

  // Info
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + displayList.length;
  document.getElementById('pid').textContent = c.patient_id;

  const badge = document.getElementById('subtype-badge');
  badge.textContent = c.subtype.toUpperCase();
  badge.className = 'info-badge badge-' + c.subtype;

  document.getElementById('gold-freq').textContent = c.gold_freq.toFixed(2) + ' Hz';

  const estEl = document.getElementById('est-freq');
  if (c.est_freq != null) {{
    const err = Math.abs(c.freq_error || 0);
    const cls = err < 0.3 ? 'val-good' : err < 0.7 ? 'val-warn' : 'val-bad';
    estEl.innerHTML = '<span class="' + cls + '">' + c.est_freq.toFixed(2) + ' Hz</span>';
  }} else {{
    estEl.textContent = 'N/A';
  }}

  document.getElementById('n-discharges').textContent = c.n_discharges || 0;
  document.getElementById('ipi-mean').textContent = c.ipi_mean != null ? c.ipi_mean.toFixed(3) + 's' : 'N/A';
  document.getElementById('ipi-cv').textContent = c.ipi_cv != null ? c.ipi_cv.toFixed(2) : 'N/A';
  document.getElementById('n-sync').textContent = c.n_sync || 0;
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ idx++; show(); e.preventDefault(); }}
  else if (e.key === 'ArrowLeft') {{ idx--; show(); e.preventDefault(); }}
  else if (e.key === 'Home') {{ idx = 0; show(); e.preventDefault(); }}
  else if (e.key === 'End') {{ idx = displayList.length - 1; show(); e.preventDefault(); }}
}});

// Init
sortChanged();
</script>
</body>
</html>'''

    return html


def main():
    print("=" * 70)
    print("Phase 4b: Discharge Timing Visualization")
    print("=" * 70)

    # Load models
    print("\nLoading discharge detector models...")
    models = load_ensemble_models()
    print(f"  Loaded {len(models)} models")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(verbose=False)
    df = dataset['df']
    segments = dataset['segments']

    # Select ~50 example patients sorted by gold frequency
    # Mix of LPD and GPD
    lpd_pats = df[df['subtype'] == 'lpd'].sort_values('gold_standard_freq')
    gpd_pats = df[df['subtype'] == 'gpd'].sort_values('gold_standard_freq')

    # Take up to 25 from each, evenly spaced
    def sample_evenly(sub_df, n=25):
        if len(sub_df) <= n:
            return sub_df
        indices = np.linspace(0, len(sub_df) - 1, n, dtype=int)
        return sub_df.iloc[indices]

    selected = []
    for sub_df in [sample_evenly(lpd_pats, 25), sample_evenly(gpd_pats, 25)]:
        for _, row in sub_df.iterrows():
            pid = row['patient_id']
            if segments.get(pid) and len(segments[pid]) > 0:
                selected.append(row)

    print(f"\nSelected {len(selected)} patients for visualization")

    # Process each patient
    cases = []
    image_data = {}
    n_done = 0

    for row in selected:
        pid = str(row['patient_id'])
        subtype = row['subtype']
        gold_freq = row['gold_standard_freq']

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            continue

        seg = pat_segs[0]  # Use first segment

        try:
            # Run detector
            probs, peaks_per_ch, freqs_per_ch, ipis_per_ch = run_detector_on_segment(models, seg)

            # Find synchronous events
            sync_events = find_synchronous_times(peaks_per_ch, tolerance_samples=5)

            # Compute summary stats
            valid_freqs = freqs_per_ch[np.isfinite(freqs_per_ch)]
            est_freq = float(np.median(valid_freqs)) if len(valid_freqs) > 0 else None
            total_peaks = sum(len(p) for p in peaks_per_ch)

            all_ipis = np.concatenate([ip for ip in ipis_per_ch if len(ip) > 0])
            ipi_mean = float(np.mean(all_ipis)) if len(all_ipis) > 0 else None
            ipi_cv_val = float(np.std(all_ipis) / np.mean(all_ipis)) if len(all_ipis) > 0 and np.mean(all_ipis) > 0 else None

            freq_error = (est_freq - gold_freq) if est_freq is not None else None

            # Generate image
            jpeg_bytes = generate_discharge_eeg_image(
                seg, probs, peaks_per_ch, freqs_per_ch, ipis_per_ch,
                sync_events, pid, subtype, gold_freq
            )
            image_data[pid] = base64.b64encode(jpeg_bytes).decode('ascii')

            cases.append({
                'patient_id': pid,
                'subtype': subtype,
                'gold_freq': round(gold_freq, 3),
                'est_freq': round(est_freq, 3) if est_freq is not None else None,
                'freq_error': round(freq_error, 3) if freq_error is not None else None,
                'n_discharges': total_peaks,
                'n_sync': len(sync_events),
                'ipi_mean': round(ipi_mean, 4) if ipi_mean is not None else None,
                'ipi_cv': round(ipi_cv_val, 3) if ipi_cv_val is not None else None,
            })

            n_done += 1
            if n_done % 10 == 0:
                print(f"  Processed {n_done}/{len(selected)} patients...")

        except Exception as e:
            print(f"  FAILED: {pid}: {e}")

    print(f"\nGenerated images for {len(cases)} patients")

    # Build HTML viewer
    print("\nBuilding HTML viewer...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    html = build_discharge_viewer_html(cases, image_data)
    output_path = RESULTS_DIR / 'discharge_timing_viewer.html'
    with open(str(output_path), 'w') as f:
        f.write(html)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nViewer saved to: {output_path} ({size_mb:.1f} MB)")
    print(f"  Cases: {len(cases)}")
    n_lpd = sum(1 for c in cases if c['subtype'] == 'lpd')
    n_gpd = sum(1 for c in cases if c['subtype'] == 'gpd')
    print(f"  LPD: {n_lpd}, GPD: {n_gpd}")

    # Summary stats
    est_freqs = [c['est_freq'] for c in cases if c['est_freq'] is not None]
    gold_freqs = [c['gold_freq'] for c in cases if c['est_freq'] is not None]
    if len(est_freqs) >= 5:
        rho, _ = spearmanr(gold_freqs, est_freqs)
        print(f"  Viewer cohort Spearman: {rho:.4f}")

    print("=" * 70)


if __name__ == '__main__':
    from scipy.stats import spearmanr
    main()
