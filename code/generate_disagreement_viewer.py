"""
Generate images and an HTML viewer for annotated EEG segments where
the B+pnorm algorithm's frequency estimate disagrees with expert
consensus by more than 0.5 Hz.

Outputs to results/disagreement_cases/:
  - EEG and pointiness PNG images for each disagreement case
  - metadata.json / metadata.js with algorithm and expert frequencies
  - viewer.html for interactive browsing and rating
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from mne.filter import notch_filter, filter_data
import hdf5storage
import warnings

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))
sys.path.insert(0, str(CODE_DIR / 'rda_detector'))

from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf
from browse_results import (
    BIPOLAR_CHANNELS, MONO_CHANNELS, LEFT_INDICES, RIGHT_INDICES,
    MIDLINE_INDICES, REGION_META, REGION_ORDER, REGION_BARE,
    LEFT_REGIONS, RIGHT_REGIONS, REGION_ACTIVE_THRESHOLD,
    generate_verbal_description, get_bipolar, detect_pd_peaks,
    compute_pointiness_trace,
)


def run_detector(segment, fs, pattern_type):
    """Run the PD detector and return a results dict (PD types only)."""
    result = pd_detect_alternate(segment, fs, pk_detect='apd')
    score_key = 'channel_pd_scores'
    row = {
        'files': 'temp',
        'type_event': result.get('type_event', np.nan),
        'event_frequency': result.get('event_frequency', np.nan),
        'acf_frequency': result.get('acf_frequency', np.nan),
        'spatial_extent': result.get('spatial_extent', np.nan),
        'laterality_index': result.get('laterality_index', np.nan),
        'left_mean_score': result.get('left_mean_score', np.nan),
        'right_mean_score': result.get('right_mean_score', np.nan),
    }
    ch_scores = result.get(score_key, {})
    ch_freqs = result.get('channel_frequencies', {})
    for ch in BIPOLAR_CHANNELS:
        row[f'score_{ch}'] = ch_scores.get(ch, np.nan)
        row[f'freq_{ch}'] = ch_freqs.get(ch, np.nan)
    for reg, score in result.get('region_scores', {}).items():
        row[f'region_{reg}'] = score
    return row


# Lazy import of draw functions to avoid fooof dependency at module level
_draw_funcs_loaded = False
draw_figure = None
draw_pointiness_figure = None

def _load_draw_functions():
    global _draw_funcs_loaded, draw_figure, draw_pointiness_figure
    if _draw_funcs_loaded:
        return
    # Mock fooof and rda1b_fft to avoid the dependency
    import types
    if 'fooof' not in sys.modules:
        fake_fooof = types.ModuleType('fooof')
        fake_fooof.FOOOF = type('FOOOF', (), {})
        sys.modules['fooof'] = fake_fooof
    from generate_test_images import draw_figure as _df, draw_pointiness_figure as _dpf
    draw_figure = _df
    draw_pointiness_figure = _dpf
    _draw_funcs_loaded = True

DATA_DIR = CODE_DIR.parent / 'data' / 'dataset_eeg'
ANN_DIR = CODE_DIR.parent / 'data' / 'annotations'
OUTPUT_DIR = CODE_DIR.parent / 'results' / 'disagreement_cases'


def load_per_expert_annotations():
    """Load annotations keeping individual expert ratings separate."""
    records = {}
    for pattern, subdir in [('LPDS', 'lpd'), ('GPDS', 'gpd')]:
        for expert_file in sorted(ANN_DIR.glob(f'{pattern}_*')):
            expert = expert_file.stem.split('_')[1]
            df = pd.read_csv(expert_file)
            for _, row in df.iterrows():
                mat_name = Path(row['files']).stem.replace('_score', '') + '.mat'
                key = (subdir, mat_name)
                if key not in records:
                    records[key] = {}
                freq = row['frequency']
                try:
                    freq = float(freq)
                except (ValueError, TypeError):
                    freq = np.nan
                records[key][expert] = {'frequency': freq}
    return records


def run_all_algorithms(data, fs):
    """Run Methods A, B, and B+pnorm. Return dict of frequencies."""
    # Method A
    rA = pd_detect_alternate(data, fs, pk_detect='apd')

    # Method B
    rB = pd_detect_pointiness_acf(
        data, fs, method='pointiness',
        lowpass_hz=15, smoothing_sigma=0.02,
        acf_min_lag=0.4, acf_peak_threshold=0.20,
        peak_height_frac=0.3, sync_threshold=0.8)

    # Method B+pnorm
    rBpn = pd_detect_pointiness_acf(
        data, fs, method='pointiness',
        lowpass_hz=15, smoothing_sigma=0.02,
        acf_min_lag=0.4, acf_peak_threshold=0.20,
        peak_height_frac=0.3, sync_threshold=0.8,
        percentile_norm=True, percentile_window_s=10.0,
        percentile_val=90)

    return {
        'A_freq': rA.get('event_frequency', np.nan),
        'B_freq': rB.get('event_frequency', np.nan),
        'Bpn_freq': rBpn.get('event_frequency', np.nan),
    }


def generate_viewer_html(cases):
    """Generate the HTML viewer file."""
    base_images_js = json.dumps([c['base_name'] for c in cases])

    html = f'''<!DOCTYPE html>
<html>
<head>
<title>B+pnorm Disagreement Viewer</title>
<style>
  body {{ margin: 0; background: #1a1a1a; color: #eee; font-family: monospace; }}
  #header {{ display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: #222; flex-wrap: wrap; gap: 8px; }}
  #info {{ font-size: 14px; }}
  #counter {{ color: #aaa; }}
  #view-label {{ font-weight: bold; }}
  #freq-panel {{ background: #333; padding: 8px 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .freq-btn {{ padding: 6px 14px; border: 2px solid #555; border-radius: 6px; background: #444; color: #eee;
         cursor: pointer; font-family: monospace; font-size: 14px; font-weight: bold; min-width: 160px; text-align: center; }}
  .freq-btn:hover {{ background: #555; border-color: #888; }}
  .freq-btn.selected-1 {{ background: #8b2020; border-color: #e03030; box-shadow: 0 0 8px #e03030; }}
  .freq-btn.selected-2 {{ background: #885510; border-color: #ee7722; box-shadow: 0 0 8px #ee7722; }}
  .freq-btn.selected-3 {{ background: #2a5a8a; border-color: #4499dd; box-shadow: 0 0 8px #4499dd; }}
  .freq-expert-label {{ padding: 6px 14px; border: 2px solid #2a6e2a; border-radius: 6px; background: #2a3a2a;
         font-family: monospace; font-size: 14px; font-weight: bold; min-width: 130px; text-align: center; color: #44cc44; }}
  .freq-individual {{ padding: 4px 10px; border: 1px solid #444; border-radius: 4px; background: #2a2a2a;
         font-family: monospace; font-size: 12px; color: #aaa; text-align: center; }}
  .freq-label {{ font-size: 10px; color: #999; display: block; margin-bottom: 2px; }}
  .freq-hz {{ font-size: 15px; }}
  #img-container {{ text-align: center; padding: 8px; }}
  #img-container img {{ max-width: 100%; max-height: calc(100vh - 140px); }}
  .key {{ background: #444; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
  #save-status {{ color: #44cc44; font-size: 12px; }}
  .which-label {{ color: #aaa; font-size: 13px; }}
  .disagreement-badge {{ background: #cc3333; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .sep {{ border-left: 1px solid #555; height: 40px; margin: 0 4px; }}
</style>
</head>
<body>
<div id="header">
  <div>
    <span class="key">&larr;</span> <span class="key">&rarr;</span> nav &nbsp;
    <span class="key">&uarr;</span> <span class="key">&darr;</span> toggle EEG/pointiness &nbsp;
    <span class="key">1</span> A best &nbsp;
    <span class="key">2</span> B best &nbsp;
    <span class="key">3</span> B+pnorm best &nbsp;
    <span class="key">E</span> export
  </div>
  <div id="info">
    <span id="view-label">EEG</span> &nbsp;
    <span id="filename"></span> &nbsp; <span id="counter"></span> &nbsp;
    <span class="disagreement-badge" id="disagree-badge"></span>
  </div>
</div>
<div id="freq-panel">
  <span class="which-label">Which freq is best?</span>
  <button class="freq-btn" id="btn-1" onclick="rate('1')">
    <span class="freq-label">Method A (peak det) [1]</span>
    <span class="freq-hz" id="freq-a-val">&mdash;</span>
  </button>
  <button class="freq-btn" id="btn-2" onclick="rate('2')">
    <span class="freq-label">Method B (ACF) [2]</span>
    <span class="freq-hz" id="freq-b-val">&mdash;</span>
  </button>
  <button class="freq-btn" id="btn-3" onclick="rate('3')">
    <span class="freq-label">B+pnorm [3]</span>
    <span class="freq-hz" id="freq-bpn-val">&mdash;</span>
  </button>
  <div class="sep"></div>
  <div class="freq-expert-label">
    <span class="freq-label">Expert Consensus</span>
    <span class="freq-hz" id="freq-expert-val">?</span>
  </div>
  <div class="freq-individual">
    <span class="freq-label">LB</span>
    <span id="freq-lb">?</span>
  </div>
  <div class="freq-individual">
    <span class="freq-label">PH</span>
    <span id="freq-ph">?</span>
  </div>
  <div class="freq-individual">
    <span class="freq-label">SZ</span>
    <span id="freq-sz">?</span>
  </div>
  <span id="save-status"></span>
</div>
<div id="img-container">
  <img id="viewer" src="" />
</div>
<script>
const BASE_IMAGES = {base_images_js};

let metadata = {{}};
let ratings = {{}};
let items = BASE_IMAGES;
let idx = 0;
let showPointiness = false;

function loadMetadata() {{
  fetch('metadata.json?t=' + Date.now())
    .then(r => {{ if (!r.ok) throw new Error('fetch failed'); return r.json(); }})
    .then(d => {{ metadata = d; show(); }})
    .catch(() => {{
      const s = document.createElement('script');
      s.src = 'metadata.js?t=' + Date.now();
      document.head.appendChild(s);
    }});
}}
loadMetadata();

try {{ ratings = JSON.parse(localStorage.getItem('disagreement_ratings') || '{{}}'); }} catch(e) {{}}

function getSrc(base) {{
  return showPointiness ? base + '_pointiness.png' : base + '.png';
}}

function show() {{
  const name = items[idx];
  document.getElementById('viewer').src = getSrc(name);
  document.getElementById('filename').textContent = name;
  document.getElementById('counter').textContent = (idx + 1) + ' / ' + items.length;
  document.getElementById('view-label').textContent = showPointiness ? 'POINTINESS' : 'EEG';
  document.getElementById('view-label').style.color = showPointiness ? '#ee7722' : '#ffcc44';

  const meta = metadata[name] || {{}};
  document.getElementById('freq-a-val').textContent = meta.freq_a != null ? meta.freq_a + ' Hz' : '\\u2014';
  document.getElementById('freq-b-val').textContent = meta.freq_b != null ? meta.freq_b + ' Hz' : '\\u2014';
  document.getElementById('freq-bpn-val').textContent = meta.freq_bpn != null ? meta.freq_bpn + ' Hz' : '\\u2014';
  document.getElementById('freq-expert-val').textContent = meta.expert_freq != null ? meta.expert_freq + ' Hz' : '?';

  document.getElementById('freq-lb').textContent = meta.expert_LB != null ? meta.expert_LB + ' Hz' : '?';
  document.getElementById('freq-ph').textContent = meta.expert_PH != null ? meta.expert_PH + ' Hz' : '?';
  document.getElementById('freq-sz').textContent = meta.expert_SZ != null ? meta.expert_SZ + ' Hz' : '?';

  const disagree = meta.disagreement != null ? meta.disagreement.toFixed(2) : '?';
  document.getElementById('disagree-badge').textContent = '\\u0394 ' + disagree + ' Hz';

  // Show current rating
  const r = ratings[name];
  for (const k of ['1','2','3']) {{
    document.getElementById('btn-' + k).classList.remove('selected-' + k);
  }}
  if (r) document.getElementById('btn-' + r).classList.add('selected-' + r);
  document.getElementById('save-status').textContent = '';
}}

function rate(choice) {{
  const name = items[idx];
  ratings[name] = choice;
  localStorage.setItem('disagreement_ratings', JSON.stringify(ratings));

  for (const k of ['1','2','3']) {{
    document.getElementById('btn-' + k).classList.remove('selected-' + k);
  }}
  document.getElementById('btn-' + choice).classList.add('selected-' + choice);

  const n = Object.keys(ratings).length;
  document.getElementById('save-status').textContent = 'Saved (' + n + ' rated)';

  if (idx < items.length - 1) {{
    setTimeout(() => {{ idx++; show(); }}, 300);
  }}
}}

function exportRatings() {{
  const blob = new Blob([JSON.stringify(ratings, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'disagreement_ratings.json';
  a.click();
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') {{ idx = Math.min(idx + 1, items.length - 1); show(); }}
  else if (e.key === 'ArrowLeft') {{ idx = Math.max(idx - 1, 0); show(); }}
  else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {{ showPointiness = !showPointiness; show(); e.preventDefault(); }}
  else if (e.key === '1') rate('1');
  else if (e.key === '2') rate('2');
  else if (e.key === '3') rate('3');
  else if (e.key === 'e' || e.key === 'E') exportRatings();
}});

show();
</script>
</body>
</html>'''
    return html


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading expert annotations...')
    records = load_per_expert_annotations()
    print(f'  Loaded {len(records)} annotated segments')

    experts_list = ['LB', 'PH', 'SZ']

    # Process each segment: run algorithms and check for disagreement
    cases = []
    n_total = 0
    n_skipped = 0
    n_no_consensus = 0

    keys = sorted(records.keys())
    print(f'\nProcessing {len(keys)} segments...')

    for i, (subdir, mat_name) in enumerate(keys):
        mat_path = DATA_DIR / subdir / mat_name
        if not mat_path.exists():
            n_skipped += 1
            continue

        expert_data = records[(subdir, mat_name)]

        # Compute expert consensus (median of experts with freq > 0)
        expert_freqs = {}
        for e in experts_list:
            if e in expert_data:
                f = expert_data[e]['frequency']
                if np.isfinite(f) and f > 0:
                    expert_freqs[e] = f

        if len(expert_freqs) == 0:
            n_no_consensus += 1
            continue

        consensus_freq = np.median(list(expert_freqs.values()))
        n_total += 1

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(keys)}] processed, {len(cases)} disagreements so far...')

        # Load data
        try:
            mat = hdf5storage.loadmat(str(mat_path))
            data = mat.get('data')
            if data is None:
                n_skipped += 1
                continue
            if data.shape[0] > data.shape[1]:
                data = data.T
            fs = 200
        except Exception as e:
            n_skipped += 1
            continue

        # Run all algorithms
        try:
            algo_freqs = run_all_algorithms(data, fs)
        except Exception as e:
            print(f'  Algorithm error on {mat_name}: {e}')
            n_skipped += 1
            continue

        bpn_freq = algo_freqs['Bpn_freq']
        if not np.isfinite(bpn_freq):
            continue

        disagreement = abs(bpn_freq - consensus_freq)
        if disagreement <= 0.5:
            continue

        # This is a disagreement case
        cases.append({
            'subdir': subdir,
            'mat_name': mat_name,
            'mat_path': str(mat_path),
            'data': data,
            'fs': fs,
            'A_freq': algo_freqs['A_freq'],
            'B_freq': algo_freqs['B_freq'],
            'Bpn_freq': bpn_freq,
            'expert_freq': consensus_freq,
            'expert_individual': expert_freqs,
            'disagreement': disagreement,
        })

    print(f'\n  Total with consensus: {n_total}')
    print(f'  Skipped (no file / error): {n_skipped}')
    print(f'  No consensus (all experts 0): {n_no_consensus}')
    print(f'  Disagreement cases (>0.5 Hz): {len(cases)}')

    if len(cases) == 0:
        print('\nNo disagreement cases found. Exiting.')
        return

    # Sort by disagreement (largest first)
    cases.sort(key=lambda c: -c['disagreement'])

    # Load draw functions (lazy to avoid fooof dependency)
    _load_draw_functions()

    # Generate images and metadata
    metadata = {}
    print(f'\nGenerating images for {len(cases)} cases...')

    for ci, case in enumerate(cases):
        subdir = case['subdir']
        mat_name = case['mat_name']
        stem = Path(mat_name).stem
        base_name = f'{subdir}_{ci+1:03d}_{stem}'
        case['base_name'] = base_name

        data = case['data']
        fs = case['fs']
        pattern_type = subdir  # 'lpd' or 'gpd'

        print(f'  [{ci+1}/{len(cases)}] {base_name} (delta={case["disagreement"]:.2f} Hz)', end='', flush=True)

        try:
            # Filter for display
            seg_filtered = notch_filter(data.astype(float), fs, 60, n_jobs=1, verbose="ERROR")
            seg_filtered = filter_data(seg_filtered, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
            seg_bi = get_bipolar(seg_filtered)

            # Run detector for the figure (uses Method A results for display)
            result_row = run_detector(data, fs, pattern_type)
            result_row['files'] = base_name

            title_extra = (f'{stem}  |  '
                           f'Expert={case["expert_freq"]:.2f}  '
                           f'B+pn={case["Bpn_freq"]:.2f}  '
                           f'delta={case["disagreement"]:.2f}')

            # EEG figure
            fig = draw_figure(result_row, seg_bi, fs, pattern_type, title_extra=title_extra)
            fig.savefig(str(OUTPUT_DIR / f'{base_name}.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Pointiness figure
            fig2 = draw_pointiness_figure(result_row, seg_bi, fs, pattern_type, title_extra=title_extra)
            fig2.savefig(str(OUTPUT_DIR / f'{base_name}_pointiness.png'), dpi=150, bbox_inches='tight')
            plt.close(fig2)

            # Metadata entry
            meta_entry = {
                'freq_a': round(case['A_freq'], 2) if np.isfinite(case['A_freq']) else None,
                'freq_b': round(case['B_freq'], 2) if np.isfinite(case['B_freq']) else None,
                'freq_bpn': round(case['Bpn_freq'], 2) if np.isfinite(case['Bpn_freq']) else None,
                'expert_freq': round(case['expert_freq'], 2),
                'disagreement': round(case['disagreement'], 2),
                'pattern': subdir,
                'mat_file': mat_name,
            }
            for e in experts_list:
                if e in case['expert_individual']:
                    meta_entry[f'expert_{e}'] = round(case['expert_individual'][e], 2)
                else:
                    meta_entry[f'expert_{e}'] = None

            metadata[base_name] = meta_entry
            print('  ok')

        except Exception as e:
            print(f'  FAILED: {e}')
            continue

    # Remove data arrays from cases (for the HTML generation)
    for c in cases:
        c.pop('data', None)

    # Save metadata
    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_DIR / 'metadata.js', 'w') as f:
        f.write(f'metadata = {json.dumps(metadata, indent=2)};\nshow();\n')

    # Generate viewer HTML
    valid_cases = [c for c in cases if c['base_name'] in metadata]
    html = generate_viewer_html(valid_cases)
    with open(OUTPUT_DIR / 'viewer.html', 'w') as f:
        f.write(html)

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'  Total segments with expert consensus: {n_total}')
    print(f'  Disagreement cases (|B+pnorm - expert| > 0.5 Hz): {len(metadata)}')
    print(f'  Output directory: {OUTPUT_DIR}')
    print(f'  Viewer: {OUTPUT_DIR / "viewer.html"}')
    print(f'\nTop 10 disagreements:')
    for c in valid_cases[:10]:
        m = metadata[c['base_name']]
        print(f'  {c["base_name"]}: B+pn={m["freq_bpn"]} vs expert={m["expert_freq"]} (delta={m["disagreement"]})')


if __name__ == '__main__':
    main()
