"""Generate HPP labeler for the 27 LPD/GPD cases missing timing labels."""

import sys, json, numpy as np, scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES, FS
from label_pipeline.hpp_discharge_marking import (
    _compute_channel_evidence, _aggregate_evidence,
    _detect_active_interval, _extract_candidates, _dp_best_sequence, _em_refine,
)
from generate_hpp_labeler import build_html, FREQ_BUTTONS

DURATION = 10.0


def hpp_with_freq(evidence, fs, freq_hz):
    if len(evidence) < 10 or freq_hz <= 0: return []
    freq_estimate = np.clip(freq_hz, 0.2, 5.0)
    active_start, active_end = _detect_active_interval(evidence, fs)
    candidates = _extract_candidates(evidence, fs, freq_estimate, active_start, active_end)
    if len(candidates) == 0: return []
    discharge_samples = _dp_best_sequence(candidates, evidence, fs, freq_estimate)
    if len(discharge_samples) == 0: return []
    if len(discharge_samples) >= 3:
        discharge_samples = _em_refine(evidence, discharge_samples, fs, freq_estimate)
    return [t for t in (discharge_samples / fs).tolist() if 0 <= t <= DURATION]


def downsample(arr, n):
    if arr.ndim == 1:
        if len(arr) <= n: return arr.tolist()
        return arr[np.linspace(0, len(arr)-1, n).astype(int)].tolist()
    if arr.shape[1] <= n: return arr.tolist()
    return arr[:, np.linspace(0, arr.shape[1]-1, n).astype(int)].tolist()


def main():
    patients = pd.read_csv(str(PROJECT_DIR / 'data/labels/patients.csv'))
    with open(str(PROJECT_DIR / 'data/labels/discharge_times.json')) as f:
        hpp = json.load(f)
    gt_pids = {pid for pid, v in hpp.items() if v.get('review_status') == 'ground_truth'}

    lpd_gpd = patients[(patients['subtype'].isin(['lpd', 'gpd'])) & (patients['excluded'] != True)]
    missing = lpd_gpd[~lpd_gpd['patient_id'].astype(str).isin(gt_pids)]
    pids = sorted(missing['patient_id'].values)
    print(f"Cases needing timing: {len(pids)}")

    dataset = load_dataset(verbose=False)
    segments = dataset['segments']

    b_lp, a_lp = butter(4, 20.0 / (FS / 2), btype='low')

    cases = []
    for pid in pids:
        row = missing[missing['patient_id'] == pid].iloc[0]
        subtype = row['subtype']
        lat = row.get('laterality', '')
        if not isinstance(lat, str) or lat not in ('left', 'right'):
            lat = None

        seg = None
        pat_segs = segments.get(str(pid), [])
        if pat_segs:
            seg = pat_segs[0]
        else:
            for suffix in ['_seg000.mat', '.mat']:
                p = PROJECT_DIR / 'data' / 'eeg' / f'{pid}{suffix}'
                if p.exists():
                    mat = sio.loadmat(str(p))
                    key = [k for k in mat.keys() if not k.startswith('_')][0]
                    seg = mat[key]
                    if seg.shape[0] > seg.shape[1]: seg = seg.T
                    seg = seg[:18, :2000]
                    break
        if seg is None:
            print(f'  SKIP {pid}')
            continue

        seg_d = np.zeros_like(seg)
        for ch in range(seg.shape[0]):
            try: seg_d[ch] = filtfilt(b_lp, a_lp, seg[ch])
            except: seg_d[ch] = seg[ch]

        n_ch = min(seg.shape[0], 18)
        evidence_all = np.zeros((n_ch, seg.shape[1]))
        for ch in range(n_ch):
            evidence_all[ch] = _compute_channel_evidence(seg[ch], FS)
        evidence = _aggregate_evidence(evidence_all, subtype, lat)
        ev_max = np.max(evidence)
        ev_display = evidence / ev_max if ev_max > 0 else evidence

        hpp_results = {}
        for freq in FREQ_BUTTONS:
            hpp_results[str(freq)] = hpp_with_freq(evidence, FS, freq)

        cases.append({
            'patient_id': str(pid),
            'est_freq': 0,
            'bin': f'{subtype.upper()} | lat={lat or "?"}',
            'eeg_data': downsample(seg_d, 1000),
            'evidence': downsample(ev_display, 500),
            'hpp_results': hpp_results,
        })

    print(f"Loaded {len(cases)} cases")

    html = build_html(cases)
    out_path = PROJECT_DIR / 'results' / 'hpp_labeler_remaining27.html'
    with open(str(out_path), 'w') as f:
        f.write(html)
    print(f"Written to {out_path}")

    import subprocess
    subprocess.run(['open', str(out_path)])


if __name__ == '__main__':
    main()
