#!/usr/bin/env python3
"""PDProfiler Optimization Contest — preprocessing/postprocessing variants.

Runs PDProfiler with different pre- and post-processing on all labeled LPD
segments. No retraining — uses existing trained weights. Evaluates laterality,
frequency, and discharge timing metrics. Generates an HTML leaderboard.

Contest entries (all use the same trained PDProfiler weights):
  A: Baseline — current PDProfiler as-is
  B: Bandpass 0.5-20 Hz before PDProfiler
  C: Bandpass 1-15 Hz before PDProfiler (aggressive HF removal)
  D: Notch 60 Hz before PDProfiler
  E: Detrend + bandpass 0.5-30 Hz before PDProfiler
  F: Edge padding — mirror-pad 1s on each side, run, trim (fix edge effects)
  G: Evidence smoothing — wider Gaussian σ (reduce spurious peaks)
  H: Relaxed DP — lower alpha (less strict periodicity prior)
  I: Strict DP — higher alpha + lower skip penalty
  J: Kitchen sink — BP 0.5-20 Hz + edge padding + relaxed DP

Usage:
    conda run -n morgoth python code/hemi_detector/retrain_contest.py
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, iirnotch, detrend as scipy_detrend
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import scipy.io as sio

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_profiler import PDProfiler
from pd_pointiness_acf import fcn_getBanana

# Constants
FS = 200
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
RESULTS_DIR = ROOT_DIR / 'results' / 'hemi_retrain_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# Preprocessing functions — applied to 18-ch bipolar EEG before PDProfiler
# ═══════════════════════════════════════════════════════════════════════

def preproc_none(seg_bi):
    """A: No preprocessing — current behavior."""
    return seg_bi

def preproc_bp_05_20(seg_bi):
    """B: Bandpass 0.5-20 Hz."""
    sos = butter(4, [0.5, 20], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, seg_bi, axis=1)

def preproc_bp_1_15(seg_bi):
    """C: Bandpass 1-15 Hz (aggressive)."""
    sos = butter(4, [1.0, 15.0], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, seg_bi, axis=1)

def preproc_notch60(seg_bi):
    """D: Notch 60 Hz."""
    b, a = iirnotch(60.0, 30.0, FS)
    from scipy.signal import filtfilt
    return filtfilt(b, a, seg_bi, axis=1)

def preproc_detrend_bp(seg_bi):
    """E: Detrend + bandpass 0.5-30 Hz."""
    detrended = scipy_detrend(seg_bi, axis=1)
    sos = butter(4, [0.5, 30], btype='bandpass', fs=FS, output='sos')
    return sosfiltfilt(sos, detrended, axis=1)

def preproc_edge_pad(seg_bi):
    """F: Mirror-pad 1s on each side to reduce edge effects.
    Pads, runs through pipeline later, then trims.
    Returns padded array with metadata for trimming."""
    pad_samples = FS  # 1 second = 200 samples
    padded = np.pad(seg_bi, ((0, 0), (pad_samples, pad_samples)), mode='reflect')
    return padded

def preproc_bp_05_20_edge_pad(seg_bi):
    """J: BP 0.5-20 Hz + edge padding."""
    sos = butter(4, [0.5, 20], btype='bandpass', fs=FS, output='sos')
    filtered = sosfiltfilt(sos, seg_bi, axis=1)
    pad_samples = FS
    padded = np.pad(filtered, ((0, 0), (pad_samples, pad_samples)), mode='reflect')
    return padded


# ═══════════════════════════════════════════════════════════════════════
# Post-processing / DP parameter overrides
# ═══════════════════════════════════════════════════════════════════════

# Default DP params (from discharge_detector.py)
DEFAULT_DP = {
    'alpha': 1.275,
    'beta': 0.3,
    'dp_lambda': 0.05,
    'max_skip': 3,
}

DP_RELAXED = {
    'alpha': 0.8,   # less strict periodicity
    'beta': 0.2,
    'dp_lambda': 0.05,
    'max_skip': 4,
}

DP_STRICT = {
    'alpha': 2.0,    # stricter periodicity
    'beta': 0.5,     # higher skip penalty
    'dp_lambda': 0.05,
    'max_skip': 2,
}

# ═══════════════════════════════════════════════════════════════════════
# Contest entries
# ═══════════════════════════════════════════════════════════════════════

ENTRIES = {
    'A_Baseline': {
        'description': 'Current PDProfiler, no changes',
        'preproc': preproc_none,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'B_BP05_20': {
        'description': 'Bandpass 0.5-20 Hz before inference',
        'preproc': preproc_bp_05_20,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'C_BP1_15': {
        'description': 'Bandpass 1-15 Hz before inference',
        'preproc': preproc_bp_1_15,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'D_Notch60': {
        'description': 'Notch 60 Hz before inference',
        'preproc': preproc_notch60,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'E_Detrend_BP': {
        'description': 'Detrend + bandpass 0.5-30 Hz',
        'preproc': preproc_detrend_bp,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'F_EdgePad': {
        'description': 'Mirror-pad 1s edges to fix boundary effects',
        'preproc': preproc_none,
        'edge_pad': True,
        'dp_params': None,
        'evidence_sigma': None,
    },
    'G_SmoothEvidence': {
        'description': 'Wider evidence smoothing (sigma=6 vs 3)',
        'preproc': preproc_none,
        'edge_pad': False,
        'dp_params': None,
        'evidence_sigma': 6,
    },
    'H_RelaxedDP': {
        'description': 'Relaxed DP (alpha=0.8, less strict periodicity)',
        'preproc': preproc_none,
        'edge_pad': False,
        'dp_params': DP_RELAXED,
        'evidence_sigma': None,
    },
    'I_StrictDP': {
        'description': 'Strict DP (alpha=2.0, stricter periodicity)',
        'preproc': preproc_none,
        'edge_pad': False,
        'dp_params': DP_STRICT,
        'evidence_sigma': None,
    },
    'J_KitchenSink': {
        'description': 'BP 0.5-20 Hz + edge pad + relaxed DP',
        'preproc': preproc_bp_05_20,
        'edge_pad': True,
        'dp_params': DP_RELAXED,
        'evidence_sigma': None,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_evaluation_data():
    """Load all labeled LPD segments and ground truth."""
    sl = pd.read_csv(ROOT_DIR / 'data' / 'labels' / 'segment_labels.csv')
    lpd = sl[(sl.subtype == 'lpd') & (sl.excluded != True)].copy()

    with open(ROOT_DIR / 'data' / 'labels' / 'discharge_times.json') as f:
        dt_json = json.load(f)

    # Build discharge_times lookup by mat_file
    import os
    eeg_files = set(os.listdir(ROOT_DIR / 'data' / 'eeg'))
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

    print(f"LPD segments: {len(lpd)}")
    print(f"  With laterality: {lpd.laterality.isin(['left','right']).sum()}")
    print(f"  With expert freq: {lpd.expert_freq_hz.notna().sum()}")
    print(f"  With discharge times: {sum(1 for m in lpd.mat_file if m in dt_by_matfile)}")

    return lpd, dt_by_matfile


# ═══════════════════════════════════════════════════════════════════════
# Run a single entry
# ═══════════════════════════════════════════════════════════════════════

def run_entry(entry_name, entry_config, lpd, dt_by_matfile):
    """Run PDProfiler with the given preprocessing/postprocessing on all LPD data."""
    print(f"\n{'='*60}")
    print(f"Entry: {entry_name} — {entry_config['description']}")
    print(f"{'='*60}")

    preproc_fn = entry_config['preproc']
    edge_pad = entry_config['edge_pad']
    dp_params = entry_config['dp_params']
    evidence_sigma = entry_config['evidence_sigma']

    # Create PDProfiler — may need to monkey-patch for DP/evidence params
    pc = PDProfiler()

    # If we need custom DP params or evidence sigma, patch the discharge_detector module
    if dp_params is not None:
        import discharge_detector as dd
        dd.DP_ALPHA = dp_params['alpha']
        dd.DP_BETA = dp_params['beta']
        dd.DP_LAMBDA = dp_params['dp_lambda']
        dd.MAX_SKIP = dp_params['max_skip']
    else:
        import discharge_detector as dd
        dd.DP_ALPHA = 1.275
        dd.DP_BETA = 0.3
        dd.DP_LAMBDA = 0.05
        dd.MAX_SKIP = 3

    if evidence_sigma is not None:
        import discharge_detector as dd
        dd.SMOOTH_SIGMA_SAMPLES = evidence_sigma
    else:
        import discharge_detector as dd
        dd.SMOOTH_SIGMA_SAMPLES = 3

    # Evaluation accumulators
    gt_labels, lat_scores = [], []
    lat_correct = 0
    freq_gt, freq_pred = [], []
    tp_total, fp_total, fn_total = 0, 0, 0
    timing_errors = []
    n_processed = 0
    test_case_result = None

    t0 = time.time()

    for _, row in lpd.iterrows():
        mat_file = row['mat_file']
        try:
            mat = sio.loadmat(str(ROOT_DIR / 'data' / 'eeg' / mat_file))
            data = mat['data']
            if data.shape[0] > data.shape[1]:
                data = data.T
            seg_bi = fcn_getBanana(data[:, :2000])

            # Apply preprocessing
            seg_input = preproc_fn(seg_bi)

            # Handle edge padding
            if edge_pad:
                pad_samples = FS  # 1 second
                seg_padded = np.pad(seg_input, ((0, 0), (pad_samples, pad_samples)), mode='reflect')
                result = pc.characterize(seg_padded, subtype='lpd')
                # Trim discharge times back to original window
                if 'discharge_times' in result and result['discharge_times']:
                    orig_times = []
                    offset = pad_samples / FS  # 1.0 second
                    for t in result['discharge_times']:
                        t_adj = t - offset
                        if 0 <= t_adj <= 10.0:
                            orig_times.append(t_adj)
                    result['discharge_times'] = orig_times
            else:
                result = pc.characterize(seg_input, subtype='lpd')

        except Exception as e:
            continue

        n_processed += 1

        # --- Laterality ---
        gt_lat = row.get('laterality')
        if gt_lat in ('left', 'right'):
            cp = result['channel_probs']
            score = np.mean([cp[i] for i in RIGHT_INDICES]) - np.mean([cp[i] for i in LEFT_INDICES])
            gt_labels.append(1 if gt_lat == 'right' else 0)
            lat_scores.append(score)
            if result['laterality'] == gt_lat:
                lat_correct += 1

        # --- Frequency ---
        gt_freq = float(row['expert_freq_hz']) if pd.notna(row.get('expert_freq_hz')) else None
        pred_freq = result.get('frequency')
        if gt_freq and pred_freq and np.isfinite(pred_freq):
            freq_gt.append(gt_freq)
            freq_pred.append(pred_freq)

        # --- Discharge timing ---
        gt_times = dt_by_matfile.get(mat_file)
        if gt_times and len(gt_times) > 0:
            pred_times = sorted(result.get('discharge_times', []))
            gt_sorted = sorted(gt_times)
            tol = 0.1  # 100ms

            matched_gt = set()
            for pt in pred_times:
                best_j, best_d = -1, tol + 1
                for j, gt_t in enumerate(gt_sorted):
                    if j not in matched_gt:
                        d = abs(pt - gt_t)
                        if d < best_d:
                            best_d = d
                            best_j = j
                if best_j >= 0 and best_d <= tol:
                    matched_gt.add(best_j)
                    timing_errors.append(best_d)

            tp = len(matched_gt)
            fp = len(pred_times) - tp
            fn = len(gt_sorted) - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

        # --- Test case ---
        if 'S0001116940915_20140820235019' in mat_file:
            test_case_result = {
                'pred_lat': result.get('laterality', '?'),
                'pred_freq': round(result.get('frequency', 0), 2),
                'n_det': len(result.get('discharge_times', [])),
                'gt_lat': 'right',
                'gt_freq': 1.75,
            }

        if n_processed % 500 == 0:
            print(f"  {n_processed} segments processed...")

    elapsed = time.time() - t0
    print(f"  Processed {n_processed} segments in {elapsed:.0f}s")

    # --- Compute metrics ---
    metrics = {
        'name': entry_name,
        'description': entry_config['description'],
        'n_processed': n_processed,
        'time_s': round(elapsed, 1),
    }

    # Laterality
    if gt_labels and len(set(gt_labels)) > 1:
        metrics['lat_auc'] = round(roc_auc_score(gt_labels, lat_scores), 3)
        metrics['lat_acc'] = round(100 * lat_correct / len(gt_labels), 1)
        metrics['lat_n'] = len(gt_labels)
    else:
        metrics['lat_auc'] = '--'
        metrics['lat_acc'] = '--'

    # Frequency
    if freq_gt:
        rho, _ = spearmanr(freq_gt, freq_pred)
        metrics['freq_rho'] = round(rho, 3)
        metrics['freq_mae'] = round(np.mean(np.abs(np.array(freq_gt) - np.array(freq_pred))), 3)
        metrics['freq_n'] = len(freq_gt)
    else:
        metrics['freq_rho'] = '--'
        metrics['freq_mae'] = '--'

    # Timing
    if tp_total > 0:
        prec = tp_total / (tp_total + fp_total)
        sens = tp_total / (tp_total + fn_total)
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        metrics['timing_f1'] = round(f1, 3)
        metrics['timing_sens'] = round(sens, 3)
        metrics['timing_prec'] = round(prec, 3)
        metrics['timing_mae_ms'] = round(np.mean(timing_errors) * 1000, 1)
        metrics['timing_tp'] = tp_total
        metrics['timing_fp'] = fp_total
        metrics['timing_fn'] = fn_total
    else:
        metrics['timing_f1'] = '--'
        metrics['timing_sens'] = '--'
        metrics['timing_prec'] = '--'
        metrics['timing_mae_ms'] = '--'

    # Test case
    metrics['test_case'] = test_case_result or {}

    # Print summary
    print(f"\n  Results for {entry_name}:")
    print(f"    Lat AUC: {metrics['lat_auc']}, Acc: {metrics['lat_acc']}%")
    print(f"    Freq rho: {metrics['freq_rho']}, MAE: {metrics['freq_mae']}")
    print(f"    Timing F1: {metrics['timing_f1']}, Sens: {metrics['timing_sens']}, Prec: {metrics['timing_prec']}")
    if test_case_result:
        print(f"    Test: lat={test_case_result['pred_lat']}, freq={test_case_result['pred_freq']}, det={test_case_result['n_det']}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════════════════════

def update_leaderboard():
    """Regenerate leaderboard HTML from all result JSONs."""
    results = []
    for f in sorted(RESULTS_DIR.glob('*.json')):
        with open(f) as fh:
            results.append(json.load(fh))

    # Sort by freq_rho descending (primary)
    def sort_key(r):
        v = r.get('freq_rho', 0)
        return v if isinstance(v, (int, float)) else -1
    results.sort(key=sort_key, reverse=True)

    n_entries = len(results)

    def _fmt(v, fmt='.3f'):
        if isinstance(v, str):
            return v
        return f'{v:{fmt}}'

    def _color_lat(v):
        if not isinstance(v, (int, float)):
            return '#888'
        return '#44cc88' if v >= 0.95 else '#cccc44' if v >= 0.8 else '#cc4444'

    def _color_freq(v):
        if not isinstance(v, (int, float)):
            return '#888'
        return '#44cc88' if v >= 0.6 else '#cccc44' if v >= 0.45 else '#cc4444'

    rows = ''
    for i, r in enumerate(results):
        tc = r.get('test_case', {})
        test_lat = tc.get('pred_lat', '?')
        test_freq = tc.get('pred_freq', '?')
        test_det = tc.get('n_det', '?')
        lat_color = '#44cc88' if test_lat == 'right' else '#cc4444'

        rows += f"""<tr>
<td>{i+1}</td>
<td><strong>{r['name']}</strong></td>
<td>{r['description']}</td>
<td style="color:{_color_lat(r.get('lat_auc'))};font-weight:bold">{_fmt(r.get('lat_auc', '--'))}</td>
<td>{_fmt(r.get('lat_acc', '--'), '.1f')}%</td>
<td style="color:{_color_freq(r.get('freq_rho'))};font-weight:bold">{_fmt(r.get('freq_rho', '--'))}</td>
<td>{_fmt(r.get('freq_mae', '--'))}</td>
<td>{_fmt(r.get('timing_f1', '--'))}</td>
<td>{_fmt(r.get('timing_sens', '--'))}</td>
<td>{_fmt(r.get('timing_prec', '--'))}</td>
<td>{_fmt(r.get('timing_mae_ms', '--'), '.1f')}</td>
<td style="color:{lat_color}">{test_lat}</td>
<td>{test_freq}</td>
<td>{test_det}</td>
</tr>"""

    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html><head><title>PDProfiler Optimization Contest</title>
<meta http-equiv="refresh" content="10">
<style>
body {{ background:#111; color:#eee; font-family:Consolas,monospace; padding:20px; }}
h1 {{ color:#44cc88; }}
h2 {{ color:#888; margin-top:30px; }}
table {{ border-collapse:collapse; width:100%; }}
th,td {{ padding:6px 10px; text-align:left; border-bottom:1px solid #333; font-size:13px; }}
th {{ background:#222; color:#888; position:sticky; top:0; }}
tr:hover {{ background:#1a1a1a; }}
.note {{ color:#888; font-size:12px; margin-top:10px; }}
</style></head><body>
<h1>PDProfiler Optimization Contest &mdash; {n_entries} entries</h1>
<p class="note">Same trained model weights throughout. Only preprocessing and DP parameters vary.</p>
<p class="note">Sorted by Freq rho (primary). Test case: sub-S0001116940915 (GT: right, 1.75 Hz, ~18 discharges)</p>

<table>
<tr>
  <th>#</th><th>Entry</th><th>Description</th>
  <th>Lat AUC</th><th>Lat Acc</th>
  <th>Freq &rho;</th><th>Freq MAE</th>
  <th>Timing F1</th><th>Timing Sens</th><th>Timing Prec</th><th>Timing MAE ms</th>
  <th>Test Lat</th><th>Test Freq</th><th>Test #Det</th>
</tr>
{rows}
</table>
<p class="note">Updated: {ts}</p>
<p class="note">
<strong>Lat AUC</strong> = ROC AUC for left vs right lateralization (higher is better)<br>
<strong>Freq &rho;</strong> = Spearman correlation vs expert frequency (higher is better)<br>
<strong>Freq MAE</strong> = Mean absolute error in Hz (lower is better)<br>
<strong>Timing F1/Sens/Prec</strong> = Discharge detection at &plusmn;100ms tolerance<br>
<strong>Timing MAE</strong> = Mean timing error of matched detections in ms (lower is better)
</p>
</body></html>"""

    with open(RESULTS_DIR / 'hemi_retrain_leaderboard.html', 'w') as f:
        f.write(html)
    print(f"  Leaderboard updated ({n_entries} entries)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    # Clean up old retraining results
    for f in RESULTS_DIR.glob('*.json'):
        f.unlink()

    print("Loading evaluation data...")
    lpd, dt_by_matfile = load_evaluation_data()

    for entry_name, entry_config in ENTRIES.items():
        result_file = RESULTS_DIR / f'{entry_name}.json'

        metrics = run_entry(entry_name, entry_config, lpd, dt_by_matfile)

        # Save results
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Update leaderboard after each entry
        update_leaderboard()

    print(f"\n{'='*60}")
    print("Contest complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
