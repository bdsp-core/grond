"""
Voltage topography and CSD-based spatial localization for LPD/GPD.

Compares 4 methods on multi-rater segments:
  1. CNN+PLV (current) -- from PDCharacterizer
  2. Voltage at peak  -- mean discharge voltage topography -> regions
  3. CSD at peak      -- CSD of mean discharge voltage -> regions
  4. Hybrid CNN+CSD   -- average of CNN+PLV and CSD scores

For each method, sweeps threshold percentiles (30-70) and reports best Jaccard.
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import mne
mne.set_log_level('ERROR')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_characterizer import PDCharacterizer
from pd_pointiness_acf import fcn_getBanana

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = PROJECT_DIR / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']

# 19-channel monopolar order matching data files
MONO_CHANNELS_19 = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2',
]

# MNE-compatible names (10-20 standard)
NAME_MAP = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
MNE_NAMES = [NAME_MAP.get(n, n) for n in MONO_CHANNELS_19]

# Region to monopolar electrode mapping
REGION_TO_ELECTRODES = {
    'LF': ['Fp1', 'F3', 'F7'], 'RF': ['Fp2', 'F4', 'F8'],
    'LT': ['T3', 'T5', 'F7'], 'RT': ['T4', 'T6', 'F8'],
    'LCP': ['C3', 'P3'], 'RCP': ['C4', 'P4'],
    'LO': ['O1', 'P3', 'T5'], 'RO': ['O2', 'P4', 'T6'],
}


def _parse_regions(spatial_str):
    """Parse spatial_channels string into set of region names."""
    if not spatial_str or str(spatial_str).strip() in ('', '0', 'na', 'NA', 'nan'):
        return set()
    tokens = str(spatial_str).replace(',', ' ').split()
    canonical = set()
    for t in tokens:
        t = t.strip().upper()
        if t in REGIONS:
            canonical.add(t)
        elif t == 'LPC':
            canonical.add('LCP')
        elif t == 'LP':
            canonical.add('LCP')
        elif t == 'LFP':
            canonical.add('LF')
    return canonical


def load_multirater_segments(verbose=True):
    """Load segments with 2+ expert spatial raters (LB, PH, SZ)."""
    annot = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))

    seg_info = {}
    for _, sr in seg_df.iterrows():
        seg_info[sr['segment_id']] = {
            'mat_file': sr['mat_file'],
            'subtype': sr['subtype'],
            'patient_id': str(sr['patient_id']),
        }

    # Group spatial annotations by segment, only expert raters
    seg_spatial = defaultdict(dict)
    for _, r in annot.iterrows():
        sc = str(r.get('spatial_channels', '')).strip()
        rater = r.get('rater', '')
        if sc and rater in ('LB', 'PH', 'SZ'):
            regions = _parse_regions(sc)
            if regions:
                seg_spatial[r['segment_id']][rater] = regions

    # Build gold standard with majority vote
    records = []
    for sid, raters in seg_spatial.items():
        if len(raters) < 2:
            continue
        info = seg_info.get(sid)
        if not info:
            continue
        if info['subtype'] not in ('lpd', 'gpd'):
            continue

        region_votes = defaultdict(int)
        for rater_regions in raters.values():
            for reg in rater_regions:
                region_votes[reg] += 1

        threshold = 2
        gold_regions = set()
        for reg, votes in region_votes.items():
            if votes >= threshold:
                gold_regions.add(reg)

        if not gold_regions:
            continue

        # Per-rater labels for breakdown
        rater_labels = {k: sorted(v) for k, v in raters.items()}

        records.append({
            'segment_id': sid,
            'patient_id': info['patient_id'],
            'subtype': info['subtype'],
            'mat_file': info['mat_file'],
            'gold_regions': sorted(gold_regions),
            'n_raters': len(raters),
            'rater_labels': rater_labels,
        })

    if verbose:
        n_lpd = sum(1 for r in records if r['subtype'] == 'lpd')
        n_gpd = sum(1 for r in records if r['subtype'] == 'gpd')
        print(f"Gold standard: {len(records)} segments ({n_lpd} LPD, {n_gpd} GPD)")

    return records


def make_mne_info():
    """Create MNE Info object for 19-channel monopolar montage."""
    info = mne.create_info(ch_names=MNE_NAMES, sfreq=FS, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)
    return info


def topo_to_region_scores(topo_19, ch_names=MONO_CHANNELS_19):
    """Convert 19-electrode topography to per-region scores in [0,1]."""
    abs_topo = np.abs(topo_19)
    mx = abs_topo.max()
    if mx > 0:
        norm = abs_topo / mx
    else:
        norm = abs_topo.copy()

    region_scores = {}
    for region, electrodes in REGION_TO_ELECTRODES.items():
        scores = []
        for e in electrodes:
            if e in ch_names:
                scores.append(norm[ch_names.index(e)])
        region_scores[region] = float(max(scores)) if scores else 0.0
    return region_scores


def predict_regions_at_threshold(region_scores, threshold_pct):
    """Predict involved regions above a percentile threshold."""
    vals = list(region_scores.values())
    threshold = np.percentile(vals, threshold_pct)
    return [r for r in REGIONS if region_scores.get(r, 0) >= threshold]


def compute_jaccard(pred_regions, gold_regions):
    """Compute Jaccard index between two region sets."""
    pred = set(pred_regions)
    gold = set(gold_regions)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    return len(pred & gold) / len(pred | gold)


def compute_per_rater_jaccard(pred_regions, rater_labels):
    """Compute Jaccard against each individual rater."""
    results = {}
    for rater, labels in rater_labels.items():
        results[rater] = compute_jaccard(pred_regions, labels)
    return results


def main():
    print("=" * 80)
    print("Voltage Topography & CSD Spatial Localization Comparison")
    print("=" * 80)

    # Load data
    records = load_multirater_segments(verbose=True)
    if not records:
        print("No multi-rater segments found!")
        return

    # Initialize
    pc = PDCharacterizer()
    mne_info = make_mne_info()

    # Storage for all methods' region scores per segment
    # method -> segment_id -> region_scores dict
    all_scores = {
        'CNN+PLV': {},
        'Voltage': {},
        'CSD': {},
        'Hybrid_CNN_CSD': {},
    }

    n_total = len(records)
    n_no_discharge = 0
    n_csd_fail = 0
    n_loaded = 0
    t0 = time.time()

    for i, rec in enumerate(records):
        sid = rec['segment_id']
        mat_path = EEG_DIR / rec['mat_file']
        if not mat_path.exists():
            continue

        try:
            mat = sio.loadmat(str(mat_path))
            dk = [k for k in mat if not k.startswith('_')][0]
            data = mat[dk].astype(np.float64)
            if data.shape[0] > data.shape[1]:
                data = data.T
        except Exception:
            continue

        # Get monopolar (19ch) and bipolar (18ch) data
        mono = data[:19, :2000].copy()
        try:
            seg_bi = np.array(fcn_getBanana(data[:, :2000]), dtype=np.float64)
        except Exception:
            continue

        n_loaded += 1

        # --- Method 1: CNN+PLV (current system) ---
        try:
            result = pc.characterize(seg_bi, subtype=rec['subtype'])
            cnn_plv_scores = result.get('region_scores', {})
            discharge_times = result.get('discharge_times', [])
        except Exception:
            cnn_plv_scores = {r: 0.0 for r in REGIONS}
            discharge_times = []

        all_scores['CNN+PLV'][sid] = cnn_plv_scores

        # --- Methods 2 & 3: Voltage and CSD at discharge peaks ---
        # Extract monopolar voltage at each discharge peak
        peak_voltages = []
        for t in discharge_times:
            sample = int(t * FS)
            if 0 <= sample < mono.shape[1]:
                peak_voltages.append(mono[:, sample])

        if peak_voltages:
            mean_topo = np.mean(peak_voltages, axis=0)  # (19,)
        else:
            # Fallback: use full-segment mean absolute voltage
            mean_topo = np.mean(np.abs(mono), axis=1)  # (19,)
            n_no_discharge += 1

        # Voltage topography scores
        voltage_scores = topo_to_region_scores(mean_topo)
        all_scores['Voltage'][sid] = voltage_scores

        # CSD topography
        try:
            evoked = mne.EvokedArray(
                mean_topo.reshape(19, 1), mne_info, tmin=0)
            evoked_csd = mne.preprocessing.compute_current_source_density(evoked)
            csd_topo = evoked_csd.data[:, 0]
            csd_scores = topo_to_region_scores(csd_topo)
        except Exception:
            csd_scores = voltage_scores.copy()
            n_csd_fail += 1

        all_scores['CSD'][sid] = csd_scores

        # --- Method 4: Hybrid CNN+PLV + CSD ---
        hybrid_scores = {}
        for r in REGIONS:
            hybrid_scores[r] = 0.5 * cnn_plv_scores.get(r, 0.0) + 0.5 * csd_scores.get(r, 0.0)
        all_scores['Hybrid_CNN_CSD'][sid] = hybrid_scores

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Progress: {i+1}/{n_total} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nProcessed {n_loaded}/{n_total} segments in {elapsed:.0f}s")
    print(f"  No discharges (used fallback): {n_no_discharge}")
    print(f"  CSD failures (used voltage): {n_csd_fail}")

    # --- Evaluation: sweep thresholds, compute Jaccard ---
    thresholds = list(range(30, 75, 5))  # 30, 35, ..., 70
    method_names = ['CNN+PLV', 'Voltage', 'CSD', 'Hybrid_CNN_CSD']

    # Build index of loaded segments
    valid_records = [r for r in records if r['segment_id'] in all_scores['CNN+PLV']]

    print(f"\nEvaluating on {len(valid_records)} segments...")
    print(f"Sweeping thresholds: {thresholds}")

    results_summary = {}

    for method in method_names:
        best_jaccard = -1
        best_threshold = None
        best_per_rater = None

        for thr in thresholds:
            jaccards = []
            per_rater_jacs = defaultdict(list)

            for rec in valid_records:
                sid = rec['segment_id']
                scores = all_scores[method].get(sid, {})
                if not scores:
                    continue

                pred = predict_regions_at_threshold(scores, thr)
                jac = compute_jaccard(pred, rec['gold_regions'])
                jaccards.append(jac)

                # Per-rater breakdown
                for rater, labels in rec['rater_labels'].items():
                    rater_jac = compute_jaccard(pred, labels)
                    per_rater_jacs[rater].append(rater_jac)

            mean_jac = float(np.mean(jaccards)) if jaccards else 0.0
            if mean_jac > best_jaccard:
                best_jaccard = mean_jac
                best_threshold = thr
                best_per_rater = {r: float(np.mean(v)) for r, v in per_rater_jacs.items()}

        results_summary[method] = {
            'best_jaccard': round(best_jaccard, 4),
            'best_threshold_pct': best_threshold,
            'per_rater_jaccard': {r: round(v, 4) for r, v in best_per_rater.items()} if best_per_rater else {},
            'n_segments': len(valid_records),
        }

    # --- Print comparison table ---
    print("\n" + "=" * 100)
    print("RESULTS: Voltage Topography & CSD Spatial Localization")
    print("=" * 100)

    raters = ['LB', 'PH', 'SZ']
    header = f"{'Method':<22} | {'Best Jaccard':>12} | {'Threshold':>9}"
    for r in raters:
        header += f" | {'vs '+r:>7}"
    header += f" | {'Mean':>6}"
    print(header)
    print("-" * len(header))

    for method in method_names:
        res = results_summary[method]
        per_r = res['per_rater_jaccard']
        rater_vals = [per_r.get(r, 0) for r in raters]
        mean_rater = float(np.mean(rater_vals)) if rater_vals else 0
        row = f"{method:<22} | {res['best_jaccard']:>12.4f} | {res['best_threshold_pct']:>7d}th"
        for r in raters:
            row += f" | {per_r.get(r, 0):>7.3f}"
        row += f" | {mean_rater:>6.3f}"
        print(row)

    print("=" * 100)

    # --- Also compute per-subtype breakdown ---
    print("\nPer-subtype breakdown (at best threshold per method):")
    print(f"{'Method':<22} | {'LPD Jaccard':>11} | {'GPD Jaccard':>11}")
    print("-" * 52)
    for method in method_names:
        thr = results_summary[method]['best_threshold_pct']
        for subtype_label in ['lpd', 'gpd']:
            subtype_recs = [r for r in valid_records if r['subtype'] == subtype_label]
            jacs = []
            for rec in subtype_recs:
                sid = rec['segment_id']
                scores = all_scores[method].get(sid, {})
                if scores:
                    pred = predict_regions_at_threshold(scores, thr)
                    jacs.append(compute_jaccard(pred, rec['gold_regions']))
            results_summary[method][f'{subtype_label}_jaccard'] = round(float(np.mean(jacs)), 4) if jacs else 0.0

        res = results_summary[method]
        print(f"{method:<22} | {res.get('lpd_jaccard', 0):>11.4f} | {res.get('gpd_jaccard', 0):>11.4f}")

    # --- Save results ---
    output_path = RESULTS_DIR / 'voltage_csd_comparison.json'
    with open(str(output_path), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
