"""
Round 2 experiment: Use Method A frequency as anchor to disambiguate Method B subharmonics.

Tests 6 fusion strategies combining Method A (pd_detect_alternate),
Method B (pd_detect_pointiness_acf with acf_thr=0.10), and peak-count frequency.
"""

import sys
import numpy as np
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'pd_detector_alternate'))

from optimization_harness import load_dataset, load_eeg_data, evaluate_predictions
from pd_detect_alternate import pd_detect_alternate
from pd_pointiness_acf import pd_detect_pointiness_acf

# ---------------------------------------------------------------------------
# Peak-count frequency (same as round 1 baseline)
# ---------------------------------------------------------------------------
from scipy.signal import find_peaks, butter, filtfilt, detrend
from scipy.ndimage import gaussian_filter1d

mono_channels = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
    'Fz', 'Cz', 'Pz',
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG',
]
bipolar_channels = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]


def fcn_getBanana(X):
    bipolar_ids = np.array([
        [mono_channels.index(bc.split('-')[0]), mono_channels.index(bc.split('-')[1])]
        for bc in bipolar_channels
    ])
    return X[bipolar_ids[:, 0]] - X[bipolar_ids[:, 1]]


def peak_count_frequency(segment, fs):
    """Simple peak-count frequency: count peaks in lowpassed bipolar channels."""
    from mne.filter import notch_filter, filter_data
    segment = notch_filter(segment, fs, 60, n_jobs=1, verbose="ERROR")
    segment = filter_data(segment, fs, 0.5, 40, n_jobs=1, verbose="ERROR")
    seg = np.array(fcn_getBanana(segment))

    b_lp, a_lp = butter(4, 15.0 / (fs / 2), btype='low')
    freqs = []
    for i in range(seg.shape[0]):
        try:
            ch = filtfilt(b_lp, a_lp, seg[i])
        except ValueError:
            continue
        ch = detrend(ch - np.mean(ch))
        pks, _ = find_peaks(ch, distance=int(0.3 * fs))
        if len(pks) >= 2:
            duration = (pks[-1] - pks[0]) / fs
            if duration > 0:
                freq = (len(pks) - 1) / duration
                if 0.3 < freq < 5.0:
                    freqs.append(freq)
    if freqs:
        return float(np.median(freqs))
    return np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading dataset...")
    dataset = load_dataset()
    print(f"Dataset: {len(dataset)} segments")

    # Collect raw predictions from all three methods
    preds_A = {}
    preds_B = {}
    preds_pk = {}
    subtypes = {}  # mat_name -> 'lpd' or 'gpd'

    for idx, entry in enumerate(dataset):
        mat_name = entry['mat_name']
        data, fs = load_eeg_data(entry)
        if data is None:
            continue

        subtypes[mat_name] = entry['subdir']

        # Method A
        try:
            res_a = pd_detect_alternate(data, fs, pk_detect='apd')
            a_freq = res_a['event_frequency']
            if a_freq is not None and np.isfinite(a_freq):
                preds_A[mat_name] = float(a_freq)
            else:
                preds_A[mat_name] = np.nan
        except Exception:
            preds_A[mat_name] = np.nan

        # Method B (acf_thr=0.10)
        try:
            res_b = pd_detect_pointiness_acf(
                data, fs,
                method='pointiness',
                lowpass_hz=15,
                smoothing_sigma=0.02,
                acf_min_lag=0.4,
                acf_peak_threshold=0.10,
                peak_height_frac=0.3,
            )
            b_freq = res_b['event_frequency']
            if b_freq is not None and np.isfinite(b_freq):
                preds_B[mat_name] = float(b_freq)
            else:
                preds_B[mat_name] = np.nan
        except Exception:
            preds_B[mat_name] = np.nan

        # Peak-count frequency
        try:
            preds_pk[mat_name] = peak_count_frequency(data, fs)
        except Exception:
            preds_pk[mat_name] = np.nan

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} segments")

    print(f"Finished computing raw predictions.")
    print(f"  A valid: {sum(1 for v in preds_A.values() if np.isfinite(v))}")
    print(f"  B valid: {sum(1 for v in preds_B.values() if np.isfinite(v))}")
    print(f"  PkCount valid: {sum(1 for v in preds_pk.values() if np.isfinite(v))}")

    # All mat_names
    all_names = set(preds_A.keys()) | set(preds_B.keys()) | set(preds_pk.keys())

    # -----------------------------------------------------------------------
    # Strategy (a): r2_A_anchor_double
    # If B_freq < A_freq * 0.6, double B_freq (A says it's subharmonic)
    # -----------------------------------------------------------------------
    preds_a_strat = {}
    for mn in all_names:
        af = preds_A.get(mn, np.nan)
        bf = preds_B.get(mn, np.nan)
        if np.isfinite(bf) and np.isfinite(af):
            if bf < af * 0.6:
                preds_a_strat[mn] = bf * 2.0
            else:
                preds_a_strat[mn] = bf
        elif np.isfinite(bf):
            preds_a_strat[mn] = bf
        elif np.isfinite(af):
            preds_a_strat[mn] = af
    evaluate_predictions(dataset, preds_a_strat, "r2_A_anchor_double")

    # -----------------------------------------------------------------------
    # Strategy (b): r2_A_anchor_select
    # If |A - B| > 0.5, use A; otherwise use B
    # -----------------------------------------------------------------------
    preds_b_strat = {}
    for mn in all_names:
        af = preds_A.get(mn, np.nan)
        bf = preds_B.get(mn, np.nan)
        if np.isfinite(af) and np.isfinite(bf):
            if abs(af - bf) > 0.5:
                preds_b_strat[mn] = af
            else:
                preds_b_strat[mn] = bf
        elif np.isfinite(af):
            preds_b_strat[mn] = af
        elif np.isfinite(bf):
            preds_b_strat[mn] = bf
    evaluate_predictions(dataset, preds_b_strat, "r2_A_anchor_select")

    # -----------------------------------------------------------------------
    # Strategy (c): r2_A_anchor_harmonic
    # If A/B close to 2 (within 20%), use A; if close to 1, use mean;
    # otherwise use whichever is closer to peak-count freq
    # -----------------------------------------------------------------------
    preds_c_strat = {}
    for mn in all_names:
        af = preds_A.get(mn, np.nan)
        bf = preds_B.get(mn, np.nan)
        pkf = preds_pk.get(mn, np.nan)
        if np.isfinite(af) and np.isfinite(bf) and bf > 0:
            ratio = af / bf
            if abs(ratio - 2.0) / 2.0 < 0.20:
                # A is likely correct, B at subharmonic
                preds_c_strat[mn] = af
            elif abs(ratio - 1.0) < 0.20:
                # Close agreement
                preds_c_strat[mn] = (af + bf) / 2.0
            else:
                # Use whichever is closer to peak-count freq
                if np.isfinite(pkf):
                    if abs(af - pkf) < abs(bf - pkf):
                        preds_c_strat[mn] = af
                    else:
                        preds_c_strat[mn] = bf
                else:
                    preds_c_strat[mn] = af  # default to A
        elif np.isfinite(af):
            preds_c_strat[mn] = af
        elif np.isfinite(bf):
            preds_c_strat[mn] = bf
    evaluate_predictions(dataset, preds_c_strat, "r2_A_anchor_harmonic")

    # -----------------------------------------------------------------------
    # Strategy (d): r2_triple_median
    # Median of (A, B, peak_count) — robust to any one being wrong
    # -----------------------------------------------------------------------
    preds_d_strat = {}
    for mn in all_names:
        vals = []
        for src in [preds_A, preds_B, preds_pk]:
            v = src.get(mn, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            preds_d_strat[mn] = float(np.median(vals))
    evaluate_predictions(dataset, preds_d_strat, "r2_triple_median")

    # -----------------------------------------------------------------------
    # Strategy (e): r2_triple_median_bayesian
    # (d) + bayesian nudge toward 1.5 Hz prior
    # -----------------------------------------------------------------------
    preds_e_strat = {}
    prior_mean = 1.5
    prior_weight = 0.15  # blend 15% toward prior
    for mn in all_names:
        vals = []
        for src in [preds_A, preds_B, preds_pk]:
            v = src.get(mn, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if vals:
            med = float(np.median(vals))
            nudged = med * (1.0 - prior_weight) + prior_mean * prior_weight
            preds_e_strat[mn] = nudged
    evaluate_predictions(dataset, preds_e_strat, "r2_triple_median_bayesian")

    # -----------------------------------------------------------------------
    # Strategy (f): r2_smart_select
    # For LPD: max(A, B, peak_count) * 0.85
    # For GPD: median(A, B, peak_count)
    # -----------------------------------------------------------------------
    preds_f_strat = {}
    for mn in all_names:
        vals = []
        for src in [preds_A, preds_B, preds_pk]:
            v = src.get(mn, np.nan)
            if np.isfinite(v):
                vals.append(v)
        if not vals:
            continue
        pattern = subtypes.get(mn, 'gpd')
        if pattern == 'lpd':
            preds_f_strat[mn] = max(vals) * 0.85
        else:
            preds_f_strat[mn] = float(np.median(vals))
    evaluate_predictions(dataset, preds_f_strat, "r2_smart_select")

    print("\nAll strategies evaluated. Run update_dashboard.py to refresh.")


if __name__ == '__main__':
    main()
