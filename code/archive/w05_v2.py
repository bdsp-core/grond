#!/usr/bin/env python3
"""W05_v2: NB-Hilbert with harmonic-disambiguation pass.

The original W05_DomOnly_IterRefine (in code/generators/labeling/
generate_rda_freq_labeler.py) sub-harmonic-locks on high-frequency LRDA
segments: when the true rhythm is at ~2.5--3 Hz, the pass-1 filter
0.5--3.5 Hz attenuates the fundamental at the band edge while passing
the 1.5 Hz sub-harmonic at full strength. Hilbert's instantaneous
frequency then tracks 1.5 Hz, and pass-2 narrowband at 1.5+/-0.4 Hz
locks the answer in.

This module wraps the original W05 and adds a harmonic-disambiguation
test: after the standard pass-2 frequency f1 is computed, we also try
f2 = 2*f1 (when f2 is in a plausible LRDA range, 1.5--4.0 Hz). For
each candidate (f1, f2) we run a pass-2-style narrowband filter
centered at the candidate, compute the variance-explained on the
dominant hemisphere, and switch to f2 only if VE(f2) is materially
higher than VE(f1) (default ratio threshold = 1.4x).

This is a conservative additive fix: on segments whose true frequency
is genuinely the original answer, VE(f2) will be very low and the
algorithm sticks with f1. On segments where the original answer was
sub-harmonic, VE(f2) will be much higher because the filter at f2
catches the actual signal energy.

Usage:
    # Test on the 7 known disagreement cases:
    conda run -n morgoth python code/evaluation/w05_v2.py --test-disagreement

    # Run on the full LRDA manifest, write predictions JSON:
    conda run -n morgoth python code/evaluation/w05_v2.py --full

    # Programmatic:
    from code.evaluation.w05_v2 import w05_v2_estimate_freq
    f, dom_side, info = w05_v2_estimate_freq(seg_18ch_2000samp)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'

# Pull constants and the v1 implementation from the labeling viewer module.
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import (  # noqa: E402
    FS, LEFT_CHS, RIGHT_CHS,
    w05_estimate_freq as w05_v1_estimate_freq,
    _hilbert_freq_cv,
    load_segment,
)

# Disambiguation parameters
F2_RANGE = (1.5, 4.0)         # only consider doubling when 2*f1 is in this band
VE_RATIO_THRESHOLD = 1.4      # switch to f2 only if VE(f2)/VE(f1) >= this
NB_HALF_BW = 0.3              # narrowband filter +/- this many Hz around the candidate


def _ve_at(seg_bi, freq, channels):
    """Variance-explained-by-narrowband proxy for `seg_bi` at `freq`.

    For each channel in `channels`, bandpass-filter the signal at
    freq +/- NB_HALF_BW, then compute (var(narrowband) / var(broadband)).
    Return the mean across channels.

    Higher value = more of the signal energy lives in the narrowband
    around `freq`.
    """
    lo = max(freq - NB_HALF_BW, 0.1)
    hi = min(freq + NB_HALF_BW, FS / 2 - 0.1)
    if lo >= hi:
        return 0.0
    sos = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')

    ratios = []
    for ch in channels:
        sig = seg_bi[ch]
        var_total = float(np.var(sig))
        if var_total < 1e-10:
            continue
        try:
            nb = sosfiltfilt(sos, sig)
        except Exception:
            continue
        var_nb = float(np.var(nb))
        ratios.append(var_nb / var_total)
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def w05_v2_estimate_freq(seg_bi):
    """W05_v2: NB-Hilbert + harmonic-disambiguation.

    Returns (freq_hz, dom_side, info) where info is a dict with the
    diagnostic fields {f1, f2, ve_f1, ve_f2, switched}.
    """
    f1, dom_side = w05_v1_estimate_freq(seg_bi)

    info = {
        'f1': float(f1),
        'dom_side_v1': dom_side,
        'f2_candidate': None,
        've_f1': None,
        've_f2': None,
        'switched_to_double': False,
    }

    # Only consider doubling when 2*f1 is in the plausible LRDA band.
    f2 = 2.0 * f1
    if not (F2_RANGE[0] <= f2 <= F2_RANGE[1]):
        return f1, dom_side, info

    info['f2_candidate'] = float(f2)

    # Apply the same prefilter the v1 algorithm uses, so the VE comparison
    # is on a comparably preprocessed signal.
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)

    # Identify dominant hemisphere channels exactly as v1 does:
    # broadband 0.5-3.5 Hz, mean-variance lateralization, top-3 channels of dom side.
    sos1 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_pre, axis=1)
    ls = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top3 = dom_chs[np.argsort(powers)[::-1][:3]]

    ve_f1 = _ve_at(seg_pre, f1, top3)
    ve_f2 = _ve_at(seg_pre, f2, top3)
    info['ve_f1'] = ve_f1
    info['ve_f2'] = ve_f2

    # Switch only if f2's narrowband captures meaningfully more variance.
    if ve_f1 > 0 and (ve_f2 / max(ve_f1, 1e-6)) >= VE_RATIO_THRESHOLD:
        info['switched_to_double'] = True
        # When switching, also re-derive dom_side from a narrowband at f2
        # (because v1 picked dom_side using a freq we now think was wrong).
        lo = max(f2 - NB_HALF_BW, 0.1)
        hi = min(f2 + NB_HALF_BW, FS / 2 - 0.1)
        sos_f2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb_f2 = sosfiltfilt(sos_f2, seg_pre, axis=1)
        ls2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb_f2[ch]))) for ch in LEFT_CHS]))
        rs2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb_f2[ch]))) for ch in RIGHT_CHS]))
        new_dom_side = 'left' if ls2 >= rs2 else 'right'
        info['dom_side_v2'] = new_dom_side
        return float(f2), new_dom_side, info

    info['dom_side_v2'] = dom_side
    return f1, dom_side, info


# ---------- Test drivers ----------

def _load_disagreement_cases():
    """Read the 7 LRDA disagreement-manifest segments + the 4-way labels."""
    manifest_csv = (PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'
                    / 'lrda' / 'disagreement_manifest.csv')
    out = []
    with open(manifest_csv) as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def _load_lrda_manifest():
    manifest_csv = (PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'
                    / 'lrda' / 'manifest.csv')
    with open(manifest_csv) as f:
        return [r for r in csv.DictReader(f)]


def test_on_disagreement():
    """Run v2 on the 7 known disagreement cases and print the result."""
    cases = _load_disagreement_cases()
    print(f"Running w05_v2 on the {len(cases)} known disagreement cases...")
    print()

    # Load each rater's labels for context
    raters_freq = {r: {} for r in ('MW', 'SZ', 'TZ', 'ALGO')}
    raters_lat = {r: {} for r in ('MW', 'SZ', 'TZ', 'ALGO')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            rater = row['rater']
            if rater not in ('MW', 'SZ', 'TZ'):
                continue
            mf = row['mat_file']
            if row['label_type'] == 'frequency_hz':
                try:
                    raters_freq[rater][mf] = float(row['value'])
                except ValueError:
                    pass
            elif row['label_type'] == 'laterality':
                v = row['value'].strip().lower()
                if v in ('left', 'right'):
                    raters_lat[rater][mf] = v

    # Pull v1 algorithm predictions from any rater export JSON
    for fp in [
        'data/labels/raw_inputs/independent_expert_v1/TZ/lrda_freq_labeling_results_TZ.json',
        'data/labels/raw_inputs/independent_expert_v1/SZ/rda_freq_labeling_results-2.json',
    ]:
        with open(PROJECT_DIR / fp) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf and sub == 'lrda':
                if mf not in raters_freq['ALGO'] and v.get('w05_freq') is not None:
                    raters_freq['ALGO'][mf] = float(v['w05_freq'])
                if mf not in raters_lat['ALGO'] and v.get('w05_laterality') in ('left', 'right'):
                    raters_lat['ALGO'][mf] = v['w05_laterality']

    print(f"{'mat_file':50s} | "
          f"{'MW':>6s} {'SZ':>6s} {'TZ':>6s} {'V1':>6s} | "
          f"{'V2':>6s} {'switch':>6s}  {'VE(f1)':>7s} {'VE(f2)':>7s}")
    print('-' * 130)

    n_switched = 0
    for case in cases:
        mf = case['mat_file']
        seg = load_segment(mf)
        if seg is None:
            print(f"{mf}: EEG not loadable")
            continue
        f, side, info = w05_v2_estimate_freq(seg)

        mw = raters_freq['MW'].get(mf)
        sz = raters_freq['SZ'].get(mf)
        tz = raters_freq['TZ'].get(mf)
        v1 = raters_freq['ALGO'].get(mf)

        sw = '✓' if info['switched_to_double'] else ''
        if info['switched_to_double']:
            n_switched += 1
        ve_f1 = info.get('ve_f1') or 0.0
        ve_f2 = info.get('ve_f2') or 0.0

        def fmt(v):
            return f"{v:.2f}" if v is not None else '   - '

        print(f"{mf:50s} | "
              f"{fmt(mw):>6s} {fmt(sz):>6s} {fmt(tz):>6s} {fmt(v1):>6s} | "
              f"{f:>6.2f} {sw:>6s}  {ve_f1:>7.4f} {ve_f2:>7.4f}")

    print()
    print(f"Switched to 2*f1 on {n_switched} of {len(cases)} cases.")
    return n_switched


def run_full_manifest(out_path=None):
    """Run v2 on every LRDA manifest segment, save predictions JSON."""
    cases = _load_lrda_manifest()
    print(f"Running w05_v2 on the full LRDA manifest ({len(cases)} segments)...")

    out = {}
    n_switched = 0
    n_skipped = 0
    for i, case in enumerate(cases):
        mf = case['mat_file']
        seg = load_segment(mf)
        if seg is None:
            n_skipped += 1
            continue
        f, side, info = w05_v2_estimate_freq(seg)
        if info['switched_to_double']:
            n_switched += 1
        out[mf] = {
            'mat_file': mf,
            'patient_id': case.get('patient_id', ''),
            'subtype': 'lrda',
            'w05_v1_freq': info['f1'],
            'w05_v1_laterality': info['dom_side_v1'],
            'w05_v2_freq': f,
            'w05_v2_laterality': side,
            'switched_to_double': info['switched_to_double'],
            've_f1': info.get('ve_f1'),
            've_f2': info.get('ve_f2'),
        }
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cases)} processed ({n_switched} switched, {n_skipped} skipped)")

    if out_path is None:
        out_path = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1' / 'algo_v2_lrda_predictions.json'
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out)} predictions to {out_path.relative_to(PROJECT_DIR)}")
    print(f"Switched to 2*f1 on {n_switched} of {len(out)} segments ({100*n_switched/len(out):.1f}%).")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-disagreement', action='store_true',
                   help='Test v2 only on the 7 known LRDA disagreement cases.')
    p.add_argument('--full', action='store_true',
                   help='Run v2 on the full 200-segment LRDA manifest and save predictions.')
    args = p.parse_args()

    if not (args.test_disagreement or args.full):
        p.print_help()
        return

    if args.test_disagreement:
        test_on_disagreement()
    if args.full:
        print()
        run_full_manifest()


if __name__ == '__main__':
    main()
