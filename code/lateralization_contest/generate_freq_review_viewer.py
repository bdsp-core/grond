#!/usr/bin/env python3
"""Generate RDA frequency review viewer using V22's per-hemisphere estimates.

Shows V22's estimated frequency as default (accepted with Enter).
Displays narrowband overlay at that frequency for visual confirmation.
Frequency buttons available if MW disagrees.

Uses the build_html() infrastructure from generate_lrda_labeler.py.

Usage:
    conda run -n morgoth python code/lateralization_contest/generate_freq_review_viewer.py
"""
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend, hilbert, sosfiltfilt

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_pointiness_acf import fcn_getBanana

# Import the labeler infrastructure
sys.path.insert(0, str(CODE_DIR / 'generators' / 'labeling'))
from generate_lrda_labeler import (
    FS, N_SAMPLES, FREQ_GRID, FREQ_BUTTONS, NB_BW,
    LEFT_CHANNELS, RIGHT_CHANNELS, BIPOLAR_CHANNELS,
    downsample, nvo_bandpass_grid, compute_button_filtered, build_html,
    narrowband_filter,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'

LEFT_CHS = np.array(LEFT_CHANNELS)
RIGHT_CHS = np.array(RIGHT_CHANNELS)


def v22_estimate(seg_bi):
    """V22/V23: envelope amplitude for lateralization, top-3 Hilbert for frequency.

    Returns (left_score, right_score, left_freq, right_freq, best_freq).
    """
    seg_f = _prefilter(seg_bi)
    sos_n = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_narrow = sosfiltfilt(sos_n, seg_f, axis=1)

    # Score: mean envelope per hemisphere (all channels)
    ls = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in LEFT_CHS]))
    rs = float(np.mean([np.mean(np.abs(hilbert(seg_f[ch]))) for ch in RIGHT_CHS]))

    # Freq: Hilbert CV on top-3 per hemisphere (narrowband)
    def hemi_freq(chs):
        powers = np.array([np.var(seg_narrow[ch]) for ch in chs])
        top3 = chs[np.argsort(powers)[::-1][:3]]
        ch_freqs = []
        for ch in top3:
            sig = seg_narrow[ch]
            if np.std(sig) < 1e-10:
                continue
            analytic = hilbert(sig)
            inst_freq = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
            mask = (inst_freq > 0.3) & (inst_freq < 4.0)
            valid = inst_freq[mask]
            if len(valid) >= 20:
                ch_freqs.append(float(np.median(valid)))
        return float(np.median(ch_freqs)) if ch_freqs else np.nan

    lf = hemi_freq(LEFT_CHS)
    rf = hemi_freq(RIGHT_CHS)

    # Best freq from dominant hemisphere
    if ls >= rs:
        freq = lf if np.isfinite(lf) else rf
    else:
        freq = rf if np.isfinite(rf) else lf

    return ls, rs, lf, rf, freq


def _prefilter(seg, lo=0.3, hi=5.0):
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    return sosfiltfilt(sos, seg, axis=1)


def load_and_preprocess(mat_file):
    """Load monopolar file, convert to bipolar, preprocess."""
    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    raw = mat[dk].astype(np.float64)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    if raw.shape[0] >= 19:
        seg_bi = np.array(fcn_getBanana(raw[:19, :N_SAMPLES]), dtype=np.float64)
    elif raw.shape[0] == 18:
        seg_bi = raw[:18, :N_SAMPLES]
    else:
        return None
    from mne.filter import notch_filter, filter_data
    seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
    seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')
    return seg_bi


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RDA Frequency Review Viewer (V22 estimates)")
    print("=" * 70)

    # Load segment labels
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    # Select RDA segments — start with highest-quality LRDA, then GRDA
    rda = sl[
        (sl['subtype'].isin(['lrda', 'grda'])) &
        (~sl['excluded'].fillna(False).astype(bool)) &
        (sl['n_votes'] >= 3)  # at least 3 votes for reasonable confidence
    ].copy()

    # Sort: LRDA first, then GRDA, by n_votes descending
    lrda = rda[rda['subtype'] == 'lrda'].sort_values('n_votes', ascending=False)
    grda = rda[rda['subtype'] == 'grda'].sort_values('n_votes', ascending=False)
    selected = pd.concat([lrda, grda], ignore_index=True)

    print(f"Cases: {len(lrda)} LRDA + {len(grda)} GRDA = {len(selected)}")

    # Process cases
    print("Processing cases...")
    cases_data = []
    n_skip = 0
    DISPLAY_HP, DISPLAY_LP = 0.3, 5.0
    b_bp, a_bp = butter(4, [DISPLAY_HP / (FS / 2), DISPLAY_LP / (FS / 2)], btype='bandpass')

    for i, (_, row) in enumerate(selected.iterrows()):
        mat_file = row['mat_file']
        subtype = row['subtype']
        n_votes = int(row['n_votes']) if pd.notna(row['n_votes']) else 0
        prev_freq = float(row['mw_freq']) if pd.notna(row.get('mw_freq')) else None
        if prev_freq is None and pd.notna(row.get('auto_freq')):
            prev_freq = float(row['auto_freq'])

        seg_bi = load_and_preprocess(mat_file)
        if seg_bi is None or seg_bi.shape != (18, N_SAMPLES):
            n_skip += 1
            continue

        # V22 frequency estimate
        try:
            ls, rs, lf, rf, v22_freq = v22_estimate(seg_bi)
        except:
            n_skip += 1
            continue

        # Bandpass for display
        seg_display = np.zeros_like(seg_bi)
        for ch in range(18):
            try:
                seg_display[ch] = filtfilt(b_bp, a_bp, seg_bi[ch])
            except ValueError:
                seg_display[ch] = seg_bi[ch]

        # NVO for VE matrix
        try:
            ve_matrix, nvo_freq, best_ve = nvo_bandpass_grid(seg_display)
        except:
            n_skip += 1
            continue

        # Use V22 frequency as default, fall back to NVO
        est_freq = v22_freq if np.isfinite(v22_freq) else nvo_freq

        # Snap to nearest button frequency
        est_freq = min(FREQ_BUTTONS, key=lambda f: abs(f - est_freq))

        # Precompute narrowband overlays
        btn_filtered = compute_button_filtered(seg_display)
        nb_signals = {}
        for freq_str, ch_dict in btn_filtered.items():
            ch_data = {}
            for ch_str, arr in ch_dict.items():
                ch_data[ch_str] = [round(float(v), 1) for v in arr]
            nb_signals[freq_str] = ch_data

        # Laterality from V22
        lat_index = (rs - ls) / (ls + rs + 1e-12)
        est_laterality = 'left' if lat_index < -0.15 else ('right' if lat_index > 0.15 else 'bilateral')

        pid = row.get('patient_id', mat_file.replace('.mat', ''))

        case = {
            'patient_id': str(pid),
            'segment_id': mat_file.replace('.mat', ''),
            'est_freq': round(float(est_freq), 2),
            'est_laterality': est_laterality,
            'lat_index': round(float(lat_index), 3),
            'annotated': False,
            'subtype': subtype,
            'rda_frac': 1.0,
            'n_experts': n_votes,
            'ensemble_prob': 1.0,
            'prev_freq': round(float(prev_freq), 2) if prev_freq and prev_freq > 0 else None,
            'eeg_data': downsample(seg_display, 500),
            've_matrix': [[round(float(v), 3) for v in ve_matrix[fi]] for fi in range(len(FREQ_GRID))],
            'freq_grid': [round(f, 2) for f in FREQ_GRID],
            'best_ve': [round(float(v), 4) for v in best_ve],
            'nb_signals': nb_signals,
        }
        cases_data.append(case)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(selected)} ({time.time() - t0:.0f}s)")

    print(f"  Total: {len(cases_data)} (skipped {n_skip})")

    # Split into batches
    BATCH_SIZE = 300
    n_batches = (len(cases_data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Splitting into {n_batches} batches of ~{BATCH_SIZE} cases...")

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(cases_data))
        batch = cases_data[start:end]

        print(f"  Building batch {batch_idx + 1}/{n_batches} ({len(batch)} cases)...")
        html = build_html(batch)

        # Patch title and storage key
        html = html.replace(
            'LRDA Labeling Tool — Frequency, Laterality &amp; Wave Morphology',
            f'RDA Freq Review (V22) — Batch {batch_idx + 1}/{n_batches} ({len(batch)} cases)')
        html = html.replace("'lrda_labeler_v1'", "'rda_freq_review_v22'")

        # Hide laterality and wave annotation buttons (frequency only)
        html = html.replace('id="laterality-buttons"',
                             'id="laterality-buttons" style="display:none"')
        html = html.replace('id="wave-tools"',
                             'id="wave-tools" style="display:none"')

        # Add subtype + n_experts info
        html = html.replace(
            "document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;",
            """document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;
    const infoExtra = document.getElementById('info-extra');
    if (infoExtra) {
      const sub = (c.subtype || '').toUpperCase();
      const prevF = c.prev_freq ? c.prev_freq.toFixed(2) + ' Hz' : 'none';
      infoExtra.innerHTML = '<span style="color:#ff8844;font-weight:bold">' + sub + '</span>'
        + ' | Prev freq: <span style="color:#cc8833">' + prevF + '</span>'
        + ' | N=' + (c.n_experts || '1');
    }"""
        )

        # Insert info-extra element
        html = html.replace(
            '<div id="info-laterality"',
            '<div id="info-extra" style="font-size:13px;color:#aaa;margin-top:4px"></div><div id="info-laterality"')

        out_path = OUT_DIR / f'rda_freq_review_v22_batch{batch_idx + 1}.html'
        with open(str(out_path), 'w') as f:
            f.write(html)
        print(f"  Saved: {out_path}")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
