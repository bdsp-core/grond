#!/usr/bin/env python3
"""LRDA per-segment featurizer for the hard-case classifier (Plan A).

Computes a fixed set of features summarizing how confident V1 (W05/NB-Hilbert)
should be on a given LRDA segment. Features include V1 internal signals,
whole-segment and sliding-window spectral peak prominence, hemisphere
asymmetry, per-channel rhythmicity dispersion, and IIIC ambiguity.

CLI: regenerates the feature cache.

    conda run -n morgoth python code/evaluation/lrda_features.py

Output:
    data/labels/independent_expert_v1/lrda_features.csv  (one row per LRDA manifest segment)
"""

import csv
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
OUT_PATH = LABELS_DIR / 'independent_expert_v1' / 'lrda_features.csv'

FEATURE_NAMES = [
    # V1 internals
    'v1_freq', 'v1_dom_side_left', 'v1_if_cv_pass1', 'v1_if_cv_pass2',
    'v1_hemi_var_ratio', 'v1_pass_delta',
    # Whole-segment spectral
    'welch_peak_prominence', 'welch_n_peaks', 'welch_peak_freq',
    'welch_peak_freq_minus_v1_freq',
    # Sliding-window stability
    'rhythm_stability_mean', 'rhythm_stability_std',
    'rhythm_max_min_ratio', 'rhythm_best_window_start',
    # Hemisphere asymmetry
    'lr_prominence_ratio_log', 'lr_prominence_diff', 'dom_hemi_signed_score',
    # Per-channel dispersion (within V1's chosen hemisphere)
    'ch_prominence_std', 'ch_prominence_max_min_ratio', 'ch_top3_v_bot3_ratio',
    # Pre-existing context
    'iiic_plurality_frac', 'iiic_n_votes',
]


def _hilbert_if_cv(sig: np.ndarray) -> tuple[float, float]:
    """Median Hilbert IF + coefficient of variation."""
    if np.std(sig) < 1e-10:
        return float('nan'), 1.0
    analytic = hilbert(sig)
    inst = np.diff(np.unwrap(np.angle(analytic))) * FS / (2 * np.pi)
    mask = (inst > 0.3) & (inst < 4.0)
    valid = inst[mask]
    if len(valid) < 20:
        return float('nan'), 1.0
    med = float(np.median(valid))
    cv = float(np.std(valid) / max(med, 1e-6))
    return med, cv


def _welch_peak(sig: np.ndarray, nperseg: int = 1024) -> tuple[float, float, int]:
    """Largest spectral peak in 0.5-4 Hz: returns (peak_freq, peak_prominence_ratio, n_peaks)."""
    nperseg = min(nperseg, len(sig))
    f, pxx = welch(sig, fs=FS, nperseg=nperseg)
    mask = (f >= 0.5) & (f <= 4.0)
    f_m = f[mask]
    pxx_m = pxx[mask]
    if not len(pxx_m) or np.max(pxx_m) <= 0:
        return float('nan'), 0.0, 0
    peaks, props = find_peaks(pxx_m, prominence=np.max(pxx_m) * 0.05)
    if not len(peaks):
        return float(f_m[np.argmax(pxx_m)]), 0.0, 0
    bp = int(np.argmax(props['prominences']))
    return (
        float(f_m[peaks[bp]]),
        float(props['prominences'][bp]) / np.max(pxx_m),
        int(len(peaks)),
    )


def _per_channel_narrowband_var_ratio(seg_pre: np.ndarray, freq: float, channels: list[int]) -> np.ndarray:
    """For each channel, return narrowband(freq±0.4)/broadband variance ratio."""
    bw = 0.4
    lo = max(freq - bw, 0.1)
    hi = min(freq + bw, FS / 2 - 0.1)
    if lo >= hi:
        return np.zeros(len(channels))
    sos = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos, seg_pre, axis=1)
    out = []
    for ch in channels:
        bv = np.var(seg_pre[ch])
        nv = np.var(seg_nb[ch])
        out.append(nv / bv if bv > 1e-9 else 0.0)
    return np.array(out)


def featurize(seg_bi: np.ndarray, iiic_plurality_frac: float = 0.0,
              iiic_n_votes: float = 0.0) -> dict:
    """Compute the feature dict for one segment. Pure function, no I/O."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)
    sos1 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_pre, axis=1)

    # V1 hemisphere selection
    ls = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    dom_chs = LEFT_CHS if ls >= rs else RIGHT_CHS
    dom_side_left = 1 if ls >= rs else 0
    hemi_var_ratio = ls / rs if rs > 1e-9 else 999.0
    if hemi_var_ratio < 1.0:
        hemi_var_ratio = 1.0 / max(hemi_var_ratio, 1e-9)  # always >= 1

    # V1 pass-1 freq
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top3 = dom_chs[np.argsort(powers)[::-1][:3]]
    sig_p1 = np.mean(seg_n[top3], axis=0)
    f1, cv1 = _hilbert_if_cv(sig_p1)
    if not np.isfinite(f1):
        f1 = 1.5

    # V1 pass-2 narrowband
    bw = 0.4
    lo = max(f1 - bw, 0.1)
    hi = min(f1 + bw, FS / 2 - 0.1)
    if lo < hi:
        sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
        seg_nb = sosfiltfilt(sos2, seg_pre, axis=1)
        powers2 = np.array([np.var(seg_nb[ch]) for ch in dom_chs])
        top3_2 = dom_chs[np.argsort(powers2)[::-1][:3]]
        sig_p2 = np.mean(seg_nb[top3_2], axis=0)
        f2, cv2 = _hilbert_if_cv(sig_p2)
    else:
        f2, cv2 = f1, 1.0
    if not np.isfinite(f2):
        f2 = f1
    v1_freq = float(np.clip(f2, 0.25, 4.0))
    v1_pass_delta = float(abs(f1 - f2))

    # Whole-segment Welch peak prominence on dominant top-3 mean
    sig_full = np.mean(seg_pre[top3], axis=0)
    welch_freq, welch_prom, welch_n = _welch_peak(sig_full)

    # Sliding-window stability
    win_samp = int(3.0 * FS)
    hop_samp = int(0.5 * FS)
    n_samp = seg_pre.shape[1]
    proms = []
    best_prom = -1
    best_start = 0
    for ws in range(0, n_samp - win_samp + 1, hop_samp):
        we = ws + win_samp
        sig_w = np.mean(seg_pre[top3, ws:we], axis=0)
        _, p, _ = _welch_peak(sig_w, nperseg=256)
        proms.append(p)
        if p > best_prom:
            best_prom = p
            best_start = ws
    proms_arr = np.array(proms) if proms else np.array([0.0])
    rhythm_max_min = (np.max(proms_arr) / max(np.min(proms_arr), 1e-6)) if len(proms_arr) > 1 else 1.0

    # Hemisphere asymmetry by spectral peak prominence
    sig_left = np.mean(seg_pre[LEFT_CHS][np.argsort([np.var(seg_pre[ch]) for ch in LEFT_CHS])[::-1][:3]], axis=0)
    sig_right = np.mean(seg_pre[RIGHT_CHS][np.argsort([np.var(seg_pre[ch]) for ch in RIGHT_CHS])[::-1][:3]], axis=0)
    _, prom_l, _ = _welch_peak(sig_left)
    _, prom_r, _ = _welch_peak(sig_right)
    lr_diff = abs(prom_l - prom_r)
    lr_log_ratio = np.log((prom_l + 1e-3) / (prom_r + 1e-3))
    dom_signed = (prom_l - prom_r) if dom_side_left else (prom_r - prom_l)
    # positive means V1's hemisphere choice agrees with the spectral-prominence choice
    # negative means V1 went one way but spectral says the other

    # Per-channel narrowband variance ratio dispersion within V1's hemisphere
    ch_ratios = _per_channel_narrowband_var_ratio(seg_pre, v1_freq, list(dom_chs))
    sorted_r = np.sort(ch_ratios)
    if len(sorted_r) >= 6:
        top3_mean = float(np.mean(sorted_r[-3:]))
        bot3_mean = float(np.mean(sorted_r[:3]))
        top_bot_ratio = top3_mean / max(bot3_mean, 1e-6)
    else:
        top_bot_ratio = 1.0
    ch_max_min = (np.max(ch_ratios) / max(np.min(ch_ratios), 1e-6)) if len(ch_ratios) > 1 else 1.0

    return {
        'v1_freq': v1_freq,
        'v1_dom_side_left': dom_side_left,
        'v1_if_cv_pass1': cv1,
        'v1_if_cv_pass2': cv2,
        'v1_hemi_var_ratio': hemi_var_ratio,
        'v1_pass_delta': v1_pass_delta,
        'welch_peak_prominence': welch_prom,
        'welch_n_peaks': welch_n,
        'welch_peak_freq': welch_freq if np.isfinite(welch_freq) else 1.5,
        'welch_peak_freq_minus_v1_freq': (welch_freq - v1_freq) if np.isfinite(welch_freq) else 0.0,
        'rhythm_stability_mean': float(np.mean(proms_arr)),
        'rhythm_stability_std': float(np.std(proms_arr)),
        'rhythm_max_min_ratio': float(rhythm_max_min),
        'rhythm_best_window_start': float(best_start / FS),
        'lr_prominence_ratio_log': float(lr_log_ratio),
        'lr_prominence_diff': float(lr_diff),
        'dom_hemi_signed_score': float(dom_signed),
        'ch_prominence_std': float(np.std(ch_ratios)),
        'ch_prominence_max_min_ratio': float(ch_max_min),
        'ch_top3_v_bot3_ratio': float(top_bot_ratio),
        'iiic_plurality_frac': float(iiic_plurality_frac),
        'iiic_n_votes': float(iiic_n_votes),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_DIR / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    print(f'Featurizing {len(rows)} LRDA manifest segments...')

    out_rows = []
    for i, row in enumerate(rows):
        mf = row['mat_file']
        seg = load_segment(mf)
        if seg is None:
            print(f'  WARNING: {mf} unloadable; skipping')
            continue
        try:
            iiic_p = float(row.get('iiic_plurality_frac') or 0)
        except ValueError:
            iiic_p = 0.0
        try:
            iiic_n = float(row.get('iiic_n_votes') or 0)
        except ValueError:
            iiic_n = 0.0
        feats = featurize(seg, iiic_plurality_frac=iiic_p, iiic_n_votes=iiic_n)
        feats['mat_file'] = mf
        feats['patient_id'] = row['patient_id']
        out_rows.append(feats)
        if (i + 1) % 25 == 0:
            print(f'  {i+1}/{len(rows)}')

    fields = ['mat_file', 'patient_id'] + FEATURE_NAMES
    with open(OUT_PATH, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f'Wrote {OUT_PATH}  ({len(out_rows)} rows)')


if __name__ == '__main__':
    main()
