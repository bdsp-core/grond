#!/usr/bin/env python3
"""LRDA laterality feature extractor (Plan A.2 — learned laterality detector).

Computes ~15 per-segment features summarizing left vs right hemisphere
content. The features are then fed to a binary classifier
(train_lrda_laterality_classifier.py) that learns which discriminator
metric to trust on each segment.

Features include the four canonical hand-coded discriminators (variance,
envelope amplitude, spectral peak prominence, narrowband variance),
robustness signals (pass-1-vs-pass-2 agreement, top-3-vs-all-8
consistency), artifact-detection signals (low-frequency drift), and
per-channel rhythmicity dispersion.

Output: data/labels/independent_expert_v1/lrda_laterality_features.csv

    conda run -n morgoth python code/evaluation/lrda_laterality_features.py
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS, w05_estimate_freq  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
OUT_PATH = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'

FEATURE_NAMES = [
    'pass1_var_log_ratio',          # log(left_var / right_var) at 0.5-3.5 Hz
    'pass2_env_log_ratio',          # log(left_env_amp / right_env_amp) at est_freq +- 0.4
    'narrowband_var_log_ratio',     # log(left_nb_var / right_nb_var) at est_freq +- 0.3
    'top3_var_log_ratio',           # log of top-3 channels variance ratio per hemisphere
    'spectral_peak_prom_log_ratio', # log of Welch peak prominence ratio
    'pass1_pass2_agreement',        # 1 if pass-1 and pass-2 picked the same side, else 0
    'top3_uniform_agreement',       # 1 if top-3 and all-8 mean give same side, else 0
    'left_artifact_score',          # left LF drift power / left mid-band power
    'right_artifact_score',         # right LF drift power / right mid-band power
    'left_ch_dispersion',           # std of per-channel narrowband variance (left)
    'right_ch_dispersion',          # std of per-channel narrowband variance (right)
    'est_freq',                     # the estimated frequency itself (low-freq cases may be more robust)
    'est_freq_if_cv',               # Hilbert IF coefficient of variation (estimator confidence)
    'left_max_ch_var',              # log of max single-channel variance on left
    'right_max_ch_var',             # log of max single-channel variance on right
    'lr_max_ch_log_ratio',          # log of max-channel-variance ratio left/right
]


def _safe_log_ratio(a: float, b: float) -> float:
    eps = 1e-9
    return float(np.log(max(a, eps) / max(b, eps)))


def _hilbert_if_cv(sig: np.ndarray) -> tuple[float, float]:
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


def _welch_peak_prominence(sig: np.ndarray, fs: int = FS) -> float:
    """Max spectral peak prominence in 0.5-4 Hz, normalized by max spectrum value."""
    n = min(len(sig), 1024)
    f, pxx = welch(sig, fs=fs, nperseg=n)
    mask = (f >= 0.5) & (f <= 4.0)
    pxx_m = pxx[mask]
    if not len(pxx_m) or np.max(pxx_m) <= 0:
        return 0.0
    peaks, props = find_peaks(pxx_m, prominence=np.max(pxx_m) * 0.05)
    if not len(peaks):
        return 0.0
    return float(np.max(props['prominences']) / np.max(pxx_m))


def featurize_lat(seg_bi: np.ndarray) -> dict:
    """Compute laterality features for one bipolar EEG segment."""
    sos_pre = butter(4, [0.3 / (FS / 2), 5.0 / (FS / 2)], btype='bandpass', output='sos')
    seg_pre = sosfiltfilt(sos_pre, seg_bi, axis=1)

    sos1 = butter(4, [0.5 / (FS / 2), 3.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_n = sosfiltfilt(sos1, seg_pre, axis=1)

    # Pass-1 variance per hemisphere
    ls1 = float(np.mean([np.var(seg_n[ch]) for ch in LEFT_CHS]))
    rs1 = float(np.mean([np.var(seg_n[ch]) for ch in RIGHT_CHS]))
    pass1_left_dom = ls1 >= rs1

    # Estimate frequency (using same method as W05)
    dom_chs = LEFT_CHS if pass1_left_dom else RIGHT_CHS
    powers = np.array([np.var(seg_n[ch]) for ch in dom_chs])
    top3 = dom_chs[np.argsort(powers)[::-1][:3]]
    sig_p1 = np.mean(seg_n[top3], axis=0)
    f1, cv1 = _hilbert_if_cv(sig_p1)
    if not np.isfinite(f1):
        f1 = 1.5
    est_freq = float(np.clip(f1, 0.25, 4.0))

    # Pass-2 narrowband at est_freq +- 0.4
    bw = 0.4
    lo = max(est_freq - bw, 0.1)
    hi = min(est_freq + bw, FS / 2 - 0.1)
    sos2 = butter(3, [lo / (FS / 2), hi / (FS / 2)], btype='bandpass', output='sos')
    seg_nb = sosfiltfilt(sos2, seg_pre, axis=1)
    # Pass-2 envelope amplitude per hemisphere
    ls2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in LEFT_CHS]))
    rs2 = float(np.mean([np.mean(np.abs(hilbert(seg_nb[ch]))) for ch in RIGHT_CHS]))
    pass2_left_dom = ls2 >= rs2

    # Narrowband variance per hemisphere (slightly different from envelope)
    ls_nb_var = float(np.mean([np.var(seg_nb[ch]) for ch in LEFT_CHS]))
    rs_nb_var = float(np.mean([np.var(seg_nb[ch]) for ch in RIGHT_CHS]))

    # Top-3 channel variance per hemisphere (the freq estimator's input)
    left_top3_var = float(np.mean(np.sort([np.var(seg_n[ch]) for ch in LEFT_CHS])[::-1][:3]))
    right_top3_var = float(np.mean(np.sort([np.var(seg_n[ch]) for ch in RIGHT_CHS])[::-1][:3]))
    top3_left_dom = left_top3_var >= right_top3_var

    # Spectral peak prominence per hemisphere (top-3 channel mean signal)
    left_top3 = LEFT_CHS[np.argsort([np.var(seg_pre[ch]) for ch in LEFT_CHS])[::-1][:3]]
    right_top3 = RIGHT_CHS[np.argsort([np.var(seg_pre[ch]) for ch in RIGHT_CHS])[::-1][:3]]
    left_prom = _welch_peak_prominence(np.mean(seg_pre[left_top3], axis=0))
    right_prom = _welch_peak_prominence(np.mean(seg_pre[right_top3], axis=0))

    # Artifact: low-freq drift power / mid-band power per hemisphere
    sos_lf = butter(4, [0.3 / (FS / 2), 0.5 / (FS / 2)], btype='bandpass', output='sos')
    seg_lf = sosfiltfilt(sos_lf, seg_bi, axis=1)
    left_lf_pow = float(np.mean([np.var(seg_lf[ch]) for ch in LEFT_CHS]))
    right_lf_pow = float(np.mean([np.var(seg_lf[ch]) for ch in RIGHT_CHS]))
    left_artifact = left_lf_pow / max(ls1, 1e-9)
    right_artifact = right_lf_pow / max(rs1, 1e-9)

    # Per-channel narrowband-VE dispersion (within each hemisphere)
    left_ch_var = np.array([np.var(seg_nb[ch]) for ch in LEFT_CHS])
    right_ch_var = np.array([np.var(seg_nb[ch]) for ch in RIGHT_CHS])
    left_disp = float(np.std(left_ch_var) / max(np.mean(left_ch_var), 1e-9))
    right_disp = float(np.std(right_ch_var) / max(np.mean(right_ch_var), 1e-9))

    # Max-single-channel variance per hemisphere (captures focal rhythm)
    left_max_ch = float(np.max(left_ch_var))
    right_max_ch = float(np.max(right_ch_var))

    return {
        'pass1_var_log_ratio': _safe_log_ratio(ls1, rs1),
        'pass2_env_log_ratio': _safe_log_ratio(ls2, rs2),
        'narrowband_var_log_ratio': _safe_log_ratio(ls_nb_var, rs_nb_var),
        'top3_var_log_ratio': _safe_log_ratio(left_top3_var, right_top3_var),
        'spectral_peak_prom_log_ratio': _safe_log_ratio(left_prom + 0.01, right_prom + 0.01),
        'pass1_pass2_agreement': float(pass1_left_dom == pass2_left_dom),
        'top3_uniform_agreement': float(top3_left_dom == pass1_left_dom),
        'left_artifact_score': left_artifact,
        'right_artifact_score': right_artifact,
        'left_ch_dispersion': left_disp,
        'right_ch_dispersion': right_disp,
        'est_freq': est_freq,
        'est_freq_if_cv': cv1,
        'left_max_ch_var': float(np.log(max(left_max_ch, 1e-9))),
        'right_max_ch_var': float(np.log(max(right_max_ch, 1e-9))),
        'lr_max_ch_log_ratio': _safe_log_ratio(left_max_ch, right_max_ch),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_DIR / 'manifest.csv') as f:
        rows = list(csv.DictReader(f))
    print(f'Featurizing laterality on {len(rows)} LRDA manifest segments...')

    out_rows = []
    for i, row in enumerate(rows):
        mf = row['mat_file']
        seg = load_segment(mf)
        if seg is None:
            print(f'  WARNING: {mf} unloadable; skipping')
            continue
        feats = featurize_lat(seg)
        feats['mat_file'] = mf
        feats['patient_id'] = row['patient_id']
        out_rows.append(feats)
        if (i + 1) % 50 == 0:
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
