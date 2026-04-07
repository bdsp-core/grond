"""
Generate interactive RDA frequency annotation viewer.

Selects high-quality LRDA (>=5 experts) and GRDA (>=10 experts) cases,
pre-scores frequency using NVO bandpass VE, and builds the same interactive
canvas-based viewer used for LRDA labeling (with live narrowband overlay,
frequency buttons, accept/reject).

Cases sorted by vote agreement (highest first).

Usage:
    conda run -n foe python code/generate_rda_freq_viewer.py
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

# Import everything from the LRDA labeler — same processing + HTML builder
from generate_lrda_labeler import (
    FS, N_SAMPLES, FREQ_GRID, FREQ_BUTTONS, NB_BW,
    LEFT_CHANNELS, RIGHT_CHANNELS, MIDLINE_CHANNELS,
    BIPOLAR_CHANNELS,
    load_segment, downsample,
    nvo_bandpass_grid, compute_button_filtered,
    build_html,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RDA Frequency Viewer Generator (Interactive)")
    print("  Uses LRDA labeler canvas viewer with NVO + narrowband overlay")
    print("=" * 70)

    # ── Load patients ──
    df_pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_pat['patient_id'] = df_pat['patient_id'].astype(str)
    df_pat['n_expert_votes'] = pd.to_numeric(
        df_pat['n_expert_votes'], errors='coerce').fillna(0).astype(int)
    df_pat['vote_agreement'] = pd.to_numeric(
        df_pat['vote_agreement'], errors='coerce').fillna(0)

    rda = df_pat[
        (df_pat['subtype'].isin(['lrda', 'grda'])) &
        (df_pat['excluded'] != True)
    ].copy()

    # ── Select cases ──
    # LRDA: >=5 experts, all available
    lrda = rda[(rda['subtype'] == 'lrda') & (rda['n_expert_votes'] >= 5)].copy()
    lrda = lrda.sort_values('vote_agreement', ascending=False)

    # GRDA: >=10 experts, top 500 by agreement
    grda = rda[(rda['subtype'] == 'grda') & (rda['n_expert_votes'] >= 10)].copy()
    grda = grda.sort_values('vote_agreement', ascending=False).head(500)

    print(f"\nLRDA candidates (>=5 experts): {len(lrda)}")
    print(f"GRDA candidates (>=10 experts, top 500): {len(grda)}")

    # Exclude cases that already have MW frequency annotations
    # The canonical source is annotations.csv (all labels merged there)
    df_ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    mw_rda_ann = df_ann[
        (df_ann['rater'] == 'MW') &
        (df_ann['frequency_hz'].notna()) &
        (df_ann['frequency_hz'] > 0)
    ]
    already_labeled = set(mw_rda_ann['patient_id'].astype(str))
    # Also add segment_id matches (some patient_ids are segment-level IDs)
    already_labeled.update(set(mw_rda_ann['segment_id'].astype(str)))

    print(f"Already labeled by MW (from annotations.csv): {len(already_labeled)}")

    lrda = lrda[~lrda['patient_id'].isin(already_labeled)]
    grda = grda[~grda['patient_id'].isin(already_labeled)]
    print(f"After excluding: LRDA={len(lrda)}, GRDA={len(grda)}")

    # Combine, sorted by agreement (highest first)
    selected = pd.concat([lrda, grda], ignore_index=True)
    selected = selected.sort_values('vote_agreement', ascending=False).reset_index(drop=True)
    print(f"Total to generate: {len(selected)}")

    # ── Get segment files ──
    df_seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_seg['patient_id'] = df_seg['patient_id'].astype(str)

    # ── Process cases ──
    print(f"\nProcessing cases (NVO + narrowband precomputation)...")
    cases_data = []
    n_skipped = 0

    # Bandpass 0.3-5Hz: removes baseline drift AND high-freq noise
    DISPLAY_LP = 5.0
    DISPLAY_HP = 0.3
    b_bp, a_bp = butter(4, [DISPLAY_HP / (FS / 2), DISPLAY_LP / (FS / 2)], btype='bandpass')

    for i, (_, row) in enumerate(selected.iterrows()):
        pid = str(row['patient_id'])
        subtype = row['subtype']
        agreement = row['vote_agreement']
        n_experts = row['n_expert_votes']

        # Find EEG file
        pid_segs = df_seg[df_seg['patient_id'] == pid]
        mat_file = None
        montage = 'monopolar'
        for _, sr in pid_segs.iterrows():
            p = EEG_DIR / sr['mat_file']
            if p.exists():
                mat_file = sr['mat_file']
                montage = sr.get('montage', 'monopolar')
                break
        if mat_file is None:
            # Try default name
            p = EEG_DIR / f'{pid}_seg000.mat'
            if p.exists():
                mat_file = f'{pid}_seg000.mat'
                montage = 'bipolar'
        if mat_file is None:
            n_skipped += 1
            continue

        # Auto-detect montage from channel count (many IIIC files are already bipolar)
        _path = EEG_DIR / mat_file
        try:
            _mat = sio.loadmat(str(_path))
            _dk = [k for k in _mat if not k.startswith('_')][0]
            n_ch = min(_mat[_dk].shape)
        except Exception:
            n_skipped += 1
            continue
        auto_montage = 'monopolar' if n_ch >= 19 else 'bipolar'
        seg_bi = load_segment(mat_file, auto_montage)
        if seg_bi is None or seg_bi.shape != (18, N_SAMPLES):
            n_skipped += 1
            continue

        # Lowpass at 5Hz for display
        seg_display = np.zeros_like(seg_bi)
        for ch in range(18):
            try:
                seg_display[ch] = filtfilt(b_bp, a_bp, seg_bi[ch])
            except ValueError:
                seg_display[ch] = seg_bi[ch]

        # NVO bandpass VE search on the lowpassed signal
        try:
            ve_matrix, best_freq, best_ve = nvo_bandpass_grid(seg_display)
        except Exception:
            n_skipped += 1
            continue

        # Precompute narrowband filtered signals
        btn_filtered = compute_button_filtered(seg_display)

        # Serialize filtered signals
        nb_signals = {}
        for freq_str, ch_dict in btn_filtered.items():
            ch_data = {}
            for ch_str, arr in ch_dict.items():
                ch_data[ch_str] = [round(float(v), 1) for v in arr]
            nb_signals[freq_str] = ch_data

        # Laterality estimate
        left_ve = np.mean(best_ve[LEFT_CHANNELS])
        right_ve = np.mean(best_ve[RIGHT_CHANNELS])
        denom = left_ve + right_ve
        lat_index = (right_ve - left_ve) / denom if denom > 0 else 0.0
        est_laterality = ('left' if lat_index < -0.15
                          else ('right' if lat_index > 0.15 else 'bilateral'))

        case = {
            'patient_id': pid,
            'segment_id': f'{pid}_seg000',
            'est_freq': round(float(best_freq), 2),
            'est_laterality': est_laterality,
            'lat_index': round(float(lat_index), 3),
            'annotated': False,
            'subtype': subtype,
            'agreement': round(float(agreement), 3),
            'n_experts': int(n_experts),
            'eeg_data': downsample(seg_display, 500),
            've_matrix': [[round(float(v), 3) for v in ve_matrix[fi]]
                          for fi in range(len(FREQ_GRID))],
            'freq_grid': [round(f, 2) for f in FREQ_GRID],
            'best_ve': [round(float(v), 4) for v in best_ve],
            'nb_signals': nb_signals,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(selected)} processed ({elapsed:.0f}s)")

    n_lrda = sum(1 for c in cases_data if c.get('subtype') == 'lrda')
    n_grda = sum(1 for c in cases_data if c.get('subtype') == 'grda')
    print(f"\n  Total: {len(cases_data)} ({n_lrda} LRDA + {n_grda} GRDA), skipped {n_skipped}")

    # ── Build HTML ──
    print("\nBuilding HTML viewer...")
    html = build_html(cases_data)

    # Patch title and storage key to distinguish from LRDA-only labeler
    html = html.replace(
        'LRDA Labeling Tool — Frequency, Laterality &amp; Wave Morphology',
        'RDA Frequency Viewer — LRDA + GRDA (sorted by expert agreement)')
    html = html.replace("'lrda_labeler_v1'", "'rda_freq_viewer_v1'")

    # Hide laterality buttons — subtype already determined by expert consensus
    html = html.replace('id="laterality-buttons"',
                         'id="laterality-buttons" style="display:none"')

    # Inject subtype display into the show() function
    # Add after the info-laterality update
    html = html.replace(
        "document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;",
        ("document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;\n"
         "    if (document.getElementById('info-subtype')) {\n"
         "      const st = c.subtype || '';\n"
         "      const el = document.getElementById('info-subtype');\n"
         "      el.textContent = st.toUpperCase() + ' (expert consensus)';\n"
         "      el.style.color = st === 'lrda' ? '#88aaff' : '#ff8844';\n"
         "    }")
    )

    # Add subtype element in the info panel
    html = html.replace(
        '<span class="info-item">Laterality: <strong id="info-laterality"',
        '<span class="info-item">Type: <strong id="info-subtype" style="color:#ff8844;">--</strong></span> '
        '<span class="info-item" style="display:none">Laterality: <strong id="info-laterality"'
    )

    out_path = OUT_DIR / 'rda_freq_viewer.html'
    with open(str(out_path), 'w') as f:
        f.write(html)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print(f"  DONE ({elapsed:.0f}s)")
    print(f"  Cases: {len(cases_data)} ({n_lrda} LRDA + {n_grda} GRDA)")
    print(f"  Viewer: {out_path} ({size_mb:.1f} MB)")
    print(f"  Sorted by: vote agreement (highest first)")
    print(f"{'=' * 70}")

    import subprocess
    subprocess.Popen(['open', str(out_path)])
    print("  Opened in browser.")


if __name__ == '__main__':
    main()
