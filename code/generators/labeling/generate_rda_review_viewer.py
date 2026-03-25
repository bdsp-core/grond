"""
RDA Review Viewer — Re-label frequencies with ensemble predictions.

Shows ensemble-predicted frequency as default, MW's previous label,
narrowband overlay at f±0.5Hz, RDA probability vs expert agreement.
GRDA first, then LRDA, ordered by expert RDA agreement.

Usage:
    conda run -n foe python code/generate_rda_review_viewer.py
"""
import sys, json, time, numpy as np, pandas as pd, scipy.io as sio
from pathlib import Path
from scipy.signal import butter, filtfilt, detrend

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from generate_lrda_labeler import (
    FS, N_SAMPLES, FREQ_GRID, FREQ_BUTTONS, NB_BW,
    LEFT_CHANNELS, RIGHT_CHANNELS, BIPOLAR_CHANNELS,
    load_segment, downsample, nvo_bandpass_grid, compute_button_filtered, build_html,
)

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'
OUT_DIR = PROJECT_DIR / 'results'


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RDA Review Viewer — Ensemble-guided frequency re-labeling")
    print("=" * 70)

    # Load ensemble predictions
    pred_path = OUT_DIR / 'rda_contest' / 'ensemble_predictions.json'
    with open(str(pred_path)) as f:
        preds = json.load(f)
    print(f"Ensemble predictions: {len(preds)}")

    # Load patients
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    for c in ['n_expert_votes', 'vote_lrda', 'vote_grda', 'gold_standard_freq']:
        pat[c] = pd.to_numeric(pat[c], errors='coerce').fillna(0)
    pat['rda_frac'] = (pat['vote_lrda'] + pat['vote_grda']) / pat['n_expert_votes'].clip(lower=1)

    # Select RDA cases with predictions
    rda = pat[
        (pat['subtype'].isin(['lrda', 'grda'])) &
        (pat['excluded'] != True) &
        (pat['n_expert_votes'] >= 5) &
        (pat['patient_id'].isin(preds.keys()))
    ].copy()

    # Sort: GRDA first (sorted by rda_frac desc), then LRDA (sorted by rda_frac desc)
    grda = rda[rda['subtype'] == 'grda'].sort_values('rda_frac', ascending=False)
    lrda = rda[rda['subtype'] == 'lrda'].sort_values('rda_frac', ascending=False)
    selected = pd.concat([grda, lrda], ignore_index=True)
    print(f"Cases: {len(grda)} GRDA + {len(lrda)} LRDA = {len(selected)}")

    # Load segments
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    # Build cases
    print("Processing cases...")
    cases_data = []
    n_skip = 0
    DISPLAY_HP, DISPLAY_LP = 0.3, 5.0
    b_bp, a_bp = butter(4, [DISPLAY_HP / (FS/2), DISPLAY_LP / (FS/2)], btype='bandpass')

    for i, (_, row) in enumerate(selected.iterrows()):
        pid = str(row['patient_id'])
        subtype = row['subtype']
        rda_frac = row['rda_frac']
        n_experts = int(row['n_expert_votes'])
        prev_freq = row['gold_standard_freq'] if row['gold_standard_freq'] > 0 else None
        pred = preds.get(pid, {})
        ensemble_freq = pred.get('ensemble_freq')
        ensemble_prob = pred.get('ensemble_rda_prob', 0)

        # Find EEG
        pid_segs = seg_df[seg_df['patient_id'] == pid]
        mat_file = None
        montage = 'monopolar'
        for _, sr in pid_segs.iterrows():
            if (EEG_DIR / sr['mat_file']).exists():
                mat_file = sr['mat_file']
                montage = sr.get('montage', 'monopolar')
                break
        if mat_file is None:
            if (EEG_DIR / f'{pid}_seg000.mat').exists():
                mat_file = f'{pid}_seg000.mat'
                montage = 'bipolar'
        if mat_file is None:
            n_skip += 1
            continue

        # Auto-detect montage
        try:
            _mat = sio.loadmat(str(EEG_DIR / mat_file))
            _dk = [k for k in _mat if not k.startswith('_')][0]
            n_ch = min(_mat[_dk].shape)
            auto_montage = 'monopolar' if n_ch >= 19 else 'bipolar'
        except:
            n_skip += 1
            continue

        seg_bi = load_segment(mat_file, auto_montage)
        if seg_bi is None or seg_bi.shape != (18, N_SAMPLES):
            n_skip += 1
            continue

        # Bandpass for display
        seg_display = np.zeros_like(seg_bi)
        for ch in range(18):
            try:
                seg_display[ch] = filtfilt(b_bp, a_bp, seg_bi[ch])
            except ValueError:
                seg_display[ch] = seg_bi[ch]

        # NVO for VE matrix (needed for display)
        try:
            ve_matrix, nvo_freq, best_ve = nvo_bandpass_grid(seg_display)
        except:
            n_skip += 1
            continue

        # Use ensemble frequency as the default selected frequency
        est_freq = ensemble_freq if ensemble_freq and np.isfinite(ensemble_freq) else nvo_freq

        # Precompute narrowband (wider: f±0.5Hz as requested)
        btn_filtered = compute_button_filtered(seg_display)

        nb_signals = {}
        for freq_str, ch_dict in btn_filtered.items():
            ch_data = {}
            for ch_str, arr in ch_dict.items():
                ch_data[ch_str] = [round(float(v), 1) for v in arr]
            nb_signals[freq_str] = ch_data

        # Laterality
        left_ve = np.mean(best_ve[LEFT_CHANNELS])
        right_ve = np.mean(best_ve[RIGHT_CHANNELS])
        denom = left_ve + right_ve
        lat_index = (right_ve - left_ve) / denom if denom > 0 else 0.0
        est_laterality = 'left' if lat_index < -0.15 else ('right' if lat_index > 0.15 else 'bilateral')

        case = {
            'patient_id': pid,
            'segment_id': f'{pid}_seg000',
            'est_freq': round(float(est_freq), 2),
            'est_laterality': est_laterality,
            'lat_index': round(float(lat_index), 3),
            'annotated': False,
            'subtype': subtype,
            'rda_frac': round(float(rda_frac), 3),
            'n_experts': n_experts,
            'ensemble_prob': round(float(ensemble_prob), 3),
            'prev_freq': round(float(prev_freq), 2) if prev_freq and prev_freq > 0 else None,
            'eeg_data': downsample(seg_display, 500),
            've_matrix': [[round(float(v), 3) for v in ve_matrix[fi]] for fi in range(len(FREQ_GRID))],
            'freq_grid': [round(f, 2) for f in FREQ_GRID],
            'best_ve': [round(float(v), 4) for v in best_ve],
            'nb_signals': nb_signals,
        }
        cases_data.append(case)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(selected)} ({time.time()-t0:.0f}s)")

    print(f"  Total: {len(cases_data)} (skipped {n_skip})")

    # Split into batches of ~200 cases each
    BATCH_SIZE = 200
    n_batches = (len(cases_data) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Splitting into {n_batches} batches of ~{BATCH_SIZE} cases...")

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(cases_data))
        batch = cases_data[start:end]

        print(f"  Building batch {batch_idx+1}/{n_batches} ({len(batch)} cases)...")
        html = build_html(batch)

        # Patch title and storage key (shared across batches so labels persist)
        html = html.replace(
            'LRDA Labeling Tool — Frequency, Laterality &amp; Wave Morphology',
            f'RDA Review Viewer — Batch {batch_idx+1}/{n_batches} ({len(batch)} cases)')
        html = html.replace("'lrda_labeler_v1'", "'rda_review_v1'")

        # Hide laterality buttons
        html = html.replace('id="laterality-buttons"',
                             'id="laterality-buttons" style="display:none"')

        # Add ensemble prob + rda_frac + prev_freq to the info display
        html = html.replace(
            "document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;",
            """document.getElementById('info-laterality').textContent = selectedLaterality || c.est_laterality;
    // Show ensemble info
    const infoExtra = document.getElementById('info-extra');
    if (infoExtra) {
      const sub = (c.subtype || '').toUpperCase();
      const rfPct = (c.rda_frac * 100).toFixed(0);
      const epPct = (c.ensemble_prob * 100).toFixed(0);
      const prevF = c.prev_freq ? c.prev_freq.toFixed(2) + ' Hz' : 'none';
      infoExtra.innerHTML = '<span style="color:#ff8844;font-weight:bold">' + sub + '</span>'
        + ' | Expert RDA: <span style="color:#44cc88;font-weight:bold">' + rfPct + '%</span>'
        + ' | Model RDA: <span style="color:#8888ff;font-weight:bold">' + epPct + '%</span>'
        + ' | Prev freq: <span style="color:#cc8833">' + prevF + '</span>'
        + ' | N=' + (c.n_experts || '--');
    }"""
        )

        # Add info-extra element to the HTML
        html = html.replace(
            '<span class="info-item">Laterality: <strong id="info-laterality"',
            '<span id="info-extra" class="info-item" style="font-size:13px"></span>'
            '<span class="info-item" style="display:none">Laterality: <strong id="info-laterality"'
        )

        suffix = f'_batch{batch_idx+1}' if n_batches > 1 else ''
        out_path = OUT_DIR / f'rda_review_viewer{suffix}.html'
        with open(str(out_path), 'w') as f:
            f.write(html)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE ({elapsed:.0f}s)")
    print(f"  Cases: {len(cases_data)} in {n_batches} batches")
    print(f"  Order: GRDA first (by rda_frac desc), then LRDA (by rda_frac desc)")
    print(f"{'='*70}")

    # Open first batch
    import subprocess
    first_batch = OUT_DIR / 'rda_review_viewer_batch1.html'
    if first_batch.exists():
        subprocess.Popen(['open', str(first_batch)])
    else:
        subprocess.Popen(['open', str(OUT_DIR / 'rda_review_viewer.html')])


if __name__ == '__main__':
    main()
