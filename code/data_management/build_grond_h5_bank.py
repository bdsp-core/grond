#!/usr/bin/env python3
"""Build the GROND reproducibility data bank: grond_data.h5.

Self-contained .h5 file pairing each EEG segment with all of its labels,
algorithm predictions, cohort flags, and metadata.

Schema:
  /segments/{segment_id}/
       eeg              (19, 2000) float32 monopolar referential EEG at 200 Hz
       attrs: patient_id, subtype, fs_hz=200, eeg_source, eeg_file,
              montage, mat_file, has_discharge_timing

       The bank stores the canonical 19-channel monopolar referential
       montage. Downstream code is expected to derive whichever montage
       it needs (bipolar banana, Laplacian, average reference, etc.) at
       runtime. Bipolar banana derivation is documented in
       /metadata/bipolar_pair_definitions_json.
  /labels/{segment_id}/
       freq_hz          float32 — expert frequency (consensus across raters)
       freq_per_rater   variable-length attr — JSON {rater: freq_hz}
       laterality       str — 'left'/'right'/'bilateral'/''
       discharge_times  variable-length (N,) float32 — discharge times in seconds
       spatial_channels str — channel involvement labels (raw form, optional)
       attrs: review_status, source
  /predictions/pdchar/{segment_id}/
       pdchar_freq_hz, pdchar_laterality, pdchar_spatial_extent,
       pdchar_channel_probs (18,) (where available)
  /predictions/tautan/{segment_id}/
       tautan_freq_hz, tautan_spatial_extent
  /predictions/rda_plv/{segment_id}/
       rda_plv_spatial_extent
  /cohorts/
       pd_gt              (N,) bool — discharge-timing GT cohort flag
       irr_canonical      (N,) bool — canonical IRR cohort flag
       irr_tautan         (N,) bool — Tautan IRR cohort flag
       segment_ids        (N,) str — ordered segment IDs (key for cohort flags)
  /metadata/
       channel_names_bipolar  (18,) str
       bipolar_pair_definitions  str (JSON describing each bipolar pair)
       channel_names_mono     (19,) str
       pipeline_version       str
       label_provenance       str (JSON)
       citation               str

Usage:
    conda run -n morgoth python code/data_management/build_grond_h5_bank.py [--out OUT_PATH]
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import scipy.io as sio

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
SEG_CSV = PROJECT_DIR / 'data' / 'labels' / 'segments.csv'
LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'labels.csv'
SEG_LABELS_CSV = PROJECT_DIR / 'data' / 'labels' / 'segment_labels.csv'
DT_JSON = PROJECT_DIR / 'data' / 'labels' / 'discharge_times.json'
IRR_ROOT = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'

OUT_PATH_DEFAULT = PROJECT_DIR / 'data' / 'grond_data.h5'

MONO_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz',
                 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']
BIPOLAR_PAIRS = [
    ('Fp1','F7'), ('F7','T3'), ('T3','T5'), ('T5','O1'),
    ('Fp2','F8'), ('F8','T4'), ('T4','T6'), ('T6','O2'),
    ('Fp1','F3'), ('F3','C3'), ('C3','P3'), ('P3','O1'),
    ('Fp2','F4'), ('F4','C4'), ('C4','P4'), ('P4','O2'),
    ('Fz','Cz'), ('Cz','Pz'),
]
BIPOLAR_NAMES = [f'{a}-{b}' for a, b in BIPOLAR_PAIRS]
BIPOLAR_INDICES = np.array([[MONO_CHANNELS.index(a), MONO_CHANNELS.index(b)]
                            for a, b in BIPOLAR_PAIRS])

PID_RE = re.compile(r'sub-S0001(\d+)_')


def load_eeg(mat_file):
    """Load a .mat file from data/eeg/; return (19, 2000) float32 monopolar referential.

    The bank stores 19-channel monopolar EEG so that downstream code can
    choose its own derivation (bipolar banana, Laplacian, average reference,
    etc.) at runtime. For consumers that want the canonical 18-channel
    bipolar banana montage, use the bipolar_pair_definitions in
    /metadata/ to derive it on the fly:

        mono = h5['segments'][seg_id]['eeg'][:]                # (19, 2000)
        bip = mono[BIPOLAR_INDICES[:,0]] - mono[BIPOLAR_INDICES[:,1]]  # (18, 2000)
    """
    p = EEG_DIR / mat_file
    if not p.exists():
        return None
    try:
        m = sio.loadmat(str(p))
    except Exception:
        return None
    data = m.get('data')
    if data is None:
        return None
    data = data.astype(np.float64)
    if data.shape[0] == data.shape[1]:
        return None  # ambiguous
    if data.shape[0] > data.shape[1]:
        data = data.T
    if data.shape[1] != 2000:
        return None
    if data.shape[0] == 19:
        # Already canonical 19-ch monopolar
        return data.astype(np.float32)
    if data.shape[0] == 20:
        # Drop EKG (last channel) → canonical 19-ch monopolar
        return data[:19].astype(np.float32)
    if data.shape[0] == 18:
        # File has been pre-derived to bipolar; the monopolar original is
        # not recoverable from the bipolar signal alone (the bipolar set has
        # 18 independent equations on 19 unknowns). Mark for caller to skip.
        return None
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT_PATH_DEFAULT),
                        help='Output .h5 path (default: data/grond_data.h5)')
    parser.add_argument('--max-segments', type=int, default=0,
                        help='Limit to N segments for testing (0 = all)')
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print('Loading inputs...')
    seg_df = pd.read_csv(SEG_CSV)
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)
    # Drop duplicate mat_file rows (keeping the row with the most-complete metadata).
    # Prefer rows where eeg_source is one of the recovery sources (since those
    # were appended for the missing PD-GT cohort and override the original blank rows).
    # Simple strategy: keep last occurrence.
    n_before = len(seg_df)
    seg_df = seg_df.drop_duplicates(subset=['mat_file'], keep='last').reset_index(drop=True)
    if len(seg_df) < n_before:
        print(f'  Deduplicated mat_file: {n_before} → {len(seg_df)} rows '
              f'({n_before - len(seg_df)} duplicates dropped)')
    labels_df = pd.read_csv(LABELS_CSV)
    sl_df = pd.read_csv(SEG_LABELS_CSV)
    sl_df['patient_id'] = sl_df['patient_id'].astype(str)
    with open(DT_JSON) as f:
        dt = json.load(f)
    print(f'  segments.csv: {len(seg_df)} rows')
    print(f'  labels.csv: {len(labels_df)} rows')
    print(f'  segment_labels.csv: {len(sl_df)} rows')
    print(f'  discharge_times.json: {len(dt)} entries')

    # Build IRR cohort flags
    irr_canonical_pids = set()
    for sub in ('lpd', 'gpd', 'lrda', 'grda'):
        m = pd.read_csv(IRR_ROOT / sub / 'manifest.csv')
        for _, row in m.iterrows():
            ps = str(row['patient_id'])
            if ps.startswith('sub-S0001'):
                mt = re.match(r'sub-S0001(\d+)', ps)
                if mt: irr_canonical_pids.add(mt.group(1))
            else:
                irr_canonical_pids.add(ps)
    pd_gt_pids = {pid for pid, v in dt.items()
                  if v.get('review_status') == 'ground_truth'
                  and v.get('subtype') in ('lpd', 'gpd')}

    if args.max_segments > 0:
        seg_df = seg_df.head(args.max_segments)
        print(f'  LIMITED to first {len(seg_df)} segments')

    # Index labels.csv by mat_file → list of (rater, label_type, value)
    labels_by_mat = {}
    for _, r in labels_df.iterrows():
        labels_by_mat.setdefault(r['mat_file'], []).append({
            'rater': str(r['rater']),
            'label_type': str(r['label_type']),
            'value': str(r['value']),
            'round': str(r.get('round', '')),
            'date': str(r.get('date', '')),
        })

    # Index segment_labels.csv by mat_file
    sl_by_mat = {}
    for _, r in sl_df.iterrows():
        sl_by_mat[r['mat_file']] = r.to_dict()

    print(f'\nWriting {out_path} (this may take a few minutes)...')
    t0 = time.time()
    with h5py.File(out_path, 'w') as f:
        # ─── /metadata ───────────────────────────────────────────────────
        meta = f.create_group('metadata')
        meta.attrs['pipeline_version'] = 'grond_2026.06'
        meta.attrs['fs_hz'] = 200.0
        meta.attrs['segment_duration_s'] = 10.0
        meta.attrs['n_samples_per_segment'] = 2000
        meta.attrs['n_bipolar_channels'] = 18
        meta.attrs['n_monopolar_channels'] = 19
        meta.create_dataset('channel_names_bipolar',
                            data=np.array(BIPOLAR_NAMES, dtype='S16'))
        meta.create_dataset('channel_names_mono',
                            data=np.array(MONO_CHANNELS, dtype='S16'))
        meta.attrs['bipolar_pair_definitions_json'] = json.dumps(
            [{'index': i, 'name': BIPOLAR_NAMES[i],
              'anode': BIPOLAR_PAIRS[i][0], 'cathode': BIPOLAR_PAIRS[i][1]}
             for i in range(18)])
        meta.attrs['citation'] = (
            'GROND: Generalized Rhythmic and Oscillatory Neurophysiology '
            'Descriptor. Westover et al. (2026).')
        meta.attrs['repo'] = 'https://github.com/bdsp-core/grond'
        meta.attrs['eeg_source_legend'] = (
            's3_morgoth/dataset_eeg/external_drive = original; '
            'alex_recovery/s3_iiic_freq3_recovery/pathC_morgoth1_s3 = '
            'recovered post-cleanup, see REPRODUCIBILITY.md for caveats.')

        # ─── /segments + /labels + /predictions per row ──────────────────
        seg_group = f.create_group('segments')
        lab_group = f.create_group('labels')
        pred_pdchar = f.create_group('predictions/pdchar')
        pred_tautan = f.create_group('predictions/tautan')
        pred_rda_plv = f.create_group('predictions/rda_plv')

        segment_ids = []
        pd_gt_flags = []
        irr_flags = []
        n_written = 0
        n_no_eeg = 0
        n_bad_shape = 0

        for i, row in seg_df.iterrows():
            mat_file = row['mat_file']
            pid = str(row['patient_id'])
            segment_id = mat_file.replace('.mat', '')

            eeg = load_eeg(mat_file)
            if eeg is None:
                n_no_eeg += 1
                continue
            if eeg.shape != (19, 2000):
                n_bad_shape += 1
                continue

            # /segments/{id}
            sg = seg_group.create_group(segment_id)
            sg.create_dataset('eeg', data=eeg,
                              compression='gzip', compression_opts=4)
            sg.attrs['patient_id'] = pid
            sg.attrs['subtype'] = str(row.get('subtype', ''))
            sg.attrs['subtype_source'] = str(row.get('subtype_source', ''))
            sg.attrs['eeg_source'] = str(row.get('eeg_source', ''))
            sg.attrs['eeg_file'] = str(row.get('eeg_file', '') or '')
            sg.attrs['montage'] = str(row.get('montage', ''))
            sg.attrs['mat_file'] = mat_file
            sg.attrs['has_discharge_timing'] = bool(pid in pd_gt_pids)

            # /labels/{id}
            lg = lab_group.create_group(segment_id)
            # Frequency labels from labels.csv
            freqs = [lab for lab in labels_by_mat.get(mat_file, [])
                     if lab['label_type'] == 'frequency_hz']
            if freqs:
                freq_vals = []
                per_rater = {}
                for lab in freqs:
                    try:
                        v = float(lab['value'])
                        if np.isfinite(v):
                            freq_vals.append(v)
                            per_rater[lab['rater']] = v
                    except ValueError:
                        pass
                if freq_vals:
                    lg.attrs['freq_hz_consensus'] = float(np.median(freq_vals))
                    lg.attrs['freq_hz_n_raters'] = len(freq_vals)
                    lg.attrs['freq_per_rater_json'] = json.dumps(per_rater)
            # Laterality from labels.csv
            lats = [lab for lab in labels_by_mat.get(mat_file, [])
                    if lab['label_type'] == 'laterality']
            if lats:
                vals = [lab['value'] for lab in lats]
                lg.attrs['laterality_per_rater_json'] = json.dumps(
                    {lab['rater']: lab['value'] for lab in lats})
                # Take most common
                from collections import Counter
                lg.attrs['laterality_consensus'] = Counter(vals).most_common(1)[0][0]
            # From segment_labels.csv: spatial + algo freq etc
            slr = sl_by_mat.get(mat_file, {})
            if pd.notna(slr.get('spatial_channels')) and str(slr['spatial_channels']) not in ('', 'nan'):
                lg.attrs['spatial_channels'] = str(slr['spatial_channels'])
            if pd.notna(slr.get('expert_freq_hz')):
                try:
                    lg.attrs['expert_freq_hz'] = float(slr['expert_freq_hz'])
                except (TypeError, ValueError):
                    pass
            # Discharge times for PD-GT
            if pid in dt:
                v = dt[pid]
                if v.get('review_status') == 'ground_truth' and v.get('subtype') in ('lpd', 'gpd'):
                    gt = v.get('global_times', [])
                    if isinstance(gt, list) and len(gt) > 0:
                        lg.create_dataset('discharge_times',
                                          data=np.array(gt, dtype=np.float32))
                        lg.attrs['review_status'] = str(v.get('review_status', ''))
                        lg.attrs['review_source'] = str(v.get('review_source', ''))
                        lg.attrs['source'] = str(v.get('source', ''))
                        if 'selected_freq' in v:
                            try:
                                lg.attrs['selected_freq'] = float(v['selected_freq'])
                            except (TypeError, ValueError):
                                pass
                        if 'laterality' in v and v['laterality']:
                            lg.attrs['gt_laterality'] = str(v['laterality'])

            # /predictions/pdchar/{id}
            if pd.notna(row.get('pdchar_freq_hz')) and str(row['pdchar_freq_hz']) not in ('', 'nan'):
                pg = pred_pdchar.create_group(segment_id)
                try:
                    pg.attrs['pdchar_freq_hz'] = float(row['pdchar_freq_hz'])
                except (TypeError, ValueError):
                    pass
                if pd.notna(row.get('pdchar_laterality')):
                    pg.attrs['pdchar_laterality'] = str(row['pdchar_laterality'])
                if pd.notna(row.get('pdchar_spatial_extent')):
                    try:
                        pg.attrs['pdchar_spatial_extent'] = float(row['pdchar_spatial_extent'])
                    except (TypeError, ValueError):
                        pass
            # /predictions/tautan/{id}
            if pd.notna(row.get('tautan_freq_hz')) and str(row['tautan_freq_hz']) not in ('', 'nan'):
                tg = pred_tautan.create_group(segment_id)
                try:
                    tg.attrs['tautan_freq_hz'] = float(row['tautan_freq_hz'])
                except (TypeError, ValueError):
                    pass
                if pd.notna(row.get('tautan_spatial_extent')):
                    try:
                        tg.attrs['tautan_spatial_extent'] = float(row['tautan_spatial_extent'])
                    except (TypeError, ValueError):
                        pass
            # /predictions/rda_plv/{id}
            if pd.notna(row.get('rda_plv_spatial_extent')) and str(row['rda_plv_spatial_extent']) not in ('', 'nan'):
                rg = pred_rda_plv.create_group(segment_id)
                try:
                    rg.attrs['rda_plv_spatial_extent'] = float(row['rda_plv_spatial_extent'])
                except (TypeError, ValueError):
                    pass

            # Cohort flags
            segment_ids.append(segment_id)
            pd_gt_flags.append(pid in pd_gt_pids)
            irr_flags.append(pid in irr_canonical_pids)

            n_written += 1
            if n_written % 500 == 0:
                elapsed = time.time() - t0
                print(f'  wrote {n_written}/{len(seg_df)}  ({elapsed:.0f}s)')

        # ─── /cohorts ────────────────────────────────────────────────────
        co_group = f.create_group('cohorts')
        co_group.create_dataset('segment_ids',
                                data=np.array(segment_ids, dtype='S64'))
        co_group.create_dataset('pd_gt',
                                data=np.array(pd_gt_flags, dtype=np.bool_))
        co_group.create_dataset('irr_canonical',
                                data=np.array(irr_flags, dtype=np.bool_))
        co_group.attrs['pd_gt_description'] = (
            'PD-GT discharge-timing ground-truth cohort '
            '(review_status=ground_truth in discharge_times.json)')
        co_group.attrs['irr_canonical_description'] = (
            'Canonical IRR evaluation cohort '
            '(200 stratified segments × 4 subtypes; rated by MW, SZ, TZ, AS)')

    print(f'\nDone.')
    print(f'  Wrote {n_written} segments to {out_path}')
    print(f'  Skipped: {n_no_eeg} missing EEG file, {n_bad_shape} bad shape')
    print(f'  Size: {out_path.stat().st_size / 1e9:.2f} GB')
    print(f'  Time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
