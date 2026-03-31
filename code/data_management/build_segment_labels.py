#!/usr/bin/env python3
"""Build segment_labels.csv — the single canonical label file.

One row per EEG file on disk, consolidating all label sources:
- segments.csv (file registry — metadata merged into output)
- annotations.csv (per-segment per-rater labels)
- patients.csv (patient-level labels — laterality, exclusion, auto-freq)
- list_events CSV (IIIC per-segment crowd votes)
- JSON label files (discharge timing, wave timing, channel involvement)
- Laterality review batch JSONs

Column naming conventions:
- iiic_*: IIIC crowd vote data (pattern class labels from 10-30 experts)
- expert_*: MW or expert-reviewed labels
- algo_*: Algorithm-assigned labels
- source_*: Provenance/audit trail

Usage:
    python code/data_management/build_segment_labels.py
"""
import os
import sys
import ast
import json
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
EEG_DIR = DATA_DIR / 'eeg'

EXPERT_RATERS = {'MW', 'mw', 'LB', 'PH', 'SZ'}
AUTO_RATER_SOURCES = {'iiic_dataset', 'harvest'}


def main():
    print("=" * 70)
    print("  Building segment_labels.csv")
    print("=" * 70)

    # ── Load all sources ──

    seg = pd.read_csv(str(LABELS_DIR / 'segments.csv'), dtype=str)
    pat_path = LABELS_DIR / 'patients.csv'
    if not pat_path.exists():
        pat_path = LABELS_DIR / 'archive_labels' / 'patients.csv'
    pat = pd.read_csv(str(pat_path), dtype=str)
    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'), dtype=str)

    # IIIC per-segment votes
    iiic_path = DATA_DIR / 'list_events_20241129.csv'
    if not iiic_path.exists():
        iiic_path = LABELS_DIR / 'archive_labels' / 'list_events_20241129.csv'
    if iiic_path.exists():
        events = pd.read_csv(str(iiic_path))
        events['votes'] = events['label ([other,seizure,lpd,gpd,lrda,grda])'].apply(
            lambda s: ast.literal_eval(str(s)))
        iiic_by_filename = {row['file_name']: row['votes'] for _, row in events.iterrows()}
        print(f"IIIC events loaded: {len(events)}")
    else:
        iiic_by_filename = {}
        print("WARNING: No IIIC events file found")

    # JSON label files
    discharge_times = _load_json(LABELS_DIR / 'discharge_times.json')
    wave_labels = _load_json(LABELS_DIR / 'rda_wave_labels.json')
    channel_inv = _load_json(LABELS_DIR / 'channel_involvement.json')

    # Laterality review batches (mat_file -> {label, lat_index, pred})
    lat_review = {}
    for batch_file in sorted(LABELS_DIR.glob('*_laterality_batch*.json')):
        batch = _load_json(batch_file)
        if 'decisions' in batch:
            lat_review.update(batch['decisions'])
            print(f"Laterality review loaded: {batch_file.name} ({len(batch['decisions'])} decisions)")
    # Also check archive_labels
    for batch_file in sorted((LABELS_DIR / 'archive_labels').glob('*_laterality_batch*.json')):
        batch = _load_json(batch_file)
        if 'decisions' in batch:
            lat_review.update(batch['decisions'])
            print(f"Laterality review loaded: archive_labels/{batch_file.name} ({len(batch['decisions'])} decisions)")

    # LPD lat+timing+freq review results (segment_id -> {laterality, selected_freq, global_times, ...})
    lat_timing_review = {}
    for search_dir in [LABELS_DIR, LABELS_DIR / 'archive_labels']:
        for rf in sorted(search_dir.glob('*_lat_timing_*_results*.json')):
            data = _load_json(rf)
            # Keys are segment_ids (without .mat)
            for sid, entry in data.items():
                if isinstance(entry, dict) and 'laterality' in entry:
                    lat_timing_review[sid] = entry
            print(f"Lat+timing review loaded: {rf.name} ({len(data)} entries)")

    # ── Handle EEG files not in segments.csv ──
    eeg_files_set = set(f for f in os.listdir(str(EEG_DIR)) if f.endswith('.mat'))
    seg_mat_files = set(seg['mat_file'])
    orphan_eeg = eeg_files_set - seg_mat_files
    if orphan_eeg:
        print(f"EEG files not in segments.csv: {len(orphan_eeg)} (adding them)")
        new_rows = []
        for mat in orphan_eeg:
            pid = mat.split('_seg')[0] if '_seg' in mat else mat.replace('.mat', '')
            new_rows.append({
                'segment_id': mat.replace('.mat', ''),
                'patient_id': pid,
                'mat_file': mat,
                'subtype': '', 'subtype_source': '', 'original_source': 'orphan',
                'original_filename': '',
            })
        seg = pd.concat([seg, pd.DataFrame(new_rows)], ignore_index=True)

    # ── Index sources for fast lookup ──

    seg_by_mat = {r['mat_file']: r for _, r in seg.iterrows()}
    pat_by_id = {r['patient_id']: r for _, r in pat.iterrows()}

    ann_by_seg = {}
    ann_by_pat = {}
    for _, r in ann.iterrows():
        ann_by_seg.setdefault(r['segment_id'], []).append(r)
        ann_by_pat.setdefault(r['patient_id'], []).append(r)

    # ── Build rows: one per EEG file on disk ──

    eeg_files = sorted(f for f in os.listdir(str(EEG_DIR)) if f.endswith('.mat'))
    print(f"EEG files on disk: {len(eeg_files)}")

    rows = []
    for mat_file in eeg_files:
        row = {}

        # ── Identity / file info ──
        seg_info = seg_by_mat.get(mat_file)
        if seg_info is not None:
            row['mat_file'] = mat_file
            row['patient_id'] = seg_info['patient_id']
            row['subtype_source'] = seg_info.get('subtype_source', '')
            # Merge segments.csv metadata
            row['original_source'] = seg_info.get('original_source', '')
            row['source_filename'] = seg_info.get('original_filename', '')
            row['montage'] = seg_info.get('montage', '')
            row['duration_sec'] = seg_info.get('duration_sec', '')
            row['fs'] = seg_info.get('fs', '')
            row['n_channels'] = seg_info.get('n_channels', '')
        else:
            row['mat_file'] = mat_file
            row['patient_id'] = mat_file.split('_seg')[0] if '_seg' in mat_file else mat_file.replace('.mat', '')
            row['subtype_source'] = ''
            row['original_source'] = ''
            row['source_filename'] = ''
            row['montage'] = ''
            row['duration_sec'] = ''
            row['fs'] = ''
            row['n_channels'] = ''

        pid = row['patient_id']
        sid = mat_file.replace('.mat', '')
        orig_base = str(row['source_filename']).replace('.mat', '') if row['source_filename'] else ''

        # ── IIIC crowd votes (pattern class) ──
        cats = ['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']
        mat_base = mat_file.replace('.mat', '')
        iiic_votes = iiic_by_filename.get(orig_base) or iiic_by_filename.get(mat_base)
        if iiic_votes:
            for i, cat in enumerate(cats):
                row[f'iiic_vote_{cat}'] = iiic_votes[i]
            total = sum(iiic_votes)
            row['iiic_n_votes'] = total
            winner_idx = max(range(6), key=lambda i: iiic_votes[i])
            row['iiic_plurality'] = cats[winner_idx]
            row['iiic_plurality_frac'] = round(iiic_votes[winner_idx] / total, 3) if total > 0 else ''
        elif seg_info is not None and seg_info.get('subtype', '') in cats:
            assigned_subtype = seg_info['subtype']
            for cat in cats:
                row[f'iiic_vote_{cat}'] = 1 if cat == assigned_subtype else 0
            row['iiic_n_votes'] = 1
            row['iiic_plurality'] = assigned_subtype
            row['iiic_plurality_frac'] = 1.0
        else:
            for cat in cats:
                row[f'iiic_vote_{cat}'] = ''
            row['iiic_n_votes'] = ''
            row['iiic_plurality'] = ''
            row['iiic_plurality_frac'] = ''

        # ── Subtype (best available) ──
        if iiic_votes:
            row['subtype'] = row['iiic_plurality']
            row['subtype_source'] = 'iiic_segment_vote'
        elif seg_info is not None:
            row['subtype'] = seg_info.get('subtype', '')
            source = seg_info.get('subtype_source', '')
            row['subtype_source'] = source if source else ''
        else:
            row['subtype'] = ''
            row['subtype_source'] = ''

        # ── Frequency (from annotations.csv) ──
        expert_freq, expert_rater = _get_expert_freq(sid, pid, ann_by_seg, ann_by_pat)
        # Lat+timing review overrides (higher priority — MW reviewed)
        if sid in lat_timing_review:
            ltr = lat_timing_review[sid]
            if not ltr.get('rejected') and ltr.get('selected_freq') is not None:
                expert_freq = round(float(ltr['selected_freq']), 2)
                expert_rater = 'MW'
        row['expert_freq_hz'] = expert_freq
        row['expert_freq_rater'] = expert_rater

        # Auto-assigned freq from patients.csv
        pat_info = pat_by_id.get(pid)
        if pat_info is not None:
            subtype_rater = str(pat_info.get('subtype_rater', ''))
            freq_val = pd.to_numeric(pat_info.get('gold_standard_freq'), errors='coerce')
            has_freq = pd.notna(freq_val) and freq_val > 0
            if has_freq and subtype_rater in AUTO_RATER_SOURCES:
                row['algo_freq_hz'] = round(float(freq_val), 2)
            elif has_freq and not row['expert_freq_hz']:
                row['algo_freq_hz'] = round(float(freq_val), 2)
            else:
                row['algo_freq_hz'] = ''
        else:
            row['algo_freq_hz'] = ''

        # ── Spatial annotations ──
        spatial_info = _get_spatial(sid, pid, ann_by_seg, ann_by_pat)
        row['spatial_channels'] = spatial_info[0]
        row['spatial_raters'] = spatial_info[1]

        # ── Laterality ──
        lat = ''
        lat_rater = ''
        if pat_info is not None:
            p_lat = pat_info.get('laterality', '')
            if p_lat in ('left', 'right', 'bilateral'):
                lat = p_lat
                lat_rater = pat_info.get('laterality_rater', '')
        if not lat and pid in channel_inv:
            ci_lat = channel_inv[pid].get('laterality', '')
            if ci_lat in ('left', 'right', 'bilateral'):
                lat = ci_lat
                lat_rater = channel_inv[pid].get('review_source', 'channel_inv')
        if mat_file in lat_review:
            review_label = lat_review[mat_file].get('label', '')
            if review_label in ('left', 'right'):
                lat = review_label
                lat_rater = 'MW'
            elif review_label == 'reject':
                lat = ''
                lat_rater = ''
        # Lat+timing review (keyed by segment_id)
        if sid in lat_timing_review:
            ltr = lat_timing_review[sid]
            if not ltr.get('rejected') and ltr.get('laterality') in ('left', 'right'):
                lat = ltr['laterality']
                lat_rater = 'MW'
            elif ltr.get('rejected'):
                lat = ''
                lat_rater = ''
        row['laterality'] = lat
        row['laterality_rater'] = lat_rater

        # ── Exclusion ──
        excluded = False
        excl_reason = ''
        if pat_info is not None:
            excluded = str(pat_info.get('excluded', 'False')) == 'True'
            excl_reason = pat_info.get('exclusion_reason', '') if excluded else ''
        if mat_file in lat_review and lat_review[mat_file].get('label') == 'reject':
            excluded = True
            excl_reason = 'not_lrda_mw_review'
        if sid in lat_timing_review and lat_timing_review[sid].get('rejected'):
            excluded = True
            excl_reason = 'rejected_mw_review'
        row['excluded'] = excluded
        row['exclusion_reason'] = excl_reason if excluded else ''

        # ── Audit trail ──
        if pat_info is not None:
            so = pat_info.get('subtype_original', '')
            row['subtype_original'] = so if so and so != 'nan' else ''
            fo = pat_info.get('gold_standard_freq_original', '')
            fo_val = pd.to_numeric(fo, errors='coerce')
            row['freq_original'] = round(float(fo_val), 2) if pd.notna(fo_val) and fo_val > 0 else ''
            lo = pat_info.get('laterality_original', '')
            row['laterality_original'] = lo if lo in ('left', 'right', 'bilateral') else ''
        else:
            row['subtype_original'] = ''
            row['freq_original'] = ''
            row['laterality_original'] = ''

        # ── Other labels (JSON files + lat_timing review) ──
        has_dt = pid in discharge_times or sid in discharge_times
        if sid in lat_timing_review:
            ltr = lat_timing_review[sid]
            if not ltr.get('rejected') and ltr.get('global_times') and len(ltr['global_times']) > 0:
                has_dt = True
                # Also merge into discharge_times dict for downstream use
                discharge_times[sid] = {
                    'global_times': ltr['global_times'],
                    'frequency': ltr.get('selected_freq'),
                    'source': 'lpd_lat_timing_review',
                }
        row['has_discharge_timing'] = has_dt
        row['has_wave_timing'] = pid in wave_labels or sid in wave_labels
        row['has_channel_involvement'] = pid in channel_inv or sid in channel_inv

        # ── Annotators ──
        raters = set()
        for a in ann_by_seg.get(sid, []):
            raters.add(a['rater'])
        row['annotators'] = ','.join(sorted(raters)) if raters else ''

        rows.append(row)

    # ── Write output ──
    df = pd.DataFrame(rows)

    col_order = [
        'mat_file', 'patient_id',
        'subtype', 'subtype_source',
        'iiic_vote_other', 'iiic_vote_seizure', 'iiic_vote_lpd', 'iiic_vote_gpd',
        'iiic_vote_lrda', 'iiic_vote_grda', 'iiic_n_votes', 'iiic_plurality', 'iiic_plurality_frac',
        'expert_freq_hz', 'expert_freq_rater', 'algo_freq_hz',
        'spatial_channels', 'spatial_raters',
        'laterality', 'laterality_rater',
        'excluded', 'exclusion_reason',
        'subtype_original', 'freq_original', 'laterality_original',
        'has_discharge_timing', 'has_wave_timing', 'has_channel_involvement',
        'annotators',
        'original_source', 'source_filename',
        'montage', 'duration_sec', 'fs', 'n_channels',
    ]
    df = df[col_order]

    out_path = LABELS_DIR / 'segment_labels.csv'
    df.to_csv(str(out_path), index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}")

    # Save updated discharge_times.json (may have new entries from lat_timing review)
    dt_path = LABELS_DIR / 'discharge_times.json'
    with open(str(dt_path), 'w') as f:
        json.dump(discharge_times, f, indent=2)
    print(f"Updated: {dt_path} ({len(discharge_times)} entries)")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print("  Coverage Summary")
    print(f"{'=' * 50}")
    print(f"Total segments:              {len(df)}")
    nv = pd.to_numeric(df['iiic_n_votes'], errors='coerce')
    print(f"With IIIC votes (>=1):       {(nv >= 1).sum()}")
    print(f"With IIIC votes (>=10):      {(nv >= 10).sum()}")
    print(f"With expert frequency:       {(df['expert_freq_hz'] != '').sum()}")
    print(f"With algo frequency:         {(df['algo_freq_hz'] != '').sum()}")
    print(f"With spatial annotations:    {(df['spatial_channels'] != '').sum()}")
    print(f"With laterality:             {(df['laterality'] != '').sum()}")
    print(f"Excluded:                    {df['excluded'].sum()}")
    print(f"With discharge timing:       {df['has_discharge_timing'].sum()}")
    print(f"With wave timing:            {df['has_wave_timing'].sum()}")
    print(f"With channel involvement:    {df['has_channel_involvement'].sum()}")
    print()
    print(f"Subtype distribution:")
    st = df[df['subtype'] != '']['subtype'].value_counts()
    for k, v in st.items():
        print(f"  {k}: {v}")
    print()
    print(f"Subtype source distribution:")
    ss = df[df['subtype_source'] != '']['subtype_source'].value_counts()
    for k, v in ss.items():
        print(f"  {k}: {v}")


def _get_expert_freq(seg_id, pat_id, ann_by_seg, ann_by_pat):
    """Get MW/expert frequency for a segment. Prefers segment-level, falls back to patient-level."""
    for a in ann_by_seg.get(seg_id, []):
        if a['rater'] in EXPERT_RATERS:
            freq = pd.to_numeric(a.get('frequency_hz'), errors='coerce')
            if pd.notna(freq) and freq > 0:
                return round(float(freq), 2), a['rater']

    for a in ann_by_pat.get(pat_id, []):
        if a['rater'] in EXPERT_RATERS:
            freq = pd.to_numeric(a.get('frequency_hz'), errors='coerce')
            if pd.notna(freq) and freq > 0:
                return round(float(freq), 2), a['rater']

    return '', ''


def _get_spatial(seg_id, pat_id, ann_by_seg, ann_by_pat):
    """Get spatial channel annotations."""
    channels = []
    raters = []
    for a in ann_by_seg.get(seg_id, []):
        sc = str(a.get('spatial_channels', ''))
        if sc and sc != 'nan':
            channels.append(sc)
            raters.append(a['rater'])
    if channels:
        return '; '.join(f'{r}:{c}' for r, c in zip(raters, channels)), ','.join(raters)

    for a in ann_by_pat.get(pat_id, []):
        sc = str(a.get('spatial_channels', ''))
        if sc and sc != 'nan':
            channels.append(sc)
            raters.append(a['rater'])
    if channels:
        return '; '.join(f'{r}:{c}' for r, c in zip(raters, channels)), ','.join(raters)

    return '', ''


def _load_json(path):
    if path.exists():
        with open(str(path)) as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    main()
