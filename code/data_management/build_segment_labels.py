#!/usr/bin/env python3
"""Build segment_labels.csv — the single canonical label file.

One row per EEG file on disk, consolidating all label sources:
- segments.csv (file registry)
- annotations.csv (per-segment per-rater labels)
- patients.csv (patient-level labels — laterality, exclusion, auto-freq)
- list_events CSV (IIIC per-segment crowd votes)
- JSON label files (discharge timing, wave timing, channel involvement)

Usage:
    python code/build_segment_labels.py
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

    # segments.csv by mat_file
    seg_by_mat = {r['mat_file']: r for _, r in seg.iterrows()}

    # patients.csv by patient_id
    pat_by_id = {r['patient_id']: r for _, r in pat.iterrows()}

    # annotations.csv grouped by segment_id and by patient_id
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
            row['segment_id'] = seg_info['segment_id']
            row['patient_id'] = seg_info['patient_id']
            row['original_source'] = seg_info.get('original_source', '')
            row['original_filename'] = seg_info.get('original_filename', '')
        else:
            row['mat_file'] = mat_file
            row['segment_id'] = mat_file.replace('.mat', '')
            row['patient_id'] = mat_file.split('_seg')[0] if '_seg' in mat_file else mat_file.replace('.mat', '')
            row['original_source'] = ''
            row['original_filename'] = ''

        pid = row['patient_id']
        sid = row['segment_id']
        orig_base = str(row['original_filename']).replace('.mat', '') if row['original_filename'] else ''

        # ── Per-segment vote vector ──
        # Source 1: IIIC crowd votes (matched by original_filename)
        # Source 2: MW single-rater assignment (from folder/subtype_source)
        # Both stored in the same format: [other, seizure, lpd, gpd, lrda, grda]
        cats = ['other', 'seizure', 'lpd', 'gpd', 'lrda', 'grda']
        iiic_votes = iiic_by_filename.get(orig_base)
        if iiic_votes:
            for i, cat in enumerate(cats):
                row[f'vote_{cat}'] = iiic_votes[i]
            total = sum(iiic_votes)
            row['n_votes'] = total
            winner_idx = max(range(6), key=lambda i: iiic_votes[i])
            row['plurality'] = cats[winner_idx]
            row['plurality_frac'] = round(iiic_votes[winner_idx] / total, 3) if total > 0 else ''
        elif seg_info is not None and seg_info.get('subtype', '') in cats:
            # No IIIC crowd votes — create a single-vote vector from the folder assignment
            # This represents MW's single-rater classification
            assigned_subtype = seg_info['subtype']
            for cat in cats:
                row[f'vote_{cat}'] = 1 if cat == assigned_subtype else 0
            row['n_votes'] = 1
            row['plurality'] = assigned_subtype
            row['plurality_frac'] = 1.0
        else:
            for cat in cats:
                row[f'vote_{cat}'] = ''
            row['n_votes'] = ''
            row['plurality'] = ''
            row['plurality_frac'] = ''

        # ── Subtype (best available) ──
        if iiic_votes:
            row['subtype'] = row['plurality']
            row['subtype_source'] = 'iiic_segment_vote'
        elif seg_info is not None:
            row['subtype'] = seg_info.get('subtype', '')
            source = seg_info.get('subtype_source', '')
            row['subtype_source'] = source if source else ''
        else:
            row['subtype'] = ''
            row['subtype_source'] = ''

        # ── Frequency (from annotations.csv) ──
        # Look for per-segment annotations first, then patient-level
        mw_freq, mw_rater = _get_expert_freq(sid, pid, ann_by_seg, ann_by_pat)
        row['mw_freq'] = mw_freq
        row['mw_freq_rater'] = mw_rater

        # Auto-assigned freq from patients.csv
        pat_info = pat_by_id.get(pid)
        if pat_info is not None:
            subtype_rater = str(pat_info.get('subtype_rater', ''))
            freq_val = pd.to_numeric(pat_info.get('gold_standard_freq'), errors='coerce')
            has_freq = pd.notna(freq_val) and freq_val > 0
            if has_freq and subtype_rater in AUTO_RATER_SOURCES:
                row['auto_freq'] = round(float(freq_val), 2)
            elif has_freq and not row['mw_freq']:
                # Freq exists in patients.csv but not in annotations — treat as auto
                row['auto_freq'] = round(float(freq_val), 2)
            else:
                row['auto_freq'] = ''
        else:
            row['auto_freq'] = ''

        # ── Spatial annotations ──
        spatial_info = _get_spatial(sid, pid, ann_by_seg, ann_by_pat)
        row['spatial_channels'] = spatial_info[0]
        row['spatial_raters'] = spatial_info[1]

        # ── Laterality (from patients.csv) ──
        if pat_info is not None:
            lat = pat_info.get('laterality', '')
            row['laterality'] = lat if lat in ('left', 'right', 'bilateral') else ''
            row['laterality_rater'] = pat_info.get('laterality_rater', '') if row['laterality'] else ''
        else:
            row['laterality'] = ''
            row['laterality_rater'] = ''

        # ── Exclusion ──
        if pat_info is not None:
            row['excluded'] = str(pat_info.get('excluded', 'False')) == 'True'
            row['exclusion_reason'] = pat_info.get('exclusion_reason', '') if row['excluded'] else ''
        else:
            row['excluded'] = False
            row['exclusion_reason'] = ''

        # ── Audit trail (_original columns from patients.csv) ──
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

        # ── Other labels (JSON files) ──
        row['has_discharge_timing'] = pid in discharge_times or sid in discharge_times
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
        'mat_file', 'segment_id', 'patient_id',
        'subtype', 'subtype_source',
        'vote_other', 'vote_seizure', 'vote_lpd', 'vote_gpd',
        'vote_lrda', 'vote_grda', 'n_votes', 'plurality', 'plurality_frac',
        'mw_freq', 'mw_freq_rater', 'auto_freq',
        'spatial_channels', 'spatial_raters',
        'laterality', 'laterality_rater',
        'excluded', 'exclusion_reason',
        'subtype_original', 'freq_original', 'laterality_original',
        'has_discharge_timing', 'has_wave_timing', 'has_channel_involvement',
        'annotators',
        'original_source', 'original_filename',
    ]
    df = df[col_order]

    out_path = LABELS_DIR / 'segment_labels.csv'
    df.to_csv(str(out_path), index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}")

    # ── Summary ──
    print(f"\n{'=' * 50}")
    print("  Coverage Summary")
    print(f"{'=' * 50}")
    print(f"Total segments:              {len(df)}")
    print(f"With IIIC per-segment votes: {(df['n_votes'] != '').sum()}")
    print(f"With MW/expert frequency:    {(df['mw_freq'] != '').sum()}")
    print(f"With auto frequency:         {(df['auto_freq'] != '').sum()}")
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
    # Segment-level first
    for a in ann_by_seg.get(seg_id, []):
        if a['rater'] in EXPERT_RATERS:
            freq = pd.to_numeric(a.get('frequency_hz'), errors='coerce')
            if pd.notna(freq) and freq > 0:
                return round(float(freq), 2), a['rater']

    # Patient-level fallback (annotations for other segments of same patient)
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

    # Patient-level fallback
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
