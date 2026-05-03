#!/usr/bin/env python3
"""Pre-retrain data inventory check.

Verifies that every mat_file referenced in segment_labels.csv exists locally,
that every label file loads cleanly, and that every model checkpoint listed
in the manuscript's training-data table is present. Writes a summary to
``results/verify_local_data_report.json`` so a from-scratch retrain attempt
can be planned with a known data-coverage baseline.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
OUT_PATH = PROJECT_DIR / 'results' / 'verify_local_data_report.json'

# Model checkpoints expected to exist if the trained models are to be reused
# rather than retrained from scratch.
EXPECTED_CHECKPOINTS = [
    'data/pd_channel_cache/cnn_attn_fold0.pt',
    'data/pd_channel_cache/cnn_attn_fold1.pt',
    'data/pd_channel_cache/cnn_attn_fold2.pt',
    'data/pd_channel_cache/cnn_attn_fold3.pt',
    'data/pd_channel_cache/cnn_attn_fold4.pt',
    'data/hemi_cache/hemi_cet_v2/hemi_cet_fold0.pt',
    'data/hemi_cache/hemi_cet_v2/hemi_cet_fold1.pt',
    'data/hemi_cache/hemi_cet_v2/hemi_cet_fold2.pt',
    'data/hemi_cache/hemi_cet_v2/hemi_cet_fold3.pt',
    'data/hemi_cache/hemi_cet_v2/hemi_cet_fold4.pt',
    'data/labels/independent_expert_v1/lrda_laterality_classifier.pkl',
    'data/labels/independent_expert_v1/hard_case_classifier.pkl',
]

# Required label files for the published numbers.
REQUIRED_LABEL_FILES = [
    'data/labels/labels.csv',
    'data/labels/segments.csv',
    'data/labels/segment_labels.csv',
    'data/labels/annotations.csv',
    'data/labels/discharge_times.json',
    'data/labels/channel_involvement.json',
    'data/labels/channel_pseudolabels.json',
    'data/labels/predictions.json',
    'data/labels/rda_wave_labels.json',
    'data/labels/independent_expert_v1/v14_predictions.json',
    'data/labels/archive_labels/patients.csv',
    'data/labels/archive_labels/channel_involvement_predictions.json',
]


def report_section(title, payload, *, file=sys.stdout):
    print(f'\n{"="*60}\n  {title}\n{"="*60}', file=file)
    if isinstance(payload, dict):
        for k, v in payload.items():
            print(f'  {k}: {v}', file=file)
    else:
        for line in str(payload).splitlines():
            print(f'  {line}', file=file)


def main():
    report = {}

    # ---------------- Label-file presence ----------------
    label_status = {}
    for rel in REQUIRED_LABEL_FILES:
        p = PROJECT_DIR / rel
        ok = p.exists()
        size_mb = round(p.stat().st_size / 1e6, 3) if ok else None
        label_status[rel] = {'present': ok, 'size_mb': size_mb}
    report['label_files'] = label_status
    missing_labels = [k for k, v in label_status.items() if not v['present']]
    report_section('Label files', {k: ('OK' if v['present'] else 'MISSING') + (f" ({v['size_mb']:.2f} MB)" if v['size_mb'] is not None else '') for k, v in label_status.items()})
    if missing_labels:
        report_section('MISSING label files', missing_labels)

    # ---------------- Checkpoints ----------------
    ckpt_status = {}
    for rel in EXPECTED_CHECKPOINTS:
        p = PROJECT_DIR / rel
        ok = p.exists()
        size_mb = round(p.stat().st_size / 1e6, 3) if ok else None
        ckpt_status[rel] = {'present': ok, 'size_mb': size_mb}
    report['checkpoints'] = ckpt_status
    report_section('Model checkpoints', {k: ('OK' if v['present'] else 'MISSING') + (f" ({v['size_mb']:.2f} MB)" if v['size_mb'] is not None else '') for k, v in ckpt_status.items()})

    # ---------------- segment_labels.csv -> data/eeg/ coverage ----------------
    sl_path = LABELS_DIR / 'segment_labels.csv'
    if not sl_path.exists():
        print('segment_labels.csv missing; cannot run EEG coverage check.')
        sys.exit(1)
    sl = pd.read_csv(sl_path)
    print(f'\nsegment_labels.csv loaded: {len(sl)} rows, {sl["mat_file"].nunique()} unique mat_file refs')

    # Build flat list of all .mat files anywhere under data/eeg
    eeg_files = set()
    if EEG_DIR.exists():
        eeg_files = {p.name for p in EEG_DIR.rglob('*.mat')}
    print(f'data/eeg/ recursive scan: {len(eeg_files)} .mat files found on disk')

    # Per-subtype coverage
    referenced = set(sl['mat_file'].dropna().unique())
    missing = referenced - eeg_files
    extra = eeg_files - referenced

    coverage = {
        'referenced_in_segment_labels': len(referenced),
        'on_disk': len(eeg_files),
        'missing_on_disk': len(missing),
        'extra_on_disk_not_referenced': len(extra),
    }
    by_subtype = {}
    for sub in ('lpd', 'gpd', 'lrda', 'grda', 'seizure', 'other', 'bipd'):
        sub_refs = set(sl[sl.subtype == sub]['mat_file'].dropna())
        sub_missing = sub_refs - eeg_files
        by_subtype[sub] = {
            'n_referenced': len(sub_refs),
            'n_missing': len(sub_missing),
        }
    coverage['by_subtype'] = by_subtype
    report['eeg_coverage'] = coverage
    report_section('EEG file coverage', coverage)
    if missing:
        sample = sorted(missing)[:10]
        report_section(f'Sample of missing mat_files ({len(missing)} total, first 10)', sample)

    # Save full lists of missing/extra for follow-up
    report['eeg_missing_full'] = sorted(missing)
    report['eeg_extra_full'] = sorted(list(extra)[:200])  # cap to keep file small

    # ---------------- Sanity: each label JSON parses ----------------
    json_load = {}
    for rel in REQUIRED_LABEL_FILES:
        if not rel.endswith('.json'):
            continue
        p = PROJECT_DIR / rel
        if not p.exists():
            json_load[rel] = 'MISSING'
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            json_load[rel] = f'OK ({len(d) if hasattr(d, "__len__") else "?"} entries)'
        except Exception as e:
            json_load[rel] = f'PARSE_ERROR: {e}'
    report['json_parse'] = json_load
    report_section('JSON parse check', json_load)

    # ---------------- IIIC coverage breakdown ----------------
    iiic_summary = {}
    if 'subtype_source' in sl.columns:
        for source, sub in sl.groupby('subtype_source'):
            iiic_summary[source] = len(sub)
    if 'iiic_n_votes' in sl.columns:
        iiic_summary['with_iiic_votes_ge_10'] = int((pd.to_numeric(sl.get('iiic_n_votes'), errors='coerce') >= 10).sum())
    report['iiic_summary'] = iiic_summary
    report_section('IIIC source breakdown (subtype_source value counts)', iiic_summary)

    # ---------------- Save ----------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nWrote full report to {OUT_PATH.relative_to(PROJECT_DIR)}')

    # Bottom-line summary
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  Label files: {sum(1 for v in label_status.values() if v["present"])}/{len(label_status)} present')
    print(f'  Checkpoints: {sum(1 for v in ckpt_status.values() if v["present"])}/{len(ckpt_status)} present')
    print(f'  EEG coverage: {len(referenced) - len(missing)}/{len(referenced)} referenced .mat files on disk ({len(missing)} missing)')


if __name__ == '__main__':
    main()
