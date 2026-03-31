#!/usr/bin/env python3
"""Label status report — shows completeness of all label types.

Usage:
    python code/data_management/label_status_report.py              # Print report
    python code/data_management/label_status_report.py --save-snapshot NAME  # Save snapshot for later comparison
    python code/data_management/label_status_report.py --verify-against NAME # Compare against saved snapshot
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
SNAPSHOT_DIR = LABELS_DIR / '.snapshots'


def load_all_sources():
    """Load segment_labels.csv and all supporting label files."""
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    jsons = {}
    for name in ['channel_involvement', 'channel_pseudolabels', 'discharge_times', 'rda_wave_labels']:
        path = LABELS_DIR / f'{name}.json'
        if path.exists():
            with open(str(path)) as f:
                jsons[name] = json.load(f)
        else:
            jsons[name] = {}

    # Load laterality batch reviews
    lat_review = {}
    for batch_dir in [LABELS_DIR, LABELS_DIR / 'archive_labels']:
        for bf in sorted(batch_dir.glob('*_laterality_batch*.json')):
            with open(str(bf)) as f:
                batch = json.load(f)
            if 'decisions' in batch:
                lat_review.update(batch['decisions'])

    ann = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))

    return sl, jsons, lat_review, ann


def compute_stats(sl, jsons, lat_review, ann):
    """Compute all label statistics. Returns a dict suitable for JSON serialization."""
    ci = jsons['channel_involvement']
    cp = jsons['channel_pseudolabels']
    dt = jsons['discharge_times']
    rw = jsons['rda_wave_labels']

    eeg_on_disk = {f.replace('.mat', '') for f in os.listdir(str(DATA_DIR / 'eeg')) if f.endswith('.mat')}
    batch_rejects = {m for m, v in lat_review.items() if v.get('label') == 'reject'}

    sl_od = sl[sl.mat_file.str.replace('.mat', '', regex=False).isin(eeg_on_disk)].copy()
    sl_od['batch_excluded'] = sl_od.mat_file.isin(batch_rejects)
    sl_active = sl_od[(sl_od.excluded != True) & (~sl_od.batch_excluded)]

    nv = pd.to_numeric(sl_active.iiic_n_votes, errors='coerce').fillna(0)

    ann_by_seg = ann.groupby('segment_id') if len(ann) > 0 else {}

    stats = {
        'timestamp': datetime.now().isoformat(),
        'total_on_disk': len(sl_od),
        'total_active': len(sl_active),
        'total_excluded': len(sl_od) - len(sl_active),
        'subtypes': {},
    }

    for sub in ['lpd', 'gpd', 'lrda', 'grda']:
        m = sl_active[sl_active.subtype == sub]
        total = len(m)
        if total == 0:
            continue
        pids = set(m.patient_id)
        sids = set(m.mat_file.str.replace('.mat', '', regex=False))
        mats = set(m.mat_file)
        m_nv = pd.to_numeric(m.iiic_n_votes, errors='coerce').fillna(0)

        # Pattern class multi-rater
        pattern_class = {
            'total': total,
            'n_patients': len(pids),
            'ge1_expert': int((m_nv >= 1).sum()),
            'ge5_experts': int((m_nv >= 5).sum()),
            'ge10_experts': int((m_nv >= 10).sum()),
        }

        # Laterality
        has_lat_sl = set(m[m.laterality.notna() & m.laterality.isin(['left', 'right', 'bilateral'])].mat_file)
        has_lat_ci = {row['mat_file'] for _, row in m.iterrows()
                      if row['patient_id'] in ci and ci[row['patient_id']].get('laterality') in ('left', 'right', 'bilateral')}
        has_lat_batch = {mf for mf in mats if mf in lat_review and lat_review[mf].get('label') in ('left', 'right')}
        lat_count = len(has_lat_sl | has_lat_ci | has_lat_batch)
        laterality = {'total': lat_count, 'pct': round(100 * lat_count / total, 1)}

        # Frequency
        has_expert = m.expert_freq_hz.notna()
        has_algo = m.algo_freq_hz.notna()
        freq_any = int((has_expert | has_algo).sum())
        freq_expert = int(has_expert.sum())
        freq_algo_only = int((has_algo & ~has_expert).sum())

        # Multi-rater frequency
        freq_raters = {}
        for sid in sids:
            if isinstance(ann_by_seg, pd.core.groupby.DataFrameGroupBy) and sid in ann_by_seg.groups:
                grp = ann_by_seg.get_group(sid)
                fr = grp[pd.to_numeric(grp['frequency_hz'], errors='coerce').notna()]['rater'].nunique()
                if fr > 0:
                    freq_raters[sid] = fr

        frequency = {
            'total': freq_any,
            'pct': round(100 * freq_any / total, 1),
            'expert': freq_expert,
            'algo_only': freq_algo_only,
            'ge1_expert_rater': sum(1 for v in freq_raters.values() if v >= 1),
            'ge3_expert_raters': sum(1 for v in freq_raters.values() if v >= 3),
            'ge5_expert_raters': sum(1 for v in freq_raters.values() if v >= 5),
        }

        # Spatial
        has_spat_sl = set(m[m.spatial_channels.notna()].patient_id)
        has_spat_cp = {pid for pid in pids if pid in cp}
        has_spat_ci = {pid for pid in pids if pid in ci}
        spat_count = m[m.patient_id.isin(has_spat_sl | has_spat_cp | has_spat_ci)].shape[0]

        spat_raters = {}
        for sid in sids:
            if isinstance(ann_by_seg, pd.core.groupby.DataFrameGroupBy) and sid in ann_by_seg.groups:
                grp = ann_by_seg.get_group(sid)
                sr = grp[grp['spatial_channels'].notna() & (grp['spatial_channels'] != '')]['rater'].nunique()
                if sr > 0:
                    spat_raters[sid] = sr

        spatial = {
            'total': spat_count,
            'pct': round(100 * spat_count / total, 1),
            'ge1_expert_rater': sum(1 for v in spat_raters.values() if v >= 1),
            'ge3_expert_raters': sum(1 for v in spat_raters.values() if v >= 3),
            'ge5_expert_raters': sum(1 for v in spat_raters.values() if v >= 5),
        }

        # Timing
        timing = {}
        if sub in ['lpd', 'gpd']:
            has_timing = {pid for pid in pids if pid in dt}
            tc = m[m.patient_id.isin(has_timing)].shape[0]
            timing = {'discharge_timing': tc, 'pct': round(100 * tc / total, 1)}
        elif sub in ['lrda', 'grda']:
            rw_keys = set(rw.keys())
            wc = m[m.mat_file.str.replace('.mat', '', regex=False).isin(rw_keys)].shape[0]
            timing = {'wave_timing': wc, 'pct': round(100 * wc / total, 1)}

        stats['subtypes'][sub] = {
            'pattern_class': pattern_class,
            'laterality': laterality,
            'frequency': frequency,
            'spatial': spatial,
            'timing': timing,
        }

    return stats


def print_report(stats):
    """Print a formatted label status report."""
    print()
    print("=" * 75)
    print(f"  LABEL STATUS REPORT — {stats['timestamp'][:10]}")
    print(f"  {stats['total_active']} active segments, {stats['total_excluded']} excluded")
    print("=" * 75)

    for sub in ['lpd', 'gpd', 'lrda', 'grda']:
        if sub not in stats['subtypes']:
            continue
        s = stats['subtypes'][sub]
        pc = s['pattern_class']
        print(f"\n{'─' * 75}")
        print(f"  {sub.upper()}: {pc['total']} segments ({pc['n_patients']} patients)")
        print(f"{'─' * 75}")

        print(f"  Pattern class:    {pc['total']:>5} (100%)    "
              f">=1: {pc['ge1_expert']:>5}   >=5: {pc['ge5_experts']:>5}   >=10: {pc['ge10_experts']:>5}")

        lat = s['laterality']
        print(f"  Laterality:       {lat['total']:>5} ({lat['pct']:>5.1f}%)  [single-rater MW]")

        freq = s['frequency']
        print(f"  Frequency:        {freq['total']:>5} ({freq['pct']:>5.1f}%)  "
              f"[expert: {freq['expert']}, algo-only: {freq['algo_only']}]")
        print(f"                    >=1: {freq['ge1_expert_rater']:>5}   "
              f">=3: {freq['ge3_expert_raters']:>5}   >=5: {freq['ge5_expert_raters']:>5}  expert raters")

        sp = s['spatial']
        print(f"  Spatial:          {sp['total']:>5} ({sp['pct']:>5.1f}%)")
        print(f"                    >=1: {sp['ge1_expert_rater']:>5}   "
              f">=3: {sp['ge3_expert_raters']:>5}   >=5: {sp['ge5_expert_raters']:>5}  expert raters")

        t = s['timing']
        if 'discharge_timing' in t:
            print(f"  Discharge timing: {t['discharge_timing']:>5} ({t['pct']:>5.1f}%)")
        elif 'wave_timing' in t:
            print(f"  Wave timing:      {t['wave_timing']:>5} ({t['pct']:>5.1f}%)")

    print()


def save_snapshot(stats, name):
    """Save stats as a JSON snapshot for later comparison."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f'{name}.json'
    with open(str(path), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Snapshot saved: {path}")


def verify_against(stats, name):
    """Compare current stats against a saved snapshot."""
    path = SNAPSHOT_DIR / f'{name}.json'
    if not path.exists():
        print(f"ERROR: No snapshot found at {path}")
        sys.exit(1)

    with open(str(path)) as f:
        old = json.load(f)

    print()
    print("=" * 75)
    print(f"  VERIFICATION: comparing against snapshot '{name}'")
    print("=" * 75)

    all_ok = True

    # Check total rows
    if stats['total_on_disk'] != old['total_on_disk']:
        print(f"  WARNING: total_on_disk changed: {old['total_on_disk']} -> {stats['total_on_disk']}")
        all_ok = False
    else:
        print(f"  Total on disk: {stats['total_on_disk']} (unchanged)")

    print(f"  Active: {old['total_active']} -> {stats['total_active']} "
          f"(excluded: {old['total_excluded']} -> {stats['total_excluded']})")

    for sub in ['lpd', 'gpd', 'lrda', 'grda']:
        if sub not in stats['subtypes'] or sub not in old['subtypes']:
            continue
        s_new = stats['subtypes'][sub]
        s_old = old['subtypes'][sub]

        changes = []

        # Check each label type
        for label_type in ['pattern_class', 'laterality', 'frequency', 'spatial', 'timing']:
            old_vals = s_old.get(label_type, {})
            new_vals = s_new.get(label_type, {})
            for key in set(list(old_vals.keys()) + list(new_vals.keys())):
                if key == 'pct' or key == 'n_patients':
                    continue
                ov = old_vals.get(key, 0)
                nv = new_vals.get(key, 0)
                if ov != nv:
                    changes.append((label_type, key, ov, nv))

        if changes:
            print(f"\n  {sub.upper()}:")
            for label_type, key, ov, nv in changes:
                direction = "+" if nv > ov else ""
                symbol = "OK" if nv >= ov else "WARNING"
                print(f"    {symbol}: {label_type}.{key}: {ov} -> {nv} ({direction}{nv - ov})")
                if nv < ov:
                    all_ok = False

    print()
    if all_ok:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED — review warnings above")
    print()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Label status report')
    parser.add_argument('--save-snapshot', metavar='NAME', help='Save current stats as named snapshot')
    parser.add_argument('--verify-against', metavar='NAME', help='Verify against a saved snapshot')
    args = parser.parse_args()

    sl, jsons, lat_review, ann = load_all_sources()
    stats = compute_stats(sl, jsons, lat_review, ann)

    if args.save_snapshot:
        save_snapshot(stats, args.save_snapshot)
        print_report(stats)
    elif args.verify_against:
        ok = verify_against(stats, args.verify_against)
        print_report(stats)
        if not ok:
            sys.exit(1)
    else:
        print_report(stats)


if __name__ == '__main__':
    main()
