"""
Build comprehensive channel-level pseudolabel JSON for all patients.

For each patient in patients.csv, assigns per-channel pd_label and rda_label
with source and confidence metadata.

Sources (in priority order for LPD):
  1. Ground truth MW review (from channel_involvement.json, review_status=ground_truth)
  2. Spatial annotation channels (from annotations.csv spatial_channels column)
  3. Human laterality label (from patients.csv or channel_involvement.json)
  4. Predicted laterality (from channel_involvement_predictions.json)
  5. Bilateral / unknown fallbacks

Output: data/labels/channel_pseudolabels.json
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_DIR / 'code'
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'

sys.path.insert(0, str(CODE_DIR))
from pd_pointiness_acf import region_channel_map

LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]
MIDLINE_INDICES = [16, 17]
ALL_CHANNELS = list(range(18))


def get_spatial_channel_indices(spatial_channels_str):
    """Parse spatial_channels string (e.g. 'LF RF LT') into channel indices."""
    if pd.isna(spatial_channels_str) or not str(spatial_channels_str).strip():
        return None
    regions = str(spatial_channels_str).strip().split()
    indices = set()
    for region in regions:
        region = region.strip()
        if region in region_channel_map:
            indices.update(region_channel_map[region])
    return sorted(indices) if indices else None


def get_annotation_spatial_channels(df_annotations, patient_id):
    """Get spatial_channels from annotations for a patient (first non-empty, non-skipped)."""
    pat_annots = df_annotations[df_annotations['patient_id'] == str(patient_id)]
    for _, row in pat_annots.iterrows():
        if pd.notna(row.get('spatial_channels')) and str(row['spatial_channels']).strip():
            if row.get('no_pd') is True or row.get('skipped') is True:
                continue
            return str(row['spatial_channels'])
    return None


def make_channel_entry(pd_label, rda_label, source, confidence):
    """Create a single channel label entry."""
    return {
        'pd_label': pd_label,
        'rda_label': rda_label,
        'source': source,
        'confidence': confidence,
    }


def assign_ground_truth_channels(patient_id, ci_entry, pred_entry):
    """Assign channels for MW-reviewed ground_truth cases.

    Uses the corrected n_involved + laterality from channel_involvement.json.
    For channel indices, reconstructs from prediction probabilities by taking
    top-N channels by probability (for corrected cases) or using prediction
    involved_channels directly (for 'correct' cases).
    """
    channels = {}
    n_involved = ci_entry['n_involved']
    laterality = ci_entry.get('laterality', 'none')
    subtype = ci_entry.get('subtype', 'unknown')
    review_source = ci_entry.get('review_source', 'unknown')

    if pred_entry and review_source == 'correct':
        # MW confirmed the prediction was correct - use prediction channels
        involved = set(pred_entry.get('involved_channels', []))
    elif pred_entry and n_involved > 0:
        # MW corrected n_involved - reconstruct by taking top-N by prob
        probs = pred_entry.get('channel_probs', [])
        if probs and len(probs) == 18:
            sorted_indices = sorted(range(18), key=lambda i: probs[i], reverse=True)
            involved = set(sorted_indices[:n_involved])
        else:
            involved = set()
    elif n_involved == 0:
        involved = set()
    else:
        # No prediction data available - fall back to laterality-based assignment
        involved = None

    # Subtype determines whether "involved" means PD or RDA
    is_rda_subtype = subtype in ('grda', 'lrda')

    if involved is not None:
        for ch in ALL_CHANNELS:
            if is_rda_subtype:
                # For GRDA/LRDA: involved = RDA-positive, all PD-negative
                if ch in involved:
                    channels[str(ch)] = make_channel_entry(0, 1, 'ground_truth_mw', 'ground_truth')
                else:
                    channels[str(ch)] = make_channel_entry(0, 0, 'ground_truth_mw', 'ground_truth')
            else:
                # For GPD/LPD: involved = PD-positive, all RDA-negative
                if ch in involved:
                    channels[str(ch)] = make_channel_entry(1, 0, 'ground_truth_mw', 'ground_truth')
                else:
                    channels[str(ch)] = make_channel_entry(0, 0, 'ground_truth_mw', 'ground_truth')
    else:
        # Fallback: use laterality + n_involved as best approximation
        if laterality in ('left', 'right'):
            pos = LEFT_INDICES if laterality == 'left' else RIGHT_INDICES
            neg = RIGHT_INDICES if laterality == 'left' else LEFT_INDICES
            pd_val = 0 if is_rda_subtype else 1
            rda_val = 1 if is_rda_subtype else 0
            for ch in pos:
                channels[str(ch)] = make_channel_entry(pd_val, rda_val, 'ground_truth_mw', 'ground_truth')
            for ch in neg:
                channels[str(ch)] = make_channel_entry(0, 0, 'ground_truth_mw', 'ground_truth')
            for ch in MIDLINE_INDICES:
                channels[str(ch)] = make_channel_entry(None, 0 if not is_rda_subtype else None,
                                                       'ground_truth_mw_midline', None)
        elif laterality == 'bilateral':
            pd_val = 0 if is_rda_subtype else 1
            rda_val = 1 if is_rda_subtype else 0
            for ch in range(16):
                channels[str(ch)] = make_channel_entry(pd_val, rda_val, 'ground_truth_mw', 'ground_truth')
            for ch in MIDLINE_INDICES:
                channels[str(ch)] = make_channel_entry(None, None if is_rda_subtype else 0,
                                                       'ground_truth_mw_midline', None)
        else:
            # No laterality, no predictions: mark all unknown
            for ch in ALL_CHANNELS:
                channels[str(ch)] = make_channel_entry(None, None if is_rda_subtype else 0,
                                                       'ground_truth_mw_unknown', None)

    return channels


def assign_gpd_channels():
    """All 18 channels PD-positive, RDA-negative."""
    channels = {}
    for ch in ALL_CHANNELS:
        channels[str(ch)] = make_channel_entry(1, 0, 'gpd_all_channels', 'medium')
    return channels


def assign_grda_channels():
    """All 18 channels PD-negative, RDA-positive."""
    channels = {}
    for ch in ALL_CHANNELS:
        channels[str(ch)] = make_channel_entry(0, 1, 'grda_all_channels', 'medium')
        channels[str(ch)]['pd_source'] = 'rda_negative_pd_case'
        channels[str(ch)]['pd_confidence'] = 'high'
    # Flatten: use separate source/confidence for pd and rda
    result = {}
    for ch in ALL_CHANNELS:
        result[str(ch)] = {
            'pd_label': 0,
            'rda_label': 1,
            'source': 'rda_negative_pd_case',
            'confidence': 'high',
            'rda_source': 'grda_all_channels',
            'rda_confidence': 'medium',
        }
    return result


def assign_lrda_channels():
    """All 18 channels PD-negative, RDA-positive (low confidence - lateralized but unknown side)."""
    result = {}
    for ch in ALL_CHANNELS:
        result[str(ch)] = {
            'pd_label': 0,
            'rda_label': 1,
            'source': 'rda_negative_pd_case',
            'confidence': 'high',
            'rda_source': 'lrda_all_channels',
            'rda_confidence': 'low',
        }
    return result


def assign_lpd_lateral_channels(laterality, source_prefix, confidence):
    """Assign LPD channels based on laterality (left/right/bilateral)."""
    channels = {}
    if laterality == 'left':
        for ch in LEFT_INDICES:
            channels[str(ch)] = make_channel_entry(1, 0, f'ipsilateral_{source_prefix}', confidence)
        for ch in RIGHT_INDICES:
            channels[str(ch)] = make_channel_entry(0, 0, f'contralateral_{source_prefix}', confidence)
        for ch in MIDLINE_INDICES:
            channels[str(ch)] = make_channel_entry(None, 0, 'midline_excluded', None)
    elif laterality == 'right':
        for ch in RIGHT_INDICES:
            channels[str(ch)] = make_channel_entry(1, 0, f'ipsilateral_{source_prefix}', confidence)
        for ch in LEFT_INDICES:
            channels[str(ch)] = make_channel_entry(0, 0, f'contralateral_{source_prefix}', confidence)
        for ch in MIDLINE_INDICES:
            channels[str(ch)] = make_channel_entry(None, 0, 'midline_excluded', None)
    elif laterality == 'bilateral':
        for ch in range(16):
            channels[str(ch)] = make_channel_entry(1, 0, 'bilateral_all', 'medium')
        for ch in MIDLINE_INDICES:
            channels[str(ch)] = make_channel_entry(None, 0, 'midline_excluded', None)
    return channels


def assign_lpd_spatial_channels(spatial_str):
    """Assign LPD channels from spatial annotation region codes."""
    involved = set(get_spatial_channel_indices(spatial_str))
    channels = {}
    for ch in ALL_CHANNELS:
        if ch in involved:
            channels[str(ch)] = make_channel_entry(1, 0, 'spatial_annotation', 'high')
        else:
            channels[str(ch)] = make_channel_entry(0, 0, 'spatial_annotation_negative', 'medium-high')
    return channels


def assign_lpd_unknown_channels():
    """LPD with no laterality info - all null."""
    channels = {}
    for ch in ALL_CHANNELS:
        channels[str(ch)] = make_channel_entry(None, 0, 'unknown', None)
    return channels


def main():
    print("Building comprehensive channel-level pseudolabel JSON...")
    print(f"Project dir: {PROJECT_DIR}")

    # Load data sources
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)

    df_annotations = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    df_annotations['patient_id'] = df_annotations['patient_id'].astype(str)

    with open(str(LABELS_DIR / 'channel_involvement.json')) as f:
        channel_involvement = json.load(f)

    with open(str(LABELS_DIR / 'channel_involvement_predictions.json')) as f:
        channel_predictions = json.load(f)

    print(f"\nData sources loaded:")
    print(f"  patients.csv: {len(df_patients)} patients")
    print(f"  annotations.csv: {len(df_annotations)} rows")
    print(f"  channel_involvement.json: {len(channel_involvement)} entries")
    print(f"  channel_involvement_predictions.json: {len(channel_predictions)} entries")

    # Build output
    output = {}
    source_counts = Counter()
    confidence_counts = Counter()
    pd_positive = 0
    pd_negative = 0
    pd_null = 0
    rda_positive = 0
    rda_negative = 0
    rda_null = 0
    label_type_counts = Counter()
    subtype_counts = Counter()

    for _, pat_row in df_patients.iterrows():
        pid = str(pat_row['patient_id'])
        subtype = str(pat_row['subtype']).lower().strip()
        laterality = pat_row.get('laterality', '')
        if pd.isna(laterality):
            laterality = ''
        laterality = str(laterality).lower().strip()

        ci_entry = channel_involvement.get(pid)
        pred_entry = channel_predictions.get(pid)

        patient_record = {
            'subtype': subtype,
            'label_type': None,
            'channels': {},
        }

        # Determine if ground truth or pseudolabel
        is_ground_truth = (ci_entry is not None and
                           ci_entry.get('review_status') == 'ground_truth')

        if is_ground_truth:
            patient_record['label_type'] = 'ground_truth'
            patient_record['channels'] = assign_ground_truth_channels(
                pid, ci_entry, pred_entry)

        elif subtype == 'gpd':
            patient_record['label_type'] = 'pseudolabel'
            patient_record['channels'] = assign_gpd_channels()

        elif subtype == 'grda':
            patient_record['label_type'] = 'pseudolabel'
            patient_record['channels'] = assign_grda_channels()

        elif subtype == 'lrda':
            patient_record['label_type'] = 'pseudolabel'
            patient_record['channels'] = assign_lrda_channels()

        elif subtype == 'lpd':
            patient_record['label_type'] = 'pseudolabel'

            # Priority 1: spatial_channels from annotations
            spatial_str = get_annotation_spatial_channels(df_annotations, pid)
            spatial_indices = get_spatial_channel_indices(spatial_str) if spatial_str else None

            if spatial_indices:
                patient_record['channels'] = assign_lpd_spatial_channels(spatial_str)

            # Priority 2: human laterality label from patients.csv
            elif laterality in ('left', 'right'):
                patient_record['channels'] = assign_lpd_lateral_channels(
                    laterality, 'human_lat', 'medium-high')

            elif laterality == 'bilateral':
                patient_record['channels'] = assign_lpd_lateral_channels(
                    laterality, 'human_lat', 'medium')

            # Priority 3: predicted laterality from CNN predictions
            elif pred_entry and pred_entry.get('predicted_laterality') in ('left', 'right'):
                pred_lat = pred_entry['predicted_laterality']
                patient_record['channels'] = assign_lpd_lateral_channels(
                    pred_lat, 'predicted_lat', 'medium')

            # Priority 4: no info
            else:
                patient_record['channels'] = assign_lpd_unknown_channels()

        else:
            # Unknown subtype - skip
            continue

        # Tally statistics
        label_type_counts[patient_record['label_type']] += 1
        subtype_counts[subtype] += 1

        for ch_str, ch_data in patient_record['channels'].items():
            src = ch_data.get('source', 'unknown')
            conf = ch_data.get('confidence')
            source_counts[src] += 1
            confidence_counts[str(conf)] += 1

            pd_val = ch_data.get('pd_label')
            if pd_val == 1:
                pd_positive += 1
            elif pd_val == 0:
                pd_negative += 1
            else:
                pd_null += 1

            rda_val = ch_data.get('rda_label')
            if rda_val == 1:
                rda_positive += 1
            elif rda_val == 0:
                rda_negative += 1
            else:
                rda_null += 1

        output[pid] = patient_record

    # Save
    out_path = LABELS_DIR / 'channel_pseudolabels.json'
    with open(str(out_path), 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    total_patients = len(output)
    total_channels = sum(len(v['channels']) for v in output.values())

    print(f"\n{'='*70}")
    print(f"Channel pseudolabels saved to: {out_path}")
    print(f"{'='*70}")

    print(f"\n  Total patients: {total_patients}")
    print(f"  Total channels: {total_channels}")

    print(f"\n  By label_type:")
    for lt, cnt in sorted(label_type_counts.items()):
        print(f"    {lt}: {cnt}")

    print(f"\n  By subtype:")
    for st, cnt in sorted(subtype_counts.items()):
        print(f"    {st}: {cnt}")

    print(f"\n  PD labels:")
    print(f"    pd_label=1 (positive): {pd_positive}")
    print(f"    pd_label=0 (negative): {pd_negative}")
    print(f"    pd_label=null:         {pd_null}")

    print(f"\n  RDA labels:")
    print(f"    rda_label=1 (positive): {rda_positive}")
    print(f"    rda_label=0 (negative): {rda_negative}")
    print(f"    rda_label=null:         {rda_null}")

    print(f"\n  By source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src}: {cnt}")

    print(f"\n  By confidence:")
    for conf, cnt in sorted(confidence_counts.items(), key=lambda x: -x[1]):
        print(f"    {conf}: {cnt}")

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
