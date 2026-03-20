"""
Build a channel-level PD (periodic discharge) dataset.

For each patient, loads first segment, extracts each of 18 bipolar channels,
and assigns a binary PD label per channel based on subtype and laterality.

Label logic:
  - GPD: all 18 channels = positive (1), subsampled to max 6 per patient
  - LPD with laterality label: ipsilateral = positive, contralateral = negative,
    midline [16,17] = excluded
  - LPD with spatial_channels: parse region codes -> channel indices via
    region_channel_map; involved = positive, non-involved = negative
  - LPD without any label: predict laterality using GBM classifier trained
    on all LPD patients with human laterality labels, then assign channels
  - LRDA/GRDA: all channels = negative (0) -- hard negatives
  - Other / excluded: skip

Saves to data/pd_channel_cache/channel_dataset.npz
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from optimization_harness_v2 import (
    _load_mat_as_bipolar, LEFT_INDICES, RIGHT_INDICES,
    LABELS_DIR, EEG_DIR, compute_sp_features,
    ALL_FEATURE_COLS, LATERALITY_FEATURE_COLS,
)
from pd_pointiness_acf import region_channel_map

CACHE_DIR = PROJECT_DIR / 'data' / 'pd_channel_cache'
MAX_GPD_CHANNELS = 6  # subsample GPD to avoid class imbalance

# Features used by the laterality classifier (laterality + frequency features)
LAT_CLASSIFIER_FEAT_COLS = LATERALITY_FEATURE_COLS + ['f_B', 'f_peaks', 'f_fft', 'f_tkeo', 'f_coh']


def _get_spatial_channel_indices(spatial_channels_str):
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


def _get_annotation_spatial_channels(df_annotations, patient_id):
    """Get spatial_channels from annotations for a patient (first non-empty)."""
    pat_annots = df_annotations[df_annotations['patient_id'] == str(patient_id)]
    for _, row in pat_annots.iterrows():
        if pd.notna(row.get('spatial_channels')) and str(row['spatial_channels']).strip():
            # Skip if no_pd or skipped
            if row.get('no_pd') == True or row.get('skipped') == True:
                continue
            return str(row['spatial_channels'])
    return None


def _train_laterality_classifier(df_patients, df_segments):
    """Train a GBM laterality classifier on all LPD patients with human labels.

    Returns:
        clf: trained GradientBoostingClassifier
        feat_medians: array of median values for imputation
    """
    from sklearn.ensemble import GradientBoostingClassifier

    print("\n  Training GBM laterality classifier on labeled LPD patients...")

    # Get LPD patients with left/right labels
    lpd_labeled = df_patients[
        (df_patients['subtype'].str.lower().str.strip() == 'lpd') &
        (df_patients['laterality'].isin(['left', 'right']))
    ].copy()

    print(f"    Found {len(lpd_labeled)} LPD patients with laterality labels")

    # Compute features for each labeled patient (using all segments)
    feat_rows = []
    lat_labels = []

    for _, pat_row in lpd_labeled.iterrows():
        pid = str(pat_row['patient_id'])
        laterality = pat_row['laterality'].lower().strip()
        lat_label = 0 if laterality == 'left' else 1  # 0=left, 1=right

        pat_segs = df_segments[df_segments['patient_id'] == pid]
        for _, seg_row in pat_segs.iterrows():
            mat_path = EEG_DIR / seg_row['mat_file']
            if not mat_path.exists():
                continue
            try:
                seg = _load_mat_as_bipolar(
                    mat_path, seg_row['montage'], seg_row['n_channels'])
            except Exception:
                continue

            if seg.shape[1] < 2000:
                continue

            try:
                feats = compute_sp_features(seg[:, :2000], is_gpd=0)
                feat_vec = [feats.get(c, np.nan) for c in LAT_CLASSIFIER_FEAT_COLS]
                feat_rows.append(feat_vec)
                lat_labels.append(lat_label)
            except Exception:
                continue

    X = np.array(feat_rows, dtype=float)
    y = np.array(lat_labels, dtype=int)

    print(f"    Training samples: {len(y)} segments (left={np.sum(y==0)}, right={np.sum(y==1)})")

    # Impute NaN with column medians
    feat_medians = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = np.isfinite(col)
        feat_medians[j] = np.median(col[finite]) if np.any(finite) else 0.0
        X[~finite, j] = feat_medians[j]

    # Train GBM with balanced class weights
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42,
    )
    # Compute inverse class frequency weights
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sw = np.array([weight_map[yi] for yi in y])
    clf.fit(X, y, sample_weight=sw)

    # Report training accuracy
    train_preds = clf.predict(X)
    train_acc = np.mean(train_preds == y)
    print(f"    Training accuracy: {train_acc:.3f}")

    return clf, feat_medians


def _predict_laterality(clf, feat_medians, seg, df_segments_for_patient=None):
    """Predict laterality for an LPD patient using the GBM classifier.

    Args:
        clf: trained classifier
        feat_medians: array of medians for imputation
        seg: (18, N) segment array (already loaded, first segment)

    Returns:
        'left' or 'right'
    """
    try:
        feats = compute_sp_features(seg[:, :2000].astype(np.float64), is_gpd=0)
        feat_vec = np.array([feats.get(c, np.nan) for c in LAT_CLASSIFIER_FEAT_COLS],
                           dtype=float).reshape(1, -1)

        # Impute NaN
        for j in range(feat_vec.shape[1]):
            if not np.isfinite(feat_vec[0, j]):
                feat_vec[0, j] = feat_medians[j]

        prob_right = clf.predict_proba(feat_vec)[0, 1]
        return 'right' if prob_right >= 0.5 else 'left'
    except Exception:
        return None


def main():
    print("Building channel-level PD dataset (with predicted laterality)...")

    # Load CSVs
    df_patients = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    df_patients['patient_id'] = df_patients['patient_id'].astype(str)

    df_segments = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    df_segments['patient_id'] = df_segments['patient_id'].astype(str)

    df_annotations = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    df_annotations['patient_id'] = df_annotations['patient_id'].astype(str)

    # Filter excluded patients
    df_patients = df_patients[df_patients['excluded'] != True].copy()

    # Step 1: Train laterality classifier on all labeled LPD patients
    lat_clf, lat_feat_medians = _train_laterality_classifier(df_patients, df_segments)

    # Collect channels
    all_channels = []
    all_labels = []
    all_patient_ids = []
    all_channel_indices = []
    all_subtypes = []

    stats = {
        'gpd_positive': 0, 'lpd_lateral_positive': 0, 'lpd_lateral_negative': 0,
        'lpd_spatial_positive': 0, 'lpd_spatial_negative': 0,
        'lpd_predicted_positive': 0, 'lpd_predicted_negative': 0,
        'rda_negative': 0, 'skipped_no_eeg': 0, 'skipped_other': 0,
        'lpd_no_label_info': 0,
        'lpd_predicted_left': 0, 'lpd_predicted_right': 0,
    }

    rng = np.random.RandomState(42)
    n_human_lat = 0
    n_predicted_lat = 0

    for idx, pat_row in df_patients.iterrows():
        pid = str(pat_row['patient_id'])
        subtype = str(pat_row['subtype']).lower().strip()

        # Skip "other" or unexpected subtypes
        if subtype not in ('gpd', 'lpd', 'grda', 'lrda'):
            stats['skipped_other'] += 1
            continue

        # Get first segment for this patient
        pat_segs = df_segments[df_segments['patient_id'] == pid]
        if len(pat_segs) == 0:
            stats['skipped_no_eeg'] += 1
            continue

        seg_row = pat_segs.iloc[0]
        mat_path = EEG_DIR / seg_row['mat_file']
        if not mat_path.exists():
            stats['skipped_no_eeg'] += 1
            continue

        try:
            seg = _load_mat_as_bipolar(
                mat_path, seg_row['montage'], seg_row['n_channels'])
        except Exception as e:
            stats['skipped_no_eeg'] += 1
            continue

        # seg is (18, N) -- ensure exactly 2000 samples
        if seg.shape[1] < 2000:
            stats['skipped_no_eeg'] += 1
            continue
        seg = seg[:, :2000].astype(np.float32)
        n_ch = seg.shape[0]

        # Assign labels per channel
        if subtype == 'gpd':
            # All channels positive, subsample to MAX_GPD_CHANNELS
            ch_indices = list(range(n_ch))
            if len(ch_indices) > MAX_GPD_CHANNELS:
                ch_indices = sorted(rng.choice(ch_indices, MAX_GPD_CHANNELS, replace=False))
            for ch in ch_indices:
                all_channels.append(seg[ch])
                all_labels.append(1)
                all_patient_ids.append(pid)
                all_channel_indices.append(ch)
                all_subtypes.append(subtype)
                stats['gpd_positive'] += 1

        elif subtype == 'lpd':
            laterality = pat_row.get('laterality', '')
            if pd.isna(laterality):
                laterality = ''
            laterality = str(laterality).lower().strip()

            if laterality in ('left', 'right'):
                # Use human laterality label
                n_human_lat += 1
                if laterality == 'left':
                    pos_indices = LEFT_INDICES
                    neg_indices = RIGHT_INDICES
                else:
                    pos_indices = RIGHT_INDICES
                    neg_indices = LEFT_INDICES
                # Midline [16,17] excluded
                for ch in pos_indices:
                    if ch < n_ch:
                        all_channels.append(seg[ch])
                        all_labels.append(1)
                        all_patient_ids.append(pid)
                        all_channel_indices.append(ch)
                        all_subtypes.append(subtype)
                        stats['lpd_lateral_positive'] += 1
                for ch in neg_indices:
                    if ch < n_ch:
                        all_channels.append(seg[ch])
                        all_labels.append(0)
                        all_patient_ids.append(pid)
                        all_channel_indices.append(ch)
                        all_subtypes.append(subtype)
                        stats['lpd_lateral_negative'] += 1

            elif laterality == 'bilateral':
                # Bilateral: all non-midline channels positive
                n_human_lat += 1
                for ch in range(min(n_ch, 16)):
                    all_channels.append(seg[ch])
                    all_labels.append(1)
                    all_patient_ids.append(pid)
                    all_channel_indices.append(ch)
                    all_subtypes.append(subtype)
                    stats['lpd_lateral_positive'] += 1

            else:
                # Try spatial_channels from annotations first
                spatial_str = _get_annotation_spatial_channels(df_annotations, pid)
                used_spatial = False
                if spatial_str:
                    pos_ch = _get_spatial_channel_indices(spatial_str)
                    if pos_ch:
                        pos_set = set(pos_ch)
                        for ch in range(min(n_ch, 18)):
                            if ch in (16, 17):
                                continue  # skip midline
                            all_channels.append(seg[ch])
                            label = 1 if ch in pos_set else 0
                            all_labels.append(label)
                            all_patient_ids.append(pid)
                            all_channel_indices.append(ch)
                            all_subtypes.append(subtype)
                            if label == 1:
                                stats['lpd_spatial_positive'] += 1
                            else:
                                stats['lpd_spatial_negative'] += 1
                        used_spatial = True
                        n_human_lat += 1

                if not used_spatial:
                    # No human label available -- predict laterality with GBM
                    predicted_lat = _predict_laterality(lat_clf, lat_feat_medians, seg)
                    if predicted_lat is not None:
                        n_predicted_lat += 1
                        if predicted_lat == 'left':
                            pos_indices = LEFT_INDICES
                            neg_indices = RIGHT_INDICES
                            stats['lpd_predicted_left'] += 1
                        else:
                            pos_indices = RIGHT_INDICES
                            neg_indices = LEFT_INDICES
                            stats['lpd_predicted_right'] += 1

                        # Midline [16,17] excluded
                        for ch in pos_indices:
                            if ch < n_ch:
                                all_channels.append(seg[ch])
                                all_labels.append(1)
                                all_patient_ids.append(pid)
                                all_channel_indices.append(ch)
                                all_subtypes.append(subtype)
                                stats['lpd_predicted_positive'] += 1
                        for ch in neg_indices:
                            if ch < n_ch:
                                all_channels.append(seg[ch])
                                all_labels.append(0)
                                all_patient_ids.append(pid)
                                all_channel_indices.append(ch)
                                all_subtypes.append(subtype)
                                stats['lpd_predicted_negative'] += 1
                    else:
                        stats['lpd_no_label_info'] += 1

        elif subtype in ('grda', 'lrda'):
            # All channels negative (hard negatives)
            # Subsample to 6 channels to match GPD subsampling
            ch_indices = list(range(min(n_ch, 18)))
            if len(ch_indices) > MAX_GPD_CHANNELS:
                ch_indices = sorted(rng.choice(ch_indices, MAX_GPD_CHANNELS, replace=False))
            for ch in ch_indices:
                all_channels.append(seg[ch])
                all_labels.append(0)
                all_patient_ids.append(pid)
                all_channel_indices.append(ch)
                all_subtypes.append(subtype)
                stats['rda_negative'] += 1

    # Convert to arrays
    channels = np.array(all_channels, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int8)
    patient_ids = np.array(all_patient_ids, dtype=str)
    channel_indices = np.array(all_channel_indices, dtype=np.int8)
    subtypes = np.array(all_subtypes, dtype=str)

    # Remove channels with NaN/Inf values
    finite_mask = np.all(np.isfinite(channels), axis=1)
    n_removed = int(np.sum(~finite_mask))
    if n_removed > 0:
        print(f"\n  Removed {n_removed} channels with NaN/Inf values")
        channels = channels[finite_mask]
        labels = labels[finite_mask]
        patient_ids = patient_ids[finite_mask]
        channel_indices = channel_indices[finite_mask]
        subtypes = subtypes[finite_mask]

    # Save
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / 'channel_dataset.npz'
    np.savez(
        str(out_path),
        channels=channels,
        labels=labels,
        patient_ids=patient_ids,
        channel_indices=channel_indices,
        subtypes=subtypes,
    )

    # Print statistics
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    n_total = len(labels)
    n_patients = len(set(patient_ids))

    print(f"\n{'='*60}")
    print(f"Channel-level PD dataset saved to: {out_path}")
    print(f"{'='*60}")
    print(f"  Total channels: {n_total}")
    print(f"  Positive (PD):  {n_pos} ({100*n_pos/n_total:.1f}%)")
    print(f"  Negative:       {n_neg} ({100*n_neg/n_total:.1f}%)")
    print(f"  Unique patients: {n_patients}")
    print(f"\n  Laterality source:")
    print(f"    Human labels:    {n_human_lat} patients")
    print(f"    GBM predicted:   {n_predicted_lat} patients")
    print(f"\n  Breakdown by source:")
    print(f"    GPD positive channels:              {stats['gpd_positive']}")
    print(f"    LPD human-lateral positive:         {stats['lpd_lateral_positive']}")
    print(f"    LPD human-lateral negative:         {stats['lpd_lateral_negative']}")
    print(f"    LPD spatial positive:               {stats['lpd_spatial_positive']}")
    print(f"    LPD spatial negative:               {stats['lpd_spatial_negative']}")
    print(f"    LPD predicted-lateral positive:     {stats['lpd_predicted_positive']}")
    print(f"    LPD predicted-lateral negative:     {stats['lpd_predicted_negative']}")
    print(f"      (predicted left: {stats['lpd_predicted_left']}, right: {stats['lpd_predicted_right']})")
    print(f"    RDA negative (hard negatives):      {stats['rda_negative']}")
    print(f"    LPD skipped (no label/prediction):  {stats['lpd_no_label_info']}")
    print(f"    Skipped (no EEG):                   {stats['skipped_no_eeg']}")
    print(f"    Skipped (other subtype):            {stats['skipped_other']}")
    print(f"{'='*60}")

    return channels, labels, patient_ids, channel_indices, subtypes


if __name__ == '__main__':
    main()
