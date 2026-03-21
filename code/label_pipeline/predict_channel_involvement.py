"""
Predict which EEG channels contain periodic discharges for each patient.

Uses an ensemble of 5-fold CNN+Attention models to get per-channel PD
probability, then determines overall laterality from the channel distribution.

Saves results to data/labels/channel_involvement_predictions.json

Usage:
    conda run -n foe_dl python code/label_pipeline/predict_channel_involvement.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

import torch

# ── Path setup ────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from optimization_harness_v2 import load_dataset, LEFT_INDICES, RIGHT_INDICES

DATA_DIR = PROJECT_DIR / 'data'
LABELS_DIR = DATA_DIR / 'labels'
CACHE_DIR = DATA_DIR / 'pd_channel_cache'
DEVICE = torch.device('cpu')

# Channel names for reference
BIPOLAR_CHANNELS = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

THRESHOLD = 0.5


def load_fold_models():
    """Load all 5 fold CNN+Attention models."""
    models = {}
    for fold in range(5):
        model_path = CACHE_DIR / f'cnn_attn_fold{fold}.pt'
        if not model_path.exists():
            print(f"  WARNING: model not found: {model_path}")
            continue
        model = ChannelPDNetAttention().to(DEVICE)
        state = torch.load(str(model_path), map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        models[fold] = model
        print(f"  Loaded fold {fold}: {model_path.name}")
    return models


def predict_laterality(left_probs, right_probs, threshold=0.15):
    """Determine laterality from left/right channel PD probabilities.

    Args:
        left_probs: PD probabilities for left hemisphere channels
        right_probs: PD probabilities for right hemisphere channels
        threshold: minimum difference in mean prob to declare laterality

    Returns:
        'left', 'right', 'bilateral', or 'midline'
    """
    mean_left = np.mean(left_probs)
    mean_right = np.mean(right_probs)
    diff = mean_left - mean_right

    if abs(diff) < threshold:
        # Both sides similar
        if mean_left > THRESHOLD and mean_right > THRESHOLD:
            return 'bilateral'
        elif mean_left < THRESHOLD and mean_right < THRESHOLD:
            return 'none'
        else:
            return 'bilateral'
    elif diff > 0:
        return 'left'
    else:
        return 'right'


@torch.no_grad()
def predict_channels_for_patient(models, segment):
    """Run ensemble of CNN models on each channel of a segment.

    Args:
        models: dict fold_idx -> model
        segment: (18, N) numpy array

    Returns:
        channel_probs: (18,) array of PD probabilities (ensemble mean)
    """
    n_ch = min(segment.shape[0], 18)
    n_models = len(models)

    # Accumulate probabilities across all models
    prob_accum = np.zeros(n_ch)

    for fold_idx, model in models.items():
        for ch in range(n_ch):
            ch_data = segment[ch, :2000].astype(np.float32).copy()

            # Skip channels with NaN/Inf
            if not np.all(np.isfinite(ch_data)):
                continue

            # Per-channel z-score normalization
            mu = np.mean(ch_data)
            std = np.std(ch_data)
            if std > 1e-8:
                ch_data = (ch_data - mu) / std
            else:
                ch_data = ch_data - mu

            # Shape: (1, 1, 2000)
            x = torch.from_numpy(ch_data[np.newaxis, np.newaxis, :])
            pd_prob, _freq_pred, _attn = model(x)
            prob_accum[ch] += pd_prob.item()

    # Average across models
    channel_probs = prob_accum / max(n_models, 1)

    # Pad to 18 if needed
    if n_ch < 18:
        full_probs = np.zeros(18)
        full_probs[:n_ch] = channel_probs
        return full_probs

    return channel_probs


def main():
    t0 = time.time()
    print("=" * 72)
    print("Channel Involvement Prediction (CNN+Attention Ensemble)")
    print("=" * 72)

    # Load models
    print("\n--- Loading CNN+Attention models ---")
    models = load_fold_models()
    if len(models) == 0:
        print("ERROR: No models found. Exiting.")
        sys.exit(1)
    print(f"  Loaded {len(models)} fold models")

    # Load dataset
    print("\n--- Loading dataset ---")
    dataset = load_dataset(verbose=True)
    df = dataset['df']
    segments = dataset['segments']

    # Load discharge times for subtype info
    print("\n--- Loading discharge times ---")
    hpp_path = LABELS_DIR / 'discharge_times_hpp.json'
    with open(str(hpp_path)) as f:
        hpp_data = json.load(f)
    print(f"  Loaded HPP data for {len(hpp_data)} patients")

    # Process each patient
    print("\n--- Predicting channel involvement ---")
    results = {}
    n_processed = 0
    n_skipped = 0

    for _, row in df.iterrows():
        pid = str(row['patient_id'])
        subtype = row['subtype']
        gold_freq = float(row['gold_standard_freq'])

        pat_segs = segments.get(pid, [])
        if not pat_segs:
            n_skipped += 1
            continue

        # Use first segment
        seg = pat_segs[0]
        if seg.shape[1] < 2000:
            n_skipped += 1
            continue

        # Predict
        channel_probs = predict_channels_for_patient(models, seg)

        # Determine involved channels
        involved = [int(i) for i in range(18) if channel_probs[i] > THRESHOLD]

        # Determine laterality
        left_probs = channel_probs[LEFT_INDICES]
        right_probs = channel_probs[RIGHT_INDICES]
        predicted_lat = predict_laterality(left_probs, right_probs)

        results[pid] = {
            'channel_probs': [round(float(p), 4) for p in channel_probs],
            'involved_channels': involved,
            'predicted_laterality': predicted_lat,
            'subtype': subtype,
            'gold_standard_freq': round(gold_freq, 4),
            'n_involved': len(involved),
            'review_status': 'auto',
        }

        n_processed += 1
        if n_processed % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {n_processed} patients ({elapsed:.0f}s)")

    # Save results
    out_path = LABELS_DIR / 'channel_involvement_predictions.json'
    with open(str(out_path), 'w') as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0

    # Summary statistics
    n_total = len(results)
    subtypes = {}
    lateralities = {}
    n_involved_list = []

    for pid, entry in results.items():
        st = entry['subtype']
        lat = entry['predicted_laterality']
        subtypes[st] = subtypes.get(st, 0) + 1
        lateralities[lat] = lateralities.get(lat, 0) + 1
        n_involved_list.append(entry['n_involved'])

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Patients processed: {n_processed}")
    print(f"  Patients skipped:   {n_skipped}")
    print(f"  Results saved to:   {out_path}")
    print(f"\n  Subtypes:")
    for st, count in sorted(subtypes.items()):
        print(f"    {st}: {count}")
    print(f"\n  Predicted laterality:")
    for lat, count in sorted(lateralities.items()):
        print(f"    {lat}: {count}")
    print(f"\n  Involved channels:")
    print(f"    Mean: {np.mean(n_involved_list):.1f}")
    print(f"    Median: {np.median(n_involved_list):.1f}")
    print(f"    Range: {np.min(n_involved_list)}-{np.max(n_involved_list)}")
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
