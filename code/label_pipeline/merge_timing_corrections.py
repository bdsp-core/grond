"""
Merge timing corrections from a browser-exported JSON into discharge_times_hpp.json.

Usage:
    conda run -n foe python code/label_pipeline/merge_timing_corrections.py <corrections.json>

The corrections JSON format (exported from the timing correction viewer):
{
    "patient_id": {
        "patient_id": "...",
        "original_times": [...],
        "corrected_times": [...],
        "status": "corrected" | "in_progress" | "accepted"
    },
    ...
}

For each patient in the corrections file:
  - Updates global_times with corrected_times
  - Sets review_status = 'ground_truth'
  - Sets review_source based on status ('round3_corrected' or 'round3_accepted')
  - Recomputes frequency from median IPI, ipi_cv, n_discharges
"""

import json
import sys
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
HPP_PATH = PROJECT_DIR / 'data' / 'labels' / 'discharge_times_hpp.json'


def recompute_stats(times):
    """Recompute frequency, ipi_cv, n_discharges from sorted discharge times."""
    times = sorted(times)
    n = len(times)
    if n < 2:
        return {
            'frequency': 0.0,
            'ipi_cv': 0.0,
            'n_discharges': n,
        }
    ipis = [times[i + 1] - times[i] for i in range(n - 1)]
    median_ipi = float(np.median(ipis))
    freq = 1.0 / median_ipi if median_ipi > 0 else 0.0
    mean_ipi = float(np.mean(ipis))
    std_ipi = float(np.std(ipis))
    cv = std_ipi / mean_ipi if mean_ipi > 0 else 0.0
    return {
        'frequency': freq,
        'ipi_cv': cv,
        'n_discharges': n,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_timing_corrections.py <corrections.json> [--round ROUND_NUM]")
        sys.exit(1)

    corrections_path = Path(sys.argv[1])

    # Parse optional round number
    round_num = 3
    if '--round' in sys.argv:
        idx = sys.argv.index('--round')
        if idx + 1 < len(sys.argv):
            round_num = int(sys.argv[idx + 1])

    if not corrections_path.exists():
        print(f"ERROR: {corrections_path} not found")
        sys.exit(1)

    # Load corrections
    with open(corrections_path) as f:
        corrections = json.load(f)
    print(f"Loaded {len(corrections)} corrections from {corrections_path}")

    # Load HPP
    with open(HPP_PATH) as f:
        hpp = json.load(f)
    print(f"Loaded HPP file with {len(hpp)} patients")

    # Count existing ground truth
    existing_gt = sum(1 for v in hpp.values() if v.get('review_status') == 'ground_truth')
    print(f"Existing ground truth: {existing_gt}")

    # Process corrections
    n_updated = 0
    n_new_gt = 0
    n_already_gt = 0
    n_not_found = 0
    status_counts = {}

    for pid, corr in corrections.items():
        status = corr.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

        corrected_times = corr.get('corrected_times', corr.get('times', []))
        if not corrected_times:
            print(f"  WARNING: {pid} has no corrected_times, skipping")
            continue

        if pid not in hpp:
            print(f"  WARNING: {pid} not in HPP file, skipping")
            n_not_found += 1
            continue

        was_gt = hpp[pid].get('review_status') == 'ground_truth'
        if was_gt:
            n_already_gt += 1

        # Update times
        corrected_times = sorted([round(t, 4) for t in corrected_times])
        hpp[pid]['global_times'] = corrected_times

        # Recompute stats
        stats = recompute_stats(corrected_times)
        hpp[pid]['frequency'] = stats['frequency']
        hpp[pid]['ipi_cv'] = stats['ipi_cv']
        hpp[pid]['n_discharges'] = stats['n_discharges']

        # Set review status
        hpp[pid]['review_status'] = 'ground_truth'

        # Set review source based on correction status
        if status in ('corrected', 'in_progress'):
            hpp[pid]['review_source'] = f'round{round_num}_corrected'
        elif status == 'accepted':
            hpp[pid]['review_source'] = f'round{round_num}_accepted'
        else:
            hpp[pid]['review_source'] = f'round{round_num}_corrected'

        n_updated += 1
        if not was_gt:
            n_new_gt += 1

    # Save
    with open(HPP_PATH, 'w') as f:
        json.dump(hpp, f, indent=2)

    # Summary
    total_gt = sum(1 for v in hpp.values() if v.get('review_status') == 'ground_truth')
    total_auto = sum(1 for v in hpp.values() if v.get('review_status') != 'ground_truth')

    print(f"\n{'=' * 60}")
    print("MERGE SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Correction statuses: {status_counts}")
    print(f"  Patients updated:    {n_updated}")
    print(f"  New ground truth:    {n_new_gt}")
    print(f"  Re-corrected (was already GT): {n_already_gt}")
    print(f"  Not found in HPP:    {n_not_found}")
    print(f"  ---")
    print(f"  Total ground truth:  {total_gt} (was {existing_gt})")
    print(f"  Total auto remaining:{total_auto}")
    print(f"  Total patients:      {len(hpp)}")
    print(f"{'=' * 60}")

    # Also show review_source breakdown
    sources = {}
    for v in hpp.values():
        src = v.get('review_source', 'none')
        sources[src] = sources.get(src, 0) + 1
    print(f"\nReview source breakdown:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")


if __name__ == '__main__':
    main()
