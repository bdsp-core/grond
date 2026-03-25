"""
Validate laterality results against expert annotations and expected behavior.

Checks:
1. Backward compatibility: binary RDA detection unchanged vs original results
2. LRDA segments should have |laterality_index| > 0 (lateralized)
3. GRDA segments should have laterality_index near 0 (generalized/symmetric)
4. Laterality index should correlate with expert spatial_area annotations
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent.parent
repo_root = script_dir.parent if script_dir.name == 'code' else script_dir
results_dir = repo_root / 'results'
annot_dir = repo_root / 'data' / 'annotations'


def check_backward_compatibility():
    """Verify that original detection results are preserved in the new output."""
    print("=" * 70)
    print("CHECK 1: Backward Compatibility")
    print("=" * 70)

    for event in ['lrda', 'grda']:
        orig_file = results_dir / f'{event}_results.csv'
        new_file = results_dir / f'{event}_laterality_results.csv'

        if not orig_file.exists():
            print(f"  {event.upper()}: Original results not found ({orig_file}). Skipping.")
            continue
        if not new_file.exists():
            print(f"  {event.upper()}: Laterality results not found ({new_file}). "
                  "Run extract_with_laterality.py first.")
            continue

        df_orig = pd.read_csv(orig_file)
        df_new = pd.read_csv(new_file)

        # Merge on filename
        merged = df_orig.merge(df_new, on='files', suffixes=('_orig', '_new'))

        # Compare event frequency (rda1b_fft column in original)
        freq_orig = merged['freq_rda1b_fft']
        freq_new = merged['event_frequency']

        both_nan = freq_orig.isna() & freq_new.isna()
        both_valid = freq_orig.notna() & freq_new.notna()
        freq_match = both_nan | (both_valid & np.isclose(freq_orig, freq_new, equal_nan=True))
        n_match = freq_match.sum()
        n_total = len(merged)

        # Compare spatial extent
        spat_orig = merged['spatial_rda1b_fft']
        spat_new = merged['spatial_extent']
        both_nan_s = spat_orig.isna() & spat_new.isna()
        both_valid_s = spat_orig.notna() & spat_new.notna()
        spat_match = both_nan_s | (both_valid_s & np.isclose(spat_orig, spat_new, equal_nan=True))
        s_match = spat_match.sum()

        print(f"\n  {event.upper()} ({n_total} segments):")
        print(f"    Frequency match: {n_match}/{n_total}")
        print(f"    Spatial extent match: {s_match}/{n_total}")

        if n_match == n_total and s_match == n_total:
            print(f"    PASS: All values match original results.")
        else:
            print(f"    WARN: Some values differ. This may indicate a change in behavior.")


def check_laterality_distributions():
    """Check that LRDA has higher |laterality| than GRDA."""
    print("\n" + "=" * 70)
    print("CHECK 2: Laterality Index Distributions")
    print("=" * 70)

    results = {}
    for event in ['lrda', 'grda']:
        f = results_dir / f'{event}_laterality_results.csv'
        if not f.exists():
            print(f"  {event.upper()}: Results not found. Run extract_with_laterality.py first.")
            continue
        df = pd.read_csv(f)
        lat = df['laterality_index'].dropna()
        results[event] = lat

        abs_lat = lat.abs()
        print(f"\n  {event.upper()} ({len(lat)} segments with laterality values):")
        print(f"    Laterality index:  mean={lat.mean():.4f}  std={lat.std():.4f}")
        print(f"    |Laterality index|: mean={abs_lat.mean():.4f}  median={abs_lat.median():.4f}")
        print(f"    Range: [{lat.min():.4f}, {lat.max():.4f}]")

    if 'lrda' in results and 'grda' in results:
        lrda_abs = results['lrda'].abs().mean()
        grda_abs = results['grda'].abs().mean()
        print(f"\n  Comparison:")
        print(f"    LRDA mean |laterality|: {lrda_abs:.4f}")
        print(f"    GRDA mean |laterality|: {grda_abs:.4f}")
        if lrda_abs > grda_abs:
            print(f"    PASS: LRDA is more lateralized than GRDA (as expected).")
        else:
            print(f"    WARN: GRDA is more lateralized than LRDA (unexpected).")


def check_laterality_vs_annotations():
    """Check whether laterality correlates with expert-annotated spatial areas."""
    print("\n" + "=" * 70)
    print("CHECK 3: Laterality vs Expert Annotations")
    print("=" * 70)

    left_regions = {'LF', 'LT', 'LCP', 'LO'}
    right_regions = {'RF', 'RT', 'RCP', 'RO'}

    for event in ['lrda', 'grda']:
        f = results_dir / f'{event}_laterality_results.csv'
        if not f.exists():
            continue

        df = pd.read_csv(f)
        df = df.dropna(subset=['laterality_index', 'spatial_areas'])

        # Parse spatial_areas from string representation
        def parse_areas(s):
            if isinstance(s, str) and s.startswith('['):
                return [x.strip().strip("'\"") for x in s.strip('[]').split(',') if x.strip()]
            return []

        df['areas_parsed'] = df['spatial_areas'].apply(parse_areas)

        # Classify each segment as left-only, right-only, bilateral, or no-detection
        def classify_laterality(areas):
            area_set = set(areas)
            has_left = bool(area_set & left_regions)
            has_right = bool(area_set & right_regions)
            if has_left and has_right:
                return 'bilateral'
            elif has_left:
                return 'left_only'
            elif has_right:
                return 'right_only'
            else:
                return 'midline_or_none'

        df['expert_side'] = df['areas_parsed'].apply(classify_laterality)

        print(f"\n  {event.upper()}:")
        for side in ['left_only', 'right_only', 'bilateral', 'midline_or_none']:
            subset = df[df['expert_side'] == side]
            if len(subset) == 0:
                continue
            lat = subset['laterality_index']
            print(f"    {side:>16s} (n={len(subset):3d}): "
                  f"mean laterality={lat.mean():+.4f}  std={lat.std():.4f}")

        # Expected: left_only should have negative laterality, right_only positive
        left_only = df[df['expert_side'] == 'left_only']['laterality_index']
        right_only = df[df['expert_side'] == 'right_only']['laterality_index']

        if len(left_only) > 0 and len(right_only) > 0:
            if left_only.mean() < right_only.mean():
                print(f"    PASS: Left-only regions have lower laterality than right-only.")
            else:
                print(f"    WARN: Expected left < right laterality, got "
                      f"left={left_only.mean():.4f} vs right={right_only.mean():.4f}.")


def main():
    print("RDA Laterality Validation")
    print("=" * 70)

    check_backward_compatibility()
    check_laterality_distributions()
    check_laterality_vs_annotations()

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
