"""
Test the laterality computation logic with synthetic data.
Does NOT require fooof/mne — tests the math only.
"""

import numpy as np
import sys

# Bipolar channel list (same as rda1b_fft.py)
bipolar_channels = [
    'Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    'Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    'Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'Fz-Cz', 'Cz-Pz',
]

left_indices = [0, 1, 2, 3, 8, 9, 10, 11]
right_indices = [4, 5, 6, 7, 12, 13, 14, 15]


def compute_laterality(rda_scores):
    """Replicate the laterality computation from rda1b_fft."""
    scores_for_lat = np.where(np.isnan(rda_scores), 1.0, rda_scores)
    left_mean = np.mean(scores_for_lat[left_indices])
    right_mean = np.mean(scores_for_lat[right_indices])
    denom = right_mean + left_mean
    if denom > 0:
        return (right_mean - left_mean) / denom
    else:
        return 0.0


def test_symmetric_scores():
    """All channels equal → laterality = 0."""
    scores = np.full(18, 2.0)
    lat = compute_laterality(scores)
    assert abs(lat) < 1e-10, f"Expected 0, got {lat}"
    print("  PASS: Symmetric scores → laterality = 0")


def test_all_nan():
    """All NaN (no peaks) → all replaced with 1.0 → laterality = 0."""
    scores = np.full(18, np.nan)
    lat = compute_laterality(scores)
    assert abs(lat) < 1e-10, f"Expected 0, got {lat}"
    print("  PASS: All NaN → laterality = 0")


def test_left_lateralized():
    """Strong left scores, weak right → negative laterality."""
    scores = np.full(18, 1.0)
    for i in left_indices:
        scores[i] = 5.0  # Strong RDA on left
    lat = compute_laterality(scores)
    assert lat < 0, f"Expected negative, got {lat}"
    print(f"  PASS: Left-lateralized → laterality = {lat:.4f} (negative)")


def test_right_lateralized():
    """Strong right scores, weak left → positive laterality."""
    scores = np.full(18, 1.0)
    for i in right_indices:
        scores[i] = 5.0  # Strong RDA on right
    lat = compute_laterality(scores)
    assert lat > 0, f"Expected positive, got {lat}"
    print(f"  PASS: Right-lateralized → laterality = {lat:.4f} (positive)")


def test_fully_left():
    """Only left has RDA, right is NaN → should be negative."""
    scores = np.full(18, np.nan)
    for i in left_indices:
        scores[i] = 3.0
    lat = compute_laterality(scores)
    # Left=3.0, Right=1.0 (NaN→1.0), so (1-3)/(1+3) = -0.5
    assert lat < 0, f"Expected negative, got {lat}"
    assert abs(lat - (-0.5)) < 1e-10, f"Expected -0.5, got {lat}"
    print(f"  PASS: Only-left RDA → laterality = {lat:.4f}")


def test_fully_right():
    """Only right has RDA, left is NaN → should be positive."""
    scores = np.full(18, np.nan)
    for i in right_indices:
        scores[i] = 3.0
    lat = compute_laterality(scores)
    assert lat > 0, f"Expected positive, got {lat}"
    assert abs(lat - 0.5) < 1e-10, f"Expected 0.5, got {lat}"
    print(f"  PASS: Only-right RDA → laterality = {lat:.4f}")


def test_range():
    """Laterality should always be in [-1, +1]."""
    rng = np.random.RandomState(42)
    for _ in range(1000):
        scores = rng.exponential(2.0, 18)
        # Randomly set some to NaN
        nan_mask = rng.random(18) < 0.3
        scores[nan_mask] = np.nan
        lat = compute_laterality(scores)
        assert -1.0 <= lat <= 1.0, f"Out of range: {lat}"
    print("  PASS: 1000 random trials all in [-1, +1]")


def test_midline_excluded():
    """Midline channels (16, 17) should not affect laterality."""
    scores_a = np.full(18, 2.0)
    scores_b = np.full(18, 2.0)
    scores_b[16] = 100.0  # Fz-Cz
    scores_b[17] = 100.0  # Cz-Pz
    lat_a = compute_laterality(scores_a)
    lat_b = compute_laterality(scores_b)
    assert abs(lat_a - lat_b) < 1e-10, f"Midline affected result: {lat_a} vs {lat_b}"
    print("  PASS: Midline channels do not affect laterality")


def main():
    print("Testing laterality computation logic")
    print("=" * 50)
    tests = [
        test_symmetric_scores,
        test_all_nan,
        test_left_lateralized,
        test_right_lateralized,
        test_fully_left,
        test_fully_right,
        test_range,
        test_midline_excluded,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"{passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
