"""Run the RDA Contest — evaluate all methods and update leaderboard.

Usage:
    conda run -n foe python code/run_rda_contest.py
    conda run -n foe python code/run_rda_contest.py --only E1_VESearch
    conda run -n foe python code/run_rda_contest.py --leaderboard
"""
import sys
import argparse
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from rda_contest.harness import load_contest_data, run_method, evaluate, save_result, print_leaderboard, update_html_leaderboard


def get_all_methods():
    """Import and instantiate all contest methods."""
    methods = []

    # Existing methods (E1-E6)
    try:
        from rda_contest.methods_existing import (
            E1_VESearch, E2_FFTPeak, E3_NVODualBand,
            E4_SpectralConcentration, E5_FOOOF, E6_ACFPeak,
        )
        methods.extend([E1_VESearch(), E2_FFTPeak(), E3_NVODualBand(),
                        E4_SpectralConcentration(), E5_FOOOF(), E6_ACFPeak()])
    except ImportError as e:
        print(f"Warning: could not import existing methods: {e}")

    # AR model methods (M1-M3)
    try:
        from rda_contest.methods_ar import (
            M1_AR2Oscillator, M2_AR2LikelihoodRatio, M3_HilbertCV,
        )
        methods.extend([M1_AR2Oscillator(), M2_AR2LikelihoodRatio(), M3_HilbertCV()])
    except ImportError as e:
        print(f"Warning: could not import AR methods: {e}")

    # Spectral + template methods (S1-S3, T1-T2)
    try:
        from rda_contest.methods_spectral import (
            S1_WaveformCorrelation, S2_EnvelopeContinuity,
            S3_SpectralEntropy, T1_TemplateMatch, T2_PeakRegularity,
        )
        methods.extend([S1_WaveformCorrelation(), S2_EnvelopeContinuity(),
                        S3_SpectralEntropy(), T1_TemplateMatch(), T2_PeakRegularity()])
    except ImportError as e:
        print(f"Warning: could not import spectral methods: {e}")

    # CNN methods (CNN1-CNN4) using UnifiedPDModel
    try:
        from rda_contest.methods_cnn import (
            CNN1_SubtypeSoftmax, CNN2_RDAChannelMax,
            CNN3_Unified, CNN4_SubtypeXChannel,
        )
        methods.extend([CNN1_SubtypeSoftmax(), CNN2_RDAChannelMax(),
                        CNN3_Unified(), CNN4_SubtypeXChannel()])
    except ImportError as e:
        print(f"Warning: could not import CNN methods: {e}")

    return methods


def main():
    parser = argparse.ArgumentParser(description='RDA Contest Runner')
    parser.add_argument('--only', type=str, help='Run single method by name')
    parser.add_argument('--leaderboard', action='store_true', help='Print leaderboard only')
    parser.add_argument('--max-rda', type=int, default=None, help='Max RDA cases to load')
    parser.add_argument('--max-neg', type=int, default=500, help='Max negative cases')
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    print("=" * 70)
    print("  RDA Analysis Contest")
    print("  Task A: Q-score vs expert agreement (Spearman)")
    print("  Task B: RDA vs non-RDA detection (AUC)")
    print("  Task C: Frequency estimation accuracy (Spearman)")
    print("=" * 70)

    # Load data
    data = load_contest_data(max_rda=args.max_rda, max_neg=args.max_neg)

    # Get methods
    all_methods = get_all_methods()
    print(f"\nMethods available: {len(all_methods)}")
    for m in all_methods:
        print(f"  {m.name}: {m.description}")

    if args.only:
        all_methods = [m for m in all_methods if m.name == args.only]
        if not all_methods:
            print(f"Method '{args.only}' not found!")
            return

    # Run each method
    t0 = time.time()
    for method in all_methods:
        print(f"\n{'─'*60}")
        print(f"Running: {method.name}")
        print(f"{'─'*60}")

        results = run_method(method, data)
        metrics = evaluate(results, data)

        print(f"  Task A (agreement ρ): {metrics['task_a_rho']}")
        print(f"  Task B (detection AUC): {metrics['task_b_auc']}")
        print(f"  Task C (freq ρ): {metrics['task_c_rho']} (MAE: {metrics['task_c_mae']})")
        print(f"  Composite: {metrics['composite']}")

        save_result(method.name, metrics)
        update_html_leaderboard(n_total=len(all_methods))

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  All methods complete ({elapsed:.0f}s)")
    print(f"{'='*70}")

    # Print final leaderboard
    print_leaderboard()


if __name__ == '__main__':
    main()
