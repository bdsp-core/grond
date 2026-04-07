"""Run the PD Spatial Localization Contest.

Usage:
    conda run -n foe python code/run_spatial_contest.py
    conda run -n foe python code/run_spatial_contest.py --only S1_PointinessMax
    conda run -n foe python code/run_spatial_contest.py --leaderboard
"""
import sys
import argparse
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from spatial_contest.harness import (
    load_spatial_data, run_method, evaluate, save_result,
    print_leaderboard, update_html_leaderboard,
)


def get_all_methods():
    """Import and instantiate all contest methods."""
    methods = []

    # Baselines (B1-B4)
    try:
        from spatial_contest.methods_baseline import (
            B1_AllRegions, B2_SubtypeDefault, B3_RMSThreshold, B4_RandomBaseline,
        )
        methods.extend([B1_AllRegions(), B2_SubtypeDefault(),
                        B3_RMSThreshold(), B4_RandomBaseline()])
    except ImportError as e:
        print(f"Warning: baselines: {e}")

    # Signal methods (S1-S8)
    try:
        from spatial_contest.methods_signal import (
            S1_PointinessMax, S2_PointinessMean, S3_FFTPeakPower,
            S4_BandpowerRatio, S5_ACFPeakHeight, S6_LineLength,
            S7_EnvelopePeakiness, S8_SpectralEntropy,
        )
        methods.extend([S1_PointinessMax(), S2_PointinessMean(), S3_FFTPeakPower(),
                        S4_BandpowerRatio(), S5_ACFPeakHeight(), S6_LineLength(),
                        S7_EnvelopePeakiness(), S8_SpectralEntropy()])
    except ImportError as e:
        print(f"Warning: signal methods: {e}")

    # Cross-channel methods (X1-X6)
    try:
        from spatial_contest.methods_crosschannel import (
            X1_CoherenceNetwork, X2_CrossCorrPeak, X3_PeakSynchrony,
            X4_TemplateCorrelation, X5_PhaseCoherence, X6_MutualInfoPDBand,
        )
        methods.extend([X1_CoherenceNetwork(), X2_CrossCorrPeak(), X3_PeakSynchrony(),
                        X4_TemplateCorrelation(), X5_PhaseCoherence(), X6_MutualInfoPDBand()])
    except ImportError as e:
        print(f"Warning: cross-channel methods: {e}")

    # Hybrid methods (H1-H8)
    try:
        from spatial_contest.methods_hybrid import (
            H1_EnsembleVote, H2_WeightedEnsemble, H3_AdaptiveThreshold,
            H4_PowerRankPercentile, H5_SymmetryAware, H6_TKEO,
            H7_VEPerChannel, H8_GradientSharpness,
        )
        methods.extend([H1_EnsembleVote(), H2_WeightedEnsemble(), H3_AdaptiveThreshold(),
                        H4_PowerRankPercentile(), H5_SymmetryAware(), H6_TKEO(),
                        H7_VEPerChannel(), H8_GradientSharpness()])
    except ImportError as e:
        print(f"Warning: hybrid methods: {e}")

    # CNN model methods (M1-M4)
    try:
        from spatial_contest.methods_model import (
            M1_CNNChannelProbs, M2_CNNAttentionWeighted,
            M3_CNNPlusPointiness, M4_CNNSymmetryGPD,
        )
        methods.extend([M1_CNNChannelProbs(), M2_CNNAttentionWeighted(),
                        M3_CNNPlusPointiness(), M4_CNNSymmetryGPD()])
    except ImportError as e:
        print(f"Warning: model methods: {e}")

    return methods


def main():
    parser = argparse.ArgumentParser(description='PD Spatial Localization Contest')
    parser.add_argument('--only', type=str, help='Run single method by name')
    parser.add_argument('--leaderboard', action='store_true', help='Print leaderboard only')
    args = parser.parse_args()

    if args.leaderboard:
        print_leaderboard()
        return

    n_methods_total = 26  # B4 + S8 + X6 + H8 + M4

    print("=" * 70)
    print("  PD Spatial Localization Contest")
    print("  Task: Predict which brain regions (LF RF LT RT LCP RCP LO RO)")
    print("        are involved in periodic discharges")
    print(f"  Methods: {n_methods_total}")
    print("=" * 70)

    # Load data
    data = load_spatial_data()

    # Get methods
    all_methods = get_all_methods()
    print(f"\nMethods loaded: {len(all_methods)}")
    for m in all_methods:
        print(f"  {m.name}: {m.description}")

    if args.only:
        all_methods = [m for m in all_methods if m.name == args.only]
        if not all_methods:
            print(f"Method '{args.only}' not found!")
            return

    # Initial leaderboard
    update_html_leaderboard(n_total=n_methods_total)

    # Run each method
    t0 = time.time()
    for i, method in enumerate(all_methods):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(all_methods)}] Running: {method.name}")
        print(f"{'─'*60}")

        results = run_method(method, data)
        metrics = evaluate(results, data)

        print(f"  Macro F1:     {metrics['macro_f1']}")
        print(f"  Micro F1:     {metrics['micro_f1']}")
        print(f"  Jaccard:      {metrics['jaccard']}")
        print(f"  Mean AUC:     {metrics['mean_auc']}")
        print(f"  Extent rho:   {metrics['extent_rho']}")
        print(f"  Exact match:  {metrics['exact_match']}")
        print(f"  Composite:    {metrics['composite']}")

        save_result(method.name, metrics)
        update_html_leaderboard(n_total=n_methods_total)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  All {len(all_methods)} methods complete ({elapsed:.0f}s)")
    print(f"{'='*70}")

    print_leaderboard()

    print(f"\nLeaderboard: results/spatial_contest_leaderboard.html")
    print(f"  open results/spatial_contest_leaderboard.html")


if __name__ == '__main__':
    main()
