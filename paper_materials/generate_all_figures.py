#!/usr/bin/env python3
"""
Generate all publication figures in order.

Usage:
    conda run -n morgoth python paper_materials/generate_all_figures.py
    conda run -n morgoth python paper_materials/generate_all_figures.py --figure 2  # single figure
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FIGURES = [
    {
        'num': 1,
        'name': 'Fig 1: EEG Examples',
        'script': 'generate_fig0_examples.py',
        'output': 'manuscript/figures/fig1_eeg_examples.png',
    },
    {
        'num': 2,
        'name': 'Fig 2: PD Pipeline Architecture',
        'script': 'build_fig2.py',
        'output': 'manuscript/figures/fig2_pd_pipeline.png',
    },
    {
        'num': 3,
        'name': 'Fig 3: RDA Pipeline Architecture',
        'script': 'build_fig3.py',
        'output': 'manuscript/figures/fig3_rda_pipeline.png',
    },
    {
        'num': 4,
        'name': 'Fig 4: Frequency Scatter Plots',
        'script': 'generate_fig6.py',
        'output': 'manuscript/figures/fig4_frequency_scatter.png',
    },
    {
        'num': 5,
        'name': 'Fig 5: Independent-expert IRR comparison (paired bars)',
        'script': 'build_fig_irr_bars.py',
        'output': 'manuscript/figures/fig5_irr_comparison.png',
    },
    {
        'num': '6-9',
        'name': 'Figs 6-9: Characterization Examples (LPD, GPD, LRDA, GRDA)',
        'script': 'render_figures.py',
        'output': 'manuscript/figures/fig6_lpd_characterization.png',
    },
    {
        'num': 'S1',
        'name': 'Fig S1: Inter-Rater Reliability',
        'script': 'generate_fig_irr.py',
        'output': 'manuscript/figures/figS1_irr_comparison.png',
    },
    {
        'num': 'S2',
        'name': 'Fig S2: Spatial Scatter Plots',
        'script': 'generate_fig_spatial_scatter.py',
        'output': 'manuscript/figures/figS2_spatial_scatter.png',
    },
    {
        'num': 'S3',
        'name': 'Fig S3: Threshold Sweep',
        'script': 'generate_threshold_sweep.py',
        'output': 'manuscript/figures/figS3_threshold_sweep.png',
    },
    {
        'num': 'S4',
        'name': 'Fig S4: Independent-Expert Task Frequency Distribution',
        'script': 'generate_fig_independent_expert_freq.py',
        'output': 'manuscript/figures/figS4_independent_expert_freq_distribution.png',
    },
]


def run_figure(fig, dry_run=False, from_scratch=False):
    script_path = SCRIPT_DIR / fig['script']
    if not script_path.exists():
        print(f"  SKIP: {fig['script']} not found")
        return False

    print(f"\n{'='*60}")
    print(f"  {fig['name']}")
    print(f"  Script: {fig['script']}")
    print(f"{'='*60}")

    if dry_run:
        print(f"  [DRY RUN] Would run: python {script_path}")
        return True

    output_path = SCRIPT_DIR / fig['output']
    # Capture the pre-run mtime so we can detect whether the script actually
    # rewrote the expected output. Without this check the wrapper has no way
    # to notice when a generator silently writes to the wrong filename.
    prev_mtime = output_path.stat().st_mtime if output_path.exists() else None

    t0 = time.time()
    env = dict(os.environ)
    if not from_scratch:
        env['USE_SPATIAL_CACHE'] = '1'
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=600,
        env=env,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED ({elapsed:.1f}s)")
        # Show last 5 lines of stderr
        for line in result.stderr.strip().split('\n')[-5:]:
            print(f"    {line}")
        return False

    if not output_path.exists():
        print(f"  FAILED ({elapsed:.1f}s): script ran cleanly but expected output "
              f"file does not exist: {fig['output']}")
        return False

    new_mtime = output_path.stat().st_mtime
    size_mb = output_path.stat().st_size / 1e6
    if prev_mtime is not None and new_mtime <= prev_mtime:
        # The expected file exists but was not touched by this run. Almost
        # certainly means the generator script writes to a different filename
        # than fig['output'] declares -- a stale-output bug.
        print(f"  FAILED ({elapsed:.1f}s): expected output {fig['output']} was not "
              f"updated by the script (writes to a different filename?)")
        return False

    print(f"  OK ({elapsed:.1f}s) -> {fig['output']} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate all publication figures')
    parser.add_argument('--figure', type=str, help='Generate only this figure (e.g., 2, S1)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run')
    parser.add_argument('--from-scratch', action='store_true',
                        help='Re-run all inference from raw EEG (slow, ~10 min). '
                             'Default: use cached intermediate results (fast, ~30 sec).')
    args = parser.parse_args()

    # Check if spatial cache exists for fast mode
    cache_path = SCRIPT_DIR / 'spatial_inference_cache.json'
    if not args.from_scratch and not cache_path.exists():
        print("No spatial_inference_cache.json found.")
        print("Run: python paper_materials/precompute_spatial_cache.py")
        print("Or use --from-scratch to re-run all inference.")
        print()

    print("=" * 60)
    mode = "FROM SCRATCH (re-running inference)" if args.from_scratch else "FROM CACHE (fast)"
    print(f"Generating Publication Figures — {mode}")
    print("=" * 60)

    figures = FIGURES
    if args.figure:
        figures = [f for f in FIGURES if str(f['num']) == args.figure]
        if not figures:
            print(f"Unknown figure: {args.figure}")
            print(f"Available: {', '.join(str(f['num']) for f in FIGURES)}")
            return

    ok, fail = 0, 0
    for fig in figures:
        if run_figure(fig, dry_run=args.dry_run, from_scratch=args.from_scratch):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"Done: {ok} succeeded, {fail} failed")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
