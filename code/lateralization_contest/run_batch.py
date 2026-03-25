#!/usr/bin/env python3
"""Run a batch of lateralization methods.

Usage:
    python run_batch.py methods_power        # run all methods in methods_power.py
    python run_batch.py methods_power L01_DeltaBandpower L02_NarrowbandPeak  # specific methods
"""
import sys
import importlib
from pathlib import Path

# Ensure code/ is on the path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from lateralization_contest.harness import (
    load_contest_data, run_method, evaluate, save_result, update_html_leaderboard
)


def get_methods_from_module(module_name):
    """Import a module and return all LateralMethod subclasses."""
    from lateralization_contest.base import LateralMethod
    mod = importlib.import_module(f'lateralization_contest.{module_name}')
    methods = []
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type) and issubclass(attr, LateralMethod)
                and attr is not LateralMethod):
            methods.append(attr())
    return methods


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_batch.py <module_name> [method_names...]")
        sys.exit(1)

    module_name = sys.argv[1]
    filter_names = set(sys.argv[2:]) if len(sys.argv) > 2 else None

    print(f"Loading data...")
    data = load_contest_data(verbose=True)

    methods = get_methods_from_module(module_name)
    if filter_names:
        methods = [m for m in methods if m.name in filter_names]

    print(f"\nRunning {len(methods)} methods from {module_name}:")
    for m in methods:
        print(f"  - {m.name}: {m.description}")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Running: {method.name}")
        print(f"{'='*60}")
        results = run_method(method, data, verbose=True)
        metrics = evaluate(results, data)
        save_result(method.name, metrics)
        print(f"  Composite: {metrics['composite']:.4f}")
        print(f"  Task A (LRDA/GRDA AUC): {metrics['task_a_lrda_vs_grda_auc']}")
        print(f"  Task B (Side AUC): {metrics['task_b_side_auc']}")
        print(f"  Task B (Side Acc): {metrics['task_b_side_acc']}")
        print(f"  Task C (Lat ρ): {metrics['task_c_lat_rho']}")
        print(f"  Task D (Correct hemi): {metrics['task_d_correct_hemi']}")
        update_html_leaderboard()

    print("\nDone!")


if __name__ == '__main__':
    main()
