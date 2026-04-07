#!/usr/bin/env python3
"""Run a batch of lateralization v2 methods.

Usage:
    python run_batch_v2.py methods_v2_tier1
    python run_batch_v2.py methods_v2_tier1 L01_NarrowbandVE L02_MultiChannelVE
"""
import sys
import importlib
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

from lateralization_contest.harness_v2 import (
    load_contest_data, run_method, evaluate,
    save_result, save_per_patient, update_html_leaderboard
)


def get_methods_from_module(module_name):
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
        print("Usage: python run_batch_v2.py <module_name> [method_names...]")
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
        print(f"\n{'=' * 60}")
        print(f"Running: {method.name}")
        print(f"{'=' * 60}")
        results = run_method(method, data, verbose=True)
        metrics = evaluate(results, data)
        save_result(method.name, metrics)
        save_per_patient(method.name, results)
        print(f"  ** AUC: {metrics['primary_auc']} **  Cohen's d: {metrics['cohens_d']}")
        print(f"  Sens: {metrics['sensitivity']}  Spec: {metrics['specificity']}")
        print(f"  Side accuracy: {metrics['side_accuracy']} ({metrics['n_side_validation']} cases)")
        print(f"  Mean asym LRDA: {metrics['mean_asym_lrda']}  GRDA: {metrics['mean_asym_grda']}")
        update_html_leaderboard()

    print("\nDone!")


if __name__ == '__main__':
    main()
