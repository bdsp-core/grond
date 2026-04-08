#!/usr/bin/env python3
"""
Generate all publication tables.

Tables are stored as markdown files in paper_materials/tables/.
This script verifies they exist and prints a summary.

For tables that can be auto-generated from data, this script
will regenerate them. For manually curated tables, it verifies
the files exist and reports their contents.

Usage:
    conda run -n morgoth python paper_materials/generate_all_tables.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TABLES_DIR = SCRIPT_DIR / 'tables'

# Tables with auto-generation scripts
AUTO_GENERATED = {
    'table1_dataset.md': 'tables/generate_table1.py',
    'table3_lateralization.md': 'tables/generate_table3.py',
    'table4_spatial.md': 'tables/generate_table4.py',
    'table5_frequency.md': 'tables/generate_table5.py',
    'table6_timing.md': 'tables/generate_table6.py',
    'table7_model_variants.md': 'tables/generate_table7.py',
}

TABLES = [
    ('table1_dataset.md', 'Table 1: Dataset Statistics'),
    ('table2_architecture.md', 'Table 2: Pipeline Architecture Components'),
    ('table3_lateralization.md', 'Table 3: Lateralization Performance'),
    ('table4_spatial.md', 'Table 4: Spatial Inter-Rater Agreement'),
    ('table5_frequency.md', 'Table 5: Frequency Estimation Performance'),
    ('table6_timing.md', 'Table 6: Discharge Timing Performance'),
    ('table7_model_variants.md', 'Table 7: Model Architecture Comparison'),
]


def main():
    print("=" * 60)
    print("Publication Tables")
    print("=" * 60)

    gen_failures = []

    # Auto-generate tables that have scripts
    for filename, script in AUTO_GENERATED.items():
        script_path = SCRIPT_DIR / script
        if not script_path.exists():
            continue

        target = TABLES_DIR / filename
        # Capture the pre-run mtime so we can detect whether the generator
        # actually rewrote the expected output. Without this check the wrapper
        # can silently miss generators that write to the wrong filename.
        prev_mtime = target.stat().st_mtime if target.exists() else None

        print(f"\n  Generating {filename}...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=60,
        )

        if result.returncode != 0:
            print(f"  FAILED: {result.stderr[-200:]}")
            gen_failures.append(filename)
            continue

        if not target.exists():
            print(f"  FAILED: script ran cleanly but expected output {filename} "
                  f"does not exist (writes to a different path?)")
            gen_failures.append(filename)
            continue

        new_mtime = target.stat().st_mtime
        if prev_mtime is not None and new_mtime <= prev_mtime:
            print(f"  FAILED: expected output {filename} was not updated by "
                  f"the script (writes to a different filename?)")
            gen_failures.append(filename)
            continue

        print(f"  OK  (auto-generated from label files)")

    # Check all tables
    print()
    all_ok = True
    for filename, title in TABLES:
        path = TABLES_DIR / filename
        if path.exists():
            lines = path.read_text().strip().split('\n')
            auto = " [auto-generated]" if filename in AUTO_GENERATED else ""
            print(f"  OK  {title}{auto}")
            print(f"      -> {filename} ({len(lines)} lines)")
        else:
            print(f"  MISSING  {title}")
            print(f"           -> {filename}")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok and not gen_failures:
        print(f"All {len(TABLES)} tables present in {TABLES_DIR}/")
    else:
        if gen_failures:
            print(f"Generator failures: {', '.join(gen_failures)}")
        if not all_ok:
            print("Some tables are missing!")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
