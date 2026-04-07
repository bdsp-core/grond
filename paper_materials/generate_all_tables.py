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

    # Auto-generate tables that have scripts
    for filename, script in AUTO_GENERATED.items():
        script_path = SCRIPT_DIR / script
        if script_path.exists():
            print(f"\n  Generating {filename}...")
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                print(f"  OK  (auto-generated from label files)")
            else:
                print(f"  FAILED: {result.stderr[-200:]}")

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
    if all_ok:
        print(f"All {len(TABLES)} tables present in {TABLES_DIR}/")
    else:
        print("Some tables are missing!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
