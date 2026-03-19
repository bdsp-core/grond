#!/usr/bin/env python3
"""Read rda_*.json from results/optimization_runs_v2/ and write rda_results_data.js."""

import json
import glob
import os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'optimization_runs_v2')
OUT_JS = os.path.join(RESULTS_DIR, 'rda_results_data.js')

def main():
    pattern = os.path.join(RESULTS_DIR, 'rda_*.json')
    files = sorted(glob.glob(pattern))

    results = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        results.append(data)

    # Sort by freq_combined_spearman descending
    results.sort(key=lambda x: x.get('freq_combined_spearman', 0), reverse=True)

    js_content = f"// Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nwindow._rda_results = {json.dumps(results, indent=2)};\nif (typeof render === 'function') render();\n"

    with open(OUT_JS, 'w') as f:
        f.write(js_content)

    print(f"Wrote {len(results)} RDA experiments to {OUT_JS}")

if __name__ == '__main__':
    main()
