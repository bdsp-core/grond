"""Update the v2 dashboard data files from all experiment JSON results in optimization_runs_v2/."""
import json
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'optimization_runs_v2'


def update():
    results = []
    json_files = sorted(RUNS_DIR.glob('*.json'))
    # Exclude any index/meta files
    json_files = [f for f in json_files if f.name not in ('index.json', 'results_data.json')]

    for f in json_files:
        try:
            with open(str(f)) as fh:
                data = json.load(fh)
                results.append(data)
        except Exception as e:
            print(f'Warning: failed to read {f.name}: {e}')

    # Write results_data.js (loadable via script tag for file:// protocol)
    out_path = RUNS_DIR / 'results_data.js'
    with open(str(out_path), 'w') as f:
        f.write('window._v2_results = ')
        json.dump(results, f)
        f.write(';\n')

    print(f'Dashboard v2 updated: {len(results)} experiments -> {out_path}')


if __name__ == '__main__':
    update()
