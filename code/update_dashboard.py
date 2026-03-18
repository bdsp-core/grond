"""Update the dashboard data files from all experiment JSON results."""
import json
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'optimization_runs'

def update():
    results = []
    json_files = sorted(RUNS_DIR.glob('*.json'))
    json_files = [f for f in json_files if f.name != 'index.json']

    for f in json_files:
        try:
            with open(str(f)) as fh:
                results.append(json.load(fh))
        except:
            pass

    # Write index.json
    with open(str(RUNS_DIR / 'index.json'), 'w') as f:
        json.dump([p.name for p in json_files], f)

    # Write results_data.js (for file:// protocol)
    with open(str(RUNS_DIR / 'results_data.js'), 'w') as f:
        f.write('window._optimization_results = ')
        json.dump(results, f)
        f.write('; if (typeof renderResults === "function") renderResults();')

    print(f'Dashboard updated: {len(results)} experiments')

if __name__ == '__main__':
    update()
