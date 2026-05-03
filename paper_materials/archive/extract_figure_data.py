#!/usr/bin/env python3
"""Extract embedded case data from HTML figure files into JSON."""

import json
import re
import os

SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_cases(html_path):
    with open(html_path, 'r') as f:
        text = f.read()
    m = re.search(r'const CASES = (\[.*?\]);\s*\n', text, re.DOTALL)
    if not m:
        raise ValueError(f"No CASES found in {html_path}")
    return json.loads(m.group(1))


def generate_verbal_description(c):
    """Replicate the JS generateVerbalDescription function."""
    REGION_BARE = {
        'LF': 'frontal', 'RF': 'frontal',
        'LT': 'temporal', 'RT': 'temporal',
        'LCP': 'centro-parietal', 'RCP': 'centro-parietal',
        'LO': 'occipital', 'RO': 'occipital',
        'MID': 'midline'
    }
    LEFT_REGS = ['LF', 'LT', 'LCP', 'LO']
    RIGHT_REGS = ['RF', 'RT', 'RCP', 'RO']

    st = c['subtype'].upper()
    regs = c.get('gt_regions', [])
    if len(regs) == 0:
        return f'{st} -- no spatial regions labeled.'

    left_sel = [r for r in regs if r in LEFT_REGS]
    right_sel = [r for r in regs if r in RIGHT_REGS]
    mid_sel = 'MID' in regs
    is_gen = st in ('GPD', 'GRDA')

    lat_str = ''
    if len(left_sel) > 0 and len(right_sel) == 0:
        lat_str = 'unilateral left'
    elif len(right_sel) > 0 and len(left_sel) == 0:
        lat_str = 'unilateral right'
    elif len(left_sel) > len(right_sel):
        lat_str = 'bilateral, left-predominant'
    elif len(right_sel) > len(left_sel):
        lat_str = 'bilateral, right-predominant'
    elif len(left_sel) > 0 and len(right_sel) > 0:
        lat_str = 'bilateral/symmetric'
    elif mid_sel:
        lat_str = 'midline'

    if is_gen:
        rs = c.get('region_scores', {})
        frontal = (rs.get('LF', 0) + rs.get('RF', 0)) / 2
        occipital = (rs.get('LO', 0) + rs.get('RO', 0)) / 2
        temporal = (rs.get('LT', 0) + rs.get('RT', 0)) / 2
        scores = {'frontally': frontal, 'occipitally': occipital, 'temporally': temporal}
        best = max(scores, key=scores.get)
        rng = max(frontal, occipital, temporal) - min(frontal, occipital, temporal)
        predom = f'{best} predominant' if rng > 0.1 else 'no regional predominance'
        return f'{st}, {predom}.'

    dom_regs = left_sel if len(left_sel) >= len(right_sel) else right_sel
    if len(dom_regs) == 0 and mid_sel:
        dom_regs = ['MID']
    scored = sorted([(r, c.get('region_scores', {}).get(r, 0)) for r in dom_regs],
                    key=lambda x: -x[1])
    top_names = []
    for r, s in scored[:2]:
        bare = REGION_BARE.get(r)
        if bare and bare not in top_names:
            top_names.append(bare)
    region_str = ('maximal in the ' + ' and '.join(top_names) +
                  ' region' + ('s' if len(top_names) > 1 else '')) if top_names else 'no region clearly dominant'

    return f'{st}, {lat_str}; {region_str}.'


def main():
    for subtype in SUBTYPES:
        html_path = os.path.join(SCRIPT_DIR, f'figure_{subtype}_examples.html')
        if not os.path.exists(html_path):
            print(f"  Skipping {subtype}: {html_path} not found")
            continue

        cases = extract_cases(html_path)
        # Add verbal description to each case
        for c in cases:
            c['verbal_description'] = generate_verbal_description(c)

        out_path = os.path.join(SCRIPT_DIR, f'figure_{subtype}_examples_data.json')
        with open(out_path, 'w') as f:
            json.dump(cases, f)
        print(f"  {subtype}: {len(cases)} cases -> {out_path}")
        for c in cases:
            print(f"    {c['difficulty']}: {c['patient_id']} | {c['verbal_description']}")


if __name__ == '__main__':
    main()
