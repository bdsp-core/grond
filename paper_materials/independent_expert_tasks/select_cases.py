#!/usr/bin/env python3
"""Select 200 segments per pattern for the independent-expert annotation tasks.

For each subtype (LPD, GPD, LRDA, GRDA), this script samples ~200 segments
approximately uniformly across the 0.5--3.0 Hz frequency range, deduplicated
by patient and quality-filtered. It writes one manifest CSV per subtype that
the labeling viewers can consume.

Stratification source per subtype:
  - LPD/GPD: MW expert_freq_hz (algo_freq_hz is not populated for PDs).
             Segments without an MW frequency are still kept as backfill;
             they go into the "unbinned" pool and are sampled last.
  - LRDA/GRDA: algo_freq_hz from the RDA-Profiler (independent of MW).

The viewer's pre-filled default at run-time still comes from the PDProfiler /
RDA-Profiler, so the colleagues never see MW's labels.

Output (per subtype):
    paper_materials/independent_expert_tasks/<subtype>/manifest.csv

Usage:
    conda run -n morgoth python paper_materials/independent_expert_tasks/select_cases.py
    conda run -n morgoth python paper_materials/independent_expert_tasks/select_cases.py --subtype lpd
    conda run -n morgoth python paper_materials/independent_expert_tasks/select_cases.py --n-cases 200 --seed 42
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
EEG_DIR = PROJECT_DIR / 'data' / 'eeg'
TASKS_DIR = Path(__file__).resolve().parent

SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']

# Stratification frequency column per subtype.
FREQ_COL = {
    'lpd': 'expert_freq_hz',
    'gpd': 'expert_freq_hz',
    'lrda': 'algo_freq_hz',
    'grda': 'algo_freq_hz',
}

# Frequency range and bin width for stratification.
FREQ_LO = 0.5
FREQ_HI = 3.0
BIN_WIDTH = 0.25  # 10 bins of 0.25 Hz each, spanning [0.5, 3.0)


def parse_float(s):
    if s is None:
        return None
    s = str(s).strip()
    if s in ('', 'nan', 'NaN', 'None'):
        return None
    try:
        v = float(s)
        return v if v == v else None  # filter NaN
    except ValueError:
        return None


def load_segments(subtype):
    """Yield (row dict) for non-excluded segments of the given subtype."""
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        for row in csv.DictReader(f):
            if row.get('subtype', '').lower() != subtype:
                continue
            if row.get('excluded', '').strip().lower() in ('true', '1', 'yes'):
                continue
            yield row


def bin_index(freq_hz):
    """Return the 0-indexed bin for a frequency in [FREQ_LO, FREQ_HI), or -1 if out of range."""
    if freq_hz is None:
        return -1
    if freq_hz < FREQ_LO or freq_hz >= FREQ_HI:
        return -1
    return int((freq_hz - FREQ_LO) / BIN_WIDTH)


def n_bins():
    return int(round((FREQ_HI - FREQ_LO) / BIN_WIDTH))


def select_subtype(subtype, n_cases, seed):
    """Select up to n_cases segments for one subtype, stratified by frequency."""
    rng = random.Random(seed)
    freq_col = FREQ_COL[subtype]

    # Step 1: load all candidate segments. Group by patient (so we can dedup).
    by_patient = defaultdict(list)
    for row in load_segments(subtype):
        pid = row.get('patient_id', '').strip()
        if not pid:
            continue
        # Quality requirement: the EEG mat file must actually exist.
        mat_file = row.get('mat_file', '').strip()
        if not mat_file or not (EEG_DIR / mat_file).exists():
            continue
        freq = parse_float(row.get(freq_col))
        # Compute IIIC agreement strength as a tiebreaker (higher is better).
        try:
            iiic_agree = float(row.get('iiic_plurality_frac') or 0)
        except ValueError:
            iiic_agree = 0.0
        try:
            iiic_n = float(row.get('iiic_n_votes') or 0)
        except ValueError:
            iiic_n = 0.0
        by_patient[pid].append({
            'mat_file': mat_file,
            'patient_id': pid,
            'subtype': subtype,
            'strat_freq_hz': freq,
            'strat_freq_source': freq_col,
            'iiic_n_votes': iiic_n,
            'iiic_plurality_frac': iiic_agree,
            'expert_freq_hz': parse_float(row.get('expert_freq_hz')),
            'expert_freq_rater': row.get('expert_freq_rater', '').strip(),
            'algo_freq_hz': parse_float(row.get('algo_freq_hz')),
            'laterality_existing': row.get('laterality', '').strip(),
        })

    # Step 2: pick one segment per patient.
    # Preference order:
    #   (a) prefer segments with a stratification frequency in [FREQ_LO, FREQ_HI)
    #   (b) within the same in-range/out-of-range tier, prefer higher IIIC agreement
    #       (and more votes), to favor cleaner cases.
    one_per_patient = []
    for pid, segs in by_patient.items():
        segs.sort(key=lambda r: (
            0 if bin_index(r['strat_freq_hz']) >= 0 else 1,
            -(r['iiic_plurality_frac'] or 0),
            -(r['iiic_n_votes'] or 0),
            r['mat_file'],  # deterministic tiebreaker
        ))
        one_per_patient.append(segs[0])

    # Step 3: bin the in-range segments.
    bins = defaultdict(list)
    out_of_range = []
    for r in one_per_patient:
        b = bin_index(r['strat_freq_hz'])
        if b >= 0:
            bins[b].append(r)
        else:
            out_of_range.append(r)

    # Sort each bin's contents (deterministic) and shuffle for sampling.
    for b in bins:
        bins[b].sort(key=lambda r: (
            -(r['iiic_plurality_frac'] or 0),
            -(r['iiic_n_votes'] or 0),
            r['mat_file'],
        ))
    out_of_range.sort(key=lambda r: (
        -(r['iiic_plurality_frac'] or 0),
        -(r['iiic_n_votes'] or 0),
        r['mat_file'],
    ))

    # Step 4: stratified sampling.
    target_per_bin = n_cases // n_bins()  # e.g., 200/10 = 20
    selected = []
    bin_supply = {b: list(bins[b]) for b in range(n_bins())}

    # First pass: take up to target_per_bin from each bin (best-quality first).
    for b in range(n_bins()):
        take = bin_supply[b][:target_per_bin]
        selected.extend(take)
        bin_supply[b] = bin_supply[b][target_per_bin:]

    # Second pass: top up from any remaining in-range segments (round-robin
    # across bins, biggest-supply-first), preserving quality order.
    while len(selected) < n_cases:
        bins_with_supply = sorted(
            ((b, len(bin_supply[b])) for b in range(n_bins()) if bin_supply[b]),
            key=lambda x: -x[1],
        )
        if not bins_with_supply:
            break
        for b, _ in bins_with_supply:
            if len(selected) >= n_cases:
                break
            selected.append(bin_supply[b].pop(0))

    # Final pass: if we are STILL short (e.g., LRDA in 2--3 Hz), pull from out-of-range.
    if len(selected) < n_cases:
        needed = n_cases - len(selected)
        selected.extend(out_of_range[:needed])

    return selected


def write_manifest(subtype, rows):
    out_path = TASKS_DIR / subtype / 'manifest.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'mat_file', 'patient_id', 'subtype',
        'strat_freq_hz', 'strat_freq_source',
        'expert_freq_hz', 'expert_freq_rater', 'algo_freq_hz',
        'iiic_n_votes', 'iiic_plurality_frac',
        'laterality_existing',
    ]
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fields})
    return out_path


def report(subtype, rows):
    print(f"\n{subtype.upper()} — selected {len(rows)} segments")
    bin_counts = defaultdict(int)
    n_in_range = 0
    n_no_strat = 0
    for r in rows:
        b = bin_index(r['strat_freq_hz'])
        if b >= 0:
            bin_counts[b] += 1
            n_in_range += 1
        elif r['strat_freq_hz'] is None:
            n_no_strat += 1
        else:
            bin_counts['out'] += 1
    print(f"  in [{FREQ_LO}, {FREQ_HI}) Hz: {n_in_range}    out of range: {bin_counts.get('out', 0)}    no strat freq: {n_no_strat}")
    print(f"  bin counts (target {200 // n_bins()} per bin):")
    for b in range(n_bins()):
        lo = FREQ_LO + b * BIN_WIDTH
        hi = lo + BIN_WIDTH
        bar = '#' * bin_counts.get(b, 0)
        print(f"    {lo:.2f}-{hi:.2f} Hz : {bin_counts.get(b, 0):3d}  {bar}")
    n_unique_pts = len({r['patient_id'] for r in rows})
    print(f"  unique patients: {n_unique_pts} (should equal selected count if dedup worked)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subtype', choices=SUBTYPES + ['all'], default='all')
    parser.add_argument('--n-cases', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    targets = SUBTYPES if args.subtype == 'all' else [args.subtype]

    print(f"Selecting {args.n_cases} cases per subtype, stratified across [{FREQ_LO}, {FREQ_HI}) Hz")
    print(f"Bin width: {BIN_WIDTH} Hz ({n_bins()} bins, target {args.n_cases // n_bins()} per bin)")

    for sub in targets:
        rows = select_subtype(sub, args.n_cases, args.seed)
        path = write_manifest(sub, rows)
        report(sub, rows)
        print(f"  -> wrote {path.relative_to(PROJECT_DIR)}")


if __name__ == '__main__':
    main()
