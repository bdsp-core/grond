#!/usr/bin/env python3
"""Independent-expert v1 inter-rater reliability (IRR) analysis.

Compares pairwise agreement among MW, SZ, TZ, and the GROND algorithm
on the four 200-segment task subsets defined in
paper_materials/independent_expert_tasks/<subtype>/manifest.csv.

Hypothesis under test:
    Each rater agrees with the algorithm at least as well as the raters
    agree with each other, on every task evaluated.

Sources:
- MW labels:  data/labels/segment_labels.csv (consolidated view) +
              data/labels/labels.csv (rater='MW') as a fallback.
- SZ, TZ:     data/labels/labels.csv (rater={SZ,TZ}, round='independent_expert_v1')
              as appended by code/data_management/ingest_independent_expert_v1.py.
- ALGO:       extracted from the rater export JSONs at
              data/labels/raw_inputs/independent_expert_v1/{SZ,TZ}/*.json,
              where the algorithm's pre-filled values (est_freq for PD,
              w05_freq + w05_laterality for RDA, laterality_index for PD lat)
              are baked into every rater's case_data and are deterministic
              per segment.

Metrics:
- frequency_hz (continuous, on segments where both sides labeled it):
      ICC(3,1), Spearman rho, mean absolute error in Hz.
- laterality (binary 'left'/'right', LPD and LRDA only):
      Cohen's kappa, percent agreement.

Pairs:
- 3 expert-expert: MW-SZ, MW-TZ, SZ-TZ
- 3 expert-algorithm: MW-ALGO, SZ-ALGO, TZ-ALGO

Each metric is reported with a bootstrap 95 percent confidence interval
(1000 resamples of segments with replacement). Tasks with insufficient
data on either side are flagged.

Outputs (under results/independent_expert_v1/):
- summary.json           full numerical results
- forest_plot.png        the headline figure: expert-expert vs expert-algo
                         intervals stacked one row per (task, metric)
- scatter_freq.png       4-panel scatter of frequency agreement per task
- coverage.png           bar chart of per-(rater, task, label_type) coverage

Usage:
    conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
TASK_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks'
RESULTS_DIR = PROJECT_DIR / 'results' / 'independent_expert_v1'

ROUND = 'independent_expert_v1'
SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']
PRETTY = {'lpd': 'LPD', 'gpd': 'GPD', 'lrda': 'LRDA', 'grda': 'GRDA'}

# Tasks where each metric is meaningful.
FREQ_TASKS = SUBTYPES
LAT_TASKS = ['lpd', 'lrda']  # GPD is bilateral, GRDA is generalized

# Pair definitions.
EXPERT_RATERS = ['MW', 'SZ', 'TZ']
EE_PAIRS = [('MW','SZ'), ('MW','TZ'), ('SZ','TZ')]
EA_PAIRS = [('MW','ALGO'), ('SZ','ALGO'), ('TZ','ALGO')]


def load_manifest(subtype):
    with open(TASK_DIR / subtype / 'manifest.csv') as f:
        return [r['mat_file'] for r in csv.DictReader(f)]


def load_labels_csv(label_type, round_filter=None, raters=None):
    """Read labels.csv and return {(rater, mat_file): value}.

    For frequency_hz the value is a float (Hz). For laterality, a string.
    """
    out = {}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            if row.get('label_type') != label_type:
                continue
            if raters is not None and row.get('rater') not in raters:
                continue
            if round_filter is not None and row.get('round') != round_filter:
                continue
            mf = row.get('mat_file', '')
            r = row.get('rater', '')
            v = row.get('value', '')
            if not mf or not r or not v:
                continue
            if label_type == 'frequency_hz':
                try:
                    out[(r, mf)] = float(v)
                except ValueError:
                    pass
            elif label_type == 'laterality':
                v = v.strip().lower()
                if v in ('left', 'right'):
                    out[(r, mf)] = v
    return out


def load_segment_labels_csv():
    """Read segment_labels.csv into {mat_file: row_dict}."""
    out = {}
    with open(LABELS_DIR / 'segment_labels.csv') as f:
        for row in csv.DictReader(f):
            out[row.get('mat_file', '')] = row
    return out


def load_rater_jsons():
    """Load all SZ + TZ raw JSONs. Used for algorithm-prediction extraction.

    Returns: dict[mat_file] -> dict with algorithm fields:
        algo_freq:     PD est_freq or RDA w05_freq (continuous)
        algo_lat:      PD: derived from laterality_index sign
                       RDA: w05_laterality
        subtype:       lpd / gpd / lrda / grda

    Algorithm predictions are deterministic per segment (the case_data was
    inlined into all rater HTMLs identically), so the SAME mat_file may
    appear in both SZ and TZ files; we take the first one we see.
    """
    algo = {}
    files = [
        ('TZ/lpd_freq_timing_results_TZ.json',     'lpd',  'pd'),
        ('TZ/gpd_freq_timing_results_TZ.json',     'gpd',  'pd'),
        ('TZ/lrda_freq_labeling_results_TZ.json',  'lrda', 'rda'),
        ('TZ/grda_freq_labeling_results_TZ.json',  'grda', 'rda'),
        ('SZ/lpd_freq_timing_batch1_results.json', 'lpd',  'pd'),
        ('SZ/gpd_freq_timing_batch1_results.json', 'gpd',  'pd'),
        ('SZ/rda_freq_labeling_results-2.json',    None,   'rda_combined'),
    ]
    for rel, subtype_tag, kind in files:
        path = RAW_DIR / rel
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        for entry in d.values():
            mf = entry.get('mat_file')
            sid = entry.get('segment_id')
            if not mf and sid:
                mf = sid + '.mat'
            if not mf:
                continue
            if kind == 'rda_combined':
                this_sub = (entry.get('subtype') or '').lower()
                if this_sub not in ('lrda', 'grda'):
                    continue
            else:
                this_sub = subtype_tag
            if mf in algo:
                continue  # already populated from earlier file

            if kind == 'pd':
                f_val = entry.get('est_freq')
                lat_idx = entry.get('laterality_index')
                # PDProfiler default: positive index -> right, negative -> left
                # (mirroring the viewer's pre-fill heuristic)
                algo_lat = None
                if lat_idx is not None:
                    if lat_idx > 0:
                        algo_lat = 'right'
                    elif lat_idx < 0:
                        algo_lat = 'left'
                algo[mf] = {
                    'algo_freq': float(f_val) if f_val is not None else None,
                    'algo_lat': algo_lat,
                    'subtype': this_sub,
                }
            else:  # rda or rda_combined
                f_val = entry.get('w05_freq')
                lat = entry.get('w05_laterality')
                if lat in ('left', 'right'):
                    algo_lat = lat
                else:
                    algo_lat = None
                algo[mf] = {
                    'algo_freq': float(f_val) if f_val is not None else None,
                    'algo_lat': algo_lat,
                    'subtype': this_sub,
                }
    return algo


def load_rater_accept_status():
    """Per-rater accept/reject status from raw export JSONs.

    The ingester drops rejected entries when writing labels.csv, so we
    read the raw JSONs to recover the reject signal. Returns:

        status[subtype][rater][mat_file] = 'accept' | 'reject_not_rda' | absent

    A segment that does not appear means the rater was not asked about it
    (which on the 200-seg manifests is rare since we built per-rater HTMLs
    that included all 200 segments per subtype).
    """
    status = {sub: {r: {} for r in ('MW', 'SZ', 'TZ')} for sub in SUBTYPES}
    files = [
        ('TZ/lpd_freq_timing_results_TZ.json',     'lpd',  'TZ', 'pd'),
        ('TZ/gpd_freq_timing_results_TZ.json',     'gpd',  'TZ', 'pd'),
        ('TZ/lrda_freq_labeling_results_TZ.json',  'lrda', 'TZ', 'rda'),
        ('TZ/grda_freq_labeling_results_TZ.json',  'grda', 'TZ', 'rda'),
        ('SZ/lpd_freq_timing_batch1_results.json', 'lpd',  'SZ', 'pd'),
        ('SZ/gpd_freq_timing_batch1_results.json', 'gpd',  'SZ', 'pd'),
        ('SZ/rda_freq_labeling_results-2.json',    None,   'SZ', 'rda_combined'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', None, 'MW', 'rda_combined'),
    ]
    for rel, subtype_tag, rater, kind in files:
        path = RAW_DIR / rel
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        for entry in d.values():
            mf = entry.get('mat_file')
            sid = entry.get('segment_id')
            if not mf and sid:
                mf = sid + '.mat'
            if not mf:
                continue
            if kind == 'rda_combined':
                this_sub = (entry.get('subtype') or '').lower()
                if this_sub not in ('lrda', 'grda'):
                    continue
            else:
                this_sub = subtype_tag
            if kind == 'pd':
                # PD viewer: 'rejected' bool + 'review_status'
                if entry.get('rejected') or entry.get('review_status') == 'rejected':
                    s = 'reject_not_rda'
                else:
                    s = 'accept'
            else:
                s = entry.get('action') or 'unknown'
            status[this_sub][rater][mf] = s
    # MW for LPD / GPD: no separate JSON (MW labels live in labels.csv only).
    # We treat all 200 manifest segments as MW-accepted for LPD/GPD since the
    # MW LPD/GPD labels in labels.csv came from regular reviews where MW
    # didn't have a 'reject' option in those viewers.
    for sub in ('lpd', 'gpd'):
        for mf in load_manifest(sub):
            status[sub]['MW'].setdefault(mf, 'accept')
    return status


def consensus_eligible(mf, status_per_rater, policy):
    """Return True if `mf` passes the consensus inclusion filter.

    Policies:
        'any'        -> at least one rater accepted (default; old behavior)
        'majority'   -> >=2 of the rated raters accepted
        'unanimous'  -> all rated raters accepted
    """
    accepts = sum(1 for r, s in status_per_rater.items() if s == 'accept')
    rejects = sum(1 for r, s in status_per_rater.items() if s == 'reject_not_rda')
    rated = accepts + rejects
    if rated == 0:
        return False
    if policy == 'any':
        return accepts >= 1
    if policy == 'majority':
        return accepts > rejects
    if policy == 'unanimous':
        return rejects == 0 and accepts >= 1
    raise ValueError(f'unknown consensus policy: {policy}')


def build_label_tables(consensus='any'):
    """Produce per-task per-rater label tables.

    Returns: tables[subtype]['freq'][rater][mat_file] = value (float Hz)
             tables[subtype]['lat'][rater][mat_file]  = 'left' or 'right'
    where rater in {'MW', 'SZ', 'TZ', 'ALGO'}.

    consensus: which segments are eligible for the IRR analysis.
        'any'       = at least one rater accepted (legacy)
        'majority'  = >=2 of the 3 raters accepted (canonical)
        'unanimous' = all rated raters accepted
    """
    print(f"Loading labels (consensus policy: {consensus})...")

    # SZ + TZ from labels.csv (round=independent_expert_v1)
    sz_tz_freq = load_labels_csv('frequency_hz', round_filter=ROUND, raters={'SZ', 'TZ'})
    sz_tz_lat  = load_labels_csv('laterality',   round_filter=ROUND, raters={'SZ', 'TZ'})

    # MW from segment_labels.csv (consolidated; uses MW from any source) +
    # from labels.csv with rater='MW' (any round) as a fallback.
    seg_labels = load_segment_labels_csv()
    mw_freq = {}
    mw_lat = {}
    for mf, row in seg_labels.items():
        if row.get('expert_freq_rater', '').strip() == 'MW':
            try:
                mw_freq[('MW', mf)] = float(row.get('expert_freq_hz') or '')
            except (ValueError, TypeError):
                pass
        # Laterality on segment_labels.csv is consolidated and not rater-tagged
        # in the column, so fall back to labels.csv for MW laterality below.
    mw_freq_csv = load_labels_csv('frequency_hz', raters={'MW'})
    mw_lat_csv = load_labels_csv('laterality', raters={'MW'})
    for k, v in mw_freq_csv.items():
        mw_freq.setdefault(k, v)
    mw_lat.update(mw_lat_csv)

    # Algorithm predictions per segment
    algo_pred = load_rater_jsons()

    # Per-rater accept/reject status from raw JSONs (used for consensus filter)
    rater_status = load_rater_accept_status()

    # Build per-task tables filtered to manifest segments
    tables = {}
    consensus_excluded = {}
    for sub in SUBTYPES:
        manifest_set = set(load_manifest(sub))
        freq_table = {r: {} for r in ['MW', 'SZ', 'TZ', 'ALGO']}
        lat_table  = {r: {} for r in ['MW', 'SZ', 'TZ', 'ALGO']}
        n_excluded = 0

        for mf in manifest_set:
            # Apply consensus filter (skip segments where insufficient experts accepted)
            status_per_rater = {r: rater_status[sub][r].get(mf) for r in ('MW', 'SZ', 'TZ')}
            if not consensus_eligible(mf, status_per_rater, consensus):
                n_excluded += 1
                continue
            # MW
            if ('MW', mf) in mw_freq:
                freq_table['MW'][mf] = mw_freq[('MW', mf)]
            if ('MW', mf) in mw_lat:
                lat_table['MW'][mf] = mw_lat[('MW', mf)]
            # SZ, TZ
            for r in ('SZ', 'TZ'):
                if (r, mf) in sz_tz_freq:
                    freq_table[r][mf] = sz_tz_freq[(r, mf)]
                if (r, mf) in sz_tz_lat:
                    lat_table[r][mf] = sz_tz_lat[(r, mf)]
            # ALGO
            ap = algo_pred.get(mf)
            if ap is not None:
                if ap['algo_freq'] is not None:
                    freq_table['ALGO'][mf] = ap['algo_freq']
                if ap['algo_lat'] is not None:
                    lat_table['ALGO'][mf] = ap['algo_lat']

        tables[sub] = {'freq': freq_table, 'lat': lat_table, 'manifest': manifest_set}
        consensus_excluded[sub] = n_excluded

    # Per-task: also filter individual rater labels to segments where THAT rater accepted.
    # If an individual rater rejected a segment that nonetheless passed the global
    # consensus filter (e.g., majority accept = MW + TZ; SZ rejected), drop SZ's
    # frequency label on that segment so SZ-ALGO IRR is computed only on segments
    # where SZ herself agreed it was a valid pattern.
    for sub in SUBTYPES:
        for r in ('MW', 'SZ', 'TZ'):
            for mf in list(tables[sub]['freq'][r]):
                if rater_status[sub][r].get(mf) == 'reject_not_rda':
                    tables[sub]['freq'][r].pop(mf, None)
            for mf in list(tables[sub]['lat'][r]):
                if rater_status[sub][r].get(mf) == 'reject_not_rda':
                    tables[sub]['lat'][r].pop(mf, None)

    # Report consensus filter exclusions
    if any(n > 0 for n in consensus_excluded.values()):
        print(f"\nConsensus filter ({consensus}) excluded segments per task:")
        for sub in SUBTYPES:
            print(f"  {sub.upper()}: {consensus_excluded[sub]} of 200 segments excluded")
            print(f"            -> {200 - consensus_excluded[sub]} eligible for IRR")

    # Print coverage
    print("\nCoverage of the 200-seg subsets per (subtype, rater, label_type):")
    print(f"{'subtype':7s} | {'rater':5s} | {'freq':>6s} | {'lat':>6s}")
    print('-' * 40)
    for sub in SUBTYPES:
        for r in ['MW', 'SZ', 'TZ', 'ALGO']:
            nf = len(tables[sub]['freq'][r])
            nl = len(tables[sub]['lat'][r])
            print(f"{sub.upper():7s} | {r:5s} | {nf:>6d} | {nl:>6d}")
    return tables


# ---- Metrics ----

def icc31(x, y):
    """Intraclass correlation coefficient ICC(3,1) for two raters.

    Two-way mixed effects, single-rater, consistency.
    Standard formula:
        ICC(3,1) = (BMS - EMS) / (BMS + (k - 1) * EMS)
    where k = number of raters (=2 here), BMS and EMS are the
    between-targets and residual mean squares from a two-way ANOVA.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 or len(y) != n:
        return np.nan
    k = 2
    # Stack into n x k matrix
    M = np.column_stack([x, y])
    grand = M.mean()
    target_means = M.mean(axis=1)  # length n
    rater_means = M.mean(axis=0)   # length k
    # Sum of squares
    ss_between = k * np.sum((target_means - grand) ** 2)
    ss_total = np.sum((M - grand) ** 2)
    ss_judges = n * np.sum((rater_means - grand) ** 2)
    ss_error = ss_total - ss_between - ss_judges
    df_between = n - 1
    df_error = (n - 1) * (k - 1)
    if df_between <= 0 or df_error <= 0:
        return np.nan
    bms = ss_between / df_between
    ems = ss_error / df_error
    if (bms + (k - 1) * ems) == 0:
        return np.nan
    return (bms - ems) / (bms + (k - 1) * ems)


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())
    if denom == 0:
        return np.nan
    return float((rx * ry).sum() / denom)


def mae(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.abs(x - y)))


def cohen_kappa(a, b):
    """Cohen's kappa for two raters on categorical (here binary) labels."""
    a = list(a); b = list(b)
    if len(a) != len(b) or len(a) < 2:
        return np.nan
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return 1.0  # all-agree on a single class is trivially perfect
    n = len(a)
    po = sum(1 for i in range(n) if a[i] == b[i]) / n
    pa = {c: a.count(c) / n for c in cats}
    pb = {c: b.count(c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    if pe >= 1:
        return np.nan
    return (po - pe) / (1 - pe)


def percent_agree(a, b):
    if len(a) != len(b) or len(a) == 0:
        return np.nan
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


# ---- Pairwise driver ----

def pairwise_freq(label_a, label_b, n_boot=1000, seed=42):
    """Compute ICC, Spearman, MAE between two rater frequency dicts.

    label_a, label_b are {mat_file: float}. Computed on the intersection.
    Returns dict with point estimates and bootstrap 95% CIs.
    """
    common = sorted(set(label_a) & set(label_b))
    n = len(common)
    if n < 5:
        return {'n': n, 'icc': None, 'spearman': None, 'mae': None,
                'icc_ci': None, 'spearman_ci': None, 'mae_ci': None}
    a = np.array([label_a[mf] for mf in common])
    b = np.array([label_b[mf] for mf in common])
    point = {
        'n': n,
        'icc': float(icc31(a, b)),
        'spearman': float(spearman(a, b)),
        'mae': float(mae(a, b)),
    }
    # Bootstrap
    rng = np.random.default_rng(seed)
    iccs, sps, maes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        iccs.append(icc31(a[idx], b[idx]))
        sps.append(spearman(a[idx], b[idx]))
        maes.append(mae(a[idx], b[idx]))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        if len(arr) < 10:
            return None
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    point['icc_ci'] = ci(iccs)
    point['spearman_ci'] = ci(sps)
    point['mae_ci'] = ci(maes)
    return point


def pairwise_lat(label_a, label_b, n_boot=1000, seed=42):
    common = sorted(set(label_a) & set(label_b))
    n = len(common)
    if n < 5:
        return {'n': n, 'kappa': None, 'percent': None,
                'kappa_ci': None, 'percent_ci': None}
    a = [label_a[mf] for mf in common]
    b = [label_b[mf] for mf in common]
    point = {
        'n': n,
        'kappa': float(cohen_kappa(a, b)),
        'percent': float(percent_agree(a, b)),
    }
    rng = np.random.default_rng(seed)
    ks, ps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = [a[i] for i in idx]; sb = [b[i] for i in idx]
        ks.append(cohen_kappa(sa, sb))
        ps.append(percent_agree(sa, sb))
    def ci(arr):
        arr = np.array([x for x in arr if not np.isnan(x)])
        if len(arr) < 10:
            return None
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    point['kappa_ci'] = ci(ks)
    point['percent_ci'] = ci(ps)
    return point


def run_analysis(tables):
    """Compute pairwise IRR for every (task, metric) and pair."""
    results = {'tasks': {}}
    for sub in SUBTYPES:
        T = tables[sub]
        task_results = {'freq': {}, 'lat': {}}
        # Frequency
        for pair in EE_PAIRS + EA_PAIRS:
            r1, r2 = pair
            task_results['freq'][f"{r1}-{r2}"] = pairwise_freq(
                T['freq'][r1], T['freq'][r2]
            )
        # Laterality (only meaningful for LPD and LRDA)
        if sub in LAT_TASKS:
            for pair in EE_PAIRS + EA_PAIRS:
                r1, r2 = pair
                task_results['lat'][f"{r1}-{r2}"] = pairwise_lat(
                    T['lat'][r1], T['lat'][r2]
                )
        else:
            task_results['lat'] = None
        results['tasks'][sub] = task_results
    return results


# ---- Visualizations ----

def plot_forest(results, out_path):
    """The headline figure: one row per (task, metric).

    Each row shows two horizontal intervals: the mean expert-expert IRR
    (with 95% CI from the pair-wise bootstraps) and the mean
    expert-algorithm IRR. If the prediction holds, the algo interval
    overlaps or exceeds the expert-expert interval on every row.
    """
    import matplotlib.pyplot as plt

    # Build the rows
    rows = []
    for sub in SUBTYPES:
        rows.append((PRETTY[sub] + ' frequency  (ICC)', sub, 'freq', 'icc'))
        rows.append((PRETTY[sub] + ' frequency  (Spearman)', sub, 'freq', 'spearman'))
    for sub in LAT_TASKS:
        rows.append((PRETTY[sub] + ' laterality (kappa)', sub, 'lat', 'kappa'))

    fig, ax = plt.subplots(figsize=(11, max(6, 0.55 * len(rows) + 2)))

    y_positions = list(range(len(rows)))[::-1]  # top row first
    EE_OFFSET = +0.18
    EA_OFFSET = -0.18

    for i, (label, sub, mtype, metric) in enumerate(rows):
        y = y_positions[i]
        task_res = results['tasks'][sub]
        if mtype == 'lat' and task_res['lat'] is None:
            continue

        # Collect EE point estimates and CIs
        ee_vals, ee_los, ee_his = [], [], []
        for pair in EE_PAIRS:
            d = task_res[mtype][f"{pair[0]}-{pair[1]}"]
            v = d.get(metric)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ee_vals.append(v)
            ci_key = metric + '_ci'
            ci = d.get(ci_key)
            if ci is not None:
                ee_los.append(ci[0]); ee_his.append(ci[1])

        ea_vals, ea_los, ea_his = [], [], []
        for pair in EA_PAIRS:
            d = task_res[mtype][f"{pair[0]}-{pair[1]}"]
            v = d.get(metric)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            ea_vals.append(v)
            ci_key = metric + '_ci'
            ci = d.get(ci_key)
            if ci is not None:
                ea_los.append(ci[0]); ea_his.append(ci[1])

        # Plot individual pair points
        ee_color = '#cc3344'
        ea_color = '#229966'
        for v in ee_vals:
            ax.plot(v, y + EE_OFFSET, 'o', color=ee_color, markersize=8, alpha=0.85)
        for v in ea_vals:
            ax.plot(v, y + EA_OFFSET, 's', color=ea_color, markersize=8, alpha=0.85)
        # Plot mean +/- range as a horizontal line behind
        if ee_vals:
            ax.hlines(y + EE_OFFSET, min(ee_los) if ee_los else min(ee_vals),
                       max(ee_his) if ee_his else max(ee_vals),
                       color=ee_color, alpha=0.35, linewidth=2)
        if ea_vals:
            ax.hlines(y + EA_OFFSET, min(ea_los) if ea_los else min(ea_vals),
                       max(ea_his) if ea_his else max(ea_vals),
                       color=ea_color, alpha=0.35, linewidth=2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.7)
    ax.axvline(1, color='gray', linestyle=':', linewidth=0.7)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel('Inter-rater reliability  (1.0 = perfect agreement)')
    ax.grid(axis='x', alpha=0.3)
    # Legend
    ee_legend = ax.plot([], [], 'o', color=ee_color, markersize=8, label='Expert--expert pair')[0]
    ea_legend = ax.plot([], [], 's', color=ea_color, markersize=8, label='Expert--algorithm pair')[0]
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_title('Independent expert v1: pairwise inter-rater reliability\n'
                 'Each row shows the 3 expert-expert pairs (red circles) and the 3 expert-algorithm pairs (green squares); '
                 'horizontal bars span the bootstrap 95% CI envelope.',
                 fontsize=11, loc='left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_freq_scatter(tables, out_path):
    """4-panel scatter, one per task, all 6 pairs overlaid."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    pair_colors = {
        ('MW','SZ'):   ('#cc3344', 'o'),
        ('MW','TZ'):   ('#cc6633', 'o'),
        ('SZ','TZ'):   ('#aa3399', 'o'),
        ('MW','ALGO'): ('#229966', 's'),
        ('SZ','ALGO'): ('#226677', 's'),
        ('TZ','ALGO'): ('#33aa44', 's'),
    }
    for ax, sub in zip(axes.flat, SUBTYPES):
        T = tables[sub]['freq']
        for (r1, r2), (color, marker) in pair_colors.items():
            common = sorted(set(T[r1]) & set(T[r2]))
            if not common:
                continue
            x = [T[r1][mf] for mf in common]
            y = [T[r2][mf] for mf in common]
            ax.scatter(x, y, color=color, marker=marker, s=18, alpha=0.6,
                        edgecolors='none', label=f'{r1}-{r2}  (n={len(common)})')
        ax.plot([0, 4], [0, 4], 'k:', linewidth=0.7, alpha=0.5)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 4)
        ax.set_xlabel('Rater 1 frequency (Hz)')
        ax.set_ylabel('Rater 2 frequency (Hz)')
        ax.set_title(PRETTY[sub])
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(alpha=0.3)
    fig.suptitle('Pairwise frequency agreement across raters and the algorithm\n'
                 '(diagonal = perfect agreement; expert-expert pairs = circles, expert-algorithm pairs = squares)',
                 fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_coverage(tables, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    raters = ['MW', 'SZ', 'TZ', 'ALGO']
    bar_colors = ['#666666', '#cc3344', '#3366cc', '#229966']
    width = 0.2
    for ax, mtype, title in [(axes[0], 'freq', 'Frequency labels'),
                              (axes[1], 'lat',  'Laterality labels')]:
        x = np.arange(len(SUBTYPES))
        for i, r in enumerate(raters):
            counts = [len(tables[sub][mtype][r]) for sub in SUBTYPES]
            ax.bar(x + (i - 1.5) * width, counts, width, label=r, color=bar_colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY[s] for s in SUBTYPES])
        ax.set_ylabel('# segments labeled (out of 200)')
        ax.set_title(title)
        ax.set_ylim(0, 210)
        ax.axhline(200, color='gray', linestyle=':', linewidth=0.6)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--algo', choices=['v1', 'v9', 'crnn'], default='v1',
                        help='Which algorithm version to use as the ALGO column. '
                             'v1 = W05 NB-Hilbert (default, baseline). '
                             'v9 = gated hybrid (Plan A). '
                             'crnn = end-to-end neural pitch detector (Plan B). '
                             'For v9/crnn, only LRDA frequency is overridden; other tasks fall back to v1.')
    parser.add_argument('--consensus', choices=['any', 'majority', 'unanimous'], default='majority',
                        help='Inclusion policy: which segments enter the IRR analysis. '
                             'any: at least one rater accepted (legacy, includes data noise). '
                             'majority (default): >=2 of 3 raters accepted (canonical). '
                             'unanimous: all 3 raters agreed it is the labeled pattern.')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tables = build_label_tables(consensus=args.consensus)

    # Override LRDA freq column with v9 or crnn predictions if requested.
    if args.algo == 'v9':
        v9_path = PROJECT_DIR / 'data/labels/independent_expert_v1/v9_predictions.json'
        if v9_path.exists():
            with open(v9_path) as f:
                v9 = json.load(f)
            n_overrides = 0
            for sid, e in v9.items():
                mf = e.get('mat_file')
                if mf in tables['lrda']['freq']['ALGO']:
                    tables['lrda']['freq']['ALGO'][mf] = float(e['v9_freq'])
                    n_overrides += 1
            print(f"Override: replaced LRDA ALGO frequency with V9 predictions ({n_overrides} segs).")
        else:
            print(f"WARNING: --algo v9 requested but {v9_path} not found; falling back to v1.")
    elif args.algo == 'crnn':
        crnn_path = PROJECT_DIR / 'data/labels/independent_expert_v1/lrda_crnn_predictions.json'
        if crnn_path.exists():
            with open(crnn_path) as f:
                cr = json.load(f)
            n_overrides = 0
            for sid, e in cr.items():
                mf = e.get('mat_file')
                if mf in tables['lrda']['freq']['ALGO']:
                    tables['lrda']['freq']['ALGO'][mf] = float(e['crnn_freq'])
                    n_overrides += 1
            print(f"Override: replaced LRDA ALGO frequency with CRNN predictions ({n_overrides} segs).")
        else:
            print(f"WARNING: --algo crnn requested but {crnn_path} not found; falling back to v1.")

    print("\nComputing pairwise IRR (bootstrap n_boot=%d)..." % args.n_boot)
    results = run_analysis(tables)

    # Print a nice text summary
    print("\n" + "=" * 80)
    print("Pairwise IRR summary")
    print("=" * 80)
    for sub in SUBTYPES:
        print(f"\n  {PRETTY[sub]} frequency:")
        for pair in EE_PAIRS + EA_PAIRS:
            d = results['tasks'][sub]['freq'][f"{pair[0]}-{pair[1]}"]
            tag = '(EE)' if pair in EE_PAIRS else '(EA)'
            if d.get('icc') is None:
                print(f"    {pair[0]:5}-{pair[1]:5} {tag}  n={d['n']:3d}  (insufficient data)")
            else:
                ci = d.get('icc_ci')
                ci_str = f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
                print(f"    {pair[0]:5}-{pair[1]:5} {tag}  n={d['n']:3d}  ICC={d['icc']:.3f}{ci_str}  rho={d['spearman']:.3f}  MAE={d['mae']:.3f} Hz")

        if sub in LAT_TASKS:
            print(f"\n  {PRETTY[sub]} laterality:")
            for pair in EE_PAIRS + EA_PAIRS:
                d = results['tasks'][sub]['lat'][f"{pair[0]}-{pair[1]}"]
                tag = '(EE)' if pair in EE_PAIRS else '(EA)'
                if d.get('kappa') is None:
                    print(f"    {pair[0]:5}-{pair[1]:5} {tag}  n={d['n']:3d}  (insufficient data)")
                else:
                    ci = d.get('kappa_ci')
                    ci_str = f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
                    print(f"    {pair[0]:5}-{pair[1]:5} {tag}  n={d['n']:3d}  kappa={d['kappa']:.3f}{ci_str}  pct={d['percent']:.3f}")

    # Save JSON
    summary_path = RESULTS_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
    print(f"\nWrote {summary_path.relative_to(PROJECT_DIR)}")

    # Plots
    plot_forest(results, RESULTS_DIR / 'forest_plot.png')
    print(f"Wrote {(RESULTS_DIR / 'forest_plot.png').relative_to(PROJECT_DIR)}")
    plot_freq_scatter(tables, RESULTS_DIR / 'scatter_freq.png')
    print(f"Wrote {(RESULTS_DIR / 'scatter_freq.png').relative_to(PROJECT_DIR)}")
    plot_coverage(tables, RESULTS_DIR / 'coverage.png')
    print(f"Wrote {(RESULTS_DIR / 'coverage.png').relative_to(PROJECT_DIR)}")


if __name__ == '__main__':
    main()
