"""Spatial Localization Contest Harness — data loading, evaluation, leaderboard."""
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = PROJECT_DIR / 'results' / 'spatial_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
REGIONS = ['LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO']


def _parse_regions(spatial_str):
    """Parse a spatial_channels string into a set of canonical region names."""
    if not spatial_str or spatial_str.strip() in ('', '0', 'na', 'NA'):
        return set()
    tokens = spatial_str.replace(',', ' ').split()
    canonical = set()
    for t in tokens:
        t = t.strip().upper()
        if t in ('LF', 'RF', 'LT', 'RT', 'LCP', 'RCP', 'LO', 'RO'):
            canonical.add(t)
        elif t == 'LPC':
            canonical.add('LCP')
        elif t == 'LP':
            canonical.add('LCP')
        elif t == 'LFP':
            canonical.add('LF')
    return canonical


def _load_mat_as_bipolar(mat_path):
    """Load a .mat file and return (18, N) bipolar array."""
    from pd_pointiness_acf import fcn_getBanana

    mat = sio.loadmat(str(mat_path))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] >= 20:
        return np.array(fcn_getBanana(seg[:20, :2000]), dtype=np.float64)
    elif seg.shape[0] == 18:
        return seg[:18, :2000].copy()
    return None


def load_spatial_data(verbose=True):
    """Load the spatial localization contest dataset.

    Gold standard: majority vote across raters (LB, PH, SZ) for each region.
    A region is 'involved' if >=2 of available raters marked it.

    Returns dict with:
        'df': DataFrame with segment_id, patient_id, subtype, gold_regions, gold_extent, n_raters
        'segments': dict mapping segment_id -> (18, 2000) numpy array
    """
    t0 = time.time()

    # Load annotations
    annot = pd.read_csv(str(LABELS_DIR / 'annotations.csv'))
    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_info = {}
    for _, sr in seg_df.iterrows():
        seg_info[sr['segment_id']] = {
            'mat_file': sr['mat_file'],
            'subtype': sr['subtype'],
            'patient_id': str(sr['patient_id']),
            'montage': sr.get('montage', 'monopolar'),
            'n_channels': int(sr.get('n_channels', 20)),
        }

    # Group spatial annotations by segment, only expert raters
    seg_spatial = defaultdict(dict)
    for _, r in annot.iterrows():
        sc = str(r.get('spatial_channels', '')).strip()
        rater = r.get('rater', '')
        if sc and rater in ('LB', 'PH', 'SZ'):
            regions = _parse_regions(sc)
            if regions:
                seg_spatial[r['segment_id']][rater] = regions

    # Build gold standard: majority vote (>=2 raters agree)
    records = []
    for sid, raters in seg_spatial.items():
        if len(raters) < 2:
            continue
        info = seg_info.get(sid)
        if not info:
            continue
        if info['subtype'] not in ('lpd', 'gpd'):
            continue

        # Count votes per region
        region_votes = defaultdict(int)
        for rater_regions in raters.values():
            for reg in rater_regions:
                region_votes[reg] += 1

        n_raters = len(raters)
        threshold = 2 if n_raters >= 3 else 2  # majority
        gold_regions = set()
        for reg, votes in region_votes.items():
            if votes >= threshold:
                gold_regions.add(reg)

        if not gold_regions:
            continue

        gold_extent = len(gold_regions) / len(REGIONS)

        records.append({
            'segment_id': sid,
            'patient_id': info['patient_id'],
            'subtype': info['subtype'],
            'mat_file': info['mat_file'],
            'gold_regions': sorted(gold_regions),
            'gold_extent': gold_extent,
            'n_raters': n_raters,
            'rater_labels': {k: sorted(v) for k, v in raters.items()},
        })

    if verbose:
        print(f"Gold standard: {len(records)} segments "
              f"({sum(1 for r in records if r['subtype']=='lpd')} LPD, "
              f"{sum(1 for r in records if r['subtype']=='gpd')} GPD)")

    # Load EEG segments
    segments = {}
    n_loaded = 0
    n_failed = 0
    for rec in records:
        sid = rec['segment_id']
        mat_path = EEG_DIR / rec['mat_file']
        if not mat_path.exists():
            n_failed += 1
            continue
        try:
            seg = _load_mat_as_bipolar(mat_path)
            if seg is not None and seg.shape == (18, 2000):
                segments[sid] = seg
                n_loaded += 1
            else:
                n_failed += 1
        except Exception:
            n_failed += 1

    # Filter records to those with loaded EEG
    records = [r for r in records if r['segment_id'] in segments]

    df = pd.DataFrame(records)

    if verbose:
        elapsed = time.time() - t0
        n_patients = df['patient_id'].nunique()
        print(f"Loaded {n_loaded} segments ({n_patients} patients) in {elapsed:.0f}s")
        if n_failed:
            print(f"  Skipped {n_failed} (no EEG)")

        # Gold standard stats
        all_gold = [set(r['gold_regions']) for r in records]
        mean_regions = np.mean([len(g) for g in all_gold])
        print(f"  Mean regions per segment: {mean_regions:.1f}")
        from collections import Counter
        region_freq = Counter()
        for g in all_gold:
            for r in g:
                region_freq[r] += 1
        for r in REGIONS:
            print(f"    {r}: {region_freq[r]} ({region_freq[r]/len(records)*100:.0f}%)")

    return {
        'df': df,
        'segments': segments,
    }


def run_method(method, data, verbose=True):
    """Run a method on all segments."""
    df = data['df']
    segments = data['segments']
    results = {}

    t0 = time.time()
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        sid = row['segment_id']
        seg = segments.get(sid)
        if seg is None:
            continue
        results[sid] = method.analyze(seg, subtype=row['subtype'])
        if verbose and (i + 1) % 100 == 0:
            print(f"  {method.name}: {i+1}/{n} ({time.time()-t0:.0f}s)")

    if verbose:
        elapsed = time.time() - t0
        print(f"  {method.name}: done ({elapsed:.1f}s, {len(results)} segments)")
    return results


def evaluate(method_results, data):
    """Evaluate spatial localization results.

    Metrics:
    1. Region F1 (macro): per-region F1 averaged across 8 regions
    2. Region F1 (micro): treating each (segment, region) as a binary prediction
    3. Jaccard (IoU): mean intersection-over-union of predicted vs gold regions
    4. Spatial extent Spearman: correlation of predicted vs gold spatial extent
    5. Per-region AUC: area under ROC for each region's score vs gold involvement
    6. Exact match: fraction of segments where predicted set == gold set
    """
    df = data['df']

    # Build binary matrices: (n_segments, 8_regions)
    gold_matrix = []
    pred_matrix = []
    pred_score_matrix = []
    gold_extents = []
    pred_extents = []
    segment_ids = []

    for _, row in df.iterrows():
        sid = row['segment_id']
        if sid not in method_results:
            continue

        result = method_results[sid]
        gold_regions = set(row['gold_regions'])

        gold_vec = [1 if r in gold_regions else 0 for r in REGIONS]
        pred_regions = set(result.get('involved_regions', []))
        pred_vec = [1 if r in pred_regions else 0 for r in REGIONS]
        score_vec = [result.get('region_scores', {}).get(r, 0.0) for r in REGIONS]

        gold_matrix.append(gold_vec)
        pred_matrix.append(pred_vec)
        pred_score_matrix.append(score_vec)
        gold_extents.append(row['gold_extent'])
        pred_extents.append(result.get('spatial_extent', 0.0))
        segment_ids.append(sid)

    if len(gold_matrix) < 10:
        return _empty_metrics()

    gold_arr = np.array(gold_matrix)
    pred_arr = np.array(pred_matrix)
    score_arr = np.array(pred_score_matrix)
    n_segs = gold_arr.shape[0]

    # 1. Region F1 (macro) — average per-region F1
    per_region_f1 = {}
    for j, r in enumerate(REGIONS):
        if gold_arr[:, j].sum() > 0:
            per_region_f1[r] = f1_score(gold_arr[:, j], pred_arr[:, j], zero_division=0)
        else:
            per_region_f1[r] = np.nan
    valid_f1s = [v for v in per_region_f1.values() if np.isfinite(v)]
    macro_f1 = float(np.mean(valid_f1s)) if valid_f1s else 0.0

    # 2. Region F1 (micro) — flatten all (segment, region) predictions
    micro_f1 = f1_score(gold_arr.ravel(), pred_arr.ravel(), zero_division=0)

    # 3. Jaccard (IoU) — per segment, then average
    jaccards = []
    for i in range(n_segs):
        g = set(np.where(gold_arr[i] == 1)[0])
        p = set(np.where(pred_arr[i] == 1)[0])
        if len(g) == 0 and len(p) == 0:
            jaccards.append(1.0)
        elif len(g | p) == 0:
            jaccards.append(0.0)
        else:
            jaccards.append(len(g & p) / len(g | p))
    mean_jaccard = float(np.mean(jaccards))

    # 4. Spatial extent Spearman
    from scipy.stats import spearmanr
    extent_rho = np.nan
    if len(gold_extents) >= 10:
        rho, _ = spearmanr(gold_extents, pred_extents)
        extent_rho = float(rho) if np.isfinite(rho) else np.nan

    # 5. Per-region AUC
    per_region_auc = {}
    for j, r in enumerate(REGIONS):
        g = gold_arr[:, j]
        s = score_arr[:, j]
        if len(set(g)) >= 2:
            try:
                per_region_auc[r] = float(roc_auc_score(g, s))
            except ValueError:
                per_region_auc[r] = np.nan
        else:
            per_region_auc[r] = np.nan
    valid_aucs = [v for v in per_region_auc.values() if np.isfinite(v)]
    mean_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # 6. Exact match
    exact = float(np.mean([
        1.0 if np.array_equal(gold_arr[i], pred_arr[i]) else 0.0
        for i in range(n_segs)
    ]))

    # Composite score (weighted average of key metrics)
    composite = 0.30 * macro_f1 + 0.25 * mean_jaccard + 0.25 * mean_auc + 0.20 * micro_f1

    return {
        'macro_f1': round(macro_f1, 4),
        'micro_f1': round(micro_f1, 4),
        'jaccard': round(mean_jaccard, 4),
        'extent_rho': round(extent_rho, 4) if np.isfinite(extent_rho) else None,
        'mean_auc': round(mean_auc, 4),
        'exact_match': round(exact, 4),
        'composite': round(composite, 4),
        'per_region_f1': {r: round(v, 4) if np.isfinite(v) else None for r, v in per_region_f1.items()},
        'per_region_auc': {r: round(v, 4) if np.isfinite(v) else None for r, v in per_region_auc.items()},
        'n_segments': n_segs,
    }


def _empty_metrics():
    return {
        'macro_f1': None, 'micro_f1': None, 'jaccard': None,
        'extent_rho': None, 'mean_auc': None, 'exact_match': None,
        'composite': 0.0, 'per_region_f1': {}, 'per_region_auc': {},
        'n_segments': 0,
    }


def save_result(method_name, metrics):
    """Save method results to JSON."""
    metrics['method'] = method_name
    metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = RESULTS_DIR / f'{method_name}.json'
    with open(str(path), 'w') as f:
        json.dump(metrics, f, indent=2)


def update_html_leaderboard(n_total=24):
    """Regenerate the live HTML leaderboard."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        try:
            with open(str(path)) as f:
                results.append(json.load(f))
        except Exception:
            continue
    if not results:
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))
    n_done = len(results)
    pct = n_done / n_total * 100

    # Per-region detail for best method
    best = results[0] if results else {}

    rows = ""
    for i, r in enumerate(results):
        def fmt(v, d=4):
            return f"{v:.{d}f}" if v is not None else "--"
        comp = r.get('composite', 0) or 0
        color = ('#44cc88' if comp > 0.6 else '#88cc44' if comp > 0.5
                 else '#cccc44' if comp > 0.4 else '#cc8844' if comp > 0.3 else '#cc4444')
        rows += (
            f"<tr>"
            f"<td>{i+1}</td>"
            f"<td style='font-weight:bold'>{r.get('method','?')}</td>"
            f"<td>{fmt(r.get('macro_f1'))}</td>"
            f"<td>{fmt(r.get('micro_f1'))}</td>"
            f"<td>{fmt(r.get('jaccard'))}</td>"
            f"<td>{fmt(r.get('mean_auc'))}</td>"
            f"<td>{fmt(r.get('extent_rho'))}</td>"
            f"<td>{fmt(r.get('exact_match'))}</td>"
            f"<td style='color:{color};font-weight:bold'>{fmt(comp)}</td>"
            f"<td>{r.get('n_segments','--')}</td>"
            f"</tr>\n"
        )

    # Per-region breakdown for top method
    region_rows = ""
    if best.get('per_region_f1'):
        for reg in REGIONS:
            f1_val = best.get('per_region_f1', {}).get(reg)
            auc_val = best.get('per_region_auc', {}).get(reg)
            f1_str = f"{f1_val:.3f}" if f1_val is not None else "--"
            auc_str = f"{auc_val:.3f}" if auc_val is not None else "--"
            region_rows += f"<tr><td>{reg}</td><td>{f1_str}</td><td>{auc_str}</td></tr>\n"

    html = f"""<!DOCTYPE html><html><head><title>Spatial Contest</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#1a1a1a;color:#eee;font-family:'Consolas','Monaco',monospace;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#ff9800;margin-bottom:5px}}
h2{{color:#44cc88;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin-top:10px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555;font-size:13px}}
td{{padding:8px 10px;border-bottom:1px solid #333;font-size:13px}}
tr:hover{{background:#2a2a2a}}
.prog{{width:100%;height:24px;background:#333;border-radius:12px;margin:10px 0;overflow:hidden}}
.bar{{height:100%;background:linear-gradient(90deg,#ff9800,#44cc88);border-radius:12px;transition:width 0.5s}}
.stats{{display:flex;gap:20px;flex-wrap:wrap;margin:15px 0}}
.stat-box{{background:#2a2a2a;padding:12px 20px;border-radius:8px;border:1px solid #444}}
.stat-val{{font-size:24px;font-weight:bold;color:#44cc88}}
.stat-label{{font-size:11px;color:#888;margin-top:2px}}
.two-col{{display:grid;grid-template-columns:1fr 300px;gap:20px}}
.small-table{{width:auto}}
.small-table td,.small-table th{{padding:5px 12px}}
p.updated{{color:#555;font-size:11px;margin:5px 0}}
</style></head><body>

<h1>PD Spatial Localization Contest</h1>
<p style="color:#aaa;margin-top:0">Which brain regions have periodic discharges? 466 segments, 8 regions, {n_total} methods.</p>

<div class="stats">
  <div class="stat-box"><div class="stat-val">{n_done}/{n_total}</div><div class="stat-label">Methods Complete</div></div>
  <div class="stat-box"><div class="stat-val">{results[0].get('composite',0):.4f}</div><div class="stat-label">Best Composite</div></div>
  <div class="stat-box"><div class="stat-val">{results[0].get('method','--')}</div><div class="stat-label">Current Leader</div></div>
</div>

<div class="prog"><div class="bar" style="width:{pct:.0f}%"></div></div>
<p class="updated">Auto-refreshes every 5s | Updated: {time.strftime('%H:%M:%S')}</p>

<div class="two-col">
<div>
<h2>Leaderboard</h2>
<table>
<tr>
  <th>#</th><th>Method</th>
  <th>Macro F1</th><th>Micro F1</th><th>Jaccard</th>
  <th>Mean AUC</th><th>Extent rho</th><th>Exact</th>
  <th>Composite</th><th>N</th>
</tr>
{rows}
</table>
</div>

<div>
<h2>Per-Region (Leader)</h2>
<table class="small-table">
<tr><th>Region</th><th>F1</th><th>AUC</th></tr>
{region_rows}
</table>

<h2 style="margin-top:20px">Metrics</h2>
<p style="color:#888;font-size:11px;line-height:1.6">
<b>Macro F1</b>: Avg F1 across 8 regions<br>
<b>Micro F1</b>: F1 over all (seg,region) pairs<br>
<b>Jaccard</b>: Mean IoU of predicted vs gold<br>
<b>Mean AUC</b>: Avg per-region AUC on scores<br>
<b>Extent rho</b>: Spearman(pred extent, gold)<br>
<b>Exact</b>: Fraction with perfect region match<br>
<b>Composite</b>: 0.30*MacF1 + 0.25*Jac + 0.25*AUC + 0.20*MicF1
</p>
</div>
</div>

</body></html>"""

    out = RESULTS_DIR.parent / 'spatial_contest_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)


def print_leaderboard():
    """Print leaderboard from saved results."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        try:
            with open(str(path)) as f:
                results.append(json.load(f))
        except Exception:
            continue
    if not results:
        print("No results yet.")
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))

    print(f"\n{'='*110}")
    print(f"  Spatial Localization Contest ({len(results)} methods)")
    print(f"{'='*110}")
    print(f"{'#':<4} {'Method':<30} {'MacF1':>7} {'MicF1':>7} {'Jacc':>7} "
          f"{'AUC':>7} {'ExtRho':>7} {'Exact':>7} {'Comp':>8} {'N':>5}")
    print(f"{'-'*110}")

    for i, r in enumerate(results):
        def fmt(v):
            return f"{v:.4f}" if v is not None else "  --  "
        print(f"{i+1:<4} {r.get('method','?'):<30} "
              f"{fmt(r.get('macro_f1')):>7} {fmt(r.get('micro_f1')):>7} "
              f"{fmt(r.get('jaccard')):>7} {fmt(r.get('mean_auc')):>7} "
              f"{fmt(r.get('extent_rho')):>7} {fmt(r.get('exact_match')):>7} "
              f"{fmt(r.get('composite')):>8} {r.get('n_segments','--'):>5}")
    print(f"{'='*110}")
