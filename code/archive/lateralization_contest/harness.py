"""Lateralization Contest Harness — data loading, evaluation, leaderboard.

Ground truth:
    - Subtype (LRDA vs GRDA) from patients.csv
    - Laterality (left/right/bilateral) from patients.csv
    - For evaluation: LRDA-left/right are clearly lateralized cases

Evaluation tasks:
    Task A: LRDA vs GRDA classification (AUC using asymmetry score)
    Task B: Side identification for lateralized LRDA (accuracy, AUC)
    Task C: Laterality index correlation with ground truth (-1=left, 0=bilateral, +1=right)
"""
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = PROJECT_DIR / 'results' / 'lateralization_contest' / '_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FS = 200


def load_segment(mat_file):
    """Load and preprocess a segment to (18, 2000) bipolar."""
    from mne.filter import notch_filter, filter_data
    from scipy.signal import detrend
    from pd_pointiness_acf import fcn_getBanana

    path = EEG_DIR / mat_file
    if not path.exists():
        return None
    mat = sio.loadmat(str(path))
    dk = [k for k in mat if not k.startswith('_')][0]
    seg = mat[dk].astype(np.float64)
    if seg.shape[0] > seg.shape[1]:
        seg = seg.T
    if seg.shape[0] >= 20:
        seg_bi = np.array(fcn_getBanana(seg[:20, :2000]), dtype=np.float64)
    elif seg.shape[0] == 18:
        seg_bi = seg[:18, :2000].copy()
    else:
        return None
    seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
    seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
    for ch in range(seg_bi.shape[0]):
        seg_bi[ch] = detrend(seg_bi[ch], type='linear')
    return seg_bi


def load_contest_data(verbose=True):
    """Load lateralization contest dataset.

    Returns dict with:
        df: DataFrame with patient_id, subtype, laterality, gt_laterality_score
        segs: dict patient_id -> (18, 2000) numpy array
    """
    cache_file = CACHE_DIR / 'lateral_data.pkl'
    if cache_file.exists():
        if verbose:
            print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    t0 = time.time()
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    pat['n_expert_votes'] = pd.to_numeric(pat['n_expert_votes'], errors='coerce').fillna(0).astype(int)

    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    # RDA patients with laterality labels
    rda = pat[
        (pat['subtype'].isin(['lrda', 'grda'])) &
        (pat['excluded'] != True) &
        (pat['laterality'].notna()) &
        (pat['n_expert_votes'] >= 5)
    ].copy()

    # Ground truth laterality score: -1=left, 0=bilateral, +1=right
    lat_map = {'left': -1.0, 'bilateral': 0.0, 'right': 1.0}
    rda['gt_laterality_score'] = rda['laterality'].map(lat_map)
    rda = rda[rda['gt_laterality_score'].notna()].copy()

    if verbose:
        print(f"RDA with laterality: {len(rda)} "
              f"({(rda['subtype'] == 'lrda').sum()} LRDA, "
              f"{(rda['subtype'] == 'grda').sum()} GRDA)")
        for st in ['lrda', 'grda']:
            sub = rda[rda['subtype'] == st]
            for lat in ['left', 'right', 'bilateral']:
                print(f"  {st} {lat}: {(sub['laterality'] == lat).sum()}")

    # Load EEG
    if verbose:
        print("Loading EEG...")
    segs = {}
    n_skip = 0
    for _, row in rda.iterrows():
        pid = row['patient_id']
        pid_segs = seg_df[seg_df['patient_id'] == pid]
        loaded = False
        for _, sr in pid_segs.iterrows():
            seg = load_segment(sr['mat_file'])
            if seg is not None and seg.shape == (18, 2000):
                segs[pid] = seg
                loaded = True
                break
        if not loaded:
            seg = load_segment(f'{pid}_seg000.mat')
            if seg is not None and seg.shape == (18, 2000):
                segs[pid] = seg
            else:
                n_skip += 1

    rda = rda[rda['patient_id'].isin(segs.keys())].copy().reset_index(drop=True)

    if verbose:
        elapsed = time.time() - t0
        print(f"Loaded {len(segs)} segments in {elapsed:.0f}s (skipped {n_skip})")

    data = {
        'df': rda[['patient_id', 'subtype', 'laterality', 'gt_laterality_score']].copy(),
        'segs': segs,
    }

    # Cache
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    if verbose:
        print(f"Cached to {cache_file}")

    return data


def run_method(method, data, verbose=True):
    """Run a method on all segments. Returns per-patient results."""
    results = {}
    segs = data['segs']
    t0 = time.time()
    n = len(segs)
    for i, (pid, seg) in enumerate(segs.items()):
        results[pid] = method.analyze(seg)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {method.name}: {i + 1}/{n} ({time.time() - t0:.0f}s)")
    if verbose:
        elapsed = time.time() - t0
        print(f"  {method.name}: done ({elapsed:.0f}s, {len(results)} segments)")
    return results


def evaluate(method_results, data):
    """Evaluate a method on all three tasks.

    Task A: LRDA vs GRDA classification (AUC using asymmetry)
    Task B: Side identification for lateralized LRDA cases (AUC, accuracy)
    Task C: Laterality index correlation with ground truth
    """
    df = data['df']

    # Collect predictions aligned with ground truth
    pids, subtypes, lats, lat_scores = [], [], [], []
    pred_left, pred_right, pred_lat_idx, pred_asym = [], [], [], []

    for _, row in df.iterrows():
        pid = row['patient_id']
        if pid not in method_results:
            continue
        r = method_results[pid]
        pids.append(pid)
        subtypes.append(row['subtype'])
        lats.append(row['laterality'])
        lat_scores.append(row['gt_laterality_score'])
        pred_left.append(r['left_score'])
        pred_right.append(r['right_score'])
        pred_lat_idx.append(r['laterality_index'])
        pred_asym.append(r['asymmetry'])

    subtypes = np.array(subtypes)
    lats = np.array(lats)
    lat_scores = np.array(lat_scores, dtype=float)
    pred_lat_idx = np.array(pred_lat_idx, dtype=float)
    pred_asym = np.array(pred_asym, dtype=float)
    pred_left = np.array(pred_left, dtype=float)
    pred_right = np.array(pred_right, dtype=float)

    # ── Task A: LRDA vs GRDA classification ──
    # LRDA should have higher asymmetry than GRDA
    is_lrda = (subtypes == 'lrda').astype(int)
    task_a_auc = np.nan
    if len(set(is_lrda)) >= 2 and len(is_lrda) >= 20:
        task_a_auc = roc_auc_score(is_lrda, pred_asym)

    # ── Task B: Side identification for clearly lateralized LRDA ──
    # Only LRDA cases labeled left or right
    lat_mask = (subtypes == 'lrda') & ((lats == 'left') | (lats == 'right'))
    task_b_auc = np.nan
    task_b_acc = np.nan
    n_lateral = lat_mask.sum()
    if n_lateral >= 10:
        # Ground truth: right=1, left=0
        gt_side = (lats[lat_mask] == 'right').astype(int)
        # Prediction: positive laterality_index → right
        pred_side_score = pred_lat_idx[lat_mask]
        if len(set(gt_side)) >= 2:
            task_b_auc = roc_auc_score(gt_side, pred_side_score)
        pred_side_binary = (pred_side_score > 0).astype(int)
        task_b_acc = accuracy_score(gt_side, pred_side_binary)

    # ── Task C: Laterality index correlation ──
    # All cases: gt_laterality_score vs predicted laterality_index
    task_c_rho = np.nan
    finite_mask = np.isfinite(pred_lat_idx) & np.isfinite(lat_scores)
    if finite_mask.sum() >= 20:
        task_c_rho, _ = spearmanr(pred_lat_idx[finite_mask], lat_scores[finite_mask])

    # ── Task D: Per-hemisphere score quality ──
    # For LRDA-left: left_score should be high, right_score should be low
    # For LRDA-right: right_score high, left_score low
    # For GRDA-bilateral: both high
    task_d_correct_hemi = np.nan
    lrda_lr = (subtypes == 'lrda') & ((lats == 'left') | (lats == 'right'))
    if lrda_lr.sum() >= 10:
        correct = 0
        total = 0
        for i in range(len(subtypes)):
            if not lrda_lr[i]:
                continue
            total += 1
            if lats[i] == 'left' and pred_left[i] > pred_right[i]:
                correct += 1
            elif lats[i] == 'right' and pred_right[i] > pred_left[i]:
                correct += 1
        task_d_correct_hemi = correct / max(total, 1)

    # Composite: weighted average of available metrics
    scores = []
    weights = []
    for val, w in [(task_a_auc, 2.0), (task_b_auc, 3.0), (task_c_rho, 2.0), (task_d_correct_hemi, 3.0)]:
        if np.isfinite(val) if isinstance(val, float) else val is not None:
            scores.append(val)
            weights.append(w)
    composite = float(np.average(scores, weights=weights)) if scores else 0.0

    return {
        'task_a_lrda_vs_grda_auc': _fmt(task_a_auc),
        'task_b_side_auc': _fmt(task_b_auc),
        'task_b_side_acc': _fmt(task_b_acc),
        'task_c_lat_rho': _fmt(task_c_rho),
        'task_d_correct_hemi': _fmt(task_d_correct_hemi),
        'composite': round(composite, 4),
        'n_total': len(pids),
        'n_lrda': int((subtypes == 'lrda').sum()),
        'n_grda': int((subtypes == 'grda').sum()),
        'n_lateral_lrda': int(n_lateral),
    }


def _fmt(v, d=4):
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return round(float(v), d)


def save_result(method_name, metrics):
    """Save method results to JSON."""
    metrics['method'] = method_name
    metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = RESULTS_DIR / f'{method_name}.json'
    with open(str(path), 'w') as f:
        json.dump(metrics, f, indent=2)


def update_html_leaderboard(n_total=25):
    """Regenerate the HTML leaderboard."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        if path.name.startswith('_'):
            continue
        with open(str(path)) as f:
            results.append(json.load(f))
    if not results:
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))
    n_done = len(results)
    pct = n_done / n_total * 100

    rows = ""
    for i, r in enumerate(results):
        def fmt(v):
            return f"{v:.4f}" if v is not None else "--"
        comp = r.get('composite', 0) or 0
        color = ('#44cc88' if comp > 0.7 else '#88cc44' if comp > 0.6 else
                 '#cccc44' if comp > 0.5 else '#cc8844')
        rows += (f"<tr><td>{i + 1}</td><td style='font-weight:bold'>{r['method']}</td>"
                 f"<td>{fmt(r.get('task_a_lrda_vs_grda_auc'))}</td>"
                 f"<td>{fmt(r.get('task_b_side_auc'))}</td>"
                 f"<td>{fmt(r.get('task_b_side_acc'))}</td>"
                 f"<td>{fmt(r.get('task_c_lat_rho'))}</td>"
                 f"<td>{fmt(r.get('task_d_correct_hemi'))}</td>"
                 f"<td style='color:{color};font-weight:bold'>{fmt(comp)}</td>"
                 f"<td>{r.get('n_total', '--')}</td></tr>\n")

    html = f"""<!DOCTYPE html><html><head><title>Lateralization Contest</title>
<meta http-equiv="refresh" content="10">
<style>body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px}}
h1{{color:#44cc88}}table{{border-collapse:collapse;width:100%;margin-top:20px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555}}
td{{padding:8px 10px;border-bottom:1px solid #333}}tr:hover{{background:#2a2a2a}}
.prog{{width:100%;height:20px;background:#333;border-radius:10px;margin:10px 0}}
.bar{{height:100%;background:#44cc88;border-radius:10px}}
.desc{{color:#888;font-size:12px;margin-top:5px}}</style></head><body>
<h1>Lateralization Contest Leaderboard</h1>
<p>Goal: Process each hemisphere independently → classify LRDA vs GRDA + identify side</p>
<p>{n_done}/{n_total} methods complete</p>
<div class="prog"><div class="bar" style="width:{pct:.0f}%"></div></div>
<p style="color:#777;font-size:12px">Updated: {time.strftime('%H:%M:%S')}</p>
<table><tr>
<th>Rank</th><th>Method</th>
<th>Task A (AUC)<br><small>LRDA vs GRDA</small></th>
<th>Task B (AUC)<br><small>Side ID</small></th>
<th>Task B (Acc)<br><small>Side ID</small></th>
<th>Task C (ρ)<br><small>Lat corr</small></th>
<th>Task D (%)<br><small>Correct hemi</small></th>
<th>Composite</th>
<th>N</th></tr>
{rows}</table>
<div class="desc">
<p>Task A: Can asymmetry score distinguish LRDA from GRDA? (AUC)</p>
<p>Task B: For LRDA-left/right cases, does laterality_index predict the correct side? (AUC + Accuracy)</p>
<p>Task C: Spearman correlation of predicted laterality_index with ground truth (-1=left, 0=bilateral, +1=right)</p>
<p>Task D: For lateralized LRDA, is the correct hemisphere's score higher?</p>
</div></body></html>"""

    out = RESULTS_DIR.parent / 'lateralization_contest_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)
    if n_done > 0:
        print(f"Leaderboard: {out}")


def print_leaderboard():
    """Print text leaderboard."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        if path.name.startswith('_'):
            continue
        with open(str(path)) as f:
            results.append(json.load(f))
    if not results:
        print("No results yet.")
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))
    print(f"\n{'=' * 110}")
    print(f"  Lateralization Contest Leaderboard ({len(results)} methods)")
    print(f"{'=' * 110}")
    print(f"{'Rank':<5} {'Method':<28} {'A:LRDA/GRDA':>11} {'B:Side AUC':>10} "
          f"{'B:Side Acc':>10} {'C:Lat ρ':>8} {'D:Hemi%':>8} {'Composite':>10}")
    print(f"{'-' * 110}")
    for i, r in enumerate(results):
        def f(k):
            v = r.get(k)
            return f"{v:.4f}" if v is not None else "  --  "
        print(f"{i + 1:<5} {r['method']:<28} {f('task_a_lrda_vs_grda_auc'):>11} "
              f"{f('task_b_side_auc'):>10} {f('task_b_side_acc'):>10} "
              f"{f('task_c_lat_rho'):>8} {f('task_d_correct_hemi'):>8} "
              f"{f('composite'):>10}")
    print(f"{'=' * 110}")
