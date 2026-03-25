"""Lateralization Contest v2 Harness — IIIC data, LRDA vs GRDA discrimination.

Data: IIIC crowd-labeled cases with >=10 expert votes and >=50% agreement.
Primary metric: AUC of asymmetry score for LRDA vs GRDA classification.
"""
import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
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
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v2'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / '_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
MIN_VOTES = 10
MIN_AGREEMENT = 0.50


def load_segment(mat_file):
    """Load and preprocess a segment to (18, 2000) bipolar."""
    import scipy.io as sio
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
    """Load v2 contest dataset from IIIC expert-majority labels.

    Selection: label_source='expert_majority', n_expert_votes >= 10,
    vote_agreement >= 0.50, subtype in (lrda, grda).

    Returns dict with:
        df: DataFrame with patient_id, subtype, vote_agreement, laterality
        segs: dict patient_id -> (18, 2000) numpy array
    """
    cache_file = CACHE_DIR / 'lateral_v2_data.pkl'
    if cache_file.exists():
        if verbose:
            print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        df = data['df']
        if verbose:
            print(f"Cached: {len(data['segs'])} segments "
                  f"({(df['subtype']=='lrda').sum()} LRDA, {(df['subtype']=='grda').sum()} GRDA)")
        return data

    t0 = time.time()
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'), dtype={'patient_id': str})
    for col in ['n_expert_votes', 'vote_agreement', 'vote_lrda', 'vote_grda']:
        pat[col] = pd.to_numeric(pat[col], errors='coerce').fillna(0)

    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'), dtype={'patient_id': str})

    # IIIC crowd-labeled LRDA/GRDA with strong agreement
    rda = pat[
        (pat['label_source'] == 'expert_majority') &
        (pat['n_expert_votes'] >= MIN_VOTES) &
        (pat['vote_agreement'] >= MIN_AGREEMENT) &
        (pat['subtype'].isin(['lrda', 'grda'])) &
        (pat['excluded'] != True)
    ].copy()

    if verbose:
        print(f"IIIC cases (>={MIN_VOTES} votes, >={MIN_AGREEMENT:.0%} agreement):")
        print(f"  LRDA: {(rda['subtype']=='lrda').sum()}")
        print(f"  GRDA: {(rda['subtype']=='grda').sum()}")

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
        print(f"  LRDA: {(rda['subtype']=='lrda').sum()}, GRDA: {(rda['subtype']=='grda').sum()}")

    # Side validation: LRDA cases with known laterality
    side_known = rda[
        (rda['subtype'] == 'lrda') &
        (rda['laterality'].isin(['left', 'right']))
    ]
    if verbose:
        print(f"  Side validation set: {len(side_known)} LRDA with known side")

    data = {
        'df': rda[['patient_id', 'subtype', 'vote_agreement', 'laterality']].copy(),
        'segs': segs,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    if verbose:
        print(f"Cached to {cache_file}")

    return data


def run_method(method, data, verbose=True):
    """Run a method on all segments."""
    results = {}
    segs = data['segs']
    t0 = time.time()
    n = len(segs)
    for i, (pid, seg) in enumerate(segs.items()):
        results[pid] = method.analyze(seg)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {method.name}: {i + 1}/{n} ({time.time() - t0:.0f}s)")
    if verbose:
        print(f"  {method.name}: done ({time.time() - t0:.0f}s, {len(results)} segments)")
    return results


def evaluate(method_results, data):
    """Evaluate on LRDA vs GRDA discrimination + side validation.

    Primary: AUC of asymmetry for LRDA(=1) vs GRDA(=0)
    Secondary: side accuracy, Cohen's d, Spearman correlation
    """
    df = data['df']

    subtypes, asym_scores, lat_indices = [], [], []
    lats, pred_sides = [], []

    for _, row in df.iterrows():
        pid = row['patient_id']
        if pid not in method_results:
            continue
        r = method_results[pid]
        subtypes.append(row['subtype'])
        asym_scores.append(r['asymmetry'])
        lat_indices.append(r['laterality_index'])
        lats.append(row.get('laterality', None))
        pred_sides.append('right' if r['laterality_index'] > 0 else 'left')

    subtypes = np.array(subtypes)
    asym_scores = np.array(asym_scores, dtype=float)
    lat_indices = np.array(lat_indices, dtype=float)

    n_total = len(subtypes)
    n_lrda = int((subtypes == 'lrda').sum())
    n_grda = int((subtypes == 'grda').sum())

    # ── Primary: LRDA vs GRDA AUC ──
    is_lrda = (subtypes == 'lrda').astype(int)
    primary_auc = np.nan
    if len(set(is_lrda)) >= 2 and n_total >= 20:
        primary_auc = roc_auc_score(is_lrda, asym_scores)

    # ── Cohen's d ──
    cohens_d = np.nan
    lrda_asym = asym_scores[subtypes == 'lrda']
    grda_asym = asym_scores[subtypes == 'grda']
    if len(lrda_asym) > 1 and len(grda_asym) > 1:
        pooled_std = np.sqrt((np.var(lrda_asym) * (len(lrda_asym) - 1) +
                              np.var(grda_asym) * (len(grda_asym) - 1)) /
                             (len(lrda_asym) + len(grda_asym) - 2))
        if pooled_std > 1e-12:
            cohens_d = (np.mean(lrda_asym) - np.mean(grda_asym)) / pooled_std

    # ── Side accuracy (validation only) ──
    side_acc = np.nan
    n_side = 0
    lrda_lr_mask = [(s == 'lrda' and l in ('left', 'right')) for s, l in zip(subtypes, lats)]
    if sum(lrda_lr_mask) >= 5:
        gt_sides = np.array([lats[i] for i in range(len(lats)) if lrda_lr_mask[i]])
        pr_sides = np.array([pred_sides[i] for i in range(len(pred_sides)) if lrda_lr_mask[i]])
        side_acc = float(np.mean(gt_sides == pr_sides))
        n_side = len(gt_sides)

    # ── Sensitivity / Specificity at optimal threshold ──
    sens, spec = np.nan, np.nan
    if np.isfinite(primary_auc) and n_lrda >= 10 and n_grda >= 10:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(is_lrda, asym_scores)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        sens = float(tpr[best_idx])
        spec = float(1 - fpr[best_idx])

    # ── Mean asymmetry by subtype ──
    mean_asym_lrda = float(np.mean(lrda_asym)) if len(lrda_asym) > 0 else None
    mean_asym_grda = float(np.mean(grda_asym)) if len(grda_asym) > 0 else None

    return {
        'primary_auc': _fmt(primary_auc),
        'cohens_d': _fmt(cohens_d),
        'sensitivity': _fmt(sens),
        'specificity': _fmt(spec),
        'side_accuracy': _fmt(side_acc),
        'n_side_validation': n_side,
        'mean_asym_lrda': _fmt(mean_asym_lrda),
        'mean_asym_grda': _fmt(mean_asym_grda),
        'n_total': n_total,
        'n_lrda': n_lrda,
        'n_grda': n_grda,
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


def save_per_patient(method_name, method_results):
    """Save per-patient left/right scores for ensemble methods."""
    cache = CACHE_DIR / f'{method_name}_scores.json'
    out = {}
    for pid, r in method_results.items():
        out[pid] = {
            'left_score': r['left_score'],
            'right_score': r['right_score'],
            'asymmetry': r['asymmetry'],
            'laterality_index': r['laterality_index'],
        }
    with open(str(cache), 'w') as f:
        json.dump(out, f)


def update_html_leaderboard(n_total=25):
    """Regenerate the auto-updating HTML leaderboard."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        if path.name.startswith('_'):
            continue
        with open(str(path)) as f:
            results.append(json.load(f))
    if not results:
        return

    results.sort(key=lambda r: -(r.get('primary_auc', 0) or 0))
    n_done = len(results)
    pct = n_done / n_total * 100

    rows = ""
    for i, r in enumerate(results):
        def fmt(v):
            return f"{v:.4f}" if v is not None else "--"
        auc = r.get('primary_auc', 0) or 0
        color = ('#44cc88' if auc > 0.7 else '#88cc44' if auc > 0.6 else
                 '#cccc44' if auc > 0.55 else '#cc8844')
        rows += (
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td style='font-weight:bold'>{r['method']}</td>"
            f"<td style='color:{color};font-weight:bold;font-size:1.1em'>{fmt(auc)}</td>"
            f"<td>{fmt(r.get('cohens_d'))}</td>"
            f"<td>{fmt(r.get('sensitivity'))}</td>"
            f"<td>{fmt(r.get('specificity'))}</td>"
            f"<td>{fmt(r.get('side_accuracy'))}</td>"
            f"<td>{fmt(r.get('mean_asym_lrda'))}</td>"
            f"<td>{fmt(r.get('mean_asym_grda'))}</td>"
            f"<td>{r.get('n_total', '--')}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html><html><head><title>Lateralization Contest v2</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#44cc88;margin-bottom:5px}}
h2{{color:#888;font-weight:normal;font-size:14px;margin-top:0}}
table{{border-collapse:collapse;width:100%;margin-top:15px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555;font-size:12px}}
td{{padding:8px 10px;border-bottom:1px solid #333}}
tr:hover{{background:#2a2a2a}}
.prog{{width:100%;height:24px;background:#333;border-radius:12px;margin:10px 0;overflow:hidden}}
.bar{{height:100%;background:linear-gradient(90deg,#44cc88,#88cc44);border-radius:12px;
      transition:width 0.5s ease}}
.meta{{color:#666;font-size:11px;margin-top:15px}}
.desc{{color:#888;font-size:12px;margin-top:15px;line-height:1.6}}
.highlight{{background:#2a3a2a}}
</style></head><body>
<h1>Lateralization Contest v2 — LRDA vs GRDA</h1>
<h2>Can hemisphere asymmetry distinguish lateralized from generalized RDA?</h2>
<p style="color:#aaa">Data: {results[0].get('n_lrda','?')} LRDA + {results[0].get('n_grda','?')} GRDA
(IIIC, ≥10 votes, ≥50% agreement)</p>
<div class="prog"><div class="bar" style="width:{pct:.0f}%"></div></div>
<p style="color:#777;font-size:12px">{n_done}/{n_total} methods · Updated {time.strftime('%H:%M:%S')}</p>
<table>
<tr>
<th>#</th><th>Method</th>
<th>AUC ↓<br><small style="color:#44cc88">PRIMARY</small></th>
<th>Cohen's d</th>
<th>Sens</th><th>Spec</th>
<th>Side Acc<br><small>({results[0].get('n_side_validation','?')} cases)</small></th>
<th>LRDA asym</th><th>GRDA asym</th>
<th>N</th>
</tr>
{rows}</table>
<div class="desc">
<p><b>Primary metric (AUC):</b> Can asymmetry between hemisphere scores separate LRDA from GRDA?</p>
<p><b>Design constraint:</b> Every method processes L/R hemispheres independently → outputs (left_score, right_score).
Asymmetry = |L-R|/(L+R). LRDA should have high asymmetry; GRDA should have low.</p>
<p><b>Side accuracy:</b> For LRDA cases with expert-labeled side, does the method pick the correct hemisphere?</p>
</div>
<div class="meta">Auto-refreshes every 5s</div>
</body></html>"""

    out = RESULTS_DIR.parent / 'lateralization_v2_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)


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

    results.sort(key=lambda r: -(r.get('primary_auc', 0) or 0))
    print(f"\n{'=' * 120}")
    print(f"  Lateralization Contest v2 — LRDA vs GRDA ({len(results)} methods)")
    print(f"{'=' * 120}")
    print(f"{'#':<4} {'Method':<28} {'AUC':>8} {'Cohen d':>8} {'Sens':>6} {'Spec':>6} "
          f"{'Side%':>6} {'LRDA ā':>8} {'GRDA ā':>8} {'N':>5}")
    print(f"{'-' * 120}")
    for i, r in enumerate(results):
        def f(k):
            v = r.get(k)
            return f"{v:.4f}" if v is not None else "  -- "
        print(f"{i + 1:<4} {r['method']:<28} {f('primary_auc'):>8} {f('cohens_d'):>8} "
              f"{f('sensitivity'):>6} {f('specificity'):>6} {f('side_accuracy'):>6} "
              f"{f('mean_asym_lrda'):>8} {f('mean_asym_grda'):>8} {r.get('n_total','--'):>5}")
    print(f"{'=' * 120}")
