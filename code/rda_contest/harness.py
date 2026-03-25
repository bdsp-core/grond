"""RDA Contest Harness — data loading, method evaluation, leaderboard."""
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

DATA_DIR = PROJECT_DIR / 'data'
EEG_DIR = DATA_DIR / 'eeg'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = PROJECT_DIR / 'results' / 'rda_contest'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def load_contest_data(max_rda=None, max_neg=500, verbose=True):
    """Load contest dataset.

    Returns dict with:
        rda_df: DataFrame of RDA patients (subtype, vote_agreement, gold_standard_freq)
        rda_segs: dict patient_id -> (18, 2000) numpy array (one segment per patient)
        neg_df: DataFrame of non-RDA patients (LPD/GPD as negatives)
        neg_segs: dict patient_id -> (18, 2000) numpy array
    """
    t0 = time.time()
    pat = pd.read_csv(str(LABELS_DIR / 'patients.csv'))
    pat['patient_id'] = pat['patient_id'].astype(str)
    pat['n_expert_votes'] = pd.to_numeric(pat['n_expert_votes'], errors='coerce').fillna(0).astype(int)
    pat['vote_agreement'] = pd.to_numeric(pat['vote_agreement'], errors='coerce').fillna(0)
    pat['gold_standard_freq'] = pd.to_numeric(pat['gold_standard_freq'], errors='coerce')
    pat['vote_lrda'] = pd.to_numeric(pat.get('vote_lrda', 0), errors='coerce').fillna(0)
    pat['vote_grda'] = pd.to_numeric(pat.get('vote_grda', 0), errors='coerce').fillna(0)
    pat['vote_other'] = pd.to_numeric(pat.get('vote_other', 0), errors='coerce').fillna(0)

    seg_df = pd.read_csv(str(LABELS_DIR / 'segments.csv'))
    seg_df['patient_id'] = seg_df['patient_id'].astype(str)

    # ── RDA positives ──
    rda = pat[
        (pat['subtype'].isin(['lrda', 'grda'])) &
        (pat['excluded'] != True) &
        (pat['n_expert_votes'] >= 5)
    ].copy().sort_values('vote_agreement', ascending=False)

    if max_rda:
        rda = rda.head(max_rda)

    if verbose:
        print(f"RDA positives: {len(rda)} ({(rda['subtype']=='lrda').sum()} LRDA, "
              f"{(rda['subtype']=='grda').sum()} GRDA)")

    # ── Non-RDA negatives: cases where experts voted >50% "other" ──
    # These are true non-epileptiform patterns (not PDs, not RDA, not seizures)
    pat['other_frac'] = pat['vote_other'] / pat['n_expert_votes'].clip(lower=1)
    neg = pat[
        (pat['other_frac'] > 0.4) &
        (pat['excluded'] != True) &
        (pat['n_expert_votes'] >= 5)
    ].copy().sort_values('other_frac', ascending=False)
    neg = neg.head(max_neg)

    if verbose:
        print(f"Non-RDA negatives: {len(neg)} ({(neg['subtype']=='lpd').sum()} LPD, "
              f"{(neg['subtype']=='gpd').sum()} GPD)")

    # ── Load EEG segments (one per patient) ──
    def load_patients(df, label):
        segs = {}
        n_skip = 0
        for _, row in df.iterrows():
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
                # Try default name
                seg = load_segment(f'{pid}_seg000.mat')
                if seg is not None and seg.shape == (18, 2000):
                    segs[pid] = seg
                else:
                    n_skip += 1
        if verbose and n_skip:
            print(f"  {label}: skipped {n_skip} (no EEG)")
        return segs

    if verbose:
        print("Loading EEG...")
    rda_segs = load_patients(rda, "RDA")
    neg_segs = load_patients(neg, "Non-RDA")

    # Filter DataFrames to only patients with loaded EEG
    rda = rda[rda['patient_id'].isin(rda_segs.keys())].copy()
    neg = neg[neg['patient_id'].isin(neg_segs.keys())].copy()

    if verbose:
        elapsed = time.time() - t0
        print(f"Loaded in {elapsed:.0f}s: {len(rda_segs)} RDA + {len(neg_segs)} non-RDA segments")

    return {
        'rda_df': rda.reset_index(drop=True),
        'rda_segs': rda_segs,
        'neg_df': neg.reset_index(drop=True),
        'neg_segs': neg_segs,
    }


def run_method(method, data, verbose=True):
    """Run a method on all segments. Returns per-patient results."""
    results = {}
    all_segs = {}
    all_segs.update(data['rda_segs'])
    all_segs.update(data['neg_segs'])

    t0 = time.time()
    n = len(all_segs)
    for i, (pid, seg) in enumerate(all_segs.items()):
        results[pid] = method.analyze(seg)
        if verbose and (i + 1) % 100 == 0:
            print(f"  {method.name}: {i+1}/{n} ({time.time()-t0:.0f}s)")

    if verbose:
        elapsed = time.time() - t0
        print(f"  {method.name}: done ({elapsed:.0f}s)")
    return results


def evaluate(method_results, data):
    """Evaluate a method's results on all three tasks.

    Returns dict with all metrics.
    """
    rda_df = data['rda_df']
    rda_pids = set(rda_df['patient_id'])
    neg_pids = set(data['neg_df']['patient_id'])

    # ── Task A: Q-score vs RDA vote fraction ──
    # rda_fraction = (vote_lrda + vote_grda) / n_expert_votes
    # This measures "how much do experts agree this IS RDA" — NOT overall agreement
    q_scores_a = []
    rda_fractions = []
    for _, row in rda_df.iterrows():
        pid = row['patient_id']
        if pid in method_results:
            q = method_results[pid]['q_score']
            n_exp = row.get('n_expert_votes', 0)
            v_lrda = row.get('vote_lrda', 0)
            v_grda = row.get('vote_grda', 0)
            if n_exp > 0 and np.isfinite(q):
                rda_frac = (v_lrda + v_grda) / n_exp
                q_scores_a.append(q)
                rda_fractions.append(rda_frac)

    task_a_rho = np.nan
    if len(q_scores_a) >= 10:
        task_a_rho, _ = spearmanr(q_scores_a, rda_fractions)

    # ── Task B: RDA vs non-RDA detection ──
    labels_b = []
    scores_b = []
    for pid in rda_pids:
        if pid in method_results:
            labels_b.append(1)
            scores_b.append(method_results[pid]['q_score'])
    for pid in neg_pids:
        if pid in method_results:
            labels_b.append(0)
            scores_b.append(method_results[pid]['q_score'])

    task_b_auc = np.nan
    if len(set(labels_b)) >= 2 and len(labels_b) >= 20:
        task_b_auc = roc_auc_score(labels_b, scores_b)

    # ── Task C: Frequency vs gold standard ──
    pred_freqs = []
    gold_freqs = []
    for _, row in rda_df.iterrows():
        pid = row['patient_id']
        gf = row['gold_standard_freq']
        if pid in method_results and np.isfinite(gf) and gf > 0:
            pf = method_results[pid]['freq']
            if np.isfinite(pf):
                pred_freqs.append(pf)
                gold_freqs.append(gf)

    task_c_rho = np.nan
    task_c_mae = np.nan
    if len(pred_freqs) >= 10:
        task_c_rho, _ = spearmanr(pred_freqs, gold_freqs)
        task_c_mae = float(np.mean(np.abs(np.array(pred_freqs) - np.array(gold_freqs))))

    # Composite
    scores = [v for v in [task_a_rho, task_b_auc, task_c_rho] if np.isfinite(v)]
    composite = float(np.mean(scores)) if scores else 0.0

    return {
        'task_a_rho': round(float(task_a_rho), 4) if np.isfinite(task_a_rho) else None,
        'task_b_auc': round(float(task_b_auc), 4) if np.isfinite(task_b_auc) else None,
        'task_c_rho': round(float(task_c_rho), 4) if np.isfinite(task_c_rho) else None,
        'task_c_mae': round(float(task_c_mae), 3) if np.isfinite(task_c_mae) else None,
        'composite': round(composite, 4),
        'n_rda': len(q_scores_a),
        'n_neg': sum(1 for l in labels_b if l == 0),
        'n_freq': len(pred_freqs),
    }


def save_result(method_name, metrics):
    """Save method results to JSON."""
    metrics['method'] = method_name
    metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = RESULTS_DIR / f'{method_name}.json'
    with open(str(path), 'w') as f:
        json.dump(metrics, f, indent=2)


def update_html_leaderboard(n_total=14):
    """Regenerate the HTML leaderboard file."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        with open(str(path)) as f:
            results.append(json.load(f))
    if not results:
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))
    n_done = len(results)
    pct = n_done / n_total * 100

    rows = ""
    for i, r in enumerate(results):
        def fmt(v, d=4):
            return f"{v:.{d}f}" if v is not None else "--"
        comp = r.get('composite', 0) or 0
        color = '#44cc88' if comp > 0.4 else '#88cc44' if comp > 0.3 else '#cccc44' if comp > 0.2 else '#cc8844'
        rows += (f"<tr><td>{i+1}</td><td style='font-weight:bold'>{r['method']}</td>"
                 f"<td>{fmt(r.get('task_a_rho'))}</td><td>{fmt(r.get('task_b_auc'))}</td>"
                 f"<td>{fmt(r.get('task_c_rho'))}</td><td>{fmt(r.get('task_c_mae'),3)}</td>"
                 f"<td style='color:{color};font-weight:bold'>{fmt(comp)}</td>"
                 f"<td>{r.get('n_rda','--')}</td></tr>\n")

    html = f"""<!DOCTYPE html><html><head><title>RDA Contest</title>
<meta http-equiv="refresh" content="10">
<style>body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px}}
h1{{color:#44cc88}}table{{border-collapse:collapse;width:100%;margin-top:20px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555}}
td{{padding:8px 10px;border-bottom:1px solid #333}}tr:hover{{background:#2a2a2a}}
.prog{{width:100%;height:20px;background:#333;border-radius:10px;margin:10px 0}}
.bar{{height:100%;background:#44cc88;border-radius:10px}}</style></head><body>
<h1>RDA Analysis Contest Leaderboard</h1>
<p>{n_done}/{n_total} methods complete</p>
<div class="prog"><div class="bar" style="width:{pct:.0f}%"></div></div>
<p style="color:#777;font-size:12px">Updated: {time.strftime('%H:%M:%S')}</p>
<table><tr><th>Rank</th><th>Method</th><th>Task A (ρ)<br><small>Q vs RDA frac</small></th><th>Task B (AUC)</th>
<th>Task C (ρ)</th><th>MAE</th><th>Composite</th><th>N</th></tr>
{rows}</table></body></html>"""

    out = RESULTS_DIR.parent / 'rda_contest_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)


def print_leaderboard():
    """Print leaderboard from saved results."""
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        with open(str(path)) as f:
            results.append(json.load(f))

    if not results:
        print("No results yet.")
        return

    results.sort(key=lambda r: -(r.get('composite', 0) or 0))

    print(f"\n{'='*90}")
    print(f"  RDA Contest Leaderboard ({len(results)} methods)")
    print(f"{'='*90}")
    print(f"{'Rank':<5} {'Method':<30} {'TaskA(ρ)':>9} {'TaskB(AUC)':>10} "
          f"{'TaskC(ρ)':>9} {'MAE':>6} {'Composite':>10}")
    print(f"{'-'*90}")

    for i, r in enumerate(results):
        a = f"{r['task_a_rho']:.4f}" if r.get('task_a_rho') is not None else "  --  "
        b = f"{r['task_b_auc']:.4f}" if r.get('task_b_auc') is not None else "  --  "
        c = f"{r['task_c_rho']:.4f}" if r.get('task_c_rho') is not None else "  --  "
        m = f"{r['task_c_mae']:.3f}" if r.get('task_c_mae') is not None else " -- "
        comp = f"{r['composite']:.4f}" if r.get('composite') else "  --  "
        print(f"{i+1:<5} {r['method']:<30} {a:>9} {b:>10} {c:>9} {m:>6} {comp:>10}")

    print(f"{'='*90}")
