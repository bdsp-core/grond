"""Lateralization Contest V4 — Clean IIIC per-segment labels.

Data: >=10 expert votes per segment, plurality LRDA or GRDA.
All segments monopolar (>=19 channels), converted to bipolar on-the-fly.
Balanced: subsample GRDA to match LRDA count.

Methods must process hemispheres independently → output (left_score, right_score).
Primary metric: AUC of asymmetry for LRDA vs GRDA classification.
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
RESULTS_DIR = PROJECT_DIR / 'results' / 'lateralization_contest_v4'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / '_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FS = 200
MIN_VOTES = 1  # Use all labeled segments; n_votes available for weighting


def monopolar_to_bipolar(seg_mono):
    """Convert 19-channel monopolar to 18-channel bipolar (longitudinal banana montage)."""
    from pd_pointiness_acf import fcn_getBanana
    if seg_mono.shape[0] > seg_mono.shape[1]:
        seg_mono = seg_mono.T
    if seg_mono.shape[0] >= 19:
        return np.array(fcn_getBanana(seg_mono[:19, :]), dtype=np.float64)
    elif seg_mono.shape[0] == 18:
        return seg_mono.astype(np.float64)
    return None


def load_contest_data(verbose=True):
    """Load V4 contest data from segment_labels.csv.

    Returns dict with:
        df: DataFrame with segment info
        segs: dict mat_file -> (18, 2000) bipolar numpy array
    """
    cache_file = CACHE_DIR / 'v4_data.pkl'
    if cache_file.exists():
        if verbose:
            print("Loading from cache...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        if verbose:
            df = data['df']
            print(f"Cached: {len(data['segs'])} segments "
                  f"({(df['subtype'] == 'lrda').sum()} LRDA, {(df['subtype'] == 'grda').sum()} GRDA)")
        return data

    t0 = time.time()
    sl = pd.read_csv(str(LABELS_DIR / 'segment_labels.csv'))

    # Select: >=10 votes, plurality LRDA or GRDA, not excluded
    rda = sl[
        (sl['n_votes'] >= MIN_VOTES) &
        (sl['plurality'].isin(['lrda', 'grda'])) &
        (~sl['excluded'].fillna(False).astype(bool))
    ].copy()
    rda['subtype'] = rda['plurality']

    if verbose:
        print(f"Segments with >={MIN_VOTES} votes: "
              f"{(rda['subtype'] == 'lrda').sum()} LRDA, {(rda['subtype'] == 'grda').sum()} GRDA")

    # Load and convert to bipolar
    if verbose:
        print("Loading EEG...")
    from mne.filter import notch_filter, filter_data
    from scipy.signal import detrend

    segs = {}
    n_skip = 0
    for _, row in rda.iterrows():
        mat_file = row['mat_file']
        path = EEG_DIR / mat_file
        if not path.exists():
            n_skip += 1
            continue
        try:
            mat = sio.loadmat(str(path))
            dk = [k for k in mat if not k.startswith('_')][0]
            raw = mat[dk].astype(np.float64)
            seg_bi = monopolar_to_bipolar(raw)
            if seg_bi is None or seg_bi.shape != (18, 2000):
                n_skip += 1
                continue
            seg_bi = notch_filter(seg_bi, FS, 60, n_jobs=1, verbose='ERROR')
            seg_bi = filter_data(seg_bi, FS, 0.5, 40, n_jobs=1, verbose='ERROR')
            for ch in range(18):
                seg_bi[ch] = detrend(seg_bi[ch], type='linear')
            segs[mat_file] = seg_bi
        except Exception as e:
            n_skip += 1

    rda = rda[rda['mat_file'].isin(segs.keys())].copy().reset_index(drop=True)

    if verbose:
        print(f"Loaded {len(segs)} segments in {time.time() - t0:.0f}s (skipped {n_skip})")
        print(f"  LRDA: {(rda['subtype'] == 'lrda').sum()}, GRDA: {(rda['subtype'] == 'grda').sum()}")

    data = {
        'df': rda[['mat_file', 'segment_id', 'patient_id', 'subtype',
                    'n_votes', 'vote_lrda', 'vote_grda', 'plurality_frac',
                    'laterality', 'mw_freq', 'auto_freq']].copy(),
        'segs': segs,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    if verbose:
        print(f"Cached to {cache_file}")

    return data


def run_method(method, data, verbose=True):
    """Run a LateralMethod on all segments."""
    results = {}
    segs = data['segs']
    t0 = time.time()
    n = len(segs)
    for i, (mat_file, seg) in enumerate(segs.items()):
        results[mat_file] = method.analyze(seg)
        if verbose and (i + 1) % 200 == 0:
            print(f"  {method.name}: {i + 1}/{n} ({time.time() - t0:.0f}s)")
    if verbose:
        print(f"  {method.name}: done ({time.time() - t0:.0f}s)")
    return results


def evaluate(method_results, data):
    """Evaluate on LRDA vs GRDA discrimination."""
    df = data['df']

    subtypes, asym_scores = [], []
    for _, row in df.iterrows():
        mf = row['mat_file']
        if mf not in method_results:
            continue
        r = method_results[mf]
        subtypes.append(row['subtype'])
        asym_scores.append(r['asymmetry'])

    subtypes = np.array(subtypes)
    asym_scores = np.array(asym_scores, dtype=float)
    is_lrda = (subtypes == 'lrda').astype(int)

    primary_auc = np.nan
    if len(set(is_lrda)) >= 2 and len(is_lrda) >= 20:
        primary_auc = roc_auc_score(is_lrda, asym_scores)

    # Cohen's d
    cohens_d = np.nan
    lrda_a = asym_scores[subtypes == 'lrda']
    grda_a = asym_scores[subtypes == 'grda']
    if len(lrda_a) > 1 and len(grda_a) > 1:
        pooled = np.sqrt((np.var(lrda_a) * (len(lrda_a) - 1) + np.var(grda_a) * (len(grda_a) - 1)) /
                         (len(lrda_a) + len(grda_a) - 2))
        if pooled > 1e-12:
            cohens_d = (np.mean(lrda_a) - np.mean(grda_a)) / pooled

    # Also evaluate on high-quality subset (>=10 votes)
    hq_mask = np.array([row['n_votes'] >= 10 for _, row in df.iterrows() if row['mat_file'] in method_results])
    hq_auc = np.nan
    if hq_mask.sum() >= 20 and len(set(is_lrda[hq_mask])) >= 2:
        hq_auc = roc_auc_score(is_lrda[hq_mask], asym_scores[hq_mask])

    # ── Frequency evaluation ──
    freq_rho = np.nan
    n_freq = 0
    pred_freqs, gold_freqs = [], []
    for _, row in df.iterrows():
        mf = row['mat_file']
        if mf not in method_results:
            continue
        r = method_results[mf]
        pred_f = r.get('extras', {}).get('freq')
        # Gold standard: prefer mw_freq, fall back to auto_freq
        gold_f = row.get('mw_freq') if pd.notna(row.get('mw_freq')) else row.get('auto_freq')
        if pred_f and np.isfinite(pred_f) and gold_f and pd.notna(gold_f):
            gold_f = float(gold_f)
            if gold_f > 0:
                pred_freqs.append(pred_f)
                gold_freqs.append(gold_f)
    if len(pred_freqs) >= 10:
        freq_rho, _ = spearmanr(pred_freqs, gold_freqs)
        n_freq = len(pred_freqs)

    return {
        'primary_auc': round(float(primary_auc), 4) if np.isfinite(primary_auc) else None,
        'hq_auc': round(float(hq_auc), 4) if np.isfinite(hq_auc) else None,
        'cohens_d': round(float(cohens_d), 4) if np.isfinite(cohens_d) else None,
        'freq_rho': round(float(freq_rho), 4) if np.isfinite(freq_rho) else None,
        'n_freq': n_freq,
        'mean_asym_lrda': round(float(np.mean(lrda_a)), 4),
        'mean_asym_grda': round(float(np.mean(grda_a)), 4),
        'n_total': len(subtypes),
        'n_lrda': int((subtypes == 'lrda').sum()),
        'n_grda': int((subtypes == 'grda').sum()),
        'n_hq': int(hq_mask.sum()),
    }


def save_result(method_name, metrics):
    metrics['method'] = method_name
    metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    path = RESULTS_DIR / f'{method_name}.json'
    with open(str(path), 'w') as f:
        json.dump(metrics, f, indent=2)


def update_html_leaderboard(n_total=25):
    results = []
    for path in sorted(RESULTS_DIR.glob('*.json')):
        if path.name.startswith('_'):
            continue
        with open(str(path)) as f:
            data = json.load(f)
        if isinstance(data, dict) and 'primary_auc' in data:
            results.append(data)
    if not results:
        return

    results.sort(key=lambda r: -(r.get('primary_auc', 0) or 0))
    n_done = len(results)

    rows = ""
    for i, r in enumerate(results):
        auc = r.get('primary_auc', 0) or 0
        color = ('#44cc88' if auc > 0.70 else '#88cc44' if auc > 0.65 else
                 '#cccc44' if auc > 0.60 else '#cc8844' if auc > 0.55 else '#cc4444')
        bar_w = max(0, (auc - 0.5) * 200)
        hq = r.get('hq_auc')
        hq_str = f"{hq:.4f}" if hq else "--"
        frho = r.get('freq_rho')
        frho_str = f"{frho:.4f}" if frho else "--"
        frho_color = '#44cc88' if frho and frho > 0.7 else '#88cc44' if frho and frho > 0.5 else '#aaa'
        rows += (
            f"<tr><td>{i + 1}</td>"
            f"<td style='font-weight:bold'>{r['method']}</td>"
            f"<td style='color:{color};font-weight:bold;font-size:1.1em'>{auc:.4f}</td>"
            f"<td>{hq_str}</td>"
            f"<td style='color:{frho_color}'>{frho_str}</td>"
            f"<td>{r.get('cohens_d', '--')}</td>"
            f"<td><div style='background:#333;border-radius:4px;height:14px;width:120px'>"
            f"<div style='background:{color};height:100%;width:{bar_w}px;border-radius:4px'></div></div></td>"
            f"<td>{r.get('n_total', '--')} ({r.get('n_hq', '--')} HQ)</td>"
            f"</tr>\n"
        )

    best = results[0]
    html = f"""<!DOCTYPE html><html><head><title>V4 Lateralization Contest</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#1a1a1a;color:#eee;font-family:'Consolas',monospace;padding:20px;max-width:1400px;margin:0 auto}}
h1{{color:#44cc88;margin-bottom:5px}}
h2{{color:#888;font-weight:normal;font-size:14px;margin-top:0}}
table{{border-collapse:collapse;width:100%;margin-top:15px}}
th{{background:#333;padding:10px;text-align:left;border-bottom:2px solid #555;font-size:12px}}
td{{padding:6px 10px;border-bottom:1px solid #333}}
tr:hover{{background:#2a2a2a}}
tr:first-child td{{background:#1a2a1a}}
.best{{color:#44cc88;font-size:24px;margin:10px 0}}
</style></head><body>
<h1>V4 Lateralization Contest — Hemisphere-Independent Methods</h1>
<h2>Clean IIIC per-segment labels (>={MIN_VOTES} expert votes) — constraint: L/R hemispheres processed independently</h2>
<div class="best">Best AUC: {best.get('primary_auc', 0):.4f} ({best.get('method', '?')})</div>
<p style="color:#777">{n_done}/{n_total} methods · Updated {time.strftime('%H:%M:%S')}</p>
<table>
<tr><th>#</th><th>Method</th><th>AUC ↓<br><small>all data</small></th><th>HQ AUC<br><small>≥10 votes</small></th>
<th>Freq ρ</th><th>Cohen's d</th><th>AUC bar</th><th>N</th></tr>
{rows}</table>
<p style="color:#555;font-size:11px;margin-top:20px">
Design: each method outputs (left_score, right_score) by processing hemispheres independently.
Asymmetry = |L-R|/(L+R). LRDA should be more asymmetric than GRDA.
</p></body></html>"""

    out = RESULTS_DIR.parent / 'v4_lateralization_leaderboard.html'
    with open(str(out), 'w') as f:
        f.write(html)
