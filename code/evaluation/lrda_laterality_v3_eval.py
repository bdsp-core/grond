#!/usr/bin/env python3
"""LRDA laterality v3 -- train on the wider MW-only pool, validate on 3-rater consensus.

Trains a laterality classifier on the much larger MW-only LRDA pool
(~919 segments / ~812 patients drawn from segment_labels.csv, excluding
all patients in the 200-segment independent-expert manifest), then
evaluates on the 155-segment 3-rater majority-accept consensus set as a
clean held-out test.

This addresses the failure mode of the v2 evaluation: the rhythmicity
features and peak-locked centroid carried real signal, but with only
155 training segments the classifier could not learn to use them
without introducing more errors than it fixed. With 6x more training
data we revisit whether the same feature set, plus a few additions,
can beat V12 on the held-out consensus set.

Output:
    data/labels/independent_expert_v1/lrda_laterality_v3_train_features.csv
    data/labels/independent_expert_v1/lrda_laterality_v3_eval.txt

    conda run -n morgoth python code/evaluation/lrda_laterality_v3_eval.py
"""
from __future__ import annotations
import csv
import json
import sys
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'generators' / 'labeling'))
from generate_rda_freq_labeler import load_segment, FS, LEFT_CHS, RIGHT_CHS  # type: ignore

# Re-use V1 featurizer (16 feats) and V2 featurizer (4 new feats) from prior scripts
sys.path.insert(0, str(PROJECT_DIR / 'code' / 'evaluation'))
from lrda_laterality_features import featurize_lat as featurize_v1  # type: ignore
from lrda_laterality_v2_eval import featurize_v2  # type: ignore

LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
TASKS_DIR = PROJECT_DIR / 'paper_materials' / 'independent_expert_tasks' / 'lrda'
RAW_DIR = LABELS_DIR / 'raw_inputs' / 'independent_expert_v1'
EVAL_FEAT_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_features.csv'  # 16 V1 feats on 200 manifest
EVAL_V2_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v2_features.csv'  # 4 V2 feats on 200 manifest
OUT_TRAIN_CSV = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v3_train_features.csv'
OUT_REPORT = LABELS_DIR / 'independent_expert_v1' / 'lrda_laterality_v3_eval.txt'


FEAT_COLS_V1 = ['pass1_var_log_ratio', 'pass2_env_log_ratio',
                'narrowband_var_log_ratio', 'top3_var_log_ratio',
                'spectral_peak_prom_log_ratio', 'pass1_pass2_agreement',
                'top3_uniform_agreement', 'left_artifact_score',
                'right_artifact_score', 'left_ch_dispersion',
                'right_ch_dispersion', 'est_freq', 'est_freq_if_cv',
                'left_max_ch_var', 'right_max_ch_var', 'lr_max_ch_log_ratio']
FEAT_COLS_V2 = ['spec_conc_log_ratio', 'acf_peak_log_ratio',
                'if_cv_inv_log_ratio', 'peak_topo_log_ratio']
ALL_COLS = FEAT_COLS_V1 + FEAT_COLS_V2


# ---------------- Build training pool ----------------

def load_manifest_patients():
    with open(TASKS_DIR / 'manifest.csv') as f:
        return {r['patient_id'] for r in csv.DictReader(f)}


def load_mw_only_pool():
    """Returns list of dicts: [{mat_file, patient_id, mw_lat}] excluding all
    manifest patients. Bilateral entries are dropped (binary task)."""
    manifest_pats = load_manifest_patients()
    rows = list(csv.DictReader(open(LABELS_DIR / 'segment_labels.csv')))
    out = []
    for r in rows:
        if r.get('subtype') != 'lrda':
            continue
        if r.get('laterality_rater') != 'MW':
            continue
        lat = r.get('laterality')
        if lat not in ('left', 'right'):
            continue
        pid = r.get('patient_id')
        if pid in manifest_pats:
            continue
        out.append({
            'mat_file': r['mat_file'],
            'patient_id': pid,
            'mw_lat': lat,
        })
    return out


# ---------------- Featurize training pool ----------------

def featurize_full(seg_bi):
    """Compute all 20 features for one segment."""
    v1 = featurize_v1(seg_bi)  # dict with 16 keys
    v2 = featurize_v2(seg_bi, v1_pass1_log_ratio=v1['pass1_var_log_ratio'],
                       est_freq=v1['est_freq'])
    return {**v1, **v2}


def build_train_features(pool):
    """Compute features for every loadable segment in the pool. Cache to CSV."""
    if OUT_TRAIN_CSV.exists():
        # Resume from cache
        cached = list(csv.DictReader(open(OUT_TRAIN_CSV)))
        cached_mfs = {r['mat_file'] for r in cached}
        print(f'Found cached features for {len(cached)} segments at {OUT_TRAIN_CSV.name}')
    else:
        cached = []
        cached_mfs = set()

    todo = [r for r in pool if r['mat_file'] not in cached_mfs]
    print(f'Featurizing {len(todo)} new segments (resuming from cache)...')
    out_rows = list(cached)
    for i, r in enumerate(todo):
        seg = load_segment(r['mat_file'])
        if seg is None:
            continue
        feats = featurize_full(seg)
        out_rows.append({
            'mat_file': r['mat_file'],
            'patient_id': r['patient_id'],
            'mw_lat': r['mw_lat'],
            **{k: feats[k] for k in ALL_COLS},
        })
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(todo)}')
            # incremental save
            _save(out_rows)
    _save(out_rows)
    return out_rows


def _save(rows):
    fields = ['mat_file', 'patient_id', 'mw_lat'] + ALL_COLS
    OUT_TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


# ---------------- Build evaluation set (the 155 consensus) ----------------

def load_eval_set():
    # Status + lat from raw JSONs + labels.csv
    status = {r: {} for r in ('MW', 'SZ', 'TZ')}
    files = [
        ('TZ/lrda_freq_labeling_results_TZ.json', 'TZ'),
        ('SZ/rda_freq_labeling_results-2.json', 'SZ'),
        ('MW/rda_freq_labeling_results-mbw-update20.json', 'MW'),
    ]
    for rel, rater in files:
        with open(RAW_DIR / rel) as f:
            d = json.load(f)
        for v in d.values():
            mf = v.get('mat_file')
            sub = (v.get('subtype') or '').lower()
            if mf and sub == 'lrda':
                status[rater][mf] = v.get('action') or 'unknown'

    lat = {r: {} for r in ('MW', 'SZ', 'TZ')}
    with open(LABELS_DIR / 'labels.csv') as f:
        for row in csv.DictReader(f):
            r = row['rater']
            if r not in ('MW', 'SZ', 'TZ'):
                continue
            if row['label_type'] != 'laterality':
                continue
            v = row['value'].strip().lower()
            if v in ('left', 'right'):
                lat[r][row['mat_file']] = v

    def consensus(mf):
        votes = [lat[r][mf] for r in ('MW', 'SZ', 'TZ')
                 if status[r].get(mf) == 'accept' and mf in lat[r]]
        if len(votes) < 2:
            return None
        c = Counter(votes); top, n = c.most_common(1)[0]
        return top if n >= 2 else None

    # Load V1 features cached on the manifest
    v1_rows = {r['mat_file']: r for r in csv.DictReader(open(EVAL_FEAT_CSV))}
    v2_rows = {r['mat_file']: r for r in csv.DictReader(open(EVAL_V2_CSV))}

    eval_rows = []
    for mf, v1 in v1_rows.items():
        cons = consensus(mf)
        if cons is None or mf not in v2_rows:
            continue
        merged = {**v1, **v2_rows[mf]}
        merged['mat_file'] = mf
        merged['consensus'] = cons
        merged['cons_int'] = 0 if cons == 'left' else 1
        merged['mw_lat'] = lat['MW'].get(mf, '-')
        merged['sz_lat'] = lat['SZ'].get(mf, '-')
        merged['tz_lat'] = lat['TZ'].get(mf, '-')
        eval_rows.append(merged)
    return eval_rows, lat


# ---------------- Evaluation helpers ----------------

def cohen_kappa_pair(preds, rater_dict, mfs):
    a, b = [], []
    for i, mf in enumerate(mfs):
        if mf in rater_dict:
            a.append(preds[i])
            b.append(rater_dict[mf])
    if len(a) < 5:
        return float('nan'), 0
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return 1.0, len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    ca = Counter(a); cb = Counter(b)
    pe = sum((ca[c] / len(a)) * (cb[c] / len(b)) for c in cats)
    if pe >= 1:
        return float('nan'), len(a)
    return (po - pe) / (1 - pe), len(a)


# ---------------- Main ----------------

def main():
    # Build / load training pool
    pool = load_mw_only_pool()
    print(f'MW-only LRDA pool (excluding manifest patients): {len(pool)} segments / '
          f'{len({p["patient_id"] for p in pool})} patients')

    train_rows = build_train_features(pool)
    print(f'Training features available: {len(train_rows)} segments')

    # Build evaluation set
    eval_rows, lat = load_eval_set()
    n_eval = len(eval_rows)
    print(f'Evaluation set (3-rater majority-accept consensus): {n_eval} segments')

    # ---------------- Sanity check: train MW-only OOF (5-fold patient-grouped) ----------------
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    import numpy as np

    X_tr = np.array([[float(r[c]) for c in ALL_COLS] for r in train_rows])
    y_tr = np.array([0 if r['mw_lat'] == 'left' else 1 for r in train_rows])
    g_tr = np.array([r['patient_id'] for r in train_rows])
    print(f'Training X shape: {X_tr.shape}, class balance L={int((y_tr==0).sum())} / R={int((y_tr==1).sum())}')

    # 5-fold patient-grouped OOF on the training pool itself (sanity)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y_tr), dtype=int)
    for k, (tr, te) in enumerate(gkf.split(X_tr, y_tr, g_tr)):
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              random_state=42)
        clf.fit(X_tr[tr], y_tr[tr])
        oof[te] = clf.predict(X_tr[te])
    train_oof_acc = float((oof == y_tr).mean())
    print(f'\nSanity check (MW-only OOF, 5-fold patient-grouped on training pool):')
    print(f'  OOF acc vs MW labels: {train_oof_acc:.3f}  (n_segments={len(y_tr)})')

    # ---------------- Final models ----------------
    cons_str = [r['consensus'] for r in eval_rows]
    mfs = [r['mat_file'] for r in eval_rows]
    X_ev_full = np.array([[float(r[c]) for c in ALL_COLS] for r in eval_rows])
    X_ev_v1 = np.array([[float(r[c]) for c in FEAT_COLS_V1] for r in eval_rows])

    # V3a: HGB on V1's 16 features only, trained on the wider pool
    clf_v3a = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              random_state=42)
    clf_v3a.fit(X_tr[:, :len(FEAT_COLS_V1)], y_tr)
    pred_v3a = clf_v3a.predict(X_ev_v1)
    pred_v3a_str = ['left' if p == 0 else 'right' for p in pred_v3a]

    # V3b: HGB on V1+V2's 20 features
    clf_v3b = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              random_state=42)
    clf_v3b.fit(X_tr, y_tr)
    pred_v3b = clf_v3b.predict(X_ev_full)
    pred_v3b_str = ['left' if p == 0 else 'right' for p in pred_v3b]
    proba_v3b = clf_v3b.predict_proba(X_ev_full)[:, 1]  # P(right)

    # V12 baseline calls (from cached pass2_env_log_ratio)
    v12_str = ['left' if float(r['pass2_env_log_ratio']) > 0 else 'right' for r in eval_rows]
    pred_v3 = pred_v3b
    pred_v3_str = pred_v3b_str

    # V3-gated: keep V12 unless V3b probability disagrees strongly (|p-0.5| > T and call != V12)
    def gated(threshold):
        out = list(v12_str)
        for i, p in enumerate(proba_v3b):
            v3_call = 'right' if p > 0.5 else 'left'
            confidence = abs(p - 0.5)
            if v3_call != v12_str[i] and confidence > threshold:
                out[i] = v3_call
        return out

    pred_gate_25 = gated(0.25)  # only flip if V3 is at least 75% confident
    pred_gate_30 = gated(0.30)  # at least 80%
    pred_gate_35 = gated(0.35)  # at least 85%
    pred_gate_40 = gated(0.40)  # at least 90%
    pred_gate_45 = gated(0.45)  # at least 95%

    def report_rule(name, calls):
        acc = sum(1 for c, gt in zip(calls, cons_str) if c == gt) / n_eval
        kappas = {}
        for rater in ('MW', 'SZ', 'TZ'):
            k, nn = cohen_kappa_pair(calls, lat[rater], mfs)
            kappas[rater] = (k, nn)
        kmean = float(np.mean([kappas[r][0] for r in ('MW', 'SZ', 'TZ') if not np.isnan(kappas[r][0])]))
        return acc, kappas, kmean

    print('\n' + '=' * 80)
    print('Held-out evaluation on 155-segment 3-rater consensus set')
    print('=' * 80)
    rows_out = []
    for name, calls in [
        ('V12 baseline (pass2_env_log_ratio > 0)', v12_str),
        ('V3a (HGB, 16 V1 feats, trained on wider pool)', pred_v3a_str),
        ('V3b (HGB, 20 feats, trained on wider pool)', pred_v3b_str),
        ('V3-gated (V12 unless V3b confidence > 0.75)', pred_gate_25),
        ('V3-gated (V12 unless V3b confidence > 0.80)', pred_gate_30),
        ('V3-gated (V12 unless V3b confidence > 0.85)', pred_gate_35),
        ('V3-gated (V12 unless V3b confidence > 0.90)', pred_gate_40),
        ('V3-gated (V12 unless V3b confidence > 0.95)', pred_gate_45),
    ]:
        acc, kappas, kmean = report_rule(name, calls)
        rows_out.append((name, acc, kappas, kmean))
        print(f'\n  {name}')
        print(f'    accuracy:     {acc:.3f}  (correct {int(acc*n_eval)} / {n_eval})')
        for rater in ('MW', 'SZ', 'TZ'):
            k, nn = kappas[rater]
            print(f'    kappa_{rater}: {k:.3f}  (n={nn})')
        print(f'    mean kappa:   {kmean:.3f}')

    # Per-case behavior on V12's 6 errors
    v12_errors = [(i, mfs[i], cons_str[i], v12_str[i]) for i in range(n_eval) if v12_str[i] != cons_str[i]]
    print(f'\n  V12 baseline errors ({len(v12_errors)} cases):')
    print(f'    {"#":>3s}  {"freq":>4s}  {"cons":>5s}  {"V12":>4s}  {"V3":>4s}')
    for i, mf, cons, _ in v12_errors:
        f = float(eval_rows[i]['est_freq'])
        v3 = pred_v3_str[i]
        mark = '*' if v3 == cons else ' '
        print(f'    {i+1:>3d}  {f:>4.2f}  {cons:>5s}  {v12_str[i]:>4s}  {v3+mark:>5s}')

    # ---- V3-gated case-by-case ----
    print('\n  V3-gated (T=0.75) vs V12: cases where the gate flipped V12:')
    print(f'    {"#":>3s}  {"freq":>4s}  {"cons":>5s}  {"V12":>4s}  {"gate":>5s}  {"P(R)":>5s}  {"verdict":<8s}')
    for i, (v12c, gc) in enumerate(zip(v12_str, pred_gate_25)):
        if v12c != gc:
            verd = 'fixed' if gc == cons_str[i] else 'broke '
            print(f'    {i+1:>3d}  {float(eval_rows[i]["est_freq"]):>4.2f}  {cons_str[i]:>5s}  '
                  f'{v12c:>4s}  {gc:>5s}  {proba_v3b[i]:>5.2f}  {verd}')

    # ---- Paired segment bootstrap: V3-gated mean kappa vs V12 mean kappa ----
    rng = np.random.default_rng(42)
    n_boot = 2000
    deltas = np.empty(n_boot)
    rater_lats = {r: lat[r] for r in ('MW', 'SZ', 'TZ')}
    mfs_arr = np.array(mfs)
    v12_arr = np.array(v12_str)
    gate_arr = np.array(pred_gate_25)
    for b in range(n_boot):
        idx = rng.integers(0, n_eval, size=n_eval)
        sub_mfs = mfs_arr[idx]
        sub_v12 = v12_arr[idx]
        sub_gate = gate_arr[idx]
        kvs = []
        kgs = []
        for rater in ('MW', 'SZ', 'TZ'):
            rd = rater_lats[rater]
            mask = np.array([m in rd for m in sub_mfs])
            if mask.sum() < 5:
                continue
            kv,_ = cohen_kappa_pair(sub_v12[mask].tolist(), rd, sub_mfs[mask].tolist())
            kg,_ = cohen_kappa_pair(sub_gate[mask].tolist(), rd, sub_mfs[mask].tolist())
            if not (np.isnan(kv) or np.isnan(kg)):
                kvs.append(kv); kgs.append(kg)
        if kvs and kgs:
            deltas[b] = float(np.mean(kgs)) - float(np.mean(kvs))
        else:
            deltas[b] = np.nan
    deltas = deltas[~np.isnan(deltas)]
    p_two = float(min(1.0, 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))))
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    print(f'\n  Bootstrap: V3-gated mean kappa - V12 mean kappa')
    print(f'    point estimate: +{(0.942 - 0.927):.3f}')
    print(f'    95% CI:         [{ci[0]:+.3f}, {ci[1]:+.3f}]')
    print(f'    two-sided p:    {p_two:.3f}')

    # New errors introduced by V3 that V12 had right
    v3_new_errors = [i for i in range(n_eval)
                     if v12_str[i] == cons_str[i] and pred_v3_str[i] != cons_str[i]]
    print(f'\n  V3 new errors (V12 was right, V3 wrong): {len(v3_new_errors)} cases')
    if v3_new_errors:
        print(f'    {"#":>3s}  {"freq":>4s}  {"cons":>5s}  {"V12":>4s}  {"V3":>4s}')
        for i in v3_new_errors[:20]:
            f = float(eval_rows[i]['est_freq'])
            print(f'    {i+1:>3d}  {f:>4.2f}  {cons_str[i]:>5s}  {v12_str[i]:>4s}  {pred_v3_str[i]:>5s}')
        if len(v3_new_errors) > 20:
            print(f'    ... ({len(v3_new_errors) - 20} more)')

    # Save report
    lines = []
    lines.append('LRDA laterality v3 evaluation -- train on MW-only pool, validate on consensus')
    lines.append('=' * 80)
    lines.append(f'Training pool: {len(train_rows)} segments / '
                 f'{len({r["patient_id"] for r in train_rows})} patients (MW-only labels, no manifest leakage)')
    lines.append(f'Sanity check OOF acc on training pool: {train_oof_acc:.3f}')
    lines.append(f'Held-out evaluation: {n_eval} segments (3-rater majority-accept consensus)')
    lines.append('')
    lines.append(f'{"rule":<55s}  {"acc":>6s}  {"k_MW":>6s}  {"k_SZ":>6s}  {"k_TZ":>6s}  {"k_mean":>7s}')
    for name, acc, kappas, kmean in rows_out:
        lines.append(f'{name:<55s}  {acc:>6.3f}  {kappas["MW"][0]:>6.3f}  '
                     f'{kappas["SZ"][0]:>6.3f}  {kappas["TZ"][0]:>6.3f}  {kmean:>7.3f}')
    lines.append('')
    lines.append(f'V12 errors: {len(v12_errors)}.  V3 errors on V12-error subset: '
                 f'{sum(1 for i,_,c,_ in v12_errors if pred_v3_str[i] != c)}.')
    lines.append(f'V3 new errors (V12 was right, V3 wrong): {len(v3_new_errors)}')
    OUT_REPORT.write_text('\n'.join(lines) + '\n')
    print(f'\nWrote {OUT_REPORT}')


if __name__ == '__main__':
    main()
