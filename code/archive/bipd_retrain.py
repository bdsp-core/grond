"""
BIPD Retrain — Retrain with MW's corrected labels, evaluate on all available GPDs.

Changes from v1:
  - Exclude cases MW relabeled as LPD/REJECT from GPD training pool
  - Add MW's newly identified BIPDs to positive examples
  - Report synthetic CV performance in detail
  - Evaluate on all 185 cached GPD detections + 14 confirmed BIPDs

Usage:
    conda run -n foe_dl python code/bipd_retrain.py
"""

import sys
import json
import time
import warnings
import numpy as np
from pathlib import Path
from collections import Counter

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from bipd_detector import (
    compute_bipd_features, compute_empirical_jitter,
    generate_synthetic_data, FEATURE_NAMES, CACHE_DIR, RESULTS_DIR,
    DATA_DIR,
)

# ── Load data ─────────────────────────────────────────────────────────

print("=" * 70)
print("  BIPD Retrain with Corrected Labels")
print("=" * 70)

# Load cached detections
with open(CACHE_DIR / 'gpd_hemi_detections.json') as f:
    gpd_det = json.load(f)
with open(CACHE_DIR / 'lpd_hemi_detections.json') as f:
    lpd_det = json.load(f)
with open(CACHE_DIR / 'bipd_hemi_detections.json') as f:
    bipd_det = json.load(f)

# Load MW's corrected labels
review_path = DATA_DIR / 'labels' / 'bipd_review_labels_mw.json'
with open(review_path) as f:
    mw_labels = json.load(f)

print(f"\nCached detections: {len(gpd_det)} GPD, {len(lpd_det)} LPD, {len(bipd_det)} BIPD")
print(f"MW reviewed: {len(mw_labels)} cases")
mw_counts = Counter(v['label'] for v in mw_labels.values())
print(f"  Labels: {dict(mw_counts)}")

# ── Clean the GPD pool ────────────────────────────────────────────────
# Remove cases MW says are LPD, REJECT, or BIPD from the GPD pool

contaminated = set()
for pid, v in mw_labels.items():
    if v['label'] in ('LPD', 'REJECT', 'BIPD') and v['original_label'] == 'GPD':
        contaminated.add(pid)

gpd_det_clean = {k: v for k, v in gpd_det.items() if k not in contaminated}
print(f"\nGPD pool: {len(gpd_det)} → {len(gpd_det_clean)} after removing {len(contaminated)} contaminated")
print(f"  Removed: {len([p for p in contaminated if mw_labels[p]['label']=='LPD'])} LPD, "
      f"{len([p for p in contaminated if mw_labels[p]['label']=='REJECT'])} REJECT, "
      f"{len([p for p in contaminated if mw_labels[p]['label']=='BIPD'])} BIPD")

# ── Build confirmed BIPD set from MW labels ───────────────────────────

confirmed_bipd_pids = set()
for pid, v in mw_labels.items():
    if v['label'] == 'BIPD':
        confirmed_bipd_pids.add(pid)
print(f"\nConfirmed BIPDs: {len(confirmed_bipd_pids)}")

# Merge BIPD detections: from bipd_det cache + any GPD cases relabeled as BIPD
all_bipd_det = {}
for pid in confirmed_bipd_pids:
    if pid in bipd_det:
        all_bipd_det[pid] = bipd_det[pid]
    elif pid in gpd_det:
        all_bipd_det[pid] = gpd_det[pid]
print(f"BIPDs with detections: {len(all_bipd_det)}")

# ── Compute empirical jitter from CLEAN GPDs ──────────────────────────

print("\nComputing empirical jitter from clean GPD cases...")
jitter = compute_empirical_jitter(gpd_det_clean)
print(f"  Empirical jitter: {jitter*1000:.1f} ms")

# ── Generate synthetic data with clean pools ──────────────────────────

print("\nGenerating synthetic training data (clean GPD pool)...")
X_synth, y_synth, synth_labels = generate_synthetic_data(
    gpd_det_clean, lpd_det, jitter)
print(f"  Feature matrix: {X_synth.shape}")
print(f"  Class balance: {np.sum(y_synth==0)} GPD, {np.sum(y_synth==1)} BIPD")

# ── Synthetic CV ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("  SYNTHETIC 5-FOLD CV")
print("=" * 70)

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = []
cv_preds_all = np.zeros(len(y_synth))
cv_probs_all = np.zeros(len(y_synth))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_synth, y_synth)):
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42)
    model.fit(X_synth[train_idx], y_synth[train_idx])
    probs = model.predict_proba(X_synth[val_idx])[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_synth[val_idx], probs)
    cv_aucs.append(auc)
    cv_probs_all[val_idx] = probs
    cv_preds_all[val_idx] = preds

    tp = np.sum((preds == 1) & (y_synth[val_idx] == 1))
    fn = np.sum((preds == 0) & (y_synth[val_idx] == 1))
    fp = np.sum((preds == 1) & (y_synth[val_idx] == 0))
    tn = np.sum((preds == 0) & (y_synth[val_idx] == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"  Fold {fold}: AUC={auc:.4f}  Sens={sens:.3f}  Spec={spec:.3f}  "
          f"TP={tp} FN={fn} FP={fp} TN={tn}")

print(f"\n  Mean CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

# Overall synthetic performance
synth_preds = (cv_probs_all >= 0.5).astype(int)
cm = confusion_matrix(y_synth, synth_preds)
print(f"\n  Synthetic confusion matrix (rows=true, cols=pred):")
print(f"              Pred GPD  Pred BIPD")
print(f"  True GPD    {cm[0,0]:>7d}  {cm[0,1]:>9d}")
print(f"  True BIPD   {cm[1,0]:>7d}  {cm[1,1]:>9d}")

overall_sens = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
overall_spec = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
print(f"\n  Overall synthetic: Sens={overall_sens:.3f}  Spec={overall_spec:.3f}")

# ── Train final model on ALL synthetic data ───────────────────────────

print("\n" + "=" * 70)
print("  TRAINING FINAL MODEL")
print("=" * 70)

final_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42)
final_model.fit(X_synth, y_synth)

# Feature importances
importances = final_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("\n  Feature importances:")
for i in sorted_idx[:10]:
    print(f"    {FEATURE_NAMES[i]:25s}: {importances[i]:.4f}")

# ── Evaluate on ALL available real cases ──────────────────────────────

print("\n" + "=" * 70)
print("  EVALUATION ON REAL DATA")
print("=" * 70)

# Build feature matrix for all GPD + BIPD cases with detections
all_det = {}
all_det.update(gpd_det)  # ALL 185 GPDs (including contaminated — we predict on all)
all_det.update(all_bipd_det)

# Ground truth: use MW labels where available, otherwise original
gt_labels = {}
for pid in gpd_det:
    if pid in mw_labels:
        gt_labels[pid] = mw_labels[pid]['label']
    else:
        gt_labels[pid] = 'GPD'  # unreviewed GPDs stay as GPD

for pid in all_bipd_det:
    if pid in mw_labels:
        gt_labels[pid] = mw_labels[pid]['label']
    else:
        gt_labels[pid] = 'BIPD'  # unreviewed BIPDs stay as BIPD

# Compute features and predict
results_by_label = {}
all_predictions = {}

for pid, det in all_det.items():
    tL = det['left']['times']
    tR = det['right']['times']
    fL = det['left']['freq']
    fR = det['right']['freq']
    if len(tL) < 1 or len(tR) < 1:
        continue

    feats = compute_bipd_features(tL, tR, fL, fR)
    row = [feats.get(fn, 0.0) for fn in FEATURE_NAMES]
    if any(not np.isfinite(v) for v in row):
        continue

    X = np.array([row])
    prob = float(final_model.predict_proba(X)[0, 1])
    pred = 'BIPD' if prob >= 0.5 else 'GPD'
    gt = gt_labels.get(pid, 'GPD')

    all_predictions[pid] = {
        'prob': prob, 'pred': pred, 'gt': gt,
    }

    if gt not in results_by_label:
        results_by_label[gt] = {'correct': 0, 'total': 0, 'probs': []}
    results_by_label[gt]['total'] += 1
    results_by_label[gt]['probs'].append(prob)
    # "Correct" for BIPD means pred=BIPD, for GPD means pred=GPD
    if (gt == 'BIPD' and pred == 'BIPD') or (gt != 'BIPD' and pred == 'GPD'):
        results_by_label[gt]['correct'] += 1

print(f"\nTotal cases evaluated: {len(all_predictions)}")
for label in sorted(results_by_label.keys()):
    r = results_by_label[label]
    acc = r['correct'] / r['total'] if r['total'] > 0 else 0
    mean_prob = np.mean(r['probs'])
    print(f"  {label:8s}: {r['correct']}/{r['total']} correct ({acc:.1%}), "
          f"mean BIPD prob={mean_prob:.3f}")

# AUC: BIPD vs everything else
y_true_real = []
y_prob_real = []
for pid, p in all_predictions.items():
    if p['gt'] in ('BIPD', 'GPD'):
        y_true_real.append(1 if p['gt'] == 'BIPD' else 0)
        y_prob_real.append(p['prob'])

if len(set(y_true_real)) >= 2:
    auc_real = roc_auc_score(y_true_real, y_prob_real)
    print(f"\n  AUC (BIPD vs GPD on real data): {auc_real:.3f}")

# Per-case BIPD results
print(f"\n  Confirmed BIPDs:")
for pid in sorted(confirmed_bipd_pids):
    if pid in all_predictions:
        p = all_predictions[pid]
        status = 'HIT' if p['pred'] == 'BIPD' else 'MISS'
        print(f"    {pid}: prob={p['prob']:.3f} → {p['pred']} [{status}]")

# False positives on unreviewed GPDs
unreviewed_fp = [(pid, p) for pid, p in all_predictions.items()
                 if p['gt'] == 'GPD' and p['pred'] == 'BIPD'
                 and pid not in mw_labels]
if unreviewed_fp:
    print(f"\n  Unreviewed GPDs predicted as BIPD ({len(unreviewed_fp)}):")
    for pid, p in sorted(unreviewed_fp, key=lambda x: -x[1]['prob'])[:20]:
        print(f"    {pid}: prob={p['prob']:.3f}")

# Save results
results_out = {
    'synthetic_cv_auc_mean': round(np.mean(cv_aucs), 4),
    'synthetic_cv_auc_std': round(np.std(cv_aucs), 4),
    'synthetic_sens': round(overall_sens, 4),
    'synthetic_spec': round(overall_spec, 4),
    'real_auc': round(auc_real, 4) if len(set(y_true_real)) >= 2 else None,
    'n_confirmed_bipd': len(confirmed_bipd_pids),
    'predictions': all_predictions,
}
out_path = RESULTS_DIR / 'bipd_retrain_results.json'
with open(out_path, 'w') as f:
    json.dump(results_out, f, indent=2)
print(f"\n  Results saved to {out_path}")
print("=" * 70)
