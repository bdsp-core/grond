#!/usr/bin/env python3
"""
R11: Comprehensive Bootstrap Statistical Analysis
==================================================
- Expert-expert pairwise Spearman with bootstrap 95% CIs
- Algorithm vs consensus Spearman with bootstrap 95% CIs
- Statistical comparison of algorithm-consensus vs expert-expert
- Power analysis for sample size requirements

All analyses at patient level (1 value per patient).
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
ANNOT_DIR = DATA_DIR / '_archive' / 'annotations'
RESULTS_DIR = PROJECT_DIR / 'results'

N_BOOTSTRAP = 10_000

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load canonical dataset
# ═══════════════════════════════════════════════════════════════════════════════
labels = pd.read_csv(DATA_DIR / '_archive' / 'canonical_dataset' / 'labels.csv')
labels = labels[labels['excluded'] == False].copy()
print(f"Canonical dataset: {len(labels)} usable patients")
print(f"  LPD: {(labels.subtype=='lpd').sum()}, GPD: {(labels.subtype=='gpd').sum()}")

original = labels[labels['source'] == 'original_dataset'].copy()
print(f"\nOriginal dataset (3-expert): {len(original)} patients")
print(f"  LPD: {(original.subtype=='lpd').sum()}, GPD: {(original.subtype=='gpd').sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Load annotation CSVs and compute patient-level expert ratings
# ═══════════════════════════════════════════════════════════════════════════════
def extract_patient_id(filepath):
    """Extract patient ID from annotation file path."""
    basename = os.path.basename(str(filepath))
    m = re.match(r'((?:pat|abn|emu)\d+)_', basename)
    if m:
        return m.group(1)
    return None


def load_expert_annotations(pattern_type, expert_initial):
    """Load annotation CSV, return DataFrame with patient_id and frequency."""
    prefix = f'{pattern_type}_{expert_initial}_'
    matches = [f for f in os.listdir(ANNOT_DIR) if f.startswith(prefix)]
    assert len(matches) == 1, f"Expected 1 file for {prefix}, got {matches}"
    df = pd.read_csv(ANNOT_DIR / matches[0])
    df['patient_id'] = df['files'].apply(extract_patient_id)
    df['freq'] = pd.to_numeric(df['frequency'], errors='coerce')
    return df[['patient_id', 'freq']].dropna(subset=['patient_id'])


def get_patient_level_expert_data(pattern_type, expert_initial):
    """Average segments per patient, return {patient_id: mean_freq}."""
    df = load_expert_annotations(pattern_type, expert_initial)
    return df.groupby('patient_id')['freq'].mean()


def bootstrap_spearman(x, y, n_boot=N_BOOTSTRAP):
    """Compute Spearman correlation with bootstrap 95% CI."""
    x, y = np.array(x), np.array(y)
    n = len(x)
    observed = stats.spearmanr(x, y)[0]
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        r, _ = stats.spearmanr(x[idx], y[idx])
        boot_rs[i] = r
    ci_lo = np.percentile(boot_rs, 2.5)
    ci_hi = np.percentile(boot_rs, 97.5)
    return observed, ci_lo, ci_hi, boot_rs


def bootstrap_difference_test(boot_a, boot_b):
    """Test if two bootstrap distributions differ (proportion of boot_a > boot_b)."""
    diff = boot_a - boot_b
    p_greater = np.mean(diff > 0)
    p_less = np.mean(diff < 0)
    # Two-sided p-value
    p_value = 2 * min(p_greater, p_less)
    return np.mean(diff), np.percentile(diff, 2.5), np.percentile(diff, 97.5), p_value


# ═══════════════════════════════════════════════════════════════════════════════
# 2a. Expert-expert pairwise Spearman (original 43 patients, patient-level)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 2: EXPERT-EXPERT PAIRWISE SPEARMAN CORRELATIONS (Patient-Level)")
print("="*80)

experts = ['LB', 'PH', 'SZ']
pairs = [('LB', 'PH'), ('LB', 'SZ'), ('PH', 'SZ')]
pattern_types = {'lpd': 'LPDS', 'gpd': 'GPDS'}

# Store results for later comparison
expert_boot_distributions = {}
expert_results = []

for subtype, pattern in pattern_types.items():
    print(f"\n--- {subtype.upper()} ---")

    # Get patient-level data for each expert
    expert_data = {}
    for e in experts:
        expert_data[e] = get_patient_level_expert_data(pattern, e)

    pair_boots = []
    for e1, e2 in pairs:
        # Get common patients
        common = expert_data[e1].index.intersection(expert_data[e2].index)
        v1 = expert_data[e1].loc[common].values
        v2 = expert_data[e2].loc[common].values

        # Filter to patients where BOTH experts rated > 0 (patient-level mean > 0)
        mask = (v1 > 0) & (v2 > 0)
        v1_filt = v1[mask]
        v2_filt = v2[mask]

        r, ci_lo, ci_hi, boot_rs = bootstrap_spearman(v1_filt, v2_filt)
        pair_boots.append(boot_rs)
        expert_results.append({
            'subtype': subtype, 'pair': f'{e1}-{e2}',
            'n': len(v1_filt), 'spearman': r, 'ci_lo': ci_lo, 'ci_hi': ci_hi
        })
        print(f"  {e1}-{e2}: rs={r:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (n={len(v1_filt)} patients)")

    # Mean expert-expert Spearman (average of bootstrap distributions)
    mean_boot = np.mean(pair_boots, axis=0)
    mean_r = np.mean([res['spearman'] for res in expert_results if res['subtype'] == subtype])
    mean_ci_lo = np.percentile(mean_boot, 2.5)
    mean_ci_hi = np.percentile(mean_boot, 97.5)
    expert_boot_distributions[subtype] = mean_boot

    expert_results.append({
        'subtype': subtype, 'pair': 'MEAN',
        'n': '-', 'spearman': mean_r, 'ci_lo': mean_ci_lo, 'ci_hi': mean_ci_hi
    })
    print(f"  MEAN:   rs={mean_r:.4f} [{mean_ci_lo:.4f}, {mean_ci_hi:.4f}]")

# Also compute combined (all subtypes pooled)
print(f"\n--- COMBINED (LPD+GPD, patient-level) ---")
all_pair_boots = []
for e1, e2 in pairs:
    all_v1, all_v2 = [], []
    for subtype, pattern in pattern_types.items():
        expert_data_e1 = get_patient_level_expert_data(pattern, e1)
        expert_data_e2 = get_patient_level_expert_data(pattern, e2)
        common = expert_data_e1.index.intersection(expert_data_e2.index)
        v1 = expert_data_e1.loc[common].values
        v2 = expert_data_e2.loc[common].values
        mask = (v1 > 0) & (v2 > 0)
        all_v1.extend(v1[mask])
        all_v2.extend(v2[mask])
    all_v1 = np.array(all_v1)
    all_v2 = np.array(all_v2)
    r, ci_lo, ci_hi, boot_rs = bootstrap_spearman(all_v1, all_v2)
    all_pair_boots.append(boot_rs)
    print(f"  {e1}-{e2}: rs={r:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (n={len(all_v1)})")

combined_mean_boot = np.mean(all_pair_boots, axis=0)
combined_mean_r = np.mean([stats.spearmanr(np.array(a), np.array(b))[0] for a, b in [(all_v1, all_v2)]])
# Recalculate properly
combined_rs = []
for boot in all_pair_boots:
    combined_rs.append(np.mean(boot))
combined_mean_r_v2 = np.mean(combined_rs)
combined_ci_lo = np.percentile(combined_mean_boot, 2.5)
combined_ci_hi = np.percentile(combined_mean_boot, 97.5)
expert_boot_distributions['combined'] = combined_mean_boot

# Actually compute combined mean properly from pair means
combined_pair_rs = []
for e1, e2 in pairs:
    all_v1, all_v2 = [], []
    for subtype, pattern in pattern_types.items():
        expert_data_e1 = get_patient_level_expert_data(pattern, e1)
        expert_data_e2 = get_patient_level_expert_data(pattern, e2)
        common = expert_data_e1.index.intersection(expert_data_e2.index)
        v1 = expert_data_e1.loc[common].values
        v2 = expert_data_e2.loc[common].values
        mask = (v1 > 0) & (v2 > 0)
        all_v1.extend(v1[mask])
        all_v2.extend(v2[mask])
    combined_pair_rs.append(stats.spearmanr(all_v1, all_v2)[0])
combined_mean_r_final = np.mean(combined_pair_rs)
print(f"  MEAN:   rs={combined_mean_r_final:.4f} [{combined_ci_lo:.4f}, {combined_ci_hi:.4f}]")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Algorithm vs Consensus (patient-level from CV predictions)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 3: ALGORITHM vs CONSENSUS (Patient-Level CV Predictions)")
print("="*80)

# Load the best algorithm predictions
cv_path = RESULTS_DIR / 'optimization_runs' / 'dl_proper_patient_cv_SP_only.json'
with open(cv_path) as f:
    cv = json.load(f)

# The predictions are segment-level from the 43 original patients (with multiple segments).
# We need to aggregate to patient level. The JSON has per-segment predictions aligned with
# the annotation CSV ordering. Let's use the labels.csv original dataset for patient mapping.

# From generate_figures.py, the pred/expert arrays map directly to the annotation CSV rows.
# The annotation CSVs have multiple segments per patient. We need to map segments -> patients.

# Strategy: load annotation CSVs, get patient IDs per segment, average per patient.
algo_results = []
algo_boot_distributions = {}

for subtype, pattern in pattern_types.items():
    prefix_key = subtype  # 'lpd' or 'gpd'
    pred = np.array(cv[f'{prefix_key}_pred_vals'])
    expert_vals = np.array(cv[f'{prefix_key}_expert_vals'])
    expert_LB = np.array(cv[f'{prefix_key}_expert_LB'])
    expert_PH = np.array(cv[f'{prefix_key}_expert_PH'])
    expert_SZ = np.array(cv[f'{prefix_key}_expert_SZ'])

    # Load corresponding annotation to get patient IDs per segment
    annot_df = load_expert_annotations(pattern, 'LB')  # LB as reference for ordering
    n_segments = len(annot_df)

    # The CV predictions should align with annotation rows
    # But lengths might differ if some filtering happened. Let's check.
    print(f"\n--- {subtype.upper()} ---")
    print(f"  Prediction array length: {len(pred)}")
    print(f"  Annotation rows (LB): {n_segments}")

    if len(pred) != n_segments:
        # The predictions might include all segments including those with freq=0.
        # The annotation CSV has all segments. Let's match by length.
        # Expert arrays in JSON have same length as predictions.
        # We need patient IDs. Use LB annotation file rows in order.
        print(f"  WARNING: Length mismatch. Using predictions directly with consensus values.")

    # Use patient IDs from annotation file (same order as expert arrays)
    patient_ids = annot_df['patient_id'].values

    if len(patient_ids) == len(pred):
        # Perfect match - aggregate by patient
        seg_df = pd.DataFrame({
            'patient_id': patient_ids,
            'pred': pred,
            'consensus': expert_vals,
            'expert_LB': expert_LB,
            'expert_PH': expert_PH,
            'expert_SZ': expert_SZ,
        })
    else:
        # Length mismatch - the predictions might correspond to a different filtering
        # Use the expert_vals which are the consensus for each segment
        # Try matching by using first n rows
        min_n = min(len(patient_ids), len(pred))
        print(f"  Using first {min_n} entries")
        seg_df = pd.DataFrame({
            'patient_id': patient_ids[:min_n],
            'pred': pred[:min_n],
            'consensus': expert_vals[:min_n],
        })

    # Patient-level: average predictions and consensus per patient
    pat_df = seg_df.groupby('patient_id').agg({
        'pred': 'mean',
        'consensus': 'mean',
    }).dropna()

    # Filter to patients where consensus > 0 (valid PDs)
    pat_df = pat_df[pat_df['consensus'] > 0]

    n_patients = len(pat_df)
    r, ci_lo, ci_hi, boot_rs = bootstrap_spearman(
        pat_df['pred'].values, pat_df['consensus'].values
    )
    algo_boot_distributions[subtype] = boot_rs
    algo_results.append({
        'subtype': subtype, 'comparison': 'algo-consensus',
        'n': n_patients, 'spearman': r, 'ci_lo': ci_lo, 'ci_hi': ci_hi
    })
    print(f"  Algorithm vs Consensus: rs={r:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (n={n_patients} patients)")

    # Also report segment-level for comparison
    seg_r = stats.spearmanr(pred, expert_vals)[0]
    print(f"  (Segment-level for reference: rs={seg_r:.4f}, n={len(pred)} segments)")

# Combined
print(f"\n--- COMBINED ---")
all_pred, all_cons = [], []
for subtype, pattern in pattern_types.items():
    prefix_key = subtype
    pred = np.array(cv[f'{prefix_key}_pred_vals'])
    expert_vals = np.array(cv[f'{prefix_key}_expert_vals'])
    annot_df = load_expert_annotations(pattern, 'LB')
    patient_ids = annot_df['patient_id'].values
    min_n = min(len(patient_ids), len(pred))
    seg_df = pd.DataFrame({
        'patient_id': patient_ids[:min_n],
        'pred': pred[:min_n],
        'consensus': expert_vals[:min_n],
    })
    pat_df = seg_df.groupby('patient_id').agg({'pred': 'mean', 'consensus': 'mean'}).dropna()
    pat_df = pat_df[pat_df['consensus'] > 0]
    all_pred.extend(pat_df['pred'].values)
    all_cons.extend(pat_df['consensus'].values)

all_pred = np.array(all_pred)
all_cons = np.array(all_cons)
r, ci_lo, ci_hi, boot_rs = bootstrap_spearman(all_pred, all_cons)
algo_boot_distributions['combined'] = boot_rs
print(f"  Algorithm vs Consensus: rs={r:.4f} [{ci_lo:.4f}, {ci_hi:.4f}] (n={len(all_pred)} patients)")


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. Statistical Comparison: Algorithm-Consensus vs Expert-Expert
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 3c: STATISTICAL COMPARISON (Algorithm vs Expert-Expert Ceiling)")
print("="*80)

for key in ['lpd', 'gpd', 'combined']:
    label = key.upper()
    if key in expert_boot_distributions and key in algo_boot_distributions:
        expert_boot = expert_boot_distributions[key]
        algo_boot = algo_boot_distributions[key]

        diff_mean, diff_ci_lo, diff_ci_hi, p_val = bootstrap_difference_test(
            algo_boot, expert_boot
        )

        algo_median = np.median(algo_boot)
        expert_median = np.median(expert_boot)

        print(f"\n  {label}:")
        print(f"    Expert-expert mean Spearman (bootstrap median): {expert_median:.4f}")
        print(f"    Algorithm-consensus Spearman (bootstrap median): {algo_median:.4f}")
        print(f"    Difference (algo - expert): {diff_mean:.4f} [{diff_ci_lo:.4f}, {diff_ci_hi:.4f}]")
        print(f"    Bootstrap p-value (two-sided): {p_val:.4f}")
        if diff_ci_hi < 0:
            print(f"    --> Algorithm is SIGNIFICANTLY WORSE than expert-expert agreement")
        elif diff_ci_lo > 0:
            print(f"    --> Algorithm is SIGNIFICANTLY BETTER than expert-expert agreement")
        else:
            print(f"    --> NO significant difference (CIs overlap zero)")

        # What fraction of expert-expert is the algorithm achieving?
        if expert_median > 0:
            pct = algo_median / expert_median * 100
            print(f"    Algorithm achieves {pct:.1f}% of expert-expert agreement")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Statistical Power Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 4: STATISTICAL POWER ANALYSIS")
print("="*80)

def power_for_spearman_diff(n, r1, r2, alpha=0.05, n_sim=5000):
    """
    Simulate power to detect difference between two Spearman correlations.

    Uses Fisher z-transform approach:
    z = arctanh(r), SE(z) ~ 1/sqrt(n-3)
    For two independent correlations: z_diff ~ N(z1-z2, sqrt(2/(n-3)))
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(2 / (n - 3))  # SE for difference of two independent z-values
    z_diff = z1 - z2

    # Under H1: z_diff is the true effect
    # Test statistic: z_diff_obs / se ~ N(z_diff/se, 1)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Power = P(|Z| > z_crit) where Z ~ N(z_diff/se, 1)
    # This is actually: Z ~ N(noncentrality, 1)
    ncp = abs(z_diff) / se
    power = 1 - stats.norm.cdf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp)
    return power


def sample_size_for_power(r1, r2, target_power=0.80, alpha=0.05):
    """Find minimum n to achieve target power for detecting r1 vs r2 difference."""
    for n in range(10, 5000):
        p = power_for_spearman_diff(n, r1, r2, alpha)
        if p >= target_power:
            return n
    return float('inf')


# Get observed correlations for power analysis
# Use the point estimates
for key in ['lpd', 'gpd', 'combined']:
    label = key.upper()

    # Get expert-expert mean
    ee_results = [r for r in expert_results if r['subtype'] == key and r['pair'] == 'MEAN']
    algo_res = [r for r in algo_results if r['subtype'] == key]

    if not ee_results and key == 'combined':
        r_expert = combined_mean_r_final
        r_algo_list = [r for r in algo_results]
        if r_algo_list:
            r_algo = np.mean([r['spearman'] for r in r_algo_list])
        else:
            continue
    elif ee_results:
        r_expert = ee_results[0]['spearman']
        r_algo = algo_res[0]['spearman'] if algo_res else None
    else:
        continue

    if r_algo is None:
        continue

    # Get current sample size
    if key == 'combined':
        current_n = len(all_pred)
    elif algo_res:
        current_n = algo_res[0]['n']
    else:
        continue

    print(f"\n  {label} (expert-expert r={r_expert:.3f}, algo-consensus r={r_algo:.3f}):")

    # Current power
    current_power = power_for_spearman_diff(current_n, r_expert, r_algo)
    print(f"    Current sample size: n={current_n}")
    print(f"    Current power to detect observed difference: {current_power:.3f}")

    # Minimum detectable difference at current n
    # Search for delta where power = 0.80
    for delta in np.arange(0.01, 1.0, 0.01):
        r_test = r_expert - delta
        if r_test < 0:
            break
        p = power_for_spearman_diff(current_n, r_expert, r_test)
        if p >= 0.80:
            print(f"    Minimum detectable difference at 80% power: delta={delta:.2f}")
            break

    # Sample sizes needed for specific deltas
    for delta in [0.05, 0.10, 0.15, 0.20]:
        r_test = r_expert - delta
        if r_test < 0:
            continue
        n_needed = sample_size_for_power(r_expert, r_test, target_power=0.80)
        print(f"    n needed to detect delta={delta:.2f} (r={r_test:.3f} vs {r_expert:.3f}): {n_needed}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary Table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("SECTION 5: SUMMARY TABLE")
print("="*80)

print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│                    COMPREHENSIVE STATISTICAL ANALYSIS                       │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print("│                                                                             │")
print("│  EXPERT-EXPERT AGREEMENT (Patient-Level, Original 43 Patients)              │")
print("│  ─────────────────────────────────────────────────────────────              │")

for r in expert_results:
    sub = r['subtype'].upper()
    pair = r['pair']
    if pair == 'MEAN':
        line = f"│  {sub:4s} Mean Expert-Expert:  rs = {r['spearman']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
    else:
        line = f"│  {sub:4s} {pair}:               rs = {r['spearman']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  n={r['n']}"
    print(f"{line:<77s}│")

print(f"│  {'COMB':4s} Mean Expert-Expert:  rs = {combined_mean_r_final:.4f}  [{combined_ci_lo:.4f}, {combined_ci_hi:.4f}]{' '*(77-68)}│")

print("│                                                                             │")
print("│  ALGORITHM vs CONSENSUS (Patient-Level CV)                                  │")
print("│  ────────────────────────────────────────                                   │")

for r in algo_results:
    sub = r['subtype'].upper()
    line = f"│  {sub:4s} Algorithm-Consensus: rs = {r['spearman']:.4f}  [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]  n={r['n']}"
    print(f"{line:<77s}│")

# Combined algo
r_comb_algo = stats.spearmanr(all_pred, all_cons)[0]
comb_ci_lo = np.percentile(algo_boot_distributions['combined'], 2.5)
comb_ci_hi = np.percentile(algo_boot_distributions['combined'], 97.5)
line = f"│  {'COMB':4s} Algorithm-Consensus: rs = {r_comb_algo:.4f}  [{comb_ci_lo:.4f}, {comb_ci_hi:.4f}]  n={len(all_pred)}"
print(f"{line:<77s}│")

print("│                                                                             │")
print("│  ALGORITHM AS % OF EXPERT-EXPERT CEILING                                   │")
print("│  ──────────────────────────────────────                                     │")

for key in ['lpd', 'gpd']:
    ee_r = [r['spearman'] for r in expert_results if r['subtype']==key and r['pair']=='MEAN'][0]
    al_r = [r['spearman'] for r in algo_results if r['subtype']==key][0]
    pct = al_r / ee_r * 100 if ee_r > 0 else float('nan')
    line = f"│  {key.upper():4s}: {al_r:.4f} / {ee_r:.4f} = {pct:.1f}%"
    print(f"{line:<77s}│")

pct_comb = r_comb_algo / combined_mean_r_final * 100
line = f"│  COMB: {r_comb_algo:.4f} / {combined_mean_r_final:.4f} = {pct_comb:.1f}%"
print(f"{line:<77s}│")

print("│                                                                             │")
print("│  KEY FINDINGS                                                               │")
print("│  ────────────                                                               │")

# Determine if significant
for key in ['lpd', 'gpd', 'combined']:
    if key in algo_boot_distributions and key in expert_boot_distributions:
        diff_mean, diff_ci_lo, diff_ci_hi, p_val = bootstrap_difference_test(
            algo_boot_distributions[key], expert_boot_distributions[key]
        )
        if diff_ci_hi < 0:
            finding = f"Algo BELOW expert ceiling (p={p_val:.3f})"
        elif diff_ci_lo > 0:
            finding = f"Algo ABOVE expert ceiling (p={p_val:.3f})"
        else:
            finding = f"No sig. difference from expert ceiling (p={p_val:.3f})"
        line = f"│  {key.upper():4s}: {finding}"
        print(f"{line:<77s}│")

print("│                                                                             │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

print("\nDone.")
