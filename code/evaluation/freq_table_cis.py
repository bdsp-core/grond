#!/usr/bin/env python3
"""Compute 95% confidence intervals and paired-bootstrap p-values for the
frequency-table headline numbers (Table 4 in the manuscript).

Replicates the quality-filtered cohort used by paper_materials/generate_fig6.py:
  A segment passes the quality filter if any of:
    - MW reviewed (rater set for that mat_file includes 'MW' in labels.csv), OR
    - LB+PH+SZ all provided frequency labels for that mat_file, OR
    - >=10 IIIC expert votes with >=80% pattern-plurality agreement.

For each subtype:
  - Spearman rho 95% CI via Fisher z-transform.
  - MAE 95% CI via patient-stratified non-parametric bootstrap (1000 reps).
  - Paired bootstrap p-value for d_rho = rho_alg - rho_tautan and
    d_mae = MAE_tautan - MAE_alg over the segments where both estimators
    produced a value.

Also writes a JSON summary to results/freq_table_cis.json.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABELS_DIR = PROJECT_DIR / 'data' / 'labels'
OUT_PATH = PROJECT_DIR / 'results' / 'freq_table_cis.json'

N_BOOT = 1000
SUBTYPES = ['lpd', 'gpd', 'lrda', 'grda']


def fisher_z_ci(r, n, alpha=0.05):
    if n < 4 or not np.isfinite(r):
        return float('nan'), float('nan')
    r = max(min(r, 0.999999), -0.999999)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zc = 1.959963984540054
    lo, hi = z - zc * se, z + zc * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def patient_clusters(df):
    return [(pid, np.array(g.index)) for pid, g in df.groupby('patient_id')]


def cluster_bootstrap_indices(clusters, rng):
    pick = rng.integers(0, len(clusters), size=len(clusters))
    return np.concatenate([clusters[i][1] for i in pick])


def load_quality_filtered():
    """Replicates load_data_and_filter() in paper_materials/generate_fig6.py."""
    labels = pd.read_csv(LABELS_DIR / 'labels.csv')
    sl = pd.read_csv(LABELS_DIR / 'segment_labels.csv')
    seg = pd.read_csv(LABELS_DIR / 'segments.csv',
                      usecols=['mat_file', 'pdchar_freq_hz', 'tautan_freq_hz'])
    sl = sl.merge(seg, on='mat_file', how='left')

    fr = labels[labels.label_type == 'frequency_hz'].copy()
    fr['value'] = pd.to_numeric(fr['value'], errors='coerce')
    fr = fr[fr['value'].notna() & (fr['value'] > 0)]

    rater_info = fr.groupby('mat_file')['rater'].apply(set).to_dict()
    expert_freq = fr.groupby('mat_file')['value'].mean().to_dict()
    n_raters = fr.groupby('mat_file')['rater'].nunique().to_dict()

    rows = []
    for _, row in sl.iterrows():
        mf = row['mat_file']
        sub = row.get('subtype')
        if sub not in SUBTYPES or row.get('excluded') == True:
            continue
        if mf not in expert_freq:
            continue
        gt = expert_freq[mf]
        if not np.isfinite(gt) or gt <= 0:
            continue
        if sub in ('lpd', 'gpd'):
            algo = pd.to_numeric(row.get('pdchar_freq_hz'), errors='coerce')
        else:
            algo = pd.to_numeric(row.get('algo_freq_hz'), errors='coerce')
        tautan = pd.to_numeric(row.get('tautan_freq_hz'), errors='coerce')
        if not (np.isfinite(algo) or np.isfinite(tautan)):
            continue
        raters = rater_info.get(mf, set())
        nv = pd.to_numeric(row.get('iiic_n_votes'), errors='coerce')
        pf = pd.to_numeric(row.get('iiic_plurality_frac'), errors='coerce')
        passes = (
            ('MW' in raters)
            or {'LB', 'PH', 'SZ'}.issubset(raters)
            or (np.isfinite(nv) and np.isfinite(pf) and nv >= 10 and pf >= 0.80)
        )
        if not passes:
            continue
        rows.append(dict(
            mat_file=mf, patient_id=row.get('patient_id'), subtype=sub,
            expert_freq_hz=float(gt),
            algo_freq_hz=float(algo) if np.isfinite(algo) else np.nan,
            tautan_freq_hz=float(tautan) if np.isfinite(tautan) else np.nan,
            n_raters=int(n_raters.get(mf, 1)),
        ))
    return pd.DataFrame(rows)


def bootstrap_metrics(df, alg_col, ref_col, baseline_col, rng, n_boot=N_BOOT):
    sub_df = df.dropna(subset=[alg_col, ref_col]).reset_index(drop=True)
    n = len(sub_df)
    rho_pt, _ = spearmanr(sub_df[alg_col], sub_df[ref_col])
    mae_pt = float(np.mean(np.abs(sub_df[alg_col] - sub_df[ref_col])))
    rho_lo, rho_hi = fisher_z_ci(rho_pt, n)

    clusters = patient_clusters(sub_df)
    rhos, maes = [], []
    drhos, dmaes = [], []
    paired = sub_df.dropna(subset=[baseline_col]).reset_index(drop=True)
    paired_clusters = patient_clusters(paired)
    for _ in range(n_boot):
        idx = cluster_bootstrap_indices(clusters, rng)
        b = sub_df.loc[idx]
        if len(b) < 5 or b[alg_col].std() == 0 or b[ref_col].std() == 0:
            continue
        r, _ = spearmanr(b[alg_col], b[ref_col])
        m = float(np.mean(np.abs(b[alg_col] - b[ref_col])))
        rhos.append(r)
        maes.append(m)
        idx2 = cluster_bootstrap_indices(paired_clusters, rng)
        b2 = paired.loc[idx2]
        if (len(b2) >= 5 and b2[alg_col].std() > 0
                and b2[ref_col].std() > 0 and b2[baseline_col].std() > 0):
            ra, _ = spearmanr(b2[alg_col], b2[ref_col])
            rb, _ = spearmanr(b2[baseline_col], b2[ref_col])
            ma = float(np.mean(np.abs(b2[alg_col] - b2[ref_col])))
            mb = float(np.mean(np.abs(b2[baseline_col] - b2[ref_col])))
            drhos.append(ra - rb)
            dmaes.append(mb - ma)
    out = dict(n=int(n), rho=float(rho_pt),
                rho_ci_fisher=[rho_lo, rho_hi],
                rho_ci_boot=[float(np.percentile(rhos, 2.5)),
                              float(np.percentile(rhos, 97.5))],
                mae=mae_pt,
                mae_ci_boot=[float(np.percentile(maes, 2.5)),
                              float(np.percentile(maes, 97.5))])
    if drhos:
        rho_pt_alg = spearmanr(paired[alg_col], paired[ref_col])[0]
        rho_pt_base = spearmanr(paired[baseline_col], paired[ref_col])[0]
        mae_pt_alg = float(np.mean(np.abs(paired[alg_col] - paired[ref_col])))
        mae_pt_base = float(np.mean(np.abs(paired[baseline_col] - paired[ref_col])))
        a = np.array(drhos)
        b = np.array(dmaes)
        out.update(
            paired_n=int(len(paired)),
            rho_baseline=float(rho_pt_base),
            mae_baseline=mae_pt_base,
            delta_rho=float(rho_pt_alg - rho_pt_base),
            delta_mae=float(mae_pt_base - mae_pt_alg),
            delta_rho_ci=[float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
            delta_mae_ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            p_delta_rho=float(min(1.0, 2 * min(np.mean(a <= 0), np.mean(a >= 0)))),
            p_delta_mae=float(min(1.0, 2 * min(np.mean(b <= 0), np.mean(b >= 0)))),
        )
    return out


def main():
    df = load_quality_filtered()
    print(f'Quality-filtered rows total: {len(df)}')
    print('Per-subtype counts:')
    for sub in SUBTYPES:
        print(f'  {sub.upper():4s} {(df.subtype == sub).sum()}')

    rng = np.random.default_rng(42)
    results = {}
    for sub in SUBTYPES:
        d = df[df.subtype == sub].reset_index(drop=True)
        if len(d) < 10:
            continue
        out = bootstrap_metrics(d, 'algo_freq_hz', 'expert_freq_hz',
                                 'tautan_freq_hz', rng=rng)
        results[sub] = out
        print(f'\n{sub.upper():4s}  n={out["n"]:5d}  '
              f'rho={out["rho"]:.3f} CI=[{out["rho_ci_fisher"][0]:.3f}, {out["rho_ci_fisher"][1]:.3f}]  '
              f'MAE={out["mae"]:.3f} CI=[{out["mae_ci_boot"][0]:.3f}, {out["mae_ci_boot"][1]:.3f}]')
        if 'delta_rho' in out:
            print(f'      paired n={out["paired_n"]}  Tautan rho={out["rho_baseline"]:.3f} MAE={out["mae_baseline"]:.3f}')
            print(f'      d_rho={out["delta_rho"]:+.3f} CI=[{out["delta_rho_ci"][0]:+.3f}, {out["delta_rho_ci"][1]:+.3f}] p={out["p_delta_rho"]:.4f}')
            print(f'      d_mae={out["delta_mae"]:+.3f} CI=[{out["delta_mae_ci"][0]:+.3f}, {out["delta_mae_ci"][1]:+.3f}] p={out["p_delta_mae"]:.4f}')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {OUT_PATH.relative_to(PROJECT_DIR)}')


if __name__ == '__main__':
    main()
