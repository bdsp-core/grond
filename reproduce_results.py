#!/usr/bin/env python3
"""
Reproduce the key results from:
Tautan et al. 2025, J. Neural Eng. 22, 066027

This script reproduces:
- Table 1: MAE between mean expert annotations and algorithm outputs
- Figures 5 & 6: ICC (expert-expert and expert-algorithm) for segments
  with 100% agreement on pattern classification
"""

import pandas as pd
import numpy as np
import pingouin as pg
import os
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOT_DIR = os.path.join(BASE_DIR, 'data', 'annotations')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# ============================================================
# Helper functions
# ============================================================
def mae_with_ci(y_true, y_pred, confidence=0.95):
    """Compute MAE with bootstrap 95% CI."""
    abs_errors = np.abs(y_true - y_pred)
    mae_val = np.nanmean(abs_errors)
    # Bootstrap CI
    n_boot = 1000
    rng = np.random.RandomState(42)
    boot_maes = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(abs_errors), len(abs_errors))
        boot_maes.append(np.nanmean(abs_errors.iloc[idx] if hasattr(abs_errors, 'iloc') else abs_errors[idx]))
    ci_lo = np.percentile(boot_maes, 2.5)
    ci_hi = np.percentile(boot_maes, 97.5)
    return mae_val, ci_lo, ci_hi


def compute_icc3(df_wide, raters, target_col='segment'):
    """Compute ICC(3,1) from wide-format dataframe using pingouin.

    df_wide: DataFrame with columns for each rater
    raters: list of rater column names to include
    Returns: ICC value and 95% CI
    """
    df = df_wide[raters].copy()
    df['segment'] = range(1, len(df) + 1)
    df_long = pd.melt(df, id_vars='segment', var_name='rater', value_name='annotation')
    df_long = df_long.dropna(subset=['annotation'])

    try:
        icc = pg.intraclass_corr(data=df_long, targets='segment', raters='rater',
                                  ratings='annotation', nan_policy='omit')
        row = icc[icc['Type'] == 'ICC3k']
        icc_val = row['ICC'].values[0]
        # Handle both CI95% and CI95 column names across pingouin versions
        ci_col = 'CI95%' if 'CI95%' in icc.columns else 'CI95'
        ci = row[ci_col].values[0]
        return icc_val, ci
    except Exception as e:
        return np.nan, [np.nan, np.nan]


# ============================================================
# 1. Load annotations
# ============================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

# LPD annotations
df_sz_lpd = pd.read_csv(os.path.join(ANNOT_DIR, 'LPDS_SZ_3_2025.csv'))
df_lb_lpd = pd.read_csv(os.path.join(ANNOT_DIR, 'LPDS_LB_2_2025.csv'))
df_ph_lpd = pd.read_csv(os.path.join(ANNOT_DIR, 'LPDS_PH_3_3025.csv'))

# GPD annotations
df_sz_gpd = pd.read_csv(os.path.join(ANNOT_DIR, 'GPDS_SZ_3_2025.csv'))
df_lb_gpd = pd.read_csv(os.path.join(ANNOT_DIR, 'GPDS_LB_2_2025.csv'))
df_ph_gpd = pd.read_csv(os.path.join(ANNOT_DIR, 'GPDS_PH_3_2025.csv'))

# LRDA annotations
df_sz_lrda = pd.read_csv(os.path.join(ANNOT_DIR, 'LRDA_SZ_3_2025.csv'))
df_lb_lrda = pd.read_csv(os.path.join(ANNOT_DIR, 'LRDA_LB_2_2025.csv'))
df_ph_lrda = pd.read_csv(os.path.join(ANNOT_DIR, 'LRDA_PH_3_2025.csv'))

# GRDA annotations
df_sz_grda = pd.read_csv(os.path.join(ANNOT_DIR, 'GRDA_SZ_3_2025.csv'))
df_lb_grda = pd.read_csv(os.path.join(ANNOT_DIR, 'GRDA_LB_2_2025.csv'))
df_ph_grda = pd.read_csv(os.path.join(ANNOT_DIR, 'GRDA_PH_3_2025.csv'))


# ============================================================
# 2. Clean filenames (strip paths and _score.png suffix)
# ============================================================
def clean_files(df):
    df = df.copy()
    df['files'] = df['files'].apply(os.path.basename).str.extract(r'(.*?)(?:_score\.png)?$')[0]
    return df

df_sz_lpd = clean_files(df_sz_lpd)
df_lb_lpd = clean_files(df_lb_lpd)
df_ph_lpd = clean_files(df_ph_lpd)
df_sz_gpd = clean_files(df_sz_gpd)
df_lb_gpd = clean_files(df_lb_gpd)
df_ph_gpd = clean_files(df_ph_gpd)
df_sz_lrda = clean_files(df_sz_lrda)
df_lb_lrda = clean_files(df_lb_lrda)
df_ph_lrda = clean_files(df_ph_lrda)
df_sz_grda = clean_files(df_sz_grda)
df_lb_grda = clean_files(df_lb_grda)
df_ph_grda = clean_files(df_ph_grda)


# ============================================================
# 3. Rename columns per annotator
# ============================================================
def rename_annot(df, suffix, drop_cols=None):
    df = df.copy()
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=[c])
    rename_map = {}
    for c in df.columns:
        if c == 'files':
            continue
        rename_map[c] = f"{c}_{suffix}"
    return df.rename(columns=rename_map)

df_lb_lpd = rename_annot(df_lb_lpd, 'lb')
df_sz_lpd = rename_annot(df_sz_lpd, 'sz', drop_cols=['spatial_original', 'spatial_area_original'])
df_ph_lpd = rename_annot(df_ph_lpd, 'ph', drop_cols=['spatial_origin'])

df_lb_gpd = rename_annot(df_lb_gpd, 'lb')
df_sz_gpd = rename_annot(df_sz_gpd, 'sz', drop_cols=['spatial_origin', 'spatial_area_origin'])
df_ph_gpd = rename_annot(df_ph_gpd, 'ph')

df_lb_lrda = rename_annot(df_lb_lrda, 'lb')
df_sz_lrda = rename_annot(df_sz_lrda, 'sz', drop_cols=['spatial_origin', 'spatial_area_origin'])
df_ph_lrda = rename_annot(df_ph_lrda, 'ph', drop_cols=['spatial_origin', 'birds'])

df_lb_grda = rename_annot(df_lb_grda, 'lb')
df_sz_grda = rename_annot(df_sz_grda, 'sz', drop_cols=['spatial_origin', 'spatial_area_origin'])
df_ph_grda = rename_annot(df_ph_grda, 'ph')


# ============================================================
# 4. Merge annotations
# ============================================================
df_lpd = df_sz_lpd.merge(df_lb_lpd, on='files').merge(df_ph_lpd, on='files')
df_gpd = df_sz_gpd.merge(df_lb_gpd, on='files').merge(df_ph_gpd, on='files')
df_lrda = df_sz_lrda.merge(df_lb_lrda, on='files').merge(df_ph_lrda, on='files')
df_grda = df_sz_grda.merge(df_lb_grda, on='files').merge(df_ph_grda, on='files', how='left')

# Round spatial values
for df in [df_lpd, df_gpd, df_lrda, df_grda]:
    for suf in ['sz', 'lb', 'ph']:
        col = f'spatial_{suf}'
        if col in df.columns:
            df[col] = df[col].round(2)


# ============================================================
# 5. Load algorithm results
# ============================================================
df_lpd_results = pd.read_csv(os.path.join(RESULTS_DIR, 'lpd_results.csv'))
df_gpd_results = pd.read_csv(os.path.join(RESULTS_DIR, 'gpd_results.csv'))
df_lrda_results = pd.read_csv(os.path.join(RESULTS_DIR, 'lrda_results.csv'))
df_grda_results = pd.read_csv(os.path.join(RESULTS_DIR, 'grda_results.csv'))

# Drop index column if present
for df_r in [df_lpd_results, df_gpd_results, df_lrda_results, df_grda_results]:
    if 'Unnamed: 0' in df_r.columns:
        df_r.drop(columns=['Unnamed: 0'], inplace=True)

# Merge results with annotations
df_lpd = df_lpd.merge(df_lpd_results, on='files')
df_gpd = df_gpd.merge(df_gpd_results, on='files')
df_lrda = df_lrda.merge(df_lrda_results, on='files', how='left')
df_grda = df_grda.merge(df_grda_results, on='files', how='left')

print(f"LPD segments: {len(df_lpd)}")
print(f"GPD segments: {len(df_gpd)}")
print(f"LRDA segments: {len(df_lrda)}")
print(f"GRDA segments: {len(df_grda)}")


# ============================================================
# 6. Filter to 100% agreement segments
#    (all 3 annotators agree: all >0 or all ==0)
# ============================================================
def filter_agreement(df, freq_cols, spatial_cols):
    """Keep segments where all annotators agree on presence (all >0 or all ==0)."""
    freq_vals = df[freq_cols]
    all_present = (freq_vals > 0).all(axis=1)
    all_absent = (freq_vals == 0).all(axis=1)
    mask = all_present | all_absent
    df_agreed = df[mask].copy()
    # For ICC/MAE, further filter to only segments where event IS present
    df_present = df[all_present].copy()
    return df_agreed, df_present

df_lpd_agreed, df_lpd_present = filter_agreement(
    df_lpd, ['frequency_sz', 'frequency_lb', 'frequency_ph'],
    ['spatial_sz', 'spatial_lb', 'spatial_ph'])
df_gpd_agreed, df_gpd_present = filter_agreement(
    df_gpd, ['frequency_sz', 'frequency_lb', 'frequency_ph'],
    ['spatial_sz', 'spatial_lb', 'spatial_ph'])
df_lrda_agreed, df_lrda_present = filter_agreement(
    df_lrda, ['frequency_sz', 'frequency_lb', 'frequency_ph'],
    ['spatial_sz', 'spatial_lb', 'spatial_ph'])
df_grda_agreed, df_grda_present = filter_agreement(
    df_grda, ['frequency_sz', 'frequency_lb', 'frequency_ph'],
    ['spatial_sz', 'spatial_lb', 'spatial_ph'])

print(f"\nSegments with 100% agreement (present):")
print(f"  LPD: {len(df_lpd_present)}  (paper: 111)")
print(f"  GPD: {len(df_gpd_present)}  (paper: 109)")
print(f"  LRDA: {len(df_lrda_present)}  (paper: 50)")
print(f"  GRDA: {len(df_grda_present)}  (paper: 119)")


# ============================================================
# 7. Compute mean annotator values (gold standard for MAE)
# ============================================================
def add_mean_annot(df):
    df = df.copy()
    df['freq_mean_annot'] = df[['frequency_sz', 'frequency_lb', 'frequency_ph']].mean(axis=1)
    df['spatial_mean_annot'] = df[['spatial_sz', 'spatial_lb', 'spatial_ph']].mean(axis=1)
    return df

df_lpd_present = add_mean_annot(df_lpd_present)
df_gpd_present = add_mean_annot(df_gpd_present)
df_lrda_present = add_mean_annot(df_lrda_present)
df_grda_present = add_mean_annot(df_grda_present)


# ============================================================
# 8. TABLE 1: MAE (on 100% agreement segments)
# ============================================================
print("\n" + "=" * 70)
print("TABLE 1: MAE (segments with 100% agreement on classification)")
print("=" * 70)

print("\n--- RDA: Frequency of Event [Hz] ---")
print(f"{'Algorithm':<15} {'LRDA MAE':>20} {'GRDA MAE':>20}")
print("-" * 55)

for algo_name, freq_col in [('RDA1a-FFT', 'freq_rda1a_fft'),
                              ('RDA1b-FFT', 'freq_rda1b_fft'),
                              ('RDA2-HHT', 'freq_rda2_hhtt')]:
    mae_lrda, lo_lrda, hi_lrda = mae_with_ci(df_lrda_present['freq_mean_annot'],
                                               df_lrda_present[freq_col])
    mae_grda, lo_grda, hi_grda = mae_with_ci(df_grda_present['freq_mean_annot'],
                                               df_grda_present[freq_col])
    print(f"{algo_name:<15} {mae_lrda:>5.2f} [{lo_lrda:.2f},{hi_lrda:.2f}]   "
          f"{mae_grda:>5.2f} [{lo_grda:.2f},{hi_grda:.2f}]")

print("\nPaper Table 1 (RDA Frequency):")
print("  RDA1a-FFT:  LRDA 0.18 [0.12,0.24]  GRDA 0.24 [0.19,0.30]")
print("  RDA1b-FFT:  LRDA 0.13 [0.09,0.17]  GRDA 0.26 [0.21,0.32]")
print("  RDA2-HHT:   LRDA 0.13 [0.09,0.16]  GRDA 0.46 [0.39,0.52]")

print("\n--- RDA: Spatial Extent ---")
print(f"{'Algorithm':<15} {'LRDA MAE':>20} {'GRDA MAE':>20}")
print("-" * 55)

for algo_name, spat_col in [('RDA1a-FFT', 'spatial_rda1a_fft'),
                              ('RDA1b-FFT', 'spatial_rda1b_fft'),
                              ('RDA2-HHT', 'spatial_rda2_hhtt')]:
    mae_lrda, lo_lrda, hi_lrda = mae_with_ci(df_lrda_present['spatial_mean_annot'],
                                               df_lrda_present[spat_col])
    mae_grda, lo_grda, hi_grda = mae_with_ci(df_grda_present['spatial_mean_annot'],
                                               df_grda_present[spat_col])
    print(f"{algo_name:<15} {mae_lrda:>5.2f} [{lo_lrda:.2f},{hi_lrda:.2f}]   "
          f"{mae_grda:>5.2f} [{lo_grda:.2f},{hi_grda:.2f}]")

print("\nPaper Table 1 (RDA Spatial):")
print("  RDA1a-FFT:  LRDA 0.25 [0.22,0.30]  GRDA 0.32 [0.28,0.36]")
print("  RDA1b-FFT:  LRDA 0.19 [0.16,0.21]  GRDA 0.09 [0.07,0.09]")
print("  RDA2-HHT:   LRDA 0.14 [0.12,0.17]  GRDA 0.08 [0.06,0.09]")


print("\n--- PD: Frequency of Event [Hz] ---")
print(f"{'Algorithm':<15} {'LPD MAE':>20} {'GPD MAE':>20}")
print("-" * 55)

for algo_name, freq_col in [('PD1', 'freq'),
                              ('PD2a', 'freq_apd'),
                              ('PD2b', 'freq_zscore')]:
    mae_lpd, lo_lpd, hi_lpd = mae_with_ci(df_lpd_present['freq_mean_annot'],
                                            df_lpd_present[freq_col])
    mae_gpd, lo_gpd, hi_gpd = mae_with_ci(df_gpd_present['freq_mean_annot'],
                                            df_gpd_present[freq_col])
    print(f"{algo_name:<15} {mae_lpd:>5.2f} [{lo_lpd:.2f},{hi_lpd:.2f}]   "
          f"{mae_gpd:>5.2f} [{lo_gpd:.2f},{hi_gpd:.2f}]")

print("\nPaper Table 1 (PD Frequency):")
print("  PD1:   LPD 1.18 [1.09,1.36]  GPD 0.98 [0.86,1.11]")
print("  PD2a:  LPD 0.41 [0.38,0.45]  GPD 0.15 [0.13,0.18]")
print("  PD2b:  LPD 0.57 [0.50,0.63]  GPD 1.25 [1.20,1.29]")

print("\n--- PD: Spatial Extent ---")
print(f"{'Algorithm':<15} {'LPD MAE':>20} {'GPD MAE':>20}")
print("-" * 55)

for algo_name, spat_col in [('PD1', 'spatial'),
                              ('PD2a', 'spatial_apd'),
                              ('PD2b', 'spatial_zscore')]:
    mae_lpd, lo_lpd, hi_lpd = mae_with_ci(df_lpd_present['spatial_mean_annot'],
                                            df_lpd_present[spat_col])
    mae_gpd, lo_gpd, hi_gpd = mae_with_ci(df_gpd_present['spatial_mean_annot'],
                                            df_gpd_present[spat_col])
    print(f"{algo_name:<15} {mae_lpd:>5.2f} [{lo_lpd:.2f},{hi_lpd:.2f}]   "
          f"{mae_gpd:>5.2f} [{lo_gpd:.2f},{hi_gpd:.2f}]")

print("\nPaper Table 1 (PD Spatial):")
print("  PD1:   LPD 0.53 [0.50,0.56]  GPD 0.01 [0.01,0.02]")
print("  PD2a:  LPD 0.17 [0.15,0.19]  GPD 0.40 [0.35,0.44]")
print("  PD2b:  LPD 0.59 [0.58,0.60]  GPD 0.01 [0.01,0.02]")


# ============================================================
# 9. ICC Analysis (Figures 5 & 6)
# ============================================================
print("\n" + "=" * 70)
print("ICC ANALYSIS (Figures 5 & 6, segments with 100% agreement)")
print("=" * 70)

# --- RDA ICC (Figure 5) ---
print("\n--- Figure 5: RDA ICC ---")

rda_algos = [('RDA1a-FFT', 'freq_rda1a_fft', 'spatial_rda1a_fft'),
             ('RDA1b-FFT', 'freq_rda1b_fft', 'spatial_rda1b_fft'),
             ('RDA2-HHT', 'freq_rda2_hhtt', 'spatial_rda2_hhtt')]

for event_name, df_evt in [('LRDA', df_lrda_present), ('GRDA', df_grda_present)]:
    print(f"\n  {event_name}:")

    # Expert-expert ICC (annotators only)
    annot_freq = df_evt[['frequency_sz', 'frequency_lb', 'frequency_ph']].copy()
    annot_freq.columns = ['Annot 1', 'Annot 2', 'Annot 3']
    icc_ee_freq, ci_ee_freq = compute_icc3(annot_freq, ['Annot 1', 'Annot 2', 'Annot 3'])

    annot_spat = df_evt[['spatial_sz', 'spatial_lb', 'spatial_ph']].copy()
    annot_spat.columns = ['Annot 1', 'Annot 2', 'Annot 3']
    icc_ee_spat, ci_ee_spat = compute_icc3(annot_spat, ['Annot 1', 'Annot 2', 'Annot 3'])

    print(f"    ee-IRR Freq: ICC={icc_ee_freq:.0%}  CI=[{ci_ee_freq[0]:.0%},{ci_ee_freq[1]:.0%}]")
    print(f"    ee-IRR Spat: ICC={icc_ee_spat:.0%}  CI=[{ci_ee_spat[0]:.0%},{ci_ee_spat[1]:.0%}]")

    # Expert-algorithm ICC
    for algo_name, freq_col, spat_col in rda_algos:
        df_wide_freq = annot_freq.copy()
        df_wide_freq[algo_name] = df_evt[freq_col].values
        icc_ea_freq, ci_ea_freq = compute_icc3(df_wide_freq, ['Annot 1', 'Annot 2', 'Annot 3', algo_name])

        df_wide_spat = annot_spat.copy()
        df_wide_spat[algo_name] = df_evt[spat_col].values
        icc_ea_spat, ci_ea_spat = compute_icc3(df_wide_spat, ['Annot 1', 'Annot 2', 'Annot 3', algo_name])

        print(f"    ea-IRR {algo_name} Freq: ICC={icc_ea_freq:.0%}  CI=[{ci_ea_freq[0]:.0%},{ci_ea_freq[1]:.0%}]")
        print(f"    ea-IRR {algo_name} Spat: ICC={icc_ea_spat:.0%}  CI=[{ci_ea_spat[0]:.0%},{ci_ea_spat[1]:.0%}]")

print("\nPaper Figure 5 reference (ICC %):")
print("  LRDA: ee-IRR Freq=88%, RDA1a=84%, RDA1b=91%, RDA2=72%")
print("  GRDA: ee-IRR Freq=92%, RDA1a=95%, RDA1b=96%, RDA2=73%")
print("  LRDA: ee-IRR Spat=89%, RDA1a=60%, RDA1b=83%, RDA2=66%")
print("  GRDA: ee-IRR Spat=15%, RDA1a=85%, RDA1b=49%, RDA2=31%")

# --- PD ICC (Figure 6) ---
print("\n--- Figure 6: PD ICC ---")

pd_algos = [('PD1', 'freq', 'spatial'),
            ('PD2a', 'freq_apd', 'spatial_apd'),
            ('PD2b', 'freq_zscore', 'spatial_zscore')]

for event_name, df_evt in [('LPD', df_lpd_present), ('GPD', df_gpd_present)]:
    print(f"\n  {event_name}:")

    annot_freq = df_evt[['frequency_sz', 'frequency_lb', 'frequency_ph']].copy()
    annot_freq.columns = ['Annot 1', 'Annot 2', 'Annot 3']
    icc_ee_freq, ci_ee_freq = compute_icc3(annot_freq, ['Annot 1', 'Annot 2', 'Annot 3'])

    annot_spat = df_evt[['spatial_sz', 'spatial_lb', 'spatial_ph']].copy()
    annot_spat.columns = ['Annot 1', 'Annot 2', 'Annot 3']
    icc_ee_spat, ci_ee_spat = compute_icc3(annot_spat, ['Annot 1', 'Annot 2', 'Annot 3'])

    print(f"    ee-IRR Freq: ICC={icc_ee_freq:.0%}  CI=[{ci_ee_freq[0]:.0%},{ci_ee_freq[1]:.0%}]")
    print(f"    ee-IRR Spat: ICC={icc_ee_spat:.0%}  CI=[{ci_ee_spat[0]:.0%},{ci_ee_spat[1]:.0%}]")

    for algo_name, freq_col, spat_col in pd_algos:
        df_wide_freq = annot_freq.copy()
        freq_vals = df_evt[freq_col].values
        df_wide_freq[algo_name] = freq_vals
        icc_ea_freq, ci_ea_freq = compute_icc3(df_wide_freq, ['Annot 1', 'Annot 2', 'Annot 3', algo_name])

        df_wide_spat = annot_spat.copy()
        spat_vals = df_evt[spat_col].values
        df_wide_spat[algo_name] = spat_vals
        icc_ea_spat, ci_ea_spat = compute_icc3(df_wide_spat, ['Annot 1', 'Annot 2', 'Annot 3', algo_name])

        print(f"    ea-IRR {algo_name} Freq: ICC={icc_ea_freq:.0%}  CI=[{ci_ea_freq[0]:.0%},{ci_ea_freq[1]:.0%}]")
        print(f"    ea-IRR {algo_name} Spat: ICC={icc_ea_spat:.0%}  CI=[{ci_ea_spat[0]:.0%},{ci_ea_spat[1]:.0%}]")

print("\nPaper Figure 6 reference (ICC %):")
print("  LPD: ee-IRR Freq=49%, PD1=40%, PD2a=86%, PD2b=46%")
print("  GPD: ee-IRR Freq=86%, PD1=0%, PD2a=61%, PD2b=55%")
print("  LPD: ee-IRR Spat=77%, PD1=65%, PD2a=77%, PD2b=76%")
print("  GPD: ee-IRR Spat=13%, PD1=12%, PD2a=11%, PD2b=0%")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
