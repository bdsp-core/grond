# Frequency Estimation for Periodic EEG Patterns: Review v13

## Critical Methodological Principle

**No method may use gold standard labels as input.** All algorithms operate from raw EEG only.

## Dataset (as of 2026-03-22)

### Labeled Data

| Type | Active | Freq | Timing | Laterality | Status |
|------|--------|------|--------|------------|--------|
| **LPD** | 437 | 437 | 437 | 437 | **Complete** |
| **GPD** | 207 | 207 | 207 | N/A | **Complete** |
| **BIPD** | 21 | — | awaiting | N/A | Screened, needs L/R timing |
| **LRDA** | 99 | 4 | 4 | 0 | Mostly unlabeled |
| **GRDA** | 119 | 14 | 14 | N/A | Mostly unlabeled |

**Discharge timing labels**: 675 ground truth cases (437 LPD + 207 GPD + 31 other), expert-reviewed through 3 rounds of model-assisted correction. Labels refined using HemiCET model predictions as reference.

### IIIC Dataset Expansion (in progress)

Downloading ~8,260 segments from the IIIC S3 dataset:
- LPD: 2,144 segments
- GPD: 2,167 segments
- LRDA: 1,311 segments
- GRDA: 2,638 segments

Each segment has IIIC expert vote data: vote counts for [other, seizure, lpd, gpd, lrda, grda] from 1-32 raters (median 3). Mean agreement: 77-96% depending on pattern type.

### EEG Data

~9,600 standardized .mat files (18 channels × 2000 samples at 200 Hz). All files verified to have consistent format (keys: data, Fs).

## What Changed Since v12

### 1. Single-Hemisphere Discharge Detection (HemiCET)

Trained a CET-UNet that takes 8 channels (one hemisphere) as input and produces a single evidence trace. Combined with existing DP pipeline for discharge detection.

**Key result**: HemiCET + DP achieves the **best timing precision** of any method — 5.0ms median timing MAE — while maintaining competitive F1.

### 2. Comprehensive Model Comparison

| Method | F1 | Sens | Prec | Freq ρ | Freq MAE | Timing MAE | Timing Med |
|--------|-----|------|------|--------|----------|-----------|------------|
| Full 18ch pipeline | **0.717** | 0.714 | **0.720** | 0.733 | **0.291** | 17.3ms | 9.9ms |
| Per-hemi baseline | 0.707 | 0.712 | 0.702 | 0.729 | 0.298 | 19.0ms | 10.0ms |
| HemiCET + DP | 0.699 | **0.746** | 0.658 | **0.758** | 0.317 | **14.0ms** | **5.0ms** |

HemiCET trades precision for sensitivity and timing accuracy. Its superior timing enables better IPI-derived frequency (ρ=0.758 vs 0.733).

### 3. End-to-End Models Investigated but Did Not Beat Pipeline

| Model | Params | F1 | Finding |
|-------|--------|-----|---------|
| PDNetV2 (18ch U-Net+Transformer) | 3.4M | 0.460 | Severe overfitting |
| HemiNet Design A (8ch U-Net+Transformer) | 2.3M | 0.624 | Overfitting |
| HemiNet Design B (8ch dilated conv) | 2.2M | 0.626 | Overfitting |
| HemiNet Design D (HPP wrapper) | 262K | 0.605 | Most stable but lowest |
| HemiNet + MAE pretrain | 2.3M | 0.620 | Slight improvement |

**Conclusion**: With ~800 labeled examples, end-to-end models cannot beat the handcrafted pipeline. The proven components (pointiness, TKEO, DP) encode domain knowledge that's hard to learn from data this size. The winning strategy is HemiCET as evidence source + existing DP framework.

### 4. Label Cleanup Through Model-Assisted Review

Three rounds of reviewing cases where HemiCET disagreed with ground truth labels:

| Round | Cases reviewed | CET accepted | Edited | Rejected (not PD) |
|-------|---------------|-------------|--------|-------------------|
| 1 | 42 | 4 | 11 | 0 |
| 2 | 91 | 20 | 15 | 0 |
| 3 | 66 | 0 | 11 | 15 |
| **Total** | **199** | **24** | **37** | **15** |

Impact on frequency correlation (GT vs HemiCET IPI frequency):
- Before cleanup: Spearman ρ = 0.765, MAE = 0.315 Hz
- After cleanup: **Spearman ρ = 0.819, MAE = 0.252 Hz**

Many "errors" were actually label noise — the model was right and the labels were wrong in ~24 cases.

### 5. Complete LPD/GPD Labeling

All LPD and GPD cases now have complete labels:
- **Frequency**: 644/644 (100%)
- **Discharge timing**: 644/644 (100%)
- **Laterality** (LPD only): 437/437 (100%)

Laterality was labeled using CNN+Attention PD probability predictions — **98% model accuracy** (224/229 cases where model prediction matched expert judgment).

### 6. BIPD Detection Plan

21 confirmed BIPD cases identified from 198 candidates (11% yield). Plan: run PD detection independently per hemisphere, then use a synthetic-data-trained classifier on timing sequences to distinguish BIPD from GPD. See BIPD_PLAN.md.

### 7. HPP-Assisted Labeling Tool

Built an efficient semi-automated labeling tool:
- User clicks a frequency button (0.25-4.5 Hz)
- HPP algorithm instantly shows inferred discharge times (precomputed for all frequencies)
- User accepts, adjusts, or rejects
- Dramatically faster than manual annotation

Used to label 327 harvested segments (72 accepted, 254 rejected, 1 BIPD) and 27 remaining LPD/GPD cases.

### 8. Data Cleanup and Standardization

- All EEG files standardized to 18ch × 2000 samples, {data, Fs} format
- 941 broken stub files deleted, 1,061 20-channel files trimmed, 1,375 duplicates removed
- patients.csv expanded to 2,865 rows (includes harvest manifests)
- segments.csv expanded to 3,313 rows
- discharge_times.json: unified timing label file with BIPD support (left_times/right_times)
- Expert vote data integrated from IIIC dataset (47,330 segments)

## Architecture: Two Detection Approaches

### Approach 1: Full 18-Channel Pipeline (F1=0.717)

```
EEG (18ch × 2000) ──┬── CNN+Attention (freq) ──→ f_cnn ──┐
                     ├── ACF on pointiness ──→ f_acf ──────┤── 0.8×f_cnn + 0.2×f_acf ──→ f_est
                     ├── HPP evidence (pointiness+TKEO) ───┤
                     └── CET-UNet evidence ────────────────┤── product-boost ──→ DP ──→ times
```

### Approach 2: HemiCET (8-Channel, Best Timing, F1=0.699)

```
EEG (8ch hemisphere) ──→ HemiCET-UNet ──→ evidence ──→ DP ──→ times
                    ──→ CNN+Attention (freq) ──→ f_est ──↗
```

HemiCET is the building block for BIPD detection (run independently on each hemisphere).

## Method Names

| Abbreviation | Full Name | What it does |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge timing |
| **CET** | CNN Evidence Trace | Learned per-channel discharge evidence (U-Net) |
| **HemiCET** | Hemisphere CET | 8-channel CET-UNet for single-hemisphere evidence |

## Nine Tasks Status

| # | Task | Status | Best Method | Performance |
|---|------|--------|-------------|-------------|
| 1 | LPD vs GPD | **Done** | RF 300 | AUC 0.931 |
| 2 | LRDA vs GRDA | TODO | — | — |
| 3 | PD channel ID | Partial | CNN+Attention | AUC 0.870 |
| 4 | RDA channel ID | Pseudolabels | CNN | AUC 0.842 |
| 5 | PD discharge timing | **Done** | Full pipeline / HemiCET | F1 0.717 / 0.699 |
| 6 | RDA wave timing | TODO | — | — |
| 7 | RDA frequency | Partial (14 cases) | FFT baseline | ρ 0.840 |
| 8 | PD frequency | **Done** | HemiCET IPI | ρ 0.819 |
| 9 | BIPD detection | Plan complete | — | — |

## Next Steps

1. **Retrain models** with cleaned labels (675 GT cases, improved from 593)
2. **Integrate IIIC S3 data** (~8,000 segments downloading) for pretraining and augmentation
3. **Self-supervised pretraining** for HemiCET using all available EEG
4. **LRDA/GRDA labeling** — build frequency labeling tools for RDA patterns
5. **BIPD timing labeling** — 21 confirmed cases need per-hemisphere timing
6. **Implement BIPD classifier** per BIPD_PLAN.md
7. **Generate paper figures** per PAPER_ROADMAP.md
