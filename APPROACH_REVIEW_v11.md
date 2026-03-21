# Frequency Estimation for Periodic EEG Patterns: Review v11

## Critical Methodological Note

**No method may use the gold standard frequency as input.** All algorithms must operate from the raw EEG alone. Previous evaluations of the HPP timing algorithm used the gold standard frequency as a period prior (`freq_estimate=gold_freq`), which inflated performance metrics significantly. This has been corrected in v11.

## Problem & Data

Estimating frequency (Hz) of periodic discharges (LPD, GPD) and rhythmic delta activity (LRDA, GRDA) in 10-second, 18-channel bipolar EEG at 200 Hz. Additionally: subtype classification, spatial localization, discharge timing, and wave timing.

### Dataset (as of 2026-03-21)

| | Patients | Segments | Labels |
|---|---|---|---|
| LPD | 450 | ~1,700 | Freq, timing, spatial (partial), laterality |
| GPD | 169 | ~830 | Freq, timing, spatial (partial) |
| LRDA | 100 + ~60 harvested | ~310+ | Subtype, partial freq |
| GRDA | 120 + harvesting | ~385+ | Subtype, partial freq |
| BIPD | 100 harvested | ~100 | Subtype only (new) |
| Other (controls) | harvesting | growing | None (true negatives) |
| **Total** | **~1,000+** | **~3,500+** | |

**Discharge timing labels**: 593 patients with MW-reviewed discharge times (5,643+ events). Labels refined through CET-UNet comparison — 66% of reviewed cases had CET-UNet labels accepted as more accurate.

### What changed since v10
- **CET-UNet (CNN Evidence Trace with U-Net)**: Trained to produce frame-level discharge evidence from raw EEG. U-Net with skip connections + sharp targets (σ=10ms) + auxiliary HPP floor loss.
- **CET-UNet labels are more accurate than HPP labels**: MW accepted CET-UNet discharge locations in 66% of 100 reviewed cases. Labels were biased toward pointiness peaks.
- **Critical finding: HPP was cheating**: HPP used gold standard frequency as input, inflating F1 from ~0.64 (fair) to 0.77 (cheating). All future evaluations must use EEG-only input.
- **MPS GPU acceleration**: Training now uses Apple Metal Performance Shaders (~15× speedup).
- **Data harvesting**: 100 BIPDs, 60+ LRDA, GRDA and Other controls being collected from S3.
- **Unified model attempted**: 4-class subtype + frequency + PD/RDA channel detection. Performed comparably to separate models but didn't exceed them. Abandoned in favor of specialized models.
- **Dataset inventory dashboard**: Live tracking of all data by pattern type.

## Current Best Results

### IMPORTANT: Fair vs Unfair Evaluation

Previous reviews reported HPP timing F1=0.795. This was **unfair** because the HPP algorithm received the gold standard frequency as an input parameter. When restricted to EEG-only input (using ACF-estimated frequency instead), HPP F1 drops to ~0.64.

**All results below use EEG-only input unless explicitly marked as "reference."**

### Discharge Timing Detection

| Method | Evidence | Freq Prior | F1 | Freq ρ | Note |
|--------|----------|-----------|-----|--------|------|
| HPP | Handcrafted (pointiness+TKEO) | ACF-estimated | **0.640** | 0.477 | Fair — EEG only |
| HPP | Handcrafted | **Gold standard** | 0.774 | 0.960 | **REFERENCE ONLY — not deployment-ready** |
| CET-UNet+HPP | CNN U-Net | Gold standard | 0.614 | 0.968 | Reference — but CET labels more accurate per MW review |

**Key insight**: The CET-UNet produces more accurate discharge locations (confirmed by MW in 66% of cases), but the automated F1 metric penalizes it because the ground truth labels were biased toward pointiness peaks. After MW corrected labels based on CET, HPP F1 dropped from 0.795 to 0.774 (labels moved away from pointiness peaks).

**TODO**: Fix the fair evaluation pipeline (current implementation has a bug). The fair comparison with CNN-estimated frequency prior needs to be completed.

### Frequency Estimation

| Method | Spearman | MAE | Input |
|--------|----------|-----|-------|
| IPI from MW timing labels | 0.968 | — | Timing labels (not automated) |
| IPI from HPP (gold freq prior) | 0.960 | 0.088 | EEG + gold freq (**reference**) |
| IPI from CET-UNet (gold freq prior) | 0.951 | 0.154 | EEG + gold freq (**reference**) |
| CNN+Attention direct | 0.640 | 0.271 | **EEG only** |
| RF on handcrafted features | 0.604 | 0.267 | **EEG only** |
| Ridge on handcrafted features | 0.589 | 0.274 | **EEG only** |

**Best fully automated**: CNN+Attention at Spearman 0.640. IPI-based methods achieve ~0.95 but require either gold freq prior or accurate timing labels.

### Subtype Classification

| Method | AUC | Note |
|--------|-----|------|
| RF 300 (LPD vs GPD) | 0.931 | EEG only |
| GBM balanced (laterality) | 0.957 | EEG only |

### Channel-Level Detection

| Method | Channel AUC | Patient AUC |
|--------|------------|-------------|
| CNN+Attention (PD) | 0.870 | 0.989 |
| CNN+Attention (RDA) | 0.842 | — |

## The Nine Tasks (Paper Roadmap)

See [docs/PAPER_ROADMAP.md](docs/PAPER_ROADMAP.md) for full details. Specialized models for each task:

1. LPD vs GPD classification
2. LRDA vs GRDA classification
3. PD channel identification
4. RDA channel identification
5. PD discharge timing (HPP + CET-UNet hybrid)
6. RDA wave timing
7. RDA frequency estimation
8. PD frequency estimation
9. BIPD analysis

## Key Findings (v11)

### 1. The gold standard frequency leak

HPP's F1 dropped from 0.774 to 0.640 when the gold standard frequency was removed as input. This means ~17% of HPP's apparent performance came from knowing the answer. **All future evaluations must be EEG-only.**

### 2. CET-UNet finds better discharge locations than pointiness

MW accepted CET-UNet locations over pointiness-based locations in 66% of cases. The pointiness trace biased the original labels toward sharp transients, but many real discharges have broader morphology that the CNN detects better.

### 3. IPI-derived frequency >> direct CNN frequency estimation

Frequency derived from inter-discharge intervals (Spearman ~0.95) massively outperforms direct CNN regression (Spearman 0.64). Counting actual intervals is more precise than learning a regression.

### 4. The complete system needs a frequency estimator

The HPP pipeline requires a frequency estimate to set its periodic prior. The current options are:
- ACF: unreliable (Spearman ~0.48 without gold freq)
- CNN+Attention: better (Spearman 0.64) but not yet integrated with HPP
- IPI from a first-pass timing detection: bootstrap approach (use rough timing → estimate freq → refine timing)

### 5. Unified model didn't improve over specialized models

The 4-class unified CNN performed comparably (freq ρ=0.630, subtype AUC=0.913) but didn't beat specialized models. Separate models that each do their job well are the better approach.

## Infrastructure

See v10 for full infrastructure listing. Key additions in v11:
- `code/cet_model/` — CET and CET-UNet models, training, evaluation, evidence comparison viewer
- `code/harvest_bipd_segments.py` — BIPD harvesting from morgoth2
- `code/harvest_iiic_rda_segments.py` — LRDA/GRDA harvesting
- `code/harvest_iiic_other_segments.py` — Control segment harvesting
- `code/generate_dataset_dashboard.py` — Dataset inventory dashboard
- `results/training_dashboard.html` — Live training progress with loss curves
- `results/evidence_comparison_viewer.html` — 3-way evidence comparison with interactive markers

## Next Steps

1. **Fix fair evaluation**: Complete the EEG-only evaluation with CNN-estimated frequency prior for HPP
2. **Integrate CNN freq → HPP**: Use CNN+Attention frequency estimate as the period prior for HPP timing
3. **Bootstrap timing→freq→timing**: Use rough HPP timing → estimate freq from IPI → re-run HPP with better freq prior
4. **Complete CET label review**: Extend the 100-case CET review to all 593 cases
5. **RDA tasks**: Frequency, timing, channel detection for LRDA/GRDA
6. **Integrate harvested data**: BIPD, LRDA/GRDA, Other controls, high-freq LPDs
