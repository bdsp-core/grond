# Paper Roadmap: Automated Characterization of Periodic and Rhythmic EEG Patterns

## Overview

This paper describes a comprehensive system for characterizing periodic discharges (PDs) and rhythmic delta activity (RDA) in continuous EEG. The system comprises a suite of specialized algorithms — each optimized for its specific task — that together provide: subtype classification, spatial localization, discharge/wave timing, and frequency estimation for both PD and RDA patterns.

The key innovations are:
1. **Rich annotation framework** — discharge-level timing labels, per-channel spatial labels, and frequency labels for 800+ patients, gathered through iterative human-in-the-loop active learning with custom HTML review tools
2. **Hidden Point Process (HPP) algorithm** — MAP inference via dynamic programming for discharge/wave timing, using approximately-periodic priors with skipped-event tolerance
3. **CNN Evidence Traces (CET)** — learned per-channel evidence signals that replace handcrafted features as input to HPP, improving timing accuracy in difficult cases
4. **Narrowband Variance Optimization (NVO)** — sinusoidal template matching for RDA frequency and phase estimation
5. **Systematic benchmarking** — "contest of agents" approach comparing 50+ algorithm variants across all tasks

---

## The Nine Tasks

### Task 1: LPD vs GPD Classification
**Goal**: Distinguish lateralized (LPD) from generalized (GPD) periodic discharges.
**Context**: Morgoth already provides excellent pattern-type classification (LPD/GPD/LRDA/GRDA/seizure). This task serves as (a) a sanity check that our features capture the right spatial patterns, and (b) a component that helps downstream tasks (e.g., spatial localization benefits from knowing subtype).
**Current best**: RF 300 trees, AUC 0.931.
**Approach**: Handcrafted features (laterality index, frequency features) + tree-based classifiers. The CNN spatial attention also implicitly learns this.

### Task 2: LRDA vs GRDA Classification
**Goal**: Distinguish lateralized (LRDA) from generalized (GRDA) rhythmic delta activity.
**Context**: Same as Task 1 — primarily a sanity check and supporting task.
**Current best**: Not yet benchmarked separately (included in 4-class unified model at 0.913 macro AUC, but LRDA accuracy was poor at 20%).
**Approach**: Similar to Task 1 but for RDA patterns. Need to develop RDA-specific spatial features. The laterality index should work similarly.

### Task 3: PD Channel Identification
**Goal**: For each of 18 bipolar EEG channels, determine whether it contains periodic discharges.
**Context**: Enables spatial localization ("left temporal predominant", "bilateral, frontal maximum") and supports verbal description generation following ACNS 2021 nomenclature. We have rules that map channel involvement to verbal descriptions.
**Training data**: 304 MW-reviewed cases (ground truth) + pseudolabels from 530+ additional cases (using laterality, subtype, and expert spatial annotations as proxies). Fine-tuning critical — the CNN was too aggressive (5% acceptance rate in MW review), over-including channels.
**Current best**: CNN+Attention, 0.865 channel AUC (on ground truth), but needs calibration to avoid over-inclusion.
**Approach**:
- Pre-train on abundant pseudolabels (LRDA/GRDA as hard negatives, LPD contralateral as negatives)
- Fine-tune on MW-reviewed ground truth + expert spatial annotations (Peter Hadar, Laura Basovic, Sahar Zafar)
- Threshold calibration to match MW's channel inclusion rate (mean 6.2 for LPD, 14.8 for GPD)

### Task 4: RDA Spatial Extent
**Goal**: For each of 18 bipolar EEG channels, determine whether it contains rhythmic delta activity. Report spatial extent as fraction of involved channels.
**Context**: Enables spatial localization of LRDA/GRDA patterns. Component of ACNS 2021 description.
**Training data**: 211 LRDA/GRDA segments with 3-rater (LB, PH, SZ) spatial_extent ground truth.
**Current best**: RDA-PLV (phase coherence), ICC 0.371 matching expert-expert ICC 0.373. MAE 0.215.
**Approach — RDA-PLV** (`code/rda_spatial_extent.py`):
1. Get frequency estimate from W05
2. Bandpass at estimated_freq ± 0.4 Hz (narrowband)
3. Identify dominant hemisphere (highest narrowband variance)
4. Compute reference signal: mean of top-3 narrowband channels on dominant side
5. Per-channel PLV: phase coherence with reference → 18 channel scores
6. Threshold (0.62) → binary involvement → spatial extent = count/18
**Key finding**: PLV (phase coherence) dominates over VE and SNR. Algorithm agreement with experts (ICC 0.371) matches expert-expert agreement (0.373).

### Task 5: PD Discharge Timing
**Goal**: Mark the precise time of each periodic discharge on each involved channel: t_1, t_2, ..., t_K.
**Context**: Enables IPI-derived frequency estimation (more accurate than FFT/ACF), regularity assessment (ACNS "nearly regular" criterion), and downstream BIPD analysis (comparing timing between hemispheres).
**Training data**: 675 MW-reviewed cases with discharge times, refined through 3 rounds of model-assisted review. Complete ground truth for LPD+GPD.
**Current best**: HemiCET+DP, F1=0.873, freq Spearman 0.885, timing accuracy <1ms median.
**Approach — HemiCET+DP (current best)**:
1. HemiCET-UNet (8-channel, one hemisphere) produces evidence trace from raw EEG
2. CNN+ACF ensemble estimates frequency (0.8×CNN + 0.2×ACF)
3. DP with approximately-periodic prior finds optimal discharge sequence
4. EM template refinement
5. Post-hoc confidence filtering
For LPD: run on affected hemisphere. For GPD: run on both, keep best. For BIPD: run independently on each hemisphere.

**Approach — Full 18ch pipeline (previous best, F1=0.717)**:
Product-boosted max(HPP handcrafted, CET-UNet) evidence + CNN+ACF freq + optimized DP. Surpassed by HemiCET approach.

**Key finding**: 8-channel hemisphere input outperforms 18-channel because it avoids noise from the uninvolved hemisphere (LPD) and learns cross-channel patterns within a hemisphere directly.

### Task 6: RDA Wave Timing
**Goal**: Mark the timing of each rhythmic delta wave — specifically the onset, peak, and offset of each wave cycle.
**Context**: Enables phase estimation of RDA (important for downstream analysis, e.g., whether fast activity occurs at a specific RDA phase). Also enables wave-by-wave frequency estimation.
**Training data**: Not yet available. Need to develop.
**Approach — NVO (Narrowband Variance Optimization)**:
Use the existing method we developed for estimating RDA frequency by finding a narrowband sinusoidal approximation that maximizes variance explained. This provides:
- Estimated frequency of the dominant rhythmic component
- Phase of the fitted sinusoid → wave onset/peak/offset times
- The fitted sinusoid serves as an "evidence trace" for RDA waves, analogous to the pointiness trace for PDs

**Approach — HPP for RDA**:
Adapt the HPP algorithm for RDA waves:
- Evidence signal: NVO-derived sinusoidal fit amplitude (or CNN evidence)
- Periodic prior: softer than PD version (RDA frequency is more variable)
- Wave markers: onset, peak, offset (3 markers per wave vs 1 for PD)
- Active interval detection: identify where rhythmic pattern is present

### Task 7: RDA Frequency Estimation
**Goal**: Estimate the frequency (Hz) of rhythmic delta activity.
**Context**: Component of ACNS 2021 standardized description.
**Training data**: 556 LRDA + 1,281 GRDA with MW-reviewed frequency labels. 68 with 3-expert frequency.
**Current best**: W05_DomOnly_IterRefine (Hilbert + iterative narrowband), ICC 0.860 matching expert-expert ICC 0.852. Fig 6: LRDA ρ=0.617, GRDA ρ=0.689.
**Approach — W05**:
1. Bandpass 0.5-3.5 Hz, coarse lateralization via variance
2. Hilbert instantaneous frequency from top-3 channels of dominant hemisphere
3. Narrowband at estimated freq ± 0.4 Hz
4. Refined lateralization via envelope amplitude + refined Hilbert frequency

### Task 8: PD Frequency Estimation
**Goal**: Estimate the frequency (Hz) of periodic discharges.
**Context**: Core deliverable. Component of ACNS 2021 description.
**Training data**: 644 cases with gold standard freq (437 LPD, 207 GPD). All complete.
**Current best**: HemiCET+DP IPI-derived frequency, Spearman 0.885, MAE 0.202 Hz.
**Approaches compared**:
- (a) Alexandra's original method (Spearman 0.353)
- (b) CNN+Attention direct prediction (Spearman 0.744)
- (c) Full 18ch pipeline IPI (Spearman 0.733)
- (d) **HemiCET+DP IPI (Spearman 0.885)** ← current best
**Key finding**: IPI-derived frequency from accurate discharge timing is far more accurate than direct CNN prediction. Label cleanup improved correlation from 0.765→0.885.

### Task 9: BIPD Analysis
**Goal**: Detect bilateral independent periodic discharges — LPDs occurring independently on left and right hemispheres with different timing/frequency.
**Context**: BIPDs have distinct clinical significance from unilateral LPDs or bilateral GPDs.
**Training data**: 21 confirmed BIPD cases (from 198 candidates, 11% yield). Awaiting per-hemisphere timing labels.
**Approach** (see BIPD_PLAN.md):
1. Run HemiCET+DP independently on each hemisphere → two timing sequences
2. Train a classifier on timing sequence pairs to distinguish BIPD from GPD
3. Use synthetic training data (cross-patient LPD pairs, phase-shifted GPD) to overcome small sample size
4. Features: phase consistency, frequency ratio, cross-correlation, matched fraction
5. GBT classifier on ~18 handcrafted features from the two timing sequences
**Status**: Plan complete, BIPD screener built, 21 cases confirmed. Need per-hemisphere timing labels before implementing classifier.

---

## Method Names and Abbreviations

| Abbreviation | Full Name | Description |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge/wave timing with periodic prior |
| **CET** | CNN Evidence Trace | Learned per-channel evidence signal replacing handcrafted features |
| **HemiCET** | Hemisphere CET | 8-channel CET-UNet for single-hemisphere evidence (current best) |
| **NVO** | Narrowband Variance Optimization | Sinusoidal template fitting for RDA frequency/phase estimation |
| **SPF** | Signal Processing Features | Handcrafted features (pointiness, ACF, FFT, TKEO, coherence) |

---

## Implementation Roadmap

### Phase 1: PD Tasks (Tasks 1, 3, 5, 8) — Complete (except spatial)

| Step | Task | Status | Result |
|------|------|--------|--------|
| 1.1 | PD frequency (SPF baselines) | **Done** | 42 experiments, RF best at 0.604 |
| 1.2 | PD frequency (CNN+Attention) | **Done** | Spearman 0.744 |
| 1.3 | PD timing (HPP) | **Done** | Superseded by HemiCET |
| 1.4 | PD channel identification | Partial | 304 GT, 290 pending review |
| 1.5 | LPD vs GPD classification | **Done** | RF AUC 0.931 |
| 1.6 | PD timing (CET+HPP) | **Done** | Full 18ch pipeline F1=0.717 |
| 1.7 | PD timing (HemiCET+DP) | **Done** | **F1=0.873**, <1ms timing, ρ=0.885 |
| 1.8 | PD frequency (HemiCET IPI) | **Done** | **Spearman 0.885**, MAE 0.202 Hz |
| 1.9 | LPD laterality | **Done** | 437/437 complete, 98% model accuracy |
| 1.10 | Label cleanup (3 rounds) | **Done** | 675 GT cases, 61 corrections, 15 rejections |
| **1.11** | **Complete spatial review** | **TODO** | 290 cases pending |

### Phase 2: RDA Tasks (Tasks 2, 4, 6, 7) — Mostly Done

| Step | Task | Status | Result |
|------|------|--------|--------|
| 2.1 | LRDA vs GRDA classification | **Done** | W05_DomOnly_IterRefine, AUC 0.837 |
| 2.2 | LRDA laterality annotation | **Done** | 1,374 LRDA segments reviewed (3 batches) |
| 2.3 | RDA spatial extent | **Done** | RDA-PLV, ICC 0.371 = expert ICC 0.373 |
| 2.4 | RDA frequency (W05/W07) | **Done** | W05 ICC 0.860 = expert ICC 0.852; W07 ρ=0.686 |
| 2.5 | RDA frequency labeling | **Done** | 453 new LRDA/GRDA labels from MW review |
| **2.6** | **RDA wave timing** | **TODO** | 549 cases labeled, need automated method |

### Phase 3: Data Expansion

| Step | What | Status |
|------|------|--------|
| 3.1 | Hi-freq LPD harvest (>2.5 Hz) | **Done** — 297 harvested, 72 accepted with timing labels |
| 3.2 | IIIC S3 data download (8,260 segments) | **In progress** — ~5,000 downloaded |
| 3.3 | Expert vote integration | **Done** — 47,330 segments with IIIC vote data |
| 3.4 | Retrain HemiCET with expanded data | **Done** — v2 on cleaned labels, F1=0.873 |
| 3.5 | Self-supervised pretraining on all EEG | TODO |
| 3.6 | Multi-segment training | TODO |

### Phase 4: BIPD Analysis (Task 9)

| Step | What | Status |
|------|------|--------|
| 4.1 | BIPD screening | **Done** — 21/198 confirmed |
| 4.2 | BIPD per-hemisphere timing labels | TODO — labeler built |
| 4.3 | Synthetic training data generation | TODO — plan in BIPD_PLAN.md |
| 4.4 | Train BIPD vs GPD classifier | TODO |

### Phase 5: Paper Writing

| Step | What |
|------|------|
| 5.1 | Generate all figures and tables |
| 5.2 | Write methods section |
| 5.3 | Write results section |
| 5.4 | Write discussion |
| 5.5 | Internal review |

---

## Paper Outline (Revised 2026-03-31)

### Framing

The paper presents **PDCharacterizer** — a unified pipeline that characterizes periodic and rhythmic EEG patterns in a single pass: laterality, spatial localization, discharge/wave timing, and frequency estimation. The pipeline combines CNNs for feature extraction with dynamic programming for temporal structure, achieving near-expert-level performance across all tasks.

### Title
"PDCharacterizer: Automated Lateralization, Spatial Localization, Timing, and Frequency Estimation for Periodic and Rhythmic EEG Patterns"

### Abstract
- Problem: Characterizing PD/RDA properties (laterality, spatial extent, frequency, timing) is subjective and time-consuming
- What we did: unified pipeline (PDCharacterizer) combining ChannelPD-Net + Hybrid CNN+PLV + HemiCET+DP, trained with iterative human-in-the-loop label refinement
- Key results: Lat AUC 0.984, Freq ρ 0.663, Timing F1 0.506, Spatial Jaccard 0.731 (97.3% of human agreement)
- Key innovation: single-hemisphere CET-UNet evidence + DP inference; spatial localization matching expert inter-rater reliability
- Significance: first system to jointly characterize all properties of periodic and rhythmic patterns with quantitative performance benchmarks against multi-rater expert annotations

### 1. Introduction
- EEG monitoring in ICU; prevalence and significance of periodic/rhythmic patterns
- ACNS 2021 terminology: laterality, spatial localization, frequency, timing
- Current limitations: manual characterization is subjective, no discharge-level timing, spatial localization is qualitative
- What this paper contributes: a unified automated system with expert-level performance

### 2. Data and Annotations
- **2.1 Dataset** — 12,983 active segments across 4 subtypes (LPD, GPD, LRDA, GRDA), 3 data sources (IIIC crowd-labeled, MW-labeled, expert dataset)
- **2.2 Multi-layer annotation framework** — IIIC crowd votes (≥10 experts for 3,731 segments), expert frequency/laterality/spatial labels, discharge timing labels
- **2.3 Active learning tools** — HTML-based interactive viewers for laterality+timing+frequency (combined labeler), spatial localization (topoplot + per-channel toggling), iterative model-assisted review
- **2.4 Pseudolabel framework** — leveraging patient-level labels for channel-level training
- **Figure 1**: LPD characterization examples — easy/medium/hard (EEG + topoplot + verbal description) — **DONE**
- **Figure 2**: GPD characterization examples — **DONE**
- **Table 1**: Dataset statistics by subtype — segments, patients, label coverage for each label type (pattern class, laterality, frequency, spatial, timing) with multi-rater counts

### 3. Methods

- **3.1 PDCharacterizer pipeline overview**
  - **Figure 3**: System diagram — EEG input → ChannelPD-Net (per-channel PD probability) → laterality + spatial (Hybrid CNN+PLV) → discharge timing (HemiCET+DP) → frequency (IPI) → ACNS verbal description
  - Single-pass inference; all components use the same 18-channel bipolar EEG input
  - **Table 2**: Component summary — architecture, parameters, training data, role

- **3.2 ChannelPD-Net** — per-channel PD detection + frequency estimation
  - 1D CNN+Attention architecture (4 conv blocks + attention pooling)
  - Multi-task: PD probability + log-frequency per channel
  - 5-fold patient-stratified CV, trained on curated 815-patient dataset

- **3.3 Spatial localization** — Discharge-locked topographic mapping
  - **Motivation**: Expert ratings of "spatial extent" (% channels involved) showed poor inter-rater reliability (ICC 0.43-0.69), making threshold-based channel counting an unreliable ground truth. Because our HemiCET+DP pipeline localizes discharge peaks with <1ms accuracy, we can directly compute the voltage topography at each discharge — a principled, ground-truth-free approach to localization.
  - **Method**: Laplacian-GFP aligned discharge averaging with two-pass template refinement and GFP²-weighted averaging to suppress phantom discharges
  - **Output**: MNE spherical spline topoplots (both monopolar and Laplacian/CSD), with standardized ACNS 2021 verbal descriptions using 16 brain regions from the morgoth-viewer IED localization module
  - **Spatial extent comparison** (retained for benchmarking against Tautan et al.): PDCharacterizer CNN+PLV threshold (T=0.62), RDA-PLV×Amp (T=0.15). PDCharacterizer exceeds expert-expert ICC for PD spatial (0.852 vs 0.845)
  - Region mapping: 16 regions including transitional zones (frontotemporal, centro-parietal, fronto-central, parieto-occipital) via morgoth-viewer
  - ACNS 2021 verbal description: laterality from PDCharacterizer, localization from discharge-locked topography

- **3.4 Discharge timing** — HemiCET+DP
  - HemiCET: 8-channel CET-UNet (one hemisphere → frame-level evidence)
  - Dual evidence: product-boosted max(HPP handcrafted, CET learned)
  - Dynamic programming with approximately-periodic prior (α=1.275, skip tolerance)
  - EM template refinement + post-hoc confidence filtering
  - Frequency: 1/median(IPI) from detected discharge times

- **3.5 RDA lateralization + frequency** — signal processing contest
  - V5 lateralization contest: 76 methods compared
  - Best lateralization: W05_DomOnly_IterRefine (two-pass envelope amplitude + Hilbert frequency), AUC 0.837
  - Best frequency: W07_AutoChannel_FreqAgreement (MAD-based channel selection + Hilbert), ρ 0.686

- **3.6 RDA spatial extent** — RDA-PLV
  - Per-channel PLV coherence with dominant-hemisphere reference signal at estimated frequency
  - Analogous to PLV refinement in PD spatial pipeline
  - Threshold optimization (0.62) against 3-rater ground truth (211 segments)

- **3.6 Optimization framework**
  - "Contest of agents" approach: systematic comparison of many algorithm variants
  - Live leaderboards for tracking experiments
  - Patient-stratified cross-validation throughout

### 4. Results

- **4.1 Lateralization**
  - PD: Lat AUC 0.984, Accuracy 95.0% (n=880)
  - RDA: Lat AUC 0.837 (V5 contest, n=4,253)
  - **Table 3**: Lateralization performance by subtype

- **4.2 Spatial localization**
  - **Discharge-locked topographic mapping** — primary localization method for PDs
    - Two-pass Laplacian-GFP alignment + GFP²-weighted averaging → MNE topoplot
    - ACNS 2021 verbal descriptions with 16-region localization (morgoth-viewer)
    - Interactive viewer with 3 montages (bipolar, average ref, Laplacian) — **DONE**
  - **Spatial extent comparison** (benchmarking against Tautan et al.)
    - Expert-expert spatial ICC: 0.845 (after removing SZ zeros)
    - PDCharacterizer (T=0.62): ICC=**0.852** (exceeds experts), MAE=0.095
    - Tautan: ICC=0.764, MAE=0.310
    - **Figure spatial scatter**: per-rater dots, 4 subtypes — **DONE**
    - **Figure IRR**: ICC/PA bars for spatial — **DONE**
    - **Figure threshold sweep**: threshold optimization curves — **DONE**
  - **Key finding**: Expert spatial extent ratings have poor IRR, motivating the discharge-locked topographic approach which bypasses subjective channel counting entirely
  - **Table 4**: Spatial extent ICC comparison (4 raters, threshold-optimized)

- **4.3 Frequency estimation**
  - **Quality-filtered evaluation** (MW-reviewed OR 3-expert consensus OR ≥80% IIIC agreement):
    - LPD: ρ=0.786, MAE=0.265 Hz (n=1,226)
    - GPD: ρ=0.846, MAE=0.172 Hz (n=1,089)
    - LRDA: ρ=0.674, MAE=0.233 Hz (n=640)
    - GRDA: ρ=0.712, MAE=0.215 Hz (n=1,310)
  - Label quality analysis: segments with <60% expert agreement have 2.4× higher discrepancy than those with 100% agreement
  - Discrepancy review: MW reviewed all |model - MW| > 0.5 Hz cases; 94% of PD and 61% of RDA accepted model over original MW label
  - **Figure 6**: 2×4 frequency scatter (PDCharacterizer vs Tautan et al., 4 subtypes) — **DONE**
  - **Figure IRR**: Inter-rater reliability comparison (ICC/PA bars) — expert-expert vs expert-algorithm for frequency and spatial extent, all 4 subtypes — **DONE**
  - **Table 5**: Frequency method comparison with quality-filtered labels

- **4.X RDA spatial localization**
  - RDA-PLV method: per-channel phase coherence with dominant hemisphere reference
  - ICC 0.371 matching expert-expert ICC 0.373 (n=211)
  - MAE 0.215, better than some expert-expert pairs (LB vs SZ MAE=0.484)

- **4.4 Discharge timing**
  - F1 0.506, Sensitivity 0.545, Precision 0.472, Timing MAE 25.4 ms (n=882)
  - **Figure 6**: LRDA/GRDA characterization examples — **DONE**
  - **Table 6**: Timing performance metrics

- **4.5 Unified PDCharacterizer evaluation**
  - All metrics on the same test set (expanded labels: 880 lat, 842 freq, 882 timing)
  - Preprocessing optimization contest: 10 variants tested, baseline wins
  - V1 confirmed best vs V3 retrained and E2E DETR attempt
  - **Table 7**: Unified comparison — V1 vs V3 vs preprocessing variants

### 5. Discussion

- **5.1 Key findings**
  - Unified pipeline achieves near-expert-level performance across all characterization tasks
  - Spatial localization matches expert inter-rater reliability (model as "virtual 4th rater")
  - Domain knowledge (DP) essential — end-to-end approaches failed with current data size
  - Curated training data quality > quantity (V1 on 815 patients > V2 on 8,060)

- **5.2 Methodological contributions**
  - "Contest of agents" approach for systematic algorithm comparison
  - Active learning HTML tools for efficient expert annotation
  - Pseudolabel framework enabling channel-level training from patient-level labels

- **5.3 Limitations**
  - Timing F1 (0.506) lower than early benchmarks (0.873) on expanded harder test set
  - RDA spatial localization not yet validated (single-rater labels only)
  - BIPD detection not yet implemented (21 confirmed cases, plan only)
  - End-to-end model needs >1000 labeled examples to compete

- **5.4 Comparison with prior work**
  - Tautan et al. 2025; other PD/RDA characterization methods
  - Novel: first system jointly characterizing all properties with expert-level benchmarks

- **5.5 Clinical implications and future directions**
  - Real-time ICU monitoring support; standardized quantitative ACNS descriptions
  - Future: BIPD analysis, RDA wave timing, phase-amplitude coupling, longitudinal tracking

### 6. Conclusions

### Supplementary Material

- **Figure S1**: HPP algorithm diagram — evidence → candidates → DP path → refined timing
- **Figure S2**: HemiCET architecture diagram (8ch → U-Net → evidence → DP → times)
- **Figure S3**: Comparison of handcrafted vs HemiCET evidence (3 example cases)
- **Figure S4**: Annotation tool screenshots (timing viewer, spatial viewer, frequency viewer)
- **Figure S5**: Feature extraction pipeline diagram
- **Figure S6**: Performance by review round (iterative label improvement)
- **Figure S7**: Failure mode examples (4 cases with commentary)
- **Figure S8**: Full contest leaderboards (lateralization 76 methods, spatial 30 methods, preprocessing 10 variants)
- **Figure S9**: End-to-end DETR model training curves (showing overfitting)
- **Figure S10**: V1 vs V3 vs preprocessing optimization results
- **Table S1**: Complete experiment results (all methods × all metrics)
- **Table S2**: Label coverage by subtype (from label_status_report)
- **Table S3**: Pseudolabel statistics by source and confidence
- **Table S4**: Per-fold cross-validation results
- **Table S5**: RDA lateralization contest full results (76 methods)
- **Suppl. Methods**: Detailed CNN architectures, hyperparameters, training procedures
- **Suppl. Code**: GitHub repository with all code, viewers, and reproducibility instructions

---

## Main Figures Summary (6 figures)

| Figure | Description | Status | File |
|--------|-------------|--------|------|
| Fig 1 | Raw EEG examples (6 panels: 4 clear + 2 ambiguous) | **DONE** | `fig1_eeg_examples.png` |
| Fig 2 | PD pipeline (ChannelPD-Net + HemiCET+DP + discharge-locked topo) | **TODO** | `fig2_pd_pipeline.png` |
| Fig 3 | RDA pipeline (W05 iterative Hilbert + PLV×Amp + narrowband topo) | **TODO** | `fig3_rda_pipeline.png` |
| Fig 4 | Frequency scatter (2×4: PDChar/W05 vs Tautan, quality-filtered) | **DONE** | `fig4_frequency_scatter.png` |
| Fig 5 | LPD characterization examples (easy/med/hard) | **DONE** | `fig5_lpd_characterization.png` |
| Fig 6 | GPD characterization examples | **DONE** | `fig6_gpd_characterization.png` |
| Fig 7 | LRDA characterization examples | **DONE** | `fig7_lrda_characterization.png` |
| Fig 8 | GRDA characterization examples | **DONE** | `fig8_grda_characterization.png` |
| Fig S1 | IRR comparison (ICC/PA bars, Tautan-style) | **DONE** | `figS1_irr_comparison.png` |
| Fig S2 | Spatial extent scatter (per-rater dots) | **DONE** | `figS2_spatial_scatter.png` |
| Fig S3 | Threshold optimization curves | **DONE** | `figS3_threshold_sweep.png` |

Each characterization figure shows 3 cases selected by IIIC crowd vote agreement (Easy ≥95%/80%, Medium 70-80%, Hard 45-60%). Panels: 18-channel bipolar EEG with discharge markers + hemisphere shading, MNE spherical spline topoplot (inferno, per-case normalized), ACNS 2021 verbal description.

Rendered via: `conda run -n morgoth python paper_materials/render_figures.py --pick '{"lpd":[1,15,9],"gpd":[17,2,9],"lrda":[11,3,6],"grda":[0,4,9]}'`

Optimized via 3-round Gemini critic loop (`paper_materials/optimize_figures.py`).

## Main Tables Summary (7 tables)

| Table | Description | Status |
|-------|-------------|--------|
| Table 1 | Dataset statistics by subtype (segments, patients, label coverage, multi-rater counts) | Can generate from label_status_report.py |
| Table 2 | PDCharacterizer component summary (architecture, params, training data, role) | TODO |
| Table 3 | Lateralization performance by subtype (PD AUC 0.984, RDA AUC 0.837) | **Data ready** |
| Table 4 | Spatial inter-rater Jaccard matrix (LB, PH, SZ, Model) | **DONE** |
| Table 5 | Frequency method comparison (Alexandra → CNN+ACF → HemiCET IPI) | **Data ready** |
| Table 6 | Discharge timing metrics (F1, Sens, Prec, MAE) | **Data ready** |
| Table 7 | Unified V1 vs V3 vs preprocessing variants comparison | **Data ready** |

## Supplementary Figures (S1–S10)

| Figure | Description | Status |
|--------|-------------|--------|
| S1 | HPP algorithm diagram (evidence → DP → timing) | TODO |
| S2 | HemiCET architecture (8ch → U-Net → evidence → DP → times) | TODO |
| S3 | Handcrafted vs HemiCET evidence comparison (3 cases) | **Data ready** |
| S4 | Annotation tool screenshots (timing, spatial, frequency viewers) | TODO (screenshots exist) |
| S5 | Feature extraction pipeline diagram | TODO |
| S6 | Performance by review round (iterative label improvement) | **Data ready** |
| S7 | Failure mode examples (4 cases with commentary) | TODO |
| S8 | Contest leaderboards (lateralization 76 methods, spatial 30, preprocessing 10) | **Data ready** |
| S9 | E2E DETR training curves (showing overfitting) | **DONE** (e2e_training_curves.html) |
| S10 | V1 vs V3 vs preprocessing optimization results | **Data ready** |

## Supplementary Tables (S1–S5)

| Table | Description | Status |
|-------|-------------|--------|
| S1 | Complete experiment results (all methods × all metrics) | **Data ready** |
| S2 | Label coverage by subtype (full report from label_status_report) | Can generate |
| S3 | Pseudolabel statistics by source and confidence | Partial |
| S4 | Per-fold CV results | **Data ready** |
| S5 | RDA lateralization contest full results (76 methods) | **DONE** |

---

## Priority Order for Remaining Work

### Tier 1 — Required for paper
1. ~~**PD timing**~~ — **DONE** (PDCharacterizer V1 F1=0.506 on expanded test set)
2. ~~**PD frequency**~~ — **DONE** (quality-filtered: LPD ρ=0.758, GPD ρ=0.767)
3. ~~**LPD laterality**~~ — **DONE** (AUC 0.984, 95% accuracy on 880 segments)
4. ~~**Spatial localization**~~ — **DONE** (Jaccard 0.731, 97.3% of human; threshold optimized)
5. ~~**Publication figures**~~ — **DONE** (Figs 1-2 characterization, Fig 5 spatial, Fig 6 frequency)
6. ~~**Frequency label review**~~ — **DONE** (313 PD + 299 RDA discrepancies reviewed, labels corrected)
7. ~~**GPD labeling expansion**~~ — **DONE** (808 GPD with freq+timing, balanced with 811 LPD)
8. **System diagram figure** (Fig 4) — PDCharacterizer pipeline overview
9. **Write methods + results sections** — using outline above

### Tier 2 — Strengthens paper
10. **RDA spatial labeling** — no multi-rater labels yet, PLV-based predictions ready
11. **RDA wave timing** — adapt HPP for RDA (design doc exists)
12. **LRDA/GRDA frequency expansion** — only 3 cases with ≥10 votes have expert freq
13. **Failure mode examples** (Fig S7) — select and annotate
14. **Algorithm diagrams** (Fig S1, S2) — HPP and HemiCET architecture

### Tier 3 — Future work
15. **BIPD timing + classifier** (21 confirmed cases, plan complete)
16. **End-to-end differentiable model** (plan at docs/PLAN_end_to_end_pdcharacterizer.md; needs >1000 labeled examples)
17. **Phase-amplitude coupling analysis**
18. **Longitudinal pattern tracking**

## Completion Summary (as of 2026-03-31)

| Category | Done | Remaining |
|----------|------|-----------|
| **PD Characterization** | PDCharacterizer V1: Lat AUC 0.984, Freq ρ 0.663, Timing F1 0.506, Spatial Jaccard 0.731 | System diagram figure |
| **RDA Characterization** | V5 lateralization contest (76 methods, AUC 0.837, Freq ρ 0.635) | Spatial labels, wave timing, freq labels |
| **Spatial Localization** | Inter-rater analysis (model=97.3% of human), threshold 0.38 optimized, 106 LPD labels | ~1750 LPD + 1034 GPD spatial labeling |
| **Labels** | 880 lat, 842 freq, 882 timing, 106 spatial (LPD); 1,214 lat (LRDA) | GPD/LRDA/GRDA spatial; RDA timing/freq expansion |
| **Model Evaluation** | V1 confirmed best; V3 retrained (worse); E2E DETR (failed); 10 preprocessing variants tested | — |
| **Figures** | 4 characterization PNGs (DONE), spatial agreement (DONE), Gemini-optimized | System diagram, freq scatter, supplementary figures |
| **Paper** | Revised outline complete, figures plan finalized | Writing, remaining figures/tables |
| **Tools** | Spatial labeler (topoplot+per-channel), lat+timing labeler, label status/ingest scripts, figure optimization loop | — |

## Notes (Historical)

The following notes document the evolution of our approach. Originally planned as a unified model, we pivoted to specialized models after finding they perform better individually.

Original note from MW:
 i don't want a "unified" model. 
 separate models that each do their job very well are fine, and already are a big advance. 

 so, here's are the tasks i want to accomplish: 

 1- LPD vs GPD classification
 2- LRDA vs GRDA classification
 3- channel identification: which channels have RDA?
 4- channel identification: which channels have PD?
 5- discharge peak timing for PDs
 6- RDA peak timing
 7- frequency estimation for RDA
 8- frequency estimation for PDs
 9- BIPD analysis - will specify later

 some notes about these (by numbers):
 
 1- when we use these algorithms, we will first apply an already extremely good model ("morgoth") that can tell use the pattern type (among: seizure, LPD, GPD, LRDA, GRDA). that's why distinguishing LPD, GPD, LRDA, GRDA is not very important for us to do. i still want to do LPD vs GPD here, because i think doing that well helps us to do other things well and is an important sanity check. 
 
 2- similar comment. 
 
 3- for this, we use the abundant pseudolabels to help ensure we do it well. but, it's important to fine tune the resulting model to ensure it's not overly aggressive in including too many channels. we can use the region labels from peter hadar, laura (forgot her last name), and sahar zafar, and the more recent channel labels that i provided, for this fine tuning. we might also need to label more examples to get this to work better. the ultimate goal here is to make the model able to spatially localize the LPDs or GPDs, and support verbal description of the localization. we have a bunch of rules that we developed to go from the channel identification to the verbal descriptions. 
 
 4- similar comments to #3. 
 
 5- our HPP model already works quite well for this. i would like to pursue creating a CNN that can provide an even better evidence trace, in hopes of getting even more accurate timing information, especially in difficult cases. we need to compare our current HPP model with the resulting CNN+HPP model. the CNN that would do this probably can be relatively simple. it needs to do these things: 
 - give narrowly concentrated signal / evidence around the discharge peaks
 - give near zero signal / evidence away from the peaks
 - give near zero signal / evidence on channels that don't have discharges -- so somehow we want this to be consistent with model #3, so that channels not involved don't get much evidence from the CNN. maybe this will just happen naturally if the CNN is already good at assigning evidence? or maybe if the CNN is good in this way, we can use it to help with channel localization? need to explore. 

 6- we haven't worked on this yet. i want to develop an HPP model for this similar to the one we have for the PDs. one thing that may help to provide an "evidence trace" to get this going is the previous method we developed for estimating the frequency of RDA, by finding a narrow-band "sinusoidal" approximation to the signal that maximizes varience explained. you should be able to find what we did along these lines previously. anyway the goal would be to put a marker at the beginning, end, and peak of each wave. this lets us e.g. estimate the phase of RDA, which will eventually be important for downstream tasks, e.g. identfying whether fast activity reliably occurs at a certain phase of RDA. 

7- i hypothesize that the CNN + HPP method will be best. we need to compare it with (a) alexandra's original method, (b) our recent best before HPP, (c) HPP (works extremely well!), (d) HPP + CNN. 

8- similar comments to 7. 

9- i will be getting a large collection of BIPDs. when i get these, we'll need to update our plan to distinguish LPDs vs GPDs vs BIPDs. probably we'll do this by using the CNN+HPP model, and trying to determine whether the phase and / or frequency of PDs are different on the left and right. 

please clean up the description of these plans, elaborate on them as needed, and make a systematic roadmap that we can follow to accomplish them all. include a list of figures and tables that we want to produce to write this up as a paper. that will help steer us to the goal. some of the things that i think will need to be included are: 
- good description of the tasks, and lots of examples (EEG images) showing for each task a range of difficulty, from easy to hard. want to show why the tasks are not trivial. some of this can go in the appendix / supplemental material. but we need abundant visual examples -- otherwise most people have no concept of why this is not easy. 
- description of the new labels, the methods and tools we developed to gather them (e.g. html tools; active learning). these are much richer labels than were available for our prior projects!
- description of the "contest of agents" approach with the leaderboard to finding good algorithms for the different tasks
- description and figure showing how the HPP method works
- description and figure showing how the CNN models training works
- description and figure showing how we can identify RDA frequency using the narrow-bandpass filter optimization method (incidentally, we need to come up with good names for each of the major methods that have good abbreviations -- similar to "HPP")
- diagram explaining how the HPP algorithm works
- table with performance metrics comparisons for all the different tasks
- scatter plots for tasks where that's appropriate
- other performance metrics plots
- plots showing successful operation of the algorithms
- plots showing algorithm failure modes, and commentary on why we think these cases happen, and opportunities for future improvements

please improve this plan, and include the paper outline / plan in our paper roadmap document. 
