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

### Task 4: RDA Channel Identification
**Goal**: Same as Task 3 but for rhythmic delta activity.
**Context**: Enables spatial localization of LRDA/GRDA patterns.
**Training data**: Pseudolabels only so far (GRDA all channels positive, LRDA all channels positive as crude approximation). Need LRDA laterality annotations to improve.
**Current best**: 0.842 channel AUC (from unified model, pseudolabels only).
**Approach**: Same as Task 3 but with RDA labels. Key need: annotate LRDA laterality (99 cases) to enable proper ipsilateral/contralateral pseudolabel split.

### Task 5: PD Discharge Timing
**Goal**: Mark the precise time of each periodic discharge on each involved channel: t_1, t_2, ..., t_K.
**Context**: Enables IPI-derived frequency estimation (more accurate than FFT/ACF), regularity assessment (ACNS "nearly regular" criterion), and downstream BIPD analysis (comparing timing between hemispheres).
**Training data**: 593 MW-reviewed cases with discharge times (5,643 total events, mean 9.5/case). Complete ground truth.
**Current best**: HPP algorithm, F1=0.795, freq Spearman 0.935 vs MW, timing accuracy 2ms.
**Approach — HPP (Hidden Point Process)**:
1. Build per-channel evidence signal E(t) from pointiness + TKEO features
2. Aggregate by subtype (bilateral for GPD, lateralized for LPD)
3. Detect active interval (where PDs are present)
4. Extract candidate peaks
5. Dynamic programming with approximately-periodic prior (allows skipped discharges)
6. EM template refinement

**Approach — CET+HPP (CNN Evidence Trace + HPP)**:
Train a CNN to output a per-channel, per-time-step discharge evidence signal. This replaces the handcrafted pointiness+TKEO with a learned, case-adaptive evidence trace. The HPP DP inference then operates on the CNN evidence instead of handcrafted features.
- CNN should produce: high, narrowly concentrated signal at discharge peaks; near-zero signal away from peaks; near-zero on uninvolved channels
- The CNN's channel-level behavior should be consistent with Task 3 (PD channel identification) — channels identified as uninvolved should not produce discharge evidence
- Explore: can the CNN evidence trace serve double duty for both timing (Task 5) and channel identification (Task 3)?

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
**Training data**: 14 GRDA + 4 LRDA with gold standard freq (very sparse). Need more.
**Current best**: FFT peak frequency, Spearman 0.840 on 23 patients (from v7).
**Approaches to compare**:
- (a) Alexandra's original method (FFT/FOOOF)
- (b) NVO — narrowband variance optimization
- (c) HPP with handcrafted evidence → IPI-derived frequency
- (d) CET+HPP — CNN evidence + HPP → IPI-derived frequency

### Task 8: PD Frequency Estimation
**Goal**: Estimate the frequency (Hz) of periodic discharges.
**Context**: Core deliverable. Component of ACNS 2021 description.
**Training data**: 594 cases with gold standard freq (425 LPD, 151 GPD).
**Current best**: CNN+Attention Spearman 0.640; HPP IPI-derived freq Spearman 0.935 vs MW timing.
**Approaches to compare**:
- (a) Alexandra's original method (Spearman ~0.47 on original dataset)
- (b) Handcrafted features + Ridge regression (Spearman 0.589)
- (c) Handcrafted features + RF/GBM (Spearman 0.604)
- (d) CNN+Attention on single channels (Spearman 0.640)
- (e) HPP with handcrafted evidence → IPI-derived frequency (Spearman 0.935 vs MW)
- (f) CET+HPP — CNN evidence + HPP → IPI-derived frequency (expected best)

### Task 9: BIPD Analysis
**Goal**: Detect bilateral independent periodic discharges — LPDs occurring independently on left and right hemispheres with different timing/frequency.
**Context**: BIPDs have distinct clinical significance from unilateral LPDs or bilateral GPDs.
**Training data**: Coming soon (MW will provide BIPD collection).
**Approach**: Use the spatial (Task 3) + timing (Task 5) outputs:
- If left and right hemisphere channels both show PDs (from Task 3)
- But the discharge timing sequences are NOT synchronized (from Task 5)
- And/or the frequencies differ between hemispheres
- → Flag as BIPD candidate
**Dependencies**: Requires Tasks 3 and 5 to work well first.

---

## Method Names and Abbreviations

| Abbreviation | Full Name | Description |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge/wave timing with periodic prior |
| **CET** | CNN Evidence Trace | Learned per-channel evidence signal replacing handcrafted features |
| **NVO** | Narrowband Variance Optimization | Sinusoidal template fitting for RDA frequency/phase estimation |
| **SPF** | Signal Processing Features | Handcrafted features (pointiness, ACF, FFT, TKEO, coherence) |

---

## Implementation Roadmap

### Phase 1: PD Tasks (Tasks 1, 3, 5, 8) — Mostly complete

| Step | Task | Status | What remains |
|------|------|--------|-------------|
| 1.1 | PD frequency (SPF baselines) | Done | 42 experiments, RF best at 0.604 |
| 1.2 | PD frequency (CNN+Attention) | Done | 0.640 Spearman |
| 1.3 | PD timing (HPP) | Done | F1=0.795, 593 GT cases |
| 1.4 | PD channel identification | Partial | 304 GT, 290 pending review. CNN needs calibration. |
| 1.5 | LPD vs GPD classification | Done | RF AUC 0.931 |
| **1.6** | **PD timing (CET+HPP)** | **TODO** | Train CNN evidence head, compare with HPP-only |
| **1.7** | **PD frequency (CET+HPP → IPI)** | **TODO** | Compare IPI-derived freq from CET+HPP vs other methods |
| **1.8** | **Complete spatial review** | **TODO** | Review remaining 290 cases, retrain channel detector |

### Phase 2: RDA Tasks (Tasks 2, 4, 6, 7) — Mostly TODO

| Step | Task | Status | What remains |
|------|------|--------|-------------|
| **2.1** | **Find NVO code** | **TODO** | Locate existing narrowband sinusoidal fitting code |
| **2.2** | **LRDA laterality annotation** | **TODO** | MW annotates 99 LRDA cases |
| **2.3** | **RDA channel identification** | Partial | 0.842 AUC from pseudolabels. Need LRDA laterality for better pseudolabels. |
| **2.4** | **RDA wave timing (HPP)** | **TODO** | Adapt HPP for RDA waves (onset/peak/offset markers) |
| **2.5** | **RDA frequency (NVO)** | **TODO** | Benchmark NVO against FFT baseline |
| **2.6** | **RDA frequency (HPP → IPI)** | **TODO** | Compare with NVO |
| **2.7** | **LRDA vs GRDA classification** | **TODO** | Benchmark with laterality features |
| **2.8** | **RDA timing (CET+HPP)** | **TODO** | CNN evidence for RDA + HPP |

### Phase 3: High-Frequency Integration (Sprint 5)

| Step | What | Status |
|------|------|--------|
| 3.1 | Seizure harvest completes | In progress (31%, 278 kept, 2/3 bins full) |
| 3.2 | Frequency annotation of harvested cases | TODO |
| 3.3 | Subtype verification | TODO |
| 3.4 | Discharge timing for new cases | TODO |
| 3.5 | Retrain all PD models with expanded data | TODO |

### Phase 4: BIPD Analysis (Task 9)

| Step | What | Status |
|------|------|--------|
| 4.1 | Receive BIPD training cases from MW | Waiting |
| 4.2 | Develop BIPD detection rules | TODO |
| 4.3 | Evaluate on BIPD cases | TODO |

### Phase 5: Paper Writing

| Step | What |
|------|------|
| 5.1 | Generate all figures and tables |
| 5.2 | Write methods section |
| 5.3 | Write results section |
| 5.4 | Write discussion |
| 5.5 | Internal review |

---

## Paper Outline

### Title
"Comprehensive Automated Characterization of Periodic and Rhythmic EEG Patterns: Discharge Timing, Spatial Localization, and Frequency Estimation"

### Abstract
- Problem: PD and RDA characterization is time-consuming and subjective
- What we did: suite of specialized algorithms for 9 tasks
- Key results: HPP timing F1=0.795, frequency Spearman 0.94, channel AUC 0.87, etc.
- Significance: first system to provide discharge-level timing and spatial localization

### 1. Introduction
- EEG monitoring in ICU
- ACNS 2021 terminology for periodic and rhythmic patterns
- Current limitations: frequency estimation is crude, no discharge-level timing, spatial localization is qualitative
- What this paper contributes

### 2. Data and Annotations
- **2.1 Dataset overview** — 839 patients, 4 pattern types, data sources
- **2.2 Annotation framework** — description of the multi-layer label system:
  - Patient-level: subtype, frequency, laterality
  - Channel-level: PD/RDA involvement per channel
  - Discharge-level: precise timing of each discharge/wave
- **2.3 Active learning annotation tools** — HTML-based interactive viewers for:
  - Binary review (correct/incorrect with C/I keyboard shortcuts)
  - Interactive marker editing (canvas-based, click-to-add/delete/move)
  - Frequency annotation (text input with live IPI feedback)
  - Spatial review (click channels to toggle involvement)
  - Iterative refinement workflow: algorithm auto-labels → human binary review → human correction → retrain → repeat
- **2.4 Pseudolabel framework** — leveraging patient-level labels to create channel-level training data:
  - LPD laterality → ipsilateral/contralateral channel labels
  - GPD → all channels (with caveats from MW review)
  - LRDA/GRDA as cross-pattern negatives
  - Confidence weighting for differential training signal
- **Figure 1**: Example EEG segments showing range of difficulty for each pattern type (LPD, GPD, LRDA, GRDA), from obvious to subtle. 8-panel figure.
- **Figure 2**: Annotation tool screenshots — timing correction viewer, spatial review viewer, frequency disagreement viewer.
- **Table 1**: Dataset statistics — patients, segments, label types, counts per category.

### 3. Methods

- **3.1 Signal processing features (SPF)**
  - Pointiness trace computation
  - ACF frequency estimation
  - FFT, TKEO, coherence features
  - Laterality index and hemisphere energy ratio
  - **Figure 3**: Feature extraction pipeline diagram

- **3.2 Hidden Point Process (HPP) algorithm**
  - Problem formulation: MAP inference over latent discharge sequence
  - Evidence signal construction (per-channel, class-aware aggregation)
  - Active interval detection
  - Candidate peak extraction
  - Dynamic programming with approximately-periodic prior
  - Skip modeling (allowing 1-3 missed discharges)
  - EM template refinement
  - **Figure 4**: HPP algorithm diagram — showing evidence signal → candidates → DP path → refined timing
  - **Figure 5**: Example HPP results — 4 cases showing detected discharge times overlaid on EEG, from easy (clear periodic) to difficult (irregular, partial window)

- **3.3 CNN Evidence Trace (CET)**
  - Architecture: per-channel 1D CNN encoder with temporal attention
  - Training: supervised on MW-reviewed discharge times
  - Output: frame-level discharge probability per channel
  - Integration with HPP: CET replaces handcrafted evidence in the HPP pipeline
  - **Figure 6**: CET architecture diagram
  - **Figure 7**: Comparison of handcrafted evidence vs CNN evidence traces for 3 example cases

- **3.4 Narrowband Variance Optimization (NVO)** for RDA
  - Sinusoidal template fitting across frequency range
  - Variance-explained criterion
  - Phase estimation → wave onset/peak/offset timing
  - **Figure 8**: NVO method illustration — raw EEG, fitted sinusoid, extracted wave markers

- **3.5 Channel identification models**
  - CNN architecture for per-channel PD/RDA detection
  - Pseudolabel training with confidence weighting
  - Fine-tuning with ground truth labels
  - Threshold calibration

- **3.6 Classification models**
  - LPD vs GPD: feature-based (RF/GBM)
  - LRDA vs GRDA: similar approach
  - Role of Morgoth as upstream classifier

- **3.7 Optimization framework**
  - "Contest of agents" approach
  - Live dashboard for tracking experiments
  - LOPO cross-validation methodology
  - Bootstrap confidence intervals

### 4. Results

- **4.1 PD Frequency Estimation**
  - **Table 2**: Comparison of all methods (Alexandra's original, SPF+Ridge, SPF+RF, CNN+Attention, HPP, CET+HPP)
  - **Figure 9**: Scatter plots — gold standard vs predicted frequency for best method, colored by subtype (LPD green, GPD blue), separate panels for LPD and GPD
  - **Figure 10**: Spearman progression across method iterations (bar chart)

- **4.2 RDA Frequency Estimation**
  - **Table 3**: Comparison (Alexandra's FFT, NVO, HPP, CET+HPP)
  - **Figure 11**: Scatter plot for RDA frequency

- **4.3 Discharge Timing Detection**
  - **Table 4**: HPP performance (sensitivity, precision, F1, timing accuracy)
  - **Table 5**: CET+HPP vs HPP-only comparison
  - **Figure 12**: Example discharge timing results — 4 cases with discharge markers overlaid on EEG
  - **Figure 13**: IPI-derived frequency vs gold standard scatter plot (ρ=0.970)

- **4.4 RDA Wave Timing**
  - **Table 6**: HPP-RDA performance metrics
  - **Figure 14**: Example RDA wave timing — onset/peak/offset markers

- **4.5 Spatial Localization**
  - **Table 7**: Channel-level PD/RDA detection AUC
  - **Figure 15**: Example spatial localization results — EEG with channels colored by predicted involvement
  - Comparison with expert spatial annotations

- **4.6 Subtype Classification**
  - **Table 8**: LPD vs GPD, LRDA vs GRDA classification metrics
  - Confusion matrices

- **4.7 BIPD Detection** (if data available)
  - Proof of concept results

- **4.8 Iterative label improvement analysis**
  - **Figure 16**: How algorithm performance improved across review rounds (timing F1 by round)
  - **Table 9**: Label statistics per review round

### 5. Discussion

- **5.1 Key findings**
  - HPP + structural priors outperform pure learned approaches for timing
  - CET+HPP hybrid combines CNN's pattern recognition with HPP's structural reasoning
  - Pseudolabels enable training with limited ground truth
  - Active learning dramatically reduces annotation effort

- **5.2 PD detection orthogonality finding**
  - PD probability does not predict frequency estimation accuracy
  - Implications for model design

- **5.3 Failure modes and limitations**
  - **Figure 17**: Algorithm failure cases — 4 examples with commentary:
    - Very slow PDs (< 0.3 Hz) — few training examples, hard to distinguish from background
    - Variable morphology — discharges that change shape across the segment
    - Bilateral LPDs with asymmetric involvement — spatial localization ambiguity
    - Transition patterns — PDs evolving into seizure or vice versa
  - High-frequency gap (few cases > 2.5 Hz)
  - LRDA classification weakness
  - Label noise in gold standard frequency

- **5.4 Comparison with prior work**
  - Tautan et al. 2025 (our prior paper)
  - Other automated PD/RDA characterization methods
  - What's new: discharge-level timing, channel-level spatial labels, HPP algorithm

- **5.5 Clinical implications**
  - Real-time ICU monitoring support
  - Standardized quantitative descriptions
  - Foundation for BIPD detection

- **5.6 Future directions**
  - BIPD analysis
  - Phase-amplitude coupling (fast activity at specific RDA phase)
  - Longitudinal tracking (pattern evolution over time)
  - Integration with clinical outcome prediction

### 6. Conclusions

### Supplementary Material

- **Suppl. Figure S1-S8**: Extended EEG examples for each task (5 per task, showing range of difficulty)
- **Suppl. Figure S9**: Full optimization dashboard screenshot
- **Suppl. Figure S10**: All 42+ experiment results from the "contest of agents"
- **Suppl. Table S1**: Complete experiment results table (all methods × all metrics)
- **Suppl. Table S2**: Pseudolabel statistics by source and confidence level
- **Suppl. Table S3**: Per-fold cross-validation results
- **Suppl. Methods**: Detailed CNN architectures, hyperparameters, training procedures
- **Suppl. Code**: GitHub repository link with all code, viewers, and reproducibility instructions

---

## Figures Summary

| Figure | Description | Status |
|--------|-------------|--------|
| Fig 1 | EEG examples by pattern type and difficulty | TODO |
| Fig 2 | Annotation tool screenshots | TODO (screenshots exist) |
| Fig 3 | Feature extraction pipeline | TODO |
| Fig 4 | HPP algorithm diagram | TODO |
| Fig 5 | HPP example results (4 cases) | TODO |
| Fig 6 | CET architecture diagram | TODO |
| Fig 7 | Handcrafted vs CNN evidence comparison | TODO (needs CET) |
| Fig 8 | NVO method illustration | TODO (needs NVO) |
| Fig 9 | PD frequency scatter plots | Partial (dashboard has these) |
| Fig 10 | Spearman progression bar chart | TODO |
| Fig 11 | RDA frequency scatter plot | TODO (needs RDA work) |
| Fig 12 | Discharge timing examples | Partial (viewer exists) |
| Fig 13 | IPI vs gold standard scatter | Done (gold_vs_ipi_frequency.html) |
| Fig 14 | RDA wave timing examples | TODO (needs RDA HPP) |
| Fig 15 | Spatial localization examples | Partial (viewer exists) |
| Fig 16 | Performance by review round | TODO |
| Fig 17 | Failure mode examples | TODO |

## Tables Summary

| Table | Description | Status |
|-------|-------------|--------|
| Table 1 | Dataset statistics | Done (in approach reviews) |
| Table 2 | PD frequency method comparison | Partial |
| Table 3 | RDA frequency method comparison | TODO |
| Table 4 | HPP timing performance | Done |
| Table 5 | CET+HPP vs HPP comparison | TODO (needs CET) |
| Table 6 | RDA HPP performance | TODO |
| Table 7 | Channel detection AUC | Partial |
| Table 8 | Subtype classification | Done |
| Table 9 | Label stats by review round | Done (in v10 review) |

---

## Priority Order for Remaining Work

### Tier 1 — Required for paper
1. **CET+HPP for PD timing** (Phase 1.6-1.7) — the key novelty claim
2. **Complete PD spatial review** (Phase 1.8) — need solid channel AUC numbers
3. **NVO for RDA** (Phase 2.1, 2.5) — find existing code, benchmark
4. **RDA HPP** (Phase 2.4, 2.6) — adapt HPP for RDA waves
5. **Generate all figures** (Phase 5.1)

### Tier 2 — Strengthens paper
6. **LRDA laterality annotation** (Phase 2.2)
7. **RDA channel identification** (Phase 2.3)
8. **High-frequency case integration** (Phase 3)
9. **LRDA vs GRDA classification** (Phase 2.7)

### Tier 3 — Can be future work
10. **BIPD analysis** (Phase 4)
11. **CET+HPP for RDA** (Phase 2.8)
12. **Phase-amplitude coupling analysis**

====== NOTES ========

i have changed my mind about the goal. 
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
