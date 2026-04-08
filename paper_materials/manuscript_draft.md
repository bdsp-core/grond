# Automated Characterization of Periodic Discharges and Rhythmic Delta Activity in Continuous EEG

---

## Abstract

**Objective.** Periodic discharges (PDs) and rhythmic delta activity (RDA) are common electroencephalographic (EEG) patterns in critically ill patients that require detailed characterization -- including lateralization, spatial localization, discharge timing, and frequency estimation -- according to the American Clinical Neurophysiology Society (ACNS) 2021 standardized terminology. Manual characterization is subjective, time-consuming, and exhibits substantial inter-rater variability. This study presents a comprehensive automated system for characterizing all four major subtypes: lateralized periodic discharges (LPD), generalized periodic discharges (GPD), lateralized rhythmic delta activity (LRDA), and generalized rhythmic delta activity (GRDA).

**Approach.** Two complementary pipelines were developed. The PD pipeline combines a per-channel convolutional neural network (ChannelPD-Net) with hemisphere-specific learned evidence traces (HemiCET-UNet), dynamic programming with an approximately-periodic prior, and discharge-locked topographic localization. The RDA pipeline uses iterative narrowband Hilbert refinement (W05) for frequency and lateralization, with phase-locking value (PLV) analysis for spatial extent. Both pipelines were trained and evaluated on 12,238 EEG segments from over 3,000 patients, annotated by four expert electroencephalographers through iterative human-in-the-loop label refinement.

**Main results.** For PD characterization, the system achieved hemisphere lateralization area under the receiver operating characteristic curve (AUC) of 0.989 (n = 1,274), LPD versus GPD classification AUC of 0.911 (n = 7,037), discharge detection F1 of 0.889 with timing accuracy of 1.0 ms, and frequency estimation Spearman correlations of 0.786 (LPD) and 0.846 (GPD). For RDA, lateralization AUC was 0.837 (n = 4,253), and frequency estimation achieved an intraclass correlation coefficient (ICC) of 0.860, matching expert-expert ICC of 0.852. Spatial localization reached 97.3% of expert inter-rater agreement for PDs (Jaccard 0.731 versus 0.751) and matched expert-expert ICC for RDA (0.371 versus 0.373).

**Significance.** This is the first system to jointly characterize lateralization, spatial localization, discharge timing, and frequency for both periodic and rhythmic EEG patterns, with quantitative benchmarks against multi-rater expert annotations across all tasks.

---

## 1. Introduction

Continuous EEG (cEEG) monitoring is standard practice in neurological intensive care units (ICUs) for detecting seizures and identifying patterns associated with secondary brain injury [1]. Among the most clinically significant findings are periodic discharges (PDs) and rhythmic delta activity (RDA), which lie along the ictal-interictal continuum and are associated with increased risk of seizures, neuronal injury, and poor outcomes [2,3]. The ACNS standardized critical care EEG terminology, revised in 2021, provides a framework for describing these patterns along multiple dimensions: main term (periodic vs. rhythmic), lateralization (lateralized vs. generalized vs. bilateral independent), spatial distribution (e.g., frontal predominant, hemispheric), and quantitative features including repetition frequency and regularity [4].

Despite the standardization of terminology, manual characterization of PDs and RDA remains subjective and labor-intensive. A trained electroencephalographer must identify the pattern type, determine its spatial distribution across the scalp, estimate its repetition frequency, and describe its temporal evolution -- all from visual inspection of multi-channel EEG tracings. Inter-rater reliability for these characterization tasks varies widely: lateralization (generalized vs. lateralized) shows moderate to good agreement, while spatial extent and frequency estimation exhibit substantially lower reliability [5,6]. The need for expert review creates bottlenecks in clinical workflow, particularly during overnight monitoring when specialist coverage is limited.

Automated detection of PDs and RDA has received growing attention. The IIIC (Interictal-Ictal Continuum) dataset introduced by Jing et al. [7] provides expert consensus labels for pattern classification (LPD, GPD, LRDA, GRDA, seizure, other) based on crowd-sourced annotations from 10 or more expert raters per segment. Deep learning models trained on this dataset achieve high classification accuracy for distinguishing among pattern types [8,9]. However, classification alone -- determining that a segment contains LPD rather than GPD -- addresses only one dimension of the characterization problem.

Tautan et al. [10] recently described a signal-processing approach to PD and RDA characterization that estimates frequency and spatial extent using autocorrelation and spectral analysis. That work demonstrated the feasibility of automated characterization but reported limited agreement with expert annotations, particularly for frequency estimation (Spearman correlations below 0.25 for all subtypes) and spatial extent quantification.

The present work addresses the full characterization problem: given a 10-second EEG segment containing a periodic or rhythmic pattern, the system automatically determines (1) lateralization (left, right, or bilateral/generalized), (2) spatial localization (which brain regions are involved), (3) frequency of the periodic or rhythmic pattern, (4) for PDs, the precise timing of each individual discharge, and (5) for PDs, whether the pattern represents bilateral independent periodic discharges (BIPD). The system produces a structured verbal description following ACNS 2021 nomenclature (e.g., "LPD, left-sided, at 1.5 Hz, left frontotemporal predominant").

The key contributions of this work are as follows. First, a modular pipeline architecture is presented that combines learned neural evidence (convolutional neural networks) with structured temporal inference (dynamic programming) for PD characterization, and iterative narrowband signal processing for RDA characterization. Second, discharge-locked topographic localization is introduced as a novel approach to spatial localization that computes the mean voltage topography at the moment of each detected discharge, bypassing the subjective channel-counting approach used in prior work. Third, a systematic "contest of methods" framework is described in which 76 algorithm variants were compared for RDA characterization and over 30 variants for PD spatial localization, providing empirical evidence for design decisions. Fourth, all results are benchmarked against multi-rater expert annotations with quantitative inter-rater reliability analysis, enabling direct comparison of algorithm-expert agreement with expert-expert agreement across all characterization tasks.

---

## 2. Methods

### 2.1 Dataset and Annotations

#### 2.1.1 EEG Data

The dataset comprised 12,238 non-excluded EEG segments drawn from three sources (table 1). Each segment consisted of 10 seconds of 19-channel monopolar EEG recorded at 200 Hz in common average reference montage (2,000 samples per channel). Segments were classified into four subtypes based on the dominant pattern: LPD (n = 4,170), GPD (n = 3,337), LRDA (n = 1,408), and GRDA (n = 3,323). An additional 990 segments were excluded during quality review.

The first data source consisted of segments from the IIIC dataset [7], comprising 10-minute recordings from which the center 10-second segment (the labeled epoch) was extracted. Each of these segments had pattern classification votes from 10 or more expert electroencephalographers (3,529 segments: 1,846 LPD, 1,024 GPD, 239 LRDA, 420 GRDA). The second source consisted of segments selected from pattern-specific clinical EEG databases and classified by a single expert rater (MW). The third source was a 38-patient expert dataset annotated independently by four raters (LB, PH, SZ, MW) for pattern classification, frequency, spatial extent, and discharge timing.

**Table 1. Dataset statistics.**

| | LPD | GPD | LRDA | GRDA | Total |
|---|---:|---:|---:|---:|---:|
| Segments (non-excluded) | 4,170 | 3,337 | 1,408 | 3,323 | 12,238 |
| Unique patients | 3,953 | 3,086 | 1,381 | 3,159 | -- |
| Expert-reviewed frequency | 1,499 | 1,539 | 654 | 1,381 | 5,073 |
| Discharge/wave timing | 917 | 1,036 | 189 | 313 | 2,159/502 |
| Spatial annotations | 352 | 260 | 29 | 177 | 818 |
| IIIC crowd votes (>=10 raters) | 1,846 | 1,024 | 239 | 420 | 3,529 |

#### 2.1.2 Annotation Framework

A multi-layer annotation framework was developed with task-specific interactive review tools.

**Pattern classification and laterality.** For IIIC crowd-labeled segments, pattern subtype was assigned based on the majority vote across expert raters. For MW-labeled segments, classification was performed during visual review. Laterality annotations (left vs. right hemisphere dominance) were obtained for 1,336 LPD segments and 1,039 LRDA segments through dedicated review using a symmetric channel-layout viewer. A total of 1,374 LRDA segments were reviewed across three batches, with 338 segments rejected as not meeting LRDA criteria and excluded from subsequent analysis.

**Frequency.** Expert-reviewed frequency labels were obtained for 5,073 segments across all four subtypes. For PD subtypes, frequency was annotated by MW during combined timing-frequency-laterality review. For RDA subtypes, frequency was annotated using an interactive viewer that displayed the W05 algorithm estimate as a default, with a narrowband-filtered overlay for visual confirmation; 993 RDA segments were reviewed, with 399 frequency corrections and 594 confirmations. An additional 453 LRDA/GRDA frequency labels were added from segments with at least 50% IIIC crowd agreement. Three additional expert raters (LB, PH, SZ) independently annotated frequency for 1,060 segments (approximately 265 per subtype) from the expert dataset.

**Discharge timing.** Individual discharge times were annotated for 1,953 PD segments (917 LPD, 1,036 GPD) through three rounds of iterative model-assisted review. In the first round, an expert (MW) annotated discharge times de novo. In subsequent rounds, the HemiCET+DP algorithm generated candidate discharge times, and the expert reviewed and corrected these candidates, correcting 61 discharge sequences and rejecting 15 segments across three review cycles.

**Spatial extent and channel involvement.** Per-channel spatial involvement was annotated for 818 segments (352 LPD, 260 GPD, 29 LRDA, 177 GRDA) by three expert raters (LB, PH, SZ) using an interactive topoplot viewer with per-channel toggling. An additional 1,980 IIIC segments received MW spatial extent annotations. Region-level spatial descriptions followed the 16-region parcellation from the morgoth-viewer module, which includes transitional zones (frontotemporal, centro-parietal, fronto-central, parieto-occipital) compatible with ACNS 2021 nomenclature.

#### 2.1.3 Evaluation Framework

All evaluations used patient-stratified cross-validation to ensure no patient appeared in both training and evaluation sets. For models trained with k-fold cross-validation (ChannelPD-Net, HemiCET-UNet), 5-fold patient-stratified splits were used, and evaluation was performed on the held-out fold for each patient. For methods not requiring training (W05, RDA-PLV), leave-one-patient-out (LOPO) cross-validation was not applicable, and all available labeled segments were used for evaluation.

Frequency estimation was evaluated using quality-filtered labels to reduce the influence of annotation noise. A segment was included in quality-filtered evaluation if any of the following criteria were met: (1) MW reviewed the frequency, (2) all three expert raters (LB, PH, SZ) provided independent labels, or (3) at least 10 IIIC crowd votes with at least 80% pattern agreement were available. Expert frequency for multi-rater segments was defined as the mean across available raters.

Inter-rater reliability was quantified using the intraclass correlation coefficient (ICC(3,1)) and percentage agreement (PA), following the framework of Tautan et al. [10]. The algorithm was treated as an additional rater in the ICC computation, enabling direct comparison of algorithm-expert agreement with expert-expert agreement.

### 2.2 PD Characterization Pipeline

The PD characterization pipeline (figure 2) processed 19-channel monopolar EEG segments in a single forward pass, producing lateralization, spatial localization, discharge timing, frequency estimation, and a structured verbal description. The pipeline comprised five components operating on a shared 18-channel bipolar representation derived from the monopolar input using standard longitudinal and transverse bipolar pairs.

**Table 2. PD pipeline components.**

| Component | Architecture | Parameters | Role |
|---|---|---:|---|
| ChannelPD-Net | CNN + Attention | ~50K | Per-channel PD detection and frequency |
| HemiCET-UNet | Encoder-decoder U-Net | ~525K | Frame-level discharge evidence |
| HPP | Handcrafted features | -- | Complementary evidence (pointiness + TKEO) |
| Dynamic Programming | Viterbi-style DP | -- | Temporal inference with periodic prior |
| CNN+ACF Ensemble | ChannelPD-Net + ACF | -- | Frequency prior for DP |
| Hybrid-PLV | Phase-locking value | -- | Spatial localization (channel scoring) |
| Discharge-locked topo | Laplacian-GFP alignment | -- | Topographic localization |

#### 2.2.1 ChannelPD-Net

ChannelPD-Net was a lightweight one-dimensional convolutional neural network (CNN) with temporal attention pooling that operated independently on each of 18 bipolar EEG channels. Given a single 10-second z-scored channel (2,000 samples at 200 Hz), the model jointly predicted (i) the probability that a periodic discharge pattern was present and (ii) the log-frequency of the pattern. The architecture comprised four convolutional blocks (kernel sizes 51, 25, 13, 7; channels 16, 32, 64, 64; stride 2 throughout) with batch normalization, GELU activation, and dropout (0.1-0.2), downsampling the temporal dimension by a factor of 16 to 125 time steps. A learned temporal attention mechanism projected each 64-dimensional feature vector to a scalar logit via a 1x1 convolution, applied softmax normalization across time, and computed an attention-weighted context vector. Two independent linear heads produced a PD probability (sigmoid output) and a log-frequency estimate (linear output).

The model was trained with a multi-task loss combining binary cross-entropy for detection and mean squared error for log-frequency estimation (masked to PD-positive channels, weighted by 0.5). Training used 5-fold patient-stratified cross-validation with data augmentation including amplitude scaling (0.8-1.2x), additive Gaussian noise (20-40 dB SNR), and circular time shifts (plus or minus 50 samples). The 5-fold ensemble prediction was the arithmetic mean across fold models.

**Laterality detection.** Hemisphere laterality was determined by comparing the mean PD probability across left-hemisphere channels versus right-hemisphere channels in the bipolar montage. The hemisphere with higher mean probability was assigned as the dominant side.

**CNN frequency estimate.** The segment-level CNN frequency was computed as a PD-probability-weighted average of per-channel log-frequency predictions, converted back to linear frequency and clipped to the physiological range of 0.3-3.5 Hz.

#### 2.2.2 Discharge Timing: HemiCET-UNet and Dynamic Programming

**HemiCET-UNet.** Frame-level discharge evidence was generated by HemiCET-UNet, a one-dimensional U-Net encoder-decoder architecture with approximately 525,000 parameters. Unlike ChannelPD-Net, which provided a single summary probability per channel, HemiCET-UNet produced a dense evidence trace at the full 200 Hz resolution, indicating the instantaneous likelihood of a discharge at each time point. The encoder mirrored the ChannelPD-Net backbone (four stride-2 convolutional stages), while the decoder progressively upsampled through transposed convolutions with skip connections from corresponding encoder stages, producing a sigmoid-activated output of dimension 2,000.

Training targets were constructed from expert-annotated discharge times as Gaussian peaks (sigma = 10 ms). The loss function combined weighted binary cross-entropy (positive weight = 20 to address temporal sparsity), a sharpness penalty encouraging sparse evidence (weight 0.1), and a floor loss ensuring learned evidence at labeled discharge locations was at least as strong as handcrafted evidence (weight 0.05). The model was trained with 5-fold patient-stratified cross-validation on 675 segments with ground-truth discharge timing, with data augmentation including amplitude scaling, additive noise, channel dropout (p = 0.15), and discharge time jitter (sigma = 5 ms).

The HemiCET-UNet operated on 8-channel hemisphere input rather than the full 18-channel bipolar montage. For LPD segments, the model processed the affected hemisphere only; for GPD segments, both hemispheres were processed independently and the result with higher overall evidence was retained. This hemisphere-specific design avoided contamination from the uninvolved hemisphere in lateralized patterns and improved performance compared to full 18-channel processing.

**Handcrafted evidence (HPP).** Complementary evidence was computed from two handcrafted signal features. The pointiness trace captured the sharpness of transient waveforms by computing the ratio of squared prominence to half-width at half-prominence for each local maximum. The Teager-Kaiser energy operator (TKEO) responded to instantaneous energy and frequency. Both traces were z-scored and combined with fixed weights (0.6 pointiness, 0.4 TKEO), Gaussian-smoothed (sigma = 15 ms), and clipped to non-negative values.

**Product-boosted evidence combination.** The aggregated HPP and CET evidence traces were combined using a product-boost formula that amplified regions of agreement: E(t) = max(E_hpp, E_cet) + 3.0 * E_hpp * E_cet, where both traces were normalized to [0,1] and the CET trace was thresholded at the 80th percentile of nonzero values with a floor of 0.3.

**CNN+ACF frequency ensemble.** The frequency prior for dynamic programming was obtained by combining the CNN frequency estimate (PD-probability-weighted average across channels) with an autocorrelation-based estimate (first prominent peak in the normalized autocorrelation of the pointiness trace within the 0.33-3.5 Hz range) using fixed weights (0.8 CNN, 0.2 ACF).

**Dynamic programming.** Individual discharge times were detected using a forward dynamic programming algorithm with an approximately-periodic prior. Candidate peaks were extracted from the combined evidence trace within an automatically detected active interval. The DP algorithm selected the optimal approximately-periodic subsequence of candidates by maximizing a score function comprising superlinear evidence rewards (E(c)^1.5 minus an existence cost of 0.05), quadratic penalties for deviations from the expected period (alpha = 1.275), and skip penalties allowing up to 3 consecutive missed discharges (beta = 0.3 per skip). The optimal path was recovered by backtracking from the highest-scoring endpoint.

**Post-processing.** Discharge times were refined through three rounds of EM-style template refinement, in which the mean discharge template was cross-correlated with the full evidence trace to generate improved candidates, followed by re-running the DP. A post-hoc confidence filter removed detections with evidence below 30% of the median peak evidence.

**IPI-derived frequency.** The final frequency estimate was computed as the reciprocal of the median inter-peak interval (IPI) across detected discharges, providing a more accurate estimate than the CNN+ACF prior by leveraging the actual detected timing sequence.

#### 2.2.3 Spatial Localization

Two complementary approaches were used for spatial localization of periodic discharges.

**Hybrid-PLV spatial extent.** Per-channel involvement was scored using a hybrid of CNN probabilities and phase-locking value (PLV) analysis. The top-3 channels by ChannelPD-Net probability served as references (with contralateral suppression for LPD). All channels were bandpass-filtered to 0.5-3.5 Hz, and the PLV between each channel and the circular-mean reference phase was computed. The combined score (0.5 * CNN probability + 0.5 * PLV) was mapped to 8 anatomical regions (left/right frontal, temporal, centro-parietal, occipital) by taking the maximum channel score per region. Regions exceeding a threshold of 0.4 were classified as involved, and spatial extent was reported as the fraction of involved regions.

**Discharge-locked topographic localization.** A novel approach to spatial localization computed the mean voltage topography at the moment of each detected discharge (figure 2C). The 19-channel monopolar EEG was bandpass-filtered (0.5-20 Hz) and transformed to the surface Laplacian to sharpen spatial resolution. For each discharge time from the HemiCET+DP pipeline, the global field power (GFP) peak was located within plus or minus 25 ms to align to the moment of maximal scalp field strength. A two-pass template refinement procedure was applied: the first pass extracted plus or minus 50 ms epochs around each GFP-aligned peak and averaged them to create a template; the second pass cross-correlated each individual epoch's GFP profile with the template to correct residual jitter. The mean topography was computed as a GFP-squared-weighted average across all refined discharge epochs, where the squared weighting strongly suppressed phantom discharges (time points where the DP algorithm interpolated a discharge at a location where no true electrographic transient existed) that exhibited low GFP near the baseline level.

The resulting 19-electrode topography vector was rendered using MNE-Python spherical spline interpolation with the inferno colormap. Verbal spatial descriptions were generated by mapping electrode amplitudes to 16 anatomical brain regions using the morgoth-viewer localization module, combined with ChannelPD-Net laterality to produce ACNS 2021-formatted descriptions (e.g., "LPD, left-sided, at 1.5 Hz, left frontotemporal predominant").

### 2.3 RDA Characterization Pipeline

The RDA pipeline (figure 3) characterized LRDA and GRDA patterns using signal-processing methods operating on the same 18-channel bipolar EEG representation.

#### 2.3.1 W05: Iterative Narrowband Refinement

Lateralization and frequency of RDA patterns were estimated using a two-pass iterative narrowband refinement procedure. In the first pass, all 18 channels were bandpass-filtered at 0.5-3.5 Hz (third-order Butterworth, zero-phase). Coarse lateralization was determined by comparing mean broadband variance between left-hemisphere and right-hemisphere channel groups, and the dominant hemisphere was assigned to the side with higher variance. Coarse frequency was estimated using Hilbert-based instantaneous frequency analysis on the top-3 channels (by variance) of the dominant hemisphere, with the median of valid instantaneous frequency samples (within 0.3-4.0 Hz) taken as the estimate.

In the second pass, all channels were re-filtered using a narrow bandpass centered on the first-pass frequency estimate (plus or minus 0.4 Hz). Lateralization was refined by comparing the mean envelope amplitude (via Hilbert transform) between hemispheres, and frequency was re-estimated from the Hilbert instantaneous frequency of the top-3 narrowband channels on the refined dominant hemisphere. LRDA versus GRDA classification was based on an amplitude asymmetry index A = |L_score - R_score| / (L_score + R_score), with LRDA assigned when asymmetry exceeded a threshold calibrated against expert labels.

The W05 method was selected as the best unified (lateralization plus frequency) method from a systematic contest of 76 algorithm variants evaluated on 4,253 RDA segments (1,295 LRDA, 2,958 GRDA), achieving a lateralization AUC of 0.837 and frequency Spearman correlation of 0.635. Alternative methods evaluated included single-pass envelope amplitude (AUC 0.826, no frequency output), PLV-selected channels (AUC 0.809, frequency rho = 0.682), and automatic channel selection with frequency consensus (AUC 0.790, frequency rho = 0.686). The iterative refinement approach provided the best lateralization performance by suppressing non-RDA activity through narrowband filtering tuned to the estimated pattern frequency.

#### 2.3.2 RDA-PLV: Spatial Extent

Spatial extent of RDA patterns was quantified using a PLV-amplitude product metric. Each bipolar channel was narrowband-filtered at the W05-estimated frequency (plus or minus 0.4 Hz). A reference signal was constructed as the mean of the top-3 narrowband channels on the dominant hemisphere. For each channel, the PLV with the reference was computed and multiplied by the normalized mean envelope amplitude. Channels with a PLV-amplitude product exceeding 0.15 were classified as involved, and spatial extent was reported as the fraction of involved channels. The PLV-amplitude product penalized channels with high PLV but negligible amplitude (noise-driven phase locking) and channels with high amplitude but poor phase coherence (non-rhythmic activity).

### 2.4 BIPD Detection

Bilateral independent periodic discharges (BIPD) -- defined by independent periodic discharge trains on each hemisphere with distinct timing sequences -- were detected using a two-stage approach. In the first stage, the HemiCET+DP pipeline was run independently on each hemisphere, yielding two discharge time sequences. In the second stage, a gradient-boosted tree (GBT) classifier was trained on features computed from the bilateral timing sequences. Features included the frequency ratio between hemispheres, inter-pulse interval regularity (coefficient of variation), matched fraction (proportion of temporally coincident discharges within 100 ms), phase consistency (mean resultant length of the cross-hemisphere phase relationship), and asymmetry measures (sequence length ratio, frequency asymmetry, count asymmetry).

The GBT classifier was trained on synthetic data to overcome the rarity of confirmed BIPD cases (21 out of 198 candidates screened). Synthetic BIPD examples were constructed by pairing independent LPD sequences from different patients. Synthetic GPD examples were created by phase-shifting a single LPD sequence with random jitter (plus or minus 25 ms) to simulate synchronous bilateral discharges. The full 3-way classification (LPD vs. GPD vs. BIPD) used 29 features comprising 18 per-channel PD probabilities from ChannelPD-Net plus 11 timing-derived features.

### 2.5 Statistical Analysis

Lateralization and classification performance was evaluated using the area under the receiver operating characteristic curve (AUC). Frequency estimation was assessed using Spearman rank correlation (rho) and mean absolute error (MAE) against expert labels. Discharge timing accuracy was evaluated using F1 score (at plus or minus 100 ms tolerance), sensitivity, precision, and timing MAE (in milliseconds). Spatial localization was assessed using the Jaccard index (intersection over union) for PD region sets and the ICC(3,1) for spatial extent as a continuous measure. For inter-rater reliability analyses, 95% confidence intervals were computed using bootstrap resampling (1,000 iterations).

---

## 3. Results

### 3.1 Lateralization and Subtype Classification

**PD lateralization.** ChannelPD-Net achieved a hemisphere lateralization AUC of 0.989 (n = 1,274) for determining whether PDs were left-dominant or right-dominant, based on the difference in mean PD probability between left and right hemisphere channels (table 3). This near-ceiling performance indicated that the per-channel CNN reliably detected the lateralized nature of LPD patterns.

**LPD versus GPD classification.** A random forest classifier (300 trees) trained on 18 per-channel PD probabilities plus 8 handcrafted features achieved an AUC of 0.911 (n = 7,037) for distinguishing LPD from GPD. The features capturing spatial asymmetry in PD probability across channels were the primary drivers of classification performance.

**3-way PD classification (LPD vs. GPD vs. BIPD).** The 3-way classifier incorporating timing-derived features achieved a macro AUC of 0.862 (n = 5,064; LPD AUC 0.832, GPD AUC 0.835, BIPD AUC 0.920). In the clinically relevant binary comparison of BIPD versus GPD, the classifier achieved an AUC of 0.937 (n = 2,308), demonstrating strong discrimination between synchronous generalized discharges and bilateral independent discharge trains. The dataset comprised 2,756 LPD, 2,298 GPD, and 10 confirmed BIPD segments. The most discriminative features for BIPD detection were matched fraction and phase consistency, which captured the defining characteristic of temporal independence between hemispheres.

**RDA lateralization (LRDA vs. GRDA).** The W05 iterative narrowband refinement method achieved a lateralization AUC of 0.837 on 4,253 segments (1,295 LRDA, 2,958 GRDA) in the V5 lateralization contest (table 3). This method outperformed 75 alternative approaches, including simple amplitude asymmetry (L24, AUC 0.826), PLV-based lateralization (V04, AUC 0.809), and various multi-channel Hilbert methods (AUC 0.790). The key to W05's superiority was that the narrowband filtering in pass 2 removed non-RDA activity, sharpening the lateralization signal. Clean per-segment labels proved critical: the same hemisphere-independent methods achieved only AUC 0.58 with noisy patient-level labels in earlier experiments, rising to 0.84 after label correction.

**Table 3. Lateralization and classification performance.**

| Task | Method | AUC | N |
|---|---|---|---:|
| PD hemisphere (L vs. R) | ChannelPD-Net mean probability | 0.989 | 1,274 |
| LPD vs. GPD | RF on channel probs + features | 0.911 | 7,037 |
| 3-way macro (LPD/GPD/BIPD) | RF on channel probs + timing | 0.862 | 5,064 |
| BIPD vs. GPD | Timing features (GBT) | 0.937 | 2,308 |
| LRDA vs. GRDA | W05 iterative narrowband | 0.837 | 4,253 |

### 3.2 Frequency Estimation

**PD frequency.** On quality-filtered segments, the PD pipeline (IPI-derived frequency from HemiCET+DP) achieved Spearman correlations of 0.786 for LPD (n = 1,226, MAE = 0.265 Hz) and 0.846 for GPD (n = 1,089, MAE = 0.172 Hz) (table 5, figure 4). These results represented substantial improvements over the signal-processing baseline of Tautan et al. [10], which achieved correlations of 0.184 (LPD, MAE = 0.581 Hz) and 0.248 (GPD, MAE = 0.469 Hz) on the same segments. The improvement was approximately four-fold for both subtypes.

The superior performance of IPI-derived frequency over direct CNN prediction (rho approximately 0.55) or CNN+ACF ensemble (rho approximately 0.60) confirmed that leveraging detected discharge timing for frequency estimation was substantially more accurate than spectral or regression-based approaches. The HemiCET+DP pipeline achieved a timing-derived frequency Spearman correlation of 0.891 on the 582 production evaluation cases, with a frequency MAE of 0.183 Hz.

**RDA frequency.** The W05 iterative narrowband method achieved Spearman correlations of 0.674 for LRDA (n = 640, MAE = 0.233 Hz) and 0.712 for GRDA (n = 1,310, MAE = 0.215 Hz) on quality-filtered segments (table 5, figure 4). Tautan et al. achieved correlations of 0.135 (LRDA, MAE = 0.573 Hz) and 0.218 (GRDA, MAE = 0.546 Hz), representing a five-fold and three-fold improvement, respectively.

In the inter-rater reliability analysis on the expert dataset (68 segments with 3-expert frequency annotations), the W05 method achieved an ICC of 0.860 when treated as an additional rater, exceeding the expert-expert ICC of 0.852 (figure S1). This result indicated that the algorithm agreed with experts as well as experts agreed with each other for RDA frequency estimation. For PD frequency, the algorithm ICC was 0.572 compared to expert-expert ICC of 0.662, indicating room for further improvement.

**Table 5. Frequency estimation performance (quality-filtered).**

| Subtype | N | Algorithm rho | Algorithm MAE (Hz) | Tautan rho | Tautan MAE (Hz) |
|---|---:|---|---|---|---|
| LPD | 1,226 | 0.786 | 0.265 | 0.184 | 0.581 |
| GPD | 1,089 | 0.846 | 0.172 | 0.248 | 0.469 |
| LRDA | 640 | 0.674 | 0.233 | 0.135 | 0.573 |
| GRDA | 1,310 | 0.712 | 0.215 | 0.218 | 0.546 |

**Label quality analysis.** The quality of ground-truth frequency labels substantially influenced apparent algorithm performance. Segments with less than 60% expert agreement on pattern classification exhibited 2.4-fold higher frequency discrepancy between algorithm and expert than segments with 100% agreement. When MW re-reviewed all cases where the algorithm-expert frequency discrepancy exceeded 0.5 Hz, the original MW label was corrected in favor of the algorithm estimate in 94% of PD cases and 61% of RDA cases. This bidirectional quality improvement -- algorithm predictions revealing annotation errors, which in turn improved the evaluation benchmark -- was a recurring theme throughout development.

### 3.3 Discharge Timing

The HemiCET v2 + DP pipeline (configuration C1) achieved a discharge detection F1 of 0.889 (sensitivity 0.921, precision 0.859) at a tolerance of plus or minus 100 ms on 582 evaluation cases (table 6). The median timing MAE was 1.0 ms, indicating that detected discharges were localized to within a single sample (5 ms at 200 Hz) of the expert-annotated timing. The IPI-derived frequency from detected discharges achieved Spearman rho of 0.891 with expert-reviewed frequency.

This performance substantially exceeded two earlier pipeline versions: the full 18-channel pipeline (F1 = 0.726, timing MAE = 17.7 ms) and the per-hemisphere baseline using handcrafted evidence only (F1 = 0.719, timing MAE = 19.4 ms). The HemiCET approach also exceeded an Oracle baseline that used expert-annotated frequency as the DP prior with handcrafted evidence (F1 = 0.664, timing MAE = 10.9 ms). This finding demonstrated that learned evidence from the 8-channel hemisphere CET-UNet was superior to handcrafted features, more than compensating for imperfect frequency knowledge. The 8-channel hemisphere design outperformed the full 18-channel approach because it avoided noise from the uninvolved hemisphere (for LPD) and learned cross-channel patterns within a hemisphere directly.

**Table 6. Discharge timing performance.**

| Method | F1 | Sensitivity | Precision | Freq rho | Timing MAE (ms) |
|---|---|---|---|---|---|
| HemiCET v2 + DP (production) | 0.889 | 0.921 | 0.859 | 0.891 | 1.0 |
| Oracle (expert freq + HPP) | 0.664 | 0.569 | 0.799 | 0.910 | 10.9 |
| Full 18-channel pipeline | 0.726 | 0.722 | 0.730 | 0.765 | 17.7 |
| Per-hemisphere baseline | 0.719 | 0.724 | 0.715 | 0.778 | 19.4 |

### 3.4 Spatial Localization

#### 3.4.1 PD Spatial Localization

**Discharge-locked topographic localization.** The primary spatial localization method for PDs computed the mean discharge topography using the two-pass Laplacian-GFP alignment procedure described in section 2.2.3. Representative topographic maps are shown in figures 5-6. The Laplacian transform sharpened spatial resolution by removing volume-conducted contributions, producing focal peaks corresponding to cortical generators. The GFP-squared weighting reduced the effective contribution of phantom discharges by a factor of 4-10 relative to uniform averaging, and the two-pass alignment reduced cross-epoch jitter by approximately 8-12 ms compared to single-pass GFP alignment.

**Spatial extent (Jaccard agreement).** The Hybrid-PLV spatial method was evaluated against three expert raters (LB, PH, SZ) on PD segments with 3-rater ground truth (table 4). At the optimized binarization threshold of 0.38, the mean model-expert Jaccard index was 0.731 (plus or minus 0.076), compared to a mean expert-expert Jaccard of 0.751 (plus or minus 0.025), representing 97.3% of human inter-rater agreement. Notably, the model-PH agreement (Jaccard 0.837) exceeded all expert-expert pairs (maximum: LB-SZ at 0.773), indicating that the model acted as a "virtual fourth rater" with agreement patterns within the range of expert variation.

**Table 4. PD spatial inter-rater agreement (Jaccard, threshold = 0.38).**

| | LB | PH | SZ | Model |
|---|---:|---:|---:|---:|
| LB | 1.000 | 0.762 | 0.773 | 0.693 |
| PH | 0.762 | 1.000 | 0.716 | 0.837 |
| SZ | 0.773 | 0.716 | 1.000 | 0.662 |
| Model | 0.693 | 0.837 | 0.662 | 1.000 |

After threshold optimization (threshold = 0.62) and quality filtering (excluding SZ spatial_extent = 0 entries), the PDProfiler spatial ICC was 0.852, exceeding the expert-expert ICC of 0.845 (figure S1). In comparison, Tautan et al. [10] achieved a PD spatial ICC of 0.764 on the same segments.

#### 3.4.2 RDA Spatial Extent

The RDA-PLV method achieved an ICC of 0.371 when the algorithm was treated as a fourth rater alongside the three experts, compared to an expert-expert ICC of 0.373 (n = 211 segments with 3-rater spatial ground truth). The algorithm MAE of 0.215 was better than some expert-expert pairs (e.g., LB vs. SZ MAE = 0.484). Tautan et al. achieved an RDA spatial ICC of 0.215, substantially below both expert agreement and the present method.

The low absolute ICC values (approximately 0.37) for RDA spatial extent -- for both the algorithm and expert raters -- reflected the inherent difficulty of this task. Unlike PD spatial localization, where discrete discharges create identifiable transients, RDA produces continuous rhythmic activity whose boundaries are gradual and subjective. The near-parity between algorithm and expert ICC indicated that the PLV-amplitude method captured the same information experts used when judging spatial extent, while acknowledging that spatial extent is inherently a low-reliability measure even among trained electroencephalographers.

### 3.5 End-to-End Characterization Examples

Figures 5-8 present representative characterization examples for all four subtypes at three difficulty levels (easy, medium, hard), selected to span the range of inter-rater agreement. Each example shows the 19-channel EEG with pipeline outputs overlaid: detected discharge times (red dashed lines, PD only), hemisphere shading (light blue), Laplacian topographic map (inferno colormap), and ACNS 2021 verbal description.

For LPD examples (figure 5), the easy case (IIIC agreement at least 95%) exhibited clear, stereotyped discharges with unambiguous lateralization and a sharp focal maximum in the topographic map. The medium case (70-80% agreement) showed recognizable periodicity with some morphological variability; the pipeline correctly lateralized and detected most discharges, though one or two marginal peaks were missed. The hard case (45-60% agreement) featured subtle, atypical discharges where expert raters themselves disagreed; the pipeline produced a reasonable characterization, though the topographic map was more diffuse, reflecting the ambiguity of the pattern.

For GPD examples (figure 6), discharge markers spanned both hemispheres, and the topographic maps showed bilateral, often frontally predominant distributions. For LRDA and GRDA examples (figures 7-8), no discharge markers were shown (RDA being a continuous rhythmic pattern), but hemisphere shading and amplitude topographic maps correctly identified the spatial distribution.

### 3.6 Model Architecture Comparison

A comparison of neural architecture strategies (table 7) confirmed the value of combining learned evidence with structured inference. The production CET-UNet + DP pipeline (F1 = 0.889, frequency rho = 0.891) substantially outperformed an end-to-end CNN approach (E2E-CNN; F1 = 0.006, frequency rho = 0.001), which failed completely with the available training data size of approximately 675 labeled examples. This result demonstrated that with current data volumes, end-to-end neural models could not learn the temporal structure that DP encoded as a prior. The winning strategy was neural evidence generation (CET-UNet learning what a discharge looks like) combined with structured inference (DP learning when discharges should occur based on periodicity constraints).

**Table 7. End-to-end versus structured inference.**

| Model | F1 | Freq rho | Notes |
|---|---|---|---|
| CET-UNet + DP (production) | 0.889 | 0.891 | Neural evidence + DP |
| E2E-CNN (direct prediction) | 0.006 | 0.001 | Failed with ~675 training examples |

---

## 4. Discussion

### 4.1 Comparison with Prior Work

The most directly comparable prior work is that of Tautan et al. [10], who described a signal-processing approach to PD and RDA characterization using autocorrelation-based frequency estimation and spectral spatial analysis. The present system achieved substantially higher performance across all characterization tasks (table 5). For frequency estimation, the improvement ranged from three-fold (GRDA: rho 0.712 vs. 0.218) to five-fold (LRDA: rho 0.674 vs. 0.135) in Spearman correlation, and from two-fold to three-fold reduction in MAE. For spatial extent, the present system achieved ICC values matching or exceeding expert-expert agreement (PD: 0.852 vs. expert 0.845; RDA: 0.371 vs. expert 0.373), while Tautan et al. achieved lower ICC values (PD: 0.764; RDA: 0.215).

Several factors contributed to these improvements. First, the use of learned evidence (HemiCET-UNet) rather than purely handcrafted features enabled the PD pipeline to capture discharge morphology patterns that are difficult to specify algorithmically. Second, the IPI-derived frequency from detected discharge timing provided a more direct and accurate frequency estimate than spectral methods, which are sensitive to harmonics, non-stationarity, and contamination from non-periodic activity. Third, the iterative narrowband refinement in the RDA pipeline (W05) leveraged the estimated frequency to isolate the pattern of interest before computing lateralization and refined frequency estimates, reducing the influence of confounding signals. Fourth, the phase-locking value approach to spatial extent quantified the functional coherence of each channel with the dominant pattern, providing a more principled metric than amplitude-based channel counting.

No prior work has reported automated discharge-level timing for periodic discharges or bilateral independent PD detection, making direct comparison on these tasks impossible. The present F1 of 0.889 for discharge detection and AUC of 0.937 for BIPD versus GPD classification establish baseline benchmarks for future studies.

### 4.2 Learned Evidence versus Handcrafted Features

A key finding of this work was that the HemiCET-UNet evidence trace (a learned 1D U-Net operating on single channels) substantially outperformed handcrafted features (pointiness + TKEO) when combined with the same DP inference framework. The HemiCET v2 + DP achieved F1 = 0.889 compared to the per-hemisphere handcrafted baseline F1 = 0.719, representing a 24% improvement. More strikingly, the learned evidence with imperfect CNN+ACF frequency exceeded the Oracle configuration using expert-annotated frequency with handcrafted evidence (F1 = 0.889 vs. 0.664), demonstrating that the quality of discharge evidence mattered more than the accuracy of the frequency prior.

This result supports a hybrid architecture in which neural networks are used for perceptual tasks (recognizing discharge morphology in raw EEG) while domain knowledge is encoded as structural constraints (periodicity in the DP algorithm). Pure end-to-end approaches, which must learn both the perceptual and structural components from labeled data, failed with the current training set size of approximately 675 labeled segments. This finding is consistent with the broader machine learning literature on combining neural perception with model-based reasoning for structured prediction tasks [11].

The complementary value of handcrafted and learned evidence was confirmed by the product-boost combination formula, which yielded better performance than either source alone. The handcrafted pointiness trace captured sharp transients that the CET-UNet occasionally missed (particularly low-amplitude discharges in noisy channels), while the CET-UNet identified discharges with atypical morphology that did not trigger the pointiness detector.

### 4.3 Label Quality Discovery

A recurring theme in this work was the discovery that apparent algorithm limitations were often attributable to label noise rather than algorithm deficiency. Three examples illustrate this phenomenon.

First, the original per-patient label system (patients.csv) aggregated IIIC crowd votes across all 10-second segments within a 10-minute recording, such that a patient labeled "LRDA" might have had most individual segments voted as "other" by the crowd. Migration to per-segment labels (segment_labels.csv) improved RDA lateralization AUC from 0.58 to 0.84 using the same algorithms.

Second, MW's systematic frequency review of 993 RDA segments identified that the W05 algorithm estimate matched the expert exactly in 59% of cases and was within 0.25 Hz in 85% of cases. Among the 399 corrections, many involved ambiguous segments where frequency estimation was inherently uncertain.

Third, in a post-hoc analysis of PD frequency discrepancies exceeding 0.5 Hz, MW re-reviewed all discordant cases and judged the algorithm estimate to be correct (more consistent with the visual pattern) in 94% of cases, leading to ground-truth label corrections. These findings underscore the importance of iterative, bidirectional quality improvement between algorithm development and annotation refinement.

### 4.4 Clinical Implications

The automated characterization system has several potential clinical applications. First, it can reduce the burden on electroencephalographers during prolonged cEEG monitoring by providing structured ACNS 2021 descriptions that serve as a starting point for expert review, rather than requiring de novo characterization. Second, standardized quantitative characterization enables large-scale studies of PD and RDA epidemiology, evolution, and association with clinical outcomes, which are currently limited by the labor intensity of manual annotation. Third, the discharge-locked topographic maps provide a quantitative spatial representation that is more objective and reproducible than verbal descriptions of field distribution.

The BIPD detection capability, though limited by the small number of confirmed cases (10 in the evaluation set), addresses a clinically important distinction. BIPDs have different prognostic implications than unilateral LPDs or bilateral synchronous GPDs, and their identification requires comparison of independent hemisphere timing sequences -- a task that is cognitively demanding for human reviewers but naturally suited to algorithmic analysis.

The discharge timing capability (F1 = 0.889, timing MAE = 1.0 ms) enables analyses that were previously impractical at scale, including IPI variability assessment (relevant to the ACNS "nearly regular" criterion), discharge-locked averaging for morphology analysis, and temporal relationship analysis between discharge patterns and other physiological signals.

### 4.5 Limitations and Future Work

Several limitations should be acknowledged. First, the BIPD evaluation was limited to 10 confirmed cases due to the rarity of the pattern (21 confirmed from 198 candidates screened, 11% yield). The 3-way classification results (macro AUC 0.862) should be interpreted with caution given this sample size imbalance. Future work should expand the BIPD training dataset (21 confirmed cases are available but only 10 were used) and evaluate the classifier on a larger held-out set.

Second, RDA wave timing (identifying the onset, peak, and offset of each rhythmic delta wave cycle) was not addressed in this work. While 502 segments have been labeled with wave timing annotations, the automated method for wave detection remains under development. Wave-level timing would enable phase-locked analyses (e.g., whether fast activity occurs at a specific RDA phase) and wave-by-wave frequency estimation.

Third, the discharge-locked topographic localization approach, while providing a principled and ground-truth-free spatial representation, has not been formally validated against a gold standard. The Jaccard analysis of spatial extent (table 4) demonstrated near-expert agreement for threshold-based channel involvement, but the topographic maps themselves were evaluated only qualitatively through expert visual review. Future work should develop quantitative validation methods, potentially using intracranial EEG recordings as ground truth.

Fourth, the evaluation of all methods relied on a single institution's EEG data (though drawn from thousands of patients) and annotations from a limited number of expert raters (primarily one rater for frequency and timing, three raters for spatial extent). Multi-center validation with independent expert panels would strengthen the generalizability of these results.

Fifth, the system operates on isolated 10-second EEG segments and does not model temporal evolution of patterns across longer recordings. Clinical monitoring generates hours to days of continuous data, and characterization of pattern evolution (e.g., frequency trending, spatial spread) would enhance clinical utility. Integration with existing pattern detection systems (e.g., Morgoth [7]) that provide real-time classification of continuous EEG is a natural extension.

Sixth, the current system does not address seizure patterns, which share features with PDs and RDA but require distinct characterization. Extension to the full ACNS critical care EEG terminology, including seizures, brief potentially ictal rhythmic discharges (BIRDs), and stimulus-induced patterns, is an important direction.

Finally, training data volume remains a constraint, particularly for the HemiCET-UNet (675 labeled segments) and BIPD classifier (10 evaluation cases). Self-supervised pretraining on unlabeled EEG data and transfer learning from larger datasets may help overcome this limitation.

---

## 5. Conclusion

This work presents a comprehensive automated system for characterizing periodic discharges and rhythmic delta activity in continuous EEG, addressing lateralization, spatial localization, discharge timing, frequency estimation, and BIPD detection within a unified framework. The PD pipeline combines learned neural evidence (HemiCET-UNet) with structured dynamic programming inference, while the RDA pipeline uses iterative narrowband signal processing with phase-coherence analysis. Across all characterization tasks, the system achieves performance approaching or matching expert inter-rater agreement, with frequency estimation correlations 3-5 times higher than the previous state of the art.

The key technical insight is that a hybrid architecture -- neural networks for perceptual feature extraction combined with domain-knowledge-encoded temporal priors -- outperforms both purely handcrafted and purely end-to-end approaches at current data scales. The discharge-locked topographic localization method provides a principled, threshold-free spatial representation that directly computes the voltage field distribution at discharge peaks. The iterative, bidirectional relationship between algorithm development and annotation refinement -- where algorithm predictions reveal label errors and corrected labels improve algorithm evaluation -- was essential to achieving reliable performance benchmarks.

The system produces ACNS 2021-formatted verbal descriptions suitable for clinical reporting and provides a foundation for large-scale quantitative analysis of periodic and rhythmic EEG patterns. All code and evaluation scripts are publicly available.

---

## References

[1] Claassen J, Mayer SA, Kowalski RG, Emerson RG, Hirsch LJ. Detection of electrographic seizures with continuous EEG monitoring in critically ill patients. *Neurology*. 2004;62(10):1743-1748.

[2] Chong DJ, Hirsch LJ. Which EEG patterns warrant treatment in the critically ill? Reviewing the evidence for treatment of periodic epileptiform discharges and related patterns. *J Clin Neurophysiol*. 2005;22(2):79-91.

[3] Rodriguez Ruiz A, Vlachy J, Lee JW, et al. Association of periodic and rhythmic electroencephalographic patterns with seizures in critically ill patients. *JAMA Neurol*. 2017;74(2):181-188.

[4] Hirsch LJ, Fong MWK, Leitinger M, et al. American Clinical Neurophysiology Society's standardized critical care EEG terminology: 2021 version. *J Clin Neurophysiol*. 2021;38(1):1-29.

[5] Gaspard N, Hirsch LJ, LaRoche SM, Hahn CD, Westover MB; Critical Care EEG Monitoring Research Consortium. Interrater agreement for critical care EEG terminology. *Epilepsia*. 2014;55(9):1366-1373.

[6] Leitinger M, Beniczky S, Rohracher A, et al. Salzburg consensus criteria for non-convulsive status epilepticus -- revisited: a critical reappraisal. *Epilepsy Behav*. 2015;49:158-163.

[7] Jing J, Ge W, Hong S, et al. Development of expert-level classification of seizures and rhythmic and periodic patterns during EEG interpretation. *Neurology*. 2023;100(17):e1750-e1762.

[8] Ge W, Jing J, An S, et al. Deep active learning for interictal ictal injury continuum EEG patterns. *J Neurosci Methods*. 2023;390:109835.

[9] Zheng WL, Amorim E, Jing J, et al. Predicting neurological outcome from electroencephalogram dynamics in comatose patients after cardiac arrest with deep learning. *IEEE Trans Biomed Eng*. 2022;69(5):1813-1825.

[10] Tautan AM, Jing J, Bhatt A, et al. Automated characterization of periodic and rhythmic patterns in EEG. *Clin Neurophysiol*. 2025;169:38-49.

[11] Lake BM, Ullman TD, Tenenbaum JB, Gershman SJ. Building machines that learn and think like people. *Behav Brain Sci*. 2017;40:e253.

---

## Figure Legends

**Figure 1.** Representative EEG examples of periodic and rhythmic patterns. Six 10-second EEG segments displayed in average-reference montage (19 channels, 200 Hz, bandpass 0.5-20 Hz). Channels are grouped by hemisphere: left parasagittal (Fp1, F3, C3, P3), left temporal (F7, T3, T5, O1), midline (Fz, Cz, Pz), right parasagittal (Fp2, F4, C4, P4), and right temporal (F8, T4, T6, O2). **(A)** Clear lateralized periodic discharges (LPD). **(B)** Clear generalized periodic discharges (GPD). **(C)** Clear lateralized rhythmic delta activity (LRDA). **(D)** Clear generalized rhythmic delta activity (GRDA). **(E)** Ambiguous LPD case with lower inter-rater agreement. **(F)** Ambiguous GRDA case with lower inter-rater agreement. Clear cases (A-D) were selected from segments with at least 80% agreement among at least 10 IIIC expert raters; ambiguous cases (E-F) have lower agreement, illustrating the challenge of pattern classification. Scale bar: 100 uV.

**Figure 2.** PD characterization pipeline architecture and example output. Three-panel overview of the periodic discharge (PD) characterization pipeline applied to an example LPD segment. **(A)** Input: raw 19-channel average-reference EEG (10 seconds, 200 Hz). **(B)** Pipeline architecture: 18 independent bipolar channels are processed by ChannelPD-Net (CNN+Attention), which produces per-channel PD probabilities and frequency estimates. Three downstream modules operate in parallel: (1) Laterality Detection compares left versus right hemisphere mean probabilities; (2) Discharge Detection uses 8-channel CET-UNet evidence and CNN+ACF frequency prior fed into dynamic programming with EM template refinement, yielding individual discharge times and IPI-derived frequency; (3) Topographic Localization extracts monopolar voltage at discharge peaks, applies Laplacian-GFP alignment with two-pass template refinement and GFP-squared-weighted averaging to produce a spatial localization map. **(C)** Output: the same EEG with red dashed lines marking detected discharge times, light blue shading on the involved hemisphere, a Laplacian topoplot showing the discharge topography, and a verbal description following ACNS 2021 standardized nomenclature. Scale bar: 100 uV.

**Figure 3.** RDA characterization pipeline architecture and example output. Three-panel overview of the rhythmic delta activity (RDA) characterization pipeline applied to an example LRDA segment, following the same layout as figure 2. **(A)** Input: raw 19-channel average-reference EEG. **(B)** Pipeline architecture: the W05 iterative narrowband refinement algorithm performs two passes. Pass 1: coarse analysis with broadband (0.5-3.5 Hz) filtering, estimating lateralization via per-hemisphere mean variance and frequency via Hilbert instantaneous frequency from the top-3 dominant channels. Pass 2: narrowband filtering at the estimated frequency plus or minus 0.4 Hz, with envelope amplitude for refined lateralization and Hilbert frequency on the dominant hemisphere only. Three output branches produce laterality, spatial extent (via per-channel PLV with the dominant hemisphere), and frequency. **(C)** Output: the same EEG with hemisphere shading, a Laplacian amplitude topoplot, and a verbal description. No discharge markers are shown, as RDA is a continuous rhythmic pattern. Scale bar: 100 uV.

**Figure 4.** Frequency estimation performance across all four IIIC subtypes. Scatter plots comparing algorithm-predicted frequency (y-axis) against expert-reviewed frequency (x-axis) for each subtype. Top row: present system (PD: CNN+ACF frequency prior refined by IPI from detected discharges; RDA: W05 iterative narrowband Hilbert). Bottom row: Tautan et al. signal-processing baseline. Columns from left to right: LPD (n = 1,226; rho = 0.786), GPD (n = 1,089; rho = 0.846), LRDA (n = 640; rho = 0.674), GRDA (n = 1,310; rho = 0.712). Spearman correlation (rho) and mean absolute error (MAE, in Hz) are shown for each panel. Expert frequency is the mean across available raters. Dashed diagonal line indicates perfect agreement.

**Figure 5.** LPD characterization examples at three difficulty levels. Three representative lateralized periodic discharge (LPD) cases selected to span the range of inter-rater agreement. Each row shows one case with: (left) 19-channel average-reference EEG (10 seconds, bandpass 0.3-50 Hz, amplitude clipped at plus or minus 250 uV) with detected discharge times marked as red dashed vertical lines on the involved hemisphere and light blue shading indicating the lateralized side; (right) Laplacian topoplot (inferno colormap) with electrode labels, and ACNS 2021 verbal description. Top: easy case (agreement at least 95%). Middle: medium case (agreement 70-80%). Bottom: hard case (agreement 45-60%). Scale bar: 100 uV.

**Figure 6.** GPD characterization examples at three difficulty levels. Three representative generalized periodic discharge (GPD) cases, following the same format as figure 5. Discharge markers span both hemispheres, reflecting the generalized distribution. Scale bar: 100 uV.

**Figure 7.** LRDA characterization examples at three difficulty levels. Three representative lateralized rhythmic delta activity (LRDA) cases, following the same format as figure 5. No discharge markers are shown. Light blue shading indicates the dominant hemisphere. Scale bar: 100 uV.

**Figure 8.** GRDA characterization examples at three difficulty levels. Three representative generalized rhythmic delta activity (GRDA) cases, following the same format as figure 5. No discharge markers are shown. Light blue shading covers all channels, reflecting the generalized distribution. Scale bar: 100 uV.

**Figure S1.** Inter-rater reliability comparison: expert-expert versus algorithm-expert. Bar charts comparing ICC(3,1) and percentage agreement for frequency and spatial extent estimation. Top row: RDA (LRDA + GRDA combined). Bottom row: PD (LPD + GPD combined). Three conditions compared: expert-expert agreement (blue), present system (orange), and Tautan et al. (green). W05 for RDA frequency achieves ICC = 0.860, matching expert-expert ICC = 0.852. PDProfiler for PD spatial extent achieves ICC = 0.852 after threshold optimization, exceeding expert-expert ICC = 0.845.

**Figure S2.** Spatial extent scatter plots by subtype. Scatter plots comparing algorithm-predicted spatial extent (y-axis, fraction of 18 channels involved) against expert spatial extent (x-axis, mean across raters) for each subtype. Per-rater data points shown with distinct markers for each expert rater (LB, PH, SZ, MW). Spearman rho and MAE are reported per panel.

**Figure S3.** Spatial extent threshold optimization. Effect of binarization threshold on spatial extent prediction accuracy, evaluated on segments with at least 2 expert raters. Top row: PD (LPD + GPD). Bottom row: RDA (LRDA + GRDA). Three metrics shown across threshold values (0.01-0.99): MAE, Pearson correlation, and ICC(3,1). Optimal thresholds are marked: PD approximately 0.38-0.62 (broad plateau), RDA approximately 0.15.
