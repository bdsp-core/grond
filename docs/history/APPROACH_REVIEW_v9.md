# Frequency Estimation for Periodic EEG Patterns: Review v9

## Problem & Data

Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Additionally: classifying subtype (LPD vs GPD), laterality (left vs right for LPD), and detecting which channels contain periodic discharges.

### Dataset (as of 2026-03-20)

| | Patients | Segments | Raters |
|---|---|---|---|
| LPD | 425 | ~1,700 | MW (all) |
| GPD | 151 | ~830 | MW (all) |
| GRDA | 115 | ~460 | Various |
| LRDA | 105 | ~420 | Various |
| **Total** | **839** | **~2,511** | |

**Frequency estimation**: 594 patients with gold standard freq labels (425 LPD + 151 GPD + 18 other).
**Laterality labels**: 143 LPD patients (83 left, 60 right). 276 additional LPD patients have GBM-predicted laterality (0.957 AUC proxy).
**Channel-level PD dataset**: 9,310 channels from 815 patients (4,524 positive, 4,786 negative).
**Gold standard**: MW's frequency rating. `_original` columns preserve first-pass annotations.

### What changed since v8
- **CNN+Temporal Attention model** achieves new best frequency estimation (Spearman 0.640)
- **Channel-level PD detector** completed through Phase 3b (logistic → CNN → CNN+Attention)
- **PD probability does NOT predict frequency estimation accuracy** (ρ = 0.011, orthogonal tasks)
- **Discharge timing detector (Phase 4)** attempted — bootstrapped from pointiness peaks, failed (Spearman -0.08)
- **PD morphological consistency scorer** built and evaluated — too strict, most cases score low
- **Expanded channel dataset** using GBM-predicted laterality for 276 unlabeled LPD patients

## Current Best Results

### Frequency Estimation — Best: CNN+Temporal Attention, Spearman 0.640

| Method | Combined ρs | LPD ρs (n=425) | GPD ρs (n=151) | MAE |
|--------|-------------|----------------|----------------|-----|
| **CNN+Temporal Attention** | **0.640** | **0.573** | **0.718** | **0.271** |
| RF (200 trees, depth 8) | 0.604 [0.542, 0.661] | 0.519 | 0.677 | 0.267 |
| GBM (200 trees, depth 3) | 0.602 [0.541, 0.656] | 0.496 | 0.698 | 0.270 |
| Ridge (α=1) | 0.589 [0.524, 0.646] | 0.488 | 0.700 | 0.274 |

The CNN+Attention learns directly from raw single-channel waveforms. For each channel it predicts both PD presence and frequency, using temporal attention to focus on the most informative time windows. Patient-level frequency is computed as a PD-probability-weighted mean across channels — channels the model is confident have PDs contribute more.

### Subtype Classification (LPD vs GPD) — Best: RF 300, AUC 0.931

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF 300 (depth 8)** | **0.931** | **86.5%** | 0.814 |
| GBM (depth 3) | 0.929 | 86.0% | 0.810 |
| GBM balanced | 0.923 | 84.5% | **0.853** |
| Logistic Ridge (α=0.1) | 0.852 | 79.0% | 0.702 |

Tree models outperform logistic regression by +0.08 AUC. Class balancing doubles GPD sensitivity (0.87 vs 0.44) at modest accuracy cost. Laterality features are useless for subtype (AUC 0.47).

### Laterality Classification (Left vs Right) — Best: GBM balanced, AUC 0.957

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF balanced** | **0.959** | 90.9% | 0.906 |
| **GBM balanced** | 0.957 | **93.0%** | **0.924** |
| lat_idx only | 0.920 | 86.7% | 0.856 |

The laterality index (single feature) achieves 0.920 AUC by itself. Tree models add ~0.04 AUC via nonlinear interactions.

### Channel-Level PD Detection — Best: CNN+Attention, Patient AUC 0.989

| Model | Channel AUC | Patient AUC | Ch Freq ρs |
|-------|------------|-------------|------------|
| Logistic regression (13 features) | 0.766 | 0.902 | — |
| 1D CNN (multi-task) | 0.867 | 0.981 | 0.566 |
| **1D CNN + Temporal Attention** | **0.870** | **0.989** | **0.590** |

## The CNN+Attention Architecture

This is our most important new result. A single 1D CNN processes individual EEG channels and jointly performs PD detection + frequency estimation.

### Architecture
```
Input: single channel (1, 2000) — 10s at 200Hz

Encoder:
  Conv1d(1→16, k=51, s=2) → BN → GELU → Dropout(0.1)    # 1000 samples
  Conv1d(16→32, k=25, s=2) → BN → GELU → Dropout(0.1)    # 500 samples
  Conv1d(32→64, k=13, s=2) → BN → GELU → Dropout(0.1)    # 250 samples
  Conv1d(64→64, k=7, s=2) → BN → GELU → Dropout(0.2)     # 125 samples

Temporal Attention:
  Conv1d(64→1, k=1) → softmax over time → weights (1, 125)
  Weighted pool: sum(features × attention, dim=time) → (64,)

Two heads:
  PD head: Linear(64→1) → Sigmoid
  Freq head: Linear(64→1) → log(frequency)
```

### Training
- **Multi-task loss**: BCE(PD prediction) + 0.5 × MSE(log freq) × PD_mask
- Frequency loss only computed on PD-positive channels
- 5-fold patient-stratified CV (not full LOPO — too expensive for CNN)
- 30 epochs, batch 128, lr=1e-3 with cosine annealing, Adam
- Data augmentation: amplitude scaling (0.8-1.2×), Gaussian noise (SNR 20-40 dB), time shift (±50 samples)
- Per-channel z-score normalization

### Patient-Level Aggregation
For each patient:
1. Run all channels of all segments through the CNN
2. Get per-channel PD probability and frequency prediction
3. Patient frequency = PD-weighted mean of per-channel log-frequency predictions
4. Channels with higher PD probability contribute more → spatial selectivity

### Why It Works
- **Temporal attention** learns to focus on the part of the 10-second window with the clearest periodic pattern, ignoring quiet flanks
- **PD-weighted aggregation** automatically selects channels with PDs, rather than averaging across all 18 channels (many of which may have no PDs)
- **Multi-task learning** forces the encoder to learn representations useful for both PD detection and frequency, improving generalization
- **Raw waveform input** captures information that handcrafted features miss (waveform morphology, subtle periodicity patterns)

## Key Findings

### 1. CNN+Attention beats handcrafted features for frequency estimation

| Approach | Combined ρs | LPD ρs | GPD ρs |
|----------|------------|--------|--------|
| Handcrafted SP features + RF | 0.604 | 0.519 | 0.677 |
| CNN+Attention (raw waveform) | **0.640** | **0.573** | **0.718** |
| Improvement | **+0.036** | **+0.054** | **+0.041** |

The biggest improvement is on LPD (+0.054), where spatial variability makes handcrafted features less reliable. The CNN learns to adapt to each channel's characteristics.

### 2. PD detection and frequency estimation are orthogonal

CNN PD probability vs frequency estimation error: ρ = 0.011 (p = 0.85). Tested with mean, max, and top-4 PD probability — all near zero correlation. AUC for predicting high-vs-low error: 0.50-0.52 (chance).

**Interpretation**: A case can have very clear PDs (high PD prob) but ambiguous frequency (e.g., irregular intervals), and vice versa. PD probability cannot serve as a confidence score for frequency estimation.

### 3. Discharge timing from bootstrapped targets doesn't work

Phase 4 attempted a U-Net style discharge detector trained on pointiness-derived peak targets:
- 5-fold CV, 30 epochs per fold
- IPI-derived frequency: Spearman **-0.08** (worse than chance)
- The model detected ~22 peaks per channel but many spurious
- Mean IPI CV = 0.438 (high variability)

**Why it failed**: Pointiness peak detection is too noisy as training targets. Many detected peaks are artifacts, not true discharges. The model learns to reproduce this noise rather than find real discharge times. Clean hand-annotated discharge times on even 20-30 cases would likely fix this.

### 4. PD morphological consistency scoring is too strict

Cross-correlation of successive discharge windows + CV of shape features:
- 54 patients scored HIGH (>0.5), 60 MODERATE (0.3-0.5), 480 LOW (<0.3)
- The 30% pointiness peak height threshold is too aggressive for many real PDs
- Not useful as a quality filter in its current form

### 5. Handcrafted feature findings (unchanged from v8)

- **Feature sets barely matter**: base 6, base 5, all 9, interactions — all within 0.003 Spearman
- **Frequency balancing hurts Ridge** (-0.09), neutral for RF (-0.004)
- **Tree models beat Ridge** consistently (+0.015 Spearman)
- **LPD-GPD gap persists**: LPD ρs ~0.52-0.57, GPD ρs ~0.68-0.72

### 6. FFT frequency estimator is unreliable for triage

FFT peak frequency on pointiness traces vs MW gold standard (265 cases):
- Spearman: 0.204 (very weak)
- Mean signed error: -0.60 Hz (systematic overestimation)
- 35% of cases off by >1 Hz

This caused our "balanced" S3 harvest to produce a heavily mid-range dataset instead of the intended uniform distribution.

## Model Progression Summary

| Version | Method | Combined ρs | N patients | Key advance |
|---------|--------|------------|-----------|-------------|
| v1-v5 | Various SP detectors | ~0.3-0.4 | 43 | Initial exploration |
| v6 | Ridge on 6 SP features | 0.476 | 43→202 | Pointiness+ACF features |
| v7 | GBM on 6 SP features | 0.686 | 335 | More data, tree models |
| v8 | RF 200 on 6 SP features | 0.604 | 594 | Even more data (harder) |
| **v9** | **CNN+Temporal Attention** | **0.640** | **594** | **Raw waveform learning** |

Note: v7→v8 Spearman dropped because the dataset expanded with harder cases, not because the model got worse.

## What We Tried (v9 additions)

### Channel-level PD detection (Phases 1-3b)
- **Phase 1**: Built channel-level dataset (9,310 channels, 815 patients) using laterality labels for ipsilateral/contralateral assignment. GBM-predicted laterality for 276 unlabeled patients.
- **Phase 2**: Logistic regression baseline (13 per-channel features) → 0.766 channel AUC, 0.902 patient AUC
- **Phase 3**: 1D CNN multi-task (PD + frequency) → 0.867 channel AUC, 0.981 patient AUC
- **Phase 3b**: 1D CNN + temporal attention → 0.870 channel AUC, 0.989 patient AUC, **0.640 patient freq Spearman**

### Discharge timing detection (Phase 4) — failed
- U-Net encoder-decoder outputting frame-level discharge probability (2000 samples)
- Trained on bootstrapped targets from pointiness peak detection
- 5-fold CV, all folds completed
- IPI-derived frequency: Spearman -0.08 (failure — noisy targets)
- Visualization built at results/discharge_timing_viewer.html

### PD morphological consistency scoring — limited utility
- Cross-correlation of successive discharge windows + CV of shape features
- 80% of cases scored LOW (<0.3) — threshold too strict
- Not useful as a quality filter

### Comprehensive "didn't help" list (all rounds)
CNN embeddings, CNN direct regression, DANN, k-NN features, >11 features, stacking, ensembles, ordinal regression, YIN/SRH, alternans detection, comb-fit scoring, DTW template matching, HMM windowed tracking, matched-filter envelope, GED standalone, HPS standalone, log feature transforms, per-expert training, frequency-bin weighting (for Ridge), oversampling rare frequencies, quantile/median regression, residual correction, piecewise models, PD probability as confidence score, PD morphological consistency scoring, discharge timing from bootstrapped peaks

## Evaluation Methodology

### Frequency estimation
- **LOPO** for handcrafted features (Ridge, RF, GBM): all segments from held-out patient excluded
- **5-fold patient-stratified CV** for CNN models (LOPO too expensive)
- Up to 5 segments per patient; patient prediction = mean of segment predictions
- MW gold standard frequency as ground truth
- Bootstrap 95% CIs on Spearman and MAE (10,000 iterations)

### Classification tasks
- **LOPO** for logistic Ridge; **5-fold CV** for tree models
- Subtype: accuracy, balanced accuracy, AUC, sensitivity/specificity
- Laterality: same metrics, on LPD patients with left/right labels only (exclude bilateral)

### Channel-level PD detection
- **5-fold patient-stratified CV** (patient-level split, not channel-level)
- Channel AUC and patient-level AUC (mean PD prob across channels)
- Multi-task: also evaluate per-channel frequency Spearman on PD+ channels

## Infrastructure

```
data/
├── eeg/                 ~2,511 .mat files (10s @ 200 Hz, 18ch bipolar)
├── labels/
│   ├── segments.csv     2,511 rows (segment registry)
│   ├── annotations.csv  3,821 rows (long format: segment × rater)
│   └── patients.csv     839 rows (gold standard + _original cols)
├── pd_channel_cache/    Channel dataset, CNN models (fold 0-4)
│   ├── channel_dataset.npz       9,310 channels, 815 patients
│   ├── cnn_fold{0-4}.pt          Basic CNN models
│   ├── cnn_attn_fold{0-4}.pt     CNN+Attention models
│   └── discharge_fold{0-4}.pt    Discharge detector models
└── _archive/            Source annotation CSVs, previous round data

code/
├── Core
│   ├── optimization_harness_v2.py       Evaluation engine (LOPO CV, bootstrap CIs)
│   ├── pd_pointiness_acf.py             Core signal processing
│   └── update_dashboard_v2.py           Dashboard data updater
│
├── Experiments
│   ├── exp_opt_freq_models.py           Frequency model experiments (14)
│   ├── exp_opt_freq_features.py         Frequency feature experiments (10)
│   ├── exp_opt_freq_bias.py             Bias correction experiments (8)
│   ├── exp_opt_freq_subtype.py          Subtype-specific frequency (5)
│   ├── exp_opt_subtype_class.py         Subtype classification (9)
│   └── exp_opt_lat_class.py             Laterality classification (9)
│
├── Channel-Level PD Detection
│   ├── pd_channel_detector/
│   │   ├── build_channel_dataset.py     Phase 1: data assembly
│   │   ├── channel_features.py          Phase 2: per-channel features
│   │   ├── train_baseline.py            Phase 2: logistic baseline
│   │   ├── channel_cnn.py              Phase 3: CNN + Attention architectures
│   │   ├── train_cnn.py                Phase 3: basic CNN training
│   │   ├── train_cnn_attention.py      Phase 3b: attention CNN training
│   │   ├── discharge_detector.py       Phase 4: U-Net discharge detector
│   │   ├── train_discharge_detector.py Phase 4: discharge training
│   │   ├── visualize_discharges.py     Phase 4: EEG overlay visualization
│   │   └── validate_quality_predictor.py Phase 4: PD prob vs freq error
│   └── pd_consistency_score.py          Morphological consistency scoring
│
├── Data Harvesting
│   ├── harvest_lpd_segments.py          S3 LPD harvesting pipeline
│   └── harvest_seizure_lpd_candidates.py Seizure folder PD candidate finder
│
└── Interactive Tools
    ├── generate_misclass_reviewer.py    Error review (3 tabs: subtype/lat/freq)
    ├── generate_freq_annotation_viewer.py  Frequency annotation tool
    └── generate_consistency_viewer.py    PD consistency score viewer

results/
├── optimization_dashboard_v2.html      Live dashboard (42+ experiments)
├── optimization_runs_v2/               JSON result files
├── misclass_reviewer.html              Interactive error reviewer
├── freq_annotation_viewer.html         Frequency annotation tool
├── discharge_timing_viewer.html        Discharge timing visualization
├── pd_prob_vs_freq_error.html          PD prob vs freq error scatter
├── consistency_viewer.html             PD consistency viewer
├── harvest_dashboard.html              LPD harvest progress tracker
├── harvest_seizure_dashboard.html      Seizure candidate tracker
├── cnn_attn_learning_curves.html       CNN training curves
└── cnn_attn_patient_freq.json          CNN frequency results
```

## Next Steps

### High priority
1. **Phase 3c: Multi-channel spatial+temporal attention** — single model jointly predicting subtype, laterality, and frequency with spatial attention across 18 channels + temporal attention within channels. Phase 3b showed +0.036 Spearman improvement, justifying this next step.
2. **Hand-annotate discharge times** on 20-30 cases to provide clean training targets for the discharge timing detector. The bootstrapped approach failed because pointiness peaks are too noisy.
3. **Receive Sahar's annotations** → MW-Sahar IRR → fair algorithm-vs-expert comparison using `_original` columns.

### Medium priority
4. **Ensemble CNN+Attention with handcrafted features** — the two approaches may capture complementary signal. A simple average or stacking model could push Spearman beyond 0.640.
5. **CNN+Attention for subtype/laterality** — the CNN already learns these implicitly through the PD detection head. Explicitly adding classification heads could match or beat the RF/GBM results.
6. **Faster CNN training** — current 5-fold CV takes ~2+ hours. Options: fewer epochs with learning rate warmup, mixed precision, or smaller model for rapid iteration.

### Lower priority
7. **High-frequency LPD gap**: very few >2 Hz cases. Seizure folder harvesting found some candidates but pass rate ~0.1%. May be an inherent data limitation.
8. **LRDA laterality annotation** — still pending.
9. **RDA frequency estimation** — not updated since v7 (Spearman 0.840 on 23 patients). Could benefit from the CNN approach.

## Questions for Review

1. **CNN+Attention (0.640) vs handcrafted RF (0.604)**: The CNN wins by +0.036. Is this enough to justify the added complexity for publication, or should we present both as complementary approaches?

2. **Phase 3c (multi-channel model)**: Is the potential gain worth the implementation complexity and training time? Or should we focus on validating what we have?

3. **Discharge timing**: The bootstrapped approach failed. Is hand-annotating 20-30 cases worth the effort, or should we accept that discharge timing is a harder problem for later?

4. **The orthogonality finding** (PD prob ⊥ freq error): This is scientifically interesting. Is it worth highlighting in the paper as a negative result?

5. **Label quality**: We've done 2 rounds of correction. The CNN was trained on the same labels it's evaluated against (5-fold CV, not LOPO). Should we be concerned about label memorization, or is 5-fold CV sufficient?
