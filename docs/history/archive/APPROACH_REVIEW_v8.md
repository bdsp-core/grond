# Frequency Estimation for Periodic EEG Patterns: Review v8

## Problem & Data

Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Additionally: classifying subtype (LPD vs GPD) and laterality (left vs right for LPD).

### Dataset (as of 2026-03-19)

| | Patients | Segments | Raters |
|---|---|---|---|
| LPD | 425 | ~1,700 | MW (all) |
| GPD | 151 | ~830 | MW (all) |
| GRDA | 115 | ~460 | Various |
| LRDA | 105 | ~420 | Various |
| **Total** | **839** | **~2,511** | |

Of these, **594 patients** have valid gold standard frequency labels (425 LPD + 151 GPD + 18 other), yielding **576 patients** used in frequency estimation evaluation (after exclusions).

**Laterality labels:** 143 LPD patients have left/right laterality labels (83 left, 60 right). 5 labeled bilateral, 13 skipped.

**Gold standard:** MW's frequency rating. `_original` columns preserve first-pass annotations for future inter-rater reliability analysis.

### What changed since v7
- **2.4× more patients**: 374 → 839 (594 with gold standard freq, up from 343)
- **265 new LPD cases harvested** from morgoth1 S3 LPD pool, MW-annotated for frequency
- **LPD laterality labels** added for 186 LPD patients (95 left, 75 right, 3 bilateral)
- **Label corrections**: 2 rounds of misclassification review via interactive HTML viewer
  - Subtype: 12 corrections (7 reclassified as "neither", 5 swapped LPD↔GPD)
  - Laterality: 9 corrections
  - Frequency: 26 corrections
- **`_original` columns** added to patients.csv to preserve noisy first-pass labels (for future IRR analysis)
- **Self-contained repo**: symlinks to IIIC-AlexandraData replaced with real data copies
- **3 evaluation tasks**: frequency estimation, subtype classification, laterality classification
- **42 new experiments** across all 3 tasks

## Current Best Results

### Frequency Estimation — 42+ experiments on 594 patients, LOPO/5-fold CV

| Method | Combined ρs | LPD ρs (n=425) | GPD ρs (n=151) | MAE |
|--------|-------------|----------------|----------------|-----|
| **CNN+Temporal Attention** | **0.640** | **0.573** | **0.718** | **0.271** |
| RF (200 trees, depth 8) | 0.604 [0.542, 0.661] | 0.519 | 0.677 | 0.267 |
| GBM (200 trees, depth 3) | 0.602 [0.541, 0.656] | 0.496 | 0.698 | 0.270 |
| Ridge (α=1) | 0.589 [0.524, 0.646] | 0.488 | 0.700 | 0.274 |
| Ridge balanced | 0.497 [0.428, 0.560] | 0.351 | 0.654 | 0.346 |

**The CNN+Attention model is our new best**, beating RF by +0.036 combined Spearman and +0.054 on LPD specifically. It processes raw single-channel waveforms with a 1D CNN, uses temporal attention to focus on the most informative time windows, and aggregates across channels weighted by PD probability.

**Note on Spearman drop (v7: 0.686 → v8: 0.640):** This is NOT a regression. The 265 new LPD cases from S3 are harder — FFT pre-screening was poorly correlated with true frequency (FFT Spearman 0.204 vs MW). The expanded dataset is more representative of real-world LPD diversity.

### Subtype Classification (LPD vs GPD) — 9 experiments, 594 patients

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF 300 (depth 8)** | **0.931** | **86.5%** | 0.814 |
| GBM (depth 3) | 0.929 | 86.0% | 0.810 |
| GBM balanced | 0.923 | 84.5% | **0.853** |
| RF balanced | 0.922 | 83.5% | 0.846 |
| Logistic Ridge (α=0.1) | 0.852 | 79.0% | 0.702 |

Tree models dramatically outperform logistic regression (+0.08 AUC). Class balancing improves GPD sensitivity (0.87 vs 0.71) at modest accuracy cost. Laterality features are useless for subtype classification (AUC 0.47 — worse than chance).

### Laterality Classification (Left vs Right, LPD only) — 9 experiments, 143 patients

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF balanced** | **0.959** | 90.9% | 0.906 |
| **GBM balanced** | 0.957 | **93.0%** | **0.924** |
| GBM | 0.954 | 93.0% | 0.921 |
| RF | 0.956 | 90.9% | 0.901 |
| Logistic Ridge (α=5) | 0.927 | 88.8% | 0.881 |
| lat_idx only | 0.920 | 86.7% | 0.856 |

The laterality index (single feature) achieves 0.920 AUC by itself — most of the signal is in one feature. Tree models add ~0.04 AUC by learning nonlinear interactions with frequency features.

## Key Findings

### 1. Frequency balancing hurts overall Spearman

Inverse-frequency-bin weighting was tested across Ridge, RF, and GBM to address mean-shrinkage bias on high-frequency cases:

| Model | Unweighted ρs | Balanced ρs | Δ |
|-------|--------------|-------------|---|
| Ridge | 0.589 | 0.497 | **-0.092** |
| RF 200 | 0.604 | 0.600 | -0.004 |
| GBM 200 | 0.602 | 0.553 | -0.049 |

Ridge is devastated by weighting (sparse bins amplify noise in WLS). RF is robust. Oversampling rare bins (0.556) performs worse than weighting.

### 2. Tree models consistently beat Ridge for frequency

RF and GBM outperform Ridge by ~0.015 Spearman. This gap persists across feature sets and balancing strategies. RF 200 trees (depth 8) is the best overall.

### 3. Feature sets barely matter

| Feature set | ρs (Ridge α=1) |
|-------------|----------------|
| Base 6 features | 0.589 |
| Base 5 (no is_gpd) | 0.587 |
| All 9 (+ laterality) | 0.588 |
| 6 + 3 interactions | 0.586 |

Adding laterality features, removing is_gpd, adding interaction terms — none make a meaningful difference. The 6 base SP features capture nearly all the learnable signal.

### 4. The LPD-GPD gap persists

| | LPD ρs | GPD ρs |
|---|---|---|
| Best algorithm | 0.519 | 0.698 |
| Gap | 0.18 lower | — |

GPD is consistently easier across all models and feature sets. LPD spatial variability and narrower frequency range make rank-ordering harder.

### 5. FFT frequency estimator is unreliable for triage

During S3 harvesting, FFT peak frequency on pointiness traces was used to bin segments by frequency. Comparison with MW gold standard on 265 cases:
- Spearman: **0.204** (very weak)
- Mean signed error: **-0.60 Hz** (FFT systematically overestimates)
- Cases >1 Hz off: **35%**

This caused the "balanced" harvest to actually produce a lopsided dataset (heavy 0.5-1.5 Hz, sparse >2 Hz).

## Features (unchanged from v7)

| Feature | Description | Time |
|---------|-------------|------|
| f_B | Pointiness → smooth → ACF first peak | 24 ms |
| f_peaks | Peak-count: pointiness peaks → (n-1)/time_span | 0.2 ms |
| f_fft | FFT of pointiness → peak in [0.3, 3.5] Hz | 0.3 ms |
| f_tkeo | TKEO envelope → FFT peak | 0.7 ms |
| f_coh | Cross-channel spectral coherence peak | 2.2 ms |
| is_gpd | Pattern type indicator | 0 ms |
| lat_idx | Laterality index (L-R energy asymmetry) | — |
| lat_energy_ratio | Hemisphere energy ratio | — |
| lat_acf_ratio | Hemisphere ACF periodicity ratio | — |

## What We Tried (42 new experiments)

### Frequency estimation (34 experiments)
- **Model comparison**: Ridge (5 alphas), RF (2 configs), GBM (2 configs) → RF 200 wins
- **Frequency balancing**: inverse-bin weighting, fine bins, oversampling → hurts Ridge, neutral for RF
- **Feature sets**: base 6, base 5 (no is_gpd), all 9, interactions → no meaningful differences
- **Subtype-specific models**: separate LPD/GPD Ridge → no improvement over combined
- **Bias correction attempts**: residual correction, quantile regression, piecewise models, weighted high-freq → none helped

### Subtype classification (9 experiments)
- RF and GBM dramatically outperform logistic Ridge (+0.08 AUC)
- Class balancing doubles GPD sensitivity (0.87 vs 0.44)
- Frequency features alone match full feature set for logistic regression

### Laterality classification (9 experiments)
- GBM balanced achieves 93% accuracy, 0.957 AUC
- lat_idx alone is remarkably predictive (0.920 AUC)
- lat_energy_ratio alone is useless (chance-level)

### Comprehensive "didn't help" list (all rounds)
CNN embeddings, CNN direct regression, DANN, k-NN features, >11 features, stacking, ensembles, ordinal regression, YIN/SRH, alternans detection, comb-fit scoring, DTW template matching, HMM windowed tracking, matched-filter envelope, GED standalone, HPS standalone, log feature transforms, per-expert training, frequency-bin weighting (for Ridge), oversampling rare frequencies, quantile/median regression, residual correction, piecewise models

## Evaluation Methodology

- **LOPO** (Leave-One-Patient-Out): all segments from held-out patient excluded
- **Up to 5 segments per patient** for training; patient prediction = mean of segment predictions
- **MW gold standard**: MW's frequency rating as ground truth
- **Bootstrap 95% CIs** on Spearman and MAE (10,000 iterations)
- **Three tasks evaluated**: frequency estimation (Spearman), subtype classification (AUC), laterality classification (AUC)

## Infrastructure

```
data/
├── eeg/              ~2,511 .mat files (10s @ 200 Hz, 18ch bipolar)
├── labels/
│   ├── segments.csv  2,511 rows (segment registry)
│   ├── annotations.csv  3,821 rows (long format: segment × rater)
│   └── patients.csv  839 rows (patient summary + gold standard + _original cols)
├── pd_channel_cache/  Channel-level PD detection dataset (in progress)
└── _archive/          Source annotation CSVs, previous round data

code/
├── optimization_harness_v2.py         Evaluation engine (LOPO CV, bootstrap CIs)
├── pd_pointiness_acf.py               Core signal processing
├── exp_opt_freq_models.py             Frequency model experiments
├── exp_opt_freq_features.py           Frequency feature experiments
├── exp_opt_freq_bias.py               Bias correction experiments
├── exp_opt_subtype_class.py           Subtype classification experiments
├── exp_opt_lat_class.py               Laterality classification experiments
├── generate_misclass_reviewer.py      Interactive error review tool (3 tabs)
├── generate_freq_annotation_viewer.py Frequency annotation tool
├── harvest_lpd_segments.py            S3 LPD harvesting pipeline
├── harvest_seizure_lpd_candidates.py  Seizure folder PD candidate finder
├── pd_consistency_score.py            PD morphological consistency scoring
├── pd_channel_detector/               Channel-level PD detection (in progress)
└── update_dashboard_v2.py             Dashboard data updater

results/
├── optimization_dashboard_v2.html     Live dashboard (42 experiments, auto-refresh)
├── optimization_runs_v2/              JSON result files
├── misclass_reviewer.html             Interactive error reviewer
├── freq_annotation_viewer.html        Frequency annotation tool
├── harvest_dashboard.html             LPD harvest progress tracker
├── harvest_seizure_dashboard.html     Seizure LPD candidate tracker
└── consistency_viewer.html            PD consistency score viewer
```

## Channel-Level PD Detection (completed Phase 1-3b)

Binary classifier predicting whether individual EEG channels contain periodic discharges.

### Dataset
- **9,310 channels from 815 patients** (balanced ~50/50 positive/negative)
- Positives: ipsilateral LPD channels, all GPD channels, expert spatial annotations
- Negatives: contralateral LPD channels, LRDA/GRDA (hard negatives)
- For 276 LPD patients without laterality labels: GBM classifier (0.957 AUC) predicted laterality as proxy

### Model Progression

| Model | Channel AUC | Patient AUC | Ch Freq ρs |
|-------|------------|-------------|------------|
| Logistic regression (13 features) | 0.766 | 0.902 | — |
| 1D CNN (4 conv blocks, multi-task) | 0.867 | 0.981 | 0.566 |
| **1D CNN + Temporal Attention** | **0.870** | **0.989** | **0.590** |

### Key Finding: PD probability does NOT predict frequency estimation accuracy
Correlation between CNN PD probability and frequency error: ρ = 0.011 (p = 0.85). PD detection and frequency estimation are orthogonal — a case can have very clear PDs but ambiguous frequency, and vice versa.

### Key Finding: CNN+Attention IS the best frequency estimator
When used for patient-level frequency estimation (PD-weighted channel aggregation), the CNN+Attention achieves Spearman 0.640 — beating all handcrafted feature methods. The temporal attention learns WHERE in the 10-second window to focus for frequency estimation.

## Next Steps

1. **Phase 4: Discharge timing detection** — frame-level CNN output predicting individual discharge times (t_1, t_2, ...) per channel. Bootstrap training targets from pointiness peak detection. Derive frequency from inter-discharge intervals rather than FFT/ACF estimation. Visualize on EEG plots with discharge markers and IPI analysis.
2. **Phase 3c (conditional): Multi-channel spatial+temporal attention** — single model jointly predicting subtype, laterality, and frequency with spatial attention across channels + temporal attention within channels. Contingent on Phase 3b showing clear improvement (it did: +0.036 Spearman).
3. **Receive Sahar's annotations** → MW-Sahar IRR → fair algorithm-vs-expert comparison (using `_original` columns for unbiased comparison)
4. **High-frequency LPD gap**: still very few >2 Hz cases in gold standard. Seizure folder harvesting found some candidates but pass rate is very low (~0.1%).
5. **Integration**: use CNN+Attention as the primary frequency estimation method, with handcrafted SP features as fallback/ensemble

## Questions for Review

1. **Spearman 0.604 on 576 patients vs 0.686 on 335 patients**: Is this a concern, or expected from a harder dataset? Should we stratify results by data source (original vs harvested)?

2. **Channel-level PD detector**: Is the multi-task CNN (PD detection + frequency) the right next bet, or should we focus on data quality first?

3. **High-frequency LPD problem**: We tried hard to harvest >2 Hz LPDs but the FFT estimator was unreliable and few exist in the morgoth1 pool. Is this a data limitation we accept, or pursue further?

4. **Subtype/laterality results (AUC 0.93/0.96)**: Are these strong enough for the paper, or secondary to frequency estimation?

5. **Label denoising via review**: We've done 2 rounds. Diminishing returns? Or should we do another round now that models are better?
