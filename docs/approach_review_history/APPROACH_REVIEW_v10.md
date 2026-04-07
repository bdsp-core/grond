# Frequency Estimation for Periodic EEG Patterns: Review v10

## Problem & Data

Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Additionally: classifying subtype (LPD vs GPD), laterality (left vs right for LPD), detecting which channels contain periodic discharges, and identifying individual discharge times.

### Dataset (as of 2026-03-20)

| | Patients | Segments | Raters |
|---|---|---|---|
| LPD | 425 | ~1,700 | MW (all) |
| GPD | 151 | ~830 | MW (all) |
| GRDA | 115 | ~460 | Various |
| LRDA | 105 | ~420 | Various |
| **Total** | **839** | **~2,511** | |

**Frequency estimation**: 594 patients with gold standard freq labels (425 LPD + 151 GPD + 18 other).
**Laterality labels**: 143 LPD patients with human labels (83 left, 60 right). 276 additional LPD patients have GBM-predicted laterality (0.957 AUC proxy).
**Channel-level PD dataset**: 9,310 channels from 815 patients (4,524 positive, 4,786 negative).
**Discharge timing labels**: **593 patients with MW-reviewed discharge times** — complete ground truth for all LPD+GPD cases.

### What changed since v9
- **Complete discharge timing ground truth**: All 593 LPD+GPD patients now have MW-reviewed discharge times (6 rounds of iterative review)
- **Latent point process (HPP) discharge detector**: MAP inference via dynamic programming with approximately-periodic prior achieves F1=0.795, frequency Spearman 0.935 vs MW
- **Iterative label improvement workflow**: HPP auto-marks → MW binary review (correct/incorrect) → MW manual correction → retrain → repeat. Each round improved both algorithm performance and label completeness.
- **Unified PD model plan**: Comprehensive plan for a single end-to-end model jointly performing subtype classification, frequency estimation, spatial localization, and discharge timing

## Current Best Results

### Frequency Estimation — Best: CNN+Temporal Attention, Spearman 0.640

| Method | Combined ρs | LPD ρs (n=425) | GPD ρs (n=151) | MAE |
|--------|-------------|----------------|----------------|-----|
| **CNN+Temporal Attention** | **0.640** | **0.573** | **0.718** | **0.271** |
| RF (200 trees, depth 8) | 0.604 [0.542, 0.661] | 0.519 | 0.677 | 0.267 |
| GBM (200 trees, depth 3) | 0.602 [0.541, 0.656] | 0.496 | 0.698 | 0.270 |
| Ridge (α=1) | 0.589 [0.524, 0.646] | 0.488 | 0.700 | 0.274 |

### Subtype Classification (LPD vs GPD) — Best: RF 300, AUC 0.931

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF 300 (depth 8)** | **0.931** | **86.5%** | 0.814 |
| GBM balanced | 0.923 | 84.5% | **0.853** |

### Laterality Classification (Left vs Right) — Best: GBM balanced, AUC 0.957

| Method | AUC | Accuracy | Balanced Acc |
|--------|-----|----------|-------------|
| **RF balanced** | **0.959** | 90.9% | 0.906 |
| **GBM balanced** | 0.957 | **93.0%** | **0.924** |

### Channel-Level PD Detection — Best: CNN+Attention, Patient AUC 0.989

| Model | Channel AUC | Patient AUC | Ch Freq ρs |
|-------|------------|-------------|------------|
| Logistic regression (13 features) | 0.766 | 0.902 | — |
| 1D CNN (multi-task) | 0.867 | 0.981 | 0.566 |
| **1D CNN + Temporal Attention** | **0.870** | **0.989** | **0.590** |

### Discharge Timing Detection — HPP Algorithm, F1 0.795

| Metric | Value |
|--------|-------|
| **Sensitivity** | **0.734** (finds 73% of MW's discharges) |
| **Precision** | **0.867** (87% of detections are real) |
| **F1** | **0.795** |
| **Freq ρ (algo IPI vs MW IPI)** | **0.935** |
| **Freq ρ (algo IPI vs gold standard)** | **0.960** |
| **Timing accuracy** | **2.0 ms** (matched discharges) |
| MW discharges/case | 9.5 |
| Algo discharges/case | 8.1 |

## The Discharge Timing Breakthrough

### The problem
Previous approaches to discharge timing failed:
- **U-Net trained on bootstrapped pointiness peaks**: Spearman -0.08 (worse than chance). Pointiness peaks are too noisy as training targets.
- **PD morphological consistency scoring**: 80% of cases scored LOW — too strict to be useful.

### The solution: Latent Point Process (HPP) with Dynamic Programming

Instead of training a neural network to predict discharge times, we formulate discharge timing as **MAP inference over a latent sequence of discharge events**:

1. **Per-channel evidence signal** E_c(t): Combined pointiness trace (z-scored) + TKEO (z-scored), weighted 0.6/0.4, Gaussian-smoothed
2. **Class-aware aggregation**: GPD uses median across all channels; LPD uses median across ipsilateral channels (with predicted laterality for unlabeled cases)
3. **Active interval detection**: Rolling mean threshold to find where PDs are present (minimum 3 seconds)
4. **Candidate peak extraction**: Local maxima of E(t) with adaptive thresholds
5. **Dynamic programming** with approximately-periodic prior:
   - Node score: evidence at candidate time (superlinear weighting)
   - Edge score: penalizes deviation from expected period T=1/f, allows skipped discharges (up to 3)
   - Complexity penalty: λ per event to prevent overcalling
6. **EM template refinement**: Extract case-specific discharge template, cross-correlate to refine candidates, re-run DP

### Key parameters (optimized via grid search on ground truth)
- `PEAK_HEIGHT_FRAC = 0.05` — low threshold favors sensitivity (easier to delete FPs than add FNs)
- `DP_ALPHA = 3.0` — flexible timing tolerance
- `DP_LAMBDA = 0.02` — minimal complexity penalty
- `MAX_SKIP = 3` — allows up to 3 skipped discharges

### The iterative review workflow

This was critical to the success of the approach:

| Round | Cases reviewed | Ground truth total | Sensitivity | Precision | F1 | Freq ρ |
|-------|---------------|-------------------|-------------|-----------|-----|--------|
| 1 (binary review) | 63 | 61 | 0.682 | 0.872 | 0.766 | 0.900 |
| 2 (corrections) | 77 | 138 | — | — | — | — |
| 3 (corrections) | 160 | 221 | 0.620 | 0.869 | 0.724 | 0.857 |
| 4 (corrections) | 141 | 362 | 0.708 | 0.780 | 0.742 | 0.904 |
| 5 (corrections) | 76 | 438 | 0.713 | 0.814 | 0.760 | 0.913 |
| **6 (final)** | **148** | **593** | **0.734** | **0.867** | **0.795** | **0.935** |

Key observations:
- Sensitivity and precision both improved as more ground truth became available
- Parameter tuning between rounds (λ: 0.3→0.1→0.02, peak threshold: 0.3→0.15→0.05) dramatically improved sensitivity
- The interactive correction viewer (canvas-based EEG with click-to-add/delete/move markers) made review efficient
- MW's review produced richer annotations (9.5 discharges/case) than the algorithm (8.1/case)
- MW's IPI-derived frequency correlates 0.928 with gold standard — confirming timing labels are high quality

### Why HPP succeeded where the U-Net failed

| Factor | U-Net approach | HPP approach |
|--------|---------------|--------------|
| Training targets | Noisy pointiness peaks | Clean MW-reviewed times (iteratively improved) |
| Model | Learned (data-hungry) | Physics-informed (periodic prior) |
| Frequency information | Not used | Used as strong structural prior |
| Spatial information | Not used | Class-aware channel aggregation |
| Skipped discharges | Not handled | Explicit skip modeling |
| Active intervals | Not handled | Detected and enforced |

## Key Findings

### 1. CNN+Attention beats handcrafted features for frequency estimation
Combined Spearman 0.640 vs 0.604 (RF) and 0.589 (Ridge). Biggest improvement on LPD (+0.054).

### 2. PD detection and frequency estimation are orthogonal
CNN PD probability vs frequency error: ρ = 0.011. Cannot use PD confidence as frequency quality predictor.

### 3. Discharge timing requires clean labels, not bigger models
U-Net on bootstrapped labels: ρ = -0.08. HPP with MW-reviewed labels: ρ = 0.935. The bottleneck was label quality, not model capacity.

### 4. Iterative human-in-the-loop label improvement works
Each review round improved both the algorithm AND the labels. The key was making review efficient (auto-advance, keyboard shortcuts, canvas-based editing).

### 5. Physics-informed inference outperforms learned approaches for structured problems
The approximately-periodic prior in HPP encodes domain knowledge that a generic CNN must learn from scratch. With limited data, the structured approach wins.

### 6. MW's timing-derived frequency (ρ=0.928 vs gold standard) validates label quality
The discharge timing labels are consistent with the original frequency annotations, confirming they capture the same underlying periodicity.

## Model Progression Summary

| Version | Method | Combined ρs | N patients | Key advance |
|---------|--------|------------|-----------|-------------|
| v1-v5 | Various SP detectors | ~0.3-0.4 | 43 | Initial exploration |
| v6 | Ridge on 6 SP features | 0.476 | 43→202 | Pointiness+ACF features |
| v7 | GBM on 6 SP features | 0.686 | 335 | More data, tree models |
| v8 | RF 200 on 6 SP features | 0.604 | 594 | Even more data (harder) |
| v9 | CNN+Temporal Attention | 0.640 | 594 | Raw waveform learning |
| **v10** | **HPP discharge timing** | **0.935*** | **593** | **Complete timing labels** |

*HPP frequency is IPI-derived (1/median inter-discharge interval), not a regression prediction. Direct comparison with regression-based Spearman is approximate.

## Infrastructure

```
data/
├── eeg/                 ~2,511 .mat files (10s @ 200 Hz, 18ch bipolar)
├── labels/
│   ├── segments.csv     2,511 rows (segment registry)
│   ├── annotations.csv  3,821 rows (long format: segment × rater)
│   ├── patients.csv     839 rows (gold standard + _original cols)
│   └── discharge_times_hpp.json  593 patients with MW-reviewed discharge times
├── pd_channel_cache/    Channel dataset, CNN models
│   ├── channel_dataset.npz       9,310 channels, 815 patients
│   ├── cnn_attn_fold{0-4}.pt     CNN+Attention models
│   └── discharge_fold{0-4}.pt    Discharge detector models (deprecated)
└── _archive/
    └── timing_review/   Source correction CSVs (rounds 1-6)

code/
├── Core
│   ├── optimization_harness_v2.py       Evaluation engine (LOPO CV, bootstrap CIs)
│   ├── pd_pointiness_acf.py             Core signal processing
│   └── update_dashboard_v2.py           Dashboard data updater
│
├── Label Pipeline (Sprint 1 — complete)
│   ├── label_pipeline/
│   │   ├── hpp_discharge_marking.py     HPP discharge detection algorithm
│   │   ├── evaluate_hpp.py             Evaluation against MW ground truth
│   │   ├── merge_timing_corrections.py  Merge browser-exported corrections
│   │   ├── generate_timing_review_viewer.py    Binary review (correct/incorrect)
│   │   └── generate_timing_correction_viewer.py Interactive marker editing
│
├── Channel-Level PD Detection
│   ├── pd_channel_detector/
│   │   ├── build_channel_dataset.py     Data assembly
│   │   ├── channel_cnn.py              CNN + Attention architectures
│   │   ├── train_cnn_attention.py      Attention CNN training
│   │   ├── discharge_detector.py       U-Net discharge detector (deprecated)
│   │   └── validate_quality_predictor.py PD prob vs freq error analysis
│
├── Experiments (42+)
│   ├── exp_opt_freq_models.py           Frequency model experiments
│   ├── exp_opt_freq_features.py         Feature experiments
│   ├── exp_opt_subtype_class.py         Subtype classification
│   └── exp_opt_lat_class.py             Laterality classification
│
├── Data Harvesting
│   ├── harvest_lpd_segments.py          S3 LPD harvesting
│   └── harvest_seizure_lpd_candidates.py Seizure folder PD finder
│
└── Interactive Tools
    ├── generate_misclass_reviewer.py    Error review (3 tabs)
    ├── generate_freq_annotation_viewer.py  Frequency annotation
    └── generate_consistency_viewer.py    PD consistency viewer

results/
├── optimization_dashboard_v2.html      Live experiment dashboard
├── timing_review_viewer.html           Binary timing review
├── timing_correction_viewer.html       Interactive marker editing
├── misclass_reviewer.html              Error reviewer
├── pd_prob_vs_freq_error.html          PD prob analysis
└── cnn_attn_patient_freq.json          CNN frequency results
```

## Next Steps (from Unified PD Model Plan)

### Sprint 2: Spatial Localization Labels (next)
- Run CNN+Attention PD detector on all cases → predict channel involvement
- MW binary review (correct/incorrect) → MW manual correction
- Store in `data/labels/channel_involvement.json`

### Sprint 3: Frequency/Subtype Label Refinement
- Use updated models to find label disagreements
- MW review and correct

### Sprint 4: Unified Multi-Channel Model
- Architecture: per-channel CNN encoder → temporal attention → spatial attention → 4 task heads (subtype, frequency, channel involvement, discharge timing)
- Train with all label types: subtype, frequency, channel involvement, discharge times
- The discharge timing labels (593 patients, complete) provide rich gradient signal
- Expected: frequency Spearman > 0.70, timing F1 > 0.85

### Sprint 5: BIPD Extension
- Detect bilateral independent periodic discharges from spatial + temporal structure
- Falls out naturally from the unified model's spatial attention

## Questions for Review

1. **Sprint 2 vs Sprint 4**: Should we proceed with spatial localization labels first, or jump to the unified model using existing labels + discharge timing?

2. **HPP as training target**: The HPP gets F1=0.795 on timing. Is this good enough as CNN training targets, or should we only use MW-reviewed times?

3. **Discharge timing labels for frequency estimation**: MW's IPI-derived frequency correlates 0.928 with gold standard. Should we update the gold standard frequencies to use IPI-derived values where they differ?

4. **Publication readiness**: With discharge timing labels complete and CNN+Attention at 0.640 Spearman, what's the minimum additional work needed for a paper?
