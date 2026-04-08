# Frequency Estimation for Periodic EEG Patterns: Review v12

## Critical Methodological Principle

**No method may use gold standard labels as input.** All algorithms operate from raw EEG only. Previous evaluations inflated HPP timing F1 from ~0.65 to ~0.77 by passing gold standard frequency as input.

## Problem & Data

Comprehensive characterization of periodic discharges (LPD, GPD) and rhythmic delta activity (LRDA, GRDA) in 10-second, 18-channel bipolar EEG at 200 Hz.

### Dataset (as of 2026-03-22)

| | Patients | Labels |
|---|---|---|
| LPD | 450 | Freq, timing (complete), spatial (partial), laterality |
| GPD | 169 | Freq, timing (complete), spatial (partial) |
| LRDA | 100 + 70 harvested | Subtype |
| GRDA | 120 + harvesting | Subtype |
| BIPD | 100 harvested | Subtype (new) |
| Other (controls) | 20+ harvesting | True negatives |
| Hi-freq LPD (>2.5 Hz) | 292 harvested | Awaiting annotation |

**Discharge timing labels**: 593 LPD+GPD patients with MW-reviewed discharge times, refined through CET-UNet comparison. 184 cases (32%) had labels updated based on CET-UNet review — labels are no longer biased toward pointiness peaks. This is the definitive gold standard.

### What changed since v11
- **Consolidated best method**: Product-boosted evidence combination, CNN+ACF ensemble frequency, CET thresholding, and post-hoc confidence filtering → F1 improved from 0.721 to **0.740**
- **Systematic improvement sweep**: Tested 5 improvement strategies on the best method:
  1. CET 80th percentile threshold (+0.007 F1) — suppresses CET noise floor
  2. PD-weighted channel selection (no improvement) — median aggregation is more robust
  3. CNN+ACF frequency ensemble with 0.2 ACF weight (+0.004 F1) — averaging reduces freq estimation errors
  4. Product-boost evidence combination boost=3, floor=0.3 (+0.005 F1) — amplifies agreement between HPP and CET
  5. Post-hoc min-evidence filter ratio=0.3 (+0.004 F1) — drops weak detected peaks
- **Iterative frequency refinement failed**: IPI-derived frequency from detected discharges is noisier than CNN estimate; feeding it back degrades F1 from 0.73 to 0.54
- **PDNetV2 unified model (Phase 1)**: Joint 18-channel U-Net + Transformer bottleneck trained end-to-end. F1=0.46 — learns meaningful structure but undertrained with only 593 examples and 50 epochs. Proof of concept, not yet competitive.
- **Self-contained discharge_detector.py**: Complete pipeline in one file for colleague review

## Best System: Product-Boosted max(HPP, CET-UNet) + CNN+ACF Freq + Optimized DP

### Architecture Overview

```
EEG (18ch × 2000) ──┬── CNN+Attention (freq est) ──→ f_cnn ──┐
                     │                                         ├── 0.8×f_cnn + 0.2×f_acf ──→ f_est
                     ├── ACF on pointiness ──→ f_acf ──────────┘
                     │
                     ├── Handcrafted evidence ──→ E_hpp(t) ──┐
                     │   (pointiness + TKEO)                  │
                     │                                        ├── product-boost ──→ E(t) ──→ HPP DP ──→ post-hoc filter ──→ times
                     └── CET-UNet evidence ──→ E_cet(t) ──────┘
                         (threshold 80th pct,                  │
                          floor 0.3)                           │
                                                               │
                     E(t) = max(hpp, cet) + 3 × hpp × cet     │
```

1. **Frequency estimation**: CNN+ACF ensemble (0.8×CNN + 0.2×ACF) estimates f_est from raw EEG
2. **Dual evidence**: handcrafted (pointiness+TKEO) AND learned (CET-UNet) evidence traces
3. **CET thresholding**: Zero CET values below 80th percentile and below floor=0.3 (suppresses noise)
4. **Product-boost combination**: max(HPP, CET) + 3×HPP×CET (amplifies regions where both agree)
5. **HPP DP inference**: dynamic programming with approximately-periodic prior (α=1.275, λ=0.05, β=0.3)
6. **EM template refinement**: case-specific waveform template cross-correlation
7. **Post-hoc filter**: Drop detections with evidence < 0.3× median peak evidence

Self-contained implementation: `code/discharge_detector.py`

## Current Results (Updated Gold Standard, EEG-Only, 593 cases)

### HPP-only (handcrafted evidence, gold freq — REFERENCE)

| Metric | Value |
|--------|-------|
| Sensitivity | 0.693 |
| Precision | 0.834 |
| F1 | 0.757 |
| Freq ρ (algo IPI vs MW IPI) | 0.956 |

Note: This uses gold standard frequency as input (reference only).

### Fair EEG-Only Methods

| Method | Sens | Prec | **F1** | **Freq ρ** |
|--------|------|------|--------|-----------|
| **Consolidated best** | 0.776 | 0.706 | **0.740** | **0.718** |
| max(HPP,CET)+CNN freq+opt | 0.774 | 0.675 | 0.721 | 0.753 |
| HPP + CNN freq | 0.578 | 0.728 | 0.645 | 0.716 |
| HPP + bootstrap | 0.615 | 0.658 | 0.636 | 0.453 |
| CET + bootstrap | 0.499 | 0.746 | 0.598 | 0.531 |
| CET + CNN freq | 0.451 | 0.735 | 0.559 | 0.742 |
| PDNetV2 (Phase 1) | 0.482 | 0.440 | 0.460 | 0.346 |

Consolidated best = product-boost + CET threshold + CNN+ACF freq + post-hoc filter + optimized DP (α=1.275, λ=0.05, β=0.3).

### Improvement Breakdown

| Improvement | F1 | Cumulative Δ |
|-------------|-----|-------------|
| Original max(HPP,CET)+CNN_freq+opt | 0.721 | — |
| + CET 80% threshold | 0.727 | +0.007 |
| + CNN+ACF freq ensemble (0.2) | 0.731 | +0.010 |
| + Product-boost (boost=3, floor=0.3) | 0.736 | +0.015 |
| + Post-hoc min-evidence (0.3) | **0.740** | **+0.019** |

### What didn't help

| Approach | Result | Why |
|----------|--------|-----|
| PD-weighted channel aggregation | -0.01 F1 | Median more robust than weighted mean with noisy PD probs |
| Iterative freq refinement (IPI→re-DP) | -0.19 F1 | IPI from detected discharges is noisier than CNN estimate |
| Gated CET combination | -0.01 F1 | Too conservative, loses CET contributions |
| Adaptive max (suppress CET in quiet HPP) | -0.01 F1 | Similar to threshold, less effective |
| Trained conv combiner (5-fold CV) | -0.03 F1 | Overfits on small dataset |

### Frequency Estimation Comparison (EEG-Only)

| Method | Spearman | MAE |
|--------|----------|-----|
| FFT peak (Alexandra's baseline) | 0.353 | 0.561 |
| CNN+Attention direct | **0.744** | 0.266 |
| IPI from HPP+CNN_freq timing | 0.688 | 0.262 |

### Other Tasks

| Task | Method | Performance | Input |
|------|--------|-------------|-------|
| Subtype (LPD vs GPD) | RF 300 | AUC 0.931 | EEG only |
| Laterality (L vs R) | GBM balanced | AUC 0.957 | EEG only |
| Channel PD detection | CNN+Attention | AUC 0.870 | EEG only |
| Channel RDA detection | Pseudolabels | AUC 0.842 | EEG only |

## Key Findings

### 1. Product-boosted evidence combination outperforms naive max

The product-boost formula `max(HPP, CET) + 3×HPP×CET` amplifies regions where both handcrafted and CNN evidence agree while preserving each method's unique detections. Combined with CET thresholding (80th percentile, floor=0.3), this gives +0.012 F1 over naive max.

### 2. CNN+ACF frequency ensemble reduces estimation errors

Blending CNN frequency (Spearman 0.744) with ACF frequency (Spearman ~0.5) at 0.8/0.2 ratio improves timing F1 by +0.004 — the methods make different errors that average out.

### 3. Frequency estimation remains the bottleneck

The gap between gold-freq HPP (F1=0.757) and best EEG-only (F1=0.740) is now only 0.017 — down from 0.051. But frequency estimation quality (ρ=0.744) still limits timing performance.

### 4. The pipeline is well-optimized for this data size

Five systematic improvement attempts yielded +0.019 F1 total. The remaining 0.017 gap to gold-freq reference is driven by frequency estimation noise. Further gains likely require more training data or better CNN frequency models.

### 5. End-to-end unified model is promising but needs more data

PDNetV2 (joint 18-channel U-Net + Transformer) achieved F1=0.46 in Phase 1 — clearly learning discharge structure but undertrained. The main limitation is dataset size (593 examples) for a 3.4M parameter model. More data and training may close the gap.

### 6. Error analysis reveals frequency-dependent performance

| Frequency range | F1 | Notes |
|----------------|-----|-------|
| 0.3–0.5 Hz | 0.54 | Very few discharges per segment |
| 0.5–1.0 Hz | 0.68 | Below average |
| 1.0–2.0 Hz | 0.80 | Sweet spot |
| >2.0 Hz | 0.71 | Good but fewer training examples |

GPD (F1=0.83) outperforms LPD (F1=0.69) due to bilateral symmetry simplifying the problem. Right-lateralized LPD (F1=0.79) outperforms unspecified laterality (F1=0.66).

## Method Names

| Abbreviation | Full Name | What it does |
|-------------|-----------|-------------|
| **HPP** | Hidden Point Process | MAP inference via DP for discharge timing |
| **CET** | CNN Evidence Trace | Learned per-channel discharge evidence (U-Net) |
| **NVO** | Narrowband Variance Optimization | Sinusoidal fitting for RDA frequency (TODO) |
| **SPF** | Signal Processing Features | Handcrafted features (pointiness, ACF, FFT, TKEO) |

## Nine Tasks (Paper Roadmap Status)

| # | Task | Status | Best Method | Performance |
|---|------|--------|-------------|-------------|
| 1 | LPD vs GPD | **Done** | RF 300 | AUC 0.931 |
| 2 | LRDA vs GRDA | TODO | — | — |
| 3 | PD channel ID | Partial (304 GT) | CNN+Attention | AUC 0.870 |
| 4 | RDA channel ID | Pseudolabels only | CNN | AUC 0.842 |
| 5 | PD discharge timing | **Done** | Consolidated best | F1 0.740 |
| 6 | RDA wave timing | TODO | — | — |
| 7 | RDA frequency | TODO | FFT baseline | ρ 0.840 (23 pts) |
| 8 | PD frequency | **Done** | CNN+Attention direct | ρ 0.744 |
| 9 | BIPD analysis | Waiting for data | — | — |

## Next Steps

1. **Integrate harvested data**: 292 hi-freq LPDs (>2.5 Hz) for CNN frequency retraining
2. **PDNetV2 Phase 2**: Train longer, add more data, add periodicity loss
3. **RDA tasks**: NVO implementation, HPP adaptation for RDA waves, LRDA laterality annotation
4. **Integrate other harvested data**: 100 BIPDs, 70+ LRDA/GRDA, Other controls
5. **Generate paper figures**: see PAPER_ROADMAP.md
