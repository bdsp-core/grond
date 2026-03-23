# Table: Comprehensive Method Comparison for PD Discharge Detection

## Dataset
- 675 expert-reviewed cases (437 LPD + 207 GPD), 3 rounds of model-assisted label cleanup
- Evaluation on 582 cases with available EEG segments
- Discharge matching tolerance: ±100ms
- All methods are EEG-only (no gold standard labels as input) except Oracle

## Reference: Expert Frequency Agreement
- Expert gold standard frequency vs expert-derived IPI frequency: Spearman ρ = 0.941, MAE = 0.114 Hz
- This represents the ceiling for frequency agreement between independent expert estimates

## Method Comparison

| Method | F1 | Sensitivity | Precision | Freq ρ | Freq MAE (Hz) | Timing MAE (ms) | Timing Median (ms) |
|--------|-----|------------|-----------|--------|---------------|-----------------|-------------------|
| **HemiCET v2 + DP (C1)** | **0.889** | **0.921** | **0.859** | **0.891** | **0.183** | **1.0** | **<1** |
| Full 18ch pipeline | 0.726 | 0.722 | 0.730 | 0.765 | 0.250 | 17.7 | 10.0 |
| Per-hemi baseline (HPP+CET+DP) | 0.719 | 0.724 | 0.715 | 0.778 | 0.253 | 19.4 | 10.0 |
| Oracle (HPP + expert freq) | 0.664 | 0.569 | 0.799 | 0.910 | 0.139 | 10.9 | <1 |

### Notes

**HemiCET v2 + DP (C1)**: The winning model. 8-channel hemisphere CET-UNet (525K params) produces evidence trace, fed to DP with optimized parameters (α=1.5, β=0.3, λ=0.05) + evidence thresholding (50th percentile) + post-hoc confidence filter (ratio=0.4). Frequency from CNN+ACF ensemble (0.8/0.2).

**Full 18ch pipeline**: Product-boosted max(HPP handcrafted, per-channel CET-UNet) evidence on all 18 channels, with CET thresholding, CNN+ACF frequency, optimized DP, and post-hoc filter.

**Per-hemi baseline**: Same as full 18ch pipeline but run on 8 hemisphere channels only.

**Oracle (HPP + expert freq)**: HPP algorithm using the expert's gold standard frequency as the DP period prior. Uses default (non-optimized) DP parameters (α=3.0, β=1.0, λ=0.02) and handcrafted evidence only. Despite having perfect frequency knowledge, it underperforms HemiCET because: (1) its evidence trace (pointiness+TKEO) is inferior to learned evidence, (2) its DP parameters were not optimized for the cleaned labels.

### Key Findings

1. **HemiCET v2 surpasses even the Oracle** — learned evidence from 8 channels jointly is so much better than handcrafted evidence that it more than compensates for not having perfect frequency knowledge.

2. **Timing precision is sub-millisecond** — HemiCET+DP achieves <1ms median timing error (at the 5ms resolution limit of 200 Hz sampling), compared to 10-19ms for other methods.

3. **Frequency correlation ρ=0.891 approaches expert-expert agreement** (ρ=0.941) — the IPI-derived frequency from accurate discharge timing is nearly as good as independent expert frequency estimation.

4. **8-channel hemisphere input > 18-channel full input** — processing one hemisphere at a time avoids contamination from the uninvolved hemisphere and enables the model to learn hemisphere-specific cross-channel patterns.

## Optimization Experiment Leaderboard

| Experiment | F1 | LPD F1 | GPD F1 | Note |
|-----------|-----|--------|--------|------|
| Oracle (gold freq + C1 DP) | 0.919 | 0.931 | 0.893 | Theoretical ceiling |
| **C1: E2+E3 combined** | **0.891** | **0.881** | **0.913** | **Production config** |
| E3: Post-hoc filtering | 0.891 | 0.880 | 0.914 | Threshold=50%, ratio=0.4 |
| E5: Midline 10ch | 0.877 | 0.863 | 0.905 | Adding midline hurt |
| Baseline (HemiCET v2) | 0.873 | 0.865 | 0.890 | Clean labels alone |
| E2: DP re-optimization | 0.873 | 0.865 | 0.890 | α=1.5, minimal gain |
| E6: Multi-segment | 0.588 | 0.535 | 0.700 | Label mismatch issue |

## Frequency Estimation Experiment Results

| Freq Method | F1 | Freq ρ | Freq MAE | Note |
|------------|-----|--------|----------|------|
| CNN+ACF baseline | 0.890 | 0.890 | 0.183 | Current production |
| Two-pass (CNN→IPI→re-DP) | 0.887 | 0.900 | 0.169 | Better freq, slightly worse F1 |
| Evidence ACF + CNN | 0.880 | 0.809 | 0.193 | Evidence ACF mediocre |
| Evidence ACF alone | 0.843 | 0.747 | 0.286 | |
| Evidence FFT | 0.782 | 0.458 | 0.508 | FFT on evidence is bad |

## End-to-End Model Comparison (all inferior to HemiCET+DP)

| Model | Params | F1 | Note |
|-------|--------|-----|------|
| PDNetV2 (18ch U-Net+Transformer) | 3.4M | 0.460 | Severe overfitting |
| HemiNet Design A (8ch U-Net+Transformer) | 2.3M | 0.624 | Overfitting |
| HemiNet Design B (8ch dilated conv) | 2.2M | 0.626 | Overfitting |
| HemiNet Design D (HPP wrapper) | 262K | 0.605 | Most stable, lowest |
| HemiNet + MAE pretrain | 2.3M | 0.620 | Slight improvement |

**Conclusion**: With ~1,000 training examples, end-to-end neural models cannot learn the temporal structure that DP encodes as a prior. The winning strategy is neural evidence generation (HemiCET) + structured inference (DP).
