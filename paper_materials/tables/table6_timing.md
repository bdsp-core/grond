# Table 6: PD Discharge Timing Performance

## HemiCET v2 + DP (Production Configuration C1)

| Metric | Combined | LPD | GPD |
|---|---|---|---|
| F1 | **0.889** | 0.881 | 0.913 |
| Sensitivity | 0.921 | — | — |
| Precision | 0.859 | — | — |
| Freq ρ (IPI-derived) | 0.891 | — | — |
| Freq MAE (Hz) | 0.183 | — | — |
| Timing MAE (ms) | 1.0 | — | — |

## Method Comparison

| Method | F1 | Timing MAE (ms) | Freq ρ | Notes |
|---|---|---|---|---|
| **HemiCET v2 + DP (C1)** | **0.889** | **1.0** | **0.891** | Production: optimized DP + post-hoc filter |
| Full 18ch pipeline | 0.726 | 17.7 | — | All channels, product-boosted evidence |
| Per-hemi baseline (HPP+CET+DP) | 0.719 | 19.4 | — | Per-hemisphere, default params |
| Oracle (HPP + expert freq) | 0.664 | 10.9 | — | Perfect frequency, handcrafted evidence |
| Expert gold standard | — | — | ρ=0.941, MAE=0.114 | IPI vs reviewed frequency |

## Configuration Details (C1 = E2 + E3)

| Component | Setting |
|---|---|
| Evidence | Product-boosted max(HPP, CET) + 3 × HPP × CET |
| Evidence threshold | 50th percentile of non-zero values |
| DP parameters | α=1.5, β=0.3, λ=0.05 |
| Post-hoc filter | Drop peaks with evidence < 0.4 × median |
| Frequency prior | 0.8 × CNN + 0.2 × ACF |

## Key Finding

HemiCET v2 surpasses even the Oracle (expert frequency + handcrafted evidence): learned evidence from 8 hemisphere channels jointly is superior to handcrafted features, more than compensating for imperfect frequency knowledge.
