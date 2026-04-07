# Table 5: Frequency Estimation Performance

## Per-Subtype Performance (Quality-Filtered)

| Subtype | N | PDChar/W05 ρ | MAE (Hz) | Tautan et al. ρ |
|---|---:|---|---|---|
| LPD | 1,226 | **0.786** | 0.192 | 0.261 |
| GPD | 1,089 | **0.846** | 0.184 | 0.555 |
| LRDA | 640 | **0.674** | 0.265 | 0.086 |
| GRDA | 1,310 | **0.712** | 0.221 | 0.274 |

Quality filter: MW-reviewed OR 3-expert consensus OR IIIC ≥80% agreement.

## ICC Comparison with Expert Inter-Rater Reliability

| Measure | Expert-Expert ICC | PDChar/W05 ICC | Tautan et al. ICC |
|---|---|---|---|
| PD Frequency | 0.662 | 0.572 | 0.451 |
| RDA Frequency | 0.852 | **0.860** | 0.816 |

W05 for RDA frequency **matches expert-expert ICC** (0.860 vs 0.852).

## PD Frequency Method Comparison

| Method | LPD ρ | GPD ρ | Source |
|---|---|---|---|
| CNN+ACF → IPI (PDCharacterizer) | **0.786** | **0.846** | IPI-derived from detected discharges |
| CNN only | 0.663 | — | ChannelPD-Net direct output |
| ACF only | — | — | Autocorrelation of pointiness |
| Tautan et al. 2025 | 0.261 | 0.555 | Published baseline |

## RDA Frequency Method Comparison (V5 Contest)

| Rank | Method | Freq ρ | AUC | Strategy |
|---|---|---|---|---|
| 1 | W07_AutoChannel_FreqAgreement | **0.686** | 0.790 | MAD-based outlier rejection |
| 2 | W03_DomOnly_QualityWeighted | 0.685 | 0.790 | Quality-weighted Hilbert |
| 3 | W02_DomOnly_AutoK | 0.683 | 0.790 | Auto-K channel selection |
| 4 | V04_PLVSelected | 0.682 | 0.809 | PLV-coherent channels |
| 5 | W05_DomOnly_IterRefine | 0.635 | **0.837** | Best lateralization |

Pareto frontier: W05 (best lateralization), V04 (balanced), W07 (best frequency).
