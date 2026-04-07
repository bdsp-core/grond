# Table 3: Lateralization Performance

## PD Lateralization (ChannelPD-Net)

| Metric | Value | N | Method |
|---|---|---:|---|
| AUC (LPD vs GPD) | 0.931 | 594 | RF 300 trees on ChannelPD-Net features |
| AUC (L vs R hemisphere) | 0.963 | 423 | ChannelPD-Net mean probability comparison |
| Channel PD detection AUC | 0.870 | 815 | Per-channel CNN+Attention |

## RDA Lateralization (LRDA vs GRDA)

| Rank | Method | AUC | Freq ρ | N |
|---|---|---|---|---:|
| 1 | W05_DomOnly_IterRefine | **0.837** | 0.635 | 4,253 |
| 2 | V12_IterativeRefine | 0.825 | 0.595 | 4,253 |
| 3 | V04_PLVSelected | 0.809 | 0.682 | 4,253 |
| 4 | W07_AutoChannel_FreqAgreement | 0.790 | **0.686** | 4,253 |

Dataset: 1,295 LRDA + 2,958 GRDA segments. Leave-one-patient-out CV.

## Impact of Label Quality on RDA Lateralization

| Contest | Labels | AUC | Notes |
|---|---|---|---|
| V1 | Noisy (patient-level aggregation) | 0.58 | Pre-cleanup |
| V4 | Clean (per-segment IIIC votes) | 0.84 | +45% relative improvement |
| V5 | Clean + MW laterality review | 0.837 | 338 non-LRDA excluded |
