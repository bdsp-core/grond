# Table 3: Lateralization Performance

*Auto-generated from `predictions.json` + `segment_labels.csv` + contest results.*
*AUC computed on demand from stored per-channel predictions.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table3.py`*

## PD Lateralization (ChannelPD-Net V1)

| Metric | Value | N | Method |
|---|---|---:|---|
| Hemisphere AUC (L vs R) | **0.989** | 1,274 | L vs R hemisphere mean PD probability |
| LPD vs GPD AUC | **0.8752** | 7,037 | RF 300 trees on ChannelPD-Net channel probabilities |
| Timing F1 (production) | 0.889 | 582 | HemiCET v2 + DP |
| Frequency ρ (IPI) | 0.891 | 582 | IPI-derived from detected discharges |

## RDA Lateralization (LRDA vs GRDA) — V5 Contest

Dataset: 1,295 LRDA + 2,958 GRDA = 4,253 segments.

### Top Unified Methods (lateralization + frequency)

| Rank | Method | AUC | Freq ρ |
|---|---|---|---|
| 1 | W05_DomOnly_IterRefine | 0.837 | 0.635 |
| 2 | V12_IterativeRefine | 0.825 | 0.595 |
| 3 | V04_PLVSelected | 0.809 | 0.682 |
| 4 | U10_MultiCh_HilbertFreq | 0.790 | 0.601 |
| 5 | V01_DomHemi_Top3Hilbert | 0.790 | 0.575 |
| 6 | V02_PowerWeightedHilbert | 0.790 | 0.619 |
| 7 | V22_EnvAmp_DomHilbert | 0.790 | 0.650 |
| 8 | V23_CherryPick | 0.790 | 0.650 |
| 9 | V24_SoftChannelWeight | 0.790 | 0.620 |
| 10 | W01_DomOnly_StrictHilbert | 0.790 | 0.650 |

### Top Lateralization-Only Methods

| Rank | Method | AUC |
|---|---|---|
| 1 | V25_FreqBandEnvRatio | 0.853 |
| 2 | L24_EnvelopeAmplitude | 0.826 |
| 3 | L05_RMSAmplitude | 0.797 |
| 4 | W10_DomOnly_EnvPeakFreq | 0.790 |
| 5 | L01_DeltaBandpower | 0.782 |

76 methods evaluated in V5 lateralization contest.

