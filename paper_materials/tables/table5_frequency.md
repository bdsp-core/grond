# Table 5: Frequency Estimation Performance

*Auto-generated from `data/labels/segment_labels.csv`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table5.py`*

## Per-Subtype Performance (Quality-Filtered)

| Subtype | N (PDChar) | PDChar/W05 ρ | PDChar MAE (Hz) | N (Tautan) | Tautan ρ | Tautan MAE (Hz) |
|---|---:|---|---|---:|---|---|
| LPD | 1,499 | **0.733** | 0.274 | 1,482 | 0.142 | 0.600 |
| GPD | 1,539 | **0.640** | 0.603 | 1,509 | 0.362 | 0.845 |
| LRDA | 654 | **0.665** | 0.241 | 500 | 0.145 | 0.571 |
| GRDA | 1,381 | **0.705** | 0.220 | 1,041 | 0.213 | 0.541 |

Quality filter: segments with expert-reviewed frequency (expert_freq_hz) and valid algorithm prediction.

## RDA Frequency — Top Methods (V5 Contest)

| Rank | Method | Lat AUC | Freq ρ |
|---|---|---|---|
| 1 | W07_AutoChannel_FreqAgreement | 0.790 | 0.686 |
| 2 | W03_DomOnly_QualityWeighted | 0.790 | 0.685 |
| 3 | W02_DomOnly_AutoK | 0.790 | 0.683 |
| 4 | V04_PLVSelected | 0.809 | 0.682 |
| 5 | U11_HilbertCV_Top3 | 0.581 | 0.674 |
| 6 | W06_AutoChannel_EnvThreshold | 0.790 | 0.663 |
| 7 | V03_ConsistencySelected | 0.787 | 0.659 |
| 8 | V22_EnvAmp_DomHilbert | 0.790 | 0.650 |
| 9 | V23_CherryPick | 0.790 | 0.650 |
| 10 | W01_DomOnly_StrictHilbert | 0.790 | 0.650 |

46 methods evaluated on LRDA vs GRDA classification + frequency estimation.

