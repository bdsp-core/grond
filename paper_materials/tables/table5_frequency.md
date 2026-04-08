# Table 5: Frequency Estimation Performance

*Auto-generated from `segment_labels.csv` + `annotations.csv` with quality filtering.*
*Quality filter: MW reviewed OR LB+PH+SZ consensus OR IIIC ≥10 votes with ≥80% agreement.*
*Expert frequency = mean across raters. Same logic as generate_fig6.py.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table5.py`*

## Per-Subtype Performance (Quality-Filtered)

| Subtype | N | PDChar/W05 ρ | MAE (Hz) | N (Tautan) | Tautan ρ | Tautan MAE |
|---|---:|---|---|---:|---|---|
| LPD | 1,226 | **0.786** | 0.265 | 1,212 | 0.184 | 0.581 |
| GPD | 1,089 | **0.846** | 0.172 | 1,061 | 0.248 | 0.469 |
| LRDA | 640 | **0.674** | 0.233 | 486 | 0.135 | 0.573 |
| GRDA | 1,310 | **0.712** | 0.215 | 971 | 0.218 | 0.546 |

## Per-Subtype Performance (All Segments, Unfiltered)

| Subtype | N | PDChar/W05 ρ | MAE (Hz) |
|---|---:|---|---|
| LPD | 1,496 | 0.796 | 0.240 |
| GPD | 1,539 | 0.879 | 0.140 |
| LRDA | 654 | 0.668 | 0.239 |
| GRDA | 1,380 | 0.711 | 0.218 |

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

