# Table 5: Frequency Estimation Performance

*Auto-generated from `segment_labels.csv` + `annotations.csv` with quality filtering.*
*Quality filter: MW reviewed OR LB+PH+SZ consensus OR IIIC ≥10 votes with ≥80% agreement.*
*Expert frequency = mean across raters. Same logic as generate_fig6.py.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table5.py`*

## Per-Subtype Performance (Quality-Filtered)

| Subtype | N | PDChar/W05 ρ | MAE (Hz) | N (Tautan) | Tautan ρ | Tautan MAE |
|---|---:|---|---|---:|---|---|
| LPD | 0 | — | — | 0 | — | — |
| GPD | 0 | — | — | 0 | — | — |
| LRDA | 0 | — | — | 0 | — | — |
| GRDA | 0 | — | — | 0 | — | — |

## Per-Subtype Performance (All Segments, Unfiltered)

| Subtype | N | PDChar/W05 ρ | MAE (Hz) |
|---|---:|---|---|
| LPD | 0 | — | — |
| GPD | 0 | — | — |
| LRDA | 0 | — | — |
| GRDA | 0 | — | — |

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

