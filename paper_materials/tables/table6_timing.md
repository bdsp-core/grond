# Table 6: PD Discharge Timing Performance

*Auto-generated from `paper_materials/method_comparison_table.json` and `data/cet_cache/`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table6.py`*

## HemiCET v2 + DP — Production Model (C1)

| Metric | Value |
|---|---|
| F1 | **0.889** |
| Sensitivity | 0.921 |
| Precision | 0.859 |
| Freq ρ (IPI-derived) | 0.891 |
| Freq MAE (Hz) | 0.183 |
| Timing MAE (ms) | 1.0 |
| N cases | 582 |

## Method Comparison

| Method | F1 | Sens | Prec | Freq ρ | Timing MAE (ms) |
|---|---|---|---|---|---|
| Oracle (HPP+gold freq) | 0.664 | 0.569 | 0.799 | 0.910 | 10.9 |
| **HemiCET v2 + DP (C1)** | **0.889** | 0.921 | 0.859 | 0.891 | 1.0 |
| Full 18ch pipeline | 0.726 | 0.722 | 0.730 | 0.765 | 17.7 |
| Per-hemi baseline (HPP+CET+DP) | 0.719 | 0.724 | 0.715 | 0.778 | 19.4 |

## Expert Gold Standard

| Metric | Value |
|---|---|
| IPI vs reviewed freq ρ | 0.940 |
| IPI vs reviewed freq MAE (Hz) | 0.114 |
| N segments with gold freq | 662 |

## Key Finding

HemiCET v2 surpasses the Oracle (expert frequency + handcrafted evidence): 
learned evidence from 8 hemisphere channels is superior to handcrafted features, 
more than compensating for imperfect frequency knowledge.

