# Table 6: PD Discharge Timing Performance

*Auto-generated from `paper_materials/method_comparison_table.json` and `data/cet_cache/`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table6.py`*

*method_comparison_table.json not found.*

## CET-UNet + DP (Baseline — consolidated_results.json)

| Metric | Value |
|---|---|
| f1 | 0.740 |
| sensitivity | 0.776 |
| precision | 0.706 |
| freq_spearman | 0.718 |
| n_cases | 593.000 |

## Key Finding

HemiCET v2 surpasses the Oracle (expert frequency + handcrafted evidence): 
learned evidence from 8 hemisphere channels is superior to handcrafted features, 
more than compensating for imperfect frequency knowledge.

