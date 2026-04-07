# Table 4: Spatial Inter-Rater Agreement (Jaccard Matrix)

*Auto-generated from `results/spatial_agreement.json`.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table4.py`*

## PD Spatial Localization (N=—, threshold=0.38)

|       | LB | PH | SZ | Model |
|-------|------:|------:|------:|------:|
| LB    | 1.000 | 0.762 | 0.773 | 0.693 |
| PH    | 0.762 | 1.000 | 0.716 | 0.837 |
| SZ    | 0.773 | 0.716 | 1.000 | 0.662 |
| Model | 0.693 | 0.837 | 0.662 | 1.000 |

| Comparison | Mean Jaccard |
|---|---|
| Expert-expert | 0.751 ± 0.025 |
| Model-expert | 0.731 ± 0.076 |
| **Model as % of expert** | **97.3%** |
| Best pair: PDChar-PH | 0.837 |

## RDA Spatial Extent

| Method | Metric | Notes |
|---|---|---|
| PLV × Amplitude | See Fig S2 for scatter plots | Threshold-based and continuous modes |
| Expert-expert ICC | 0.373 | 3-rater (LB, PH, SZ) |
| RDA-PLV ICC | 0.371 | Matches expert-expert |

*RDA spatial metrics computed from spatial_inference_cache.json. See generate_fig_irr.py for full ICC computation.*

