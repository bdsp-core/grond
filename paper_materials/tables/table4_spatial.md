# Table 4: Spatial Inter-Rater Agreement (Jaccard Matrix)

## PD Spatial Localization (N=220, threshold=0.38)

|       | LB    | PH    | SZ    | PDChar |
|-------|------:|------:|------:|-------:|
| LB    | 1.000 | 0.762 | 0.773 | 0.692 |
| PH    | 0.762 | 1.000 | 0.716 | 0.842 |
| SZ    | 0.773 | 0.716 | 1.000 | 0.664 |
| PDChar| 0.692 | 0.842 | 0.664 | 1.000 |

| Comparison | Mean Jaccard |
|---|---|
| Expert-expert (LB, PH, SZ) | 0.751 ± 0.025 |
| Model-expert (PDChar vs LB, PH, SZ) | 0.731 ± 0.076 |
| **Model as % of expert** | **97.3%** |
| Best pair: PDChar-PH | 0.842 (exceeds all expert-expert pairs) |

## RDA Spatial Extent (N=211)

| Method | MAE | Pearson r | ICC (as 4th rater) |
|---|---|---|---|
| PLV continuous | **0.178** | 0.479 | 0.349 |
| VE+PLV blend | 0.219 | **0.506** | **0.380** |
| PLV threshold=0.62 | 0.215 | 0.485 | 0.371 |
| Expert-expert ICC | — | — | 0.373 |

## ICC Comparison with Tautan et al. 2025

| Measure | Expert-Expert | PDChar/W05 | Tautan et al. |
|---|---|---|---|
| PD Spatial ICC | 0.455 | 0.852* | 0.464 |
| RDA Spatial ICC | 0.373 | 0.371 | 0.215 |

*After threshold optimization (0.62) and SZ spatial_extent=0 cleanup.
