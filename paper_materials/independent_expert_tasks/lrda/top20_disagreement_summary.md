# Top 20 LRDA expert-vs-algorithm disagreements

Sorted by `score = |mean(expert freqs) - algo freq|`. Cases at the top of the table contribute most to the LRDA-frequency expert-vs-algorithm gap on figS5.


## Summary statistics

- Algorithm BELOW expert mean: **20** of 20

- Algorithm ABOVE expert mean: **0** of 20

- Algorithm at ~half expert mean (ratio 1.6-2.4): **7** of 20  -- classic sub-harmonic locking pattern

- Expert-mean frequency distribution: min=1.08, median=2.25, max=3.75 Hz


## Per-case table (sorted by disagreement)

| # | segment_id | MW | SZ | TZ | mean exp | ALGO | score | ratio | lat (MW/SZ/TZ/ALGO) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `sub-I0002150018626_20191223215055` | 3.75 | — | — | **3.75** | **1.46** | **2.29** | 2.58× | R/—/—/R |
| 2 | `sub-I0002150014148_20191128142304` | — | — | 3.00 | **3.00** | **0.99** | **2.01** | 3.03× | —/—/L/L |
| 3 | `sub-S0001113601600_20150813165933` | 2.25 | — | — | **2.25** | **0.50** | **1.75** | 4.48× | R/—/—/R |
| 4 | `sub-S0001121878670_20190522133939` | 2.50 | 2.50 | 2.50 | **2.50** | **1.03** | **1.47** | 2.44× | L/L/L/R |
| 5 | `sub-S0002114848389_20180605000928` | 2.00 | — | 1.75 | **1.88** | **0.43** | **1.45** | 4.39× | L/—/L/R |
| 6 | `sub-I0002150008296_20200528033816` | — | — | 3.00 | **3.00** | **1.73** | **1.27** | 1.73× | R/—/R/R |
| 7 | `sub-S0002121743214_20161001083843` | 2.50 | — | 2.50 | **2.50** | **1.40** | **1.10** | 1.78× | L/—/L/L |
| 8 | `sub-I0002150014437_20170801114038` | 2.50 | — | 2.00 | **2.25** | **1.25** | **1.00** | 1.80× | R/—/R/R |
| 9 | `sub-I0002150006476_20191025074005` | 2.00 | — | — | **2.00** | **1.16** | **0.84** | 1.73× | R/—/—/R |
| 10 | `sub-S0001111970923_20181229105949` | 2.25 | — | — | **2.25** | **1.42** | **0.83** | 1.58× | L/—/—/R |
| 11 | `sub-S0002116502824_20160626105252` | 1.75 | — | — | **1.75** | **0.93** | **0.82** | 1.88× | L/—/—/R |
| 12 | `sub-I0002150016440_20200614003334` | 3.25 | 1.75 | — | **2.50** | **1.70** | **0.80** | 1.47× | L/R/—/R |
| 13 | `sub-S0002114410018_20200430105901` | 1.25 | — | — | **1.25** | **0.62** | **0.63** | 2.02× | R/—/—/L |
| 14 | `sub-S0002111416588_20181113020323` | 1.00 | 1.25 | 1.00 | **1.08** | **0.46** | **0.62** | 2.33× | R/R/R/L |
| 15 | `sub-S0001117792110_20221222091421` | 3.00 | 2.25 | 3.00 | **2.75** | **2.13** | **0.62** | 1.29× | R/R/R/R |
| 16 | `sub-I0002150005742_20210316065303` | 1.75 | — | 1.75 | **1.75** | **1.14** | **0.61** | 1.54× | L/—/L/L |
| 17 | `sub-S0002115111161_20201101205535` | 3.00 | 2.00 | 3.00 | **2.67** | **2.07** | **0.59** | 1.29× | R/R/R/R |
| 18 | `sub-S0001121058250_20130617123309` | 1.50 | — | — | **1.50** | **0.96** | **0.54** | 1.56× | R/—/—/R |
| 19 | `sub-I0002150018266_20191025022947` | 2.25 | — | 1.75 | **2.00** | **1.47** | **0.53** | 1.36× | R/—/R/R |
| 20 | `sub-I0002150018705_20200629212516` | 2.75 | 2.00 | 2.50 | **2.42** | **1.91** | **0.51** | 1.27× | L/L/L/L |

## How to read this

- **score**: the headline metric -- magnitude of disagreement in Hz.

- **ratio**: mean-expert / algorithm. Values near 2.0 indicate the algorithm is reading the sub-harmonic.

- **lat**: laterality letters in order MW / SZ / TZ / ALGO. "L"/"R"/"--".


## To review interactively

Open the focused viewer at `top20_disagreement_viewer.html` (regenerated from this manifest with the standard LRDA viewer) -- arrow keys to navigate, up/down to step through frequency buttons and watch which value the green narrowband overlay locks onto.
