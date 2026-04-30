# LRDA laterality disagreements: algorithm vs consensus

Found **6** segments where the V12/V1 algorithm laterality call disagrees with the >=2-of-3 majority-accept consensus laterality (consensus dataset = 155 segments).

Algorithm rule: `pass2_env_log_ratio > 0` means left dominant. A segment ends up wrong when several discriminators disagree, when the estimated frequency is wrong (so the pass-2 narrowband filter is centered on the wrong rhythm), or when laterality is genuinely ambiguous.

Columns:
- `pass1_var`: log(L/R) of pass-1 broadband variance (sign agrees with consensus if positive=left). `pass2_env`: dominant W05 discriminator. `nb_var`: pass-2 narrowband variance ratio. `top3_var`: top-3-channel variance ratio. `peak_prom`: spectral peak prominence ratio. `max_ch`: log of max-single-channel variance ratio.
- `agree_p1p2`: 1 if pass-1 and pass-2 picked the same side. `agree_top3`: 1 if top-3 and uniform-mean agree. `if_cv`: Hilbert IF coefficient of variation (estimator confidence; high = unreliable). `art_l`/`art_r`: low-freq drift artifact score per side.

| # | mat_file (short) | freq | consensus | algo | MW | SZ | TZ | pass1_var | pass2_env | nb_var | top3_var | peak_prom | max_ch | agree_p1p2 | agree_top3 | if_cv | art_l | art_r |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `S0002114848389_20180605000928` | 0.50 | **left** | right | left | - | left | -3.75 | -1.82 | -3.84 | -3.89 | +3.76 | -4.28 | 1 | 1 | 0.81 | 0.11 | 0.13 |
| 2 | `S0002111416588_20181113020323` | 0.63 | **right** | left | right | right | right | +0.06 | +0.74 | +1.73 | +0.34 | -4.38 | +2.30 | 1 | 1 | 0.62 | 0.34 | 0.05 |
| 3 | `S0002116248487_20200212102023` | 1.16 | **left** | right | left | - | left | -7.74 | -2.69 | -7.53 | -8.07 | +1.53 | -7.67 | 1 | 1 | 0.60 | 0.27 | 0.15 |
| 4 | `S0001121878670_20190522133939` | 1.18 | **left** | right | left | left | left | -1.36 | -0.54 | -1.95 | -1.46 | -0.95 | -2.19 | 1 | 1 | 0.47 | 0.30 | 0.35 |
| 5 | `S0001118794756_20190913151358` | 1.56 | **right** | left | right | right | right | +2.67 | +0.58 | +2.16 | +3.11 | -0.02 | +2.82 | 1 | 1 | 0.45 | 0.24 | 0.09 |
| 6 | `S0002111434248_20170505124308` | 1.67 | **left** | right | left | - | left | -0.49 | -0.07 | +0.02 | -0.50 | -0.19 | -0.19 | 1 | 1 | 0.31 | 0.11 | 0.24 |

Sign convention for the log-ratio features: positive = left dominant.

## Per-case interpretation prompts
For each row above, judge:

1. **Are the discriminators unanimous in being wrong, or split?** If split, the rule could be made more robust by combining (e.g., majority vote across pass1/pass2/peak_prom/max_ch) instead of relying solely on `pass2_env_log_ratio`.
2. **Is the est_freq correct?** If the pass-2 narrowband filter is centered on a wrong frequency, the envelope ratio measures the wrong rhythm. Check `est_freq` against the visual rhythm in the EEG figure.
3. **Is laterality genuinely binary?** If the pattern is bilateral with subtle asymmetry, the binary L/R call is ill-defined and the algorithm gets penalized by the binary metric.
4. **Is one rater the outlier?** If two raters and the algo agree, the consensus is fragile; a 4-way consensus would tighten the ground truth.
