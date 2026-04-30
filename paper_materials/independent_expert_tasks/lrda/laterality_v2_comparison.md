# LRDA laterality: Fix 1 (rhythmicity) and Fix 2 (peak-locked centroid) vs V12 baseline

Evaluated on the 155-segment majority-accept consensus laterality set
(>=2 of 3 raters accepted the segment as a valid LRDA AND >=2 of 3
raters who labeled it agree on left vs right).

## Headline result: V12 baseline wins

| rule | acc | kappa_MW | kappa_SZ | kappa_TZ | mean kappa |
|---|---|---|---|---|---|
| **V12 baseline (`pass2_env_log_ratio > 0`)** | **0.961** | **0.922** | **0.946** | **0.912** | **0.927** |
| Fix1a: `spec_conc_log_ratio` | 0.768 | 0.535 | 0.622 | 0.534 | 0.563 |
| Fix1b: `acf_peak_log_ratio` | 0.697 | 0.410 | 0.602 | 0.420 | 0.477 |
| Fix1c: `if_cv_inv_log_ratio` (per hem) | 0.581 | 0.164 | 0.189 | 0.170 | 0.174 |
| Fix1_sum (sum of all 3 Fix-1 features) | 0.729 | 0.459 | 0.621 | 0.463 | 0.515 |
| Fix2: `peak_topo_log_ratio` (neutral peak detection) | 0.929 | 0.858 | 0.874 | 0.854 | 0.862 |
| Hybrid 3-vote (pass2 + spec_conc + peak_topo) | 0.948 | 0.897 | 0.910 | 0.883 | 0.897 |
| Hybrid 5-vote (pass2 + spec_conc + peak_topo + acf + max_ch) | 0.942 | 0.884 | 0.910 | 0.869 | 0.887 |
| Trained classifier (HGB on 16 V1 + 4 V2 features, 5-fold patient-grouped OOF) | 0.955 | 0.909 | 0.928 | 0.912 | 0.917 |

**No rule beats V12.** The closest competitor is the trained classifier at 0.955 / 0.917.

## Per-rule call on the 6 V12 baseline disagreement cases

| # | freq | cons | V12 | Fix1a | Fix1b | Fix1c | Fix1_sum | Fix2 | Hybrid3 | Hybrid5 | Trained |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1.16 | L | R | **L** | R | R | **L** | R | R | R | R |
| 15 | 0.63 | R | L | L | **R** | **R** | **R** | L | L | L | L |
| 16 | 1.67 | L | R | R | R | **L** | R | R | R | R | R |
| 78 | 1.56 | R | L | **R** | **R** | L | **R** | L | L | L | L |
| 99 | 1.18 | L | R | R | R | R | R | R | R | R | R |
| 131 | 0.50 | L | R | R | R | R | R | R | R | R | **L** |

(Bold = rule got this case right.)

- **Fix 1 features collectively address 4 of the 6 V12 errors** (cases 2, 15, 16, 78), confirming the triage hypothesis that rhythmicity-based signals carry information that pass-2 envelope amplitude misses.
- **Fix 2 (improved with neutral peak detection) addresses 0 of 6 V12 errors**. It mostly mimics V12 because the peak-locked topography ends up dominated by whichever hemisphere has bigger peak-time amplitude — the same physical thing pass-2 envelope already measures.
- **The trained classifier addresses 1 of 6 V12 errors** (case 131, the 0.5 Hz extreme low).
- **Cases 99 and 131 are not addressed by any single hand-tuned rule.** Case 99 is a unanimous-amplitude-error (every feature wrong); case 131 has spectral concentration also fooled by drift at 0.5 Hz.

## Why every fix loses overall

Each individual rhythmicity feature is *noisy* on the 149 segments where V12 is right. Even Fix1a (the cleanest of the three) gets 36 of 149 V12-correct cases wrong. So while Fix 1 catches 4 of 6 V12 errors, it introduces tens of new errors on cases V12 had already nailed. Net κ is much worse.

Voting hybrids and the trained classifier dilute the noise but don't escape it: V12's `pass2_env_log_ratio` is heavily weighted because it has the cleanest signal, and the hybrids end up close to V12 plus a few extra mistakes.

## Why the rhythmicity features look promising on the triage table but fail in aggregate

The triage table showed `peak_prom` (V1's existing spectral peak prominence ratio) pointing the right way on cases 1, 2, 3. But in standalone evaluation `peak_prom_only` was 64% accurate -- it carries the right signal on those specific cases but is noisy elsewhere. The new Fix-1 features are similar: they add a true second look at rhythmicity, but each per-segment estimate is variance-dominated for cases where the rhythm is clear (so the L vs R log-ratio swings around for what should be an easy left or easy right call).

A confidence-gated rule ("trust V12 unless V12-confidence is low, then defer to rhythmicity") cannot work either: the |pass2_env| values for V12's 6 wrong cases (0.07, 0.54, 0.58, 0.74, 1.82, 2.69) are fully nested within the |pass2_env| values for V12's 149 correct cases (median 0.50, range 0.0--6.5).

## What the upper bound looks like

Across Fix 1 + Fix 2 + the trained classifier, **5 of 6 V12 errors are addressed by *some* rule** (cases 2, 15, 16, 78, 131). Only case 99 remains unaddressed by any feature in this experiment. This sets an empirical upper bound: with the 16 V1 + 4 V2 features and the current 155-segment dataset, the best plausible algorithm could hope for 1 unfixable error -- a kappa ceiling around 0.985, vs V12's 0.927 and the EE ceiling of 0.994.

The gap between achievable (≈0.985) and observed (V12's 0.927) is 6 errors. The gap between observed and EE ceiling is also ≈6 errors. So the ceiling is real and worth pursuing -- but not with the feature classes evaluated here. The classifier's failure to break through suggests the dataset is too small to learn the gating decisions needed.

## Recommendations

1. **Keep V12 (current production).** Don't ship any of the alternatives based on this evaluation.
2. **Wait for the 4th rater.** Three of the six V12 errors are 2-rater consensus segments (SZ rejected); a 4-way consensus could shrink the error count without any model change.
3. **The natural next experiments** require fundamentally different feature classes than rhythmicity or peak-locked amplitude:
   - Surface Laplacian narrowband envelope (volume-conduction reduction).
   - Source localization at the rhythm frequency.
   - Transfer-fine-tune the HemiCET-UNet hemispheric head on the 155 LRDA-laterality segments.
   - More training data (the 4th rater plus future rounds).
4. **Reframe the metric.** Cohen's κ on a 0.99 EE ceiling penalizes any binary error heavily; consider a graded laterality measure (continuous laterality index) that recognizes "subtle bilateral" segments where binary is the wrong abstraction.

## Files written

- `code/evaluation/lrda_laterality_v2_eval.py`: featurizer + rule evaluator
- `data/labels/independent_expert_v1/lrda_laterality_v2_features.csv`: 4 new features per segment
- `data/labels/independent_expert_v1/lrda_laterality_v2_eval.txt`: machine-readable evaluation report
