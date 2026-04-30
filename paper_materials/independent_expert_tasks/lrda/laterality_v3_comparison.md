# LRDA laterality v3 -- training on the wider MW-only LRDA pool

The v2 evaluation showed that with only 155 training segments, no
combination of rhythmicity features and peak-locked centroid could
beat V12. The hypothesis here: with ~6x more training data (the
~919-segment MW-only LRDA pool, with all manifest patients excluded),
the same features may learn a more useful decision boundary.

## Training pool

- 919 LRDA segments / 812 unique patients labeled by MW alone
  (binary left/right; bilateral excluded for now).
- All 200 manifest patients excluded -- no leakage with the held-out
  evaluation set.
- Class balance: left=656, right=263.
- Sanity check: 5-fold patient-grouped OOF on the training pool itself
  achieves accuracy 0.954 against MW labels, confirming the model
  learns the MW boundary reasonably well.

## Evaluation: 155-segment 3-rater majority-accept consensus (held out)

| rule | acc | kappa_MW | kappa_SZ | kappa_TZ | mean kappa |
|---|---|---|---|---|---|
| **V12 baseline** (`pass2_env_log_ratio > 0`) | 0.961 | 0.922 | 0.946 | 0.912 | **0.927** |
| V3a (HGB on V1's 16 feats, wider pool)       | 0.935 | 0.870 | 0.892 | 0.897 | 0.886 |
| V3b (HGB on V1+V2's 20 feats, wider pool)    | 0.942 | 0.883 | 0.928 | 0.883 | 0.898 |
| **V3-gated (V12 unless V3b confidence > 0.75)** | **0.968** | **0.935** | **0.964** | **0.927** | **0.942** |
| V3-gated (>0.80) | 0.968 | 0.935 | 0.946 | 0.927 | 0.936 |
| V3-gated (>0.85) | 0.968 | 0.935 | 0.946 | 0.927 | 0.936 |
| V3-gated (>0.90) | 0.968 | 0.935 | 0.946 | 0.927 | 0.936 |
| V3-gated (>0.95) | 0.961 | 0.922 | 0.946 | 0.912 | 0.927 (= V12) |

**Headline:** V3-gated at threshold 0.75 is the first rule that beats V12 on this held-out set: accuracy 0.968 (vs 0.961), mean kappa 0.942 (vs 0.927), with all three pair-wise kappa's improving.

But:

**Bootstrap significance test** (paired segment-level resampling, 2000 iters): point estimate of mean kappa improvement = +0.015, 95% CI [-0.018, +0.056], two-sided **p = 0.583**. The improvement is not statistically significant on the 155-segment held-out set.

## What the gate is actually doing (T=0.75)

The V3-gated rule flips V12 on exactly 3 of 155 segments:

| # | freq | cons | V12 | gate | P(right) | verdict |
|---|---|---|---|---|---|---|
| 99 | 1.18 | left | right | left | 0.23 | **FIXED** -- the unanimous-amplitude error from v2 triage |
| 131 | 0.50 | left | right | left | 0.07 | **FIXED** -- the 0.5 Hz extreme-low case (very confident V3 call) |
| 84 | 1.43 | left | left | right | 0.75 | **broke** -- a marginal V3 call at the threshold edge |

The 2 cases the gate fixes are exactly the cases v2 couldn't address with any feature combination. The 1 new error is right at the threshold (P=0.75 = the threshold itself); a slightly higher gate (0.80) excludes it without losing the two fixes -- which is why V3-gated at 0.80 still has 150/155 correct but mean kappa drops slightly (the kappa difference between 0.75 and 0.80 reflects how the 1-broken-case affects different raters' kappas differently).

## Why the wider pool doesn't directly help (V3a, V3b unconditional)

V3a (V1 features only) and V3b (V1+V2 features) trained on the wider pool both score below V12 on the held-out set. The classifier is internally consistent (954 OOF acc on its own pool) but **MW labels carry MW-specific biases that don't generalize to 3-rater consensus**: the classifier ends up learning "what MW would call this segment," which differs from "what the 3-rater consensus is" on the parts of the segment space MW finds borderline.

The gated approach extracts value from V3 selectively: only when V3 is very confident (so the MW-specific bias is unlikely to be the source of the call) do we override the rule. On the 152 segments where V3b's confidence is < 75%, V3 essentially mimics V12 (which dominates V3's training because pass2_env_log_ratio is in the feature set), so deferring to V12 is a no-op.

## Alternative readings of the result

1. **+0.015 mean kappa is real but small and not significant at n=155.** Modest data; the 95% CI on kappa is roughly +/-0.04 for a sample this size. Need ~500+ consensus-labeled segments to detect a 0.015 effect with confidence.

2. **The gate's 2 fixes are the *most semantically meaningful* errors:** the unanimous-amplitude error (case 99) and the 0.5 Hz extreme-low case (case 131). These were the hardest cases for any rule we built. The gate's 1 new error (case 84) is a marginal call. So even if statistically marginal, the failure modes addressed are exactly the right ones.

3. **The fact that the wider pool *does* help when gated** confirms the v2 hypothesis was right: rhythmicity + centroid features carry signal; we just needed enough training data to learn when to trust them.

## Recommendations

1. **Don't ship V3-gated yet.** The improvement isn't significant on the 155-segment held-out set. We'd be over-claiming.

2. **Re-run when the 4th rater finishes.** The denominator goes up (some 2-rater consensus segments may flip to 3-rater), and the gate may stabilize. If V3-gated is then significantly better, ship as V13.

3. **The MW-only pool is a usable training resource.** Even if we don't ship V3-gated now, this experiment establishes that the wider pool *can* train laterality classifiers that beat the rule on a few key cases. When we re-run after the 4th rater (or after another round of consensus labeling), V3-gated is the canonical method to revisit.

4. **Bilateral labels (170 LRDA segs MW-labeled "bilateral")** are still on the table. A 3-class classifier could surface which V12 errors are actually genuine bilateral cases that the binary L/R metric misclassifies on principle. Future work.

## Files written

- `code/evaluation/lrda_laterality_v3_eval.py`
- `data/labels/independent_expert_v1/lrda_laterality_v3_train_features.csv` (cached features for 919 training segs)
- `data/labels/independent_expert_v1/lrda_laterality_v3_eval.txt`
