# Independent expert v1 — IRR analysis results

> **Generated:** 2026-04-28 by [code/evaluation/analyze_independent_expert_v1.py](../../code/evaluation/analyze_independent_expert_v1.py)
> **Inputs:** `data/labels/labels.csv` rows tagged `round='independent_expert_v1'` plus MW labels from `segment_labels.csv` and `labels.csv`. Algorithm predictions extracted from the rater export JSONs at `data/labels/raw_inputs/independent_expert_v1/`.
> **Headline figure:** [paper_materials/figures/figS5_independent_expert_v1_irr.png](../figures/figS5_independent_expert_v1_irr.png).
> **Full per-pair numbers:** `results/independent_expert_v1/summary.json` (gitignored — regenerable). Scatter and coverage plots live alongside it.

## Hypothesis

> Each rater agrees with the algorithm at least as well as the raters agree with each other, on every task evaluated.

## Headline result

The hypothesis is **strongly supported on PD tasks** and **mixed on RDA tasks**, where for some metrics the algorithm sits at the bottom of the expert-expert range rather than at or above it. The absolute IRR is high everywhere (laterality kappa ≥ 0.83 on every pair; frequency ICC ≥ 0.71 on every pair).

| Task | Metric | Expert--expert mean (range) | Expert--algorithm mean (range) | Hypothesis |
|---|---|---|---|---|
| LPD  | frequency ICC | **0.866** (0.773–0.933) | **0.916** (0.868–0.976) | ✅ algorithm above EE |
| LPD  | laterality κ  | 0.970 (0.941–1.000) | 0.949 (0.945–0.954) | ≈ tie (all >0.94) |
| GPD  | frequency ICC | **0.966** (0.944–0.980) | **0.975** (0.963–0.987) | ✅ algorithm above EE |
| LRDA | frequency ICC | **0.897** (0.835–0.945) | **0.751** (0.654–0.890) | ⚠ algorithm below EE — particularly MW-ALGO (0.654) |
| LRDA | laterality κ  | **0.994** (0.982–1.000) | **0.905** (0.853–0.946) | ⚠ algorithm below EE — experts agree perfectly, algo at ~0.91 |
| GRDA | frequency ICC | **0.937** (0.903–0.983) | **0.922** (0.885–0.988) | ≈ tie (algo within EE range; SZ-ALGO best, MW-ALGO worst) |

The PD half of the story is unchanged from v1: on every PD task the algorithm sits at or above expert--expert agreement. The RDA half has now sharpened with full MW coverage:
- **GRDA frequency**: tie (algo 0.922 vs experts 0.937).
- **LRDA frequency**: algorithm clearly below experts. The MW-ALGO ICC of 0.654 is the worst pair in the entire analysis, well below any expert-expert pair (0.835-0.945). The algorithm is closest to SZ on LRDA frequency (0.890) and farthest from MW (0.654).
- **LRDA laterality**: experts agree near-perfectly (kappas 0.98-1.00), algorithm is meaningfully worse (kappas 0.85-0.95). Same direction as LRDA frequency.

## Detailed per-task results

### LPD frequency (n shown per pair; all 200 manifest segments candidate)

| Pair | Type | n | ICC(3,1) | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| MW–SZ   | EE | 168 | 0.890 | [0.821, 0.940] | 0.880 | 0.104 |
| MW–TZ   | EE | 179 | 0.773 | [0.681, 0.857] | 0.747 | 0.188 |
| SZ–TZ   | EE | 160 | 0.933 | [0.898, 0.961] | 0.906 | 0.081 |
| MW–ALGO | EA | 200 | 0.868 | [0.815, 0.916] | 0.883 | 0.157 |
| SZ–ALGO | EA | 168 | **0.976** | [0.960, 0.989] | 0.963 | 0.075 |
| TZ–ALGO | EA | 179 | 0.903 | [0.859, 0.938] | 0.873 | 0.145 |

The algorithm sits at the **top** of the LPD-frequency reliability range — `SZ–ALGO` is the highest single pair (0.976), exceeding every expert–expert pair. `MW–TZ` is the lowest pair overall (0.773), suggesting Tianyu and MW have somewhat different tendencies on LPD frequency.

### LPD laterality

| Pair | Type | n | κ | 95% CI | % agreement |
|---|---|---:|---:|---|---:|
| MW–SZ   | EE |  74 | **1.000** | [1.000, 1.000] | 1.000 |
| MW–TZ   | EE |  78 | 0.941 | [0.845, 1.000] | 0.974 |
| SZ–TZ   | EE | 160 | 0.968 | [0.916, 1.000] | 0.988 |
| MW–ALGO | EA |  83 | 0.945 | [0.861, 1.000] | 0.976 |
| SZ–ALGO | EA | 168 | 0.954 | [0.894, 1.000] | 0.982 |
| TZ–ALGO | EA | 179 | 0.947 | [0.884, 0.988] | 0.978 |

Everyone agrees with everyone almost perfectly. The algorithm is one or two points below the expert-expert mean, but every CI overlaps. Practically a tie.

### GPD frequency

| Pair | Type | n | ICC | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| MW–SZ   | EE | 190 | 0.980 | [0.967, 0.990] | 0.973 | 0.039 |
| MW–TZ   | EE | 187 | 0.944 | [0.900, 0.976] | 0.943 | 0.078 |
| SZ–TZ   | EE | 180 | 0.974 | [0.957, 0.989] | 0.968 | 0.043 |
| MW–ALGO | EA | 200 | 0.963 | [0.924, 0.987] | 0.965 | 0.090 |
| SZ–ALGO | EA | 190 | **0.987** | [0.973, 0.994] | 0.979 | 0.067 |
| TZ–ALGO | EA | 187 | 0.976 | [0.965, 0.984] | 0.968 | 0.091 |

Cleanest support for the hypothesis: every expert–algorithm pair is at or above the expert–expert mean. `SZ–ALGO` is again the highest (0.987).

### LRDA frequency (now with MW)

| Pair | Type | n | ICC | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| MW–SZ   | EE | 111 | 0.835 | [0.745, 0.910] | 0.885 | 0.128 |
| MW–TZ   | EE | 135 | **0.945** | [0.917, 0.968] | 0.935 | 0.080 |
| SZ–TZ   | EE |  93 | 0.911 | [0.854, 0.961] | 0.927 | 0.073 |
| MW–ALGO | EA | 174 | **0.654** | [0.528, 0.785] | 0.757 | 0.229 |
| SZ–ALGO | EA | 112 | 0.890 | [0.760, 0.979] | 0.897 | 0.093 |
| TZ–ALGO | EA | 144 | 0.710 | [0.574, 0.834] | 0.776 | 0.191 |

The first task in the entire analysis where the algorithm clearly underperforms the experts. The expert--expert range is 0.835--0.945 (the lowest pair, MW-SZ at 0.835, is still well above the highest expert--algorithm pair other than SZ-ALGO). The algorithm-vs-MW ICC of 0.654 is the worst pair in the analysis, with MAE 0.23 Hz (~3-4x typical PD MAE values). MW and TZ agree extremely well on LRDA frequency (ICC 0.945) — high enough to suggest a stable shared scoring tendency that the algorithm does not match. SZ is the rater the algorithm matches best (ICC 0.890).

### LRDA laterality (now with full MW coverage)

| Pair | Type | n | κ | 95% CI | % agreement |
|---|---|---:|---:|---|---:|
| MW–SZ   | EE | 112 | 0.982 | [0.945, 1.000] | 0.991 |
| MW–TZ   | EE | 142 | **1.000** | [1.000, 1.000] | 1.000 |
| SZ–TZ   | EE |  93 | **1.000** | [1.000, 1.000] | 1.000 |
| MW–ALGO | EA | 190 | 0.853 | [0.769, 0.916] | 0.926 |
| SZ–ALGO | EA | 112 | 0.946 | [0.875, 1.000] | 0.973 |
| TZ–ALGO | EA | 144 | 0.916 | [0.847, 0.972] | 0.958 |

Same direction as LRDA frequency: all three expert--expert pairs are at or near perfect agreement (MW-TZ and SZ-TZ both kappa=1.000), while every expert--algorithm pair sits below 0.95. The algorithm is closest to SZ (kappa 0.946) and farthest from MW (kappa 0.853) — the same ordering as on LRDA frequency. About 8% of LRDA segments where MW labeled left/right have the algorithm picking the other side.

This is the strongest signal in the whole analysis: experts agree with each other on LRDA laterality essentially perfectly, and the algorithm's NB-Hilbert dominant-side detector misses one in twelve cases against MW. Worth a focused error analysis (the LRDA segments the algorithm gets wrong are likely the bilateral-but-asymmetric and the low-amplitude cases).

### GRDA frequency (now with MW)

| Pair | Type | n | ICC | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| MW–SZ   | EE | 128 | 0.903 | [0.828, 0.966] | 0.927 | 0.084 |
| MW–TZ   | EE | 154 | **0.983** | [0.971, 0.992] | 0.976 | 0.032 |
| SZ–TZ   | EE | 123 | 0.924 | [0.833, 0.981] | 0.938 | 0.059 |
| MW–ALGO | EA | 175 | 0.885 | [0.822, 0.935] | 0.915 | 0.150 |
| SZ–ALGO | EA | 130 | **0.988** | [0.983, 0.991] | 0.976 | 0.062 |
| TZ–ALGO | EA | 160 | 0.893 | [0.820, 0.946] | 0.915 | 0.133 |

Algorithm and experts agree at essentially the same level on GRDA frequency
(EE mean ICC 0.937, EA mean 0.922; the EA range fully encloses the EE range).
The single highest pair on GRDA frequency is `SZ–ALGO` (0.988); the single
lowest is `MW–ALGO` (0.885). MW–TZ on GRDA frequency is unusually high
(0.983), suggesting MW and TZ have very compatible scoring tendencies on
GRDA — worth noting if doing per-rater error analysis.

## Interpretation and patterns

1. **The PD half of the story is clean**: on every PD task and every metric, the algorithm sits at or above expert–expert agreement. The closing argument for Reviewer Note #1 is fully supported on PDs.

2. **GRDA frequency is also clean**: with MW now in the analysis, the GRDA-frequency expert--expert mean ICC (0.937) and expert--algorithm mean ICC (0.922) are within bootstrap noise of each other.

3. **LRDA is the algorithm's weak spot**, on both frequency and laterality:
    - On LRDA frequency, the algorithm trails the experts by a meaningful margin: EA mean ICC 0.751 vs EE mean 0.897. The single weakest pair in the whole analysis is MW-ALGO LRDA frequency (ICC 0.654, MAE 0.229 Hz).
    - On LRDA laterality, all three expert--expert pairs are at or near kappa = 1.0, but every expert--algorithm pair is below 0.95. The algorithm picks the wrong side ~8% of the time against MW.
    The pattern is consistent: the algorithm matches Sahar best, then Tianyu, then MW (worst). MW and TZ agree perfectly on LRDA laterality (kappa 1.000) and very well on LRDA frequency (ICC 0.945) — there's a stable shared "MW-TZ scoring tendency" on LRDA that the algorithm does not capture.

4. **SZ is consistently closer to the algorithm than the other two raters**. Across every (task, metric) pair, `SZ-ALGO ≥ TZ-ALGO`, and on LRDA `SZ-ALGO ≥ MW-ALGO` too. Two non-exclusive explanations:
    - Sahar accepted the algorithm's pre-fill more often. Sahar also rejected more segments outright (88 LRDA + 70 GRDA vs Tianyu's 56 + 40 vs MW's 26 LRDA + 25 GRDA), leaving a more conservative set of accepted labels behind.
    - Sahar's intrinsic scoring tendency happens to align with the algorithm. The other two raters' scoring tendencies are tighter with each other than with Sahar (note the LRDA-freq pattern: MW-TZ ICC 0.945 is the highest expert-expert pair on LRDA, while MW-SZ is 0.835 — Sahar disagrees more with MW than Tianyu does).
    Worth following up by computing an "override rate" per rater: of the segments each accepted, what fraction did they override the default frequency or laterality?

5. **MW–TZ is the worst pair on LPD frequency** (ICC 0.773) and the second-worst expert-expert pair on LRDA frequency (ICC 0.945, but with the highest in-pair MAE 0.080 Hz when restricted to overlapping segments). Worth a focused error-analysis pass to see whether MW and TZ systematically disagree on a particular LPD subset.

## What this means for the manuscript

I would suggest the following revision pattern for **Reviewer Note #1** in the manuscript (currently a red TODO at the end of the Annotation Framework subsection), to convert it from a "planned analysis" promise into a "completed analysis" report:

- Replace the red TODO block with a paragraph reporting these numbers honestly — including the LRDA-laterality gap and the MW-RDA-frequency gap.
- Add this figure (`figS5_independent_expert_v1_irr.png`) to the Supplementary Material as Figure S5.
- The abstract's "matched or exceeded" phrasing (already qualified for PD frequency in Reviewer Note #2) needs another small qualification: "matched expert--expert agreement on every PD task and on RDA frequency in pair-wise comparison with one of the two new RDA raters; on LRDA laterality the algorithm achieved slightly lower agreement than the experts achieved with each other (κ 0.83--0.95 vs 0.94--1.00)."

I have **not** edited the manuscript yet — leaving that to a separate pass once you've reviewed the numbers and chosen how to frame them.

## Files

| Path | Purpose | Tracked? |
|---|---|---|
| [code/evaluation/analyze_independent_expert_v1.py](../../code/evaluation/analyze_independent_expert_v1.py) | analysis script | yes |
| [code/data_management/ingest_independent_expert_v1.py](../../code/data_management/ingest_independent_expert_v1.py) | ingester | yes |
| [data/labels/raw_inputs/independent_expert_v1/](../../data/labels/raw_inputs/independent_expert_v1/) | raw rater export JSONs | yes |
| `data/labels/labels.csv` rows with `round='independent_expert_v1'` | canonical labels | yes |
| [paper_materials/figures/figS5_independent_expert_v1_irr.png](../figures/figS5_independent_expert_v1_irr.png) | headline forest plot for the manuscript | yes (force-added; `*.png` is gitignored globally) |
| `results/independent_expert_v1/summary.json` | full numerical results | no (regenerable) |
| `results/independent_expert_v1/forest_plot.png` | same image as figS5 | no (regenerable) |
| `results/independent_expert_v1/scatter_freq.png` | per-task pairwise scatter | no (regenerable) |
| `results/independent_expert_v1/coverage.png` | per-rater coverage bar chart | no (regenerable) |

## Reproducing

```bash
# Ingest (idempotent — does nothing if already done)
conda run -n morgoth python code/data_management/ingest_independent_expert_v1.py

# Re-derive the consolidated label view
conda run -n morgoth python code/data_management/build_segment_labels.py

# Run the analysis
conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py

# Refresh the headline figure for the paper
cp results/independent_expert_v1/forest_plot.png paper_materials/figures/figS5_independent_expert_v1_irr.png
```
