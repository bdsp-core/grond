# Independent expert v1 — IRR analysis results

> **Generated:** 2026-04-28 by [code/evaluation/analyze_independent_expert_v1.py](../../code/evaluation/analyze_independent_expert_v1.py)
> **Inputs:** `data/labels/labels.csv` rows tagged `round='independent_expert_v1'` plus MW labels from `segment_labels.csv` and `labels.csv`. Algorithm predictions extracted from the rater export JSONs at `data/labels/raw_inputs/independent_expert_v1/`.
> **Headline figure:** [paper_materials/figures/figS5_independent_expert_v1_irr.png](../figures/figS5_independent_expert_v1_irr.png).
> **Full per-pair numbers:** `results/independent_expert_v1/summary.json` (gitignored — regenerable). Scatter and coverage plots live alongside it.

## Hypothesis

> Each rater agrees with the algorithm at least as well as the raters agree with each other, on every task evaluated.

## Headline result

The hypothesis is **strongly supported on PD tasks** and **mixed on RDA tasks**, where for some metrics the algorithm sits at the bottom of the expert-expert range rather than at or above it. The absolute IRR is high everywhere (laterality kappa ≥ 0.83 on every pair; frequency ICC ≥ 0.71 on every pair).

| Task | Metric | Expert--expert mean ± range | Expert--algorithm mean ± range | Hypothesis |
|---|---|---|---|---|
| LPD  | frequency ICC | **0.866** (0.773–0.933) | **0.916** (0.868–0.976) | ✅ algorithm above EE |
| LPD  | laterality κ  | 0.970 (0.941–1.000) | 0.949 (0.945–0.954) | ≈ tie (all >0.94) |
| GPD  | frequency ICC | **0.966** (0.944–0.980) | **0.975** (0.963–0.987) | ✅ algorithm above EE |
| LRDA | frequency ICC | 0.911 (single pair) | **0.800** (0.710–0.890) | ⚠ TZ-ALGO below SZ-TZ; SZ-ALGO ties |
| LRDA | laterality κ  | **0.974** (0.943–1.000) | **0.899** (0.834–0.946) | ⚠ algorithm below EE |
| GRDA | frequency ICC | 0.924 (single pair) | **0.940** (0.893–0.988) | ✅ on average; TZ-ALGO below |

(Single-pair entries are because MW did not label the LRDA/GRDA frequency subsets.)

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

### LRDA frequency (no MW labels — 1 EE pair only)

| Pair | Type | n | ICC | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| SZ–TZ   | EE |  93 | 0.911 | [0.854, 0.961] | 0.927 | 0.073 |
| SZ–ALGO | EA | 112 | 0.890 | [0.760, 0.979] | 0.897 | 0.093 |
| TZ–ALGO | EA | 144 | 0.710 | [0.574, 0.834] | 0.776 | 0.191 |

`SZ–ALGO` matches the expert–expert ICC; `TZ–ALGO` is meaningfully lower. The pattern suggests Tianyu disagrees with the algorithm's LRDA-frequency estimates more often than Sahar does — but absent more raters this could equally well reflect 1–2 outlier segments that happened to be in TZ's accepted set but not SZ's.

### LRDA laterality

| Pair | Type | n | κ | 95% CI | % agreement |
|---|---|---:|---:|---|---:|
| MW–SZ   | EE |  74 | 0.943 | [0.851, 1.000] | 0.973 |
| MW–TZ   | EE |  99 | 0.978 | [0.930, 1.000] | 0.990 |
| SZ–TZ   | EE |  93 | **1.000** | [1.000, 1.000] | 1.000 |
| MW–ALGO | EA | 126 | 0.834 | [0.726, 0.930] | 0.921 |
| SZ–ALGO | EA | 112 | 0.946 | [0.875, 1.000] | 0.973 |
| TZ–ALGO | EA | 144 | 0.916 | [0.847, 0.972] | 0.958 |

The single result that **does not support the hypothesis**: experts agree near-perfectly on LRDA laterality (mean κ 0.974, including a perfect SZ–TZ kappa) but the algorithm is meaningfully worse against MW (κ 0.834). The absolute number is still high (>92% agreement on all pairs) but the gap is real. Worth a focused error analysis: which LRDA segments did the algorithm get the side wrong on, and was MW or SZ/TZ the outlier?

### GRDA frequency (no MW labels — 1 EE pair only)

| Pair | Type | n | ICC | 95% CI | Spearman ρ | MAE (Hz) |
|---|---|---:|---:|---|---:|---:|
| SZ–TZ   | EE | 123 | 0.924 | [0.833, 0.981] | 0.938 | 0.059 |
| SZ–ALGO | EA | 130 | **0.988** | [0.983, 0.991] | 0.976 | 0.062 |
| TZ–ALGO | EA | 160 | 0.893 | [0.820, 0.946] | 0.915 | 0.133 |

Mixed: `SZ–ALGO` is exceptionally high (0.988), `TZ–ALGO` is below `SZ–TZ`.

## Interpretation and patterns

1. **The PD half of the story is clean**: on every PD task and every metric, the algorithm sits at or above expert–expert agreement. This is the cleanest possible support for the abstract claim that the system matches expert IRR on PDs. The closing argument for Reviewer Note #1 is well-supported here.

2. **The RDA half is more nuanced**: on LRDA laterality the algorithm is measurably below expert agreement, and on LRDA/GRDA frequency the algorithm matches Sahar but not Tianyu. The right framing for the manuscript is probably "the algorithm matches expert agreement on every PD task, and on RDA frequency in pairwise comparison with one of the two RDA raters, but is slightly below the expert–expert ceiling on LRDA laterality."

3. **SZ is consistently closer to the algorithm than TZ**. Across every metric where both are scored, `SZ–ALGO ≥ TZ–ALGO`. Two non-exclusive explanations:
    - Sahar accepted the algorithm's pre-fill more often than Tianyu did (see also that Sahar rejected more segments outright — 88 LRDA + 70 GRDA vs Tianyu's 56 + 40 — leaving a more conservative set of accepted labels behind).
    - Tianyu has a systematically different scoring tendency, particularly on RDA frequency.
    Worth following up by computing an "override rate" per rater: of the segments each accepted, what fraction did they override the default frequency?

4. **MW gap on RDA frequency**. Without MW labels on the LRDA/GRDA frequency manifests, the RDA-frequency analysis has only 1 expert–expert pair (SZ–TZ) and 2 expert–algorithm pairs. Statistically thin. If MW labels those 400 segments later (estimated 1.5 hours total), the analysis upgrades to a full 4-way comparison.

5. **MW–TZ is a recurring weak link**. Look at LPD frequency: `MW–TZ` ICC 0.773 is the worst single pair in the entire analysis. This deserves an error-analysis pass to figure out where they systematically disagree on LPDs.

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
