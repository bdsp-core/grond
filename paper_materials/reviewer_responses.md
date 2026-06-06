# GROND — reviewer-response plan (final state)

Source: `IIIC-Frequency-Functions-For-Morgoth/grond_manuscript_overleaf_full_annotated/` snapshot of Overleaf with `\overleafcomment{}` macros embedded.

- **Alexandra Tautan** (AT) — substantive review on 2026-05-14, -05-18, -05-20.
- **Chenxi Sun** (CXS) — meta-notes on her own revisions, several of which we reverted in commit `a181561`.

---

## Status of every manuscript-body comment

The full per-comment annotated breakdown lives at `paper_materials/manuscript/manuscript.tex`; below is a summary of what was *changed* in the final pass.

### Abstract

- **M02** ✅ — added GPD ICC (0.980 vs EE 0.973) inline.
- **M03** ✅ — split the dense parenthetical into frequency and laterality clauses.
- **M04** ✅ — replaced `1.0~ms median timing error` with `mean absolute timing error 1.0~ms at 200~Hz; median error 0 samples`. The published "1.0 ms median" was a mis-label of the **mean**; the median IS 0 samples (algorithm and GT both quantized to 5-ms grid). Both numbers reproduce on the 147-patient n=147 evaluation; see "Reproducibility" below.
- **M05** ✅ — clarified Jaccard (PD region set, binary) vs ICC (RDA continuous extent).
- **M06** ✅ — dropped `(see Discussion)` parenthetical; reworded trailing clause.
- **M07** ✅ — softened "first system to jointly characterize" → "first system to jointly **reach expert-level inter-rater reliability** across".
- **M08** ✅ — Chenxi softening of "substitute for and improve" was reverted to original strong claim (commit `a181561`).

### Introduction

- **M09** ✅ — dropped `$p<10^{-4}$` (out-of-context p-value).
- **M11** ✅ — Spearman attribution fixed in 3 places (Introduction, §3.2 LPD/GPD, §3.2 LRDA/GRDA): the Spearman ρ values cited come from **re-evaluating** Täutan et al.'s algorithm on our cohort, not from Täutan reporting Spearman.

### Data and Annotation (§2.1–§2.3)

- **M12 / M13 / M14 / M15** ✅ — §2.1 paragraph rewritten with pre-exclusion N (13,228), classification criteria stated, quality-review criteria stated. Third data source clarified as the prior Täutan 38-patient cohort (distinct from the 200-per-subtype canonical IRR cohort).
- **M18** ✅ — laterality annotation method added (dominant hemisphere for PDs; rhythmic-envelope hemisphere for LRDA).
- **M19** ✅ — §2.5 stray LOPO sentence replaced with correct framing for parameter-free methods.
- **M22** ✅ — §2.3 cohort-assembly sentence added.

### Methods — Pipelines (§2.3–§2.4)

- **M23** ✅ — dropped Hybrid-PLV from Table 2 (PD-Profiler components); added a caption note clarifying that DLTL is the production spatial output and Hybrid-PLV is described in appendix only as a contest entry.
- **§2.4 V12 definition** (cross-cutting #1) ✅ — added "V12: retuned NB-Hilbert hyperparameters" paragraph, including the held-out-half ICC result (0.834 out-of-sample vs 0.806 in-sample) confirming V12 does not over-fit (resolves **M20**).
- **M25 / M26** ✅ — §2.3.2 product-boost paragraph extended: E(t) range stated explicitly, boost/threshold/floor values cited to contest-of-methods provenance.
- **M27** ✅ — kernel-size schedule provenance sentence added to ChannelPD-Net description.
- **M30** ✅ — BIPD synthesis clarification added per user direction (same-side or cross-side pairing is fine; what matters is timing independence).

### Evaluation Framework (§2.5)

- **M21 / M33** ✅ — §2.5 fully rewritten to declare metrics list up front (AUC, Spearman ρ, MAE Hz, F1 at ±100 ms, sensitivity/precision/timing-MAE ms, Jaccard, ICC). MAE now appears in Methods before being used in Results.
- **M38** ✅ — both evaluation cohorts (canonical MW/SZ/TZ/AS + Täutan PHLBSZ) pre-registered with rationale.
- **Reproducibility paragraph** ✅ — added (new): documents `grond_data.h5` bank and `REPRODUCIBILITY.md` pointer.

### Results (§3)

- **M32 / M34** ✅ — V12 = retuned NB-Hilbert connection made explicit (handled in §2.4 paragraph).
- **M35** ✅ — "noisier than the canonical cohort" reworded in 3 places: PH/LB/SZ produced labels by visual cycle-counting without the interactive narrowband-overlay tools — perceptual-task difficulty, not annotator skill.
- **M37** ✅ — embedded in §2.5 + §2.3 cohort-source registration; "subset of training" framing for the 582 cases is now bounded by the §2.3 + §2.5 statements that all reported metrics are out-of-fold predictions under 5-fold patient-stratified CV.
- **M39** ✅ — §3.4 PD-spatial paragraph reworded: EE ICC of 0.00 on GPD spatial extent means the labeling task itself is not technically measurable; the algorithm's 0.09 should not be read as failure.

### Summary and Discussion (§4)

- **M40** ✅ — removed all `.csv` filename references from body text (kept in Table 1 caption as an auditable artifact name).
- **M41** ✅ — added `tab:contest_summary` to `app:contest`: one-line summary per subtask of variants tested + categories explored.

### Appendix (`appendix_math_methods.tex`)

- **Theme A — Notation block** ✅ — new "Notation and conventions" subsection at top defining F_s, N, σ, GELU, BCE, MSE, F1, GFP, U, Hilbert transform, f_gt vs f̂_log distinction (fixes **A10**), E(t) vs E[n] convention, T = 1/f, HPP, CET, with explicit note about overloaded symbols.
- **Theme E — Hyperparameter provenance** ✅ — appended a paragraph stating the across-the-board policy.
- **A55** ✅ — CNN+ACF Eq.(2) reworded to explicitly introduce `F` and `F^{-1}` as the Fourier transform.
- **A65** ✅ — Discharge-locked topo bandpass spec extended ("3rd-order Butterworth, forward–backward zero phase via scipy.signal.sosfiltfilt").
- **A69** ✅ — Hybrid-PLV "Design rationale" now includes a concrete contest example.
- **Themes F / B / C / D** — minor follow-up items addressed implicitly by the notation block + provenance paragraph.

---

## Items where we *chose not to change* (per user direction)

- **M01** — REJECT change. The majority-accept filter is symmetric (applied identically to EE and EA denominators). Methods paragraph notes this.
- **M10** — keep the long Introduction motivation paragraph (kept).
- **M36** — keep "exceeded EE" framing on GPD Δ=+0.007 (kept).

---

## Reproducibility status (post data-recovery)

A separate, substantial workstream was the data-bundle reproducibility effort. Summarized here for the reviewers and for paper trail:

### What reproduces from the public repo

| Metric | Reproducible | Cohort |
|---|---|---|
| Subtype classification labels | 100 % | 12,236 segments |
| Expert frequency labels | 100 % | 3,977 patients |
| Laterality labels | 100 % | 2,760 patients |
| Spatial extent / channel involvement | 100 % | 60 patients |
| **PD-GT discharge timing F1** | n=147 of n=582 published → **F1 = 0.876 vs published 0.889** | 147 |
| Frequency-Spearman ρ | 100 % (re-evaluated on full cohort) | full per-subtype cohorts |
| Lateralization AUC | 100 % | 4,253 RDA, 7,037 PD |
| EE-vs-EA bar chart (fig:irr_main) | 100 % | full canonical IRR cohort |
| IRR cohort frequency / κ | 100 % | 648 IRR-cohort segments |

### Why only n=147 for the timing-F1 reproduction

The published F1=0.889 was evaluated on 582 PD-GT discharge-timing patients. During the May-2026 repository cleanup, the bare-PID `{pid}_seg000.mat` files for 510 of these patients were removed from `data/eeg/`. We recovered:

- 297 segments from a collaborator's backup directory (`alex_recovery`),
- 358 segments from `s3://bdsp-opendata-credentialed/iiic-freq3/data/eeg/` for the IRR cohort,
- 158 segments via a window-finder on `s3://bdsp-opendata-credentialed/morgoth1/data/internal_dataset/{LPD,GPD,IIIC}/segments_raw/` source recordings,
- = **813 newly-recovered segments**, bringing the PD-GT-resolvable cohort from 156 to 649 of 657 (98.8 %).

But the recovered segments have residual timing-alignment offsets relative to the original `discharge_times.json` annotations (filter group delay in the backup-derived files; window-boundary uncertainty in the re-extracted files). Per-source F1 on the recovered subsets is materially lower than 0.876, with median timing error ~10-50 ms (vs 0 ms for the 147 original-source patients). This degrades the *timing* metric specifically but does not affect frequency, laterality, or spatial-extent metrics (those are robust to sub-100 ms label/data shifts).

The full recovery audit lives in `REPRODUCIBILITY.md` + `results/c1_repro/*.csv` provenance logs.

### Data bank

The full reproducible bundle is packaged as `data/grond_data.h5` (1.67 GB):
- 14,341 EEG segments at 200 Hz (18-channel bipolar, float32).
- All labels (frequency, laterality, spatial, discharge timing) keyed by segment_id.
- Algorithm + Täutan-baseline predictions for every segment.
- Cohort flags (PD-GT, canonical IRR).
- Channel definitions and source-provenance attrs.

`paper_materials/generate_all_figures.py` will regenerate every figure and table from this single file.

---

## Workflow

1. Manuscript builds (pdflatex 54 pp, no undefined refs).
2. Recovery + repro work all committed under `code/data_management/` with audit logs in `results/c1_repro/`.
3. Final commit and push pending — covers manuscript submodule (`bdsp-core/grond-manuscript`) + parent repo (`bdsp-core/grond`) including data bank, scripts, and `REPRODUCIBILITY.md`.
