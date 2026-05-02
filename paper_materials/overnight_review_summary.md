# Overnight review summary (May 1–2, 2026)

Two automated review tools were run on the V14 manuscript build (post Fig-5 promotion):

- **PAT** (Paper Assessment Tool, BDSP): 31 specialized agents on `manuscript.pdf`. Submission Readiness Score: **54%**. Verdict: **needs revision**. Report: `~/GithubRepos/PAT-PaperAssessmentTool/reports/review_manuscript_20260501_210329.{md,html}`.
- **paper-agents-figures** (BDSP): 8 vision agents × 9 main-text figures. Verdict: **major revisions**. Report: `~/GithubRepos/paper-agents-figures/reports/review_20260501_204321.md`.

I applied a small set of safe, mechanical fixes overnight; everything else is documented below for your judgment.

---

## Applied this commit

### Manuscript

| change | location | rationale |
|---|---|---|
| Define **intraclass correlation coefficients (ICCs)** at first use | abstract main-results paragraph | PAT Acronym Audit |
| Define **confidence interval (CI)** at first use | abstract main-results paragraph | PAT Acronym Audit |
| Define **dynamic programming (DP)** at first use | Methods §2.2 (Discharge timing review) | PAT Acronym Audit |
| Define **inter-peak-interval (IPI)** at first use | Fig 2 caption | PAT Acronym Audit |
| Define **global field power (GFP)** at first use | Fig 2 caption | PAT Acronym Audit |
| Define **Teager–Kaiser energy operator (TKEO)** at first use | contest-table HPP row | PAT Acronym Audit |

### Figure 5 (the new IRR-comparison figure)

| change | rationale |
|---|---|
| y-axis truncated from 0–1.08 to 0.65–1.05 | paper-agents-figures advisory #1: data clusters 0.75–1.0; differences invisible in the top sliver |
| footer text size 8pt → 9pt; sig brackets tightened against bar tops | paper-agents-figures blocking #7: footer essential-stats text was below 7pt floor |
| footer note explicitly states the y-axis truncation | clarity |

PDF rebuilt successfully (49 pages).

---

## Deferred for your morning review (not auto-applied)

These are real findings that require your judgment, your approval of substantive edits, or substantive scope.

### Manuscript: highest-priority deferred items

1. **Circular-validation framing (PAT Adversarial Reviewer #2; Synthesis theme #1)**
   The reviewers flag MW's triple role (annotator → developer → re-adjudicator) as the most serious methodological concern. The trajectory of the LRDA-laterality gap closing via "MW's two-pass review" is *correctly* described in the Re-evaluation paragraph but is presented as a virtue rather than as a bias source. **Suggested action**: add a Limitations subsection (or expand the existing Limitations section) explicitly acknowledging:
   - MW served as primary annotator + algorithm developer + post-hoc adjudicator on disagreement segments.
   - The MW-only review pass tightens MW–TZ EE agreement and shrinks the EA gap, which can be read either as label-quality improvement OR as algorithm–ground-truth co-adaptation.
   - Plan a held-out validation by the 4th rater (already scheduled) and/or a fully external dataset.
   This is a **substantive narrative change**; I did not auto-write it.

2. **BIPD section caveat (PAT Statistical Methods Review; Adversarial #2)**
   The reviewers correctly note that 10 confirmed BIPD evaluation cases is too thin to support the reported AUC 0.937 / sensitivity 63%. The existing REVIEWER NOTE #3 in `manuscript.tex` (at line 502) already lists actions for this; they have not yet been incorporated into the body text. **Suggested action**: incorporate REVIEWER NOTE #3's planned changes (synthetic-to-real gap, bootstrap CIs on BIPD metrics, multi-site consortium plan, explicit "preliminary" caveat).

3. **Multiple-comparisons correction (PAT Statistical Methods Review)**
   The contest-of-methods framework tested 76+ algorithm variants without explicit multiple-testing correction. **Suggested action**: a single sentence in Methods stating that contest results are descriptive (not hypothesis-test-corrected) and that the *production* algorithm's performance is independently evaluated on the held-out independent-expert cohort.

4. **Embedded `[REVIEWER NOTE …]` colored blocks (PAT Synthesis action #10)**
   Six `\textcolor{red}{[REVIEWER NOTE …]}` scaffolds remain in the manuscript (notes #3, #4, #5, #6, #8, plus a subset note #7 in the abstract area — listed below). Each captures a real action item that should be folded into the body text and the colored block deleted. I left them in place because folding them in is substantive, not mechanical.
   - REVIEWER NOTE #3 — line ~502 — BIPD sample size + synthetic-to-real
   - REVIEWER NOTE #4 — line ~504 — uncertainty quantification
   - REVIEWER NOTE #5 — line ~493 — Morgoth integration
   - REVIEWER NOTE #6 — line ~495 — clinical impact reframing
   - REVIEWER NOTE #8 — line ~388 — spatial extent rationale (Discussion)

5. **Stress-position fixes (PAT Sentence Architecture)**
   The agent identified 15 specific high-impact rewrites that move buried key findings to sentence-end stress positions. Several are clearly improvements; a few are debatable. I did not auto-apply because rewriting the abstract opening and the LPD/GPD frequency results sentences risks breaking voice. The full list is in `~/GithubRepos/PAT-PaperAssessmentTool/reports/review_manuscript_20260501_210329.md` lines 449–510.

6. **Conciseness / wordy-phrase pass (PAT Conciseness Audit)**
   The agent claimed extensive wordiness with reduction potential of 20–25%. I checked for the agent's specific examples ("due to the fact that", "in order to", "it should be noted", "There is/are openers", "this work presents", etc.) — **none of these phrases actually appear in the manuscript**. The agent appears to have hallucinated examples from training data. The manuscript's prose density is real but is not driven by formulaic wordy phrases — it's driven by long Methods/Results paragraphs that the user may want to leave as-is for technical completeness.

7. **Abstract length & restructuring (PAT Paragraph Quality; Abstract Quality)**
   Both agents flag the abstract as "incomprehensibly dense" with statistical detail. **Suggested action**: lead with the most striking single result (e.g., the algorithm exceeds expert–expert ICC on every PD task) and move the per-pair statistical detail to Table 4 references. This is a substantive rewrite.

8. **Introduction problem-statement delay (PAT Introduction Audit)**
   Problem statement currently emerges in paragraph 4. Agent suggests opening with the core problem and the GROND solution. **Suggested action**: restructure intro paragraphs 1–4. Substantive.

### Manuscript: lower-priority deferred (worth mentioning, not blocking)

- A few additional acronyms exist that could be expanded at first use: RF, GBT, FFT, AWS, FDG-PET. None used in headline statements; current placement is fine for a specialist audience but a journal copy editor will likely flag.
- "There is/are" and "It is important to note" weak openers don't appear in the actual text (PAT hallucinated).
- VSNC framework agent suggests a memorable slogan ("From subjective annotation to expert-level automation"). Optional / stylistic.
- Inter-agent agreement was Fleiss' κ 0.184 ("slight agreement"); take individual agent judgments with grains of salt.

### Figures: highest-priority deferred items

paper-agents-figures flagged 11 BLOCKING and 14 ADVISORY issues across 9 figures. The blocking ones are:

1. **All 9 figures are PNG; journal may require TIFF/EPS/PDF.** This depends on the target journal; J. Neural Engineering historically accepts PNG with high DPI, so this may not actually block. Confirm with submission portal before re-exporting. *No action taken.*

2. **Rasterized text in all figures.** Same caveat as #1. matplotlib defaults to vector text in PDF backend; if you re-export as PDF for submission, this should resolve automatically. *No action taken; depends on journal.*

3. **Color-blind accessibility failures (red-green pairs)** — affects fig4, fig5, fig6, fig7, fig8, fig9.
   - **fig4**: orange (LPD) vs green (LRDA) collapses under deuteranopia/protanopia. **Suggested action**: switch to Okabe-Ito palette for the 4 subtypes (orange #E69F00 / sky-blue #56B4E9 / bluish-green #009E73 / vermillion #D55E00). Edit `paper_materials/generate_fig6.py:131-136` `SUBTYPE_COLORS` dict.
   - **fig5**: green "+" / red "−" sig markers — primary annotation contrast for direction of effect. **Suggested action**: instead of green/red, use a single dark color (e.g., `#222`) and rely on the +/− sign + text alone, OR use Okabe-Ito vermillion + bluish-green which are CVD-safe.
   - **fig6–9**: green EASY / orange MEDIUM / red HARD difficulty badges in `render_figures.py`. **Suggested action**: rebuild with a CVD-safe trio.

4. **Cross-figure color-semantic conflict**: fig4 uses orange for LPD; fig6–9 use orange for MEDIUM difficulty. Same color, different meaning. **Suggested action**: unify to a single Okabe-Ito palette across all figures with explicit semantic assignments documented.

5. **Captions absent / incomplete (figs 1, 2, 4, 6, 7, 8, 9)**: This is a **false positive in our case**. The agent only saw the standalone PNGs, not the LaTeX captions. Our `manuscript.tex` and `figure_legends.md` have full captions. *No action needed.*

6. **Laplacian topomap colorbars absent** — affects fig2, fig3, fig6, fig7, fig8, fig9. The colored topographies have no labeled colorbar with units. **Suggested action**: edit `render_figures.py`, `build_fig2.py`, `build_fig3.py` to draw a small colorbar adjacent to each topo. Substantive change.

7. **fig5 footer text < 7pt** — *FIXED this commit*.

8. **fig8 (LRDA characterization gallery) ~240 mm tall, exceeds 230 mm page height limit.** The four characterization gallery figures are large. **Suggested action**: tighten inter-panel whitespace in `render_figures.py` or move to supplementary.

9. **Inconsistent panel labels across figures** — fig1 small plain "A/B/C", fig2/3 large bold "A. Input", fig5 medium "A. Frequency (ICC)", fig6–9 no A/B/C labels (only difficulty badges). **Suggested action**: standardize to bold 9–10 pt top-left across all figures. Substantive style pass.

10. **Difficulty-badge boxes overlap EEG traces in fig6–9.** Opaque colored rectangles at upper-right occlude Fp1/Fp2 channel data. **Suggested action**: edit `render_figures.py` to move difficulty labels above the panel or to the left margin outside the trace area.

11. **Annotation/topo insets overlap O2 channel in fig2, fig3 panel C.** **Suggested action**: edit `build_fig2.py` / `build_fig3.py` to relocate.

### Figures: advisory deferred items (worth doing eventually)

- **fig5 paired-data visualization** (advisory #4): the bootstrap test uses paired differences but the bars don't show pairing. Consider connected dot plots or pair-lines. Subjective design call.
- **fig5 panels A & B redundancy** (advisory #11): ICC and Spearman ρ are nearly visually identical. Could move Spearman to supplement. Saves space; reasonable but optional.
- **fig4 yellow GPD points have low luminance** (advisory #5). Switch to higher-contrast color. Bound up with the broader colorblind palette redesign.
- **fig4 redundant per-panel legends + duplicate axes** (advisory #2). Could de-duplicate to reduce non-data ink. Subjective design call.
- **fig4 overplotting in dense regions** (advisory #3). Add α ≈ 0.25–0.35 or hexbin. Subjective.
- **fig1 row/column structure invisible** (advisory #9): 2 cols × 3 rows organization not labeled. Add row headers ("High agreement", "Intermediate", "Ambiguous"). Helpful.
- **fig1 scale bars overlap O2 channel** (advisory #12).
- **fig2/fig3 amplitude calibration bars missing in panel C** (advisory #8). Add 100 µV scale bars matching panel A.
- **fig6–9 size-and-supplement question** (advisory #14): characterization galleries are large; could move to supplement.

---

## Failures / agent errors worth noting

- PAT Cross-Figure Consistency (agent #31): failed with HTTP 413 ("request too large") because some figures exceed 5 MB single-image API limit. fig6–9 are 5–6.7 MB each. Same root cause as several Figure Caption / Color / Typography agent failures.
- PAT Reporting Guideline Compliance: crashed with `'item_id'` KeyError. Skip.
- PAT Reference Quality & Correctness: "no reference list found in paper" — the bibliography is present in the .bbl/.bib but the agent appears to have parsed only the body PDF text. False negative.
- PAT Missing References: 132 uncited claims flagged across 49 paragraphs. Many are clinical-introduction claims that already cite references; the agent may not be matching its claim-extraction to existing in-text citations. **Treat as advisory, not as 132 missing citations.**

---

## What I'd tackle first when you're back

1. **Limitations subsection on the circular-validation concern** (1–2 paragraphs). This is the single most important action item from PAT and the only one that affects how reviewers will read the methodology.

2. **Fold the 6 `[REVIEWER NOTE …]` red blocks into body text** (mechanical once you decide what each becomes).

3. **Colorblind palette redesign** (fig4, fig5, fig6–9). This is an afternoon's work touching `generate_fig6.py`, `build_fig_irr_bars.py`, and `render_figures.py`. Single Okabe-Ito palette with documented semantic assignments.

4. **Topomap colorbars** (fig2, fig3, fig6–9). Mechanical once we agree on placement and units.

5. **Statistical-rigor sentence in Methods** about contest multiple-testing.

The PDF is up to date as of this commit (49 pages); fig5 visibility is improved; six undefined acronyms are now defined at first use.

— overnight agent
