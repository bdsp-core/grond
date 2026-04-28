# LRDA disagreement triage

**7 segments** where MW disagrees with the algorithm AND with at least one other expert (SZ or TZ), on either frequency (>0.5 Hz) or laterality. Sorted by severity (largest disagreement first).

For each case, all four labels are shown side-by-side. **Bold** marks any value that differs from MW by more than the disagreement threshold; *italics* marks missing labels. The "Why" column shows which metric triggered the disagreement.

| # | mat_file | Why | MW freq | SZ freq | TZ freq | ALGO freq | MW lat | SZ lat | TZ lat | ALGO lat |
|---|---|---|---:|---:|---:|---:|---|---|---|---|
| 1 | `sub-I0002150016440_20200614003334` | freq+lat | 3.25 | **1.75** | *--* | **1.70** | left | **right** | *--* | **right** |
| 2 | `sub-S0002115111161_20201101205535` | freq | 3.00 | **2.00** | 3.00 | **2.07** | right | right | right | right |
| 3 | `sub-I0002150002865_20181125150611` | freq | 2.00 | **1.00** | 1.75 | **1.09** | left | left | left | left |
| 4 | `sub-S0001117792110_20221222091421` | freq | 3.00 | **2.25** | 3.00 | **2.13** | right | right | right | right |
| 5 | `sub-S0001111599067_20151028100451` | freq | 2.00 | **1.25** | *--* | **1.16** | left | left | *--* | left |
| 6 | `sub-I0002150018705_20200629212516` | freq | 2.75 | **2.00** | 2.50 | **1.91** | left | left | left | left |
| 7 | `sub-S0002111849106_20200918135149` | freq | 2.50 | *--* | **1.75** | **1.73** | right | *--* | right | right |

## How to review

1. Generate the focused viewer for these cases:

   ```bash
   conda run -n morgoth python code/generators/labeling/generate_rda_freq_labeler.py \
       --subtype lrda \
       --manifest paper_materials/independent_expert_tasks/lrda/disagreement_manifest.csv \
       --output paper_materials/independent_expert_tasks/lrda/disagreement_viewer.html \
       --no-open
   ```

2. Open `disagreement_viewer.html` and step through the cases (← / → arrows). For each case, the table above tells you exactly which raters disagreed.

3. For each case, decide:
   - **MW labeling error**: change the row in `labels.csv` (rater=MW, label_type=frequency_hz or laterality, mat_file=...).
   - **Genuine ambiguity**: leave as-is, note in the manuscript that some LRDA segments are inherently ambiguous.
   - **Algorithm bug**: file an issue, design an error-analysis fix.

4. After any label changes, re-run:

   ```bash
   conda run -n morgoth python code/data_management/build_segment_labels.py
   conda run -n morgoth python code/evaluation/analyze_independent_expert_v1.py
   cp results/independent_expert_v1/forest_plot.png paper_materials/figures/figS5_independent_expert_v1_irr.png
   ```
