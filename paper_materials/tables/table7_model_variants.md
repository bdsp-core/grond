# Table 7: Model Architecture Comparison

*Auto-generated from model evaluation result files.*
*Regenerate: `conda run -n morgoth python paper_materials/tables/generate_table7.py`*

## End-to-End vs Structured Inference

| Model | F1 | Freq ρ | Source |
|---|---|---|---|
| **CET-UNet + DP (production)** | **0.740** | **0.718** | cet_cache/consolidated_results.json |
| PDNetV2 (18ch U-Net+Transformer) | — | — | pdnet_v2_cache/evaluation_results.json |
| E2E-CNN (phase 1) | 0.006 | 0.001 | e2e_cache/e2e_phase1_results.json |
| HemiNet exp1_1 | — | — | hemi_cache/exp1_1/eval_results.json |
| HemiNet exp1_2 | — | — | hemi_cache/exp1_2/eval_results.json |
| HemiNet exp1_4 | — | — | hemi_cache/exp1_4/eval_results.json |
| HemiNet exp1_5_finetune | — | — | hemi_cache/exp1_5_finetune/eval_results.json |

With ~1,000 training examples, end-to-end neural models cannot learn the temporal structure that DP encodes as a prior. The winning strategy is neural evidence generation (CET-UNet) + structured inference (DP).

## CET-UNet + DP Configuration Variants

| Configuration | F1 | Freq ρ | Source |
|---|---|---|---|
| iterative_freq | — | — | improvement_iterative_freq_results.json |
| learned_combine | — | — | improvement_learned_combine_results.json |
| posthoc | — | — | improvement_posthoc_results.json |
| stage_ab | — | — | improvement_stage_ab_results.json |

## RDA Pipeline — Top Methods (V5 Contest)

| Rank | Method | Lat AUC | Freq ρ |
|---|---|---|---|
| 1 | V25_FreqBandEnvRatio | 0.853 | — |
| 2 | W05_DomOnly_IterRefine | 0.837 | 0.635 |
| 3 | L24_EnvelopeAmplitude | 0.826 | — |
| 4 | V12_IterativeRefine | 0.825 | 0.595 |
| 5 | V04_PLVSelected | 0.809 | 0.682 |
| 6 | L05_RMSAmplitude | 0.797 | — |
| 7 | U10_MultiCh_HilbertFreq | 0.790 | 0.601 |
| 8 | V01_DomHemi_Top3Hilbert | 0.790 | 0.575 |
| 9 | V02_PowerWeightedHilbert | 0.790 | 0.619 |
| 10 | V22_EnvAmp_DomHilbert | 0.790 | 0.650 |

76 methods evaluated on 4,253 segments.

