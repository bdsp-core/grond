# Table 7: Model Architecture Comparison

## End-to-End vs Structured Inference

| Model | Architecture | Params | F1 | Notes |
|---|---|---:|---|---|
| **HemiCET v2 + DP** | **CET-UNet + DP** | **525K** | **0.889** | **Neural evidence + structured DP** |
| PDNetV2 | 18ch U-Net + Transformer | 3.4M | 0.460 | Severe overfitting |
| HemiNet Design A | 8ch U-Net + Transformer | 2.3M | 0.624 | Overfitting despite smaller input |
| HemiNet Design B | 8ch dilated conv | 2.2M | 0.626 | Similar to Design A |
| HemiNet Design D | HPP wrapper | 262K | 0.605 | Most stable, lowest F1 |
| HemiNet + MAE pretrain | 8ch U-Net + Transformer | 2.3M | 0.620 | Self-supervised pretraining, minimal gain |

With ~1,000 training examples, end-to-end neural models cannot learn the temporal structure that DP encodes as a prior. The winning strategy is neural evidence generation (HemiCET) + structured inference (DP).

## HemiCET+DP Configuration Variants

| Configuration | F1 | LPD | GPD | Description |
|---|---|---|---|---|
| C1 (PRODUCTION) | **0.891** | 0.881 | 0.913 | Optimized DP + evidence threshold + post-hoc filter |
| E3: Post-hoc filtering | 0.891 | 0.880 | 0.914 | Threshold=50%, ratio=0.4 |
| E5: Midline 10ch | 0.877 | 0.863 | 0.905 | Adding midline channels hurt |
| Baseline (HemiCET v2) | 0.873 | 0.865 | 0.890 | Clean labels, default DP params |
| E2: DP re-optimization | 0.873 | 0.865 | 0.890 | α=1.5 (minimal gain from DP tuning alone) |
| Oracle (gold freq + C1 DP) | 0.919 | 0.931 | 0.893 | Theoretical ceiling with perfect frequency |

## RDA Pipeline (W05) vs Alternatives

| Method | Lat AUC | Freq ρ | Strategy |
|---|---|---|---|
| **W05_DomOnly_IterRefine** | **0.837** | 0.635 | Iterative narrowband + dom-side freq |
| W07_AutoChannel_FreqAgreement | 0.790 | **0.686** | MAD-based channel selection |
| V04_PLVSelected | 0.809 | 0.682 | Phase coherence channel selection |
| L24_EnvelopeAmplitude (lat only) | 0.826 | — | Simple envelope amplitude ratio |

76 methods evaluated in V5 lateralization contest (1,295 LRDA + 2,958 GRDA).
