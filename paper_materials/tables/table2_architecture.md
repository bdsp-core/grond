# Table 2: PDProfiler Pipeline Components

| Component | Architecture | Parameters | Input | Output | Training Data | Role |
|---|---|---:|---|---|---:|---|
| ChannelPD-Net | CNN + Attention | ~70K | 1 channel (1 × 2000) | PD probability, log-frequency | 675 cases (5-fold CV) | Laterality, spatial reference, evidence weighting |
| HemiCET-UNet | Encoder-decoder U-Net | ~525K | 8-channel hemisphere (8 × 2000) | Frame-level discharge evidence ∈ [0,1] | 1,045 hemisphere examples (5-fold CV) | Discharge timing detection |
| HPP | Handcrafted features | — | 8-channel hemisphere | Pointiness + TKEO evidence | — | Complementary evidence for DP |
| Dynamic Programming | Viterbi-style DP | — | Evidence trace + frequency prior | Discharge times t₁, ..., tₙ | — | Temporal inference with periodic prior |
| CNN+ACF Ensemble | ChannelPD-Net + ACF | — | Per-channel outputs | Frequency estimate (Hz) | — | Frequency prior for DP (0.8 × CNN + 0.2 × ACF) |
| Hybrid-PLV | Phase-locking value | — | 19-channel EEG + reference | 8 region involvement scores | — | Spatial localization |
| Discharge-locked topo | Laplacian-GFP alignment | — | 19-channel monopolar + discharge times | Voltage topography (19-vector) | — | Topographic localization + verbal description |

## DP Hyperparameters (Optimized)

| Parameter | Value | Description |
|---|---|---|
| α (reward) | 1.5 | Evidence height reward |
| β (penalty) | 0.3 | Deviation from expected period penalty |
| λ (skip cost) | 0.05 | Cost per skipped expected discharge |
| max_skip | 3 | Maximum consecutive skips |
| Evidence threshold | 50th percentile | Suppress noise floor before DP |
| Post-hoc filter ratio | 0.4 | Drop peaks below 0.4 × median evidence |
