# Single-Hemisphere Discharge Detector: Design & Experiment Plan

## Goal

Build the best possible periodic discharge detector that operates on **one hemisphere** (8 channels) at a time. This becomes the core building block for:
- LPD detection (run on affected hemisphere)
- GPD detection (run on both hemispheres, combine)
- BIPD detection (run on both hemispheres independently, compare timing)

**Target: beat F1=0.740** (current best full pipeline on 593 patients).

## Data

### Training set composition

| Source | Hemisphere used | N examples | Notes |
|--------|----------------|------------|-------|
| LPD with laterality | Affected hemisphere | ~170 | Laterality from classifier (AUC 0.957) |
| LPD without laterality | max(left, right) evidence | ~280 | Use hemisphere with stronger PD signal |
| GPD — left hemisphere | Left | ~160 | Each GPD gives 2 training examples |
| GPD — right hemisphere | Right | ~160 | |
| Hi-freq LPD (new labels) | Affected hemisphere | ~72 | From HPP-assisted labeling |
| **Total** | | **~842** | vs 593 in current pipeline |

### Channel mapping

Each hemisphere has 8 channels in bipolar montage:
- **Left**: Fp1-F7, F7-T3, T3-T5, T5-O1, Fp1-F3, F3-C3, C3-P3, P3-O1
- **Right**: Fp2-F8, F8-T4, T4-T6, T6-O2, Fp2-F4, F4-C4, C4-P4, P4-O2

Input shape: **(B, 8, 2000)** — 8 channels × 10 seconds × 200 Hz.

### Labels available per example
- Discharge times (ground truth, 665 cases)
- Gold standard frequency (609 cases)
- Active intervals (derived from discharge times)

## Model Architecture: HemiNet

### Input
`(B, 8, 2000)` — one hemisphere, z-scored per channel.

### Design A: U-Net with attention bottleneck (adapted from PDNetV2)

```
Stem:     Conv1d(8→48, k=15) → BN → GELU → ResBlock(48)       → (B, 48, 2000)
Enc1:     ResBlock(48→64, stride=2)                             → (B, 64, 1000)
Enc2:     ResBlock(64→96, stride=2)                             → (B, 96, 500)
Enc3:     ResBlock(96→128, stride=2)                            → (B, 128, 250)
Enc4:     ResBlock(128→192, stride=2)                           → (B, 192, 125)

Bottleneck: 2× Transformer layers (d=192, 6 heads)             → (B, 192, 125)

Dec3:     Up(192→128) + skip(Enc3) → fuse                      → (B, 128, 250)
Dec2:     Up(128→96) + skip(Enc2) → fuse                       → (B, 96, 500)
Dec1:     Up(96→64) + skip(Enc1) → fuse                        → (B, 64, 1000)

Heads (from Dec1, 100 Hz resolution):
  event_logits:  Conv(64→32→1)    → (B, 1000)  discharge probability
  active_logits: Conv(64→32→1)    → (B, 1000)  PD-active regime

Heads (from bottleneck pool):
  freq_logit:    Linear(192→1)    → (B, 1)     log frequency
```

Smaller than PDNetV2 (8 input channels, fewer parameters) — ~1.5M params.

### Design B: Pure convolutional (no Transformer)

Replace bottleneck Transformer with dilated convolutions:
```
Bottleneck: 4× dilated ResBlocks (dilation 1,2,4,8) at 125 time steps
```
Receptive field covers ~4 seconds at the bottleneck — sufficient for periodicity.

### Design C: Lightweight per-channel + cross-channel

```
Per-channel:  1D CNN per channel → (B, 8, 32, 1000)
Cross-channel: Conv2d(8×32 → 64) across channels → (B, 64, 1000)
Then: temporal U-Net decoder → event_logits, active_logits
```
Explicitly separates within-channel morphology from cross-channel consistency.

### Design D: Current pipeline components as a neural network

Keep the proven components but make them trainable:
```
Pointiness+TKEO (frozen, handcrafted) → (B, 8, 2000) evidence
CET-UNet (trainable) → (B, 8, 2000) CNN evidence
Combination module (trainable, replaces product-boost) → (B, 1, 2000) combined
Aggregation (trainable attention over 8 channels) → (B, 2000) single trace
Temporal head → event_logits at 200 Hz
```
This is a "neural wrapper" around the existing pipeline.

## Loss Functions

### Primary: Event detection

For `event_logits` vs Gaussian bump targets (σ=2 bins at 100 Hz = 20ms):

1. **Focal BCE + Dice** (from PDNetV2):
   `L_event = focal_BCE(γ=2, α=0.75) + 0.5 × soft_dice`

2. **Soft F1 loss** (differentiable approximation of F1):
   `L_f1 = 1 - 2×sum(p×y) / (sum(p) + sum(y))`
   Directly optimizes the metric we care about.

3. **Peak matching loss**: For each GT discharge, find nearest predicted peak, penalize distance. Plus penalty for unmatched predicted peaks (FPs).

### Secondary: Active region + frequency

- `L_active = BCE(active_logits, y_active)` — weight 0.3
- `L_freq = huber(freq_pred, log(gold_freq))` — weight 0.2, segment-level

### Auxiliary: Channel attention supervision

If using cross-channel attention, supervise it with the known PD channel labels (channel_pseudolabels.json has 834 patients):
- `L_channel = BCE(channel_attention, channel_pd_labels)` — weight 0.1

## Decoding Strategies

### Strategy 1: Simple peak picking (baseline)
- `p_eff = sigmoid(event_logits) × sigmoid(active_logits)^γ`
- NMS with min distance from predicted frequency
- Threshold = 0.25

### Strategy 2: Peak picking + DP refinement
- Get initial peaks from Strategy 1
- Use predicted frequency as DP prior
- Run lightweight DP (α=1.275, λ=0.05, β=0.3)
- This combines neural detection with DP's periodicity enforcement

### Strategy 3: Learned NMS
- Train a small 1D conv to suppress non-maxima
- Input: raw event_logits + predicted frequency
- Output: refined event_logits with sharper peaks

## Experiment Plan

### Experiment 0: Establish baselines

Run the **current pipeline** in per-hemisphere mode on all cases. This is the number to beat.

```
Baseline A: Current pipeline on affected hemisphere only
Baseline B: Current pipeline on both hemispheres, pick best
```

### Phase 1: Architecture search (5 experiments)

All use the same training recipe (5-fold patient CV, AdamW, cosine LR, 100 epochs, MPS GPU).

| Exp | Architecture | Input | Loss | Decoding |
|-----|-------------|-------|------|----------|
| 1.1 | Design A (U-Net+Transformer) | 8ch | Focal+Dice | Peak picking |
| 1.2 | Design B (U-Net+dilated conv) | 8ch | Focal+Dice | Peak picking |
| 1.3 | Design C (per-channel+cross) | 8ch | Focal+Dice | Peak picking |
| 1.4 | Design D (neural wrapper) | 8ch evidence | Focal+Dice | Peak picking |
| 1.5 | Best of above | 8ch | Focal+Dice | Peak picking + DP |

**Decision**: Pick best architecture. If Design D wins, the existing pipeline components are hard to beat and we should focus on improving them rather than replacing them.

### Phase 2: Training optimization (6 experiments)

Using best architecture from Phase 1:

| Exp | Change | Hypothesis |
|-----|--------|------------|
| 2.1 | 200 epochs (vs 100) | May still be undertrained |
| 2.2 | Multi-segment: use up to 5 segments per patient | More data per patient |
| 2.3 | Heavy augmentation: time warp, channel dropout, amplitude scale, noise | Regularization |
| 2.4 | Mixup on event targets | Regularization for small dataset |
| 2.5 | Curriculum: start with easy (high-freq, clear PD) cases, add harder ones | May improve convergence |
| 2.6 | Pre-train on pseudo-labels from all ~2500 EEG segments, fine-tune on GT | Leverage unlabeled data |

### Phase 3: Loss function optimization (4 experiments)

| Exp | Loss variant | Hypothesis |
|-----|-------------|------------|
| 3.1 | Soft F1 loss (replace Focal+Dice) | Directly optimizes target metric |
| 3.2 | Peak matching loss + Focal | Better peak localization |
| 3.3 | Add periodicity consistency loss | Encourage regular spacing |
| 3.4 | Add channel attention supervision | Better channel weighting |

### Phase 4: Decoding optimization (3 experiments)

| Exp | Decoding | Hypothesis |
|-----|----------|------------|
| 4.1 | Peak picking + DP post-processing | DP enforces periodicity neural model misses |
| 4.2 | Learned NMS | Better than fixed threshold |
| 4.3 | Ensemble: neural peaks + HPP peaks, merge | Combines strengths of both approaches |

### Phase 5: Ensemble and combination (3 experiments)

| Exp | Approach | Hypothesis |
|-----|----------|------------|
| 5.1 | Ensemble 5-fold models (average predictions) | Standard improvement |
| 5.2 | Neural model + current pipeline, merge detections | Best of both worlds |
| 5.3 | Neural evidence + DP (replace CET-UNet in current pipeline) | Use neural model as better evidence source |

## Training Recipe

### Standard recipe (for all experiments unless noted)

- **Optimizer**: AdamW, lr=5e-4, weight_decay=1e-4
- **Schedule**: Cosine annealing, warmup 5 epochs
- **Batch size**: 32 (reduce to 16 if OOM)
- **Epochs**: 100 (200 for exp 2.1)
- **CV**: 5-fold patient-stratified (stratify by subtype + frequency bin)
- **Device**: MPS GPU
- **Gradient clipping**: max_norm=1.0
- **Early stopping**: patience=20, monitor validation event F1

### Data augmentation (standard)

- Amplitude scaling: ×Uniform(0.7, 1.3) per segment
- Gaussian noise: σ = 0.05 × channel_std
- Channel dropout: zero 1 random channel with p=0.15
- Time shift: shift segment by ±0.5s (wrap or pad)
- **Discharge time jitter**: Add N(0, σ=5ms) jitter to each GT discharge time when constructing Gaussian bump targets. This reflects the natural ~5-10ms label noise and effectively creates multiple slightly different training targets from the same segment. Cheap way to multiply effective training data.

### Evaluation

- **Primary metric**: Event F1 with ±100ms matching tolerance
- **Secondary**: Sensitivity, Precision, Frequency Spearman
- **Compute every 5 epochs** during training
- **Report on held-out fold** (never train+eval on same patient)

## Expected Timeline

| Phase | Experiments | Est. time per exp | Total |
|-------|-----------|-------------------|-------|
| 0: Baselines | 2 | 10 min | 20 min |
| 1: Architecture | 5 | 2-3 hours | 12 hours |
| 2: Training | 6 | 2-3 hours | 15 hours |
| 3: Loss | 4 | 2-3 hours | 10 hours |
| 4: Decoding | 3 | 30 min | 1.5 hours |
| 5: Ensemble | 3 | 30 min | 1.5 hours |

Total: ~40 hours of compute. Phases can be parallelized across experiments.

## Key Hypotheses to Test

1. **8-channel hemisphere input > 18-channel full input**: Less noise, cleaner signal for LPD.
2. **GPD 2× data helps**: Each hemisphere is an independent training example.
3. **Design D (neural wrapper) may win**: The existing pipeline components (pointiness, TKEO, DP) encode strong domain knowledge that's hard to learn from 842 examples.
4. **DP post-processing helps any neural model**: Even if the neural model is good, DP's periodicity prior adds value — this was proven with the current pipeline (max(HPP,CET) > CET alone).
5. **Experiment 5.3 may be the sweet spot**: Use the neural model as a better evidence trace within the existing DP framework, rather than trying to replace DP entirely.

## Success Criteria

- **Minimum**: Match current pipeline F1=0.740 with a cleaner single-hemisphere architecture
- **Good**: F1 > 0.760 (closing gap to gold-freq reference F1=0.757)
- **Excellent**: F1 > 0.770 (exceeding gold-freq reference, possible if the model learns better frequency estimation)

## Notes

- The current pipeline's bottleneck is frequency estimation (Spearman 0.744). A model that jointly predicts timing + frequency may implicitly share information that improves both.
- The hi-freq LPD labels (72 new cases, many >3 Hz) specifically help the underrepresented high-frequency range where the current CNN freq model is weakest.
- For deployment as the BIPD building block, the model must produce reliable timing even when only one hemisphere has PDs (the other may be flat or have different activity). Include negative examples (non-PD hemispheres) in training.
