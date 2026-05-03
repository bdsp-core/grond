# Unified PD Frequency & Classification: Redesign Plan

## Goals

A single coherent model that:
1. **Classifies**: LPD vs GPD vs BIPD
2. **Estimates frequency**: per-hemisphere for LPD/BIPD, bilateral for GPD
3. **Detects discharge timing**: one set (LPD/GPD) or two independent sets (BIPD)
4. **Works really well** for all three pattern types

## Current State

| Task | LPD | GPD | BIPD |
|------|-----|-----|------|
| Classification | AUC 0.984 (vs GPD) | — | AUC 0.840 (vs GPD) |
| Frequency ρ | **0.677** | **0.180** (broken) | Not evaluated |
| Timing F1 | 0.506 | ~0.5 | Not evaluated |
| Architecture | CNN+PLV + HemiCET+DP | Same pipeline, all channels | Two-stage: per-hemi DP → GBT |

### Root cause of GPD failure
CNN frequency estimate trained on 76% LPD data → biased to ~1.35 Hz for everything. Replacing CNN with ACF doubles GPD ρ to 0.353 but is architecturally inelegant.

## Proposed Architecture: Confidence-Weighted Dual-Estimator

### Core principle
CNN and ACF are **two noisy estimators of latent frequency**. Instead of a fixed blend (0.8 × CNN + 0.2 × ACF), learn an **adaptive weight** based on estimator confidence.

### Stage 1: Per-Hemisphere Feature Extraction

```
Input: 18-channel bipolar EEG
  │
  ├─→ ChannelPD-Net (existing, per-channel)
  │     → 18 channel PD probabilities
  │     → 18 per-channel (μ_freq, σ²_freq)  ← NEW: predict uncertainty
  │
  ├─→ Left hemisphere:
  │     CNN-weighted evidence → HemiCET+DP → {t_L}, IPI_freq_L
  │     Per-channel ACF → ACF_freq_L, ACF_confidence_L
  │     CNN ensemble → CNN_freq_L, CNN_uncertainty_L
  │
  └─→ Right hemisphere:
        Same → {t_R}, IPI_freq_R, ACF_freq_R, CNN_freq_R, uncertainties
```

### Stage 2: Confidence-Weighted Frequency Fusion (per hemisphere)

For each hemisphere h ∈ {L, R}:

**CNN uncertainty** (σ²_CNN): from the new uncertainty output of ChannelPD-Net
- Trained via heteroscedastic loss: `-log p(y|μ,σ²) = log(σ²)/2 + (y-μ)²/(2σ²)`
- The CNN learns to be uncertain for GPD (out-of-distribution) and confident for LPD

**ACF confidence** (1/σ²_ACF): from handcrafted features
- ACF peak height (higher = more periodic = more confident)
- ACF peak sharpness (narrow peak = precise frequency)
- Consistency across channels (low std of per-channel ACF peaks = confident)
- Converts to σ²_ACF via calibrated mapping

**Fusion** (inverse-variance weighting):
```
μ_fused = (μ_CNN/σ²_CNN + μ_ACF/σ²_ACF) / (1/σ²_CNN + 1/σ²_ACF)
```

This naturally handles:
- **LPD**: CNN confident (trained on LPD) → CNN dominates → ρ stays high
- **GPD**: CNN uncertain (underrepresented) → ACF dominates → ρ improves
- **BIPD**: Each hemisphere estimated independently → different frequencies allowed

### Stage 3: Classification (LPD vs GPD vs BIPD)

Uses the **timing features** from Stage 1 (existing 21-feature BIPD detector):

```
From Stage 1:
  {t_L}, {t_R}, f_L, f_R, IPI statistics
  │
  ├─→ 21 timing features (freq_ratio, phase_consistency, xcorr, matched_fraction, ...)
  │
  └─→ GBT classifier → P(LPD), P(GPD), P(BIPD)
       (3-class, trained on synthetic + real data)
```

**Classification determines output mode**:
- P(LPD) highest → report f_dominant_hemisphere, {t_dominant}
- P(GPD) highest → report f_fused_bilateral, {t_bilateral}
- P(BIPD) highest → report f_L, f_R, {t_L}, {t_R} independently

### Stage 4: Final Frequency Output

| Classification | Frequency output | Timing output |
|---------------|-----------------|---------------|
| LPD | f_fused from dominant hemisphere | {t} from dominant hemisphere |
| GPD | f_fused from bilateral (all channels) | {t} from bilateral evidence |
| BIPD | f_L from left, f_R from right | {t_L}, {t_R} independently |

For GPD bilateral fusion: instead of current 18-channel weighted average, use:
```
f_GPD = inverse_variance_weighted_mean(f_L, f_R, σ²_L, σ²_R)
```
This gracefully handles asymmetric GPD (where one hemisphere has cleaner signal).

## Implementation Plan

### Phase 1: Confidence-Weighted Fusion (no retraining)

**Effort**: ~1 day. **Impact**: GPD ρ from 0.18 to ~0.35+

1. Compute ACF confidence features:
   - Per-channel ACF peak height, sharpness, consistency
   - Calibrate to pseudo-σ² using the 86 GPD segments with 3-rater labels
   
2. For CNN uncertainty (without retraining):
   - Use ensemble disagreement across 5 folds as σ²_CNN proxy
   - High disagreement → high uncertainty → weight shifts to ACF
   
3. Implement inverse-variance fusion
4. Evaluate: must maintain LPD ρ ≥ 0.65, target GPD ρ ≥ 0.40

### Phase 2: Retrain CNN with Uncertainty Head

**Effort**: ~2 days. **Impact**: GPD ρ target 0.50+

1. Add uncertainty output to ChannelPD-Net:
   ```python
   # Current: self.freq_head = nn.Linear(hidden, 1)  # μ only
   # New:
   self.freq_head_mu = nn.Linear(hidden, 1)
   self.freq_head_logvar = nn.Linear(hidden, 1)
   ```

2. Change frequency loss from MSE to heteroscedastic NLL:
   ```python
   # Current: loss_freq = MSE(pred_freq, target_freq)
   # New:
   logvar = self.freq_head_logvar(features)
   loss_freq = 0.5 * (logvar + (pred_freq - target_freq)**2 / logvar.exp())
   ```

3. Training data:
   - Existing v1 dataset (815 patients, 9310 channels)
   - **Rebalance**: upsample GPD frequency examples 5× 
   - **Add GPD frequency labels**: 259 segments with expert freq (mean across LB, PH, SZ)
   - Keep LPD labels unchanged

4. Validation:
   - LPD freq ρ ≥ 0.65 (must not degrade)
   - GPD freq ρ ≥ 0.45 (substantial improvement)
   - Uncertainty should be higher for GPD than LPD (calibration check)

### Phase 3: 3-Class Classification (LPD/GPD/BIPD)

**Effort**: ~2 days. **Impact**: Enable BIPD detection

1. Extend existing BIPD detector (code/bipd_detector.py) to 3-class:
   - Current: binary BIPD vs GPD
   - New: 3-class LPD vs GPD vs BIPD
   
2. Synthetic training data (from BIPD_PLAN.md):
   - **GPD-like** (~1000): real GPD pairs + duplicated LPD with jitter + GPD with delay
   - **BIPD-like** (~2500): cross-patient LPD pairs + phase-shifted GPD + frequency-scaled GPD
   - **LPD-like** (~1000): real LPD with contralateral silence/low activity
   
3. 21 timing features + 3 new features:
   - CNN uncertainty ratio (left/right)
   - Laterality index from ChannelPD-Net
   - Bilateral synchrony from CET evidence cross-correlation

4. GBT classifier with 3-class output

5. Validation:
   - BIPD sensitivity ≥ 0.75 on 16-21 confirmed cases
   - GPD specificity ≥ 0.90
   - LPD accuracy maintained

### Phase 4: Integration

1. Wire confidence-weighted fusion into PDProfiler
2. Add BIPD classification output
3. Update verbal description generator for BIPD:
   - "BIPD: left hemisphere at 1.2 Hz, right hemisphere at 0.8 Hz"
4. Run full evaluation contest
5. Update paper figures and tables

## Training Data Inventory

### Available labeled data

| Label Type | LPD | GPD | BIPD | Notes |
|-----------|----:|----:|-----:|-------|
| **Segments on disk** | 4,213 | 3,357 | 2 | Non-excluded, with EEG |
| **IIIC ≥10 expert votes** | 1,860 | 1,034 | — | High-quality pattern class |
| **Expert frequency** | **839** | **259** | — | Mean across raters |
| ↳ with ≥3 raters | 140 | 86 | — | LB+PH+SZ consensus |
| ↳ with ≥2 raters | 223 | 240 | — | |
| ↳ MW-only | 566 | 0 | — | From lat+timing reviewer |
| **Discharge timing** | **893** | **158** | — | In discharge_times.json |
| **Laterality** | 942 | 253 | — | left/right/bilateral |
| **Spatial** | 276 | 259 | — | Channel involvement |
| **CNN channels (v1)** | 7,024 | 978 | — | Per-channel PD labels |
| **Confirmed cases** | 3,967 pts | 3,102 pts | 16-21 pts | |

### Synthetic data (for BIPD, from BIPD_PLAN.md)

| Source | Type | Count | Method |
|--------|------|------:|--------|
| Real GPD pairs | GPD-like | ~160 | Direct L/R from GPD cases |
| Duplicated LPD + jitter | GPD-like | ~450 | Single LPD → bilateral with σ=10-30ms jitter |
| GPD + systematic delay | GPD-like | ~480 | Add 10-50ms propagation delay |
| Cross-patient LPD pairs | BIPD-like | ~1,000 | Random L from one LPD + R from different LPD |
| Phase-shifted GPD | BIPD-like | ~800 | Shift one side by 0.25-2.0s |
| Frequency-scaled GPD | BIPD-like | ~480 | Scale one side's IPI by 1.1-2.0× |
| Similar-freq cross-LPD | BIPD-like | ~200 | Hardest case: same freq, independent phase |
| **Total synthetic** | | **~3,570** | |

### The imbalance problem

**Current CNN training data**:
- LPD: 7,024 channels (75.5%)
- GPD: 978 channels (10.5%)
- GRDA: 684 channels (7.3%)
- LRDA: 624 channels (6.7%)
- **Ratio LPD:GPD = 7.2:1** — severely imbalanced

**Frequency labels**:
- LPD: 839 segments
- GPD: 259 segments
- **Ratio LPD:GPD = 3.2:1** — moderately imbalanced

**Discharge timing labels**:
- LPD: 893 segments
- GPD: 158 segments
- **Ratio LPD:GPD = 5.7:1** — severely imbalanced

## Data Balancing Strategy

### For the CNN frequency head (Phase 2 retraining)

**Option A: Class-weighted loss** (simplest)
```python
# Weight GPD frequency loss proportional to class imbalance
n_lpd_freq = 839
n_gpd_freq = 259
gpd_weight = n_lpd_freq / n_gpd_freq  # ~3.2×

loss_freq = 0.5 * (logvar + (pred - target)² / exp(logvar))
if subtype == 'gpd':
    loss_freq *= gpd_weight
```
Pro: No data duplication. Con: GPD gradients are noisier (fewer samples, bigger weights).

**Option B: Oversampling GPD** (recommended)
- During each epoch, sample GPD frequency examples 3× (with replacement)
- Effective ratio becomes ~1:1
- Combined with augmentation (amplitude scale, noise) to reduce overfitting to repeated GPD examples

**Option C: Stratified batch sampling**
- Each training batch guaranteed to contain ~50% LPD and ~50% GPD frequency examples
- Uses a custom sampler that alternates between LPD and GPD pools
- Most stable gradients

**Recommendation**: Option B (oversampling) + Option A (2× class weight on top) = ~6× effective upweighting of GPD. This matches common practice for imbalanced medical datasets.

### For the discharge timing model (if retraining HemiCET)

- LPD: 893 segments → ~1,786 hemisphere examples (affected hemi only)
- GPD: 158 segments → ~316 hemisphere examples (both hemispheres)
- **Ratio: 5.6:1** after hemisphere expansion

**Fix**: Oversample GPD timing examples 4×, yielding ~1,264 GPD hemi examples.
Combined with augmentation: amplitude scaling (0.8-1.2), Gaussian noise (SNR 25dB), channel dropout (10%).

### For 3-class classification (Phase 3)

- Real LPD: thousands available
- Real GPD: thousands available
- Real BIPD: 16-21 cases only
- **Ratio: ~200:200:1** — extreme BIPD imbalance

**Fix**: Synthetic data generation (from BIPD_PLAN.md):
- ~1,090 GPD-like synthetic examples
- ~2,480 BIPD-like synthetic examples
- ~1,000 LPD examples (real, subsampled)
- **Final ratio: ~1:1:2.5** (LPD:GPD:BIPD) — slightly overrepresenting BIPD to compensate for synthetic vs real gap

Validate on real data only: leave-one-out on 16-21 BIPDs, held-out real GPDs.

### Cross-validation strategy

**Patient-stratified 5-fold CV** throughout:
- No patient appears in both train and val
- Each fold has proportional representation of LPD/GPD
- BIPD cases: leave-one-out (too few for 5-fold)

**Train/val split for frequency**:
- 5-fold: ~80% train, ~20% val per fold
- LPD train: ~670 segments/fold
- GPD train: ~207 segments/fold (× 3 oversampling = ~621 effective)
- GPD val: ~52 segments/fold (never oversampled — eval on real data only)

## Expected Outcomes

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|--------------|--------------|--------------|
| LPD freq ρ | 0.677 | ≥ 0.65 | ≥ 0.65 | ≥ 0.65 |
| GPD freq ρ | 0.180 | ≥ 0.35 | ≥ 0.50 | ≥ 0.50 |
| BIPD sensitivity | N/A | N/A | N/A | ≥ 0.75 |
| Lat AUC | 0.984 | 0.984 | 0.984 | 0.984 |

## Key Design Decisions

1. **Why not end-to-end?** We tried (DETR model, ρ≈0). With ~1000 labeled examples, structured models (DP + features) outperform learned decoders.

2. **Why inverse-variance fusion?** It's the Bayesian-optimal combination of two Gaussian estimates. It naturally handles the "CNN is uncertain for GPD" case without hard-coded rules.

3. **Why keep ACF?** ACF is robust for GPD (periodic signals are what ACF was designed for). The CNN adds value for LPD (where morphology matters more than periodicity). Both are useful; the question is just how to combine them.

4. **Why synthetic BIPD data?** Only 16-21 real BIPD cases. Synthetic data from cross-patient LPD pairs provides thousands of realistic "independent bilateral" examples.

5. **Why GBT for classification?** 21 handcrafted timing features + small dataset = tree-based methods dominate. Neural classifiers overfit with <100 real positive examples.
