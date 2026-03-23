# Winning Model: HemiCET v2 + DP with C1 Parameters

## Overview

The system detects periodic discharges in 10-second EEG segments and estimates their frequency. It operates on one hemisphere (8 channels) at a time, combining a learned neural evidence trace with dynamic programming for temporal inference.

**Performance**: F1=0.891, Sensitivity=0.924, Precision=0.860, Frequency Spearman ρ=0.897, Timing MAE <1ms median.

---

## System Architecture (High-Level)

```
INPUT: 18-channel bipolar EEG segment (18 × 2000 samples, 200 Hz, 10 seconds)
│
├─── Hemisphere Selection
│    ├── LPD with known laterality → select 8 ipsilateral channels
│    ├── LPD unknown laterality → run both hemispheres, keep best
│    └── GPD → run both hemispheres, keep best
│
│    Left hemisphere channels:  [Fp1-F7, F7-T3, T3-T5, T5-O1, Fp1-F3, F3-C3, C3-P3, P3-O1]
│    Right hemisphere channels: [Fp2-F8, F8-T4, T4-T6, T6-O2, Fp2-F4, F4-C4, C4-P4, P4-O2]
│
├─── BRANCH A: Frequency Estimation (CNN+ACF Ensemble)
│    │
│    ├── Per-channel CNN+Attention (×8 channels, ×5 fold ensemble)
│    │   Input:  (1, 1, 2000) — one z-scored channel
│    │   Output: PD probability p_ch, log-frequency f_ch
│    │
│    ├── PD-weighted frequency: f_CNN = exp(Σ(p_ch × f_ch) / Σ(p_ch))
│    │
│    ├── Per-channel ACF on pointiness traces → f_ACF
│    │
│    └── Ensemble: f_est = clip(0.8 × f_CNN + 0.2 × f_ACF, 0.3, 3.5 Hz)
│
├─── BRANCH B: Evidence Generation (HemiCET-UNet)
│    │
│    Input:  (1, 8, 2000) — 8 z-scored hemisphere channels
│    Output: (1, 1, 2000) — discharge evidence trace E(t) ∈ [0, 1]
│    │
│    └── 5-fold ensemble average → final evidence trace
│
├─── Evidence Thresholding
│    E(t) = 0 where E(t) < percentile_50(E(t))  [suppress noise floor]
│
├─── Active Interval Detection
│    Rolling mean (1s window) > 50% of max → find longest contiguous run
│    Expand by ±0.5s, minimum 3s length
│
├─── Candidate Peak Extraction
│    Standard peaks: height > 5% of max, min distance = 0.2 × T
│    Strong peaks:   height > 50% of max, min distance = 0.1 × T
│    Merge and deduplicate
│
├─── Dynamic Programming (Hidden Point Process)
│    Forward DP with approximately-periodic prior:
│    - Node score = E(candidate)^1.5
│    - Edge score = -α × (deviation)² - β × (skips - 1)
│    - New-sequence cost = -λ per candidate
│    Parameters: α=1.5, β=0.3, λ=0.05, max_skip=3
│    Viterbi-style backtracking → optimal discharge sequence
│
├─── EM Template Refinement
│    1. Average evidence snippets (±150ms) around detected discharges → template
│    2. Normalized cross-correlation of template with full evidence trace
│    3. Re-run peak detection + DP on correlation peaks
│
├─── Post-hoc Confidence Filter
│    Drop discharges with evidence peak < 0.4 × median peak value
│
└─── OUTPUT
     ├── Discharge times: [t₁, t₂, ..., tₖ] in seconds
     └── Frequency: 1 / median(IPI) in Hz
```

---

## Neural Network Architectures

### 1. ChannelPDNetAttention (Frequency Estimation)

**Purpose**: Per-channel PD detection probability + log-frequency prediction.

```
Input: (batch, 1, 2000) — one z-scored EEG channel

Block 1: Conv1d(1→16, kernel=51, stride=2, padding=25) → BatchNorm1d(16) → GELU → Dropout(0.1)
          Output: (batch, 16, 1000)

Block 2: Conv1d(16→32, kernel=25, stride=2, padding=12) → BatchNorm1d(32) → GELU → Dropout(0.1)
          Output: (batch, 32, 500)

Block 3: Conv1d(32→64, kernel=13, stride=2, padding=6) → BatchNorm1d(64) → GELU → Dropout(0.1)
          Output: (batch, 64, 250)

Block 4: Conv1d(64→64, kernel=7, stride=2, padding=3) → BatchNorm1d(64) → GELU → Dropout(0.2)
          Output: (batch, 64, 125)

Attention: Conv1d(64→1, kernel=1) → Softmax over time dimension
           Output: attention_weights (batch, 1, 125)

Weighted Pool: features × attention_weights, sum over time → (batch, 64)

PD Head:   Linear(64→1) → Sigmoid → p_pd ∈ [0, 1]
Freq Head: Linear(64→1) → f_log (raw log-frequency)

Output: (p_pd, f_log, attention_weights)
```

**Total parameters**: ~50K
**Training**: 5-fold patient-stratified CV, multi-task loss (BCE for PD + MSE for log-freq)
**Inference**: Ensemble average of 5 fold models per channel, then PD-weighted average across 8 channels

### 2. HemiCET (Hemisphere CNN Evidence Trace U-Net)

**Purpose**: Generate a single discharge evidence trace from 8 hemisphere channels jointly.

```
Input: (batch, 8, 2000) — 8 z-scored hemisphere channels

ENCODER
═══════
enc1: Conv1d(8→32, kernel=51, stride=2, padding=25) → BatchNorm1d(32) → GELU
      Output: e1 (batch, 32, 1000)

enc2: Conv1d(32→64, kernel=25, stride=2, padding=12) → BatchNorm1d(64) → GELU
      Output: e2 (batch, 64, 500)

enc3: Conv1d(64→128, kernel=13, stride=2, padding=6) → BatchNorm1d(128) → GELU
      Output: e3 (batch, 128, 250)

enc4: Conv1d(128→128, kernel=7, stride=2, padding=3) → BatchNorm1d(128) → GELU
      Output: e4 (batch, 128, 125)

DECODER (with skip connections)
═══════════════════════════════
up4: ConvTranspose1d(128→128, kernel=4, stride=2, padding=1) → BatchNorm1d(128) → GELU
     Concatenate with e3: (batch, 256, 250)
     skip4: Conv1d(256→128, kernel=3, padding=1) → BatchNorm1d(128) → GELU
     Output: d3 (batch, 128, 250)

up3: ConvTranspose1d(128→64, kernel=4, stride=2, padding=1) → BatchNorm1d(64) → GELU
     Concatenate with e2: (batch, 128, 500)
     skip3: Conv1d(128→64, kernel=3, padding=1) → BatchNorm1d(64) → GELU
     Output: d2 (batch, 64, 500)

up2: ConvTranspose1d(64→32, kernel=4, stride=2, padding=1) → BatchNorm1d(32) → GELU
     Concatenate with e1: (batch, 64, 1000)
     skip2: Conv1d(64→32, kernel=3, padding=1) → BatchNorm1d(32) → GELU
     Output: d1 (batch, 32, 1000)

up1: ConvTranspose1d(32→16, kernel=4, stride=2, padding=1) → BatchNorm1d(16) → GELU
     Output: (batch, 16, 2000)

head: Conv1d(16→1, kernel=1) → Sigmoid
      Output: evidence (batch, 1, 2000) — values in [0, 1]

Size matching: if decoder and encoder dimensions differ by 1 sample (due to odd sizes),
               trim the larger tensor to match.
```

**Total parameters**: ~525K
**Training**:
- 5-fold patient-stratified CV
- 80 epochs, AdamW, lr=3e-4, weight_decay=1e-4, cosine annealing
- Batch size 32, MPS GPU
- Loss = BCE(pos_weight=20) + 0.1 × mean(evidence) [sharpness penalty]
         + 0.05 × max(0, HPP_evidence - CNN_evidence)² at discharge locations [floor loss]
- Targets: sharp Gaussians (σ=2 samples = 10ms) centered at each expert-labeled discharge time
- Data: 675 expert-reviewed cases → ~1,045 hemisphere examples (LPD: affected hemi, GPD: both hemis)
- Augmentation: amplitude scale ×Uniform(0.7, 1.3), Gaussian noise σ=0.05, channel dropout p=0.15, discharge time jitter N(0, σ=5ms)

**Inference**: Ensemble average of 5 fold models → single evidence trace

---

## Dynamic Programming Details

### Formulation

Given:
- Evidence trace E(t) at 200 Hz resolution
- Candidate discharge peaks C = {c₁, c₂, ..., cₙ} (local maxima of E)
- Expected period T = 1/f_est from frequency estimation

Find the subsequence S ⊆ C that maximizes:

```
Score(S) = Σᵢ NodeScore(sᵢ) + Σᵢ EdgeScore(sᵢ, sᵢ₊₁)
```

where:
```
NodeScore(c) = E(c)^1.5 - λ                    [favor strong peaks, penalize each candidate]

EdgeScore(cᵢ, cⱼ) = max over m ∈ {1,2,3}:
    -α × ((Δt - m×T) / (m×T))² - β × (m-1)    [approximately-periodic prior with skip tolerance]

where Δt = (cⱼ - cᵢ) / fs  [time between candidates in seconds]
      m = number of expected periods between candidates (1=adjacent, 2=one skip, 3=two skips)
```

### Parameters (C1 optimized)
- α = 1.5  (timing deviation penalty — quadratic cost for deviating from expected period)
- β = 0.3  (skip penalty — linear cost per skipped discharge)
- λ = 0.05 (existence cost — per-candidate cost to prevent spurious detections)
- max_skip = 3 (allow up to 3 consecutive missed discharges)
- peak_height_frac = 0.05 (minimum evidence height for candidates)

### Solution
Forward Viterbi-style DP: O(n²) where n = number of candidates (typically 20-50 per segment).

---

## Evidence Thresholding and Post-hoc Filtering

### Evidence Thresholding (before DP)
- Compute 50th percentile of non-zero evidence values
- Set evidence below this threshold to zero
- Effect: suppresses CET noise floor, prevents false candidate peaks

### Post-hoc Confidence Filter (after DP)
- For each detected discharge, look up its evidence peak value
- Compute median peak value across all detections in the sequence
- Drop any discharge with peak value < 0.4 × median
- Effect: removes weak/uncertain detections at sequence edges

---

## Hemisphere Selection Logic

```
if subtype == 'gpd':
    left_times  = run_pipeline(segment, LEFT_INDICES)   # [0,1,2,3,8,9,10,11]
    right_times = run_pipeline(segment, RIGHT_INDICES)   # [4,5,6,7,12,13,14,15]
    return left_times if len(left_times) >= len(right_times) else right_times

elif subtype == 'lpd':
    if laterality == 'left':
        return run_pipeline(segment, LEFT_INDICES)
    elif laterality == 'right':
        return run_pipeline(segment, RIGHT_INDICES)
    else:  # unknown laterality
        left_times  = run_pipeline(segment, LEFT_INDICES)
        right_times = run_pipeline(segment, RIGHT_INDICES)
        return left_times if len(left_times) >= len(right_times) else right_times
```

---

## Channel Definitions

18-channel bipolar longitudinal montage (double banana):

| Index | Channel | Chain | Hemisphere |
|-------|---------|-------|------------|
| 0 | Fp1-F7 | Left temporal | Left |
| 1 | F7-T3 | Left temporal | Left |
| 2 | T3-T5 | Left temporal | Left |
| 3 | T5-O1 | Left temporal | Left |
| 4 | Fp2-F8 | Right temporal | Right |
| 5 | F8-T4 | Right temporal | Right |
| 6 | T4-T6 | Right temporal | Right |
| 7 | T6-O2 | Right temporal | Right |
| 8 | Fp1-F3 | Left parasagittal | Left |
| 9 | F3-C3 | Left parasagittal | Left |
| 10 | C3-P3 | Left parasagittal | Left |
| 11 | P3-O1 | Left parasagittal | Left |
| 12 | Fp2-F4 | Right parasagittal | Right |
| 13 | F4-C4 | Right parasagittal | Right |
| 14 | C4-P4 | Right parasagittal | Right |
| 15 | P4-O2 | Right parasagittal | Right |
| 16 | Fz-Cz | Midline | — |
| 17 | Cz-Pz | Midline | — |

Left hemisphere indices: [0, 1, 2, 3, 8, 9, 10, 11]
Right hemisphere indices: [4, 5, 6, 7, 12, 13, 14, 15]

---

## Training Data

- **675 expert-reviewed cases** (437 LPD + 207 GPD + 31 other)
- 3 rounds of model-assisted label cleanup (61 timing corrections, 15 rejections)
- LPD: use affected hemisphere only (laterality known for all 437 cases)
- GPD: use both hemispheres as separate training examples
- Total: ~1,045 hemisphere-level training examples
- 5-fold patient-stratified cross-validation (stratified by subtype)

---

## Preprocessing

Per-channel z-score normalization:
```
for each channel i in [0..7]:
    μ = mean(channel_i)
    σ = std(channel_i)
    if σ > 1e-8:
        channel_i = (channel_i - μ) / σ
    else:
        channel_i = channel_i - μ
```

No other preprocessing (no filtering, no artifact rejection, no re-referencing).
The raw bipolar montage data at 200 Hz is fed directly to the models.

---

## Key Design Decisions and Their Justification

1. **8 channels (one hemisphere) > 18 channels (full)**: Avoids noise from uninvolved hemisphere in LPD. The 18ch pipeline (F1=0.717) is significantly worse than HemiCET (F1=0.891).

2. **Neural evidence + DP > End-to-end neural**: With ~1,000 training examples, end-to-end models overfit (F1=0.46-0.63). The DP's periodicity prior encodes domain knowledge that compensates for limited training data.

3. **Joint 8-channel input > Per-channel + aggregation**: HemiCET processes all 8 channels simultaneously, learning cross-channel patterns. Per-channel CET + median aggregation gives F1=0.707 vs HemiCET's 0.891.

4. **Evidence thresholding at 50th percentile**: Suppresses the neural evidence noise floor that causes false positive peaks. Improves precision from 0.823→0.860 with minimal sensitivity cost.

5. **Post-hoc confidence filter at 0.4× median**: Removes weak detections at segment boundaries. Combined with thresholding, gives +0.018 F1 over baseline.

6. **CNN+ACF frequency ensemble**: CNN provides the primary estimate (ρ=0.744); ACF provides a complementary view that averages out CNN errors. The 0.8/0.2 blend was optimized empirically.

7. **Label cleanup was critical**: Retraining on cleaned labels improved F1 from 0.733→0.873 (+0.14). Many original labels had timing errors or included non-PD segments.
