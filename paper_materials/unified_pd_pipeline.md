# Unified PD Characterization Pipeline

## Overview

Given a 10-second, 18-channel bipolar EEG segment classified as having periodic discharges (LPD or GPD), this pipeline produces four outputs:

1. **Laterality** (LPD only): left vs right hemisphere
2. **Spatial localization**: which of 8 brain regions are involved
3. **Discharge timing**: when each periodic discharge occurs
4. **Frequency**: repetition rate in Hz

## Pipeline Components

### Component 1: ChannelPD-Net (Per-Channel PD Detector)

**Architecture**: CNN+Attention, ~50K parameters, 5-fold ensemble
**Input**: Single EEG channel (1 × 2000 samples at 200 Hz), z-scored
**Output**: PD probability (0-1) + log-frequency estimate

This network processes each of the 18 bipolar channels independently, producing a per-channel probability that the channel carries periodic discharges. These 18 probabilities serve three downstream purposes:

- **Laterality**: Comparing left vs right hemisphere mean probabilities determines which side has the PDs (AUC = 0.98 on 437 cases)
- **Spatial reference**: The channels with highest PD probability become reference channels for phase-locking analysis
- **Evidence weighting**: Channels with higher PD probability receive more weight when aggregating discharge evidence

### Component 2: Hybrid-PLV (Spatial Localizer)

**Method**: Phase-Locking Value with CNN-guided reference selection
**Input**: 18-channel EEG + ChannelPD-Net probabilities + laterality (from Component 1)
**Output**: Per-region involvement scores for 8 brain regions

Steps:
1. For LPD: restrict reference channel candidates to the **ipsilateral hemisphere** (determined by laterality detection). For GPD: use all channels.
2. ChannelPD-Net identifies the top 3 channels by PD probability within the permitted set → these become the **reference channels**
3. Bandpass filter all channels at 0.5–3.5 Hz (PD frequency range)
4. Extract instantaneous phase via Hilbert transform on each channel
5. Compute **phase-locking value (PLV)** between each channel and the mean reference phase
6. Combine: channel score = 0.5 × PD probability + 0.5 × PLV
7. Map 18 channel scores to 8 brain regions (max score across contributing channels per region)

The laterality-guided reference selection ensures the spatial localizer seeds from the hemisphere where PDs are actually present, preventing contralateral noise from corrupting the phase reference.

Performance: Composite = 0.811 (MacroF1 = 0.847, Jaccard = 0.736, AUC = 0.814) on 465 segments with 3-rater expert annotations.

### Component 3: HemiCET+DP (Discharge Detector)

**Architecture**: Two evidence streams combined with dynamic programming

**Evidence Stream A — Handcrafted (HPP)**:
For each channel, compute:
- Pointiness trace: prominence² / width at each peak of the rectified signal
- TKEO (Teager-Kaiser Energy Operator) on 20 Hz lowpassed signal
- Weighted combination → Gaussian smooth → E_hpp(t)

**Evidence Stream B — Neural (HemiCET)**:
- CET-UNet: encoder-decoder CNN with skip connections (~500K params)
- Input: single z-scored channel (1 × 2000)
- Output: frame-level discharge evidence E_cet(t) ∈ [0, 1]
- 5-fold ensemble, sigmoid output

**Evidence Combination**:
1. Threshold CET at 80th percentile (suppress noise floor)
2. Apply CET floor at 0.3
3. Product-boost: E(t) = max(HPP, CET) + 3 × HPP × CET

**Channel Aggregation** (CNN-weighted, laterality-guided):
- Weight each channel's evidence by its ChannelPD-Net probability
- For LPD: aggregate only **ipsilateral hemisphere** channels (determined by laterality detection, weighted average)
- For GPD: aggregate all 18 channels (weighted average)

The laterality detection step directly controls which channels contribute to evidence aggregation, ensuring discharge detection focuses on the hemisphere with PDs.

**Dynamic Programming Inference**:
1. Detect active interval (region with sustained high evidence)
2. Extract candidate peaks (local maxima above 5% of max evidence)
3. Forward DP with approximately-periodic prior:
   - Reward: evidence height at candidate
   - Penalty: deviation from expected IPI (α=1.275, β=0.3)
   - Skip penalty: λ=0.05 per skipped expected discharge
4. EM template refinement (3 iterations, ±150ms window)
5. Post-hoc filter: drop peaks below 0.3× median evidence

**Frequency Estimation**:
- Primary: CNN+ACF ensemble (0.8 × CNN frequency + 0.2 × ACF frequency)
- Output: median(1 / IPI) from detected discharge times

Performance: F1 = 0.74 on 593 patients with expert-reviewed discharge times.

### Component 4: Frequency Estimator (CNN+ACF Ensemble)

**CNN frequency**: ChannelPD-Net's log-frequency output, aggregated across channels weighted by PD probability
**ACF frequency**: Autocorrelation of pointiness traces, first peak in delta range
**Ensemble**: 0.8 × CNN + 0.2 × ACF

Used as the frequency prior for DP inference. Final frequency is IPI-derived (from detected discharge times).

Performance: Spearman ρ = 0.744 (CNN+Attention direct) on 594 patients.

## Complete Pipeline Flow

```
EEG Segment (18 channels × 2000 samples @ 200 Hz)
│
├─── ChannelPD-Net (×18 channels, 5-fold ensemble)
│    → 18 per-channel PD probabilities
│    → 18 per-channel frequency estimates
│    │
│    ├─ IF LPD: Laterality Detection
│    │  Compare mean(left 8 probs) vs mean(right 8 probs)
│    │  → laterality = 'left' or 'right'
│    │  │
│    │  ├──── ipsilateral seed ────→ Hybrid-PLV
│    │  └──── hemisphere selection → HemiCET+DP
│    │
│    ├─ Hybrid-PLV Spatial Localization
│    │  Top-3 CNN channels from ipsilateral hemisphere → reference phase
│    │  PLV to each channel → combined scores
│    │  → 8 region involvement scores
│    │  → list of involved regions
│    │
│    ├─ CNN+ACF Frequency Ensemble
│    │  0.8 × CNN freq + 0.2 × ACF freq
│    │  → frequency prior for DP
│    │
│    └─ Evidence Weighting
│       Channel weights = PD probabilities
│       │
│       ├─ Per-Channel HPP Evidence (pointiness + TKEO)
│       ├─ Per-Channel CET Evidence (HemiCET U-Net)
│       │
│       ├─ CNN-Weighted Aggregation
│       │  LPD: weighted avg of ipsilateral channels
│       │  GPD: weighted avg of all channels
│       │
│       ├─ Product-Boost Combination
│       │  E(t) = max(HPP, CET) + 3 × HPP × CET
│       │
│       └─ DP + EM Inference
│          → discharge times
│          → IPI-derived frequency
│
Output:
  ├─ laterality (LPD: left/right; GPD: N/A)
  ├─ regions: list of involved brain regions
  ├─ region_scores: per-region confidence (0-1)
  ├─ discharge_times: list of seconds
  ├─ frequency: Hz (median 1/IPI)
  └─ channel_probs: per-channel PD probability
```

## Performance Summary

| Task | Metric | Value | N | Method |
|------|--------|-------|---|--------|
| Laterality (L vs R) | AUC | **0.963** | 423 | ChannelPD-Net hemisphere comparison |
| Spatial Localization | Composite | **0.811** | 465 | Hybrid-PLV (CNN ref + PLV) |
| Spatial Localization | AUC (mean) | **0.814** | 465 | Hybrid-PLV |
| Discharge Timing | F1 | **0.684** | 651 | HemiCET+DP (CNN-weighted) |
| Frequency | Spearman ρ | **0.681** | 500 | CNN+ACF → IPI |

## Code

Standalone callable: `code/pd_characterizer.py`

Pipeline figure: `paper_materials/figures/fig2-unified-pipeline.png`

Contest results: `results/leaderboards/spatial_contest/` (26 methods), `results/leaderboards/rda_contest/` (45 methods)

```python
from pd_characterizer import PDCharacterizer

charzer = PDCharacterizer()
result = charzer.characterize(eeg_18ch, subtype='lpd')

# result keys:
#   laterality, laterality_confidence,
#   regions, region_scores,
#   discharge_times, frequency, freq_estimate_input,
#   n_discharges, channel_probs
```

## For Figure Generation (paper-banana)

**Title**: Unified Periodic Discharge Characterization Pipeline

**Layout**: Top-to-bottom flow diagram with the following blocks:

**Input block** (top):
- "18-Channel Bipolar EEG (10 sec, 200 Hz)"
- Show a small EEG montage icon

**Block 1** — ChannelPD-Net (wide block spanning full width):
- "ChannelPD-Net: Per-Channel CNN+Attention (5-fold ensemble)"
- Input arrow from EEG, showing "×18 channels"
- Output: "18 PD probabilities + 18 frequency estimates"
- Color: blue
- This block feeds into all downstream blocks via arrows

**Block 2** — Three parallel outputs from ChannelPD-Net (side by side):

- **2a** (left, green): "Laterality Detection"
  - "Compare L vs R hemisphere mean probabilities"
  - Output: "Left / Right"
  - Label: "AUC = 0.98"
  - Note: "LPD only"
  - **CRITICAL**: Draw dashed arrow from 2a to 2b labeled "ipsilateral seed"
  - **CRITICAL**: Draw dashed arrow from 2a to 2c labeled "hemisphere selection"

- **2b** (center, orange): "Hybrid-PLV Spatial Localizer"
  - "Reference channels from ipsilateral hemisphere"
  - "Phase-locking value to each channel"
  - "0.5 × PD prob + 0.5 × PLV"
  - Output: "8 region scores"
  - Show 8 small boxes: LF RF LT RT LCP RCP LO RO
  - Label: "Composite = 0.811"

- **2c** (right, purple): "CNN+ACF Frequency Prior"
  - "0.8 × CNN freq + 0.2 × ACF"
  - Output arrow labeled "freq prior" going down to Block 3

**Block 3** — HemiCET+DP (large block, below center):
- "HemiCET+DP: Discharge Detection"
- Inside, show two parallel streams merging:
  - Left stream: "HPP (Pointiness + TKEO)" — handcrafted
  - Right stream: "HemiCET U-Net" — neural
  - Merge arrow: "Product-boost: max(HPP,CET) + 3×HPP×CET"
- Below merge: "CNN-weighted channel aggregation"
- Below that: "DP + EM → discharge sequence"
- Arrow from Block 2c feeding in "freq prior"
- Color: red
- Label: "F1 = 0.74"

**Output block** (bottom):
- Four output boxes in a row:
  1. "Laterality" (green)
  2. "Regions" (orange)
  3. "Discharge Times" (red)
  4. "Frequency (Hz)" (red)

**Arrows**:
- ChannelPD-Net probabilities feed into: Laterality, Hybrid-PLV (as reference), and HemiCET+DP (as channel weights)
- This "one CNN feeding everything" is the key architectural insight to emphasize

**Key visual emphasis**:
- The ChannelPD-Net is the backbone — highlight that a single per-channel CNN serves triple duty (laterality, spatial reference, evidence weighting)
- **Laterality Detection feeds forward** into both Spatial Localizer (ipsilateral seed) and Discharge Detector (hemisphere selection) via dashed arrows — this is the key inter-module dependency
- The Hybrid-PLV combines learned (CNN) and physics-based (PLV) features
- The HemiCET+DP combines handcrafted (HPP) and neural (CET) evidence
