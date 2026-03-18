# Frequency Estimation for Periodic EEG Patterns: Review v3

## Problem Statement

We are estimating the **frequency** (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG recordings from critically ill patients. The pattern type is known. Our goal is to match expert-level inter-rater agreement.

## Data

- **556 annotated segments** (260 LPD, 296 GPD), 10s at 200 Hz, 18 bipolar channels
- **3 expert neurophysiologists** (LB, PH, SZ) independently annotated frequency
- Expert frequencies: discrete values (0.25, 0.5, 0.75, 1.0, 1.25, ... 3.0+ Hz)

## Expert-Expert Agreement (target)

| | LPD Spearman | LPD MAE | GPD Spearman | GPD MAE |
|---|---|---|---|---|
| All 3 pairs pooled | **0.525** | 0.325 | **0.476** | 0.191 |
| LOO (expert vs mean of other 2) | 0.742 | 0.297 | 0.663 | 0.163 |

## Current Best Results (138 experiments across 6 rounds)

| Method | LPD rs | GPD rs | Mean rs | LPD MAE | GPD MAE |
|--------|--------|--------|---------|---------|---------|
| Expert-Expert (pooled) | **0.525** | 0.476 | 0.50 | 0.325 | 0.191 |
| **Our best balanced** | 0.416 | **0.511** | **0.449** | 0.373 | **0.176** |
| Our best LPD-only | **0.425** | 0.232 | 0.328 | 0.380 | 0.254 |
| Method A baseline (PD2a) | 0.234 | -0.145 | 0.045 | 0.537 | 0.274 |

**GPD: We beat experts** (rs=0.511 vs 0.476, MAE=0.176 vs 0.191).
**LPD: Gap remains** (rs=0.416-0.425 vs 0.525). This is our main challenge.

## Architecture of Current Best Model

**Ridge regression on log(expert_freq)** with LOO-CV (alpha=1.0), using 10 frequency estimators as features. Predictions = exp(output), clamped to [0.2, 4.0] Hz.

### Features (with ridge coefficients, ranked by importance):

| Rank | Feature | Coefficient | Description |
|------|---------|------------|-------------|
| 1 | f_B (ACF thr=0.10) | +0.101 | Pointiness trace → smooth → ACF → first peak after 0.4s |
| 2 | f_tkeo_fft | +0.055 | TKEO trace (x²(n)-x(n-1)x(n+1)) → FFT peak in [0.3,3.5] Hz |
| 3 | f_peaks | +0.050 | Peak-count: pointiness peaks → (n-1)/time_span |
| 4 | f_spectral_coh | +0.047 | Cross-channel spectral coherence peak in [0.3,3.5] Hz |
| 5 | f_A | +0.041 | Method A: adaptive threshold on derivative → mean(1/intervals) |
| 6 | f_fft | +0.032 | FFT of pointiness trace → peak in [0.3,3.5] Hz |
| 7 | f_envelope | +0.030 | Matched-filter envelope (template bank) → FFT peak |
| 8 | is_gpd | +0.023 | Pattern type indicator |
| 9 | f_ged | +0.010 | GED spatial filter → pointiness → FFT |
| 10 | f_hps3 | +0.007 | Harmonic Product Spectrum P(f)·P(2f)·P(3f) peak |

All features except f_A use the same preprocessing: notch 60Hz → bandpass 0.5-40Hz → bipolar montage → 15Hz lowpass.

## What We Tried (6 rounds, 138 experiments)

### Round 1-2: Signal Processing Foundations
- ACF of pointiness trace → subharmonic locking (73% of errors)
- Lowering ACF threshold (0.20→0.10) → big improvement
- Peak-count frequency → noisy alone, great in combination
- FFT of pointiness → better than ACF, avoids subharmonics
- Ensemble/median of multiple estimates → robust
- Bayesian nudge toward pattern-type prior → helps MAE, not correlation

### Round 3: ML Breakthrough
- **Ridge regression on log(freq)** with 5 diverse frequency estimates → Spearman jumped from 0.36 to 0.415
- Random Forest didn't beat ridge (not enough data)
- Cepstral analysis failed (signals too short)

### Round 4: Template Matching
- Built data-driven (k-means on extracted discharge snippets) and synthetic template banks
- Matched-filter envelope helps LPD slightly, adds marginal value as ridge feature
- **YIN/CMNDF failed** — envelope too noisy for threshold-based period detection
- **SRH (subharmonic ratio) failed** — worse than plain FFT on short signals
- Cross-channel salience aggregation — theory good, execution poor

### Round 5: Feature Engineering + Target Transforms (hit ceiling)
- 25 morphological/temporal/spatial features → no improvement
- Predict period (1/f) → worse than log-freq
- Ordinal regression → same ceiling
- Stacking → same ceiling
- More templates (8→24) → marginal
- **Conclusion: bottleneck is the frequency estimates themselves, not how we combine them**

### Round 6: New Feature Traces (current round, breakthrough)
- **TKEO (Teager-Kaiser Energy Operator)**: x²(n)-x(n-1)x(n+1) → |·| → smooth → FFT. **2nd most important feature** in the ridge model. Suppresses 1/f background, sharper pulses.
- **HPS (Harmonic Product Spectrum)**: P(f)·P(2f)·P(3f) on pointiness FFT. Adds small value as ridge feature. HPS2 (2 harmonics) > HPS3 (3 harmonics) on short signals.
- **GED spatial filtering**: Narrowband/broadband covariance ratio → eigenvector → filtered signal. Poor standalone (rs=0.10), adds small value in ridge. Chicken-and-egg problem: needs rough freq estimate first.
- **Cross-channel spectral coherence**: scipy.signal.coherence averaged across adjacent channel pairs, peak frequency. **4th most important feature**. Captures spatial propagation patterns.
- **Mega ridge with all 10 features** → **Spearman 0.449** (up from 0.418 ceiling)
- Dropping Method A actually helps (Spearman 0.449 vs 0.445 with it)

## Progress Across Rounds

| Round | Mean rs | LPD rs | GPD rs | Key Insight |
|-------|---------|--------|--------|-------------|
| 1 | 0.33 | 0.29 | 0.45 | ACF threshold + peak-count |
| 2 | 0.36 | 0.43 | 0.40 | Triple median |
| 3 | 0.42 | 0.43 | 0.49 | Ridge on log-freq (ML breakthrough) |
| 4 | 0.42 | 0.38 | 0.49 | Matched-filter envelope |
| 5 | 0.42 | 0.38 | 0.49 | **(ceiling — 36 experiments, no gain)** |
| **6** | **0.449** | **0.42** | **0.51** | TKEO + HPS + coherence (broke ceiling) |
| Expert | 0.50 | 0.53 | 0.48 | — |

## Key Learnings

1. **Diverse feature traces matter more than clever combination**: Adding TKEO and spectral coherence (genuinely different signal representations) broke the ceiling that 36 experiments of feature engineering, target transforms, and stacking couldn't.

2. **TKEO > pointiness for transient detection**: TKEO (one-sample energy operator) suppresses 1/f background and produces sharper discharge pulses. It's now our 2nd most important feature.

3. **Spectral coherence captures what per-channel methods miss**: The frequency at which adjacent channels are most coherent adds information that no single-channel estimator provides.

4. **Ridge on log-freq is the right framework**: It naturally handles multiplicative errors and learns optimal weights across diverse estimators. More complex ML (RF, GBM, stacking) didn't help with 556 samples.

5. **GPD is solved**: rs=0.511 beats expert-expert 0.476. Multiple independent estimators all agree well for GPD.

6. **LPD remains hard**: rs=0.42 vs expert 0.53. The remaining gap is likely in the fundamental representation — our feature traces still struggle with polyphasic LPD morphology.

## The Remaining LPD Gap

**LPD Spearman: 0.42 vs expert 0.53 (81% of expert level)**

### What we think experts do that we can't:
- Recognize the complete discharge complex (sharp wave + slow wave = one cycle) regardless of amplitude variation or polyphasic morphology
- Use spatial propagation patterns (which channel leads, how the discharge spreads)
- Count "one stereotyped complex per cycle" rather than individual peaks

### What we've tried that didn't help LPD:
- More/different templates (8→24, PCA, synthetic)
- Morphological features (peak width, polyphasic index, amplitude ratio)
- Temporal consistency features (frequency stability across segment halves)
- Spatial features (channel agreement, laterality balance)
- YIN, SRH, alternans detection
- Separate LPD-specific models

### What remains untested:
- **Deep learning on raw EEG** (small CNN with augmentation — time-stretching to create synthetic frequencies)
- **DTW-based template matching** (handles morphological jitter better than cross-correlation)
- **Proper GED** with better initialization (current version has chicken-and-egg problem)
- **Longer analysis windows** (current 10s may have only 5-10 cycles for slow LPDs)

## Questions

1. **What would you try next for the LPD gap?** We've exhausted classical signal processing approaches. The remaining gap seems to require either (a) a fundamentally better discharge representation or (b) learning from raw data (CNN).

2. **Is the 1D-CNN with time-stretch augmentation worth trying?** We have 260 LPD segments. Time-stretching could give us ~2600 synthetic training samples. Would a small EEGNet-style architecture work, or would it overfit?

3. **DTW-based template matching** — your colleague suggested this might handle LPD morphological jitter. Is the computational cost worth it for 18 channels × 10s × 200Hz?

4. **Is 81% of expert-expert agreement a reasonable stopping point?** Or do you think there's a clear path to 90%+?

5. **For the paper**: we now have a method that beats experts on GPD and reaches 81% on LPD. What's the right framing — "matches expert performance" (true for GPD, debatable for LPD) or "approaches expert performance" (more honest)?

## Technical Summary

```
Best model: Ridge regression on log(expert_freq), LOO-CV, alpha=1.0
Features:   10 frequency estimators + metadata
Training:   556 segments (260 LPD, 296 GPD)
Target:     log(median expert frequency)

Preprocessing: notch 60Hz → bandpass 0.5-40Hz → bipolar montage → 15Hz lowpass
Feature traces: pointiness, TKEO, matched-filter envelope
Frequency extraction: ACF, FFT, peak-count, HPS, spectral coherence, GED

Results:
  GPD: Spearman=0.511 (vs expert 0.476), MAE=0.176 Hz (vs expert 0.191)
  LPD: Spearman=0.416 (vs expert 0.525), MAE=0.373 Hz (vs expert 0.325)
```
