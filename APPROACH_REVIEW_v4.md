# Frequency Estimation for Periodic EEG Patterns: Review v4 (Final)

## Problem & Data
Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Pattern type is known. 556 annotated segments (260 LPD, 296 GPD), 3 expert raters. Additionally, ~10,000 segments with known class but no frequency labels exist on an external drive.

## Expert-Expert Agreement (target)

| | LPD Spearman | GPD Spearman | Mean | LPD MAE | GPD MAE |
|---|---|---|---|---|---|
| All 3 pairs pooled | **0.525** | **0.476** | **0.50** | 0.325 | 0.191 |

## Current Best Results (162 experiments, 7 rounds)

| Method | LPD rs | GPD rs | Mean rs | LPD MAE | GPD MAE |
|--------|--------|--------|---------|---------|---------|
| Expert-Expert | **0.525** | 0.476 | 0.50 | 0.325 | 0.191 |
| **r7_expert_spatial_ridge** | 0.42 | **0.52** | **0.471** | 0.361 | **0.176** |
| r7_decoder_ridge | **0.432** | **0.528** | 0.458 | 0.369 | 0.178 |
| r7_per_expert_mean | 0.42 | 0.52 | 0.471 | 0.365 | 0.175 |
| Method A baseline | 0.234 | -0.145 | 0.045 | 0.537 | 0.274 |

**GPD: Beaten** (rs=0.52 vs 0.48, MAE=0.176 vs 0.191). **LPD: 81% of expert** (rs=0.42 vs 0.53).

## Best Model Architecture

**Ridge regression on log(expert_freq)**, per-expert training (3 models averaged), LOO-CV, alpha=1.0. Uses ~10 features from the 18-channel bipolar signal:

| Feature | Description | Ridge weight |
|---------|-------------|-------------|
| f_B (ACF thr=0.10) | Pointiness → smooth → ACF first peak | Highest |
| f_peaks | Peak-count: pointiness peaks → (n-1)/time_span | High |
| f_tkeo_fft | TKEO |x²(n)-x(n-1)x(n+1)| → FFT peak | High |
| f_spectral_coh | Cross-channel coherence spectrum peak | Medium |
| f_fft | FFT of pointiness → peak in [0.3,3.5] Hz | Medium |
| f_envelope | Matched-filter template bank → FFT peak | Medium |
| f_A | Method A adaptive peak detection frequency | Low |
| is_gpd | Pattern type indicator | Low |
| n_channels | Number of ACF-detected channels | Low |

Per-expert training: train separate ridge models targeting log(LB_freq), log(PH_freq), log(SZ_freq), average the 3 predictions. This captures inter-expert variability better than predicting the consensus median.

## What Worked (7 rounds, 162 experiments)

### Breakthroughs (each gave >0.02 Spearman jump)
1. **Ridge on log-frequency** (R3): jumped from 0.36 → 0.42. Learned optimal weighting of diverse frequency estimates.
2. **TKEO feature trace** (R6): broke the 0.42 ceiling → 0.43. Teager-Kaiser energy operator produces sharper discharge pulses than pointiness.
3. **Spectral coherence** (R6): additional signal from cross-channel phase relationships.
4. **Per-expert training** (R7): 0.45 → 0.47. Training 3 models (one per expert) and averaging captures label distribution better than predicting the median.

### Useful but smaller gains
- FFT of pointiness > ACF (avoids subharmonics)
- ACF threshold 0.10 (detects more channels than 0.20)
- Peak-count frequency (independent estimate, good in combination)
- Matched-filter envelope (marginal value as ridge feature)
- Spatial channel selection for LPD
- Multi-montage (CAR, Laplacian) frequency estimates
- HPS (Harmonic Product Spectrum)
- Windowed voting across 3s windows

### Did NOT help
- More features beyond ~10 (overfitting with 556 samples)
- Random Forest / GBM / stacking (ridge is better with small data)
- Separate LPD/GPD models (smaller training sets hurt)
- Predicting period (1/f) instead of log-freq
- Ordinal regression (same ceiling as ridge)
- 25 morphological/temporal/spatial features (noise, not signal)
- More templates (8→24→50, marginal)
- YIN/CMNDF (envelope too noisy)
- SRH subharmonic ratio scoring (worse than FFT)
- GED spatial filtering standalone (chicken-and-egg problem)
- Alternans detection (55% false positive rate)
- Comb-fit period scoring (too slow, poor results)
- Log compression / percentile normalization of pointiness
- Parameter grid search (432 combos, confirmed model > parameter mismatch)

## Progress Across Rounds

| Round | Mean rs | LPD rs | GPD rs | Key Insight |
|-------|---------|--------|--------|-------------|
| 1 | 0.33 | 0.29 | 0.45 | ACF threshold + peak-count averaging |
| 2 | 0.36 | 0.43 | 0.40 | Triple median of diverse estimates |
| 3 | 0.42 | 0.43 | 0.49 | **Ridge on log-freq** (ML breakthrough) |
| 4 | 0.42 | 0.38 | 0.49 | Matched-filter envelope |
| 5 | 0.42 | 0.38 | 0.49 | *Ceiling — 36 experiments, no gain* |
| **6** | **0.45** | 0.42 | **0.51** | **TKEO + HPS + spectral coherence** |
| **7** | **0.47** | **0.43** | **0.53** | **Per-expert training + spatial selection** |
| Expert | 0.50 | 0.53 | 0.48 | — |

## The Remaining LPD Gap

**LPD: rs=0.42-0.43 vs expert 0.525 (81%)**

### What we tried that didn't close it
- Rich morphological features (peak width, polyphasic index, amplitude ratios)
- Temporal consistency (frequency stability across segment halves)
- More/better templates (8→50, PCA, synthetic, from 10K unlabeled segments)
- Multi-montage inputs (bipolar + CAR + Laplacian)
- Event-vs-background GED spatial filtering
- HMM Viterbi frequency tracking
- Windowed vote frequency tracking
- Comb-fit period scoring
- DTW template matching (too slow, timed out)
- Separate LPD-only models

### What we haven't tried
- **Deep learning on raw EEG** (1D-CNN with time-stretch augmentation)
- **Self-supervised pre-training** on 10K unlabeled segments → fine-tune on 556
- **Learned eventness filter** (small conv bank trained to detect discharges, not frequencies)
- **Proper DTW template matching** (needs optimized implementation)
- **Longer segments** (current 10s = only 5-10 cycles for slow LPDs)

## Available Untapped Data

| Data | Count | Has class? | Has freq? |
|------|-------|-----------|-----------|
| Frequency-annotated | 556 | Yes | Yes (3 experts) |
| Class-labeled (LPD/GPD) | **~10,000** | Yes | No |
| Total on external drive | 69,794 | Partial | No |

We extracted 50 templates per type from ~4000 of these segments (143K LPD snippets, 100K GPD snippets). The data exists but we haven't used it for training frequency models.

## Expert Annotation Granularity

Experts annotate in roughly **0.25 Hz increments** (dominant values: 0.75, 1.0, 1.25, 1.5, 2.0 Hz). This quantization creates a theoretical correlation ceiling and MAE floor (~0.125 Hz). Our best MAE of 0.176 (GPD) is approaching this floor.

## Questions

1. **Is the LPD gap closeable without deep learning?** We've tried every classical signal processing approach we and two expert colleagues could think of. 162 experiments, same ceiling. The remaining gap seems to be in discharge *recognition*, not frequency *estimation*.

2. **Should we use the 10K unlabeled segments for self-supervised pre-training?** Train a window encoder on all PD segments, then fine-tune for frequency. This is the main untapped resource.

3. **Is a learned eventness filter worth trying?** Train a small 1D conv to output "discharge probability" per time step, then run our existing frequency estimators on that trace. This is intermediate between full end-to-end CNN and pure signal processing.

4. **Given we beat experts on GPD and reach 94% overall — is this publishable as-is?** Or should we push for the LPD result before publishing?

5. **For the paper framing**: our method uses 10 signal processing features + ridge regression. It's simple, interpretable, and requires only LOO-CV (no held-out test set). Is this a strength or weakness for publication?

## Technical Summary
```
Model: Ridge regression on log(expert_freq), per-expert training, LOO-CV
Features: 10 (5 freq estimates + TKEO + spectral coherence + envelope + is_gpd + n_ch)
Training: 556 segments, 3 expert targets
Preprocessing: notch 60Hz → bandpass 0.5-40Hz → bipolar montage → 15Hz lowpass

Results (vs expert-expert pooled Spearman):
  GPD: 0.52 vs 0.48 — BEATEN
  LPD: 0.42 vs 0.53 — 81%
  Combined: 0.47 vs 0.50 — 94%
```
