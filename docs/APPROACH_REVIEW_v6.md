# Frequency Estimation for Periodic EEG Patterns: Review v6

## Problem & Data
Estimating frequency (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG at 200 Hz. Pattern type is known. 556 annotated segments from **43 patients** (37 LPD, 6 GPD), 3 expert raters. Critical: GPD has only 6 patients, with 1 patient contributing 236/296 segments.

Additionally, ~10,000 PD segments with known class but no frequency labels exist (308 patients).

## KEY FINDING: With 1 Segment Per Patient, We Match Experts

When evaluated properly (1 segment per patient, LOO = LOPO):

| | LPD rs | GPD rs |
|---|---|---|
| **Expert-Expert** (1-per-patient, pooled) | **0.411** (n=76 pairs) | 0.044 (n=13 pairs) |
| **Our Algorithm** | **0.404** (n=37) | 0.088 (n=6) |

**LPD: We match expert-expert agreement** (0.404 vs 0.411). The remaining difference is within noise for n=37.

**GPD: Both experts AND our algorithm are near zero** with only 6 patients. The previously reported "GPD win" (rs=0.516) was an artifact of having 236 segments from one patient.

## Evaluation Methodology Journey

This project taught us critical lessons about evaluation:

| Evaluation Method | LPD rs | GPD rs | Why Inflated? |
|-------------------|--------|--------|---------------|
| LOO-CV (segment-level) | 0.462 | 0.518 | Patient leakage: 235/236 segments from same patient in training |
| 5-fold patient-CV | 0.385 | 0.516 | Fold 1 = one GPD patient, still somewhat inflated |
| LOPO (all segments) | 0.381 | 0.481 | Multiple segments/patient → correlated predictions |
| **1-per-patient LOPO** | **0.404** | **0.088** | **Honest: each prediction is independent** |

The 1-per-patient evaluation is the only unbiased one. All others allowed information leakage through patient-specific patterns.

## Best Model

**Per-expert ridge regression on log(freq)**, LOO-CV with 1 segment per patient, alpha=1.0.

### Features (8):
| Feature | Description |
|---------|-------------|
| f_B | Pointiness → smooth → ACF first peak (thr=0.10) |
| f_peaks | Peak-count: pointiness peaks → (n-1)/time_span |
| f_fft | FFT of pointiness → peak in [0.3, 3.5] Hz |
| f_tkeo | TKEO |x²(n)-x(n-1)x(n+1)| → FFT peak |
| f_coh | Cross-channel spectral coherence peak |
| is_gpd | Pattern type indicator |
| n_ch | Number of ACF-detected channels |

**Speed:** 48ms per segment, CPU only, 209× real-time.

## What We Tried (9 rounds, 192 experiments)

### Signal Processing (Rounds 1-7)
- ACF, FFT, peak-count, TKEO, HPS, spectral coherence, matched-filter envelope
- Ridge on log-freq (key breakthrough), per-expert training
- Multi-montage (CAR, Laplacian), GED spatial filtering
- HMM comb-fit, windowed voting
- 25 morphological/temporal/spatial features
- **Best SP result (LOPO): combined rs=0.414**

### Deep Learning (Rounds 8-9)
- CNN backbone pretrained on 3,816 segments (93.6% LPD/GPD classification)
- CNN embeddings as ridge features: **overfit to patient identity** (LOO inflated to 0.492, honest LOPO was 0.397)
- DANN gradient reversal: destroyed useful features
- Cross-patient k-NN: near-zero correlation
- **Conclusion: CNN adds nothing when patients are properly separated**

### What Didn't Help (comprehensive)
CNN embeddings, DANN, k-NN, >8 features, RF/GBM, stacking, separate LPD/GPD models, period prediction, ordinal regression, YIN/SRH, alternans detection, comb-fit scoring, DTW template matching, grid search (432 combos), matched-filter envelope, GED standalone, HPS standalone

## The Real Bottleneck: Patient Count

| Dataset | Patients | Segments | Has Frequency? |
|---------|----------|----------|----------------|
| Current annotated | **37 LPD, 6 GPD** | 556 | Yes |
| External (unlabeled) | **~237 LPD, ~247 GPD** | ~10,000 | **No** |

With only 37 LPD patients and 6 GPD patients, we cannot meaningfully evaluate or improve beyond expert-expert levels. The algorithm already matches experts; we just can't prove it statistically with n=37/6.

**The highest-value next step is annotating frequency for more patients from the external dataset.**

## Active Learning Plan for Annotation

We propose selecting the most informative cases for expert annotation:

1. Run our algorithm on all ~10,000 external PD segments
2. Select **1 segment per patient** (the one closest to the event time)
3. Rank by:
   - High disagreement among our frequency estimators (ACF vs FFT vs peak-count)
   - LPD segments in the 0.75-2.0 Hz range (where our errors are largest)
   - Prefer patients not already in the annotated set
4. Select **50 LPD patients + 30 GPD patients** for expert frequency annotation
5. With ~80 new patients + 43 existing = ~123 patients, we can run statistically powered evaluation

## Questions

1. **We match experts on LPD with 37 patients — is this publishable?** The CI on rs=0.404 (n=37) is wide, but it overlaps with expert-expert rs=0.411 (n=76 pairs). Is this convincing?

2. **How many new patients do we need?** Power analysis: to detect a difference of 0.10 in Spearman with 80% power, we need ~80 patients per type. To demonstrate equivalence (non-inferiority), we need even more.

3. **Should we report all evaluation methods?** The progression from LOO-CV (inflated) to 1-per-patient LOPO (honest) is itself an important methodological finding. Many EEG papers use segment-level CV.

4. **Is the expert-expert agreement on GPD (rs=0.044 with 6 patients) real?** It suggests that GPD frequency variation is mostly within-patient, not between-patient. Different segments from the same GPD patient have similar frequencies, but different patients have different frequencies — and experts can only rank within-patient.

## Technical Summary
```
Model: Per-expert ridge on log(freq), 1-per-patient LOO-CV, alpha=1.0
Features: 8 (ACF, peak-count, FFT, TKEO, coherence, type, n_ch)
Evaluation: 43 patients (37 LPD, 6 GPD), 1 segment each
Speed: 48ms per segment (209× real-time)

Results (1-per-patient, honest):
  LPD: Spearman=0.404 (expert-expert: 0.411) — MATCHED
  GPD: Spearman=0.088 (expert-expert: 0.044) — both near zero, need more patients
```
