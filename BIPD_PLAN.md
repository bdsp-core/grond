# BIPD Detection Plan

## Problem Statement

Distinguish **BIPD** (bilateral independent periodic discharges) from **GPD** (generalized periodic discharges). Both show periodic discharges on both hemispheres, but:

- **GPD**: A single generator drives bilateral discharges. Left and right timing is phase-locked with only propagation delays (typically 10–50ms). Same frequency on both sides.
- **BIPD**: Two independent generators, one per hemisphere. Left and right have independent timing. Frequencies may differ. Phase relationship is unstable or absent.

## Data Challenge

Only **21 confirmed BIPD cases** from 198 candidates (11% yield from harvest). Far too few for supervised learning from raw EEG. The approach below solves this by operating on **timing sequences** rather than raw EEG, enabling unlimited synthetic training data.

## Architecture: Two-Stage Pipeline

### Stage 1: Per-Hemisphere Discharge Detection (existing pipeline)

Run the best discharge detector independently on each hemisphere:

```
EEG (18 channels) ──┬── Left hemisphere (channels 0-3, 8-11)  ──→ Detector ──→ {t_L1, t_L2, ...}, f_L
                     └── Right hemisphere (channels 4-7, 12-15) ──→ Detector ──→ {t_R1, t_R2, ...}, f_R
```

This uses the existing consolidated pipeline (product-boosted max(HPP,CET) + CNN+ACF freq + optimized DP) but applied to each hemisphere separately. The detector outputs:

- **Discharge times**: Ordered sequence of event times for each hemisphere
- **Frequency**: IPI-derived frequency per hemisphere
- **Evidence quality**: Peak evidence values (confidence proxy)

### Stage 2: BIPD vs GPD Classifier

**Input**: Two sequences of discharge times (left and right). Nothing else — no raw EEG.

**Output**: Probability of BIPD (vs GPD).

## Feature Engineering

From the two timing sequences T_L = {t_L1, t_L2, ...} and T_R = {t_R1, t_R2, ...}, compute:

### Frequency features
1. **f_L, f_R**: Median IPI frequency per hemisphere
2. **freq_ratio**: max(f_L, f_R) / min(f_L, f_R) — near 1.0 for GPD, can be >1 for BIPD
3. **freq_diff**: |f_L - f_R| in Hz
4. **log_freq_ratio**: log(freq_ratio) — more Gaussian for classification

### Phase relationship features
5. **nearest_delay_median**: For each L discharge, find nearest R discharge, take median of these delays
6. **nearest_delay_std**: Std of nearest-neighbor delays — low for GPD (stable propagation), high for BIPD (independent)
7. **nearest_delay_iqr**: IQR of nearest-neighbor delays (robust to outliers)
8. **nearest_delay_mad**: Median absolute deviation of delays
9. **phase_consistency**: Circular variance of the phase φ = (delay mod T) / T — near 0 for GPD (consistent phase), near 1 for BIPD (random phase)

### Cross-correlation features
10. **xcorr_peak**: Peak of cross-correlogram between L and R event trains (smoothed with Gaussian kernel σ=20ms)
11. **xcorr_peak_lag**: Lag at peak — small for GPD, variable for BIPD
12. **xcorr_ratio**: Peak / mean — high for GPD (strong periodicity in cross-correlation), low for BIPD

### Independence features
13. **ipi_correlation**: Spearman correlation between consecutive IPIs on L vs R (after alignment). High for GPD (shared modulation), low for BIPD.
14. **matched_fraction**: Fraction of L discharges that have a R discharge within ±50ms — high for GPD, can be low for BIPD
15. **unmatched_L, unmatched_R**: Fraction of discharges on each side with no partner within ±100ms

### Count features
16. **n_L, n_R**: Number of discharges per hemisphere
17. **count_ratio**: max(n_L, n_R) / min(n_L, n_R)
18. **total_discharges**: n_L + n_R

## Synthetic Training Data Generation

### Source data
- **GPD cases**: ~160 patients with bilateral discharge timing from per-hemisphere detection
- **LPD cases**: ~450 patients with unilateral discharge timing

### Negative examples (GPD-like)

1. **Real GPD pairs**: Run detector on both hemispheres of each GPD case. The L and R sequences will naturally be near-synchronous. ~160 examples.

2. **Duplicated LPD with jitter**: Take single-hemisphere LPD timing, duplicate as both L and R, add small Gaussian jitter (σ = 10–30ms, derived from real GPD propagation delays) to simulate bilateral propagation. ~450 examples.

3. **Real GPD with propagation delay variation**: Take GPD pairs, add systematic delay of 10–50ms to one side (simulating varying propagation). ~160 × 3 delay values = ~480 examples.

### Positive examples (BIPD-like)

4. **Cross-patient LPD pairs**: Randomly pair L timing from one LPD patient with R timing from a different LPD patient. This creates examples with different frequencies and independent timing. Sample ~1000 pairs.

5. **Phase-shifted GPD**: Take real GPD pairs, shift one side by Δt ~ Uniform(0.25s, 2.0s). This creates examples with same frequency but independent phase. ~160 × 5 shifts = ~800 examples.

6. **Frequency-scaled GPD**: Take real GPD pairs, scale one side's times by a factor (1.1–2.0× the IPI), creating frequency differences. ~160 × 3 factors = ~480 examples.

7. **Cross-patient LPD with similar frequencies**: Pair LPD cases that have similar frequencies (within 0.3 Hz) — the hardest case for the classifier (same freq, independent phase). ~200 examples.

### Jitter augmentation (applied to all examples)

For every synthetic example, add per-discharge jitter:
- Sample σ_jitter from the empirical distribution of timing noise in real data
- Estimate this from GPD cases: the std of (L_time - nearest_R_time) across all matched pairs gives the natural timing variability
- Apply: t' = t + N(0, σ_jitter) for each discharge independently
- Also randomly drop 0–15% of discharges from each side (simulating missed detections)

### Expected dataset size
- ~1000 negative (GPD-like) examples
- ~2500 positive (BIPD-like) examples
- Balance with oversampling or class weights

## Algorithm

### Recommended: Gradient Boosted Trees (XGBoost/LightGBM)

**Why:**
- Works well with ~18 handcrafted features
- Handles class imbalance (weight parameter)
- Interpretable (feature importances reveal what matters)
- Robust to synthetic data distribution shift
- No need for GPU or complex training
- Easy to validate with the 21 real BIPD cases

**Evaluation:**
- Train on synthetic data
- Validate on held-out synthetic data (to tune hyperparameters)
- Test on 21 real BIPD cases + ~160 real GPD cases (the ground truth)
- Primary metric: AUC, with attention to false positive rate at high recall

### Alternative: Small MLP on features

If GBT doesn't work well enough:
- Input: 18 features → 64 → 32 → 1 (sigmoid)
- Train with focal loss (class imbalance)
- Same features as above

### Not recommended (yet): Sequence model

A model that takes raw timing sequences (e.g., set transformer, RNN) could theoretically learn richer representations, but:
- Synthetic data may not capture the true distribution well enough
- Only 21 real positives makes validation unreliable
- Feature-based approach is more transparent and debuggable

Could revisit if more real BIPD cases are collected.

## Validation Strategy

### Primary validation: Leave-One-Out on real cases

For each of the 21 real BIPD cases:
1. Hold it out
2. Train on all synthetic data
3. Predict on the held-out BIPD case + all real GPD cases
4. Record: was the BIPD correctly identified?

This gives a realistic estimate of sensitivity.

### Secondary: Synthetic cross-validation

Standard 5-fold CV on synthetic data to tune hyperparameters and select features.

### Key metrics
- **Sensitivity on real BIPDs**: Out of 21, how many correctly identified?
- **Specificity on real GPDs**: Out of ~160, how many correctly classified as GPD?
- **AUC on real data**: Combined discrimination

### Success criteria
- Sensitivity ≥ 0.80 on real BIPDs (≥17/21 correct)
- Specificity ≥ 0.90 on real GPDs (≤16 false positives)
- If these aren't met, examine feature distributions of real vs synthetic to diagnose distribution shift

## Edge Cases to Consider

1. **GPD with slight asymmetry**: Some GPDs have slightly different amplitudes L vs R, which could cause the per-hemisphere detector to find slightly different numbers of discharges. Features should be robust to this.

2. **BIPD with similar frequencies**: When L and R have nearly the same frequency, the main distinguishing feature is phase independence. The phase_consistency feature is critical here.

3. **BIPD with intermittent activity**: One hemisphere may have gaps. The matched_fraction and unmatched features handle this.

4. **Very slow discharges (<0.5 Hz)**: Few discharges per 10s segment means fewer data points for statistics. Features may be noisy. Consider requiring minimum 3 discharges per hemisphere.

5. **Propagation delay mimicking phase offset**: In GPD, propagation delay is typically <50ms and consistent. In BIPD, "delay" is >100ms and variable. The delay_std feature distinguishes these.

## Implementation Order

1. **Run per-hemisphere detection** on all GPD and LPD cases → save L/R timing sequences
2. **Compute empirical jitter** from GPD cases (std of L-R delays)
3. **Generate synthetic dataset** (~3500 examples)
4. **Compute features** for all examples
5. **Train GBT classifier** with hyperparameter tuning
6. **Evaluate on 21 real BIPDs + ~160 real GPDs**
7. **Analyze errors** — which real cases are misclassified? Why?
8. **Iterate** on features or synthetic data generation if needed

## Dependencies

- Per-hemisphere discharge detector must work well independently (Stage 1 prerequisite)
- Need to run detector on both hemispheres of all GPD cases (not currently done)
- Need the 21 confirmed BIPD cases for validation (screening complete, timing labeling pending)

## Notes

- This approach scales naturally: as we collect more real BIPD cases, they can be added to validation
- The feature-based classifier is lightweight and interpretable — easy for clinicians to understand why a case was classified as BIPD
- The synthetic data approach means we can generate as many training examples as needed, varying difficulty
- Future work: if BIPD detection performance is insufficient, consider adding raw EEG features (e.g., morphology similarity between L and R discharges) as additional classifier inputs
