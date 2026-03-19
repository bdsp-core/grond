# Frequency Estimation for Periodic EEG Patterns: Updated Review (v2)

## Problem Statement

We are estimating the **frequency** (Hz) of periodic discharges (LPD, GPD) in 10-second, 18-channel bipolar EEG recordings from critically ill patients. The pattern type is known (from an independent classifier). Our goal is to match or beat expert-level inter-rater agreement on frequency estimation.

## Data

- **556 annotated segments** (260 LPD, 296 GPD), each 10s at 200 Hz, 18 bipolar channels
- **3 expert neurophysiologists** (LB, PH, SZ) independently annotated frequency and spatial extent
- Expert frequencies: discrete values (0.25, 0.5, 0.75, 1.0, 1.25, ... 3.0+ Hz)

## Expert-Expert Agreement (our target to match)

**Pooled across all 3 pairwise comparisons:**

| | LPD Spearman | LPD MAE | GPD Spearman | GPD MAE |
|---|---|---|---|---|
| All 3 pairs pooled | 0.525 | 0.325 Hz | 0.476 | 0.191 Hz |
| LB vs PH | 0.76 | 0.24 | 0.46 | 0.19 |
| LB vs SZ | 0.61 | 0.34 | 0.61 | 0.18 |
| PH vs SZ | 0.57 | 0.45 | 0.61 | 0.22 |
| LOO (expert vs mean of other 2) | 0.74 | 0.30 | 0.66 | 0.16 |

## Current Best Results (after ~100 experiments across 5 rounds)

| Method | LPD rs | GPD rs | Mean rs | LPD MAE | GPD MAE |
|--------|--------|--------|---------|---------|---------|
| Expert-Expert (pooled) | **0.525** | **0.476** | **0.50** | 0.325 | 0.191 |
| **Our best (ridge on log-freq)** | 0.377 | **0.489** | **0.418** | 0.379 | **0.180** |
| Best LPD-specific method | **0.425** | 0.232 | 0.328 | 0.380 | 0.254 |
| Method A baseline (PD2a) | 0.234 | -0.145 | 0.045 | 0.537 | 0.274 |

**GPD: We match/beat experts** (rs=0.489 vs 0.476, MAE=0.180 vs 0.191).
**LPD: Significant gap remains** (rs=0.377-0.425 vs 0.525). This is our main challenge.

## What We Tried (5 rounds, ~100 experiments)

### Round 1: Signal Processing Approaches
| Approach | Best rs | Verdict |
|----------|---------|---------|
| ACF of pointiness trace (original Method B) | 0.14 | Subharmonic locking dominates |
| ACF with lower threshold (0.10 vs 0.20) | 0.29 | **Big improvement** — detects more channels |
| Direct peak-count frequency | 0.21 | Noisy alone, good in combination |
| Average of ACF + peak-count | 0.32 | **Key finding**: averaging corrects subharmonics |
| Ensemble: pick higher of A and B | 0.36 | Confirms underestimation is the main error |
| Bayesian nudge (0.7×algo + 0.3×prior) | 0.36 | Helps MAE but not correlation much |
| Subharmonic correction (check L/2, L/3, L/4) | 0.45 (MAE) | Modest improvement |
| Channel aggregation (weighted mean by ACF score) | 0.42 (MAE) | Better than median |
| Log compression before ACF | worse | Amplifies noise on quiet channels |
| Percentile normalization | worse | Same problem |
| Peak equalization (binary peaks → ACF) | 0.32 | Decent but not better than simpler methods |
| Inter-peak interval histogram | 0.20 | Noisy; useful as additional estimate |

### Round 2: Combination Strategies
| Approach | Best rs | Verdict |
|----------|---------|---------|
| Triple median (Method A, ACF, peak-count) | 0.36 | Best non-ML signal processing approach |
| Quad/five-way median | 0.38 | More estimates = more robust |
| Type-specific routing (different method per LPD/GPD) | 0.34 | Didn't help as much as expected |

### Round 3: New Frequency Estimators + ML
| Approach | Best rs | Verdict |
|----------|---------|---------|
| **FFT of pointiness trace** | **0.38** | **Better than ACF** — naturally avoids subharmonics |
| FFT + peak-count average | **0.375** | Best single signal processing method |
| Cepstral analysis | -0.01 | **Failed** — signals too short for reliable cepstrum |
| Random Forest regression (LOO-CV) | 0.36 | Didn't beat simple FFT+peaks |
| Ridge regression on log(freq) | **0.415** | **Breakthrough** — learned optimal combination |

### Round 4: Template Matching + Advanced Methods
| Approach | Best rs | Verdict |
|----------|---------|---------|
| Matched-filter discharge envelope (8 templates) | 0.33 LPD | Helps LPD, hurts GPD |
| YIN/CMNDF on matched-filter envelope | failed | Envelope too noisy for YIN |
| SRH (subharmonic ratio) scoring | 0.24 | Worse than FFT |
| Cross-channel salience aggregation | 0.26 | Theory good, execution poor |
| SRH on cross-channel salience | 0.04 | **Failed** — collapsed to noise |
| Alternans detection + correction | 0.15 | Too aggressive (55% false positive rate) |
| Ridge with envelope FFT as feature | **0.415** | Envelope adds marginal value |
| Type-specific (best-for-LPD + best-for-GPD) | 0.34 | Ridge implicitly does this better |

### Round 5: Feature Engineering + Target Transforms
| Approach | Best rs | Verdict |
|----------|---------|---------|
| 25 rich features (morphological, temporal, spatial) | 0.417 | **No improvement** — features don't help |
| More templates (16, 24, PCA) | 0.413 | Marginal |
| Predict period (1/f) instead of frequency | 0.36 | **Worse** than log-freq |
| Predict sqrt(freq) | 0.40 | Nearly as good as log-freq |
| Ordinal regression (match expert bins) | 0.415 | Same ceiling |
| Snap predictions to expert grid | 0.39 rs, **0.273 MAE** | Best MAE but not correlation |
| Stacking (two-level ridge) | 0.41 | No improvement |
| Take-max for LPD (assume underestimate) | 0.37 | **Hurt** — not all errors are underestimates |
| Consistency weighting | **0.418** | Tiny improvement, same ceiling |
| Separate LPD/GPD models | 0.40 | Worse (smaller training sets) |

## What We've Learned

### 1. The dominant failure mode is subharmonic locking (73% of errors)
The ACF picks peaks at 2×, 3×, 4× the true period, returning freq/2, freq/3, etc. This affects LPD much more than GPD.

### 2. FFT > ACF for frequency estimation
FFT of the pointiness trace finds the spectral peak directly in [0.3, 3.5] Hz, naturally avoiding subharmonics. This was the single biggest algorithmic improvement.

### 3. Log-frequency is the right prediction target
Errors are multiplicative (algorithm says 0.5 when truth is 1.0), so log-transform linearizes them. Ridge regression on log(freq) was our breakthrough.

### 4. The optimal method is ridge regression on 5 diverse frequency estimates
Features: Method A freq, ACF freq (thr=0.10), peak-count freq, FFT-of-pointiness freq, matched-filter-envelope freq. Trained on log(expert_freq) with LOO-CV. Coefficients: ACF=0.32, PeakCount=0.28, Envelope=0.19, FFT=0.17, MethodA=0.16.

### 5. We've hit a ceiling at Spearman ~0.42 with current features
25 additional features (morphological, temporal, spatial), better templates (8→24), different target transforms (period, sqrt, ordinal), stacking, type-specific models — none broke through. The bottleneck is the frequency estimates themselves, not how we combine them.

### 6. GPD is solved; LPD is the remaining challenge
- GPD rs=0.489 matches expert-expert 0.476 ✓
- LPD rs=0.377-0.425 vs expert 0.525 — 72-81% of expert level

### 7. What experts see that our algorithm misses
Experts recognize complete discharge complexes (sharp wave + after-going slow wave = one cycle) regardless of amplitude variation. Our features (pointiness, d²x/dt², FFT) all struggle with:
- Polyphasic/triphasic LPD morphology (multiple peaks per complex)
- Amplitude alternation (strong/weak/strong → ACF doubles the period)
- Complex background activity that contaminates feature traces

## Your Previous Advice and What Happened

You recommended (in priority order):

1. **Replace pointiness with discharge-likelihood (matched filter/template bank)** — We built both data-driven (k-means on extracted snippets) and synthetic template banks. The matched-filter envelope helped LPD slightly (rs=0.334 as standalone) and adds marginal value as a ridge feature, but didn't transform results. The envelope may not be clean enough — we used simple cross-correlation with 8-24 templates.

2. **Anti-subharmonic period search (YIN/CMNDF + SRH)** — YIN failed completely (envelope too noisy for CMNDF threshold). SRH gave rs=0.24 — worse than plain FFT. The 10-second segments may be too short for these methods to work reliably.

3. **Aggregate salience across channels before deciding frequency (GED/DSS spatial filtering)** — We tried per-channel salience → weighted sum → argmax. Results were poor (rs=0.26). We did NOT try GED/DSS spatial filtering — this remains untested.

4. **Alternans model** — Our alternans detector fired on 55% of segments (too aggressive), and correction hurt performance. We didn't implement a proper alternans state in a generative model.

5. **Stacking model over candidates and confidences** — We did this (two-level ridge stacking). Results: rs=0.41, same as single-level ridge. The candidates themselves are the bottleneck.

## The Remaining Gap

**LPD Spearman: 0.377-0.425 vs expert 0.525**

The gap appears to be in the **raw frequency estimation for LPD**, not in the post-processing. All our frequency estimators (ACF, FFT, peak-count, matched-filter, Method A) produce LPD frequency estimates that correlate with expert consensus at rs=0.19-0.43. No amount of combining them gets above ~0.42.

Possible explanations:
- LPD complexes have more diverse morphology → our feature traces miss or double-count components
- LPD often has irregular intervals → 10 seconds may have only 5-10 cycles, with significant jitter
- Experts may use spatial context (which channels show the discharge) to interpret ambiguous morphology
- Some LPD segments may be near the boundary of expert agreement anyway (expert rs=0.525, not 0.90)

## Questions for Your Updated Advice

1. **Given that matched-filter + YIN/SRH didn't work well on these short signals, what would you suggest as the next approach for the discharge-likelihood envelope?** Should we try a learned convolutional filter instead of template cross-correlation? Or a different envelope extraction approach?

2. **GED/DSS spatial filtering is the one recommendation we haven't tried.** Do you think spatial filtering to create 1-3 optimally weighted channels could help? The idea of computing the frequency estimate from a spatially optimized signal rather than per-channel-then-median is appealing.

3. **Is there a simpler anti-subharmonic method than YIN/SRH?** Something that works on short (10s, 5-10 cycle) signals? Maybe just: "if FFT peak at f and also peak at 2f, prefer 2f"?

4. **Should we try deep learning?** With 556 segments we could train a small CNN on the raw EEG (or on the pointiness traces) to directly predict frequency. Data augmentation (time-shifting, adding noise) could help. Or would you expect this to overfit?

5. **Is the remaining LPD gap (~0.42 vs 0.53) actually closeable with signal processing?** Or is this the point where we need fundamentally different input (longer segments, different features, temporal context from adjacent segments)?

6. **Any quick wins we're missing?** We've done 100 experiments and hit a wall. Sometimes a fresh pair of eyes sees something obvious.

## Technical Details

### Frequency Estimators (5 currently used as ridge features)
1. **Method A (PD2a):** Adaptive threshold on first derivative → peak detection → mean(1/intervals)
2. **ACF (thr=0.10):** Pointiness trace → 15Hz lowpass → smooth → ACF → first peak after 0.4s
3. **Peak-count:** Pointiness peaks (height>30% max, distance>0.2s) → (n-1)/time_span
4. **FFT of pointiness:** Power spectrum peak in [0.3, 3.5] Hz
5. **Matched-filter envelope FFT:** Cross-correlate with template bank → max envelope → FFT peak

### Best Model
Ridge regression (alpha=1.0) on log(expert_freq), LOO-CV, 7 features (5 freqs + pattern_type + n_channels). Predictions = exp(ridge_output), clamped to [0.2, 4.0] Hz.

### Repository
```
code/
  pd_detector_alternate/pd_detect_alternate.py  — Method A
  pd_pointiness_acf.py                          — Method B (pointiness + ACF)
  optimization_harness.py                       — evaluation framework
  build_template_banks.py                       — template bank construction
data/
  dataset_eeg/lpd/, gpd/                        — 556 .mat files
  annotations/                                  — expert CSVs
  templates_C_lpd.npy, templates_C_gpd.npy      — data-driven templates
  templates_D.npy                               — synthetic templates
results/
  optimization_runs/                            — ~100 experiment JSONs
  optimization_dashboard.html                   — live dashboard
```
