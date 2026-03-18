# Frequency Estimation for Periodic and Rhythmic EEG Patterns: Approach Review

## Problem Statement

We are building algorithms to automatically estimate the **frequency** (in Hz) of periodic and rhythmic patterns observed in 10-second EEG recordings from critically ill patients. These patterns include:

- **LPD** (lateralized periodic discharges) — sharp discharges repeating periodically on one side of the brain
- **GPD** (generalized periodic discharges) — sharp discharges repeating periodically across the whole brain
- **LRDA** (lateralized rhythmic delta activity) — rhythmic slow waves on one side
- **GRDA** (generalized rhythmic delta activity) — rhythmic slow waves across the whole brain

Clinically, the frequency matters because higher frequency periodic discharges (>1.5 Hz) are associated with increased seizure risk and worse outcomes.

**We can assume the pattern type (LPD/GPD/LRDA/GRDA) is known** — in practice, an independent classifier (deep neural network) provides this. Our task is purely frequency estimation.

## Data

### EEG Segments
- **556 annotated segments** (260 LPD, 296 GPD) — each is 10 seconds of 19-channel EEG at 200 Hz
- Re-referenced to **18 bipolar channels** (longitudinal bipolar montage)
- Preprocessing: 60 Hz notch filter, 0.5-40 Hz bandpass

### Expert Annotations
- **3 expert neurophysiologists** (LB, PH, SZ) independently annotated each segment
- Each expert provided: frequency (Hz), spatial extent (fraction of channels involved)
- Expert consensus = median of experts who rated frequency > 0

### Expert Agreement (our ceiling to match):

| Pair | LPD Spearman r | LPD MAE | GPD Spearman r | GPD MAE |
|------|---------------|---------|---------------|---------|
| LB vs PH | 0.76 | 0.24 Hz | 0.46 | 0.19 Hz |
| LB vs SZ | 0.61 | 0.34 Hz | 0.61 | 0.18 Hz |
| PH vs SZ | 0.57 | 0.45 Hz | 0.61 | 0.22 Hz |

Expert frequencies range from 0.25 to 4.0 Hz, with most between 0.5 and 2.0 Hz.

## Approaches Tried

### Method A: Adaptive Peak Detection (PD2a — baseline from published paper)
- Detrend + Savitzky-Golay smooth the raw EEG signal
- Compute first derivative
- Adaptive threshold (local mean + 3*local_std in sliding 1s window)
- Detect peaks exceeding threshold
- Filter by regularity (std of inter-peak intervals < 1s)
- Frequency = mean(1/intervals), median across channels
- **Results: LPD Spearman=0.23, GPD Spearman=-0.15, Combined MAE=0.406**

### Method B: Pointiness + ACF
- Compute "pointiness" feature per sample: prominence²/width at each local max
- Smooth with Gaussian (sigma=0.02s)
- Apply 15 Hz lowpass before feature extraction (reduces frontal artifact)
- Autocorrelate the smoothed pointiness trace
- Find first ACF peak after min_lag → frequency = fs/lag
- Frequency = median across channels where ACF found periodicity
- **Best variant (ACF threshold=0.10): LPD Spearman=0.29, GPD Spearman=0.45, Combined MAE=0.312**

### Variants explored:
- **Log compression** of pointiness before ACF — hurt L/G classification
- **Percentile normalization** (divide by running 95th percentile) — marginal improvement
- **Peak equalization** (binary 0/1 at peak locations, smooth, ACF) — Spearman 0.32
- **Inter-peak interval histogram** — bypasses ACF, Spearman ~0.20 alone
- **Subharmonic correction** (check ACF at L/2, L/3, L/4) — modest improvement
- **Channel aggregation** variants (weighted mean by ACF score, best channel, mode binning)
- **Ensemble of A+B** (mean, max, pattern-specific weighting)
- **Bayesian nudge** (0.7*algorithm + 0.3*prior where LPD prior=1.25, GPD prior=1.0)
- **Triple median** (median of Method A, Method B, peak-count frequency) — best signal processing approach

### Current best signal processing result:
**Triple median + Bayesian nudge: Combined Spearman ~0.36, Combined MAE ~0.30**

- GPD MAE (0.20) is at expert-expert level
- LPD Spearman (~0.36) is still well below expert-expert (0.57-0.76)

## Dominant Failure Mode

**73% of frequency estimation failures are due to subharmonic locking** — the ACF picks a peak at 2x, 3x, or 4x the true period, returning freq/2, freq/3, or freq/4.

- 93% of errors are underestimates
- Failure rate for expert freq >1.25 Hz: 73%
- Failure rate for expert freq >2.0 Hz: 79%
- LPD failure rate (53%) is much higher than GPD (30%)

The pointiness trace often has alternating strong/weak discharges, causing the ACF to lock onto the every-other-discharge period rather than the fundamental.

## What We're Trying Next

1. **FFT / Cepstral analysis** of the pointiness trace — cepstrum is specifically designed to separate fundamentals from harmonics
2. **ML regression** (Random Forest / GBM with leave-one-out CV) using features extracted from multiple signal processing methods
3. **Deep parameter grid search** (432 combinations) over the triple-median approach

## Questions for Review

1. **Are there better approaches for periodic frequency estimation from noisy, non-stationary signals?** We've tried ACF, direct peak counting, inter-peak intervals, and are now trying FFT/cepstral. What else?

2. **The subharmonic locking problem**: The ACF of the pointiness trace often has its strongest peak at 2x the true period. Standard ACF doesn't distinguish fundamentals from subharmonics. The cepstrum should help — are there other methods?

3. **Is there a better feature than "pointiness" (prominence²/width)?** We also tried |d²x/dt²| (second derivative) which performed similarly. The pointiness trace is sparse (nonzero only at local maxima), which may contribute to subharmonic issues.

4. **For the ML approach**: with 556 samples and ~15 features, what models/regularization would you recommend? We're planning Ridge, Random Forest, and Gradient Boosting with LOO-CV. Anything else?

5. **Should we reconsider the problem formulation?** Currently we estimate frequency per-channel then aggregate (median/weighted mean). Would it be better to work on the multi-channel signal jointly? For example, spatial filtering (like a beamformer) to create one optimally-weighted signal, then estimate frequency from that?

6. **The LPD problem**: LPD frequencies are more variable (0.5-2.5 Hz) and discharges have more complex morphology (polyphasic, triphasic). Experts agree better on LPD frequency (Spearman 0.57-0.76) than our algorithm does (0.23-0.36). What could experts be doing differently that we're missing?

7. **Any completely different paradigms?** For example:
   - Fitting a parametric model (e.g., spike + slow wave template at unknown frequency)?
   - Phase-locked loop / adaptive frequency tracking?
   - Time-frequency analysis (STFT, wavelets) to handle non-stationarity?
   - Bayesian inference with a generative model of periodic discharges?

## Repository Structure

```
code/
  pd_detector_alternate/pd_detect_alternate.py  — Method A (PD2a)
  pd_pointiness_acf.py                          — Method B (pointiness + ACF)
  optimization_harness.py                       — evaluation framework
  reproduce_paper_figures.py                    — ICC/PA analysis
data/
  dataset_eeg/lpd/, gpd/                        — 556 .mat files (HDF5 format)
  annotations/                                  — expert CSV annotations (LB, PH, SZ)
results/
  optimization_runs/                            — JSON results from experiments
  optimization_dashboard.html                   — live dashboard
```

## Key Constraints
- 10-second segments (limited data per segment)
- 200 Hz sampling rate
- 18 bipolar channels
- Frequency range: 0.3-3.5 Hz (known a priori)
- Pattern type is known
- Must work on single segments (no temporal context from adjacent segments)
