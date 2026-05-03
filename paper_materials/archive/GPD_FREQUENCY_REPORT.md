# GPD Frequency Estimation: Diagnostic Report

## Problem

PDProfiler achieves excellent frequency estimation for LPD (Spearman ρ = 0.677, MAE = 0.224 Hz) but performs poorly on GPD (ρ = 0.180, MAE = 0.441 Hz). GPD frequency estimation should be *easier* than LPD since the discharges are bilateral and typically more prominent.

## How the Current System Works

### PDProfiler Pipeline for Frequency Estimation

```
Input: 18-channel bipolar EEG (10 seconds @ 200 Hz)
   │
   ├─→ ChannelPD-Net (per-channel CNN+Attention)
   │     → 18 channel PD probabilities
   │     → 18 per-channel log-frequency estimates
   │     → Weighted average → CNN frequency estimate
   │
   ├─→ ACF frequency (per-channel autocorrelation on pointiness traces)
   │     → Median across channels → ACF frequency estimate
   │
   ├─→ Frequency prior: 0.8 × CNN_freq + 0.2 × ACF_freq
   │
   ├─→ Per-channel evidence:
   │     HPP (pointiness + TKEO) ─┐
   │     CET-UNet (learned CNN) ──┤→ Product-boosted combination
   │                               │
   │     CNN-weighted aggregation ─→ Combined evidence trace (2000 samples)
   │
   │     For GPD: weighted average across ALL 18 channels
   │     For LPD: weighted average across ipsilateral hemisphere only (8 channels)
   │
   ├─→ Dynamic Programming:
   │     Active interval detection → Candidate peaks → DP with periodic prior
   │     → EM template refinement → Post-hoc filtering
   │     → Discharge times: t₁, t₂, ..., tₖ
   │
   └─→ Frequency output: 1 / median(IPI)
         where IPI = inter-peak intervals = diff(discharge_times)
```

### What Goes Wrong for GPD

**Observation**: PDProfiler predicts ~1.35 Hz for 69% of GPD cases, regardless of actual frequency.

```
GPD Expert GT:    mean=0.98 Hz, std=0.31, range=[0.33, 2.45]
PDProfiler:  mean=1.39 Hz, std=0.23, range=[0.83, 2.60]  ← very narrow!
Alexandra's:      mean=1.00 Hz, std=0.43, range=[0.47, 3.48]  ← much wider
```

**Root cause analysis**:

1. **CNN frequency estimate is biased toward LPD frequencies**. The CNN was trained on:
   - 7,024 LPD channels (76%)
   - 978 GPD channels (10%)
   - 684 GRDA channels (7%)
   - 624 LRDA channels (7%)
   
   With 76% LPD training data, the CNN learned LPD frequency patterns and doesn't generalize to GPD. GPD discharges have different morphology (typically sharper, more synchronous across channels, often higher amplitude) than LPD.

2. **The frequency prior (0.8 × CNN + 0.2 × ACF) is dominated by the biased CNN**. Even when ACF gives a correct estimate, the 80/20 blend pulls it toward the CNN's ~1.35 Hz.

3. **Evidence aggregation across all 18 channels** may dilute the signal. For GPD, discharges are synchronous, so averaging should help — but if the evidence computation (pointiness + TKEO) doesn't capture GPD waveform morphology well, the aggregation amplifies noise.

4. **The DP periodicity prior uses the biased frequency estimate**, so it preferentially selects discharge sequences matching ~1.35 Hz, even when the actual frequency is 0.5 Hz or 2.0 Hz.

## Contest Results: 25 Approaches Tested

All methods keep LPD path unchanged; only modify GPD frequency estimation.

| Rank | Method | GPD ρ | GPD MAE | Description |
|------|--------|:-----:|:-------:|-------------|
| 1 | **B1_ACFPrior** | **0.353** | 0.330 | Pure ACF as freq prior (no CNN) |
| 2 | A2_HemiMore | 0.292 | 0.469 | DP on both hemispheres, take freq from hemi with more discharges |
| 3 | E1_AlexPrior | 0.269 | 0.321 | Alexandra's freq as DP prior |
| 4 | A4_HemiCNNWt | 0.269 | 0.432 | Both hemi DP, CNN-probability-weighted average freq |
| 5 | A1_HemiAvg | 0.268 | 0.432 | Both hemi DP, average freq |
| 6 | D2_Alpha20 | 0.264 | 0.368 | Stricter DP (α=2.0) |
| 7 | C1_MedianEv | 0.232 | 0.428 | Median evidence instead of weighted mean |
| ... | | | | |
| 14 | **Baseline** | **0.180** | **0.441** | Current PDProfiler |
| ... | | | | |
| 26 | B4_WelchPrior | 0.094 | 0.535 | Welch PSD peak as prior |

### Key findings:

1. **Replacing the CNN frequency prior with ACF doubles performance** (B1: ρ=0.353 vs baseline 0.180). This confirms the CNN is the bottleneck.

2. **Hemisphere-based strategies help moderately** (A1-A5: ρ=0.20-0.29). Running DP on each hemisphere independently and combining gives some improvement, suggesting the all-channel aggregation for GPD is suboptimal.

3. **Alexandra's frequency estimate as prior** (E1: ρ=0.269, MAE=0.321) gives the best MAE, suggesting her signal processing approach captures GPD frequency better than our CNN.

4. **Stricter DP periodicity** (D2: ρ=0.264) helps — the default α=1.275 may be too lenient, allowing the DP to select non-periodic sequences that don't reflect the true frequency.

5. **Welch PSD is terrible** (B4: ρ=0.094) — spectral peak frequency doesn't work for 10-second segments with transient patterns.

6. **Best GPD ρ = 0.353 is still much lower than LPD ρ = 0.677**. The gap is not fully closed by any post-hoc modification.

## Recommendations for Improvement

### Short-term (no retraining):
- **Adopt B1 (pure ACF prior)** for GPD cases → immediate improvement from ρ=0.180 to ρ=0.353
- Consider combining B1 + A2 (ACF prior + hemisphere strategy) for further gains

### Medium-term (retraining required):
- **Retrain ChannelPD-Net with balanced LPD/GPD data**. Currently 76% LPD / 10% GPD. Rebalancing to ~50/50 (or using class-weighted loss) should fix the CNN frequency bias.
- We have 259 GPD segments with expert frequency labels (86 with ≥3 raters) — enough for supervised frequency training.
- Critical: the multi-task loss `BCE(pd_prob) + α × MSE(log_freq)` should weight GPD frequency examples more heavily, or train a separate frequency head for GPD.

### Long-term:
- **GPD-specific evidence model**: GPD discharges have different morphology than LPD (sharper, bilateral). A CET-UNet trained on GPD discharge timing (which we have for 267 segments in discharge_times.json) could produce better evidence traces.
- **Frequency-aware DP**: Instead of using a single frequency prior, consider a multi-hypothesis approach that evaluates multiple candidate frequencies and selects the one with best DP score.

## Question: Would Retraining Help?

**Yes, almost certainly.** The evidence is strong:

1. The CNN frequency estimate is the primary bottleneck (replacing it with ACF doubles ρ)
2. The training data is severely imbalanced (76% LPD, 10% GPD)
3. We have adequate GPD frequency labels (259 segments, 86 with 3-rater consensus)
4. The architecture (CNN+Attention) is sound — it works excellently for LPD

**Proposed retraining approach**:
- Keep the same ChannelPD-Net architecture
- Use the existing v1 curated dataset (815 patients, 9310 channels)
- Add GPD frequency labels from all available raters (mean across LB, PH, SZ)
- Use class-weighted frequency loss: upweight GPD frequency examples 5-10×
- OR: train separate frequency heads for LPD and GPD
- Validate: LPD ρ must stay ≥ 0.65, GPD ρ target ≥ 0.50
