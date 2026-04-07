# Hybrid-PLV: CNN-Guided Phase-Locking Value for Spatial Localization

## Overview

Spatial localization of periodic discharges requires identifying which brain regions participate in the discharge pattern.
We employ a hybrid approach that combines per-channel PD probabilities from ChannelPD-Net with phase-locking value (PLV) analysis to produce region-level involvement scores.
The CNN selects reference channels most likely to contain discharges, and PLV quantifies phase coherence between each channel and this reference, identifying regions that are synchronously involved in the periodic pattern.

## Reference Channel Selection

The top-3 channels by ChannelPD-Net probability are selected as references.
For LPD segments with known laterality, contralateral channels are suppressed before ranking to ensure the reference is seeded from the correct hemisphere:

$$\tilde{p}_c = \begin{cases} p_c & \text{if } c \in \mathcal{H}_{\mathrm{ipsi}} \text{ or subtype is GPD} \\ 0 & \text{if } c \in \mathcal{H}_{\mathrm{contra}} \text{ and subtype is LPD} \end{cases} \tag{1}$$

$$\mathcal{R} = \operatorname{argtop\text{-}3}(\tilde{p}_c) \tag{2}$$

where $\mathcal{H}_{\mathrm{ipsi}}$ and $\mathcal{H}_{\mathrm{contra}}$ are the ipsilateral and contralateral hemisphere channel index sets, and $\mathcal{R}$ is the set of three reference channels.

## Bandpass Filtering

All 18 channels are bandpass-filtered to the periodic discharge frequency range using a 4th-order Butterworth filter:

$$x_c^{\mathrm{bp}}(t) = \mathrm{BPF}_{[0.5,\, 3.5]\,\text{Hz}}\!\big(x_c(t)\big) \tag{3}$$

This isolates the slow periodic component while rejecting faster EEG rhythms and DC drift.

## Phase Extraction via Hilbert Transform

The instantaneous phase of each channel is obtained from the analytic signal:

$$z_c(t) = x_c^{\mathrm{bp}}(t) + j \cdot \mathcal{H}\!\big\{x_c^{\mathrm{bp}}(t)\big\} \tag{4}$$

$$\varphi_c(t) = \arg\!\big(z_c(t)\big) \in (-\pi, \pi] \tag{5}$$

where $\mathcal{H}\{\cdot\}$ denotes the Hilbert transform.

## Reference Phase Construction

The reference phase is constructed as the circular mean of the three reference channel phases:

$$\varphi_{\mathrm{ref}}(t) = \arg\!\left(\sum_{k \in \mathcal{R}} \exp\!\big(j \cdot \varphi_k(t)\big)\right) \tag{6}$$

This produces a robust reference that is less sensitive to noise in any single channel than using a single reference.

## Per-Channel Phase-Locking Value

The PLV between each channel and the reference quantifies the consistency of their phase relationship across time:

$$\mathrm{PLV}_c = \left|\frac{1}{N} \sum_{t=1}^{N} \exp\!\Big(j \cdot \big(\varphi_c(t) - \varphi_{\mathrm{ref}}(t)\big)\Big)\right| \in [0, 1] \tag{7}$$

where $N = 2000$ is the number of samples.
$\mathrm{PLV}_c = 1$ indicates perfect phase synchrony (constant phase difference), while $\mathrm{PLV}_c \approx 0$ indicates no consistent phase relationship.

## Combined Channel Score

The CNN probability and PLV are combined with equal weight to produce a per-channel involvement score:

$$s_c = 0.5 \cdot p_c + 0.5 \cdot \mathrm{PLV}_c \tag{8}$$

This combination leverages the CNN's ability to detect discharge morphology (even without phase information) and the PLV's sensitivity to temporal synchrony (even in channels with atypical morphology).

## Region Mapping

The 18 bipolar channels are mapped to 8 anatomical regions corresponding to the standard double-banana montage:

| Region | Abbreviation | Channel Indices |
|--------|-------------|-----------------|
| Left Frontal | LF | Fp1-F7 (0), Fp1-F3 (8) |
| Right Frontal | RF | Fp2-F8 (4), Fp2-F4 (12) |
| Left Temporal | LT | F7-T3 (1), T3-T5 (2) |
| Right Temporal | RT | F8-T4 (5), T4-T6 (6) |
| Left Centro-Parietal | LCP | F3-C3 (9), C3-P3 (10) |
| Right Centro-Parietal | RCP | F4-C4 (13), C4-P4 (14) |
| Left Occipital | LO | T5-O1 (3), P3-O1 (11) |
| Right Occipital | RO | T6-O2 (7), P4-O2 (15) |

The region score is the maximum channel score among channels belonging to that region:

$$S_r = \max_{c \in \mathcal{C}_r} s_c, \quad r \in \{\text{LF, RF, LT, RT, LCP, RCP, LO, RO}\} \tag{9}$$

where $\mathcal{C}_r$ is the set of channel indices mapped to region $r$.
Taking the maximum (rather than mean) ensures that a region is considered involved if any of its constituent channels shows strong evidence.

## Binary Involvement and Spatial Extent

Each region is classified as involved if its score exceeds a fixed threshold:

$$\mathrm{involved}(r) = \mathbb{1}[S_r > \theta], \quad \theta = 0.4 \tag{10}$$

The spatial extent of the discharge pattern is the fraction of regions involved:

$$\mathrm{extent} = \frac{\sum_{r=1}^{8} \mathbb{1}[S_r > \theta]}{8} \in [0, 1] \tag{11}$$

## Algorithm Pseudocode

```
Algorithm: Hybrid-PLV Spatial Localization
Input: segment_18ch (18 x 2000), channel_probs p (18,), laterality
Output: region_scores (8,), involved_regions, spatial_extent

  // 1. Reference selection
  if LPD and laterality known:
      suppress contralateral channel probs to 0
  ref_channels = argsort(p, descending)[:3]

  // 2. Bandpass filter 0.5-3.5 Hz
  For each channel c = 1, ..., 18:
      x_bp[c] = butterworth_bandpass(segment_18ch[c], 0.5, 3.5 Hz)

  // 3. Phase extraction
  For each channel c = 1, ..., 18:
      phi[c] = angle(hilbert(x_bp[c]))

  // 4. Reference phase (circular mean of top-3)
  phi_ref = angle( sum_{k in ref_channels} exp(j * phi[k]) )

  // 5. PLV computation
  For each channel c = 1, ..., 18:
      PLV[c] = |mean_t( exp(j * (phi[c] - phi_ref)) )|
      s[c] = 0.5 * p[c] + 0.5 * PLV[c]

  // 6. Region mapping
  For each region r in {LF, RF, LT, RT, LCP, RCP, LO, RO}:
      S[r] = max(s[c] for c in region_channels[r])
      involved[r] = (S[r] > 0.4)

  // 7. Spatial extent
  extent = sum(involved) / 8

  return S, involved, extent
```

## Design Rationale

Phase-locking value was selected from a 26-method spatial localization contest (including amplitude correlation, coherence, Granger causality, CSD analysis, and others) as the best-performing single spatial metric.
The hybrid combination with CNN probabilities outperforms either component alone: the CNN provides morphology-based evidence that is robust to phase noise, while PLV captures temporal synchrony patterns that the CNN's channel-independent architecture cannot directly model.

The 0.5/0.5 weighting was determined empirically.
The threshold $\theta = 0.4$ was chosen to balance sensitivity and specificity for region involvement classification.

## Performance

The Hybrid-PLV method achieves a composite spatial localization score of approximately 0.79 and a region-level AUC of approximately 0.71, evaluated against expert spatial annotations.
PLV alone achieves AUC approximately 0.68, while CNN probabilities alone achieve AUC approximately 0.65, confirming the complementary value of the two information sources.
