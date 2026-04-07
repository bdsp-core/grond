# CNN+ACF Frequency Ensemble

## Overview

Accurate estimation of the discharge repetition frequency is critical both as a clinical descriptor and as the periodic prior $T = 1/f$ that governs the dynamic programming discharge detector.
We combine two complementary frequency estimates -- one learned (CNN) and one handcrafted (autocorrelation of the pointiness trace) -- into a robust ensemble that serves as the frequency prior for downstream processing.

## CNN Frequency Estimate

### Per-Channel Prediction

Each of the 18 bipolar channels is processed independently through the 5-fold ChannelPD-Net ensemble (see ChannelPD-Net method), yielding per-channel PD probability $p_c$ and log-frequency prediction $\hat{f}_{\log,c}$.

### PD-Weighted Aggregation

The patient-level CNN frequency is a PD-weighted average in log-space, which down-weights channels lacking periodic patterns:

$$f_{\mathrm{CNN}} = \exp\!\left(\frac{\sum_{c=1}^{C} p_c \cdot \hat{f}_{\log,c}}{\sum_{c=1}^{C} p_c}\right) \tag{1}$$

where $C = 18$ is the number of bipolar channels.
If $\sum_c p_c < 10^{-6}$, the unweighted mean is used as a fallback.
The estimate is clipped to the physiological range $[0.3, 3.5]$ Hz.

**Rationale for log-space averaging.** Discharge frequencies span nearly an order of magnitude (0.3--3.5 Hz).
Averaging in log-space prevents high-frequency outliers from dominating and provides a more symmetric error distribution.

## ACF Frequency Estimate

### Per-Channel Autocorrelation

For each channel, we compute the autocorrelation of the pointiness trace $\mathrm{pt}(t)$ (see Dynamic Programming method, Eq. 3) via the Wiener-Khinchin theorem:

$$R(\tau) = \mathcal{F}^{-1}\!\Big\{\big|\mathcal{F}\{\mathrm{pt}(t) - \bar{\mathrm{pt}}\}\big|^2\Big\} \tag{2}$$

The autocorrelation is normalized by its zero-lag value:

$$\hat{R}(\tau) = \frac{R(\tau)}{R(0)} \tag{3}$$

so that $\hat{R}(0) = 1$.

### Peak Detection

The first prominent peak in the normalized ACF within the physiologically plausible lag range indicates the dominant repetition period:

$$\tau^* = \arg\max_{\tau \in [\tau_{\min},\, \tau_{\max}]} \hat{R}(\tau), \quad \text{subject to } \hat{R}(\tau^*) > 0.1 \tag{4}$$

where $\tau_{\min} = \lfloor F_s / 3.5 \rfloor$ samples (maximum 3.5 Hz) and $\tau_{\max} = \lfloor F_s / 0.33 \rfloor$ samples (minimum 0.33 Hz).
If multiple peaks exist, the shortest-lag peak above threshold is selected to avoid subharmonic aliasing.

The per-channel ACF frequency is:

$$f_{\mathrm{ACF}}^{(c)} = \frac{F_s}{\tau^*_c} \tag{5}$$

### Multi-Channel Aggregation

The patient-level ACF frequency is the median of valid per-channel estimates:

$$f_{\mathrm{ACF}} = \mathrm{median}\!\big(\{f_{\mathrm{ACF}}^{(c)} : f_{\mathrm{ACF}}^{(c)} \text{ is finite}\}\big) \tag{6}$$

The median provides robustness against channels with artifact or absent periodic patterns.
If no channel yields a valid ACF peak, $f_{\mathrm{ACF}}$ is treated as missing.

## Ensemble Combination

The final frequency prior combines the two estimates with fixed weights favoring the CNN:

$$f_{\mathrm{prior}} = \begin{cases} 0.8 \cdot f_{\mathrm{CNN}} + 0.2 \cdot f_{\mathrm{ACF}} & \text{if } f_{\mathrm{ACF}} \text{ is finite} \\ f_{\mathrm{CNN}} & \text{otherwise} \end{cases} \tag{7}$$

The combined estimate is clipped to the physiological range:

$$f_{\mathrm{prior}} = \mathrm{clip}(f_{\mathrm{prior}},\; 0.3,\; 3.5) \tag{8}$$

This $f_{\mathrm{prior}}$ determines the expected period $T = 1 / f_{\mathrm{prior}}$ used as the periodic prior in the DP algorithm (see Dynamic Programming method, Eq. 8).

## Algorithm Pseudocode

```
Algorithm: CNN+ACF Frequency Ensemble
Input: segment_18ch (18 x 2000), subtype, laterality
Output: f_prior (Hz)

  // CNN frequency
  For each channel c = 1, ..., 18:
      z-score normalize channel c
      For each fold k = 1, ..., 5:
          (p_c^k, f_log_c^k, _) = ChannelPDNet_k(channel_c)
      p_c = mean(p_c^1, ..., p_c^5)
      f_log_c = mean(f_log_c^1, ..., f_log_c^5)
  f_CNN = exp( sum(p_c * f_log_c) / sum(p_c) )
  f_CNN = clip(f_CNN, 0.3, 3.5)

  // ACF frequency
  acf_freqs = []
  For each channel c = 1, ..., 18:
      pt_c = pointiness_trace(lowpass_20Hz(channel_c))
      R_c = IFFT(|FFT(pt_c - mean(pt_c))|^2)
      R_c = R_c / R_c[0]
      Find first peak tau* in R_c[Fs/3.5 : Fs/0.33] with height > 0.1
      if peak found:
          acf_freqs.append(Fs / tau*)
  f_ACF = median(acf_freqs) if len(acf_freqs) > 0 else NaN

  // Ensemble
  if f_ACF is finite:
      f_prior = 0.8 * f_CNN + 0.2 * f_ACF
  else:
      f_prior = f_CNN
  f_prior = clip(f_prior, 0.3, 3.5)
  return f_prior
```

## Design Rationale

The 80/20 weighting reflects empirical findings that the CNN frequency estimate is more accurate on average (having been trained on expert labels), while the ACF provides a useful corrective signal in cases where the CNN mispredicts -- particularly for segments with unusual morphology or low PD probability across channels.
The ACF is parameter-free and does not require labeled data, making it a complementary regularizer.

## Performance

| Method | Spearman $\rho$ vs. Expert |
|--------|---------------------------|
| CNN alone | ~0.55 |
| ACF alone | ~0.40 |
| CNN+ACF ensemble (0.8/0.2) | ~0.60 |
| Full pipeline (IPI from DP) | ~0.65 |

The CNN+ACF ensemble provides the prior that seeds the DP algorithm; the final IPI-derived frequency from the DP output further improves correlation with expert ratings by leveraging the detected discharge times directly.
