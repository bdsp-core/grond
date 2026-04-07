# Method 8: RDA-PLV Spatial Extent

## Overview

We quantify the spatial extent of rhythmic delta activity (RDA) using a phase-locking value (PLV) metric weighted by narrowband amplitude. Each bipolar channel is compared to a reference signal derived from the dominant-hemisphere channels in the narrowband around the estimated RDA frequency. Channels whose PLV-amplitude product exceeds a threshold are classified as "involved," and spatial extent is defined as the fraction of involved channels. This approach captures both phase coherence (indicating true rhythmic coupling) and amplitude (indicating signal strength), avoiding false positives from low-amplitude channels with spuriously high PLV.

## Input

- 18-channel bipolar EEG, $\mathbf{x}(t) \in \mathbb{R}^{18}$
- Estimated RDA frequency $\hat{f}\,(\mathrm{Hz})$ from the W05 pipeline (Method 7)
- Dominant hemisphere and top-3 channels from W05

## Mathematical Formulation

### Step 1: Narrowband Filtering

Apply a 3rd-order Butterworth bandpass filter centered on the estimated frequency:

$$x_{\mathrm{nb}}^{(c)}(t) = \mathrm{BPF}_{[\hat{f} - 0.4,\;\; \hat{f} + 0.4]}^{(\mathrm{Butter},\, 3)}\!\left[x^{(c)}(t)\right], \quad c \in \{1,\ldots,18\} \tag{1}$$

### Step 2: Reference Signal Construction

The reference signal is the arithmetic mean of the narrowband-filtered top-3 channels on the dominant hemisphere, $\{c_1, c_2, c_3\}$:

$$x_{\mathrm{ref}}(t) = \frac{1}{3} \sum_{i=1}^{3} x_{\mathrm{nb}}^{(c_i)}(t) \tag{2}$$

### Step 3: Phase Extraction

Compute the instantaneous phase for each channel and the reference via the Hilbert transform:

$$\phi^{(c)}(t) = \arg\!\left(x_{\mathrm{nb}}^{(c)}(t) + j \cdot \mathcal{H}\!\left\{x_{\mathrm{nb}}^{(c)}\right\}(t)\right) \tag{3}$$

$$\phi_{\mathrm{ref}}(t) = \arg\!\left(x_{\mathrm{ref}}(t) + j \cdot \mathcal{H}\!\left\{x_{\mathrm{ref}}\right\}(t)\right) \tag{4}$$

### Step 4: Phase-Locking Value

The PLV for each channel quantifies the consistency of the phase difference relative to the reference over $N$ time samples:

$$\mathrm{PLV}_c = \left|\frac{1}{N} \sum_{t=1}^{N} \exp\!\left(j \cdot \left[\phi^{(c)}(t) - \phi_{\mathrm{ref}}(t)\right]\right)\right| \tag{5}$$

$\mathrm{PLV}_c \in [0,1]$, where 1 indicates perfect phase locking and 0 indicates uniformly distributed phase differences.

### Step 5: Amplitude Ratio

Compute the mean envelope amplitude per channel and normalize by the maximum across channels:

$$\bar{a}_c = \frac{1}{N} \sum_{t=1}^{N} \left|x_{\mathrm{nb}}^{(c)}(t) + j \cdot \mathcal{H}\!\left\{x_{\mathrm{nb}}^{(c)}\right\}(t)\right| \tag{6}$$

$$a_c = \frac{\bar{a}_c}{\max_{c'} \bar{a}_{c'}} \tag{7}$$

### Step 6: Combined Score

The per-channel involvement score is the product of PLV and normalized amplitude:

$$s_c = \mathrm{PLV}_c \cdot a_c \tag{8}$$

This product penalizes channels that have high PLV but negligible amplitude (noise-driven phase locking) and channels with high amplitude but poor phase coherence (non-rhythmic activity).

### Step 7: Variance Explained

As an auxiliary quality metric, fit a sinusoidal model to each narrowband channel:

$$\hat{x}^{(c)}(t) = A_c \sin(2\pi \hat{f} \cdot t + \phi_{0,c}) + B_c \tag{9}$$

where $A_c$, $\phi_{0,c}$, and $B_c$ are estimated via least squares. The variance explained is:

$$\mathrm{VE}_c = 1 - \frac{\mathrm{var}\!\left(x_{\mathrm{nb}}^{(c)} - \hat{x}^{(c)}\right)}{\mathrm{var}\!\left(x_{\mathrm{nb}}^{(c)}\right)} \tag{10}$$

### Step 8: Thresholding and Spatial Extent

A channel is classified as involved if its combined score exceeds a fixed threshold:

$$\mathcal{I} = \left\{c : s_c > \theta_s\right\}, \quad \theta_s = 0.62 \tag{11}$$

The spatial extent is:

$$\mathrm{SE} = \frac{|\mathcal{I}|}{18} \tag{12}$$

where the denominator is the total number of bipolar channels.

## Algorithm

```
RDA_PLV(x[c,t], f_hat, top3_channels):
    1. x_nb[c,t] = BPF(x[c,t], [f_hat-0.4, f_hat+0.4], order=3)   [Eq. (1)]
    2. x_ref(t) = mean(x_nb[top3_channels, t])                      [Eq. (2)]
    3. For each channel c = 1 to 18:
         phi_c(t) = arg(hilbert(x_nb[c,t]))                         [Eq. (3)]
         phi_ref(t) = arg(hilbert(x_ref(t)))                        [Eq. (4)]
         PLV_c = |mean(exp(j*(phi_c - phi_ref)))|                   [Eq. (5)]
         a_c = mean(|hilbert(x_nb[c,t])|)                           [Eq. (6)]
    4. Normalize: a_c = a_c / max(a_c)                               [Eq. (7)]
    5. s_c = PLV_c * a_c                                             [Eq. (8)]
    6. (Optional) Fit sinusoid, compute VE_c                      [Eqs. (9-10)]
    7. Involved = {c : s_c > 0.62}                                  [Eq. (11)]
    8. SE = |Involved| / 18                                          [Eq. (12)]
    return SE, {s_c}, {PLV_c}, {VE_c}
```

## Output

- **Spatial extent**: fraction of channels involved, $\mathrm{SE} \in [0, 1]$
- **Per-channel scores**: $\{s_c\}$, $\{\mathrm{PLV}_c\}$, $\{a_c\}$, $\{\mathrm{VE}_c\}$

## Performance Note

The threshold $\theta_s = 0.62$ was calibrated to maximize agreement with expert spatial extent ratings. The automated spatial extent estimates achieve an intraclass correlation coefficient (ICC) of 0.371 against expert consensus, compared to an inter-rater ICC of 0.373 between independent expert raters. This near-parity with human agreement indicates that the PLV-amplitude method captures the same information experts use when judging RDA spatial extent, while acknowledging that spatial extent is inherently a low-reliability measure even among trained electroencephalographers.
