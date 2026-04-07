# Method 10: Evidence Combination (Product-Boosted Max)

## Overview

We combine two complementary evidence signals for discharge detection: a handcrafted heuristic peak-pointiness (HPP) trace and a learned convolutional evidence trace (CET) from the HemiCET-UNet ensemble. HPP excels at detecting sharp electrographic transients via morphological features, while CET captures complex learned discharge patterns. The two are fused using a product-boosted maximum operator, which selects the stronger signal at each timepoint while amplifying regions where both methods agree via a quadratic interaction term. This combination yields a single evidence trace suitable for downstream dynamic programming (DP) path optimization.

## Input

- Single-hemisphere EEG (8 bipolar channels), 10 s at 200 Hz

## Mathematical Formulation

### HPP Evidence

#### Pointiness Trace

For each local maximum at time $t_p$ in the 20 Hz lowpass-filtered signal, compute the pointiness score from its prominence $\rho(t_p)$ and half-prominence width $w(t_p)$:

$$p(t_p) = \frac{\rho(t_p)^2}{w(t_p)} \tag{1}$$

The pointiness trace $P(t)$ is constructed by placing $p(t_p)$ at each peak location and zero elsewhere, then taking the channel-wise maximum across all 8 channels.

#### Teager-Kaiser Energy Operator (TKEO)

The TKEO captures instantaneous energy and is sensitive to rapid amplitude and frequency changes. Applied to the 20 Hz lowpass signal $x(t)$:

$$\mathrm{tkeo}[n] = \left|x[n]^2 - x[n-1] \cdot x[n+1]\right| \tag{2}$$

The TKEO trace is the channel-wise maximum across all 8 channels.

#### HPP Combination

The two components are z-scored and combined as a weighted sum:

$$E_{\mathrm{hpp}}(t) = 0.6 \cdot \tilde{P}(t) + 0.4 \cdot \widetilde{\mathrm{tkeo}}(t) \tag{3}$$

where $\tilde{\cdot}$ denotes z-scoring (zero mean, unit variance). The combined trace is smoothed with a Gaussian kernel ($\sigma = 3$ samples, i.e., 15 ms at 200 Hz):

$$E_{\mathrm{hpp}}(t) \leftarrow (E_{\mathrm{hpp}} * g_\sigma)(t), \quad g_\sigma(t) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{t^2}{2\sigma^2}\right) \tag{4}$$

### CET Evidence

The CET evidence $E_{\mathrm{cet}}(t)$ is the output of the HemiCET-UNet ensemble, a U-Net trained to predict discharge probability at each timepoint. The ensemble averages predictions from multiple folds for robustness.

### Preprocessing

Both evidence traces are normalized to $[0,1]$:

$$E_{\mathrm{hpp}}(t) \leftarrow \frac{E_{\mathrm{hpp}}(t) - \min(E_{\mathrm{hpp}})}{\max(E_{\mathrm{hpp}}) - \min(E_{\mathrm{hpp}})} \tag{5}$$

$$E_{\mathrm{cet}}(t) \leftarrow \frac{E_{\mathrm{cet}}(t) - \min(E_{\mathrm{cet}})}{\max(E_{\mathrm{cet}}) - \min(E_{\mathrm{cet}})} \tag{6}$$

The CET trace is then thresholded at the 80th percentile of its non-zero values to suppress low-confidence predictions, and a hard floor is applied:

$$\tau_{80} = \mathrm{percentile}_{80}\!\left(\{E_{\mathrm{cet}}(t) : E_{\mathrm{cet}}(t) > 0\}\right) \tag{7}$$

$$E_{\mathrm{cet}}(t) \leftarrow \begin{cases} E_{\mathrm{cet}}(t) & \text{if } E_{\mathrm{cet}}(t) \geq \max(\tau_{80},\; 0.3) \\ 0 & \text{otherwise} \end{cases} \tag{8}$$

### Product-Boosted Max Combination

The final evidence trace is:

$$E(t) = \max\!\left(E_{\mathrm{hpp}}(t),\; E_{\mathrm{cet}}(t)\right) + \lambda \cdot E_{\mathrm{hpp}}(t) \cdot E_{\mathrm{cet}}(t) \tag{9}$$

where $\lambda = 3.0$ is an empirically optimized boost coefficient.

The two terms serve distinct roles:

1. **Max term**: At each timepoint, the stronger evidence source is selected. This ensures that a discharge detected by either method alone is preserved in the combined trace, preventing loss of sensitivity.

2. **Product term**: The multiplicative interaction $E_{\mathrm{hpp}}(t) \cdot E_{\mathrm{cet}}(t)$ is nonzero only when both methods provide positive evidence at the same timepoint. Since both inputs are in $[0,1]$, the product is also in $[0,1]$ and is large only when both are simultaneously large. The coefficient $\lambda = 3.0$ amplifies this agreement signal, creating a strong synergistic boost.

The product term effectively implements a soft AND gate: timepoints where both HPP and CET agree receive substantially higher evidence than those supported by only one method. This is motivated by the observation that true discharges activate both the morphological (sharp transient) and learned (contextual pattern) detectors, while artifacts tend to trigger only one.

## Algorithm

```
EVIDENCE_COMBINATION(eeg[c,t]):
    # --- HPP Evidence ---
    1. x_lp = lowpass(eeg, 20 Hz)
    2. For each channel c:
         Find local maxima {t_p}
         Compute pointiness: p(t_p) = prominence(t_p)^2 / width(t_p)     [Eq. (1)]
         Compute TKEO: tkeo[n] = |x[n]^2 - x[n-1]*x[n+1]|               [Eq. (2)]
    3. P(t) = max over channels of pointiness traces
    4. tkeo(t) = max over channels of TKEO traces
    5. E_hpp = 0.6*zscore(P) + 0.4*zscore(tkeo)                          [Eq. (3)]
    6. E_hpp = gaussian_smooth(E_hpp, sigma=3)                            [Eq. (4)]

    # --- CET Evidence ---
    7. E_cet = HemiCET_UNet_ensemble(eeg)

    # --- Preprocessing ---
    8. Normalize E_hpp, E_cet to [0, 1]                                [Eqs. (5-6)]
    9. Threshold E_cet at max(80th percentile, 0.3)                    [Eqs. (7-8)]

    # --- Combination ---
   10. E(t) = max(E_hpp(t), E_cet(t)) + 3.0 * E_hpp(t) * E_cet(t)      [Eq. (9)]

    return E(t)
```

## Output

- Combined evidence trace $E(t) \in \mathbb{R}^+$, used as input to the dynamic programming path optimizer for discharge time estimation.

## Performance Note

The product-boosted max combination outperforms both individual evidence sources and simpler fusion strategies (e.g., weighted sum, max alone). The boost coefficient $\lambda = 3.0$ was selected via grid search over $\lambda \in \{0.5, 1.0, 2.0, 3.0, 4.0, 5.0\}$, optimizing discharge detection F1 score on a held-out validation set. The key advantage is that the method preserves the sensitivity of either individual detector (via the max term) while substantially boosting precision in regions of agreement (via the product term), yielding a favorable precision-recall tradeoff for downstream periodic discharge characterization.
