# Method 7: W05 Iterative Narrowband Refinement (RDA)

## Overview

We estimate the frequency and lateralization of rhythmic delta activity (LRDA/GRDA) using a two-pass iterative narrowband refinement procedure. The first pass applies a broad delta bandpass to obtain coarse estimates of dominant hemisphere and frequency via Hilbert-based instantaneous frequency analysis. The second pass re-filters the signal in a narrow band centered on the first-pass frequency estimate, yielding refined lateralization scores and a more precise frequency measurement. The method also classifies each segment as LRDA or GRDA based on an amplitude asymmetry index.

## Input

- 18-channel bipolar EEG, $\mathbf{x}(t) \in \mathbb{R}^{18}$, sampled at $f_s = 200\,\mathrm{Hz}$, with segment duration $T = 10\,\mathrm{s}$.
- Channels are partitioned into left-hemisphere set $\mathcal{L}$ and right-hemisphere set $\mathcal{R}$.

## Mathematical Formulation

### Pass 1: Coarse Estimation

#### Broadband Delta Filtering

Apply a bandpass filter at 0.5--3.5 Hz (3rd-order Butterworth, zero-phase) to all channels:

$$x_{\mathrm{broad}}^{(c)}(t) = \mathrm{BPF}_{[0.5,\, 3.5]}\!\left[x^{(c)}(t)\right], \quad c \in \{1,\ldots,18\} \tag{1}$$

#### Coarse Lateralization

Compute the mean variance for each hemisphere:

$$\sigma_L^2 = \frac{1}{|\mathcal{L}|} \sum_{c \in \mathcal{L}} \mathrm{var}\!\left(x_{\mathrm{broad}}^{(c)}\right), \qquad \sigma_R^2 = \frac{1}{|\mathcal{R}|} \sum_{c \in \mathcal{R}} \mathrm{var}\!\left(x_{\mathrm{broad}}^{(c)}\right) \tag{2}$$

The dominant hemisphere is:

$$\mathrm{dom} = \underset{h \in \{L, R\}}{\operatorname{argmax}}\; \sigma_h^2 \tag{3}$$

Select the top-3 channels by variance from the dominant hemisphere: $\{c_1, c_2, c_3\} \subset \mathcal{H}_{\mathrm{dom}}$.

#### Hilbert-Based Instantaneous Frequency

For each selected channel $c_i$, compute the analytic signal:

$$z^{(c_i)}(t) = x_{\mathrm{broad}}^{(c_i)}(t) + j \cdot \mathcal{H}\!\left\{x_{\mathrm{broad}}^{(c_i)}\right\}(t) \tag{4}$$

where $\mathcal{H}\{\cdot\}$ denotes the Hilbert transform. The instantaneous phase and frequency are:

$$\phi^{(c_i)}(t) = \arg\!\left(z^{(c_i)}(t)\right) \tag{5}$$

$$f^{(c_i)}(t) = \frac{f_s}{2\pi} \cdot \Delta\!\left(\mathrm{unwrap}\!\left(\phi^{(c_i)}\right)\right)(t) \tag{6}$$

where $\Delta$ denotes the first-order backward difference operator. The coarse frequency estimate is obtained by pooling valid instantaneous frequency samples across the three channels:

$$\hat{f}_{\mathrm{coarse}} = \underset{t,\, i}{\mathrm{median}}\;\left\{f^{(c_i)}(t) : f^{(c_i)}(t) \in [0.3,\; 4.0]\,\mathrm{Hz}\right\} \tag{7}$$

### Pass 2: Narrowband Refinement

#### Narrowband Filtering

Re-filter all channels using a narrow bandpass centered on the coarse estimate:

$$x_{\mathrm{narrow}}^{(c)}(t) = \mathrm{BPF}_{[\hat{f}_{\mathrm{coarse}} - 0.4,\;\; \hat{f}_{\mathrm{coarse}} + 0.4]}\!\left[x^{(c)}(t)\right] \tag{8}$$

#### Refined Lateralization

Compute the analytic signal for each narrowband channel and extract the instantaneous envelope:

$$\mathrm{env}^{(c)}(t) = \left|z_{\mathrm{narrow}}^{(c)}(t)\right| = \left|x_{\mathrm{narrow}}^{(c)}(t) + j \cdot \mathcal{H}\!\left\{x_{\mathrm{narrow}}^{(c)}\right\}(t)\right| \tag{9}$$

Hemisphere amplitude scores:

$$L_{\mathrm{score}} = \frac{1}{|\mathcal{L}|} \sum_{c \in \mathcal{L}} \overline{\mathrm{env}^{(c)}}, \qquad R_{\mathrm{score}} = \frac{1}{|\mathcal{R}|} \sum_{c \in \mathcal{R}} \overline{\mathrm{env}^{(c)}} \tag{10}$$

where $\overline{\mathrm{env}^{(c)}} = \frac{1}{T} \int_0^T \mathrm{env}^{(c)}(t)\, dt$ is the temporal mean envelope amplitude.

The refined dominant hemisphere is $\mathrm{dom}' = \operatorname{argmax}_{h}(L_{\mathrm{score}}, R_{\mathrm{score}})$.

#### Refined Frequency

Select the top-3 narrowband channels by variance on the refined dominant hemisphere. Repeat the Hilbert instantaneous frequency estimation (Eqs. 4--7) on these narrowband signals to obtain:

$$\hat{f}_{\mathrm{refined}} = \underset{t,\, i}{\mathrm{median}}\;\left\{f_{\mathrm{narrow}}^{(c_i)}(t) : f_{\mathrm{narrow}}^{(c_i)}(t) \in [0.3,\; 4.0]\,\mathrm{Hz}\right\} \tag{11}$$

#### LRDA vs. GRDA Classification

Compute the asymmetry index:

$$A = \frac{|L_{\mathrm{score}} - R_{\mathrm{score}}|}{L_{\mathrm{score}} + R_{\mathrm{score}}} \tag{12}$$

The subtype classification is:

$$\mathrm{subtype} = \begin{cases} \text{LRDA} & \text{if } A > \theta_A \\ \text{GRDA} & \text{otherwise} \end{cases} \tag{13}$$

where $\theta_A$ is a threshold calibrated against expert consensus labels.

## Algorithm

```
W05_RDA(x[c,t], f_s=200):
    # --- PASS 1: Coarse ---
    1. x_broad[c,t] = BPF(x[c,t], [0.5, 3.5] Hz)           [Eq. (1)]
    2. Compute sigma_L^2, sigma_R^2                           [Eq. (2)]
    3. dom = argmax(sigma_L^2, sigma_R^2)                     [Eq. (3)]
    4. Select top-3 channels {c1,c2,c3} by variance on dom
    5. For each c_i:
         z(t) = x_broad(t) + j*Hilbert(x_broad(t))           [Eq. (4)]
         phi(t) = arg(z(t))                                   [Eq. (5)]
         f(t) = (f_s/2pi) * diff(unwrap(phi))                 [Eq. (6)]
    6. f_coarse = median({f(t) : f(t) in [0.3, 4.0]})        [Eq. (7)]

    # --- PASS 2: Narrowband ---
    7. x_narrow[c,t] = BPF(x[c,t], [f_coarse-0.4, f_coarse+0.4])  [Eq. (8)]
    8. env[c](t) = |hilbert(x_narrow[c,t])|                   [Eq. (9)]
    9. L_score = mean(env_left), R_score = mean(env_right)     [Eq. (10)]
   10. dom' = argmax(L_score, R_score)
   11. Select top-3 narrowband channels on dom'
   12. f_refined = Hilbert freq on narrowband top-3             [Eq. (11)]
   13. A = |L - R| / (L + R)                                   [Eq. (12)]
   14. subtype = LRDA if A > theta_A else GRDA                 [Eq. (13)]

    return subtype, laterality(dom'), f_refined
```

## Output

- **Subtype**: LRDA or GRDA
- **Laterality**: left or right dominant hemisphere (for LRDA)
- **Frequency**: refined estimate $\hat{f}_{\mathrm{refined}}$ in Hz

## Performance Note

The two-pass approach improves frequency estimation accuracy by 15--20% relative to single-pass broadband Hilbert analysis, as measured by Spearman correlation against expert-labeled frequencies. The narrowband filtering in Pass 2 suppresses harmonic contamination and adjacent-band activity that bias instantaneous frequency estimates in the broadband case. In the V5 lateralization contest (76 methods), W05 achieved the best unified weighted AUC of 0.837 with frequency Spearman $\rho = 0.635$.
