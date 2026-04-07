# Method 9: BIPD Detection

## Overview

Bilateral independent periodic discharges (BIPD) are defined by the presence of independent periodic discharge trains on each hemisphere, with distinct timing sequences that are not phase-locked to one another. We detect BIPD using a two-stage approach: (1) independent per-hemisphere discharge detection using the HemiCET+DP pipeline, and (2) gradient-boosted tree (GBT) classification of the resulting bilateral timing sequences. The classifier distinguishes BIPD (independent bilateral timing) from GPD (synchronous bilateral discharges) based on features capturing frequency asymmetry, temporal independence, and phase consistency.

## Input

- 18-channel bipolar EEG partitioned into left-hemisphere channels $\mathcal{L}$ ($|\mathcal{L}| = 8$, plus 2 midline) and right-hemisphere channels $\mathcal{R}$ ($|\mathcal{R}| = 8$, plus 2 midline)

## Mathematical Formulation

### Stage 1: Per-Hemisphere Discharge Detection

Run the HemiCET+DP pipeline independently on each hemisphere, yielding two discharge time sequences:

$$\text{Left:}\quad \{t_1^L, t_2^L, \ldots, t_m^L\} \tag{1}$$

$$\text{Right:}\quad \{t_1^R, t_2^R, \ldots, t_n^R\} \tag{2}$$

The inter-pulse intervals (IPIs) for each hemisphere are:

$$\mathrm{IPI}_i^L = t_{i+1}^L - t_i^L, \quad i = 1,\ldots,m-1 \tag{3}$$

$$\mathrm{IPI}_j^R = t_{j+1}^R - t_j^R, \quad j = 1,\ldots,n-1 \tag{4}$$

### Stage 2: Feature Extraction

The following features are computed from the bilateral timing sequences:

#### Feature 1: Frequency Ratio

Per-hemisphere frequency is estimated from the median IPI:

$$f_L = \frac{1}{\mathrm{median}(\mathrm{IPI}^L)}, \qquad f_R = \frac{1}{\mathrm{median}(\mathrm{IPI}^R)} \tag{5}$$

$$r_f = \frac{f_L}{f_R} \tag{6}$$

For BIPD, the frequency ratio typically deviates from unity since the two hemispheres generate discharges at independent rates.

#### Feature 2: Regularity (Coefficient of Variation)

$$\mathrm{CoV}_L = \frac{\mathrm{std}(\mathrm{IPI}^L)}{\mathrm{mean}(\mathrm{IPI}^L)}, \qquad \mathrm{CoV}_R = \frac{\mathrm{std}(\mathrm{IPI}^R)}{\mathrm{mean}(\mathrm{IPI}^R)} \tag{7}$$

Low CoV indicates regular periodicity; BIPD hemispheres may differ in regularity.

#### Feature 3: Matched Fraction

Quantifies the temporal overlap between left and right discharge times using a 100 ms coincidence window:

$$\mathrm{MF} = \frac{\left|\left\{(i,j) : |t_i^L - t_j^R| < 100\,\mathrm{ms}\right\}\right|}{\max(m, n)} \tag{8}$$

For GPD (synchronous), $\mathrm{MF} \approx 1$; for BIPD (independent), $\mathrm{MF}$ is low and determined by chance coincidence.

#### Feature 4: Phase Consistency

Compute the nearest-neighbor phase for each left discharge relative to the right hemisphere cycle:

$$\psi_i = 2\pi \cdot \frac{t_i^L - t_{j^*(i)}^R}{\mathrm{IPI}_{j^*(i)}^R} \tag{9}$$

where $j^*(i) = \operatorname{argmin}_j |t_i^L - t_j^R|$ subject to $t_j^R \leq t_i^L$. Phase consistency is quantified by the mean resultant length:

$$\mathrm{PC} = \left|\frac{1}{m} \sum_{i=1}^{m} \exp(j \psi_i)\right| \tag{10}$$

For GPD, $\mathrm{PC} \approx 1$ (fixed phase relationship); for BIPD, $\mathrm{PC} \approx 0$ (uniformly distributed phases).

#### Features 5--7: Asymmetry Measures

$$\text{Sequence length ratio:} \quad r_n = \frac{\min(m,n)}{\max(m,n)} \tag{11}$$

$$\text{Frequency asymmetry:} \quad A_f = \frac{|f_L - f_R|}{f_L + f_R} \tag{12}$$

$$\text{Count asymmetry:} \quad A_n = \frac{|m - n|}{m + n} \tag{13}$$

### Stage 3: GBT Classification

The feature vector $\mathbf{x} = [r_f,\; \mathrm{CoV}_L,\; \mathrm{CoV}_R,\; \mathrm{MF},\; \mathrm{PC},\; r_n,\; A_f,\; A_n]^{\!\top} \in \mathbb{R}^8$ is input to a gradient-boosted tree classifier:

$$P(\text{BIPD} \mid \mathbf{x}) = \sigma\!\left(\sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})\right) \tag{14}$$

where $\sigma$ is the logistic function, $h_m$ are regression trees, $\eta$ is the learning rate, and $M$ is the number of boosting rounds.

### Training Data Generation

The GBT is trained on synthetic data constructed from real single-hemisphere discharge sequences:

- **Synthetic BIPD**: Draw one LPD sequence from Patient A (left) and an independent LPD sequence from Patient B (right). The cross-patient pairing ensures genuinely independent timing.
- **Synthetic GPD**: Take a single LPD sequence and create a bilateral version by phase-shifting: $t_j^R = t_j^L + \delta$ where $\delta \sim \mathcal{U}(-25\,\mathrm{ms},\; 25\,\mathrm{ms})$. This produces synchronized bilateral discharges with slight jitter.

## Algorithm

```
BIPD_DETECT(x_left[c,t], x_right[c,t]):
    # --- Stage 1: Per-hemisphere detection ---
    1. {t^L_1,...,t^L_m} = HemiCET_DP(x_left)        [Eq. (1)]
    2. {t^R_1,...,t^R_n} = HemiCET_DP(x_right)        [Eq. (2)]

    # --- Stage 2: Feature extraction ---
    3. IPI^L, IPI^R from discharge times               [Eqs. (3-4)]
    4. f_L, f_R = 1/median(IPI^L), 1/median(IPI^R)    [Eq. (5)]
    5. r_f = f_L / f_R                                 [Eq. (6)]
    6. CoV_L, CoV_R = std/mean of IPIs                 [Eq. (7)]
    7. MF = matched fraction with 100ms window          [Eq. (8)]
    8. PC = phase consistency (mean resultant length)    [Eqs. (9-10)]
    9. r_n, A_f, A_n = asymmetry features               [Eqs. (11-13)]

    # --- Stage 3: Classification ---
   10. x = [r_f, CoV_L, CoV_R, MF, PC, r_n, A_f, A_n]
   11. P(BIPD) = GBT.predict_proba(x)                   [Eq. (14)]

    return P(BIPD), {t^L}, {t^R}
```

## Output

- $P(\text{BIPD})$: probability of bilateral independent periodic discharges
- Per-hemisphere discharge times for downstream analysis

## Performance Note

The BIPD classifier achieves an AUC of 0.840 on held-out data, with 63% sensitivity at the operating threshold. From 198 candidate segments flagged by the detector, 21 were confirmed as true BIPD by expert review. The most discriminative features are matched fraction ($\mathrm{MF}$) and phase consistency ($\mathrm{PC}$), which together capture the defining characteristic of BIPD: temporal independence between hemispheres. The synthetic training strategy avoids the need for large annotated BIPD datasets, which are rare in clinical practice.
