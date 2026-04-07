# Dynamic Programming for Periodic Discharge Detection

## Overview

We detect individual periodic discharges within each 10-second EEG segment using a Viterbi-style forward dynamic programming algorithm with an approximately-periodic prior.
The algorithm receives a combined evidence trace $E(t)$ (from handcrafted and learned sources), identifies candidate discharge peaks, and selects the optimal approximately-periodic subsequence that balances evidence strength against temporal regularity.
Post-processing includes EM-style template refinement and confidence-based filtering.

## Handcrafted Evidence: HPP

The handcrafted periodic pattern (HPP) evidence trace is computed per-channel from two complementary signal features.

### Pointiness Trace

For each local maximum at sample $n$ in the (optionally lowpass-filtered) signal $x$, we compute:

$$\mathrm{prom}(n) = x(n) - \max\!\big(\min_{j \in [n-w, n)} x(j),\;\; \min_{j \in (n, n+w]} x(j)\big) \tag{1}$$

$$\mathrm{width}(n) = \big|\{j : |j - n| \leq w,\; x(j) > x(n) - \tfrac{1}{2}\,\mathrm{prom}(n)\}\big| \tag{2}$$

$$\mathrm{pt}(n) = \frac{\mathrm{prom}(n)^2}{\mathrm{width}(n)} \tag{3}$$

where $w = 8$ samples (40 ms) is the half-window for valley search, and $\mathrm{pt}(n) = 0$ at non-peak locations.
Pointiness captures the sharpness of transient waveforms: narrow, prominent peaks (characteristic of discharges) receive high scores.

### Teager-Kaiser Energy Operator (TKEO)

The TKEO is applied to the 20 Hz lowpass-filtered signal $x_{\mathrm{lp}}$:

$$\mathrm{tkeo}[n] = \big|x_{\mathrm{lp}}[n]^2 - x_{\mathrm{lp}}[n-1] \cdot x_{\mathrm{lp}}[n+1]\big| \tag{4}$$

TKEO responds to instantaneous energy and frequency, producing transient peaks at discharge locations.

### Evidence Combination

Both traces are z-scored within the 10-second window and combined with fixed weights:

$$E_{\mathrm{hpp}}(t) = \max\!\Big(0,\;\; G_{\sigma} * \big[0.6 \cdot \tilde{\mathrm{pt}}(t) + 0.4 \cdot \widetilde{\mathrm{tkeo}}(t)\big]\Big) \tag{5}$$

where $\tilde{\cdot}$ denotes z-scoring, $G_{\sigma}$ is a Gaussian kernel with $\sigma = 3$ samples (15 ms), and $*$ denotes convolution.
Negative values are clipped to zero.

## Neural Evidence: CET-UNet

Per-channel CET evidence $E_{\mathrm{cet}}(t)$ is obtained from the HemiCET-UNet ensemble (see HemiCET-UNet method).

## Evidence Aggregation and Combination

### Channel Aggregation

Per-channel evidence is aggregated to a single trace depending on subtype:

- **GPD:** median across all 18 channels
- **LPD with known laterality:** median across the 8 ipsilateral channels
- **LPD without laterality:** $\max\!\big(\mathrm{median}_{\mathcal{L}},\; \mathrm{median}_{\mathcal{R}}\big)$ pointwise

### Product-Boosted Combination

The aggregated HPP and CET evidence traces are combined using a product-boost formula that amplifies regions where both sources agree:

$$E(t) = \max\!\big(\hat{E}_{\mathrm{hpp}}(t),\; \hat{E}_{\mathrm{cet}}(t)\big) + \beta_{\mathrm{boost}} \cdot \hat{E}_{\mathrm{hpp}}(t) \cdot \hat{E}_{\mathrm{cet}}(t) \tag{6}$$

where $\hat{E}_{\mathrm{hpp}}$ and $\hat{E}_{\mathrm{cet}}$ are normalized to $[0, 1]$, the CET trace is thresholded at the 80th percentile of nonzero values (to suppress noise floor) and floored at 0.3, and $\beta_{\mathrm{boost}} = 3.0$.

## Active Interval Detection

Before candidate extraction, we identify the longest contiguous interval of sustained discharge activity:

1. Compute a rolling mean of $E(t)$ with a 1-second window ($w = 200$ samples)
2. Threshold at 50% of the global rolling maximum
3. Find the longest contiguous above-threshold run
4. Expand by 0.5 s on each side; require minimum 3 s duration (else use full segment)

## Candidate Extraction

Candidate discharge peaks are extracted from $E(t)$ within the active interval via two passes:

1. **Standard pass:** all local maxima with height $> 0.05 \cdot \max(E)$ and minimum inter-peak distance $0.2 \cdot T$ samples
2. **Strong pass:** all local maxima with height $> 0.50 \cdot \max(E)$ and minimum distance $0.1 \cdot T$ samples

Both sets are merged and deduplicated, yielding candidate set $\mathcal{S} = \{c_1, c_2, \ldots, c_n\}$ ordered by time.

## Dynamic Programming Formulation

### State Space and Scoring

Let $T = 1 / f_{\mathrm{est}}$ be the expected discharge period (from the CNN+ACF frequency ensemble).
For each candidate $c \in \mathcal{S}$, define:

**Node score** (superlinear reward minus existence cost):

$$R(c) = E(c)^{1.5} - \lambda \tag{7}$$

The exponent 1.5 provides superlinear scaling that disproportionately favors strong evidence peaks over marginal ones.

**Edge score** (transition from $c_i$ to $c_j$, allowing up to $M = 3$ skipped periods):

$$Q(c_i, c_j) = \max_{m \in \{1, 2, 3\}} \left[-\alpha \cdot \left(\frac{\Delta t - m \cdot T}{m \cdot T}\right)^{\!2} - \beta \cdot (m - 1)\right] \tag{8}$$

where $\Delta t = (c_j - c_i) / F_s$ is the inter-candidate interval in seconds.
The first term penalizes deviations from an integer multiple of the expected period (quadratic in fractional deviation).
The second term penalizes skips: $m = 1$ incurs no penalty, $m = 2$ costs $\beta$, and $m = 3$ costs $2\beta$.
The best skip count $m$ is chosen greedily.

### Recurrence

The optimal score at candidate $c_j$ is:

$$V(c_j) = \max\!\bigg(R(c_j),\;\; \max_{c_i < c_j,\; \Delta t \leq 4T} \big[V(c_i) + Q(c_i, c_j) + R(c_j)\big]\bigg) \tag{9}$$

with the first case allowing $c_j$ to start a new sequence.

### Backtracking

The optimal path is recovered by backtracking from $c^* = \arg\max_j V(c_j)$.

### Parameters

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Timing deviation penalty | $\alpha$ | 1.275 |
| Skip penalty | $\beta$ | 0.3 |
| Existence cost | $\lambda$ | 0.05 |
| Maximum skip | $M$ | 3 |

## Algorithm Pseudocode

```
Algorithm: Forward DP with Approximately-Periodic Prior
Input: candidates S = {c_1, ..., c_n}, evidence E(t), freq_est f
Output: optimal discharge sequence D

  T = 1 / f                              // expected period
  For j = 1 to n:
      V[j] = E(c_j)^1.5 - lambda          // start new sequence
      prev[j] = -1

  For j = 2 to n:
      For i = 1 to j-1:
          dt = (c_j - c_i) / Fs
          if dt <= 0 or dt > 4T: continue
          edge = max over m in {1,2,3} of:
              -alpha * ((dt - m*T) / (m*T))^2 - beta * (m - 1)
          score = V[i] + edge + E(c_j)^1.5 - lambda
          if score > V[j]:
              V[j] = score
              prev[j] = i

  // Backtrack from best endpoint
  j* = argmax(V)
  D = []
  while j* >= 0:
      D.prepend(c_{j*})
      j* = prev[j*]
  return D
```

## Post-Processing

### EM Template Refinement

After the initial DP pass, discharge times are refined through template-based cross-correlation (3 iterations):

1. **E-step:** Extract evidence snippets $E[d_k \pm 150\,\text{ms}]$ around each detected discharge $d_k$ and compute the mean template $\bar{\tau}$
2. **M-step:** Compute normalized cross-correlation of $\bar{\tau}$ with the full evidence trace $E(t)$; extract new candidate peaks from the correlation signal
3. Re-run the DP on the refined candidates

$$r(t) = \frac{\sum_{\delta} (\bar{\tau}(\delta) - \mu_{\bar{\tau}})(E(t + \delta) - \mu_{E,t})}{\|\bar{\tau} - \mu_{\bar{\tau}}\| \cdot \|E_t - \mu_{E,t}\|} \tag{10}$$

### Post-Hoc Confidence Filter

Detections with weak evidence are removed:

$$\mathcal{D}_{\mathrm{filtered}} = \big\{d \in \mathcal{D} : E(d) \geq \eta \cdot \mathrm{median}_{d' \in \mathcal{D}} E(d')\big\} \tag{11}$$

with $\eta = 0.3$ (minimum 30% of the median peak evidence).

## Final Frequency Estimate

The output frequency is derived from the inter-peak intervals (IPIs) of the final discharge sequence:

$$f_{\mathrm{IPI}} = \frac{1}{\mathrm{median}\!\big(\{d_{k+1} - d_k\}_{k=1}^{K-1}\big)} \tag{12}$$

where $d_k$ are discharge times in seconds.

### Per-Channel Timing

Global discharge times are refined per-channel by searching for the local evidence maximum within $\pm 50$ ms of each global time, accommodating inter-channel propagation delays.

## Performance

The full pipeline (product-boosted HPP+CET evidence, DP with CNN+ACF frequency prior, EM refinement, post-hoc filtering) achieves discharge detection F1 of approximately 0.74 at $\pm 100$ ms tolerance, evaluated on 593 patients with expert-reviewed discharge annotations.
The IPI-derived frequency achieves Spearman $\rho \approx 0.65$ with expert frequency ratings.
