# HemiCET-UNet: Frame-Level Discharge Evidence via Encoder-Decoder Network

## Overview

HemiCET-UNet is a 1D U-Net encoder-decoder architecture that produces a dense, frame-level discharge evidence trace from a single z-scored EEG channel.
Unlike ChannelPD-Net, which provides a single summary probability per channel, HemiCET-UNet outputs a temporal signal $E(t) \in [0,1]^N$ at the full input resolution ($N = 2000$ samples, 200 Hz), indicating the instantaneous likelihood of a periodic discharge at each time point.
The model has approximately 525,000 trainable parameters and is deployed as a 5-fold patient-stratified ensemble.

## Input Representation

The input is identical to ChannelPD-Net: a single bipolar EEG channel $\mathbf{x} \in \mathbb{R}^{1 \times N}$ with $N = 2000$ samples at $F_s = 200$ Hz, z-scored per-channel:

$$x_n = \frac{x_n^{\mathrm{raw}} - \hat{\mu}}{\hat{\sigma}}, \quad n = 1, \ldots, N \tag{1}$$

## Architecture

### Encoder

The encoder mirrors the ChannelPD-Net backbone: four convolutional stages with stride-2 downsampling, batch normalization, and GELU activation.
Each stage halves the temporal dimension:

| Stage | Operation | Output Shape |
|-------|-----------|-------------|
| $\mathbf{e}_1$ | $\mathrm{Conv1d}(1 \to 16,\; k{=}51,\; s{=}2,\; p{=}25) \to \mathrm{BN} \to \mathrm{GELU}$ | $16 \times 1000$ |
| $\mathbf{e}_2$ | $\mathrm{Conv1d}(16 \to 32,\; k{=}25,\; s{=}2,\; p{=}12) \to \mathrm{BN} \to \mathrm{GELU}$ | $32 \times 500$ |
| $\mathbf{e}_3$ | $\mathrm{Conv1d}(32 \to 64,\; k{=}13,\; s{=}2,\; p{=}6) \to \mathrm{BN} \to \mathrm{GELU}$ | $64 \times 250$ |
| $\mathbf{e}_4$ | $\mathrm{Conv1d}(64 \to 64,\; k{=}7,\; s{=}2,\; p{=}3) \to \mathrm{BN} \to \mathrm{GELU}$ | $64 \times 125$ |

### Decoder with Skip Connections

The decoder progressively upsamples back to the input resolution using transposed convolutions.
At each stage, the upsampled feature map is concatenated with the corresponding encoder feature map (skip connection) and fused through a $1 \times 3$ convolution:

| Stage | Upsample | Skip Concat | Fusion | Output Shape |
|-------|----------|-------------|--------|-------------|
| $\mathbf{d}_4$ | $\mathrm{ConvTranspose1d}(64 \to 64,\; k{=}4,\; s{=}2,\; p{=}1)$ | $[\mathbf{d}_4 \,\|\, \mathbf{e}_3]$ ($128$ ch) | $\mathrm{Conv1d}(128 \to 64,\; k{=}3,\; p{=}1) \to \mathrm{BN} \to \mathrm{GELU}$ | $64 \times 250$ |
| $\mathbf{d}_3$ | $\mathrm{ConvTranspose1d}(64 \to 32,\; k{=}4,\; s{=}2,\; p{=}1)$ | $[\mathbf{d}_3 \,\|\, \mathbf{e}_2]$ ($64$ ch) | $\mathrm{Conv1d}(64 \to 32,\; k{=}3,\; p{=}1) \to \mathrm{BN} \to \mathrm{GELU}$ | $32 \times 500$ |
| $\mathbf{d}_2$ | $\mathrm{ConvTranspose1d}(32 \to 16,\; k{=}4,\; s{=}2,\; p{=}1)$ | $[\mathbf{d}_2 \,\|\, \mathbf{e}_1]$ ($32$ ch) | $\mathrm{Conv1d}(32 \to 16,\; k{=}3,\; p{=}1) \to \mathrm{BN} \to \mathrm{GELU}$ | $16 \times 1000$ |
| $\mathbf{d}_1$ | $\mathrm{ConvTranspose1d}(16 \to 8,\; k{=}4,\; s{=}2,\; p{=}1)$ | -- | $\mathrm{Conv1d}(8 \to 1,\; k{=}1) \to \sigma$ | $1 \times 2000$ |

When encoder and decoder feature maps differ in length due to rounding, the longer map is truncated to match.

### Output

The final layer applies a sigmoid activation to produce the evidence trace:

$$E(t) = \sigma\!\big(\mathrm{Conv1d}(8 \to 1,\; k{=}1)\big) \in [0, 1]^N \tag{2}$$

## Training

### Target Construction

Training targets are constructed from expert-annotated discharge times $\{t_1^*, t_2^*, \ldots, t_K^*\}$ (in samples).
Each discharge generates a Gaussian peak in the target trace:

$$y(t) = \max_{k=1}^{K} \exp\!\left(-\frac{(t - t_k^*)^2}{2\sigma_{\mathrm{target}}^2}\right) \tag{3}$$

where $\sigma_{\mathrm{target}} = 2$ samples (10 ms at 200 Hz), producing narrow, well-localized targets.

### Loss Function

The composite loss has three terms:

$$\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \gamma_1 \cdot \mathcal{L}_{\mathrm{sharp}} + \gamma_2 \cdot \mathcal{L}_{\mathrm{floor}} \tag{4}$$

**Term 1: Weighted binary cross-entropy.**
Because discharge events are temporally sparse (a few dozen samples out of 2000), we apply asymmetric weighting with positive weight $w_+ = 20$:

$$\mathcal{L}_{\mathrm{BCE}} = -\frac{1}{N} \sum_{t=1}^{N} \Big[ w_+ \cdot y(t) \log E(t) + (1 - y(t)) \log(1 - E(t)) \Big] \tag{5}$$

**Term 2: Sharpness penalty.**
Encourages sparse evidence by penalizing the mean activation:

$$\mathcal{L}_{\mathrm{sharp}} = \frac{1}{N} \sum_{t=1}^{N} E(t) \tag{6}$$

with $\gamma_1 = 0.1$.

**Term 3: Floor loss.**
At expert-labeled discharge locations, the CET evidence should be at least as strong as the handcrafted HPP evidence, ensuring the learned model does not miss obvious discharges:

$$\mathcal{L}_{\mathrm{floor}} = \frac{1}{K} \sum_{k=1}^{K} \max\!\big(0,\; E_{\mathrm{hpp}}(t_k^*) - E(t_k^*)\big)^2 \tag{7}$$

with $\gamma_2 = 0.05$.

### Data Augmentation

Four augmentation strategies are applied stochastically during training:

1. **Amplitude scaling:** $x \leftarrow s \cdot x$ with $s \sim \mathcal{U}(0.7, 1.3)$
2. **Additive Gaussian noise:** $x \leftarrow x + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.05^2)$
3. **Channel dropout:** each channel is zeroed with probability $p = 0.15$
4. **Discharge jitter:** expert discharge times are perturbed by $\delta \sim \mathcal{N}(0, \sigma_j^2)$ with $\sigma_j = 1$ sample (5 ms), modeling annotation uncertainty

### Cross-Validation and Ensembling

The model is trained with 5-fold patient-stratified cross-validation.
At inference, the ensemble evidence trace is the arithmetic mean:

$$E_{\mathrm{ensemble}}(t) = \frac{1}{5} \sum_{k=1}^{5} E^{(k)}(t) \tag{8}$$

## Inference Pipeline

```
Input: single channel x in R^{2000}
  1. z-score normalize x
  2. For each fold model k = 1, ..., 5:
       E^(k) = CETUNet_k(x)           // forward pass
  3. E_ensemble = mean(E^(1), ..., E^(5))
Output: E_ensemble in [0,1]^{2000}    // frame-level evidence
```

The per-channel evidence traces are subsequently aggregated across channels and combined with handcrafted evidence (see Dynamic Programming method) for discharge detection.

## Performance

HemiCET-UNet evidence, when combined with handcrafted HPP evidence via the product-boost formula and processed through dynamic programming, achieves a discharge detection F1 of approximately 0.74 at a tolerance of $\pm 100$ ms.
Using CET evidence alone (without HPP) yields F1 approximately 0.65, demonstrating that the learned and handcrafted evidence sources provide complementary information.
