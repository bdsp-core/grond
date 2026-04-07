# ChannelPD-Net: CNN with Temporal Attention for Per-Channel PD Detection

## Overview

ChannelPD-Net is a lightweight 1D convolutional neural network with temporal attention pooling that operates independently on each bipolar EEG channel.
Given a single 10-second z-scored channel, it jointly predicts (i) the probability that a periodic discharge pattern is present and (ii) the log-frequency of that pattern.
The model has approximately 50,000 trainable parameters and is deployed as a 5-fold patient-stratified ensemble.
Its per-channel outputs serve as the foundation for lateralization, spatial localization, and frequency estimation in the full characterization pipeline.

## Input Representation

Each input is a single bipolar EEG channel resampled to $F_s = 200$ Hz, yielding $N = 2000$ samples for a 10-second epoch.
The channel is z-scored to zero mean and unit variance:

$$x_n = \frac{x_n^{\mathrm{raw}} - \hat{\mu}}{\hat{\sigma}}, \quad n = 1, \ldots, N \tag{1}$$

where $\hat{\mu}$ and $\hat{\sigma}$ are the sample mean and standard deviation of the channel.
The model receives $\mathbf{x} \in \mathbb{R}^{1 \times N}$ as a single-channel 1D signal.

## Architecture

### Convolutional Backbone

The backbone consists of four 1D convolutional blocks, each applying convolution, batch normalization, GELU activation, and dropout.
Stride-2 convolutions progressively downsample the temporal dimension by a factor of $2^4 = 16$.

$$\mathbf{h}^{(1)} = \mathrm{Drop}_{0.1}\!\Big(\mathrm{GELU}\!\big(\mathrm{BN}\!\big(\mathrm{Conv1d}(\mathbf{x};\; C_{\mathrm{out}}{=}16,\; k{=}51,\; s{=}2,\; p{=}25)\big)\big)\Big) \tag{2}$$

$$\mathbf{h}^{(2)} = \mathrm{Drop}_{0.1}\!\Big(\mathrm{GELU}\!\big(\mathrm{BN}\!\big(\mathrm{Conv1d}(\mathbf{h}^{(1)};\; 32,\; k{=}25,\; s{=}2,\; p{=}12)\big)\big)\Big) \tag{3}$$

$$\mathbf{h}^{(3)} = \mathrm{Drop}_{0.1}\!\Big(\mathrm{GELU}\!\big(\mathrm{BN}\!\big(\mathrm{Conv1d}(\mathbf{h}^{(2)};\; 64,\; k{=}13,\; s{=}2,\; p{=}6)\big)\big)\Big) \tag{4}$$

$$\mathbf{h}^{(4)} = \mathrm{Drop}_{0.2}\!\Big(\mathrm{GELU}\!\big(\mathrm{BN}\!\big(\mathrm{Conv1d}(\mathbf{h}^{(3)};\; 64,\; k{=}7,\; s{=}2,\; p{=}3)\big)\big)\Big) \tag{5}$$

After the backbone, $\mathbf{h}^{(4)} \in \mathbb{R}^{64 \times T'}$ where $T' = \lceil N / 16 \rceil = 125$.

### Temporal Attention Pooling

Rather than global average pooling, we employ a learned temporal attention mechanism that allows the model to focus on the most informative time steps within the 10-second window.

A $1 \times 1$ convolution projects each 64-dimensional feature vector to a scalar logit, followed by softmax normalization across time:

$$\alpha_t = \frac{\exp\!\big(w^{\top} \mathbf{h}^{(4)}_t + b\big)}{\sum_{t'=1}^{T'} \exp\!\big(w^{\top} \mathbf{h}^{(4)}_{t'} + b\big)}, \quad t = 1, \ldots, T' \tag{6}$$

where $w \in \mathbb{R}^{64}$ and $b \in \mathbb{R}$ are the parameters of $\mathrm{Conv1d}(64 \to 1, k{=}1)$.
The context vector is the attention-weighted sum of feature vectors:

$$\mathbf{z} = \sum_{t=1}^{T'} \alpha_t \, \mathbf{h}^{(4)}_t \in \mathbb{R}^{64} \tag{7}$$

### Dual-Head Prediction

Two independent linear heads operate on the context vector $\mathbf{z}$:

**PD detection head.** Produces a probability that the channel contains a periodic discharge pattern:

$$p_{\mathrm{pd}} = \sigma\!\big(\mathbf{w}_{\mathrm{pd}}^{\top} \mathbf{z} + b_{\mathrm{pd}}\big) \in [0, 1] \tag{8}$$

**Frequency estimation head.** Predicts the log-frequency of the discharge pattern (trained only on PD-positive channels):

$$\hat{f}_{\log} = \mathbf{w}_f^{\top} \mathbf{z} + b_f \in \mathbb{R} \tag{9}$$

where $\hat{f}_{\log} \approx \log(f_{\mathrm{gt}})$ in Hz.

## Training

### Loss Function

The multi-task loss jointly optimizes detection and frequency estimation:

$$\mathcal{L} = \mathrm{BCE}(p_{\mathrm{pd}},\; y_{\mathrm{pd}}) + \lambda \cdot m \cdot \mathrm{MSE}(\hat{f}_{\log},\; \log f_{\mathrm{gt}}) \tag{10}$$

where $y_{\mathrm{pd}} \in \{0, 1\}$ is the binary PD label, $f_{\mathrm{gt}}$ is the expert-annotated frequency, $m \in \{0, 1\}$ is a mask that is 1 only when $y_{\mathrm{pd}} = 1$ and a valid frequency label exists, and $\lambda = 0.5$ balances the two tasks.

### Cross-Validation and Ensembling

Training uses 5-fold patient-stratified cross-validation, ensuring no patient appears in both training and validation folds.
At inference, the ensemble prediction is the arithmetic mean across all five fold models:

$$\bar{p}_{\mathrm{pd}} = \frac{1}{5} \sum_{k=1}^{5} p_{\mathrm{pd}}^{(k)}, \qquad \bar{f}_{\log} = \frac{1}{5} \sum_{k=1}^{5} \hat{f}_{\log}^{(k)} \tag{11}$$

### Data Augmentation

During training, three augmentations are applied stochastically:
- **Amplitude scaling:** $x \leftarrow s \cdot x$ with $s \sim \mathcal{U}(0.8, 1.2)$
- **Additive Gaussian noise:** SNR drawn from $\mathcal{U}(20, 40)$ dB
- **Circular time shift:** $x \leftarrow \mathrm{roll}(x, \delta)$ with $\delta \sim \mathcal{U}(-50, 50)$ samples

## Downstream Usage

### Laterality Detection

For LPD segments, laterality is determined by comparing the mean PD probability of left-hemisphere channels versus right-hemisphere channels:

$$\mathrm{laterality} = \begin{cases} \texttt{left} & \text{if } \bar{p}_{\mathrm{left}} > \bar{p}_{\mathrm{right}} \\ \texttt{right} & \text{otherwise} \end{cases} \tag{12}$$

where $\bar{p}_{\mathrm{left}} = \frac{1}{|\mathcal{L}|}\sum_{c \in \mathcal{L}} p_{\mathrm{pd}}^{(c)}$ and $\mathcal{L}, \mathcal{R}$ are the index sets of left and right hemisphere channels in the 18-channel bipolar montage.

### CNN Frequency Estimate

The patient-level CNN frequency estimate is a PD-weighted log-space average across all channels:

$$f_{\mathrm{CNN}} = \exp\!\left(\frac{\sum_{c=1}^{C} p_{\mathrm{pd}}^{(c)} \cdot \hat{f}_{\log}^{(c)}}{\sum_{c=1}^{C} p_{\mathrm{pd}}^{(c)}}\right) \tag{13}$$

This weighting ensures that channels with high PD probability contribute more to the frequency estimate, effectively ignoring channels dominated by artifact or background activity.
The final estimate is clipped to the physiological range $[0.3, 3.5]$ Hz.

## Performance

ChannelPD-Net achieves a channel-level PD detection AUC of approximately 0.82 and a patient-level AUC of approximately 0.93.
The CNN frequency estimate alone achieves a Spearman correlation of approximately 0.55 with expert frequency ratings; this improves substantially when combined with the ACF frequency estimate (see CNN+ACF Frequency Ensemble method).
