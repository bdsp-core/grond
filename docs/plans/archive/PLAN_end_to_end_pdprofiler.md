# End-to-End Differentiable PDProfiler

## Goal

Replace the modular PDProfiler pipeline with a single end-to-end differentiable model that can be trained with gradient descent from raw EEG to discharge times, frequency, and laterality. Run training on cloud GPU (AWS or GCP).

## Current Pipeline (Modular, Partially Non-Differentiable)

```
EEG (18ch, 2000 samples)
  |
  v
ChannelPDNetAttention (CNN)         <-- differentiable
  |-> channel_probs (18,)
  |-> log_freq per channel
  |
  v
Evidence Traces:
  HPP (pointiness + TKEO)           <-- NOT differentiable (peak finding)
  CETUNet (CNN)                      <-- differentiable
  |
  v
Weighted Aggregation                 <-- differentiable
  |-> combined evidence (2000,)
  |
  v
Active Interval Detection            <-- NOT differentiable (thresholding)
  |
  v
Peak Detection (find_peaks)          <-- NOT differentiable
  |
  v
Dynamic Programming                  <-- NOT differentiable (argmax, discrete)
  |
  v
EM Refinement                        <-- NOT differentiable (template matching)
  |
  v
discharge_times, frequency, laterality
```

## Proposed Architecture: DETR-Style Discharge Detector

### Overview

Replace everything below ChannelPDNetAttention with a single learned decoder. Keep HPP as a frozen auxiliary feature.

```
EEG (18ch, 2000 samples)
  |
  v
Backbone Encoder (shared)
  |-> Channel features (18, D, T)    <- learns PD-relevant features
  |-> HPP features (18, 2000)        <- frozen, handcrafted (no grad)
  |
  v
Feature Fusion
  |-> Concatenate CNN features + HPP along channel dim
  |-> 1D conv to fuse -> (D', T)
  |
  v
Transformer Decoder (DETR-style)
  |-> N learned discharge queries (N=30, max possible discharges in 10s)
  |-> Cross-attention to fused features
  |-> Self-attention between queries
  |
  v
Prediction Heads (per query):
  |-> discharge_time: sigmoid * 10.0  (continuous, 0-10s)
  |-> confidence: sigmoid             (is this a real discharge?)
  |-> Output: variable-length set of (time, confidence) pairs
  |
  v
Auxiliary Heads (from encoder):
  |-> laterality: from channel attention weights
  |-> frequency: 1/median(IPI) from predicted times, or direct regression
```

### Loss Function: Hungarian Matching

Following DETR (Carion et al., 2020):

1. **Bipartite matching**: Use Hungarian algorithm to match predicted discharges to GT discharges (minimizing time error + confidence cost)
2. **Matched pairs**: L1 loss on time + BCE on confidence (target=1)
3. **Unmatched predictions**: BCE on confidence (target=0)
4. **Auxiliary losses**:
   - Laterality: BCE on R-L channel attention difference
   - Frequency: Huber loss on predicted vs GT frequency
   - Evidence reconstruction: optional MSE on intermediate evidence trace vs HPP (regularization)

```python
# Pseudocode
matched_pred, matched_gt = hungarian_match(pred_times, gt_times)
loss_time = L1(matched_pred.times, matched_gt.times)
loss_conf_pos = BCE(matched_pred.conf, 1.0)
loss_conf_neg = BCE(unmatched_pred.conf, 0.0)
loss_lat = BCE(lat_pred, lat_gt)
loss_freq = huber(freq_pred, freq_gt)

total = w_time * loss_time + w_conf * (loss_conf_pos + loss_conf_neg)
      + w_lat * loss_lat + w_freq * loss_freq
```

### Handling Non-Differentiable Components

| Component | Strategy |
|-----------|----------|
| Dynamic Programming | Replaced entirely by Transformer decoder |
| Peak detection | Replaced by learned discharge queries |
| Active interval | Replaced by confidence head (low confidence = inactive) |
| Pointiness (HPP) | Keep frozen as input feature (no grad needed) |
| TKEO | Keep frozen as input feature |
| Frequency from IPI | Differentiable: 1/median(diff(sorted(pred_times))) with soft-sort, OR direct regression head |

### Architecture Details

**Backbone Encoder** (reuse/finetune existing weights):
- Start from pretrained ChannelPDNetAttention conv layers
- Remove classification head, keep conv1-conv4 as feature extractor
- Output: (batch, 64, 125) per channel -> reshape to (batch, 18*64, 125)
- Add positional encoding (sinusoidal, 125 positions)

**Feature Fusion**:
- HPP trace: downsample 2000 -> 125 (avg pool)
- Concatenate: (batch, 18*64 + 18, 125) = (batch, 1170, 125)
- 1D conv: 1170 -> 256, kernel=1

**Transformer Decoder**:
- 4 layers, 8 heads, dim=256
- N=30 learned queries (max ~30 discharges at 3 Hz in 10s)
- Cross-attention to encoder features
- Self-attention between queries (learn to space out)

**Prediction Heads** (per query, 3 small MLPs):
- Time: Linear(256, 1) + sigmoid * 10.0
- Confidence: Linear(256, 1) + sigmoid
- (Optional) Local waveform class: Linear(256, K) for morphology

### Training Strategy

**Phase 1: Warm start (freeze backbone)**
- Initialize backbone from pretrained ChannelPDNetAttention weights
- Freeze backbone, train only decoder + heads
- 50 epochs, lr=1e-3
- This learns the decoder without destroying pretrained features

**Phase 2: End-to-end finetuning**
- Unfreeze backbone with lower lr (1e-5 for backbone, 1e-4 for decoder)
- 100 epochs with cosine annealing
- Gradient clipping = 1.0

**Phase 3: Hard example mining**
- Identify segments where model fails (high time error or missed detections)
- Upsample these in training batches
- 50 more epochs

### Data Requirements

Current labeled data:
- 882 segments with discharge timing (GT discharge times)
- 880 with laterality
- 842 with expert frequency

This is borderline for a Transformer — augmentation is critical:
- Time shifting (circular shift +-0.5s)
- Amplitude scaling (0.7-1.3x)
- Channel dropout (15%)
- Gaussian noise (SNR 20-40 dB)
- Mixup on evidence features
- Random segment cropping (8-10s window from longer recordings if available)

Consider: pretrain on the full 4213 LPD segments using HPP pseudo-labels before finetuning on 882 expert-labeled segments.

## Cloud Deployment Plan

### Option A: AWS (preferred if account unblocked)

**Instance**: p3.2xlarge (V100 16GB) or g4dn.xlarge (T4 16GB)
- V100: ~$3/hr, faster training
- T4: ~$0.50/hr, sufficient for this model size

**Setup**:
```bash
# Launch with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0296250c0b9cc776b \
  --instance-type g4dn.xlarge \
  --key-name hemi-contest \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100}}]'

# Upload code + data (~4GB)
rsync -avz --progress \
  code/ data/labels/ data/eeg/ data/pd_channel_cache/ \
  ubuntu@<ip>:~/pdchar/

# Train
nohup python train_e2e.py --phase 1 --epochs 50 > train.log 2>&1 &
nohup python train_e2e.py --phase 2 --epochs 100 --resume > train.log 2>&1 &
```

**Estimated training time**: ~2-4 hours on V100, ~6-10 hours on T4

### Option B: GCP

**Instance**: n1-standard-4 + T4 GPU
- Requires GPU quota increase (request 1 GPU in us-east1)
- Quota approval: usually minutes for small requests

**Setup**:
```bash
# Launch with PyTorch DL image
gcloud compute instances create pdchar-train \
  --zone=us-east1-b \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE

# Upload via gcloud scp
gcloud compute scp --recurse code/ data/ pdchar-train:~/pdchar/

# Train via SSH
gcloud compute ssh pdchar-train -- 'cd pdchar && nohup python train_e2e.py &'
```

### Option C: Google Colab Pro (quickest to start)

- A100 GPU available with Pro+ ($50/mo)
- Upload data to Google Drive, mount in Colab
- No infrastructure setup needed
- Downside: session timeouts, need to checkpoint frequently

## Implementation Plan

### Step 1: Scaffold (1 day)
- [ ] Create `code/e2e_model/` directory
- [ ] Implement `E2EDischargeDetector` model class
- [ ] Implement Hungarian matching loss
- [ ] Implement dataset class that loads EEG + discharge_times GT

### Step 2: Training loop (1 day)
- [ ] Implement training script with Phase 1/2/3
- [ ] Add logging (wandb or tensorboard)
- [ ] Add checkpointing every 10 epochs
- [ ] Add evaluation (same metrics as contest: F1, MAE, freq rho, lat AUC)

### Step 3: Local validation (1 day)
- [ ] Train Phase 1 locally on MPS (small batch, few epochs) to verify convergence
- [ ] Compare against PDProfiler baseline on held-out fold
- [ ] Debug any issues with Hungarian matching or gradient flow

### Step 4: Cloud training (1 day)
- [ ] Set up cloud instance (AWS or GCP)
- [ ] Upload code + data
- [ ] Run full training (Phase 1 + 2 + 3)
- [ ] Download trained weights

### Step 5: Integration (1 day)
- [ ] Wrap trained model in PDProfiler-compatible interface
- [ ] Run full contest evaluation to compare
- [ ] If better: make default; if worse: keep as experimental

## Expected Outcomes

**Optimistic**: End-to-end model beats modular pipeline on all metrics because it can learn features optimized for the final task rather than intermediate objectives.

**Realistic**: Comparable or slightly better on timing F1 and frequency, similar on laterality. Main win is simpler inference (one forward pass vs multi-stage pipeline).

**Pessimistic**: Worse due to insufficient training data (882 labeled segments is small for a Transformer). Mitigated by warm-starting from pretrained weights and heavy augmentation.

## References

- Carion et al., "End-to-End Object Detection with Transformers" (DETR), ECCV 2020
- Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series", ICML 2017
- Stewart et al., "End-to-End People Detection in Crowded Scenes", CVPR 2016
