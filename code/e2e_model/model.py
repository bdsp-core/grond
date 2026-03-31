"""
E2E Discharge Detector — DETR-style model for periodic discharge detection.

Architecture:
  - Backbone: Pretrained ChannelPDNetAttention conv layers (per-channel CNN)
  - HPP features: Frozen handcrafted pointiness + TKEO evidence
  - Feature fusion: Linear(1170, 256)
  - Positional encoding: sinusoidal, 125 positions
  - Transformer decoder: 4 layers, 8 heads, dim=256, 30 learned queries
  - Prediction heads: time, confidence, frequency

Input: (batch, 18, 2000) bipolar EEG
Output: discharge times, confidences, frequency estimate, laterality logit
"""

import math
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'code'))

from pd_channel_detector.channel_cnn import ChannelPDNetAttention
from discharge_detector import compute_channel_evidence

# Channel layout for laterality
LEFT_INDICES = [0, 1, 2, 3, 8, 9, 10, 11]
RIGHT_INDICES = [4, 5, 6, 7, 12, 13, 14, 15]


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class E2EDischargeDetector(nn.Module):
    """DETR-style end-to-end discharge detector."""

    def __init__(self, n_queries=30, d_model=256, nhead=8, num_decoder_layers=4,
                 pretrained_path=None):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.n_channels = 18

        # --- Backbone: reuse ChannelPDNetAttention conv blocks ---
        self._build_backbone(pretrained_path)

        # --- Feature fusion ---
        # CNN features: 18 * 64 = 1152, HPP features: 18 -> total 1170
        self.feature_fusion = nn.Sequential(
            nn.Linear(1170, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # --- Positional encoding ---
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=200)

        # --- Transformer decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, activation='gelu', batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # --- Learned discharge queries ---
        self.discharge_queries = nn.Embedding(n_queries, d_model)

        # --- Prediction heads ---
        self.time_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Frequency: pool all queries -> single freq prediction
        self.freq_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Laterality: from backbone channel attention weights (not a learned head)
        # We store per-channel PD probs during forward and derive laterality

    def _build_backbone(self, pretrained_path=None):
        """Extract conv blocks from pretrained ChannelPDNetAttention."""
        # Create source model to copy weights from
        source = ChannelPDNetAttention()
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            source.load_state_dict(state)

        # Copy conv blocks (shared across all 18 channels)
        self.conv_block1 = source.block1
        self.conv_block2 = source.block2
        self.conv_block3 = source.block3
        self.conv_block4 = source.block4

        # Also keep the attention conv for laterality
        self.channel_attn_conv = source.attn_conv
        self.channel_pd_head = source.pd_head

    def _extract_cnn_features(self, eeg):
        """Run CNN backbone on all 18 channels.

        Args:
            eeg: (batch, 18, 2000) bipolar EEG

        Returns:
            features: (batch, 18, 64, T) where T=125
            channel_pd_probs: (batch, 18) PD probability per channel
        """
        B, C, S = eeg.shape  # batch, 18, 2000
        # Process all channels as a batch
        x = eeg.reshape(B * C, 1, S)  # (B*18, 1, 2000)

        x = self.conv_block1(x)   # (B*18, 16, 1000)
        x = self.conv_block2(x)   # (B*18, 32, 500)
        x = self.conv_block3(x)   # (B*18, 64, 250)
        x = self.conv_block4(x)   # (B*18, 64, 125)

        T = x.shape[-1]  # should be 125

        # Get channel PD probs for laterality
        attn_logits = self.channel_attn_conv(x)  # (B*18, 1, T)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        pooled = (x * attn_weights).sum(dim=-1)  # (B*18, 64)
        channel_pd_probs = torch.sigmoid(self.channel_pd_head(pooled)).squeeze(-1)  # (B*18,)
        channel_pd_probs = channel_pd_probs.reshape(B, C)  # (B, 18)

        features = x.reshape(B, C, 64, T)  # (B, 18, 64, T)
        return features, channel_pd_probs

    def _compute_hpp_features(self, eeg_np):
        """Compute HPP (pointiness + TKEO) features for each channel.

        Args:
            eeg_np: numpy array (batch, 18, 2000)

        Returns:
            hpp: numpy array (batch, 18, 125) — downsampled evidence
        """
        B, C, S = eeg_np.shape
        hpp = np.zeros((B, C, 125), dtype=np.float32)
        pool_size = S // 125  # 16

        for b in range(B):
            for c in range(C):
                evidence = compute_channel_evidence(eeg_np[b, c])
                # Downsample via average pooling: 2000 -> 125
                n = len(evidence)
                # Pad to multiple of pool_size if needed
                if n % pool_size != 0:
                    pad = pool_size - (n % pool_size)
                    evidence = np.pad(evidence, (0, pad), mode='constant')
                hpp[b, c] = evidence[:pool_size * 125].reshape(125, pool_size).mean(axis=1)

        return hpp

    def compute_laterality_logit(self, channel_pd_probs):
        """Compute laterality logit from channel PD probabilities.

        Returns positive for right lateralization, negative for left.
        """
        left_mean = channel_pd_probs[:, LEFT_INDICES].mean(dim=1)
        right_mean = channel_pd_probs[:, RIGHT_INDICES].mean(dim=1)
        # logit = log(right / left) ~ right - left in probability space
        lat_logit = right_mean - left_mean  # (B,)
        return lat_logit

    def forward(self, eeg, hpp=None):
        """Forward pass.

        Args:
            eeg: (batch, 18, 2000) bipolar EEG tensor
            hpp: (batch, 18, 125) precomputed HPP features tensor.
                 If None, computes from eeg (slow).

        Returns:
            dict with:
                pred_times: (batch, n_queries) predicted discharge times in seconds
                pred_confs: (batch, n_queries) confidence scores
                pred_freq: (batch, 1) predicted frequency in Hz
                lat_logit: (batch,) laterality logit (positive = right)
                channel_pd_probs: (batch, 18) per-channel PD probability
        """
        B = eeg.shape[0]
        device = eeg.device

        # 1. CNN backbone features
        cnn_feats, channel_pd_probs = self._extract_cnn_features(eeg)
        # cnn_feats: (B, 18, 64, 125)

        # 2. HPP features (frozen, no grad)
        if hpp is None:
            eeg_np = eeg.detach().cpu().numpy()
            with torch.no_grad():
                hpp_np = self._compute_hpp_features(eeg_np)
            hpp = torch.from_numpy(hpp_np).to(device)  # (B, 18, 125)
        else:
            hpp = hpp.to(device)

        # 3. Reshape and concatenate
        # CNN: (B, 18, 64, 125) -> (B, 18*64, 125) -> (B, 125, 1152)
        T = cnn_feats.shape[-1]
        cnn_flat = cnn_feats.reshape(B, self.n_channels * 64, T)  # (B, 1152, 125)
        cnn_flat = cnn_flat.permute(0, 2, 1)  # (B, 125, 1152)

        # HPP: (B, 18, 125) -> (B, 125, 18)
        hpp_flat = hpp.permute(0, 2, 1)  # (B, 125, 18)

        # Concatenate: (B, 125, 1170)
        fused = torch.cat([cnn_flat, hpp_flat], dim=-1)

        # 4. Feature fusion: (B, 125, 1170) -> (B, 125, 256)
        memory = self.feature_fusion(fused)

        # 5. Positional encoding
        memory = self.pos_enc(memory)

        # 6. Transformer decoder
        queries = self.discharge_queries.weight.unsqueeze(0).expand(B, -1, -1)
        # (B, n_queries, d_model)
        decoded = self.transformer_decoder(queries, memory)
        # (B, n_queries, d_model)

        # 7. Prediction heads
        pred_times = torch.sigmoid(self.time_head(decoded).squeeze(-1)) * 10.0
        # (B, n_queries) in [0, 10] seconds

        pred_confs = torch.sigmoid(self.conf_head(decoded).squeeze(-1))
        # (B, n_queries) in [0, 1]

        # Frequency: pool confident queries
        # Use attention-weighted mean of query features
        conf_weights = pred_confs.unsqueeze(-1)  # (B, n_queries, 1)
        weighted_queries = (decoded * conf_weights).sum(dim=1) / (
            conf_weights.sum(dim=1) + 1e-6
        )  # (B, d_model)
        pred_freq = self.freq_head(weighted_queries)  # (B, 1)

        # Laterality from channel attention
        lat_logit = self.compute_laterality_logit(channel_pd_probs)

        return {
            'pred_times': pred_times,
            'pred_confs': pred_confs,
            'pred_freq': pred_freq.squeeze(-1),
            'lat_logit': lat_logit,
            'channel_pd_probs': channel_pd_probs,
        }

    def freeze_backbone(self):
        """Freeze CNN backbone for Phase 1 training."""
        for module in [self.conv_block1, self.conv_block2,
                       self.conv_block3, self.conv_block4,
                       self.channel_attn_conv, self.channel_pd_head]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze CNN backbone for Phase 2 training."""
        for module in [self.conv_block1, self.conv_block2,
                       self.conv_block3, self.conv_block4,
                       self.channel_attn_conv, self.channel_pd_head]:
            for param in module.parameters():
                param.requires_grad = True

    def get_param_groups(self, lr_backbone=1e-5, lr_decoder=1e-4):
        """Get parameter groups with different learning rates."""
        backbone_params = []
        decoder_params = []

        backbone_modules = {self.conv_block1, self.conv_block2,
                            self.conv_block3, self.conv_block4,
                            self.channel_attn_conv, self.channel_pd_head}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_backbone = any(
                param is p for m in backbone_modules for p in m.parameters()
            )
            if is_backbone:
                backbone_params.append(param)
            else:
                decoder_params.append(param)

        return [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': decoder_params, 'lr': lr_decoder},
        ]
