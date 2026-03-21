"""
Unified multi-task CNN for joint PD/RDA analysis.

Multi-channel architecture that processes all 18 bipolar EEG channels jointly.

Four task heads:
  1. 4-class subtype classification (LPD=0, GPD=1, LRDA=2, GRDA=3)
  2. Frequency estimation (log Hz, masked for cases without frequency labels)
  3. Per-channel PD detection (P(has PDs) per channel)
  4. Per-channel RDA detection (P(has RDA) per channel)

Input: (batch, 18, 2000) -- all 18 bipolar channels, 10s at 200Hz
"""

import torch
import torch.nn as nn


class UnifiedPDModel(nn.Module):
    """Multi-channel CNN with spatial+temporal attention for unified PD/RDA analysis.

    Input: (batch, 18, 2000) -- all 18 bipolar channels, 10s at 200Hz

    Per-channel encoder (shared weights):
      Conv1d(1->16, k=51, s=2) -> BN -> GELU -> Dropout(0.1)
      Conv1d(16->32, k=25, s=2) -> BN -> GELU -> Dropout(0.1)
      Conv1d(32->64, k=13, s=2) -> BN -> GELU -> Dropout(0.1)
      Conv1d(64->64, k=7, s=2) -> BN -> GELU -> Dropout(0.2)

    Temporal attention (per channel):
      Conv1d(64->1, k=1) -> softmax -> weights
      Weighted pool -> channel embeddings (18, 64)

    Per-channel heads:
      PD head: Linear(64->1) -> Sigmoid -- P(PD) per channel
      RDA head: Linear(64->1) -> Sigmoid -- P(RDA) per channel

    Spatial attention (across channels):
      MLP(64->32->1) -> softmax -> (18,)
      Weighted pool -> patient embedding (64,)

    Patient-level heads:
      Subtype head: Linear(64->4) -- 4-class (LPD=0, GPD=1, LRDA=2, GRDA=3)
      Frequency head: Linear(64->1) -- log(Hz)
    """

    def __init__(self):
        super().__init__()

        # Per-channel encoder (shared weights across all 18 channels)
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Temporal attention (per channel)
        self.temporal_attn = nn.Conv1d(64, 1, kernel_size=1)

        # Per-channel heads
        self.pd_head = nn.Linear(64, 1)
        self.rda_head = nn.Linear(64, 1)

        # Spatial attention (across channels)
        self.spatial_attn = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Patient-level heads
        self.subtype_head = nn.Linear(64, 4)  # 4-class: LPD=0, GPD=1, LRDA=2, GRDA=3
        self.freq_head = nn.Linear(64, 1)     # log(Hz)

    def _encode_channel(self, x):
        """Encode a single channel through the shared backbone + temporal attention.

        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            embedding: (batch, 64) channel embedding
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (batch, 64, T)

        # Temporal attention pooling
        attn_logits = self.temporal_attn(x)  # (batch, 1, T)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (batch, 1, T)
        pooled = (x * attn_weights).sum(dim=-1)  # (batch, 64)

        return pooled

    def forward(self, x):
        """
        Args:
            x: (batch, 18, 2000) all 18 bipolar channels

        Returns:
            subtype_logits: (batch, 4) raw logits for 4-class subtype
            freq_pred: (batch, 1) log(Hz) frequency prediction
            pd_channel_probs: (batch, 18) P(PD) per channel
            rda_channel_probs: (batch, 18) P(RDA) per channel
        """
        batch_size = x.shape[0]
        n_channels = x.shape[1]  # 18

        # Encode each channel through shared backbone
        channel_embeddings = []
        for ch in range(n_channels):
            ch_input = x[:, ch:ch+1, :]  # (batch, 1, 2000)
            ch_emb = self._encode_channel(ch_input)  # (batch, 64)
            channel_embeddings.append(ch_emb)

        # Stack: (batch, 18, 64)
        channel_embs = torch.stack(channel_embeddings, dim=1)

        # Per-channel PD and RDA predictions
        pd_logits = self.pd_head(channel_embs).squeeze(-1)  # (batch, 18)
        pd_channel_probs = torch.sigmoid(pd_logits)

        rda_logits = self.rda_head(channel_embs).squeeze(-1)  # (batch, 18)
        rda_channel_probs = torch.sigmoid(rda_logits)

        # Spatial attention across channels
        spatial_logits = self.spatial_attn(channel_embs).squeeze(-1)  # (batch, 18)
        spatial_weights = torch.softmax(spatial_logits, dim=-1)  # (batch, 18)

        # Weighted pool across channels -> patient embedding
        # spatial_weights: (batch, 18) -> (batch, 18, 1)
        patient_emb = (channel_embs * spatial_weights.unsqueeze(-1)).sum(dim=1)  # (batch, 64)

        # Patient-level predictions
        subtype_logits = self.subtype_head(patient_emb)  # (batch, 4)
        freq_pred = self.freq_head(patient_emb)  # (batch, 1)

        return subtype_logits, freq_pred, pd_channel_probs, rda_channel_probs
