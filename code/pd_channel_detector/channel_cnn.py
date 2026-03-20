"""
1D CNN for single-channel PD detection + optional frequency estimation.

Input: (batch, 1, 2000) - single channel, 10s at 200Hz

Two heads:
  - PD head: Linear(64, 1) -> Sigmoid  (PD probability)
  - Freq head: Linear(64, 1)  (log frequency prediction, only trained on PD+ channels)
"""

import torch
import torch.nn as nn


class ChannelPDNet(nn.Module):
    """1D CNN for single-channel PD detection + optional frequency estimation.

    Input: (batch, 1, 2000) - single channel, 10s at 200Hz

    Architecture:
      Conv1d(1, 16, kernel=51, stride=2, pad=25) -> BatchNorm -> GELU -> Dropout(0.1)
      Conv1d(16, 32, kernel=25, stride=2, pad=12) -> BatchNorm -> GELU -> Dropout(0.1)
      Conv1d(32, 64, kernel=13, stride=2, pad=6) -> BatchNorm -> GELU -> Dropout(0.1)
      Conv1d(64, 64, kernel=7, stride=2, pad=3) -> BatchNorm -> GELU -> Dropout(0.2)
      AdaptiveAvgPool1d(1) -> Flatten

    Two heads:
      - PD head: Linear(64, 1) -> Sigmoid  (PD probability)
      - Freq head: Linear(64, 1)  (log frequency prediction, only trained on PD+ channels)
    """

    def __init__(self):
        super().__init__()

        # Backbone
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

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        # PD detection head
        self.pd_head = nn.Linear(64, 1)

        # Frequency estimation head (log frequency)
        self.freq_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            pd_prob: (batch, 1) PD probability (after sigmoid)
            freq_pred: (batch, 1) log frequency prediction (raw)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        x = self.flatten(x)  # (batch, 64)

        pd_prob = torch.sigmoid(self.pd_head(x))  # (batch, 1)
        freq_pred = self.freq_head(x)              # (batch, 1)

        return pd_prob, freq_pred


class ChannelPDNetAttention(nn.Module):
    """1D CNN with temporal attention for PD detection + frequency estimation.

    Same conv backbone as ChannelPDNet, but instead of AdaptiveAvgPool:
      - Attention branch: Conv1d(64, 1, kernel=1) -> softmax over time -> attention weights
      - Weighted pool: sum(features * attention_weights, dim=time) -> (64,)
      - Two heads: PD detection (sigmoid), frequency estimation (log Hz)

    The attention weights indicate WHERE in the 10-second window the model
    focuses for PD detection and frequency estimation.
    """

    def __init__(self):
        super().__init__()

        # Backbone (identical to ChannelPDNet)
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

        # Attention branch (replaces AdaptiveAvgPool1d)
        self.attn_conv = nn.Conv1d(64, 1, kernel_size=1)

        # PD detection head
        self.pd_head = nn.Linear(64, 1)

        # Frequency estimation head (log frequency)
        self.freq_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            pd_prob: (batch, 1) PD probability (after sigmoid)
            freq_pred: (batch, 1) log frequency prediction (raw)
            attention_weights: (batch, 1, T) attention weights over time
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # (batch, 64, T)

        # Attention pooling
        attn_logits = self.attn_conv(x)  # (batch, 1, T)
        attention_weights = torch.softmax(attn_logits, dim=-1)  # (batch, 1, T)
        pooled = (x * attention_weights).sum(dim=-1)  # (batch, 64)

        pd_prob = torch.sigmoid(self.pd_head(pooled))  # (batch, 1)
        freq_pred = self.freq_head(pooled)              # (batch, 1)

        return pd_prob, freq_pred, attention_weights
