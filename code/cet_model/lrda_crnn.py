"""LRDA frequency CRNN — end-to-end neural pitch detector.

Architecture mirrors the existing ChannelPD-Net (CNN backbone) but adds a
recurrent stage (BiGRU) over the temporally-pooled feature sequence to
capture sub-second rhythmic structure. Output is scalar log-frequency.

Input:  (B, 18, 2000)  -- 18 bipolar channels x 10s x 200 Hz
Output: (B,)           -- predicted log-frequency in [log(0.5), log(4.0)]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=2, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class TemporalAttention(nn.Module):
    """Learnable attention pool over the time dimension."""
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Conv1d(dim, 1, 1)

    def forward(self, x):                           # (B, C, T)
        a = self.proj(x).squeeze(1)                 # (B, T)
        a = F.softmax(a, dim=1)                     # (B, T)
        return torch.einsum('bct,bt->bc', x, a)     # (B, C)


class LRDAFreqCRNN(nn.Module):
    """CNN(4 blocks) -> BiGRU(1 layer) -> attention pool -> log-freq regression."""
    def __init__(self, n_channels: int = 18, dropout: float = 0.1):
        super().__init__()
        # Per-channel CNN backbone applied jointly across channels:
        # we treat the 18 channels as the Conv1d input channels and do
        # cross-channel conv (multi-channel temporal CNN). This is the same
        # design as ChannelPD-Net but with the channel dimension as features
        # rather than per-channel independent convs.
        self.conv = nn.Sequential(
            ConvBlock(n_channels, 16, kernel=51, stride=2, dropout=dropout),
            ConvBlock(16, 32, kernel=25, stride=2, dropout=dropout),
            ConvBlock(32, 64, kernel=13, stride=2, dropout=dropout),
            ConvBlock(64, 64, kernel=7,  stride=2, dropout=dropout),
        )
        # 2000 / 16 = 125 time steps, 64 channels.
        self.gru = nn.GRU(64, 64, num_layers=1, bidirectional=True, batch_first=False)
        # output: (T, B, 128)
        self.attn = TemporalAttention(128)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 18, 2000)
        x = self.conv(x)                  # (B, 64, 125)
        x = x.permute(2, 0, 1)            # (T, B, 64)
        x, _ = self.gru(x)                # (T, B, 128)
        x = x.permute(1, 2, 0)            # (B, 128, T)
        x = self.attn(x)                  # (B, 128)
        out = self.head(x).squeeze(-1)    # (B,)
        # Bound to log-freq range [log(0.4), log(4.5)] ≈ [-0.92, 1.50]
        # via 2.5*tanh(x) which maps to [-2.5, 2.5] covering the range with margin.
        return 2.5 * torch.tanh(out / 2.0)


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
