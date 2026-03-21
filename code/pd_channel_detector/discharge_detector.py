"""
Frame-level discharge detector: U-Net style encoder-decoder that produces
a per-sample discharge probability signal from single-channel EEG.

Input:  (batch, 1, 2000) - single channel, 10s at 200Hz
Output: (batch, 1, 2000) - discharge probability at each sample
"""

import torch
import torch.nn as nn


class DischargeDetector(nn.Module):
    """Frame-level discharge detection from single-channel EEG.

    Input: (batch, 1, 2000) - single channel, 10s at 200Hz
    Output: (batch, 1, 2000) - discharge probability at each sample

    Architecture (encoder-decoder / U-Net style):
      Encoder (same as ChannelPDNet):
        Conv1d(1, 16, k=51, s=2, p=25) -> BN -> GELU     # 1000 samples
        Conv1d(16, 32, k=25, s=2, p=12) -> BN -> GELU     # 500 samples
        Conv1d(32, 64, k=13, s=2, p=6) -> BN -> GELU      # 250 samples
        Conv1d(64, 64, k=7, s=2, p=3) -> BN -> GELU       # 125 samples

      Decoder (upsample back to original resolution):
        ConvTranspose1d(64+64, 32, k=4, s=2, p=1) -> BN -> GELU  # 250
        ConvTranspose1d(32+64, 16, k=4, s=2, p=1) -> BN -> GELU  # 500
        ConvTranspose1d(16+32, 8, k=4, s=2, p=1) -> BN -> GELU   # 1000
        ConvTranspose1d(8+16, 1, k=4, s=2, p=1) -> Sigmoid       # 2000

    Skip connections from encoder to decoder (U-Net style).
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # Decoder with skip connections
        # dec4 input: enc4 output (64) + enc3 output (64) = 128
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        # dec3 input: dec4 output (32) + enc2 output (32) = 64
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(32 + 32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        # dec2 input: dec3 output (16) + enc1 output (16) = 32
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(16 + 16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.GELU(),
        )
        # dec1 input: dec2 output (8) + original input (1) = 9
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(8 + 1, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            prob: (batch, 1, 2000) discharge probability (after sigmoid)
        """
        # Encoder
        e1 = self.enc1(x)    # (B, 16, 1000)
        e2 = self.enc2(e1)   # (B, 32, 500)
        e3 = self.enc3(e2)   # (B, 64, 250)
        e4 = self.enc4(e3)   # (B, 64, 125)

        # Decoder with skip connections
        # Upsample e4 and concat with e3
        d4 = self.dec4(torch.cat([e4, _match_length(e3, e4)], dim=1))  # (B, 32, 250)
        d3 = self.dec3(torch.cat([d4, _match_length(e2, d4)], dim=1))  # (B, 16, 500)
        d2 = self.dec2(torch.cat([d3, _match_length(e1, d3)], dim=1))  # (B, 8, 1000)
        d1 = self.dec1(torch.cat([d2, _match_length(x, d2)], dim=1))   # (B, 1, 2000)

        # Ensure output matches input length exactly
        if d1.shape[2] != x.shape[2]:
            d1 = d1[:, :, :x.shape[2]]

        prob = torch.sigmoid(d1)
        return prob


def _match_length(skip, target):
    """Trim or pad skip connection to match target's temporal dimension."""
    if skip.shape[2] == target.shape[2]:
        return skip
    elif skip.shape[2] > target.shape[2]:
        return skip[:, :, :target.shape[2]]
    else:
        # Pad with zeros
        pad = torch.zeros(skip.shape[0], skip.shape[1],
                          target.shape[2] - skip.shape[2],
                          device=skip.device, dtype=skip.dtype)
        return torch.cat([skip, pad], dim=2)
