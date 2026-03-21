"""
CNN Evidence Trace (CET) — produces frame-level discharge evidence.

Two architectures:
  CETModel: Original encoder-decoder (no skip connections)
  CETUNet:  U-Net with skip connections for sharper temporal resolution

Input:  (batch, 1, 2000) — single channel, 10s at 200 Hz
Output: (batch, 1, 2000) — discharge evidence in [0, 1]
"""

import torch
import torch.nn as nn


class CETModel(nn.Module):
    """CNN Evidence Trace — original encoder-decoder (no skip connections).

    Kept for backward compatibility and comparison.
    """

    def __init__(self):
        super().__init__()

        # Encoder (identical to ChannelPDNet backbone)
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Decoder (lightweight upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.GELU(),
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(8, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            evidence: (batch, 1, 2000) discharge evidence in [0, 1]
        """
        # Encode
        x = self.enc1(x)   # (B, 16, 1000)
        x = self.enc2(x)   # (B, 32, 500)
        x = self.enc3(x)   # (B, 64, 250)
        x = self.enc4(x)   # (B, 64, 125)

        # Decode
        x = self.dec1(x)   # (B, 32, 250)
        x = self.dec2(x)   # (B, 16, 500)
        x = self.dec3(x)   # (B, 8, 1000)
        x = self.dec4(x)   # (B, 1, 2000)

        return x


class CETUNet(nn.Module):
    """U-Net for frame-level discharge evidence at full resolution.

    Encoder (same conv blocks):
      e1: Conv1d(1->16, k=51, s=2, p=25) -> BN -> GELU    # 1000
      e2: Conv1d(16->32, k=25, s=2, p=12) -> BN -> GELU    # 500
      e3: Conv1d(32->64, k=13, s=2, p=6) -> BN -> GELU     # 250
      e4: Conv1d(64->64, k=7, s=2, p=3) -> BN -> GELU      # 125

    Decoder with skip connections:
      d4: ConvTranspose1d(64->64, k=4, s=2, p=1) -> BN -> GELU   # 250
          cat(d4, e3) -> Conv1d(128->64, k=3, p=1) -> BN -> GELU
      d3: ConvTranspose1d(64->32, k=4, s=2, p=1) -> BN -> GELU   # 500
          cat(d3, e2) -> Conv1d(64->32, k=3, p=1) -> BN -> GELU
      d2: ConvTranspose1d(32->16, k=4, s=2, p=1) -> BN -> GELU   # 1000
          cat(d2, e1) -> Conv1d(32->16, k=3, p=1) -> BN -> GELU
      d1: ConvTranspose1d(16->8, k=4, s=2, p=1) -> BN -> GELU    # 2000
          Conv1d(8->1, k=1) -> Sigmoid

    Skip connections concatenate encoder features with decoder features
    at matching resolutions, preserving high-frequency temporal detail.

    Handle size mismatches: if encoder and decoder sizes differ by 1 sample
    due to odd input sizes, trim the larger one.
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
        # d4: upsample from 125 -> 250, then cat with e3 (64+64=128 -> 64)
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.skip4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # d3: upsample from 250 -> 500, then cat with e2 (32+32=64 -> 32)
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.skip3 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # d2: upsample from 500 -> 1000, then cat with e1 (16+16=32 -> 16)
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.skip2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )

        # d1: upsample from 1000 -> 2000, then 1x1 conv to output
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(8),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(8, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _match_size(decoder_feat, encoder_feat):
        """Crop or pad so decoder and encoder features have the same length."""
        d_len = decoder_feat.shape[2]
        e_len = encoder_feat.shape[2]
        if d_len == e_len:
            return decoder_feat, encoder_feat
        # Trim the longer one
        min_len = min(d_len, e_len)
        return decoder_feat[:, :, :min_len], encoder_feat[:, :, :min_len]

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 2000) single-channel EEG

        Returns:
            evidence: (batch, 1, 2000) discharge evidence in [0, 1]
        """
        # Encode — save activations for skip connections
        e1 = self.enc1(x)    # (B, 16, 1000)
        e2 = self.enc2(e1)   # (B, 32, 500)
        e3 = self.enc3(e2)   # (B, 64, 250)
        e4 = self.enc4(e3)   # (B, 64, 125)

        # Decode with skip connections
        d4 = self.up4(e4)                        # (B, 64, 250)
        d4, e3_matched = self._match_size(d4, e3)
        d4 = self.skip4(torch.cat([d4, e3_matched], dim=1))  # (B, 64, 250)

        d3 = self.up3(d4)                        # (B, 32, 500)
        d3, e2_matched = self._match_size(d3, e2)
        d3 = self.skip3(torch.cat([d3, e2_matched], dim=1))  # (B, 32, 500)

        d2 = self.up2(d3)                        # (B, 16, 1000)
        d2, e1_matched = self._match_size(d2, e1)
        d2 = self.skip2(torch.cat([d2, e1_matched], dim=1))  # (B, 16, 1000)

        d1 = self.up1(d2)                        # (B, 8, 2000)
        out = self.head(d1)                      # (B, 1, 2000)

        return out
