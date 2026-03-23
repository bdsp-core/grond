"""
HemiCET: CET-UNet adapted for 8-channel hemisphere input.

Experiment 5.3 — single-hemisphere neural evidence model.

Input:  (B, 8, 2000) — one hemisphere, z-scored per channel
Output: (B, 1, 2000) — frame-level discharge evidence in [0, 1]

Architecture mirrors CETUNet but with 8-channel input and larger
channel widths to capture cross-channel patterns within a hemisphere.

Encoder:
  enc1: Conv1d(8→32,  k=51, s=2, p=25) → BN → GELU   # 1000
  enc2: Conv1d(32→64, k=25, s=2, p=12) → BN → GELU   # 500
  enc3: Conv1d(64→128, k=13, s=2, p=6) → BN → GELU   # 250
  enc4: Conv1d(128→128, k=7, s=2, p=3) → BN → GELU   # 125

Decoder with skip connections:
  up4: ConvTranspose1d(128→128) → BN → GELU → cat(enc3) → Conv1d(256→128) → BN → GELU
  up3: ConvTranspose1d(128→64)  → BN → GELU → cat(enc2) → Conv1d(128→64)  → BN → GELU
  up2: ConvTranspose1d(64→32)   → BN → GELU → cat(enc1) → Conv1d(64→32)   → BN → GELU
  up1: ConvTranspose1d(32→16)   → BN → GELU → Conv1d(16→1) → Sigmoid

Total: ~500K parameters
"""

import torch
import torch.nn as nn


class HemiCET(nn.Module):
    """Multi-channel CET-UNet for one hemisphere (8 channels → evidence trace).

    The key difference from the original CETUNet is the 8-channel input,
    which lets the model learn cross-channel evidence patterns within a
    hemisphere (e.g., spatial propagation, channel-specific sharpness).
    """

    def __init__(self, in_channels: int = 8):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=51, stride=2, padding=25),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )  # → (B, 32, 1000)

        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )  # → (B, 64, 500)

        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=13, stride=2, padding=6),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )  # → (B, 128, 250)

        self.enc4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )  # → (B, 128, 125)

        # Decoder with skip connections
        # up4: 125 → 250, then cat enc3 (128+128=256 → 128)
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.skip4 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # up3: 250 → 500, then cat enc2 (128+64=192 → 64)
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.skip3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # up2: 500 → 1000, then cat enc1 (64+32=96 → 32)
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.skip2 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        # up1: 1000 → 2000, then 1x1 to output
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _match_size(decoder_feat, encoder_feat):
        """Crop so decoder and encoder features have the same length."""
        d_len = decoder_feat.shape[2]
        e_len = encoder_feat.shape[2]
        if d_len == e_len:
            return decoder_feat, encoder_feat
        min_len = min(d_len, e_len)
        return decoder_feat[:, :, :min_len], encoder_feat[:, :, :min_len]

    def forward(self, x):
        """
        Args:
            x: (B, 8, 2000) — hemisphere EEG, z-scored per channel

        Returns:
            evidence: (B, 1, 2000) — discharge evidence in [0, 1]
        """
        # Encode
        e1 = self.enc1(x)    # (B, 32, 1000)
        e2 = self.enc2(e1)   # (B, 64, 500)
        e3 = self.enc3(e2)   # (B, 128, 250)
        e4 = self.enc4(e3)   # (B, 128, 125)

        # Decode with skip connections
        d4 = self.up4(e4)                              # (B, 128, 250)
        d4, e3m = self._match_size(d4, e3)
        d4 = self.skip4(torch.cat([d4, e3m], dim=1))  # (B, 128, 250)

        d3 = self.up3(d4)                              # (B, 64, 500)
        d3, e2m = self._match_size(d3, e2)
        d3 = self.skip3(torch.cat([d3, e2m], dim=1))  # (B, 64, 500)

        d2 = self.up2(d3)                              # (B, 32, 1000)
        d2, e1m = self._match_size(d2, e1)
        d2 = self.skip2(torch.cat([d2, e1m], dim=1))  # (B, 32, 1000)

        d1 = self.up1(d2)    # (B, 16, 2000)
        out = self.head(d1)  # (B, 1, 2000)

        return out


def count_parameters(model):
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    import torch
    model = HemiCET()
    n_params = count_parameters(model)
    print(f"HemiCET parameters: {n_params:,}")

    x = torch.randn(4, 8, 2000)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    assert y.shape == (4, 1, 2000), f"Unexpected output shape: {y.shape}"
    assert y.min() >= 0.0 and y.max() <= 1.0, "Output out of [0, 1]"
    print("Shape and range checks passed.")
