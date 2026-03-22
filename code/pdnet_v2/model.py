"""
PDNetV2 - Joint 18-channel temporal U-Net with attention bottleneck.

Architecture:
  Input: (B, 18, 2000)
  Stem → Encoder (4 levels) → Bottleneck (Transformer) → Decoder (3 levels)
  Output heads: event_logits, active_logits, freq_loghz (temporal, 1000 bins)
                subtype_logits (2), lat_logits (3) (segment-level)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """Residual block: Conv(stride) -> BN -> GELU -> Dropout -> Conv(1) -> BN + skip -> GELU."""

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1,
                               padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection (project if needed)
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class FusionBlock(nn.Module):
    """Fuse upsampled + skip connection after concatenation."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class PDNetV2(nn.Module):
    """
    Unified periodic discharge detection network.

    Input: (B, 18, 2000) - 10 seconds at 200 Hz, 18 bipolar channels
    Output:
        event_logits  (B, 1000) - discharge center log-prob at 100 Hz
        active_logits (B, 1000) - PD-active regime log-prob at 100 Hz
        freq_loghz    (B, 1000) - local log-frequency at 100 Hz
        subtype_logits (B, 2)   - LPD=0, GPD=1
        lat_logits     (B, 3)   - left=0, right=1, unknown=2
    """

    def __init__(self):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(18, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualBlock1D(64, 64, kernel_size=7, stride=1),
        )

        # ── Encoder ───────────────────────────────────────────────────────
        # Each block downsamples by stride=2
        self.enc1 = ResidualBlock1D(64, 96, kernel_size=7, stride=2)    # (B, 96, 1000)
        self.enc2 = ResidualBlock1D(96, 128, kernel_size=7, stride=2)   # (B, 128, 500)
        self.enc3 = ResidualBlock1D(128, 192, kernel_size=7, stride=2)  # (B, 192, 250)
        self.enc4 = ResidualBlock1D(192, 256, kernel_size=7, stride=2)  # (B, 256, 125)

        # ── Bottleneck: Transformer ────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512,
            dropout=0.1, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Decoder ───────────────────────────────────────────────────────
        # up3: 256 -> 192, fuse with e3 skip (192) -> 192, output (B, 192, 250)
        self.up3 = nn.ConvTranspose1d(256, 192, kernel_size=2, stride=2)
        self.fuse3 = FusionBlock(192 + 192, 192)

        # up2: 192 -> 128, fuse with e2 skip (128) -> 128, output (B, 128, 500)
        self.up2 = nn.ConvTranspose1d(192, 128, kernel_size=2, stride=2)
        self.fuse2 = FusionBlock(128 + 128, 128)

        # up1: 128 -> 96, fuse with e1 skip (96) -> 96, output (B, 96, 1000)
        self.up1 = nn.ConvTranspose1d(128, 96, kernel_size=2, stride=2)
        self.fuse1 = FusionBlock(96 + 96, 96)

        # ── Temporal output heads (from d1, shape (B, 96, 1000)) ──────────
        self.event_head = nn.Sequential(
            nn.Conv1d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )
        self.active_head = nn.Sequential(
            nn.Conv1d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )
        self.freq_head = nn.Sequential(
            nn.Conv1d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, 1, 1),
        )

        # ── Segment heads (from bottleneck mean pool) ─────────────────────
        self.seg_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.subtype_head = nn.Linear(128, 2)
        self.lat_head = nn.Linear(128, 3)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 18, 2000) EEG tensor

        Returns:
            event_logits  (B, 1000)
            active_logits (B, 1000)
            freq_loghz    (B, 1000)
            subtype_logits (B, 2)
            lat_logits     (B, 3)
        """
        # Stem
        s = self.stem(x)          # (B, 64, 2000)

        # Encoder
        e1 = self.enc1(s)         # (B, 96, 1000)
        e2 = self.enc2(e1)        # (B, 128, 500)
        e3 = self.enc3(e2)        # (B, 192, 250)
        e4 = self.enc4(e3)        # (B, 256, 125)

        # Bottleneck: Transformer expects (B, T, C)
        # MPS may have issues with Transformer — run on CPU if needed
        b_in = e4.permute(0, 2, 1)  # (B, 125, 256)
        try:
            b_out = self.transformer(b_in)
        except Exception:
            # Fallback: run transformer on CPU
            cpu_in = b_in.cpu()
            b_out_cpu = self.transformer.cpu()(cpu_in)
            b_out = b_out_cpu.to(e4.device)
            self.transformer = self.transformer.to(e4.device)
        b_out = b_out.permute(0, 2, 1)  # (B, 256, 125)

        # Decoder
        d3 = self.up3(b_out)                            # (B, 192, 250)
        # Handle potential size mismatch
        if d3.shape[2] != e3.shape[2]:
            d3 = F.interpolate(d3, size=e3.shape[2], mode='nearest')
        d3 = self.fuse3(torch.cat([d3, e3], dim=1))    # (B, 192, 250)

        d2 = self.up2(d3)                               # (B, 128, 500)
        if d2.shape[2] != e2.shape[2]:
            d2 = F.interpolate(d2, size=e2.shape[2], mode='nearest')
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))    # (B, 128, 500)

        d1 = self.up1(d2)                               # (B, 96, 1000)
        if d1.shape[2] != e1.shape[2]:
            d1 = F.interpolate(d1, size=e1.shape[2], mode='nearest')
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))    # (B, 96, 1000)

        # Temporal heads
        event_logits = self.event_head(d1).squeeze(1)   # (B, 1000)
        active_logits = self.active_head(d1).squeeze(1)  # (B, 1000)
        freq_loghz = self.freq_head(d1).squeeze(1)       # (B, 1000)

        # Segment heads (mean pool over bottleneck time)
        pooled = b_out.mean(dim=2)                       # (B, 256)
        seg_feat = self.seg_mlp(pooled)                  # (B, 128)
        subtype_logits = self.subtype_head(seg_feat)     # (B, 2)
        lat_logits = self.lat_head(seg_feat)              # (B, 3)

        return event_logits, active_logits, freq_loghz, subtype_logits, lat_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = PDNetV2()
    print(f"PDNetV2 parameters: {count_parameters(model):,}")

    x = torch.randn(2, 18, 2000)
    outputs = model(x)
    names = ['event_logits', 'active_logits', 'freq_loghz', 'subtype_logits', 'lat_logits']
    for name, out in zip(names, outputs):
        print(f"  {name}: {out.shape}")
