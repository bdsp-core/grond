"""
HemiNet: U-Net with Transformer bottleneck for single-hemisphere discharge detection.

Design A (Experiment 1.1):
    Input:  (B, 8, 2000) — 8-channel hemisphere EEG, z-scored
    Output: event_logits  (B, 1000) — discharge center logits at 100 Hz
            active_logits (B, 1000) — active-regime logits at 100 Hz
            freq_logit    (B, 1)    — log-frequency (from bottleneck pool)

Architecture:
    Stem: Conv1d(8→48, k=15) → BN → GELU → ResBlock(48)
    Enc1: ResBlock(48→64, stride=2)    → (B, 64, 1000)
    Enc2: ResBlock(64→96, stride=2)    → (B, 96, 500)
    Enc3: ResBlock(96→128, stride=2)   → (B, 128, 250)
    Enc4: ResBlock(128→192, stride=2)  → (B, 192, 125)
    Bottleneck: 2× TransformerEncoderLayer (d=192, 6 heads)
    Dec3: Up(192→128) + skip(Enc3) → ResBlock
    Dec2: Up(128→96)  + skip(Enc2) → ResBlock
    Dec1: Up(96→64)   + skip(Enc1) → ResBlock
    event_head:  Conv(64→32→1) → (B, 1000)
    active_head: Conv(64→32→1) → (B, 1000)
    freq_head:   Linear(192→1) → (B, 1)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """1D residual block with optional stride.

    Conv1d(in, out, k=7, stride, pad=3) → BN → GELU → Dropout(0.1)
    → Conv1d(out, out, k=7, pad=3) → BN
    Skip: Conv1d(in, out, k=1, stride) if in!=out or stride!=1
    Output: GELU(main + skip)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()

        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act2(out + residual)


class UpBlock(nn.Module):
    """Upsample + skip connection + ResBlock.

    ConvTranspose1d(in_ch, out_ch, k=4, s=2, p=1) → BN → GELU
    → cat(skip) → ResBlock(out_ch + skip_ch, out_ch)
    """

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int, dropout: float = 0.1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.up_bn = nn.BatchNorm1d(out_ch)
        self.up_act = nn.GELU()
        self.res = ResBlock(out_ch + skip_ch, out_ch, stride=1, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.up_bn(x)
        x = self.up_act(x)
        # Align temporal dimension in case of rounding
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class HemiNet(nn.Module):
    """Single-hemisphere U-Net discharge detector (Design A).

    Parameters
    ----------
    in_channels : int
        Number of EEG channels (default 8, one hemisphere).
    dropout : float
        Dropout probability in ResBlocks (default 0.1).
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()

        # ── Stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            ResBlock(48, 48, stride=1, dropout=dropout),
        )

        # ── Encoder ────────────────────────────────────────────────────
        self.enc1 = ResBlock(48, 64, stride=2, dropout=dropout)    # → (B, 64, 1000)
        self.enc2 = ResBlock(64, 96, stride=2, dropout=dropout)    # → (B, 96, 500)
        self.enc3 = ResBlock(96, 128, stride=2, dropout=dropout)   # → (B, 128, 250)
        self.enc4 = ResBlock(128, 192, stride=2, dropout=dropout)  # → (B, 192, 125)

        # ── Bottleneck: 2× Transformer encoder layers ──────────────────
        # batch_first=True → input shape (B, T, D)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN (more stable training)
        )
        self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Decoder ────────────────────────────────────────────────────
        self.dec3 = UpBlock(192, 128, skip_ch=128, dropout=dropout)  # → (B, 128, 250)
        self.dec2 = UpBlock(128, 96, skip_ch=96, dropout=dropout)    # → (B, 96, 500)
        self.dec1 = UpBlock(96, 64, skip_ch=64, dropout=dropout)     # → (B, 64, 1000)

        # ── Output heads ───────────────────────────────────────────────
        self.event_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.active_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        # Frequency from bottleneck global average pool
        self.freq_head = nn.Linear(192, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000)

        Returns
        -------
        event_logits  : (B, 1000)
        active_logits : (B, 1000)
        freq_logit    : (B, 1)
        """
        # ── Stem ───────────────────────────────────────────────────────
        s = self.stem(x)            # (B, 48, 2000)

        # ── Encoder ────────────────────────────────────────────────────
        e1 = self.enc1(s)           # (B, 64, 1000)
        e2 = self.enc2(e1)          # (B, 96, 500)
        e3 = self.enc3(e2)          # (B, 128, 250)
        e4 = self.enc4(e3)          # (B, 192, 125)

        # ── Bottleneck ──────────────────────────────────────────────────
        # Transformer expects (B, T, D)
        b = e4.permute(0, 2, 1)     # (B, 125, 192)
        b = self.bottleneck(b)      # (B, 125, 192)
        b_seq = b.permute(0, 2, 1)  # (B, 192, 125) — for decoder skip

        # Global average pool for frequency head
        b_pool = b.mean(dim=1)      # (B, 192)
        freq_logit = self.freq_head(b_pool)  # (B, 1)

        # ── Decoder ────────────────────────────────────────────────────
        d3 = self.dec3(b_seq, e3)   # (B, 128, 250)
        d2 = self.dec2(d3, e2)      # (B, 96, 500)
        d1 = self.dec1(d2, e1)      # (B, 64, 1000)

        # ── Heads ──────────────────────────────────────────────────────
        event_logits = self.event_head(d1).squeeze(1)   # (B, 1000)
        active_logits = self.active_head(d1).squeeze(1) # (B, 1000)

        return event_logits, active_logits, freq_logit


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# Design B — Dilated Convolutions (Experiment 1.2)
# ══════════════════════════════════════════════════════════════════════════════

class DilatedResBlock(nn.Module):
    """1D residual block with dilation (no stride).

    Two conv layers: dilation applied to first conv, kernel_size=k.
    Skip connection is identity (channels must match).
    """

    def __init__(self, channels: int, dilation: int = 1, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               dilation=dilation, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                               dilation=1, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.act2(out + residual)


class HemiNetB(nn.Module):
    """Single-hemisphere U-Net discharge detector (Design B).

    Replaces Transformer bottleneck with 4 dilated ResBlocks (dilation=1,2,4,8).
    Fewer parameters (~800K) to reduce overfitting on small datasets.

    Parameters
    ----------
    in_channels : int
        Number of EEG channels (default 8, one hemisphere).
    dropout : float
        Dropout probability in ResBlocks (default 0.1).
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()

        # ── Stem ───────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            ResBlock(48, 48, stride=1, dropout=dropout),
        )

        # ── Encoder ────────────────────────────────────────────────────
        self.enc1 = ResBlock(48, 64, stride=2, dropout=dropout)    # → (B, 64, 1000)
        self.enc2 = ResBlock(64, 96, stride=2, dropout=dropout)    # → (B, 96, 500)
        self.enc3 = ResBlock(96, 128, stride=2, dropout=dropout)   # → (B, 128, 250)
        self.enc4 = ResBlock(128, 128, stride=2, dropout=dropout)  # → (B, 128, 125) keep 128

        # ── Bottleneck: 4 dilated ResBlocks (no Transformer) ───────────
        # Use kernel_size=5 to keep parameter count reasonable (~800K target)
        self.bottleneck = nn.Sequential(
            DilatedResBlock(128, dilation=1, kernel_size=5, dropout=dropout),
            DilatedResBlock(128, dilation=2, kernel_size=5, dropout=dropout),
            DilatedResBlock(128, dilation=4, kernel_size=5, dropout=dropout),
            DilatedResBlock(128, dilation=8, kernel_size=5, dropout=dropout),
        )

        # ── Decoder ────────────────────────────────────────────────────
        self.dec3 = UpBlock(128, 128, skip_ch=128, dropout=dropout)  # → (B, 128, 250)
        self.dec2 = UpBlock(128, 96, skip_ch=96, dropout=dropout)    # → (B, 96, 500)
        self.dec1 = UpBlock(96, 64, skip_ch=64, dropout=dropout)     # → (B, 64, 1000)

        # ── Output heads ───────────────────────────────────────────────
        self.event_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.active_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        # Frequency from bottleneck global average pool
        self.freq_head = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000)

        Returns
        -------
        event_logits  : (B, 1000)
        active_logits : (B, 1000)
        freq_logit    : (B, 1)
        """
        # ── Stem ───────────────────────────────────────────────────────
        s = self.stem(x)            # (B, 48, 2000)

        # ── Encoder ────────────────────────────────────────────────────
        e1 = self.enc1(s)           # (B, 64, 1000)
        e2 = self.enc2(e1)          # (B, 96, 500)
        e3 = self.enc3(e2)          # (B, 128, 250)
        e4 = self.enc4(e3)          # (B, 128, 125)

        # ── Bottleneck (dilated convs) ───────────────────────────────────
        b = self.bottleneck(e4)     # (B, 128, 125)

        # Global average pool for frequency head
        b_pool = b.mean(dim=-1)     # (B, 128)
        freq_logit = self.freq_head(b_pool)  # (B, 1)

        # ── Decoder ────────────────────────────────────────────────────
        d3 = self.dec3(b, e3)       # (B, 128, 250)
        d2 = self.dec2(d3, e2)      # (B, 96, 500)
        d1 = self.dec1(d2, e1)      # (B, 64, 1000)

        # ── Heads ──────────────────────────────────────────────────────
        event_logits = self.event_head(d1).squeeze(1)   # (B, 1000)
        active_logits = self.active_head(d1).squeeze(1) # (B, 1000)

        return event_logits, active_logits, freq_logit


# ══════════════════════════════════════════════════════════════════════════════
# Design D — Neural Wrapper Around Existing Pipeline (Experiment 1.4)
# ══════════════════════════════════════════════════════════════════════════════

class SmallUNet(nn.Module):
    """Small 3-level U-Net operating on raw EEG channels.

    Enc: 8 → 32 → 64 → 64 (stride=2 each level)
    Dec: 64 → 32 → 8 with skip connections
    Output: (B, 8, 2000) learned evidence
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.enc1 = ResBlock(in_channels, 32, stride=2, dropout=dropout)  # → (B, 32, 1000)
        self.enc2 = ResBlock(32, 64, stride=2, dropout=dropout)            # → (B, 64, 500)
        self.enc3 = ResBlock(64, 64, stride=2, dropout=dropout)            # → (B, 64, 250)

        self.dec2 = UpBlock(64, 64, skip_ch=64, dropout=dropout)           # → (B, 64, 500)
        self.dec1 = UpBlock(64, 32, skip_ch=32, dropout=dropout)           # → (B, 32, 1000)
        # Final upsample to restore 2000 samples
        self.out_up = nn.Sequential(
            nn.ConvTranspose1d(32, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)           # (B, 32, 1000)
        e2 = self.enc2(e1)          # (B, 64, 500)
        e3 = self.enc3(e2)          # (B, 64, 250)

        d2 = self.dec2(e3, e2)      # (B, 64, 500)
        d1 = self.dec1(d2, e1)      # (B, 32, 1000)
        out = self.out_up(d1)       # (B, 8, 2000)
        return out


class HemiNetD(nn.Module):
    """Neural Wrapper Around Existing Pipeline (Design D).

    Combines frozen handcrafted HPP evidence (pointiness+TKEO) with a small
    learnable U-Net, then combines via channel attention into event/active/freq
    outputs.

    CRITICAL: Uses _compute_channel_evidence from label_pipeline.hpp_discharge_marking
    (identical to the existing pipeline) — computed once per segment, no gradient.

    Parameters
    ----------
    in_channels : int
        EEG channels per hemisphere (default 8).
    dropout : float
        Dropout probability (default 0.1).
    fs : float
        EEG sample rate in Hz (default 200).
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1, fs: float = 200.0):
        super().__init__()
        self.in_channels = in_channels
        self.fs = fs

        # Branch 2: small trainable U-Net on raw EEG
        self.small_unet = SmallUNet(in_channels=in_channels, dropout=dropout)

        # Combination: cat(hpp_evidence(8ch), learned_evidence(8ch)) → (B, 16, 2000)
        # Pointwise channel attention → (B, 1, 2000)
        self.combiner = nn.Sequential(
            nn.Conv1d(in_channels * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )

        # Output heads at 100Hz (after 2× downsample)
        self.event_head = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.active_head = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        # Frequency from global pool of combined evidence
        self.freq_head = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # 2× average pool for downsampling to 100Hz
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _compute_hpp_evidence(self, x_np: 'np.ndarray') -> 'np.ndarray':
        """Compute HPP evidence for each channel using the existing pipeline function.

        Parameters
        ----------
        x_np : (8, 2000) numpy float32

        Returns
        -------
        evidence : (8, 2000) float32
        """
        import sys
        from pathlib import Path
        code_dir = Path(__file__).resolve().parent.parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from label_pipeline.hpp_discharge_marking import _compute_channel_evidence

        n_ch, n_t = x_np.shape
        evidence = np.zeros((n_ch, n_t), dtype=np.float32)
        for ch in range(n_ch):
            ev = _compute_channel_evidence(x_np[ch].astype(np.float64), self.fs)
            evidence[ch] = ev.astype(np.float32)
        return evidence

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000)

        Returns
        -------
        event_logits  : (B, 1000)
        active_logits : (B, 1000)
        freq_logit    : (B, 1)
        """
        import numpy as np

        B = x.shape[0]
        device = x.device

        # ── Branch 1: frozen HPP evidence (no gradient) ──────────────
        x_cpu = x.detach().cpu().numpy()  # (B, 8, 2000) numpy
        hpp_list = []
        for b in range(B):
            ev = self._compute_hpp_evidence(x_cpu[b])  # (8, 2000)
            hpp_list.append(ev)
        hpp_evidence = torch.from_numpy(np.stack(hpp_list, axis=0)).to(device)  # (B, 8, 2000)

        # ── Branch 2: trainable small U-Net ──────────────────────────
        learned_evidence = self.small_unet(x)  # (B, 8, 2000)

        # ── Combination ──────────────────────────────────────────────
        combined = torch.cat([hpp_evidence, learned_evidence], dim=1)  # (B, 16, 2000)
        fused = self.combiner(combined)  # (B, 1, 2000)

        # ── Downsample to 100Hz ───────────────────────────────────────
        fused_ds = self.downsample(fused)  # (B, 1, 1000)

        # ── Frequency from global pool ────────────────────────────────
        freq_pool = fused_ds.mean(dim=-1)  # (B, 1)
        freq_logit = self.freq_head(freq_pool)  # (B, 1)

        # ── Output heads ─────────────────────────────────────────────
        event_logits = self.event_head(fused_ds).squeeze(1)   # (B, 1000)
        active_logits = self.active_head(fused_ds).squeeze(1) # (B, 1000)

        return event_logits, active_logits, freq_logit


# ══════════════════════════════════════════════════════════════════════════════
# Design A + MAE Pretraining (Experiment 1.5)
# ══════════════════════════════════════════════════════════════════════════════

class HemiNetEncoder(nn.Module):
    """Shared encoder from Design A, extracted as a standalone module.

    Used by both the MAE pretraining phase and HemiNetPretrained fine-tuning.
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 48, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(48),
            nn.GELU(),
            ResBlock(48, 48, stride=1, dropout=dropout),
        )
        # Encoder
        self.enc1 = ResBlock(48, 64, stride=2, dropout=dropout)    # → (B, 64, 1000)
        self.enc2 = ResBlock(64, 96, stride=2, dropout=dropout)    # → (B, 96, 500)
        self.enc3 = ResBlock(96, 128, stride=2, dropout=dropout)   # → (B, 128, 250)
        self.enc4 = ResBlock(128, 192, stride=2, dropout=dropout)  # → (B, 192, 125)

        # Bottleneck: 2× Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=192,
            nhead=6,
            dim_feedforward=384,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000) — optionally with masked patches

        Returns
        -------
        s   : (B, 48, 2000)  — stem output
        e1  : (B, 64, 1000)
        e2  : (B, 96, 500)
        e3  : (B, 128, 250)
        b_seq : (B, 192, 125) — bottleneck output (for decoder)
        b_pool : (B, 192)     — global avg pool (for freq head)
        """
        s = self.stem(x)            # (B, 48, 2000)
        e1 = self.enc1(s)           # (B, 64, 1000)
        e2 = self.enc2(e1)          # (B, 96, 500)
        e3 = self.enc3(e2)          # (B, 128, 250)
        e4 = self.enc4(e3)          # (B, 192, 125)

        b = e4.permute(0, 2, 1)     # (B, 125, 192)
        b = self.bottleneck(b)      # (B, 125, 192)
        b_seq = b.permute(0, 2, 1)  # (B, 192, 125)
        b_pool = b.mean(dim=1)      # (B, 192)

        return s, e1, e2, e3, b_seq, b_pool


class MAEDecoder(nn.Module):
    """Lightweight decoder for masked autoencoder pretraining only.

    Reconstructs (B, 8, 2000) from bottleneck + skip features.
    This module is discarded after pretraining.
    """

    def __init__(self, out_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dec3 = UpBlock(192, 128, skip_ch=128, dropout=dropout)
        self.dec2 = UpBlock(128, 96, skip_ch=96, dropout=dropout)
        self.dec1 = UpBlock(96, 64, skip_ch=64, dropout=dropout)

        # Final up to 2000 samples
        self.out_up = nn.ConvTranspose1d(64, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, b_seq, e3, e2, e1):
        d3 = self.dec3(b_seq, e3)   # (B, 128, 250)
        d2 = self.dec2(d3, e2)      # (B, 96, 500)
        d1 = self.dec1(d2, e1)      # (B, 64, 1000)
        out = self.out_up(d1)       # (B, 8, 2000)
        return out


class HemiNetMAE(nn.Module):
    """Masked Autoencoder for pretraining the Design A encoder.

    Input:  (B, 8, 2000)
    Output: (B, 8, 2000) reconstruction

    Usage:
        mae = HemiNetMAE()
        recon, mask = mae(x)  # mask is (B, 1, 2000) bool: True = masked
        loss = F.mse_loss(recon[mask.expand_as(recon)], x[mask.expand_as(x)])
    """

    PATCH_SIZE = 100   # 0.5s at 200Hz
    MASK_RATIO = 0.20  # mask 20% of patches

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = HemiNetEncoder(in_channels=in_channels, dropout=dropout)
        self.decoder = MAEDecoder(out_channels=in_channels, dropout=dropout)

    def _make_mask(self, B: int, T: int, device) -> torch.Tensor:
        """Create a random patch mask.

        Returns
        -------
        mask : (B, 1, T) bool — True where masked (to be reconstructed)
        """
        n_patches = T // self.PATCH_SIZE
        n_mask = max(1, int(n_patches * self.MASK_RATIO))

        mask = torch.zeros(B, n_patches, dtype=torch.bool, device=device)
        for b in range(B):
            idx = torch.randperm(n_patches, device=device)[:n_mask]
            mask[b, idx] = True

        # Expand patches to sample-level: (B, n_patches) → (B, 1, T)
        # Each patch covers PATCH_SIZE consecutive samples
        mask_full = mask.repeat_interleave(self.PATCH_SIZE, dim=1)  # (B, T)
        mask_full = mask_full.unsqueeze(1)  # (B, 1, T)
        return mask_full

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000)

        Returns
        -------
        recon : (B, 8, 2000) — full reconstruction
        mask  : (B, 1, 2000) bool — True at masked positions
        """
        B, C, T = x.shape
        device = x.device

        # Create mask
        mask = self._make_mask(B, T, device)  # (B, 1, T)

        # Apply mask: zero out masked patches in the input
        x_masked = x * (~mask).float()  # (B, 8, 2000)

        # Encode (with masked input)
        s, e1, e2, e3, b_seq, b_pool = self.encoder(x_masked)

        # Decode → reconstruct
        recon = self.decoder(b_seq, e3, e2, e1)  # (B, 8, 2000)

        return recon, mask


class HemiNetPretrained(nn.Module):
    """Design A encoder initialized from MAE pretraining, with task heads.

    Load pretrained encoder weights via load_pretrained_encoder().
    Fine-tune with lower lr (1e-4 instead of 5e-4).
    """

    def __init__(self, in_channels: int = 8, dropout: float = 0.1):
        super().__init__()

        # Shared encoder (same as Design A / MAE)
        self.encoder = HemiNetEncoder(in_channels=in_channels, dropout=dropout)

        # Decoder (same as Design A)
        self.dec3 = UpBlock(192, 128, skip_ch=128, dropout=dropout)
        self.dec2 = UpBlock(128, 96, skip_ch=96, dropout=dropout)
        self.dec1 = UpBlock(96, 64, skip_ch=64, dropout=dropout)

        # Output heads
        self.event_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.active_head = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.freq_head = nn.Linear(192, 1)

        self._init_heads()

    def _init_heads(self):
        """Initialize only the task-specific heads (encoder initialized separately)."""
        for m in [self.dec3, self.dec2, self.dec1, self.event_head, self.active_head, self.freq_head]:
            for p in m.modules():
                if isinstance(p, nn.Conv1d):
                    nn.init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(p, nn.ConvTranspose1d):
                    nn.init.kaiming_normal_(p.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(p, nn.BatchNorm1d):
                    nn.init.ones_(p.weight)
                    nn.init.zeros_(p.bias)
                elif isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)

    def load_pretrained_encoder(self, checkpoint_path: str, device=None):
        """Load encoder weights from a saved MAE checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=device or 'cpu', weights_only=False)
        enc_state = {k.replace('encoder.', '', 1): v
                     for k, v in ckpt['model_state_dict'].items()
                     if k.startswith('encoder.')}
        missing, unexpected = self.encoder.load_state_dict(enc_state, strict=True)
        print(f"Loaded pretrained encoder: {len(enc_state)} tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 8, 2000)

        Returns
        -------
        event_logits  : (B, 1000)
        active_logits : (B, 1000)
        freq_logit    : (B, 1)
        """
        s, e1, e2, e3, b_seq, b_pool = self.encoder(x)

        freq_logit = self.freq_head(b_pool)  # (B, 1)

        d3 = self.dec3(b_seq, e3)   # (B, 128, 250)
        d2 = self.dec2(d3, e2)      # (B, 96, 500)
        d1 = self.dec1(d2, e1)      # (B, 64, 1000)

        event_logits = self.event_head(d1).squeeze(1)   # (B, 1000)
        active_logits = self.active_head(d1).squeeze(1) # (B, 1000)

        return event_logits, active_logits, freq_logit


if __name__ == '__main__':
    import numpy as np

    model = HemiNet(in_channels=8)
    n_params = count_parameters(model)
    print(f"HemiNet (Design A) parameters: {n_params:,}")

    model_b = HemiNetB(in_channels=8)
    print(f"HemiNetB (Design B) parameters: {count_parameters(model_b):,}")

    model_d = HemiNetD(in_channels=8)
    print(f"HemiNetD (Design D) parameters: {count_parameters(model_d):,}")

    model_pre = HemiNetPretrained(in_channels=8)
    print(f"HemiNetPretrained (Design A+MAE) parameters: {count_parameters(model_pre):,}")

    x = torch.randn(4, 8, 2000)
    ev, ac, fr = model(x)
    print(f"\nDesign A: event_logits={ev.shape}, active_logits={ac.shape}, freq_logit={fr.shape}")

    ev_b, ac_b, fr_b = model_b(x)
    print(f"Design B: event_logits={ev_b.shape}, active_logits={ac_b.shape}, freq_logit={fr_b.shape}")

    ev_pre, ac_pre, fr_pre = model_pre(x)
    print(f"Design A+MAE: event_logits={ev_pre.shape}")
