"""
EEG Frequency Estimation Model: Depthwise-Separable 1D CNN with dual heads.

Architecture:
- Backbone: 4 depthwise-separable conv blocks (18→32→64→128→128 channels)
- Head A (eventness): Transposed convs back to (1, 2000) — discharge probability per timestep
- Head B (frequency): 3 expert-specific heads predicting log(freq)
- Classification head (Phase 1 only): Binary LPD/GPD classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1D convolution: depthwise + pointwise."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, stride=stride,
                                    padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ConvBlock(nn.Module):
    """Single conv block: DepthwiseSepConv → BN → GELU → Dropout."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=2, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.bn(self.conv(x))))


class EEGBackbone(nn.Module):
    """4-block depthwise-separable CNN backbone for 18-channel bipolar EEG."""
    def __init__(self, in_channels=18, dropout=0.1):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 32, kernel_size=51, stride=2, dropout=dropout)
        self.block2 = ConvBlock(32, 64, kernel_size=25, stride=2, dropout=dropout)
        self.block3 = ConvBlock(64, 128, kernel_size=13, stride=2, dropout=dropout)
        self.block4 = ConvBlock(128, 128, kernel_size=7, stride=2, dropout=dropout)

    def forward(self, x):
        """Input: (batch, 18, 2000) → Output: (batch, 128, 125)"""
        x = self.block1(x)   # (B, 32, 1000)
        x = self.block2(x)   # (B, 64, 500)
        x = self.block3(x)   # (B, 128, 250)
        x = self.block4(x)   # (B, 128, 125)
        return x


class ClassificationHead(nn.Module):
    """Binary LPD/GPD classification head (Phase 1)."""
    def __init__(self, in_features=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """Input: (batch, 128, T) → Output: (batch, 1)"""
        x = self.pool(x).squeeze(-1)  # (B, 128)
        return self.fc(x)             # (B, 1)


class EventnessHead(nn.Module):
    """Eventness trace decoder: upsamples back to original temporal resolution."""
    def __init__(self, in_channels=128, target_len=2000):
        super().__init__()
        self.target_len = target_len
        self.up1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.up3 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.up4 = nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        """Input: (batch, 128, 125) → Output: (batch, 1, 2000)"""
        x = self.act(self.bn1(self.up1(x)))  # (B, 64, 250)
        x = self.act(self.bn2(self.up2(x)))  # (B, 32, 500)
        x = self.act(self.bn3(self.up3(x)))  # (B, 16, 1000)
        x = self.up4(x)                       # (B, 1, 2000)
        # Interpolate to exact target length if needed
        if x.shape[-1] != self.target_len:
            x = F.interpolate(x, size=self.target_len, mode='linear', align_corners=False)
        return torch.sigmoid(x)


class FrequencyHead(nn.Module):
    """Single frequency regression head: predicts log(frequency)."""
    def __init__(self, in_features=128, dropout=0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_features, 64)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """Input: (batch, 128, T) → Output: (batch, 1) — log(frequency)"""
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.drop(self.act(self.fc1(x)))  # (B, 64)
        return self.fc2(x)  # (B, 1)


class EEGClassifier(nn.Module):
    """Phase 1: Classification model (LPD vs GPD)."""
    def __init__(self, in_channels=18, dropout=0.1):
        super().__init__()
        self.backbone = EEGBackbone(in_channels, dropout)
        self.head = ClassificationHead(128)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class EEGFrequencyModel(nn.Module):
    """Phase 2: Frequency estimation with eventness and per-expert heads."""
    def __init__(self, in_channels=18, dropout=0.1, n_experts=3):
        super().__init__()
        self.backbone = EEGBackbone(in_channels, dropout)
        self.eventness_head = EventnessHead(128, target_len=2000)
        self.freq_heads = nn.ModuleList([FrequencyHead(128, dropout=0.2) for _ in range(n_experts)])

    def forward(self, x):
        """
        Input: (batch, 18, 2000)
        Returns:
            eventness: (batch, 1, 2000) — discharge probability per timestep
            freq_preds: list of (batch, 1) — log(frequency) per expert
        """
        features = self.backbone(x)
        eventness = self.eventness_head(features)
        freq_preds = [head(features) for head in self.freq_heads]
        return eventness, freq_preds

    def load_pretrained_backbone(self, classifier_state_dict):
        """Load backbone weights from a pretrained classifier."""
        backbone_dict = {}
        for k, v in classifier_state_dict.items():
            if k.startswith('backbone.'):
                backbone_dict[k.replace('backbone.', '')] = v
        self.backbone.load_state_dict(backbone_dict, strict=True)
        print(f"Loaded {len(backbone_dict)} backbone parameters from pretrained classifier.")

    def freeze_early_blocks(self):
        """Freeze blocks 1-2, keep blocks 3-4 trainable."""
        for param in self.backbone.block1.parameters():
            param.requires_grad = False
        for param in self.backbone.block2.parameters():
            param.requires_grad = False
        print("Froze backbone blocks 1-2.")
