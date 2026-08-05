"""Deeper 3D U-Net with bottleneck multi-head self-attention.

Written for NeuroScan AI.  The architecture is intentionally different from the
linked neuro-voxel repository: it uses residual GroupNorm convolution blocks,
four encoder levels, a spatial positional convolution, PyTorch MultiheadAttention
at the compact bottleneck, and the project's existing Streamlit/Plotly pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.projection = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.projection(x)
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class BottleneckMultiHeadAttention3D(nn.Module):
    """Global self-attention over the compact D×H×W bottleneck token grid."""

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.positional_conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        x = x + self.positional_conv(x)
        tokens = x.flatten(2).transpose(1, 2)  # [B, D*H*W, C]

        normalised = self.norm1(tokens)
        attended, _ = self.attention(
            normalised,
            normalised,
            normalised,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.feed_forward(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.fuse = ResidualConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class AttentionUNet3D(nn.Module):
    """Four-level residual 3D U-Net with bottleneck multi-head attention."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 16,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.enc1 = ResidualConvBlock3D(in_channels, c1)
        self.enc2 = ResidualConvBlock3D(c1, c2)
        self.enc3 = ResidualConvBlock3D(c2, c3)
        self.enc4 = ResidualConvBlock3D(c3, c4, dropout=0.05)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = ResidualConvBlock3D(c4, c5, dropout=0.10)
        self.global_attention = BottleneckMultiHeadAttention3D(
            channels=c5,
            num_heads=num_heads,
            dropout=0.10,
        )

        self.dec4 = UpBlock3D(c5, c4, c4)
        self.dec3 = UpBlock3D(c4, c3, c3)
        self.dec2 = UpBlock3D(c3, c2, c2)
        self.dec1 = UpBlock3D(c2, c1, c1)
        self.output = nn.Conv3d(c1, out_channels, kernel_size=1)

        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bottleneck = self.bottleneck(self.pool(e4))
        bottleneck = self.global_attention(bottleneck)

        x = self.dec4(bottleneck, e4)
        x = self.dec3(x, e3)
        x = self.dec2(x, e2)
        x = self.dec1(x, e1)
        return self.output(x)


if __name__ == "__main__":
    model = AttentionUNet3D()
    sample = torch.randn(1, 4, 32, 160, 160)
    with torch.no_grad():
        output = model(sample)
    print("Input:", tuple(sample.shape))
    print("Output:", tuple(output.shape))
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
