import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention2D(nn.Module):
    """
    Multi-head self-attention applied at the U-Net bottleneck.

    Input:
        [B, C, H, W]

    Attention expects:
        [B, tokens, C]

    Therefore H*W pixels become tokens.
    """

    def __init__(self, channels=256, num_heads=8, dropout=0.1):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        # Depth-wise positional convolution gives attention
        # some information about spatial position.
        self.positional_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False
        )

        self.norm1 = nn.LayerNorm(channels)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(channels)

        self.feed_forward = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, channels, height, width = x.shape

        # Add learnable spatial positional information
        x = x + self.positional_conv(x)

        # [B,C,H,W] -> [B,H*W,C]
        tokens = x.flatten(2).transpose(1, 2)

        normalized = self.norm1(tokens)

        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False
        )

        # Residual connection
        tokens = tokens + attended

        # Transformer-style feed-forward residual
        tokens = tokens + self.feed_forward(
            self.norm2(tokens)
        )

        # [B,H*W,C] -> [B,C,H,W]
        return (
            tokens
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )


class AttentionUNet2D(nn.Module):
    """
    Existing NeuroScan 2D U-Net structure +
    multi-head self-attention at the bottleneck.
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_heads=8
    ):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        self.global_attention = MultiHeadAttention2D(
            channels=256,
            num_heads=num_heads,
            dropout=0.1
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            256, 128,
            kernel_size=2,
            stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64,
            kernel_size=2,
            stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32,
            kernel_size=2,
            stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        self.final = nn.Conv2d(
            32,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # NEW
        b = self.global_attention(b)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)


if __name__ == "__main__":
    model = AttentionUNet2D()

    x = torch.randn(1, 1, 256, 256)

    with torch.no_grad():
        y = model(x)

    print("Input:", x.shape)
    print("Output:", y.shape)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )
