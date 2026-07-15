import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    A repeated convolution block used throughout the U-Net model.

    The block applies:
    1. Convolution
    2. Batch normalization
    3. ReLU activation
    4. A second convolution
    5. Batch normalization
    6. ReLU activation

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Pass input data through the double convolution block.

        Args:
            x: Input tensor.

        Returns:
            Tensor after two convolution layers.
        """
        return self.layers(x)


class UNet(nn.Module):
    """
    U-Net architecture for binary brain tumor segmentation.

    The encoder reduces the spatial size of the image while learning deeper
    features. The decoder restores the original size and combines decoder
    features with encoder features using skip connections.

    Args:
        in_channels: Number of image input channels. Default is 1 for grayscale MRI.
        out_channels: Number of output channels. Default is 1 for binary segmentation.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # Encoder path
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Middle part of the network
        self.bottleneck = DoubleConv(128, 256)

        # Decoder path
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # Final 1x1 convolution produces one prediction score per pixel
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Perform the forward pass of the U-Net model.

        Args:
            x: Input MRI tensor of shape [batch, channels, height, width].

        Returns:
            Raw segmentation output before sigmoid activation.
        """

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
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