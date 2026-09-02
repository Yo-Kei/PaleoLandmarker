import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class LightweightDilatedContext(nn.Module):
    """Low-cost multi-scale context at the bottleneck."""

    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        if len(dilations) != 3:
            raise ValueError("the lightweight context module requires three branches")
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        context = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.relu(x + self.fuse(context))


class LightweightResizeProjection(nn.Module):
    """Bilinear resize followed by depthwise/pointwise projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.projection(x)


class _EncoderControlMixin:
    def encoder_parameters(self):
        for module in self.encoder_modules():
            yield from module.parameters()

    def set_encoder_trainable(self, trainable):
        self.encoder_frozen = not trainable
        for module in self.encoder_modules():
            for param in module.parameters():
                param.requires_grad = trainable

    def freeze_encoder(self):
        self.set_encoder_trainable(False)

    def unfreeze_encoder(self):
        self.set_encoder_trainable(True)

    def train(self, mode=True):
        super().train(mode)
        if mode and getattr(self, "encoder_frozen", False):
            for module in self.encoder_modules():
                module.eval()
        return self


class OriginalLandmarkUNet(_EncoderControlMixin, nn.Module):
    """Original transposed-convolution U-Net retained for controlled ablations."""

    def __init__(self, num_landmarks, channels, dropout=0.08, residual_blocks=2):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(1, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = nn.Sequential(
            ConvBlock(c3, c4),
            *[ResidualBlock(c4) for _ in range(residual_blocks)],
        )
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(c4, c3, 2, 2)
        self.dec2 = ConvBlock(c3 * 2, c3)
        self.up1 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.dec1 = ConvBlock(c2 * 2, c2)
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(c2, num_landmarks, 1))

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        d = self.dec2(torch.cat([self.up2(z), c], dim=1))
        return self.head(self.dec1(torch.cat([self.up1(d), b], dim=1)))

    def encoder_modules(self):
        return (self.enc1, self.enc2, self.enc3, self.bottleneck)


class LandmarkUNet(_EncoderControlMixin, nn.Module):
    """Official context-and-resize heatmap U-Net used by all model sizes."""

    def __init__(self, num_landmarks, channels, dropout=0.08, residual_blocks=2):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.enc1 = ConvBlock(1, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = nn.Sequential(
            ConvBlock(c3, c4),
            *[ResidualBlock(c4) for _ in range(residual_blocks)],
        )
        self.context = LightweightDilatedContext(c4, dilations=(1, 2, 4))
        self.pool = nn.MaxPool2d(2)
        self.up2 = LightweightResizeProjection(c4, c3)
        self.dec2 = ConvBlock(c3 * 2, c3)
        self.up1 = LightweightResizeProjection(c3, c2)
        self.dec1 = ConvBlock(c2 * 2, c2)
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(c2, num_landmarks, 1))

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        z = self.context(self.bottleneck(self.pool(c)))
        d = self.dec2(torch.cat([self.up2(z), c], dim=1))
        return self.head(self.dec1(torch.cat([self.up1(d), b], dim=1)))

    def encoder_modules(self):
        return (self.enc1, self.enc2, self.enc3, self.bottleneck, self.context)


MODEL_SIZE_SPECS = {
    "tiny": ((24, 48, 72, 96), 0.08, 2),
    "small": ((32, 64, 96, 128), 0.08, 2),
    "medium": ((52, 104, 156, 208), 0.08, 2),
    "large": ((72, 144, 216, 288), 0.10, 3),
}


class TinyLandmarkUNet(LandmarkUNet):
    def __init__(self, num_landmarks=12):
        super().__init__(num_landmarks, *MODEL_SIZE_SPECS["tiny"])


class SmallLandmarkUNet(LandmarkUNet):
    def __init__(self, num_landmarks=12):
        super().__init__(num_landmarks, *MODEL_SIZE_SPECS["small"])


class MediumLandmarkUNet(LandmarkUNet):
    def __init__(self, num_landmarks=12):
        super().__init__(num_landmarks, *MODEL_SIZE_SPECS["medium"])


class LargeLandmarkUNet(LandmarkUNet):
    def __init__(self, num_landmarks=12):
        super().__init__(num_landmarks, *MODEL_SIZE_SPECS["large"])


def build_original_unet(model_size, num_landmarks=12):
    """Build the pre-context U-Net for an explicitly labelled ablation."""
    if model_size not in MODEL_SIZE_SPECS:
        raise ValueError(f"unknown original U-Net size: {model_size}")
    return OriginalLandmarkUNet(num_landmarks, *MODEL_SIZE_SPECS[model_size])
