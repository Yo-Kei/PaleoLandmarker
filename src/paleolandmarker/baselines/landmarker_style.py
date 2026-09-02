"""Landmarker stacked-hourglass baseline.

The architecture follows the ``Hourglass`` and ``StackedHourglass`` modules in
the code release accompanying Jonkers et al. (2025), landmarker v0.2.1:
https://github.com/ElsevierSoftwareX/SOFTX-D-24-00590

Only the channel widths and number of stacks are scaled down to match the
parameter budget of PaleoLandmarker-Medium. The down/skip/neck/up topology,
nearest-neighbour upsampling, per-stack outputs, and prediction-to-input
feedback follow the source implementation.
"""

import torch
from torch import nn
import torch.nn.functional as F

from paleolandmarker.baselines.torch_heatmap import TorchHeatmapBaseline


class LandmarkerResidualUnit(nn.Module):
    """Two-subunit residual block matching MONAI ResidualUnit defaults."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=True)
        )

    def forward(self, x):
        return self.body(x) + self.residual(x)


class ConvBatchNorm(nn.Sequential):
    """Landmarker code's 1x1 Convolution with ``adn_ordering='N'``."""

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )


class LandmarkerHourglass(nn.Module):
    """Multi-resolution hourglass ported from landmarker v0.2.1."""

    def __init__(self, in_channels, out_channels, channels):
        super().__init__()
        channels = tuple(int(channel) for channel in channels)
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("landmarker channels must be a non-empty positive sequence")
        self.channels = channels
        decoder_channels = channels[-1]

        self.down_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        previous_channels = in_channels
        for channel in channels:
            self.down_blocks.append(
                nn.Sequential(
                    LandmarkerResidualUnit(previous_channels, channel),
                    nn.MaxPool2d(2),
                )
            )
            self.skip_blocks.append(
                LandmarkerResidualUnit(channel, decoder_channels)
            )
            previous_channels = channel

        self.neck_block = nn.Sequential(
            LandmarkerResidualUnit(decoder_channels, decoder_channels),
            ConvBatchNorm(decoder_channels, decoder_channels),
            ConvBatchNorm(decoder_channels, decoder_channels),
        )
        self.up_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    LandmarkerResidualUnit(decoder_channels, decoder_channels),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                )
                for _ in channels
            ]
        )
        self.last_layer = nn.Sequential(
            ConvBatchNorm(decoder_channels, decoder_channels // 2),
            nn.Conv2d(decoder_channels // 2, out_channels, 1, bias=True),
        )

    def forward(self, x):
        down_outputs = []
        for down_block, skip_block in zip(self.down_blocks, self.skip_blocks):
            x = down_block(x)
            down_outputs.append(skip_block(x))
        x = self.neck_block(x)
        for up_block, skip_output in zip(
            reversed(self.up_blocks), reversed(down_outputs)
        ):
            x = up_block(x + skip_output)
        return self.last_layer(x)


class StackedHourglassHeatmap(nn.Module):
    """Parameter-matched port of landmarker v0.2.1 ``StackedHourglass``."""

    def __init__(
        self,
        num_landmarks,
        channels=(11, 22, 44, 86),
        stacks=2,
        output_size=112,
    ):
        super().__init__()
        if int(stacks) < 1:
            raise ValueError("landmarker_stacks must be at least one")
        self.output_size = int(output_size)
        self.stacks = nn.ModuleList(
            [
                LandmarkerHourglass(
                    in_channels=1,
                    out_channels=num_landmarks,
                    channels=channels,
                )
                for _ in range(int(stacks))
            ]
        )
        self.output_to_feature = nn.ModuleList(
            [ConvBatchNorm(num_landmarks, 1) for _ in range(int(stacks) - 1)]
        )

    def forward(self, x):
        # The source Hourglass preserves its input resolution. Our shared
        # benchmark uses 224px images and 112px target heatmaps, so resize the
        # input without adding a non-source learnable stem.
        if x.shape[-2:] != (self.output_size, self.output_size):
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        source = x
        next_x = x
        outputs = []
        for index, stack in enumerate(self.stacks):
            prediction = stack(next_x)
            outputs.append(prediction)
            if index < len(self.output_to_feature):
                next_x = source + self.output_to_feature[index](prediction)
        return outputs


class LandmarkerStyleBaseline(TorchHeatmapBaseline):
    """Landmarker v0.2.1 stacked-hourglass initialized from scratch."""

    name = "landmarker_scratch"

    def build_model(self, cfg):
        return StackedHourglassHeatmap(
            num_landmarks=cfg["num_landmarks"],
            channels=cfg.get("landmarker_channels", (11, 22, 44, 86)),
            stacks=cfg.get("landmarker_stacks", 2),
            output_size=cfg.get("heatmap_size", 112),
        )


class LandmarkerSyntheticBaseline(LandmarkerStyleBaseline):
    """Landmarker stacked-hourglass initialized by Synthetic pretraining."""

    name = "landmarker_synthetic"

    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["synthetic_pretrain"] = True
        super().__init__(cfg)
