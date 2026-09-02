import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from paleolandmarker.baselines.torch_heatmap import TorchHeatmapBaseline


class DLCMobileNetV3SmallHeatmap(nn.Module):
    """MobileNetV3-Small encoder plus deconvolution heatmap head."""

    def __init__(self, num_landmarks, pretrained=False, width_mult=1.0):
        super().__init__()
        if float(width_mult) != 1.0:
            raise ValueError(
                "DeepLabCut-style comparisons use one shared MobileNetV3-Small "
                "1.0x architecture; dlc_width_mult must be 1.0"
            )
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_small(
            weights=weights,
            width_mult=1.0,
        ).features
        last_channels = self.features[-1].out_channels
        self.head = nn.Sequential(
            nn.ConvTranspose2d(last_channels, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_landmarks, 1),
        )
        self.pretrained = pretrained
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        # Keep preprocessing identical across scratch, Synthetic, and
        # ImageNet initialization so initialization is the only comparison
        # variable. Synthetic checkpoints are trained with this transform too.
        x = (x - self.mean) / self.std
        return self.head(self.features(x))


class DeepLabCutStyleBaseline(TorchHeatmapBaseline):
    """DeepLabCut-like transfer baseline without requiring DLC project files."""

    name = "deeplabcut_scratch"

    def build_model(self, cfg):
        backbone = cfg.get("dlc_backbone", "mobilenet_v3_small")
        if backbone != "mobilenet_v3_small":
            raise ValueError(
                f"unsupported DeepLabCut-style backbone: {backbone!r}"
            )
        return DLCMobileNetV3SmallHeatmap(
            num_landmarks=cfg["num_landmarks"],
            pretrained=cfg.get("dlc_pretrained", False),
            width_mult=cfg.get("dlc_width_mult", 1.0),
        )


class DeepLabCutImageNetBaseline(DeepLabCutStyleBaseline):
    """DeepLabCut-style baseline initialized from ImageNet MobileNetV3-Small."""

    name = "deeplabcut_imagenet"

    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["dlc_pretrained"] = True
        cfg["dlc_width_mult"] = 1.0
        super().__init__(cfg)


class DeepLabCutSyntheticBaseline(DeepLabCutStyleBaseline):
    """DeepLabCut-style baseline initialized by all-domain synthetic pretraining."""

    name = "deeplabcut_synthetic"

    def __init__(self, cfg):
        cfg = dict(cfg)
        cfg["dlc_pretrained"] = False
        cfg["dlc_width_mult"] = 1.0
        cfg["synthetic_pretrain"] = True
        super().__init__(cfg)
