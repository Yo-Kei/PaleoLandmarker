from paleolandmarker.models.unet import (
    LargeLandmarkUNet,
    MediumLandmarkUNet,
    SmallLandmarkUNet,
    TinyLandmarkUNet,
    build_original_unet,
)


FORMAL_MODEL_ARCHITECTURE = "context_resize"
ORIGINAL_MODEL_ARCHITECTURE = "original_unet"

MODEL_REGISTRY = {
    "tiny": TinyLandmarkUNet,
    "small": SmallLandmarkUNet,
    "medium": MediumLandmarkUNet,
    "large": LargeLandmarkUNet,
}

def build_model(cfg):
    model_cfg = cfg.get("model", "tiny")
    name = model_cfg.get("name", "tiny") if isinstance(model_cfg, dict) else model_cfg
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model: {name}; choose from {sorted(MODEL_REGISTRY)}")
    architecture = cfg.get(
        "model_architecture",
        model_cfg.get("architecture", FORMAL_MODEL_ARCHITECTURE)
        if isinstance(model_cfg, dict)
        else FORMAL_MODEL_ARCHITECTURE,
    )
    if architecture == FORMAL_MODEL_ARCHITECTURE:
        return MODEL_REGISTRY[name](cfg["num_landmarks"])
    if architecture == ORIGINAL_MODEL_ARCHITECTURE:
        return build_original_unet(name, cfg["num_landmarks"])
    raise ValueError(
        f"unknown model architecture: {architecture}; choose "
        f"{FORMAL_MODEL_ARCHITECTURE!r} or {ORIGINAL_MODEL_ARCHITECTURE!r}"
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
