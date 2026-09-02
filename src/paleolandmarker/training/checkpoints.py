from pathlib import Path

import torch


def load_compatible(model, checkpoint, device):
    source = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    target = model.state_dict()
    usable = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(usable, strict=False)
    print(f"loaded {len(usable)}/{len(target)} tensors from {checkpoint}; reset={len(missing)} unexpected={len(unexpected)}")
    return len(usable), len(target)


def save_checkpoint(path, model, epoch, val_error, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_error": val_error, "config": cfg}, path)

