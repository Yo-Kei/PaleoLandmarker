import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from paleolandmarker.data.dataset import LandmarkDataset
from paleolandmarker.evaluation.decode import heatmaps_to_coords
from paleolandmarker.evaluation.metrics import compute_metrics
from paleolandmarker.evaluation.visualize import save_prediction
from paleolandmarker.models import build_model
from paleolandmarker.training.checkpoints import load_compatible, save_checkpoint
from paleolandmarker.training.losses import compute_loss
from paleolandmarker.utils import seed_everything


def get_device(requested=None):
    requested = (requested or os.environ.get("PALEOLANDMARKER_DEVICE", "auto")).lower()
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device {requested!r}; choose auto, cpu, cuda, or mps.")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Check the WSL NVIDIA driver "
                "and install a CUDA-enabled PyTorch build."
            )
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(cfg, split, augment=False, shuffle=False):
    ds = LandmarkDataset(
        Path(cfg["data_dir"]) / f"source_{split}.npz",
        cfg["image_size"],
        cfg["heatmap_size"],
        cfg["heatmap_sigma"],
        augment=augment,
    )
    shot_count = cfg.get("shot_count")
    if split == "train" and shot_count not in (None, "all"):
        shot_count = min(int(shot_count), len(ds))
        rng = np.random.default_rng(cfg.get("seed", 42))
        ds.indices = rng.permutation(len(ds))[:shot_count]
    if split != "train":
        max_samples = cfg.get(f"{split}_max_samples", cfg.get("eval_max_samples"))
        if max_samples not in (None, "all"):
            max_samples = min(int(max_samples), len(ds))
            rng = np.random.default_rng(cfg.get("seed", 42) + {"val": 1, "test": 2}.get(split, 3))
            ds.indices = rng.permutation(len(ds))[:max_samples]
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = cfg.get("pin_memory")
    if pin_memory is None:
        pin_memory = get_device(cfg.get("device")).type == "cuda"
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def validation_error(model, dataloader, device, cfg):
    model.eval()
    total = 0.0
    n = 0
    for batch in dataloader:
        pred = heatmaps_to_coords(
            model(batch["image"].to(device)),
            cfg["image_size"],
            cfg.get("decode", {}).get("method", "softargmax"),
            cfg.get("decode", {}).get("temperature", 40.0),
        ).cpu()
        total += torch.linalg.vector_norm(pred - batch["keypoints"], dim=-1).sum().item()
        n += pred.numel() / 2
    return total / n


def optimizer_parameter_groups(model, cfg):
    """Build optional discriminative-LR groups for transfer learning."""
    learning_rate = float(cfg["learning_rate"])
    multiplier = float(cfg.get("encoder_learning_rate_multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("encoder_learning_rate_multiplier must be positive")
    if multiplier == 1.0:
        return model.parameters()
    if not hasattr(model, "encoder_parameters"):
        raise ValueError(
            "encoder_learning_rate_multiplier requires a model with "
            "encoder_parameters()"
        )
    encoder = list(model.encoder_parameters())
    encoder_ids = {id(parameter) for parameter in encoder}
    task = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_ids
    ]
    if not encoder or not task:
        raise ValueError("could not split encoder and task parameters")
    return [
        {
            "params": encoder,
            "lr": learning_rate * multiplier,
            "group_name": "encoder",
        },
        {
            "params": task,
            "lr": learning_rate,
            "group_name": "task",
        },
    ]


def parameter_group_learning_rates(optimizer):
    rates = {
        group.get("group_name", f"group_{index}"): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    if "task" not in rates:
        rates["task"] = float(optimizer.param_groups[-1]["lr"])
    if "encoder" not in rates:
        rates["encoder"] = float(optimizer.param_groups[0]["lr"])
    return rates


def train_model(cfg, checkpoint):
    seed_everything(cfg.get("seed", 42))
    device = get_device(cfg.get("device"))
    model = build_model(cfg).to(device)
    loaded = total = 0
    if cfg.get("init_checkpoint"):
        loaded, total = load_compatible(model, cfg["init_checkpoint"], device)
    freeze_encoder_epochs = int(cfg.get("freeze_encoder_epochs", 0))
    if freeze_encoder_epochs < 0:
        raise ValueError("freeze_encoder_epochs cannot be negative")
    if freeze_encoder_epochs:
        if not hasattr(model, "freeze_encoder") or not hasattr(model, "unfreeze_encoder"):
            raise ValueError(
                "freeze_encoder_epochs requires a model with freeze_encoder() "
                "and unfreeze_encoder()"
            )
        model.freeze_encoder()
    train_loader = make_loader(cfg, "train", augment=cfg.get("augmentation", True), shuffle=True)
    val_loader = make_loader(cfg, "val")
    opt = torch.optim.AdamW(
        optimizer_parameter_groups(model, cfg),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    learning_rate_schedule = cfg.get("learning_rate_schedule", "constant")
    if learning_rate_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(cfg.get("epochs", cfg.get("base_epochs", 1))),
            eta_min=float(cfg.get("minimum_learning_rate", 0.00001)),
        )
    elif learning_rate_schedule == "constant":
        scheduler = None
    else:
        raise ValueError(f"unknown learning-rate schedule: {learning_rate_schedule}")
    best = (float("inf"), None, 0)
    history = {
        "loss": [],
        "val_error": [],
        "learning_rate": [],
        "encoder_learning_rate": [],
        "encoder_frozen": [],
        "heatmap_sigma": [],
    }
    epochs = int(cfg.get("epochs", cfg.get("base_epochs", 1)))
    for epoch in range(1, epochs + 1):
        if freeze_encoder_epochs and epoch == freeze_encoder_epochs + 1:
            model.unfreeze_encoder()
        sigma_end = cfg.get("heatmap_sigma_end")
        if sigma_end is None:
            current_sigma = float(cfg["heatmap_sigma"])
        else:
            progress = (epoch - 1) / max(epochs - 1, 1)
            sigma_schedule = cfg.get("heatmap_sigma_schedule", "cosine")
            if sigma_schedule == "linear":
                fraction = progress
            elif sigma_schedule == "cosine":
                fraction = 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                raise ValueError(f"unknown heatmap-sigma schedule: {sigma_schedule}")
            current_sigma = float(cfg["heatmap_sigma"]) + (
                float(sigma_end) - float(cfg["heatmap_sigma"])
            ) * fraction
        train_loader.dataset.sigma = current_sigma
        learning_rates = parameter_group_learning_rates(opt)
        current_learning_rate = learning_rates["task"]
        current_encoder_learning_rate = learning_rates["encoder"]
        encoder_frozen = bool(getattr(model, "encoder_frozen", False))
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"{cfg.get('run_name','train')} {epoch}/{epochs}", leave=False, disable=True):
            opt.zero_grad(set_to_none=True)
            pred = model(batch["image"].to(device))
            loss, _ = compute_loss(pred, batch, cfg)
            loss.backward()
            opt.step()
            running += loss.item() * len(batch["image"])
        val_error = validation_error(model, val_loader, device, cfg)
        train_loss = running / len(train_loader.dataset)
        history["loss"].append(train_loss)
        history["val_error"].append(val_error)
        history["learning_rate"].append(current_learning_rate)
        history["encoder_learning_rate"].append(current_encoder_learning_rate)
        history["encoder_frozen"].append(encoder_frozen)
        history["heatmap_sigma"].append(current_sigma)
        print(
            f"epoch={epoch} loss={train_loss:.5f} val_error={val_error:.3f}px "
            f"lr={current_learning_rate:.6g} encoder_lr={current_encoder_learning_rate:.6g} "
            f"encoder_frozen={encoder_frozen}"
        )
        if val_error < best[0]:
            best = (val_error, copy.deepcopy(model.state_dict()), epoch)
            save_checkpoint(checkpoint, model, epoch, val_error, cfg)
        if scheduler is not None:
            scheduler.step()
    model.load_state_dict(best[1])
    return model, history, {
        "best_val_error_px": best[0],
        "best_epoch": best[2],
        "device": str(device),
        "init_checkpoint": cfg.get("init_checkpoint", ""),
        "init_tensors_loaded": loaded,
        "model_tensors": total,
        "shot_count": cfg.get("shot_count", "all"),
    }


def evaluate_model(model, cfg, split="test", visualize_dir=None, max_visualizations=0):
    device = next(model.parameters()).device
    loader = make_loader(cfg, split)
    model.eval()
    preds, gts, boxes = [], [], []
    with torch.no_grad():
        seen = 0
        for batch in loader:
            pred = heatmaps_to_coords(
                model(batch["image"].to(device)),
                cfg["image_size"],
                cfg.get("decode", {}).get("method", "softargmax"),
                cfg.get("decode", {}).get("temperature", 40.0),
            ).cpu().numpy()
            preds.append(pred)
            gts.append(batch["keypoints"].numpy())
            boxes.append(batch["bbox"].numpy())
            if visualize_dir is not None and seen < max_visualizations:
                for j in range(min(len(pred), max_visualizations - seen)):
                    image = (batch["image"][j, 0].numpy() * 255).astype(np.uint8)
                    save_prediction(image, batch["keypoints"][j].numpy(), pred[j], Path(visualize_dir) / f"sample_{seen+j:03d}.png")
                seen += min(len(pred), max_visualizations - seen)
    return compute_metrics(
        np.concatenate(preds),
        np.concatenate(gts),
        cfg["image_size"],
        cfg.get("pck_thresholds", [3, 5, 10]),
        np.concatenate(boxes),
    )


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
