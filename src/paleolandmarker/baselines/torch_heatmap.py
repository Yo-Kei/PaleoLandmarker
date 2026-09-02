import copy
import json
from pathlib import Path

import torch
from torch import nn

from paleolandmarker.data.synthetic.generate import VERSION as SYNTHETIC_VERSION
from paleolandmarker.data.synthetic.generate import ensure_dataset
from paleolandmarker.training.checkpoints import save_checkpoint
from paleolandmarker.training.engine import (
    evaluate_model,
    get_device,
    make_loader,
    validation_error,
)
from paleolandmarker.training.losses import compute_loss
from paleolandmarker.utils import seed_everything


def make_baseline_loader(cfg, split, augment=False, shuffle=False):
    """Use exactly the same split sampling and loader settings as our models."""
    return make_loader(cfg, split, augment=augment, shuffle=shuffle)


class TorchHeatmapBaseline:
    name = "torch_heatmap"

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.device = get_device(self.cfg.get("device"))
        self.model = self.build_model(self.cfg).to(self.device)
        self.history = {"loss": [], "val_error": []}
        self.best = {}

    def build_model(self, cfg):
        raise NotImplementedError

    def _train(self, model, cfg, checkpoint, run_name):
        train_loader = make_baseline_loader(cfg, "train", augment=cfg.get("augmentation", True), shuffle=True)
        val_loader = make_baseline_loader(cfg, "val")
        epochs = int(cfg.get("baseline_epochs", cfg.get("epochs", 60)))
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("baseline_learning_rate", cfg["learning_rate"]),
            weight_decay=cfg["weight_decay"],
        )
        best_error = float("inf")
        best_state = None
        best_epoch = 0
        history = {"loss": [], "val_error": []}
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                pred = model(batch["image"].to(self.device))
                loss, _ = compute_loss(pred, batch, cfg)
                loss.backward()
                opt.step()
                running += loss.item() * len(batch["image"])
            val_error = validation_error(model, val_loader, self.device, cfg)
            train_loss = running / len(train_loader.dataset)
            history["loss"].append(train_loss)
            history["val_error"].append(val_error)
            print(f"{run_name} epoch={epoch} loss={train_loss:.5f} val_error={val_error:.3f}px")
            if val_error < best_error:
                best_error = val_error
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                save_checkpoint(checkpoint, model, epoch, val_error, cfg)
        if best_state is not None:
            model.load_state_dict(best_state)
        summary = {"best_val_error_px": best_error, "best_epoch": best_epoch, "device": str(self.device)}
        return history, summary

    def _synthetic_pretrain_cfg(self):
        cfg = dict(self.cfg)
        cfg.update(
            {
                "data_dir": self.cfg["synthetic_pretrain_data_dir"],
                "num_landmarks": self.cfg.get("synthetic_pretrain_num_landmarks", 18),
                "baseline_epochs": self.cfg.get("synthetic_pretrain_epochs", 50),
                "synthetic_generator": "latest",
                "synthetic_generator_version": SYNTHETIC_VERSION,
                "batch_size": self.cfg.get(
                    "synthetic_pretrain_batch_size",
                    self.cfg["batch_size"],
                ),
                "baseline_learning_rate": self.cfg.get("synthetic_pretrain_learning_rate", self.cfg.get("baseline_learning_rate", self.cfg["learning_rate"])),
                "loss": self.cfg.get("synthetic_pretrain_loss", {"name": "heatmap_mse"}),
                "augmentation": self.cfg.get(
                    "synthetic_pretrain_augmentation",
                    self.cfg.get("augmentation", True),
                ),
                "shot_count": "all",
            }
        )
        for key in ("source_domains", "target_domain", "source_counts", "target_counts"):
            pretrain_key = f"synthetic_pretrain_{key}"
            if pretrain_key in self.cfg:
                cfg[key] = self.cfg[pretrain_key]
        return cfg

    def _ensure_synthetic_pretrain(self):
        pre_cfg = self._synthetic_pretrain_cfg()
        pre_dir = Path(self.cfg.get("synthetic_pretrain_checkpoint_dir", self.cfg.get("baseline_checkpoint", "artifacts/outputs/baselines"))) / self.name / "synthetic_pretrain"
        checkpoint = pre_dir / "best.pt"
        config_path = pre_dir / "config.json"
        signature_cfg = {
            key: value
            for key, value in pre_cfg.items()
            if key not in {"output_dir", "baseline_checkpoint"}
        }
        config_json = json.dumps(signature_cfg, sort_keys=True, default=str)
        if (
            checkpoint.exists()
            and config_path.exists()
            and config_path.read_text(encoding="utf-8") == config_json
        ):
            print(f"{self.name} reuse synthetic pretrain checkpoint: {checkpoint}")
            return checkpoint
        if checkpoint.exists():
            print(f"{self.name} rerun synthetic pretrain because config changed: {checkpoint}")
        if self.cfg.get("synthetic_pretrain_generate", False):
            ensure_dataset(pre_cfg)
        pre_model = self.build_model(pre_cfg).to(self.device)
        self._train(pre_model, pre_cfg, checkpoint, f"{self.name}_synthetic_pretrain")
        config_path.write_text(config_json, encoding="utf-8")
        return checkpoint

    def _load_compatible(self, checkpoint):
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)["model"]
        target = self.model.state_dict()
        usable = {k: v for k, v in state.items() if k in target and target[k].shape == v.shape}
        target.update(usable)
        self.model.load_state_dict(target)
        print(f"{self.name} loaded {len(usable)}/{len(target)} compatible tensors from {checkpoint}")

    def fit(self):
        seed_everything(self.cfg.get("seed", 42))
        if self.cfg.get("synthetic_pretrain", False):
            self._load_compatible(self._ensure_synthetic_pretrain())
        checkpoint = Path(self.cfg.get("baseline_checkpoint", "artifacts/outputs/baselines")) / self.name / "best.pt"
        self.history, self.best = self._train(self.model, self.cfg, checkpoint, self.name)
        return self

    def evaluate(self, split="test", visualize_dir=None, max_visualizations=8):
        return evaluate_model(self.model, self.cfg, split, visualize_dir, max_visualizations)


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))
