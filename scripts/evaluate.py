import argparse
from pathlib import Path

import torch

import _bootstrap  # noqa: F401
from paleolandmarker.models import build_model
from paleolandmarker.training.engine import evaluate_model, get_device, save_json
from paleolandmarker.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--visualizations", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    vis_dir = Path(cfg["output_dir"]) / "visualizations" / args.split
    metrics = evaluate_model(model, cfg, args.split, vis_dir, args.visualizations)
    save_json(Path(cfg["output_dir"]) / f"metrics_{args.split}.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()

