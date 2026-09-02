import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from paleolandmarker.training.engine import evaluate_model, save_json, train_model
from paleolandmarker.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None, help="Optional compatible checkpoint used to initialize the model before training.")
    parser.add_argument("--shots", default=None, help="Number of real training samples to use, or all.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.init_checkpoint is not None:
        cfg["init_checkpoint"] = args.init_checkpoint
    if args.shots is not None:
        cfg["shot_count"] = "all" if args.shots == "all" else int(args.shots)
    checkpoint = Path(args.checkpoint or Path(cfg["output_dir"]) / "checkpoints" / "best.pt")
    model, history, summary = train_model(cfg, checkpoint)
    metrics = evaluate_model(model, cfg, "test", Path(cfg["output_dir"]) / "visualizations" / "test", 8)
    save_json(Path(cfg["output_dir"]) / "history.json", history)
    save_json(Path(cfg["output_dir"]) / "metrics_test.json", {**summary, **metrics})


if __name__ == "__main__":
    main()
