import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _bootstrap  # noqa: F401
from paleolandmarker.evaluation.visualize import save_prediction
from paleolandmarker.inference import image_paths, predict_paths, write_tps
from paleolandmarker.models import build_model
from paleolandmarker.training.engine import get_device
from paleolandmarker.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--image-dir", "--input-dir", dest="image_dir", type=Path)
    parser.add_argument("--output", default="artifacts/outputs/inference.png")
    parser.add_argument(
        "--tps-output",
        type=Path,
        default=None,
        help="TPS output path; directory inference defaults to <image-dir>/predictions.tps.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=None,
        help="Optional directory for source-resolution prediction overlays.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    model = build_model(cfg).to(device).eval()
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    paths = [args.image] if args.image is not None else image_paths(args.image_dir)
    records = predict_paths(model, paths, cfg, device, args.batch_size)

    if args.image is not None:
        with Image.open(args.image) as image:
            source_image = np.asarray(image.convert("L"))
        save_prediction(source_image, None, records[0]["points"], args.output)
        print(f"saved {args.output}")
        print(records[0]["points"].tolist())

    if args.visualization_dir is not None:
        args.visualization_dir.mkdir(parents=True, exist_ok=True)
        for record in records:
            with Image.open(record["path"]) as image:
                source_image = np.asarray(image.convert("L"))
            save_prediction(
                source_image,
                None,
                record["points"],
                args.visualization_dir / f"{record['path'].stem}_pred.png",
            )

    tps_output = args.tps_output
    if args.image_dir is not None and tps_output is None:
        tps_output = args.image_dir / "predictions.tps"
    if tps_output is not None:
        write_tps(records, tps_output)
        print(f"saved {len(records)} TPS records to {tps_output}")


if __name__ == "__main__":
    main()
