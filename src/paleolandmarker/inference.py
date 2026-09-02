"""Inference helpers that preserve image geometry and write tpsDig files."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from paleolandmarker.data.real.common import IMAGE_SUFFIXES, letterbox
from paleolandmarker.evaluation.decode import heatmaps_to_coords


def image_paths(directory: Path):
    """Return a deterministic list of supported raster files in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def prepare_image(path: Path, image_size: int):
    """Letterbox one image and retain the transform needed for inversion."""
    with Image.open(path) as image:
        canvas, _, original_size, scale, offset = letterbox(
            image,
            np.empty((0, 2), dtype=np.float32),
            image_size,
        )
    tensor = torch.from_numpy(canvas.copy()).float().unsqueeze(0) / 255.0
    transform = {
        "original_size": original_size,
        "scale": scale,
        "offset": offset,
    }
    return tensor, canvas, transform


def restore_coordinates(points: np.ndarray, transform):
    """Map model-space letterbox coordinates back into the source raster."""
    width, height = transform["original_size"]
    scale = np.asarray(transform["scale"], dtype=np.float32)
    offset = np.asarray(transform["offset"], dtype=np.float32)
    restored = (np.asarray(points, dtype=np.float32) - offset) / scale
    restored[:, 0] = np.clip(restored[:, 0], 0, width - 1)
    restored[:, 1] = np.clip(restored[:, 1], 0, height - 1)
    return restored


@torch.no_grad()
def predict_paths(model, paths, cfg, device, batch_size=None):
    """Predict source-raster landmark coordinates for one or more images."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("No images were provided for inference")
    batch_size = int(batch_size or cfg.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    records = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        prepared = [prepare_image(path, cfg["image_size"]) for path in batch_paths]
        tensors = torch.stack([item[0] for item in prepared]).to(device)
        model_points = heatmaps_to_coords(
            model(tensors),
            cfg["image_size"],
            cfg.get("decode", {}).get("method", "softargmax"),
            cfg.get("decode", {}).get("temperature", 40.0),
        ).cpu().numpy()
        for path, (_, canvas, transform), points in zip(batch_paths, prepared, model_points):
            records.append({
                "path": path,
                "letterboxed_image": canvas,
                "letterboxed_points": points,
                "points": restore_coordinates(points, transform),
                "original_size": transform["original_size"],
            })
    return records


def tps_coordinates(points: np.ndarray, image_height: int):
    """Convert upper-left raster coordinates to tpsDig lower-left coordinates."""
    converted = np.asarray(points, dtype=np.float32).copy()
    converted[:, 1] = image_height - 1 - converted[:, 1]
    return converted


def write_tps(records, output: Path):
    """Write predicted source-image coordinates as a multi-record TPS file."""
    records = list(records)
    if not records:
        raise ValueError("Cannot write an empty TPS file")
    counts = {len(record["points"]) for record in records}
    if len(counts) != 1:
        raise ValueError(f"Inconsistent landmark counts: {sorted(counts)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, record in enumerate(records, 1):
        width, height = record["original_size"]
        points = tps_coordinates(record["points"], height)
        relative_image = os.path.relpath(record["path"], output.parent)
        lines.append(f"LM={len(points)}")
        lines.extend(f"{x:.5f} {y:.5f}" for x, y in points)
        lines.append(f"IMAGE={Path(relative_image).as_posix()}")
        lines.append(f"ID={record.get('id', index)}")
    # CRLF maximizes compatibility with the Windows tpsDig tools.
    with output.open("w", encoding="utf-8", newline="") as file:
        file.write("\r\n".join(lines) + "\r\n")
    return output
