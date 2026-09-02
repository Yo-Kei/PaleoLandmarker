"""Convert a TPS landmark file and its images into a training NPZ dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from paleolandmarker.data.real.common import IMAGE_SUFFIXES, save_dataset, tps_records


def _normalise_path(value: str) -> str:
    """Make Windows paths usable when a TPS file is read on another OS."""
    return value.strip().strip('"').replace("\\", "/")


def _image_index(image_dir: Path) -> dict[str, Path]:
    index = {}
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.casefold(), path)
            index.setdefault(path.stem.casefold(), path)
    return index


def _ordered_images(image_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.as_posix().casefold(),
    )


def _resolve_image(value: str | None, tps_path: Path, image_dir: Path, index: dict[str, Path]) -> Path | None:
    if not value:
        return None
    normalised = _normalise_path(value)
    raw_path = Path(normalised)
    candidates = [
        image_dir / normalised,
        image_dir / raw_path.name,
        tps_path.parent / normalised,
        tps_path.parent / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return index.get(raw_path.name.casefold()) or index.get(raw_path.stem.casefold())


def _convert_records(records, tps_path: Path, image_dir: Path, coordinate_system: str, pair_by_order: bool):
    index = _image_index(image_dir)
    ordered_images = _ordered_images(image_dir)
    if pair_by_order and len(ordered_images) != len(records):
        raise ValueError(
            "--pair-by-order requires exactly one image per TPS record; "
            f"found {len(ordered_images)} images and {len(records)} records"
        )
    samples = []
    skipped = []
    for record_index, record in enumerate(records, 1):
        image = _resolve_image(record.get("image"), tps_path, image_dir, index)
        if image is None and pair_by_order:
            image = ordered_images[record_index - 1]
        if image is None:
            skipped.append({
                "record": record_index,
                "image": record.get("image", ""),
                "reason": "IMAGE entry could not be matched to an image file",
            })
            continue
        points = record["points"].copy()
        if coordinate_system == "tpsdig":
            with Image.open(image) as source:
                points[:, 1] = source.height - 1 - points[:, 1]
        sample_id = record.get("id") or image.stem
        samples.append({"id": str(sample_id), "image": image, "points": points})
    return samples, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert TPS records and matching images into dataset.npz."
    )
    parser.add_argument("--tps", type=Path, required=True, help="Input TPS file.")
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing the images referenced by TPS IMAGE fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory, for example data/real/processed_224/my_dataset.",
    )
    parser.add_argument("--dataset", default="my_dataset", help="Dataset label stored in the NPZ file.")
    parser.add_argument("--size", type=int, default=224, help="Square output image size.")
    parser.add_argument(
        "--coordinate-system",
        choices=("tpsdig", "raster"),
        default="tpsdig",
        help="Coordinate convention in the TPS file; default: tpsdig lower-left.",
    )
    parser.add_argument(
        "--pair-by-order",
        action="store_true",
        help="Pair sorted images with TPS records when IMAGE fields are absent.",
    )
    args = parser.parse_args()

    if not args.tps.is_file():
        raise FileNotFoundError(args.tps)
    if not args.images.is_dir():
        raise NotADirectoryError(args.images)
    records = tps_records(args.tps)
    if not records:
        raise ValueError(f"No LM records found in {args.tps}")
    samples, skipped = _convert_records(
        records, args.tps, args.images, args.coordinate_system, args.pair_by_order
    )
    if not samples:
        raise ValueError("No TPS records could be matched to images")

    counts = sorted({len(sample["points"]) for sample in samples})
    if len(counts) != 1:
        raise ValueError(f"Inconsistent landmark counts in matched records: {counts}")
    if any(not np.isfinite(sample["points"]).all() for sample in samples):
        raise ValueError("TPS contains non-finite landmark coordinates")

    save_dataset(
        samples,
        args.output,
        args.dataset,
        [f"LM{i}" for i in range(1, counts[0] + 1)],
        args.size,
    )
    audit = {
        "tps": str(args.tps),
        "images": str(args.images),
        "output": str(args.output),
        "converted": len(samples),
        "skipped": skipped,
        "coordinate_system": args.coordinate_system,
        "pair_by_order": args.pair_by_order,
        "image_size": args.size,
        "num_landmarks": counts[0],
    }
    audit_path = args.output / "conversion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"converted {len(samples)} records with {counts[0]} landmarks to {args.output}")
    if skipped:
        print(f"skipped {len(skipped)} TPS records; see {audit_path}")


if __name__ == "__main__":
    main()
