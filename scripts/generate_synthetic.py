"""Generate or preview the target-aware Synthetic dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import _bootstrap  # noqa: F401
from paleolandmarker.data.synthetic.generate import (
    ACQUISITION_ARTIFACT_PROBABILITIES,
    BASE_SIZE,
    DOMAINS,
    LANDMARK_NAMES,
    LAYOUT_RANGES,
    REVISION_NOTES,
    SECOND_ACQUISITION_ARTIFACT_PROBABILITY,
    STYLE_WEIGHTS,
    VERSION,
    generate,
    make_sample,
    resize_sample,
)
from paleolandmarker.utils import load_config


DEFAULT_OUTPUT = Path("artifacts/figures/synthetic_preview")
COLORS = [
    (230, 25, 25),
    (0, 114, 178),
    (0, 158, 115),
    (230, 159, 0),
    (204, 121, 167),
    (86, 180, 233),
]


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def annotate(image, keypoints):
    output = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(output)
    radius = max(2, round(output.width / 90))
    for index, (x, y) in enumerate(keypoints):
        color = COLORS[index % len(COLORS)]
        draw.ellipse(
            (x - radius - 1, y - radius - 1, x + radius + 1, y + radius + 1),
            fill="white",
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    return output


def contact_sheet(rows, annotated, samples_per_domain, destination):
    tile = BASE_SIZE
    label_width = 270
    header = 52
    gap = 8
    canvas = Image.new(
        "RGB",
        (
            label_width + samples_per_domain * (tile + gap),
            header + len(DOMAINS) * (tile + gap),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    kind = "landmark overlay" if annotated else "unannotated"
    draw.text(
        (10, 10),
        f"PaleoLandmarker Synthetic preview — {kind}",
        fill="black",
        font=_font(24),
    )
    grouped = {domain: [] for domain in DOMAINS}
    for row in rows:
        grouped[row["domain"]].append(row)
    for domain_index, domain in enumerate(DOMAINS):
        y = header + domain_index * (tile + gap)
        draw.text((10, y + 92), domain.replace("_", " "), fill="black", font=_font(20))
        for column, row in enumerate(grouped[domain]):
            source = row["annotated_path"] if annotated else row["raw_path"]
            with Image.open(source) as image:
                tile_image = image.convert("RGB")
            x = label_width + column * (tile + gap)
            canvas.paste(tile_image, (x, y))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)


def preservation_sheet(rows, destination):
    """Collect sparse edge-loss samples plus complex trilobite cephala."""
    selected = []
    damaged_domains = {
        "conodont_element",
        "gastropod_shell",
        "microtus_dentition",
    }
    for domain in damaged_domains:
        candidates = [
            row
            for row in rows
            if row["domain"] == domain
            and json.loads(row["metadata"]).get("edge_loss_count", 0) > 0
        ]
        selected.extend(candidates[:2])
    selected.extend([row for row in rows if row["domain"] == "trilobite"][:3])
    if not selected:
        return None
    tile, gap, header = BASE_SIZE, 8, 52
    canvas = Image.new(
        "RGB", (len(selected) * (tile + gap), header + tile), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        f"Synthetic {VERSION} — sparse edge loss and 3-D cephala",
        fill="black",
        font=_font(22),
    )
    for index, row in enumerate(selected):
        with Image.open(row["annotated_path"]) as image:
            canvas.paste(image.convert("RGB"), (index * (tile + gap), header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return destination


def acquisition_artifact_sheet(rows, destination):
    """Show the sparse scale bars and plate/specimen labels separately."""
    candidates = [
        row
        for row in rows
        if json.loads(row["metadata"]).get("acquisition_artifact_count", 0) > 0
    ]
    selected = []
    represented_domains = set()
    for row in candidates:
        if row["domain"] not in represented_domains:
            selected.append(row)
            represented_domains.add(row["domain"])
    for row in candidates:
        if row not in selected:
            selected.append(row)
        if len(selected) >= 12:
            break
    if not selected:
        return None
    columns = min(4, len(selected))
    rows_count = int(np.ceil(len(selected) / columns))
    tile, label_height, gap, header = BASE_SIZE, 42, 8, 52
    canvas = Image.new(
        "RGB",
        (
            columns * (tile + gap),
            header + rows_count * (tile + label_height + gap),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10),
        f"Synthetic {VERSION} — sparse scale and plate artifacts",
        fill="black",
        font=_font(22),
    )
    for index, row in enumerate(selected):
        column = index % columns
        row_index = index // columns
        x = column * (tile + gap)
        y = header + row_index * (tile + label_height + gap)
        with Image.open(row["raw_path"]) as image:
            canvas.paste(image.convert("RGB"), (x, y))
        metadata = json.loads(row["metadata"])
        types = ", ".join(
            artifact["type"] for artifact in metadata["acquisition_artifacts"]
        )
        draw.text(
            (x + 3, y + tile + 3),
            f"{row['domain']}\n{types}",
            fill="black",
            font=_font(13),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, optimize=True)
    return destination


def preview(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_root = args.output_dir / "raw"
    annotated_root = args.output_dir / "annotated"
    raw_root.mkdir(parents=True, exist_ok=True)
    annotated_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for domain in DOMAINS:
        background_styles = list(STYLE_WEIGHTS[domain])
        for index in range(args.samples_per_domain):
            forced_variant = None
            if domain == "bryozoan_zooid":
                forced_variant = "single" if index == 0 else "colony"
            sample = resize_sample(
                make_sample(
                    domain,
                    rng,
                    forced_style=background_styles[index % len(background_styles)],
                    forced_variant=forced_variant,
                ),
                args.image_size,
            )
            image, mask, keypoints, bbox, metadata = sample
            raw_path = raw_root / domain / f"{index + 1:02d}.png"
            annotated_path = annotated_root / domain / f"{index + 1:02d}.png"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(raw_path, optimize=True)
            annotate(image, keypoints).save(annotated_path, optimize=True)
            rows.append(
                {
                    "domain": domain,
                    "sample": index + 1,
                    "style": metadata["style"],
                    "raw_path": str(raw_path.resolve()),
                    "annotated_path": str(annotated_path.resolve()),
                    "bbox": json.dumps([float(value) for value in bbox]),
                    "keypoints": json.dumps(keypoints.tolist()),
                    "metadata": json.dumps(metadata, sort_keys=True),
                    "mask_fraction": float(mask.mean()),
                }
            )
    with (args.output_dir / "preview_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "generator_version": VERSION,
        "mode": "preview_only_no_training",
        "seed": args.seed,
        "image_size": args.image_size,
        "samples_per_domain": args.samples_per_domain,
        "domains": DOMAINS,
        "landmark_names": LANDMARK_NAMES,
        "style_weights": STYLE_WEIGHTS,
        "acquisition_artifact_probabilities": ACQUISITION_ARTIFACT_PROBABILITIES,
        "second_acquisition_artifact_probability": SECOND_ACQUISITION_ARTIFACT_PROBABILITY,
        "layout_ranges": LAYOUT_RANGES,
        "preview_background_sampling": "stratified_cycle_over_available_styles",
        "preview_bryozoan_variants": "first_single_then_colony",
        "revision_notes": REVISION_NOTES,
        "sample_count": len(rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    raw_sheet = args.output_dir / "synthetic_raw_contact_sheet.png"
    annotated_sheet = args.output_dir / "synthetic_landmarks_contact_sheet.png"
    preservation_path = args.output_dir / "synthetic_preservation_contact_sheet.png"
    artifact_path = args.output_dir / "synthetic_acquisition_artifacts.png"
    contact_sheet(rows, False, args.samples_per_domain, raw_sheet)
    contact_sheet(rows, True, args.samples_per_domain, annotated_sheet)
    preservation_sheet(rows, preservation_path)
    acquisition_artifact_sheet(rows, artifact_path)
    print(f"saved Synthetic preview: {raw_sheet}")
    print(f"saved Synthetic landmark preview: {annotated_sheet}")
    print(f"saved Synthetic preservation preview: {preservation_path}")
    print(f"saved Synthetic acquisition-artifact preview: {artifact_path}")
    print("preview only: no training dataset or experiment was started")
    return raw_sheet, annotated_sheet


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "dataset"], default="preview")
    parser.add_argument("--config", default="configs/synthetic_multidomain.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-domain", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--image-size", type=int, default=BASE_SIZE)
    args = parser.parse_args(argv)
    if args.samples_per_domain < 1 or args.image_size < 64:
        parser.error("samples-per-domain must be >= 1 and image-size must be >= 64")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "preview":
        return preview(args)
    cfg = load_config(args.config)
    generate(cfg)
    return None


if __name__ == "__main__":
    main()
