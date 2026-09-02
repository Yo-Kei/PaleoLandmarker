"""Convert downloaded real landmark datasets to NPZ.

TriloMorph curve control points / TPS CURVES blocks are deliberately ignored;
only the fixed ``landmarks.pixel`` matrix or the initial TPS ``LM=`` block is
read.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def letterbox(image: Image.Image, points: np.ndarray, size: int):
    # Coordinates were digitized in the stored raster coordinate system; do
    # not apply EXIF rotation unless the landmarks are rotated identically.
    image = image.convert("L")
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new("L", (size, size), 0)
    canvas.paste(image.resize((nw, nh), Image.Resampling.LANCZOS), (ox, oy))
    sx, sy = nw / w, nh / h
    out_points = points.astype(np.float32) * np.array([sx, sy], np.float32) + np.array([ox, oy], np.float32)
    return np.asarray(canvas), out_points, (w, h), (sx, sy), (ox, oy)


def save_dataset(samples, output: Path, dataset: str, names, size: int):
    output.mkdir(parents=True, exist_ok=True)
    converted, manifest = [], []
    for sample in samples:
        with Image.open(sample["image"]) as im:
            image, points, original_size, scale, offset = letterbox(im, sample["points"], size)
        visibility = np.isfinite(points).all(axis=1)
        if not visibility.all() or np.any(points < 0) or np.any(points >= size):
            raise ValueError(f"Invalid transformed coordinates: {sample['id']}")
        lo, hi = points.min(0), points.max(0)
        converted.append((image, points, visibility, np.array([*lo, *hi], np.float32), sample))
        manifest.append({"id": sample["id"], "image": str(sample["image"]),
                         "original_width": original_size[0], "original_height": original_size[1],
                         "scale_x": scale[0], "scale_y": scale[1],
                         "offset_x": offset[0], "offset_y": offset[1]})
    np.savez_compressed(
        output / "dataset.npz",
        images=np.stack([x[0] for x in converted]),
        keypoints=np.stack([x[1] for x in converted]),
        visibility=np.stack([x[2] for x in converted]),
        masks=np.ones((len(converted), size, size), dtype=np.uint8),
        bbox=np.stack([x[3] for x in converted]),
        labels=np.array([x[4]["id"] for x in converted]),
        domains=np.array([dataset] * len(converted)),
        landmark_names=np.array(names),
    )
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    summary = {"dataset": dataset, "samples": len(samples), "image_size": size,
               "num_landmarks": len(names), "landmark_names": list(names),
               "split": None,
               "mask_note": "all-one placeholder; source provides no segmentation mask",
               "bbox_note": "tight bounding box around fixed landmarks"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def save_image_inventory(samples, output: Path, dataset: str):
    """Write an audit-only image list for sources that lack landmark truth."""
    output.mkdir(parents=True, exist_ok=True)
    fields = ["id", "image", "taxon", "note"]
    with (output / "image_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({k: sample.get(k, "") for k in fields})
    summary = {
        "dataset": dataset,
        "images": len(samples),
        "converted_to_npz": False,
        "reason": "source download contains images but no landmark coordinates",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def honey_samples(root: Path):
    images = root / "images"
    out = []
    with (root / "EC-raw-coordinates.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            pts = np.array([[float(row[f"x{i}"]), float(row[f"y{i}"])] for i in range(1, 20)])
            path = images / row["file"]
            if not path.exists(): raise FileNotFoundError(path)
            # IdentiFly exports Cartesian coordinates with the origin at the
            # lower left; raster/Pillow coordinates start at the upper left.
            with Image.open(path) as im:
                height = im.height
            pts[:, 1] = height - 1 - pts[:, 1]
            out.append({"id": row["file"], "image": path, "points": pts})
    return out


def fixed_points(path: Path):
    def parse_coord(value: str) -> float:
        if value.upper() == "NA":
            return float("nan")
        return float(value)

    text = path.read_text(errors="replace")
    if path.suffix.lower() == ".tps":
        m = re.search(r"(?mi)^LM=(\d+)\s*$", text)
        if not m: raise ValueError(f"No LM block: {path}")
        n = int(m.group(1)); tail = text[m.end():].splitlines()
        pts = []
        for line in (line for line in tail if line.strip()):
            if len(pts) == n:
                break
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid TPS coordinate {line!r}: {path}")
            pts.append([parse_coord(fields[0]), parse_coord(fields[1])])
        if len(pts) != n: raise ValueError(f"Expected {n} landmarks in {path}, got {len(pts)}")
    else:
        m = re.search(r"<landmarks\.pixel\b[^>]*nrow=(\d+)[^>]*>(.*?)</landmarks\.pixel>", text, re.S)
        if not m: raise ValueError(f"No landmarks.pixel block: {path}")
        n = int(m.group(1)); pts = []
        for line in m.group(2).strip().splitlines():
            fields = line.split()
            if len(fields) >= 3: pts.append([parse_coord(fields[-2]), parse_coord(fields[-1])])
        if len(pts) != n: raise ValueError(f"Expected {n} landmarks in {path}, got {len(pts)}")
    return np.asarray(pts, np.float32)


def trilomorph_samples(root: Path, part: str, selected_image_dir: Path | None = None):
    base = root / part
    images = {p.stem.lower(): p for p in (base / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    landmark_files = {p.stem.lower(): p for p in (base / "landmarks").iterdir()
                      if p.suffix.lower() in {".txt", ".tps"}}
    out, skipped = [], []

    if selected_image_dir is None:
        records = [(path, images.get(path.stem.lower()), None)
                   for path in sorted((base / "landmarks").iterdir())
                   if path.suffix.lower() in {".txt", ".tps"}]
    else:
        selected = sorted(p for p in selected_image_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        records = []
        seen_stems = {}
        for image in selected:
            stem = image.stem
            key = stem.lower()
            if key in seen_stems:
                skipped.append({"image": str(image), "reason": f"duplicate selected image stem; already using {seen_stems[key]}"})
                continue
            seen_stems[key] = str(image)
            path = landmark_files.get(key)
            match_rule = "exact selected image stem"
            if path is None and not stem.endswith(f"_{part[0]}"):
                path = landmark_files.get(f"{key}_{part[0].lower()}")
                match_rule = f"selected image stem plus _{part[0]} suffix"
            records.append((path, image, match_rule))

    for path, image, match_rule in records:
        if path is None:
            skipped.append({"image": str(image), "reason": "matching landmark file not found"})
            continue
        if image is None:
            skipped.append({"landmark_file": str(path), "reason": "matching image not found"}); continue
        try: points = fixed_points(path)
        except (ValueError, IndexError) as e:
            skipped.append({"landmark_file": str(path), "reason": str(e)}); continue
        sample = {"id": path.stem, "image": image, "points": points}
        if match_rule is not None:
            sample["match_rule"] = match_rule
        out.append(sample)
    counts = sorted({len(x["points"]) for x in out})
    # TriloMorph omits unavailable fixed landmarks instead of padding them with
    # NA. A heatmap head needs one stable channel set, so retain the complete
    # template (the maximum observed count) and audit every incomplete record.
    expected = max(counts)
    complete = []
    for sample in out:
        if len(sample["points"]) != expected:
            skipped.append({"landmark_file": sample["id"],
                            "reason": f"incomplete fixed landmarks: {len(sample['points'])}/{expected}"})
            continue
        with Image.open(sample["image"]) as im:
            w, h = im.size
        pts = sample["points"]
        if not np.isfinite(pts).all():
            skipped.append({"landmark_file": sample["id"],
                            "reason": "one or more fixed landmarks are NA"})
            continue
        if np.any(pts < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] >= h):
            skipped.append({"landmark_file": sample["id"],
                            "reason": f"fixed landmark outside image bounds {w}x{h}"})
            continue
        complete.append(sample)
    return complete, skipped


def read_tps_text(path: Path):
    """Read TPS text without corrupting non-ASCII image filenames.

    Most public TPS files in this project are UTF-8 or plain ASCII. Files
    exported by the Windows/Chinese tpsDig workflow may instead use GB18030.
    Decode strictly so that a failed encoding cannot silently replace parts of
    an ``IMAGE=`` filename and break image matching.
    """
    raw = path.read_bytes()
    errors = []
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as error:
            errors.append(f"{encoding}: {error}")
    raise UnicodeError(f"Could not decode TPS file {path}: {'; '.join(errors)}")


def tps_records(path: Path):
    """Read every LM record plus its IMAGE/ID fields from a TPS file."""
    lines = read_tps_text(path).splitlines()
    records = []
    i = 0
    while i < len(lines):
        match = re.match(r"(?i)^LM=(\d+)\s*$", lines[i].strip())
        if not match:
            i += 1
            continue
        n = int(match.group(1))
        if i + n >= len(lines):
            raise ValueError(f"Truncated LM block at line {i + 1}: {path}")
        points = []
        for line in lines[i + 1:i + 1 + n]:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid TPS coordinate {line!r}: {path}")
            points.append([float(fields[0]), float(fields[1])])
        record = {"points": np.asarray(points, np.float32), "image": None, "id": None}
        i += n + 1
        while i < len(lines) and not re.match(r"(?i)^LM=\d+\s*$", lines[i].strip()):
            field = re.match(r"(?i)^(IMAGE|ID)=(.*)$", lines[i].strip())
            if field:
                record[field.group(1).lower()] = field.group(2).strip()
            i += 1
        records.append(record)
    return records


def ml_morph_sdd_specimens(root: Path):
    sdd = next(root.glob("*_sdd.xml"))
    tree = ET.parse(sdd)
    ns = "{http://ns.tdwg.org/UBIF/2006}"
    doc = tree.getroot()

    def label(elem):
        lab = elem.find(f"{ns}Label") if elem is not None else None
        return " ".join("".join(lab.itertext()).split()) if lab is not None else ""

    media = {elem.attrib["id"]: elem for elem in doc.find(f".//{ns}MediaObjects").findall(f"{ns}MediaObject")}
    taxon_names = {elem.attrib["id"]: label(elem) for elem in doc.find(f".//{ns}TaxonNames").findall(f"{ns}TaxonName")}
    specimens = {}
    for specimen in doc.find(f".//{ns}Specimens").findall(f"{ns}Specimen"):
        taxon = specimen.find(f"{ns}TaxonName")
        taxon_label = taxon_names.get(taxon.attrib["ref"], "") if taxon is not None else ""
        refs = [obj.attrib["ref"] for obj in specimen.find(f"{ns}Representation").findall(f"{ns}MediaObject")]
        images = []
        for ref in refs:
            image_id = int(ref[1:])
            data = media[ref].find(f"{ns}Data")
            source_name = Path(data.attrib.get("href", "")).name if data is not None else ""
            source_stem = re.sub(r"^M\d+_", "", Path(source_name).stem)
            path = root / "media" / f"3598_{image_id}_original.jpg"
            if not path.exists():
                raise FileNotFoundError(path)
            images.append({"id": ref, "image": path, "taxon": taxon_label,
                           "source_name": source_name, "source_stem": source_stem})
        specimens[taxon_label] = images
    return specimens


def ml_morph_fish_images(root: Path):
    specimens = ml_morph_sdd_specimens(root)
    return [{**sample, "note": "Pseudanthias media from MorphoBank SDD; no landmark coordinates in downloaded files"}
            for sample in specimens.get("<i>Pseudanthias</i>", [])]


def ml_morph_bryozoan_samples(root: Path):
    """Crop Steginoporella zooid boxes and convert absolute XML landmarks."""
    specimens = ml_morph_sdd_specimens(root)
    images = {sample["source_stem"].lower(): sample
              for sample in specimens.get("<i>Steginoporella</i> <i>magnifica</i>", [])}
    landmark_path = root / "bryozoan_landmark.xml"
    xml_images = ET.parse(landmark_path).getroot().findall(".//image")

    samples, skipped = [], []
    part_names = sorted({part.attrib["name"] for image in xml_images for box in image.findall("box")
                         for part in box.findall("part")}, key=lambda x: int(x))
    matched_xml_images = 0
    for image_index, xml_image in enumerate(xml_images):
        xml_stem = Path(xml_image.attrib.get("file", "")).stem.lower()
        media_record = images.get(xml_stem)
        if media_record is None:
            skipped.append({"xml_image": xml_image.attrib.get("file", ""), "image_index": image_index,
                            "reason": "matching MorphoBank media source filename not found"})
            continue
        matched_xml_images += 1
        image_path = media_record["image"]
        with Image.open(image_path) as im:
            width, height = im.size
        for box_index, box in enumerate(xml_image.findall("box")):
            left = int(float(box.attrib["left"]))
            top = int(float(box.attrib["top"]))
            box_width = int(float(box.attrib["width"]))
            box_height = int(float(box.attrib["height"]))
            parts = {part.attrib["name"]: (float(part.attrib["x"]), float(part.attrib["y"]))
                     for part in box.findall("part")}
            missing = [name for name in part_names if name not in parts]
            if missing:
                skipped.append({"image": str(image_path), "box": box_index,
                                "reason": f"missing landmark parts: {','.join(missing)}"})
                continue
            absolute_points = np.asarray([[parts[name][0], parts[name][1]] for name in part_names], np.float32)
            if left < 0 or top < 0 or left + box_width > width or top + box_height > height:
                skipped.append({"image": str(image_path), "box": box_index,
                                "reason": f"box outside image bounds {width}x{height}"})
                continue
            crop_left = max(0, int(np.floor(min(left, float(absolute_points[:, 0].min())))))
            crop_top = max(0, int(np.floor(min(top, float(absolute_points[:, 1].min())))))
            crop_right = min(width, int(np.ceil(max(left + box_width, float(absolute_points[:, 0].max()) + 1))))
            crop_bottom = min(height, int(np.ceil(max(top + box_height, float(absolute_points[:, 1].max()) + 1))))
            crop_width = crop_right - crop_left
            crop_height = crop_bottom - crop_top
            pad = max(16, round(0.12 * max(crop_width, crop_height)))
            crop_left = max(0, crop_left - pad)
            crop_top = max(0, crop_top - pad)
            crop_right = min(width, crop_right + pad)
            crop_bottom = min(height, crop_bottom + pad)
            points = absolute_points - np.asarray([crop_left, crop_top], np.float32)
            if np.any(points < 0) or np.any(points[:, 0] >= box_width) or np.any(points[:, 1] >= box_height):
                crop_width = crop_right - crop_left
                crop_height = crop_bottom - crop_top
                if not (np.all(points >= 0) and np.all(points[:, 0] < crop_width) and np.all(points[:, 1] < crop_height)):
                    skipped.append({"image": str(image_path), "box": box_index,
                                    "reason": f"landmark outside expanded crop bounds {crop_width}x{crop_height}"})
                    continue
            if crop_right <= crop_left or crop_bottom <= crop_top:
                skipped.append({"image": str(image_path), "box": box_index,
                                "reason": "empty expanded crop"})
                continue
            samples.append({
                "id": f"{media_record['id']}:box{box_index:02d}",
                "image": image_path,
                "points": points,
                "crop": (crop_left, crop_top, crop_right, crop_bottom),
                "source_box": (left, top, left + box_width, top + box_height),
                "source_image_file": xml_image.attrib.get("file", ""),
                "media_source_name": media_record.get("source_name", ""),
                "box_index": box_index,
            })
    return samples, skipped, [f"part_{name}" for name in part_names], matched_xml_images, len(xml_images)


def save_cropped_dataset(samples, output: Path, dataset: str, names, size: int, source_note: str):
    output.mkdir(parents=True, exist_ok=True)
    converted, manifest = [], []
    for sample in samples:
        with Image.open(sample["image"]) as im:
            crop = im.crop(sample["crop"])
            image, points, original_size, scale, offset = letterbox(crop, sample["points"], size)
        visibility = np.isfinite(points).all(axis=1)
        if not visibility.all() or np.any(points < 0) or np.any(points >= size):
            raise ValueError(f"Invalid transformed coordinates: {sample['id']}")
        lo, hi = points.min(0), points.max(0)
        converted.append((image, points, visibility, np.array([*lo, *hi], np.float32), sample))
        manifest.append({"id": sample["id"], "image": str(sample["image"]),
                         "source_image_file": sample.get("source_image_file", ""),
                         "media_source_name": sample.get("media_source_name", ""),
                         "box_index": sample.get("box_index", ""),
                         "crop_left": sample["crop"][0], "crop_top": sample["crop"][1],
                         "crop_right": sample["crop"][2], "crop_bottom": sample["crop"][3],
                         "source_box_left": sample.get("source_box", ("", "", "", ""))[0],
                         "source_box_top": sample.get("source_box", ("", "", "", ""))[1],
                         "source_box_right": sample.get("source_box", ("", "", "", ""))[2],
                         "source_box_bottom": sample.get("source_box", ("", "", "", ""))[3],
                         "original_width": original_size[0], "original_height": original_size[1],
                         "scale_x": scale[0], "scale_y": scale[1],
                         "offset_x": offset[0], "offset_y": offset[1]})
    np.savez_compressed(
        output / "dataset.npz",
        images=np.stack([x[0] for x in converted]),
        keypoints=np.stack([x[1] for x in converted]),
        visibility=np.stack([x[2] for x in converted]),
        masks=np.ones((len(converted), size, size), dtype=np.uint8),
        bbox=np.stack([x[3] for x in converted]),
        labels=np.array([x[4]["id"] for x in converted]),
        domains=np.array([dataset] * len(converted)),
        landmark_names=np.array(names),
    )
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    summary = {"dataset": dataset, "samples": len(samples), "image_size": size,
               "num_landmarks": len(names), "landmark_names": list(names),
               "split": None,
               "source_note": source_note,
               "mask_note": "all-one placeholder; source provides no segmentation mask",
               "bbox_note": "tight bounding box around fixed landmarks"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def expanded_crop_from_points(points: np.ndarray, width: int, height: int, pad_fraction: float = 0.75):
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    box_w = float(hi[0] - lo[0])
    box_h = float(hi[1] - lo[1])
    pad = max(32, round(pad_fraction * max(box_w, box_h)))
    left = max(0, int(np.floor(lo[0] - pad)))
    top = max(0, int(np.floor(lo[1] - pad)))
    right = min(width, int(np.ceil(hi[0] + pad + 1)))
    bottom = min(height, int(np.ceil(hi[1] + pad + 1)))
    if right <= left or bottom <= top:
        raise ValueError(f"Empty landmark crop for bounds {width}x{height}")
    return left, top, right, bottom


def save_auto_cropped_dataset(samples, output: Path, dataset: str, names, size: int, source_note: str,
                              pad_fraction: float = 0.75):
    output.mkdir(parents=True, exist_ok=True)
    converted, manifest = [], []
    for sample in samples:
        with Image.open(sample["image"]) as im:
            width, height = im.size
            crop = expanded_crop_from_points(sample["points"], width, height, pad_fraction)
            crop_image = im.crop(crop)
            crop_points = sample["points"].astype(np.float32) - np.array([crop[0], crop[1]], np.float32)
            image, points, original_size, scale, offset = letterbox(crop_image, crop_points, size)
        visibility = np.isfinite(points).all(axis=1)
        if not visibility.all() or np.any(points < 0) or np.any(points >= size):
            raise ValueError(f"Invalid transformed coordinates: {sample['id']}")
        lo, hi = points.min(0), points.max(0)
        converted.append((image, points, visibility, np.array([*lo, *hi], np.float32), sample, crop, original_size, scale, offset))
        manifest.append({"id": sample["id"], "image": str(sample["image"]),
                         "crop_left": crop[0], "crop_top": crop[1],
                         "crop_right": crop[2], "crop_bottom": crop[3],
                         "original_width": original_size[0], "original_height": original_size[1],
                         "scale_x": scale[0], "scale_y": scale[1],
                         "offset_x": offset[0], "offset_y": offset[1],
                         "match_rule": sample.get("match_rule", "")})
    np.savez_compressed(
        output / "dataset.npz",
        images=np.stack([x[0] for x in converted]),
        keypoints=np.stack([x[1] for x in converted]),
        visibility=np.stack([x[2] for x in converted]),
        masks=np.ones((len(converted), size, size), dtype=np.uint8),
        bbox=np.stack([x[3] for x in converted]),
        labels=np.array([x[4]["id"] for x in converted]),
        domains=np.array([dataset] * len(converted)),
        landmark_names=np.array(names),
    )
    with (output / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    summary = {"dataset": dataset, "samples": len(samples), "image_size": size,
               "num_landmarks": len(names), "landmark_names": list(names),
               "split": None,
               "source_note": source_note,
               "crop_note": f"adaptive landmark crop with pad_fraction={pad_fraction}; coordinates translated before letterboxing",
               "mask_note": "all-one placeholder; source provides no segmentation mask",
               "bbox_note": "tight bounding box around fixed landmarks after crop and letterbox"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def ml_morph_fish_samples(root: Path):
    specimens = ml_morph_sdd_specimens(root)
    images = {sample["source_stem"].lower(): sample
              for sample in specimens.get("<i>Pseudanthias</i>", [])}
    landmark_path = root / "fish_landmark.xml"
    xml_images = ET.parse(landmark_path).getroot().findall(".//image")
    samples, skipped = [], []
    part_names = sorted({part.attrib["name"] for image in xml_images for box in image.findall("box")
                         for part in box.findall("part")}, key=lambda x: int(x))
    matched_xml_images = 0
    for image_index, xml_image in enumerate(xml_images):
        xml_stem = Path(xml_image.attrib.get("file", "")).stem.lower()
        media_record = images.get(xml_stem)
        if media_record is None:
            skipped.append({"xml_image": xml_image.attrib.get("file", ""), "image_index": image_index,
                            "reason": "matching MorphoBank media source filename not found"})
            continue
        matched_xml_images += 1
        boxes = xml_image.findall("box")
        if len(boxes) != 1:
            skipped.append({"xml_image": xml_image.attrib.get("file", ""), "image_index": image_index,
                            "reason": f"expected one fish box, got {len(boxes)}"})
            continue
        box = boxes[0]
        parts = {part.attrib["name"]: (float(part.attrib["x"]), float(part.attrib["y"]))
                 for part in box.findall("part")}
        missing = [name for name in part_names if name not in parts]
        if missing:
            skipped.append({"xml_image": xml_image.attrib.get("file", ""), "image_index": image_index,
                            "reason": f"missing landmark parts: {','.join(missing)}"})
            continue
        points = np.asarray([[parts[name][0], parts[name][1]] for name in part_names], np.float32)
        with Image.open(media_record["image"]) as im:
            width, height = im.size
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            skipped.append({"xml_image": xml_image.attrib.get("file", ""), "image_index": image_index,
                            "reason": f"landmark outside image bounds {width}x{height}"})
            continue
        samples.append({"id": media_record["id"], "image": media_record["image"], "points": points,
                        "match_rule": f"XML image stem matched MorphoBank href ({media_record.get('source_name', '')})"})
    return samples, skipped, [f"part_{name}" for name in part_names], matched_xml_images, len(xml_images)


def microtus_image_root(raw: Path, tps_path: Path):
    if "Fossils" in tps_path.parts:
        return raw / "Specimen_images" / "P23_Deposit1_fossils"
    name = tps_path.name.lower()
    if name.startswith("nikon"):
        return raw / "Specimen_images" / "Nikon"
    if "tilted" in name:
        return raw / "Specimen_images" / "Dinolite_tilted"
    return raw / "Specimen_images" / "Dinolite_normal"


def microtus_samples(raw: Path, tps_path: Path):
    image_root = microtus_image_root(raw, tps_path)
    images = {p.name.lower(): p for p in image_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    samples, skipped = [], []
    for index, record in enumerate(tps_records(tps_path)):
        image_name = record["image"]
        image = images.get(image_name.lower()) if image_name else None
        if image is None:
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": f"matching image not found in {image_root}"})
            continue
        with Image.open(image) as im:
            width, height = im.size
        points = record["points"].copy()
        points[:, 1] = height - 1 - points[:, 1]
        if not np.isfinite(points).all():
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": "one or more landmarks are not finite"})
            continue
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": f"landmark outside image bounds {width}x{height}"})
            continue
        samples.append({"id": f"{tps_path.stem}:{record['id'] or index}", "image": image, "points": points,
                        "match_rule": "TPS IMAGE filename in acquisition-specific image directory"})
    return samples, skipped


def fusulinid_samples(root: Path, tps_path: Path | None = None):
    """Pair fusulinid TPS records with PNGs and normalize TPS coordinates.

    The source was digitized in tpsDig's lower-left coordinate system. A few
    landmarks use the outer pixel boundary (``x == width`` or ``y == height``);
    these are clipped by one pixel and reported separately for auditability.
    Records that do not contain the complete landmark template are rejected.
    """
    tps_path = tps_path or root / "test.TPS"
    records = tps_records(tps_path)
    if not records:
        raise ValueError(f"No TPS records found: {tps_path}")
    expected_landmarks = max(len(record["points"]) for record in records)
    if expected_landmarks == 0:
        raise ValueError(f"No landmark coordinates found: {tps_path}")
    images = {
        path.name.lower(): path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    samples, skipped, adjusted = [], [], []
    seen_images = set()
    for index, record in enumerate(records):
        image_name = record["image"]
        audit = {"record": index, "id": record["id"], "image": image_name}
        if len(record["points"]) != expected_landmarks:
            skipped.append({
                **audit,
                "reason": (
                    f"incomplete fixed landmarks: "
                    f"{len(record['points'])}/{expected_landmarks}"
                ),
            })
            continue
        image = images.get(image_name.lower()) if image_name else None
        if image is None:
            skipped.append({**audit, "reason": "matching image not found"})
            continue
        image_key = image.name.lower()
        if image_key in seen_images:
            skipped.append({**audit, "reason": "duplicate TPS IMAGE record"})
            continue
        with Image.open(image) as im:
            width, height = im.size
        points = record["points"].copy()
        if not np.isfinite(points).all():
            skipped.append({**audit, "reason": "one or more landmarks are not finite"})
            continue
        if (
            np.any(points < 0)
            or np.any(points[:, 0] > width)
            or np.any(points[:, 1] > height)
        ):
            skipped.append({
                **audit,
                "reason": f"landmark outside TPS image bounds 0..{width} x 0..{height}",
            })
            continue
        boundary = np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height)
        if boundary:
            points[:, 0] = np.minimum(points[:, 0], width - 1)
            points[:, 1] = np.minimum(points[:, 1], height - 1)
            adjusted.append({
                **audit,
                "reason": "TPS outer-boundary coordinate clipped to final raster pixel",
            })
        points[:, 1] = height - 1 - points[:, 1]
        sample_id = record["id"] or Path(image_name).stem
        samples.append({
            "id": f"fusulinid:{sample_id}",
            "image": image,
            "points": points,
            "match_rule": "exact TPS IMAGE filename",
        })
        seen_images.add(image_key)
    return samples, skipped, adjusted


def conodont_samples(root: Path, tps_path: Path | None = None):
    """Pair Conodont TPS records with images and convert tpsDig coordinates.

    Conodont landmarks were digitized in tpsDig's Cartesian coordinate system
    (origin at the lower left). Training arrays use raster coordinates (origin
    at the upper left), so the y axis is inverted after exact filename
    matching. The complete 27-landmark template is required for every sample.
    """
    tps_path = tps_path or root / "Final.tps"
    records = tps_records(tps_path)
    if not records:
        raise ValueError(f"No TPS records found: {tps_path}")
    expected_landmarks = max(len(record["points"]) for record in records)
    images = {
        path.name.casefold(): path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    samples, skipped = [], []
    seen_images = set()
    for index, record in enumerate(records):
        image_name = record["image"]
        audit = {"record": index, "id": record["id"], "image": image_name}
        if len(record["points"]) != expected_landmarks:
            skipped.append({
                **audit,
                "reason": f"incomplete fixed landmarks: {len(record['points'])}/{expected_landmarks}",
            })
            continue
        image = images.get(image_name.casefold()) if image_name else None
        if image is None:
            skipped.append({**audit, "reason": "matching image not found"})
            continue
        image_key = image.name.casefold()
        if image_key in seen_images:
            skipped.append({**audit, "reason": "duplicate TPS IMAGE record"})
            continue
        with Image.open(image) as im:
            width, height = im.size
        points = record["points"].copy()
        if not np.isfinite(points).all():
            skipped.append({**audit, "reason": "one or more landmarks are not finite"})
            continue
        if (
            np.any(points < 0)
            or np.any(points[:, 0] > width)
            or np.any(points[:, 1] > height)
        ):
            skipped.append({
                **audit,
                "reason": f"landmark outside TPS image bounds 0..{width} x 0..{height}",
            })
            continue
        # tpsDig can place a point on the outer pixel boundary. Keep the
        # sample while mapping that boundary to the final valid raster pixel.
        points[:, 0] = np.minimum(points[:, 0], width - 1)
        points[:, 1] = np.minimum(points[:, 1], height - 1)
        points[:, 1] = height - 1 - points[:, 1]
        sample_id = record["id"] or Path(image_name).stem
        samples.append({
            "id": f"conodont:{sample_id}",
            "image": image,
            "points": points,
            "match_rule": "exact TPS IMAGE filename",
        })
        seen_images.add(image_key)
    return samples, skipped


def slugify(value: str):
    value = value.lower().replace(".tps", "")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def land_snail_samples(root: Path, species: str):
    """Pair Dryad TPS records with rasters, auditing every rejected record."""
    base = root / species
    tps_path = base / f"{species}.TPS"
    images = {p.name.lower(): p for p in base.iterdir()
              if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    samples, skipped = [], []
    for index, record in enumerate(tps_records(tps_path)):
        image_name = record["image"]
        if not image_name:
            skipped.append({"record": index, "id": record["id"],
                            "reason": "TPS record has no IMAGE field; source does not publish specimen images"})
            continue
        image = images.get(image_name.lower())
        if image is None:
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": "matching image not found"})
            continue
        points = record["points"]
        with Image.open(image) as im:
            width, height = im.size
        # tpsDig2 stores Cartesian coordinates with the origin at the lower
        # left, while Pillow arrays use an upper-left origin.
        points = points.copy()
        points[:, 1] = height - 1 - points[:, 1]
        if not np.isfinite(points).all():
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": "one or more landmarks are not finite"})
            continue
        if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
            skipped.append({"record": index, "id": record["id"], "image": image_name,
                            "reason": f"landmark outside image bounds {width}x{height}"})
            continue
        sample_id = record["id"] or Path(image_name).stem
        samples.append({"id": f"{species}:{sample_id}", "image": image,
                        "points": points, "match_rule": "exact TPS IMAGE filename"})
    return samples, skipped


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=Path("real_data/raw"))
    p.add_argument("--output", type=Path, default=Path("real_data/processed")); p.add_argument("--size", type=int, default=128)
    a = p.parse_args()
    honey = honey_samples(a.root / "honey_bee")
    save_dataset(honey, a.output / "honey_bee", "honey_bee", [f"wing_LM{i}" for i in range(1, 20)], a.size)
    tri_root = a.root / "trilomorph" / "v1.2"
    selected_cephala = a.root / "trilomorph" / "selected"
    all_skipped = {}
    for part, slug in (("Cephala", "trilomorph_cephala"), ("Pygidia", "trilomorph_pygidia")):
        selected_dir = selected_cephala if part == "Cephala" and selected_cephala.exists() else None
        samples, skipped = trilomorph_samples(tri_root, part, selected_dir); all_skipped[slug] = skipped
        n = len(samples[0]["points"])
        save_dataset(samples, a.output / slug, slug, [f"LM{i}" for i in range(1, n + 1)], a.size)
    (a.output / "trilomorph_skipped.json").write_text(json.dumps(all_skipped, indent=2) + "\n")
    snail_root = a.root / "land_snails_dryad"
    snail_skipped = {}
    for species in ("Haplotrema_concavum",):
        samples, skipped = land_snail_samples(snail_root, species)
        slug = f"land_snail_{species.lower()}"
        snail_skipped[slug] = skipped
        if not samples:
            continue
        counts = sorted({len(sample["points"]) for sample in samples})
        if len(counts) != 1:
            raise ValueError(f"Inconsistent landmark counts for {species}: {counts}")
        save_dataset(samples, a.output / slug, slug,
                     [f"LM{i}" for i in range(1, counts[0] + 1)], a.size)
    (a.output / "land_snails_skipped.json").write_text(json.dumps(snail_skipped, indent=2) + "\n")

    ml_root = a.root / "ml_morph_morphobank"
    if ml_root.exists():
        fish_landmarks = ml_root / "fish_landmark.xml"
        if fish_landmarks.exists():
            fish_samples, fish_skipped, fish_names, fish_matched, fish_total = ml_morph_fish_samples(ml_root)
            save_auto_cropped_dataset(fish_samples, a.output / "ml_morph_pseudanthias",
                                      "ml_morph_pseudanthias", fish_names, a.size,
                                      f"fish XML image names were matched to MorphoBank media href source names ({fish_matched}/{fish_total} images matched); each sample is a padded adaptive crop around the fish landmarks",
                                      pad_fraction=0.18)
        else:
            fish = ml_morph_fish_images(ml_root)
            save_image_inventory(fish, a.output / "ml_morph_pseudanthias_images", "ml_morph_pseudanthias_images")
            fish_skipped = []
        bryo_samples, bryo_skipped, bryo_names, matched, total = ml_morph_bryozoan_samples(ml_root)
        save_cropped_dataset(bryo_samples, a.output / "ml_morph_bryozoan_zooids",
                             "ml_morph_bryozoan_zooids", bryo_names, a.size,
                             f"bryozoan XML image names were matched to MorphoBank media href source names ({matched}/{total} images matched); XML boxes are detector boxes, not always full landmark bounds, so each crop is the padded union of the source box and all absolute landmark coordinates before letterboxing")
        (a.output / "ml_morph_skipped.json").write_text(json.dumps({
            "ml_morph_pseudanthias": fish_skipped,
            "ml_morph_bryozoan_zooids": bryo_skipped,
        }, indent=2) + "\n")

    micro_root = a.root / "microtus_dryad"
    if micro_root.exists():
        micro_skipped = {}
        selected = [
            (micro_root / "ProcANOVA" / "Dinolite_NoTilt_EO_T2.tps",
             "microtus_dinolite_notilt_eo_t2",
             "single preferred recent-specimen dataset: standardized Dino-Lite images, experienced observer, second digitizing iteration"),
            (micro_root / "Fossils" / "P23Dep1_Dinolite_NoTilt_EO_T1.TPS",
             "microtus_fossils_dinolite_notilt_eo_t1",
             "fossil unknowns photographed with the same Dino-Lite device and digitized by the experienced observer"),
        ]
        for tps_path, slug, source_note in selected:
            samples, skipped = microtus_samples(micro_root, tps_path)
            micro_skipped[slug] = skipped
            if not samples:
                continue
            counts = sorted({len(sample["points"]) for sample in samples})
            if len(counts) != 1:
                raise ValueError(f"Inconsistent landmark counts for {tps_path}: {counts}")
            save_auto_cropped_dataset(samples, a.output / slug, slug, [f"LM{i}" for i in range(1, counts[0] + 1)],
                                      a.size, source_note, pad_fraction=0.75)
        (a.output / "microtus_skipped.json").write_text(json.dumps(micro_skipped, indent=2) + "\n")


if __name__ == "__main__": main()
