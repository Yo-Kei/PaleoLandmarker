"""Create deterministic train/validation/test files for the real datasets."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np


SPLIT_FORMAT_VERSION = 2
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_NAMES = ("train", "val", "test")


def subset_npz(source: Path, destination: Path, indices: np.ndarray) -> None:
    with np.load(source) as data:
        arrays = {}
        n = len(data["images"])
        for key in data.files:
            value = data[key]
            arrays[key] = value[indices] if value.ndim and len(value) == n else value
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp.npz"
    )
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)


def split_indices(n: int, seed: int, fractions=DEFAULT_FRACTIONS):
    order = np.random.default_rng(seed).permutation(n)
    n_train = round(n * fractions[0]); n_val = round(n * fractions[1])
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]


def wing_group_splits(manifest: Path, seed: int):
    with manifest.open(newline="", encoding="utf-8") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    groups = {}
    for i, sample_id in enumerate(ids):
        match = re.match(r"^(EC-\d+)-", sample_id)
        if not match: raise ValueError(f"Cannot derive wing group from {sample_id}")
        groups.setdefault(match.group(1), []).append(i)
    group_names = np.array(sorted(groups))
    train_g, val_g, test_g = split_indices(len(group_names), seed)
    return tuple(np.array(sorted(i for g in group_names[gidx] for i in groups[g]), dtype=int)
                 for gidx in (train_g, val_g, test_g)), groups


def source_identity(source: Path) -> dict:
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def split_identity(
    source_dir: Path,
    splits,
    split_method: str,
    seed: int,
) -> dict:
    source = source_dir / "dataset.npz"
    return {
        "split_format_version": SPLIT_FORMAT_VERSION,
        "source": str(source),
        "source_identity": source_identity(source),
        "seed": seed,
        "fractions": list(DEFAULT_FRACTIONS),
        "split_method": split_method,
        "counts": {
            name: int(len(indices))
            for name, indices in zip(SPLIT_NAMES, splits)
        },
        "indices": {
            name: indices.tolist()
            for name, indices in zip(SPLIT_NAMES, splits)
        },
    }


def write_dataset(
    source_dir: Path,
    output_dir: Path,
    splits,
    split_method: str,
    seed: int,
):
    for name, indices in zip(SPLIT_NAMES, splits):
        subset_npz(source_dir / "dataset.npz", output_dir / f"source_{name}.npz", indices)
    summary = split_identity(source_dir, splits, split_method, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "split.json"
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    print(output_dir, summary["counts"])


def dataset_splits(source_dir: Path, dataset: str, seed: int):
    if dataset == "honey_bee":
        splits, groups = wing_group_splits(source_dir / "manifest.csv", seed)
        method = f"grouped by EC specimen/colony prefix ({len(groups)} groups)"
        return splits, method
    with np.load(source_dir / "dataset.npz", mmap_mode="r") as data:
        splits = split_indices(len(data["images"]), seed)
    return splits, "random independent records"


def split_is_current(
    source_dir: Path,
    output_dir: Path,
    splits,
    split_method: str,
    seed: int,
) -> bool:
    manifest = output_dir / "split.json"
    try:
        actual = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = split_identity(source_dir, splits, split_method, seed)
    if actual != expected:
        return False
    for name, indices in zip(SPLIT_NAMES, splits):
        path = output_dir / f"source_{name}.npz"
        try:
            with np.load(path, mmap_mode="r") as data:
                if len(data["images"]) != len(indices):
                    return False
        except (OSError, KeyError, ValueError):
            return False
    return True


def ensure_dataset_split(
    processed_root: Path,
    split_root: Path,
    dataset: str,
    seed: int,
) -> Path:
    """Materialize and reuse the deterministic real-data split for one seed."""
    source_dir = Path(processed_root) / dataset
    source = source_dir / "dataset.npz"
    if not source.is_file():
        raise FileNotFoundError(
            f"standardized real dataset is missing: {source}; run its converter first"
        )
    output_dir = Path(split_root) / f"seed_{seed}" / dataset
    splits, method = dataset_splits(source_dir, dataset, seed)
    if split_is_current(source_dir, output_dir, splits, method, seed):
        print(f"reuse real split seed={seed}: {output_dir}")
        return output_dir
    print(f"create real split seed={seed}: {output_dir}")
    write_dataset(source_dir, output_dir, splits, method, seed)
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", type=Path, default=Path("real_data/processed"))
    p.add_argument("--output", type=Path, default=Path("real_data/splits"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    wing_splits, groups = wing_group_splits(args.processed / "honey_bee" / "manifest.csv", args.seed)
    write_dataset(args.processed / "honey_bee", args.output / "honey_bee", wing_splits,
                  f"grouped by EC specimen/colony prefix ({len(groups)} groups)", args.seed)
    with np.load(args.processed / "trilomorph_cephala" / "dataset.npz") as data:
        ceph_splits = split_indices(len(data["images"]), args.seed)
    write_dataset(args.processed / "trilomorph_cephala", args.output / "trilomorph_cephala",
                  ceph_splits, "random independent specimens", args.seed)
    for slug in ("land_snail_haplotrema_concavum",):
        source = args.processed / slug / "dataset.npz"
        with np.load(source) as data:
            splits = split_indices(len(data["images"]), args.seed)
        write_dataset(args.processed / slug, args.output / slug, splits,
                      "random independent museum specimens", args.seed)
    for source_dir in sorted(p for p in args.processed.iterdir() if (p / "dataset.npz").exists()):
        slug = source_dir.name
        if slug in {"honey_bee", "trilomorph_cephala", "land_snail_haplotrema_concavum"}:
            continue
        if slug.startswith("microtus_procanova_") or slug.startswith("microtus_fossils_p23dep1_"):
            continue
        with np.load(source_dir / "dataset.npz") as data:
            splits = split_indices(len(data["images"]), args.seed)
        write_dataset(source_dir, args.output / slug, splits,
                      "random independent source records; repeated-measurement TPS files are kept as separate dataset slugs",
                      args.seed)


if __name__ == "__main__": main()
