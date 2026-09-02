import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from paleolandmarker.data.real.split import split_indices, wing_group_splits, write_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/real/processed"))
    parser.add_argument("--output", type=Path, default=Path("data/real/splits"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = args.processed / args.dataset
    if args.dataset == "honey_bee":
        splits, groups = wing_group_splits(source_dir / "manifest.csv", args.seed)
        method = f"grouped by EC specimen/colony prefix ({len(groups)} groups)"
    else:
        with np.load(source_dir / "dataset.npz") as data:
            splits = split_indices(len(data["images"]), args.seed)
        method = "random independent records"
    write_dataset(source_dir, args.output / args.dataset, splits, method, args.seed)


if __name__ == "__main__":
    main()

