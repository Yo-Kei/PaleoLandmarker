#!/usr/bin/env python3
"""Compare morphospaces from paired manual and predicted landmark arrays."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def preshape(coordinates):
    centered = coordinates - coordinates.mean(axis=1, keepdims=True)
    size = np.linalg.norm(centered, axis=(1, 2), keepdims=True)
    if np.any(size <= 1e-12):
        raise ValueError("Each specimen must have nonzero centroid size.")
    return centered / size


def align(shapes, mean):
    u, _, vt = np.linalg.svd(np.einsum("nki,kj->nij", shapes, mean))
    correction = np.broadcast_to(np.eye(2), (len(shapes), 2, 2)).copy()
    correction[:, -1, -1] = np.linalg.det(u @ vt)
    return shapes @ (u @ correction @ vt)


def generalized_procrustes(coordinates):
    shapes = preshape(coordinates)
    mean = shapes[0].copy()
    for _ in range(1000):
        updated = align(shapes, mean).mean(axis=0)
        updated /= np.linalg.norm(updated)
        change = np.linalg.norm(updated - mean)
        mean = updated
        if change < 1e-10:
            return align(shapes, mean), mean
    raise RuntimeError("Generalized Procrustes analysis did not converge.")


def tangent_coordinates(shapes, mean):
    flat, pole = shapes.reshape(len(shapes), -1), mean.ravel()
    return flat - np.outer(flat @ pole, pole)


def pca(coordinates):
    center = coordinates.mean(axis=0)
    _, singular, axes = np.linalg.svd(coordinates - center, full_matrices=False)
    if singular[0] <= 1e-12:
        raise ValueError("Morphospace comparison requires variation among specimens.")
    return center, axes[:2], singular[:2] ** 2 / np.sum(singular ** 2)


def procrustes_r(first, second):
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    first_norm, second_norm = np.linalg.norm(first), np.linalg.norm(second)
    if min(first_norm, second_norm) <= 1e-12:
        raise ValueError("Morphospace comparison requires variation among specimens.")
    singular = np.linalg.svd((first / first_norm).T @ (second / second_norm),
                             compute_uv=False)
    return float(np.clip(singular.sum(), 0, 1))


def compare(manual, predicted):
    """Fit a manual GPA reference and compare separately fitted two-axis PCAs."""
    manual_shapes, mean = generalized_procrustes(manual)
    predicted_shapes = align(preshape(predicted), mean)
    manual_tangent = tangent_coordinates(manual_shapes, mean)
    predicted_tangent = tangent_coordinates(predicted_shapes, mean)
    center, axes, variance = pca(manual_tangent)
    predicted_center, predicted_axes, _ = pca(predicted_tangent)
    manual_scores = (manual_tangent - center) @ axes.T
    predicted_scores = (predicted_tangent - predicted_center) @ predicted_axes.T
    similarity = procrustes_r(manual_scores, predicted_scores)
    # Both sets use the manual PCA axes for the paired plot.
    common_scores = (predicted_tangent - center) @ axes.T
    return similarity, manual_scores, common_scores, variance


def plot_morphospace(manual, predicted, variance, output):
    fig, ax = plt.subplots(figsize=(5, 4))
    for first, second in zip(manual, predicted):
        ax.plot([first[0], second[0]], [first[1], second[1]],
                color="0.75", linewidth=0.6, zorder=1)
    ax.scatter(*manual.T, s=22, label="Manual", zorder=2)
    ax.scatter(*predicted.T, s=28, marker="x", label="Predicted", zorder=2)
    ax.set(xlabel=f"PC1 ({variance[0]:.1%})", ylabel=f"PC2 ({variance[1]:.1%})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "morphospace.png", dpi=300)
    fig.savefig(output / "morphospace.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True, help="Manual .npy array (N, K, 2)")
    parser.add_argument("--predicted", type=Path, required=True, help="Paired predicted .npy array")
    parser.add_argument("--sample-size", type=int, help="Specimens per draw (default: min(20, N))")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/downstream"))
    args = parser.parse_args()
    manual = np.asarray(np.load(args.manual, allow_pickle=False), dtype=float)
    predicted = np.asarray(np.load(args.predicted, allow_pickle=False), dtype=float)
    if (manual.shape != predicted.shape or manual.ndim != 3 or manual.shape[-1] != 2
            or manual.shape[0] < 3 or manual.shape[1] < 3
            or not np.isfinite(manual).all() or not np.isfinite(predicted).all()):
        parser.error("Inputs must be finite paired (N, K, 2) arrays, with N >= 3 and K >= 3.")
    size = args.sample_size if args.sample_size is not None else min(20, len(manual))
    if not 3 <= size <= len(manual) or args.repeats < 1:
        parser.error("Require 3 <= sample-size <= N and repeats >= 1.")
    rng = np.random.default_rng(args.seed)
    similarities = []
    for _ in range(args.repeats):
        indices = rng.choice(len(manual), size=size, replace=False)
        similarities.append(compare(manual[indices], predicted[indices])[0])
    full_r, manual_scores, predicted_scores, variance = compare(manual, predicted)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "subsamples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["repeat", "procrustes_r"])
        writer.writerows(enumerate(similarities, start=1))
    plot_morphospace(manual_scores, predicted_scores, variance, args.output)
    q10, median, q90 = np.quantile(similarities, [0.1, 0.5, 0.9])
    print(f"Full-sample Procrustes r: {full_r:.4f}")
    print(f"{args.repeats} draws of {size}: median r={median:.4f}, "
          f"10th-90th percentiles {q10:.4f}-{q90:.4f}")
    print(f"Saved subsamples.csv and morphospace.png/.pdf to {args.output}")


if __name__ == "__main__":
    main()
