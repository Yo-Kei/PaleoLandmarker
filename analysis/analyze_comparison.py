#!/usr/bin/env python3
"""Reproducible eight-seed analysis of the Comparison experiment.

The script uses only the Python standard library. Its principal estimand is an
equal-dataset (macro) average computed within each seed. Across-seed summaries
then report the arithmetic mean and sample standard deviation (ddof=1). Paired
differences use the same seed, dataset split, shot count, and test subset.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(
    os.environ.get(
        "PALEOLANDMARKER_RESULTS_ROOT",
        str(REPO_ROOT / "artifacts/reproduction/multiseed"),
    )
)
OUTPUT_ROOT = Path(__file__).resolve().parent
PROPOSED = "medium_synthetic"
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
SHOTS = (20, 50, 100)
DATASETS = (
    "conodont",
    "fusulinid",
    "land_snail_haplotrema_concavum",
    "microtus_dinolite_notilt_eo_t2",
    "ml_morph_bryozoan_zooids",
    "trilomorph_cephala",
)
METHODS = (
    "medium_synthetic",
    "medium_scratch",
    "deeplabcut_imagenet",
    "deeplabcut_synthetic",
    "deeplabcut_scratch",
    "landmarker_synthetic",
    "landmarker_scratch",
    "vandaele2018",
    "ml_morph",
)
METRICS = (
    "mean_error_px",
    "normalized_mean_error",
    "pck@3px",
    "pck@5px",
    "pck@10px",
)
T_CRITICAL_DF7_975 = 2.364624251

DISPLAY_NAMES = {
    "medium_synthetic": "PaleoLandmarker-M (synthetic)",
    "medium_scratch": "PaleoLandmarker-M (scratch)",
    "deeplabcut_imagenet": "DeepLabCut-style (ImageNet)",
    "deeplabcut_synthetic": "DeepLabCut-style (synthetic)",
    "deeplabcut_scratch": "DeepLabCut-style (scratch)",
    "landmarker_synthetic": "Landmarker-style (synthetic)",
    "landmarker_scratch": "Landmarker-style (scratch)",
    "vandaele2018": "Vandaele-style ExtraTrees",
    "ml_morph": "ML-morph-style random forest",
}


def mean(values):
    return statistics.fmean(values)


def sample_sd(values):
    return statistics.stdev(values)


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows():
    files = sorted(glob.glob(str(RESULT_ROOT / "seed_*/comparison/comparison_results.csv")))
    assert len(files) == len(SEEDS), f"expected 8 comparison files, found {len(files)}"
    rows = []
    per_file_counts = {}
    for filename in files:
        seed_from_path = int(Path(filename).parts[-3].split("_")[-1])
        with open(filename, newline="", encoding="utf-8") as handle:
            file_rows = list(csv.DictReader(handle))
        per_file_counts[seed_from_path] = len(file_rows)
        for row in file_rows:
            row["seed"] = int(row["seed"])
            row["data_split_seed"] = int(row["data_split_seed"])
            row["shots"] = int(row["shots"])
            for field in ("train_size", "val_size", "test_size", "total_size", "num_landmarks"):
                row[field] = int(row[field])
            for field in METRICS:
                row[field] = float(row[field])
                assert math.isfinite(row[field]), f"non-finite {field}: {row}"
            assert row["seed"] == seed_from_path == row["data_split_seed"]
            rows.append(row)
    return files, rows, per_file_counts


def validate_completeness(files, rows, per_file_counts):
    expected_per_seed = len(DATASETS) * len(SHOTS) * len(METHODS)
    assert set(per_file_counts) == set(SEEDS)
    assert set(per_file_counts.values()) == {expected_per_seed}
    assert len(rows) == len(SEEDS) * expected_per_seed
    assert {row["dataset"] for row in rows} == set(DATASETS)
    assert {row["shots"] for row in rows} == set(SHOTS)
    assert {row["method"] for row in rows} == set(METHODS)

    keys = [(r["seed"], r["dataset"], r["shots"], r["method"]) for r in rows]
    assert len(keys) == len(set(keys)), "duplicate seed/dataset/shot/method rows"
    assert len(set(keys)) == len(SEEDS) * len(DATASETS) * len(SHOTS) * len(METHODS)

    invariant_fields = {
        "experiment": {"comparison"},
        "epochs": {"400"},
        "batch_size": {"8"},
        "learning_rate": {"0.001"},
        "weight_decay": {"1e-05"},
        "augmentation": {"True"},
        "augmentation_mode": {"bounded_affine"},
        "image_size": {"224"},
        "heatmap_size": {"112"},
        "heatmap_sigma": {"8.0"},
        "test_max_samples": {"100"},
        "val_max_samples": {"100"},
    }
    for field, expected in invariant_fields.items():
        actual = {row[field] for row in rows}
        assert actual == expected, f"unexpected {field}: {actual}"

    expected_initialization = {
        "medium_synthetic": "synthetic_multidomain",
        "medium_scratch": "scratch",
        "deeplabcut_imagenet": "imagenet",
        "deeplabcut_synthetic": "synthetic_multidomain",
        "deeplabcut_scratch": "scratch",
        "landmarker_synthetic": "synthetic_multidomain",
        "landmarker_scratch": "scratch",
        "vandaele2018": "classical",
        "ml_morph": "classical",
    }
    for method, initialization in expected_initialization.items():
        assert {r["initialization"] for r in rows if r["method"] == method} == {initialization}

    # Publication matrices are a pivot of the same NME values, not an independent
    # result source. Check every cell to detect stale or partially written files.
    by_key = {(r["seed"], r["method"], r["dataset"], r["shots"]): r for r in rows}
    publication_cell_count = 0
    for seed in SEEDS:
        path = RESULT_ROOT / f"seed_{seed}/comparison/comparison_publication.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            publication_rows = list(csv.DictReader(handle))
        assert len(publication_rows) == len(METHODS)
        for row in publication_rows:
            method = row["method"]
            assert method in METHODS
            for dataset in DATASETS:
                for shot in SHOTS:
                    column = f"{dataset}_{shot}shots_nme"
                    actual = float(row[column])
                    expected = by_key[seed, method, dataset, shot]["normalized_mean_error"]
                    assert actual == expected
                    publication_cell_count += 1

    manifest = json.loads((RESULT_ROOT / "combined_manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_commit"] == "bd9e10974de0933f893f9973307fa963a8866aee"
    assert manifest["seeds"] == list(SEEDS)
    assert all(item["metrics_test_files"] == 279 for item in manifest["per_seed"].values())

    status_files = sorted((RESULT_ROOT / "source_runs").glob("*/status.json"))
    statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_files]
    assert statuses and all(status["state"] == "completed" for status in statuses)
    comparison_statuses = [status for status in statuses if status["task"] == "all"]
    assert sorted(seed for status in comparison_statuses for seed in status["seeds"]) == list(SEEDS)
    assert all(job["state"] == "completed" and job["returncode"] == 0
               for status in comparison_statuses for job in status["jobs"])

    uniqueness = {
        method: len({r["normalized_mean_error"] for r in rows if r["method"] == method})
        for method in METHODS
    }
    assert set(uniqueness.values()) == {len(SEEDS) * len(DATASETS) * len(SHOTS)}

    return {
        "comparison_files": len(files),
        "long_rows": len(rows),
        "rows_per_seed": expected_per_seed,
        "unique_primary_keys": len(set(keys)),
        "publication_cells_verified": publication_cell_count,
        "source_status_files": len(statuses),
        "comparison_jobs_completed": sum(len(status["jobs"]) for status in comparison_statuses),
        "unique_nme_values_per_method": uniqueness,
        "manifest_commit": manifest["code_commit"],
    }


def build_index(rows):
    return {
        (row["seed"], row["dataset"], row["shots"], row["method"]): row
        for row in rows
    }


def build_seed_macro(index, datasets=DATASETS):
    output = {}
    for seed in SEEDS:
        for shot in SHOTS:
            for method in METHODS:
                cells = [index[seed, dataset, shot, method] for dataset in datasets]
                output[seed, shot, method] = {
                    metric: mean(cell[metric] for cell in cells)
                    for metric in METRICS
                }
    return output


def export_seed_macro(seed_macro):
    rows = []
    for seed in SEEDS:
        for shot in SHOTS:
            for method in METHODS:
                rows.append({
                    "seed": seed,
                    "shots": shot,
                    "method": method,
                    **seed_macro[seed, shot, method],
                })
    write_csv(
        OUTPUT_ROOT / "seed_macro.csv",
        ["seed", "shots", "method", *METRICS],
        rows,
    )


def export_main_table(seed_macro):
    rows = []
    for method in METHODS:
        row = {"method": method, "display_name": DISPLAY_NAMES[method]}
        for shot in SHOTS:
            for metric in ("normalized_mean_error", "pck@5px"):
                values = [seed_macro[seed, shot, method][metric] for seed in SEEDS]
                short = "nme" if metric == "normalized_mean_error" else "pck5"
                row[f"{shot}_{short}_mean"] = mean(values)
                row[f"{shot}_{short}_sd"] = sample_sd(values)
        rows.append(row)
    fields = ["method", "display_name"]
    for shot in SHOTS:
        fields += [f"{shot}_nme_mean", f"{shot}_nme_sd", f"{shot}_pck5_mean", f"{shot}_pck5_sd"]
    write_csv(OUTPUT_ROOT / "main_table.csv", fields, rows)

    lines = [
        "| Method | 20-shot NME | 20-shot PCK@5 | 50-shot NME | 50-shot PCK@5 | 100-shot NME | 100-shot PCK@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cells = [row["display_name"]]
        for shot in SHOTS:
            cells.extend([
                f'{row[f"{shot}_nme_mean"]:.4f} ± {row[f"{shot}_nme_sd"]:.4f}',
                f'{row[f"{shot}_pck5_mean"]:.4f} ± {row[f"{shot}_pck5_sd"]:.4f}',
            ])
        lines.append("| " + " | ".join(cells) + " |")
    (OUTPUT_ROOT / "main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_dataset_summary(index):
    rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            for method in METHODS:
                row = {"dataset": dataset, "shots": shot, "method": method}
                for metric in METRICS:
                    values = [index[seed, dataset, shot, method][metric] for seed in SEEDS]
                    row[f"{metric}_mean"] = mean(values)
                    row[f"{metric}_sd"] = sample_sd(values)
                rows.append(row)
    fields = ["dataset", "shots", "method"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_sd"])
    write_csv(OUTPUT_ROOT / "dataset_summary.csv", fields, rows)


def export_paired_deltas(index, seed_macro):
    rows = []
    for shot in SHOTS:
        for comparator in METHODS:
            if comparator == PROPOSED:
                continue
            differences = [
                seed_macro[seed, shot, comparator]["normalized_mean_error"]
                - seed_macro[seed, shot, PROPOSED]["normalized_mean_error"]
                for seed in SEEDS
            ]
            difference_mean = mean(differences)
            difference_sd = sample_sd(differences)
            half_width = T_CRITICAL_DF7_975 * difference_sd / math.sqrt(len(SEEDS))
            comparator_mean = mean(
                seed_macro[seed, shot, comparator]["normalized_mean_error"]
                for seed in SEEDS
            )
            cell_differences = [
                index[seed, dataset, shot, comparator]["normalized_mean_error"]
                - index[seed, dataset, shot, PROPOSED]["normalized_mean_error"]
                for seed in SEEDS
                for dataset in DATASETS
            ]
            rows.append({
                "shots": shot,
                "comparator": comparator,
                "nme_difference_comparator_minus_proposed_mean": difference_mean,
                "nme_difference_sd": difference_sd,
                "nme_difference_95ci_low": difference_mean - half_width,
                "nme_difference_95ci_high": difference_mean + half_width,
                "relative_nme_reduction_vs_comparator_percent": 100 * difference_mean / comparator_mean,
                "seed_macro_wins": sum(value > 0 for value in differences),
                "seed_macro_ties": sum(value == 0 for value in differences),
                "seed_macro_losses": sum(value < 0 for value in differences),
                "dataset_seed_cell_wins": sum(value > 0 for value in cell_differences),
                "dataset_seed_cell_ties": sum(value == 0 for value in cell_differences),
                "dataset_seed_cell_losses": sum(value < 0 for value in cell_differences),
            })
    write_csv(OUTPUT_ROOT / "paired_deltas.csv", list(rows[0]), rows)


def export_rank_summary(index):
    ranks = defaultdict(list)
    wins = Counter()
    for dataset in DATASETS:
        for shot in SHOTS:
            method_means = {
                method: mean(index[seed, dataset, shot, method]["normalized_mean_error"] for seed in SEEDS)
                for method in METHODS
            }
            for rank, (method, _) in enumerate(sorted(method_means.items(), key=lambda item: item[1]), start=1):
                ranks[method].append(rank)
                wins[method] += rank == 1
    rows = [{
        "method": method,
        "mean_rank_over_18_dataset_shot_conditions": mean(ranks[method]),
        "dataset_shot_wins": wins[method],
        "dataset_shot_top3": sum(rank <= 3 for rank in ranks[method]),
    } for method in METHODS]
    write_csv(OUTPUT_ROOT / "rank_summary.csv", list(rows[0]), rows)


def export_excluding_bryozoan_sensitivity(index):
    datasets = tuple(dataset for dataset in DATASETS if dataset != "ml_morph_bryozoan_zooids")
    seed_macro = build_seed_macro(index, datasets=datasets)
    rows = []
    for shot in SHOTS:
        ordered = []
        for method in METHODS:
            values = [seed_macro[seed, shot, method]["normalized_mean_error"] for seed in SEEDS]
            ordered.append((mean(values), method, sample_sd(values)))
        for rank, (value_mean, method, value_sd) in enumerate(sorted(ordered), start=1):
            rows.append({
                "shots": shot,
                "rank": rank,
                "method": method,
                "five_dataset_nme_mean": value_mean,
                "five_dataset_nme_sd": value_sd,
            })
    write_csv(OUTPUT_ROOT / "exclude_bryozoan_sensitivity.csv", list(rows[0]), rows)


def export_monotonicity(index):
    rows = []
    for method in METHODS:
        violations = []
        for dataset in DATASETS:
            for seed in SEEDS:
                for lower, higher in ((20, 50), (50, 100)):
                    low_value = index[seed, dataset, lower, method]["normalized_mean_error"]
                    high_value = index[seed, dataset, higher, method]["normalized_mean_error"]
                    if high_value > low_value:
                        violations.append({
                            "method": method,
                            "dataset": dataset,
                            "seed": seed,
                            "transition": f"{lower}_to_{higher}",
                            "lower_shot_nme": low_value,
                            "higher_shot_nme": high_value,
                            "increase": high_value - low_value,
                        })
        rows.extend(violations)
    fields = ["method", "dataset", "seed", "transition", "lower_shot_nme", "higher_shot_nme", "increase"]
    write_csv(OUTPUT_ROOT / "shot_monotonicity_violations.csv", fields, rows)


def combination_ratio(n, k, total):
    """Return C(n, k) / C(total, k) using log-gamma arithmetic."""
    if k > n or k < 0:
        return 0.0
    return math.exp(
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        - math.lgamma(total + 1)
        + math.lgamma(k + 1)
        + math.lgamma(total - k + 1)
    )


def export_bryozoan_group_audit():
    manifest_path = REPO_ROOT / "data/real/processed_224/ml_morph_bryozoan_zooids/manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = Counter(row["source_image_file"] for row in rows)
    n = len(rows)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    expected_overlap_groups = 0.0
    expected_test_crops_with_train_source = 0.0
    for group_size in groups.values():
        p_no_train = combination_ratio(n - n_train, group_size, n)
        p_no_test = combination_ratio(n - n_test, group_size, n)
        p_neither = combination_ratio(n_val, group_size, n)
        expected_overlap_groups += 1 - p_no_train - p_no_test + p_neither
        p_has_train_given_test_crop = 1 - combination_ratio(n - group_size, n_train, n - 1)
        expected_test_crops_with_train_source += (
            group_size * n_test / n * p_has_train_given_test_crop
        )
    audit = {
        "split_method_in_code": "random independent records",
        "crop_records": n,
        "unique_source_images": len(groups),
        "mean_crops_per_source_image": mean(groups.values()),
        "min_crops_per_source_image": min(groups.values()),
        "max_crops_per_source_image": max(groups.values()),
        "source_images_with_multiple_crops": sum(value > 1 for value in groups.values()),
        "nominal_train_records": n_train,
        "nominal_validation_records": n_val,
        "nominal_test_records": n_test,
        "expected_source_images_overlapping_train_and_test": expected_overlap_groups,
        "expected_test_crops_with_at_least_one_training_crop_from_same_source": expected_test_crops_with_train_source,
        "expected_fraction_of_test_crops_with_training_source": expected_test_crops_with_train_source / n_test,
        "interpretation": (
            "Expectation under the implemented record-level random split; this is a design audit, "
            "not reconstruction of the eight archived split manifests."
        ),
    }
    (OUTPUT_ROOT / "bryozoan_group_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main():
    files, rows, per_file_counts = load_rows()
    validation = validate_completeness(files, rows, per_file_counts)
    index = build_index(rows)
    seed_macro = build_seed_macro(index)
    export_seed_macro(seed_macro)
    export_main_table(seed_macro)
    export_dataset_summary(index)
    export_paired_deltas(index, seed_macro)
    export_rank_summary(index)
    export_excluding_bryozoan_sensitivity(index)
    export_monotonicity(index)
    validation["bryozoan_group_audit"] = export_bryozoan_group_audit()
    (OUTPUT_ROOT / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Validated and analyzed {len(rows)} comparison rows.")
    print(f"Outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
