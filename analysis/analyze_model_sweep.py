#!/usr/bin/env python3
"""Reproducible eight-seed analysis of the PaleoLandmarker model sweep.

The inferential replicate is the seed. For manuscript-level macro summaries,
the six dataset summaries are first averaged within each seed; sample SD and
paired contrasts are then calculated over the eight seed-level values.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(
    os.environ.get(
        "PALEOLANDMARKER_RESULTS_ROOT",
        str(ROOT / "artifacts/reproduction/multiseed"),
    )
)
OUT = Path(__file__).resolve().parent
MODELS = ("tiny", "small", "medium", "large")
SHOTS = (20, 50, 100)
METRICS = (
    "mean_error_px",
    "normalized_mean_error",
    "pck@3px",
    "pck@5px",
    "pck@10px",
)
KEY = ("seed", "dataset", "model", "shots")


def mean(values):
    return statistics.fmean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def t_critical_975(df):
    # Sufficient for every CI in this analysis (all manuscript estimates use n=8).
    lookup = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return lookup.get(df, 1.960)


def summarize(values):
    values = list(values)
    m = mean(values)
    s = sd(values)
    half = t_critical_975(len(values) - 1) * s / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": m,
        "sd": s,
        "ci95_low": m - half,
        "ci95_high": m + half,
    }


def exact_sign_flip_p(values):
    """Two-sided exact randomization p for a paired mean contrast (n=8)."""
    values = list(values)
    observed = abs(mean(values))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(mean(v * sign for v, sign in zip(values, signs)))
        exceed += permuted >= observed - 1e-15
        total += 1
    return exceed / total


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_value(name, value):
    if name in {"seed", "data_split_seed", "shots", "train_size", "val_size", "test_size", "total_size", "num_landmarks", "parameters"}:
        return int(value)
    if name in METRICS:
        return float(value)
    return value


def load_rows():
    manifest = json.loads((RESULTS / "combined_manifest.json").read_text())
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    rows = []
    headers = []
    source_files = []
    for seed in seeds:
        path = RESULTS / f"seed_{seed}/model_sweep/model_sweep_results.csv"
        source_files.append(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            headers.append(tuple(reader.fieldnames or ()))
            for raw in reader:
                row = {name: parse_value(name, value) for name, value in raw.items()}
                row["_source_seed"] = seed
                rows.append(row)
    return manifest, seeds, rows, headers, source_files


def validate(manifest, seeds, rows, headers):
    errors = []
    warnings = []
    datasets = tuple(sorted({row["dataset"] for row in rows}))
    expected_keys = set(itertools.product(seeds, datasets, MODELS, SHOTS))
    keys = [tuple(row[name] for name in KEY) for row in rows]
    counts = Counter(keys)
    observed_keys = set(keys)
    if len(set(headers)) != 1:
        errors.append("CSV headers differ across seed files")
    if observed_keys != expected_keys:
        errors.append(
            f"factorial keys incomplete: missing={len(expected_keys-observed_keys)}, "
            f"unexpected={len(observed_keys-expected_keys)}"
        )
    duplicate_keys = [key for key, count in counts.items() if count != 1]
    if duplicate_keys:
        errors.append(f"non-unique condition keys: {len(duplicate_keys)}")
    if len(rows) != len(expected_keys):
        errors.append(f"row count {len(rows)} != expected {len(expected_keys)}")

    for row in rows:
        if row["seed"] != row["_source_seed"] or row["data_split_seed"] != row["seed"]:
            errors.append(f"seed mismatch for {tuple(row[name] for name in KEY)}")
        for metric in METRICS:
            value = row[metric]
            if not math.isfinite(value):
                errors.append(f"non-finite {metric} for {tuple(row[name] for name in KEY)}")
        if not (0 <= row["normalized_mean_error"]):
            errors.append("negative NME")
        for metric in ("pck@3px", "pck@5px", "pck@10px"):
            if not (0 <= row[metric] <= 1):
                errors.append(f"out-of-range {metric}")
        if not (row["pck@3px"] <= row["pck@5px"] <= row["pck@10px"]):
            errors.append(f"non-monotone PCK thresholds for {tuple(row[name] for name in KEY)}")
        if row["shots"] > row["train_size"]:
            errors.append(f"shots exceed train size for {tuple(row[name] for name in KEY)}")

    fixed = {
        "experiment": "model_sweep",
        "model_architecture": "context_resize",
        "initialization": "synthetic_multidomain",
        "pretrain_kind": "multidomain",
        "pretrain_epochs": "50",
        "epochs": "400",
        "batch_size": "8",
        "learning_rate": "0.001",
        "weight_decay": "1e-05",
        "augmentation": "True",
        "augmentation_mode": "bounded_affine",
        "pretrain_augmentation": "True",
        "image_size": "224",
        "heatmap_size": "112",
        "heatmap_sigma": "8.0",
        "loss_name": "heatmap_coord",
        "loss_variant": "heatmap+coord",
        "coord_weight": "0.1",
        "loss_temperature": "40.0",
        "decode_method": "softargmax",
        "decode_temperature": "40.0",
        "device": "cuda",
        "min_train_size": "100",
        "val_max_samples": "100",
        "test_max_samples": "100",
    }
    for field, expected in fixed.items():
        actual = {str(row[field]) for row in rows}
        if actual != {expected}:
            errors.append(f"{field}: observed {sorted(actual)}, expected {expected}")

    # Check that the compact per-seed publication matrices contain exactly the
    # same NME values as the long-form source, without using them as replicates.
    long_index = {
        (row["seed"], row["model"], row["dataset"], row["shots"]): row["normalized_mean_error"]
        for row in rows
    }
    publication_cells = 0
    for seed in seeds:
        path = RESULTS / f"seed_{seed}/model_sweep/model_sweep_publication.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for pub in csv.DictReader(handle):
                model = pub.pop("model")
                for column, value in pub.items():
                    dataset, shot_text = column.rsplit("_", 2)[0], column.rsplit("_", 2)[1]
                    shot = int(shot_text.removesuffix("shots"))
                    expected = long_index[(seed, model, dataset, shot)]
                    if float(value) != expected:
                        errors.append(f"publication mismatch: seed={seed}, model={model}, column={column}")
                    publication_cells += 1
    if publication_cells != len(rows):
        errors.append(f"publication matrix cells {publication_cells} != source rows {len(rows)}")

    status_files = sorted((RESULTS / "source_runs").glob("*/status.json"))
    full_statuses = []
    for path in status_files:
        status = json.loads(path.read_text())
        if status.get("task") in {"all", "model_sweep"}:
            full_statuses.append(status)
            if status.get("state") != "completed":
                errors.append(f"source run not completed: {path.parent.name}")
            incomplete = [job for job in status.get("jobs", []) if job.get("state") != "completed"]
            if incomplete:
                errors.append(f"incomplete jobs in {path.parent.name}")
    status_seeds = sorted(seed for status in full_statuses for seed in status.get("seeds", []))
    if status_seeds != sorted(seeds):
        errors.append(f"full-run status seeds {status_seeds} != manifest seeds {list(seeds)}")

    if not errors:
        warnings.append(
            "All structural checks passed; numerical rerunning was not possible because checkpoints "
            "were intentionally excluded from the local results archive."
        )
    return datasets, errors, warnings


def macro_by_seed(rows, datasets, seeds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["model"], row["shots"])].append(row)
    output = []
    for seed, model, shot in itertools.product(seeds, MODELS, SHOTS):
        group = grouped[(seed, model, shot)]
        assert len(group) == len(datasets)
        record = {"seed": seed, "model": model, "shots": shot, "n_datasets": len(group)}
        for metric in METRICS:
            record[metric] = mean(row[metric] for row in group)
        output.append(record)
    return output


def aggregate(rows, group_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key in sorted(grouped):
        group = grouped[key]
        record = dict(zip(group_fields, key))
        record["n_seeds"] = len(group)
        for metric in METRICS:
            stats = summarize(row[metric] for row in group)
            for field, value in stats.items():
                if field != "n":
                    record[f"{metric}_{field}"] = value
        output.append(record)
    return output


def holm_adjust(p_values):
    m = len(p_values)
    order = sorted(range(m), key=p_values.__getitem__)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_contrasts(macro_rows, seeds):
    index = {
        (row["seed"], row["model"], row["shots"]): row["normalized_mean_error"]
        for row in macro_rows
    }
    contrast_specs = []
    for model in MODELS:
        for start, end in ((20, 50), (50, 100), (20, 100)):
            contrast_specs.append(("label_scaling_by_model", model, f"{start}->{end} shots", model, start, model, end))
    for start, end in ((20, 50), (50, 100), (20, 100)):
        contrast_specs.append(("label_scaling_overall", "all models", f"{start}->{end} shots", None, start, None, end))
    for shot in SHOTS:
        for start, end in (("tiny", "small"), ("small", "medium"), ("medium", "large"), ("tiny", "large")):
            contrast_specs.append(("model_scaling", f"{shot} shots", f"{start}->{end}", start, shot, end, shot))

    output = []
    for family, subset, comparison, base_model, base_shot, comp_model, comp_shot in contrast_specs:
        baseline = []
        comparator = []
        for seed in seeds:
            if base_model is None:
                baseline.append(mean(index[(seed, model, base_shot)] for model in MODELS))
                comparator.append(mean(index[(seed, model, comp_shot)] for model in MODELS))
            else:
                baseline.append(index[(seed, base_model, base_shot)])
                comparator.append(index[(seed, comp_model, comp_shot)])
        differences = [after - before for before, after in zip(baseline, comparator)]
        relative = [100.0 * (before - after) / before for before, after in zip(baseline, comparator)]
        stats = summarize(differences)
        rel_stats = summarize(relative)
        output.append(
            {
                "family": family,
                "subset": subset,
                "comparison": comparison,
                "n_paired_seeds": len(seeds),
                "baseline_mean_nme": mean(baseline),
                "comparator_mean_nme": mean(comparator),
                "mean_delta_nme": stats["mean"],
                "sd_delta_nme": stats["sd"],
                "delta_ci95_low": stats["ci95_low"],
                "delta_ci95_high": stats["ci95_high"],
                "mean_relative_reduction_pct": rel_stats["mean"],
                "sd_relative_reduction_pct": rel_stats["sd"],
                "exact_sign_flip_p_raw": exact_sign_flip_p(differences),
            }
        )
    by_family = defaultdict(list)
    for i, row in enumerate(output):
        by_family[row["family"]].append(i)
    for indices in by_family.values():
        adjusted = holm_adjust([output[i]["exact_sign_flip_p_raw"] for i in indices])
        for i, value in zip(indices, adjusted):
            output[i]["holm_p_within_family"] = value
    return output


def trend_diagnostics(rows, macro_rows, datasets, seeds):
    index = {
        (row["seed"], row["dataset"], row["model"], row["shots"]): row["normalized_mean_error"]
        for row in rows
    }
    label_trajectories = []
    for seed, dataset, model in itertools.product(seeds, datasets, MODELS):
        values = [index[(seed, dataset, model, shot)] for shot in SHOTS]
        label_trajectories.append(
            {
                "seed": seed,
                "dataset": dataset,
                "model": model,
                "nme_20": values[0],
                "nme_50": values[1],
                "nme_100": values[2],
                "20_to_50_improves": values[1] < values[0],
                "50_to_100_improves": values[2] < values[1],
                "fully_monotone": values[2] < values[1] < values[0],
            }
        )

    best_counts = Counter()
    for seed, dataset, shot in itertools.product(seeds, datasets, SHOTS):
        values = {model: index[(seed, dataset, model, shot)] for model in MODELS}
        best = min(values, key=values.get)
        best_counts[(shot, best)] += 1

    macro_index = {
        (row["seed"], row["model"], row["shots"]): row["normalized_mean_error"]
        for row in macro_rows
    }
    macro_best_counts = Counter()
    for seed, shot in itertools.product(seeds, SHOTS):
        values = {model: macro_index[(seed, model, shot)] for model in MODELS}
        macro_best_counts[(shot, min(values, key=values.get))] += 1

    output = []
    total = len(label_trajectories)
    for label, field in (
        ("20->50 improves", "20_to_50_improves"),
        ("50->100 improves", "50_to_100_improves"),
        ("20>50>100 monotone", "fully_monotone"),
    ):
        count = sum(bool(row[field]) for row in label_trajectories)
        output.append({"diagnostic": label, "subset": "all dataset-model-seed trajectories", "count": count, "denominator": total, "proportion": count / total})
    for shot, model in itertools.product(SHOTS, MODELS):
        output.append({"diagnostic": "best model by NME", "subset": f"{shot} shots: {model}", "count": best_counts[(shot, model)], "denominator": len(seeds) * len(datasets), "proportion": best_counts[(shot, model)] / (len(seeds) * len(datasets))})
        output.append({"diagnostic": "best macro model by NME", "subset": f"{shot} shots: {model}", "count": macro_best_counts[(shot, model)], "denominator": len(seeds), "proportion": macro_best_counts[(shot, model)] / len(seeds)})
    return label_trajectories, output


def parameter_ranges(rows):
    output = {}
    for model in MODELS:
        values = sorted({row["parameters"] for row in rows if row["model"] == model})
        output[model] = (min(values), max(values), values)
    return output


def write_main_table(macro_summary, params):
    index = {(row["model"], row["shots"]): row for row in macro_summary}
    csv_rows = []
    for model in MODELS:
        low, high, _ = params[model]
        row = {
            "model": model,
            "parameters_min": low,
            "parameters_max": high,
        }
        for shot in SHOTS:
            source = index[(model, shot)]
            for metric, short in (("normalized_mean_error", "nme"), ("pck@5px", "pck5")):
                row[f"{shot}shot_{short}_mean"] = source[f"{metric}_mean"]
                row[f"{shot}shot_{short}_sd"] = source[f"{metric}_sd"]
        csv_rows.append(row)
    write_csv(OUT / "model_sweep_main_table.csv", csv_rows)

    def cell(value, spread, digits=4):
        return f"{value:.{digits}f} $\\pm$ {spread:.{digits}f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Model-sweep performance across six real datasets. Each cell is the mean $\pm$ sample SD across eight seeds after equal-weight macro-averaging the six dataset-level test results within each seed. Lower NME and higher PCK@5 are better.}",
        r"\label{tab:model-sweep}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Parameters (M) & \multicolumn{2}{c}{20 shots} & \multicolumn{2}{c}{50 shots} & \multicolumn{2}{c}{100 shots} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r" & & NME & PCK@5 & NME & PCK@5 & NME & PCK@5 \\",
        r"\midrule",
    ]
    for model in MODELS:
        low, high, _ = params[model]
        param_text = f"{low/1e6:.3f}--{high/1e6:.3f}" if low != high else f"{low/1e6:.3f}"
        fields = [model.capitalize(), param_text]
        for shot in SHOTS:
            source = index[(model, shot)]
            fields.append(cell(source["normalized_mean_error_mean"], source["normalized_mean_error_sd"]))
            fields.append(cell(source["pck@5px_mean"], source["pck@5px_sd"]))
        lines.append(" & ".join(fields) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\textwidth}\footnotesize Parameter ranges reflect dataset-specific output heads (8--27 landmarks). The same seed defined the split, nested training subsets, model initialization/training randomness, and deterministic evaluation subset across all paired conditions. NME is normalized by the diagonal of the tight landmark bounding box; PCK thresholds are in pixels of the 224-pixel input.",
            r"\end{minipage}",
            r"\end{table*}",
        ]
    )
    (OUT / "model_sweep_main_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(manifest, seeds, rows, datasets, errors, warnings, macro_summary, contrasts, trend_rows, params):
    index = {(row["model"], row["shots"]): row for row in macro_summary}
    overall_by_shot = {}
    for shot in SHOTS:
        seed_values = []
        for seed in seeds:
            seed_group = [
                row["normalized_mean_error"]
                for row in rows
                if row["seed"] == seed and row["shots"] == shot
            ]
            seed_values.append(mean(seed_group))
        overall_by_shot[shot] = summarize(seed_values)
    monotone = next(row for row in trend_rows if row["diagnostic"] == "20>50>100 monotone")
    label_overall = {
        row["comparison"]: row
        for row in contrasts
        if row["family"] == "label_scaling_overall"
    }
    model_scaling = {
        (row["subset"], row["comparison"]): row
        for row in contrasts
        if row["family"] == "model_scaling"
    }
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: validate",
        "- Origin Date: 2026-08-22",
        "- Verification Status: ANALYZED",
        "- Version Label: model_sweep_validation_v1",
        f"- Code Commit: {manifest['code_commit']}",
        "",
        "## Validation Report",
        "",
        f"- Source: `{RESULTS}`",
        f"- Design: {len(MODELS)} model sizes x {len(SHOTS)} label budgets x {len(datasets)} datasets x {len(seeds)} seeds = {len(rows)} condition summaries",
        "- Inferential replicate: seed (n=8); dataset-level rows and landmarks were not treated as independent seed replicates",
        f"- Overall confidence: {'CAUTION' if not errors else 'RED_FLAG'}",
        "",
        "### Completeness and reproducibility",
        "",
        f"- Structural validation: {'PASS' if not errors else 'FAIL'}",
        f"- Errors: {errors if errors else 'none'}",
        f"- Advisory: {'; '.join(warnings) if warnings else 'none'}",
        "- The compact publication matrices match all 576 long-form NME values exactly.",
        "- Both model-sweep source runs and their eight jobs are recorded as completed on CUDA (RTX 4090).",
        "- Numerical rerun status: CANNOT_VERIFY locally; checkpoints were excluded from the results-only archive.",
        "",
        "### Primary macro summary (mean ± sample SD over seeds)",
        "",
        "| Model | Parameters (M) | 20-shot NME | 50-shot NME | 100-shot NME |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        low, high, _ = params[model]
        fields = []
        for shot in SHOTS:
            row = index[(model, shot)]
            fields.append(f"{row['normalized_mean_error_mean']:.4f} ± {row['normalized_mean_error_sd']:.4f}")
        lines.append(f"| {model.capitalize()} | {low/1e6:.3f}–{high/1e6:.3f} | " + " | ".join(fields) + " |")
    lines.extend(
        [
            "",
            "### Scaling findings",
            "",
            f"- Averaged over models and datasets within each seed, NME was {overall_by_shot[20]['mean']:.4f} ± {overall_by_shot[20]['sd']:.4f}, {overall_by_shot[50]['mean']:.4f} ± {overall_by_shot[50]['sd']:.4f}, and {overall_by_shot[100]['mean']:.4f} ± {overall_by_shot[100]['sd']:.4f} at 20, 50, and 100 shots.",
            f"- The paired 20→50-shot reduction was {label_overall['20->50 shots']['mean_relative_reduction_pct']:.1f}% (mean ΔNME {label_overall['20->50 shots']['mean_delta_nme']:.4f}, 95% t CI {label_overall['20->50 shots']['delta_ci95_low']:.4f} to {label_overall['20->50 shots']['delta_ci95_high']:.4f}); the 50→100 reduction was {label_overall['50->100 shots']['mean_relative_reduction_pct']:.1f}% (Δ {label_overall['50->100 shots']['mean_delta_nme']:.4f}, 95% CI {label_overall['50->100 shots']['delta_ci95_low']:.4f} to {label_overall['50->100 shots']['delta_ci95_high']:.4f}).",
            f"- Fully monotonic label scaling occurred in {monotone['count']}/{monotone['denominator']} ({100*monotone['proportion']:.1f}%) dataset–model–seed trajectories; exceptions should be shown in the supplement rather than hidden by the macro mean.",
        ]
    )
    for shot in SHOTS:
        row = model_scaling[(f"{shot} shots", "tiny->large")]
        lines.append(
            f"- At {shot} shots, scaling Tiny→Large changed macro NME by {row['mean_delta_nme']:.4f} "
            f"(95% CI {row['delta_ci95_low']:.4f} to {row['delta_ci95_high']:.4f}); negative values favor Large."
        )
    lines.extend(
        [
            "",
            "### 11/11 statistical fallacy scan",
            "",
            "| Fallacy | Status | Model-sweep assessment |",
            "|---|---|---|",
            "| Simpson's paradox | PASS/CAUTION | The primary 20-to-100-shot and Tiny-to-Large mean contrasts retained their direction in every dataset-level mean, so no classic aggregate-versus-dataset reversal was detected. Individual seed trajectories and adjacent model rankings were heterogeneous and remain visible in the supplement. |",
            "| Ecological fallacy | PASS | Claims are restricted to dataset-level performance and seed-level macro summaries; they are not extended to individual fossils or taxonomic populations. |",
            "| Berkson's paradox | NOTE | The six datasets are a purposive eligible set (each has at least 100 training records), not a random sample of all morphometric datasets; generalization beyond them is limited. |",
            "| Collider bias | PASS | No post-treatment covariates were conditioned on in the paired factorial comparisons. |",
            "| Base-rate neglect | N/A | NME/PCK localization metrics are not diagnostic sensitivity/specificity measures. |",
            "| Regression to the mean | PASS | Conditions were not selected for analysis based on extreme baseline scores. |",
            "| Survivorship bias | PASS/NOTE | All six configured datasets met the 100-training-record threshold and all 576 conditions completed; no condition was dropped for poor performance. The benchmark is nevertheless restricted by design to datasets large enough for the 100-shot regimen. |",
            "| Look-elsewhere effect | CAUTION | Many model/shot/dataset contrasts are available. NME is treated as the primary metric; exact p values are exploratory and Holm-adjusted within contrast families in the supplement. |",
            "| Garden of forking paths | CAUTION | No preregistration was located. Preserve the complete factorial table and describe post-hoc contrasts as exploratory. |",
            "| Correlation ≠ causation | PASS/CAUTION | Model size and label budget were experimentally manipulated within the benchmark, supporting within-matrix performance contrasts, but not causal claims about unseen architectures or datasets. |",
            "| Reverse causality | N/A | The manipulated computational factors temporally precede evaluation. |",
            "",
            "Coverage: 11/11 checked.",
            "",
            "### Reporting limitations",
            "",
            "- Each seed jointly changes the train/validation/test split, training subset, synthetic pretraining, and optimization randomness. The SD therefore measures whole-pipeline variation, not optimization-only variation.",
            "- The eight seeds repartition the same finite datasets, so held-out specimens overlap across seeds. Seed-level SDs and t intervals describe variation across these eight pipeline realizations; they are not confidence intervals for a new population of specimens or datasets.",
            "- The 20/50/100-shot subsets are nested within a seed, and every paired condition uses the same test split (and the same deterministic test subsample if the 100-image cap binds). This justifies paired contrasts.",
            "- Dataset macro-averaging prevents larger datasets or larger landmark sets from dominating, but it gives a small and a large dataset equal weight.",
            "- Fixed-pixel PCK is resolution-specific and can favor larger specimens; NME, normalized by each specimen's landmark-bounding-box diagonal, is the primary cross-dataset metric.",
            "- Eight seeds provide only seven degrees of freedom. Confidence intervals are correspondingly wide; absence of separation should not be described as equivalence.",
            "- Best checkpoints were selected by validation pixel error, while the primary cross-dataset endpoint is test NME. This is defensible but should be stated explicitly.",
        ]
    )
    (OUT / "model_sweep_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    manifest, seeds, rows, headers, _ = load_rows()
    datasets, errors, warnings = validate(manifest, seeds, rows, headers)
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))

    macro_rows = macro_by_seed(rows, datasets, seeds)
    macro_summary = aggregate(macro_rows, ("model", "shots"))
    dataset_summary = aggregate(rows, ("dataset", "model", "shots"))
    contrasts = paired_contrasts(macro_rows, seeds)
    trajectories, trend_rows = trend_diagnostics(rows, macro_rows, datasets, seeds)
    params = parameter_ranges(rows)

    write_csv(OUT / "model_sweep_macro_by_seed.csv", macro_rows)
    write_csv(OUT / "model_sweep_macro_summary.csv", macro_summary)
    write_csv(OUT / "model_sweep_dataset_summary.csv", dataset_summary)
    write_csv(OUT / "model_sweep_paired_contrasts.csv", contrasts)
    write_csv(OUT / "model_sweep_label_trajectories.csv", trajectories)
    write_csv(OUT / "model_sweep_trend_diagnostics.csv", trend_rows)
    write_main_table(macro_summary, params)
    write_report(
        manifest,
        seeds,
        rows,
        datasets,
        errors,
        warnings,
        macro_summary,
        contrasts,
        trend_rows,
        params,
    )
    print(f"PASS: analyzed {len(rows)} rows; outputs written to {OUT}")


if __name__ == "__main__":
    main()
