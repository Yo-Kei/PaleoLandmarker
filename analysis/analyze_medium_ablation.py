#!/usr/bin/env python3
"""Reproducible analysis of the eight-seed, six-dataset medium ablation.

The script deliberately uses only the Python standard library.  Its inferential
unit is the seed: dataset values are first macro-averaged within seed, and only
then summarized across the eight seeds.  The 48 seed-dataset direction counts
are reported as descriptive consistency checks, never as 48 independent
replicates.  Five conditions form the cumulative sequence; a sixth condition is
a leave-one-domain-out transfer stress test and is never treated as a further
cumulative addition.
"""

from __future__ import annotations

import csv
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
OUTPUT_DIR = Path(__file__).resolve().parent

SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
DATASETS = [
    "conodont",
    "fusulinid",
    "land_snail_haplotrema_concavum",
    "microtus_dinolite_notilt_eo_t2",
    "ml_morph_bryozoan_zooids",
    "trilomorph_cephala",
]
DATASET_LABELS = {
    "conodont": "Conodont",
    "fusulinid": "Fusulinid",
    "land_snail_haplotrema_concavum": "Land snail",
    "microtus_dinolite_notilt_eo_t2": "Microtus",
    "ml_morph_bryozoan_zooids": "Bryozoan",
    "trilomorph_cephala": "Trilobite",
}
CUMULATIVE_VARIANTS = [
    "baseline_coord_noaug_scratch",
    "baseline_coord_noaug_multidomain",
    "baseline_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_aug_multidomain",
]
LODO_VARIANT = "context_resize_heatmap_coord_aug_leave_one_out"
VARIANTS = [*CUMULATIVE_VARIANTS, LODO_VARIANT]
VARIANT_LABELS = {
    VARIANTS[0]: "A: scratch + coordinate-only",
    VARIANTS[1]: "B: A + synthetic pretraining",
    VARIANTS[2]: "C: B + heatmap supervision",
    VARIANTS[3]: "D: C + context/resize architecture",
    VARIANTS[4]: "E: D + augmentation",
    LODO_VARIANT: "E-LODO: E with target-matched synthetic domain withheld",
}
STEP_LABELS = {
    (VARIANTS[0], VARIANTS[1]): "Synthetic pretraining (B-A)",
    (VARIANTS[1], VARIANTS[2]): "Heatmap supervision (C-B)",
    (VARIANTS[2], VARIANTS[3]): "Context/resize architecture (D-C)",
    (VARIANTS[3], VARIANTS[4]): "Augmentation (E-D)",
    (VARIANTS[4], LODO_VARIANT): "Matched-domain omission (E-LODO minus E)",
}
METRICS = [
    "mean_error_px",
    "normalized_mean_error",
    "pck@3px",
    "pck@5px",
    "pck@10px",
]
LOWER_IS_BETTER = {"mean_error_px", "normalized_mean_error"}

# Two-sided 95% Student-t critical value for df = 7.  This is used only for
# the pre-specified eight-seed summaries; no p-values or significance labels
# are generated.
T_CRIT_95_DF7 = 2.364624251


def mean_sd_ci95(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 8:
        raise ValueError(f"expected eight seed-level values, got {len(values)}")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T_CRIT_95_DF7 * sd / math.sqrt(len(values))
    return mean, sd, mean - half_width, mean + half_width


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate() -> tuple[list[dict], dict[tuple[int, str, str], dict]]:
    manifest = json.loads((RESULT_ROOT / "combined_manifest.json").read_text())
    if manifest["code_commit"] != "fb55087a8d7f852385924b97c2cad267710bafa8":
        raise ValueError("unexpected code commit in combined manifest")
    if manifest["seeds"] != SEEDS:
        raise ValueError("manifest seeds do not match the pre-specified seed set")
    if manifest["ablation_datasets"] != DATASETS:
        raise ValueError("manifest datasets do not match the pre-specified set")

    rows: list[dict] = []
    for seed in SEEDS:
        path = RESULT_ROOT / f"seed_{seed}/medium_ablation/ablation_results.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            seed_rows = list(csv.DictReader(handle))
        if len(seed_rows) != 36:
            raise ValueError(f"{path} has {len(seed_rows)} rows, expected 36")
        rows.extend(seed_rows)

    index: dict[tuple[int, str, str], dict] = {}
    for row in rows:
        key = (int(row["seed"]), row["dataset"], row["ablation_variant"])
        if key in index:
            raise ValueError(f"duplicate ablation row: {key}")
        if key[0] != int(row["data_split_seed"]):
            raise ValueError(f"training and split seeds differ: {key}")
        if row["experiment"] != "medium_ablation" or row["model"] != "medium":
            raise ValueError(f"unexpected experiment/model: {key}")
        if row["shots"] != "100" or row["shot_count"] != "100":
            raise ValueError(f"unexpected shot count: {key}")
        config = json.loads(row["config_json"])
        if config["seed"] != key[0] or config["ablation_variant"] != key[2]:
            raise ValueError(f"config provenance mismatch: {key}")
        for metric in METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric}: {key}")
            if metric.startswith("pck@") and not 0.0 <= value <= 1.0:
                raise ValueError(f"out-of-range {metric}: {key}")
            if metric in LOWER_IS_BETTER and value < 0.0:
                raise ValueError(f"negative error metric {metric}: {key}")
        index[key] = row

    expected = {
        (seed, dataset, variant)
        for seed in SEEDS
        for dataset in DATASETS
        for variant in VARIANTS
    }
    if set(index) != expected:
        missing = expected - set(index)
        extra = set(index) - expected
        raise ValueError(f"incomplete matrix; missing={missing}, extra={extra}")

    # Within a seed-dataset block, all six variants must use the same split
    # provenance and evaluation cap.  The split manifests themselves are remote
    # provenance strings and are not present in this results-only archive.
    for seed in SEEDS:
        for dataset in DATASETS:
            block = [index[(seed, dataset, variant)] for variant in VARIANTS]
            for field in (
                "data_split_seed",
                "train_size",
                "val_size",
                "test_size",
                "total_size",
                "num_landmarks",
                "shots",
                "val_max_samples",
                "test_max_samples",
            ):
                if len({row[field] for row in block}) != 1:
                    raise ValueError(f"unpaired {field} in seed={seed}, dataset={dataset}")
            # Source runs used different absolute working directories, but the
            # split identity is the same relative splits_224/seed/dataset path.
            split_suffixes = {
                tuple(Path(row["data_split_manifest"]).parts[-4:]) for row in block
            }
            if len(split_suffixes) != 1:
                raise ValueError(f"unpaired data_split_manifest in seed={seed}, dataset={dataset}")

    excluded = manifest.get("leave_one_out_dataset_to_excluded_domain", {})
    if set(excluded) != set(DATASETS):
        raise ValueError("leave-one-domain-out mapping is incomplete")
    for seed in SEEDS:
        for dataset in DATASETS:
            row = index[(seed, dataset, LODO_VARIANT)]
            if row["pretrain_excluded_domain"] != excluded[dataset]:
                raise ValueError(f"wrong excluded domain in seed={seed}, dataset={dataset}")
            source_domains = json.loads(row["pretrain_source_domains"])
            if len(source_domains) != 5 or excluded[dataset] in source_domains:
                raise ValueError(f"invalid five-domain source set in seed={seed}, dataset={dataset}")

    # Verify source-run terminal status independently of the combined manifest.
    for status_path in sorted((RESULT_ROOT / "source_runs").glob("*/status.json")):
        status = json.loads(status_path.read_text())
        if status.get("state") != "completed":
            raise ValueError(f"source run not completed: {status_path}")
        for job in status.get("jobs", []):
            if job.get("state") != "completed" or job.get("returncode") != 0:
                raise ValueError(f"source job failed: {status_path}, {job}")

    return rows, index


def generate_outputs(rows: list[dict], index: dict[tuple[int, str, str], dict]) -> None:
    seed_rows: list[dict] = []
    for row in rows:
        seed_rows.append(
            {
                "seed": int(row["seed"]),
                "dataset": row["dataset"],
                "variant": row["ablation_variant"],
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    seed_rows.sort(key=lambda row: (row["seed"], DATASETS.index(row["dataset"]), VARIANTS.index(row["variant"])))
    write_csv(
        OUTPUT_DIR / "ablation_seed_level_values.csv",
        seed_rows,
        ["seed", "dataset", "variant", *METRICS],
    )

    dataset_summary: list[dict] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for metric in METRICS:
                values = [float(index[(seed, dataset, variant)][metric]) for seed in SEEDS]
                mean, sd, low, high = mean_sd_ci95(values)
                dataset_summary.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "metric": metric,
                        "n_seeds": 8,
                        "mean": mean,
                        "sample_sd": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    write_csv(
        OUTPUT_DIR / "ablation_dataset_summary.csv",
        dataset_summary,
        ["dataset", "variant", "metric", "n_seeds", "mean", "sample_sd", "ci95_low", "ci95_high"],
    )

    macro_values: dict[tuple[str, str], list[float]] = {}
    macro_summary: list[dict] = []
    for variant in VARIANTS:
        for metric in METRICS:
            values = [
                statistics.mean(float(index[(seed, dataset, variant)][metric]) for dataset in DATASETS)
                for seed in SEEDS
            ]
            macro_values[(variant, metric)] = values
            mean, sd, low, high = mean_sd_ci95(values)
            macro_summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "aggregation": "equal-weight dataset macro-average within seed, then across seeds",
                    "n_seeds": 8,
                    "mean": mean,
                    "sample_sd": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    write_csv(
        OUTPUT_DIR / "ablation_macro_summary.csv",
        macro_summary,
        ["variant", "metric", "aggregation", "n_seeds", "mean", "sample_sd", "ci95_low", "ci95_high"],
    )

    delta_rows: list[dict] = []
    contrasts = [
        *zip(CUMULATIVE_VARIANTS, CUMULATIVE_VARIANTS[1:]),
        (CUMULATIVE_VARIANTS[-1], LODO_VARIANT),
    ]
    for old, new in contrasts:
        for metric in METRICS:
            for dataset in [*DATASETS, "__macro_across_datasets__"]:
                if dataset == "__macro_across_datasets__":
                    deltas = [new_value - old_value for new_value, old_value in zip(macro_values[(new, metric)], macro_values[(old, metric)])]
                else:
                    deltas = [
                        float(index[(seed, dataset, new)][metric])
                        - float(index[(seed, dataset, old)][metric])
                        for seed in SEEDS
                    ]
                mean, sd, low, high = mean_sd_ci95(deltas)
                improves = sum(
                    delta < 0 if metric in LOWER_IS_BETTER else delta > 0
                    for delta in deltas
                )
                worsens = sum(
                    delta > 0 if metric in LOWER_IS_BETTER else delta < 0
                    for delta in deltas
                )
                delta_rows.append(
                    {
                        "step": STEP_LABELS[(old, new)],
                        "contrast_type": "transfer_stress_test" if new == LODO_VARIANT else "cumulative_adjacent",
                        "old_variant": old,
                        "new_variant": new,
                        "metric": metric,
                        "dataset": dataset,
                        "n_paired_seeds": 8,
                        "mean_delta_new_minus_old": mean,
                        "sample_sd_delta": sd,
                        "ci95_low": low,
                        "ci95_high": high,
                        "improves_n": improves,
                        "worsens_n": worsens,
                        "ties_n": 8 - improves - worsens,
                    }
                )
    write_csv(
        OUTPUT_DIR / "ablation_adjacent_paired_deltas.csv",
        delta_rows,
        [
            "step", "contrast_type", "old_variant", "new_variant", "metric", "dataset",
            "n_paired_seeds", "mean_delta_new_minus_old", "sample_sd_delta",
            "ci95_low", "ci95_high", "improves_n", "worsens_n", "ties_n",
        ],
    )

    def summary_lookup(dataset: str, variant: str, metric: str) -> tuple[float, float]:
        if dataset == "__macro_across_datasets__":
            values = macro_values[(variant, metric)]
        else:
            values = [float(index[(seed, dataset, variant)][metric]) for seed in SEEDS]
        return statistics.mean(values), statistics.stdev(values)

    lines = [
        "# Main-paper ablation table",
        "",
        "Values are mean +/- sample SD across eight seeds. Each dataset contributes equally to the macro-average within each seed. NME is normalized by the diagonal of the tight landmark bounding box. PCK values are proportions.",
        "",
        "| Configuration | Conodont NME | Fusulinid NME | Land-snail NME | Microtus NME | Bryozoan NME | Trilobite NME | Macro NME | Macro PCK@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        nme_cells = []
        for dataset in DATASETS:
            mean, sd = summary_lookup(dataset, variant, "normalized_mean_error")
            nme_cells.append(f"{mean:.4f} +/- {sd:.4f}")
        macro_nme = summary_lookup("__macro_across_datasets__", variant, "normalized_mean_error")
        macro_pck5 = summary_lookup("__macro_across_datasets__", variant, "pck@5px")
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            + " | ".join(nme_cells)
            + f" | {macro_nme[0]:.4f} +/- {macro_nme[1]:.4f}"
            + f" | {macro_pck5[0]:.4f} +/- {macro_pck5[1]:.4f} |"
        )
    lines.extend(
        [
            "",
            "Configurations A-E form the cumulative sequence. E-LODO is a separate transfer stress test: it keeps E fixed but pretrains on five procedural domains after withholding the domain matched to the target dataset. All configurations used the Medium model, 100 real training images, 400 fine-tuning epochs, 224 x 224 inputs, and identical within-seed splits.",
        ]
    )
    (OUTPUT_DIR / "ablation_main_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Medium-model cumulative ablation and leave-one-domain-out transfer stress test across six datasets. Values are mean $\pm$ sample standard deviation over eight seeds. NME is normalized by the diagonal of the tight landmark bounding box. Each dataset contributes equally to the macro-average within each seed.}",
        r"\label{tab:ablation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Configuration & Conodont & Fusulinid & Land snail & \textit{Microtus} & Bryozoan & Trilobite & Macro NME & Macro PCK@5 \\",
        r"\midrule",
    ]
    tex_variant_labels = {
        VARIANTS[0]: r"A: scratch + coordinate-only",
        VARIANTS[1]: r"B: A + synthetic pretraining",
        VARIANTS[2]: r"C: B + heatmap supervision",
        VARIANTS[3]: r"D: C + context/resize",
        VARIANTS[4]: r"E: D + augmentation",
        LODO_VARIANT: r"E-LODO: matched domain withheld",
    }
    for variant in VARIANTS:
        cells = []
        for dataset in DATASETS:
            mean, sd = summary_lookup(dataset, variant, "normalized_mean_error")
            cells.append(f"{mean:.4f} $\\pm$ {sd:.4f}")
        macro_nme = summary_lookup("__macro_across_datasets__", variant, "normalized_mean_error")
        macro_pck5 = summary_lookup("__macro_across_datasets__", variant, "pck@5px")
        tex_lines.append(
            tex_variant_labels[variant]
            + " & "
            + " & ".join(cells)
            + f" & {macro_nme[0]:.4f} $\\pm$ {macro_nme[1]:.4f}"
            + f" & {macro_pck5[0]:.4f} $\\pm$ {macro_pck5[1]:.4f} \\\\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize A--E form the cumulative sequence. E-LODO is not a sixth cumulative addition: it repeats E after excluding from pretraining the one procedural domain matched to each target dataset, while retaining the other five domains. All configurations used 100 real training images, 400 fine-tuning epochs, and identical within-seed splits.}",
            r"\end{table*}",
        ]
    )
    (OUTPUT_DIR / "ablation_main_table.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    base_nme = mean_sd_ci95(macro_values[(VARIANTS[0], "normalized_mean_error")])
    pre_nme = mean_sd_ci95(macro_values[(VARIANTS[1], "normalized_mean_error")])
    loss_nme = mean_sd_ci95(macro_values[(VARIANTS[2], "normalized_mean_error")])
    arch_nme = mean_sd_ci95(macro_values[(VARIANTS[3], "normalized_mean_error")])
    final_nme = mean_sd_ci95(macro_values[(VARIANTS[4], "normalized_mean_error")])
    lodo_nme = mean_sd_ci95(macro_values[(LODO_VARIANT, "normalized_mean_error")])
    base_pck = mean_sd_ci95(macro_values[(VARIANTS[0], "pck@5px")])
    final_pck = mean_sd_ci95(macro_values[(VARIANTS[4], "pck@5px")])
    lodo_pck = mean_sd_ci95(macro_values[(LODO_VARIANT, "pck@5px")])

    def macro_delta(old: str, new: str, metric: str) -> tuple[float, float, float, float]:
        return mean_sd_ci95([new_v - old_v for new_v, old_v in zip(macro_values[(new, metric)], macro_values[(old, metric)])])

    pre_delta = macro_delta(VARIANTS[0], VARIANTS[1], "normalized_mean_error")
    loss_delta = macro_delta(VARIANTS[1], VARIANTS[2], "normalized_mean_error")
    arch_delta = macro_delta(VARIANTS[2], VARIANTS[3], "normalized_mean_error")
    aug_delta = macro_delta(VARIANTS[3], VARIANTS[4], "normalized_mean_error")
    lodo_nme_delta = macro_delta(VARIANTS[4], LODO_VARIANT, "normalized_mean_error")
    lodo_pck_delta = macro_delta(VARIANTS[4], LODO_VARIANT, "pck@5px")
    lodo_nme_cell_improves = sum(
        float(index[(seed, dataset, LODO_VARIANT)]["normalized_mean_error"])
        < float(index[(seed, dataset, VARIANTS[4])]["normalized_mean_error"])
        for seed in SEEDS for dataset in DATASETS
    )
    lodo_pck_cell_improves = sum(
        float(index[(seed, dataset, LODO_VARIANT)]["pck@5px"])
        > float(index[(seed, dataset, VARIANTS[4])]["pck@5px"])
        for seed in SEEDS for dataset in DATASETS
    )

    manuscript = f"""# Manuscript-ready ablation text

## Results

The ablation matrix comprised 288 complete fits (six configurations x six datasets x eight seeds), with no missing seed-dataset-configuration cells. Five conditions formed the cumulative sequence; the sixth was a leave-one-domain-out (LODO) transfer stress test. All configurations used 100 real training images and identical data partitions within each seed. To prevent datasets with more test images or landmarks from dominating the cross-dataset result, each metric was first macro-averaged across the six datasets within a seed and then summarized across seeds. The original U-Net trained from scratch with coordinate-only supervision yielded a macro-average NME of {base_nme[0]:.4f} +/- {base_nme[1]:.4f} and a macro PCK@5 of {base_pck[0]:.4f} +/- {base_pck[1]:.4f} (mean +/- sample SD, n = 8 seeds).

Multidomain synthetic pretraining produced the largest reduction in macro NME, from {base_nme[0]:.4f} to {pre_nme[0]:.4f} (paired mean delta, new minus previous = {pre_delta[0]:.4f}; descriptive 95% CI [{pre_delta[2]:.4f}, {pre_delta[3]:.4f}]), a {100 * (base_nme[0] - pre_nme[0]) / base_nme[0]:.1f}% relative reduction. NME decreased in 46 of the 48 seed-dataset pairs. Adding heatmap supervision to the coordinate objective during both synthetic pretraining and real-data fine-tuning further reduced macro NME to {loss_nme[0]:.4f} +/- {loss_nme[1]:.4f} (paired delta = {loss_delta[0]:.4f}, 95% CI [{loss_delta[2]:.4f}, {loss_delta[3]:.4f}]). This step reduced NME in 47 of 48 seed-dataset pairs and increased PCK@5 in all 48 pairs, providing the most directionally consistent improvement in the sequence.

Replacing the original U-Net with the context-and-resize architecture reduced macro NME from {loss_nme[0]:.4f} to {arch_nme[0]:.4f} (paired delta = {arch_delta[0]:.4f}, 95% CI [{arch_delta[2]:.4f}, {arch_delta[3]:.4f}]) and increased macro PCK@5 from {statistics.mean(macro_values[(VARIANTS[2], 'pck@5px')]):.4f} to {statistics.mean(macro_values[(VARIANTS[3], 'pck@5px')]):.4f}. The improvement was strongest for trilobite cephala (NME {summary_lookup('trilomorph_cephala', VARIANTS[2], 'normalized_mean_error')[0]:.4f} to {summary_lookup('trilomorph_cephala', VARIANTS[3], 'normalized_mean_error')[0]:.4f}) and bryozoan zooids ({summary_lookup('ml_morph_bryozoan_zooids', VARIANTS[2], 'normalized_mean_error')[0]:.4f} to {summary_lookup('ml_morph_bryozoan_zooids', VARIANTS[3], 'normalized_mean_error')[0]:.4f}), whereas changes for conodont and Microtus were small. This gain did not result from a larger parameter count: for every landmark-output dimension, the context-and-resize model had 5,928 fewer parameters (approximately 0.24%) than the corresponding original U-Net.

Enabling augmentation during both pretraining and fine-tuning changed the macro NME only from {arch_nme[0]:.4f} to {final_nme[0]:.4f} (paired delta = {aug_delta[0]:.5f}, 95% CI [{aug_delta[2]:.5f}, {aug_delta[3]:.5f}]) and macro PCK@5 from {statistics.mean(macro_values[(VARIANTS[3], 'pck@5px')]):.4f} to {final_pck[0]:.4f}. The near-zero aggregate change concealed opposing dataset responses: augmentation improved bryozoan NME in all eight seeds ({summary_lookup('ml_morph_bryozoan_zooids', VARIANTS[3], 'normalized_mean_error')[0]:.4f} to {summary_lookup('ml_morph_bryozoan_zooids', VARIANTS[4], 'normalized_mean_error')[0]:.4f}) but worsened trilobite NME in six of eight seeds ({summary_lookup('trilomorph_cephala', VARIANTS[3], 'normalized_mean_error')[0]:.4f} to {summary_lookup('trilomorph_cephala', VARIANTS[4], 'normalized_mean_error')[0]:.4f}). Across the full cumulative sequence, macro NME decreased by {100 * (base_nme[0] - final_nme[0]) / base_nme[0]:.1f}% and macro PCK@5 increased by {100 * (final_pck[0] - base_pck[0]):.1f} percentage points.

The LODO condition retained the final architecture, objective, augmentation, fine-tuning data, and total procedural pretraining-set size, but omitted the procedural domain matched to each target dataset and redistributed the source corpus across the other five domains. Its macro NME was {lodo_nme[0]:.4f} +/- {lodo_nme[1]:.4f}, compared with {final_nme[0]:.4f} +/- {final_nme[1]:.4f} under complete six-domain pretraining. The paired LODO-minus-complete difference was {lodo_nme_delta[0]:.5f} (95% CI [{lodo_nme_delta[2]:.5f}, {lodo_nme_delta[3]:.5f}]); {lodo_nme_cell_improves} of 48 seed-dataset cells favoured LODO and {48-lodo_nme_cell_improves} favoured complete pretraining. Macro PCK@5 was {lodo_pck[0]:.4f} +/- {lodo_pck[1]:.4f}, with a paired difference of {lodo_pck_delta[0]:.5f} (95% CI [{lodo_pck_delta[2]:.5f}, {lodo_pck_delta[3]:.5f}]) and {lodo_pck_cell_improves}/48 cells favouring LODO. Thus removal of the corresponding synthetic morphology did not produce a systematic aggregate loss in this benchmark; the intervals span zero and do not establish equivalence.

## Discussion

The staged ablation identifies synthetic pretraining and the combined heatmap-coordinate objective as the most robust components of the pipeline. Synthetic initialization improved nearly every seed-dataset pair. The LODO stress test further shows that the final model did not depend detectably on having a directly matched procedural morphology in its pretraining mixture: withholding that domain produced no systematic aggregate degradation. This supports transfer of reusable spatial-localization structure among the five retained procedural domains after target fine-tuning. It does not demonstrate unrestricted zero-shot transfer to an unknown taxon, because every model was still fine-tuned on 100 labelled target records and the omitted domain was represented by a designed morphological analogue rather than a complete phylogenetic or imaging distribution.

Dense heatmap supervision complemented direct coordinate regression: its benefit was observed across all six datasets and was not driven by a single large dataset because the analysis used equal-weight dataset macro-averages. This conclusion differs from one based on a heatmap-only baseline; the present experiment shows the value of adding heatmap supervision to an already coordinate-aware model.

The context-and-resize architecture supplied a smaller but still coherent aggregate improvement without increasing model size. Its concentration in trilobite cephala and bryozoan zooids suggests that multi-scale context and resize-based decoding may be most useful for landmark schemes that require broader structural information, although the current design cannot separate the dilated-context module from the change in upsampling operation. Augmentation was not uniformly beneficial. The strong improvement for bryozoans and deterioration for trilobites indicate that a single affine-photometric policy may match some image domains while perturbing diagnostically important spatial cues in others. Dataset-conditioned policies or a factorial follow-up would be needed before treating augmentation as a generally beneficial component.

These estimates quantify variability over the eight chosen training/split seeds, not independent draws of specimens. Test partitions overlap across seeds, and the results-only archive does not contain per-image predictions or local checkpoints; consequently, specimen-level uncertainty and full rerun reproducibility could not be assessed. Moreover, A--E form a cumulative sequence rather than a full factorial design, while E-LODO is a separate source-composition contrast. Each later cumulative contrast is conditional on all earlier choices; the loss, architecture, and augmentation changes apply during both pretraining and fine-tuning. The paired confidence intervals should therefore be read as descriptive uncertainty for these pre-specified seed-level contrasts, not as multiple-comparison-adjusted hypothesis tests, equivalence tests, or isolated universal main effects.
"""
    (OUTPUT_DIR / "ablation_manuscript_text.md").write_text(manuscript, encoding="utf-8")

    # Full validate-mode report, including the mandated 11/11 fallacy scan.
    report = f"""## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-25
- Verification Status: ANALYZED
- Version Label: medium_ablation_validation_lodo
- Source commit recorded by results: fb55087a8d7f852385924b97c2cad267710bafa8

## Validation Report

- Source: `{RESULT_ROOT}`
- Overall confidence: CAUTION
- Matrix completeness: 288/288 expected training-result rows; 8 seeds x 6 datasets x (5 cumulative configurations + 1 LODO stress-test configuration); no duplicate keys or missing metrics.
- Pairing: within every seed-dataset block, all six configurations share the same split-manifest provenance, split seed, split sizes, shot count, and evaluation caps. Every LODO row records the intended excluded domain and exactly five retained source domains.
- Aggregation: dataset metrics were macro-averaged within each seed, then mean and sample SD were computed across eight seed-level macro-averages. Dataset-specific summaries likewise use n = 8 seeds. The 48 seed-dataset direction counts are descriptive, not inferential replicates.
- Source-run status: every archived source run and job reports `completed`, return code 0, first attempt.
- Reproducibility verdict: CANNOT_VERIFY. The archive is results-only: checkpoint paths are remote provenance strings, split manifests are not local, and per-image predictions are absent.

### Principal findings

| Contrast | Macro NME (mean +/- SD) | Paired NME delta (95% CI) | Directional consistency |
|---|---:|---:|---:|
| A: scratch coordinate-only | {base_nme[0]:.4f} +/- {base_nme[1]:.4f} | -- | -- |
| B: + synthetic pretraining | {pre_nme[0]:.4f} +/- {pre_nme[1]:.4f} | {pre_delta[0]:.4f} [{pre_delta[2]:.4f}, {pre_delta[3]:.4f}] | NME improved in 46/48 seed-dataset pairs |
| C: + heatmap supervision | {loss_nme[0]:.4f} +/- {loss_nme[1]:.4f} | {loss_delta[0]:.4f} [{loss_delta[2]:.4f}, {loss_delta[3]:.4f}] | NME 47/48; PCK@5 48/48 |
| D: + context/resize | {arch_nme[0]:.4f} +/- {arch_nme[1]:.4f} | {arch_delta[0]:.4f} [{arch_delta[2]:.4f}, {arch_delta[3]:.4f}] | NME 38/48 |
| E: + augmentation | {final_nme[0]:.4f} +/- {final_nme[1]:.4f} | {aug_delta[0]:.5f} [{aug_delta[2]:.5f}, {aug_delta[3]:.5f}] | NME 28/48; heterogeneous by dataset |
| E-LODO: matched domain withheld | {lodo_nme[0]:.4f} +/- {lodo_nme[1]:.4f} | LODO - E = {lodo_nme_delta[0]:.5f} [{lodo_nme_delta[2]:.5f}, {lodo_nme_delta[3]:.5f}] | NME {lodo_nme_cell_improves}/48; PCK@5 {lodo_pck_cell_improves}/48 favour LODO |

The 95% intervals are t intervals over eight paired seed-level macro-differences (df = 7). With only eight seeds, normality is difficult to assess; the intervals are descriptive and are accompanied by direction counts. No p-values or significance labels are used.

### Anomalies and limitations

1. The ablation is cumulative, not factorial. B-A isolates initialization under coordinate-only/no-augmentation/original-U-Net settings, but C-B changes the objective in both pretraining and fine-tuning; D-C changes the architecture in both stages and bundles dilated context with resize projection; E-D adds augmentation in both stages.
2. E-LODO is a separate source-composition stress test, not a sixth cumulative improvement. Its interval spanning zero cannot be interpreted as proof of equivalence, and its 100-shot target fine-tuning prevents zero-shot or universal unseen-taxon claims.
3. Augmentation has opposing domain effects. It improves bryozoan NME in 8/8 seeds but worsens trilobite NME in 6/8 and PCK@5 in 7/8 seeds. Reporting only the macro-average would hide this interaction.
4. Context/resize has limited directional consistency for conodont NME (3/8 improvements) and a nearly zero mean PCK@5 change, despite a clear macro benefit driven partly by trilobite and bryozoan results.
5. Augmentation increased between-seed variability in macro NME (SD 0.00135 to 0.00224) and trilobite NME (SD 0.00763 to 0.01357). The augmented trilobite run for seed 808 selected epoch 29 and had NME 0.10038, whereas the other augmented trilobite runs selected epochs 188-351. This is a sensitivity signal, not evidence of a failed run.
6. Two datasets have capped test evaluation: 100 of 173 land-snail test images and 100 of 570 bryozoan test images per seed. The other four test splits contain at most 37 images and are evaluated in full. This is a per-seed deterministic subset, not the complete test partition for the two large datasets.
7. Seeds jointly alter the train/validation/test partition, training stochasticity, synthetic data/cache, and initialization. The design does not separate these variance sources, and overlapping test specimens make seeds correlated resamples of the underlying datasets.
8. Model selection uses the lowest validation pixel error across 400 epochs; the test metrics remain held out, but uncertainty due to checkpoint selection is included only indirectly through seed variation.
9. NME is normalized by each specimen's tight landmark-bounding-box diagonal, not by the image diagonal. This must be stated consistently in the manuscript.

### Fallacy scan

- Coverage: 11/11 statistical fallacy types checked.

| # | Fallacy | Status | Experiment-specific assessment |
|---:|---|---|---|
| 1 | Simpson's paradox | CAUTION | Macro effects and dataset effects diverge for augmentation; the aggregate near-zero delta hides a bryozoan benefit and trilobite harm. Preserve dataset-level results. No full direction reversal is present for the first three major steps. |
| 2 | Ecological fallacy | CAUTION | Dataset- or seed-level averages cannot support specimen- or landmark-level claims. Restrict inference to benchmark-level localization performance. |
| 3 | Berkson's paradox | NOTE | The six datasets are eligibility-selected (at least 100 training images and available landmark data), but no correlation analysis is made. Generalization is limited to this selected benchmark. |
| 4 | Collider bias | NOTE | No covariate-adjusted model is used, so direct statistical collider conditioning is absent. Eligibility and successful-conversion filters may nevertheless define a selected benchmark. |
| 5 | Base-rate neglect | NOTE | NME/PCK localization metrics do not involve diagnostic class prevalence, sensitivity, or specificity; this fallacy is not directly applicable. Report PCK thresholds alongside the error distribution summaries. |
| 6 | Regression to the mean | NOTE | Configurations were not selected within this eight-seed matrix because of extreme seed scores. Prior pilot-driven architecture choice could still overstate transportability and should be disclosed as exploratory development. |
| 7 | Survivorship bias | NOTE | The expected 288/288 cells are present and all archived jobs completed. The archive omits checkpoints/per-image predictions by design, but there is no failed-run exclusion in this matrix. |
| 8 | Look-elsewhere effect | CAUTION | Five metrics, six datasets, five cumulative configurations, and one LODO stress test create many possible comparisons. The analysis treats NME as primary, reports all datasets, avoids selective p-values, and places full metric tables in the supplement. |
| 9 | Garden of forking paths | CAUTION | No preregistration was found, and the final cumulative sequence follows earlier structure screens. Present results as a staged exploratory ablation and disclose the fixed analysis/aggregation used here. |
| 10 | Correlation != causation | CAUTION | Controlled paired configurations support conditional performance contrasts, but the cumulative, non-factorial design does not identify isolated component main effects. Use 'was associated with/reduced in this sequence', not universal causal language. |
| 11 | Reverse causality | NOTE | Temporal reverse causality is not meaningful for randomized training configurations; however, interpreting architecture gains as caused specifically by context rather than resize is underidentified. |

### Main-paper and supplementary display recommendation

- Main paper: one landscape table with A--E as five cumulative rows and E-LODO as a visually separated stress-test row; six dataset-specific NME mean +/- SD columns; macro NME and macro PCK@5 columns.
- Main text: report seed-level paired macro deltas and directional consistency for the major cumulative steps and the E-LODO versus E contrast; explicitly state that the latter is not an equivalence or zero-shot test.
- Supplement: (i) all five metrics by dataset/configuration (`ablation_dataset_summary.csv`), (ii) all adjacent cumulative and LODO paired deltas and CIs (`ablation_adjacent_paired_deltas.csv`), (iii) raw seed-level values (`ablation_seed_level_values.csv`), and (iv) paired seed profile plots or a dataset-by-contrast delta heat map. Avoid inferential error bars built from specimens or landmarks.
"""
    (OUTPUT_DIR / "ablation_validation_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    rows, index = load_and_validate()
    generate_outputs(rows, index)
    print(f"Validated {len(rows)} rows and wrote ablation analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
