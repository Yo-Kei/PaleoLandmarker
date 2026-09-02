import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401
from paleolandmarker.baselines import CANONICAL_BASELINE_REGISTRY, canonical_baseline_name
from paleolandmarker.data.synthetic.generate import (
    DOMAINS as SYNTHETIC_DOMAINS,
    GENERATORS,
    VERSION as SYNTHETIC_GENERATOR_VERSION,
    ensure_dataset as ensure_synthetic_dataset,
)
from paleolandmarker.data.real.split import ensure_dataset_split
from paleolandmarker.models import (
    FORMAL_MODEL_ARCHITECTURE,
    ORIGINAL_MODEL_ARCHITECTURE,
    build_model,
    count_parameters,
)
from paleolandmarker.training.engine import evaluate_model, get_device, save_json, train_model


DATASETS = [
    "conodont",
    "fusulinid",
    "land_snail_haplotrema_concavum",
    "microtus_dinolite_notilt_eo_t2",
    "ml_morph_bryozoan_zooids",
    "trilomorph_cephala",
]

MODEL_SIZES = ["tiny", "small", "medium", "large"]

ABLATION_DATASETS = list(DATASETS)

DEFAULT_ABLATION_SINGLE_DOMAINS = [
    ("conodont", "conodont_element"),
    ("fusulinid", "fusulinid_section"),
    ("land_snail_haplotrema_concavum", "gastropod_shell"),
    ("microtus_dinolite_notilt_eo_t2", "microtus_dentition"),
    ("ml_morph_bryozoan_zooids", "bryozoan_zooid"),
    ("trilomorph_cephala", "trilobite"),
]

ABLATION_ARCHITECTURES = {
    "baseline": ORIGINAL_MODEL_ARCHITECTURE,
    "context_resize": FORMAL_MODEL_ARCHITECTURE,
}
ABLATION_VARIANTS = {
    f"{architecture}_{loss}_{augmentation}_{initialization}": {
        "architecture": model_architecture,
        "loss": loss,
        "augmentation": augmentation == "aug",
        "initialization": initialization,
    }
    for architecture, model_architecture in ABLATION_ARCHITECTURES.items()
    for loss in ("heatmap_mse", "coord", "heatmap_coord")
    for augmentation in ("noaug", "aug")
    for initialization in (
        "scratch",
        "multidomain",
        "single_domain",
        "leave_one_out",
    )
}
DEFAULT_ABLATION_VARIANTS = [
    "baseline_coord_noaug_scratch",
    "baseline_coord_noaug_multidomain",
    "baseline_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_aug_multidomain",
    "context_resize_heatmap_coord_aug_leave_one_out",
]

SINGLE_DOMAIN_ALIASES = {
    "cephalon": "trilobite",
    "trilobite_cephalon": "trilobite",
}

LEAVE_ONE_OUT_KIND_PREFIX = "multidomain_without_"

COMPARISON_BASELINES = [
    "vandaele2018",
    "ml_morph",
    "landmarker_scratch",
    "landmarker_synthetic",
    "deeplabcut_scratch",
    "deeplabcut_imagenet",
    "deeplabcut_synthetic",
]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_HEATMAP_SIZE = 112
DEFAULT_HEATMAP_SIGMA = 8.0
DEFAULT_REAL_DATA_DIR = "data/real/processed_224"
DEFAULT_REAL_SPLITS_DIR = "data/real/splits_224"
DEFAULT_SYNTHETIC_PRETRAIN_DATA_DIR = "data/synthetic/multidomain"
DEFAULT_AUGMENTATION = True
SHOT_COUNTS = (20, 50, 100)
MIN_TRAIN_SIZE = max(SHOT_COUNTS)
PUBLICATION_METRICS = [
    "mean_error_px",
    "normalized_mean_error",
    "pck@3px",
    "pck@5px",
    "pck@10px",
]
DEFAULT_FINETUNE_EPOCHS = 400
DEFAULT_FINETUNE_BATCH_SIZE = 8
DEFAULT_FINETUNE_LEARNING_RATE = 0.001
DEFAULT_BASELINE_EPOCHS = DEFAULT_FINETUNE_EPOCHS
DEFAULT_PRETRAIN_EPOCHS = 50
DEFAULT_PRETRAIN_BATCH_SIZE = 16
DEEP_BASELINE_PIPELINE_VERSION = 4
MIN_OPEN_FILE_LIMIT = 65536


def ensure_open_file_limit(minimum=MIN_OPEN_FILE_LIMIT):
    """Raise RLIMIT_NOFILE for long DataLoader experiment matrices on Unix."""
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= minimum:
        return soft
    target = minimum if hard == resource.RLIM_INFINITY else min(minimum, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError) as exc:
        print(f"warning: could not raise open-file limit from {soft}: {exc}")
        return soft
    return target


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def configured_image_size():
    return int(os.environ.get("PALEOLANDMARKER_IMAGE_SIZE", DEFAULT_IMAGE_SIZE))


def configured_synthetic_data_dir():
    return os.environ.get(
        "PALEOLANDMARKER_SYNTHETIC_DATA_DIR",
        DEFAULT_SYNTHETIC_PRETRAIN_DATA_DIR,
    )


def single_domain_pretrain_data_dir(domain):
    seed_root = os.environ.get("PALEOLANDMARKER_SYNTHETIC_SEED_ROOT")
    if seed_root:
        return str(Path(seed_root) / domain)
    multidomain_dir = Path(configured_synthetic_data_dir())
    return str(
        multidomain_dir.with_name(
            f"image_size_{configured_image_size()}_{domain}"
        )
    )


def common_cfg(seed=42):
    return {
        "seed": seed,
        "device": os.environ.get("PALEOLANDMARKER_DEVICE", "auto"),
        "image_size": configured_image_size(),
        "heatmap_size": int(
            os.environ.get("PALEOLANDMARKER_HEATMAP_SIZE", DEFAULT_HEATMAP_SIZE)
        ),
        "heatmap_sigma": float(
            os.environ.get("PALEOLANDMARKER_HEATMAP_SIGMA", DEFAULT_HEATMAP_SIGMA)
        ),
        "batch_size": int(
            os.environ.get(
                "PALEOLANDMARKER_BATCH_SIZE",
                DEFAULT_FINETUNE_BATCH_SIZE,
            )
        ),
        "num_workers": int(os.environ.get("PALEOLANDMARKER_NUM_WORKERS", 0)),
        "learning_rate": float(
            os.environ.get(
                "PALEOLANDMARKER_LEARNING_RATE",
                DEFAULT_FINETUNE_LEARNING_RATE,
            )
        ),
        "weight_decay": 0.00001,
        "augmentation": env_bool("PALEOLANDMARKER_AUGMENTATION", DEFAULT_AUGMENTATION),
        "augmentation_mode": "bounded_affine",
        "pretrain_augmentation": env_bool(
            "PALEOLANDMARKER_PRETRAIN_AUGMENTATION", DEFAULT_AUGMENTATION
        ),
        "pck_thresholds": [3, 5, 10],
        "decode": {"method": "softargmax", "temperature": 40.0},
        "loss": {"name": "heatmap_coord", "coord_weight": 0.10, "softargmax_temperature": 40.0},
        "val_max_samples": 100,
        "test_max_samples": 100,
    }


def dataset_info(dataset, seed=42):
    processed_root = Path(
        os.environ.get("PALEOLANDMARKER_REAL_DATA_DIR", DEFAULT_REAL_DATA_DIR)
    )
    split_root = Path(
        os.environ.get("PALEOLANDMARKER_REAL_SPLITS_DIR", DEFAULT_REAL_SPLITS_DIR)
    )
    root = ensure_dataset_split(processed_root, split_root, dataset, seed)
    expected_size = configured_image_size()
    arrays = {}
    for split in ("train", "val", "test"):
        path = root / f"source_{split}.npz"
        with np.load(path, mmap_mode="r") as data:
            images = data["images"]
            if images.shape[-2:] != (expected_size, expected_size):
                raise ValueError(
                    f"expected {expected_size}x{expected_size} images in {path}, "
                    f"got {images.shape[-2:]}"
                )
            arrays[split] = {
                "size": len(images),
                "num_landmarks": int(data["keypoints"].shape[1]),
            }
    return {
        "dataset": dataset,
        "data_dir": str(root),
        "train_size": arrays["train"]["size"],
        "val_size": arrays["val"]["size"],
        "test_size": arrays["test"]["size"],
        "total_size": sum(split["size"] for split in arrays.values()),
        "num_landmarks": arrays["train"]["num_landmarks"],
        "data_split_seed": seed,
        "data_split_manifest": str(root / "split.json"),
    }


def shot_plan(train_size):
    if train_size < MIN_TRAIN_SIZE:
        return []
    return list(SHOT_COUNTS)


def shot_name(shot):
    return "allshots" if shot == "all" else f"{shot}shots"


def loss_cfg(name):
    if name == "heatmap_mse":
        return {"name": "heatmap_mse"}
    if name == "coord":
        return {"name": "coord", "softargmax_temperature": 40.0}
    if name == "heatmap_coord":
        return {"name": "heatmap_coord", "coord_weight": 0.10, "softargmax_temperature": 40.0}
    raise ValueError(name)


def parse_ablation_single_domain(value):
    dataset, separator, domain = value.partition("=")
    domain = SINGLE_DOMAIN_ALIASES.get(domain, domain)
    if not separator or not dataset or not domain:
        raise argparse.ArgumentTypeError("expected DATASET=DOMAIN")
    if domain not in GENERATORS:
        choices = ", ".join(sorted(GENERATORS))
        raise argparse.ArgumentTypeError(
            f"unknown synthetic domain {domain!r}; choose from {choices}"
        )
    return dataset, domain


def single_domain_pretrain_kind(domain):
    return domain


def leave_one_out_pretrain_kind(domain):
    domain = SINGLE_DOMAIN_ALIASES.get(domain, domain)
    if domain not in GENERATORS:
        raise ValueError(f"unknown excluded synthetic domain: {domain}")
    return f"{LEAVE_ONE_OUT_KIND_PREFIX}{domain}"


def leave_one_out_pretrain_data_dir(domain):
    domain = SINGLE_DOMAIN_ALIASES.get(domain, domain)
    kind = leave_one_out_pretrain_kind(domain)
    seed_root = os.environ.get("PALEOLANDMARKER_SYNTHETIC_SEED_ROOT")
    if seed_root:
        return str(Path(seed_root) / kind)
    multidomain_dir = Path(configured_synthetic_data_dir())
    return str(
        multidomain_dir.with_name(
            f"image_size_{configured_image_size()}_{kind}"
        )
    )


def synthetic_cfg(
    model,
    data_dir,
    output_dir,
    domains,
    loss_name,
    epochs,
    seed=42,
    augmentation=None,
    model_architecture=FORMAL_MODEL_ARCHITECTURE,
):
    cfg = common_cfg(seed)
    if augmentation is None:
        augmentation = env_bool(
            "PALEOLANDMARKER_PRETRAIN_AUGMENTATION", DEFAULT_AUGMENTATION
        )
    cfg.update(
        {
            "model": model,
            "model_architecture": model_architecture,
            "num_landmarks": 18,
            "data_dir": data_dir,
            "output_dir": str(output_dir),
            "source_domains": domains,
            "source_counts": {"train": 3000, "val": 500, "test": 500},
            "synthetic_generator": "latest",
            "synthetic_generator_version": SYNTHETIC_GENERATOR_VERSION,
            "epochs": epochs,
            "batch_size": DEFAULT_PRETRAIN_BATCH_SIZE,
            "learning_rate": 0.001,
            "augmentation": augmentation,
            "pretrain_augmentation": augmentation,
            "augmentation_mode": "bounded_affine",
            "loss": loss_cfg(loss_name),
            "val_max_samples": 100,
            "test_max_samples": 100,
            "run_name": f"pretrain_{model}_{loss_name}",
        }
    )
    return cfg


def ensure_synthetic_pretrain(
    model,
    kind,
    root,
    loss_name,
    epochs=DEFAULT_PRETRAIN_EPOCHS,
    seed=42,
    augmentation=None,
    model_architecture=FORMAL_MODEL_ARCHITECTURE,
):
    augmentation_is_ablation_factor = augmentation is not None
    if kind == "multidomain":
        data_dir = configured_synthetic_data_dir()
        domains = list(SYNTHETIC_DOMAINS)
    elif kind.startswith(LEAVE_ONE_OUT_KIND_PREFIX):
        excluded_domain = kind[len(LEAVE_ONE_OUT_KIND_PREFIX):]
        excluded_domain = SINGLE_DOMAIN_ALIASES.get(
            excluded_domain, excluded_domain
        )
        if excluded_domain not in GENERATORS:
            raise ValueError(
                f"unknown excluded synthetic domain: {excluded_domain}"
            )
        data_dir = leave_one_out_pretrain_data_dir(excluded_domain)
        domains = [d for d in SYNTHETIC_DOMAINS if d != excluded_domain]
    else:
        domain = SINGLE_DOMAIN_ALIASES.get(kind, kind)
        if domain not in GENERATORS:
            raise ValueError(f"unknown single pretraining domain: {domain}")
        data_dir = single_domain_pretrain_data_dir(domain)
        domains = [domain]
    if augmentation is None:
        augmentation = env_bool(
            "PALEOLANDMARKER_PRETRAIN_AUGMENTATION", DEFAULT_AUGMENTATION
        )
    out = root / "pretrain" / model / model_architecture / kind / loss_name
    # The enabled variant is the historical default, so keep its path reusable.
    # Only the new disabled variant needs a separate cache directory.
    if augmentation_is_ablation_factor and not augmentation:
        out /= "augmentation_off"
    ckpt = out / "checkpoints" / "best.pt"
    metrics_path = out / "metrics_test.json"
    cfg = synthetic_cfg(
        model,
        data_dir,
        out,
        domains,
        loss_name,
        epochs,
        seed,
        augmentation,
        model_architecture,
    )
    cfg.update(
        {
            "experiment": "synthetic_pretrain",
            "initialization": kind,
            "pretrain_kind": kind,
            "pretrain_epochs": epochs,
            "pretrain_loss_name": loss_name,
            "model_architecture": model_architecture,
        }
    )
    settings = experiment_settings(cfg)
    if ckpt.exists() and metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("config_json") == settings["config_json"]:
            print(f"reuse pretrain: {ckpt}")
            return ckpt
        print(f"rerun pretrain because config changed: {metrics_path}")
    ensure_synthetic_dataset(cfg)
    model_obj, history, summary = train_model(cfg, ckpt)
    metrics = evaluate_model(model_obj, cfg, "test")
    save_json(out / "history.json", history)
    save_json(out / "metrics_test.json", {**settings, **summary, **metrics})
    return ckpt


def real_cfg(
    info,
    model,
    output_dir,
    shot=100,
    init_checkpoint=None,
    epochs=DEFAULT_FINETUNE_EPOCHS,
    seed=42,
):
    cfg = common_cfg(seed)
    cfg.update(
        {
            "model": model,
            "model_architecture": FORMAL_MODEL_ARCHITECTURE,
            "num_landmarks": info["num_landmarks"],
            "data_dir": info["data_dir"],
            "output_dir": str(output_dir),
            "epochs": epochs,
            "shot_count": shot,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
            "run_name": f"{model}_{info['dataset']}_{shot_name(shot)}",
            "data_split_seed": info.get("data_split_seed", seed),
            "data_split_manifest": info.get("data_split_manifest", ""),
        }
    )
    return cfg


def experiment_settings(cfg):
    loss = cfg.get("loss", {})
    decode = cfg.get("decode", {})
    loss_variants = {
        "heatmap_mse": "heatmap",
        "coord": "coord",
        "heatmap_coord": "heatmap+coord",
    }
    return {
        "experiment": cfg.get("experiment", ""),
        "ablation_variant": cfg.get("ablation_variant", ""),
        "model_architecture": cfg.get("model_architecture", ""),
        "initialization": cfg.get("initialization", ""),
        "pretrain_kind": cfg.get("pretrain_kind", ""),
        "pretrain_excluded_domain": cfg.get("pretrain_excluded_domain", ""),
        "pretrain_source_domains": cfg.get("pretrain_source_domains", ""),
        "pretrain_epochs": cfg.get("pretrain_epochs", cfg.get("synthetic_pretrain_epochs", "")),
        "pretrain_loss_name": cfg.get("pretrain_loss_name", ""),
        "seed": cfg.get("seed", ""),
        "data_split_seed": cfg.get("data_split_seed", ""),
        "data_split_manifest": cfg.get("data_split_manifest", ""),
        "epochs": cfg.get("epochs", cfg.get("baseline_epochs", "")),
        "batch_size": cfg.get("batch_size", ""),
        "num_workers": cfg.get("num_workers", ""),
        "learning_rate": cfg.get("learning_rate", cfg.get("baseline_learning_rate", "")),
        "weight_decay": cfg.get("weight_decay", ""),
        "augmentation": cfg.get("augmentation", ""),
        "augmentation_mode": cfg.get("augmentation_mode", ""),
        "pretrain_augmentation": cfg.get("pretrain_augmentation", ""),
        "image_size": cfg.get("image_size", ""),
        "heatmap_size": cfg.get("heatmap_size", ""),
        "heatmap_sigma": cfg.get("heatmap_sigma", ""),
        "loss_name": loss.get("name", ""),
        "loss_variant": loss_variants.get(loss.get("name", ""), loss.get("name", "")),
        "coord_weight": loss.get("coord_weight", ""),
        "loss_temperature": loss.get("softargmax_temperature", ""),
        "decode_method": decode.get("method", ""),
        "decode_temperature": decode.get("temperature", ""),
        "val_max_samples": cfg.get("val_max_samples", ""),
        "test_max_samples": cfg.get("test_max_samples", ""),
        "requested_device": cfg.get("device", os.environ.get("PALEOLANDMARKER_DEVICE", "auto")),
        "config_json": json.dumps(cfg, sort_keys=True, default=str),
    }


def run_ours(
    info,
    model,
    shot,
    init_checkpoint,
    output_dir,
    epochs=DEFAULT_FINETUNE_EPOCHS,
    cfg_updates=None,
    seed=42,
):
    cfg = real_cfg(info, model, output_dir, shot, init_checkpoint, epochs, seed)
    if cfg_updates:
        cfg.update(cfg_updates)
    settings = experiment_settings(cfg)
    metrics_path = output_dir / "metrics_test.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("config_json") == settings["config_json"]:
            print(f"skip completed: {metrics_path}")
            return previous
        print(f"rerun because config changed: {metrics_path}")
    ckpt = output_dir / "checkpoints" / "best.pt"
    model_obj, history, summary = train_model(cfg, ckpt)
    metrics = evaluate_model(model_obj, cfg, "test", output_dir / "visualizations" / "test", 8)
    row = {
        "dataset": info["dataset"],
        "method": cfg.get("method_label", model),
        "model": model,
        "shots": shot,
        "train_size": info["train_size"],
        "val_size": info["val_size"],
        "test_size": info["test_size"],
        "total_size": info["total_size"],
        "num_landmarks": info["num_landmarks"],
        "checkpoint": str(ckpt),
        "parameters": count_parameters(model_obj),
        **settings,
        **summary,
        **metrics,
    }
    save_json(output_dir / "history.json", history)
    save_json(metrics_path, row)
    return row


def baseline_cfg(
    info,
    output_dir,
    shot,
    seed=42,
    epochs=DEFAULT_BASELINE_EPOCHS,
    pretrain_epochs=DEFAULT_PRETRAIN_EPOCHS,
):
    cfg = common_cfg(seed)
    finetune_batch_size = cfg["batch_size"]
    finetune_learning_rate = cfg["learning_rate"]
    cfg.update(
        {
            "model": "medium",
            "num_landmarks": info["num_landmarks"],
            "data_dir": info["data_dir"],
            "output_dir": str(output_dir),
            "shot_count": shot,
            "baseline_epochs": epochs,
            "batch_size": finetune_batch_size,
            "learning_rate": finetune_learning_rate,
            "baseline_learning_rate": finetune_learning_rate,
            "baseline_checkpoint": str(output_dir / "checkpoints"),
            "synthetic_pretrain_checkpoint_dir": str(output_dir.parents[2] / "_shared_baseline_pretrain"),
            "n_estimators": 300,
            "feature_size": 32,
            "n_jobs": -1,
            "dlc_backbone": "mobilenet_v3_small",
            "dlc_pretrained": False,
            "dlc_width_mult": 1.0,
            "landmarker_channels": [11, 22, 44, 86],
            "landmarker_stacks": 2,
            "synthetic_pretrain": False,
            "synthetic_pretrain_generate": True,
            "synthetic_pretrain_data_dir": configured_synthetic_data_dir(),
            "synthetic_pretrain_num_landmarks": 18,
            "synthetic_pretrain_epochs": pretrain_epochs,
            "synthetic_pretrain_generator": "latest",
            "synthetic_pretrain_generator_version": SYNTHETIC_GENERATOR_VERSION,
            "synthetic_pretrain_batch_size": DEFAULT_PRETRAIN_BATCH_SIZE,
            "synthetic_pretrain_learning_rate": 0.001,
            "synthetic_pretrain_augmentation": env_bool(
                "PALEOLANDMARKER_PRETRAIN_AUGMENTATION", DEFAULT_AUGMENTATION
            ),
            "synthetic_pretrain_loss": loss_cfg("heatmap_coord"),
            "synthetic_pretrain_source_domains": list(SYNTHETIC_DOMAINS),
            "synthetic_pretrain_source_counts": {"train": 3000, "val": 500, "test": 500},
            "data_split_seed": info.get("data_split_seed", seed),
            "data_split_manifest": info.get("data_split_manifest", ""),
            "mlmorph_feature_size": 48,
            "mlmorph_n_estimators": 250,
            "mlmorph_max_depth": 18,
            "mlmorph_min_samples_leaf": 2,
        }
    )
    return cfg


def configure_baseline_variant(cfg, baseline_name, include_pipeline_version=True):
    baseline_name = canonical_baseline_name(baseline_name)
    initialization = "classical"
    if baseline_name == "landmarker_scratch":
        initialization = "scratch"
        cfg["synthetic_pretrain"] = False
    elif baseline_name == "landmarker_synthetic":
        initialization = "synthetic_multidomain"
        cfg["synthetic_pretrain"] = True
    elif baseline_name == "deeplabcut_scratch":
        initialization = "scratch"
        cfg.update({"dlc_pretrained": False, "dlc_width_mult": 1.0, "synthetic_pretrain": False})
    elif baseline_name == "deeplabcut_imagenet":
        initialization = "imagenet"
        cfg.update({"dlc_pretrained": True, "dlc_width_mult": 1.0, "synthetic_pretrain": False})
    elif baseline_name == "deeplabcut_synthetic":
        initialization = "synthetic_multidomain"
        cfg.update({"dlc_pretrained": False, "dlc_width_mult": 1.0, "synthetic_pretrain": True})
    if include_pipeline_version and baseline_name.startswith(
        ("deeplabcut_", "landmarker_")
    ):
        # Version 4 uses one MobileNetV3-Small 1.0x architecture and identical
        # normalization for all DLC initializations, and replaces the old wide
        # decoder with a parameter-matched 128/96/64/64 decoder. It also ports
        # Landmarker v0.2.1's multi-level stacked hourglass. Keeping this in
        # config_json prevents stale deep-baseline results from being reused.
        cfg["deep_baseline_pipeline_version"] = DEEP_BASELINE_PIPELINE_VERSION
    cfg.update(
        {
            "experiment": "comparison",
            "initialization": initialization,
            "pretrain_kind": (
                "multidomain"
                if initialization == "synthetic_multidomain"
                else initialization if initialization == "imagenet" else "none"
            ),
            "pretrain_epochs": (
                cfg.get("synthetic_pretrain_epochs", 0)
                if initialization == "synthetic_multidomain"
                else 0
            ),
            "pretrain_loss_name": (
                cfg.get("synthetic_pretrain_loss", {}).get("name", "")
                if initialization == "synthetic_multidomain"
                else ""
            ),
        }
    )
    return cfg


def baseline_cache_identity(config_json, baseline_name):
    """Return the training-relevant baseline config, ignoring naming-only fields."""
    try:
        cfg = json.loads(config_json)
    except (TypeError, json.JSONDecodeError):
        return None
    # Preserve whether the saved config actually contained a loader protocol.
    # Otherwise an old result would be upgraded in memory and incorrectly
    # treated as cache-compatible with a version-2 deep baseline.
    cfg = configure_baseline_variant(
        cfg, baseline_name, include_pipeline_version=False
    )
    for key in ("output_dir", "baseline_checkpoint", "initialization", "pretrain_kind", "pretrain_epochs"):
        cfg.pop(key, None)
    return cfg


def run_baseline(
    info,
    baseline_name,
    shot,
    output_dir,
    seed=42,
    epochs=DEFAULT_BASELINE_EPOCHS,
    pretrain_epochs=DEFAULT_PRETRAIN_EPOCHS,
):
    baseline_name = canonical_baseline_name(baseline_name)
    cfg = configure_baseline_variant(
        baseline_cfg(info, output_dir, shot, seed, epochs, pretrain_epochs), baseline_name
    )
    settings = experiment_settings(cfg)
    metrics_path = output_dir / "metrics_test.json"
    if metrics_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        exact_match = previous.get("config_json") == settings["config_json"]
        compatible_rename = baseline_cache_identity(
            previous.get("config_json"), baseline_name
        ) == baseline_cache_identity(settings["config_json"], baseline_name)
        if exact_match or compatible_rename:
            previous.update(
                {
                    "method": baseline_name,
                    "initialization": cfg["initialization"],
                    **settings,
                }
            )
            save_json(metrics_path, previous)
            print(f"skip completed: {metrics_path}")
            return previous
        print(f"rerun because config changed: {metrics_path}")
    baseline = CANONICAL_BASELINE_REGISTRY[baseline_name](cfg).fit()
    metrics = baseline.evaluate("test", output_dir / "visualizations" / "test", 8)
    row = {
        "experiment": "comparison",
        "initialization": cfg["initialization"],
        "dataset": info["dataset"],
        "method": baseline_name,
        "model": "",
        "shots": shot,
        "train_size": info["train_size"],
        "val_size": info["val_size"],
        "test_size": info["test_size"],
        "total_size": info["total_size"],
        "num_landmarks": info["num_landmarks"],
        "checkpoint": "",
        **settings,
        **metrics,
    }
    save_json(metrics_path, row)
    return row


def load_model_sweep_medium(output_dir, dataset, shot):
    metrics_path = Path(output_dir) / "model_sweep" / dataset / "medium" / shot_name(shot) / "metrics_test.json"
    if not metrics_path.exists():
        return None
    row = json.loads(metrics_path.read_text(encoding="utf-8"))
    row["method"] = "medium"
    row["model"] = "medium"
    row["shots"] = shot
    return row


def append_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "experiment",
        "ablation_variant",
        "model_architecture",
        "initialization",
        "pretrain_kind",
        "pretrain_excluded_domain",
        "pretrain_source_domains",
        "pretrain_epochs",
        "dataset",
        "method",
        "model",
        "shots",
        "train_size",
        "val_size",
        "test_size",
        "total_size",
        "num_landmarks",
        "seed",
        "data_split_seed",
        "data_split_manifest",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "augmentation",
        "augmentation_mode",
        "pretrain_augmentation",
        "image_size",
        "heatmap_size",
        "heatmap_sigma",
        "loss_name",
        "loss_variant",
        "coord_weight",
        "loss_temperature",
        "decode_method",
        "decode_temperature",
        "requested_device",
        "device",
        "parameters",
        "best_epoch",
        "best_val_error_px",
        "mean_error_px",
        "normalized_mean_error",
        "pck@3px",
        "pck@5px",
        "pck@10px",
        "checkpoint",
        "init_checkpoint",
        "config_json",
    ]
    extra_fields = sorted(set().union(*(row.keys() for row in rows)) - set(preferred_fields))
    fields = preferred_fields + extra_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                    for key, value in ((key, row.get(key, "")) for key in fields)
                }
            )


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordered_unique(values):
    return list(dict.fromkeys(values))


def write_ablation_publication_csvs(root, rows):
    """Write one compact, metric-rich ablation table per dataset."""
    fields = [
        "ablation_variant",
        "model_architecture",
        "initialization",
        "pretrain_excluded_domain",
        "loss",
        "augmentation",
    ] + PUBLICATION_METRICS
    for dataset in ordered_unique(row["dataset"] for row in rows):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        publication_rows = [
            {
                "ablation_variant": row["ablation_variant"],
                "model_architecture": row["model_architecture"],
                "initialization": row["initialization"],
                "pretrain_excluded_domain": row.get("pretrain_excluded_domain", ""),
                "loss": row["loss_variant"],
                "augmentation": row["augmentation"],
                **{metric: row.get(metric, "") for metric in PUBLICATION_METRICS},
            }
            for row in dataset_rows
        ]
        write_csv(
            root / f"ablation_publication_{dataset}.csv",
            fields,
            publication_rows,
        )


def write_nme_publication_matrix(path, rows, row_field):
    """Pivot experiment rows into a paper-ready dataset/shot NME matrix."""
    if not rows:
        return
    dataset_shots = ordered_unique((row["dataset"], row["shots"]) for row in rows)
    column_for = {
        (dataset, shot): f"{dataset}_{shot_name(shot)}_nme"
        for dataset, shot in dataset_shots
    }
    row_values = ordered_unique(row[row_field] for row in rows)
    indexed = {
        (row[row_field], row["dataset"], row["shots"]): row.get("normalized_mean_error", "")
        for row in rows
    }
    publication_rows = [
        {
            row_field: row_value,
            **{
                column_for[(dataset, shot)]: indexed.get(
                    (row_value, dataset, shot), ""
                )
                for dataset, shot in dataset_shots
            },
        }
        for row_value in row_values
    ]
    write_csv(path, [row_field, *column_for.values()], publication_rows)


def run_medium_ablation(args):
    root = Path(args.output_dir) / "medium_ablation"
    pretrain_root = Path(args.output_dir) / "_pretrain_cache"
    pretrains = {}

    def get_pretrain(kind, loss_name, augmentation, model_architecture):
        key = (kind, loss_name, augmentation, model_architecture)
        if key not in pretrains:
            pretrains[key] = ensure_synthetic_pretrain(
                "medium",
                kind,
                pretrain_root,
                loss_name,
                args.pretrain_epochs,
                args.seed,
                augmentation,
                model_architecture,
            )
        return pretrains[key]

    rows = []
    for dataset in args.ablation_datasets:
        info = dataset_info(dataset, args.seed)
        if info["train_size"] < MIN_TRAIN_SIZE:
            print(
                f"skip medium ablation dataset={dataset}: train_size={info['train_size']} "
                f"< {MIN_TRAIN_SIZE}"
            )
            continue
        shot = 100
        dataset_root = root / dataset
        single_domain = args.ablation_single_domains.get(dataset)
        single_kind = (
            single_domain_pretrain_kind(single_domain)
            if single_domain is not None
            else None
        )
        for variant_name in args.ablation_variants:
            variant = ABLATION_VARIANTS[variant_name]
            loss_name = variant["loss"]
            augmentation = variant["augmentation"]
            model_architecture = variant["architecture"]
            initialization = variant["initialization"]
            if initialization == "scratch":
                label = "scratch"
                pretrain_kind = "none"
                init = None
                excluded_domain = None
                pretrain_source_domains = []
            elif initialization == "multidomain":
                label = "synthetic_multidomain"
                pretrain_kind = "multidomain"
                excluded_domain = None
                pretrain_source_domains = list(SYNTHETIC_DOMAINS)
                init = get_pretrain(
                    pretrain_kind,
                    loss_name,
                    augmentation,
                    model_architecture,
                )
            elif initialization == "single_domain":
                if single_kind is None:
                    raise ValueError(
                        f"single-domain ablation requested without a mapping for {dataset}"
                    )
                label = f"synthetic_{single_kind}"
                pretrain_kind = single_kind
                excluded_domain = None
                pretrain_source_domains = [single_kind]
                init = get_pretrain(
                    pretrain_kind,
                    loss_name,
                    augmentation,
                    model_architecture,
                )
            else:
                if single_domain is None:
                    raise ValueError(
                        f"leave-one-out ablation requested without a mapping for {dataset}"
                    )
                excluded_domain = SINGLE_DOMAIN_ALIASES.get(
                    single_domain, single_domain
                )
                pretrain_kind = leave_one_out_pretrain_kind(excluded_domain)
                label = f"synthetic_{pretrain_kind}"
                pretrain_source_domains = [
                    domain
                    for domain in SYNTHETIC_DOMAINS
                    if domain != excluded_domain
                ]
                init = get_pretrain(
                    pretrain_kind,
                    loss_name,
                    augmentation,
                    model_architecture,
                )
            run_settings = {
                "experiment": "medium_ablation",
                "ablation_variant": variant_name,
                "model_architecture": model_architecture,
                "initialization": label,
                "pretrain_kind": pretrain_kind,
                "pretrain_excluded_domain": excluded_domain or "",
                "pretrain_source_domains": pretrain_source_domains,
                "pretrain_epochs": args.pretrain_epochs if init else 0,
                "pretrain_loss_name": loss_name if init else "",
                "augmentation": augmentation,
                "pretrain_augmentation": augmentation if init else "",
                "loss": loss_cfg(loss_name),
            }
            row = run_ours(
                info,
                "medium",
                shot,
                init,
                dataset_root / variant_name,
                args.epochs,
                run_settings,
                args.seed,
            )
            row["min_train_size"] = MIN_TRAIN_SIZE
            rows.append(row)
    append_csv(root / "ablation_results.csv", rows)
    write_ablation_publication_csvs(root, rows)


def run_model_sweep(args):
    root = Path(args.output_dir) / "model_sweep"
    rows = []
    eligible = []
    for dataset in args.datasets:
        info = dataset_info(dataset, args.seed)
        if info["train_size"] < MIN_TRAIN_SIZE:
            print(
                f"skip model sweep dataset={dataset}: train_size={info['train_size']} "
                f"< {MIN_TRAIN_SIZE}"
            )
            continue
        eligible.append(info)
    if not eligible:
        print("no model sweep datasets meet the minimum train size")
        return

    pretrain_root = Path(args.output_dir) / "_pretrain_cache"
    pretrains = {
        model: ensure_synthetic_pretrain(
            model,
            "multidomain",
            pretrain_root,
            "heatmap_coord",
            args.pretrain_epochs,
            args.seed,
        )
        for model in args.models
    }
    for info in eligible:
        for model in args.models:
            for shot in shot_plan(info["train_size"]):
                out = root / info["dataset"] / model / shot_name(shot)
                row = run_ours(
                    info,
                    model,
                    shot,
                    pretrains[model],
                    out,
                    args.epochs,
                    {
                        "experiment": "model_sweep",
                        "initialization": "synthetic_multidomain",
                        "pretrain_kind": "multidomain",
                        "pretrain_epochs": args.pretrain_epochs,
                        "pretrain_loss_name": "heatmap_coord",
                    },
                    args.seed,
                )
                row["min_train_size"] = MIN_TRAIN_SIZE
                rows.append(row)
                append_csv(root / "model_sweep_results.csv", rows)
                write_nme_publication_matrix(
                    root / "model_sweep_publication.csv", rows, "model"
                )
    append_csv(root / "model_sweep_results.csv", rows)
    write_nme_publication_matrix(
        root / "model_sweep_publication.csv", rows, "model"
    )


def run_comparison(args):
    root = Path(args.output_dir) / "comparison"
    rows = []
    eligible = []
    for dataset in args.datasets:
        info = dataset_info(dataset, args.seed)
        if info["train_size"] < MIN_TRAIN_SIZE:
            print(
                f"skip comparison dataset={dataset}: train_size={info['train_size']} "
                f"< {MIN_TRAIN_SIZE}"
            )
            continue
        eligible.append(info)
    if not eligible:
        print("no comparison datasets meet the minimum train size")
        return

    pretrain_root = Path(args.output_dir) / "_pretrain_cache"
    medium_pretrain = ensure_synthetic_pretrain(
        "medium",
        "multidomain",
        pretrain_root,
        "heatmap_coord",
        args.pretrain_epochs,
        args.seed,
    )
    for info in eligible:
        for shot in shot_plan(info["train_size"]):
            synthetic_row = run_ours(
                info,
                "medium",
                shot,
                medium_pretrain,
                root / info["dataset"] / "medium_synthetic" / shot_name(shot),
                args.epochs,
                {
                    "experiment": "comparison",
                    "method_label": "medium_synthetic",
                    "initialization": "synthetic_multidomain",
                    "pretrain_kind": "multidomain",
                    "pretrain_epochs": args.pretrain_epochs,
                    "pretrain_loss_name": "heatmap_coord",
                },
                args.seed,
            )
            synthetic_row["min_train_size"] = MIN_TRAIN_SIZE
            rows.append(synthetic_row)

            scratch_row = run_ours(
                info,
                "medium",
                shot,
                None,
                root / info["dataset"] / "medium_scratch" / shot_name(shot),
                args.epochs,
                {
                    "experiment": "comparison",
                    "method_label": "medium_scratch",
                    "initialization": "scratch",
                    "pretrain_kind": "none",
                    "pretrain_epochs": 0,
                },
                args.seed,
            )
            scratch_row["min_train_size"] = MIN_TRAIN_SIZE
            rows.append(scratch_row)

            for requested_baseline_name in args.baselines:
                baseline_name = canonical_baseline_name(requested_baseline_name)
                baseline_row = run_baseline(
                    info,
                    baseline_name,
                    shot,
                    root / info["dataset"] / baseline_name / shot_name(shot),
                    args.seed,
                    args.baseline_epochs,
                    args.pretrain_epochs,
                )
                baseline_row["min_train_size"] = MIN_TRAIN_SIZE
                rows.append(baseline_row)
            append_csv(root / "comparison_results.csv", rows)
            write_nme_publication_matrix(
                root / "comparison_publication.csv", rows, "method"
            )
    append_csv(root / "comparison_results.csv", rows)
    write_nme_publication_matrix(
        root / "comparison_publication.csv", rows, "method"
    )

def main():
    open_file_limit = ensure_open_file_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["medium_ablation", "model_sweep", "comparison", "all"], default="all")
    parser.add_argument("--output-dir", default="artifacts/reproduction/full_experiments_224")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--ablation-datasets", nargs="+", default=ABLATION_DATASETS)
    parser.add_argument(
        "--ablation-variants",
        nargs="+",
        choices=sorted(ABLATION_VARIANTS),
        default=DEFAULT_ABLATION_VARIANTS,
        help=(
            "complete architecture/loss/augmentation/initialization configurations; "
            "defaults to six cumulative experiments, including leave-one-domain-out pretraining"
        ),
    )
    parser.add_argument(
        "--ablation-single-domains",
        nargs="+",
        type=parse_ablation_single_domain,
        metavar="DATASET=DOMAIN",
        default=DEFAULT_ABLATION_SINGLE_DOMAINS,
    )
    parser.add_argument("--models", nargs="+", default=MODEL_SIZES)
    parser.add_argument(
        "--baselines", nargs="+", type=canonical_baseline_name,
        choices=sorted(CANONICAL_BASELINE_REGISTRY), default=COMPARISON_BASELINES,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_FINETUNE_EPOCHS,
        help="epochs for real-data training of this study's models",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=DEFAULT_BASELINE_EPOCHS,
        help="epochs for prior deep-learning baselines",
    )
    parser.add_argument(
        "--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_FINETUNE_BATCH_SIZE,
        help="batch size for real-data training of this study's models",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_FINETUNE_LEARNING_RATE,
        help="learning rate for real-data training of this study's models",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--heatmap-size", type=int, default=DEFAULT_HEATMAP_SIZE)
    parser.add_argument("--heatmap-sigma", type=float, default=DEFAULT_HEATMAP_SIGMA)
    parser.add_argument(
        "--real-data-dir",
        default=DEFAULT_REAL_DATA_DIR,
        help="standardized real datasets containing one dataset.npz per dataset",
    )
    parser.add_argument("--real-splits-dir", default=DEFAULT_REAL_SPLITS_DIR)
    parser.add_argument(
        "--synthetic-data-dir", default=DEFAULT_SYNTHETIC_PRETRAIN_DATA_DIR
    )
    parser.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUGMENTATION,
        help="use the current boundary-safe affine and photometric augmentation",
    )
    parser.add_argument(
        "--pretrain-augmentation",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_AUGMENTATION,
        help="use the same augmentation during synthetic pretraining",
    )
    args = parser.parse_args()

    if len(args.ablation_variants) != len(set(args.ablation_variants)):
        parser.error("--ablation-variants cannot contain duplicates")

    single_domain_datasets = [dataset for dataset, _ in args.ablation_single_domains]
    if len(single_domain_datasets) != len(set(single_domain_datasets)):
        parser.error("--ablation-single-domains contains duplicate datasets")
    args.ablation_single_domains = dict(args.ablation_single_domains)
    missing_single_domains = [
        dataset for dataset in args.ablation_datasets
        if dataset not in args.ablation_single_domains
    ]
    uses_dataset_domain_mapping = any(
        ABLATION_VARIANTS[variant]["initialization"] in {"single_domain", "leave_one_out"}
        for variant in args.ablation_variants
    )
    if uses_dataset_domain_mapping and missing_single_domains:
        parser.error(
            "--ablation-single-domains is missing mappings for: "
            + ", ".join(missing_single_domains)
        )

    os.environ["PALEOLANDMARKER_DEVICE"] = args.device
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.baseline_epochs < 1:
        parser.error("--baseline-epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.image_size < 1 or args.heatmap_size < 1 or args.heatmap_sigma <= 0:
        parser.error("image size, heatmap size, and heatmap sigma must be positive")
    if args.image_size != 2 * args.heatmap_size:
        parser.error("--heatmap-size must be half of --image-size for all compared models")

    os.environ["PALEOLANDMARKER_BATCH_SIZE"] = str(args.batch_size)
    os.environ["PALEOLANDMARKER_LEARNING_RATE"] = str(args.learning_rate)
    os.environ["PALEOLANDMARKER_NUM_WORKERS"] = str(args.num_workers)
    os.environ["PALEOLANDMARKER_IMAGE_SIZE"] = str(args.image_size)
    os.environ["PALEOLANDMARKER_HEATMAP_SIZE"] = str(args.heatmap_size)
    os.environ["PALEOLANDMARKER_HEATMAP_SIGMA"] = str(args.heatmap_sigma)
    os.environ["PALEOLANDMARKER_REAL_DATA_DIR"] = args.real_data_dir
    os.environ["PALEOLANDMARKER_REAL_SPLITS_DIR"] = args.real_splits_dir
    synthetic_seed_root = Path(args.synthetic_data_dir) / f"seed_{args.seed}"
    os.environ["PALEOLANDMARKER_SYNTHETIC_SEED_ROOT"] = str(synthetic_seed_root)
    os.environ["PALEOLANDMARKER_SYNTHETIC_DATA_DIR"] = str(
        synthetic_seed_root / "multidomain"
    )
    os.environ["PALEOLANDMARKER_AUGMENTATION"] = str(args.augmentation)
    os.environ["PALEOLANDMARKER_PRETRAIN_AUGMENTATION"] = str(
        args.pretrain_augmentation
    )
    device = get_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    print(
        f"device={device} ({device_name}) "
        f"batch_size={args.batch_size} learning_rate={args.learning_rate} "
        f"num_workers={args.num_workers} image_size={args.image_size} "
        f"heatmap_size={args.heatmap_size} heatmap_sigma={args.heatmap_sigma} "
        f"augmentation={args.augmentation} "
        f"pretrain_augmentation={args.pretrain_augmentation} "
        f"open_file_limit={open_file_limit}"
    )

    if args.task in ("medium_ablation", "all"):
        run_medium_ablation(args)
    if args.task in ("model_sweep", "all"):
        run_model_sweep(args)
    if args.task in ("comparison", "all"):
        run_comparison(args)


if __name__ == "__main__":
    main()
