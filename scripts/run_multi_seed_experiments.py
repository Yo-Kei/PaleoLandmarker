import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DATASETS = [
    "conodont",
    "fusulinid",
    "land_snail_haplotrema_concavum",
    "microtus_dinolite_notilt_eo_t2",
    "ml_morph_bryozoan_zooids",
    "trilomorph_cephala",
]

ABLATION_DATASETS = list(DATASETS)

DEFAULT_ABLATION_SINGLE_DOMAINS = [
    "conodont=conodont_element",
    "fusulinid=fusulinid_section",
    "land_snail_haplotrema_concavum=gastropod_shell",
    "microtus_dinolite_notilt_eo_t2=microtus_dentition",
    "ml_morph_bryozoan_zooids=bryozoan_zooid",
    "trilomorph_cephala=trilobite",
]

DEFAULT_ABLATION_VARIANTS = [
    "baseline_coord_noaug_scratch",
    "baseline_coord_noaug_multidomain",
    "baseline_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_noaug_multidomain",
    "context_resize_heatmap_coord_aug_multidomain",
    "context_resize_heatmap_coord_aug_leave_one_out",
]
ABLATION_VARIANTS = [
    f"{architecture}_{loss}_{augmentation}_{initialization}"
    for architecture in ("baseline", "context_resize")
    for loss in ("heatmap_mse", "coord", "heatmap_coord")
    for augmentation in ("noaug", "aug")
    for initialization in (
        "scratch",
        "multidomain",
        "single_domain",
        "leave_one_out",
    )
]
MODEL_SIZES = ["tiny", "small", "medium", "large"]
COMPARISON_BASELINES = [
    "vandaele2018",
    "ml_morph",
    "landmarker_scratch",
    "landmarker_synthetic",
    "deeplabcut_scratch",
    "deeplabcut_imagenet",
    "deeplabcut_synthetic",
]
DEFAULT_PRETRAIN_EPOCHS = 50
DEFAULT_SYNTHETIC_PRETRAIN_DATA_DIR = (
    "data/synthetic/multidomain"
)
MIN_OPEN_FILE_LIMIT = 65536


def ensure_open_file_limit(minimum=MIN_OPEN_FILE_LIMIT):
    """Raise RLIMIT_NOFILE before spawning long-running DataLoader jobs."""
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

EXPERIMENT_ARGUMENTS = (
    "seeds",
    "task",
    "datasets",
    "ablation_datasets",
    "ablation_variants",
    "ablation_single_domains",
    "models",
    "baselines",
    "epochs",
    "baseline_epochs",
    "pretrain_epochs",
    "device",
    "batch_size",
    "learning_rate",
    "num_workers",
    "image_size",
    "heatmap_size",
    "heatmap_sigma",
    "real_data_dir",
    "real_splits_dir",
    "synthetic_data_dir",
    "augmentation",
    "pretrain_augmentation",
)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def experiment_arguments(args):
    source = vars(args) if isinstance(args, argparse.Namespace) else args
    return {key: source.get(key) for key in EXPERIMENT_ARGUMENTS}


def find_resumable_run(output_root, requested_command):
    if not output_root.exists():
        return None
    candidates = []
    for command_path in output_root.glob("*/command.json"):
        saved_command = read_json(command_path)
        if not isinstance(saved_command, dict):
            continue
        if saved_command.get("run_name"):
            continue
        if experiment_arguments(saved_command) == requested_command:
            candidates.append(command_path.parent)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def unused_timestamped_run_root(output_root, seeds, started_at):
    stem = f"seeds_{'_'.join(str(seed) for seed in seeds)}_{started_at}"
    candidate = output_root / stem
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def completed_job(status, seed):
    if not isinstance(status, dict):
        return None
    for job in reversed(status.get("jobs", [])):
        if job.get("seed") == seed and job.get("state") == "completed":
            return job
    return None


def completed_seed_count(status, seeds):
    return sum(
        any(
            job.get("seed") == seed and job.get("state") == "completed"
            for job in status.get("jobs", [])
        )
        for seed in seeds
    )


def process_is_running(pid, expected_tokens=()):
    """Return true only when the PID is alive and belongs to this run.

    PIDs can be reused after moving a persistent run directory to a new
    instance. Checking the command line prevents an unrelated process with the
    same numeric PID from blocking a safe resume.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    if expected_tokens:
        try:
            command = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            return False
        return all(str(token) in command for token in expected_tokens)
    return True


def accelerator_status():
    try:
        import torch

        status = {
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
        if status["cuda_available"]:
            status["cuda_devices"] = [
                torch.cuda.get_device_name(index) for index in range(status["cuda_device_count"])
            ]
        return status
    except Exception as exc:
        return {"error": repr(exc), "cuda_available": False, "mps_available": False}


def main():
    open_file_limit = ensure_open_file_limit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--task", choices=["medium_ablation", "model_sweep", "comparison", "all"], default="all")
    parser.add_argument("--output-root", default="artifacts/reproduction/multiseed")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--ablation-datasets", nargs="+", default=ABLATION_DATASETS)
    parser.add_argument(
        "--ablation-variants",
        nargs="+",
        choices=ABLATION_VARIANTS,
        default=DEFAULT_ABLATION_VARIANTS,
        help=(
            "complete architecture/loss/augmentation/initialization configurations "
            "forwarded to the full experiment runner"
        ),
    )
    parser.add_argument(
        "--ablation-single-domains",
        nargs="+",
        metavar="DATASET=DOMAIN",
        default=DEFAULT_ABLATION_SINGLE_DOMAINS,
    )
    parser.add_argument("--models", nargs="+", default=MODEL_SIZES)
    parser.add_argument("--baselines", nargs="+", default=COMPARISON_BASELINES)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--baseline-epochs", type=int, default=400)
    parser.add_argument(
        "--pretrain-epochs", type=int, default=DEFAULT_PRETRAIN_EPOCHS
    )
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--heatmap-size", type=int, default=112)
    parser.add_argument("--heatmap-sigma", type=float, default=8.0)
    parser.add_argument("--real-data-dir", default="data/real/processed_224")
    parser.add_argument("--real-splits-dir", default="data/real/splits_224")
    parser.add_argument(
        "--synthetic-data-dir", default=DEFAULT_SYNTHETIC_PRETRAIN_DATA_DIR
    )
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--pretrain-augmentation", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume a matching run (default); use --no-resume to force a new run",
    )
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds cannot contain duplicates")
    if len(args.ablation_variants) != len(set(args.ablation_variants)):
        parser.error("--ablation-variants cannot contain duplicates")
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
    if args.require_cuda and args.require_mps:
        parser.error("--require-cuda and --require-mps cannot be used together")

    if args.require_cuda:
        args.device = "cuda"
    elif args.require_mps:
        args.device = "mps"

    started_at = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    requested_command = experiment_arguments(args)
    if args.run_name:
        run_root = output_root / args.run_name
        if run_root.exists() and not args.resume:
            parser.error(
                f"run already exists: {run_root}; choose another --run-name or enable --resume"
            )
    elif args.resume:
        run_root = find_resumable_run(output_root, requested_command)
        if run_root is None:
            run_root = unused_timestamped_run_root(output_root, args.seeds, started_at)
    else:
        run_root = unused_timestamped_run_root(output_root, args.seeds, started_at)

    logs_dir = run_root / "logs"
    status_path = run_root / "status.json"
    command_path = run_root / "command.json"
    pid_path = run_root / "pid.txt"
    accelerator = accelerator_status()
    previous_command = read_json(command_path)
    previous_status = read_json(status_path)

    if run_root.exists() and args.resume:
        if not isinstance(previous_command, dict):
            parser.error(f"cannot safely resume {run_root}: command.json is missing or invalid")
        if experiment_arguments(previous_command) != requested_command:
            parser.error(
                f"cannot resume {run_root}: requested experiment settings differ from command.json"
            )
        expected_process_tokens = ["run_multi_seed_experiments.py"]
        if args.run_name:
            expected_process_tokens.append(args.run_name)
        if (
            isinstance(previous_status, dict)
            and previous_status.get("state") in {"starting", "running"}
            and process_is_running(
                previous_status.get("pid"), expected_process_tokens
            )
        ):
            parser.error(
                f"cannot resume {run_root}: process {previous_status['pid']} is still running"
            )

    resuming = isinstance(previous_status, dict)
    status = dict(previous_status) if resuming else {}
    status.update(
        {
            "state": "starting",
            "pid": os.getpid(),
            "started_at": status.get("started_at", started_at),
            "run_root": str(run_root),
            "seeds": args.seeds,
            "task": args.task,
            "ablation_datasets": args.ablation_datasets,
            "ablation_variants": args.ablation_variants,
            "ablation_single_domains": args.ablation_single_domains,
            "models": args.models,
            "baselines": args.baselines,
            "epochs": args.epochs,
            "baseline_epochs": args.baseline_epochs,
            "pretrain_epochs": args.pretrain_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_workers": args.num_workers,
            "image_size": args.image_size,
            "heatmap_size": args.heatmap_size,
            "heatmap_sigma": args.heatmap_sigma,
            "real_data_dir": args.real_data_dir,
            "real_splits_dir": args.real_splits_dir,
            "synthetic_data_dir": args.synthetic_data_dir,
            "augmentation": args.augmentation,
            "pretrain_augmentation": args.pretrain_augmentation,
            "requested_device": args.device,
            "accelerator": accelerator,
            "open_file_limit": open_file_limit,
            "jobs": status.get("jobs", []),
            "resume_enabled": args.resume,
            "resuming": resuming,
        }
    )
    status.pop("error", None)
    status.pop("failed_seed", None)
    status.pop("interrupted_seed", None)
    status.pop("finished_at", None)
    if resuming:
        status["resumed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["resume_count"] = status.get("resume_count", 0) + 1
    else:
        status["resume_count"] = 0

    run_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    write_json(command_path, vars(args))
    write_json(status_path, status)

    unavailable = (
        (args.device == "cuda" and not accelerator.get("cuda_available", False))
        or (args.device == "mps" and not accelerator.get("mps_available", False))
    )
    if unavailable:
        status["state"] = "failed"
        status["error"] = f"{args.device.upper()} was requested but is unavailable."
        write_json(status_path, status)
        raise RuntimeError(status["error"])

    script = Path(__file__).with_name("run_full_experiments.py")
    status["state"] = "running"
    write_json(status_path, status)

    for seed in args.seeds:
        seed_output = run_root / f"seed_{seed}"
        seed_log = logs_dir / f"seed_{seed}.log"
        cmd = [
            sys.executable,
            "-u",
            str(script),
            "--task",
            args.task,
            "--output-dir",
            str(seed_output),
            "--epochs",
            str(args.epochs),
            "--baseline-epochs",
            str(args.baseline_epochs),
            "--pretrain-epochs",
            str(args.pretrain_epochs),
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--num-workers",
            str(args.num_workers),
            "--image-size",
            str(args.image_size),
            "--heatmap-size",
            str(args.heatmap_size),
            "--heatmap-sigma",
            str(args.heatmap_sigma),
            "--real-splits-dir",
            args.real_splits_dir,
            "--real-data-dir",
            args.real_data_dir,
            "--synthetic-data-dir",
            args.synthetic_data_dir,
        ]
        cmd.append("--augmentation" if args.augmentation else "--no-augmentation")
        cmd.append(
            "--pretrain-augmentation"
            if args.pretrain_augmentation
            else "--no-pretrain-augmentation"
        )
        if args.datasets:
            cmd.extend(["--datasets", *args.datasets])
        if args.ablation_datasets:
            cmd.extend(["--ablation-datasets", *args.ablation_datasets])
        if args.ablation_variants:
            cmd.extend(["--ablation-variants", *args.ablation_variants])
        if args.ablation_single_domains:
            cmd.extend(["--ablation-single-domains", *args.ablation_single_domains])
        if args.models:
            cmd.extend(["--models", *args.models])
        if args.baselines:
            cmd.extend(["--baselines", *args.baselines])

        previous_completed_job = completed_job(previous_status, seed)
        if args.resume and previous_completed_job is not None:
            previous_completed_job["resume_skipped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            status["current_seed"] = seed
            status["completed_seed_count"] = completed_seed_count(status, args.seeds)
            print(f"skip completed seed={seed}: {seed_output}")
            write_json(status_path, status)
            continue

        attempt = 1 + sum(
            job.get("seed") == seed
            for job in status["jobs"]
        )
        job = {
            "seed": seed,
            "state": "running",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir": str(seed_output),
            "log": str(seed_log),
            "command": cmd,
            "attempt": attempt,
        }
        status["jobs"].append(job)
        status["current_seed"] = seed
        write_json(status_path, status)

        with seed_log.open("a", encoding="utf-8") as log:
            if seed_log.stat().st_size:
                log.write("\n")
            log.write(json.dumps({"command": cmd, "seed": seed, "output_dir": str(seed_output)}) + "\n")
            log.flush()
            try:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            except KeyboardInterrupt:
                job["state"] = "interrupted"
                job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                status["state"] = "interrupted"
                status["interrupted_seed"] = seed
                status["finished_at"] = job["finished_at"]
                write_json(status_path, status)
                raise
            except Exception as exc:
                job["state"] = "failed"
                job["error"] = repr(exc)
                job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                status["state"] = "failed"
                status["failed_seed"] = seed
                status["error"] = repr(exc)
                status["finished_at"] = job["finished_at"]
                write_json(status_path, status)
                raise

        job["state"] = "completed" if result.returncode == 0 else "failed"
        job["returncode"] = result.returncode
        job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["completed_seed_count"] = completed_seed_count(status, args.seeds)
        write_json(status_path, status)
        if result.returncode != 0:
            status["state"] = "failed"
            status["failed_seed"] = seed
            status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_json(status_path, status)
            raise SystemExit(result.returncode)

    status["state"] = "completed"
    status.pop("current_seed", None)
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(status_path, status)


if __name__ == "__main__":
    main()
