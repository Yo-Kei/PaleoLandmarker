# Core package architecture

The `paleolandmarker` package implements the complete data-to-prediction path used by the command-line scripts. The main flow is:

```text
processed NPZ data
        │
        ▼
LandmarkDataset ──► model ──► heatmaps ──► decoded landmark coordinates
        │              │                         │
        └── targets ───┘                         └── metrics / visualisation / TPS output
```

## Package map

```text
paleolandmarker/
├── data/
│   ├── dataset.py              NPZ loader, targets, and training augmentation
│   ├── real/
│   │   ├── common.py           processed-data helpers and coordinate transforms
│   │   └── split.py            deterministic train/validation/test materialisation
│   └── synthetic/
│       └── generate.py         six-domain Synthetic generator and dataset writer
├── models/
│   ├── unet.py                 formal and original U-Net implementations
│   └── registry.py             model-name and architecture dispatch
├── training/
│   ├── engine.py               seeding, loaders, training, validation, and evaluation
│   ├── losses.py               heatmap and coordinate losses
│   └── checkpoints.py           checkpoint saving and compatible-weight loading
├── evaluation/
│   ├── decode.py               heatmap-to-coordinate decoding
│   ├── metrics.py              landmark metrics and aggregation
│   └── visualize.py            prediction overlays
├── baselines/                  controlled baseline implementations and registry
├── inference.py                image preprocessing, prediction, coordinate restoration, and TPS output
└── utils.py                    configuration loading and global random seeding
```

## Data and targets

`data/dataset.py` reads a processed `source_<split>.npz` file. It converts the stored grayscale image to a normalized tensor, applies the configured bounded affine/photometric augmentation during training, and creates Gaussian heatmap targets from the landmark coordinates. The loader returns the image, heatmaps, coordinates, valid-landmark mask, bounding box, and source index in one batch dictionary.

`data/real/split.py` creates the split-specific NPZ files from a processed dataset. For the paper protocol it uses the fixed seeded 70/15/15 partition and preserves the source arrays; the shot setting is applied by the training loader with the same configuration seed. `data/real/common.py` contains the image-space helpers used when a prediction must be mapped back to the original image geometry.

`data/synthetic/generate.py` creates the six Synthetic domains, their landmark annotations, and the source NPZ files consumed by `LandmarkDataset`. The public command-line entry point is `scripts/generate_synthetic.py`. The full experiment runner creates a seed-specific Synthetic working directory when a requested seed has not already been materialised.

## Models

`models/registry.py` is the single dispatch point. The formal paper models use the `context_resize` architecture and are selected with `tiny`, `small`, `medium`, or `large`. The registry passes `num_landmarks` to the selected constructor, so the prediction head is sized for the current task. The model output has shape:

```text
[batch, number_of_landmarks, heatmap_height, heatmap_width]
```

The `original_unet` option remains available for the comparison configuration that requires it. Model construction should go through `build_model(cfg)` so that architecture selection and landmark dimensions remain consistent with the experiment configuration.

## Training and transfer learning

`training/engine.py` provides the shared training implementation. `train_model(cfg, checkpoint)` performs the following operations:

1. seed Python, NumPy, and PyTorch using `cfg["seed"]`;
2. build the configured model and optionally load a compatible initial checkpoint;
3. construct deterministic train and validation loaders;
4. compute the configured heatmap-plus-coordinate loss and update the model;
5. select the checkpoint with the lowest validation coordinate error; and
6. return the best model together with history and run metadata.

`training/losses.py` contains the supervision terms and reads their weights from the nested `loss` configuration. `evaluation/decode.py` decodes predicted heatmaps using the configured method and temperature. `evaluation/metrics.py` then compares decoded coordinates with the stored targets, while `evaluation/visualize.py` writes optional overlays.

`training/checkpoints.py` implements transfer loading. `load_compatible` imports only tensors whose names and shapes match between the source and target models. This allows the pretrained encoder to be reused while the final task head is created for the target dataset's landmark count. The four checkpoint files in `checkpoints/` therefore work as initialisations for transfer training rather than as target-dataset predictors without a matching head.

## Inference path

`inference.py` exposes the reusable functions used by `scripts/infer.py`:

```text
image_paths
    ▼
load and letterbox image
    ▼
model forward pass
    ▼
heatmaps_to_coords
    ▼
restore coordinates to source-image space
    ▼
prediction overlay and/or TPS records
```

The inference script loads the model architecture from the configuration, loads the complete target checkpoint, accepts one image or a directory, and can write source-resolution overlays and TPS landmark records.

## Configuration contract

The YAML files provide the values consumed by the package. The most important fields are:

| Field | Used by |
|---|---|
| `data_dir` | training, evaluation, and dataset loading |
| `output_dir` | checkpoints, histories, metrics, and visualisations |
| `model`, `model_architecture`, `num_landmarks` | model construction |
| `image_size`, `heatmap_size`, `heatmap_sigma` | image and target preparation |
| `batch_size`, `epochs`, `learning_rate`, `weight_decay` | optimisation |
| `augmentation`, `loss`, `decode` | training and coordinate decoding |
| `seed`, `device`, `num_workers` | reproducibility and execution |
| `init_checkpoint` | compatible transfer initialisation |

Keeping these values in the configuration, rather than changing package code, is the intended way to run another dataset, shot count, model size, or transfer experiment.

## Public entry points

The package is normally accessed through the repository scripts:

- `scripts/prepare_splits.py` materialises deterministic real-data splits.
- `scripts/convert_tps.py` converts TPS annotations and their images into the processed NPZ format.
- `scripts/generate_synthetic.py` generates Synthetic data or visual previews.
- `scripts/train.py` trains a model and evaluates its test split.
- `scripts/evaluate.py` evaluates an existing checkpoint.
- `scripts/infer.py` predicts landmarks for images.
- `scripts/run_full_experiments.py` runs one complete experiment configuration.
- `scripts/run_multi_seed_experiments.py` expands the paper protocol over its seed list and experiment components.

These scripts import the package through `scripts/_bootstrap.py`, so commands should be run from the repository root as shown in the top-level README.

## Reproducibility boundary

The package deliberately centralises random-state setup in `utils.seed_everything` and calls it at the start of each training job. Synthetic generation and real-data split materialisation also receive an explicit seed. To reproduce a run, preserve the configuration, seed, data version, device type, worker count, and dependency versions together; changing those inputs can produce small numerical differences even when the high-level experiment is unchanged.
