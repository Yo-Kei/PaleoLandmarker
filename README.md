# PaleoLandmarker

PaleoLandmarker is a landmark-localization toolkit for fossil images. It provides the model architectures, Synthetic data generator, training and evaluation pipeline, baseline implementations, transfer-learning checkpoints, and scripts used to reproduce the paper experiments.

## Project structure

~~~text
src/paleolandmarker/     Core package: data, models, training, evaluation, baselines
scripts/                 Data preparation, training, evaluation, inference, experiment runners
configs/                 Dataset and transfer-learning configurations
analysis/                Paper-metric aggregation and validation scripts
data/                    Processed real datasets and Synthetic datasets
checkpoints/             Tiny, Small, Medium, and Large pretrained models
~~~

## Installation

Python 3.10 or newer is required. A CUDA-enabled PyTorch installation is recommended for the complete experiment matrix.

~~~bash
git clone <repository-url>
cd paleolandmarker
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
~~~

Install PyTorch according to your hardware before installing the remaining dependencies. Use the official [PyTorch Start Locally](https://pytorch.org/get-started/locally/) selector, choose your operating system, Pip, Python, and compute platform, then run the generated command inside the activated virtual environment. Select `CPU` if CUDA is not available; for an NVIDIA GPU, select the CUDA version supported by your driver. Verify the installation with:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
~~~

The complete paper experiment command uses `--require-cuda` and will stop if PyTorch cannot access CUDA.

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

Run all commands from the repository root.

## Transfer learning and inference

This is the shortest path for applying PaleoLandmarker to a new landmark dataset.

### 1. Download a pretrained checkpoint

From the repository's GitHub Releases page, download `paleolandmarker-pretrained-checkpoints.tar.gz` and extract its four `.pt` files into `checkpoints/`. Alternatively, download one checkpoint and place it there directly.

The available formal `context_resize` checkpoints are:

| Model | Parameters | Checkpoint |
|---|---:|---|
| Tiny | 529,769 | `checkpoints/PaleoLandmarker-Tiny-pretrained.pt` |
| Small | 937,001 | `checkpoints/PaleoLandmarker-Small-pretrained.pt` |
| Medium | 2,459,641 | `checkpoints/PaleoLandmarker-Medium-pretrained.pt` |
| Large | 4,792,940 | `checkpoints/PaleoLandmarker-Large-pretrained.pt` |

All four models were pretrained at 224x224 image resolution with 112x112 heatmaps. Their pretraining head has 18 Synthetic landmarks. During transfer learning, the compatible checkpoint loader imports matching encoder tensors and creates a target-specific prediction head, so the source and target datasets can have different landmark counts.

### 2. Prepare a dataset

The easiest route for researchers using tpsDig or another TPS-producing tool is to convert the TPS file directly. Place the source images in a directory and run:

~~~bash
python scripts/convert_tps.py \
  --tps /path/to/landmarks.tps \
  --images /path/to/images \
  --output data/real/processed_224/my_dataset \
  --dataset my_dataset \
  --size 224 \
  --coordinate-system tpsdig
~~~

The converter matches each TPS `IMAGE=` entry to an image filename, recursively searches the image directory, converts the default tpsDig lower-left coordinates to raster upper-left coordinates, and writes `dataset.npz`, `manifest.csv`, and a conversion audit. Use `--coordinate-system raster` when the TPS coordinates already use an upper-left raster origin. If the TPS file has no `IMAGE=` fields, add `--pair-by-order` when the sorted image directory contains exactly one image per TPS record. The converter assigns generic landmark names `LM1`, `LM2`, and so on in the order stored in the TPS file.

The resulting training file has this layout:

~~~text
data/real/processed_224/my_dataset/dataset.npz
~~~

The required arrays are:

| Array | Shape and meaning |
|---|---|
| `images` | `[N, 224, 224]` grayscale images, normally `uint8` |
| `keypoints` | `[N, K, 2]` landmark coordinates in `(x, y)` image pixels |
| `bbox` | `[N, 4]` tight landmark bounding boxes as `[x_min, y_min, x_max, y_max]` |

The coordinates must use the same 224x224 image space as `images`; `K` is the number of landmarks and must match `num_landmarks` in the configuration. Optional arrays such as `visibility`, `masks`, `labels`, and `landmark_names` are preserved when splits are created. Each image and its landmark array must have the same record index.

Create deterministic train/validation/test files for the new dataset:

~~~bash
python scripts/prepare_splits.py \
  --processed data/real/processed_224 \
  --output data/real/splits_224 \
  --dataset my_dataset \
  --seed 42
~~~

This creates `data/real/splits_224/my_dataset/source_{train,val,test}.npz` and a split record. The default split is 70/15/15; the paper runner creates the corresponding split for every experiment seed.

### 3. Configure and run transfer training

Copy the supplied configuration and update the dataset-specific fields:

~~~bash
cp configs/transfer_training.yaml configs/my_dataset_medium_transfer.yaml
~~~

Set at least `data_dir`, `output_dir`, `num_landmarks`, `model`, `seed`, and `init_checkpoint`. The example configuration is a 50-shot Medium transfer run. Start training with:

~~~bash
python scripts/train.py --config configs/my_dataset_medium_transfer.yaml
~~~

To change the number of labelled training examples without editing YAML:

~~~bash
python scripts/train.py --config configs/my_dataset_medium_transfer.yaml --shots 20
~~~

Use `--shots all` to train on every training record. The run directory contains the best checkpoint, training history, test metrics, and optional prediction visualisations.

### 4. Evaluate and run inference

Evaluate a trained checkpoint on a prepared split:

~~~bash
python scripts/evaluate.py \
  --config configs/my_dataset_medium_transfer.yaml \
  --checkpoint artifacts/reproduction/transfer/my_dataset_medium_50shots/checkpoints/best.pt \
  --split test
~~~

Run inference on one image, save a source-resolution overlay, and write its landmark coordinates as a TPS record:

~~~bash
python scripts/infer.py \
  --config configs/my_dataset_medium_transfer.yaml \
  --checkpoint artifacts/reproduction/transfer/my_dataset_medium_50shots/checkpoints/best.pt \
  --image /path/to/image.png \
  --output artifacts/reproduction/inference/example_prediction.png \
  --tps-output artifacts/reproduction/inference/example_prediction.tps
~~~

Single-image inference writes the overlay, prints the numeric `(x, y)` coordinates to the terminal, and writes a TPS file when `--tps-output` is supplied. For a directory, replace `--image` with `--image-dir`; a TPS file is then written automatically to `<image-dir>/predictions.tps` unless `--tps-output` is supplied. Add `--visualization-dir` to save one source-resolution overlay per image. Inference does not require an NPZ file; it loads the model configuration and checkpoint, letterboxes each raster image, and restores predictions to the original image coordinates.

## Reproducing the paper experiments

Before running the complete matrix, download and extract the processed real-data and Synthetic-data archives from the repository's GitHub Releases page:

~~~bash
mkdir -p data/real data/synthetic
tar -xzf paleolandmarker-real-processed-224.tar.gz -C data/real
tar -xzf paleolandmarker-synthetic-multidomain.tar.gz -C data/synthetic
~~~

The experiment protocol uses:

* six real datasets and six Synthetic domains;
* seeds `101 202 303 404 505 606 707 808`;
* 224x224 images, 112x112 heatmaps, and heatmap sigma 8;
* Tiny, Small, Medium, and Large models;
* 20-, 50-, and 100-shot transfer settings;
* 400 fine-tuning epochs with batch size 8 and learning rate 0.001;
* 50 Synthetic pretraining epochs with batch size 16;
* bounded affine/photometric augmentation; and
* heatmap-plus-coordinate supervision with coordinate weight 0.10 and softargmax temperature 40.

Run the complete experiment matrix with:

~~~bash
python scripts/run_multi_seed_experiments.py --seeds 101 202 303 404 505 606 707 808 --task all --output-root artifacts/reproduction/multiseed --real-data-dir data/real/processed_224 --real-splits-dir data/real/splits_224 --synthetic-data-dir data/synthetic/multidomain --device cuda --require-cuda --num-workers 4
~~~

The `all` task runs:

1. the six-configuration Medium ablation, including leave-one-domain-out pretraining;
2. the Tiny/Small/Medium/Large model sweep at 20, 50, and 100 shots; and
3. the comparison against the baseline implementations.

Run one component with `--task medium_ablation`, `--task model_sweep`, or `--task comparison`. For a small CPU smoke run:

~~~bash
python scripts/run_multi_seed_experiments.py --seeds 101 --task medium_ablation --output-root artifacts/reproduction/smoke --real-data-dir data/real/processed_224 --real-splits-dir data/real/splits_224 --synthetic-data-dir data/synthetic/multidomain --device cpu --num-workers 0
~~~

The experiment output contains per-seed configurations, split metadata, metrics, checkpoints, and visualisations. Aggregate the outputs with:

~~~bash
export PALEOLANDMARKER_RESULTS_ROOT=artifacts/reproduction/multiseed
python analysis/analyze_medium_ablation.py
python analysis/analyze_model_sweep.py
python analysis/analyze_comparison.py
~~~

## Data

The real-data archive contains six processed datasets. Each `dataset.npz` stores the resized grayscale images, landmark coordinates, visibility, masks, bounding boxes, labels, and landmark names required by the training and evaluation loaders.

| Dataset | Records | Landmarks |
|---|---:|---:|
| `conodont` | 173 | 27 |
| `fusulinid` | 240 | 8 |
| `land_snail_haplotrema_concavum` | 1,157 | 20 |
| `microtus_dinolite_notilt_eo_t2` | 247 | 21 |
| `ml_morph_bryozoan_zooids` | 3,800 | 23 |
| `trilomorph_cephala` | 150 | 16 |

The processed data contain standardized arrays rather than the original images. The Synthetic dataset can be regenerated locally and uses six domains corresponding to the real transfer tasks:

~~~text
conodont_element, fusulinid_section, gastropod_shell,
microtus_dentition, bryozoan_zooid, trilobite
~~~

## Synthetic data generation

The current Synthetic implementation is in `src/paleolandmarker/data/synthetic/generate.py`, with the command-line wrapper in `scripts/generate_synthetic.py`.

Create a visual preview with:

~~~bash
python scripts/generate_synthetic.py --mode preview --seed 20260813 --output-dir artifacts/reproduction/synthetic_preview
~~~

For the paper experiments, `run_full_experiments.py` generates or reuses the Synthetic working data for each requested seed and configuration.

## Reproducibility

The training engine seeds Python, NumPy, and PyTorch for each job. Real-data splits and Synthetic data generation are deterministic for a fixed configuration and seed. For the closest numerical match, keep the image size, heatmap size, batch size, worker count, device type, augmentation settings, optimizer settings, and seed unchanged. Small floating-point differences may occur across PyTorch, CUDA, driver, and hardware versions.
