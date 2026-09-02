# Data layout

`real/processed_224/` contains the six processed real datasets. Each
`dataset.npz` is self-contained and provides the images, landmark coordinates,
visibility, masks, bounding boxes, labels, and landmark names required by the
loaders.

`real/splits_224/` contains deterministic train/validation/test subsets created
by `scripts/prepare_splits.py` for each experiment seed.

`synthetic/multidomain/` contains the six-domain Synthetic dataset used by the
training pipeline and can be regenerated with `scripts/generate_synthetic.py`.
