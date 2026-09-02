from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import ExtraTreesRegressor

from paleolandmarker.evaluation.metrics import compute_metrics
from paleolandmarker.evaluation.visualize import save_prediction


class Vandaele2018StyleBaseline:
    """Classical tree-ensemble landmark regressor in the spirit of Cytomine LDM.

    This is not a byte-for-byte port of the Python 2 Cytomine app. It keeps the
    relevant comparison surface: non-deep, intensity-feature, ExtraTrees-based
    landmark coordinate regression on the same NPZ splits.
    """

    name = "vandaele2018"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = ExtraTreesRegressor(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", None),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            random_state=cfg.get("seed", 42),
            n_jobs=cfg.get("n_jobs", -1),
        )
        self.feature_size = cfg.get("feature_size", 32)

    def _load_npz(self, split):
        path = Path(self.cfg["data_dir"]) / f"source_{split}.npz"
        return np.load(path)

    def _train_indices(self, n):
        shot_count = self.cfg.get("shot_count")
        if shot_count in (None, "all"):
            return np.arange(n)
        rng = np.random.default_rng(self.cfg.get("seed", 42))
        return rng.permutation(n)[: min(int(shot_count), n)]

    def _eval_indices(self, split, n):
        max_samples = self.cfg.get(f"{split}_max_samples", self.cfg.get("eval_max_samples"))
        if max_samples in (None, "all"):
            return np.arange(n)
        rng = np.random.default_rng(self.cfg.get("seed", 42) + {"val": 1, "test": 2}.get(split, 3))
        return rng.permutation(n)[: min(int(max_samples), n)]

    def _features(self, images):
        feats = []
        for image in images:
            im = Image.fromarray(image.astype(np.uint8)).resize((self.feature_size, self.feature_size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            # Add coarse finite-difference texture cues without requiring scipy.
            gx = np.diff(arr, axis=1, append=arr[:, -1:])
            gy = np.diff(arr, axis=0, append=arr[-1:, :])
            feats.append(np.concatenate([arr.ravel(), gx.ravel(), gy.ravel()]))
        return np.asarray(feats, dtype=np.float32)

    def fit(self):
        data = self._load_npz("train")
        idx = self._train_indices(len(data["images"]))
        x = self._features(data["images"][idx])
        y = data["keypoints"][idx].reshape(len(idx), -1)
        self.model.fit(x, y)
        return self

    def evaluate(self, split="test", visualize_dir=None, max_visualizations=8):
        data = self._load_npz(split)
        idx = self._eval_indices(split, len(data["images"]))
        x = self._features(data["images"][idx])
        pred = self.model.predict(x).reshape(data["keypoints"][idx].shape).astype(np.float32)
        metrics = compute_metrics(
            pred,
            data["keypoints"][idx],
            self.cfg["image_size"],
            self.cfg.get("pck_thresholds", [3, 5, 10]),
            data["bbox"][idx],
        )
        if visualize_dir is not None:
            for i in range(min(len(idx), max_visualizations)):
                j = idx[i]
                save_prediction(data["images"][j], data["keypoints"][j], pred[i], Path(visualize_dir) / f"sample_{i:03d}.png")
        return metrics
