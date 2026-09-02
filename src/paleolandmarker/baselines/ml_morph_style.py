from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from paleolandmarker.evaluation.metrics import compute_metrics
from paleolandmarker.evaluation.visualize import save_prediction


class MLMorphStyleBaseline:
    """Dlib shape-predictor style baseline using local image regression trees.

    The original ML-morph workflow trains a dlib ensemble shape predictor from
    XML annotations. This implementation keeps that comparison role without a
    fragile dlib build dependency: a multi-output random forest predicts all
    landmark coordinates from low-resolution intensity and edge features.
    """

    name = "ml_morph"

    def __init__(self, cfg):
        self.cfg = cfg
        self.feature_size = cfg.get("mlmorph_feature_size", 48)
        base = RandomForestRegressor(
            n_estimators=cfg.get("mlmorph_n_estimators", 250),
            max_depth=cfg.get("mlmorph_max_depth", 18),
            min_samples_leaf=cfg.get("mlmorph_min_samples_leaf", 2),
            random_state=cfg.get("seed", 42),
            n_jobs=cfg.get("n_jobs", -1),
        )
        self.model = MultiOutputRegressor(base, n_jobs=cfg.get("n_jobs", -1))

    def _load_npz(self, split):
        return np.load(Path(self.cfg["data_dir"]) / f"source_{split}.npz")

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

    def _features(self, images, bbox):
        feats = []
        for image, box in zip(images, bbox):
            im = Image.fromarray(image.astype(np.uint8)).resize((self.feature_size, self.feature_size), Image.Resampling.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            gx = np.diff(arr, axis=1, append=arr[:, -1:])
            gy = np.diff(arr, axis=0, append=arr[-1:, :])
            # Coarse quadrants add shape context similar to a full-object bounding box predictor.
            h = self.feature_size // 2
            quadrants = np.array(
                [
                    arr[:h, :h].mean(),
                    arr[:h, h:].mean(),
                    arr[h:, :h].mean(),
                    arr[h:, h:].mean(),
                    arr.std(),
                    np.abs(gx).mean(),
                    np.abs(gy).mean(),
                ],
                dtype=np.float32,
            )
            box = np.asarray(box, dtype=np.float32) / float(self.cfg["image_size"])
            feats.append(np.concatenate([arr.ravel(), gx.ravel(), gy.ravel(), quadrants, box]))
        return np.asarray(feats, dtype=np.float32)

    def fit(self):
        data = self._load_npz("train")
        idx = self._train_indices(len(data["images"]))
        x = self._features(data["images"][idx], data["bbox"][idx])
        y = data["keypoints"][idx].reshape(len(idx), -1)
        self.model.fit(x, y)
        return self

    def evaluate(self, split="test", visualize_dir=None, max_visualizations=8):
        data = self._load_npz(split)
        idx = self._eval_indices(split, len(data["images"]))
        x = self._features(data["images"][idx], data["bbox"][idx])
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
