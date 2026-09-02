from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


class LandmarkDataset(Dataset):
    def __init__(self, path, image_size=128, heatmap_size=64, sigma=1.8, augment=False, indices=None):
        z = np.load(Path(path))
        self.images = z["images"]
        self.keypoints = z["keypoints"].astype(np.float32)
        self.bbox = z["bbox"].astype(np.float32)
        self.indices = np.arange(len(self.images)) if indices is None else np.asarray(indices)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.augment = augment
        yy, xx = np.mgrid[: self.heatmap_size, : self.heatmap_size]
        self.xx = xx.astype(np.float32)
        self.yy = yy.astype(np.float32)

    def __len__(self):
        return len(self.indices)

    def _transform_keypoints(self, keypoints, angle, scale, translation):
        a = np.deg2rad(angle)
        center = np.array([self.image_size / 2, self.image_size / 2], dtype=np.float32)
        rotation = np.array(
            [[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]],
            dtype=np.float32,
        )
        return (
            ((keypoints - center) @ rotation) * scale
            + center
            + np.asarray(translation, dtype=np.float32)
        ).astype(np.float32)

    def _feasible_translation(self, keypoints):
        upper_boundary = self.image_size - 1
        lower = np.maximum(
            -8,
            np.ceil(-keypoints.min(axis=0)).astype(np.int64),
        )
        upper = np.minimum(
            8,
            np.floor(upper_boundary - keypoints.max(axis=0)).astype(np.int64),
        )
        if np.any(lower > upper):
            return None
        return lower, upper

    def _sample_affine(self, keypoints, max_attempts=32):
        zero_translation = np.zeros(2, dtype=np.float32)
        for _ in range(max_attempts):
            angle = float(np.random.uniform(-25, 25))
            scale = float(np.random.uniform(0.9, 1.1))
            transformed = self._transform_keypoints(
                keypoints, angle, scale, zero_translation
            )
            feasible = self._feasible_translation(transformed)
            if feasible is not None:
                lower, upper = feasible
                translation = np.array(
                    [
                        np.random.randint(lower[0], upper[0] + 1),
                        np.random.randint(lower[1], upper[1] + 1),
                    ],
                    dtype=np.float32,
                )
                return angle, scale, translation

        angle = 0.0
        scale = 0.9
        transformed = self._transform_keypoints(
            keypoints, angle, scale, zero_translation
        )
        feasible = self._feasible_translation(transformed)
        if feasible is None:
            raise ValueError(
                "landmarks cannot fit inside the image with the configured affine range"
            )
        lower, upper = feasible
        translation = np.clip(np.zeros(2, dtype=np.int64), lower, upper).astype(
            np.float32
        )
        return angle, scale, translation

    def _augment(self, image, keypoints):
        angle, scale, translation = self._sample_affine(keypoints)
        image = TF.affine(
            image,
            angle,
            translation.astype(int).tolist(),
            scale,
            0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        keypoints = self._transform_keypoints(
            keypoints, angle, scale, translation
        )
        image = TF.adjust_brightness(image, float(np.random.uniform(0.85, 1.15)))
        image = TF.adjust_contrast(image, float(np.random.uniform(0.85, 1.15)))
        image = (image + torch.randn_like(image) * float(np.random.uniform(0, 0.025))).clamp(0, 1)
        return image, keypoints.astype(np.float32)

    def _heatmaps(self, keypoints):
        ratio = self.heatmap_size / self.image_size
        points = keypoints.astype(np.float32) * ratio
        dx = self.xx[None, :, :] - points[:, 0, None, None]
        dy = self.yy[None, :, :] - points[:, 1, None, None]
        return torch.from_numpy(np.exp(-(dx**2 + dy**2) / (2 * self.sigma**2)).astype(np.float32))

    def __getitem__(self, i):
        j = self.indices[i]
        image = torch.from_numpy(self.images[j].copy()).float().unsqueeze(0) / 255.0
        keypoints = self.keypoints[j].copy()
        if self.augment:
            image, keypoints = self._augment(image, keypoints)
        keypoint_valid = (
            (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] <= self.image_size - 1)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] <= self.image_size - 1)
        )
        return {
            "image": image,
            "heatmaps": self._heatmaps(keypoints),
            "keypoints": torch.from_numpy(keypoints),
            "keypoint_valid": torch.from_numpy(keypoint_valid),
            "bbox": torch.from_numpy(self.bbox[j].copy()),
            "index": int(j),
        }
