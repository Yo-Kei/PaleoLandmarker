import numpy as np


def compute_metrics(pred, gt, image_size, thresholds=(3, 5, 10), bbox=None):
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    d = np.linalg.norm(pred - gt, axis=-1)
    if bbox is None:
        denom = np.full(d.shape[0], image_size * np.sqrt(2), dtype=np.float32)
    else:
        denom = np.sqrt((bbox[:, 2] - bbox[:, 0]) ** 2 + (bbox[:, 3] - bbox[:, 1]) ** 2).clip(1)
    result = {
        "mean_error_px": float(d.mean()),
        "normalized_mean_error": float((d / denom[:, None]).mean()),
        "per_landmark_error": [float(x) for x in d.mean(0)],
    }
    result.update({f"pck@{t}px": float((d <= t).mean()) for t in thresholds})
    return result

