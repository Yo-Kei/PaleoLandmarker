import torch
import torch.nn.functional as F

from paleolandmarker.evaluation.decode import heatmaps_to_coords


def masked_landmark_mean(values, batch):
    valid = batch.get("keypoint_valid")
    if valid is None:
        return values.mean()
    valid = valid.to(values.device).bool()
    if valid.any():
        return values[valid].mean()
    return values.sum() * 0.0


def compute_loss(pred_heatmaps, batch, cfg):
    if isinstance(pred_heatmaps, (list, tuple)):
        if not pred_heatmaps:
            raise ValueError("heatmap output sequence cannot be empty")
        stack_results = [
            compute_loss(stack_heatmaps, batch, cfg)
            for stack_heatmaps in pred_heatmaps
        ]
        loss = torch.stack([result[0] for result in stack_results]).mean()
        details = {
            "stack_losses": [
                float(result[0].detach().cpu()) for result in stack_results
            ]
        }
        detail_names = set.intersection(
            *(set(result[1]) for result in stack_results)
        )
        for detail_name in detail_names:
            details[detail_name] = sum(
                result[1][detail_name] for result in stack_results
            ) / len(stack_results)
        return loss, details

    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "heatmap_mse")
    if name == "heatmap_mse":
        target_heatmaps = batch["heatmaps"].to(pred_heatmaps.device)
        heatmap_per_landmark = F.mse_loss(
            pred_heatmaps, target_heatmaps, reduction="none"
        ).mean((-2, -1))
        return masked_landmark_mean(heatmap_per_landmark, batch), {}
    if name in {"coord", "heatmap_coord"}:
        image_size = cfg["image_size"]
        temperature = loss_cfg.get("softargmax_temperature", 40.0)
        pred_coords = heatmaps_to_coords(pred_heatmaps, image_size, "softargmax", temperature)
        target_keypoints = batch["keypoints"].to(pred_heatmaps.device)
        coord_per_landmark = F.smooth_l1_loss(
            pred_coords / image_size,
            target_keypoints / image_size,
            reduction="none",
        ).mean(-1)
        coord_loss = masked_landmark_mean(coord_per_landmark, batch)
        if name == "coord":
            return coord_loss, {"coord_loss": float(coord_loss.detach().cpu())}

        target_heatmaps = batch["heatmaps"].to(pred_heatmaps.device)
        heatmap_per_landmark = F.mse_loss(
            pred_heatmaps, target_heatmaps, reduction="none"
        ).mean((-2, -1))
        heatmap_loss = masked_landmark_mean(heatmap_per_landmark, batch)
        coord_weight = loss_cfg.get("coord_weight", 0.10)
        return heatmap_loss + coord_weight * coord_loss, {
            "heatmap_loss": float(heatmap_loss.detach().cpu()),
            "coord_loss": float(coord_loss.detach().cpu()),
        }
    raise ValueError(f"unknown loss: {name}")
