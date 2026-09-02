import torch


def heatmaps_to_coords(heatmaps, image_size, method="argmax", temperature=40.0):
    if isinstance(heatmaps, (list, tuple)):
        if not heatmaps:
            raise ValueError("heatmap output sequence cannot be empty")
        heatmaps = heatmaps[-1]
    b, k, h, w = heatmaps.shape
    if method == "softargmax":
        prob = torch.softmax(heatmaps.reshape(b, k, -1) * temperature, dim=-1).reshape(b, k, h, w)
        yy, xx = torch.meshgrid(
            torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype),
            torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype),
            indexing="ij",
        )
        x = (prob * xx).sum((-2, -1)) * (image_size / w)
        y = (prob * yy).sum((-2, -1)) * (image_size / h)
        return torch.stack([x, y], -1)
    if method != "argmax":
        raise ValueError(f"unknown heatmap decoder: {method}")
    idx = heatmaps.reshape(b, k, -1).argmax(-1)
    x = (idx % w).float() * (image_size / w)
    y = (idx // w).float() * (image_size / h)
    return torch.stack([x, y], -1)
