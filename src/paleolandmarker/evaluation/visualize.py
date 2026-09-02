from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_prediction(image, gt, pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if gt is not None:
        for x, y in gt:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(30, 200, 80))
    for x, y in pred:
        draw.line((x - 3, y, x + 3, y), fill=(230, 40, 40), width=1)
        draw.line((x, y - 3, x, y + 3), fill=(230, 40, 40), width=1)
    canvas.save(path)

