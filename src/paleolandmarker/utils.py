import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def project_path(*parts):
    return Path(__file__).resolve().parents[2].joinpath(*parts)

