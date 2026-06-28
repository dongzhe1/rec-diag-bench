"""General utilities: seeding, I/O, device detection, array normalization."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create directory and all parent directories."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """Save dict to JSON file with pretty formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load dict from JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_device(device_cfg: str = "auto") -> str:
    """Get device string: 'cuda' if available, else 'cpu'."""
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize values to zero mean and unit variance."""
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + eps)


def minmax(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    values = np.asarray(values, dtype=np.float64)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    return (values - lo) / (hi - lo + eps)
