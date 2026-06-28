"""Path resolution and configuration loading for experiment pipelines."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config and resolve absolute paths."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("project_root", ".")).resolve()
    cfg["project_root"] = str(root)
    cfg["data_dir_abs"] = str((root / cfg.get("data_dir", "data")).resolve())
    cfg["output_dir_abs"] = str((root / cfg.get("output_dir", "outputs")).resolve())
    return cfg


def dataset_dir(cfg: Dict[str, Any], dataset: str) -> Path:
    """Return path to processed dataset directory."""
    return Path(cfg["data_dir_abs"]) / "processed" / dataset


def raw_dataset_dir(cfg: Dict[str, Any], dataset: str) -> Path:
    """Return path to raw dataset directory."""
    return Path(cfg["data_dir_abs"]) / "raw" / dataset


def output_dir(cfg: Dict[str, Any], dataset: str) -> Path:
    """Return path to output directory and create if necessary."""
    out = Path(cfg["output_dir_abs"]) / dataset
    out.mkdir(parents=True, exist_ok=True)
    return out
