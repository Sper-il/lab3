"""Tiện ích đọc config YAML dùng chung cho toàn repo.

Single source of truth: mọi tham số nằm trong `configs/*.yaml`, code chỉ đọc.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def repo_path(relative: str) -> Path:
    """Trả về đường dẫn tuyệt đối từ đường dẫn tương đối so với gốc repo."""
    p = Path(relative)
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_config(path: str) -> Dict[str, Any]:
    """Đọc file YAML config (đường dẫn tương đối so với gốc repo nếu cần)."""
    p = repo_path(path)
    if not p.exists() and not p.is_absolute():
        p = CONFIG_DIR / Path(path).name
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: str = "configs/model_config.yaml") -> Dict[str, Any]:
    """Đọc `configs/model_config.yaml` và gộp thêm đường dẫn splits đã resolve."""
    cfg = load_config(path)
    paths = cfg.get("paths", {})
    splits_dir = repo_path(paths.get("splits_dir", "data/data/splits"))
    cfg["paths"] = {
        **paths,
        "splits_dir": str(splits_dir),
        "train_file": str(splits_dir / paths.get("train_file", "train.csv")),
        "val_file": str(splits_dir / paths.get("val_file", "val.csv")),
        "test_file": str(splits_dir / paths.get("test_file", "test.csv")),
    }
    return cfg
