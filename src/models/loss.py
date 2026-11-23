"""Hàm loss cho bài toán phân loại nhị phân.

Theo config `configs/model_config.yaml`: `cross_entropy` (CrossEntropyLoss).
CrossEntropyLoss nhận trực tiếp `logits` (B, 2) + `labels` (B,) — đúng output của
model nên `Trainer` dùng luôn loss mặc định của Hugging Face cũng tương thích.
"""

import torch
import torch.nn as nn

from src.utils.config import load_model_config

DEFAULT_CONFIG = "configs/model_config.yaml"

SUPPORTED = {"cross_entropy", "ce", "crossentropy"}


def build_loss(config=None, loss_name: str = None):
    """Trả về `nn.Module` loss theo config.

    Args:
        config: dict config (`load_model_config()`), None -> đọc file mặc định.
        loss_name: ghi đè tên loss trong config.
    """
    cfg = config or load_model_config(DEFAULT_CONFIG)
    name = (loss_name or cfg.get("loss", {}).get("name", "cross_entropy")).lower()

    if name not in SUPPORTED:
        raise ValueError(
            f"Loss chưa hỗ trợ: {name!r}. Hiện hỗ trợ: {sorted(SUPPORTED)}"
        )
    return nn.CrossEntropyLoss()


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, loss_fn=None) -> torch.Tensor:
    """Tính loss từ logits (B, num_labels) và labels (B,)."""
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    return loss_fn(logits, labels)
