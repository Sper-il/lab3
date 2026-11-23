"""Dựng backbone + classification head cho phân loại nhị phân.

Output của model: `logits` shape (B, num_labels) = (B, 2).
"""

import torch
from transformers import AutoModelForSequenceClassification

from src.utils.config import load_model_config

DEFAULT_CONFIG = "configs/model_config.yaml"

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def build_model(config=None, model_name: str = None, num_labels: int = None):
    """Khởi tạo `AutoModelForSequenceClassification` từ backbone pre-trained.

    Args:
        config: dict config (`load_model_config()`), None -> đọc file mặc định.
        model_name: ghi đè backbone (dùng cho test).
        num_labels: ghi đè số nhãn (mặc định lấy trong config = 2).

    Returns:
        (model, effective_model_name)
    """
    cfg = config or load_model_config(DEFAULT_CONFIG)
    mcfg = cfg.get("model", {})

    name = model_name or mcfg.get("model_name", "distilbert-base-uncased")
    fallback = mcfg.get("fallback_model_name")
    n_labels = int(num_labels if num_labels is not None else mcfg.get("num_labels", 2))
    problem_type = mcfg.get("problem_type", "single_label_classification")
    dtype = DTYPE_MAP.get(str(mcfg.get("dtype", "float32")).lower(), torch.float32)

    kwargs = dict(
        num_labels=n_labels,
        problem_type=problem_type,
        dtype=dtype,
    )

    last_err = None
    for candidate in [c for c in [name, fallback] if c]:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(candidate, **kwargs)
            # Ghi rõ tên backbone đã dùng (cho metrics.json / checkpoints)
            model.config._name_or_path = candidate
            return model, candidate
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"Không tải được model từ Hugging Face Hub ({name}): {last_err}. "
        "Kiểm tra mạng hoặc dùng cache local."
    )


def count_parameters(model) -> int:
    """Tổng số tham số có thể train (kể cả bias)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Số tham số được train (thường = tổng với full fine-tune)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model, dtype: torch.dtype = None) -> float:
    """Ước lượng dung lượng model (MB) theo dtype hiện tại của tham số."""
    n_params = count_parameters(model)
    bytes_per = torch.tensor([], dtype=dtype or torch.float32).element_size()
    return n_params * bytes_per / (1024**2)


@torch.no_grad()
def forward_check(model, batch: dict, device: str = "cpu") -> torch.Tensor:
    """Chạy forward 1 lần để kiểm tra shape, trả về logits (B, num_labels)."""
    model.eval()
    inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
    return model(**inputs).logits
