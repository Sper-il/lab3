"""Dựng tokenizer + tokenize batch thành tensor sẵn sàng cho model.

Kiến trúc theo Design:
    text (list[str]) -> input_ids (B, 256), attention_mask (B, 256)

Ràng buộc: `padding="max_length"` để shape luôn cố định (B, max_length),
`truncation=True` để không vỡ khi review dài hơn 256 token.
"""

from typing import Dict, List, Sequence

import torch
from transformers import AutoTokenizer

from src.utils.config import load_model_config

DEFAULT_CONFIG = "configs/model_config.yaml"


def build_tokenizer(config=None, model_name: str = None, cache_dir: str = None):
    """Khởi tạo AutoTokenizer cho backbone.

    Args:
        config: dict config đã load (`load_model_config()`), nếu None sẽ tự đọc file mặc định.
        model_name: ghi đè tên model (dùng cho test).
        cache_dir: thư mục cache HF, mặc định lấy từ config/env.

    Returns:
        (tokenizer, effective_model_name)
    """
    cfg = config or load_model_config(DEFAULT_CONFIG)
    mcfg = cfg.get("model", {})
    tcfg = cfg.get("tokenizer", {})

    name = model_name or mcfg.get("model_name", "distilbert-base-uncased")
    fallback = mcfg.get("fallback_model_name")
    kwargs = dict(
        lowercase=bool(tcfg.get("lowercase", True)),
        cache_dir=cache_dir,
    )

    last_err = None
    for candidate in [n for n in [name, fallback] if n]:
        try:
            tok = AutoTokenizer.from_pretrained(candidate, **kwargs)
            return tok, candidate
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(
        f"Không tải được tokenizer từ Hugging Face Hub ({name}): {last_err}. "
        "Kiểm tra mạng hoặc đặt HF_HUB_OFFLINE=1 nếu đã có cache local."
    )


def tokenize_batch(
    texts: Sequence[str],
    tokenizer,
    max_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True,
    return_tensors: str = "pt",
    device: str = None,
) -> Dict[str, torch.Tensor]:
    """Tokenize một batch văn bản thành tensor.

    Args:
        texts: danh sách văn bản đầu vào.
        tokenizer: đối tượng AutoTokenizer.
        max_length: độ dài cắt (khớp Design: 256).
        padding: "max_length" (mặc định) để shape cố định (B, max_length).
        truncation: cắt chuỗi dài.
        return_tensors: "pt" trả về torch.Tensor.
        device: nếu không None, chuyển tensor sang device (ví dụ "cpu").

    Returns:
        dict với `input_ids`, `attention_mask` (và `token_type_ids` nếu tokenizer có).
        Shape: (B, max_length).
    """
    enc = tokenizer(
        list(texts),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    batch = {k: v for k, v in enc.items()}
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def tokenize_dataframe(
    df,
    tokenizer,
    text_col: str = "text",
    max_length: int = 256,
    padding: str = "max_length",
    truncation: bool = True,
) -> Dict[str, List[List[int]]]:
    """Tokenize cả DataFrame (dùng cho Trainer: dùng `datasets.Dataset.map`).

    Trả về dict các cột giống `datasets.Dataset.map` mong đợi, giữ nguyên index.
    """
    enc = tokenizer(
        df[text_col].tolist(),
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )
    out = {k: list(v) for k, v in enc.items()}
    out["labels"] = list(df["label"].tolist())
    return out
