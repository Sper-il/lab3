"""Module models: backbone + tokenizer + loss cho bài toán phân loại nhị phân.

Bám theo sơ đồ phụ thuộc của kế hoạch (G1 Model):
    data/splits -> tokenize -> build_model -> logits (B, num_labels)
"""

from .architecture import build_model, count_parameters, model_size_mb
from .loss import build_loss
from .tokenizer_utils import build_tokenizer, tokenize_batch

__all__ = [
    "build_model",
    "build_tokenizer",
    "build_loss",
    "tokenize_batch",
    "count_parameters",
    "model_size_mb",
]
