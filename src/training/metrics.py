"""Hàm tính metrics đánh giá trong quá trình huấn luyện (Gói G2).

Sử dụng scikit-learn để tính toán offline không cần kết nối mạng.
"""

from typing import Any, Dict, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """Hàm callback tính metrics cho Hugging Face Trainer.

    Args:
        eval_pred: EvalPrediction hoặc tuple (logits, labels)

    Returns:
        Dict gồm accuracy, f1, precision, recall
    """
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    # Nếu logits là tuple (một số model HF trả tuple)
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels).flatten()

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def calculate_metrics(y_true: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]) -> Dict[str, float]:
    """Tính các chỉ số phân loại nhị phân độc lập."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }

