import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    targets: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Tính accuracy, macro/binary precision/recall/F1, per-class và confusion matrix."""
    targets = targets or {"accuracy": 0.85, "f1": 0.85}

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    acc_target = float(targets.get("accuracy", 0.85))
    f1_target = float(targets.get("f1", 0.85))

    # Schema đúng yêu cầu G3: {metric, value, target, passed}
    summary_items = [
        {
            "metric": "accuracy",
            "value": round(acc, 4),
            "target": acc_target,
            "passed": bool(acc >= acc_target),
        },
        {
            "metric": "f1",
            "value": round(f1, 4),
            "target": f1_target,
            "passed": bool(f1 >= f1_target),
        },
        {
            "metric": "precision",
            "value": round(prec, 4),
            "target": None,
            "passed": None,
        },
        {
            "metric": "recall",
            "value": round(rec, 4),
            "target": None,
            "passed": None,
        },
    ]

    per_class = {
        "class_0_negative": {
            "precision": round(float(p_cls[0]), 4),
            "recall": round(float(r_cls[0]), 4),
            "f1": round(float(f1_cls[0]), 4),
            "support": int(sup_cls[0]),
        },
        "class_1_positive": {
            "precision": round(float(p_cls[1]), 4),
            "recall": round(float(r_cls[1]), 4),
            "f1": round(float(f1_cls[1]), 4),
            "support": int(sup_cls[1]),
        },
    }

    return {
        "summary": summary_items,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def extract_error_cases(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Lấy top_k mẫu dự đoán sai có độ tự tin (confidence) cao nhất để phân tích lỗi."""
    errors: List[Dict[str, Any]] = []
    for idx, (true_lbl, pred_lbl) in enumerate(zip(y_true, y_pred)):
        if int(true_lbl) != int(pred_lbl):
            conf = float(np.max(probs[idx]))
            errors.append(
                {
                    "index": int(idx),
                    "text": texts[idx],
                    "true_label": int(true_lbl),
                    "pred_label": int(pred_lbl),
                    "confidence": round(conf, 4),
                    "prob_class_0": round(float(probs[idx][0]), 4),
                    "prob_class_1": round(float(probs[idx][1]), 4),
                }
            )
    errors.sort(key=lambda item: item["confidence"], reverse=True)
    return errors[:top_k]


class LatencyTimer:
    """Đo thời gian suy luận trên CPU/device (ms)."""

    def __init__(self) -> None:
        self.batch_times_ms: List[float] = []
        self.total_samples: int = 0

    def record_batch(self, elapsed_sec: float, batch_size: int) -> None:
        self.batch_times_ms.append(elapsed_sec * 1000.0)
        self.total_samples += batch_size

    def report(self) -> Dict[str, float]:
        total_ms = float(sum(self.batch_times_ms))
        avg_ms_per_sample = total_ms / max(self.total_samples, 1)
        throughput_samples_per_sec = (
            self.total_samples / (total_ms / 1000.0) if total_ms > 0 else 0.0
        )
        return {
            "total_inference_ms": round(total_ms, 2),
            "avg_latency_ms_per_sample": round(avg_ms_per_sample, 2),
            "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
            "num_samples": int(self.total_samples),
        }