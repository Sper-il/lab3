"""Module Huấn luyện mô hình (Gói G2 — Training).

Cung cấp API huấn luyện, tính metrics và vẽ biểu đồ loss/metric.
"""

from .metrics import calculate_metrics, compute_metrics
from .plot_utils import plot_loss_curve, plot_metrics_curve
from .trainer import (
    DEFAULT_TRAIN_CONFIG,
    LoggingTrackerCallback,
    build_training_args,
    train_model,
)

__all__ = [
    "DEFAULT_TRAIN_CONFIG",
    "LoggingTrackerCallback",
    "build_training_args",
    "calculate_metrics",
    "compute_metrics",
    "plot_loss_curve",
    "plot_metrics_curve",
    "train_model",
]

