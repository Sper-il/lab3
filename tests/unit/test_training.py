"""Unit tests cho module training (Gói G2).

Kiểm tra:
- Tính metrics (accuracy, f1, precision, recall) trên numpy và torch tensor.
- Hàm load_train_config và resolve path.
- Hàm build_training_args tạo đúng tham số Transformers 5.x.
- Trích xuất dữ liệu biểu đồ và xuất file PNG offline.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.training.metrics import calculate_metrics, compute_metrics
from src.training.plot_utils import (
    extract_curves_from_history,
    plot_loss_curve,
    plot_metrics_curve,
)
from src.training.trainer import build_training_args
from src.utils.config import load_train_config


def test_calculate_metrics_perfect():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    res = calculate_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert res["f1"] == 1.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0


def test_calculate_metrics_partial():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    res = calculate_metrics(y_true, y_pred)
    assert res["accuracy"] == 0.5
    assert 0.0 < res["f1"] < 1.0


def test_compute_metrics_numpy():
    # 3 samples, 2 classes
    logits = np.array([
        [2.5, -1.0],  # pred 0
        [-0.5, 3.0],  # pred 1
        [1.2, 0.8],   # pred 0
    ])
    labels = np.array([0, 1, 1])
    res = compute_metrics((logits, labels))
    # pred: [0, 1, 0], true: [0, 1, 1] -> 2/3 đúng
    assert abs(res["accuracy"] - 2 / 3) < 1e-4
    assert res["f1"] > 0.0


def test_compute_metrics_torch_tensor():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    res = compute_metrics((logits, labels))
    assert res["accuracy"] == 1.0
    assert res["f1"] == 1.0


def test_load_train_config():
    cfg = load_train_config("configs/train_config.yaml")
    assert "training" in cfg
    assert "data" in cfg
    assert cfg["training"]["learning_rate"] == 2e-5
    assert cfg["training"]["metric_for_best_model"] == "f1"
    assert cfg["data"]["include_test"] is False


def test_build_training_args(tmp_path):
    cfg = load_train_config("configs/train_config.yaml")
    targs = build_training_args(cfg, tmp_path, epochs=1.5, batch_size=8)
    assert targs.num_train_epochs == 1.5
    assert targs.per_device_train_batch_size == 8
    assert targs.metric_for_best_model == "f1"
    assert targs.eval_strategy == "epoch"
    assert targs.save_strategy == "epoch"
    assert targs.load_best_model_at_end is True


def test_extract_and_plot_curves(tmp_path):
    mock_history = [
        {"epoch": 0.5, "step": 10, "loss": 0.69},
        {"epoch": 1.0, "step": 20, "loss": 0.50},
        {"epoch": 1.0, "eval_loss": 0.48, "eval_accuracy": 0.80, "eval_f1": 0.79},
        {"epoch": 1.5, "step": 30, "loss": 0.35},
        {"epoch": 2.0, "step": 40, "loss": 0.25},
        {"epoch": 2.0, "eval_loss": 0.30, "eval_accuracy": 0.88, "eval_f1": 0.87},
    ]
    (tr_steps, tr_losses), (ev_eps, ev_losses), (ev_eps2, ev_f1s, ev_accs) = (
        extract_curves_from_history(mock_history)
    )
    assert len(tr_steps) == 4
    assert len(ev_losses) == 2
    assert ev_f1s == [0.79, 0.87]

    loss_file = tmp_path / "plots" / "loss_curve.png"
    metrics_file = tmp_path / "plots" / "metrics_curve.png"

    plot_loss_curve(mock_history, str(loss_file))
    plot_metrics_curve(mock_history, str(metrics_file))

    assert loss_file.exists() and loss_file.stat().st_size > 0
    assert metrics_file.exists() and metrics_file.stat().st_size > 0

