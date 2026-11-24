"""Tiện ích vẽ biểu đồ huấn luyện (loss curve, metric curve) cho G2.

Lưu ảnh offline vào `results/runs/<run_dir>/plots/`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_curves_from_history(history: List[Dict[str, Any]]):
    """Trích xuất loss và metrics từ log_history của Trainer.

    Returns:
        (train_steps, train_losses), (eval_epochs, eval_losses), (eval_epochs, eval_f1s, eval_accs)
    """
    train_steps, train_losses = [], []
    eval_epochs, eval_losses = [], []
    eval_f1s, eval_accs = [], []

    for entry in history:
        # Step training log
        if "loss" in entry and ("step" in entry or "epoch" in entry):
            train_steps.append(entry.get("epoch", entry.get("step")))
            train_losses.append(entry["loss"])

        # Evaluation log
        if "eval_loss" in entry:
            ep = entry.get("epoch", len(eval_epochs) + 1)
            eval_epochs.append(ep)
            eval_losses.append(entry["eval_loss"])
            if "eval_f1" in entry:
                eval_f1s.append(entry["eval_f1"])
            if "eval_accuracy" in entry:
                eval_accs.append(entry["eval_accuracy"])

    return (train_steps, train_losses), (eval_epochs, eval_losses), (eval_epochs, eval_f1s, eval_accs)


def plot_loss_curve(history: List[Dict[str, Any]], output_path: str):
    """Vẽ đường cong loss huấn luyện và validation."""
    (tr_steps, tr_losses), (ev_eps, ev_losses), _ = extract_curves_from_history(history)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if tr_steps and tr_losses:
        plt.plot(tr_steps, tr_losses, label="Train Loss", color="#1f77b4", alpha=0.8, linewidth=1.8)
    if ev_eps and ev_losses:
        plt.plot(ev_eps, ev_losses, label="Validation Loss", color="#d62728", marker="o", linewidth=2.0)

    plt.title("Biểu đồ hàm mất mát (Loss Curve) — G2 Training", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (CrossEntropy)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_metrics_curve(history: List[Dict[str, Any]], output_path: str):
    """Vẽ đường cong metrics (F1 và Accuracy trên tập val) theo từng epoch."""
    _, _, (ev_eps, eval_f1s, eval_accs) = extract_curves_from_history(history)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    if ev_eps:
        if eval_f1s:
            plt.plot(ev_eps, eval_f1s, label="Val F1-Score", color="#2ca02c", marker="s", linewidth=2.0)
        if eval_accs:
            plt.plot(ev_eps, eval_accs, label="Val Accuracy", color="#ff7f0e", marker="^", linewidth=2.0)

    plt.title("Chỉ số đánh giá Validation (Accuracy & F1) — G2 Training", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

