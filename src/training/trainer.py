"""Logic huấn luyện chính (Gói G2 — Training).

Fine-tune mô hình nhị phân IMDb bằng Hugging Face Trainer với:
- Checkpoint best theo F1-score trên tập validation.
- Early stopping để chống overfit.
- Tự động lưu metrics.json, config.yaml, logs.txt và biểu đồ loss_curve.png.
- Hỗ trợ resume từ checkpoint và giới hạn số mẫu (subset) để ước lượng thời gian.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import yaml
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.models.architecture import build_model
from src.models.dataset_builder import build_datasets
from src.models.tokenizer_utils import build_tokenizer
from src.training.metrics import compute_metrics
from src.training.plot_utils import plot_loss_curve, plot_metrics_curve
from src.utils.config import load_model_config, load_train_config, repo_path

DEFAULT_TRAIN_CONFIG = "configs/train_config.yaml"


class LoggingTrackerCallback(TrainerCallback):
    """Callback lưu lại nhật ký từng epoch ra stdout và file logs.txt."""

    def __init__(self, log_file_path: Path):
        super().__init__()
        self.log_file_path = log_file_path

    def _append_log(self, text: str):
        print(text, flush=True)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        msg = f"[Epoch {state.epoch:.1f}/{args.num_train_epochs}] Step {state.global_step} hoàn tất."
        self._append_log(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            acc = metrics.get("eval_accuracy", 0.0)
            f1 = metrics.get("eval_f1", 0.0)
            loss = metrics.get("eval_loss", 0.0)
            msg = f"  -> Eval: loss={loss:.4f} | accuracy={acc:.4f} | f1={f1:.4f}"
            self._append_log(msg)


def build_training_args(
    train_cfg: Dict[str, Any],
    run_dir: Path,
    **overrides,
) -> TrainingArguments:
    """Tạo đối tượng TrainingArguments chuẩn cho Hugging Face 5.x."""
    tc = train_cfg.get("training", {})
    targs_dict = {
        "output_dir": str(run_dir / "checkpoints"),
        "learning_rate": float(overrides.get("learning_rate") or tc.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(
            overrides.get("batch_size") or tc.get("per_device_train_batch_size", 16)
        ),
        "per_device_eval_batch_size": int(
            overrides.get("eval_batch_size") or tc.get("per_device_eval_batch_size", 32)
        ),
        "num_train_epochs": float(overrides.get("epochs") or tc.get("num_train_epochs", 3)),
        "weight_decay": float(tc.get("weight_decay", 0.01)),
        "warmup_steps": int(tc.get("warmup_steps", 0)),
        "max_grad_norm": float(tc.get("max_grad_norm", 1.0)),
        "seed": int(tc.get("seed", 42)),
        "logging_steps": int(tc.get("logging_steps", 10)),
        "eval_strategy": str(tc.get("eval_strategy", "epoch")),
        "save_strategy": str(tc.get("save_strategy", "epoch")),
        "save_total_limit": int(tc.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(tc.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(tc.get("metric_for_best_model", "f1")),
        "greater_is_better": bool(tc.get("greater_is_better", True)),
        "use_cpu": bool(tc.get("use_cpu", True)),
        "report_to": tc.get("report_to", []),
        "disable_tqdm": False,
    }
    return TrainingArguments(**targs_dict)


def train_model(
    config_path: str = DEFAULT_TRAIN_CONFIG,
    resume_from_checkpoint: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    epochs: Optional[float] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    run_name: Optional[str] = None,
    save_plots: bool = True,
) -> Tuple[Trainer, Path, Dict[str, Any]]:
    """Hàm huấn luyện end-to-end cho Gói G2.

    Returns:
        (trainer, run_dir, metrics_summary)
    """
    train_cfg = load_train_config(config_path)
    model_cfg_path = train_cfg.get("model_config_path", "configs/model_config.yaml")
    model_cfg = load_model_config(model_cfg_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or train_cfg.get("training", {}).get("run_name", "g2-training")
    run_id = f"{timestamp}_{name}"

    base_out = repo_path(train_cfg.get("training", {}).get("output_dir", "results/runs"))
    run_dir = base_out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_file = run_dir / "logs.txt"

    # Lưu lại bản snapshot config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(train_cfg, f, allow_unicode=True, indent=2)

    with open(logs_file, "w", encoding="utf-8") as f:
        f.write(f"=== BẮT ĐẦU HUẤN LUYỆN G2 — RUN: {run_id} ===\n")
        f.write(f"Thời gian: {datetime.now().isoformat()}\n")
        f.write(f"Backbone : {model_cfg.get('model', {}).get('model_name')}\n\n")

    print("=" * 70)
    print(f"G2 TRAINING — RUN: {run_id}")
    print(f"Thư mục lưu kết quả: {run_dir}")
    print("=" * 70, flush=True)

    # 1. Load Tokenizer & Model
    print("\n[1] Nạp Tokenizer và Backbone Model...")
    tokenizer, tok_name = build_tokenizer(model_cfg)
    model, effective_model_name = build_model(model_cfg)
    print(f"    Backbone: {effective_model_name} ({tok_name})")

    # 2. Load Dataset (Khóa split test, chỉ nạp train & val)
    print("\n[2] Nạp dữ liệu splits (train & val)...")
    n_train = max_train_samples or train_cfg.get("data", {}).get("max_train_samples")
    n_eval = max_eval_samples or train_cfg.get("data", {}).get("max_eval_samples")
    max_samples_dict = {}
    if n_train:
        max_samples_dict["train"] = n_train
    if n_eval:
        max_samples_dict["val"] = n_eval

    datasets = build_datasets(
        config=model_cfg,
        tokenizer=tokenizer,
        include_test=False,
        max_samples=max_samples_dict if max_samples_dict else None,
    )
    train_ds = datasets["train"]
    val_ds = datasets["val"]
    print(f"    Train dataset     : {len(train_ds)} mẫu")
    print(f"    Validation dataset: {len(val_ds)} mẫu")

    # 3. TrainingArguments
    targs = build_training_args(
        train_cfg,
        run_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    # 4. Callbacks: Early Stopping + Logging Tracker
    tc = train_cfg.get("training", {})
    patience = int(tc.get("early_stopping_patience", 2))
    threshold = float(tc.get("early_stopping_threshold", 0.001))
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold,
        ),
        LoggingTrackerCallback(logs_file),
    ]

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 6. Huấn luyện
    print(f"\n[3] Bắt đầu fine-tuning (Epochs: {targs.num_train_epochs}, Batch: {targs.per_device_train_batch_size})...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    total_time = time.time() - start_time

    print(f"\n[4] Huấn luyện hoàn tất trong {total_time:.1f}s!")
    print(f"    Train loss cuối cùng: {train_result.training_loss:.4f}")

    # 7. Đánh giá cuối cùng trên Validation
    print("\n[5] Đánh giá tốt nhất trên Validation...")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 8. Lưu Checkpoint Best Model & Tokenizer
    best_model_path = run_dir / "best_model"
    print(f"\n[6] Lưu best model và tokenizer vào: {best_model_path}")
    trainer.save_model(str(best_model_path))
    tokenizer.save_pretrained(str(best_model_path))

    # 9. Vẽ biểu đồ Loss Curve & Metrics Curve
    history = trainer.state.log_history
    if save_plots and history:
        loss_curve_path = str(plots_dir / "loss_curve.png")
        metrics_curve_path = str(plots_dir / "metrics_curve.png")
        plot_loss_curve(history, loss_curve_path)
        plot_metrics_curve(history, metrics_curve_path)
        print(f"    Đã lưu biểu đồ: {loss_curve_path}")
        print(f"    Đã lưu biểu đồ: {metrics_curve_path}")

    # 10. Tạo metrics.json theo đúng quy ước DoD
    best_metric_name = targs.metric_for_best_model
    best_metric_val = eval_metrics.get(f"eval_{best_metric_name}", eval_metrics.get(best_metric_name))

    metrics_summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "backbone": effective_model_name,
        "num_epochs": float(targs.num_train_epochs),
        "batch_size": int(targs.per_device_train_batch_size),
        "learning_rate": float(targs.learning_rate),
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "train_runtime_sec": round(total_time, 2),
        "train_samples_per_second": round(len(train_ds) * targs.num_train_epochs / max(total_time, 0.001), 2),
        "train_loss": round(float(train_result.training_loss), 4),
        "best_metric": {
            "name": best_metric_name,
            "value": round(float(best_metric_val), 4) if best_metric_val is not None else None,
        },
        "validation_metrics": {
            "accuracy": round(float(eval_metrics.get("eval_accuracy", 0.0)), 4),
            "f1": round(float(eval_metrics.get("eval_f1", 0.0)), 4),
            "precision": round(float(eval_metrics.get("eval_precision", 0.0)), 4),
            "recall": round(float(eval_metrics.get("eval_recall", 0.0)), 4),
            "loss": round(float(eval_metrics.get("eval_loss", 0.0)), 4),
        },
        "best_checkpoint": str(trainer.state.best_model_checkpoint or best_model_path),
        "status": "COMPLETED",
        "log_history": history,
    }

    metrics_json_path = run_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    with open(logs_file, "a", encoding="utf-8") as f:
        f.write("\n=== KẾT QUẢ HUẤN LUYỆN ===\n")
        f.write(json.dumps(metrics_summary["validation_metrics"], indent=2) + "\n")
        f.write(f"Best checkpoint: {metrics_summary['best_checkpoint']}\n")
        f.write("Hoàn thành thành công.\n")

    print("\n" + "=" * 70)
    print(f"G2 TRAINING THÀNH CÔNG — Metrics: {metrics_json_path}")
    print("=" * 70)

    return trainer, run_dir, metrics_summary

