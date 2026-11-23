"""Smoke test: chứng minh code G1 chạy được với Trainer (dành cho G2).

Mục đích: trả lời câu hỏi "người sau có thể train được ngay không?".
Chạy vài step trên subset nhỏ để kiểm tra TẤT CẢ tích hợp:
    build_datasets() -> build_model() -> Trainer -> compute_metrics -> train()

Cách chạy:
    python -m src.models.smoke_train
    python -m src.models.smoke_train --max-steps 5 --n-train 64 --n-eval 32
"""

import argparse
import json
import sys
import time

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import Trainer, TrainingArguments

from src.models.architecture import build_model
from src.models.dataset_builder import build_datasets
from src.utils.config import load_model_config, repo_path

DEFAULT_CONFIG = "configs/model_config.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Smoke test Trainer cho G2")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def compute_metrics(eval_pred):
    """Metric cho G2/G3 — dùng sklearn, không cần network."""
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.config)
    torch.manual_seed(int(cfg["runtime"].get("seed", 42)))

    print("=" * 70)
    print("SMOKE TEST — Trainer tương thích với code G1?")
    print("=" * 70)

    # ---- 1. Dataset (đúng hàm G1 đã bàn giao) --------------------------
    print(f"\n[1] build_datasets() ...", flush=True)
    ds = build_datasets(config=cfg)
    print(f"    {ds}")

    # ---- 2. Model (đúng hàm G1 đã bàn giao) ----------------------------
    print(f"\n[2] build_model() ...", flush=True)
    model, model_name = build_model(cfg)
    print(f"    {model_name}")

    # ---- 3. Thu nhỏ subset cho nhanh ------------------------------------
    train_ds = ds["train"].select(range(min(args.n_train, len(ds["train"]))))
    eval_ds = ds["val"].select(range(min(args.n_eval, len(ds["val"]))))
    print(f"\n[3] Subset: train={len(train_ds)} | eval={len(eval_ds)}")

    # ---- 4. Trainer -------------------------------------------------------
    print(f"\n[4] TrainingArguments + Trainer ...", flush=True)
    targs = TrainingArguments(
        output_dir=repo_path("results/runs/g1-model/smoke"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=1,
        max_steps=args.max_steps,
        weight_decay=0.01,
        seed=42,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.max_steps,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    # ---- 5. Train thật ----------------------------------------------------
    print(f"\n[5] trainer.train() — {args.max_steps} step ...", flush=True)
    t0 = time.time()
    out = trainer.train()
    dt = time.time() - t0
    print(f"    xong trong {dt:.1f}s | train_loss = {out.training_loss:.4f}")

    # ---- 6. Evaluate ------------------------------------------------------
    print(f"\n[6] trainer.evaluate() ...", flush=True)
    ev = trainer.evaluate()
    for k, v in ev.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ---- 7. Predict -------------------------------------------------------
    print(f"\n[7] trainer.predict() ...", flush=True)
    p = trainer.predict(eval_ds)
    preds = p.predictions.argmax(axis=-1)
    print(f"    predictions shape: {tuple(p.predictions.shape)} | preds shape: {tuple(preds.shape)}")
    print(f"    labels shape     : {tuple(p.label_ids.shape)}")

    # ---- 8. Verdict --------------------------------------------------------
    ok = True
    checks = {
        "dataset có train/val": "train" in ds and "val" in ds,
        "model build được": model.config.num_labels == 2,
        "train chạy, loss hữu hạn": torch.isfinite(torch.tensor(out.training_loss)).item(),
        "evaluate trả metric": "eval_f1" in ev and "eval_accuracy" in ev,
        "predict shape khớp (B,2)": tuple(p.predictions.shape)[1] == 2,
    }
    for k, v in checks.items():
        print(f"    [{'PASS' if v else 'FAIL'}] {k}")
        ok = ok and v

    result = {
        "module": "G1_smoke_train",
        "backbone": model_name,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "max_steps": args.max_steps,
        "train_seconds": round(dt, 1),
        "train_loss": round(float(out.training_loss), 4),
        "eval": {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in ev.items()},
        "predictions_shape": list(p.predictions.shape),
        "checks": checks,
        "status": "PASS" if ok else "FAIL",
    }

    print("\n" + "=" * 70)
    print(f"{'PASS' if ok else 'FAIL'} — G1 sẵn sàng cho G2 (Training)")
    print("=" * 70)

    if args.out:
        out_path = repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Đã ghi: {out_path}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
