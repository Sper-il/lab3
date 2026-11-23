"""Đánh giá nhanh model trên dữ liệu THẬT (chưa fine-tune) để xác nhận pipeline chạy đúng end-to-end.

Mục đích của G1 (Model) không phải đạt accuracy cao — mà là chứng minh:
    splits -> tokenize -> forward -> logits -> loss -> metric   chạy trơn tru trên dữ liệu thật.

Vì classification head được khởi tạo ngẫu nhiên, accuracy gần 50% là KẾT QUẢ ĐÚNG
và mong đợi (baseline "random"). Muốn accuracy cao thì phải qua G2 (Training).

Cách chạy:
    python -m src.models.eval_quick --n 200 --split val
    python -m src.models.eval_quick --n 500 --split val --out results/runs/g1-model/eval_quick.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.models.architecture import build_model
from src.models.loss import build_loss
from src.models.tokenizer_utils import build_tokenizer, tokenize_batch
from src.utils.config import load_model_config, repo_path

DEFAULT_CONFIG = "configs/model_config.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="G1 eval nhanh trên dữ liệu thật")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=200, help="Số mẫu đánh giá")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, df, tokenizer, cfg, batch_size: int, device: str = "cpu"):
    """Chạy inference trên toàn bộ df, trả về logits, labels, loss, timing."""
    model.eval()
    max_len = int(cfg["tokenizer"]["max_length"])
    loss_fn = build_loss(cfg)

    all_logits, all_labels, total_loss, n_batches = [], [], 0.0, 0
    t0 = time.time()
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        batch = tokenize_batch(
            chunk["text"].tolist(),
            tokenizer,
            max_length=max_len,
            padding=cfg["tokenizer"].get("padding", "max_length"),
            truncation=cfg["tokenizer"].get("truncation", True),
            device=device,
        )
        labels = torch.tensor(chunk["label"].tolist(), dtype=torch.long, device=device)
        logits = model(**batch).logits
        total_loss += loss_fn(logits, labels).item()
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    dt = time.time() - t0
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=-1)
    return logits, labels, preds, total_loss / max(n_batches, 1), dt


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.config)
    device = cfg["runtime"].get("device", "cpu")
    torch.manual_seed(int(cfg["runtime"].get("seed", 42)))

    split_path = Path(cfg["paths"][f"{args.split}_file"])
    df = pd.read_csv(split_path)
    n = min(args.n, len(df))
    sample = df.head(n).reset_index(drop=True)  # lấy head để tái lập được (seed cố định)

    print("=" * 70)
    print(f"G1 EVAL QUICK — {args.split} | {n}/{len(df):,} mẫu | batch {args.batch_size}")
    print("=" * 70)

    tokenizer, tok_name = build_tokenizer(cfg)
    model, model_name = build_model(cfg)
    model.to(device)

    logits, labels, preds, avg_loss, dt = evaluate(model, sample, tokenizer, cfg, args.batch_size, device)

    y_true = labels.numpy()
    y_pred = preds.numpy()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"\nModel       : {model_name}  (CHƯA fine-tune — head random init)")
    print(f"Tokenize    : {tok_name} | max_len={cfg['tokenizer']['max_length']}")
    print(f"Loss (avg)  : {avg_loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1          : {f1:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"Confusion   : [[TN={cm[0,0]}, FP={cm[0,1]}], [FN={cm[1,0]}, TP={cm[1,1]}]]")
    print(f"Thời gian    : {dt:.1f}s cho {n} mẫu = {dt/n*1000:.1f} ms/sample (CPU, inference)")
    print(f"label dist  : {dict(sample['label'].value_counts().sort_index().items())}")

    print("\n⚠  Lưu ý: đây là model CHƯA fine-tune. Accuracy ~50% là đúng — đây là")
    print("   baseline random, chứng minh pipeline chạy đúng. G2 (Training) mới đạt")
    print("   target accuracy/F1 >= 0.85.")

    result = {
        "module": "G1_model_eval_quick",
        "backbone": model_name,
        "tokenizer": tok_name,
        "fine_tuned": False,
        "split": args.split,
        "n_samples": n,
        "batch_size": args.batch_size,
        "metrics": {
            "accuracy": round(float(acc), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "avg_loss": round(float(avg_loss), 4),
        },
        "confusion_matrix": cm.tolist(),
        "inference_seconds": round(dt, 2),
        "ms_per_sample": round(dt / n * 1000, 1),
        "label_distribution": {str(k): int(v) for k, v in sample["label"].value_counts().items()},
        "note": "Random baseline tren model chua fine-tune. Target >= 0.85 se dat sau G2 (Training).",
    }
    print("\n" + "=" * 70)

    if args.out:
        out_path = repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Đã ghi: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
