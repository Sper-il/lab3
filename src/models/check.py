"""Kiểm tra Gói G1 (Model): tokenize 1 batch thật -> forward -> shape (B, 2).

Cách chạy:
    python -m src.models.check
    python -m src.models.check --config configs/model_config.yaml --batch-size 4
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.architecture import (
    build_model,
    count_parameters,
    count_trainable_parameters,
    model_size_mb,
)
from src.models.loss import build_loss
from src.models.tokenizer_utils import build_tokenizer, tokenize_batch
from src.utils.config import load_model_config, repo_path

SAMPLE_SENTENCES = [
    "I loved this movie, it was fantastic!",
    "This film was a complete waste of time and money.",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="G1 Model check: tokenize + forward shape")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None, help="Ghi kết quả ra file JSON (tuỳ chọn)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.config)

    max_len = int(cfg["tokenizer"]["max_length"])
    batch_size = args.batch_size or int(cfg["runtime"].get("check_batch_size", 4))
    device = args.device or cfg["runtime"].get("device", "cpu")
    seed = int(cfg["runtime"].get("seed", 42))
    torch.manual_seed(seed)

    label_names = cfg.get("label_names", {0: "NEGATIVE", 1: "POSITIVE"})

    print("=" * 70)
    print("G1 MODEL CHECK — tokenize + forward shape")
    print("=" * 70)
    print(f"Python        : {platform.python_version()} | torch {torch.__version__}")
    print(f"Device        : {device}")
    print(f"max_length    : {max_len}")
    print(f"batch_size    : {batch_size}")

    # ---- 1. Tokenizer -------------------------------------------------
    t0 = time.time()
    tokenizer, tok_name = build_tokenizer(cfg)
    print(f"\n[1] Tokenizer  : {tok_name} (load {time.time() - t0:.1f}s)")
    print(f"    vocab_size : {tokenizer.vocab_size}")
    print(f"    model_max  : {tokenizer.model_max_length}")
    print(f"    sample tok : {tokenizer.tokenize(SAMPLE_SENTENCES[0])[:12]}")

    # ---- 2. Model -----------------------------------------------------
    t0 = time.time()
    model, model_name = build_model(cfg)
    model.to(device)
    print(f"\n[2] Model      : {model_name} (load {time.time() - t0:.1f}s)")
    n_params = count_parameters(model)
    n_train = count_trainable_parameters(model)
    size_mb = model_size_mb(model)
    print(f"    params     : {n_params:,} (trainable {n_train:,})")
    print(f"    ~{n_params / 1e6:.1f}M params | ~{size_mb:.0f} MB (float32)")
    print(f"    num_labels : {model.config.num_labels} | problem_type={model.config.problem_type}")

    # ---- 3. Load 1 batch thật từ splits -------------------------------
    split_path = Path(cfg["paths"][f"{args.split}_file"])
    if not split_path.exists():
        split_path = repo_path(f"data/data/splits/{args.split}.csv")
    df = pd.read_csv(split_path)
    batch_df = df.head(batch_size)
    print(f"\n[3] Data       : {split_path} (tổng {len(df):,} dòng)")

    batch = tokenize_batch(
        batch_df["text"].tolist(),
        tokenizer,
        max_length=max_len,
        padding=cfg["tokenizer"].get("padding", "max_length"),
        truncation=cfg["tokenizer"].get("truncation", True),
        device=device,
    )
    labels = torch.tensor(batch_df["label"].tolist(), dtype=torch.long, device=device)
    print(f"    input_ids  : {tuple(batch['input_ids'].shape)}  dtype={batch['input_ids'].dtype}")
    print(f"    attn_mask  : {tuple(batch['attention_mask'].shape)}")
    if "token_type_ids" in batch:
        print(f"    token_type : {tuple(batch['token_type_ids'].shape)}")
    assert batch["input_ids"].shape == (batch_size, max_len), "Sai shape input_ids!"
    assert batch["attention_mask"].shape == (batch_size, max_len), "Sai shape attention_mask!"

    # ---- 4. Forward ----------------------------------------------------
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        out = model(**batch)
    dt = time.time() - t0
    logits = out.logits
    print(f"\n[4] Forward    : logits {tuple(logits.shape)}  (kỳ vọng ({batch_size}, 2))")
    print(f"    thời gian  : {dt:.3f}s | ~{dt / batch_size * 1000:.1f} ms/sample")
    assert logits.shape == (batch_size, 2), "Sai shape logits!"
    assert torch.isfinite(logits).all(), "logits có NaN/Inf!"

    # ---- 5. Loss + argmax ----------------------------------------------
    loss_fn = build_loss(cfg)
    loss = loss_fn(logits, labels)
    probs = torch.softmax(logits, dim=-1)
    preds = logits.argmax(dim=-1)
    print(f"\n[5] Loss       : {loss_fn.__class__.__name__} = {loss.item():.4f}")
    print(f"    softmax    : {tuple(probs.shape)}, mỗi hàng tổng = {probs.sum(-1)[0].item():.4f}")
    for i in range(min(2, batch_size)):
        txt = batch_df["text"].iloc[i][:70].replace("\n", " ")
        print(
            f"    - \"{txt}...\"\n"
            f"      label={int(labels[i])} ({label_names.get(int(labels[i]), '?')}) | "
            f"pred={int(preds[i])} ({label_names.get(int(preds[i]), '?')}) | "
            f"p(pos)={probs[i, 1].item():.4f}"
        )

    # ---- 6. Kết quả ----------------------------------------------------
    result = {
        "module": "G1_model",
        "backbone": model_name,
        "tokenizer": tok_name,
        "num_labels": 2,
        "max_length": max_len,
        "batch_size": batch_size,
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "logits_shape": list(logits.shape),
        "num_parameters": n_params,
        "num_trainable_parameters": n_train,
        "params_millions": round(n_params / 1e6, 2),
        "model_size_mb_float32": round(size_mb, 1),
        "loss_check": round(float(loss.item()), 4),
        "device": device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "forward_seconds": round(dt, 3),
        "status": "PASS",
    }
    print("\n" + "=" * 70)
    print(f"TẤT CẢ CHECK PASS — {result['params_millions']}M params, "
          f"logits {tuple(logits.shape)}, loss {result['loss_check']}")
    print("=" * 70)

    if args.out:
        out_path = repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Đã ghi kết quả: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
