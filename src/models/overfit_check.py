"""Chứng minh model THỰC SỰ HỌC ĐƯỢC — overfit 1 batch nhỏ.

Đây là sanity check quan trọng nhất trước khi train full (theo `remaining_plan.md` §G2:
"overfit 1 batch nhỏ sanity check"). Nếu model không thể overfit được 8 mẫu thì
chắc chắn sẽ không train được 39.662 mẫu.

Nếu loss giảm mạnh về 0 → backbone + head + tokenizer + loss đều đúng,
G2 chỉ cần chạy Trainer là có model tốt.

Cách chạy:
    python -m src.models.overfit_check --n 8 --epochs 30
"""

import argparse
import json
import sys
import time

import torch
from torch.utils.data import DataLoader

from src.models.architecture import build_model
from src.models.dataset_builder import build_datasets
from src.models.loss import build_loss
from src.utils.config import load_model_config, repo_path

DEFAULT_CONFIG = "configs/model_config.yaml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Overfit 1 batch để chứng minh model học được")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--n", type=int, default=8, help="Số mẫu (một batch)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_model_config(args.config)
    seed = int(cfg["runtime"].get("seed", 42))
    torch.manual_seed(seed)

    print("=" * 70)
    print(f"OVERFIT CHECK — {args.n} mẫu, {args.epochs} epoch, lr={args.lr}")
    print("=" * 70)

    ds = build_datasets(config=cfg)
    batch = ds["train"].select(range(args.n))

    model, model_name = build_model(cfg)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    loader = DataLoader(batch, batch_size=args.n, shuffle=False)

    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for b in loader:
            inputs = {k: v for k, v in b.items() if k != "labels"}
            labels = b["labels"]
            logits = model(**inputs).logits
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        history.append(round(loss.item(), 4))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d} | loss = {loss.item():.4f}")

    dt = time.time() - t0

    # ---- Đánh giá trên đúng batch đó ---------------------------------
    # Lưu ý: dataset format=torch -> ds[0] trả về tensor 1D (256,), cần thêm batch dim
    model.eval()
    with torch.no_grad():
        single = batch[0]
        inputs = {
            k: (v.unsqueeze(0) if torch.is_tensor(v) else torch.tensor([v]))
            for k, v in single.items()
            if k != "labels"
        }
        logits = model(**inputs).logits
        preds = logits.argmax(-1)
        labels = torch.tensor([single["labels"]]) if not torch.is_tensor(single["labels"]) else single["labels"].unsqueeze(0)
        acc = (preds == labels).float().mean().item()

    first, last = history[0], history[-1]
    print("\n" + "-" * 70)
    print(f"Loss đầu : {first}")
    print(f"Loss cuối: {last}")
    print(f"Accuracy trên batch đó: {acc:.4f}")
    print(f"Thời gian : {dt:.1f}s")

    can_learn = last < 0.05 and acc == 1.0
    print(f"\nKết luận  : model {'CÓ THỂ học' if can_learn else 'CHƯA học được'} (loss {first} → {last})")

    result = {
        "module": "G1_overfit_check",
        "backbone": model_name,
        "n_samples": args.n,
        "epochs": args.epochs,
        "lr": args.lr,
        "loss_first": first,
        "loss_last": last,
        "accuracy_on_batch": round(acc, 4),
        "seconds": round(dt, 1),
        "history": history,
        "can_learn": bool(can_learn),
        "conclusion": (
            "Backbone + head + tokenizer + loss deu dung -> G2 co the train binh thuong."
            if can_learn
            else "Chua overfit duoc -> can kiem tra lai learning rate / so epoch / loss."
        ),
    }

    if args.out:
        out_path = repo_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Đã ghi: {out_path}")

    return 0 if can_learn else 1


if __name__ == "__main__":
    sys.exit(main())
