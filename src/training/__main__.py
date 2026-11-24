"""Giao diện dòng lệnh CLI cho module Huấn luyện (Gói G2).

Cách dùng:
    # 1. Huấn luyện thông thường
    python -m src.training --config configs/train_config.yaml

    # 2. Overfit sanity check trên 1 batch nhỏ (bước 1 bắt buộc)
    python -m src.training --overfit-check

    # 3. Ước lượng thời gian train trên 5% dữ liệu (bước 2 bắt buộc)
    python -m src.training --estimate-time

    # 4. Huấn luyện với tham số tùy chỉnh / subset
    python -m src.training --epochs 2 --batch-size 8 --max-train-samples 1000

    # 5. Tiếp tục huấn luyện từ checkpoint
    python -m src.training --resume results/runs/<timestamp>/checkpoints/checkpoint-500
"""

import argparse
import sys
import time
from pathlib import Path

from src.models.overfit_check import main as run_overfit_check
from src.training.trainer import DEFAULT_TRAIN_CONFIG, train_model
from src.utils.config import load_model_config, load_train_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLI Huấn luyện mô hình Hugging Face (Gói G2)")
    p.add_argument("--config", default=DEFAULT_TRAIN_CONFIG, help="Đường dẫn file config train")
    p.add_argument("--resume", default=None, help="Đường dẫn checkpoint để tiếp tục huấn luyện")
    p.add_argument("--epochs", type=float, default=None, help="Ghi đè số epoch")
    p.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch size train")
    p.add_argument("--lr", type=float, default=None, help="Ghi đè learning rate")
    p.add_argument("--max-train-samples", type=int, default=None, help="Cắt tập train theo số lượng mẫu")
    p.add_argument("--max-eval-samples", type=int, default=None, help="Cắt tập eval theo số lượng mẫu")
    p.add_argument("--run-name", default=None, help="Tên tiền tố nhận diện run")
    p.add_argument(
        "--overfit-check",
        action="store_true",
        help="Chạy sanity check overfit 1 batch nhỏ (bước 1 theo kế hoạch)",
    )
    p.add_argument(
        "--estimate-time",
        action="store_true",
        help="Chạy benchmark để đo tốc độ và ước lượng tổng thời gian chạy full",
    )
    p.add_argument(
        "--benchmark-samples",
        type=int,
        default=200,
        help="Số mẫu dùng cho benchmark đo tốc độ (mặc định 200 mẫu ~ 3 phút)",
    )
    return p.parse_args(argv)


def run_time_estimation(config_path: str, n_train_samples: int = 200):
    """Bước 2 bắt buộc theo kế hoạch G2:

    Train thử 1 epoch trên subset train để ước lượng thời gian chạy full.
    """
    print("\n" + "=" * 70)
    print(f"BƯỚC 2: ƯỚC LƯỢNG THỜI GIAN HUẤN LUYỆN ({n_train_samples} MẪU TRAIN / 1 EPOCH)")
    print("=" * 70, flush=True)

    train_cfg = load_train_config(config_path)
    model_cfg = load_model_config(train_cfg.get("model_config_path", "configs/model_config.yaml"))

    n_eval_sample = min(100, n_train_samples // 2)

    print(f"Chạy benchmark trên {n_train_samples} mẫu train, eval trên {n_eval_sample} mẫu val...")
    t0 = time.time()
    trainer, run_dir, summary = train_model(
        config_path=config_path,
        max_train_samples=n_train_samples,
        max_eval_samples=n_eval_sample,
        epochs=1.0,
        run_name=f"benchmark_{n_train_samples}_samples",
    )
    elapsed = time.time() - t0

    samples_per_sec = n_train_samples / max(summary["train_runtime_sec"], 0.001)
    full_epoch_sec = 39662 / max(samples_per_sec, 0.001)
    full_3epochs_sec = full_epoch_sec * 3

    print("\n" + "=" * 70)
    print("KẾT QUẢ ƯỚC LƯỢNG THỜI GIAN (CPU LAB)")
    print("=" * 70)
    print(f"1. Tốc độ thực tế        : {samples_per_sec:.2f} samples/giây")
    print(f"2. Thời gian 1 epoch full: {full_epoch_sec / 3600:.2f} giờ ({full_epoch_sec / 60:.1f} phút)")
    print(f"3. Thời gian 3 epoch full: {full_3epochs_sec / 3600:.2f} giờ ({full_3epochs_sec / 60:.1f} phút)")
    print(f"4. Thư mục benchmark     : {run_dir}")
    print("=" * 70 + "\n")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.overfit_check:
        print("Đang chạy overfit sanity check trên 1 batch nhỏ...")
        return run_overfit_check(["--n", "8", "--epochs", "40"])

    if args.estimate_time:
        return run_time_estimation(args.config, args.benchmark_samples)

    # Chạy huấn luyện chính
    trainer, run_dir, summary = train_model(
        config_path=args.config,
        resume_from_checkpoint=args.resume,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        run_name=args.run_name,
    )
    return 0 if summary.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())

