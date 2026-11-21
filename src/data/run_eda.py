"""Script EDA + làm sạch: raw -> processed -> splits + plots + metrics.json.

Chạy: python -m src.data.run_eda
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.insert(0, os.getcwd())

from src.data.clean import clean_text  # noqa: E402
from src.data.dataset import load_raw  # noqa: E402
from src.data.split import make_splits  # noqa: E402


def load_config(path="configs/data_config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    os.makedirs(cfg["splits_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["processed_path"]), exist_ok=True)
    os.makedirs(os.path.join(cfg["results_dir"], "plots"), exist_ok=True)
    os.makedirs(cfg["raw_dir"], exist_ok=True)

    print("Đang tải IMDb thô từ Hugging Face Hub...")
    raw = load_raw(cfg["dataset"], cfg.get("fallback_dataset", "imdb"))
    df_raw = raw.to_pandas()
    print(f"Raw: {len(df_raw)} mẫu, cột {list(df_raw.columns)}")

    # Snapshot raw bất biến (mẫu nhỏ để truy vết)
    df_raw.head(1000).to_csv(os.path.join(cfg["raw_dir"], "imdb_raw_sample.csv"), index=False)

    # --- EDA trước làm sạch ---
    n_total = len(df_raw)
    n_null = int(df_raw["text"].isna().sum())
    n_empty_raw = int((df_raw["text"].fillna("").str.strip() == "").sum())
    n_dup_raw = int(df_raw.duplicated(subset=["text"]).sum())
    class_raw = df_raw["label"].value_counts(normalize=True).to_dict()
    lens_raw = df_raw["text"].fillna("").str.split().str.len()

    # --- Làm sạch ---
    df = df_raw.copy()
    df["text"] = df["text"].fillna("").map(clean_text)
    df = df[~df["text"].str.strip().eq("")]
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    n_dup_removed = before_dedup - len(df)

    df.to_csv(cfg["processed_path"], index=False)

    # --- EDA sau làm sạch ---
    n_clean = len(df)
    class_clean = df["label"].value_counts(normalize=True).to_dict()
    lens = df["text"].str.split().str.len()
    len_stats = {
        "mean": float(lens.mean()),
        "median": float(lens.median()),
        "p90": float(lens.quantile(0.9)),
        "p99": float(lens.quantile(0.99)),
        "max": int(lens.max()),
    }

    # --- Splits ---
    train_df, val_df, test_df = make_splits(df, tuple(cfg["split_ratio"]), cfg["seed"])
    train_df.to_csv(os.path.join(cfg["splits_dir"], "train.csv"), index=False)
    val_df.to_csv(os.path.join(cfg["splits_dir"], "val.csv"), index=False)
    test_df.to_csv(os.path.join(cfg["splits_dir"], "test.csv"), index=False)

    def _dist(d):
        return d["label"].value_counts(normalize=True).to_dict()

    metrics = {
        "raw": {
            "n_total": n_total,
            "n_null": n_null,
            "n_empty": n_empty_raw,
            "n_dup_text": n_dup_raw,
            "class_ratio": {str(k): float(v) for k, v in class_raw.items()},
            "len_mean": float(lens_raw.mean()),
            "len_median": float(lens_raw.median()),
        },
        "clean": {
            "n_clean": n_clean,
            "n_dup_removed": int(n_dup_removed),
            "n_empty_after": 0,
            "class_ratio": {str(k): float(v) for k, v in class_clean.items()},
            "len_stats": len_stats,
        },
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "train_dist": {str(k): float(v) for k, v in _dist(train_df).items()},
            "val_dist": {str(k): float(v) for k, v in _dist(val_df).items()},
            "test_dist": {str(k): float(v) for k, v in _dist(test_df).items()},
            "seed": cfg["seed"],
        },
    }
    with open(os.path.join(cfg["results_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # --- Plots ---
    sample = df.sample(min(len(df), cfg.get("sample_size_for_eda_plot", 5000)), random_state=cfg["seed"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(["negative(0)", "positive(1)"],
                [int((df['label'] == 0).sum()), int((df['label'] == 1).sum())])
    axes[0].set_title("Phan bo lop sau lam sach")
    axes[0].set_ylabel("So mau")
    axes[1].hist(sample["text"].str.split().str.len(), bins=50)
    axes[1].set_title("Phan bo do dai review (so tu)")
    axes[1].set_xlabel("So tu")
    axes[1].set_ylabel("Tan suat")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["results_dir"], "plots", "eda_overview.png"), dpi=150)
    plt.close()

    print("=== KẾT QUẢ EDA + LÀM SẠCH ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Đã lưu splits + plots vào {cfg['results_dir']}")


if __name__ == "__main__":
    main()
