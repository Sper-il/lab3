"""Chuẩn bị dataset Hugging Face đã tokenize sẵn cho người làm G2 (Training).

G1 chỉ cần kiểm tra shape, nhưng để người tiếp theo (Training) không phải
làm lại phần tokenize, module này cung cấp:

    build_datasets(config) -> datasets.DatasetDict
        {"train": Dataset(input_ids, attention_mask, label), "val": ..., "test": ...}

Dùng:
    from src.models.dataset_builder import build_datasets
    ds = build_datasets()
    print(ds["train"].features)
"""

import pandas as pd
from datasets import Dataset, DatasetDict

from src.models.tokenizer_utils import build_tokenizer, tokenize_dataframe
from src.utils.config import load_model_config

DEFAULT_CONFIG = "configs/model_config.yaml"


def load_split(path, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Đọc 1 file split, chuẩn hoá cột text/label."""
    df = pd.read_csv(path)
    cols = {"text": text_col, "label": label_col}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise KeyError(f"Thiếu cột {missing} trong {path}. Có: {list(df.columns)}")
    df = df.rename(columns=cols)[["text", "label"]]
    df["label"] = df["label"].astype("int64")
    return df


def build_datasets(
    config=None,
    tokenizer=None,
    include_test: bool = True,
    tokenize: bool = True,
    format: str = "torch",
):
    """Dựng DatasetDict train/val/test từ `data/splits`, đã tokenize sẵn.

    Args:
        config: dict config, None -> đọc `configs/model_config.yaml`.
        tokenizer: tokenizer dùng chung (tránh load 2 lần).
        include_test: có kèm split test không (mặc định True).
        tokenize: False -> trả về raw text/label (chỉ để EDA).
        format: "torch" (mặc định) -> dataset trả về `torch.Tensor` sẵn, dùng được
            cả với `Trainer` lẫn vòng lặp PyTorch thuần.
            "python" -> trả về list (như `Dataset.from_dict` mặc định).

    Returns:
        datasets.DatasetDict
    """
    cfg = config or load_model_config(DEFAULT_CONFIG)
    tok = tokenizer or build_tokenizer(cfg)[0]
    max_len = int(cfg["tokenizer"]["max_length"])
    padding = cfg["tokenizer"].get("padding", "max_length")
    truncation = cfg["tokenizer"].get("truncation", True)

    paths = cfg["paths"]
    wanted = ["train", "val"] + (["test"] if include_test else [])

    out = {}
    for split in wanted:
        df = load_split(paths[f"{split}_file"])
        if tokenize:
            cols = tokenize_dataframe(
                df, tok, max_length=max_len, padding=padding, truncation=truncation
            )
        else:
            cols = {"text": df["text"].tolist(), "label": df["label"].tolist()}
        ds = Dataset.from_dict(cols)
        # Quan trọng: định dạng torch để truyền thẳng vào DataLoader / model(**batch)
        # mà không cần tự chuyển đổi (Trainer cũng nhận format này).
        if format == "torch" and tokenize:
            ds.set_format(type="torch", columns=list(cols.keys()))
        out[split] = ds

    return DatasetDict(out)
