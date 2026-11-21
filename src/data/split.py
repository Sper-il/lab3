"""Chia stratified train/val/test cố định bằng seed."""

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(df: pd.DataFrame, ratios=(0.8, 0.1, 0.1), seed=42):
    """Chia stratified, tránh leakage (tách sau khi clean, trước tokenize)."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "Tỉ lệ split phải cộng lại bằng 1"
    train_ratio, val_ratio, _ = ratios
    train_df, temp_df = train_test_split(
        df, test_size=1 - train_ratio, random_state=seed, stratify=df["label"]
    )
    val_rel = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - val_rel, random_state=seed, stratify=temp_df["label"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
