"""Load dữ liệu thô IMDb từ Hugging Face Hub."""

from datasets import load_dataset


def load_raw(dataset_name="stanfordnlp/imdb", fallback="imdb"):
    """Tải split train/test gốc, gộp lại để EDA + split lại stratified.

    Trả về datasets.DatasetDict với cột text/label chuẩn hoá.
    """
    last_err = None
    for name in [dataset_name, fallback]:
        try:
            ds = load_dataset(name)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            ds = None
    if ds is None:
        raise RuntimeError(f"Không tải được IMDb từ Hub: {last_err}")

    # Chuẩn hoá tên cột về text/label
    def _norm(split):
        cols = split.column_names
        text_col = "text" if "text" in cols else cols[0]
        label_col = "label" if "label" in cols else cols[-1]
        keep = split.select_columns([text_col, label_col])
        return keep.rename_columns({text_col: "text", label_col: "label"})

    import datasets as _ds

    # Chỉ dùng split có nhãn (train/test), bỏ unsupervised (nhãn -1)
    labeled_keys = [k for k in ds.keys() if k in ("train", "test", "validation")]
    if not labeled_keys:  # fallback: lọc nhãn hợp lệ
        full_all = _ds.concatenate_datasets([_norm(ds[k]) for k in ds.keys()])
        full_all = full_all.filter(lambda x: x["label"] in (0, 1))
        return full_all

    parts = [_norm(ds[k]) for k in labeled_keys]
    full = _ds.concatenate_datasets(parts)
    # Phòng thủ: chỉ giữ nhãn nhị phân 0/1
    full = full.filter(lambda x: x["label"] in (0, 1))
    return full
