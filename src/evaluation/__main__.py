import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import (
    LatencyTimer,
    compute_classification_metrics,
    extract_error_cases,
)
from src.models.architecture import build_model
from src.models.tokenizer_utils import build_tokenizer, tokenize_batch


class EvalTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            return
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            return
    model.load_state_dict(state)


def plot_confusion_matrix(cm: list[list[int]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm),
        display_labels=["NEGATIVE (0)", "POSITIVE (1)"],
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def run_evaluation(config_path: str, checkpoint_path_arg: str | None) -> Path:
    cfg = load_yaml(config_path)
    device = torch.device(cfg.get("model", {}).get("device", "cpu"))

    ckpt_str = checkpoint_path_arg or cfg.get("checkpoint", {}).get("path")
    if not ckpt_str:
        raise ValueError("Cần truyền --checkpoint hoặc khai báo checkpoint.path trong eval_config.yaml.")
    ckpt_path = Path(ckpt_str)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    # Nếu checkpoint nằm trong results/runs/<ts>/..., lưu output ngay tại run_dir đó
    run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "checkpoints" else ckpt_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["test_path"])
    texts = df[data_cfg["text_col"]].fillna("").astype(str).tolist()
    labels = df[data_cfg["label_col"]].astype(int).to_numpy()

    model_cfg = cfg["model"]
    tokenizer = build_tokenizer(model_cfg["model_name"])
    model = build_model(model_cfg).to(device)
    load_checkpoint_weights(model, ckpt_path, device)
    model.eval()

    dataset = EvalTextDataset(texts, labels)

    def collate_fn(batch: list[Tuple[str, int]]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_texts = [item[0] for item in batch]
        batch_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        encoded = tokenize_batch(
            tokenizer,
            batch_texts,
            max_length=int(model_cfg.get("max_length", 256)),
        )
        return encoded, batch_labels

    loader = DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    timer = LatencyTimer()

    with torch.inference_mode():
        for encoded, _ in loader:
            inputs = {k: v.to(device) for k, v in encoded.items()}
            start = time.perf_counter()
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            probs = torch.softmax(logits, dim=-1)
            elapsed = time.perf_counter() - start

            batch_size = next(iter(inputs.values())).shape[0]
            timer.record_batch(elapsed, batch_size)

            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_probs.append(probs.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_probs = np.concatenate(all_probs, axis=0)

    targets = cfg.get("eval", {}).get("targets", {"accuracy": 0.85, "f1": 0.85})
    metrics_data = compute_classification_metrics(labels, y_pred, targets=targets)
    latency_data = timer.report()

    top_k = int(cfg.get("eval", {}).get("error_analysis_top_k", 5))
    error_cases = extract_error_cases(texts, labels, y_pred, y_probs, top_k=top_k)

    cm_rel_path = Path(cfg.get("output", {}).get("plots_dir", "plots")) / "confusion_matrix.png"
    plot_confusion_matrix(metrics_data["confusion_matrix"], run_dir / cm_rel_path)

    metrics_payload = {
        "run_id": run_dir.name,
        "checkpoint": str(ckpt_path),
        "split": "test",
        "metrics": metrics_data["summary"],
        "per_class": metrics_data["per_class"],
        "confusion_matrix": metrics_data["confusion_matrix"],
        "latency": latency_data,
        "error_analysis_top_k": error_cases,
        "artifacts": {
            "confusion_matrix_plot": str(cm_rel_path),
        },
    }

    metrics_filename = cfg.get("output", {}).get("metrics_filename", "metrics.json")
    metrics_path = run_dir / metrics_filename
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    errors_filename = cfg.get("output", {}).get("errors_filename", "error_cases.json")
    with open(run_dir / errors_filename, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)

    print("=== G3 EVALUATION SUMMARY (TEST SET) ===")
    for item in metrics_data["summary"]:
        print(
            f"- {item['metric']}: {item['value']} "
            f"(target: {item['target']}, passed: {item['passed']})"
        )
    print(f"- Latency: {latency_data['avg_latency_ms_per_sample']} ms/sample "
          f"({latency_data['throughput_samples_per_sec']} samples/s)")
    print(f"- Saved metrics to: {metrics_path}")
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test split (G3)")
    parser.add_argument("--config", required=True, help="Path to eval_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to trained checkpoint (.pt/.bin)")
    args = parser.parse_args()
    run_evaluation(args.config, args.checkpoint)


if __name__ == "__main__":
    main()