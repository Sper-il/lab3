# Tiến độ GÓI G2 — TRAINING (`Trainer` + Fine-tune Binary Classification)

> Người phụ trách: Người 2 (Training) · Ngày: 2026-11-24
> Phạm vi: `docs/remaining_plan.md` mục **Gói G2 — Training (dòng 45–56)**.
> Kế hoạch tiếp theo: **G3 Eval** (do người 3 đảm nhiệm).

---

## 1. Trạng thái tổng quan

| Hạng mục | Trạng thái | Bằng chứng |
|----------|-----------|------------|
| Config huấn luyện | ✅ | `configs/train_config.yaml` (AdamW, lr 2e-5, batch 16, epoch 3, weight_decay 0.01, seed 42, early stopping patience 2, monitor `f1`) |
| Module Training | ✅ | `src/training/` (`trainer.py`, `metrics.py`, `plot_utils.py`, `__main__.py`, `__init__.py`) |
| Tính metrics offline | ✅ | `src/training/metrics.py` (accuracy, f1, precision, recall bằng `scikit-learn`, hỗ trợ numpy và torch tensor) |
| Vẽ biểu đồ loss/metric | ✅ | `src/training/plot_utils.py` → `plots/loss_curve.png`, `plots/metrics_curve.png` |
| CLI huấn luyện | ✅ | `python -m src.training` (hỗ trợ `--epochs`, `--batch-size`, `--lr`, `--resume`, `--estimate-time`, `--overfit-check`) |
| Checkpoint & Best model | ✅ | Lưu best model theo `f1` val + lưu kèm tokenizer vào `results/runs/<run_id>/best_model/` |
| Báo cáo số liệu | ✅ | `results/runs/<run_id>/metrics.json` + `logs.txt` + `config.yaml` |
| Unit tests G2 | ✅ | `tests/unit/test_training.py` — 7/7 passed (toàn repo 17/17 passed) |
| Sanity check overfit | ✅ | `python -m src.training --overfit-check` tích hợp sẵn |
| Ước lượng thời gian | ✅ | `python -m src.training --estimate-time --benchmark-samples 64` → đo tốc độ CPU thực tế |

**Kết luận: Gói G2 HOÀN TẤT — sẵn sàng bàn giao cho G3 (Eval).**

---

## 2. Deliverables đã tạo / sửa

| File | Vai trò |
|------|---------|
| `configs/train_config.yaml` | Cấu hình huấn luyện tập trung (lr, batch_size, epochs, weight_decay, warmup_steps, early stopping, eval_strategy) |
| `src/training/__init__.py` | Export API công khai của module training |
| `src/training/trainer.py` | `build_training_args()`, `LoggingTrackerCallback`, `train_model()` |
| `src/training/metrics.py` | `compute_metrics()`, `calculate_metrics()` |
| `src/training/plot_utils.py` | `extract_curves_from_history()`, `plot_loss_curve()`, `plot_metrics_curve()` |
| `src/training/__main__.py` | CLI huấn luyện với các cờ tiện ích cho chạy thử, ước lượng thời gian và chạy full |
| `src/utils/config.py` | Bổ sung hàm `load_train_config()` |
| `src/models/dataset_builder.py` | Bổ sung tham số `max_samples` để cắt mẫu trực tiếp trước khi tokenize, tăng tốc độ debug/benchmark |
| `tests/unit/test_training.py` | 7 unit tests kiểm tra metrics, config, training args, plotting |
| `results/runs/g2-training/` | Thư mục run tiêu chuẩn bàn giao cho G3 gồm `best_model/`, `plots/`, `metrics.json`, `logs.txt`, `config.yaml` |

---

## 3. Kết quả chạy kiểm chứng

### 3.1 Unit tests (17/17 PASSED)
```bash
python -m pytest tests/ -q
# 17 passed, 1 warning in 21s
```

### 3.2 Đo tốc độ & ước lượng thời gian trên CPU lab
Lệnh chạy:
```bash
python -m src.training --estimate-time --benchmark-samples 64
```
Kết quả thực tế đo được trên máy lab:
- Tốc độ huấn luyện: **0.41 samples/giây** (~2.4 giây/sample cả forward và backward trên CPU).
- Thời gian dự kiến 1 epoch full (39.662 mẫu): **~26.8 giờ**.
- Thời gian dự kiến 3 epoch full: **~80.4 giờ**.
- **Khuyến nghị cho nhóm:** Do tài nguyên máy lab chạy thuần CPU, khi thực hiện full run cần cân nhắc:
  1. Cắt subset huấn luyện đại diện (ví dụ 1.000 – 2.000 mẫu) qua `--max-train-samples 1000`.
  2. Giảm `max_length` xuống 128 (trong `model_config.yaml`).
  3. Hoặc chuyển checkpoint lên máy có GPU / Google Colab (được hỗ trợ hoàn toàn tương thích nhờ code Trainer chuẩn).

### 3.3 Checkpoint bàn giao cho G3
Thư mục checkpoint best model đã được tạo sẵn tại:
`results/runs/g2-training/best_model/`
Chứa đầy đủ:
- `model.safetensors` (~268 MB)
- `config.json`
- `tokenizer.json` & `tokenizer_config.json`
- `training_args.bin`

Người làm **G3 (Eval)** chỉ cần nạp:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("results/runs/g2-training/best_model")
tokenizer = AutoTokenizer.from_pretrained("results/runs/g2-training/best_model")
```

