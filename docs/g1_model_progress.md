# Tiến độ GÓI G1 — MODEL (Tokenizer + Backbone + Loss)

> Người phụ trách: Người 1 (Model) · Ngày: 2026-11-23
> Phạm vi: `docs/remaining_plan.md` mục **Gói G1 — Model (dòng 24–34)**.
> Kế hoạch tiếp theo: **G2 Training** (do người 2 đảm nhiệm).

---

## 1. Trạng thái tổng quan

| Hạng mục | Trạng thái | Bằng chứng |
|----------|-----------|------------|
| Config model | ✅ | `configs/model_config.yaml` |
| `build_tokenizer()` | ✅ | `src/models/tokenizer_utils.py` |
| `tokenize_batch()` → `(B, 256)` | ✅ | check in: `input_ids (4, 256)`, `attention_mask (4, 256)` |
| `build_model(config)` | ✅ | `src/models/architecture.py` → `logits (4, 2)` |
| `build_loss()` = CrossEntropy | ✅ | `src/models/loss.py` |
| Params / dung lượng | ✅ | 66,955,010 params (~67.0M) · ~255 MB float32 |
| Notebook kiểm tra | ✅ | `notebooks/02_tokenize_check.ipynb` (15 cells) |
| Script kiểm tra CLI | ✅ | `python -m src.models.check` |
| Dataset tokenize sẵn cho G2 | ✅ | `src/models/dataset_builder.py` → `DatasetDict` train/val/test (format `torch`) |
| ⭐ Smoke test Trainer (tương thích G2) | ✅ | `src/models/smoke_train.py` → PASS 5/5 checks |
| ⭐⭐ Overfit check (model học được) | ✅ | `src/models/overfit_check.py` → loss 0.7017 → 0.0027, acc 1.0000 |
| Unit tests | ✅ | `tests/unit/test_models.py` — 7 passed |
| Metrics JSON | ✅ | `results/runs/g1-model/metrics.json` + `eval_quick.json` + `smoke_train.json` + `overfit_check.json` |

**Kết luận: Gói G1 HOÀN TẤT — sẵn sàng bàn giao cho G2 (Training).**

> 🟢 **Đã chứng minh bằng chạy thật, không phải suy đoán:**
> - `Trainer` của Hugging Face chạy được với đúng output của G1 (3 step, PASS 5/5 checks)
> - Model **thực sự học được**: overfit 8 mẫu → loss 0.7017 → 0.0027, accuracy 1.0000
> - Gradient ổn định (`grad_norm` 1.7 → 3.3), không NaN
> → **Người làm G2 không cần sửa code G1.**

---

## 2. Files đã tạo / sửa

### Tạo mới
| File | Vai trò |
|------|---------|
| `configs/model_config.yaml` | Single source of truth: backbone, num_labels, max_length, loss, paths splits, seed, device |
| `src/models/__init__.py` | Export API công khai của module models |
| `src/models/tokenizer_utils.py` | `build_tokenizer()`, `tokenize_batch()`, `tokenize_dataframe()` |
| `src/models/architecture.py` | `build_model()`, `count_parameters()`, `model_size_mb()`, `forward_check()` |
| `src/models/loss.py` | `build_loss()` (CrossEntropy), `compute_loss()` |
| `src/models/dataset_builder.py` | `build_datasets()` → `DatasetDict` tokenize sẵn cho Trainer (format `torch`) |
| `src/models/check.py` | CLI kiểm tra G1 (`python -m src.models.check`) |
| `src/models/eval_quick.py` | CLI đánh giá nhanh trên dữ liệu thật |
| `src/models/smoke_train.py` | ⭐ Smoke test `Trainer` — chứng minh tương thích G2 |
| `src/models/overfit_check.py` | ⭐⭐ Overfit 1 batch — chứng minh model học được |
| `src/utils/config.py` | `load_config()`, `load_model_config()`, `repo_path()` — đọc YAML, resolve path |
| `src/utils/__init__.py` | Export utils |
| `notebooks/02_tokenize_check.ipynb` | Notebook kiểm tra shape (2 câu mẫu Ex1 + dự đoán) |
| `tests/unit/test_models.py` | 7 unit test (offline, không cần mạng) |
| `results/runs/g1-model/metrics.json` | Số liệu G1 để truy vết |
| `results/runs/g1-model/eval_quick.json` | Eval 500 mẫu val thật |
| `results/runs/g1-model/smoke_train.json` | Kết quả smoke test Trainer |
| `results/runs/g1-model/overfit_check.json` | Lịch sử loss overfit |
| `requirements.txt` | Pin version môi trường |
| `docs/AUDIT_LOG.md` | Nhật ký làm việc chi tiết |
| `docs/RUN_GUIDE.md` | Hướng dẫn cài đặt / chạy / quy trình |

### Sửa
| File | Thay đổi |
|------|----------|
| `.gitignore` | Bổ sung path thực tế `data/data/**` (repo đang để dữ liệu ở `data/data/`), bỏ qua `*.safetensors` + `results/runs/**/checkpoint*/` |

---

## 3. Kiểm chứng thực tế (đã chạy, không giả lập)

Lệnh: `python -m src.models.check`

```
Python        : 3.11.9 | torch 2.12.0+cpu
Device        : cpu
max_length    : 256   |   batch_size : 4

[1] Tokenizer  : distilbert-base-uncased
    vocab_size : 30522 | model_max : 512
    sample tok : ['i','loved','this','movie',',','it','was','fantastic','!']

[2] Model      : distilbert-base-uncased
    params     : 66,955,010 (trainable 66,955,010)
    ~67.0M params | ~255 MB (float32)
    num_labels : 2 | problem_type=single_label_classification

[3] Data       : data/data/splits/train.csv (39,662 dòng)
    input_ids  : (4, 256)  dtype=torch.int64
    attn_mask  : (4, 256)
    token_type : (4, 256)

[4] Forward    : logits (4, 2)   (kỳ vọng (4, 2))
    thời gian  : ~0.50s | ~126 ms/sample (CPU, batch 4)

[5] Loss       : CrossEntropyLoss = 0.7036
    softmax    : (4, 2), mỗi hàng tổng = 1.0000
```

### Đối chiếu Definition of Done của G1
| DoD trong kế hoạch | Kết quả |
|---------------------|---------|
| 1 batch `tokenize` đúng `(B,256)` + `attention_mask` | ✅ `(4, 256)` cả hai |
| Forward ra `(B,2)` | ✅ `(4, 2)` |
| Ước lượng params (~66M) | ✅ 66.96M |
| Ước lượng dung lượng (~260MB) | ✅ ~255 MB float32 |
| Decisions Log ghi lý do chọn `distilbert` thay vì `bert-base` | ✅ xem mục 5 |
| `notebooks/02_tokenize_check.ipynb` (in shape 1 batch, 2 câu mẫu Ex1) | ✅ 15 cells |
| Rủi ro RAM → `torch.float32` CPU, batch nhỏ | ✅ dtype=float32, check_batch_size=4 |

---

## 4. Quyết định thiết kế & lý do

| # | Quyết định | Lựa chọn khác | Lý do |
|---|------------|---------------|-------|
| 1 | Chọn `distilbert-base-uncased` làm backbone | `bert-base-uncased` | Nhẹ hơn ~40% (67M vs 110M), train CPU trong phòng lab khả thi, vẫn ~97% hiệu năng BERT trên IMDB |
| 2 | `max_length = 256`, `padding="max_length"` | `padding="longest"` (biến thiên) | Shape cố định `(B, 256)` đúng Design → kiểm tra shape đơn giản, Trainer không phải xử lý padding động |
| 3 | `dtype = float32` | `float16`/`bfloat16` | CPU máy lab không có tensor core; float32 ổn định, tránh NaN, đúng ghi chú rủi ro trong kế hoạch |
| 4 | `problem_type = single_label_classification` | mặc định `None` | HF tự tính CrossEntropy khi forward → tương thích trực tiếp với `build_loss()` |
| 5 | Fallback `bert-base-uncased` khi Hub lỗi | báo lỗi luôn | An toàn cho môi trường offline có cache |
| 6 | Tách `dataset_builder.build_datasets()` | để người G2 tự tokenize | Người 2 không phải viết lại logic tokenize; `DatasetDict` sẵn sàng đưa vào `Trainer` |
| 7 | `src/utils/config.py` đọc YAML tập trung | hardcode trong từng file | Config là single source of truth, người G2 chỉ cần sửa YAML |

---

## 5. ⭐ Chứng minh model sẵn sàng cho G2 (chạy thật, 2026-11-23 13:22–13:27)

Câu hỏi: *"Model đã ổn để người sau train chưa?"* → Đã kiểm chứng bằng 2 script mới.

### 5.1 Smoke test `Trainer` — `src/models/smoke_train.py`
Chạy `Trainer` thật 3 step với đúng output của G1 (`build_datasets()` + `build_model()`):
```
{'loss': '0.69', 'grad_norm': '1.706', 'learning_rate': '2e-05', 'epoch': '0.0625'}
{'loss': '0.6774', 'grad_norm': '2.743', 'learning_rate': '1.333e-05', 'epoch': '0.125'}
{'loss': '0.7312', 'grad_norm': '3.275', 'learning_rate': '6.667e-06', 'epoch': '0.1875'}
{'eval_loss': '0.6928', 'eval_accuracy': '0.5312', 'eval_f1': '0.2857', ...}
{'train_runtime': '11.21', 'train_samples_per_second': '1.07', 'train_loss': '0.6995'}
predictions shape: (32, 2) | preds shape: (32,) | labels shape: (32,)
    [PASS] dataset có train/val
    [PASS] model build được
    [PASS] train chạy, loss hữu hện
    [PASS] evaluate trả metric
    [PASS] predict shape khớp (B,2)
```
→ **G2 không cần sửa code G1.**

### 5.2 Overfit 1 batch — `src/models/overfit_check.py`
Sanity check bắt buộc trước khi train full (theo `remaining_plan.md` §G2):
```
  epoch   0 | loss = 0.7017
  epoch  10 | loss = 0.1328
  epoch  20 | loss = 0.0139
  epoch  30 | loss = 0.0039
  epoch  39 | loss = 0.0027

Loss đầu : 0.7017
Loss cuối: 0.0027
Accuracy trên batch đó: 1.0000
Kết luận  : model CÓ THỂ học
```
→ Backbone + head + tokenizer + loss + gradient **đều đúng**. Nếu không overfit được
8 mẫu thì chắc chắn không train được 39.662 mẫu.

### 5.3 🐛 Bug phát hiện & đã sửa trong lúc kiểm chứng

| Bug | Triệu chứng | Nguyên nhân | Cách sửa |
|-----|-------------|-------------|----------|
| 1 | `TypeError: embedding(): argument 'indices' must be Tensor, not list` | `build_datasets()` trả **list** (mặc định `Dataset.from_dict`). `Trainer` tự xử lý được, nhưng PyTorch thuần thì không | Thêm `ds.set_format(type="torch", columns=...)` trong `dataset_builder.py` → dataset trả tensor sẵn, dùng được cho **cả** `Trainer` lẫn `DataLoader` |
| 2 | `RuntimeError: size of tensor a (256) must match b (512)` | Format torch → `ds[0]` trả tensor **1D** (256,), model cần batch dim | `unsqueeze(0)` khi lấy 1 mẫu để eval |

> Bug 1 đáng lưu ý: nếu không sửa, người làm G2 dùng `build_datasets()` rồi tự viết
> vòng lặp PyTorch sẽ gặp lỗi ngay. Sau khi sửa, cả 2 đường đều dùng được.

### 5.4 Số liệu đo được để G2 ước lượng thời gian
| Phép đo | Kết quả |
|---------|---------|
| Inference (forward thuần) | ~135 ms/sample |
| **Train thật** (Trainer, batch 4) | **1.07 samples/sec** (~935 ms/sample) |
| Overfit 8 mẫu × 40 epoch | 278.5s (~7s/epoch) |

→ Ước lượng **~10 h/epoch** full fine-tune trên máy này (39.662 ÷ 1.07).
→ **Khuyến nghị:** G2 nên (1) chạy 5% data để đo thật trên máy của bạn, (2) cân nhắc
`max_length: 128` nếu cần tăng tốc, (3) `gradient_checkpointing=True` nếu thiếu RAM.

---

## 7. Cấu trúc dữ liệu đầu ra cho người làm G2

`build_datasets()` trả về:

```python
DatasetDict({
    train: Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows: 39662}),
    val:   Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows:  4958}),
    test:  Dataset({features: ['input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows:  4958}),
})
```

- `input_ids` / `attention_mask` / `token_type_ids`: `List(int)` độ dài 256
- `labels`: `int64` ∈ {0, 1} → 0 = NEGATIVE, 1 = POSITIVE
- ⚠️ **Đã set format `torch`**: `ds[i]["input_ids"]` trả `torch.Tensor` sẵn.
  Dùng được cho **cả** Hugging Face `Trainer` (khuyến nghị) **và** PyTorch `DataLoader` thuần.
  Nếu muốn list python: `build_datasets(format="python")`.

> `token_type_ids` có thể bị Trainer bỏ qua — DistilBERT không dùng nó (đã đúng chuẩn; `bert-base` mới cần).

---

## 8. Gợi ý cho người làm G2 (Training)

```python
# 1) dataset
from src.models.dataset_builder import build_datasets
ds = build_datasets()          # đã tokenize sẵn, KHÔNG cần map() lại

# 2) model + loss
from src.models.architecture import build_model
from src.models.loss import build_loss
model, name = build_model()     # trả về (model, tên backbone)
loss_fn = build_loss()          # CrossEntropyLoss

# 3) Trainer — HF tự dùng CrossEntropy khi problem_type='single_label_classification'
from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    output_dir="results/runs/<timestamp>",
    learning_rate=2e-5, per_device_train_batch_size=16,
    per_device_eval_batch_size=32, num_train_epochs=3,
    weight_decay=0.01, seed=42, logging_steps=50,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="f1",
    greater_is_better=True, save_total_limit=2,
)
trainer = Trainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["val"],
    compute_metrics=...,   # accuracy + f1
)
trainer.train()
```

Lưu ý quan trọng:
- `transformers` **5.24.0** → tham số là `eval_strategy` (không phải `evaluation_strategy`).
- `datasets` **5.2.0** → `datasets` đã hợp nhất các `load_dataset`; `load_metric` không còn, dùng `evaluate.load("accuracy")`.
- `test` vẫn nên **khóa** tới G3 mới dùng (theo quy tắc trong `remaining_plan.md`).
- `compute_metrics` dùng `sklearn` (đã test chạy được, không cần network) — xem mẫu ở `src/models/smoke_train.py`.
- Mẫu Trainer ở trên **đã được verify chạy thật** → xem `src/models/smoke_train.py` + `results/runs/g1-model/smoke_train.json`.

---

## 9. Lệnh kiểm tra lại (cho người review)

```bash
# 1) Check shape end-to-end (cần mạng/cache để tải backbone lần đầu)
python -m src.models.check

# 2) Lưu metrics + xuất JSON
python -m src.models.check --out results/runs/g1-model/metrics.json

# 3) Eval nhanh trên dữ liệu thật
python -m src.models.eval_quick --n 500 --split val

# 4) ⭐ Smoke test Trainer (tương thích G2)
python -m src.models.smoke_train

# 5) ⭐⭐ Overfit check (chứng minh model học được)
python -m src.models.overfit_check --n 8 --epochs 40

# 6) Unit tests (offline, ~3s)
python -m pytest tests/unit/test_models.py -q

# 7) Toàn bộ tests
python -m pytest tests/ -q

# 8) Mở notebook
jupyter lab notebooks/02_tokenize_check.ipynb
```

Kết quả mong đợi:
- `TẤT CẢ CHECK PASS — 66.96M params, logits (4, 2)`
- `PASS — G1 sẵn sàng cho G2 (Training)`
- `model CÓ THỂ học (loss 0.7017 → 0.0027)`
- `10 passed`

---

## 10. Rủi ro còn lại & cách xử lý

| Rủi ro | Mức độ | Xử lý đã chuẩn bị |
|--------|--------|---------------------|
| Hub lỗi mạng khi tải backbone | Trung bình | Có fallback `bert-base-uncased`; ghi rõ lỗi trong exception; dùng `HF_HUB_OFFLINE=1` nếu có cache |
| RAM khi load model | Thấp | `float32` CPU, `check_batch_size=4`; người G2 nên dùng batch 16 và `gradient_checkpointing` nếu máy yếu |
| **Train CPU lâu** | 🔴 Cao | **Đo thật ở G1: Trainer chạy 1.07 samples/sec ⇒ ~10 h/epoch full fine-tune.** G2 nên: (1) chạy 5% data để đo thật, (2) cân nhắc `max_length: 128`, (3) `gradient_checkpointing=True` |
| Head random init nên dự đoán ban đầu gần ngẫu nhiên | Bình thường | Đúng kỳ vọng pre-fine-tune. Log report đã ghi rõ; sau G2 mới có dự đoán có nghĩa |
| `data/data/splits` (thư mục lồng) lệch với Design `data/splits` | Thấp | `configs/model_config.yaml` có `paths.splits_dir` — sửa 1 dòng nếu ai đó di dời thư mục |

---

## 11. Checklist bàn giao

- [x] `configs/model_config.yaml` tồn tại, đầy đủ tham số
- [x] `build_model()` / `build_tokenizer()` / `build_loss()` / `tokenize_batch()` public API
- [x] Chạy được trên 1 batch thật từ `data/data/splits/train.csv`
- [x] Shape khớp Design: `(B, 256)` → `(B, 2)`
- [x] Params + dung lượng đã ước lượng
- [x] Eval trên dữ liệu thật (500 mẫu val) — pipeline end-to-end OK
- [x] ⭐ Smoke test `Trainer` PASS 5/5 — tương thích G2
- [x] ⭐⭐ Overfit check PASS — model thực sự học được (loss → 0.0027, acc 1.0)
- [x] Dataset builder trả format `torch` — dùng được cho cả Trainer lẫn DataLoader
- [x] Notebook kiểm tra tồn tại, đã execute thật
- [x] Unit tests pass (7/7), toàn repo 10/10
- [x] Metrics JSON lưu trong `results/runs/g1-model/` (4 file)
- [x] Decisions Log cập nhật
- [x] `.gitignore` cập nhật
- [x] `docs/g1_model_progress.md` (báo cáo)
- [x] `docs/AUDIT_LOG.md` (nhật ký)
- [x] `docs/RUN_GUIDE.md` (hướng dẫn chạy)
- [ ] Commit (Conventional Commit) — chưa commit, chờ người review
