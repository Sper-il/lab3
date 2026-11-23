# HƯỚNG DẪN CHẠY — Dành cho thành viên tiếp theo

> Đọc file này là đủ để **cài đặt, chạy thử, hiểu quy trình** mà không cần đào lại code.
> Mục tiêu hiện tại: gói **G1 (Model)** đã xong. Gói tiếp theo: **G2 (Training)**.

---

## 0. Tóm tắt 30 giây

```bash
# 1. Cài thư viện
python -m pip install -r requirements.txt

# 2. Kiểm tra Model (tokenize + forward shape)
python -m src.models.check

# 3. Test
python -m pytest tests/ -q
```

Kết quả mong đợi: `TẤT CẢ CHECK PASS — 66.96M params, logits (4, 2), loss 0.7036`

---

## 0b. 🟢 Câu trả lời ngắn: "Model đã ổn để train chưa?"

**CÓ.** Đã kiểm chứng bằng 2 lần chạy thật:

| Kiểm chứng | Lệnh | Kết quả |
|-------------|------|---------|
| `Trainer` tương thích với code G1 | `python -m src.models.smoke_train` | ✅ PASS 5/5 |
| Model **thực sự học được** | `python -m src.models.overfit_check` | ✅ loss 0.7017 → **0.0027**, acc **1.0000** |
| Forward shape | `python -m src.models.check` | ✅ `(B,256)` → `(B,2)` |
| Tests | `python -m pytest tests/ -q` | ✅ 10/10 |

→ Người tiếp theo (G2) **không cần sửa code G1**, chỉ cần dựng `TrainingArguments` + `Trainer` và chạy.
→ Mẫu Trainer đầy đủ ở mục 6.2.

---

## 1. Bối cảnh: project này làm gì?

| | |
|---|---|
| **Bài toán** | Phân loại cảm xúc văn bản nhị phân (POSITIVE / NEGATIVE) trên tập IMDb |
| **Đề bài** | `đe` — Exercise 1 (sentiment pipeline) + Exercise 2 (fine-tune binary classification) |
| **Backbone** | `distilbert-base-uncased` (66.96M params) |
| **Input** | Văn bản review phim tiếng Anh, `max_length = 256` token |
| **Output** | `logits` shape `(B, 2)` → softmax → `{label: POSITIVE|NEGATIVE, score: 0.xx}` |
| **Nhãn** | `0 = NEGATIVE`, `1 = POSITIVE` |

### Quy trình tổng thể (6 gói)

```
[G0] Data: raw → clean → splits        ✅ XONG (39.662 / 4.958 / 4.958)
  ↓
[G1] Model: tokenize → backbone → loss ✅ XONG  ← BẠN ĐANG Ở ĐÂY
  ↓
[G2] Training: Trainer + fine-tune      ⬜ TIẾP THEO
  ↓
[G3] Eval: accuracy/F1 + confusion      ⬜
  ↓
[G4] Visualize: report.html              ⬜
  ↓
[G5] Demo: 1 lệnh                       ⬜
  ↓
[G6] Document: README + AI_Usage        ⬜
```

---

## 2. Cài đặt môi trường

### Yêu cầu
- **Python** ≥ 3.10 (đã test với 3.11.9)
- **RAM** ≥ 8 GB (khuyến nghị 16 GB cho G2)
- **Ổ cứng** ≥ 3 GB trống (model weights ~255 MB)
- Kết nối mạng **lần đầu** để tải backbone từ Hugging Face Hub

### Cài đặt

```bash
# Windows (PowerShell)
cd C:\Users\hoait\Downloads\lab3
python -m pip install -r requirements.txt
```

Các gói chính và vai trò:

| Gói | Version | Vai trò |
|-----|---------|---------|
| `torch` | 2.12.0 (CPU) | Tensor, forward/backward |
| `transformers` | 5.24.0 | Tokenizer, Model, Trainer |
| `datasets` | 5.2.0 | DatasetDict cho Trainer |
| `evaluate` | 0.4.6 | Metric accuracy/F1 cho G3 |
| `accelerate` | 1.16.0 | Backend train của Trainer |
| `scikit-learn` | 1.9.1 | Split + metric (G0, G3) |
| `pandas` / `numpy` | 2.3.3 / 2.3.5 | Xử lý dữ liệu |
| `PyYAML` | 6.0.3 | Đọc config |
| `pytest` | ≥ 8.0 | Unit test |

> ⚠️ **Lưu ý version quan trọng:** `transformers` **5.x** đã đổi API so với 4.x. Xem mục 8.

### Kiểm tra môi trường
```bash
python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
```
Kỳ vọng: `2.12.0+cpu 5.24.0 5.2.0`

---

## 3. Cấu trúc thư mục (sau khi G1)

```
lab3/
├── configs/
│   ├── data_config.yaml       # config cho G0 (EDA + split)
│   └── model_config.yaml      # config cho G1 (model)  ← MỚI
├── data/
│   └── data/                  # ⚠️ data thực tế nằm lồng 1 cấp
│       ├── raw/imdb_raw_sample.csv
│       ├── processed/clean_imdb.csv
│       └── splits/{train,val,test}.csv
├── docs/
│   ├── AUDIT_LOG.md           # nhật ký làm việc  ← MỚI
│   ├── RUN_GUIDE.md           # file này              ← MỚI
│   ├── g1_model_progress.md   # báo cáo G1            ← MỚI
│   ├── progress_status.md     # single source of truth
│   └── remaining_plan.md      # kế hoạch 6 gói
├── notebooks/
│   └── 02_tokenize_check.ipynb # kiểm tra shape        ← MỚI
├── results/runs/
│   ├── eda-baseline/metrics.json
│   └── g1-model/              # ← MỚI
│       ├── metrics.json
│       ├── eval_quick.json
│       ├── smoke_train.json
│       └── overfit_check.json
├── src/
│   ├── data/                  # G0 (đã xong)
│   ├── models/                # G1                   ← MỚI
│   │   ├── __init__.py
│   │   ├── architecture.py    # build_model()
│   │   ├── tokenizer_utils.py # build_tokenizer(), tokenize_batch()
│   │   ├── loss.py            # build_loss()
│   │   ├── dataset_builder.py # build_datasets() cho Trainer
│   │   ├── check.py           # CLI kiểm tra
│   │   ├── eval_quick.py      # CLI eval nhanh
│   │   ├── smoke_train.py     # ⭐ smoke test Trainer  ← MỚI
│   │   └── overfit_check.py   # ⭐⭐ chứng minh học được ← MỚI
│   ├── utils/                 # ← MỚI
│   │   └── config.py          # load_model_config()
│   └── ...
├── tests/unit/
│   ├── test_data.py
│   └── test_models.py         # ← MỚI
├── requirements.txt           # ← MỚI
└── đe, đề.txt
```

---

## 4. Các lệnh chạy (copy-paste được)

### 4.1 Kiểm tra Model — **lệnh quan trọng nhất**

```bash
python -m src.models.check
```

In ra:
```
[1] Tokenizer  : distilbert-base-uncased
[2] Model      : distilbert-base-uncased
    params     : 66,955,010 (trainable 66,955,010)
    ~67.0M params | ~255 MB (float32)
[3] Data       : ...\data\data\splits\train.csv (tổng 39,662 dòng)
    input_ids  : (4, 256)  dtype=torch.int64
    attn_mask  : (4, 256)
[4] Forward    : logits (4, 2)  (kỳ vọng (4, 2))
[5] Loss       : CrossEntropyLoss = 0.7036

TẤT CẢ CHECK PASS
```

Tuỳ chọn:
```bash
python -m src.models.check --batch-size 8            # batch khác
python -m src.models.check --split val                # split khác
python -m src.models.check --out results/runs/g1-model/metrics.json
```

### 4.2 Đánh giá nhanh trên dữ liệu thật

```bash
python -m src.models.eval_quick --n 500 --split val
```

Kết quả (model **chưa** fine-tune):
```
Loss (avg)  : 0.6959
Accuracy    : 0.4280
F1          : 0.4704
Confusion   : [[TN=87, FP=155], [FN=131, TP=127]]
Thời gian    : 67.4s cho 500 mẫu = 134.8 ms/sample (CPU)
```

> **Đừng hoảng:** accuracy ~43-50% là **đúng** vì classification head random init.
> Lệnh này nhằm chứng minh pipeline chạy đúng trên dữ liệu thật, không phải đo chất lượng.
> Target ≥ 0.85 chỉ đạt được sau **G2 (Training)**.

### 4.3 ⭐ Chứng minh model học được (overfit 8 mẫu)

```bash
python -m src.models.overfit_check --n 8 --epochs 40
```

Kết quả thực tế:
```
  epoch   0 | loss = 0.7017
  epoch  20 | loss = 0.0139
  epoch  39 | loss = 0.0027
Loss đầu : 0.7017
Loss cuối: 0.0027
Accuracy trên batch đó: 1.0000

Kết luận  : model CÓ THỂ học (loss 0.7017 → 0.0027)
```

Đây là **sanity check bắt buộc** trước khi train full. Nếu model không overfit được
8 mẫu thì chắc chắn không train được 39.662 mẫu. Mất ~5 phút.

### 4.4 ⭐ Smoke test Trainer (kiểm tra tương thích với G2)

```bash
python -m src.models.smoke_train --max-steps 3 --n-train 64 --n-eval 32
```

Kết quả:
```
    [PASS] dataset có train/val
    [PASS] model build được
    [PASS] train chạy, loss hữu hạn
    [PASS] evaluate trả metric
    [PASS] predict shape khớp (B,2)

PASS — G1 sẵn sàng cho G2 (Training)
```

Chạy `Trainer` thật 3 step với đúng output của G1 → xác nhận G2 không cần sửa code G1.

### 4.5 Test

```bash
python -m pytest tests/ -q          # toàn bộ (10 test)
python -m pytest tests/unit/test_models.py -q   # riêng module models (7 test)
```

### 4.6 Notebook

```bash
pip install jupyterlab ipykernel
jupyter lab notebooks/02_tokenize_check.ipynb
```

Chạy từ trên xuống (`Shift+Enter`). Các cell đã được verify chạy được hết.
Muốn chạy không cần mở giao diện:
```bash
python -m nbconvert --to notebook --execute --output _out.ipynb notebooks/02_tokenize_check.ipynb
```

### 4.7 Dùng như thư viện (Python)

```python
from src.utils.config import load_model_config
from src.models.architecture import build_model, count_parameters, model_size_mb
from src.models.tokenizer_utils import build_tokenizer, tokenize_batch
from src.models.loss import build_loss
from src.models.dataset_builder import build_datasets

cfg = load_model_config("configs/model_config.yaml")

tokenizer, tok_name = build_tokenizer(cfg)
model, model_name = build_model(cfg)
model.to("cpu")

batch = tokenize_batch(
    ["I loved this movie!", "Terrible waste of time."],
    tokenizer, max_length=256, padding="max_length", truncation=True,
)
with torch.no_grad():
    logits = model(**batch).logits     # (2, 2)

print(count_parameters(model), model_size_mb(model))   # 66955010, 255.4

# Dataset sẵn cho Trainer (G2)
ds = build_datasets(config=cfg, tokenizer=tokenizer)
print(ds)   # DatasetDict(train/val/test)
```

---

## 5. Quy trình G1 đã làm (để hiểu *tại sao* code viết thế)

### Bước 1 — Config là single source of truth
`configs/model_config.yaml` chứa mọi tham số. Code **không** hardcode giá trị.
Đổi backbone / `max_length` chỉ cần sửa YAML, không sửa code.

### Bước 2 — Tokenizer → shape cố định
```
text (list[str])
  ↓ tokenizer(padding="max_length", truncation=True, max_length=256)
input_ids      (B, 256)   int64
attention_mask (B, 256)   int64   ← 0 = padding, 1 = token thật
token_type_ids (B, 256)   int64
```
- `padding="max_length"` (không phải `"longest"`) → shape luôn `(B, 256)` đúng Design, Trainer không phải xử lý padding động.
- `truncation=True` → review dài hơn 256 token không làm vỡ batch.

### Bước 3 — Model → logits
```
input_ids, attention_mask
  ↓ DistilBertModel (6 layer, 768 hidden)
  ↓ pre_classifier (768→768) + ReLU
  ↓ classifier (768→2)
logits (B, 2)
```
- `num_labels=2`, `problem_type="single_label_classification"` → HF tự dùng CrossEntropy khi forward.
- `classifier.*` báo `MISSING` khi nạp → **đúng**, đây là head random init, sẽ học ở G2.
- `vocab_transform.*` báo `UNEXPECTED` → **đúng**, phần MLM không dùng cho classification.

### Bước 4 — Loss
`CrossEntropyLoss(logits (B,2), labels (B,))` → scalar.
Tương thích trực tiếp với loss mặc định của HF `Trainer`.

### Bước 5 — Kiểm chứng
`check.py` chạy 6 nhóm assertion và **fail loud** nếu shape sai:
1. tokenizer load được
2. model load được, đếm params
3. đọc được batch thật từ `train.csv`
4. `input_ids.shape == (B, 256)` ✅
5. `logits.shape == (B, 2)`, không NaN/Inf ✅
6. loss tính được, softmax tổng = 1.0

---

## 6. Quy trình G2 (Training) — dành cho người tiếp theo

### 6.1 Dữ liệu đã tokenize sẵn
```python
from src.models.dataset_builder import build_datasets
ds = build_datasets()   # KHÔNG cần .map() lại
```
```
DatasetDict({
    train: Dataset({features: ['input_ids','token_type_ids','attention_mask','labels'], num_rows: 39662}),
    val:   Dataset({features: [...], num_rows: 4958}),
    test:  Dataset({features: [...], num_rows: 4958}),
})
```
- `input_ids`/`attention_mask`/`token_type_ids`: `List(int)` độ dài 256
- `labels`: `int64` ∈ {0, 1}
- ⚠️ **Format đã set sẵn `torch`**: `ds[0]["input_ids"]` trả về `torch.Tensor` 1D, `ds[i]` trả tensor.
  Dùng được cho **cả** `Trainer` (khuyến nghị) **và** PyTorch `DataLoader` thuần.
  Nếu muốn list python: `build_datasets(format="python")`.

### 6.2 Mẫu Trainer (tham khảo)
```python
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.architecture import build_model
from src.models.dataset_builder import build_datasets

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.softmax(torch.tensor(logits), dim=-1).argmax(-1).numpy()
    labels = labels.numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }

ds = build_datasets()
model, name = build_model()

args = TrainingArguments(
    output_dir="results/runs/<timestamp>",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    logging_steps=50,
    eval_strategy="epoch",          # ← transformers 5.x
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, metric_for_best_model="f1")],
)
trainer.train()
trainer.save_model("results/runs/<timestamp>/best_model")
```

### 6.3 Thứ tự chạy bắt buộc (theo `remaining_plan.md` §G2)
1. **Overfit 1 batch nhỏ** — sanity check ← **đã kiểm chứng ở G1: PASS (loss → 0.0027)**
2. **Train 1 epoch trên 5% train** — ước lượng thời gian
3. **Train full 3 epoch**

> Bước 1 đã làm sẵn và PASS. Bạn vẫn nên chạy lại khi chạy full để chắc chắn,
> nhưng có thể tin là phần model **đã học được** (xem `results/runs/g1-model/overfit_check.json`).

### 6.3.1 Mẫu `compute_metrics` đã test chạy được (trong `smoke_train.py`)
```python
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
```
→ Dùng `sklearn` (không cần network) thay vì `evaluate.load()`.

### 6.4 Ước lượng thời gian (CPU) — **đo thật ở G1**
| Phép đo | Kết quả |
|---------|---------|
| Inference (forward thuần) | ~135 ms/sample |
| **Train thật** (Trainer, batch 4) | **1.07 samples/sec** = ~935 ms/sample |
| Overfit 8 mẫu × 40 epoch | 278.5s (~7s/epoch cho 8 mẫu) |

→ Ước lượng **~1.9 h/epoch** full fine-tune (39.662 mẫu ÷ 1.07 samples/sec ≈ 10,3 giờ cho 1 epoch
forward+backward trên máy này — nên **ưu tiên giảm `max_length` xuống 128** nếu cần tăng tốc).

| Máy | Gợi ý |
|-----|-------|
| CPU yếu (4-6 core) | `batch_size 8`, `epochs 2`, `gradient_checkpointing=True`, cân nhắc `max_length 128` |
| CPU khá (8+ core) | `batch_size 16`, `epochs 3` |

> ⏱️ **Khuyến nghị mạnh:** trước khi train full, hãy chạy bước 2 (5% data) để đo thời gian thật
> của máy bạn, rồi tính ngược ra tổng thời gian. Đừng launch full 3 epoch rồi mới biết.

### 6.5 Quy tắc quan trọng
- **`test.csv` khóa** — chỉ dùng ở G3. G2 chỉ dùng `train` + `val`.
- Mỗi run một folder mới `results/runs/<timestamp>/`, không ghi đè.
- Log đủ: loss, metric, lr, thời gian.
- Checkpoint best theo `f1` val.

---

## 7. Cảnh báo quan trọng (đọc trước khi code G2)

| # | Cảnh báo | Chi tiết |
|---|----------|----------|
| 1 | **`transformers` 5.x đổi API** | Dùng `eval_strategy=` **không phải** `evaluation_strategy=`. `load_metric` **đã bị bỏ** → dùng `evaluate.load("accuracy")` hoặc `sklearn` (khuyến nghị sklearn, không cần network) |
| 2 | **`datasets` 5.x** | `Dataset.from_dict` OK. `Dataset.map` vẫn dùng được. Nhưng G1 đã tokenize sẵn → không cần `map` |
| 3 | **Tải model lần đầu chậm** | 242 giây lần đầu (tải từ Hub), 3.7 giây lần sau (cache). Đừng hoảng nếu lần đầu lâu |
| 4 | **Offline** | Sau lần tải đầu, đặt `HF_HUB_OFFLINE=1` để không gọi mạng: `$env:HF_HUB_OFFLINE=1` (PowerShell) |
| 5 | **Cột `labels`** | `Trainer` tự tìm cột `labels` trong dataset. `build_datasets()` đã đặt đúng tên này |
| 6 | **`token_type_ids`** | DistilBERT không dùng `token_type_ids`. Trainer sẽ tự bỏ qua — không lỗi |
| 7 | **Path data lồng cấp** | Data thật ở `data/data/splits/`. Nếu ai đó di dời, sửa `paths.splits_dir` trong `configs/model_config.yaml` |
| 8 | **Đừng tune trên `test`** | Đề bài chấm điểm chính là eval trên test. Chỉ G3 mới mở test |

---

## 8. Xử lý sự cố

| Triệu chứng | Nguyên nhân | Cách sửa |
|-------------|-----------|----------|
| `ModuleNotFoundError: transformers` | Chưa cài | `pip install -r requirements.txt` |
| `OSError: We couldn't connect to 'huggingface.co'` | Mạng / firewall | Đợi hoặc dùng cache: `HF_HUB_OFFLINE=1` |
| `FileNotFoundError: data/data/splits/train.csv` | Chạy sai thư mục | `cd` vào gốc repo trước khi chạy |
| `RuntimeError: No such file or directory: 'configs/model_config.yaml'` | Sai cwd | Như trên |
| Load model lâu > 1 phút | Tải lần đầu | Bình thường, chờ. Lần sau có cache |
| `ImportError: cannot import name 'evaluation_strategy'` | Dùng API 4.x | Đổi sang `eval_strategy=` |
| Loss = NaN khi train | Learning rate cao / fp16 | Giảm `lr` xuống 2e-5, bật `max_grad_norm=1.0`, giữ `float32` |
| Train quá chậm | CPU | `batch_size` nhỏ hơn, `epochs` ít hơn, `gradient_checkpointing=True` |
| `RuntimeError: out of memory` | RAM | Đóng notebook, giảm batch, `gradient_checkpointing=True` |
| Muốn chạy không cần mạng | — | `HF_HUB_OFFLINE=1` (sau lần tải đầu) |

---

## 9. Quy ước cho người viết code tiếp theo

1. **Đọc config, không hardcode.** Mọi tham số vào `configs/*.yaml`.
2. **Đặt code vào `src/<module>/`**, mỗi module một thư mục, có `__init__.py` export API.
3. **CLI bắt đầu bằng `python -m src.<module>`** — ví dụ `python -m src.training --config configs/train_config.yaml`.
4. **Kết quả chạy → `results/runs/<timestamp>/`** gồm `config.yaml`, `logs.txt`, `metrics.json`, `plots/`.
5. **Mỗi gói một commit** theo Conventional Commit: `feat(model): ...`, `feat(training): ...`.
6. **Ghi quyết định vào `docs/progress_status.md`** (Decisions Log) và cập nhật Module Log.
7. **Test trước khi commit**: `python -m pytest tests/ -q`.
8. **Không commit** dữ liệu lớn hay model weights — đã khai trong `.gitignore`.

---

## 10. Danh sách tài liệu liên quan

| File | Nội dung |
|------|----------|
| `đe` | Đề bài gốc (Exercise 1 & 2) |
| `đề.txt` | Đề bài bản text (rỗng) |
| `docs/remaining_plan.md` | Kế hoạch 6 gói + tiêu chí DoD từng gói |
| `docs/progress_status.md` | Single source of truth: Module Log, Decisions Log, Risks, Open Questions |
| `docs/g1_model_progress.md` | Báo cáo kỹ thuật G1 |
| `docs/AUDIT_LOG.md` | Nhật ký làm việc chi tiết + bằng chứng thực thi |
| `docs/RUN_GUIDE.md` | **File này** — hướng dẫn chạy |
| `configs/model_config.yaml` | Tham số model |
| `configs/data_config.yaml` | Tham số data |
| `results/runs/g1-model/metrics.json` | Số liệu G1 |
| `results/runs/g1-model/eval_quick.json` | Số liệu eval nhanh trên data thật |
