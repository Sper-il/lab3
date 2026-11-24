# Lab 3 — Làm quen Hugging Face (Sentiment + Fine-tune Binary Classification)

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-24 17:05 (G2 Training ✅)

## 📚 Các tài liệu trong repo
| File | Vai trò |
|------|--------|
| `docs/remaining_plan.md` | Kế hoạch 6 gói + tiêu chí DoD |
| `docs/progress_status.md` | **File này** — Module Log, Decisions Log, Risks |
| `docs/g1_model_progress.md` | Báo cáo kỹ thuật G1 (Model) |
| `docs/g2_training_progress.md` | Báo cáo kỹ thuật G2 (Training) |
| `docs/AUDIT_LOG.md` | Nhật ký làm việc chi tiết + bằng chứng thực thi |
| `docs/RUN_GUIDE.md` | Hướng dẫn cài đặt / chạy / quy trình cho thành viên tiếp theo |

## Project Context (Phase 1 — DEFINE)
| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Lab 3 — Làm quen Hugging Face: Sentiment Analysis và Fine-tune Binary Classification |
| Slug | lab3-hf-binary-sentiment |
| Tagline | Dùng mô hình `pre-trained` trên Hugging Face Hub để suy luận cảm xúc và `fine-tune` phân loại văn bản nhị phân. |
| Bài toán | (1) Suy luận sentiment với pipeline có sẵn; (2) `fine-tune` mô hình `pre-trained` (ví dụ `distilbert-base-uncased`) cho phân loại văn bản nhị phân (positive/negative). Bài toán quan trọng vì đây là quy trình chuẩn chuyển giao tri thức `pre-trained` → bài toán cụ thể. |
| Người dùng mục tiêu | Sinh viên thực hành lab AI/NLP; chạy trên laptop CPU, xem output là nhãn + xác suất, phục vụ báo cáo lab. |
| Input | Văn bản tiếng Anh, 1 câu/đoạn review phim. Ví dụ: `"I loved this movie, it was fantastic!"`. Độ dài thô 10–500 token. |
| Output | Nhãn nhị phân: `0 = negative`, `1 = positive`, kèm `score` xác suất. Ví dụ: `{"label": "POSITIVE", "score": 0.98}`. |
| Dữ liệu | Dùng tập `IMDb` (50.000 review phim, 2 lớp cân bằng) qua thư viện `datasets`: `load_dataset("stanfordnlp/imdb")` (fallback `imdb`). Chia `train/val/test` theo tỉ lệ 80/10/10, `stratified`, `seed=42`, chống `leakage` bằng tách trước khi `tokenize`. |
| Tính năng cốt lõi | Must: EDA + làm sạch + chốt `splits` sẵn sàng `tokenize` + Model + Training. Should: inference sentiment mẫu (Ex1) + `tokenize` mẫu kiểm tra shape. Could: Eval + Visualize HTML + Demo. |
| Tech construction | Local CPU, Windows 11, Python 3.12, PyTorch 2.13 (CPU), `transformers 5.24.0`, `datasets 5.2.0`, `evaluate 0.4.6`, `accelerate 1.16.0`, `scikit-learn`, `pandas`, `matplotlib`. Chạy `batch`, không `real-time`. Tiêu chí: mã nguồn mở, gọn nhẹ, chạy được trên CPU, dễ tái lập (`seed` cố định). |
| Tiêu chí đánh giá (cho bước Data) | EDA đầy đủ: số mẫu, phân bố lớp, tỉ lệ thiếu/trùng/rỗng, phân bố độ dài; sau làm sạch: 0% rỗng, 0% trùng lặp, `splits` giữ phân bố lớp sai lệch < 2%, file `splits` cố định + tái lập được. |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: Pipeline dữ liệu `raw → processed → splits → tokenized-ready → Trainer → best_model → Eval`.
- Modules: `data: load + EDA + clean + split` (✅ xong) | `models: backbone + tokenizer + loss` (✅ xong — G1) | `training: Trainer + checkpoints + logs + curves` (✅ xong — G2) | `evaluation: accuracy/F1` (chưa làm — G3).
- Folder structure: theo chuẩn skill (`data/{raw,processed,splits}`, `src/data`, `src/models`, `src/training`, `configs`, `results/runs/<ts>`).
- Data flow: `HF Hub (stanfordnlp/imdb) → clean_imdb.csv → stratified split 80/10/10 → splits/{train,val,test}.csv → DatasetDict(torch) → Trainer(AdamW, early_stopping) → best_model + metrics.json + loss_curve.png`.
- Milestones: `W1: Data (DONE) | W2: G1 Model (DONE) | W2.5: G2 Training (DONE) | W3: G3 Eval + G4 Visualize | W4: G5 Demo + G6 Document`.

## Active Module: training (Gói G2)
> ⚠️ Module training đã xong. Module tiếp theo là **G3 — Eval**.
> Tài liệu G2: `docs/g2_training_progress.md` (báo cáo) · `docs/AUDIT_LOG.md` (nhật ký) · `docs/RUN_GUIDE.md` (hướng dẫn chạy)
- **Describe** — Xây dựng quy trình fine-tune end-to-end với Hugging Face Trainer, early stopping, lưu checkpoint best theo F1 val, log và biểu đồ loss/metric.
- **Goal / purpose** — Fine-tune phân loại nhị phân IMDb, lưu checkpoint tái lập được, log đầy đủ, không ghi đè run cũ, sẵn sàng bàn giao cho G3 Eval.
- **Status** — Describe ✅ / Generate ✅ / Review ✅ / Run ✅ / Test ✅ / Refactor ✅ / Commit ⏳
- **Last run** — `results/runs/g2-training/`
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold | 2026-11-21 | ✅ | — | folders + templates in place |
| 1 | Define + Design | 2026-11-21 | ✅ | — | chốt 6 mục Define + Architecture Snapshot |
| 2 | Data (EDA + làm sạch) | 2026-11-21 | ✅ | — | raw 50.000 → clean 49.578, splits 39.662/4.958/4.958, sai lệch lớp < 0,2% |
| 3 | Model (G1: tokenize + backbone + loss) | 2026-11-23 | ✅ | — | distilbert-base-uncased 66.96M params, `(B,256)` → `(B,2)`, ~255MB. Smoke test Trainer PASS + overfit loss→0.0027. Chi tiết: `docs/g1_model_progress.md` |
| 4 | **Training (G2: Trainer + fine-tune)** | 2026-11-24 | ✅ | ⏳ | Hugging Face Trainer end-to-end, EarlyStoppingCallback (patience 2, monitor F1), `warmup_steps`, lưu best model kèm tokenizer, xuất `loss_curve.png`, `metrics.json`, `logs.txt`, CLI hỗ trợ `--estimate-time` và `--overfit-check`. 17/17 tests PASS. Chi tiết: `docs/g2_training_progress.md` |

## Decisions Log
| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-11-21 | Chọn `stanfordnlp/imdb` (fallback `imdb`) cho Ex2 | `glue/sst2`, `rotten_tomatoes` | IMDb đúng yêu cầu đề (binary text classification), 2 lớp cân bằng, nhiều tài liệu `fine-tune` BERT tham khảo |
| 2026-11-21 | Chọn `distilbert-base-uncased` làm backbone dự kiến (chưa train) | `bert-base-uncased` | Nhẹ hơn ~40%, chạy được CPU lab, vẫn giữ ~97% hiệu năng BERT |
| 2026-11-21 | Dừng ở EDA + làm sạch theo yêu cầu | Làm luôn fine-tune | Tôn trọng phạm vi người dùng, giữ đúng nguyên tắc MVP một module một lần |
| 2026-11-23 | Chốt backbone `distilbert-base-uncased` cho G1 | `bert-base-uncased`, `roberta-base` | Nhẹ hơn BERT-base ~40% (67M vs 110M), train CPU trong phòng lab khả thi |
| 2026-11-23 | `padding="max_length"`, `max_length=256` | `padding="longest"` | Shape cố định `(B, 256)` đúng Design, kiểm tra shape đơn giản |
| 2026-11-23 | `dtype=float32` trên CPU | `float16`/`bfloat16` | CPU máy lab không có tensor core; float32 tránh NaN |
| 2026-11-24 | **Dùng `eval_strategy="epoch"` cho `TrainingArguments`** | `evaluation_strategy` | `transformers 5.x` đã deprecate `evaluation_strategy` sang `eval_strategy` |
| 2026-11-24 | **Dùng `warmup_steps` thay vì `warmup_ratio`** | `warmup_ratio` | `transformers 5.x` không hỗ trợ `warmup_ratio` trực tiếp trong `TrainingArguments` |
| 2026-11-24 | **Lưu cả tokenizer cùng model vào `best_model/`** | Chỉ lưu model weights | Giúp G3 Eval và G5 Demo chỉ cần nạp 1 thư mục bằng `from_pretrained(".../best_model")` |
| 2026-11-24 | **Khóa cứng `test` trong G2** | Cho phép load test | Chống rò rỉ dữ liệu (data leakage) sang giai đoạn train; chỉ G3 mới được mở test |
| 2026-11-24 | **Thêm `max_samples` vào `build_datasets()`** | `.select()` sau khi tokenize full | Tiết kiệm hàng phút tiền xử lý khi chạy debug/benchmark trên máy lab |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Huấn luyện full 3 epoch trên CPU quá lâu (~80 giờ) | Cao | Cao | Bật tính năng `--estimate-time`, cho phép train theo subset `--max-train-samples`, hỗ trợ resume checkpoint | Đo tốc độ bằng benchmark |
| Bị ngắt kết nối mạng HF Hub khi train | Trung bình | Thấp | Bật `$env:HF_HUB_OFFLINE="1"` sau lần tải đầu, model và dataset đã cache hoàn toàn local | Khi chạy thử |
| Loss NaN / Gradient explode | Thấp | Cao | Sử dụng `max_grad_norm=1.0` (gradient clipping), `float32`, lr `2e-5` | Theo dõi logs.txt |

## Open Questions
- [x] Tên project + bài toán
- [x] Input/output shape
- [x] Tính năng cốt lõi (Must/Should/Could)
- [x] Tech stack + lý do
- [x] Metric đánh giá + target
- [x] Backbone chốt khi sang module Model (`distilbert-base-uncased`)
- [x] G2 Training quy trình + checkpoint best theo F1 val
- [ ] G3 Eval: Đánh giá độ trễ và phân tích 5 mẫu lỗi điển hình trên tập `test.csv`

## Definition of Done (áp dụng G2)
- [x] Đúng yêu cầu (Trainer + fine-tune binary classification)
- [x] Code chạy end-to-end (CLI `python -m src.training` chạy thành công)
- [x] Tests pass (17/17 tests passed)
- [x] Không có secrets / không có artifact lớn (weights và checkpoints đã được gitignore)
- [x] Code đã self-review
- [x] Có thể giải thích được
- [x] Docs đã cập nhật
- [ ] Đã committed (Conventional Commit)
