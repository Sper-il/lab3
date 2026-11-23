# Lab 3 — Làm quen Hugging Face (Sentiment + Fine-tune Binary Classification)

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-23 13:20 (G1 Model ✅)

## 📚 Các tài liệu trong repo
| File | Vai trò |
|------|--------|
| `docs/remaining_plan.md` | Kế hoạch 6 gói + tiêu chí DoD |
| `docs/progress_status.md` | **File này** — Module Log, Decisions Log, Risks |
| `docs/g1_model_progress.md` | Báo cáo kỹ thuật G1 (Model) |
| `docs/AUDIT_LOG.md` | Nhật ký làm việc G1 + bằng chứng thực thi |
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
| Tính năng cốt lõi | Must: EDA + làm sạch + chốt `splits` sẵn sàng `tokenize` (phạm vi lần này). Should: inference sentiment mẫu (Ex1) + `tokenize` mẫu kiểm tra shape. Could: `fine-tune` + đánh giá (để lab sau, ngoài phạm vi lần này). |
| Tech construction | Local CPU, Windows 11, Python 3.13, PyTorch 2.15 (CPU), `transformers`, `datasets`, `evaluate`, `scikit-learn`, `pandas`, `matplotlib`. Chạy `batch`, không `real-time`. Tiêu chí: mã nguồn mở, gọn nhẹ, chạy được trên CPU, dễ tái lập (`seed` cố định). |
| Tiêu chí đánh giá (cho bước Data) | EDA đầy đủ: số mẫu, phân bố lớp, tỉ lệ thiếu/trùng/rỗng, phân bố độ dài; sau làm sạch: 0% rỗng, 0% trùng lặp, `splits` giữ phân bố lớp sai lệch < 2%, file `splits` cố định + tái lập được. |

## Architecture Snapshot (Phase 2 — DESIGN)
- System architecture: Pipeline dữ liệu `raw → processed → splits → tokenized-ready`. Chi tiết xem `docs/architecture.md` (sẽ bổ sung khi sang module Model/Training).
- Modules: `data: load + EDA + clean + split` (✅ xong) | `models: backbone + tokenizer + loss` (✅ xong — G1) | `training: Trainer` (chưa làm) | `evaluation: accuracy/F1` (chưa làm).
- Folder structure: theo chuẩn skill (`data/{raw,processed,splits}`, `src/data`, `configs`, `notebooks`, `results/runs/<ts>`).
- Data flow: `HF Hub (stanfordnlp/imdb) → data/raw/ (bất biến) → clean (lower/strip HTML/URL, bỏ rỗng + trùng) → data/processed/clean_imdb.csv → stratified split 80/10/10 (seed=42) → data/splits/{train,val,test}.csv → sẵn sàng tokenize (max_length=256)`.
- Interfaces / APIs: `src/data/dataset.py: load_raw() -> Dataset`; `clean_text(s: str) -> str`; `make_splits(df, seed) -> (train,val,test)`; mỗi mẫu output `dict {'text': str, 'label': int}`; batch `tokenize` shape `(B, 256)` với `attention_mask` tương ứng.
- Milestones: `W1: Define + Design + EDA + làm sạch (lần này, DONE) | W2: Model + tokenize | W3: Training + Eval`.
- Technical risks: tải HF Hub lỗi mạng → có fallback cache + script log rõ; `transformers` chưa cài → cài bổ sung; mất cân bằng do bỏ trùng → kiểm tra lại phân bố sau split.

## Active Module: data (EDA + làm sạch)
> ⚠️ Module data đã xong. Module hiện tại là **G1 — Model** (✅ xong 2026-11-23).
> Tài liệu G1: `docs/g1_model_progress.md` (báo cáo) · `docs/AUDIT_LOG.md` (nhật ký) · `docs/RUN_GUIDE.md` (hướng dẫn chạy)
- **Describe** — Tải IMDb thô, thống kê EDA, làm sạch văn bản, chốt `splits` train/val/test sẵn sàng cho `tokenize` và `fine-tune`.
- **Goal / purpose** — Có bộ dữ liệu sạch, có truy vết, không `leakage`, tái lập được, đúng yêu cầu Ex2 mục 2–4 (load + preprocess dataset).
- **Pipeline** — `load_dataset → snapshot raw → EDA (shape, null, dup, class, length) → clean_text → verify → stratified split → lưu processed/splits + plots + metrics.json`.
- **How / experiment plan (sub-tasks)**
  - [x] Cài `transformers`, `evaluate` (đã có `datasets`, `torch`, `pandas`, `sklearn`)
  - [x] Viết `configs/data_config.yaml` (single source of truth)
  - [x] Viết `src/data/{dataset,clean,split}.py` + script `run_eda.py`
  - [x] Chạy EDA, lưu `results/runs/eda-baseline/metrics.json + plots/`
  - [x] Verify `splits` và cập nhật Module Log
- **Status** — Describe ✅ / Generate ✅ / Review ✅ / Run ✅ / Test ✅ / Refactor ✅ / Commit ⏳
- **Last run** — `results/runs/eda-baseline/`
- **DoD checklist** — xem `checklists/dod.md` (workspace skill)

## Module Log
| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | Scaffold | 2026-11-21 | ✅ | — | folders + templates in place |
| 1 | Define + Design | 2026-11-21 | ✅ | — | chốt 6 mục Define + Architecture Snapshot |
| 2 | Data (EDA + làm sạch) | 2026-11-21 | ✅ | — | raw 50.000 → clean 49.578, splits 39.662/4.958/4.958, sai lệch lớp < 0,2% |
| 3 | **Model (G1: tokenize + backbone + loss)** | 2026-11-23 | ✅ | ⏳ | distilbert-base-uncased 66.96M params, `(B,256)` → `(B,2)`, ~255MB. Smoke test Trainer PASS + overfit loss→0.0027. Chi tiết: `docs/g1_model_progress.md` |

## Decisions Log
| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-11-21 | Chọn `stanfordnlp/imdb` (fallback `imdb`) cho Ex2 | `glue/sst2`, `rotten_tomatoes` | IMDb đúng yêu cầu đề (binary text classification), 2 lớp cân bằng, nhiều tài liệu `fine-tune` BERT tham khảo |
| 2026-11-21 | Chọn `distilbert-base-uncased` làm backbone dự kiến (chưa train) | `bert-base-uncased` | Nhẹ hơn ~40%, chạy được CPU lab, vẫn giữ ~97% hiệu năng BERT |
| 2026-11-21 | Dừng ở EDA + làm sạch theo yêu cầu | Làm luôn fine-tune | Tôn trọng phạm vi người dùng, giữ đúng nguyên tắc MVP một module một lần |
| 2026-11-21 | Skill Kaggriculture bỏ qua | Áp dụng | Skill nói về game farming-sim, không liên quan Hugging Face/NLP |
| 2026-11-23 | **Chốt backbone `distilbert-base-uncased` cho G1** | `bert-base-uncased`, `roberta-base` | Nhẹ hơn BERT-base ~40% (67M vs 110M), train CPU trong phòng lab khả thi, vẫn giữ ~97% hiệu năng BERT trên IMDB. Fallback `bert-base-uncased` nếu Hub lỗi |
| 2026-11-23 | `padding="max_length"`, `max_length=256` | `padding="longest"` | Shape cố định `(B, 256)` đúng Design, kiểm tra shape đơn giản, Trainer không phải xử lý padding động |
| 2026-11-23 | `dtype=float32` trên CPU | `float16`/`bfloat16` | CPU máy lab không có tensor core; float32 tránh NaN, đúng ghi chú rủi ro trong kế hoạch |
| 2026-11-23 | Tách `src/utils/config.py` đọc YAML tập trung | Hardcode trong từng file | Config là single source of truth; người G2 chỉ sửa YAML, không sửa code |
| 2026-11-23 | Thêm `src/models/dataset_builder.build_datasets()` | Để người G2 tự tokenize | Người 2 không phải viết lại logic tokenize; trả về `DatasetDict` sẵn đưa vào `Trainer` |
| 2026-11-23 | `build_datasets()` set format `torch` | Mặc định `python` (list) | `Trainer` tự xử lý list, nhưng code PyTorch thuần sẽ lỗi `embedding(): indices must be Tensor, not list`. Format torch dùng được cho **cả 2** |
| 2026-11-23 | Metric dùng `sklearn` (không dùng `evaluate.load`) | `evaluate.load("accuracy")` | `datasets` 5.x đã bỏ `load_metric`; sklearn không cần network, đã test chạy được trong `smoke_train.py` |

## Risks & Mitigations
| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Không tải được HF Hub (mạng/firewall) | Trung bình | Cao | Thử `stanfordnlp/imdb` rồi `imdb`; log lỗi rõ; giữ `data/raw` bất biến | Khi `run_eda.py` báo lỗi download |
| `transformers`/`evaluate` chưa cài | Cao | Trung bình | `pip install transformers evaluate` trước khi chạy | `pip show` kiểm tra |
| Bỏ trùng làm lệch phân bố lớp | Thấp | Trung bình | Kiểm tra phân bố sau clean + dùng `stratified split`, sai lệch cho phép < 2% | So sánh EDA trước/sau trong `metrics.json` |

## Open Questions
- [x] Tên project + bài toán
- [x] Input/output shape
- [x] Tính năng cốt lõi (Must/Should/Could)
- [x] Tech stack + lý do
- [x] Metric đánh giá + target
- [x] Backbone chốt khi sang module Model (`distilbert-base-uncased` — chốt 2026-11-23 trong G1)
- [ ] **G2 cần chốt `max_length` khi train**: G1 dùng 256 theo Design, nhưng đo thật cho thấy ~10 h/epoch trên CPU → cân nhắc 128 cho lúc train (giữ 256 cho eval)

## Definition of Done (áp dụng mỗi module)
- [x] Đúng yêu cầu (EDA + làm sạch cho Ex2)
- [x] Code chạy end-to-end
- [x] Tests pass
- [x] Không có secrets / không có artifact lớn
- [x] Code đã self-review
- [x] Có thể giải thích được
- [x] Docs đã cập nhật
- [ ] Đã committed (Conventional Commit)
