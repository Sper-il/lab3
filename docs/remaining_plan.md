# Kế hoạch công việc còn lại — Lab 3 Hugging Face (sau EDA + làm sạch)

> Phạm vi đã xong: Define, Design, Data (EDA + làm sạch) với `data/splits/{train,val,test}.csv` sẵn sàng `tokenize`.
> Phạm vi còn lại: Model → Training → Eval + Visualize HTML → Demo → Document, bám đúng `ai-fusion-workflow`.

## 0. Ảnh chụp tiến độ hiện tại
- Dữ liệu sạch: `clean 49.578` mẫu, splits `39.662/4.958/4.958`, `seed=42`, sai lệch lớp < 0,2%.
- Đã verify: `tokenize distilbert-base-uncased` shape `(B, 256)`, `pipeline sentiment-analysis` chạy đúng POSITIVE/NEGATIVE.
- Config hiện có: `configs/data_config.yaml`. Còn thiếu: `model_config.yaml`, `train_config.yaml`, `eval_config.yaml`.

## 1. Sơ đồ phụ thuộc
```
[Hoàn tất] Data sạch
    → G1 Model (tokenize + backbone + head)
        → G2 Training (Trainer + checkpoint)
            → G3 Eval (accuracy/F1 + confusion matrix + latency)
                → G4 Visualize report.html
                    → G5 Demo 1 lệnh
                        → G6 Document (README + AI_Usage + progress_status)
```

## 2. Chia gói việc (6 gói, làm tuần tự, mỗi gói 1 commit)

### Gói G1 — Model: chốt backbone + `tokenize` + kiểm tra shape
- Mục tiêu: có `build_model()` và `tokenize` chạy được trên 1 batch thật, khớp Design `input (B,256)` → `logits (B,2)`.
- Input: `data/splits/train.csv`; Output: `logits` shape `(B,2)`.
- Files tạo/sửa:
  - `configs/model_config.yaml` (model_name `distilbert-base-uncased`, num_labels 2, max_length 256)
  - `src/models/tokenizer_utils.py` (`build_tokenizer()`, `tokenize_batch()`)
  - `src/models/architecture.py` (`build_model(config)`)
  - `src/models/loss.py` (`build_loss()` = CrossEntropy)
  - `notebooks/02_tokenize_check.ipynb` (in shape 1 batch, 2 câu mẫu Ex1)
- Tiêu chí xong (DoD): 1 batch `tokenize` đúng `(B,256)` + `attention_mask`; forward ra `(B,2)`; ước lượng params (~66M) + dung lượng (~260MB); Decisions Log ghi lý do chọn `distilbert` thay vì `bert-base`.
- Rủi ro: thiếu RAM khi load model → dùng `torch.float32` CPU, `batch_size` nhỏ khi check.
- Ước lượng: 1 buổi lab.

### Gói G2 — Training: `Trainer` + `fine-tune` binary classification
- Mục tiêu: `fine-tune` chạy end-to-end, sinh checkpoint best + log tái lập được.
- Files tạo/sửa:
  - `configs/train_config.yaml` (optimizer AdamW, lr 2e-5, batch 16, epoch 3, weight_decay 0.01, seed 42, early_stopping patience 2, monitor `f1`)
  - `src/training/trainer.py` (`train_one_epoch()`, `validate()`, `save_checkpoint()`, hỗ trợ `--resume`)
  - `src/training/__main__.py` (CLI `python -m src.training --config configs/train_config.yaml`)
  - `results/runs/<ts>/` (`config.yaml`, `logs.txt`, `metrics.json`, `plots/loss_curve.png`)
- Thứ tự chạy bắt buộc: (1) overfit 1 batch nhỏ sanity check, (2) train thử 1 epoch trên 5% train để ước lượng giờ, (3) train full 3 epoch.
- Tiêu chí xong: loss giảm, best checkpoint theo `f1` val, mỗi run 1 folder mới không ghi đè, log đủ loss/metric/lr/thời gian.
- Rủi ro: loss NaN / overfit / train lâu trên CPU → giảm batch xuống 8, bật gradient clipping 1.0, giảm epoch còn 2 nếu kẹt.
- Ước lượng: 2–3 buổi (train CPU là bước lâu nhất).

### Gói G3 — Eval: 2 lớp model + pipeline
- Mục tiêu: số liệu cuối cùng trên `test` khóa riêng, kèm latency và error analysis.
- Files tạo/sửa:
  - `configs/eval_config.yaml` (checkpoint path, batch_size, target: accuracy ≥ 0,85, F1 ≥ 0,85)
  - `src/evaluation/metrics.py` (accuracy, precision, recall, F1, confusion matrix)
  - `src/evaluation/__main__.py` (CLI `python -m src.evaluation --config ... --checkpoint ...`)
  - `results/runs/<ts>/metrics.json` đúng schema `{metric, value, target, passed}`, `plots/confusion_matrix.png`
- Tiêu chí xong: metric tính trên `test` (không tuning trên test), có bảng target vs đạt được, có per-class metric, latency đo trên đúng CPU lab, liệt kê 5 case sai điển hình.
- Ước lượng: 1 buổi.

### Gói G4 — Visualize: `report.html` offline 1 file
- Mục tiêu: 1 file `results/runs/<ts>/report.html` mở bằng double-click, số khớp `metrics.json`.
- Files tạo/sửa:
  - `src/evaluation/visualize_html.py` (`build_report(run_dir)`, dùng `string.Template`, ảnh base64)
  - `results/runs/<ts>/report.html` (header + bảng metric pass/fail + loss curve + confusion matrix + 5 mẫu tốt + 5 mẫu tệ + limitations + next steps)
- Tiêu chí xong: offline 100%, ảnh resize cạnh dài ≤ 800px, tổng file < 10MB, có run id + checkpoint path để truy vết.
- Ước lượng: 0,5 buổi, làm ngay sau G3.

### Gói G5 — Demo: 1 lệnh chạy được
- Mục tiêu: người mới clone repo chạy demo dưới 5 phút (không tính tải model).
- Files tạo/sửa:
  - `data/processed/demo/` (3–5 câu mẫu: dễ/khó/biên, file nhẹ commit kèm repo)
  - `src/demo.py` (`python -m src.demo --input ... --checkpoint ...`, in label + score + latency)
  - `notebooks/demo.ipynb` (chạy từ trên xuống không sửa code)
  - `results/demo/<ts>/` (output + metrics mini)
- Tiêu chí xong: path tương đối, có in `inference_time_ms` + `end_to_end_ms`, có case khó minh họa giới hạn, khối Example Input → Output đồng bộ README.
- Ước lượng: 0,5 buổi.

### Gói G6 — Document: chốt README + AI_Usage + progress_status
- Mục tiêu: đủ bộ nộp bài theo skill.
- Files tạo/sửa:
  - `README.md` (10 mục chuẩn: mô tả, cài đặt, dữ liệu, train, eval, kết quả, demo, limitations, cấu trúc, tài liệu tham khảo)
  - `docs/AI_Usage.md` (7 mục: công cụ AI dùng, prompt, phần nào AI viết, phần nào người sửa, verify thế nào)
  - `docs/progress_status.md` (cập nhật Module Log, Decisions Log, Risks, DoD tick hết)
  - `docs/CHANGELOG.md` (version model/dataset)
- Tiêu chí xong: số trong README khớp `metrics.json` run tốt nhất, link `report.html` + checkpoint rõ ràng, commit Conventional Commit.
- Ước lượng: 0,5 buổi, làm song song từ G3 tới cuối.

## 3. Thứ tự đề xuất theo buổi
| Buổi | Gói | Lệnh kiểm tra |
|------|-----|---------------|
| 1 | G1 Model | `python -m src.models.check` (in shape batch) |
| 2–3 | G2 Training | `python -m src.training --config configs/train_config.yaml` |
| 4 | G3 Eval + G4 Visualize | `python -m src.evaluation ...` rồi mở `report.html` |
| 5 | G5 Demo + G6 Document | `python -m src.demo --input data/processed/demo/samples.json` |

## 4. Quy tắc thực hiện (từ skill)
- Một gói một commit, không gộp Model + Training vào một lần.
- Mỗi gói đi đủ vòng Describe → Generate → Review → Run → Test → Refactor → Commit.
- Mọi quyết định ghi vào `docs/progress_status.md` Decisions Log, mọi số ghi vào `results/runs/<ts>/metrics.json`.
- `data/raw` bất biến; `test.csv` khóa tới G3 mới dùng.
