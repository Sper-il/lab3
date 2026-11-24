import argparse
import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

try:
    from PIL import Image
except ImportError:
    Image = None


LABEL_MAP = {
    0: "NEGATIVE (0)",
    1: "POSITIVE (1)",
}

MAX_IMAGE_SIDE = 800
MAX_REPORT_BYTES = 10 * 1024 * 1024  # 10 MB


def encode_image_to_base64(image_path: Path, max_side: int = MAX_IMAGE_SIDE) -> Optional[str]:
    """Đọc ảnh, resize sao cho cạnh dài <= 800px rồi chuyển sang Data URI Base64."""
    if not image_path.exists() or not image_path.is_file():
        return None

    if Image is not None:
        with Image.open(image_path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return f"data:image/png;base64,{b64}"

    # Fallback nếu môi trường chưa cài Pillow
    suffix = image_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else (suffix or "png")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{b64}"


def load_json_safe(file_path: Path) -> Optional[Any]:
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _render_metrics_section(metrics_payload: Optional[Dict[str, Any]]) -> str:
    if not metrics_payload:
        return (
            '<div class="empty-box">'
            "<strong>Chưa có dữ liệu Metrics (chờ G3 chạy evaluation)</strong><br>"
            "<span>Không tìm thấy file <code>metrics.json</code> trong thư mục run.</span>"
            "</div>"
        )

    summary: List[Dict[str, Any]] = metrics_payload.get("metrics", [])
    per_class: Dict[str, Any] = metrics_payload.get("per_class", {})
    latency: Dict[str, Any] = metrics_payload.get("latency", {})

    rows: List[str] = []
    for item in summary:
        m_name = html.escape(str(item.get("metric", "")).upper())
        val = item.get("value")
        val_str = f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"
        target = item.get("target")
        target_str = f"&ge; {target:.2f}" if isinstance(target, (int, float)) else "—"
        passed = item.get("passed")

        if passed is True:
            badge = '<span class="badge pass">PASS</span>'
        elif passed is False:
            badge = '<span class="badge fail">FAIL</span>'
        else:
            badge = '<span class="badge info">INFO</span>'

        rows.append(
            f"<tr><td><strong>{m_name}</strong></td>"
            f'<td class="num">{val_str}</td>'
            f'<td class="num">{target_str}</td>'
            f"<td>{badge}</td></tr>"
        )

    per_class_html = ""
    if per_class:
        pc_rows = "".join(
            f"<tr><td>{html.escape(cls_key)}</td>"
            f'<td class="num">{stats.get("precision", "N/A")}</td>'
            f'<td class="num">{stats.get("recall", "N/A")}</td>'
            f'<td class="num">{stats.get("f1", "N/A")}</td>'
            f'<td class="num">{stats.get("support", "N/A")}</td></tr>'
            for cls_key, stats in per_class.items()
        )
        per_class_html = (
            "<h3>Chi tiết theo từng lớp (Per-Class Metrics)</h3>"
            "<table><thead><tr>"
            '<th>Lớp (Class)</th><th class="num">Precision</th>'
            '<th class="num">Recall</th><th class="num">F1-Score</th><th class="num">Support</th>'
            f"</tr></thead><tbody>{pc_rows}</tbody></table>"
        )

    latency_html = ""
    if latency:
        avg_ms = latency.get("avg_latency_ms_per_sample", "N/A")
        thr = latency.get("throughput_samples_per_sec", "N/A")
        latency_html = (
            '<div class="meta-strip">'
            f"<span><strong>Latency:</strong> {avg_ms} ms/mẫu</span>"
            f"<span><strong>Throughput:</strong> {thr} mẫu/giây</span>"
            "</div>"
        )

    return (
        "<table><thead><tr>"
        '<th>Chỉ số (Metric)</th><th class="num">Giá trị thực tế</th>'
        '<th class="num">Ngưỡng mục tiêu</th><th>Trạng thái</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"{latency_html}{per_class_html}"
    )


def _render_plots_section(run_dir: Path, metrics_payload: Optional[Dict[str, Any]]) -> str:
    loss_path = run_dir / "plots" / "loss_curve.png"
    cm_rel = "plots/confusion_matrix.png"
    if metrics_payload and isinstance(metrics_payload.get("artifacts"), dict):
        cm_rel = metrics_payload["artifacts"].get("confusion_matrix_plot", cm_rel)
    cm_path = run_dir / cm_rel

    loss_b64 = encode_image_to_base64(loss_path, max_side=MAX_IMAGE_SIDE)
    cm_b64 = encode_image_to_base64(cm_path, max_side=MAX_IMAGE_SIDE)

    loss_block = (
        f'<img src="{loss_b64}" alt="Training & Validation Loss Curve">'
        if loss_b64
        else (
            '<div class="empty-box plot-empty">'
            "<strong>Chưa có dữ liệu Loss Curve (chờ G2)</strong><br>"
            "<span>Thiếu file <code>plots/loss_curve.png</code></span>"
            "</div>"
        )
    )

    cm_block = (
        f'<img src="{cm_b64}" alt="Confusion Matrix">'
        if cm_b64
        else (
            '<div class="empty-box plot-empty">'
            "<strong>Chưa có dữ liệu Confusion Matrix (chờ G3)</strong><br>"
            f"<span>Thiếu file <code>{html.escape(str(cm_rel))}</code></span>"
            "</div>"
        )
    )

    return (
        '<div class="grid-2">'
        f'<div class="plot-card"><h3>1. Loss Curve (G2)</h3>{loss_block}</div>'
        f'<div class="plot-card"><h3>2. Confusion Matrix — Test Set (G3)</h3>{cm_block}</div>'
        "</div>"
    )


def _render_sample_cards(samples: List[Dict[str, Any]], is_correct: bool) -> str:
    if not samples:
        msg = (
            "Chưa có dữ liệu 5 mẫu tốt (chờ G3 xuất correct_cases.json)."
            if is_correct
            else "Chưa có dữ liệu 5 mẫu tệ (chờ G3 xuất error_cases.json)."
        )
        return f'<div class="empty-box"><span>{msg}</span></div>'

    badge_cls = "pass" if is_correct else "fail"
    border_cls = "correct-border" if is_correct else "error-border"
    cards: List[str] = []

    for idx, item in enumerate(samples[:5], start=1):
        raw_text = str(item.get("text", ""))
        display_text = html.escape(raw_text[:280] + ("..." if len(raw_text) > 280 else ""))
        true_lbl = item.get("true_label")
        pred_lbl = item.get("pred_label")
        true_str = LABEL_MAP.get(true_lbl, str(true_lbl))
        pred_str = LABEL_MAP.get(pred_lbl, str(pred_lbl))
        conf = item.get("confidence")
        conf_str = f"{conf * 100:.2f}%" if isinstance(conf, (int, float)) else "N/A"
        sample_idx = item.get("index", "?")

        cards.append(
            f'<div class="sample-card {border_cls}">'
            f'<div class="sample-header"><span><strong>#{idx}</strong> (Index: {sample_idx})</span>'
            f'<span class="badge {badge_cls}">Độ tự tin: {conf_str}</span></div>'
            f'<p class="sample-text">"{display_text}"</p>'
            f'<div class="sample-labels"><span>Nhãn thật: <strong>{html.escape(true_str)}</strong></span>'
            f'<span>Dự đoán: <strong>{html.escape(pred_str)}</strong></span></div>'
            "</div>"
        )
    return f'<div class="sample-list">{"".join(cards)}</div>'


def _render_samples_section(
    metrics_payload: Optional[Dict[str, Any]],
    error_cases_raw: Optional[Any],
    correct_cases_raw: Optional[Any],
) -> str:
    incorrect_samples: List[Dict[str, Any]] = []
    correct_samples: List[Dict[str, Any]] = []

    if isinstance(error_cases_raw, list):
        incorrect_samples = error_cases_raw
    elif isinstance(error_cases_raw, dict):
        incorrect_samples = error_cases_raw.get("incorrect_samples", [])
        correct_samples = error_cases_raw.get("correct_samples", [])
    elif metrics_payload and isinstance(metrics_payload.get("error_analysis_top_k"), list):
        incorrect_samples = metrics_payload["error_analysis_top_k"]

    if isinstance(correct_cases_raw, list):
        correct_samples = correct_cases_raw

    return (
        '<div class="grid-2">'
        f"<div><h3>5 Mẫu Tốt (Dự đoán đúng tự tin nhất)</h3>{_render_sample_cards(correct_samples, True)}</div>"
        f"<div><h3>5 Mẫu Tệ (Dự đoán sai tự tin nhất)</h3>{_render_sample_cards(incorrect_samples, False)}</div>"
        "</div>"
    )


REPORT_TEMPLATE = Template("""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Report — $run_id</title>
    <style>
        :root {
            --bg: #f8fafc; --card: #ffffff; --text: #0f172a; --muted: #64748b;
            --border: #e2e8f0; --primary: #2563eb;
            --pass-bg: #dcfce7; --pass-text: #166534;
            --fail-bg: #fee2e2; --fail-text: #991b1b;
            --info-bg: #e0f2fe; --info-text: #075985;
        }
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg); color: var(--text); margin: 0; padding: 24px; line-height: 1.5;
        }
        .container { max-width: 1140px; margin: 0 auto; }
        header {
            background: var(--card); border: 1px solid var(--border); border-radius: 10px;
            padding: 20px 24px; margin-bottom: 20px; border-left: 5px solid var(--primary);
        }
        header h1 { margin: 0 0 8px 0; font-size: 22px; }
        .meta { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: var(--muted); }
        section {
            background: var(--card); border: 1px solid var(--border); border-radius: 10px;
            padding: 20px 24px; margin-bottom: 20px;
        }
        h2 { margin: 0 0 16px 0; font-size: 17px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
        h3 { margin: 16px 0 10px 0; font-size: 14px; color: #334155; }
        table { width: 100%; border-collapse: collapse; font-size: 14px; }
        th, td { padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: left; }
        th { background: #f1f5f9; font-weight: 600; }
        .num { text-align: right; font-variant-numeric: tabular-nums; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
        .badge.pass { background: var(--pass-bg); color: var(--pass-text); }
        .badge.fail { background: var(--fail-bg); color: var(--fail-text); }
        .badge.info { background: var(--info-bg); color: var(--info-text); }
        .meta-strip {
            margin-top: 12px; padding: 10px 14px; background: #f1f5f9;
            border-radius: 6px; display: flex; gap: 24px; font-size: 13px;
        }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 820px) { .grid-2 { grid-template-columns: 1fr; } }
        .plot-card img { width: 100%; height: auto; border: 1px solid var(--border); border-radius: 6px; }
        .empty-box {
            border: 1.5px dashed #cbd5e1; background: #f8fafc; border-radius: 8px;
            padding: 24px 16px; text-align: center; color: var(--muted); font-size: 13px;
        }
        .plot-empty { min-height: 240px; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .sample-list { display: flex; flex-direction: column; gap: 12px; }
        .sample-card { border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; background: #fff; }
        .correct-border { border-left: 4px solid #22c55e; }
        .error-border { border-left: 4px solid #ef4444; }
        .sample-header { display: flex; justify-content: space-between; align-items: center; font-size: 12px; margin-bottom: 6px; }
        .sample-text { margin: 6px 0; font-size: 13px; color: #1e293b; background: #f8fafc; padding: 8px 10px; border-radius: 6px; }
        .sample-labels { display: flex; gap: 16px; font-size: 12px; color: var(--muted); }
        ul.notes { margin: 0; padding-left: 20px; font-size: 14px; color: #334155; }
        ul.notes li { margin-bottom: 6px; }
        code { background: #e2e8f0; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Báo Cáo Đánh Giá Mô Hình Phân Loại Văn Bản (Lab 3 — G4)</h1>
            <div class="meta">
                <span><strong>Run ID:</strong> $run_id</span>
                <span><strong>Checkpoint:</strong> <code>$checkpoint_path</code></span>
                <span><strong>Thư mục run:</strong> <code>$run_dir</code></span>
                <span><strong>Tạo lúc:</strong> $generated_at</span>
            </div>
        </header>

        <section>
            <h2>1. Bảng Metric Pass / Fail</h2>
            $metrics_section
        </section>

        <section>
            <h2>2. Loss Curve &amp; Confusion Matrix</h2>
            $plots_section
        </section>

        <section>
            <h2>3. Phân Tích 5 Mẫu Tốt &amp; 5 Mẫu Tệ</h2>
            $samples_section
        </section>

        <section>
            <h2>4. Limitations &amp; Next Steps</h2>
            <div class="grid-2">
                <div>
                    <h3>Limitations</h3>
                    <ul class="notes">
                        <li>Độ dài chuỗi đầu vào giới hạn ở <code>max_length = 256</code> token, có thể cắt mất phần kết luận ở văn bản dài.</li>
                        <li>Chưa tối ưu ngưỡng phân loại (threshold tuning) theo từng lớp để cân bằng Precision/Recall.</li>
                        <li>Các câu chứa từ lóng, phủ định kép hoặc mỉa mai dễ bị dự đoán sai với độ tự tin cao.</li>
                    </ul>
                </div>
                <div>
                    <h3>Next Steps</h3>
                    <ul class="notes">
                        <li>Phân tích cụm lỗi trong <code>error_cases.json</code> để bổ sung dữ liệu huấn luyện cho các mẫu khó.</li>
                        <li>Thử nghiệm tăng <code>max_length = 512</code> hoặc fine-tune kiến trúc mạnh hơn (RoBERTa / DeBERTa).</li>
                        <li>Tự động gọi <code>build_report(run_dir)</code> ngay cuối pipeline G3 sau khi evaluation hoàn tất.</li>
                    </ul>
                </div>
            </div>
        </section>
    </div>
</body>
</html>
""")


def build_report(run_dir: str | Path) -> Path:
    """Tạo file report.html offline 1 file duy nhất trong thư mục run_dir."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    metrics_payload = load_json_safe(run_path / "metrics.json")
    error_cases_raw = load_json_safe(run_path / "error_cases.json")
    correct_cases_raw = load_json_safe(run_path / "correct_cases.json")

    run_id = html.escape(str((metrics_payload or {}).get("run_id", run_path.name)))
    ckpt_path = html.escape(str((metrics_payload or {}).get("checkpoint", "Chưa có checkpoint (chờ G2/G3)")))

    rendered_html = REPORT_TEMPLATE.safe_substitute(
        run_id=run_id,
        checkpoint_path=ckpt_path,
        run_dir=html.escape(str(run_path)),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics_section=_render_metrics_section(metrics_payload),
        plots_section=_render_plots_section(run_path, metrics_payload),
        samples_section=_render_samples_section(metrics_payload, error_cases_raw, correct_cases_raw),
    )

    report_path = run_path / "report.html"
    report_path.write_text(rendered_html, encoding="utf-8")

    size_bytes = report_path.stat().st_size
    size_kb = size_bytes / 1024
    print(f"[G4] Đã tạo: {report_path.resolve()} ({size_kb:.1f} KB)")
    if size_bytes >= MAX_REPORT_BYTES:
        print(f"[CẢNH BÁO] Kích thước report.html ({size_bytes / (1024*1024):.2f} MB) vượt ngưỡng 10MB!")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone offline HTML report (G4)")
    parser.add_argument("--run-dir", required=True, help="Thư mục kết quả run (ví dụ: results/runs/g1-model)")
    args = parser.parse_args()
    build_report(args.run_dir)


if __name__ == "__main__":
    main()