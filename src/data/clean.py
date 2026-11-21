"""Module data: load IMDb thô, làm sạch, chia splits.

Quy ước: data/raw bất biến, data/processed đã làm sạch,
data/splits chứa train/val/test cố định bằng seed.
"""

import re
import html as html_lib

HTML_TAG_RE = re.compile(r"<.*?>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def clean_text(s: str) -> str:
    """Chuẩn hoá văn bản review: gỡ HTML/URL, khoảng trắng thừa, lower giữ nguyên ý.

    Giữ chữ hoa/thường gốc cho tokenizer cased sau này? Ở đây lower để EDA ổn định.
    """
    if s is None:
        return ""
    s = str(s)
    s = html_lib.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def is_empty(s: str) -> bool:
    return len(s.strip()) == 0
