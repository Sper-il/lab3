from src.data.clean import clean_text
from src.data.split import make_splits
import pandas as pd


def test_clean_text_go_html_url():
    s = "<br />I loved it! Visit https://example.com  GREAT   "
    assert clean_text(s) == "i loved it! visit great"


def test_clean_text_none_ve_rong():
    assert clean_text(None) == ""
    assert clean_text("   ") == ""


def test_make_splits_giu_phan_bo():
    df = pd.DataFrame({"text": [f"mau {i}" for i in range(200)], "label": [0] * 100 + [1] * 100})
    train, val, test = make_splits(df, (0.8, 0.1, 0.1), seed=42)
    assert len(train) == 160 and len(val) == 20 and len(test) == 20
    for d in (train, val, test):
        ratio = d["label"].mean()
        assert abs(ratio - 0.5) < 0.02
