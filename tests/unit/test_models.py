"""Unit tests cho module models (Gói G1).

Các test này KHÔNG tải model từ Hub (offline-friendly) — chỉ kiểm tra logic
tokenize/loss/shape với tokenizer giả lập, trừ test có đánh dấu `network`.
"""

import numpy as np
import pytest
import torch

from src.models.loss import build_loss, compute_loss
from src.models.tokenizer_utils import tokenize_batch


class DummyTokenizer:
    """Tokenizer giả lập: mỗi token = 1 id, đủ để kiểm tra shape/padding."""

    def __call__(self, texts, padding=None, truncation=None, max_length=None,
                 return_tensors=None):
        n = len(texts)
        ids = torch.ones((n, max_length), dtype=torch.long)
        mask = torch.ones((n, max_length), dtype=torch.long)
        if padding == "max_length":
            # giả lập 1 sample ngắn hơn -> có padding 0
            ids[0, 10:] = 0
            mask[0, 10:] = 0
        return {"input_ids": ids, "attention_mask": mask}


def test_tokenize_batch_shape():
    tok = DummyTokenizer()
    out = tokenize_batch(["a b c", "d e"], tok, max_length=16)
    assert out["input_ids"].shape == (2, 16)
    assert out["attention_mask"].shape == (2, 16)
    assert out["input_ids"].dtype == torch.int64


def test_tokenize_batch_padding_mask_is_zero():
    tok = DummyTokenizer()
    out = tokenize_batch(["a b c", "d e"], tok, max_length=16)
    assert out["attention_mask"][0, 10:].sum().item() == 0
    assert out["attention_mask"][1].sum().item() == 16


def test_tokenize_batch_to_device():
    tok = DummyTokenizer()
    out = tokenize_batch(["a"], tok, max_length=8, device="cpu")
    assert out["input_ids"].device.type == "cpu"


def test_build_loss_is_cross_entropy():
    loss_fn = build_loss({"loss": {"name": "cross_entropy"}})
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)


def test_build_loss_unsupported_raises():
    with pytest.raises(ValueError):
        build_loss({"loss": {"name": "focal"}})


def test_compute_loss_value():
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    labels = torch.tensor([0, 1])
    loss = compute_loss(logits, labels)
    assert loss.item() < 1e-3
    assert np.isfinite(loss.item())


def test_compute_loss_gradient_flows():
    logits = torch.randn(4, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0, 1])
    loss = compute_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
