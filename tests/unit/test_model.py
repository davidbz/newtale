"""Unit tests for model/transformer.py — run on CPU, no GPU required."""
from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.transformer import NewTaleForCausalLM, RMSNorm, SwiGLU


@pytest.fixture()
def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=32,
    )


def test_rmsnorm_output_shape() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_norm() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_swiglu_shape(tiny_cfg: ModelConfig) -> None:
    ffn = SwiGLU(tiny_cfg)
    x = torch.randn(2, 8, 64)
    out = ffn(x)
    assert out.shape == x.shape


def test_model_forward_no_labels(tiny_cfg: ModelConfig) -> None:
    model = NewTaleForCausalLM(tiny_cfg)
    model.eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    loss, logits = model(ids)
    assert loss is None
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)


def test_model_forward_with_labels(tiny_cfg: ModelConfig) -> None:
    model = NewTaleForCausalLM(tiny_cfg)
    model.eval()
    ids = torch.randint(0, tiny_cfg.vocab_size, (2, 16))
    loss, logits = model(ids, labels=ids)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0
    assert logits.shape == (2, 16, tiny_cfg.vocab_size)


def test_model_loss_decreases_with_overfit(tiny_cfg: ModelConfig) -> None:
    """Model should overfit a single batch given enough steps."""
    model = NewTaleForCausalLM(tiny_cfg)
    model.train()
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss, _ = model(ids, labels=ids)
        assert loss is not None
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "loss did not decrease during overfitting"


def test_model_gradient_checkpointing(tiny_cfg: ModelConfig) -> None:
    model = NewTaleForCausalLM(tiny_cfg, gradient_checkpointing=True)
    model.train()
    ids = torch.randint(0, tiny_cfg.vocab_size, (1, 8))
    loss, _ = model(ids, labels=ids)
    assert loss is not None
    loss.backward()


def test_model_param_count(tiny_cfg: ModelConfig) -> None:
    model = NewTaleForCausalLM(tiny_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0
