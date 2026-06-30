"""Unit tests for config.py."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import DataConfig, DataSourceConfig, ModelConfig, TrainingConfig


def _source(weight: float, name: str = "test") -> DataSourceConfig:
    return DataSourceConfig(path="dummy/path", weight=weight, name=name)


def test_weights_sum_to_one() -> None:
    cfg = DataConfig(
        tokenizer_dir="tok/",
        sources=[_source(0.7, "a"), _source(0.3, "b")],
    )
    assert len(cfg.sources) == 2


def test_weights_not_summing_raises() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        DataConfig(
            tokenizer_dir="tok/",
            sources=[_source(0.5, "a"), _source(0.3, "b")],
        )


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.vocab_size == 100_000
    assert cfg.rope_theta == 500_000.0


def test_model_config_computed_properties() -> None:
    cfg = ModelConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    assert cfg.head_dim == 16
    assert cfg.num_kv_groups == 2


def test_training_config_new_fields() -> None:
    cfg = TrainingConfig(output_dir="out", max_steps=100)
    assert cfg.compile_mode == "reduce-overhead"
    assert cfg.fp8_training is False
    assert cfg.profile_steps is None
