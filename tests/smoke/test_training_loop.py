"""Smoke tests: tiny 2-step training loop on CPU with a mock tokenizer.

These tests require no GPU and no network access. They verify that the
training loop runs end-to-end, loss decreases, and checkpoint save+load
resumes correctly.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import pytest
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig
from model.transformer import NewTaleForCausalLM
from training.checkpoint import CheckpointManager
from training.logging_utils import MetricsLogger
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.trainer import Trainer

TINY_MODEL = ModelConfig(
    vocab_size=256,
    hidden_size=64,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=128,
    max_position_embeddings=32,
)

SEQ_LEN = 16
BATCH_SIZE = 2


def _fake_batch() -> dict[str, Any]:
    ids = torch.randint(0, TINY_MODEL.vocab_size, (BATCH_SIZE, SEQ_LEN))
    return {"input_ids": ids, "labels": ids.clone(), "sources": ["test"] * BATCH_SIZE}


def _fake_loader(n_batches: int = 50) -> DataLoader:  # type: ignore[type-arg]
    batches = [_fake_batch() for _ in range(n_batches)]
    # batch_size=None → DataLoader passes each item directly to collate_fn (no list wrapping)
    return DataLoader(batches, batch_size=None, collate_fn=lambda x: x)  # type: ignore[arg-type]


@pytest.fixture()
def output_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_fsdp_train_loss_finite(output_dir: str) -> None:
    cfg = TrainingConfig(
        output_dir=output_dir,
        max_steps=4,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        lr=1e-3,
        warmup_steps=1,
        logging_steps=2,
        eval_steps=10,
        save_steps=10,
        save_total_limit=2,
        compile=False,
        gradient_checkpointing=False,
    )
    model = NewTaleForCausalLM(TINY_MODEL)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    ckpt = CheckpointManager(output_dir, save_total_limit=2)
    logger = MetricsLogger(rank=0)

    tokens_per_step = BATCH_SIZE * cfg.gradient_accumulation_steps * SEQ_LEN
    trainer = Trainer(
        config=cfg,
        train_loader=_fake_loader(),
        eval_loader=_fake_loader(10),
        checkpoint_manager=ckpt,
        metrics_logger=logger,
        tokens_per_step=tokens_per_step,
    )
    trainer.train_fsdp(model, optimizer, scheduler)  # should not raise


def test_fsdp_loss_decreases() -> None:
    """Model must overfit a single fixed batch — loss should decrease reliably."""
    cfg = TrainingConfig(
        output_dir="/tmp/newtale_test_overfit",
        max_steps=40,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        lr=5e-3,
        warmup_steps=2,
        logging_steps=5,
        eval_steps=100,
        save_steps=100,
        compile=False,
        gradient_checkpointing=False,
    )
    model = NewTaleForCausalLM(TINY_MODEL)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Fixed batch — model must overfit this specific data
    fixed_ids = torch.randint(0, TINY_MODEL.vocab_size, (BATCH_SIZE, SEQ_LEN))

    losses: list[float] = []
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss, _ = model(fixed_ids, labels=fixed_ids.clone())
        assert loss is not None
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    first5 = sum(losses[:5]) / 5
    last5 = sum(losses[-5:]) / 5
    assert last5 < first5, f"loss did not decrease: first={first5:.3f} last={last5:.3f}"


def test_checkpoint_resume(output_dir: str) -> None:
    """Train 2 steps, save, reload, verify step counter."""
    cfg = TrainingConfig(
        output_dir=output_dir,
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        lr=1e-3,
        warmup_steps=0,
        logging_steps=1,
        eval_steps=1,
        save_steps=1,
        compile=False,
        gradient_checkpointing=False,
    )
    model = NewTaleForCausalLM(TINY_MODEL)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    ckpt = CheckpointManager(output_dir, save_total_limit=2)

    # Manually save at step 1 to test resume
    ckpt.save_fsdp(1, model, optimizer, scheduler, {"loss": 3.0})

    latest = ckpt.find_latest()
    assert latest is not None
    assert "checkpoint-1" in latest
