"""NewTale pretraining entrypoint.

Usage:
    python train.py --config configs/3b.yaml
    python train.py --config configs/tiny.yaml  # CPU smoke-test
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import load_config
from data.collator import DataCollatorForCLM
from data.dataset import PackedStreamingDataset
from model.transformer import NewTaleForCausalLM
from tokenizer.tokenizer import NewTaleTokenizer
from training.checkpoint import CheckpointManager
from training.distributed import (
    build_device_mesh,
    init_deepspeed,
    setup_distributed,
    wrap_fsdp,
)
from training.logging_utils import MetricsLogger
from training.optimizer import build_optimizer
from training.scheduler import build_scheduler
from training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    rank, local_rank, world_size = setup_distributed()
    _logger.info("Rank %d/%d (local=%d) initialised", rank, local_rank, world_size)

    # Per-rank seed: same model init, different data order
    set_seed(config.training.seed + rank)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = NewTaleTokenizer(config.data.tokenizer_dir)
    # Always sync model vocab to the actual tokenizer — avoids mismatch when
    # using a stand-in tokenizer (e.g. GPT-2) whose vocab differs from the YAML.
    config.model.vocab_size = tokenizer.vocab_size
    _logger.info("Tokenizer vocab_size: %d (model synced)", tokenizer.vocab_size)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = PackedStreamingDataset(
        sources=config.data.sources,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
        seed=config.data.seed,
        rank=rank,
        world_size=world_size,
        num_workers=config.training.dataloader_num_workers,
        dedup_max_entries=config.data.dedup_max_entries,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.per_device_train_batch_size,
        collate_fn=DataCollatorForCLM(),
        num_workers=config.training.dataloader_num_workers,
        pin_memory=torch.cuda.is_available()
        and config.training.dataloader_num_workers > 0,
        prefetch_factor=2 if config.training.dataloader_num_workers > 0 else None,
    )

    eval_dataset = PackedStreamingDataset(
        sources=config.data.sources,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
        seed=config.data.seed + 999,
        rank=rank,
        world_size=world_size,
        num_workers=config.training.dataloader_num_workers,
        dedup_max_entries=config.data.dedup_max_entries,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.training.per_device_train_batch_size,
        collate_fn=DataCollatorForCLM(),
        num_workers=config.training.dataloader_num_workers,
        pin_memory=torch.cuda.is_available()
        and config.training.dataloader_num_workers > 0,
    )

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    use_deepspeed = config.training.distributed_backend == "deepspeed"

    if use_deepspeed and config.training.zero_stage == 3:
        try:
            import deepspeed  # type: ignore[import-untyped]

            with deepspeed.zero.Init():
                model = NewTaleForCausalLM(
                    config.model,
                    gradient_checkpointing=config.training.gradient_checkpointing,
                )
        except ImportError:
            _logger.warning("deepspeed not available; falling back to standard init")
            model = NewTaleForCausalLM(
                config.model,
                gradient_checkpointing=config.training.gradient_checkpointing,
            )
    else:
        model = NewTaleForCausalLM(
            config.model,
            gradient_checkpointing=config.training.gradient_checkpointing,
        )
        if torch.cuda.is_available():
            model = model.cuda()

    param_count = sum(p.numel() for p in model.parameters())
    _logger.info("Model parameters: %.2fB", param_count / 1e9)

    # ------------------------------------------------------------------
    # Backend-specific wrapping: FSDP2 → FP8 (must precede compile)
    # ------------------------------------------------------------------
    if not use_deepspeed:
        if torch.cuda.is_available() and world_size > 1:
            device_mesh = build_device_mesh(world_size)
            wrap_fsdp(model, inner_modules=list(model.layers), device_mesh=device_mesh)
        elif torch.cuda.is_available():
            model = model.bfloat16()  # type: ignore[assignment]

        if config.training.fp8_training:
            try:
                from torchao.float8 import convert_to_float8_training  # type: ignore[import-untyped]

                model = convert_to_float8_training(model)  # type: ignore[assignment]
                _logger.info("FP8 training enabled via torchao")
            except ImportError:
                _logger.warning("torchao not installed; fp8_training=True ignored")

    # compile applies to both backends (FSDP2 already applied above)
    if config.training.compile:
        _logger.info("Compiling model (mode=%s)...", config.training.compile_mode)
        model = cast("nn.Module", torch.compile(model, mode=config.training.compile_mode))

    optimizer = build_optimizer(model, config.training)  # type: ignore[arg-type]
    scheduler = build_scheduler(optimizer, config.training)

    # ------------------------------------------------------------------
    # Checkpoint manager
    # ------------------------------------------------------------------
    ckpt_manager = CheckpointManager(
        config.training.output_dir, config.training.save_total_limit
    )
    start_step = 0
    resume_path = config.training.resume_from_checkpoint or ckpt_manager.find_latest()

    # ------------------------------------------------------------------
    # Metrics logger
    # ------------------------------------------------------------------
    metrics_logger = MetricsLogger(
        rank=rank,
        tensorboard_dir=config.logging.tensorboard_dir,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project,
        run_name=config.logging.wandb_run_name,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    tokens_per_step = (
        config.training.per_device_train_batch_size
        * world_size
        * config.training.gradient_accumulation_steps
        * config.data.seq_length
    )
    trainer = Trainer(
        config=config.training,
        train_loader=train_loader,
        eval_loader=eval_loader,
        checkpoint_manager=ckpt_manager,
        metrics_logger=metrics_logger,
        start_step=start_step,
        tokens_per_step=tokens_per_step,
    )

    if use_deepspeed:
        engine, _, _ = init_deepspeed(model, optimizer, scheduler, config.training)  # type: ignore[arg-type]
        if resume_path:
            tag = resume_path.split("/")[-1] if "/" in resume_path else resume_path
            state = ckpt_manager.load_deepspeed(engine, tag=tag)
            start_step = state.get("global_step", 0)
            trainer.start_step = start_step
        trainer.train_deepspeed(engine)
    else:
        if resume_path:
            state = ckpt_manager.load_fsdp(resume_path, model, optimizer, scheduler)
            start_step = state.get("global_step", 0)
            trainer.start_step = start_step
        trainer.train_fsdp(model, optimizer, scheduler)

    _logger.info("Training complete.")


if __name__ == "__main__":
    main()
