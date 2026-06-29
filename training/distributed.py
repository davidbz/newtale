from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from config import TrainingConfig


def setup_distributed() -> tuple[int, int, int]:
    """Initialise NCCL process group. Returns (rank, local_rank, world_size)."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return rank, local_rank, world_size


def generate_deepspeed_config(config: TrainingConfig) -> dict[str, Any]:
    zero_config: dict[str, Any] = {"stage": config.zero_stage}
    if config.zero_stage >= 2:
        zero_config["allgather_partitions"] = True
        zero_config["reduce_scatter"] = True
        zero_config["overlap_comm"] = True
    if config.zero_stage == 3:
        zero_config["stage3_prefetch_bucket_size"] = 5e7
        zero_config["stage3_param_persistence_threshold"] = 1e6

    return {
        "train_micro_batch_size_per_gpu": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.max_grad_norm,
        "bf16": {"enabled": True},
        "zero_optimization": zero_config,
        "steps_per_print": config.logging_steps,
    }


def write_deepspeed_config(config: TrainingConfig, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(generate_deepspeed_config(config), indent=2))
    return path


def init_deepspeed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: TrainingConfig,
) -> tuple[Any, Any, Any]:
    try:
        import deepspeed  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("deepspeed is required for the deepspeed backend") from e

    ds_config = generate_deepspeed_config(config)
    engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    return engine, ds_optimizer, ds_scheduler


def wrap_fsdp(model: torch.nn.Module) -> torch.nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=policy,
        device_id=torch.cuda.current_device(),
    )
