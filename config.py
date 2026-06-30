from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from pathlib import Path


class ModelConfig(BaseModel):
    vocab_size: int = 100_000
    hidden_size: int = 3072
    num_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    rope_theta: float = 500_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


class DataSourceConfig(BaseModel):
    path: str
    weight: float
    name: str
    split: str = "train"
    subset: str | None = None  # HF dataset config name passed as `name` to load_dataset
    text_column: str = "text"  # column that holds the document text
    dedup: bool = False  # enable exact-hash dedup for this source


class DataConfig(BaseModel):
    tokenizer_dir: str
    sources: list[DataSourceConfig]
    seq_length: int = 4096
    seed: int = 42
    dedup_max_entries: int = 500_000  # per-source dedup cap; best-effort beyond this

    @field_validator("sources")
    @classmethod
    def weights_sum_to_one(cls, v: list[DataSourceConfig]) -> list[DataSourceConfig]:
        total = sum(s.weight for s in v)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Source weights must sum to 1.0, got {total:.4f}")
        return v


class TrainingConfig(BaseModel):
    output_dir: str
    max_steps: int
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    distributed_backend: str = "deepspeed"
    zero_stage: int = 2
    gradient_checkpointing: bool = True
    compile: bool = True
    compile_mode: str = "reduce-overhead"  # or "max-autotune" for production runs
    fp8_training: bool = False  # requires torchao; H100/H200/B-series only
    profile_steps: int | None = None  # export chrome trace for this many steps after warmup
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None
    seed: int = 42
    dataloader_num_workers: int = 4


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    wandb_project: str = "newtale"
    wandb_run_name: str | None = None
    tensorboard_dir: str | None = None


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)
