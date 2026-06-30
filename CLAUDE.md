# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (uv required)
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# Lint / format
uv run ruff check .
uv run ruff format .

# Type check
uv run pyright

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_model.py -x -q

# Run a single test by name
uv run pytest tests/smoke/test_training_loop.py::test_checkpoint_resume -x

# Training (single GPU)
python train.py --config configs/1b-single-gpu.yaml

# Training (multi-GPU, FSDP2)
torchrun --nproc_per_node=8 train.py --config configs/3b.yaml
```

## Architecture

See `ARCHITECTURE.md` for a detailed breakdown with diagrams. Key points:

**Entrypoint**: `train.py` — loads config, builds data loaders, wraps model with FSDP2 or DeepSpeed, compiles, creates optimizer/scheduler, then delegates to `Trainer`.

**Config**: `config.py` — three nested pydantic models (`ModelConfig`, `DataConfig`, `TrainingConfig`) + `LoggingConfig`. YAML files in `configs/` map directly to these. `DataConfig.sources` is a list of `DataSourceConfig` entries with per-source `weight`, `text_column`, and optional `dedup`. `load_config()` merges YAML with CLI overrides.

**Data pipeline** (`data/`):
- `PackedStreamingDataset` — streams from HF Hub with `streaming=True`, shards across ranks/workers (`total_shards = world_size × num_workers`), runs the preprocessing filter chain, then packs documents end-to-end (with EOS between them) into fixed-length `seq_length` chunks. No padding.
- `WeightedDatasetMixer` — token-budgeted interleaving (not sample-budgeted) so configured weights reflect actual token proportions.
- `DataCollatorForCLM` — stacks chunks into `{input_ids, labels, sources}`; `labels = input_ids.clone()` because the model shifts internally.

**Model** (`model/`): Decoder-only Transformer (LLaMA-3 style): RMSNorm pre-norm, GQA, SwiGLU FFN, RoPE (θ=500 000). `NewTaleForCausalLM.forward()` returns `(loss, logits)`. Shift (`logits[:, :-1]` vs `labels[:, 1:]`) is done inside the model.

**Distributed** (`training/distributed.py`):
- FSDP2 path: `build_device_mesh` → `wrap_fsdp(model, inner_modules=list(model.layers), device_mesh=...)`. Inner layers are sharded first, then the root model. Must happen **before** `torch.compile`.
- DeepSpeed path: engine is built in `train.py` via `deepspeed.initialize`; config from `configs/deepspeed_zero{2,3}.json`.

**Checkpointing** (`training/checkpoint.py`):
- FSDP2: `save_fsdp` / `load_fsdp` use `torch.distributed.checkpoint` (DCP) — per-rank shard files, no rank-0 OOM. Scheduler/RNG/trainer state saved by rank 0 separately.
- Eval-only weight loading: `load_weights(path, model)` — no optimizer/scheduler needed.
- `find_latest()` scans `checkpoint-{N}` dirs; `checkpoint-best/` is never rotated.

**Trainer** (`training/trainer.py`): `Trainer` owns the loop for both backends (`train_fsdp`, `train_deepspeed`). NaN/Inf detection raises `TrainingInstabilityError` after 3 consecutive bad steps. `tokens_per_step` is passed in from `train.py` (computed as `batch × world_size × grad_accum × seq_length`).

## Key constraints

- **compile ordering**: `torch.compile` must come after FSDP2 wrapping in `train.py`; `model.layers` is inaccessible once `OptimizedModule` wraps the model.
- **`batch_size=None` in DataLoader**: smoke tests pass `batch_size=None` so items are not wrapped in a list by the DataLoader; `collate_fn=lambda x: x` passes items directly.
- **No DeepSpeed optimizer in eval**: `eval_benchmarks.py` uses `ckpt.load_weights()`, not `load_fsdp()`.
- **FP8**: `fp8_training: true` in config requires `torchao` and H100+. Applied after FSDP2, before compile.
