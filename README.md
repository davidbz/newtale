# NewTale

Pretraining pipeline for a decoder-only Transformer. No instruction tuning, no RLHF.

## What's included

| Component | Location | Description |
|---|---|---|
| Model | `model/` | RMSNorm, RoPE, Grouped-Query Attention, SwiGLU FFN |
| Data | `data/` | Token-budgeted mixing, NFKC normalization, exact dedup, sequence packing |
| Tokenizer | `tokenizer/` | 100k-vocab byte-level BPE trainer + wrapper |
| Training | `training/` | AdamW, cosine+warmup, FSDP2 (default) and DeepSpeed ZeRO-2/3 |
| Configs | `configs/` | `3b.yaml`, `1b.yaml`, `1b-single-gpu.yaml`, `small.yaml`, `tiny.yaml` |
| Scripts | `scripts/` | SLURM launcher, benchmark eval, HF conversion |

## Architecture (3B default)

```
vocab_size=100k  hidden=3072  layers=28  heads=24  kv_heads=8  ffn=8192  ctx=4096
~3.13B parameters
```

- **Attention**: GQA (24Q / 8KV heads), flash-attn when available, SDPA fallback
- **FFN**: SwiGLU
- **Norm**: Pre-norm RMSNorm, no bias
- **Positional**: RoPE (θ=500 000)
- **Precision**: bfloat16; optional FP8 via `fp8_training: true` (requires torchao, H100+)

Data is streamed from HuggingFace, filtered, hash-deduplicated, and packed into non-overlapping 4096-token chunks with no padding.

## Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt

# Optional but recommended
uv pip install flash-attn --no-build-isolation
uv pip install wandb torchao
```

## Running

**Train tokenizer** (requires `HF_TOKEN`):
```bash
python -m tokenizer.train_tokenizer --output_dir tokenizer/ --vocab_size 100000
```

**Smoke test** (CPU, no GPU):
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2').save_pretrained('tokenizer/')"
python train.py --config configs/tiny.yaml
```

**Single-node 8-GPU (FSDP2)**:
```bash
torchrun --nproc_per_node=8 train.py --config configs/3b.yaml
```

**Multi-node (SLURM)**:
```bash
sbatch scripts/launch_slurm.sh
```

**Benchmarks** (HellaSwag, ARC, MMLU, Winogrande):
```bash
python scripts/eval_benchmarks.py --checkpoint checkpoints/checkpoint-best --config configs/3b.yaml
```

## Scale estimates

| Tokens | Steps | Wall time (64× A100) |
|---|---|---|
| 10B | ~1 200 | ~6 h |
| 50B | ~6 000 | ~30 h |
| 300B | ~35 700 | ~7.5 days |

Effective batch size: 2 048 sequences × 4 096 tokens ≈ **8.4M tokens/step** (64 GPUs, `per_device_batch=2`, `grad_accum=16`).
