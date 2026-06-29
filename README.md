# NewTale 3B

Pretraining pipeline for a 3-billion-parameter decoder-only Transformer trained from scratch on FineWeb-Edu + StarCoder data using causal language modelling. No instruction tuning, no RLHF.

## What's included

| Component | Location | Description |
|---|---|---|
| Model | `model/` | RMSNorm, RoPE, Grouped-Query Attention, SwiGLU FFN |
| Data | `data/` | Token-budgeted mixing, NFKC normalization, exact dedup, sequence packing |
| Tokenizer | `tokenizer/` | 50k-vocab byte-level BPE trainer + wrapper |
| Training | `training/` | AdamW, cosine+warmup schedule, DeepSpeed ZeRO-2/3 and FSDP backends |
| Configs | `configs/` | `3b.yaml`, `1b.yaml`, `tiny.yaml` (CPU smoke-test) |
| Scripts | `scripts/` | Multi-node launcher, dataset sharding |

## Architecture at a glance

```
vocab_size=50k  hidden=3072  layers=28  heads=24  kv_heads=8  ffn=8192  ctx=4096
~3.13B parameters
```

- **Attention**: Grouped-Query Attention (24Q / 8KV heads), flash-attn when available, SDPA fallback
- **FFN**: SwiGLU (`down(silu(gate(x)) * up(x))`)
- **Norm**: Pre-norm RMSNorm, no bias
- **Positional**: RoPE (θ=10 000), applied to Q and K before attention
- **Precision**: bfloat16 throughout; gradient accumulation in fp32

Data is streamed from HuggingFace, filtered, hash-deduplicated, and packed into non-overlapping 4096-token chunks with no padding (100% token utilisation). The mixer tracks tokens per source so the 70/30 FineWeb-Edu / StarCoder split is accurate at the token level, not the sample level.

## Setup

```bash
# Clone and create environment
uv venv --python 3.12 && source .venv/bin/activate

# Install PyTorch with your CUDA version (check: nvidia-smi)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Core dependencies
uv pip install -r requirements.txt

# Optional but strongly recommended
uv pip install flash-attn --no-build-isolation
uv pip install wandb
```

## Running

**Train the tokenizer** (requires `HF_TOKEN`):
```bash
export HF_TOKEN=hf_...
python -m tokenizer.train_tokenizer --output_dir tokenizer/ --vocab_size 50000 --train_size_gb 10
```

**Smoke test** (CPU, 10 steps, no GPU needed):
```bash
# Quick stand-in tokenizer
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2').save_pretrained('tokenizer/')"

python train.py --config configs/tiny.yaml
```

**Single-node 8-GPU run**:
```bash
bash scripts/launch_training.sh --config configs/3b.yaml --backend deepspeed --num_gpus 8
```

**Multi-node**:
```bash
MASTER_ADDR=node0 NNODES=8 bash scripts/launch_training.sh \
  --config configs/3b.yaml --backend deepspeed --num_gpus 8
```

**Evaluation**:
```bash
python eval.py --config configs/3b.yaml \
               --checkpoint checkpoints/3b/checkpoint-best \
               --eval_file data/validation.jsonl
```

## Scalability

| Token budget | Steps | Wall time (64× A100) |
|---|---|---|
| 10B | ~1 200 | ~6 h |
| 50B | ~6 000 | ~30 h |
| 300B | ~35 700 | ~7.5 days |

Effective batch size: 2 048 sequences × 4 096 tokens = **~8.4M tokens/step** (64 GPUs, `per_device_batch=2`, `grad_accum=16`).
