# NewTale 3B — Pretraining Pipeline Spec

## Overview

Phase 1: Train a 3-billion-parameter decoder-only Transformer from scratch on
FineWeb-Edu + code data using causal language modelling (CLM). No instruction
tuning, no RLHF.

---

## 1. Repository Layout

```
newtale/
├── configs/
│   ├── 3b.yaml                 # production config (~3.1B params)
│   ├── 1b.yaml                 # GPU dev config
│   ├── tiny.yaml               # CPU smoke-test (2 layers, hidden 128)
│   ├── deepspeed_zero2.json    # ZeRO-2 template (generated at runtime)
│   └── deepspeed_zero3.json    # ZeRO-3 template
├── model/
│   ├── __init__.py
│   ├── config.py               # ModelConfig dataclass
│   ├── rope.py                 # RotaryEmbedding + apply_rotary_emb
│   ├── attention.py            # GroupedQueryAttention (FlashAttn2 / SDPA fallback)
│   └── transformer.py          # RMSNorm, SwiGLU, TransformerBlock, NewTaleForCausalLM
├── data/
│   ├── __init__.py
│   ├── preprocessing.py        # normalize, filter, HTML strip, repetition detect, hash dedup
│   ├── dataset.py              # StreamingTextDataset + PackedStreamingDataset
│   ├── mixing.py               # WeightedDatasetMixer (token-budgeted)
│   └── collator.py             # DataCollatorForCLM
├── tokenizer/
│   ├── __init__.py
│   ├── train_tokenizer.py      # train BPE from stratified streaming corpus
│   └── tokenizer.py            # NewTaleTokenizer wrapper
├── training/
│   ├── __init__.py
│   ├── optimizer.py            # AdamW with param-group splits
│   ├── scheduler.py            # cosine + linear-warmup LR schedule
│   ├── checkpoint.py           # save / load / best-checkpoint / rotation
│   ├── logging_utils.py        # MetricsLogger (console + TensorBoard + W&B, per-dataset loss)
│   ├── distributed.py          # DeepSpeed / FSDP setup + ds_config generation
│   └── trainer.py              # Trainer: training loop, eval, grad accum, NaN detection
├── scripts/
│   ├── launch_training.sh      # torchrun / deepspeed launcher
│   └── prepare_data.sh         # shard data to local disk before training
├── config.py                   # top-level Config (pydantic) + YAML loader
├── train.py                    # entrypoint: python train.py --config configs/3b.yaml
├── eval.py                     # perplexity + prompt completions
├── requirements.txt
└── README.md
```

---

## 2. Key Design Decisions

### 2.1 Config loading — pydantic

Plain `yaml.safe_load` → `@dataclass` doesn't work without manual unpacking.
Use `pydantic` `BaseModel` with `model_validate` — YAML dict maps cleanly, gives
free type coercion, and validates at load time.

```python
class ModelConfig(BaseModel):
    vocab_size: int = 50_000
    hidden_size: int = 3072
    num_layers: int = 28
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    max_position_embeddings: int = 4096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

class DataConfig(BaseModel):
    tokenizer_dir: str
    sources: list[dict]          # [{path, weight, name}, ...]
    seq_length: int = 4096
    seed: int = 42

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
    distributed_backend: str = "deepspeed"   # "deepspeed" | "fsdp"
    zero_stage: int = 2
    gradient_checkpointing: bool = True
    compile: bool = True                     # torch.compile (disable for debugging)
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None
    seed: int = 42

class LoggingConfig(BaseModel):
    use_wandb: bool = False
    wandb_project: str = "newtale"
    tensorboard_dir: str | None = None

class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    data: DataConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()

def load_config(path: str) -> Config:
    with open(path) as f:
        return Config.model_validate(yaml.safe_load(f))
```

### 2.2 DeepSpeed-specific paths

DeepSpeed diverges from plain PyTorch in four places that must be handled separately:

1. **ZeRO-3 model init**: wrap model construction in `deepspeed.zero.Init()` — never materialize the full model before sharding.
2. **Loss scaling**: DeepSpeed manages `gradient_accumulation_steps` scaling internally. Never divide loss manually when using the DS engine.
3. **Grad clipping**: configured in the DS JSON (`"gradient_clipping"`), not called separately in the loop.
4. **Scheduler stepping**: pass scheduler to `deepspeed.initialize()` — engine calls `scheduler.step()` after each `engine.step()`. Do NOT also call it in the loop.

### 2.3 Checkpointing — two paths

- **DeepSpeed**: `engine.save_checkpoint(output_dir, tag=f"checkpoint-{step}")`. For eval, run `zero_to_fp32.py` to consolidate, or use `engine.save_16bit_model()`.
- **FSDP**: `FULL_STATE_DICT` policy on rank 0, then `torch.save`.

### 2.4 Flash-attn tensor layout

Flash-attn expects `(batch, seqlen, nheads, head_dim)`.
SDPA expects `(batch, nheads, seqlen, head_dim)`.
The attention module transposes explicitly before each path.

### 2.5 bf16 + FSDP — no gradient scaler

bf16 has sufficient dynamic range for LLM training. `GradScaler` is for fp16 only. FSDP path: plain `loss.backward()`.

### 2.6 Tokenizer — single API

Use `tokenizers` low-level API (`Tokenizer` + `BpeTrainer` + `ByteLevel` pre-tokenizer). Save via `tokenizer.save("tokenizer.json")`. Wrap in `PreTrainedTokenizerFast` for the Python interface. Do NOT mix with the high-level `ByteLevelBPETokenizer` class.

### 2.7 Mixing — token-budgeted, not sample-budgeted

FineWeb-Edu documents average ~1 500 tokens; StarCoder files can be as short as 20 tokens. Sample-based 70/30 weighting produces a very different actual token ratio. `WeightedDatasetMixer` tracks tokens yielded per source and selects the source with the largest token deficit, so configured weights accurately reflect the token distribution.

### 2.8 Tokenizer training — stratified sampling

Taking the first N bytes from a streaming HF dataset is biased by internal shard order. The tokenizer trainer interleaves multiple shards from each dataset (round-robin) before feeding the byte counter, ensuring the vocabulary reflects a uniform draw across the corpus.

### 2.9 Dedup — exact hash set

Use `xxhash.xxh64(text.encode()).hexdigest()` → Python `set`. This is exact dedup, not a bloom filter. For near-dedup at scale, `datasketch` MinHash is a separate optional step.

---

## 3. Model Architecture

### 3.1 Parameter Budget

| Hyperparameter           | 3B value    | Notes                          |
|--------------------------|-------------|--------------------------------|
| `vocab_size`             | 50 000      | trained BPE tokenizer          |
| `hidden_size`            | 3 072       |                                |
| `num_layers`             | 28          | yields ~3.13B total params     |
| `num_attention_heads`    | 24          | head_dim = 128                 |
| `num_key_value_heads`    | 8           | GQA — 3× KV-cache reduction    |
| `intermediate_size`      | 8 192       | SwiGLU gate + up projections   |
| `max_position_embeddings`| 4 096       |                                |
| `rope_theta`             | 10 000      |                                |
| `rms_norm_eps`           | 1e-5        |                                |
| `tie_word_embeddings`    | false       |                                |

```
embed_tokens          :   50 000 × 3 072  = 153.6 M
28 layers
  attn (q+k+v+o)     :  (9.44 + 3.15 + 3.15 + 9.44) M × 28 = 704.6 M
  ffn  (gate+up+down):   25.2 M × 3 × 28                    = 2 116.8 M
  norms               :   negligible
lm_head (untied)      :   50 000 × 3 072  = 153.6 M
─────────────────────────────────────────────────────────
TOTAL                 ≈  3 128 M  (~3.13B)
```

### 3.2 `model/rope.py`

```python
class RotaryEmbedding(nn.Module):
    # precompute cos/sin tables up to max_position_embeddings as buffers
    # forward(seq_len) → cos, sin sliced to seq_len
    # extend cache on demand if seq_len exceeds precomputed length

def apply_rotary_emb(q, k, cos, sin):
    # q, k: (batch, nheads, seqlen, head_dim)
    # rotate_half pattern: split last dim into 2 halves
    # q_rot = q * cos + rotate_half(q) * sin  (same for k)
```

### 3.3 `model/attention.py` — GroupedQueryAttention

```python
# q_proj: hidden → num_attention_heads * head_dim  (bias=False)
# k_proj: hidden → num_key_value_heads * head_dim  (bias=False)
# v_proj: hidden → num_key_value_heads * head_dim  (bias=False)
# o_proj: num_attention_heads * head_dim → hidden  (bias=False)

def forward(self, x, cos, sin):
    B, T, _ = x.shape
    q = q_proj(x).view(B, T, num_heads, head_dim)      # (B,T,H,D)
    k = k_proj(x).view(B, T, num_kv_heads, head_dim)
    v = v_proj(x).view(B, T, num_kv_heads, head_dim)

    q, k = apply_rotary_emb(q, k, cos, sin)            # RoPE on (B,T,H,D)

    if flash_attn_available:
        # flash_attn_func expects (B,T,H,D) — already correct
        out = flash_attn_func(q, k, v, causal=True)    # native GQA
    else:
        # SDPA expects (B,H,T,D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2).repeat_interleave(num_kv_groups, dim=1)
        v = v.transpose(1, 2).repeat_interleave(num_kv_groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2)                       # back to (B,T,H,D)

    return o_proj(out.reshape(B, T, num_heads * head_dim))
```

### 3.4 `model/transformer.py`

- **RMSNorm**: `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight`
- **SwiGLU**: `down_proj(F.silu(gate_proj(x)) * up_proj(x))`
- **TransformerBlock**: pre-norm → GQA → residual → pre-norm → SwiGLU → residual; wraps with `checkpoint(..., use_reentrant=False)` when `gradient_checkpointing=True`
- **NewTaleForCausalLM**:
  - `embed_tokens: nn.Embedding(vocab_size, hidden_size)`
  - N `TransformerBlock`s, final `RMSNorm`, `lm_head = nn.Linear(hidden_size, vocab_size, bias=False)` (untied)
  - `forward(input_ids, labels)`: shift labels inside model — `loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, V), labels[..., 1:].reshape(-1))`. Returns `(loss, logits)`.

---

## 4. Tokenizer

- Algorithm: **Byte-level BPE** via `tokenizers` low-level API
- Vocabulary: 50 000 tokens
- Special tokens: `<unk>`, `<s>` (BOS), `</s>` (EOS), `<pad>`
- Saved format: `tokenizer/tokenizer.json`
- Training corpus: stratified interleave across shards of FineWeb-Edu (10 GB) + StarCoder (2 GB)

```python
# tokenizer/train_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel()
trainer = BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
)
# text_iterator: interleaves N shards per dataset, counts bytes, stops at train_size_gb
tokenizer.train_from_iterator(text_iterator, trainer=trainer)
tokenizer.save(str(output_dir / "tokenizer.json"))
```

Requires `HF_TOKEN` env var — raises `EnvironmentError` clearly if missing.

Train command:
```bash
python -m tokenizer.train_tokenizer \
    --output_dir tokenizer/ \
    --vocab_size 50000 \
    --train_size_gb 10
```

---

## 5. Data Pipeline

### 5.1 Sources and Mix

| Dataset                         | HF Path                          | Default weight |
|---------------------------------|----------------------------------|----------------|
| FineWeb-Edu (sample-10BT)       | `HuggingFaceFW/fineweb-edu`      | 0.70           |
| StarCoder data (code)           | `bigcode/starcoderdata`          | 0.30           |

Weights are token-budgeted: the mixer tracks tokens yielded per source and always
pulls from the source with the largest token deficit vs its target fraction.
This makes the configured weights accurate at the token level, not the sample level.

### 5.2 Preprocessing (`data/preprocessing.py`)

Pure functions applied in order via a single `preprocess(text, dedup) -> str | None` entry point:

1. **Normalize** — NFKC Unicode normalization, strip C0/C1 control chars (preserve `\t \n \r`), collapse `>2` blank lines
2. **Length filter** — drop if `len(text) < 200` or `len(text) > 100 000`
3. **HTML junk** — strip tags with regex; drop if > 20% of chars are inside `<…>`
4. **Repetition detector** — top character 5-gram frequency; drop if `count(top_5gram) / total_5grams > 0.3`
5. **Exact dedup** — `xxhash.xxh64` hash set; skip if hash seen this run

Note: FineWeb-Edu is already cleaned by HuggingFace (language-filtered, near-deduped, quality-scored). These filters are a lightweight safety net, not a primary cleaning stage.

### 5.3 Mixing (`data/mixing.py`)

```python
class WeightedDatasetMixer:
    def __init__(self, datasets, weights, names, seed=42):
        self._rng = random.Random(seed)          # instance RNG, not global
        self._iters = [iter(itertools.cycle(ds)) for ds in datasets]
        self._targets = weights
        self._names = names
        self._token_counts = [0] * len(datasets)

    def __iter__(self):
        while True:
            total = sum(self._token_counts) or 1
            deficits = [self._targets[i] - self._token_counts[i] / total
                        for i in range(len(self._iters))]
            idx = deficits.index(max(deficits))
            sample = next(self._iters[idx])
            sample["_source"] = self._names[idx]
            self._token_counts[idx] += len(sample.get("text", "")) // 4  # approx
            yield sample
```

### 5.4 Tokenisation and Packing (`data/dataset.py`)

`PackedStreamingDataset` wraps the mixer:
- Tokenizes each document, appends `</s>` (EOS)
- Appends tokens to a rolling buffer; tracks source tag per token
- Yields non-overlapping `seq_length=4096` chunks (no padding — 100% token utilisation)
- Each yielded item: `{"input_ids": tensor, "source": dominant_source_name}`
- Per-worker RNG re-seeding via `get_worker_info().id` for independent streams

### 5.5 Collation (`data/collator.py`)

```python
@dataclass
class DataCollatorForCLM:
    def __call__(self, features):
        input_ids = torch.stack([f["input_ids"] for f in features])
        sources = [f["source"] for f in features]   # passed through for per-dataset loss
        return {"input_ids": input_ids, "labels": input_ids.clone(), "sources": sources}
```

DataLoader config: `num_workers=4`, `pin_memory=True`, `prefetch_factor=2`.

---

## 6. Training Infrastructure

### 6.1 Distributed Training

Two backends, selectable via `training.distributed_backend`:

| Backend      | When to use                              |
|--------------|------------------------------------------|
| DeepSpeed    | Default; ZeRO-2/3, overlapped comms      |
| PyTorch FSDP | No extra dependencies; multi-node clean  |

`setup_distributed()` must call `torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))` before `dist.init_process_group("nccl")`.

**DeepSpeed config generated at runtime** by `generate_deepspeed_config(config)`:
```python
{
    "train_micro_batch_size_per_gpu": per_device_train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "gradient_clipping": max_grad_norm,   # DS handles clipping; do NOT clip manually
    "bf16": {"enabled": True},
    "zero_optimization": {"stage": zero_stage, ...}
}
```

**ZeRO-3 model init** — wrap in `deepspeed.zero.Init()` before constructing the model to avoid momentarily materializing the full 3B model on every GPU.

**FSDP**: `ShardingStrategy.FULL_SHARD` + `MixedPrecision(param_dtype=bfloat16)`. No `GradScaler` — bf16 does not need loss scaling.

### 6.2 Optimiser

**AdamW** with two parameter groups:
- Group A (weight matrices, `ndim >= 2`, not embedding): `weight_decay=0.1`
- Group B (norms, biases, embeddings): `weight_decay=0.0`

`lr=3e-4`, `betas=(0.9, 0.95)`, `eps=1e-8`

### 6.3 Learning Rate Schedule

Linear warmup for `warmup_steps` (default 2 000), then cosine decay to `min_lr = lr × min_lr_ratio` (default 0.1) over `max_steps`.

```python
def lr_lambda(step):
    if step < warmup: return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine
```

With DeepSpeed: pass scheduler to `deepspeed.initialize()` — engine steps it automatically. Do NOT call `scheduler.step()` in the loop.

### 6.4 `torch.compile`

Applied after model construction, before `deepspeed.initialize()` or `FSDP()`:

```python
if config.training.compile:
    model = torch.compile(model, mode="reduce-overhead")
```

Controlled by `compile: bool = True` in `TrainingConfig`. Disable for debugging.

### 6.5 Batch Size

| Variable                          | 3B run |
|-----------------------------------|--------|
| `per_device_train_batch_size`     | 2      |
| `gradient_accumulation_steps`     | 16     |
| GPUs (target)                     | 64     |
| Effective batch size (sequences)  | 2 048  |
| Tokens per step                   | ~8.4 M |

---

## 7. Training Loop

### DeepSpeed

```python
nan_count = 0
for global_step in range(start_step, max_steps):
    for micro_step in range(grad_accum_steps):
        batch = next(data_iter)
        sources = batch.pop("sources")
        loss, _ = engine(**batch)               # DS scales loss internally
        engine.backward(loss)
        for src in sources:
            logger.record_source_loss(src, loss.item())

    if torch.isnan(loss) or torch.isinf(loss):
        nan_count += 1
        logger.log(global_step, {"nan_count": nan_count})
        if nan_count >= 3:
            raise TrainingInstabilityError(f"step {global_step}")
        engine.zero_grad(); continue
    nan_count = 0

    engine.step()                               # grad clip + optimizer + scheduler (all DS-managed)

    if global_step % logging_steps == 0:
        logger.log(global_step, {
            "loss": loss.item(),
            "lr": engine.get_lr()[0],
            "grad_norm": engine.get_global_grad_norm(),
        })
    if global_step % eval_steps == 0:
        ppl = evaluate(eval_loader)
        logger.log(global_step, {"perplexity": ppl})
        checkpoint.maybe_save_best(ppl)
    if global_step % save_steps == 0:
        checkpoint.save(global_step)
```

### FSDP

Same structure, but:
- `(loss / grad_accum_steps).backward()` — manual scaling
- `clip_grad_norm_(model.parameters(), max_grad_norm)` — manual clipping
- `optimizer.step(); scheduler.step(); optimizer.zero_grad()` — manual stepping
- No `GradScaler`

---

## 8. Checkpointing

Directory per checkpoint: `{output_dir}/checkpoint-{step}/`

```
checkpoint-5000/
├── model/          # state_dict (FSDP) or DS sharded files
├── optimizer/
├── scheduler.pt
├── rng_state.pt    # dict with cpu/cuda/python/numpy state per rank
└── trainer_state.json   # global_step, loss, lr, best_val_loss
```

**Best checkpoint**: after every eval, if `current_ppl < best_val_loss`, atomically replace `{output_dir}/checkpoint-best/`. The rotation logic (`save_total_limit`) never deletes `checkpoint-best/`.

**RNG state** — per-rank:
```python
def get_rng_states():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "python": random.getstate(), "numpy": np.random.get_state()}
```

**DeepSpeed eval**: consolidate ZeRO shards with `zero_to_fp32.py` before loading into a plain model for evaluation.

---

## 9. Logging

`MetricsLogger` dispatches to console, TensorBoard, W&B — rank 0 only.

Per-dataset loss (`loss/fineweb-edu`, `loss/starcoderdata`) is accumulated per micro-step and flushed at each `logging_steps` call. This makes it easy to detect if one source is hurting the other.

NaN/Inf tracking: `nan_count` is logged as a metric. Training stops after 3 consecutive NaN steps with a `TrainingInstabilityError`.

---

## 10. Evaluation

`eval.py` computes:

1. **Perplexity** on a held-out validation set:
   ```
   PPL = exp( mean( cross_entropy_loss ) )
   ```

2. **Few-shot prompt completions** (qualitative check):
   - `"def fibonacci(n):"` → model continues
   - `"Explain what a transformer is:"` → model continues

Metrics logged to console and TensorBoard/W&B.

```bash
python eval.py \
    --config configs/3b.yaml \
    --checkpoint checkpoints/3b/checkpoint-best \
    --eval_file data/validation.jsonl
```

---

## 11. Launching

### Single-node, 8 GPU (DeepSpeed)
```bash
bash scripts/launch_training.sh \
    --config configs/3b.yaml \
    --backend deepspeed \
    --num_gpus 8
```

### Multi-node (torchrun)
```bash
torchrun \
    --nnodes 8 \
    --nproc_per_node 8 \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:29500 \
    train.py --config configs/3b.yaml
```

### Tokenizer training
```bash
python -m tokenizer.train_tokenizer \
    --output_dir tokenizer/ \
    --vocab_size 50000 \
    --train_size_gb 10
```

---

## 12. Scalability Notes

| Token budget | Steps (8.4M tok/step) | Wall time (64 A100s, ~35 tok/s/GPU) |
|--------------|-----------------------|--------------------------------------|
| 10 B         | ~1 200                | ~6 h                                 |
| 50 B         | ~6 000                | ~30 h                                |
| 300 B        | ~35 700               | ~7.5 days                            |

Multi-node extension:
- DeepSpeed: add `--hostfile hostfile` to launcher
- FSDP: use `torchrun` with `--nnodes` (shown above)
- No code changes required for either backend

---

## 13. Dependencies

Core: `torch>=2.2`, `pydantic>=2.0`, `transformers`, `datasets`, `tokenizers`,
`deepspeed`, `accelerate`, `pyyaml`, `xxhash`, `tensorboard`, `tqdm`

Optional: `flash-attn>=2.1` (strongly recommended), `wandb`, `datasketch`
(MinHash near-dedup)

---

## 14. Verification Checklist

1. **Param count**: `sum(p.numel() for p in model.parameters())` ∈ [3.1B, 3.2B]
2. **Smoke run**: `python train.py --config configs/tiny.yaml` — 10 steps on CPU, loss printed, checkpoint saved
3. **Packing + source tags**: each batch has shape `(B, 4096)` and a `sources` list of length `B`
4. **Token-budgeted mixing ratio**: run mixer for 10 000 samples; token counts per source within 5% of configured weights
5. **Dedup**: same text inserted twice → second call returns duplicate
6. **Unicode normalization**: `normalize_text("ﬁle")` → `"file"`; control chars stripped
7. **Scheduler**: LR at step 0 ≈ 0, at `warmup_steps` ≈ `lr`, at `max_steps` ≈ `lr × min_lr_ratio`
8. **NaN detection**: injected NaN loss → `TrainingInstabilityError` raised after 3 occurrences
9. **Best checkpoint**: after 30 steps with eval every 10, `checkpoint-best/` exists and survives rotation
10. **ZeRO-3 init**: tiny config with `zero_stage: 3` initializes without OOM
11. **Checkpoint round-trip** (FSDP): save → load → `state_dict()` keys and shapes match
12. **`torch.compile` sanity**: forward pass output numerically close to uncompiled
