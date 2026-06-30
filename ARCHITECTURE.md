# Architecture

End-to-end flow for training a 3B decoder-only Transformer from scratch on raw text.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         train.py (entrypoint)                           │
│                                                                         │
│  load_config ──► DataSourceConfig × N                                   │
│                       │                                                 │
│                        ──► PackedStreamingDataset                       │
│                                   │                                     │
│                                   ▼                                     │
│                             DataLoader                                  │
│                                   │                                     │
│            ┌──────────────────────┘                                     │
│            ▼                                                            │
│      NewTaleForCausalLM                                                 │
│            │                                                            │
│            ▼                                                            │
│         Trainer ──► loss ──► backward ──► optimizer ──► scheduler      │
│            │                                                            │
│            ├──► CheckpointManager (save / rotate / best)               │
│            └──► MetricsLogger (console / TensorBoard / W&B)            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Configuration (`config.py`)

```
configs/1b-single-gpu.yaml
         │
         ▼
    load_config()
         │
         ▼
┌─────────────────────────────────────────────┐
│ Config (pydantic BaseModel)                 │
│                                             │
│  ModelConfig      → architecture dims       │
│  DataConfig       → sources, seq_length     │
│    DataSourceConfig × N                     │
│      path, weight, subset                   │
│      text_column  → which field has text    │
│      dedup        → opt-in hash dedup       │
│  TrainingConfig   → LR, steps, batch size   │
│  LoggingConfig    → W&B / TensorBoard       │
└─────────────────────────────────────────────┘
```

**Why pydantic:** YAML maps cleanly to nested `BaseModel` classes with free type coercion and validation at load time. Wrong types or missing required fields raise immediately.

**Why `text_column`:** Different HF datasets use different field names (`text` for FineWeb-Edu, `content` for StarCoderData). The pipeline normalises to `"text"` so nothing downstream needs to know which dataset it's reading.

---

## 2. Data Pipeline

The data pipeline has three concerns: **what to load**, **how to mix it**, and **how to pack it** into fixed-length training chunks.

### 2.1 Lazy Sharded Loading (inside `PackedStreamingDataset.__iter__`)

```
  HuggingFace Hub (parquet files, up to TB-scale)
           │
           │  load_dataset(streaming=True)          ← never downloads the full dataset
           ▼
  IterableDataset (lazy parquet reader)
           │
           │  .select_columns([src.text_column])    ← column pruning at parquet read
           │                                          time — other columns never enter
           │                                          RAM (url, score, id, etc.)
           ▼
  IterableDataset (text column only)
           │
           │  .rename_column(text_column, "text")   ← normalise field name
           │                                          (only if text_column != "text")
           ▼
  IterableDataset (uniform "text" field)
           │
           │  .shard(num_shards=total, index=i)     ← disjoint parquet file slices
           │                                          per (rank, worker) pair
           ▼
  shard for this worker only
```

**Sharding formula:**
```
total_shards  = world_size × max(1, num_workers)
shard_index   = rank × max(1, num_workers) + worker_id
```
With `num_workers=0` (single process): collapses to `total_shards=world_size`, `shard_index=rank`. Each GPU reads a different slice with no overlap and no redundant buffering.

**Why load inside `__iter__`:** DataLoader forks workers *after* the dataset is constructed. Building the HF dataset object inside `__iter__` means each worker builds its own independent iterator — no shared state, no race conditions.

---

### 2.2 Weighted Mixing (`data/mixing.py`)

```
  source A (fineweb-edu shard)  ─────┐
  source B (starcoderdata shard) ────┤
                                     ▼
                         WeightedDatasetMixer
                         ─────────────────────
                         Tracks tokens yielded per source.
                         At each step picks the source with
                         the largest deficit vs its target
                         fraction (e.g. 70% / 30%).
                         Adds {"_source": name} to each sample.
                                     │
                                     ▼
                         interleaved sample stream
```

**Why token-budgeted (not sample-budgeted):** FineWeb-Edu documents average ~1,500 tokens; StarCoder files can be as short as 20. A 70/30 sample ratio would produce a very different token ratio. Tracking actual tokens yielded per source makes the configured weights mean exactly what they say — 70% of training tokens are web text, 30% are code.

---

### 2.3 Preprocessing (`data/preprocessing.py`)

Each document passes a stateless filter chain before tokenisation:

```
  raw text
     │
     ▼  normalize_text()
     │    NFKC unicode normalisation (ﬁ → fi, ² → 2, etc.)
     │    strip C0/C1 control characters (keep \t \n \r)
     │    collapse runs of 3+ blank lines to 2
     │
     ▼  length_filter()
     │    drop if len < 200 or > 100,000 chars
     │    (too short = boilerplate; too long = code dumps / legal docs)
     │
     ▼  strip_html()
     │    drop if >20% of chars are inside HTML tags
     │    strip remaining tags
     │
     ▼  repetition_filter()
     │    compute 5-gram frequency distribution
     │    drop if top 5-gram > 30% of all 5-grams
     │    (catches copy-paste spam and template pages)
     │
     ▼  ExactDedup.is_duplicate()   ← only if src.dedup = true
     │    xxh64 hash → Python set
     │    bounded at dedup_max_entries (best-effort beyond cap)
     │    off by default — FineWeb-Edu and StarCoderData are already deduped
     │
     ▼  clean text (or None → skip document)
```

---

### 2.4 Packing (`PackedStreamingDataset`)

Language model training requires fixed-length sequences. The naive approach (pad each document) wastes tokens on short documents. Packing avoids this:

```
  document 1: [tok tok tok tok EOS]
  document 2: [tok tok EOS]
  document 3: [tok tok tok tok tok tok tok EOS]
  document 4: [tok tok tok ...

  rolling buffer: ─────────────────────────────────────────────►
                  [d1..EOS | d2..EOS | d3..EOS | d4....
                   └──── seq_length = 2048 ────┘
                         yield chunk
                                     └──── seq_length = 2048 ────┘
                                           yield chunk

  Each yielded chunk: {"input_ids": tensor(2048,), "source": dominant_source}
```

**Why EOS between documents:** The causal mask is a single continuous window. Without EOS, the model can attend across document boundaries and learn spurious correlations between unrelated texts.

**Why track source per token:** Each yielded chunk majority-votes its `source` field from the per-token source tags tracked in a parallel buffer. This flows through to the trainer for per-dataset loss logging (`loss/fineweb-edu`, `loss/starcoderdata`).

---

### 2.5 Collation (`data/collator.py`)

```
  [chunk₁, chunk₂, ..., chunkB]   ← list of B dicts from DataLoader
            │
            ▼
  DataCollatorForCLM
            │
            ▼
  {
    input_ids : tensor(B, T)
    labels    : tensor(B, T)   ← same as input_ids (CLM: predict the next token)
    sources   : list[str]      ← per-sample dataset name, for loss breakdown
  }
```

**Why `labels = input_ids.clone()`:** The model does the shift internally (`logits[..., :-1, :]` vs `labels[..., 1:]`), so collation just stacks and clones. The collator is stateless and trivially correct.

---

## 3. Model Architecture (`model/`)

```
  input_ids  (B, T)
       │
       ▼
  embed_tokens                  nn.Embedding(vocab_size, hidden_size)
       │
       │ + RoPE (cos, sin)      precomputed for seq_len, shared across all layers
       ▼
  ┌──────────────────────────────────────────────────────────┐
  │  TransformerBlock  ×  num_layers                         │
  │                                                          │
  │  ┌─ pre-norm (RMSNorm) ──────────────────────────────┐  │
  │  │                                                    │  │
  │  │  GroupedQueryAttention                             │  │
  │  │    q_proj  → (B, T, n_heads,    head_dim)         │  │
  │  │    k_proj  → (B, T, n_kv_heads, head_dim)  ◄─ GQA: fewer KV heads
  │  │    v_proj  → (B, T, n_kv_heads, head_dim)         │  │
  │  │    apply_rotary_emb(q, k, cos, sin)                │  │
  │  │    FlashAttention2  (if installed, fp16/bf16)       │  │
  │  │      OR F.scaled_dot_product_attention             │  │
  │  │         + repeat_interleave (expand KV heads)      │  │
  │  │    o_proj → (B, T, hidden_size)                    │  │
  │  │                                                    │  │
  │  └────────────────────── + residual ──────────────────┘  │
  │                                                          │
  │  ┌─ pre-norm (RMSNorm) ──────────────────────────────┐  │
  │  │                                                    │  │
  │  │  SwiGLU FFN                                        │  │
  │  │    gate_proj(x) → silu activation (gate)          │  │
  │  │    up_proj(x)   → values                          │  │
  │  │    down_proj(gate ⊙ values) → (B, T, hidden_size) │  │
  │  │                                                    │  │
  │  └────────────────────── + residual ──────────────────┘  │
  └──────────────────────────────────────────────────────────┘
       │
       ▼
  final RMSNorm
       │
       ▼
  lm_head                       nn.Linear(hidden_size, vocab_size, bias=False)
       │
       ▼
  logits  (B, T, vocab_size)
       │
       ▼  cross_entropy(logits[:, :-1], labels[:, 1:])
       │  shift done inside model — predict token t+1 from tokens 0..t
       ▼
  loss (scalar)
```

**Why RMSNorm instead of LayerNorm:** Skips mean subtraction — cheaper, and empirically equivalent. Pre-norm (before attention/FFN, not after) stabilises training at large scale.

**Why GQA (Grouped Query Attention):** n_kv_heads < n_heads means the KV cache is smaller during inference. At 1B: 16 Q heads, 8 KV heads (2:1 ratio). At 3B: 24 Q heads, 8 KV heads (3:1). Full attention quality, fraction of the memory cost.

**Why SwiGLU:** Gated activation — the gate tensor controls which values flow through, giving the network a learned sparse-ish activation. Empirically 1–2% better perplexity than GeLU at equivalent parameter count, hence used in LLaMA, Mistral, Gemma.

**Why RoPE:** Position encoding is applied to Q and K after projection, not added to embeddings. Relative position is encoded in the dot product, so the model generalises better to sequence lengths beyond training. No learned position embeddings = one less thing that can overfit.

---

## 4. Training Loop (`training/trainer.py`)

```
  data_iter = iter(train_loader)

  for global_step in range(start_step, max_steps):

    ┌─── gradient accumulation (cfg.gradient_accumulation_steps) ───┐
    │                                                                │
    │  batch = next(data_iter)                                       │
    │  loss, _ = model(input_ids, labels)                            │
    │  (loss / grad_accum_steps).backward()    ← FSDP path          │
    │  loss.backward()                         ← DeepSpeed path     │
    │  accumulate per-source loss for logging                        │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘
           │
           ▼  NaN / Inf check
           │  if nan_count >= 3 → raise TrainingInstabilityError
           │
           ▼
    clip_grad_norm_(model, max_grad_norm)   ← FSDP only; DeepSpeed handles internally
    optimizer.step()
    scheduler.step()                        ← cosine decay with linear warmup
    optimizer.zero_grad()

    every logging_steps  → log loss, lr, grad_norm, per-source loss
    every eval_steps     → eval loop → compute PPL → update best checkpoint
    every save_steps     → save checkpoint, rotate oldest beyond save_total_limit
```

**Why gradient accumulation:** Effective batch size = `per_device_train_batch_size × grad_accum_steps × world_size`. On a single A10G with 23GB VRAM, `batch=1, accum=8` → effective batch of 8, without needing 8× the memory.

**Why NaN detection:** A single NaN propagates forward through all subsequent steps silently. Three consecutive NaNs indicates a fundamental instability (LR too high, bad data batch, numerical overflow) rather than a transient spike — raising there avoids training silently on garbage.

**LR schedule:**
```
  0 ──► warmup_steps     : linear 0 → lr
  warmup_steps ──► max_steps : cosine lr → lr × min_lr_ratio
```
Warmup prevents large gradient steps before the model has stabilised. Cosine decay is smooth and avoids sharp drops.

---

## 5. Distributed Strategy (`training/distributed.py`)

```
  world_size == 1 (single GPU)
       model.cuda().bfloat16()
       standard optimizer.step() loop

  world_size > 1, backend = "fsdp"   ← FSDP2 (composable API, PyTorch 2.4+)
       build_device_mesh(world_size)  → DeviceMesh("cuda", [world_size])
       fully_shard(layer, mesh=mesh, mp_policy=bf16)  ← each TransformerBlock
       fully_shard(model, mesh=mesh, mp_policy=bf16)  ← root model last
       weights sharded across ranks — each rank holds 1/N of every layer
       all-gather before forward, reduce-scatter after backward

  world_size > 1, backend = "deepspeed"
       ZeRO-2: optimizer state + gradients sharded
       ZeRO-3: weights + optimizer state + gradients sharded
               → model must be constructed inside deepspeed.zero.Init()
```

**FSDP2 vs FSDP1:** FSDP2 uses the composable `fully_shard` API instead of `FullyShardedDataParallel`. Inner modules (individual `TransformerBlock` layers) must be sharded before the root model — reversing the order silently leaves layers unsharded. `torch.compile` must come after all `fully_shard` calls; the model cannot be rewrapped after `OptimizedModule` wraps it.

**FP8 (optional, H100+):** `fp8_training: true` in config calls `torchao.float8.convert_to_float8_training(model)` after FSDP2 wrapping and before `torch.compile`. Requires `torchao` installed.

**Why bf16 (not fp16):** bf16 has the same exponent range as fp32 — no gradient scaler needed, no inf/nan from overflow in activations. fp16 has 5-bit exponent vs bf16's 8-bit; LLM activations regularly exceed fp16 range.

**Why compile=False in single-GPU configs:** `torch.compile` adds 2–5 min of warmup on first run. For short experiments or debugging, the overhead isn't worth it. Enable for long production runs.

---

## 6. Checkpointing & Export (`training/checkpoint.py`, `scripts/convert_to_hf.py`)

```
  Every save_steps:

  FSDP2 path                         DeepSpeed path
  ──────────────────                 ──────────────────
  get_state_dict(model, optimizer)   engine.save_checkpoint()
  dcp.save({"model": …,              creates DS-native shards
             "optimizer": …},
            checkpoint_id=ckpt_dir)
  ← per-rank shard files (.distcp)
    no rank-0 gather, no OOM

  rank 0 only:
    scheduler.pt
    rng_state.pt
    trainer_state.json
  dist.barrier()

  Rotation: keep save_total_limit newest checkpoints
            never delete "checkpoint-best/"

  After eval: if PPL < best → copy to checkpoint-best/

  Eval-only load (no optimizer/scheduler):
    CheckpointManager.load_weights(path, model)
    → dcp.load({"model": {}}, …) + set_model_state_dict(model, …)

  ──────────────────────────────────────────────────────────────────────
  Export (scripts/convert_to_hf.py):

  ⚠ convert_to_hf.py currently expects a consolidated model.pt and
    predates DCP. Before running it, consolidate the DCP checkpoint:

    python -c "
    import torch, torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import set_model_state_dict
    from model.transformer import NewTaleForCausalLM
    from config import load_config
    cfg = load_config('configs/3b.yaml')
    m = NewTaleForCausalLM(cfg.model)
    state = {'model': {}}
    dcp.load(state, checkpoint_id='checkpoints/checkpoint-best')
    set_model_state_dict(m, state['model'])
    torch.save(m.state_dict(), 'checkpoints/checkpoint-best/model.pt')
    "

  checkpoint-best/model.pt
       │
       ▼  remap weight keys  (our naming → HF LLaMA naming)
       │    e.g. layers.0.attn.q_proj.weight
       │      → model.layers.0.self_attn.q_proj.weight
       │
       ▼  save as safetensors
       │
       ▼  write config.json  (HF LlamaConfig schema)
       │
       ▼  copy tokenizer files
       │
       ▼  hf-model/
              │
              ▼  convert_hf_to_gguf.py  (llama.cpp)
              ▼  llama-quantize Q4_K_M
              ▼  llama-cli -m model-q4_k_m.gguf
```

---

## 7. Key Numbers (1B single-GPU config)

| Parameter | Value | Why |
|---|---|---|
| `hidden_size` | 2048 | ~1B params with 22 layers |
| `num_layers` | 22 | depth vs width tradeoff |
| `num_attention_heads` | 16 | 128-dim per head |
| `num_key_value_heads` | 8 | GQA 2:1 ratio |
| `intermediate_size` | 5632 | SwiGLU: ~2.75× hidden (not 4×) |
| `max_position_embeddings` | 2048 | fits in A10G at batch=1 |
| `seq_length` | 2048 | matches max_position_embeddings |
| `per_device_train_batch_size` | 1 | VRAM budget |
| `gradient_accumulation_steps` | 8 | effective batch = 8 |
| `dataloader_num_workers` | 0 | avoids RAM OOM via worker forking |

---

## 8. RAM Optimisation Notes (streaming datasets)

The main source of system RAM pressure is the HuggingFace parquet reader. Three mitigations applied:

1. **`select_columns([text_column])`** — pyarrow never reads metadata columns (url, score, id, etc.) from disk. Applied before sharding so only the text column chunks are loaded per shard.

2. **`.shard(num_shards, index)`** — each worker reads only `1/N` of the parquet files. With `world_size=1, num_workers=0`: N=1 (full dataset), but `subset: sample-10BT` keeps FineWeb-Edu to ~400 shards instead of 2400+.

3. **`subset: sample-10BT`** — the pre-packaged 10B-token slice of FineWeb-Edu. Same quality as the full dataset (uniform sample), far fewer files to buffer.

Set before launching:
```bash
export HF_DATASETS_CACHE=/path/to/disk/.cache  # avoid tmpfs
export TOKENIZERS_PARALLELISM=false             # no background tokenizer threads
```
