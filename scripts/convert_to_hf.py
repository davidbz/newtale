"""Convert a NewTale checkpoint to HuggingFace LLaMA format.

The output directory can be:
  - loaded with AutoModelForCausalLM.from_pretrained()
  - passed to llama.cpp's convert_hf_to_gguf.py

Weight key mapping (ours → HF LLaMA):
  embed_tokens.weight                         → model.embed_tokens.weight
  layers.{i}.attn_norm.weight                 → model.layers.{i}.input_layernorm.weight
  layers.{i}.attn.{q,k,v,o}_proj.weight       → model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
  layers.{i}.ffn_norm.weight                  → model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.ffn.{gate,up,down}_proj.weight   → model.layers.{i}.mlp.{gate,up,down}_proj.weight
  norm.weight                                 → model.norm.weight
  lm_head.weight                              → lm_head.weight

Usage:
    python scripts/convert_to_hf.py \\
        --checkpoint checkpoints/1b-single/checkpoint-500 \\
        --config     configs/1b-single-gpu.yaml \\
        --output     hf-model/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_config

_TOPLEVEL: dict[str, str] = {
    "embed_tokens.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}


def remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        if "rotary" in key:
            continue  # cos/sin buffers are recomputed from config at load time

        if key in _TOPLEVEL:
            out[_TOPLEVEL[key]] = tensor
            continue

        if key.startswith("layers."):
            _, idx, rest = key.split(".", 2)
            if rest == "attn_norm.weight":
                out[f"model.layers.{idx}.input_layernorm.weight"] = tensor
            elif rest == "ffn_norm.weight":
                out[f"model.layers.{idx}.post_attention_layernorm.weight"] = tensor
            elif rest.startswith("attn."):
                out[f"model.layers.{idx}.self_attn.{rest[5:]}"] = tensor
            elif rest.startswith("ffn."):
                out[f"model.layers.{idx}.mlp.{rest[4:]}"] = tensor
            else:
                out[f"model.layers.{idx}.{rest}"] = tensor
            continue

        out[key] = tensor

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NewTale checkpoint → HF LLaMA format"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Checkpoint directory containing model.pt"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training YAML config used to produce the checkpoint",
    )
    parser.add_argument("--output", required=True, help="Output directory for HF model")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    m = cfg.model

    model_pt = ckpt_dir / "model.pt"
    if not model_pt.exists():
        sys.exit(f"ERROR: {model_pt} not found. Did training save a checkpoint?")

    print(f"Loading {model_pt} …")
    sd: dict[str, torch.Tensor] = torch.load(
        model_pt, map_location="cpu", weights_only=True
    )
    print(f"  {len(sd)} keys loaded")

    remapped = remap_state_dict(sd)
    print(f"  {len(remapped)} keys after remapping")

    vocab_size: int = remapped["model.embed_tokens.weight"].shape[0]
    print(f"  vocab_size (from weights): {vocab_size}")

    # ------------------------------------------------------------------ #
    # Save weights                                                         #
    # ------------------------------------------------------------------ #
    try:
        from safetensors.torch import save_file  # type: ignore[import-untyped]

        out_weights = output_dir / "model.safetensors"
        save_file(remapped, str(out_weights))
        print(f"Saved {out_weights}  ({out_weights.stat().st_size / 1e9:.2f} GB)")
    except ImportError:
        out_weights = output_dir / "pytorch_model.bin"
        torch.save(remapped, out_weights)
        print(f"safetensors not installed — saved as {out_weights}")
        print("  pip install safetensors   ← llama.cpp prefers this format")

    # ------------------------------------------------------------------ #
    # config.json  (HF LlamaConfig schema)                                #
    # ------------------------------------------------------------------ #
    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_act": "silu",
        "hidden_size": m.hidden_size,
        "intermediate_size": m.intermediate_size,
        "max_position_embeddings": m.max_position_embeddings,
        "num_attention_heads": m.num_attention_heads,
        "num_hidden_layers": m.num_layers,
        "num_key_value_heads": m.num_key_value_heads,
        "rms_norm_eps": m.rms_norm_eps,
        "rope_theta": m.rope_theta,
        "tie_word_embeddings": m.tie_word_embeddings,
        "vocab_size": vocab_size,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2))
    print("Saved config.json")

    # ------------------------------------------------------------------ #
    # generation_config.json                                              #
    # ------------------------------------------------------------------ #
    (output_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
            },
            indent=2,
        )
    )

    # ------------------------------------------------------------------ #
    # Tokenizer files                                                      #
    # ------------------------------------------------------------------ #
    tokenizer_dir = Path(cfg.data.tokenizer_dir)
    copied = []
    for f in sorted(tokenizer_dir.iterdir()):
        if f.suffix in {".json", ".txt", ".model"}:
            shutil.copy(f, output_dir / f.name)
            copied.append(f.name)
    print(f"Copied tokenizer files: {copied}")

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    size_gb = sum(t.numel() * t.element_size() for t in remapped.values()) / 1e9
    print(f"\n{'─' * 50}")
    print(f"Output : {output_dir.resolve()}")
    print(f"Params : {sum(t.numel() for t in remapped.values()) / 1e9:.2f}B")
    print(f"Size   : {size_gb:.2f} GB (bf16)")
    print(f"{'─' * 50}")
    print("\nNext steps:\n")
    print("  # Verify it loads:")
    print(
        f"  python -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{output_dir}'); print(m.config)\""
    )
    print()
    print("  # Convert to GGUF (run from llama.cpp repo):")
    print(
        f"  python convert_hf_to_gguf.py {output_dir.resolve()} --outfile newtale-1b-f16.gguf --outtype bf16"
    )
    print()
    print("  # Quantize:")
    print("  ./llama-quantize newtale-1b-f16.gguf newtale-1b-q4_k_m.gguf Q4_K_M")
    print()
    print("  # Run:")
    print('  ./llama-cli -m newtale-1b-q4_k_m.gguf -p "Once upon a time" -n 200')


if __name__ == "__main__":
    main()
