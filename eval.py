"""Evaluation script: perplexity + few-shot completions.

Usage:
    python eval.py --config configs/3b.yaml \
                   --checkpoint checkpoints/3b/checkpoint-best \
                   --eval_file data/validation.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset

from config import load_config
from data.collator import collate_for_clm
from model.transformer import NewTaleForCausalLM
from tokenizer.tokenizer import NewTaleTokenizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
_logger = logging.getLogger(__name__)

_FEW_SHOT_PROMPTS = [
    "def fibonacci(n):",
    "Explain what a transformer is:",
]


def load_model_from_checkpoint(
    checkpoint_path: Path, config_path: Path
) -> NewTaleForCausalLM:
    """Load a consolidated (non-sharded) model checkpoint."""
    config = load_config(config_path)
    model = NewTaleForCausalLM(config.model)

    # Try loading model.pt (FSDP style) or pytorch_model.bin (HF style)
    model_pt = checkpoint_path / "model.pt"
    if not model_pt.exists():
        raise FileNotFoundError(
            f"No model.pt found in {checkpoint_path}. "
            "For DeepSpeed ZeRO checkpoints, run zero_to_fp32.py first to consolidate shards."
        )
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def compute_perplexity(
    model: NewTaleForCausalLM,
    eval_file: Path,
    tokenizer: NewTaleTokenizer,
    seq_length: int,
    batch_size: int,
    device: torch.device,
) -> float:
    class JsonlDataset(IterableDataset):  # type: ignore[type-arg]
        def __init__(self, path: Path, tok: NewTaleTokenizer, seq_len: int) -> None:
            self._path = path
            self._tok = tok
            self._seq_len = seq_len

        def __iter__(self):  # type: ignore[override]
            buffer: list[int] = []
            with open(self._path) as f:
                for line in f:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    ids = [*self._tok.encode(text), self._tok.eos_id]
                    buffer.extend(ids)
                    while len(buffer) >= self._seq_len:
                        yield {
                            "input_ids": torch.tensor(
                                buffer[: self._seq_len], dtype=torch.long
                            ),
                            "source": "eval",
                        }
                        buffer = buffer[self._seq_len :]

    dataset = JsonlDataset(eval_file, tokenizer, seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_for_clm)

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch.pop("sources", None)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss, _ = model(input_ids, labels)
            total_loss += loss.item()
            n_batches += 1

    if n_batches == 0:
        _logger.warning("Eval file produced no batches — check file format")
        return float("inf")

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)
    return ppl


@torch.no_grad()
def generate(
    model: NewTaleForCausalLM,
    tokenizer: NewTaleTokenizer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    ids = [tokenizer.bos_id, *tokenizer.encode(prompt)]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        _, logits = model(input_ids)
        next_id = int(logits[0, -1].argmax())
        if next_id == tokenizer.eos_id:
            break
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], device=device)], dim=1
        )

    generated = input_ids[0, len(ids) :].tolist()
    return tokenizer.decode(generated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval_file", type=Path, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info("Device: %s", device)

    tokenizer = NewTaleTokenizer(config.data.tokenizer_dir)
    model = load_model_from_checkpoint(args.checkpoint, args.config)
    model = model.to(device)
    model.eval()

    # Perplexity
    if args.eval_file is not None:
        ppl = compute_perplexity(
            model,
            args.eval_file,
            tokenizer,
            seq_length=config.data.seq_length,
            batch_size=args.batch_size,
            device=device,
        )
        _logger.info("Perplexity: %.2f", ppl)

    # Few-shot completions
    for prompt in _FEW_SHOT_PROMPTS:
        completion = generate(model, tokenizer, prompt, args.max_new_tokens, device)
        print(f"\n--- Prompt: {prompt!r} ---\n{completion}\n")


if __name__ == "__main__":
    main()
