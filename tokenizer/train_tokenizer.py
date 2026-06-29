"""Train a byte-level BPE tokenizer on FineWeb-Edu + StarCoder data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def _check_hf_token() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise OSError(
            "HF_TOKEN environment variable is required to access HuggingFace datasets. "
            "Run: export HF_TOKEN=<your_token>"
        )


def _make_text_iterator(
    train_size_bytes: int,
    num_fineweb_shards: int = 10,
    num_starcoderdata_shards: int = 5,
) -> Iterator[str]:
    """Stratified iterator that interleaves shards from each dataset."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    token = os.environ["HF_TOKEN"]

    fineweb_shards = [
        load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True,
            token=token,
        )
        for _ in range(num_fineweb_shards)
    ]
    starcoder_shards = [
        load_dataset(
            "bigcode/starcoderdata",
            split="train",
            streaming=True,
            token=token,
        )
        for _ in range(num_starcoderdata_shards)
    ]

    iters = [iter(ds) for ds in fineweb_shards + starcoder_shards]
    consumed = 0
    idx = 0

    while consumed < train_size_bytes:
        it = iters[idx % len(iters)]
        idx += 1
        try:
            sample = next(it)
        except StopIteration:
            continue
        text = sample.get("text") or sample.get("content") or ""
        if text:
            consumed += len(text.encode())
            yield text


def train(output_dir: Path, vocab_size: int, train_size_gb: float) -> None:
    _check_hf_token()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)  # type: ignore[assignment]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )

    train_size_bytes = int(train_size_gb * 1e9)
    text_iter = _make_text_iterator(train_size_bytes)
    tokenizer.train_from_iterator(text_iter, trainer=trainer)

    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"Tokenizer saved to {output_dir / 'tokenizer.json'} (vocab={vocab_size})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--train_size_gb", type=float, default=10.0)
    args = parser.parse_args()
    train(args.output_dir, args.vocab_size, args.train_size_gb)


if __name__ == "__main__":
    main()
