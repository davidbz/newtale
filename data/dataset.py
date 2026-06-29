from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import IterableDataset

from data.preprocessing import ExactDedup, preprocess

if TYPE_CHECKING:
    from collections.abc import Iterator

    from data.mixing import WeightedDatasetMixer
    from tokenizer.tokenizer import NewTaleTokenizer


class PackedStreamingDataset(IterableDataset):  # type: ignore[type-arg]
    """Packs tokenised documents into fixed-length chunks with no padding."""

    def __init__(
        self,
        mixer: WeightedDatasetMixer,
        tokenizer: NewTaleTokenizer,
        seq_length: int = 4096,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._mixer = mixer
        self._tokenizer = tokenizer
        self._seq_length = seq_length
        self._seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self._mixer.reseed(self._seed + worker_info.id)

        dedup = ExactDedup()
        buffer: list[int] = []
        source_buffer: list[str] = []

        for sample in self._mixer:
            text = sample.get("text", "")
            source = sample.get("_source", "unknown")

            cleaned = preprocess(text, dedup)
            if cleaned is None:
                continue

            ids = [*self._tokenizer.encode(cleaned), self._tokenizer.eos_id]
            buffer.extend(ids)
            source_buffer.extend([source] * len(ids))

            while len(buffer) >= self._seq_length:
                chunk_sources = source_buffer[: self._seq_length]
                dominant = max(set(chunk_sources), key=chunk_sources.count)
                yield {
                    "input_ids": torch.tensor(
                        buffer[: self._seq_length], dtype=torch.long
                    ),
                    "source": dominant,
                }
                buffer = buffer[self._seq_length :]
                source_buffer = source_buffer[self._seq_length :]


def make_streaming_dataset(
    hf_path: str,
    split: str = "train",
    **kwargs: Any,
) -> Any:
    from datasets import load_dataset  # type: ignore[import-untyped]

    return load_dataset(hf_path, split=split, streaming=True, **kwargs)


