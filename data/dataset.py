from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import IterableDataset

from data.mixing import WeightedDatasetMixer
from data.preprocessing import ExactDedup, preprocess

if TYPE_CHECKING:
    from collections.abc import Iterator

    from config import DataSourceConfig
    from tokenizer.tokenizer import NewTaleTokenizer


class PackedStreamingDataset(IterableDataset):  # type: ignore[type-arg]
    """Packs tokenised documents into fixed-length chunks with no padding.

    Datasets are loaded lazily inside __iter__ so each DataLoader worker only
    buffers its own disjoint shard of the source parquet files.
    """

    def __init__(
        self,
        sources: list[DataSourceConfig],
        tokenizer: NewTaleTokenizer,
        seq_length: int = 4096,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        dedup_max_entries: int = 500_000,
    ) -> None:
        super().__init__()
        self._sources = sources
        self._tokenizer = tokenizer
        self._seq_length = seq_length
        self._seed = seed
        self._rank = rank
        self._world_size = world_size
        self._num_workers = num_workers
        self._dedup_max_entries = dedup_max_entries

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        effective_num_workers = max(1, self._num_workers)

        total_shards = self._world_size * effective_num_workers
        shard_index = self._rank * effective_num_workers + worker_id

        from datasets import load_dataset  # type: ignore[import-untyped]

        hf_datasets = []
        for src in self._sources:
            ds = load_dataset(
                src.path, name=src.subset, split=src.split, streaming=True
            )
            ds = ds.select_columns([src.text_column])
            if src.text_column != "text":
                ds = ds.rename_column(src.text_column, "text")
            if total_shards > 1:
                ds = ds.shard(num_shards=total_shards, index=shard_index)
            hf_datasets.append(ds)

        mixer = WeightedDatasetMixer(
            datasets=hf_datasets,
            weights=[s.weight for s in self._sources],
            names=[s.name for s in self._sources],
            seed=self._seed + shard_index,
        )

        source_dedup: dict[str, ExactDedup | None] = {
            src.name: ExactDedup(max_entries=self._dedup_max_entries)
            if src.dedup
            else None
            for src in self._sources
        }

        buffer: list[int] = []
        source_buffer: list[str] = []

        for sample in mixer:
            text = sample.get("text", "")
            source = sample.get("_source", "unknown")

            cleaned = preprocess(text, dedup=source_dedup.get(source))
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
