from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator


class WeightedDatasetMixer:
    """Token-budgeted dataset mixer.

    Tracks tokens yielded per source and always pulls from the source with the
    largest deficit vs its target fraction, so configured weights reflect the
    actual token distribution rather than sample counts.
    """

    def __init__(
        self,
        datasets: list[Any],
        weights: list[float],
        names: list[str],
        seed: int = 42,
    ) -> None:
        if len(datasets) != len(weights) or len(datasets) != len(names):
            raise ValueError("datasets, weights, and names must have the same length")
        if abs(sum(weights) - 1.0) > 1e-4:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights):.4f}")

        self._rng = random.Random(seed)
        self._iters: list[Iterator[Any]] = [iter(itertools.cycle(ds)) for ds in datasets]
        self._targets = weights
        self._names = names
        self._token_counts = [0] * len(datasets)

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            total = sum(self._token_counts) or 1
            deficits = [
                self._targets[i] - self._token_counts[i] / total
                for i in range(len(self._iters))
            ]
            idx = deficits.index(max(deficits))
            sample: dict[str, Any] = next(self._iters[idx])
            sample["_source"] = self._names[idx]
            # Approximate token count from text length before actual tokenisation
            self._token_counts[idx] += len(sample.get("text", "")) // 4
            yield sample
