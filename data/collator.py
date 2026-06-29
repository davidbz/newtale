from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorForCLM:
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        sources = [f["source"] for f in features]
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "sources": sources,
        }
