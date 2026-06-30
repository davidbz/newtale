from __future__ import annotations

from typing import Any

import torch


def collate_for_clm(features: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = torch.stack([f["input_ids"] for f in features])
    sources = [f["source"] for f in features]
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "sources": sources,
    }
