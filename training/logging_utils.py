from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from typing import Any

_logger = logging.getLogger(__name__)

_SummaryWriter: type | None = None
with contextlib.suppress(ImportError):
    from torch.utils.tensorboard import (
        SummaryWriter as _SummaryWriter,  # type: ignore[import-untyped,assignment]
    )

_wandb: Any = None
with contextlib.suppress(ImportError):
    import wandb as _wandb  # type: ignore[import-untyped]


class MetricsLogger:
    def __init__(
        self,
        rank: int,
        tensorboard_dir: str | None = None,
        use_wandb: bool = False,
        wandb_project: str = "newtale",
        run_name: str | None = None,
    ) -> None:
        self._rank = rank
        self._tb: Any = None
        self._wb: Any = None
        self._source_loss_accum: dict[str, list[float]] = defaultdict(list)

        if rank != 0:
            return

        if tensorboard_dir is not None and _SummaryWriter is not None:
            self._tb = _SummaryWriter(log_dir=tensorboard_dir)

        if use_wandb and _wandb is not None:
            self._wb = _wandb.init(project=wandb_project, name=run_name)

    def record_source_loss(self, source: str, loss: float) -> None:
        if self._rank == 0:
            self._source_loss_accum[source].append(loss)

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        if self._rank != 0:
            return

        for source, losses in self._source_loss_accum.items():
            metrics[f"loss/{source}"] = sum(losses) / len(losses)
        self._source_loss_accum.clear()

        parts = [f"step={step}"] + [
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ]
        _logger.info("  ".join(parts))

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, global_step=step)

        if self._wb is not None:
            self._wb.log({"step": step, **metrics})

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wb is not None:
            self._wb.finish()
