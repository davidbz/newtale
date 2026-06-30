from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def get_rng_states() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    torch.set_rng_state(states["cpu"])
    if states.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(states["cuda"])
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])


class CheckpointManager:
    def __init__(self, output_dir: str | Path, save_total_limit: int = 3) -> None:
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.best_val_loss: float = float("inf")
        self._rank: int = dist.get_rank() if dist.is_initialized() else 0

    def _ckpt_dir(self, step: int) -> Path:
        return self.output_dir / f"checkpoint-{step}"

    def _best_dir(self) -> Path:
        return self.output_dir / "checkpoint-best"

    # ------------------------------------------------------------------
    # FSDP2 path — uses torch.distributed.checkpoint (DCP)
    # ------------------------------------------------------------------

    def save_fsdp(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        trainer_state: dict[str, Any],
    ) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict

        ckpt_dir = self._ckpt_dir(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Per-rank distributed save — no rank-0 gather, no OOM on large models.
        model_sd, optim_sd = get_state_dict(model, optimizer)
        dcp.save(  # type: ignore[attr-defined]
            {"model": model_sd, "optimizer": optim_sd},
            checkpoint_id=str(ckpt_dir),
        )

        # Non-distributed metadata: scheduler, RNG, trainer state (rank-0 only).
        if self._rank == 0:
            torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
            torch.save(get_rng_states(), ckpt_dir / "rng_state.pt")
            trainer_state["global_step"] = step
            trainer_state["best_val_loss"] = self.best_val_loss
            (ckpt_dir / "trainer_state.json").write_text(json.dumps(trainer_state))

        if dist.is_initialized():
            dist.barrier()

        if self._rank == 0:
            self._rotate()

    def load_weights(self, path: str | Path, model: torch.nn.Module) -> None:
        """Load only model weights from a DCP checkpoint (no optimizer/scheduler)."""
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        ckpt_dir = Path(path)
        state: dict[str, Any] = {"model": {}}
        dcp.load(state, checkpoint_id=str(ckpt_dir))  # type: ignore[attr-defined]
        set_model_state_dict(model, state["model"])  # type: ignore[attr-defined]

    def load_fsdp(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> dict[str, Any]:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict

        ckpt_dir = Path(path)
        state: dict[str, Any] = {"model": {}, "optimizer": {}}
        dcp.load(state, checkpoint_id=str(ckpt_dir))  # type: ignore[attr-defined]
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )

        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler.pt", weights_only=True)
        )
        rng_states = torch.load(ckpt_dir / "rng_state.pt", weights_only=False)
        restore_rng_states(rng_states)
        trainer_state: dict[str, Any] = json.loads(
            (ckpt_dir / "trainer_state.json").read_text()
        )
        self.best_val_loss = trainer_state.get("best_val_loss", float("inf"))
        return trainer_state

    # ------------------------------------------------------------------
    # DeepSpeed path
    # ------------------------------------------------------------------

    def save_deepspeed(
        self,
        step: int,
        engine: Any,
        trainer_state: dict[str, Any],
    ) -> None:
        engine.save_checkpoint(str(self.output_dir), tag=f"checkpoint-{step}")
        if self._rank == 0:
            ckpt_dir = self._ckpt_dir(step)
            trainer_state["global_step"] = step
            trainer_state["best_val_loss"] = self.best_val_loss
            (ckpt_dir / "trainer_state.json").write_text(json.dumps(trainer_state))
            self._rotate()

    def load_deepspeed(self, engine: Any, tag: str | None = None) -> dict[str, Any]:
        _, client_state = engine.load_checkpoint(str(self.output_dir), tag=tag)
        state: dict[str, Any] = client_state or {}
        self.best_val_loss = state.get("best_val_loss", float("inf"))
        return state

    # ------------------------------------------------------------------
    # Best checkpoint
    # ------------------------------------------------------------------

    def maybe_save_best(self, val_loss: float, step: int) -> bool:
        if val_loss >= self.best_val_loss:
            return False
        self.best_val_loss = val_loss
        if self._rank == 0:
            best = self._best_dir()
            if best.exists():
                shutil.rmtree(best)
            src = self._ckpt_dir(step)
            if src.exists():
                shutil.copytree(src, best)
        if dist.is_initialized():
            dist.barrier()
        return True

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _rotate(self) -> None:
        """Delete oldest checkpoints beyond save_total_limit."""
        pattern = sorted(
            [
                d
                for d in self.output_dir.iterdir()
                if d.name.startswith("checkpoint-") and d.name != "checkpoint-best"
            ],
            key=lambda d: int(d.name.split("-")[1]),
        )
        while len(pattern) > self.save_total_limit:
            shutil.rmtree(pattern.pop(0))

    # ------------------------------------------------------------------
    # Latest detection
    # ------------------------------------------------------------------

    def find_latest(self) -> str | None:
        candidates = (
            [
                d
                for d in self.output_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint-")
                and d.name != "checkpoint-best"
            ]
            if self.output_dir.exists()
            else []
        )
        if not candidates:
            return None
        latest = max(candidates, key=lambda d: int(d.name.split("-")[1]))
        return str(latest)
