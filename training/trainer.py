from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from config import TrainingConfig
    from training.checkpoint import CheckpointManager
    from training.logging_utils import MetricsLogger

_logger = logging.getLogger(__name__)


class TrainingInstabilityError(RuntimeError):
    pass


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,  # type: ignore[type-arg]
        eval_loader: DataLoader,  # type: ignore[type-arg]
        checkpoint_manager: CheckpointManager,
        metrics_logger: MetricsLogger,
        start_step: int = 0,
        tokens_per_step: int = 0,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.ckpt = checkpoint_manager
        self.log = metrics_logger
        self.start_step = start_step
        self.tokens_per_step = tokens_per_step

    # ------------------------------------------------------------------
    # DeepSpeed training loop
    # ------------------------------------------------------------------

    def train_deepspeed(self, engine: Any) -> None:
        cfg = self.config
        data_iter = iter(self.train_loader)
        nan_count = 0
        tokens_seen = self.start_step * self.tokens_per_step
        step_start = time.perf_counter()
        prof = self._maybe_start_profiler()

        for global_step in range(self.start_step, cfg.max_steps):
            engine.train()
            step_loss = 0.0

            for _ in range(cfg.gradient_accumulation_steps):
                raw_batch = next(data_iter)
                sources: list[str] = raw_batch.pop("sources")
                gpu_batch = {k: v.cuda() for k, v in raw_batch.items()}
                loss, _ = engine(**gpu_batch)
                engine.backward(loss)
                step_loss += loss.item()
                for src in sources:
                    self.log.record_source_loss(src, loss.item())

            if math.isnan(step_loss) or math.isinf(step_loss):
                nan_count += 1
                _logger.warning(
                    "NaN/Inf loss at step %d (count=%d)", global_step, nan_count
                )
                self.log.log(global_step, {"nan_count": nan_count})
                if nan_count >= 3:
                    raise TrainingInstabilityError(
                        f"3 consecutive NaN losses starting at step {global_step}"
                    )
                engine.zero_grad()
                continue
            nan_count = 0

            engine.step()
            tokens_seen += self.tokens_per_step

            if global_step % cfg.logging_steps == 0:
                elapsed = time.perf_counter() - step_start
                tps = self.tokens_per_step * cfg.logging_steps / max(elapsed, 1e-6)
                step_start = time.perf_counter()
                self.log.log(
                    global_step,
                    {
                        "loss": step_loss / cfg.gradient_accumulation_steps,
                        "lr": engine.get_lr()[0],
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": tps,
                    },
                )

            if global_step % cfg.eval_steps == 0 and global_step > 0:
                ppl = self.evaluate_deepspeed(engine)
                self.log.log(global_step, {"perplexity": ppl})
                self.ckpt.maybe_save_best(ppl, global_step)

            if global_step % cfg.save_steps == 0 and global_step > 0:
                self.ckpt.save_deepspeed(
                    global_step,
                    engine,
                    {"loss": step_loss / cfg.gradient_accumulation_steps},
                )

            if prof is not None:
                prof.step()

        self.log.close()

    def evaluate_deepspeed(self, engine: Any) -> float:
        engine.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for raw_batch in self.eval_loader:
                raw_batch.pop("sources")
                gpu_batch = {k: v.cuda() for k, v in raw_batch.items()}
                loss, _ = engine(**gpu_batch)
                total_loss += loss.item()
                n_batches += 1
        if n_batches == 0:
            return float("inf")
        return math.exp(total_loss / n_batches)

    # ------------------------------------------------------------------
    # FSDP2 training loop
    # ------------------------------------------------------------------

    def train_fsdp(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> None:
        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_iter = iter(self.train_loader)
        nan_count = 0
        tokens_seen = self.start_step * self.tokens_per_step
        step_start = time.perf_counter()
        prof = self._maybe_start_profiler()

        for global_step in range(self.start_step, cfg.max_steps):
            model.train()
            total_loss = 0.0

            for _ in range(cfg.gradient_accumulation_steps):
                raw_batch = next(data_iter)
                sources: list[str] = raw_batch.pop("sources")
                gpu_batch = {k: v.to(device) for k, v in raw_batch.items()}
                loss, _ = model(**gpu_batch)
                (loss / cfg.gradient_accumulation_steps).backward()
                total_loss += loss.item()
                for src in sources:
                    self.log.record_source_loss(src, loss.item())

            if math.isnan(total_loss) or math.isinf(total_loss):
                nan_count += 1
                _logger.warning(
                    "NaN/Inf loss at step %d (count=%d)", global_step, nan_count
                )
                self.log.log(global_step, {"nan_count": nan_count})
                if nan_count >= 3:
                    raise TrainingInstabilityError(
                        f"3 consecutive NaN losses starting at step {global_step}"
                    )
                optimizer.zero_grad()
                continue
            nan_count = 0

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tokens_seen += self.tokens_per_step

            if global_step % cfg.logging_steps == 0:
                elapsed = time.perf_counter() - step_start
                tps = self.tokens_per_step * cfg.logging_steps / max(elapsed, 1e-6)
                step_start = time.perf_counter()
                self.log.log(
                    global_step,
                    {
                        "loss": total_loss / cfg.gradient_accumulation_steps,
                        "lr": scheduler.get_last_lr()[0],
                        "grad_norm": float(grad_norm),
                        "tokens_seen": tokens_seen,
                        "tokens_per_sec": tps,
                    },
                )

            if global_step % cfg.eval_steps == 0 and global_step > 0:
                ppl = self.evaluate_fsdp(model, device)
                self.log.log(global_step, {"perplexity": ppl})
                self.ckpt.save_fsdp(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    {"loss": total_loss / cfg.gradient_accumulation_steps},
                )
                self.ckpt.maybe_save_best(ppl, global_step)

            elif global_step % cfg.save_steps == 0 and global_step > 0:
                self.ckpt.save_fsdp(
                    global_step,
                    model,
                    optimizer,
                    scheduler,
                    {"loss": total_loss / cfg.gradient_accumulation_steps},
                )

            if prof is not None:
                prof.step()

        self.log.close()

    def evaluate_fsdp(self, model: nn.Module, device: torch.device) -> float:
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for raw_batch in self.eval_loader:
                raw_batch.pop("sources")
                gpu_batch = {k: v.to(device) for k, v in raw_batch.items()}
                loss, _ = model(**gpu_batch)
                total_loss += loss.item()
                n_batches += 1
        if n_batches == 0:
            return float("inf")
        return math.exp(total_loss / n_batches)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def _maybe_start_profiler(self) -> Any:
        cfg = self.config
        if cfg.profile_steps is None:
            return None
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=cfg.warmup_steps,
                warmup=1,
                active=cfg.profile_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.output_dir),
            with_stack=False,
        )
        prof.start()
        return prof
