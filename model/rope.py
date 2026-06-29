from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(
        self, head_dim: int, max_seq_len: int, theta: float = 10_000.0
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )  # type: ignore[attr-defined]
        freqs = torch.outer(t, self.inv_freq)  # type: ignore[attr-defined]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:  # type: ignore[attr-defined]
            self._build_cache(seq_len)
        cos: torch.Tensor = self.cos_cached[:seq_len]  # type: ignore[attr-defined]
        sin: torch.Tensor = self.sin_cached[:seq_len]  # type: ignore[attr-defined]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (batch, nheads, seqlen, head_dim)
    # cos, sin: (seqlen, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
