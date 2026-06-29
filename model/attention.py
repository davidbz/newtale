from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RotaryEmbedding, apply_rotary_emb

if TYPE_CHECKING:
    from config import ModelConfig

_flash_attn_func = None
with contextlib.suppress(ImportError):
    from flash_attn import (  # pyright: ignore[reportMissingImports]
        flash_attn_func as _flash_attn_func,  # type: ignore[import-untyped,assignment]
    )


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_kv_groups
        self.head_dim = config.head_dim
        hidden = config.hidden_size

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and reshape to (B, H, T, D) for RoPE
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if _flash_attn_func is not None:
            # flash_attn_func expects (B, T, H, D)
            out = _flash_attn_func(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                causal=True,
            )  # (B, T, H, D)
            out = out.reshape(B, T, self.num_heads * self.head_dim)
        else:
            # SDPA expects (B, H, T, D); expand KV heads to match Q heads
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)

        return self.o_proj(out)
