"""Neural-network building blocks for Qwen3.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from reasoning.qwen3.rope import apply_rope


class RMSNorm(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        eps: float = 1e-6,
        bias: bool = False,
        qwen3_compatible: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps) * self.scale
        if self.shift is not None:
            normalized = normalized + self.shift
        return normalized.to(input_dtype)


class FeedForward(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.fc1 = nn.Linear(
            config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            config["emb_dim"], config["hidden_dim"], dtype=config["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            config["hidden_dim"], config["emb_dim"], dtype=config["dtype"], bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_heads: int,
        num_kv_groups: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_groups != 0:
            raise ValueError("num_heads must be divisible by num_kv_groups")
        if head_dim is None:
            if d_in % num_heads != 0:
                raise ValueError("d_in must be divisible by num_heads")
            head_dim = d_in // num_heads

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim) if qk_norm else None
        self.k_norm = RMSNorm(head_dim) if qk_norm else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)
        keys_new = keys.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        values_new = values.view(
            batch_size, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)

        if self.q_norm is not None:
            queries = self.q_norm(queries)
        if self.k_norm is not None:
            keys_new = self.k_norm(keys_new)

        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is None:
            keys, values = keys_new, values_new
        else:
            previous_keys, previous_values = cache
            keys = torch.cat([previous_keys, keys_new], dim=2)
            values = torch.cat([previous_values, values_new], dim=2)
        next_cache = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores = attention_scores.masked_fill(mask, -torch.inf)
        attention_weights = torch.softmax(attention_scores / self.head_dim**0.5, dim=-1)
        context = (
            (attention_weights @ values)
            .transpose(1, 2)
            .reshape(batch_size, num_tokens, self.d_out)
        )
        return self.out_proj(context), next_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=config["emb_dim"],
            num_heads=config["n_heads"],
            head_dim=config["head_dim"],
            num_kv_groups=config["n_kv_groups"],
            qk_norm=config["qk_norm"],
            dtype=config["dtype"],
        )
        self.ff = FeedForward(config)
        self.norm1 = RMSNorm(config["emb_dim"])
        self.norm2 = RMSNorm(config["emb_dim"])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        shortcut = x
        x, next_cache = self.att(
            self.norm1(x), mask, cos, sin, start_pos=start_pos, cache=cache
        )
        x = x + shortcut
        return x + self.ff(self.norm2(x)), next_cache
