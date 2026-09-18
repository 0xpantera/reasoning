"""Readable, from-scratch Qwen3 model.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from reasoning.qwen3.cache import KVCache
from reasoning.qwen3.layers import RMSNorm, TransformerBlock
from reasoning.qwen3.rope import compute_rope_params


class Qwen3Model(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(
            config["vocab_size"], config["emb_dim"], dtype=config["dtype"]
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config["n_layers"])]
        )
        self.final_norm = RMSNorm(config["emb_dim"])
        self.out_head = nn.Linear(
            config["emb_dim"], config["vocab_size"], bias=False, dtype=config["dtype"]
        )

        head_dim = config["head_dim"] or config["emb_dim"] // config["n_heads"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=config["rope_base"],
            context_length=config["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = dict(config)
        self.current_pos = 0

    def forward(
        self, in_idx: torch.Tensor, cache: KVCache | None = None
    ) -> torch.Tensor:
        x = self.tok_emb(in_idx)
        num_tokens = x.shape[1]
        if cache is None:
            pos_start = 0
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        else:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool),
                diagonal=1,
            )[pos_start:pos_end, :pos_end]
        mask = mask[None, None, :, :]

        for index, block in enumerate(self.trf_blocks):
            block_cache = cache.get(index) if cache is not None else None
            x, next_cache = block(
                x,
                mask,
                self.cos,
                self.sin,
                start_pos=pos_start,
                cache=block_cache,
            )
            if cache is not None:
                cache.update(index, next_cache)

        return self.out_head(self.final_norm(x).to(self.cfg["dtype"]))

    def reset_kv_cache(self) -> None:
        self.current_pos = 0
