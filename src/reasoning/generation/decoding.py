"""Greedy decoding routines introduced in Chapter 2.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

from collections.abc import Iterator
from typing import Protocol

import torch

from reasoning.qwen3 import KVCache


class CacheableModel(Protocol):
    cfg: dict

    def eval(self): ...
    def reset_kv_cache(self) -> None: ...
    def __call__(self, token_ids: torch.Tensor, cache=None) -> torch.Tensor: ...


@torch.inference_mode()
def generate_text_basic(
    model: CacheableModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    input_length = token_ids.shape[1]
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(token_ids)[:, -1]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_cache(
    model: CacheableModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    input_length = token_ids.shape[1]
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(token_ids, cache=cache)[:, -1]
    generated_tokens = []

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        generated_tokens.append(next_token)
        logits = model(next_token, cache=cache)[:, -1]

    if generated_tokens:
        return torch.cat(generated_tokens, dim=1)
    return token_ids[:, input_length:]


@torch.inference_mode()
def generate_text_basic_stream(
    model: CacheableModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[torch.Tensor]:
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(token_ids)[:, -1]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        yield next_token
        token_ids = torch.cat([token_ids, next_token], dim=1)


@torch.inference_mode()
def generate_text_basic_stream_cache(
    model: CacheableModel,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> Iterator[torch.Tensor]:
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_kv_cache()
    logits = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_new_tokens):
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
        yield next_token
        logits = model(next_token, cache=cache)[:, -1]
