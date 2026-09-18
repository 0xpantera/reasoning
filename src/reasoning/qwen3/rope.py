"""Rotary positional embeddings used by Qwen3.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

import torch


def compute_rope_params(
    head_dim: int,
    theta_base: float = 10_000,
    context_length: int = 4096,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for split-halves RoPE."""
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even")

    dimensions = torch.arange(0, head_dim, 2, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (dimensions.float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Apply split-halves rotary positional embeddings."""
    _, _, seq_len, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even")

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return ((x * cos) + (rotated * sin)).to(dtype=x.dtype)
