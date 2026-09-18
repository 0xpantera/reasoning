"""Key-value cache for autoregressive decoding.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

import torch

LayerCache = tuple[torch.Tensor, torch.Tensor]


class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.cache: list[LayerCache | None] = [None] * n_layers

    def get(self, layer_idx: int) -> LayerCache | None:
        return self.cache[layer_idx]

    def update(self, layer_idx: int, value: LayerCache) -> None:
        self.cache[layer_idx] = value

    def get_all(self) -> list[LayerCache | None]:
        return self.cache

    def reset(self) -> None:
        for index in range(len(self.cache)):
            self.cache[index] = None
