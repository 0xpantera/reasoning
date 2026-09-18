import torch

from reasoning import get_device


def test_get_device_returns_torch_device() -> None:
    assert isinstance(get_device(enable_tensor_cores=False), torch.device)
