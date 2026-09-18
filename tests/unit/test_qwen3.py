import torch

from reasoning.generation import generate_text_basic, generate_text_basic_cache
from reasoning.qwen3 import Qwen3Model, RMSNorm, apply_rope, compute_rope_params


def tiny_config() -> dict:
    return {
        "vocab_size": 32,
        "context_length": 8,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 32,
        "head_dim": 4,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 1_000_000.0,
        "dtype": torch.float32,
    }


def deterministic_model() -> Qwen3Model:
    model = Qwen3Model(tiny_config())
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32)
            values = ((values + 7 * parameter_index) % 23 - 11) / 32
            parameter.copy_(values.reshape_as(parameter))
    return model.eval()


def test_rope_preserves_shape_and_dtype() -> None:
    values = torch.randn(2, 4, 3, 8, dtype=torch.float32)
    cos, sin = compute_rope_params(head_dim=8, context_length=6)
    result = apply_rope(values, cos, sin, offset=2)
    assert result.shape == values.shape
    assert result.dtype == values.dtype


def test_rms_norm_shape() -> None:
    values = torch.randn(2, 3, 8)
    assert RMSNorm(8)(values).shape == values.shape


def test_greedy_generation_matches_reference_fixture() -> None:
    expected = torch.tensor([[15, 11, 1, 1, 1]])
    prompt = torch.tensor([[1, 2, 3]])

    assert torch.equal(
        generate_text_basic(deterministic_model(), prompt, max_new_tokens=5),
        expected,
    )
    assert torch.equal(
        generate_text_basic_cache(deterministic_model(), prompt, max_new_tokens=5),
        expected,
    )
