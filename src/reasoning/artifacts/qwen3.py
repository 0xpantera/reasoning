"""Qwen3 artifact locations used by the book.

Adapted from Sebastian Raschka's reasoning-from-scratch project.
"""

from pathlib import Path
from typing import Literal

from reasoning.artifacts.download import download_file


def download_qwen3_small(
    kind: Literal["base", "reasoning"] = "base",
    tokenizer_only: bool = False,
    out_dir: str | Path = ".",
) -> dict[str, Path]:
    """Download the book's Qwen3 0.6B tokenizer and optionally its weights."""
    files = {
        "base": {
            "model": "qwen3-0.6B-base.pth",
            "tokenizer": "tokenizer-base.json",
        },
        "reasoning": {
            "model": "qwen3-0.6B-reasoning.pth",
            "tokenizer": "tokenizer-reasoning.json",
        },
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    targets = ("tokenizer",) if tokenizer_only else ("model", "tokenizer")
    downloaded = {}
    for target in targets:
        filename = files[kind][target]
        primary = (
            f"https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/{filename}"
        )
        backup = (
            "https://f001.backblazeb2.com/file/reasoning-from-scratch/"
            f"qwen3-0.6B/{filename}"
        )
        downloaded[target] = download_file(primary, out_dir, backup)
    return downloaded
